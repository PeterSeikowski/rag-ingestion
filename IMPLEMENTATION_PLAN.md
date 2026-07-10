# Implementation Plan — rag-ingestion-service

This document explains the architecture, components, assumptions, and build sequence for the RAG ingestion service. It is the durable design record; `PROGRESS.md` tracks day-to-day execution status against this plan.

## 1. Purpose

A standalone service that accepts PDFs or metadata-only records, parses/normalizes them, chunks them, creates embeddings, and indexes the results into a VectorStore (Elasticsearch 8.x for v1). Ingestion is fully asynchronous: API calls return a `job_id` immediately; a Celery worker does the actual parse → chunk → embed → index work.

This is v1 of a system that will eventually support multimodal ingestion (tables, images, page images, ColPali-style multi-vector embeddings) and additional VectorStore/parser/chunker/embedder backends. The architecture is chosen so those can be added later **without changing the domain model's shape or the ports' contracts** — only new adapters and enum values.

## 2. Architecture: strict hexagonal / clean architecture

```
            ┌─────────────┐        ┌──────────────┐
  HTTP ───► │   api/      │        │  workers/    │ ◄─── Celery broker (Redis)
            │ (FastAPI)   │        │  (Celery)    │
            └──────┬──────┘        └──────┬───────┘
                   │  calls                │  calls
                   ▼                       ▼
            ┌─────────────────────────────────────┐
            │           application/                │
            │  ingestion_pipeline / ingestion_service │
            │  / job_service  (orchestration only)   │
            └───────────────┬─────────────────────┘
                            │ depends on (Protocols)
                            ▼
            ┌─────────────────────────────────────┐
            │              ports/                    │
            │  parser / chunker / embedder /         │
            │  vector_store_writer / job_repository  │
            └───────────────┬─────────────────────┘
                            │ implemented by
                            ▼
            ┌─────────────────────────────────────┐
            │             adapters/                  │
            │  docling_pdf_parser | chunkers |       │
            │  litellm_embedder | elasticsearch_writer│
            │  | redis_job_repository                 │
            └─────────────────────────────────────┘

            ┌─────────────────────────────────────┐
            │              domain/                   │
            │  pure Pydantic models & enums,         │
            │  imported by every layer above,        │
            │  imports nothing from any of them      │
            └─────────────────────────────────────┘
```

**Dependency rule**: arrows only point "inward" toward `domain/`. `domain/` has zero knowledge of FastAPI, Celery, Elasticsearch, Docling, or LiteLLM. `application/` knows only `domain/` and `ports/` (Protocols), never a concrete adapter class. `api/` and `workers/` know `application/` and `domain/`, and touch concrete adapters **only** through a single composition root (`bootstrap.py`) that wires ports to adapters at startup.

If a change ever requires importing `elasticsearch`, `docling`, or `fastapi` into `domain/` or `application/`, that is a signal the abstraction is wrong — stop and fix the port instead.

## 3. Components

| Layer | Responsibility | Must NOT contain |
|---|---|---|
| `api/` | FastAPI routes, request/response (de)serialization, HTTP status mapping | ES calls, Docling calls, embedding calls, business rules |
| `application/` | Orchestrates parse→chunk→embed→index; job lifecycle; request validation | Elasticsearch DSL, Docling objects, Celery-specific code |
| `domain/` | `DocumentRecord`, `ChunkRecord`, `EmbeddingRecord`, etc. — pure data + validation | Any infrastructure import |
| `ports/` | `Protocol` interfaces: `DocumentParser`, `Chunker`, `Embedder`, `VectorStoreWriter`, `JobRepository` | Any concrete implementation |
| `adapters/` | Concrete infra implementations, one infra library per adapter file | Leaking infra types past the adapter boundary |
| `workers/` | Celery app + tasks; task bodies call `application/` services | Parsing/chunking/embedding/indexing logic itself |
| `config/` | `Settings` (env vars), structured JSON logging setup | Business logic |

## 4. Domain model overview

Six files under `domain/`, matching the required repo structure exactly:

- **`enums.py`** — `Modality` (text/table/image/page_image/mixed), `OwnerType` (chunk/page/image/table), `ElementType`, `JobStatus`, `JobStage`, `ChunkingStrategy`, `ChunkType` (parent/child/leaf), `SourceType` (pdf/metadata).
- **`provenance.py`** — `BoundingBox`, `SourceReference` (page numbers, bounding boxes, section path, element IDs). Bounding boxes are **never invented**: if Docling doesn't expose one for an element, `bounding_boxes` is an empty list.
- **`documents.py`** — `DocumentRecord` (top-level document metadata), `ParsedDocument`/`ParsedElement` (canonical parser output), `MetadataIngestionRecord` (input shape for metadata-only ingestion).
- **`chunks.py`** — `ChunkRecord`, `ChunkingConfig`.
- **`embeddings.py`** — `EmbeddingRecord`, `EmbeddingConfig`. `EmbeddingRecord` carries `vector: list[float]` **and** a currently-unused `multi_vector: list[list[float]] | None` field, plus `owner_type`, so ColPali-style page/image multi-vector embeddings can be added later without a schema migration.
- **`jobs.py`** — `IngestionJobStatus`.

**Key separation**: `ChunkRecord` never holds a vector. `EmbeddingRecord` is a distinct record type, written to a distinct Elasticsearch index, referencing its owner (`owner_id`, `owner_type`) rather than embedding the vector inline in the chunk. This is what lets the same chunk later gain multiple embeddings (different models, or a ColPali page-image embedding alongside a text embedding) without restructuring chunk storage.

## 5. Elasticsearch index strategy

Three indices, created idempotently by `ElasticsearchVectorStoreWriter.ensure_indices()`:

- `rag_documents_v1` — one doc per `DocumentRecord`.
- `rag_chunks_v1` — one doc per `ChunkRecord`. **No vector field.** Canonical chunk text lives here.
- `rag_embeddings_v1` — one doc per `EmbeddingRecord`, with a `dense_vector` field. Duplicates `owner_id`, `document_id`, `modality`, `model_name` from the owning chunk for retrieval-time joins/filters, but not the chunk text itself.

**Known MVP constraint**: `dense_vector` requires a fixed `dims` at index-creation time. This deployment supports one embedding model at a time; `dimensions` is resolved from the configured `Embedder` at bootstrap. Switching embedding models later requires creating a new embeddings index and re-ingesting (documented as a production TODO, not solved now).

Idempotency: documents/chunks/embeddings use deterministic IDs and `elasticsearch.helpers.bulk` upserts. On re-ingestion of an existing `document_id`, the pipeline calls `delete_document(document_id)` **before** writing new chunks/embeddings, so a change in chunking parameters (different chunk count) can't leave orphaned old chunks behind — ID-overwrite alone does not guarantee that.

## 6. Chunking framework

`Chunker` is a `Protocol` (`name`, `version`, `chunk(parsed_document, config) -> list[ChunkRecord]`). Two implementations for v1:

- **`SectionBasedHierarchicalChunker`** (default for PDFs) — walks `ParsedDocument.elements` tracking a heading-level stack to build `section_path`; produces one **parent** `ChunkRecord` per section (full section text, provenance aggregated across its elements) and multiple **child** `ChunkRecord`s under each parent (section text split into token windows), wiring `parent_chunk_id`/`child_chunk_ids` both ways. Table elements are never split — one chunk per table, even if it exceeds `chunk_size_tokens`. Parent chunks are not embedded by default (`ChunkingConfig.embed_parent_chunks = False`); they exist for small-to-big retrieval-time context expansion.
- **`RecursiveTextChunker`** (fallback, used when section structure is weak, and for metadata-only records) — token-window splits with configurable size/overlap, best-effort `SourceReference` from whichever elements a window overlaps.

Both share a pure helper, `adapters/chunkers/_token_windowing.py::window_text(...)`, instead of one chunker importing the other — keeps the two strategies independently swappable and the windowing arithmetic independently testable. `adapters/chunkers/registry.py` maps `ChunkingStrategy -> Chunker` so `application/ingestion_pipeline.py` can select a strategy per-request without a code change; new strategies register here.

## 7. Parsing framework

`DocumentParser` is a `Protocol` (`name`, `version`, `supported_formats`, `parse(file_path, source_reference) -> ParsedDocument`). Only `DoclingPdfParser` exists for v1.

Docling's own object model is not stable across versions, so the adapter wraps each element's field access defensively: a `_safe_extract_item()` helper logs and skips a malformed element rather than aborting the whole document, and `docling` is pinned to an exact version in `requirements.txt`. Page numbers, bounding boxes, and section/heading structure are preserved when Docling provides them; when it doesn't, we store an empty bounding-box list rather than invent coordinates. Tables and pictures become `ParsedElement`s with `element_type` `TABLE`/`IMAGE` even though only text is embedded in v1 — this is what makes future multimodal embedding additive rather than a rewrite.

## 8. Embedding framework

`Embedder` is a `Protocol` (`name`, `provider`, `model`, `dimensions` property, `embed_texts(texts, config) -> list[list[float]]`). Deliberately returns raw vectors, not `EmbeddingRecord` — the embedder has no concept of "chunk" or "owner"; `application/ingestion_pipeline.py` zips vectors with chunk/document metadata to build `EmbeddingRecord`s. This keeps `LiteLLMEmbedder` a pure "text in, vectors out" component, reusable regardless of what's being embedded (chunks today, page images later).

`LiteLLMEmbedder` batches texts (`EMBEDDING_BATCH_SIZE`), calls `litellm.embedding(...)`, asserts the returned vector count matches the input count, and fails fast and clearly if required config (`LITELLM_MODEL`, `LITELLM_API_KEY`) is missing. It does not import or know about Elasticsearch.

## 9. Async processing

Both ingestion endpoints follow the same shape: validate request → (PDF only) save upload to `TEMP_UPLOAD_DIR/{job_id}/...` → create an `IngestionJobStatus` (status `PENDING`) → enqueue a Celery task → return `{job_id, status}` immediately.

Two Celery tasks in `workers/tasks.py`: `ingest_pdf_task` and `ingest_metadata_task`. Both are thin — they resolve adapters via the shared `bootstrap.py` composition root and call `application/ingestion_pipeline.py`, updating job status via `application/job_service.py` at each stage (`parsing` → `chunking` → `embedding` → `indexing` → `completed`/`failed`). Temp files are deleted in a `finally` block regardless of outcome.

`JobRepository` is Redis-backed (`adapters/jobs/redis_job_repository.py`), storing `IngestionJobStatus` as JSON keyed by `job_id` with a TTL. It is the **sole source of truth** for job state — status is never read back via Celery's own `AsyncResult`, keeping `application/job_service.py` decoupled from Celery internals (the entire point of routing it through a port).

**Deployment note**: because the API process writes the temp upload and the worker process reads it, they must share the temp upload directory (a Docker named volume in `docker-compose.yml`). The Celery worker runs with `--pool=solo` for MVP to avoid fork-after-construct issues with long-lived adapter clients (Redis/ES sockets) built once at worker startup; multi-process scaling is a documented production TODO.

## 10. Configuration

`config/settings.py` uses `pydantic-settings`, reading everything from environment variables — no hardcoded URLs or credentials. Beyond the settings the spec requires, a few are added because they're operationally necessary, not scope creep: Elasticsearch index name overrides, `JOB_STATUS_TTL_SECONDS`, `MAX_UPLOAD_SIZE_BYTES`, `DOCLING_ENABLE_OCR`. `config/logging.py` sets up structured JSON logging and binds `job_id`/`document_id` via `contextvars` so every log line in a request/task is automatically correlated.

## 11. Assumptions and known MVP limitations

- Elasticsearch is external and already running; this repo never starts it.
- One embedding model per deployment (dims are fixed at index-creation time).
- No auth, no object storage, no Kubernetes manifests, no Unstructured fallback, no ColPali implementation — only forward-compatible fields.
- Local verification in this sandbox environment: Docker is not installed here, so `Dockerfile`/`docker-compose.yml` are reviewed by careful reading rather than built; `pytest` runs against a lightweight dependency set with hand-written fakes for the heavy adapters (Docling/Elasticsearch/LiteLLM/Celery broker) rather than installing the full (torch-heavy) `requirements.txt`. Both are documented explicitly in the README as the path a developer with Docker/full deps would use instead.
- Temp file cleanup is best-effort (`finally` blocks); a periodic sweep for orphaned files after an API-process crash is a documented production TODO, not implemented.

## 12. Build sequence

Steps 0–13 as enumerated in the project brief; tracked live in `PROGRESS.md`. Executed sequentially by a single author (not fanned out across independent agents) because every later layer depends on the exact shape of the domain models and ports defined early on — this is a tightly-coupled system, not an independently-parallelizable set of tasks. A final architecture-compliance sweep (grep-based checks for each hard rule, plus a code-review pass) runs after core implementation and before the README is finalized.
