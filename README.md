# rag-ingestion-service

A standalone RAG ingestion service: accepts PDFs or metadata-only records, parses/normalizes them, chunks them, creates embeddings, and indexes everything into Elasticsearch. Built with strict hexagonal/clean architecture (FastAPI + Celery + Redis + Docling + LiteLLM + Elasticsearch 8.x), fully asynchronous — every ingestion endpoint returns a `job_id` immediately and does the real work in a Celery worker.

See [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for the full architecture rationale and design decisions, and [`PROGRESS.md`](PROGRESS.md) for the step-by-step build log. [`CLAUDE.md`](CLAUDE.md) has quick-reference rules for anyone (human or AI) extending this codebase.

## Architecture overview

Strict hexagonal / clean architecture: dependencies only point inward, toward `domain/`.

```
  HTTP ──► api/ (FastAPI)         workers/ (Celery) ◄── Redis broker
              │                          │
              ▼                          ▼
         application/  (orchestration — ingestion_pipeline, ingestion_service, job_service)
              │  depends on
              ▼
           ports/  (Protocol interfaces: parser, chunker, embedder, vector_store_writer, job_repository)
              │  implemented by
              ▼
         adapters/  (Docling, LiteLLM, Elasticsearch, Redis — one infra library per adapter file)

         domain/  (pure Pydantic models & enums — imported by every layer above, imports nothing from them)
```

- **`domain/`** — `DocumentRecord`, `ChunkRecord`, `EmbeddingRecord`, `ParsedDocument`/`ParsedElement`, `IngestionJobStatus`, etc. Zero infrastructure imports.
- **`ports/`** — five `typing.Protocol` interfaces: `DocumentParser`, `Chunker`, `Embedder`, `VectorStoreWriter`, `JobRepository`.
- **`adapters/`** — the only place concrete infra libraries are imported: `docling` only in `adapters/parsers/docling_pdf_parser.py`, `elasticsearch` only in `adapters/vectorstores/elasticsearch_writer.py`, `litellm` only in `adapters/embedders/litellm_embedder.py`, `redis` only in `adapters/jobs/redis_job_repository.py`.
- **`application/`** — `ingestion_pipeline.py` (parse → chunk → embed → index orchestration), `ingestion_service.py` (request validation, temp file handling, job creation, Celery enqueue), `job_service.py` (job status transitions).
- **`api/`** — FastAPI routes and Pydantic request/response schemas only. No business logic.
- **`workers/`** — Celery app + tasks. Task bodies are thin wrappers that call `application/` services.
- **`bootstrap.py`** — the single composition root that wires `ports/` to concrete `adapters/` from `Settings`.

**Key separation**: a `ChunkRecord` never holds a vector. Embeddings are separate `EmbeddingRecord`s, written to a separate Elasticsearch index (`rag_embeddings_v1`), referencing their owner by ID. Parser output (`ParsedDocument`) is fully decoupled from chunking — the chunker never sees a Docling object.

## Requirements

- Python 3.11+ (developed against 3.12)
- Redis (bundled in `docker-compose.yml`)
- **An already-running Elasticsearch 8.x cluster** — this repo never starts one itself, only connects to `ELASTICSEARCH_URL`
- Docker + Docker Compose, for the full-stack path

## Local setup

### Option A — Docker Compose (recommended for the full stack)

```bash
cp .env.example .env
# edit .env: set LITELLM_API_KEY (and LITELLM_MODEL/LITELLM_API_BASE if not using OpenAI),
# and point ELASTICSEARCH_URL at a real cluster (see below for a one-off local ES container)

docker compose up --build
```

This starts `api` (port 8000), `worker`, and `redis`. Elasticsearch is **not** included — point `ELASTICSEARCH_URL` at a cluster you already run. For a quick local single-node ES 8.x for development:

```bash
docker run -d --name es8 -p 9200:9200 \
  -e discovery.type=single-node -e xpack.security.enabled=false \
  docker.elastic.co/elasticsearch/elasticsearch:8.15.0
```

...then set `ELASTICSEARCH_URL=http://host.docker.internal:9200` in `.env` (already the default in `.env.example`).

### Option B — Running the API/worker directly (no Docker)

```bash
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt      # full install, includes docling (heavy — pulls in torch)

# in one terminal
.venv\Scripts\uvicorn rag_ingestion.api.main:app --reload

# in another terminal (Redis and Elasticsearch must already be reachable)
.venv\Scripts\celery -A rag_ingestion.workers.celery_app worker --pool=solo --loglevel=info
```

`--pool=solo` matters here, not just in Docker — see [Development notes](#development-notes).

## Environment variables

All configuration is via environment variables (`config/settings.py`, `pydantic-settings`) — see `.env.example` for the full list with defaults. No URL or credential is ever hardcoded in source.

| Variable | Purpose |
|---|---|
| `APP_ENV`, `LOG_LEVEL` | environment name, structured JSON log level |
| `REDIS_URL` | job status store |
| `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND` | Celery (Redis-backed; Celery's own result backend is unused for job status — see below) |
| `JOB_STATUS_TTL_SECONDS` | how long a completed/failed job's status stays in Redis |
| `ELASTICSEARCH_URL`, `ELASTICSEARCH_USERNAME`, `ELASTICSEARCH_PASSWORD`, `ELASTICSEARCH_VERIFY_CERTS` | external Elasticsearch cluster |
| `ES_DOCUMENTS_INDEX`, `ES_CHUNKS_INDEX`, `ES_EMBEDDINGS_INDEX` | index names (defaults: `rag_documents_v1`/`rag_chunks_v1`/`rag_embeddings_v1`) |
| `LITELLM_MODEL`, `LITELLM_API_KEY`, `LITELLM_API_BASE`, `LITELLM_API_VERSION`, `LITELLM_PROVIDER` | embedding model routing |
| `EMBEDDING_BATCH_SIZE`, `EMBEDDING_DIMENSIONS` | embedding batching; `EMBEDDING_DIMENSIONS` is a manual override when the model isn't in the built-in known-dimensions table |
| `DEFAULT_CHUNK_SIZE_TOKENS`, `DEFAULT_CHUNK_OVERLAP_TOKENS` | chunking defaults when a request doesn't override them |
| `DOCLING_ENABLE_OCR` | skip OCR init for text-only PDFs (faster) |
| `TEMP_UPLOAD_DIR`, `MAX_UPLOAD_SIZE_BYTES` | temporary PDF storage; must be a volume shared by `api` and `worker` (already wired in `docker-compose.yml`) |

## API

No authentication in this MVP.

### `POST /v1/documents/pdf` — ingest a PDF

`multipart/form-data`:

```bash
curl -X POST http://localhost:8000/v1/documents/pdf \
  -F "file=@/path/to/report.pdf" \
  -F 'metadata={"title": "Q3 Report", "external_id": "report-q3-2026", "tenant_id": "acme"}' \
  -F 'chunking_config={"strategy": "section_based_hierarchical", "chunk_size_tokens": 512}'
```

Response (`202 Accepted`):

```json
{"job_id": "e0f00517-59b3-4490-8273-0b97c752ca3d", "status": "pending"}
```

`metadata` and `chunking_config` are optional JSON-encoded form fields. `embedding_config` is accepted but currently reserved — the embedding model/dimensions are server-controlled (see [Index design](#elasticsearch-index-design) for why). Supplying `metadata.external_id` makes re-ingestion idempotent: uploading the same `external_id` again derives the same internal `document_id` and replaces the previous chunks/embeddings rather than duplicating them.

### `POST /v1/records/metadata` — ingest a metadata-only record

```bash
curl -X POST http://localhost:8000/v1/records/metadata \
  -H "Content-Type: application/json" \
  -d '{"text_content": "Acme Corp, founded 1998, HQ in Springfield.", "title": "Acme Corp fact sheet", "external_id": "acme-facts"}'
```

Response (`202 Accepted`): same shape as above. No PDF, no parser involved — the record becomes exactly one chunk.

### `GET /v1/jobs/{job_id}` — check status

```bash
curl http://localhost:8000/v1/jobs/e0f00517-59b3-4490-8273-0b97c752ca3d
```

```json
{
  "job_id": "e0f00517-59b3-4490-8273-0b97c752ca3d",
  "status": "completed",
  "stage": null,
  "document_id": "478dd634-7840-50cd-b651-a3244a629e8d",
  "source_format": "pdf",
  "total_pages": 12,
  "total_chunks": 47,
  "total_embeddings": 31,
  "created_at": "...", "updated_at": "...", "completed_at": "..."
}
```

`status` is one of `pending` / `in_progress` / `completed` / `failed`; while `in_progress`, `stage` narrows it to `parsing` / `chunking` / `embedding` / `indexing`. `total_embeddings` is typically less than `total_chunks` — parent/section chunks aren't embedded by default (see below).

### `GET /health/live` / `GET /health/ready`

`/health/live` is a pure process check (no dependencies). `/health/ready` pings Redis and Elasticsearch through their ports and returns `503` with a per-dependency breakdown if either is down:

```json
{"status": "not_ready", "checks": {"redis": true, "elasticsearch": false}}
```

## Elasticsearch index design

Three indices, created idempotently at process startup (best-effort — see [Development notes](#development-notes)):

| Index | Contents |
|---|---|
| `rag_documents_v1` | One `DocumentRecord` per ingested document. |
| `rag_chunks_v1` | One `ChunkRecord` per chunk. **No vector field** — canonical chunk text lives here. |
| `rag_embeddings_v1` | One `EmbeddingRecord` per embedding, with a `dense_vector` field. Duplicates `owner_id`/`document_id`/`modality`/`model_name` from the owning chunk for retrieval-time joins — not the chunk text itself. |

**Known MVP limitation**: Elasticsearch's `dense_vector` mapping requires a fixed `dims` at index-creation time, so this deployment supports **one embedding model at a time**. `EMBEDDING_DIMENSIONS`/the model's known dimensions are baked into `rag_embeddings_v1`'s mapping the first time it's created. Switching embedding models later requires creating a new embeddings index (e.g. bump `ES_EMBEDDINGS_INDEX`) and re-ingesting everything — there is no in-place migration.

**Idempotent re-ingestion**: writes use deterministic IDs (`chunk_id`, `embedding_id`, `document_id`), but ID-overwrite alone isn't sufficient if a re-ingestion produces a different number of chunks than before — old chunks could be orphaned. So `application/ingestion_pipeline.py` calls `delete_document(document_id)` (which removes every chunk/embedding for that document, plus the document record itself) **before** writing the new set, every time.

## Chunking

Two pluggable strategies, selected via `ChunkingConfig.strategy` (`adapters/chunkers/registry.py`):

- **`section_based_hierarchical`** (default) — groups elements by heading structure, producing one **parent** chunk per section (full section text) and multiple **child** chunks under it (token-windowed). Falls back to a single implicit section when there's no heading structure. Tables are never split. Parent chunks are **not embedded by default** (`embed_parent_chunks: false`) — they exist for small-to-big retrieval-time context expansion (vector-match on children, return the parent for fuller context).
- **`recursive_text`** — plain token-window splitting, no hierarchy. Used for PDFs with weak section structure, and internally for metadata-only records (which always become exactly one chunk, bypassing the chunker entirely).

Both share `adapters/chunkers/_token_windowing.py` — a pure, provider-agnostic tokenizer (`tiktoken`/`cl100k_base`) used only for sizing, not because it matches every embedding model's real tokenizer exactly.

## Parsing

`adapters/parsers/docling_pdf_parser.py` wraps [Docling](https://github.com/docling-project/docling), converting its object model into this service's own `ParsedDocument`/`ParsedElement`. Page numbers, bounding boxes, and section/heading structure are preserved when Docling provides them; when it doesn't, an empty bounding-box list is stored rather than invented coordinates. Tables and pictures become `ParsedElement`s (`element_type=table`/`image`) even though only text is embedded today — this is what makes future multimodal ingestion additive rather than a rewrite.

Docling's public API has shifted across releases, and this specific version (pinned exactly in `requirements.txt`) hasn't been installed against a real PDF in this build's development environment (see [Development notes](#development-notes)). Field access is wrapped defensively (`# TODO(docling-api)` markers in the source) — re-verify against a real document before depending on this in production.

## Embeddings

`adapters/embedders/litellm_embedder.py` wraps [LiteLLM](https://github.com/BerriAI/litellm), giving one interface across OpenAI/Azure/Cohere/self-hosted embedding providers. The port (`ports/embedder.py`) returns plain vectors, not `EmbeddingRecord`s — the embedder has no concept of "chunk" or "owner"; `application/ingestion_pipeline.py` builds the full `EmbeddingRecord` by pairing vectors with chunk metadata.

## Multi-tenancy / access-control fields

`DocumentRecord` and `ChunkRecord` carry `tenant_id`, `allowed_groups`, and `classification` fields end-to-end (populated from request metadata, stored in Elasticsearch), so a future retrieval service can filter on them. This ingestion service does not enforce access control itself — no authentication or authorization is implemented in this MVP.

## Development notes

- **Docker is not available in this environment**, so `Dockerfile`/`docker-compose.yml` were verified by careful reading (multi-stage build, non-root user, shared temp-upload volume, `--pool=solo` worker command) rather than an actual `docker compose up --build`. Please run that build once in a real Docker environment before deploying.
- **`docling` is not installed for local test runs** — it pulls in `torch` and is slow/heavy to install. `adapters/parsers/docling_pdf_parser.py` is tested against a hand-built fake Docling object graph (see `tests/unit/test_docling_parser.py`); re-verify against a real PDF (`tests/integration/`, marked `@pytest.mark.docling_integration`) before production use.
- **`litellm` and `elasticsearch` ARE installed and verified** in this environment (litellm is lightweight enough; the `elasticsearch` client is pinned to the 8.x line to match the Elasticsearch 8.x requirement — installing it unpinned would pull the 9.x client, which targets ES 9 and should not be used against an 8.x cluster).
- **Elasticsearch index creation is best-effort at startup**, not a hard requirement to boot: `bootstrap.ensure_vector_store_ready()` logs and swallows failures so the API/worker processes stay up (and `GET /health/live` reachable) even if Elasticsearch isn't up yet. `GET /health/ready` and individual ingestion jobs will surface the problem instead.
- **Celery runs with `--pool=solo`** in `docker-compose.yml` deliberately: the worker builds one long-lived `AppContainer` (Redis/Elasticsearch clients) at process start; Celery's default `prefork` pool forks worker processes after that, which can corrupt already-open sockets. Multi-process worker scaling (via `worker_process_init`, rebuilding adapters post-fork) is a production TODO, not implemented here.
- **Job status lives entirely in Redis** via the `JobRepository` port — never read back through Celery's own `AsyncResult`. This keeps job status meaningful even if the task queue implementation changes.

## Running tests

```bash
py -m venv .venv
.venv\Scripts\pip install fastapi uvicorn python-multipart pydantic pydantic-settings celery redis elasticsearch litellm tiktoken python-json-logger httpx pytest pytest-mock
.venv\Scripts\pytest -q
```

This is the fast local path — 72 tests, no live Redis/Elasticsearch/Docling required (every adapter is tested against a hand-built fake or a real-but-network-idle client instance with its methods monkeypatched). It intentionally does **not** install `docling` from `requirements.txt` (heavy, pulls `torch`).

For the full dependency set (as Docker actually installs) plus integration tests against real infrastructure:

```bash
.venv\Scripts\pip install -r requirements.txt   # includes docling — slow, several GB
docker compose up -d redis   # + a real Elasticsearch cluster
pytest -m "integration or docling_integration"
```

See [`tests/integration/README.md`](tests/integration/README.md) for what those tests need.

## Extension points

Every adapter is swappable behind its port without touching `application/`, `api/`, or `domain/`:

| To add... | Do this |
|---|---|
| A new file format parser (e.g. DOCX) | Implement `ports/parser.py::DocumentParser` in a new `adapters/parsers/` module; wire it into `bootstrap.py`. |
| A new chunking strategy | Implement `ports/chunker.py::Chunker` in `adapters/chunkers/`, add a `ChunkingStrategy` enum value, register it in `adapters/chunkers/registry.py`. |
| A different embedding provider (bypassing LiteLLM) | Implement `ports/embedder.py::Embedder` directly. |
| A different vector store (e.g. pgvector, Qdrant) | Implement `ports/vector_store_writer.py::VectorStoreWriter`; the three-index split (documents/chunks/embeddings) and delete-then-insert idempotency rule are conventions worth keeping, not requirements of the port itself. |
| A different job-status backend | Implement `ports/job_repository.py::JobRepository`. |

## What remains TODO for production

- Authentication/authorization (none in this MVP).
- Object storage for uploaded PDFs (currently local temp disk only, cleaned up after processing; no durability across a crash between upload and enqueue beyond a best-effort cleanup).
- Multi-process Celery worker scaling (`--pool=solo` is single-process by design here).
- Multi-embedding-model support (one model's dimensions are baked into the embeddings index at creation time).
- A periodic sweep for orphaned temp files (if the API process crashes between writing a file and enqueueing its task).
- Kubernetes manifests.
- Real verification of `docling_pdf_parser.py` against the actual pinned Docling version (developed and tested against a hand-built fake — see Development notes).
- An actual `docker compose up --build` run (this environment has no Docker installed).
- Retry-tuning and dead-letter handling for Celery tasks beyond the basic `autoretry_for`/backoff already wired up.
- **No locking against concurrent ingestion of the same `document_id`.** `_embed_and_index`'s delete-then-insert sequence (`delete_document` → three `upsert_*` calls) is not atomic; two overlapping jobs that resolve to the same `document_id` (e.g. a client retry racing the original request for the same `external_id`) can interleave and corrupt or lose data — one job's `delete_document` can run after the other has already written its chunks/embeddings. The shipped `docker-compose.yml` happens to serialize this away today (`--pool=solo`, one worker process), but that's a deployment detail, not a code-level guarantee; a real fix needs a distributed lock (e.g. a Redis-backed lock keyed on `document_id`) before scaling to multiple worker processes.
- **No compensating rollback if `_embed_and_index`'s indexing step fails after `delete_document` already succeeded.** A previously-indexed document's records are gone before the new ones are confirmed written; a deterministic write failure (caught as `PermanentIngestionError`, not retried — see `ingestion_pipeline.py`) still leaves nothing in their place. A full fix needs either a two-phase write (write new records under a staging suffix, then atomically swap) or accepting eventual reconciliation via re-ingestion.

## What remains TODO for multimodal / ColPali

Forward-compatible fields already exist so this is additive, not a rewrite:

- `Modality` already includes `table`/`image`/`page_image`/`mixed` (only `text`/`table` are actually produced today).
- `EmbeddingRecord.owner_type` already includes `page`/`image`/`table` (only `chunk` is used today).
- `EmbeddingRecord.multi_vector: list[list[float]] | None` is reserved for ColPali-style multi-vector page/image embeddings — present in the schema, unused, and not mapped specially in the Elasticsearch index yet.
- Docling's picture/table elements are already parsed into `ParsedElement`s with the right `element_type` — a future image embedder just needs to read from there.
- Not implemented: any actual ColPali model integration, page-image rendering/storage, or multi-vector Elasticsearch querying (`rag_embeddings_v1`'s mapping would need a `nested`/multi-vector field activated).
