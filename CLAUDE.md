# CLAUDE.md

Guidance for AI coding agents working in this repository. See `IMPLEMENTATION_PLAN.md` for full architecture rationale.

## Architecture — read before editing

This is a strict hexagonal / clean architecture service. Layers under `src/rag_ingestion/`:

- `domain/` — pure Pydantic models and enums. **Zero imports** of `fastapi`, `elasticsearch`, `celery`, `docling`, `litellm`, or anything else infrastructure-flavored.
- `ports/` — `typing.Protocol` interfaces only. No concrete logic.
- `adapters/` — the *only* place concrete infra libraries are imported. Docling only in `adapters/parsers/docling_pdf_parser.py`. Elasticsearch only in `adapters/vectorstores/elasticsearch_writer.py`. LiteLLM only in `adapters/embedders/litellm_embedder.py`.
- `application/` — orchestration. Talks to `ports/` (Protocols), never to a concrete adapter class or to Elasticsearch DSL directly.
- `api/` — FastAPI routes + request/response schemas only. No business logic, no direct calls to Elasticsearch/Docling/embedding libraries.
- `workers/` — Celery app + tasks. Task bodies call `application/` services; they do not contain parsing/chunking/embedding/indexing logic themselves.
- `bootstrap.py` — the single composition root that wires `ports/` to concrete `adapters/` from `Settings`. Both `api/main.py` (via FastAPI `lifespan`) and `workers/celery_app.py` build their adapters through this one place.

**Hard rule**: if you're about to import an infra library into `domain/` or `application/`, stop — the port is missing something, fix the port instead.

**Hard rule**: chunking is independent from parsing. The parser produces a canonical `ParsedDocument`; the chunker only ever consumes `ParsedDocument`, never a Docling object.

**Hard rule**: embeddings are never stored inside `ChunkRecord`. They're separate `EmbeddingRecord`s in a separate Elasticsearch index (`rag_embeddings_v1`), referencing their owner by ID.

## Conventions

- Type hints everywhere. Domain/port/adapter public classes and non-trivial application services get a docstring explaining *why*, not a restatement of the signature.
- No comments that just restate the code. A comment is for a non-obvious constraint, invariant, or workaround.
- Plain `pip` + `requirements.txt`. No Poetry/uv. No Ruff/pre-commit config.
- New chunking/parsing/embedding/vector-store strategies are added as new adapter files + a `Protocol`-conforming class, never by branching inside an existing adapter.

## Commands

```bash
# Create a venv and install deps (full set, including Docling — heavy, pulls torch)
py -m venv .venv
.venv/Scripts/pip install -r requirements.txt

# Run unit tests (see README for the lightweight-dependency fast path)
.venv/Scripts/pytest tests/unit -v

# Run the full stack (API + worker + Redis; Elasticsearch must already be running externally)
docker compose up --build
```

## Where things live

| I need to... | Look in |
|---|---|
| Add a field to a domain record | `domain/*.py` — check `IMPLEMENTATION_PLAN.md` section 4 for the file mapping |
| Add a new chunking strategy | `adapters/chunkers/`, register it in `adapters/chunkers/registry.py`, add the enum value to `domain/enums.py::ChunkingStrategy` |
| Add a new parser (new file format) | `adapters/parsers/`, implement `ports/parser.py::DocumentParser` |
| Change Elasticsearch mappings | `adapters/vectorstores/elasticsearch_writer.py` only |
| Change job state machine | `domain/jobs.py` (shape) + `application/job_service.py` (transitions) + `adapters/jobs/redis_job_repository.py` (storage) |
| Add a Celery task | `workers/tasks.py` — keep it a thin wrapper around an `application/` service call |
