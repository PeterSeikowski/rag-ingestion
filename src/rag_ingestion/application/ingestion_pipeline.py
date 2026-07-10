"""Ingestion pipeline orchestration: parse -> chunk -> embed -> index.

This is where the four adapter ports (DocumentParser, Chunker, Embedder,
VectorStoreWriter) get called in sequence, via `bootstrap.AppContainer`
(whose fields are all port-typed — this module never imports a concrete
adapter class, only the container that holds them). It never touches
Elasticsearch DSL or a Docling object directly.

Failure classification (transient vs. permanent — see
application/errors.py) happens centrally here rather than in each adapter,
so Celery's retry policy (workers/tasks.py) has one consistent place to
reason about: parsing/chunking failures are treated as permanent (a
malformed PDF won't parse differently on retry); embedding/indexing
failures are treated as transient (the embedding provider or Elasticsearch
being briefly unreachable is exactly what retries are for).
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

import tiktoken

from rag_ingestion.application.errors import IngestionError, PermanentIngestionError, TransientIngestionError
from rag_ingestion.application.job_service import JobService
from rag_ingestion.bootstrap import AppContainer
from rag_ingestion.domain.chunks import ChunkingConfig, ChunkRecord
from rag_ingestion.domain.documents import DocumentRecord, MetadataIngestionRecord, ParsedDocument
from rag_ingestion.domain.embeddings import EmbeddingConfig, EmbeddingRecord
from rag_ingestion.domain.enums import ChunkLevel, JobStage, Modality, OwnerType, SourceFormat
from rag_ingestion.domain.provenance import SourceReference

logger = logging.getLogger(__name__)

# Fixed, arbitrary namespace UUID for this service's uuid5 document_id
# derivation — must never change once deployed, or previously-derived
# document_ids stop matching their external_id on re-ingestion.
_EXTERNAL_ID_NAMESPACE = uuid.UUID("6f6a1e2e-6d8a-4c8b-9a7a-8f6f6a2e2e6f")

_KNOWN_METADATA_KEYS = ("external_id", "title", "tenant_id", "allowed_groups", "classification")


def run_pdf_ingestion(
    container: AppContainer,
    *,
    job_service: JobService,
    job_id: str,
    file_path: str,
    metadata: dict,
    chunking_config: ChunkingConfig | None,
) -> None:
    """Parse `file_path` with `container.parser`, chunk the result with the
    strategy selected in `chunking_config` (falling back to
    Settings-derived defaults), embed the resulting chunks with
    `container.embedder`, and write documents/chunks/embeddings via
    `container.vector_store` — deleting any prior records for the same
    document_id first (see IMPLEMENTATION_PLAN.md decision 7).
    """
    external_id, title, tenant_id, allowed_groups, classification, custom_metadata = _extract_known_fields(metadata)
    document_id = _resolve_document_id(job_service, job_id, external_id)
    job_service.set_document_id(job_id, document_id)

    job_service.mark_stage(job_id, JobStage.PARSING)
    try:
        parsed_document = container.parser.parse(
            Path(file_path), SourceReference(document_id=document_id)
        )
    except Exception as exc:
        _reraise_as(exc, PermanentIngestionError, "Parsing")

    config = chunking_config or ChunkingConfig(
        chunk_size_tokens=container.settings.default_chunk_size_tokens,
        chunk_overlap_tokens=container.settings.default_chunk_overlap_tokens,
    )

    job_service.mark_stage(job_id, JobStage.CHUNKING)
    try:
        chunker = container.chunkers[config.strategy]
        chunks = chunker.chunk(parsed_document, config)
    except Exception as exc:
        _reraise_as(exc, PermanentIngestionError, "Chunking")

    # The Chunker port has no notion of tenant/ACL/custom metadata (it only
    # sees the parsed document + chunking config), so those fields never
    # make it onto a ChunkRecord from chunker.chunk() itself — back-fill
    # them here from the same request metadata DocumentRecord gets, so
    # every chunk of a document carries the same ACL fields as its parent.
    chunks = [
        chunk.model_copy(
            update={
                "tenant_id": tenant_id,
                "allowed_groups": allowed_groups,
                "classification": classification,
                "custom_metadata": custom_metadata,
            }
        )
        for chunk in chunks
    ]

    document_record = DocumentRecord(
        document_id=document_id,
        external_id=external_id,
        source_format=SourceFormat.PDF,
        filename=Path(file_path).name,
        title=title,
        page_count=parsed_document.page_count,
        chunk_count=len(chunks),
        content_hash=_content_hash(_full_text(parsed_document)),
        tenant_id=tenant_id,
        allowed_groups=allowed_groups,
        classification=classification,
        custom_metadata=custom_metadata,
        ingested_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    _embed_and_index(
        container,
        job_service=job_service,
        job_id=job_id,
        document_id=document_id,
        document_record=document_record,
        chunks=chunks,
        config=config,
        total_pages=parsed_document.page_count,
    )


def run_metadata_ingestion(
    container: AppContainer,
    *,
    job_service: JobService,
    job_id: str,
    record: MetadataIngestionRecord,
) -> None:
    """Convert `record.text_content` into exactly one ChunkRecord (no
    parser, no multi-window splitting — a metadata record is always one
    retrievable unit, per the spec for POST /v1/records/metadata), embed
    it, and write document/chunk/embedding via `container.vector_store`.
    """
    document_id = _resolve_document_id(job_service, job_id, record.external_id)
    job_service.set_document_id(job_id, document_id)

    job_service.mark_stage(job_id, JobStage.CHUNKING)
    now = datetime.now(timezone.utc)
    text = record.text_content
    source_ref = SourceReference(document_id=document_id)
    chunk = ChunkRecord(
        chunk_id=f"{document_id}:metadata:0",
        document_id=document_id,
        chunk_index=0,
        chunk_level=ChunkLevel.LEAF,
        text=text,
        token_count=_token_count(text),
        modality=Modality.TEXT,
        source_refs=[source_ref],
        chunker_name="metadata_direct",
        chunker_version="1.0",
        content_hash=_content_hash(text),
        tenant_id=record.tenant_id,
        allowed_groups=record.allowed_groups,
        classification=record.classification,
        custom_metadata=record.custom_metadata,
        created_at=now,
    )

    document_record = DocumentRecord(
        document_id=document_id,
        external_id=record.external_id,
        source_format=SourceFormat.METADATA,
        title=record.title,
        chunk_count=1,
        content_hash=_content_hash(text),
        tenant_id=record.tenant_id,
        allowed_groups=record.allowed_groups,
        classification=record.classification,
        custom_metadata=record.custom_metadata,
        ingested_at=now,
        updated_at=now,
    )

    config = record.chunking_config or ChunkingConfig(
        chunk_size_tokens=container.settings.default_chunk_size_tokens,
        chunk_overlap_tokens=container.settings.default_chunk_overlap_tokens,
    )

    _embed_and_index(
        container,
        job_service=job_service,
        job_id=job_id,
        document_id=document_id,
        document_record=document_record,
        chunks=[chunk],
        config=config,
        total_pages=None,
    )


def _embed_and_index(
    container: AppContainer,
    *,
    job_service: JobService,
    job_id: str,
    document_id: str,
    document_record: DocumentRecord,
    chunks: list[ChunkRecord],
    config: ChunkingConfig,
    total_pages: int | None,
) -> None:
    embeddable_chunks = [c for c in chunks if config.embed_parent_chunks or c.chunk_level != ChunkLevel.PARENT]

    job_service.mark_stage(job_id, JobStage.EMBEDDING)
    try:
        embedding_config = EmbeddingConfig(
            model=container.settings.litellm_model,
            dimensions=container.embedder.dimensions,
            batch_size=container.settings.embedding_batch_size,
        )
        vectors = container.embedder.embed_texts([c.text for c in embeddable_chunks], embedding_config)
    except Exception as exc:
        _reraise_as(exc, TransientIngestionError, "Embedding")

    now = datetime.now(timezone.utc)
    embeddings = [
        EmbeddingRecord(
            embedding_id=f"{chunk.chunk_id}:{container.embedder.name}",
            owner_id=chunk.chunk_id,
            owner_type=OwnerType.CHUNK,
            document_id=document_id,
            modality=chunk.modality,
            model_name=container.embedder.model,
            provider=container.embedder.provider,
            dimensions=len(vector),
            vector=vector,
            created_at=now,
        )
        for chunk, vector in zip(embeddable_chunks, vectors, strict=True)
    ]

    job_service.mark_stage(job_id, JobStage.INDEXING)
    try:
        # Delete-then-insert: makes re-ingestion idempotent even when the
        # chunk count changes between runs, and makes this whole function
        # safe to retry on a TransientIngestionError (see
        # IMPLEMENTATION_PLAN.md decision 7).
        container.vector_store.delete_document(document_id)
        container.vector_store.upsert_documents([document_record])
        container.vector_store.upsert_chunks(chunks)
        container.vector_store.upsert_embeddings(embeddings)
    except RuntimeError as exc:
        # ElasticsearchVectorStoreWriter raises plain RuntimeError
        # specifically for a deterministic bulk-item failure (e.g. a
        # mapping conflict) — retrying the exact same write would fail
        # identically every time, and delete_document has very likely
        # already removed this document's prior (previously-good) records
        # by this point, so blindly retrying wastes work, delays surfacing
        # a real problem, and burns through max_retries for nothing.
        # Everything else caught below (connection errors, timeouts) is
        # genuinely transient. Note: this does not roll back a delete that
        # already succeeded — see IMPLEMENTATION_PLAN.md's production TODO
        # on this for the full compensating-transaction fix.
        _reraise_as(exc, PermanentIngestionError, "Indexing")
    except Exception as exc:
        _reraise_as(exc, TransientIngestionError, "Indexing")

    job_service.mark_completed(
        job_id, total_pages=total_pages, total_chunks=len(chunks), total_embeddings=len(embeddings)
    )


def _resolve_document_id(job_service: JobService, job_id: str, external_id: str | None) -> str:
    """Resolve the document_id for this run, in priority order:

    1. Reuse the job's already-stored document_id, if this is a retry of
       an already-started job (a prior attempt already called
       set_document_id below). Without this, a Celery retry with no
       external_id would mint a fresh random document_id every attempt,
       so delete_document() in _embed_and_index would never find (and
       therefore never clean up) the previous attempt's partial writes —
       silently duplicating/orphaning data instead of safely retrying.
    2. Derive a stable id from external_id (uuid5), for idempotent
       re-ingestion across separate requests.
    3. A fresh uuid4() for a first attempt with no external_id.
    """
    existing_job = job_service.get_job(job_id)
    if existing_job and existing_job.document_id:
        return existing_job.document_id
    if external_id:
        return str(uuid.uuid5(_EXTERNAL_ID_NAMESPACE, external_id))
    return str(uuid.uuid4())


def _extract_known_fields(metadata: dict) -> tuple[str | None, str | None, str | None, list[str], str | None, dict]:
    """Pulls well-known optional keys out of the PDF endpoint's free-form
    `metadata` JSON object; whatever remains becomes DocumentRecord's
    custom_metadata verbatim.
    """
    remaining = {k: v for k, v in metadata.items() if k not in _KNOWN_METADATA_KEYS}
    return (
        metadata.get("external_id"),
        metadata.get("title"),
        metadata.get("tenant_id"),
        metadata.get("allowed_groups", []),
        metadata.get("classification"),
        remaining,
    )


def _full_text(parsed_document: ParsedDocument) -> str:
    return "\n\n".join(element.text for element in parsed_document.elements if element.text)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _token_count(text: str, encoding_name: str = "cl100k_base") -> int:
    if not text:
        return 0
    return len(tiktoken.get_encoding(encoding_name).encode(text))


def _reraise_as(exc: Exception, error_cls: type[IngestionError], stage: str) -> NoReturn:
    if isinstance(exc, IngestionError):
        raise exc
    raise error_cls(f"{stage} failed: {exc}") from exc
