"""Elasticsearch-based VectorStoreWriter adapter.

The only file allowed to import `elasticsearch`. Owns all three indices
(rag_documents_v1, rag_chunks_v1, rag_embeddings_v1) and every piece of
Elasticsearch-specific mapping/query DSL — nothing outside this module
knows an index name, a mapping shape, or a query body.

Elasticsearch's `dense_vector` field type requires a fixed `dims` at
index-creation time, so `embedding_dimensions` must be known when this
adapter is constructed (resolved from the configured Embedder — see
bootstrap.py and IMPLEMENTATION_PLAN.md decision 10). Changing the
embedding model later requires a new embeddings index and a full
re-ingestion; this adapter does not attempt to migrate an existing
dense_vector mapping.
"""

from __future__ import annotations

import logging
from typing import Any

from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch.helpers import bulk

from rag_ingestion.domain.chunks import ChunkRecord
from rag_ingestion.domain.documents import DocumentRecord
from rag_ingestion.domain.embeddings import EmbeddingRecord

logger = logging.getLogger(__name__)


class ElasticsearchVectorStoreWriter:
    """VectorStoreWriter adapter backed by Elasticsearch 8.x.

    Elasticsearch itself is external — this adapter only ever connects to
    an already-running cluster via `elasticsearch_url`; nothing in this
    codebase starts one (see docker-compose.yml).
    """

    def __init__(
        self,
        *,
        elasticsearch_url: str,
        documents_index: str,
        chunks_index: str,
        embeddings_index: str,
        embedding_dimensions: int,
        username: str | None = None,
        password: str | None = None,
        verify_certs: bool = True,
    ) -> None:
        self._documents_index = documents_index
        self._chunks_index = chunks_index
        self._embeddings_index = embeddings_index
        self._embedding_dimensions = embedding_dimensions

        basic_auth = (username, password) if username and password else None
        self._client = Elasticsearch(hosts=[elasticsearch_url], basic_auth=basic_auth, verify_certs=verify_certs)

    def ensure_indices(self) -> None:
        """Idempotently create all three indices if they don't already
        exist. Safe to call on every process startup.
        """
        self._ensure_index(self._documents_index, _DOCUMENTS_MAPPING)
        self._ensure_index(self._chunks_index, _CHUNKS_MAPPING)
        self._ensure_index(self._embeddings_index, _embeddings_mapping(self._embedding_dimensions))

    def _ensure_index(self, index_name: str, mapping: dict[str, Any]) -> None:
        if self._client.indices.exists(index=index_name):
            return
        logger.info("Creating Elasticsearch index %s", index_name)
        self._client.indices.create(index=index_name, mappings=mapping)

    def upsert_documents(self, documents: list[DocumentRecord]) -> None:
        self._bulk_upsert(self._documents_index, documents, id_field="document_id")

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        self._bulk_upsert(self._chunks_index, chunks, id_field="chunk_id")

    def upsert_embeddings(self, embeddings: list[EmbeddingRecord]) -> None:
        self._bulk_upsert(self._embeddings_index, embeddings, id_field="embedding_id")

    def _bulk_upsert(self, index_name: str, records: list[Any], *, id_field: str) -> None:
        """Deterministic-ID bulk upsert: re-writing a record with the same
        ID overwrites it. This alone is not sufficient for idempotent
        re-ingestion when the *set* of IDs changes between runs (e.g. a
        different chunk count) — see delete_document(), which the ingestion
        pipeline calls first for that reason.
        """
        if not records:
            return
        actions = (
            {"_index": index_name, "_id": getattr(record, id_field), "_source": record.model_dump(mode="json")}
            for record in records
        )
        success_count, errors = bulk(self._client, actions, raise_on_error=False)
        if errors:
            raise RuntimeError(
                f"Elasticsearch bulk upsert into {index_name!r} had {len(errors)} failures: {errors[:3]}"
            )
        logger.info("Upserted %s records into %s", success_count, index_name)

    def delete_document(self, document_id: str) -> None:
        """Delete every document/chunk/embedding record for `document_id`.

        Called by the ingestion pipeline before writing a re-ingested
        document's new records, so a change in chunking parameters can't
        leave orphaned chunks from a previous run behind (see
        IMPLEMENTATION_PLAN.md decision 7).
        """
        for index_name in (self._chunks_index, self._embeddings_index):
            self._client.delete_by_query(
                index=index_name, query={"term": {"document_id": document_id}}, conflicts="proceed"
            )
        try:
            self._client.delete(index=self._documents_index, id=document_id)
        except NotFoundError:
            pass

    def ping(self, timeout_seconds: float = 2.0) -> bool:
        try:
            return bool(self._client.ping(request_timeout=timeout_seconds))
        except Exception:
            # Any connectivity failure means "not ready" for
            # GET /health/ready — never let it raise into the caller.
            return False


_DOCUMENTS_MAPPING: dict[str, Any] = {
    "properties": {
        "document_id": {"type": "keyword"},
        "external_id": {"type": "keyword"},
        "source_format": {"type": "keyword"},
        "filename": {"type": "keyword"},
        "title": {"type": "text"},
        "page_count": {"type": "integer"},
        "chunk_count": {"type": "integer"},
        "content_hash": {"type": "keyword"},
        "tenant_id": {"type": "keyword"},
        "allowed_groups": {"type": "keyword"},
        "classification": {"type": "keyword"},
        "custom_metadata": {"type": "object", "enabled": True},
        "schema_version": {"type": "keyword"},
        "ingested_at": {"type": "date"},
        "updated_at": {"type": "date"},
    }
}

_CHUNKS_MAPPING: dict[str, Any] = {
    "properties": {
        "chunk_id": {"type": "keyword"},
        "document_id": {"type": "keyword"},
        "chunk_index": {"type": "integer"},
        "chunk_level": {"type": "keyword"},
        "parent_chunk_id": {"type": "keyword"},
        "child_chunk_ids": {"type": "keyword"},
        "text": {"type": "text"},
        "token_count": {"type": "integer"},
        "modality": {"type": "keyword"},
        # Kept as a plain (non-nested) object array for v1: we don't yet
        # need independent per-element queries into source_refs, only
        # retrieval + display. Revisit as "nested" if that changes.
        "source_refs": {"type": "object", "enabled": True},
        "page_start": {"type": "integer"},
        "page_end": {"type": "integer"},
        "section_path": {"type": "keyword"},
        "section_title": {"type": "keyword"},
        "chunker_name": {"type": "keyword"},
        "chunker_version": {"type": "keyword"},
        "content_hash": {"type": "keyword"},
        "schema_version": {"type": "keyword"},
        "tenant_id": {"type": "keyword"},
        "allowed_groups": {"type": "keyword"},
        "classification": {"type": "keyword"},
        "custom_metadata": {"type": "object", "enabled": True},
        "created_at": {"type": "date"},
    }
}


def _embeddings_mapping(dimensions: int) -> dict[str, Any]:
    return {
        "properties": {
            "embedding_id": {"type": "keyword"},
            "owner_id": {"type": "keyword"},
            "owner_type": {"type": "keyword"},
            "document_id": {"type": "keyword"},
            "modality": {"type": "keyword"},
            "model_name": {"type": "keyword"},
            "provider": {"type": "keyword"},
            "dimensions": {"type": "integer"},
            "vector": {"type": "dense_vector", "dims": dimensions, "index": True, "similarity": "cosine"},
            # multi_vector (ColPali-style) is intentionally left dynamically
            # mapped rather than given an explicit mapping — not
            # implemented in v1 (see IMPLEMENTATION_PLAN.md section 11).
            "schema_version": {"type": "keyword"},
            "created_at": {"type": "date"},
        }
    }
