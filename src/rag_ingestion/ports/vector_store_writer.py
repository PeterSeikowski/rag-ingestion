"""Port for the vector store write path.

A VectorStoreWriter is the only thing in this codebase allowed to know
about a specific vector database's query/index DSL. Concrete
implementations live under adapters/vectorstores/ — e.g.
ElasticsearchVectorStoreWriter. Nothing outside that adapter module may
import a vector-store-specific client library.
"""

from __future__ import annotations

from typing import Protocol

from rag_ingestion.domain.chunks import ChunkRecord
from rag_ingestion.domain.documents import DocumentRecord
from rag_ingestion.domain.embeddings import EmbeddingRecord


class VectorStoreWriter(Protocol):
    """Writes DocumentRecords, ChunkRecords, and EmbeddingRecords to the
    configured vector store, and supports deleting all records for a
    document.
    """

    def ensure_indices(self) -> None:
        """Idempotently create any indices/collections this writer needs.
        Called once at startup by both the API process and the worker.
        """
        ...

    def upsert_documents(self, documents: list[DocumentRecord]) -> None: ...

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None: ...

    def upsert_embeddings(self, embeddings: list[EmbeddingRecord]) -> None: ...

    def delete_document(self, document_id: str) -> None:
        """Delete every document/chunk/embedding record for `document_id`.

        Called by the ingestion pipeline before writing a re-ingested
        document's new records, so a change in chunking parameters can't
        leave orphaned chunks from a previous run behind.
        """
        ...

    def ping(self, timeout_seconds: float = 2.0) -> bool:
        """Best-effort connectivity check for GET /health/ready."""
        ...
