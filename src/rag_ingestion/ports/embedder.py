"""Port for pluggable text embedders.

An Embedder turns plain text into vectors — nothing more. It has no concept
of "chunk" or "owner"; the application layer
(application/ingestion_pipeline.py) is responsible for zipping the returned
vectors with chunk/document metadata to build EmbeddingRecords. This keeps
the embedder reusable for anything that needs "text in, vectors out" —
chunks today, other multimodal text later.
"""

from __future__ import annotations

from typing import Protocol

from rag_ingestion.domain.embeddings import EmbeddingConfig


class Embedder(Protocol):
    """Embeds a batch of texts into vectors.

    `dimensions` must be resolvable without a network call (e.g. a static
    lookup table or explicit config), because the vector store adapter
    needs it at startup to create the Elasticsearch dense_vector mapping,
    before any embedding has actually happened.
    """

    name: str
    provider: str
    model: str

    @property
    def dimensions(self) -> int | None:
        ...

    def embed_texts(self, texts: list[str], config: EmbeddingConfig) -> list[list[float]]:
        """Embed `texts`, returning one vector per input in the same order.

        Implementations must raise clearly if the returned vector count
        does not match `len(texts)`, and must fail fast with a clear error
        if required configuration (e.g. an API key) is missing.
        """
        ...
