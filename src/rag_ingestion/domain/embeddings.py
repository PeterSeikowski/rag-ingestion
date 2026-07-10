"""Embedding-level domain models.

Pure data — no infrastructure imports. Deliberately separate from
ChunkRecord: an EmbeddingRecord references its owner by ID rather than a
chunk holding its own vector, so the same chunk can later gain additional
embeddings (a different model, or a future ColPali-style multi-vector page
embedding) without restructuring chunk storage.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .enums import Modality, OwnerType


class EmbeddingConfig(BaseModel):
    """Configuration for a single embedding run.

    `model`/`dimensions` are sourced from Settings, not exposed as
    client-settable fields on the public API (see ports/embedder.py) —
    this MVP supports one embedding model per deployment because
    Elasticsearch's dense_vector field requires fixed dims at
    index-creation time. They still live on this config object so the
    embedder and vector store adapters share one typed contract.
    """

    model: str
    dimensions: int | None = None
    batch_size: int = Field(default=32, gt=0)
    extra_params: dict[str, Any] = Field(default_factory=dict)


class EmbeddingRecord(BaseModel):
    """One embedding vector, stored in the `rag_embeddings_v1` Elasticsearch
    index. Owns a reference to whatever it embeds (a chunk today; a page,
    image, or table under future multimodal ingestion) rather than being
    embedded inline in that owner's own record.
    """

    embedding_id: str
    owner_id: str = Field(description="chunk_id when owner_type=CHUNK; a page/image/table identifier otherwise")
    owner_type: OwnerType = OwnerType.CHUNK
    document_id: str
    modality: Modality = Modality.TEXT
    model_name: str
    provider: str | None = None
    dimensions: int = Field(gt=0)
    vector: list[float]
    multi_vector: list[list[float]] | None = Field(
        default=None,
        description="Reserved for future ColPali-style multi-vector embeddings; unused in v1",
    )
    schema_version: str = "1.0"
    created_at: datetime

    @model_validator(mode="after")
    def _check_vector_matches_dimensions(self) -> "EmbeddingRecord":
        if len(self.vector) != self.dimensions:
            raise ValueError(
                f"vector length {len(self.vector)} does not match declared dimensions {self.dimensions}"
            )
        return self
