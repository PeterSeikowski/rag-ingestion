"""Chunk-level domain models and the chunking configuration contract.

Pure data — no infrastructure imports. A Chunker (ports/chunker.py)
consumes a ParsedDocument and a ChunkingConfig and produces ChunkRecords;
it never touches an embedding vector or an Elasticsearch client.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .enums import ChunkingStrategy, ChunkLevel, Modality
from .provenance import SourceReference


class ChunkingConfig(BaseModel):
    """Configuration for a single chunking run.

    `chunk_size_tokens`/`chunk_overlap_tokens`'s field defaults (512/64)
    are a fallback of last resort: application/ingestion_pipeline.py
    always builds a config from Settings.DEFAULT_CHUNK_SIZE_TOKENS /
    DEFAULT_CHUNK_OVERLAP_TOKENS when a caller omits chunking_config
    entirely, so these field defaults are only actually reached when a
    caller supplies a *partial* chunking_config JSON that omits just
    those two keys — if you change the deployment's env-var defaults,
    partial client overrides of unrelated fields won't pick that change up
    for chunk_size/overlap.
    """

    strategy: ChunkingStrategy = ChunkingStrategy.SECTION_BASED_HIERARCHICAL
    chunk_size_tokens: int = Field(default=512, gt=0)
    chunk_overlap_tokens: int = Field(default=64, ge=0)
    min_chunk_size_tokens: int = Field(
        default=32, ge=0, description="Trailing windows smaller than this are merged into the previous window"
    )
    embed_parent_chunks: bool = Field(
        default=False,
        description="Parent/section chunks exist for retrieval-time context expansion and are not embedded by default",
    )
    tokenizer_encoding: str = "cl100k_base"

    @model_validator(mode="after")
    def _check_overlap_smaller_than_size(self) -> "ChunkingConfig":
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError(
                f"chunk_overlap_tokens ({self.chunk_overlap_tokens}) must be smaller than "
                f"chunk_size_tokens ({self.chunk_size_tokens}) — an overlap this large (or larger) "
                "degrades to a near-duplicate window per token instead of meaningful chunking"
            )
        return self


class ChunkRecord(BaseModel):
    """One retrievable unit of text, stored in the `rag_chunks_v1`
    Elasticsearch index. Never holds a vector — see EmbeddingRecord in
    embeddings.py, which references a ChunkRecord by chunk_id instead.
    """

    chunk_id: str
    document_id: str
    chunk_index: int = Field(description="Position among chunks produced for this document by this chunker run")
    chunk_level: ChunkLevel
    parent_chunk_id: str | None = None
    child_chunk_ids: list[str] = Field(default_factory=list)
    text: str
    token_count: int = Field(ge=0)
    modality: Modality = Modality.TEXT
    source_refs: list[SourceReference] = Field(
        default_factory=list, description="Provenance of every element this chunk draws from"
    )
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)
    section_title: str | None = None
    chunker_name: str
    chunker_version: str
    content_hash: str
    schema_version: str = "1.0"
    tenant_id: str | None = None
    allowed_groups: list[str] = Field(default_factory=list)
    classification: str | None = None
    custom_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
