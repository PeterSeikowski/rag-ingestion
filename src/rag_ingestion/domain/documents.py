"""Document-level domain models: canonical parser output, top-level document
records, and the input shape for metadata-only ingestion.

Pure data — no infrastructure imports. ParsedDocument/ParsedElement are the
canonical, parser-agnostic representation every DocumentParser adapter must
produce; the chunking layer never sees a parser-specific object (e.g. a
Docling type), only these.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from .chunks import ChunkingConfig
from .enums import ElementType, SourceFormat
from .provenance import SourceReference


class ParsedElement(BaseModel):
    """One atomic unit of content extracted by a DocumentParser (a
    paragraph, heading, table, or picture), in document order.
    """

    element_id: str
    element_type: ElementType
    text: str = Field(description="Raw or markdown-rendered text; markdown for tables")
    heading_level: int | None = Field(default=None, description="Set only for HEADING/TITLE elements")
    order_index: int = Field(description="Position in document order; used to rebuild section paths")
    source: SourceReference
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Parser-internal passthrough, e.g. a label confidence score"
    )


class ParsedDocument(BaseModel):
    """Canonical, parser-agnostic output of a DocumentParser. The chunking
    layer consumes only this — never a parser-specific object.
    """

    document_id: str
    source_filename: str
    page_count: int
    elements: list[ParsedElement] = Field(default_factory=list)
    parser_name: str
    parser_version: str | None = None
    parsed_at: datetime
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Document-level parser metadata, e.g. detected language or title"
    )


class DocumentRecord(BaseModel):
    """Top-level record for one ingested document (PDF or metadata-only),
    stored in the `rag_documents_v1` Elasticsearch index.
    """

    document_id: str
    external_id: str | None = Field(
        default=None, description="Client-supplied stable ID; enables idempotent re-ingestion of the same document"
    )
    source_format: SourceFormat
    filename: str | None = None
    title: str | None = None
    page_count: int | None = None
    chunk_count: int | None = None
    content_hash: str | None = None
    tenant_id: str | None = None
    allowed_groups: list[str] = Field(default_factory=list)
    classification: str | None = None
    custom_metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "1.0"
    ingested_at: datetime
    updated_at: datetime


class MetadataIngestionRecord(BaseModel):
    """Input payload for POST /v1/records/metadata: one metadata-only record
    that becomes exactly one retrievable chunk-like unit (no PDF, no parser
    involved).
    """

    external_id: str | None = None
    title: str | None = None
    text_content: str = Field(description="The content to chunk and embed")
    tenant_id: str | None = None
    allowed_groups: list[str] = Field(default_factory=list)
    classification: str | None = None
    custom_metadata: dict[str, Any] = Field(default_factory=dict)
    chunking_config: ChunkingConfig | None = None
