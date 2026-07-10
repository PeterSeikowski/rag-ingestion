"""Port for pluggable chunking strategies.

A Chunker consumes a canonical ParsedDocument and produces ChunkRecords. It
never sees a parser-specific object — only the domain-level ParsedDocument.
Concrete implementations live under adapters/chunkers/; new strategies
register in adapters/chunkers/registry.py plus a new
domain.enums.ChunkingStrategy value — nowhere else needs to change.
"""

from __future__ import annotations

from typing import Protocol

from rag_ingestion.domain.chunks import ChunkingConfig, ChunkRecord
from rag_ingestion.domain.documents import ParsedDocument


class Chunker(Protocol):
    """Turns a ParsedDocument into a list of ChunkRecords per `config`."""

    name: str
    version: str

    def chunk(self, parsed_document: ParsedDocument, config: ChunkingConfig) -> list[ChunkRecord]:
        ...
