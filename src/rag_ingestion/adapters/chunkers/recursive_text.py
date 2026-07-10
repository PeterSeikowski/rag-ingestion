"""RecursiveTextChunker: token-window fallback chunking strategy.

Used when section structure is weak. Produces only LEAF chunks — no
parent/child hierarchy. Provenance is best-effort: a window's source_refs
are every ParsedElement whose text it overlaps. Table elements are never
token-windowed — one chunk per table, unsplit, even if it exceeds
chunk_size_tokens.
"""

from __future__ import annotations

from datetime import datetime, timezone

from rag_ingestion.adapters.chunkers._token_windowing import (
    aggregate_source_refs,
    concatenate_with_spans,
    content_hash_for,
    elements_overlapping,
    token_count,
    window_text,
)
from rag_ingestion.domain.chunks import ChunkingConfig, ChunkRecord
from rag_ingestion.domain.documents import ParsedDocument, ParsedElement
from rag_ingestion.domain.enums import ChunkLevel, ElementType, Modality


class RecursiveTextChunker:
    """Splits a ParsedDocument's non-table text into overlapping token
    windows, independent of section structure.
    """

    name = "recursive_text"
    version = "1.0"

    def chunk(self, parsed_document: ParsedDocument, config: ChunkingConfig) -> list[ChunkRecord]:
        text_elements = [e for e in parsed_document.elements if e.element_type != ElementType.TABLE]
        table_elements = [e for e in parsed_document.elements if e.element_type == ElementType.TABLE]

        chunks = self._chunk_text(parsed_document, text_elements, config, next_index=0)
        chunks.extend(self._chunk_tables(parsed_document, table_elements, config, next_index=len(chunks)))
        return chunks

    def _chunk_text(
        self,
        parsed_document: ParsedDocument,
        elements: list[ParsedElement],
        config: ChunkingConfig,
        *,
        next_index: int,
    ) -> list[ChunkRecord]:
        text, spans = concatenate_with_spans(elements)
        windows = window_text(
            text,
            chunk_size_tokens=config.chunk_size_tokens,
            overlap_tokens=config.chunk_overlap_tokens,
            min_chunk_size_tokens=config.min_chunk_size_tokens,
            encoding_name=config.tokenizer_encoding,
        )

        now = datetime.now(timezone.utc)
        chunks: list[ChunkRecord] = []
        for offset, window in enumerate(windows):
            overlapping = elements_overlapping(spans, window.start_char, window.end_char)
            source_refs, page_start, page_end, section_path, section_title = aggregate_source_refs(overlapping)
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{parsed_document.document_id}:{self.name}:{next_index + offset}",
                    document_id=parsed_document.document_id,
                    chunk_index=next_index + offset,
                    chunk_level=ChunkLevel.LEAF,
                    text=window.text,
                    token_count=window.token_count,
                    modality=Modality.TEXT,
                    source_refs=source_refs,
                    page_start=page_start,
                    page_end=page_end,
                    section_path=section_path,
                    section_title=section_title,
                    chunker_name=self.name,
                    chunker_version=self.version,
                    content_hash=content_hash_for(window.text),
                    created_at=now,
                )
            )
        return chunks

    def _chunk_tables(
        self,
        parsed_document: ParsedDocument,
        table_elements: list[ParsedElement],
        config: ChunkingConfig,
        *,
        next_index: int,
    ) -> list[ChunkRecord]:
        now = datetime.now(timezone.utc)
        chunks: list[ChunkRecord] = []
        for offset, element in enumerate(e for e in table_elements if e.text):
            source_refs, page_start, page_end, section_path, section_title = aggregate_source_refs([element])
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{parsed_document.document_id}:{self.name}:{next_index + offset}",
                    document_id=parsed_document.document_id,
                    chunk_index=next_index + offset,
                    chunk_level=ChunkLevel.LEAF,
                    text=element.text,
                    token_count=token_count(element.text, config.tokenizer_encoding),
                    modality=Modality.TABLE,
                    source_refs=source_refs,
                    page_start=page_start,
                    page_end=page_end,
                    section_path=section_path,
                    section_title=section_title,
                    chunker_name=self.name,
                    chunker_version=self.version,
                    content_hash=content_hash_for(element.text),
                    created_at=now,
                )
            )
        return chunks
