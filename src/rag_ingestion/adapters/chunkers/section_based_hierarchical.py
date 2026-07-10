"""SectionBasedHierarchicalChunker: default PDF chunking strategy.

Builds one PARENT ChunkRecord per detected section (the section's full
concatenated text, aggregated provenance) and multiple CHILD ChunkRecords
under each parent (the section text split into token windows via the
shared _token_windowing helper, plus one CHILD per table in the section),
wiring parent_chunk_id/child_chunk_ids both ways. A document with no
headings produces exactly one implicit section covering everything —
callers never need to pre-detect "weak section structure" themselves.

Parent (section) chunks are not embedded by default
(ChunkingConfig.embed_parent_chunks=False); they exist for small-to-big
retrieval-time context expansion. Tables are never token-windowed.
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


class SectionBasedHierarchicalChunker:
    """Groups ParsedElements into sections by their shared section_path,
    producing a parent chunk per section plus token-windowed child chunks.
    """

    name = "section_based_hierarchical"
    version = "1.0"

    def chunk(self, parsed_document: ParsedDocument, config: ChunkingConfig) -> list[ChunkRecord]:
        now = datetime.now(timezone.utc)
        chunks: list[ChunkRecord] = []
        chunk_index = 0

        for section_path, section_elements in _group_by_section(parsed_document.elements):
            text_elements = [e for e in section_elements if e.element_type != ElementType.TABLE]
            table_elements = [e for e in section_elements if e.element_type == ElementType.TABLE and e.text]

            section_text, spans = concatenate_with_spans(text_elements)
            if not section_text.strip() and not table_elements:
                continue

            source_refs, page_start, page_end, _path, section_title = aggregate_source_refs(section_elements)

            parent_id = f"{parsed_document.document_id}:{self.name}:{chunk_index}"
            parent_index = chunk_index
            chunk_index += 1

            child_ids: list[str] = []
            child_records: list[ChunkRecord] = []

            windows = window_text(
                section_text,
                chunk_size_tokens=config.chunk_size_tokens,
                overlap_tokens=config.chunk_overlap_tokens,
                min_chunk_size_tokens=config.min_chunk_size_tokens,
                encoding_name=config.tokenizer_encoding,
            )
            for window in windows:
                overlapping = elements_overlapping(spans, window.start_char, window.end_char)
                child_source_refs, child_page_start, child_page_end, _cp, _ct = aggregate_source_refs(overlapping)
                child_id = f"{parsed_document.document_id}:{self.name}:{chunk_index}"
                child_ids.append(child_id)
                child_records.append(
                    ChunkRecord(
                        chunk_id=child_id,
                        document_id=parsed_document.document_id,
                        chunk_index=chunk_index,
                        chunk_level=ChunkLevel.CHILD,
                        parent_chunk_id=parent_id,
                        text=window.text,
                        token_count=window.token_count,
                        modality=Modality.TEXT,
                        source_refs=child_source_refs,
                        page_start=child_page_start,
                        page_end=child_page_end,
                        section_path=section_path,
                        section_title=section_title,
                        chunker_name=self.name,
                        chunker_version=self.version,
                        content_hash=content_hash_for(window.text),
                        created_at=now,
                    )
                )
                chunk_index += 1

            for element in table_elements:
                child_id = f"{parsed_document.document_id}:{self.name}:{chunk_index}"
                child_ids.append(child_id)
                element_source_refs, element_page_start, element_page_end, _ep, _et = aggregate_source_refs(
                    [element]
                )
                child_records.append(
                    ChunkRecord(
                        chunk_id=child_id,
                        document_id=parsed_document.document_id,
                        chunk_index=chunk_index,
                        chunk_level=ChunkLevel.CHILD,
                        parent_chunk_id=parent_id,
                        text=element.text,
                        token_count=token_count(element.text, config.tokenizer_encoding),
                        modality=Modality.TABLE,
                        source_refs=element_source_refs,
                        page_start=element_page_start,
                        page_end=element_page_end,
                        section_path=section_path,
                        section_title=section_title,
                        chunker_name=self.name,
                        chunker_version=self.version,
                        content_hash=content_hash_for(element.text),
                        created_at=now,
                    )
                )
                chunk_index += 1

            if not child_records:
                # Every element in this section was blank/unusable; don't
                # emit a parent chunk with no children under it.
                chunk_index = parent_index
                continue

            chunks.append(
                ChunkRecord(
                    chunk_id=parent_id,
                    document_id=parsed_document.document_id,
                    chunk_index=parent_index,
                    chunk_level=ChunkLevel.PARENT,
                    child_chunk_ids=child_ids,
                    text=section_text,
                    token_count=token_count(section_text, config.tokenizer_encoding),
                    modality=Modality.TEXT,
                    source_refs=source_refs,
                    page_start=page_start,
                    page_end=page_end,
                    section_path=section_path,
                    section_title=section_title,
                    chunker_name=self.name,
                    chunker_version=self.version,
                    content_hash=content_hash_for(section_text),
                    created_at=now,
                )
            )
            chunks.extend(child_records)

        return chunks


def _group_by_section(elements: list[ParsedElement]) -> list[tuple[list[str], list[ParsedElement]]]:
    """Group consecutive elements sharing the same section_path into one
    section. A document with no headings produces exactly one section
    (every element shares an empty section_path).
    """
    sections: list[tuple[list[str], list[ParsedElement]]] = []
    for element in elements:
        path = element.source.section_path
        if sections and sections[-1][0] == path:
            sections[-1][1].append(element)
        else:
            sections.append((path, [element]))
    return sections
