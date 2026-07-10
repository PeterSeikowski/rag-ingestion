"""Chunker behavior: section grouping, parent/child linkage, table
atomicity, the no-heading fallback, and the strategy registry.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rag_ingestion.adapters.chunkers.recursive_text import RecursiveTextChunker
from rag_ingestion.adapters.chunkers.registry import build_chunker_registry
from rag_ingestion.adapters.chunkers.section_based_hierarchical import SectionBasedHierarchicalChunker
from rag_ingestion.domain.chunks import ChunkingConfig
from rag_ingestion.domain.documents import ParsedDocument, ParsedElement
from rag_ingestion.domain.enums import ChunkingStrategy, ChunkLevel, ElementType, Modality
from rag_ingestion.domain.provenance import SourceReference

LONG_PARAGRAPH = " ".join(f"word{i}" for i in range(2000))


def _element(element_id, order_index, element_type, text, section_path, section_title, page):
    return ParsedElement(
        element_id=element_id,
        element_type=element_type,
        text=text,
        order_index=order_index,
        source=SourceReference(
            document_id="doc-1",
            element_id=element_id,
            page_numbers=[page],
            section_path=section_path,
            section_title=section_title,
        ),
    )


@pytest.fixture
def parsed_document_with_sections() -> ParsedDocument:
    elements = [
        _element("e0", 0, ElementType.TITLE, "My Report", ["My Report"], "My Report", 1),
        _element("e1", 1, ElementType.HEADING, "Introduction", ["My Report", "Introduction"], "Introduction", 1),
        _element("e2", 2, ElementType.TEXT, LONG_PARAGRAPH, ["My Report", "Introduction"], "Introduction", 1),
        _element(
            "e3", 3, ElementType.TABLE, "| a | b |\n|---|---|\n| 1 | 2 |",
            ["My Report", "Introduction"], "Introduction", 2,
        ),
        _element("e4", 4, ElementType.HEADING, "Conclusion", ["My Report", "Conclusion"], "Conclusion", 3),
        _element("e5", 5, ElementType.TEXT, "Short concluding remarks.", ["My Report", "Conclusion"], "Conclusion", 3),
    ]
    return ParsedDocument(
        document_id="doc-1",
        source_filename="test.pdf",
        page_count=3,
        elements=elements,
        parser_name="docling_pdf",
        parser_version="1.0",
        parsed_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def config() -> ChunkingConfig:
    return ChunkingConfig(chunk_size_tokens=256, chunk_overlap_tokens=32, min_chunk_size_tokens=16)


class TestSectionBasedHierarchicalChunker:
    def test_produces_one_parent_per_section(self, parsed_document_with_sections, config):
        chunks = SectionBasedHierarchicalChunker().chunk(parsed_document_with_sections, config)
        parents = [c for c in chunks if c.chunk_level == ChunkLevel.PARENT]
        # The standalone TITLE element forms its own micro-section (it
        # pushes itself onto the heading stack before section_path is
        # computed, per docling_pdf_parser's convention) alongside
        # Introduction and Conclusion.
        assert len(parents) == 3

    def test_long_section_produces_multiple_children(self, parsed_document_with_sections, config):
        chunks = SectionBasedHierarchicalChunker().chunk(parsed_document_with_sections, config)
        intro_parent = next(
            c for c in chunks if c.chunk_level == ChunkLevel.PARENT and c.section_title == "Introduction"
        )
        assert len(intro_parent.child_chunk_ids) > 1

    def test_short_section_produces_single_child(self, parsed_document_with_sections, config):
        chunks = SectionBasedHierarchicalChunker().chunk(parsed_document_with_sections, config)
        conclusion_parent = next(
            c for c in chunks if c.chunk_level == ChunkLevel.PARENT and c.section_title == "Conclusion"
        )
        assert len(conclusion_parent.child_chunk_ids) == 1

    def test_table_is_never_split(self, parsed_document_with_sections, config):
        chunks = SectionBasedHierarchicalChunker().chunk(parsed_document_with_sections, config)
        table_chunks = [c for c in chunks if c.modality == Modality.TABLE]
        assert len(table_chunks) == 1
        assert "| a | b |" in table_chunks[0].text

    def test_parent_child_linkage_is_bidirectional(self, parsed_document_with_sections, config):
        chunks = SectionBasedHierarchicalChunker().chunk(parsed_document_with_sections, config)
        by_id = {c.chunk_id: c for c in chunks}
        for parent in (c for c in chunks if c.chunk_level == ChunkLevel.PARENT):
            for child_id in parent.child_chunk_ids:
                assert by_id[child_id].parent_chunk_id == parent.chunk_id

    def test_every_referenced_child_id_exists_in_output(self, parsed_document_with_sections, config):
        chunks = SectionBasedHierarchicalChunker().chunk(parsed_document_with_sections, config)
        by_id = {c.chunk_id: c for c in chunks}
        for parent in (c for c in chunks if c.chunk_level == ChunkLevel.PARENT):
            for child_id in parent.child_chunk_ids:
                assert child_id in by_id

    def test_no_headings_collapses_to_one_implicit_section(self, config):
        flat_doc = ParsedDocument(
            document_id="doc-2",
            source_filename="flat.pdf",
            page_count=1,
            elements=[_element("f0", 0, ElementType.TEXT, "Just a flat document.", [], None, 1)],
            parser_name="docling_pdf",
            parser_version="1.0",
            parsed_at=datetime.now(timezone.utc),
        )
        chunks = SectionBasedHierarchicalChunker().chunk(flat_doc, config)
        parents = [c for c in chunks if c.chunk_level == ChunkLevel.PARENT]
        assert len(parents) == 1


class TestRecursiveTextChunker:
    def test_produces_only_leaf_chunks(self, parsed_document_with_sections, config):
        chunks = RecursiveTextChunker().chunk(parsed_document_with_sections, config)
        assert chunks
        assert all(c.chunk_level == ChunkLevel.LEAF for c in chunks)

    def test_table_is_never_split(self, parsed_document_with_sections, config):
        chunks = RecursiveTextChunker().chunk(parsed_document_with_sections, config)
        table_chunks = [c for c in chunks if c.modality == Modality.TABLE]
        assert len(table_chunks) == 1

    def test_long_text_produces_multiple_chunks(self, parsed_document_with_sections, config):
        chunks = RecursiveTextChunker().chunk(parsed_document_with_sections, config)
        text_chunks = [c for c in chunks if c.modality == Modality.TEXT]
        assert len(text_chunks) > 1


def test_chunker_registry_has_both_strategies():
    registry = build_chunker_registry()
    assert set(registry) == {ChunkingStrategy.SECTION_BASED_HIERARCHICAL, ChunkingStrategy.RECURSIVE_TEXT}
    assert registry[ChunkingStrategy.RECURSIVE_TEXT].name == "recursive_text"
    assert registry[ChunkingStrategy.SECTION_BASED_HIERARCHICAL].name == "section_based_hierarchical"
