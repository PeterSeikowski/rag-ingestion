"""Chunker strategy registry.

Maps ChunkingStrategy -> Chunker instance. New strategies register here;
nothing else in the codebase needs to change to add one (see
IMPLEMENTATION_PLAN.md decision 3).
"""

from __future__ import annotations

from rag_ingestion.adapters.chunkers.recursive_text import RecursiveTextChunker
from rag_ingestion.adapters.chunkers.section_based_hierarchical import SectionBasedHierarchicalChunker
from rag_ingestion.domain.enums import ChunkingStrategy
from rag_ingestion.ports.chunker import Chunker


def build_chunker_registry() -> dict[ChunkingStrategy, Chunker]:
    """Construct one instance of every registered chunker. Called once by
    bootstrap.build_container(); application.ingestion_pipeline looks up
    the right one per-request by ChunkingConfig.strategy.
    """
    return {
        ChunkingStrategy.SECTION_BASED_HIERARCHICAL: SectionBasedHierarchicalChunker(),
        ChunkingStrategy.RECURSIVE_TEXT: RecursiveTextChunker(),
    }
