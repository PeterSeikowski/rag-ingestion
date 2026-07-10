"""Enumerations shared across the domain model.

Kept in a single module so every domain file, and every adapter, imports
valid values from one source of truth. Adding a new value here (e.g. a new
`ChunkingStrategy`) is the only domain-level change a new adapter needs.
"""

from __future__ import annotations

from enum import StrEnum


class Modality(StrEnum):
    """What kind of content a ChunkRecord or EmbeddingRecord represents.

    TABLE, IMAGE, PAGE_IMAGE and MIXED are defined now — even though v1 only
    ever produces TEXT chunks/embeddings — so future multimodal ingestion
    (including ColPali-style page/image embeddings) is additive rather than
    a schema migration.
    """

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    PAGE_IMAGE = "page_image"
    MIXED = "mixed"


class OwnerType(StrEnum):
    """What kind of thing an EmbeddingRecord's vector belongs to."""

    CHUNK = "chunk"
    PAGE = "page"
    IMAGE = "image"
    TABLE = "table"


class ElementType(StrEnum):
    """Type of a single ParsedElement produced by a DocumentParser."""

    TEXT = "text"
    HEADING = "heading"
    TITLE = "title"
    TABLE = "table"
    IMAGE = "image"
    CAPTION = "caption"
    LIST_ITEM = "list_item"
    FOOTNOTE = "footnote"
    CODE = "code"
    FORMULA = "formula"
    OTHER = "other"


class SourceFormat(StrEnum):
    """Origin format of a DocumentRecord — which ingestion endpoint created it."""

    PDF = "pdf"
    METADATA = "metadata"


class ChunkLevel(StrEnum):
    """Position of a ChunkRecord within its chunk hierarchy.

    LEAF chunks have no parent and no children (the RecursiveTextChunker
    only ever produces LEAF chunks). PARENT/CHILD are produced by
    hierarchical strategies such as SectionBasedHierarchicalChunker.
    """

    LEAF = "leaf"
    PARENT = "parent"
    CHILD = "child"


class ChunkingStrategy(StrEnum):
    """Registered chunking strategies.

    New strategies register a new value here plus an entry in
    adapters/chunkers/registry.py — nowhere else needs to change.
    """

    SECTION_BASED_HIERARCHICAL = "section_based_hierarchical"
    RECURSIVE_TEXT = "recursive_text"


class JobStatus(StrEnum):
    """Coarse-grained lifecycle state of an ingestion job."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStage(StrEnum):
    """Fine-grained progress within an IN_PROGRESS job."""

    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
