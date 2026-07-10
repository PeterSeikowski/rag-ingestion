"""Provenance models: where a piece of content came from in its source document.

Pure data — no infrastructure imports. Used by both the parsing layer
(one SourceReference per ParsedElement) and the chunking layer (a
ChunkRecord aggregates the SourceReferences of every element it draws from).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """A single bounding box on one page of a source document.

    Coordinates are in whatever unit the parser reports; ``coord_origin``
    records which corner is (0, 0) so a renderer can interpret them
    correctly. When a parser cannot supply real coordinates for an element,
    no BoundingBox is created for it — callers get an empty list rather than
    invented coordinates (see SourceReference.bounding_boxes).
    """

    page_no: int = Field(..., ge=1, description="1-indexed page number")
    left: float
    top: float
    right: float
    bottom: float
    coord_origin: Literal["TOPLEFT", "BOTTOMLEFT"] | None = None


class SourceReference(BaseModel):
    """Traceability from one parsed element (or a chunk's aggregate) back to
    its position in the source document.
    """

    document_id: str
    element_id: str | None = None
    page_numbers: list[int] = Field(
        default_factory=list, description="Pages this reference touches, in ascending order"
    )
    bounding_boxes: list[BoundingBox] = Field(
        default_factory=list, description="Empty when the parser could not supply real coordinates"
    )
    section_path: list[str] = Field(
        default_factory=list, description="Heading breadcrumb, e.g. ['Chapter 2', '2.1 Methods']"
    )
    section_title: str | None = None
