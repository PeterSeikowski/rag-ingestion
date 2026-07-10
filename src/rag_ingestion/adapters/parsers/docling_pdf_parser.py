"""Docling-based PDF parser adapter.

The only file allowed to import `docling`. Converts Docling's own object
model into the domain-level ParsedDocument/ParsedElement so the rest of the
codebase (chunkers, application layer) never sees a Docling type.

Docling's public API has shifted across releases, and this sandbox cannot
install the (large, torch-dependent) `docling` package to verify field
names against the pinned version in requirements.txt. Every Docling field
access below is wrapped defensively: a malformed/unexpected element is
logged and skipped by `_safe_extract_item` rather than aborting the whole
document, and a missing/misshapen bounding box is left out rather than
invented (see IMPLEMENTATION_PLAN.md section 7). Re-verify this mapping
against the real `docling==2.15.1` API before relying on it in production —
the `# TODO(docling-api)` markers below are exactly the spots to check.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_ingestion.domain.documents import ParsedDocument, ParsedElement
from rag_ingestion.domain.enums import ElementType
from rag_ingestion.domain.provenance import BoundingBox, SourceReference

logger = logging.getLogger(__name__)

# TODO(docling-api): confirm these label strings against docling==2.15.1's
# DocItemLabel values. Unrecognized labels fall back to ElementType.OTHER
# rather than raising, so an unexpected new label degrades gracefully.
_LABEL_TO_ELEMENT_TYPE: dict[str, ElementType] = {
    "text": ElementType.TEXT,
    "paragraph": ElementType.TEXT,
    "section_header": ElementType.HEADING,
    "title": ElementType.TITLE,
    "table": ElementType.TABLE,
    "picture": ElementType.IMAGE,
    "image": ElementType.IMAGE,
    "caption": ElementType.CAPTION,
    "list_item": ElementType.LIST_ITEM,
    "footnote": ElementType.FOOTNOTE,
    "code": ElementType.CODE,
    "formula": ElementType.FORMULA,
}

_VALID_COORD_ORIGINS = {"TOPLEFT", "BOTTOMLEFT"}


class DoclingPdfParser:
    """DocumentParser adapter backed by IBM Docling.

    Preserves page numbers, bounding boxes, and section/heading structure
    when Docling exposes them for a given element. Tables and pictures
    become ParsedElements (with markdown / caption text respectively) even
    though v1 only embeds plain text chunks, so multimodal ingestion can
    consume this same parser output later without a parser change.
    """

    name = "docling_pdf"
    version = "1.0"
    supported_formats = ["pdf"]

    def __init__(self, *, enable_ocr: bool = False) -> None:
        self._enable_ocr = enable_ocr

    def parse(self, file_path: Path, source_reference: SourceReference) -> ParsedDocument:
        # Imported lazily: importing this module never requires
        # docling/torch to be installed unless parse() actually runs.
        from docling.document_converter import DocumentConverter  # TODO(docling-api)

        converter = DocumentConverter()
        result = converter.convert(str(file_path))
        document = result.document

        elements: list[ParsedElement] = []
        heading_stack: list[tuple[int, str]] = []  # (level, title), outermost first
        order_index = 0

        for raw_item, _level in _iter_items(document):
            parsed = _safe_extract_item(
                raw_item,
                document_id=source_reference.document_id,
                order_index=order_index,
                heading_stack=heading_stack,
            )
            if parsed is None:
                continue
            elements.append(parsed)
            order_index += 1

        return ParsedDocument(
            document_id=source_reference.document_id,
            source_filename=file_path.name,
            page_count=_safe_page_count(document),
            elements=elements,
            parser_name=self.name,
            parser_version=self.version,
            parsed_at=datetime.now(timezone.utc),
            metadata=_safe_document_metadata(document),
        )


def _iter_items(document: Any):
    """Yield (item, level) pairs in document reading order.

    Prefers `document.iterate_items()` (Docling's own reading-order
    traversal); falls back to concatenating `.texts`/`.tables`/`.pictures`
    if that method isn't present on the installed Docling version — order
    is not guaranteed to match reading order in that fallback path.
    """
    if hasattr(document, "iterate_items"):
        yield from document.iterate_items()  # TODO(docling-api)
        return
    logger.warning(
        "docling document has no iterate_items(); falling back to per-type lists (reading order not guaranteed)"
    )
    for item in (
        list(getattr(document, "texts", []))
        + list(getattr(document, "tables", []))
        + list(getattr(document, "pictures", []))
    ):
        yield item, 0


def _safe_extract_item(
    raw_item: Any,
    *,
    document_id: str,
    order_index: int,
    heading_stack: list[tuple[int, str]],
) -> ParsedElement | None:
    """Convert one Docling item into a ParsedElement.

    Returns None (and logs) if the item's shape doesn't match what we
    expect — one malformed element must never abort the whole document.
    """
    try:
        label = str(getattr(raw_item, "label", "text")).lower()
        element_type = _LABEL_TO_ELEMENT_TYPE.get(label, ElementType.OTHER)

        heading_level: int | None = None
        if element_type is ElementType.HEADING:
            heading_level = int(getattr(raw_item, "level", 1))
        elif element_type is ElementType.TITLE:
            heading_level = 0

        text = _safe_item_text(raw_item, element_type)

        # Pop and push together, only when there's real heading text to
        # push: popping unconditionally on any HEADING/TITLE element (even
        # one with empty/missing text) would remove same-or-deeper-level
        # entries from the shared heading_stack without restoring
        # anything, permanently truncating section_path/section_title for
        # every element that follows for the rest of the document. An
        # empty-text heading is instead treated as structurally invisible
        # — the current section context passes through unchanged.
        if heading_level is not None and text:
            while heading_stack and heading_stack[-1][0] >= heading_level:
                heading_stack.pop()
            heading_stack.append((heading_level, text))

        section_path = [title for _lvl, title in heading_stack]
        section_title = heading_stack[-1][1] if heading_stack else None

        page_numbers, bounding_boxes = _safe_provenance(raw_item)

        element_id = str(getattr(raw_item, "self_ref", None) or f"{document_id}:element:{order_index}")

        source = SourceReference(
            document_id=document_id,
            element_id=element_id,
            page_numbers=page_numbers,
            bounding_boxes=bounding_boxes,
            section_path=section_path,
            section_title=section_title,
        )

        return ParsedElement(
            element_id=element_id,
            element_type=element_type,
            text=text,
            heading_level=heading_level if element_type is ElementType.HEADING else None,
            order_index=order_index,
            source=source,
            metadata={"docling_label": label},
        )
    except Exception:
        logger.exception("Skipping unparseable docling element at order_index=%s", order_index)
        return None


def _safe_item_text(raw_item: Any, element_type: ElementType) -> str:
    if element_type is ElementType.TABLE:
        export = getattr(raw_item, "export_to_markdown", None)  # TODO(docling-api)
        if callable(export):
            try:
                return str(export())
            except Exception:
                logger.warning("Table markdown export failed; storing empty text for this element")
                return ""
        return ""
    if element_type is ElementType.IMAGE:
        caption_fn = getattr(raw_item, "caption_text", None)  # TODO(docling-api)
        if callable(caption_fn):
            try:
                return str(caption_fn())
            except Exception:
                return ""
        return ""
    return str(getattr(raw_item, "text", "") or "")


def _safe_provenance(raw_item: Any) -> tuple[list[int], list[BoundingBox]]:
    prov_list = getattr(raw_item, "prov", None) or []
    page_numbers: list[int] = []
    bounding_boxes: list[BoundingBox] = []
    for prov in prov_list:
        page_no = getattr(prov, "page_no", None)
        if isinstance(page_no, int):
            page_numbers.append(page_no)

        bbox = getattr(prov, "bbox", None)
        if bbox is None:
            continue
        try:
            bounding_boxes.append(
                BoundingBox(
                    page_no=page_no or 1,
                    left=float(bbox.l),
                    top=float(bbox.t),
                    right=float(bbox.r),
                    bottom=float(bbox.b),
                    coord_origin=_safe_coord_origin(bbox),
                )
            )
        except Exception:
            # Bounding box shape didn't match what we expect — leave it out
            # rather than invent coordinates (hard rule; see
            # IMPLEMENTATION_PLAN.md section 7).
            continue
    return page_numbers, bounding_boxes


def _safe_coord_origin(bbox: Any) -> str | None:
    raw = getattr(bbox, "coord_origin", None)
    if raw is None:
        return None
    value = getattr(raw, "name", None) or str(raw)
    return value if value in _VALID_COORD_ORIGINS else None


def _safe_page_count(document: Any) -> int:
    pages = getattr(document, "pages", None)
    if pages is None:
        return 0
    try:
        return len(pages)
    except TypeError:
        return 0


def _safe_document_metadata(document: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    name = getattr(document, "name", None)
    if name:
        metadata["docling_document_name"] = str(name)
    return metadata
