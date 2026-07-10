"""Port for pluggable document parsers.

A DocumentParser turns a source file into the canonical ParsedDocument
representation (domain/documents.py). Concrete implementations live under
adapters/parsers/ — e.g. DoclingPdfParser. Nothing outside that adapter
module may import a parser-specific library (Docling, PyMuPDF, ...).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from rag_ingestion.domain.documents import ParsedDocument
from rag_ingestion.domain.provenance import SourceReference


class DocumentParser(Protocol):
    """Parses a source file into a ParsedDocument.

    `name`/`version` are stamped onto the resulting ParsedDocument
    (`parser_name`/`parser_version`) for provenance, and flow transitively
    into every ChunkRecord's provenance.
    """

    name: str
    version: str
    supported_formats: list[str]

    def parse(self, file_path: Path, source_reference: SourceReference) -> ParsedDocument:
        """Parse `file_path` into a ParsedDocument.

        `source_reference.document_id` is the document_id the resulting
        ParsedDocument and all of its elements must be stamped with — the
        parser does not mint its own document_id.
        """
        ...
