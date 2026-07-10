"""DoclingPdfParser: defensive extraction against a hand-built fake
Docling object graph. The real `docling` package (pinned exactly in
requirements.txt — see its module docstring for why) is heavy and not
installed in this environment; a `@pytest.mark.docling_integration` test
against the real package belongs in tests/integration/ once it is.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from rag_ingestion.domain.provenance import SourceReference


class _FakeBBox:
    def __init__(self, l, t, r, b, coord_origin="TOPLEFT"):
        self.l, self.t, self.r, self.b = l, t, r, b
        self.coord_origin = types.SimpleNamespace(name=coord_origin)


class _FakeProv:
    def __init__(self, page_no, bbox=None):
        self.page_no = page_no
        self.bbox = bbox


class _FakeItem:
    def __init__(self, label, text="", level=None, prov=None, self_ref=None):
        self.label = label
        self.text = text
        if level is not None:
            self.level = level
        self.prov = prov or []
        self.self_ref = self_ref


class _FakeTableItem(_FakeItem):
    def __init__(self, markdown, **kwargs):
        super().__init__("table", **kwargs)
        self._markdown = markdown

    def export_to_markdown(self):
        return self._markdown


class _FakeBrokenItem:
    """Simulates a genuinely malformed docling item: accessing `.label`
    itself raises, which _safe_extract_item must catch and skip.
    """

    @property
    def label(self):
        raise RuntimeError("simulated malformed docling item")


class _FakeDocument:
    def __init__(self, items):
        self._items = items
        self.pages = {1: object(), 2: object()}
        self.name = "fake-doc"

    def iterate_items(self):
        for item in self._items:
            yield item, 0


@pytest.fixture
def fake_docling_module(monkeypatch):
    title = _FakeItem("title", text="My Report", self_ref="#/texts/0")
    section = _FakeItem(
        "section_header", text="Introduction", level=1, self_ref="#/texts/1",
        prov=[_FakeProv(1, _FakeBBox(10, 10, 100, 30))],
    )
    para = _FakeItem(
        "text", text="This is the intro paragraph.", self_ref="#/texts/2",
        prov=[_FakeProv(1, _FakeBBox(10, 40, 200, 60))],
    )
    table = _FakeTableItem("| a | b |\n|---|---|\n| 1 | 2 |", self_ref="#/tables/0", prov=[_FakeProv(2)])
    bad_item = _FakeBrokenItem()
    doc = _FakeDocument([title, section, para, table, bad_item])

    class _FakeConverter:
        def convert(self, source):
            return types.SimpleNamespace(document=doc)

    fake_converter_module = types.ModuleType("docling.document_converter")
    fake_converter_module.DocumentConverter = _FakeConverter
    fake_docling_pkg = types.ModuleType("docling")
    fake_docling_pkg.document_converter = fake_converter_module
    monkeypatch.setitem(sys.modules, "docling", fake_docling_pkg)
    monkeypatch.setitem(sys.modules, "docling.document_converter", fake_converter_module)
    return doc


@pytest.fixture
def parser(fake_docling_module):
    from rag_ingestion.adapters.parsers.docling_pdf_parser import DoclingPdfParser

    return DoclingPdfParser()


@pytest.fixture
def parsed(parser):
    return parser.parse(Path("fake.pdf"), SourceReference(document_id="doc-1"))


def test_page_count_and_parser_metadata(parsed):
    assert parsed.page_count == 2
    assert parsed.parser_name == "docling_pdf"
    assert parsed.parser_version == "1.0"


def test_malformed_element_is_skipped_not_fatal(parsed):
    # 5 items in the fake graph, one deliberately broken -> 4 survive.
    assert len(parsed.elements) == 4


def test_section_path_tracks_nested_headings(parsed):
    paragraph = parsed.elements[2]
    assert paragraph.source.section_path == ["My Report", "Introduction"]


def test_bounding_box_and_coord_origin_extracted(parsed):
    paragraph = parsed.elements[2]
    assert paragraph.source.page_numbers == [1]
    assert paragraph.source.bounding_boxes[0].coord_origin == "TOPLEFT"


def test_table_element_renders_markdown(parsed):
    table_element = parsed.elements[3]
    assert table_element.element_type.value == "table"
    assert "| a | b |" in table_element.text


def test_module_imports_without_docling_installed():
    # Importing the adapter module must never require docling — it's
    # imported lazily inside parse(). Deliberately does not use the
    # fake_docling_module fixture.
    import rag_ingestion.adapters.parsers.docling_pdf_parser  # noqa: F401
