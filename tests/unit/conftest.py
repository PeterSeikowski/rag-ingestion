"""Shared fixtures for unit tests: fakes conforming to each port, used
instead of real Docling/LiteLLM/Elasticsearch/Redis. No test under
tests/unit/ touches real infrastructure — that's what tests/integration/
is for.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rag_ingestion.domain.documents import ParsedDocument, ParsedElement
from rag_ingestion.domain.enums import ElementType
from rag_ingestion.domain.provenance import SourceReference


class FakeJobRepository:
    """In-memory JobRepository fake."""

    def __init__(self) -> None:
        self._store: dict[str, object] = {}

    def save(self, job) -> None:
        self._store[job.job_id] = job.model_copy(deep=True)

    def get(self, job_id: str):
        return self._store.get(job_id)

    def ping(self, timeout_seconds: float = 2.0) -> bool:
        return True


class FakeParser:
    """DocumentParser fake: returns a fixed two-element ParsedDocument
    (one heading, one paragraph) regardless of input.
    """

    name = "fake_parser"
    version = "1.0"
    supported_formats = ["pdf"]

    def parse(self, file_path, source_reference: SourceReference) -> ParsedDocument:
        return ParsedDocument(
            document_id=source_reference.document_id,
            source_filename=file_path.name,
            page_count=1,
            elements=[
                ParsedElement(
                    element_id="e0",
                    element_type=ElementType.HEADING,
                    text="Intro",
                    order_index=0,
                    heading_level=1,
                    source=SourceReference(
                        document_id=source_reference.document_id,
                        page_numbers=[1],
                        section_path=["Intro"],
                        section_title="Intro",
                    ),
                ),
                ParsedElement(
                    element_id="e1",
                    element_type=ElementType.TEXT,
                    text="Some body text about the report.",
                    order_index=1,
                    source=SourceReference(
                        document_id=source_reference.document_id,
                        page_numbers=[1],
                        section_path=["Intro"],
                        section_title="Intro",
                    ),
                ),
            ],
            parser_name=self.name,
            parser_version=self.version,
            parsed_at=datetime.now(timezone.utc),
        )


class FailingParser:
    """DocumentParser fake that always raises, for permanent-failure tests."""

    name = "failing_parser"
    version = "1.0"
    supported_formats = ["pdf"]

    def parse(self, file_path, source_reference: SourceReference) -> ParsedDocument:
        raise ValueError("corrupt PDF")


class FakeEmbedder:
    """Embedder fake: deterministic vectors, fixed dimensions=3."""

    name = "fake_embedder"
    provider = "fake"
    model = "fake-model"

    @property
    def dimensions(self) -> int:
        return 3

    def embed_texts(self, texts, config):
        return [[float(i), 0.0, 0.0] for i, _ in enumerate(texts, start=1)]


class FailingEmbedder(FakeEmbedder):
    """Embedder fake that always raises, for transient-failure tests."""

    def embed_texts(self, texts, config):
        raise ConnectionError("embedding provider unreachable")


class FakeVectorStore:
    """VectorStoreWriter fake: records every call (name + args) in order,
    for asserting call sequencing (e.g. delete-then-insert).
    """

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def ensure_indices(self) -> None:
        self.calls.append(("ensure_indices",))

    def upsert_documents(self, documents) -> None:
        self.calls.append(("upsert_documents", list(documents)))

    def upsert_chunks(self, chunks) -> None:
        self.calls.append(("upsert_chunks", list(chunks)))

    def upsert_embeddings(self, embeddings) -> None:
        self.calls.append(("upsert_embeddings", list(embeddings)))

    def delete_document(self, document_id: str) -> None:
        self.calls.append(("delete_document", document_id))

    def ping(self, timeout_seconds: float = 2.0) -> bool:
        return True


@pytest.fixture
def fake_job_repository() -> FakeJobRepository:
    return FakeJobRepository()


@pytest.fixture
def fake_parser() -> FakeParser:
    return FakeParser()


@pytest.fixture
def failing_parser() -> FailingParser:
    return FailingParser()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def failing_embedder() -> FailingEmbedder:
    return FailingEmbedder()


@pytest.fixture
def fake_vector_store() -> FakeVectorStore:
    return FakeVectorStore()
