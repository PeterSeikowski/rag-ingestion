"""ElasticsearchVectorStoreWriter: mapping shapes, bulk upsert wiring, and
delete-then-insert behavior — against a real (but network-idle)
Elasticsearch client instance with its methods monkeypatched. No live
cluster required (see tests/integration/README.md for that).
"""

from __future__ import annotations

import types
from datetime import datetime, timezone

import pytest
from elasticsearch import NotFoundError

import rag_ingestion.adapters.vectorstores.elasticsearch_writer as es_writer_module
from rag_ingestion.domain.chunks import ChunkRecord
from rag_ingestion.domain.documents import DocumentRecord
from rag_ingestion.domain.enums import ChunkLevel, SourceFormat


@pytest.fixture
def writer():
    return es_writer_module.ElasticsearchVectorStoreWriter(
        elasticsearch_url="http://localhost:9200",
        documents_index="rag_documents_v1",
        chunks_index="rag_chunks_v1",
        embeddings_index="rag_embeddings_v1",
        embedding_dimensions=3,
    )


@pytest.fixture
def captured_bulk_calls(monkeypatch):
    calls: list[list[dict]] = []

    def fake_bulk(client, actions, raise_on_error=False):
        actions = list(actions)
        calls.append(actions)
        return len(actions), []

    monkeypatch.setattr(es_writer_module, "bulk", fake_bulk)
    return calls


def _document_record() -> DocumentRecord:
    now = datetime.now(timezone.utc)
    return DocumentRecord(document_id="doc-1", source_format=SourceFormat.PDF, ingested_at=now, updated_at=now)


def _chunk_record() -> ChunkRecord:
    return ChunkRecord(
        chunk_id="doc-1:c0",
        document_id="doc-1",
        chunk_index=0,
        chunk_level=ChunkLevel.LEAF,
        text="hello",
        token_count=1,
        chunker_name="recursive_text",
        chunker_version="1.0",
        content_hash="abc",
        created_at=datetime.now(timezone.utc),
    )


def test_ensure_indices_only_creates_missing_ones(writer):
    created: list[tuple[str, dict]] = []
    existing = {"rag_documents_v1"}
    writer._client.indices.exists = lambda index: index in existing
    writer._client.indices.create = lambda index, mappings: created.append((index, mappings))

    writer.ensure_indices()

    assert [name for name, _ in created] == ["rag_chunks_v1", "rag_embeddings_v1"]


def test_ensure_indices_embeddings_mapping_has_correct_dims(writer):
    created: dict[str, dict] = {}
    writer._client.indices.exists = lambda index: False
    writer._client.indices.create = lambda index, mappings: created.__setitem__(index, mappings)

    writer.ensure_indices()

    vector_field = created["rag_embeddings_v1"]["properties"]["vector"]
    assert vector_field == {"type": "dense_vector", "dims": 3, "index": True, "similarity": "cosine"}


def test_upsert_chunks_never_carries_a_vector_field(writer, captured_bulk_calls):
    writer.upsert_chunks([_chunk_record()])

    action = captured_bulk_calls[0][0]
    assert action["_index"] == "rag_chunks_v1"
    assert action["_id"] == "doc-1:c0"
    assert "vector" not in action["_source"]
    assert action["_source"]["chunk_level"] == "leaf"


def test_upsert_documents_serializes_enum_as_plain_string(writer, captured_bulk_calls):
    writer.upsert_documents([_document_record()])

    assert captured_bulk_calls[0][0]["_source"]["source_format"] == "pdf"


def test_upsert_with_empty_list_does_not_call_bulk(writer, monkeypatch):
    called = []
    monkeypatch.setattr(es_writer_module, "bulk", lambda *a, **k: called.append(True))
    writer.upsert_chunks([])
    assert called == []


def test_bulk_errors_raise_runtime_error(writer, monkeypatch):
    monkeypatch.setattr(
        es_writer_module,
        "bulk",
        lambda client, actions, raise_on_error=False: (0, [{"index": {"error": "mapper_parsing_exception"}}]),
    )
    with pytest.raises(RuntimeError):
        writer.upsert_documents([_document_record()])


def test_delete_document_queries_chunks_and_embeddings_then_deletes_document(writer):
    dbq_calls = []
    delete_calls = []
    writer._client.delete_by_query = lambda index, query, conflicts: dbq_calls.append((index, query, conflicts))
    writer._client.delete = lambda index, id: delete_calls.append((index, id))

    writer.delete_document("doc-1")

    assert dbq_calls == [
        ("rag_chunks_v1", {"term": {"document_id": "doc-1"}}, "proceed"),
        ("rag_embeddings_v1", {"term": {"document_id": "doc-1"}}, "proceed"),
    ]
    assert delete_calls == [("rag_documents_v1", "doc-1")]


def test_delete_document_swallows_404_on_documents_index(writer):
    def raising_delete(index, id):
        raise NotFoundError("not found", types.SimpleNamespace(status=404), None)

    writer._client.delete_by_query = lambda **kwargs: None
    writer._client.delete = raising_delete

    writer.delete_document("doc-missing")  # must not raise


def test_ping_returns_true_on_success(writer):
    writer._client.ping = lambda request_timeout: True
    assert writer.ping() is True


def test_ping_returns_false_on_connectivity_failure(writer):
    def raising_ping(request_timeout):
        raise ConnectionError("no cluster")

    writer._client.ping = raising_ping
    assert writer.ping() is False
