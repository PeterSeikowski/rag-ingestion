"""Ingestion pipeline: parse -> chunk -> embed -> index orchestration,
against fakes for every port (see conftest.py). Covers the
delete-then-insert idempotency rule, external_id -> stable document_id
derivation, failure classification (permanent vs transient), and
metadata-only ingestion's exactly-one-chunk conversion.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rag_ingestion.adapters.chunkers.registry import build_chunker_registry
from rag_ingestion.application import ingestion_pipeline
from rag_ingestion.application.errors import PermanentIngestionError, TransientIngestionError
from rag_ingestion.application.job_service import JobService
from rag_ingestion.bootstrap import AppContainer
from rag_ingestion.config.settings import Settings
from rag_ingestion.domain.documents import MetadataIngestionRecord
from rag_ingestion.domain.enums import ChunkLevel, JobStatus, SourceFormat
from rag_ingestion.domain.jobs import IngestionJobStatus


@pytest.fixture
def settings() -> Settings:
    return Settings(litellm_model="fake-model", litellm_api_key="fake-key")


def _make_container(settings, job_repository, parser, embedder, vector_store) -> AppContainer:
    return AppContainer(
        settings=settings,
        job_repository=job_repository,
        parser=parser,
        chunkers=build_chunker_registry(),
        embedder=embedder,
        vector_store=vector_store,
    )


def _seed_job(job_service: JobService, job_id: str, source_format: SourceFormat) -> None:
    now = datetime.now(timezone.utc)
    job_service._job_repository.save(
        IngestionJobStatus(job_id=job_id, source_format=source_format, created_at=now, updated_at=now)
    )


class TestPdfIngestionHappyPath:
    def test_reaches_completed_with_counts(
        self, settings, fake_job_repository, fake_parser, fake_embedder, fake_vector_store
    ):
        container = _make_container(settings, fake_job_repository, fake_parser, fake_embedder, fake_vector_store)
        job_service = JobService(fake_job_repository)
        _seed_job(job_service, "job-1", SourceFormat.PDF)

        ingestion_pipeline.run_pdf_ingestion(
            container,
            job_service=job_service,
            job_id="job-1",
            file_path="fake.pdf",
            metadata={"title": "My Doc", "external_id": "ext-1", "tenant_id": "t1"},
            chunking_config=None,
        )

        job = job_service.get_job("job-1")
        assert job.status == JobStatus.COMPLETED
        assert job.document_id is not None
        assert job.total_chunks and job.total_chunks > 0
        assert job.total_embeddings and job.total_embeddings > 0

    def test_vector_store_calls_are_delete_then_insert(
        self, settings, fake_job_repository, fake_parser, fake_embedder, fake_vector_store
    ):
        container = _make_container(settings, fake_job_repository, fake_parser, fake_embedder, fake_vector_store)
        job_service = JobService(fake_job_repository)
        _seed_job(job_service, "job-1", SourceFormat.PDF)

        ingestion_pipeline.run_pdf_ingestion(
            container, job_service=job_service, job_id="job-1", file_path="fake.pdf", metadata={}, chunking_config=None,
        )

        call_names = [c[0] for c in fake_vector_store.calls]
        assert call_names == ["delete_document", "upsert_documents", "upsert_chunks", "upsert_embeddings"]

    def test_known_metadata_keys_land_on_document_record_not_custom_metadata(
        self, settings, fake_job_repository, fake_parser, fake_embedder, fake_vector_store
    ):
        container = _make_container(settings, fake_job_repository, fake_parser, fake_embedder, fake_vector_store)
        job_service = JobService(fake_job_repository)
        _seed_job(job_service, "job-1", SourceFormat.PDF)

        ingestion_pipeline.run_pdf_ingestion(
            container,
            job_service=job_service,
            job_id="job-1",
            file_path="fake.pdf",
            metadata={"title": "My Doc", "external_id": "ext-1", "tenant_id": "t1", "custom_field": "x"},
            chunking_config=None,
        )

        doc_record = [c[1] for c in fake_vector_store.calls if c[0] == "upsert_documents"][0][0]
        assert doc_record.title == "My Doc"
        assert doc_record.external_id == "ext-1"
        assert doc_record.tenant_id == "t1"
        assert doc_record.custom_metadata == {"custom_field": "x"}

    def test_chunks_inherit_document_level_acl_and_custom_metadata_fields(
        self, settings, fake_job_repository, fake_parser, fake_embedder, fake_vector_store
    ):
        # Regression test: Chunker.chunk() has no way to receive
        # tenant/ACL/metadata (it only sees ParsedDocument + ChunkingConfig),
        # so run_pdf_ingestion must back-fill these onto every chunk itself
        # — otherwise every PDF-derived chunk in Elasticsearch would have
        # null/empty ACL fields regardless of what the request specified.
        container = _make_container(settings, fake_job_repository, fake_parser, fake_embedder, fake_vector_store)
        job_service = JobService(fake_job_repository)
        _seed_job(job_service, "job-1", SourceFormat.PDF)

        ingestion_pipeline.run_pdf_ingestion(
            container,
            job_service=job_service,
            job_id="job-1",
            file_path="fake.pdf",
            metadata={
                "tenant_id": "acme",
                "allowed_groups": ["engineering"],
                "classification": "internal",
                "custom_field": "value",
            },
            chunking_config=None,
        )

        chunks = [c[1] for c in fake_vector_store.calls if c[0] == "upsert_chunks"][0]
        assert chunks, "expected at least one chunk"
        for chunk in chunks:
            assert chunk.tenant_id == "acme"
            assert chunk.allowed_groups == ["engineering"]
            assert chunk.classification == "internal"
            assert chunk.custom_metadata == {"custom_field": "value"}

    def test_parent_chunks_excluded_from_embedding_by_default(
        self, settings, fake_job_repository, fake_parser, fake_embedder, fake_vector_store
    ):
        container = _make_container(settings, fake_job_repository, fake_parser, fake_embedder, fake_vector_store)
        job_service = JobService(fake_job_repository)
        _seed_job(job_service, "job-1", SourceFormat.PDF)

        ingestion_pipeline.run_pdf_ingestion(
            container, job_service=job_service, job_id="job-1", file_path="fake.pdf", metadata={}, chunking_config=None,
        )

        chunks = [c[1] for c in fake_vector_store.calls if c[0] == "upsert_chunks"][0]
        embeddings = [c[1] for c in fake_vector_store.calls if c[0] == "upsert_embeddings"][0]
        parents = [c for c in chunks if c.chunk_level == ChunkLevel.PARENT]
        assert len(embeddings) == len(chunks) - len(parents)

    def test_same_external_id_derives_same_document_id(
        self, settings, fake_job_repository, fake_parser, fake_embedder, fake_vector_store
    ):
        container = _make_container(settings, fake_job_repository, fake_parser, fake_embedder, fake_vector_store)
        job_service = JobService(fake_job_repository)
        _seed_job(job_service, "job-1", SourceFormat.PDF)
        _seed_job(job_service, "job-2", SourceFormat.PDF)

        ingestion_pipeline.run_pdf_ingestion(
            container, job_service=job_service, job_id="job-1", file_path="fake.pdf",
            metadata={"external_id": "ext-1"}, chunking_config=None,
        )
        ingestion_pipeline.run_pdf_ingestion(
            container, job_service=job_service, job_id="job-2", file_path="fake.pdf",
            metadata={"external_id": "ext-1"}, chunking_config=None,
        )

        assert job_service.get_job("job-1").document_id == job_service.get_job("job-2").document_id

    def test_no_external_id_derives_different_document_ids(
        self, settings, fake_job_repository, fake_parser, fake_embedder, fake_vector_store
    ):
        container = _make_container(settings, fake_job_repository, fake_parser, fake_embedder, fake_vector_store)
        job_service = JobService(fake_job_repository)
        _seed_job(job_service, "job-1", SourceFormat.PDF)
        _seed_job(job_service, "job-2", SourceFormat.PDF)

        ingestion_pipeline.run_pdf_ingestion(
            container, job_service=job_service, job_id="job-1", file_path="fake.pdf", metadata={}, chunking_config=None,
        )
        ingestion_pipeline.run_pdf_ingestion(
            container, job_service=job_service, job_id="job-2", file_path="fake.pdf", metadata={}, chunking_config=None,
        )

        assert job_service.get_job("job-1").document_id != job_service.get_job("job-2").document_id


class TestDocumentIdRetryStability:
    """Regression coverage for: without reusing the job's already-stored
    document_id, a Celery retry with no external_id would mint a fresh
    random document_id every attempt, so delete_document() would never
    find (and therefore never clean up) a prior attempt's partial writes.
    """

    def test_document_id_stable_across_repeated_calls_with_no_external_id(
        self, settings, fake_job_repository, fake_parser, fake_embedder, fake_vector_store
    ):
        container = _make_container(settings, fake_job_repository, fake_parser, fake_embedder, fake_vector_store)
        job_service = JobService(fake_job_repository)
        _seed_job(job_service, "job-1", SourceFormat.PDF)

        ingestion_pipeline.run_pdf_ingestion(
            container, job_service=job_service, job_id="job-1", file_path="fake.pdf", metadata={}, chunking_config=None,
        )
        first_document_id = job_service.get_job("job-1").document_id

        # Simulates a Celery retry: the same job_id is processed again.
        ingestion_pipeline.run_pdf_ingestion(
            container, job_service=job_service, job_id="job-1", file_path="fake.pdf", metadata={}, chunking_config=None,
        )
        second_document_id = job_service.get_job("job-1").document_id

        assert first_document_id == second_document_id

    def test_retry_after_indexing_failure_cleans_up_the_same_document_id(
        self, settings, fake_job_repository, fake_parser, fake_embedder
    ):
        class FlakyVectorStore:
            def __init__(self):
                self.calls = []
                self._fail_next_upsert_chunks = True

            def delete_document(self, document_id):
                self.calls.append(("delete_document", document_id))

            def upsert_documents(self, documents):
                self.calls.append(("upsert_documents", list(documents)))

            def upsert_chunks(self, chunks):
                self.calls.append(("upsert_chunks", list(chunks)))
                if self._fail_next_upsert_chunks:
                    self._fail_next_upsert_chunks = False
                    raise ConnectionError("transient ES hiccup")

            def upsert_embeddings(self, embeddings):
                self.calls.append(("upsert_embeddings", list(embeddings)))

            def ping(self, timeout_seconds=2.0):
                return True

        vector_store = FlakyVectorStore()
        container = _make_container(settings, fake_job_repository, fake_parser, fake_embedder, vector_store)
        job_service = JobService(fake_job_repository)
        _seed_job(job_service, "job-1", SourceFormat.PDF)

        with pytest.raises(TransientIngestionError):
            ingestion_pipeline.run_pdf_ingestion(
                container, job_service=job_service, job_id="job-1", file_path="fake.pdf", metadata={}, chunking_config=None,
            )
        document_id_after_failure = job_service.get_job("job-1").document_id
        assert document_id_after_failure is not None

        # Simulated Celery retry of the same task invocation.
        ingestion_pipeline.run_pdf_ingestion(
            container, job_service=job_service, job_id="job-1", file_path="fake.pdf", metadata={}, chunking_config=None,
        )

        delete_calls = [c for c in vector_store.calls if c[0] == "delete_document"]
        assert len(delete_calls) == 2
        assert delete_calls[0][1] == delete_calls[1][1] == document_id_after_failure
        assert job_service.get_job("job-1").status == JobStatus.COMPLETED


class TestPdfIngestionFailures:
    def test_parsing_failure_raises_permanent_error(
        self, settings, fake_job_repository, failing_parser, fake_embedder, fake_vector_store
    ):
        container = _make_container(settings, fake_job_repository, failing_parser, fake_embedder, fake_vector_store)
        job_service = JobService(fake_job_repository)
        _seed_job(job_service, "job-1", SourceFormat.PDF)

        with pytest.raises(PermanentIngestionError):
            ingestion_pipeline.run_pdf_ingestion(
                container, job_service=job_service, job_id="job-1", file_path="bad.pdf", metadata={}, chunking_config=None,
            )

    def test_embedding_failure_raises_transient_error_with_no_partial_writes(
        self, settings, fake_job_repository, fake_parser, failing_embedder, fake_vector_store
    ):
        container = _make_container(settings, fake_job_repository, fake_parser, failing_embedder, fake_vector_store)
        job_service = JobService(fake_job_repository)
        _seed_job(job_service, "job-1", SourceFormat.PDF)

        with pytest.raises(TransientIngestionError):
            ingestion_pipeline.run_pdf_ingestion(
                container, job_service=job_service, job_id="job-1", file_path="fake.pdf", metadata={}, chunking_config=None,
            )

        assert fake_vector_store.calls == []

    def test_indexing_deterministic_bulk_failure_raises_permanent_error(
        self, settings, fake_job_repository, fake_parser, fake_embedder
    ):
        # A plain RuntimeError is exactly what ElasticsearchVectorStoreWriter
        # raises for a deterministic bulk-item failure (e.g. a mapping
        # conflict) — retrying the identical write would fail identically
        # every time, so this must NOT be classified as transient/retryable.
        class BulkFailureVectorStore:
            def delete_document(self, document_id):
                pass

            def upsert_documents(self, documents):
                pass

            def upsert_chunks(self, chunks):
                raise RuntimeError("Elasticsearch bulk upsert into 'rag_chunks_v1' had 1 failures")

            def upsert_embeddings(self, embeddings):
                pass

            def ping(self, timeout_seconds=2.0):
                return True

        container = _make_container(
            settings, fake_job_repository, fake_parser, fake_embedder, BulkFailureVectorStore()
        )
        job_service = JobService(fake_job_repository)
        _seed_job(job_service, "job-1", SourceFormat.PDF)

        with pytest.raises(PermanentIngestionError):
            ingestion_pipeline.run_pdf_ingestion(
                container, job_service=job_service, job_id="job-1", file_path="fake.pdf", metadata={}, chunking_config=None,
            )

    def test_indexing_connectivity_failure_raises_transient_error(
        self, settings, fake_job_repository, fake_parser, fake_embedder
    ):
        class ConnectionFailureVectorStore:
            def delete_document(self, document_id):
                raise ConnectionError("Elasticsearch unreachable")

            def upsert_documents(self, documents):
                pass

            def upsert_chunks(self, chunks):
                pass

            def upsert_embeddings(self, embeddings):
                pass

            def ping(self, timeout_seconds=2.0):
                return True

        container = _make_container(
            settings, fake_job_repository, fake_parser, fake_embedder, ConnectionFailureVectorStore()
        )
        job_service = JobService(fake_job_repository)
        _seed_job(job_service, "job-1", SourceFormat.PDF)

        with pytest.raises(TransientIngestionError):
            ingestion_pipeline.run_pdf_ingestion(
                container, job_service=job_service, job_id="job-1", file_path="fake.pdf", metadata={}, chunking_config=None,
            )


class TestMetadataIngestion:
    def test_produces_exactly_one_chunk_and_embedding(
        self, settings, fake_job_repository, fake_parser, fake_embedder, fake_vector_store
    ):
        container = _make_container(settings, fake_job_repository, fake_parser, fake_embedder, fake_vector_store)
        job_service = JobService(fake_job_repository)
        _seed_job(job_service, "job-5", SourceFormat.METADATA)
        record = MetadataIngestionRecord(text_content="A short metadata record.", title="Metadata Title")

        ingestion_pipeline.run_metadata_ingestion(container, job_service=job_service, job_id="job-5", record=record)

        job = job_service.get_job("job-5")
        assert job.status == JobStatus.COMPLETED
        assert job.total_chunks == 1
        assert job.total_embeddings == 1
        chunks = [c[1] for c in fake_vector_store.calls if c[0] == "upsert_chunks"][0]
        assert len(chunks) == 1
        assert chunks[0].text == "A short metadata record."

    def test_no_parser_involved(
        self, settings, fake_job_repository, fake_embedder, fake_vector_store
    ):
        # A parser that raises on any call — proves run_metadata_ingestion
        # never touches container.parser.
        class ExplodingParser:
            name = version = "should-not-be-called"
            supported_formats: list[str] = []

            def parse(self, *args, **kwargs):
                raise AssertionError("metadata ingestion must never call the parser")

        container = _make_container(settings, fake_job_repository, ExplodingParser(), fake_embedder, fake_vector_store)
        job_service = JobService(fake_job_repository)
        _seed_job(job_service, "job-6", SourceFormat.METADATA)
        record = MetadataIngestionRecord(text_content="No parser needed here.")

        ingestion_pipeline.run_metadata_ingestion(container, job_service=job_service, job_id="job-6", record=record)

        assert job_service.get_job("job-6").status == JobStatus.COMPLETED
