"""JobService: every state transition over a fake JobRepository, including
mark_failed — previously exercised nowhere in the test suite even though
it's the only code path that sets JobStatus.FAILED/error_message/error_type
(called from workers/tasks.py's on_failure hook and
application/ingestion_service.py's enqueue-failure handling).
"""

from __future__ import annotations

import pytest

from rag_ingestion.application.job_service import JobService
from rag_ingestion.domain.enums import JobStage, JobStatus, SourceFormat


@pytest.fixture
def job_service(fake_job_repository) -> JobService:
    return JobService(fake_job_repository)


def test_create_job_starts_pending_with_no_document_id(job_service):
    job = job_service.create_job("job-1", SourceFormat.PDF)
    assert job.status == JobStatus.PENDING
    assert job.stage is None
    assert job.document_id is None
    assert job_service.get_job("job-1") == job


def test_create_job_can_set_document_id_upfront(job_service):
    job = job_service.create_job("job-1", SourceFormat.METADATA, document_id="doc-1")
    assert job.document_id == "doc-1"


def test_get_job_returns_none_for_unknown_job(job_service):
    assert job_service.get_job("does-not-exist") is None


def test_set_document_id_updates_existing_job(job_service):
    job_service.create_job("job-1", SourceFormat.PDF)
    job_service.set_document_id("job-1", "doc-42")
    assert job_service.get_job("job-1").document_id == "doc-42"


def test_mark_stage_transitions_to_in_progress(job_service):
    job_service.create_job("job-1", SourceFormat.PDF)
    job_service.mark_stage("job-1", JobStage.EMBEDDING)
    job = job_service.get_job("job-1")
    assert job.status == JobStatus.IN_PROGRESS
    assert job.stage == JobStage.EMBEDDING


def test_mark_completed_sets_counts_and_clears_stage(job_service):
    job_service.create_job("job-1", SourceFormat.PDF)
    job_service.mark_stage("job-1", JobStage.INDEXING)
    job_service.mark_completed("job-1", total_pages=10, total_chunks=42, total_embeddings=30)

    job = job_service.get_job("job-1")
    assert job.status == JobStatus.COMPLETED
    assert job.stage is None
    assert job.total_pages == 10
    assert job.total_chunks == 42
    assert job.total_embeddings == 30
    assert job.completed_at is not None


def test_mark_completed_only_overwrites_supplied_counts(job_service):
    job_service.create_job("job-1", SourceFormat.PDF)
    job_service.mark_completed("job-1", total_chunks=5)
    job_service.mark_completed("job-1", total_embeddings=3)

    job = job_service.get_job("job-1")
    assert job.total_chunks == 5  # not clobbered by the second call
    assert job.total_embeddings == 3


def test_mark_failed_sets_failed_status_and_error_fields(job_service):
    job_service.create_job("job-1", SourceFormat.PDF)
    job_service.mark_failed("job-1", RuntimeError("elasticsearch bulk upsert failed"))

    job = job_service.get_job("job-1")
    assert job.status == JobStatus.FAILED
    assert job.error_message == "elasticsearch bulk upsert failed"
    assert job.error_type == "RuntimeError"
    assert job.completed_at is not None


def test_mark_failed_preserves_error_type_across_exception_classes(job_service):
    job_service.create_job("job-1", SourceFormat.PDF)

    class MyCustomError(Exception):
        pass

    job_service.mark_failed("job-1", MyCustomError("custom failure"))

    job = job_service.get_job("job-1")
    assert job.error_type == "MyCustomError"
    assert job.error_message == "custom failure"


def test_mark_failed_on_unknown_job_raises_key_error(job_service):
    with pytest.raises(KeyError):
        job_service.mark_failed("does-not-exist", RuntimeError("x"))


def test_mark_stage_on_unknown_job_raises_key_error(job_service):
    with pytest.raises(KeyError):
        job_service.mark_stage("does-not-exist", JobStage.PARSING)
