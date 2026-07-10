"""Job status orchestration.

A thin, typed wrapper around the JobRepository port — the only place
application code should read or write IngestionJobStatus. Kept separate
from ingestion_pipeline.py/ingestion_service.py so both the API (read-only
status lookups) and the worker (status transitions during processing)
depend on the same small surface.
"""

from __future__ import annotations

from datetime import datetime, timezone

from rag_ingestion.domain.enums import JobStage, JobStatus, SourceFormat
from rag_ingestion.domain.jobs import IngestionJobStatus
from rag_ingestion.ports.job_repository import JobRepository


class JobService:
    """Reads and mutates IngestionJobStatus records via a JobRepository port."""

    def __init__(self, job_repository: JobRepository) -> None:
        self._job_repository = job_repository

    def create_job(
        self, job_id: str, source_format: SourceFormat, *, document_id: str | None = None
    ) -> IngestionJobStatus:
        now = datetime.now(timezone.utc)
        job = IngestionJobStatus(
            job_id=job_id,
            status=JobStatus.PENDING,
            document_id=document_id,
            source_format=source_format,
            created_at=now,
            updated_at=now,
        )
        self._job_repository.save(job)
        return job

    def get_job(self, job_id: str) -> IngestionJobStatus | None:
        return self._job_repository.get(job_id)

    def set_document_id(self, job_id: str, document_id: str) -> None:
        job = self._require_job(job_id)
        job.document_id = document_id
        job.updated_at = datetime.now(timezone.utc)
        self._job_repository.save(job)

    def mark_stage(self, job_id: str, stage: JobStage) -> None:
        job = self._require_job(job_id)
        job.status = JobStatus.IN_PROGRESS
        job.stage = stage
        job.updated_at = datetime.now(timezone.utc)
        self._job_repository.save(job)

    def mark_completed(
        self,
        job_id: str,
        *,
        total_pages: int | None = None,
        total_chunks: int | None = None,
        total_embeddings: int | None = None,
    ) -> None:
        job = self._require_job(job_id)
        now = datetime.now(timezone.utc)
        job.status = JobStatus.COMPLETED
        job.stage = None
        if total_pages is not None:
            job.total_pages = total_pages
        if total_chunks is not None:
            job.total_chunks = total_chunks
        if total_embeddings is not None:
            job.total_embeddings = total_embeddings
        job.updated_at = now
        job.completed_at = now
        self._job_repository.save(job)

    def mark_failed(self, job_id: str, error: Exception) -> None:
        job = self._require_job(job_id)
        now = datetime.now(timezone.utc)
        job.status = JobStatus.FAILED
        job.error_message = str(error)
        job.error_type = type(error).__name__
        job.updated_at = now
        job.completed_at = now
        self._job_repository.save(job)

    def _require_job(self, job_id: str) -> IngestionJobStatus:
        job = self._job_repository.get(job_id)
        if job is None:
            raise KeyError(f"No job found for job_id={job_id!r}")
        return job
