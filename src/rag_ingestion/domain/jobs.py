"""Ingestion job status domain model.

Pure data — no infrastructure imports, no Celery types. The JobRepository
port (ports/job_repository.py) persists this; application/job_service.py
owns its state transitions.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from .enums import JobStage, JobStatus, SourceFormat


class IngestionJobStatus(BaseModel):
    """Current status of one asynchronous ingestion job.

    This is the sole source of truth for job progress. It is intentionally
    not derived from Celery's own AsyncResult — that would couple job
    status to the task queue implementation, defeating the purpose of
    routing it through JobRepository.
    """

    job_id: str
    status: JobStatus = JobStatus.PENDING
    stage: JobStage | None = None
    document_id: str | None = None
    source_format: SourceFormat
    error_message: str | None = None
    error_type: str | None = None
    total_pages: int | None = None
    total_chunks: int | None = None
    total_embeddings: int | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
