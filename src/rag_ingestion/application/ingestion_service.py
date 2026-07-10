"""Ingestion request orchestration: validates input, persists a temp
upload (PDF path only), creates the job record, and enqueues the async
task. No parsing/chunking/embedding/indexing logic lives here — that's
application.ingestion_pipeline, run by the Celery worker.

Takes the Celery tasks' `.delay` methods as plain injected callables
(`pdf_task_delay`/`metadata_task_delay`) rather than importing
workers.tasks directly: that keeps Celery-specific imports out of the
application layer entirely (see api/dependencies.py, which is where the
concrete task callables are wired in).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Callable

from rag_ingestion.application.job_service import JobService
from rag_ingestion.config.settings import Settings
from rag_ingestion.domain.chunks import ChunkingConfig
from rag_ingestion.domain.documents import MetadataIngestionRecord
from rag_ingestion.domain.enums import SourceFormat
from rag_ingestion.domain.jobs import IngestionJobStatus


class IngestionService:
    """Accepts and enqueues ingestion requests. Constructed with a
    JobService, Settings, and the two task-delay callables it invokes.
    """

    def __init__(
        self,
        *,
        job_service: JobService,
        settings: Settings,
        pdf_task_delay: Callable[..., Any],
        metadata_task_delay: Callable[..., Any],
    ) -> None:
        self._job_service = job_service
        self._settings = settings
        self._pdf_task_delay = pdf_task_delay
        self._metadata_task_delay = metadata_task_delay

    def submit_pdf_ingestion(
        self,
        *,
        file_bytes: bytes,
        original_filename: str,
        metadata: dict[str, Any] | None,
        chunking_config: ChunkingConfig | None,
    ) -> IngestionJobStatus:
        job_id = str(uuid.uuid4())
        job = self._job_service.create_job(job_id, SourceFormat.PDF)

        upload_dir = Path(self._settings.temp_upload_dir) / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / _safe_filename(original_filename)
        file_path.write_bytes(file_bytes)

        try:
            self._pdf_task_delay(
                job_id=job_id,
                file_path=str(file_path),
                metadata=metadata or {},
                chunking_config=chunking_config.model_dump(mode="json") if chunking_config else None,
            )
        except Exception as exc:
            # Enqueue itself failed (e.g. broker unreachable) — nothing
            # will ever process this file, so clean it up immediately
            # rather than leaving it orphaned in TEMP_UPLOAD_DIR, and mark
            # the job FAILED rather than leaving it stuck at PENDING for
            # up to JOB_STATUS_TTL_SECONDS with no worker ever going to
            # pick it up.
            file_path.unlink(missing_ok=True)
            self._job_service.mark_failed(job_id, exc)
            raise

        return job

    def submit_metadata_ingestion(self, record: MetadataIngestionRecord) -> IngestionJobStatus:
        job_id = str(uuid.uuid4())
        job = self._job_service.create_job(job_id, SourceFormat.METADATA)
        try:
            self._metadata_task_delay(job_id=job_id, record=record.model_dump(mode="json"))
        except Exception as exc:
            self._job_service.mark_failed(job_id, exc)
            raise
        return job


def _safe_filename(original_filename: str) -> str:
    # Only the basename is trusted; anything path-like is stripped so a
    # crafted filename can't write outside TEMP_UPLOAD_DIR/{job_id}/ (the
    # job_id subdirectory already isolates uploads from each other).
    return Path(original_filename).name or "upload.pdf"
