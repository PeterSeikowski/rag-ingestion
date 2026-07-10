"""Celery task definitions.

Thin wrappers: each task resolves the shared bootstrap.AppContainer and
delegates the actual parse/chunk/embed/index work to
application.ingestion_pipeline. No parsing/chunking/embedding/indexing
logic lives here.

Temp-file cleanup and job-failure marking happen in Task.on_success/
on_failure (see _IngestionTask below) rather than a plain `finally` in the
task body: Celery calls those hooks exactly once, at the task's true end
(success, or final failure after retries are exhausted) — never on an
intermediate retry attempt, which still needs the temp upload file to be
present on disk.
"""

from __future__ import annotations

import logging
import os

from celery import Task

from rag_ingestion.application import ingestion_pipeline
from rag_ingestion.application.errors import PermanentIngestionError, TransientIngestionError
from rag_ingestion.application.job_service import JobService
from rag_ingestion.bootstrap import get_container
from rag_ingestion.config.logging import bind_context, clear_context
from rag_ingestion.domain.chunks import ChunkingConfig
from rag_ingestion.domain.documents import MetadataIngestionRecord
from rag_ingestion.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _remove_temp_file(file_path: str | None) -> None:
    """Remove the uploaded file and its per-job directory
    (TEMP_UPLOAD_DIR/{job_id}/, created by
    application.ingestion_service.IngestionService.submit_pdf_ingestion).
    Removing only the file and leaving the directory behind would leak one
    empty directory per PDF ingestion job, forever.
    """
    if not file_path or not os.path.exists(file_path):
        return
    try:
        os.remove(file_path)
    except OSError:
        logger.warning("Failed to remove temp upload %s", file_path)
        return
    try:
        os.rmdir(os.path.dirname(file_path))
    except OSError:
        # Non-empty (unexpected extra file) or already gone — either way,
        # not worth failing the task over.
        pass


class _IngestionTask(Task):
    """Marks the job FAILED on the task's terminal failure (a
    PermanentIngestionError, or a TransientIngestionError with retries
    exhausted) and removes the temp upload file, if any, on every terminal
    outcome — success or failure.
    """

    def on_success(self, retval, task_id, args, kwargs) -> None:  # noqa: ANN001
        _remove_temp_file(kwargs.get("file_path"))

    def on_failure(self, exc, task_id, args, kwargs, einfo) -> None:  # noqa: ANN001
        job_id = kwargs.get("job_id")
        if job_id:
            job_service = JobService(get_container().job_repository)
            job_service.mark_failed(job_id, exc)
        _remove_temp_file(kwargs.get("file_path"))


@celery_app.task(
    base=_IngestionTask,
    name="rag_ingestion.ingest_pdf",
    autoretry_for=(TransientIngestionError,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def ingest_pdf_task(
    *,
    job_id: str,
    file_path: str,
    metadata: dict,
    chunking_config: dict | None,
) -> None:
    """Celery entry point for POST /v1/documents/pdf. `metadata` and
    `chunking_config` are plain dicts (Celery's JSON serializer can't carry
    Pydantic models) — validated back into domain types here and inside
    ingestion_pipeline.run_pdf_ingestion.
    """
    container = get_container()
    job_service = JobService(container.job_repository)
    bind_context(job_id=job_id)
    try:
        config = ChunkingConfig.model_validate(chunking_config) if chunking_config else None
        ingestion_pipeline.run_pdf_ingestion(
            container,
            job_service=job_service,
            job_id=job_id,
            file_path=file_path,
            metadata=metadata,
            chunking_config=config,
        )
    except PermanentIngestionError as exc:
        logger.error("PDF ingestion failed permanently for job %s", job_id, exc_info=exc)
        raise
    finally:
        clear_context()


@celery_app.task(
    base=_IngestionTask,
    name="rag_ingestion.ingest_metadata",
    autoretry_for=(TransientIngestionError,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def ingest_metadata_task(*, job_id: str, record: dict) -> None:
    """Celery entry point for POST /v1/records/metadata. `record` is a
    plain dict (JSON-serialized MetadataIngestionRecord).
    """
    container = get_container()
    job_service = JobService(container.job_repository)
    bind_context(job_id=job_id)
    try:
        parsed_record = MetadataIngestionRecord.model_validate(record)
        ingestion_pipeline.run_metadata_ingestion(
            container,
            job_service=job_service,
            job_id=job_id,
            record=parsed_record,
        )
    except PermanentIngestionError as exc:
        logger.error("Metadata ingestion failed permanently for job %s", job_id, exc_info=exc)
        raise
    finally:
        clear_context()
