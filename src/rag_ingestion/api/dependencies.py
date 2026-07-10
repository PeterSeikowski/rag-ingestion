"""FastAPI dependency providers.

Routes obtain application services exclusively through these functions,
never by importing an adapter directly — keeps api/ honest about only
depending on application/ and the bootstrap container's public surface.
This is also the one place in api/ allowed to import workers.tasks: the
Celery task objects' `.delay` methods are how IngestionService enqueues
work, and wiring them in here (rather than in application/) keeps
Celery-specific imports out of the application layer entirely.
"""

from __future__ import annotations

from rag_ingestion.application.ingestion_service import IngestionService
from rag_ingestion.application.job_service import JobService
from rag_ingestion.bootstrap import get_container
from rag_ingestion.workers.tasks import ingest_metadata_task, ingest_pdf_task


def get_job_service() -> JobService:
    return JobService(get_container().job_repository)


def get_ingestion_service() -> IngestionService:
    container = get_container()
    return IngestionService(
        job_service=JobService(container.job_repository),
        settings=container.settings,
        pdf_task_delay=ingest_pdf_task.delay,
        metadata_task_delay=ingest_metadata_task.delay,
    )
