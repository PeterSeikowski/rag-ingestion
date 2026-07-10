"""Celery application instance.

The only file that configures Celery itself. Built once at module import
time (not per-request) so the long-lived adapter clients constructed via
`bootstrap.get_container()` are shared across every task this worker
process runs — see IMPLEMENTATION_PLAN.md section 9 on the `--pool=solo`
tradeoff this implies.
"""

from __future__ import annotations

from celery import Celery

from rag_ingestion.bootstrap import ensure_vector_store_ready
from rag_ingestion.config.logging import configure_logging
from rag_ingestion.config.settings import get_settings

_settings = get_settings()
configure_logging(_settings.log_level)
ensure_vector_store_ready()

celery_app = Celery(
    "rag_ingestion",
    broker=_settings.celery_broker_url,
    backend=_settings.celery_result_backend,
    include=["rag_ingestion.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Job progress/results live in JobRepository (Redis, via the port), not
    # in Celery's own result backend — see domain/jobs.py and
    # ports/job_repository.py docstrings for why.
    task_ignore_result=True,
)
