"""Structured JSON logging setup.

Every log line is emitted as one JSON object. `bind_context` attaches
job_id/document_id to the current context (a Celery task on entry, a
FastAPI request via middleware) so every subsequent log record in that
request/task carries them automatically — critical for grep'ing a single
ingestion's logs across parse/chunk/embed/index stages, especially once
retries are involved.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any

from pythonjsonlogger import jsonlogger

_job_id_var: ContextVar[str | None] = ContextVar("job_id", default=None)
_document_id_var: ContextVar[str | None] = ContextVar("document_id", default=None)


def bind_context(*, job_id: str | None = None, document_id: str | None = None) -> None:
    """Bind job_id/document_id onto the current context (thread/async task
    local). Call at the start of a request or Celery task.
    """
    if job_id is not None:
        _job_id_var.set(job_id)
    if document_id is not None:
        _document_id_var.set(document_id)


def clear_context() -> None:
    """Reset bound context. Call in a `finally` block so a thread/worker
    reused for the next request/task doesn't leak the previous job_id.
    """
    _job_id_var.set(None)
    _document_id_var.set(None)


class _ContextFilter(logging.Filter):
    """Attaches the currently-bound job_id/document_id to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.job_id = _job_id_var.get()
        record.document_id = _document_id_var.get()
        return True


class _JsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["timestamp"] = self.formatTime(record, self.datefmt)
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        if getattr(record, "job_id", None):
            log_record["job_id"] = record.job_id
        if getattr(record, "document_id", None):
            log_record["document_id"] = record.document_id


def configure_logging(log_level: str = "INFO") -> None:
    """Configure the root logger for structured JSON output on stdout.

    Call exactly once at process startup (FastAPI `lifespan`, Celery
    `worker_process_init` / module import).
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_ContextFilter())
    handler.setFormatter(_JsonFormatter("%(message)s"))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(log_level.upper())
