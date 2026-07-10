"""Port for ingestion job status persistence.

Concrete implementations live under adapters/jobs/ — e.g.
RedisJobRepository. This is the single source of truth for job status; it
is deliberately not backed by Celery's own result backend (see
domain/jobs.py), so job status stays meaningful even if the task queue
implementation changes.
"""

from __future__ import annotations

from typing import Protocol

from rag_ingestion.domain.jobs import IngestionJobStatus


class JobRepository(Protocol):
    """Persists and retrieves IngestionJobStatus records."""

    def save(self, job: IngestionJobStatus) -> None:
        """Upsert `job`. Used for both initial creation and every
        subsequent status/stage update — IngestionJobStatus is small enough
        that always writing the full object avoids partial-update bugs.
        """
        ...

    def get(self, job_id: str) -> IngestionJobStatus | None: ...

    def ping(self, timeout_seconds: float = 2.0) -> bool:
        """Best-effort connectivity check for GET /health/ready."""
        ...
