"""Redis-backed JobRepository.

The only file allowed to import `redis`. Stores each IngestionJobStatus as
a JSON blob keyed by job_id, with a TTL so completed/failed job records
don't accumulate in Redis forever.
"""

from __future__ import annotations

import redis

from rag_ingestion.domain.jobs import IngestionJobStatus

_KEY_PREFIX = "rag_ingestion:job:"


class RedisJobRepository:
    """JobRepository implementation backed by Redis.

    Conforms structurally to ports.job_repository.JobRepository (a
    Protocol) — no explicit inheritance needed.
    """

    def __init__(self, redis_url: str, *, ttl_seconds: int = 86400) -> None:
        self._redis_url = redis_url
        self._ttl_seconds = ttl_seconds
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def save(self, job: IngestionJobStatus) -> None:
        self._client.set(_KEY_PREFIX + job.job_id, job.model_dump_json(), ex=self._ttl_seconds)

    def get(self, job_id: str) -> IngestionJobStatus | None:
        raw = self._client.get(_KEY_PREFIX + job_id)
        if raw is None:
            return None
        return IngestionJobStatus.model_validate_json(raw)

    def ping(self, timeout_seconds: float = 2.0) -> bool:
        # Reuses the pooled client rather than opening a fresh connection
        # per health check; timeout_seconds is therefore advisory, bounded
        # by the client's socket timeout rather than applied per-call.
        try:
            return bool(self._client.ping())
        except redis.RedisError:
            return False
