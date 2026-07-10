"""Liveness and readiness endpoints.

No business logic: /health/live is a pure process check with no
dependencies — it never calls bootstrap.get_container(), so it stays
reachable even if Redis/Elasticsearch/embedding config are all broken.
/health/ready delegates connectivity checks to the JobRepository/
VectorStoreWriter ports (via the bootstrap container) rather than talking
to Redis/Elasticsearch directly — even a health check respects the "no ES
outside the ES adapter" rule.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Response
from fastapi.concurrency import run_in_threadpool

from rag_ingestion.api.schemas import HealthLiveResponse, HealthReadyResponse
from rag_ingestion.bootstrap import get_container

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=HealthLiveResponse)
async def liveness() -> HealthLiveResponse:
    return HealthLiveResponse(status="ok")


@router.get("/ready", response_model=HealthReadyResponse)
async def readiness(response: Response) -> HealthReadyResponse:
    try:
        container = get_container()
        checks = {
            "redis": await run_in_threadpool(container.job_repository.ping),
            "elasticsearch": await run_in_threadpool(container.vector_store.ping),
        }
    except Exception:
        logger.exception("Readiness check failed while building the dependency container")
        checks = {"redis": False, "elasticsearch": False}

    ready = all(checks.values())
    response.status_code = 200 if ready else 503
    return HealthReadyResponse(status="ready" if ready else "not_ready", checks=checks)
