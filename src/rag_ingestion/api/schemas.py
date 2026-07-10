"""FastAPI request/response schemas.

Kept separate from domain models where the public HTTP contract
deliberately differs from the internal domain shape — e.g. `EmbeddingConfig`
is never exposed here, because model/dimensions are server-controlled (see
IMPLEMENTATION_PLAN.md decision 10: Elasticsearch's dense_vector mapping is
fixed to one model's dimensions per deployment). Where the HTTP contract
and the domain shape are identical, routes reuse the domain model directly
(`MetadataIngestionRecord` as a request body, `IngestionJobStatus` as a
response) instead of duplicating it here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from rag_ingestion.domain.enums import JobStatus


class IngestionAcceptedResponse(BaseModel):
    """Returned by both ingestion endpoints: the job has been accepted and
    queued for asynchronous processing, not yet completed.
    """

    job_id: str
    status: JobStatus = JobStatus.PENDING


class HealthLiveResponse(BaseModel):
    """Process liveness — no dependency checks."""

    status: str = "ok"


class HealthReadyResponse(BaseModel):
    """Dependency readiness. `checks` maps a dependency name (e.g. "redis",
    "elasticsearch") to whether it responded within its timeout.
    """

    status: str
    checks: dict[str, bool] = Field(default_factory=dict)
