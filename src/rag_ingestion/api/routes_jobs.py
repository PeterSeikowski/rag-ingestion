"""GET /v1/jobs/{job_id} — job status lookup.

job_service only needs JobRepository (Redis) — no parser/chunker/embedder
— so this route was wired first, in Step 6, well before the ingestion
endpoints in routes_documents.py/routes_metadata.py.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from rag_ingestion.api.dependencies import get_job_service
from rag_ingestion.application.job_service import JobService
from rag_ingestion.domain.jobs import IngestionJobStatus

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=IngestionJobStatus)
async def get_job_status(
    job_id: str, job_service: JobService = Depends(get_job_service)
) -> IngestionJobStatus:
    job = await run_in_threadpool(job_service.get_job, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No job found for job_id={job_id!r}")
    return job
