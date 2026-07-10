"""POST /v1/records/metadata — metadata-only ingestion endpoint.

Accepts one MetadataIngestionRecord (the domain model reused directly as
the request body — no PDF, no parser involved) and delegates to
application.ingestion_service, which converts it into exactly one
chunk-like retrievable unit.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from fastapi.concurrency import run_in_threadpool

from rag_ingestion.api.dependencies import get_ingestion_service
from rag_ingestion.api.schemas import IngestionAcceptedResponse
from rag_ingestion.application.ingestion_service import IngestionService
from rag_ingestion.domain.documents import MetadataIngestionRecord

router = APIRouter(prefix="/v1/records", tags=["records"])


@router.post("/metadata", response_model=IngestionAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_metadata_record(
    record: MetadataIngestionRecord, ingestion_service: IngestionService = Depends(get_ingestion_service)
) -> IngestionAcceptedResponse:
    job = await run_in_threadpool(ingestion_service.submit_metadata_ingestion, record)
    return IngestionAcceptedResponse(job_id=job.job_id, status=job.status)
