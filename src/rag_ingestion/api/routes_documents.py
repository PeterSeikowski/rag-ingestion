"""POST /v1/documents/pdf — PDF ingestion endpoint.

Validates and deserializes the multipart request (this is request
deserialization, not business logic, so it's allowed at the API boundary),
then delegates to application.ingestion_service. Never touches Docling,
Elasticsearch, or an embedding provider directly.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from rag_ingestion.api.dependencies import get_ingestion_service
from rag_ingestion.api.schemas import IngestionAcceptedResponse
from rag_ingestion.application.ingestion_service import IngestionService
from rag_ingestion.config.settings import get_settings
from rag_ingestion.domain.chunks import ChunkingConfig

router = APIRouter(prefix="/v1/documents", tags=["documents"])

_ACCEPTED_CONTENT_TYPES = {"application/pdf", "application/octet-stream", None}


@router.post("/pdf", response_model=IngestionAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_pdf(
    file: UploadFile = File(..., description="PDF file to ingest"),
    metadata: str | None = Form(default=None, description="Optional JSON-encoded custom metadata object"),
    chunking_config: str | None = Form(default=None, description="Optional JSON-encoded ChunkingConfig override"),
    embedding_config: str | None = Form(
        default=None,
        description="Reserved for forward compatibility; embedding model/dimensions are server-controlled",
    ),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> IngestionAcceptedResponse:
    if file.content_type not in _ACCEPTED_CONTENT_TYPES:
        raise HTTPException(status_code=422, detail=f"Unsupported content type: {file.content_type}")

    metadata_dict = _parse_json_object(metadata, "metadata")
    parsed_chunking_config = _parse_chunking_config(chunking_config)
    _ = embedding_config  # reserved; see IMPLEMENTATION_PLAN.md decision 10

    settings = get_settings()
    file_bytes = await _read_upload_within_limit(file, settings.max_upload_size_bytes)
    if not file_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    job = await run_in_threadpool(
        ingestion_service.submit_pdf_ingestion,
        file_bytes=file_bytes,
        original_filename=file.filename or "upload.pdf",
        metadata=metadata_dict,
        chunking_config=parsed_chunking_config,
    )
    return IngestionAcceptedResponse(job_id=job.job_id, status=job.status)


async def _read_upload_within_limit(file: UploadFile, max_bytes: int) -> bytes:
    """Read `file` in bounded chunks, rejecting as soon as the configured
    limit is exceeded rather than after buffering the entire body —
    `await file.read()` with no size argument would read an arbitrarily
    large upload into memory in full before the size check ever ran.
    """
    chunk_size = 1024 * 1024
    total = 0
    chunks: list[bytes] = []
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds MAX_UPLOAD_SIZE_BYTES={max_bytes}",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _parse_json_object(raw: str | None, field_name: str) -> dict:
    if raw is None:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"{field_name} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail=f"{field_name} must be a JSON object")
    return payload


def _parse_chunking_config(raw: str | None) -> ChunkingConfig | None:
    if raw is None:
        return None
    payload = _parse_json_object(raw, "chunking_config")
    try:
        return ChunkingConfig.model_validate(payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"chunking_config failed validation: {exc}") from exc
