"""Domain model validation, plus a static check that domain/ never imports
infrastructure — the hard rule the whole architecture depends on.
"""

from __future__ import annotations

import ast
import pathlib
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from rag_ingestion.domain.chunks import ChunkingConfig, ChunkRecord
from rag_ingestion.domain.embeddings import EmbeddingRecord
from rag_ingestion.domain.enums import ChunkLevel, JobStatus, SourceFormat
from rag_ingestion.domain.jobs import IngestionJobStatus
from rag_ingestion.domain.provenance import BoundingBox

_FORBIDDEN_IMPORTS = {"fastapi", "celery", "elasticsearch", "docling", "litellm", "redis"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def test_embedding_record_rejects_vector_length_mismatch():
    with pytest.raises(ValidationError):
        EmbeddingRecord(
            embedding_id="e1",
            owner_id="c1",
            document_id="d1",
            model_name="m",
            dimensions=4,
            vector=[0.1, 0.2, 0.3],
            created_at=_now(),
        )


def test_embedding_record_accepts_matching_vector_length():
    record = EmbeddingRecord(
        embedding_id="e1",
        owner_id="c1",
        document_id="d1",
        model_name="m",
        dimensions=3,
        vector=[0.1, 0.2, 0.3],
        created_at=_now(),
    )
    assert record.dimensions == 3


def test_chunk_record_defaults():
    chunk = ChunkRecord(
        chunk_id="c1",
        document_id="d1",
        chunk_index=0,
        chunk_level=ChunkLevel.LEAF,
        text="hello",
        token_count=1,
        chunker_name="x",
        chunker_version="1.0",
        content_hash="abc",
        created_at=_now(),
    )
    assert chunk.parent_chunk_id is None
    assert chunk.child_chunk_ids == []
    assert chunk.modality.value == "text"
    assert chunk.schema_version == "1.0"


def test_chunking_config_defaults_match_spec():
    config = ChunkingConfig()
    assert config.strategy.value == "section_based_hierarchical"
    assert config.chunk_size_tokens == 512
    assert config.chunk_overlap_tokens == 64
    assert config.embed_parent_chunks is False


def test_bounding_box_requires_page_no_at_least_one():
    with pytest.raises(ValidationError):
        BoundingBox(page_no=0, left=0, top=0, right=1, bottom=1)


def test_job_status_defaults_to_pending():
    job = IngestionJobStatus(job_id="j1", source_format=SourceFormat.PDF, created_at=_now(), updated_at=_now())
    assert job.status == JobStatus.PENDING
    assert job.stage is None
    assert job.document_id is None


def test_domain_package_has_no_infrastructure_imports():
    domain_dir = pathlib.Path(__file__).resolve().parents[2] / "src" / "rag_ingestion" / "domain"
    assert domain_dir.is_dir(), domain_dir

    offenders: dict[str, set[str]] = {}
    for path in domain_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module.split(".")[0])
        hit = found & _FORBIDDEN_IMPORTS
        if hit:
            offenders[path.name] = hit

    assert not offenders, f"domain/ files import forbidden infrastructure: {offenders}"
