"""Celery task layer: _IngestionTask's on_success/on_failure hooks (temp
file + directory cleanup, job-failure marking), and that each task body
correctly resolves the container and delegates to
application.ingestion_pipeline rather than containing business logic
itself. No real Celery broker/worker or Redis is used — task bodies are
called directly (bypassing .delay()/the broker), and on_success/on_failure
are exercised as plain method calls on a fresh _IngestionTask instance.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rag_ingestion.application import ingestion_pipeline as ingestion_pipeline_module
from rag_ingestion.application.errors import PermanentIngestionError, TransientIngestionError
from rag_ingestion.domain.chunks import ChunkingConfig
from rag_ingestion.domain.documents import MetadataIngestionRecord
from rag_ingestion.domain.enums import SourceFormat
from rag_ingestion.domain.jobs import IngestionJobStatus
from rag_ingestion.workers import tasks as tasks_module


class _FakeContainer:
    def __init__(self, job_repository) -> None:
        self.job_repository = job_repository


@pytest.fixture
def patched_container(monkeypatch, fake_job_repository):
    container = _FakeContainer(fake_job_repository)
    monkeypatch.setattr(tasks_module, "get_container", lambda: container)
    return container


def test_remove_temp_file_removes_file_and_its_directory(tmp_path):
    job_dir = tmp_path / "job-123"
    job_dir.mkdir()
    file_path = job_dir / "upload.pdf"
    file_path.write_bytes(b"content")

    tasks_module._remove_temp_file(str(file_path))

    assert not file_path.exists()
    assert not job_dir.exists()


def test_remove_temp_file_survives_missing_file(tmp_path):
    tasks_module._remove_temp_file(str(tmp_path / "does-not-exist.pdf"))  # must not raise


def test_remove_temp_file_survives_none(tmp_path):
    tasks_module._remove_temp_file(None)  # must not raise


def test_remove_temp_file_leaves_nonempty_directory(tmp_path):
    job_dir = tmp_path / "job-456"
    job_dir.mkdir()
    file_path = job_dir / "upload.pdf"
    file_path.write_bytes(b"content")
    (job_dir / "other_file.txt").write_bytes(b"unexpected")

    tasks_module._remove_temp_file(str(file_path))

    assert not file_path.exists()
    assert job_dir.exists()  # non-empty: left alone rather than raising


def test_on_success_removes_temp_file(tmp_path, patched_container):
    job_dir = tmp_path / "job-1"
    job_dir.mkdir()
    file_path = job_dir / "upload.pdf"
    file_path.write_bytes(b"x")

    task = tasks_module._IngestionTask()
    task.on_success(None, "task-id", (), {"job_id": "job-1", "file_path": str(file_path)})

    assert not file_path.exists()
    assert not job_dir.exists()


def test_on_failure_marks_job_failed_and_removes_temp_file(tmp_path, fake_job_repository, patched_container):
    now = datetime.now(timezone.utc)
    fake_job_repository.save(
        IngestionJobStatus(job_id="job-2", source_format=SourceFormat.PDF, created_at=now, updated_at=now)
    )
    job_dir = tmp_path / "job-2"
    job_dir.mkdir()
    file_path = job_dir / "upload.pdf"
    file_path.write_bytes(b"x")

    task = tasks_module._IngestionTask()
    task.on_failure(RuntimeError("boom"), "task-id", (), {"job_id": "job-2", "file_path": str(file_path)}, None)

    job = fake_job_repository.get("job-2")
    assert job.status.value == "failed"
    assert job.error_message == "boom"
    assert job.error_type == "RuntimeError"
    assert not file_path.exists()
    assert not job_dir.exists()


def test_on_failure_without_job_id_does_not_raise(patched_container):
    task = tasks_module._IngestionTask()
    task.on_failure(RuntimeError("boom"), "task-id", (), {}, None)  # no job_id in kwargs


def test_ingest_pdf_task_delegates_to_ingestion_pipeline(monkeypatch, patched_container):
    calls = []

    def fake_run_pdf_ingestion(container, *, job_service, job_id, file_path, metadata, chunking_config):
        calls.append((container, job_id, file_path, metadata, chunking_config))

    monkeypatch.setattr(ingestion_pipeline_module, "run_pdf_ingestion", fake_run_pdf_ingestion)

    tasks_module.ingest_pdf_task(job_id="job-1", file_path="/tmp/x.pdf", metadata={"a": 1}, chunking_config=None)

    assert len(calls) == 1
    container, job_id, file_path, metadata, chunking_config = calls[0]
    assert container is patched_container
    assert job_id == "job-1"
    assert file_path == "/tmp/x.pdf"
    assert metadata == {"a": 1}
    assert chunking_config is None


def test_ingest_pdf_task_validates_chunking_config_dict(monkeypatch, patched_container):
    captured = {}

    def fake_run_pdf_ingestion(container, *, job_service, job_id, file_path, metadata, chunking_config):
        captured["chunking_config"] = chunking_config

    monkeypatch.setattr(ingestion_pipeline_module, "run_pdf_ingestion", fake_run_pdf_ingestion)

    tasks_module.ingest_pdf_task(
        job_id="job-1",
        file_path="/tmp/x.pdf",
        metadata={},
        chunking_config={"chunk_size_tokens": 256, "chunk_overlap_tokens": 32},
    )

    assert isinstance(captured["chunking_config"], ChunkingConfig)
    assert captured["chunking_config"].chunk_size_tokens == 256


def test_ingest_pdf_task_permanent_error_propagates(monkeypatch, patched_container):
    def failing(*args, **kwargs):
        raise PermanentIngestionError("bad pdf")

    monkeypatch.setattr(ingestion_pipeline_module, "run_pdf_ingestion", failing)

    with pytest.raises(PermanentIngestionError):
        tasks_module.ingest_pdf_task(job_id="job-1", file_path="/tmp/x.pdf", metadata={}, chunking_config=None)


def test_ingest_metadata_task_delegates_to_ingestion_pipeline(monkeypatch, patched_container):
    calls = []

    def fake_run_metadata_ingestion(container, *, job_service, job_id, record):
        calls.append((job_id, record))

    monkeypatch.setattr(ingestion_pipeline_module, "run_metadata_ingestion", fake_run_metadata_ingestion)

    tasks_module.ingest_metadata_task(job_id="job-1", record={"text_content": "hi"})

    assert len(calls) == 1
    job_id, record = calls[0]
    assert job_id == "job-1"
    assert isinstance(record, MetadataIngestionRecord)
    assert record.text_content == "hi"


def test_tasks_are_configured_for_autoretry_on_transient_errors():
    assert tasks_module.ingest_pdf_task.autoretry_for == (TransientIngestionError,)
    assert tasks_module.ingest_pdf_task.retry_backoff is True
    assert tasks_module.ingest_pdf_task.retry_kwargs == {"max_retries": 3}
    assert tasks_module.ingest_metadata_task.autoretry_for == (TransientIngestionError,)
    assert isinstance(tasks_module.ingest_pdf_task, tasks_module._IngestionTask)
    assert isinstance(tasks_module.ingest_metadata_task, tasks_module._IngestionTask)
