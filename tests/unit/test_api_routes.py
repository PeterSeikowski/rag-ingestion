"""API route tests: both ingestion endpoints return job_id + 202, job
status lookup works end-to-end (simulating the Celery worker by running
application.ingestion_pipeline directly against the captured task
payload), and request validation errors return the right status codes.
Fakes are injected via FastAPI's dependency_overrides — no real Celery
broker, Redis, or Elasticsearch needed.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from rag_ingestion.adapters.chunkers.registry import build_chunker_registry
from rag_ingestion.api.dependencies import get_ingestion_service, get_job_service
from rag_ingestion.api.main import app
from rag_ingestion.application import ingestion_pipeline
from rag_ingestion.application.ingestion_service import IngestionService
from rag_ingestion.application.job_service import JobService
from rag_ingestion.bootstrap import AppContainer
from rag_ingestion.config.settings import Settings
from rag_ingestion.domain.chunks import ChunkingConfig
from rag_ingestion.domain.documents import MetadataIngestionRecord


@pytest.fixture
def captured_tasks():
    return {"pdf": [], "metadata": []}


@pytest.fixture
def api_fixture(tmp_path, fake_job_repository, fake_parser, fake_embedder, fake_vector_store, captured_tasks):
    settings = Settings(litellm_model="fake-model", litellm_api_key="fake-key", temp_upload_dir=str(tmp_path))
    container = AppContainer(
        settings=settings,
        job_repository=fake_job_repository,
        parser=fake_parser,
        chunkers=build_chunker_registry(),
        embedder=fake_embedder,
        vector_store=fake_vector_store,
    )
    job_service = JobService(fake_job_repository)
    ingestion_service = IngestionService(
        job_service=job_service,
        settings=settings,
        pdf_task_delay=lambda **kwargs: captured_tasks["pdf"].append(kwargs),
        metadata_task_delay=lambda **kwargs: captured_tasks["metadata"].append(kwargs),
    )

    app.dependency_overrides[get_job_service] = lambda: job_service
    app.dependency_overrides[get_ingestion_service] = lambda: ingestion_service
    try:
        yield TestClient(app), container, job_service
    finally:
        app.dependency_overrides.clear()


def test_pdf_ingestion_returns_job_id_and_writes_temp_file(api_fixture, captured_tasks):
    client, _container, _job_service = api_fixture

    response = client.post(
        "/v1/documents/pdf",
        files={"file": ("report.pdf", b"%PDF-1.4 fake pdf bytes", "application/pdf")},
        data={"metadata": '{"title": "Report"}'},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["job_id"]
    assert body["status"] == "pending"
    assert len(captured_tasks["pdf"]) == 1
    assert os.path.exists(captured_tasks["pdf"][0]["file_path"])


def test_metadata_ingestion_returns_job_id(api_fixture, captured_tasks):
    client, _container, _job_service = api_fixture

    response = client.post("/v1/records/metadata", json={"text_content": "A quick fact."})

    assert response.status_code == 202
    assert response.json()["job_id"]
    assert len(captured_tasks["metadata"]) == 1


def test_full_round_trip_pdf_ingestion_to_completed(api_fixture, captured_tasks):
    client, container, job_service = api_fixture

    response = client.post(
        "/v1/documents/pdf", files={"file": ("report.pdf", b"%PDF-1.4 fake pdf bytes", "application/pdf")}
    )
    job_id = response.json()["job_id"]

    # Simulate the Celery worker running against the exact captured payload.
    task_kwargs = captured_tasks["pdf"][0]
    chunking_config = (
        ChunkingConfig.model_validate(task_kwargs["chunking_config"]) if task_kwargs["chunking_config"] else None
    )
    ingestion_pipeline.run_pdf_ingestion(
        container,
        job_service=job_service,
        job_id=job_id,
        file_path=task_kwargs["file_path"],
        metadata=task_kwargs["metadata"],
        chunking_config=chunking_config,
    )

    response = client.get(f"/v1/jobs/{job_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["document_id"]


def test_full_round_trip_metadata_ingestion_to_completed(api_fixture, captured_tasks):
    client, container, job_service = api_fixture

    response = client.post("/v1/records/metadata", json={"text_content": "A quick fact.", "title": "Fact"})
    job_id = response.json()["job_id"]

    record = MetadataIngestionRecord.model_validate(captured_tasks["metadata"][0]["record"])
    ingestion_pipeline.run_metadata_ingestion(container, job_service=job_service, job_id=job_id, record=record)

    response = client.get(f"/v1/jobs/{job_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["total_chunks"] == 1


def test_unknown_job_returns_404(api_fixture):
    client, _container, _job_service = api_fixture
    response = client.get("/v1/jobs/does-not-exist")
    assert response.status_code == 404


def test_malformed_chunking_config_returns_422(api_fixture):
    client, _container, _job_service = api_fixture
    response = client.post(
        "/v1/documents/pdf",
        files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")},
        data={"chunking_config": "{not valid json"},
    )
    assert response.status_code == 422


def test_empty_file_returns_422(api_fixture):
    client, _container, _job_service = api_fixture
    response = client.post("/v1/documents/pdf", files={"file": ("empty.pdf", b"", "application/pdf")})
    assert response.status_code == 422


def test_health_live_always_returns_ok(api_fixture):
    client, _container, _job_service = api_fixture
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


class _PingableContainer:
    """Minimal stand-in for bootstrap.AppContainer exposing only what
    GET /health/ready actually touches: job_repository.ping() and
    vector_store.ping().
    """

    def __init__(self, *, redis_ok: bool, elasticsearch_ok: bool) -> None:
        self.job_repository = type("_R", (), {"ping": staticmethod(lambda timeout_seconds=2.0: redis_ok)})()
        self.vector_store = type("_V", (), {"ping": staticmethod(lambda timeout_seconds=2.0: elasticsearch_ok)})()


@pytest.fixture
def health_client():
    return TestClient(app)


def test_health_ready_returns_200_when_all_dependencies_up(monkeypatch, health_client):
    import rag_ingestion.api.routes_health as routes_health_module

    monkeypatch.setattr(
        routes_health_module, "get_container", lambda: _PingableContainer(redis_ok=True, elasticsearch_ok=True)
    )

    response = health_client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"] == {"redis": True, "elasticsearch": True}


def test_health_ready_returns_503_when_elasticsearch_down(monkeypatch, health_client):
    import rag_ingestion.api.routes_health as routes_health_module

    monkeypatch.setattr(
        routes_health_module, "get_container", lambda: _PingableContainer(redis_ok=True, elasticsearch_ok=False)
    )

    response = health_client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"] == {"redis": True, "elasticsearch": False}


def test_health_ready_returns_503_when_container_build_raises(monkeypatch, health_client):
    import rag_ingestion.api.routes_health as routes_health_module

    def raising_get_container():
        raise ValueError("LiteLLMEmbedder requires an api_key (LITELLM_API_KEY)")

    monkeypatch.setattr(routes_health_module, "get_container", raising_get_container)

    response = health_client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"] == {"redis": False, "elasticsearch": False}
