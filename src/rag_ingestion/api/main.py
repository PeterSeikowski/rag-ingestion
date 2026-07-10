"""FastAPI application factory.

The only place `docs`/`openapi`/router wiring happens. No business logic
lives here — routes delegate to application/ services resolved through
`api/dependencies.py` once the bootstrap composition root is built (Step 11).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag_ingestion.api import routes_documents, routes_health, routes_jobs, routes_metadata
from rag_ingestion.bootstrap import ensure_vector_store_ready
from rag_ingestion.config.logging import configure_logging
from rag_ingestion.config.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    # Best-effort only (see bootstrap.ensure_vector_store_ready) — must not
    # prevent the app from starting if Elasticsearch isn't reachable yet.
    # The container itself is built lazily, per-request, via
    # api/dependencies.py -> bootstrap.get_container(), not eagerly here:
    # that keeps GET /health/live reachable even when no dependency is up.
    ensure_vector_store_ready()
    yield


def create_app() -> FastAPI:
    """Build the FastAPI app. A function (not a bare module-level app) so
    tests can construct fresh instances with different lifespans/state.
    """
    app = FastAPI(
        title="rag-ingestion-service",
        version="0.1.0",
        description="Standalone RAG ingestion service: parse, chunk, embed, and index documents.",
        lifespan=lifespan,
    )
    app.include_router(routes_health.router)
    app.include_router(routes_documents.router)
    app.include_router(routes_metadata.router)
    app.include_router(routes_jobs.router)
    return app


app = create_app()
