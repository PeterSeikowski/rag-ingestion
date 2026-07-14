"""Composition root: wires ports to concrete adapters from Settings.

Both `api/main.py` (via `api/dependencies.py`) and `workers/celery_app.py` /
`workers/tasks.py` obtain adapters exclusively through `get_container()`,
built once per process. This is the only module outside `adapters/` itself
allowed to import adapter classes — application/, api/, and workers/ code
depends only on the AppContainer's port-typed fields.

`build_container()` never makes a network call (Redis/Elasticsearch client
construction is lazy; embedder construction only validates config
strings), so it can't block or crash process startup. The one genuinely
network-dependent step — creating Elasticsearch indices — is a separate,
explicit, best-effort call: `ensure_vector_store_ready()`, invoked once by
each process's entry point. It logs and swallows failures rather than
raising, so a not-yet-ready Elasticsearch cluster doesn't prevent the API
or worker process from starting (GET /health/ready surfaces the problem
instead).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

from rag_ingestion.adapters.chunkers.registry import build_chunker_registry
from rag_ingestion.adapters.embedders.litellm_embedder import LiteLLMEmbedder
from rag_ingestion.adapters.embedders.azure_openai_embedder import AzureOpenAIEmbedder
from rag_ingestion.adapters.jobs.redis_job_repository import RedisJobRepository
from rag_ingestion.adapters.parsers.docling_pdf_parser import DoclingPdfParser
from rag_ingestion.adapters.vectorstores.elasticsearch_writer import ElasticsearchVectorStoreWriter
from rag_ingestion.application.errors import ConfigurationError
from rag_ingestion.config.settings import Settings, get_settings
from rag_ingestion.domain.enums import ChunkingStrategy
from rag_ingestion.ports.chunker import Chunker
from rag_ingestion.ports.embedder import Embedder
from rag_ingestion.ports.job_repository import JobRepository
from rag_ingestion.ports.parser import DocumentParser
from rag_ingestion.ports.vector_store_writer import VectorStoreWriter

logger = logging.getLogger(__name__)


@dataclass
class AppContainer:
    """Every adapter instance the API and worker processes need, built once
    from Settings. Application/API/worker code depends on this object's
    port-typed fields, never on a concrete adapter import.
    """

    settings: Settings
    job_repository: JobRepository
    parser: DocumentParser
    chunkers: dict[ChunkingStrategy, Chunker]
    embedder: Embedder
    vector_store: VectorStoreWriter


def build_container(settings: Settings | None = None) -> AppContainer:
    """Construct a fresh AppContainer from Settings.

    Prefer `get_container()` in application/API/worker code; this is
    exposed separately so tests can build an isolated container without
    touching the process-wide cache.

    Raises ValueError (from LiteLLMEmbedder) if required embedding config
    is missing, or ConfigurationError if the embedding model's dimensions
    can't be resolved — both fail fast and clearly, per spec, rather than
    deferring to the first ingestion attempt.
    """
    settings = settings or get_settings()

    job_repository = RedisJobRepository(settings.redis_url, ttl_seconds=settings.job_status_ttl_seconds)
    parser = DoclingPdfParser(enable_ocr=settings.docling_enable_ocr)
    chunkers = build_chunker_registry()
    embedder = _build_embedder(settings=settings)

    dimensions = embedder.dimensions
    if dimensions is None:
        raise ConfigurationError(
            f"Cannot resolve embedding dimensions for model={settings.litellm_model!r}; "
            "set EMBEDDING_DIMENSIONS explicitly (Elasticsearch's dense_vector mapping needs a "
            "fixed dimension count at index-creation time)."
        )

    vector_store = ElasticsearchVectorStoreWriter(
        elasticsearch_url=settings.elasticsearch_url,
        documents_index=settings.es_documents_index,
        chunks_index=settings.es_chunks_index,
        embeddings_index=settings.es_embeddings_index,
        embedding_dimensions=dimensions,
        username=settings.elasticsearch_username,
        password=settings.elasticsearch_password,
        verify_certs=settings.elasticsearch_verify_certs,
    )

    return AppContainer(
        settings=settings,
        job_repository=job_repository,
        parser=parser,
        chunkers=chunkers,
        embedder=embedder,
        vector_store=vector_store,
    )


@lru_cache
def get_container() -> AppContainer:
    """Process-wide cached AppContainer.

    Built once per process (API or worker) — see IMPLEMENTATION_PLAN.md
    section 9 on why a single long-lived container per process is required
    for the worker's `--pool=solo` choice to make sense.
    """
    return build_container()


def ensure_vector_store_ready() -> None:
    """Best-effort Elasticsearch index creation, called once at process
    startup by both api/main.py's lifespan and workers/celery_app.py.

    Never raises: a failure here (e.g. Elasticsearch not reachable yet) is
    logged and left for GET /health/ready and individual ingestion jobs to
    surface — it must not prevent the process itself from starting.
    """
    try:
        get_container().vector_store.ensure_indices()
    except Exception:
        logger.exception("Failed to ensure Elasticsearch indices exist at startup; will retry on first use")

def _build_embedder(settings: Settings) -> Embedder:
    """Factory: Instantiates the correct Embedder adapter based on config."""
    
    if settings.embedding_provider == "azureopenai":
        if not all([settings.azure_openai_endpoint, settings.azure_openai_api_key, settings.azure_openai_deployment_name]):
            raise ConfigurationError("Missing Azure OpenAI configuration variables in environment.")
            
        return AzureOpenAIEmbedder(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            deployment_name=settings.embedding_model,
            dimensions_override=settings.embedding_dimensions
        )
        
    elif settings.embedding_provider == "litellm":
        return LiteLLMEmbedder(
            model=settings.embedding_model,
            api_key=settings.litellm_api_key,
            api_base=settings.litellm_api_base,
            api_version=settings.litellm_api_version,
            provider=settings.litellm_provider,
            dimensions_override=settings.embedding_dimensions,
        )
        
    else:
        # Fallback für ungültige Konfiguration
        raise ConfigurationError(f"Unknown embedding_provider: {settings.embedding_provider}")
