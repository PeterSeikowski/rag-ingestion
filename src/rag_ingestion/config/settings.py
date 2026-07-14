"""Application configuration, sourced entirely from environment variables.

No hardcoded service URLs or credentials: every adapter and application
service receives configuration through a `Settings` instance built here
(see `bootstrap.py`), never by reading `os.environ` directly.
"""

from __future__ import annotations

from functools import lru_cache

from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated environment configuration.

    Field names are lower_snake_case; pydantic-settings matches them to
    environment variables case-insensitively (e.g. `redis_url` <-
    `REDIS_URL`), so the required env var names from the project spec work
    unchanged.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # .env.example documents optional fields as `KEY=` (empty) for
        # discoverability; without this, an empty string fails validation
        # for non-str optional fields (e.g. embedding_dimensions: int | None).
        env_ignore_empty=True,
    )

    # --- App ---
    app_env: str = "local"
    log_level: str = "INFO"

    # --- Redis / Celery ---
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    job_status_ttl_seconds: int = 86400

    # --- Elasticsearch (external; this repo never starts it) ---
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_username: str | None = None
    elasticsearch_password: str | None = None
    elasticsearch_verify_certs: bool = True
    es_documents_index: str = "rag_documents_v1"
    es_chunks_index: str = "rag_chunks_v1"
    es_embeddings_index: str = "rag_embeddings_v1"

    # --- Embeddings ---
    embedding_provider: Literal["litellm", "azureopenai"] = "azureopenai"
    embedding_model: Literal["text-embedding-3-large"] = "text-embedding-3-large"
    embedding_batch_size: int = 32
    embedding_dimensions: int | None = None
    # --- LiteLLM embeddings ---
    litellm_api_key: str | None = None
    litellm_api_base: str | None = None
    litellm_api_version: str | None = None
    litellm_provider: str | None = None
    # --- AzureOpenAI embeddings ---
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    


    # --- Chunking defaults ---
    default_chunk_size_tokens: int = 512
    default_chunk_overlap_tokens: int = 64

    # --- Parsing ---
    docling_enable_ocr: bool = False

    # --- Uploads (temporary local disk only; no object storage in MVP) ---
    temp_upload_dir: str = "/tmp/rag_ingestion_uploads"
    max_upload_size_bytes: int = 50 * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Process-wide cached Settings instance.

    Both `api/main.py` (via FastAPI `lifespan`) and `workers/celery_app.py`
    call this once at startup to build the shared `bootstrap` composition
    root — see IMPLEMENTATION_PLAN.md section 9 on why each process builds
    its own long-lived instance rather than sharing one across processes.
    """
    return Settings()
