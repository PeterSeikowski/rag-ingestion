"""Every adapter can be instantiated directly with plain/mocked
dependencies — no live Redis/Elasticsearch/Docling/LiteLLM required.
Complements test_ingestion_pipeline.py, which exercises them through the
AppContainer/port interfaces rather than the concrete classes directly.
"""

from __future__ import annotations

import pytest

from rag_ingestion.adapters.chunkers.recursive_text import RecursiveTextChunker
from rag_ingestion.adapters.chunkers.section_based_hierarchical import SectionBasedHierarchicalChunker
from rag_ingestion.adapters.embedders.litellm_embedder import LiteLLMEmbedder
from rag_ingestion.adapters.jobs.redis_job_repository import RedisJobRepository
from rag_ingestion.adapters.parsers.docling_pdf_parser import DoclingPdfParser
from rag_ingestion.adapters.vectorstores.elasticsearch_writer import ElasticsearchVectorStoreWriter
from rag_ingestion.application.errors import ConfigurationError
from rag_ingestion.bootstrap import build_container
from rag_ingestion.config.settings import Settings


def test_docling_pdf_parser_instantiates():
    parser = DoclingPdfParser(enable_ocr=False)
    assert parser.name == "docling_pdf"
    assert parser.supported_formats == ["pdf"]


def test_chunkers_instantiate():
    assert RecursiveTextChunker().name == "recursive_text"
    assert SectionBasedHierarchicalChunker().name == "section_based_hierarchical"


def test_litellm_embedder_instantiates_with_valid_config():
    embedder = LiteLLMEmbedder(model="text-embedding-3-small", api_key="sk-fake")
    assert embedder.name == "litellm_embedder"
    assert embedder.dimensions == 1536


def test_redis_job_repository_instantiates_without_connecting():
    # redis-py's transport is lazy — construction alone must not require a
    # live Redis server (if it weren't, this call itself would raise).
    repo = RedisJobRepository("redis://localhost:6379/0", ttl_seconds=3600)
    assert repo._ttl_seconds == 3600
    assert repo._redis_url == "redis://localhost:6379/0"


def test_elasticsearch_vector_store_writer_instantiates_without_connecting():
    # elasticsearch-py's transport is likewise lazy at construction time
    # (if it weren't, this call itself would raise).
    writer = ElasticsearchVectorStoreWriter(
        elasticsearch_url="http://localhost:9200",
        documents_index="rag_documents_v1",
        chunks_index="rag_chunks_v1",
        embeddings_index="rag_embeddings_v1",
        embedding_dimensions=1536,
    )
    assert writer._documents_index == "rag_documents_v1"
    assert writer._embedding_dimensions == 1536


def test_build_container_constructs_all_adapters():
    settings = Settings(litellm_model="text-embedding-3-small", litellm_api_key="sk-fake")
    container = build_container(settings)

    assert container.parser is not None
    assert container.chunkers
    assert container.embedder is not None
    assert container.vector_store is not None
    assert container.job_repository is not None


def test_build_container_raises_value_error_when_api_key_missing():
    settings = Settings(litellm_model="text-embedding-3-small", litellm_api_key=None)
    with pytest.raises(ValueError):
        build_container(settings)


def test_build_container_raises_configuration_error_when_dimensions_unresolvable():
    settings = Settings(litellm_model="totally-unknown-model", litellm_api_key="sk-fake", embedding_dimensions=None)
    with pytest.raises(ConfigurationError):
        build_container(settings)


def test_build_container_respects_explicit_dimensions_override_for_unknown_model():
    settings = Settings(litellm_model="totally-unknown-model", litellm_api_key="sk-fake", embedding_dimensions=768)
    container = build_container(settings)
    assert container.embedder.dimensions == 768
