"""LiteLLMEmbedder: fail-fast config validation, dims resolution, batching,
order preservation, and provider-response error handling — against a
hand-built fake `litellm` module (the real package is not installed in
this environment; see requirements.txt).
"""

from __future__ import annotations

import sys
import types

import pytest

from rag_ingestion.adapters.embedders.litellm_embedder import LiteLLMEmbedder
from rag_ingestion.domain.embeddings import EmbeddingConfig


def test_missing_model_raises_value_error():
    with pytest.raises(ValueError, match="model"):
        LiteLLMEmbedder(model="", api_key="k")


def test_missing_api_key_raises_value_error():
    with pytest.raises(ValueError, match="api_key"):
        LiteLLMEmbedder(model="text-embedding-3-small", api_key="")


def test_dimensions_resolved_from_known_model_table():
    embedder = LiteLLMEmbedder(model="text-embedding-3-small", api_key="sk-fake")
    assert embedder.dimensions == 1536


def test_dimensions_none_for_unknown_model_before_first_embed():
    embedder = LiteLLMEmbedder(model="some-custom-model", api_key="sk-fake")
    assert embedder.dimensions is None


def test_dimensions_override_takes_precedence_over_known_table():
    embedder = LiteLLMEmbedder(model="text-embedding-3-small", api_key="sk-fake", dimensions_override=42)
    assert embedder.dimensions == 42


@pytest.fixture
def fake_litellm(monkeypatch):
    calls: list[list[str]] = []

    def fake_embedding(model, input, api_key, api_base, api_version, **extra):
        calls.append(list(input))
        return types.SimpleNamespace(data=[{"embedding": [float(len(t)), 0.5, 0.25]} for t in input])

    module = types.ModuleType("litellm")
    module.embedding = fake_embedding
    monkeypatch.setitem(sys.modules, "litellm", module)
    return calls


def test_embed_texts_batches_and_preserves_order(fake_litellm):
    embedder = LiteLLMEmbedder(model="some-custom-model", api_key="sk-fake")
    config = EmbeddingConfig(model="some-custom-model", batch_size=2)
    texts = ["a", "bb", "ccc", "dddd", "e"]

    vectors = embedder.embed_texts(texts, config)

    assert len(vectors) == len(texts)
    assert fake_litellm == [["a", "bb"], ["ccc", "dddd"], ["e"]]
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0, 4.0, 1.0]


def test_embed_texts_infers_dimensions_from_first_response(fake_litellm):
    embedder = LiteLLMEmbedder(model="some-custom-model", api_key="sk-fake")
    assert embedder.dimensions is None
    embedder.embed_texts(["a"], EmbeddingConfig(model="some-custom-model"))
    assert embedder.dimensions == 3


def test_embed_texts_empty_input_short_circuits(fake_litellm):
    embedder = LiteLLMEmbedder(model="some-custom-model", api_key="sk-fake")
    assert embedder.embed_texts([], EmbeddingConfig(model="some-custom-model")) == []
    assert fake_litellm == []


def test_embed_texts_vector_count_mismatch_raises_runtime_error(monkeypatch):
    def broken_embedding(model, input, api_key, api_base, api_version, **extra):
        return types.SimpleNamespace(data=[{"embedding": [0.1]}])

    module = types.ModuleType("litellm")
    module.embedding = broken_embedding
    monkeypatch.setitem(sys.modules, "litellm", module)

    embedder = LiteLLMEmbedder(model="some-custom-model", api_key="sk-fake")
    with pytest.raises(RuntimeError):
        embedder.embed_texts(["x", "y"], EmbeddingConfig(model="some-custom-model"))


def test_embed_texts_handles_attribute_style_response_items(monkeypatch):
    def attr_style_embedding(model, input, api_key, api_base, api_version, **extra):
        item = types.SimpleNamespace(embedding=[9.0, 9.0, 9.0])
        return types.SimpleNamespace(data=[item for _ in input])

    module = types.ModuleType("litellm")
    module.embedding = attr_style_embedding
    monkeypatch.setitem(sys.modules, "litellm", module)

    embedder = LiteLLMEmbedder(model="some-custom-model", api_key="sk-fake")
    vectors = embedder.embed_texts(["x"], EmbeddingConfig(model="some-custom-model"))
    assert vectors == [[9.0, 9.0, 9.0]]


def test_embed_texts_against_real_litellm_response_types(monkeypatch):
    # The real `litellm` package (unlike `docling`) is light enough to
    # install in this environment — this test mocks only the network call
    # (litellm.embedding itself) but constructs its return value from
    # litellm's *real* response classes, giving real confidence in
    # _extract_vector's dict-vs-attribute handling rather than relying
    # solely on a hand-built types.SimpleNamespace fake.
    litellm = pytest.importorskip("litellm")
    from litellm.types.utils import Embedding, EmbeddingResponse

    def real_shaped_embedding(model, input, api_key, api_base, api_version, **extra):
        return EmbeddingResponse(
            model=model,
            data=[Embedding(embedding=[1.0, 2.0, 3.0], index=i, object="embedding") for i, _ in enumerate(input)],
            object="list",
        )

    monkeypatch.setattr(litellm, "embedding", real_shaped_embedding)

    embedder = LiteLLMEmbedder(model="text-embedding-3-small", api_key="sk-fake")
    vectors = embedder.embed_texts(["a", "b"], EmbeddingConfig(model="text-embedding-3-small"))
    assert vectors == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
