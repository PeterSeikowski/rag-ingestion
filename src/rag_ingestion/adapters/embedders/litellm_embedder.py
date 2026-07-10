"""LiteLLM-based embedding adapter.

The only file allowed to import `litellm`. Turns batches of text into
vectors — nothing more; it has no concept of "chunk" or "owner" (see
ports/embedder.py). Fails fast with a plain ValueError if required
configuration is missing, and never silently returns fewer/more vectors
than input texts.

Deliberately raises plain exceptions rather than the
application.errors.IngestionError hierarchy: adapters depend only on
ports/domain, never on application/ — dependencies point inward.
application.ingestion_pipeline (Step 11) is the single place that
classifies a failure as transient/permanent for Celery's retry policy.
"""

from __future__ import annotations

import logging

from rag_ingestion.domain.embeddings import EmbeddingConfig

logger = logging.getLogger(__name__)

# TODO(litellm-dims): static fallback for embedding models whose
# dimensions LiteLLM doesn't expose without a live call. Extend as needed;
# an explicit dimensions_override (from EMBEDDING_DIMENSIONS in Settings)
# always takes precedence over this table.
_KNOWN_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
}


class LiteLLMEmbedder:
    """Embedder adapter backed by LiteLLM, giving this service one
    interface across OpenAI/Azure/Cohere/self-hosted embedding providers.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        api_base: str | None = None,
        api_version: str | None = None,
        provider: str | None = None,
        dimensions_override: int | None = None,
    ) -> None:
        if not model:
            raise ValueError("LiteLLMEmbedder requires a model (LITELLM_MODEL)")
        if not api_key:
            raise ValueError("LiteLLMEmbedder requires an api_key (LITELLM_API_KEY)")

        self.name = "litellm_embedder"
        self.model = model
        self.provider = provider or "litellm"
        self._api_key = api_key
        self._api_base = api_base
        self._api_version = api_version
        self._dimensions = dimensions_override or _KNOWN_MODEL_DIMENSIONS.get(model)

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    def embed_texts(self, texts: list[str], config: EmbeddingConfig) -> list[list[float]]:
        """Embed `texts` in batches of `config.batch_size`, preserving
        input order. Raises RuntimeError if a provider response doesn't
        contain the expected number of vectors for a batch.
        """
        if not texts:
            return []

        import litellm  # imported lazily; keeps this module importable without litellm installed

        vectors: list[list[float]] = []
        batch_size = config.batch_size or len(texts)
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = litellm.embedding(
                model=self.model,
                input=batch,
                api_key=self._api_key,
                api_base=self._api_base,
                api_version=self._api_version,
                **config.extra_params,
            )
            batch_vectors = [_extract_vector(item) for item in response.data]
            if len(batch_vectors) != len(batch):
                raise RuntimeError(
                    f"Embedding provider returned {len(batch_vectors)} vectors for {len(batch)} input "
                    f"texts (model={self.model!r})"
                )
            vectors.extend(batch_vectors)

        if self._dimensions is None and vectors:
            self._dimensions = len(vectors[0])

        return vectors


def _extract_vector(item: object) -> list[float]:
    """LiteLLM's embedding response mirrors OpenAI's shape, where each item
    in `.data` supports both dict-style and attribute access. Handle both
    defensively rather than assuming one. # TODO(litellm-api): re-verify
    against the pinned litellm version.
    """
    if isinstance(item, dict):
        return item["embedding"]
    return item.embedding
