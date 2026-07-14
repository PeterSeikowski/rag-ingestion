"""Azure OpenAI-based embedding adapter.

The only file allowed to import the `openai` package for embeddings.
Turns batches of text into vectors via Azure's specific endpoints.
Fails fast if configuration is missing and ensures precise mapping 
between input texts and returned vectors.
"""

from __future__ import annotations

import logging
from typing import Any

from rag_ingestion.domain.embeddings import EmbeddingConfig

logger = logging.getLogger(__name__)

# Azure allows custom deployment names (e.g., "my-din-embedding-model").
# If the deployment name matches standard model names, we can guess the dimensions.
# Otherwise, the application layer should pass dimensions_override, or we 
# dynamically learn it after the first API call.
_KNOWN_AZURE_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class AzureOpenAIEmbedder:
    """Embedder adapter directly backed by the official Azure OpenAI SDK."""

    def __init__(
        self,
        *,
        azure_endpoint: str,
        api_key: str,
        deployment_name: str,
        api_version: str = "2024-12-01-preview",
        dimensions_override: int | None = None,
    ) -> None:
        """
        Args:
            azure_endpoint: The base URL for your Azure OpenAI resource.
            api_key: Azure API key.
            deployment_name: The custom name you gave the model deployment in Azure.
            api_version: Azure REST API version.
            dimensions_override: Explicit dimension count if using a custom model name.
        """
        if not azure_endpoint or not api_key or not deployment_name:
            raise ValueError(
                "AzureOpenAIEmbedder requires azure_endpoint, api_key, and deployment_name."
            )

        self.name = "azure_openai_embedder"
        self.provider = "azure"
        self.model = deployment_name  # In Azure, the deployment name acts as the model parameter

        self._azure_endpoint = azure_endpoint
        self._api_key = api_key
        self._api_version = api_version
        self._dimensions = dimensions_override or _KNOWN_AZURE_DIMENSIONS.get(deployment_name)
        
        # Defer client instantiation so we don't import 'openai' or make network calls at import time
        self._client: Any = None

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    def _get_client(self) -> Any:
        """Lazy-loads the AzureOpenAI client."""
        if self._client is None:
            from openai import AzureOpenAI  # Imported lazily

            self._client = AzureOpenAI(
                api_key=self._api_key,
                api_version=self._api_version,
                azure_endpoint=self._azure_endpoint,
            )
        return self._client

    def embed_texts(self, texts: list[str], config: EmbeddingConfig) -> list[list[float]]:
        """Embeds a batch of texts using Azure OpenAI."""
        if not texts:
            return []

        client = self._get_client()
        vectors: list[list[float]] = []
        
        # Protect against Azure's strict rate limits and max input array sizes
        batch_size = config.batch_size or len(texts)

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            
            try:
                response = client.embeddings.create(
                    input=batch,
                    model=self.model,
                    **config.extra_params
                )
            except Exception as e:
                # Log the specific Azure error before raising a clean RuntimeError for the app layer
                logger.error("Azure OpenAI API error during embedding", exc_info=True)
                raise RuntimeError(f"Failed to fetch embeddings from Azure: {e}") from e

            # Extract vectors directly from the standardized Pydantic response of the new OpenAI SDK
            batch_vectors = [item.embedding for item in response.data]

            if len(batch_vectors) != len(batch):
                raise RuntimeError(
                    f"Azure returned {len(batch_vectors)} vectors for {len(batch)} input "
                    f"texts (deployment={self.model!r})"
                )

            vectors.extend(batch_vectors)

        # Dynamically cache dimensions if they were unknown at startup
        if self._dimensions is None and vectors:
            self._dimensions = len(vectors[0])

        return vectors