"""Embedding providers package.

This package provides embedding generation capabilities for semantic memory,
including OpenAI and Ollama providers with caching support.

Example:
    from agents_framework.memory.embeddings import (
        EmbeddingConfig,
        OpenAIEmbeddingProvider,
        CachedEmbeddingProvider,
    )

    # Create an OpenAI embedding provider
    config = EmbeddingConfig(model="text-embedding-3-small", api_key="...")
    provider = OpenAIEmbeddingProvider(config)

    # Wrap with caching
    cached_provider = CachedEmbeddingProvider(provider)

    # Generate embeddings
    embedding = await cached_provider.embed("Hello, world!")
"""

from .base import (
    EmbeddingConfig,
    EmbeddingResult,
    EmbeddingProvider,
    EmbeddingProviderError,
    BaseEmbeddingProvider,
    OpenAIEmbeddingProvider,
    OllamaEmbeddingProvider,
    EmbeddingCache,
    CachedEmbeddingProvider,
)

__all__ = [
    # Configuration
    "EmbeddingConfig",
    "EmbeddingResult",
    # Protocol
    "EmbeddingProvider",
    # Errors
    "EmbeddingProviderError",
    # Base class
    "BaseEmbeddingProvider",
    # Providers
    "OpenAIEmbeddingProvider",
    "OllamaEmbeddingProvider",
    # Caching
    "EmbeddingCache",
    "CachedEmbeddingProvider",
]
