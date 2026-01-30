"""Embedding provider interfaces and implementations.

This module defines the EmbeddingProvider protocol and provides implementations
for OpenAI and Ollama embedding services, along with an embedding cache for
improved efficiency.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

# Optional imports for providers
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


@dataclass
class EmbeddingConfig:
    """Configuration for embedding providers.

    Attributes:
        model: The embedding model to use.
        api_key: API key for the provider.
        base_url: Base URL for the API.
        dimensions: Output embedding dimensions (if supported).
        batch_size: Maximum batch size for embedding requests.
        timeout: Request timeout in seconds.
    """

    model: str = "text-embedding-3-small"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    dimensions: Optional[int] = None
    batch_size: int = 100
    timeout: float = 60.0


@dataclass
class EmbeddingResult:
    """Result of an embedding operation.

    Attributes:
        embedding: The embedding vector.
        text: The original text that was embedded.
        model: The model used for embedding.
        usage: Token usage information.
    """

    embedding: List[float]
    text: str
    model: str
    usage: Optional[Dict[str, int]] = None


class EmbeddingProviderError(Exception):
    """Base exception for embedding provider errors."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers.

    All embedding providers must implement this interface to be used
    with the memory system.
    """

    config: EmbeddingConfig

    async def embed(self, text: str) -> List[float]:
        """Generate an embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector.
        """
        ...

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        ...

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        ...


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._client: Any = None

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate an embedding for a single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        pass


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider.

    Supports:
    - text-embedding-3-small (default)
    - text-embedding-3-large
    - text-embedding-ada-002 (legacy)
    """

    # Default dimensions for OpenAI models
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, config: EmbeddingConfig):
        """Initialize OpenAI embedding provider.

        Args:
            config: Embedding configuration.

        Raises:
            ImportError: If openai package is not installed.
        """
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required for OpenAIEmbeddingProvider. "
                "Install it with: pip install openai"
            )
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )
        self._dimensions = config.dimensions or self.MODEL_DIMENSIONS.get(
            config.model, 1536
        )

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        return self._dimensions

    async def embed(self, text: str) -> List[float]:
        """Generate an embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector.
        """
        try:
            request_params: Dict[str, Any] = {
                "model": self.config.model,
                "input": text,
            }
            if self.config.dimensions and "3-" in self.config.model:
                request_params["dimensions"] = self.config.dimensions

            response = await self._client.embeddings.create(**request_params)
            return response.data[0].embedding
        except Exception as e:
            raise EmbeddingProviderError(
                f"OpenAI embedding error: {str(e)}",
                provider="openai",
            )

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        embeddings: List[List[float]] = []
        batch_size = self.config.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                request_params: Dict[str, Any] = {
                    "model": self.config.model,
                    "input": batch,
                }
                if self.config.dimensions and "3-" in self.config.model:
                    request_params["dimensions"] = self.config.dimensions

                response = await self._client.embeddings.create(**request_params)
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                raise EmbeddingProviderError(
                    f"OpenAI batch embedding error: {str(e)}",
                    provider="openai",
                )

        return embeddings


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Ollama embedding provider for local models.

    Supports:
    - nomic-embed-text
    - mxbai-embed-large
    - all-minilm
    - Other Ollama-supported embedding models
    """

    DEFAULT_BASE_URL = "http://localhost:11434"

    # Approximate dimensions for common models
    MODEL_DIMENSIONS = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
        "snowflake-arctic-embed": 1024,
    }

    def __init__(self, config: EmbeddingConfig):
        """Initialize Ollama embedding provider.

        Args:
            config: Embedding configuration.

        Raises:
            ImportError: If httpx package is not installed.
        """
        super().__init__(config)
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx package is required for OllamaEmbeddingProvider. "
                "Install it with: pip install httpx"
            )
        self._base_url = config.base_url or self.DEFAULT_BASE_URL
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=config.timeout,
        )
        self._dimensions = config.dimensions or self.MODEL_DIMENSIONS.get(
            config.model, 768
        )

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        return self._dimensions

    async def embed(self, text: str) -> List[float]:
        """Generate an embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector.
        """
        try:
            response = await self._client.post(
                "/api/embeddings",
                json={
                    "model": self.config.model,
                    "prompt": text,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("embedding", [])
        except Exception as e:
            raise EmbeddingProviderError(
                f"Ollama embedding error: {str(e)}",
                provider="ollama",
            )

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Ollama doesn't support batch embedding natively, so we process
        texts concurrently with rate limiting.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        # Process in parallel with semaphore for rate limiting
        semaphore = asyncio.Semaphore(self.config.batch_size)

        async def embed_with_semaphore(text: str) -> List[float]:
            async with semaphore:
                return await self.embed(text)

        tasks = [embed_with_semaphore(text) for text in texts]
        return await asyncio.gather(*tasks)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


@dataclass
class CacheEntry:
    """Entry in the embedding cache.

    Attributes:
        embedding: The cached embedding vector.
        timestamp: When the entry was created.
        hits: Number of cache hits.
    """

    embedding: List[float]
    timestamp: float
    hits: int = 0


class EmbeddingCache:
    """LRU cache for embeddings.

    Provides caching of embeddings to reduce API calls and improve
    performance for repeated texts.

    Attributes:
        max_size: Maximum number of entries in the cache.
        ttl: Time-to-live for cache entries in seconds.
    """

    def __init__(
        self,
        max_size: int = 10000,
        ttl: Optional[int] = 3600,
    ):
        """Initialize the embedding cache.

        Args:
            max_size: Maximum number of entries to cache.
            ttl: Time-to-live in seconds (None for no expiration).
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []

    def _compute_key(self, text: str, model: str) -> str:
        """Compute a cache key for text and model.

        Args:
            text: The text to embed.
            model: The embedding model.

        Returns:
            The cache key.
        """
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get an embedding from the cache.

        Args:
            text: The text that was embedded.
            model: The embedding model.

        Returns:
            The cached embedding if found and valid, None otherwise.
        """
        key = self._compute_key(text, model)
        entry = self._cache.get(key)

        if entry is None:
            return None

        # Check TTL
        if self.ttl is not None:
            if time.time() - entry.timestamp > self.ttl:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return None

        # Update access order (LRU)
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        entry.hits += 1

        return entry.embedding

    def set(self, text: str, model: str, embedding: List[float]) -> None:
        """Store an embedding in the cache.

        Args:
            text: The text that was embedded.
            model: The embedding model.
            embedding: The embedding vector.
        """
        key = self._compute_key(text, model)

        # Evict if at capacity
        while len(self._cache) >= self.max_size and self._access_order:
            oldest_key = self._access_order.pop(0)
            self._cache.pop(oldest_key, None)

        self._cache[key] = CacheEntry(
            embedding=embedding,
            timestamp=time.time(),
        )
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._cache.clear()
        self._access_order.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        total_hits = sum(entry.hits for entry in self._cache.values())
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "total_hits": total_hits,
            "ttl": self.ttl,
        }


class CachedEmbeddingProvider:
    """Embedding provider wrapper with caching.

    Wraps an embedding provider with a cache layer to reduce
    redundant API calls.
    """

    def __init__(
        self,
        provider: BaseEmbeddingProvider,
        cache: Optional[EmbeddingCache] = None,
    ):
        """Initialize cached embedding provider.

        Args:
            provider: The underlying embedding provider.
            cache: Optional cache instance (creates default if not provided).
        """
        self._provider = provider
        self._cache = cache or EmbeddingCache()

    @property
    def config(self) -> EmbeddingConfig:
        """Get the underlying provider's config."""
        return self._provider.config

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        return self._provider.dimensions

    async def embed(self, text: str) -> List[float]:
        """Generate an embedding with cache lookup.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector.
        """
        # Check cache first
        cached = self._cache.get(text, self._provider.config.model)
        if cached is not None:
            return cached

        # Generate embedding
        embedding = await self._provider.embed(text)

        # Cache the result
        self._cache.set(text, self._provider.config.model, embedding)

        return embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with cache lookup.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # Check cache for each text
        for i, text in enumerate(texts):
            cached = self._cache.get(text, self._provider.config.model)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Embed uncached texts
        if uncached_texts:
            embeddings = await self._provider.embed_batch(uncached_texts)
            for i, embedding in zip(uncached_indices, embeddings):
                results[i] = embedding
                self._cache.set(texts[i], self._provider.config.model, embedding)

        return [r for r in results if r is not None]

    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.stats()

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
