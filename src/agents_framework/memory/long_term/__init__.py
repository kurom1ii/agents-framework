"""Long-term memory storage package.

This package provides persistent memory storage backends including
Redis for key-value storage and ChromaDB for vector-based semantic search.

Example:
    from agents_framework.memory.long_term import (
        RedisMemoryStore,
        RedisMemoryConfig,
        VectorMemoryStore,
        VectorMemoryConfig,
    )

    # Redis storage
    redis_config = RedisMemoryConfig(host="localhost", port=6379)
    async with RedisMemoryStore(redis_config) as redis_store:
        await redis_store.store(item)

    # Vector storage with ChromaDB
    vector_config = VectorMemoryConfig(persist_directory="./chroma_data")
    vector_store = VectorMemoryStore(vector_config, embedding_provider=provider)
    results = await vector_store.search("similar memories", limit=10)
"""

from .redis import (
    RedisMemoryConfig,
    RedisMemoryStore,
    RedisMemoryStoreError,
)
from .vector import (
    VectorMemoryConfig,
    VectorMemoryStore,
    VectorMemoryStoreError,
)

__all__ = [
    # Redis
    "RedisMemoryConfig",
    "RedisMemoryStore",
    "RedisMemoryStoreError",
    # Vector
    "VectorMemoryConfig",
    "VectorMemoryStore",
    "VectorMemoryStoreError",
]
