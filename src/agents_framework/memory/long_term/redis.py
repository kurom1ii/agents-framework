"""Redis-based long-term memory store.

This module provides a Redis-backed implementation of the MemoryStore protocol
with connection pooling, key namespacing, and TTL support for persistent
agent memory.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..base import MemoryConfig, MemoryItem, MemoryQuery, MemoryStore, MemoryType

# Optional Redis import
try:
    import redis.asyncio as aioredis
    from redis.asyncio import ConnectionPool, Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None
    ConnectionPool = None
    Redis = None


class RedisMemoryConfig(MemoryConfig):
    """Configuration for Redis memory store.

    Attributes:
        host: Redis server host.
        port: Redis server port.
        db: Redis database number.
        password: Optional Redis password.
        ssl: Whether to use SSL.
        max_connections: Maximum connections in the pool.
        key_prefix: Prefix for all Redis keys.
        socket_timeout: Socket timeout in seconds.
        socket_connect_timeout: Connection timeout in seconds.
    """

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    max_connections: int = 10
    key_prefix: str = "agents_memory"
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0


class RedisMemoryStoreError(Exception):
    """Exception for Redis memory store errors."""
    pass


class RedisMemoryStore(MemoryStore):
    """Redis-backed memory store with connection pooling.

    Provides persistent storage for memory items using Redis with
    automatic key namespacing and TTL support.

    Attributes:
        config: Redis memory configuration.
    """

    def __init__(self, config: Optional[RedisMemoryConfig] = None):
        """Initialize Redis memory store.

        Args:
            config: Optional Redis configuration.

        Raises:
            ImportError: If redis package is not installed.
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "redis package is required for RedisMemoryStore. "
                "Install it with: pip install redis[hiredis]"
            )

        self.config = config or RedisMemoryConfig()
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[Redis] = None

    async def connect(self) -> None:
        """Establish connection to Redis with connection pooling."""
        if self._pool is not None:
            return

        self._pool = ConnectionPool(
            host=self.config.host,
            port=self.config.port,
            db=self.config.db,
            password=self.config.password,
            max_connections=self.config.max_connections,
            socket_timeout=self.config.socket_timeout,
            socket_connect_timeout=self.config.socket_connect_timeout,
            decode_responses=True,
        )
        self._client = Redis(connection_pool=self._pool)

        # Test connection
        try:
            await self._client.ping()
        except Exception as e:
            await self.disconnect()
            raise RedisMemoryStoreError(f"Failed to connect to Redis: {e}")

    async def disconnect(self) -> None:
        """Close Redis connection and pool."""
        if self._client:
            await self._client.close()
            self._client = None
        if self._pool:
            await self._pool.disconnect()
            self._pool = None

    async def _ensure_connected(self) -> Redis:
        """Ensure connection is established and return client.

        Returns:
            The Redis client.

        Raises:
            RedisMemoryStoreError: If not connected.
        """
        if self._client is None:
            await self.connect()
        return self._client  # type: ignore

    def _build_key(self, item_id: str, namespace: Optional[str] = None) -> str:
        """Build a namespaced Redis key.

        Args:
            item_id: The item ID.
            namespace: Optional namespace.

        Returns:
            The full Redis key.
        """
        parts = [self.config.key_prefix]
        ns = namespace or self.config.namespace
        if ns:
            parts.append(ns)
        parts.append(item_id)
        return ":".join(parts)

    def _build_index_key(self, namespace: Optional[str] = None) -> str:
        """Build the index key for a namespace.

        Args:
            namespace: Optional namespace.

        Returns:
            The index key.
        """
        parts = [self.config.key_prefix]
        ns = namespace or self.config.namespace
        if ns:
            parts.append(ns)
        parts.append("_index")
        return ":".join(parts)

    def _serialize_item(self, item: MemoryItem) -> str:
        """Serialize a memory item to JSON.

        Args:
            item: The memory item.

        Returns:
            JSON string.
        """
        data = item.model_dump()
        # Convert datetime to ISO format
        data["timestamp"] = item.timestamp.isoformat()
        return json.dumps(data)

    def _deserialize_item(self, data: str) -> MemoryItem:
        """Deserialize a memory item from JSON.

        Args:
            data: JSON string.

        Returns:
            The memory item.
        """
        parsed = json.loads(data)
        # Convert timestamp back to datetime
        if "timestamp" in parsed:
            parsed["timestamp"] = datetime.fromisoformat(parsed["timestamp"])
        return MemoryItem(**parsed)

    async def store(self, item: MemoryItem) -> str:
        """Store a memory item in Redis.

        Args:
            item: The MemoryItem to store.

        Returns:
            The ID of the stored item.
        """
        client = await self._ensure_connected()

        # Set namespace and type
        item.namespace = item.namespace or self.config.namespace
        item.memory_type = MemoryType.LONG_TERM

        key = self._build_key(item.id, item.namespace)
        index_key = self._build_index_key(item.namespace)

        # Serialize and store
        serialized = self._serialize_item(item)

        # Use pipeline for atomic operation
        async with client.pipeline() as pipe:
            if item.ttl or self.config.default_ttl:
                ttl = item.ttl or self.config.default_ttl
                pipe.setex(key, ttl, serialized)
            else:
                pipe.set(key, serialized)

            # Add to index set with timestamp score
            pipe.zadd(index_key, {item.id: item.timestamp.timestamp()})

            await pipe.execute()

        return item.id

    async def retrieve(self, query: MemoryQuery) -> List[MemoryItem]:
        """Retrieve memory items matching the query.

        Args:
            query: The MemoryQuery specifying search criteria.

        Returns:
            List of matching MemoryItem objects.
        """
        client = await self._ensure_connected()

        namespace = query.namespace or self.config.namespace
        index_key = self._build_index_key(namespace)

        # Get item IDs from index
        # Use time range if specified
        min_score = "-inf"
        max_score = "+inf"
        if query.start_time:
            min_score = str(query.start_time.timestamp())
        if query.end_time:
            max_score = str(query.end_time.timestamp())

        item_ids = await client.zrangebyscore(
            index_key,
            min_score,
            max_score,
            start=query.offset,
            num=query.limit * 2,  # Fetch extra for filtering
        )

        if not item_ids:
            return []

        # Fetch items
        results: List[MemoryItem] = []
        for item_id in item_ids:
            key = self._build_key(item_id, namespace)
            data = await client.get(key)
            if data:
                item = self._deserialize_item(data)

                # Apply filters
                if query.metadata_filters:
                    match = all(
                        item.metadata.get(k) == v
                        for k, v in query.metadata_filters.items()
                    )
                    if not match:
                        continue

                # Text search
                if query.query_text and query.query_text.lower() not in item.content.lower():
                    continue

                results.append(item)

                if len(results) >= query.limit:
                    break

        return results

    async def delete(self, item_id: str) -> bool:
        """Delete a memory item by ID.

        Args:
            item_id: The ID of the item to delete.

        Returns:
            True if the item was deleted, False if not found.
        """
        client = await self._ensure_connected()

        # Try to find the item first to get its namespace
        item = await self.get(item_id)
        if not item:
            return False

        key = self._build_key(item_id, item.namespace)
        index_key = self._build_index_key(item.namespace)

        async with client.pipeline() as pipe:
            pipe.delete(key)
            pipe.zrem(index_key, item_id)
            await pipe.execute()

        return True

    async def clear(self, namespace: Optional[str] = None) -> None:
        """Clear all memory items.

        Args:
            namespace: Optional namespace to clear. If None, clears all items.
        """
        client = await self._ensure_connected()

        ns = namespace or self.config.namespace
        pattern = f"{self.config.key_prefix}:{ns}:*" if ns else f"{self.config.key_prefix}:*"

        # Use SCAN to find keys
        cursor = 0
        while True:
            cursor, keys = await client.scan(cursor, match=pattern, count=100)
            if keys:
                await client.delete(*keys)
            if cursor == 0:
                break

    async def get(self, item_id: str) -> Optional[MemoryItem]:
        """Get a specific memory item by ID.

        Args:
            item_id: The ID of the item to retrieve.

        Returns:
            The MemoryItem if found, None otherwise.
        """
        client = await self._ensure_connected()

        # Try current namespace first
        key = self._build_key(item_id, self.config.namespace)
        data = await client.get(key)

        if data:
            return self._deserialize_item(data)

        # Search in all namespaces
        pattern = f"{self.config.key_prefix}:*:{item_id}"
        cursor = 0
        while True:
            cursor, keys = await client.scan(cursor, match=pattern, count=100)
            for key in keys:
                data = await client.get(key)
                if data:
                    return self._deserialize_item(data)
            if cursor == 0:
                break

        return None

    async def count(self, namespace: Optional[str] = None) -> int:
        """Count the number of stored items.

        Args:
            namespace: Optional namespace to count items in.

        Returns:
            The number of items.
        """
        client = await self._ensure_connected()

        ns = namespace or self.config.namespace
        index_key = self._build_index_key(ns)

        return await client.zcard(index_key)

    async def get_namespaces(self) -> List[str]:
        """Get all namespaces.

        Returns:
            List of namespace names.
        """
        client = await self._ensure_connected()

        # Find all index keys
        pattern = f"{self.config.key_prefix}:*:_index"
        namespaces = []

        cursor = 0
        while True:
            cursor, keys = await client.scan(cursor, match=pattern, count=100)
            for key in keys:
                # Extract namespace from key
                parts = key.split(":")
                if len(parts) >= 2:
                    ns = parts[1] if parts[1] != "_index" else "default"
                    if ns not in namespaces:
                        namespaces.append(ns)
            if cursor == 0:
                break

        return namespaces

    async def set_ttl(self, item_id: str, ttl: int) -> bool:
        """Set TTL for an existing item.

        Args:
            item_id: The item ID.
            ttl: Time-to-live in seconds.

        Returns:
            True if TTL was set, False if item not found.
        """
        client = await self._ensure_connected()

        item = await self.get(item_id)
        if not item:
            return False

        key = self._build_key(item_id, item.namespace)
        await client.expire(key, ttl)
        return True

    async def get_ttl(self, item_id: str) -> Optional[int]:
        """Get remaining TTL for an item.

        Args:
            item_id: The item ID.

        Returns:
            Remaining TTL in seconds, -1 if no TTL, None if not found.
        """
        client = await self._ensure_connected()

        item = await self.get(item_id)
        if not item:
            return None

        key = self._build_key(item_id, item.namespace)
        ttl = await client.ttl(key)
        return ttl if ttl >= -1 else None

    async def __aenter__(self) -> "RedisMemoryStore":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()
