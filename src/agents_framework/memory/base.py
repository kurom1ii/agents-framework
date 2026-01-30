"""Memory protocols and base interfaces for the agents framework.

This module defines the core abstractions for memory storage and retrieval,
including the MemoryStore protocol, MemoryItem model, and MemoryQuery for
search operations.
"""

from abc import abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Types of memory storage."""

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    VECTOR = "vector"


class MemoryItem(BaseModel):
    """Represents a single memory item stored in any memory backend.

    Attributes:
        id: Unique identifier for the memory item.
        content: The actual content/text of the memory.
        metadata: Additional metadata associated with the memory.
        embedding: Optional vector embedding for semantic search.
        timestamp: When the memory was created.
        memory_type: The type of memory store this item belongs to.
        namespace: Optional namespace for organizing memories.
        ttl: Optional time-to-live in seconds.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    memory_type: MemoryType = MemoryType.SHORT_TERM
    namespace: Optional[str] = None
    ttl: Optional[int] = None

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class MemoryQuery(BaseModel):
    """Query parameters for retrieving memories.

    Attributes:
        query_text: Optional text to search for (semantic or keyword).
        namespace: Optional namespace to filter by.
        memory_type: Optional memory type filter.
        metadata_filters: Optional metadata key-value filters.
        limit: Maximum number of results to return.
        offset: Number of results to skip (for pagination).
        start_time: Optional start time for time-range queries.
        end_time: Optional end time for time-range queries.
        similarity_threshold: Minimum similarity score for vector search.
    """

    query_text: Optional[str] = None
    namespace: Optional[str] = None
    memory_type: Optional[MemoryType] = None
    metadata_filters: Dict[str, Any] = Field(default_factory=dict)
    limit: int = 10
    offset: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    similarity_threshold: float = 0.7


class MemoryConfig(BaseModel):
    """Configuration for memory stores.

    Attributes:
        namespace: Default namespace for this memory store.
        max_items: Maximum number of items to store (for bounded stores).
        default_ttl: Default TTL for items in seconds.
        enable_embeddings: Whether to compute embeddings for stored items.
    """

    namespace: str = "default"
    max_items: Optional[int] = None
    default_ttl: Optional[int] = None
    enable_embeddings: bool = False


@runtime_checkable
class MemoryStore(Protocol):
    """Protocol defining the interface for memory storage backends.

    All memory store implementations must adhere to this protocol,
    providing async methods for storing, retrieving, deleting, and
    clearing memory items.
    """

    @abstractmethod
    async def store(self, item: MemoryItem) -> str:
        """Store a memory item.

        Args:
            item: The MemoryItem to store.

        Returns:
            The ID of the stored item.
        """
        ...

    @abstractmethod
    async def retrieve(self, query: MemoryQuery) -> List[MemoryItem]:
        """Retrieve memory items matching the query.

        Args:
            query: The MemoryQuery specifying search criteria.

        Returns:
            List of matching MemoryItem objects.
        """
        ...

    @abstractmethod
    async def delete(self, item_id: str) -> bool:
        """Delete a memory item by ID.

        Args:
            item_id: The ID of the item to delete.

        Returns:
            True if the item was deleted, False if not found.
        """
        ...

    @abstractmethod
    async def clear(self, namespace: Optional[str] = None) -> None:
        """Clear all memory items, optionally filtered by namespace.

        Args:
            namespace: Optional namespace to clear. If None, clears all items.
        """
        ...

    @abstractmethod
    async def get(self, item_id: str) -> Optional[MemoryItem]:
        """Get a specific memory item by ID.

        Args:
            item_id: The ID of the item to retrieve.

        Returns:
            The MemoryItem if found, None otherwise.
        """
        ...

    @abstractmethod
    async def count(self, namespace: Optional[str] = None) -> int:
        """Count the number of stored items.

        Args:
            namespace: Optional namespace to count items in.

        Returns:
            The number of items.
        """
        ...


class Message(BaseModel):
    """Represents a conversation message.

    Attributes:
        role: The role of the message sender (user, assistant, system).
        content: The text content of the message.
        metadata: Additional metadata for the message.
        timestamp: When the message was created.
        token_count: Estimated token count for the message.
    """

    role: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    token_count: Optional[int] = None

    def estimate_tokens(self) -> int:
        """Estimate the number of tokens in this message.

        Uses a simple heuristic of ~4 characters per token.

        Returns:
            Estimated token count.
        """
        if self.token_count is not None:
            return self.token_count
        # Simple estimation: ~4 chars per token on average
        return len(self.content) // 4 + 1
