"""Memory package for the agents framework.

This package provides memory storage and retrieval capabilities for agents,
including short-term session memory, long-term persistent memory, and
vector-based semantic memory.

Example:
    from agents_framework.memory import MemoryStore, MemoryItem, MemoryQuery

    # Create a memory item
    item = MemoryItem(
        content="User prefers Python for data analysis",
        metadata={"topic": "preferences", "importance": "high"},
    )

    # Store and retrieve memories
    await store.store(item)
    results = await store.retrieve(MemoryQuery(query_text="preferences"))
"""

from .base import (
    MemoryType,
    MemoryItem,
    MemoryQuery,
    MemoryConfig,
    MemoryStore,
    Message,
)

__all__ = [
    # Types
    "MemoryType",
    # Models
    "MemoryItem",
    "MemoryQuery",
    "MemoryConfig",
    "Message",
    # Protocols
    "MemoryStore",
]
