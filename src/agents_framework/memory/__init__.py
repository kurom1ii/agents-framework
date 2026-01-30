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

    # Use session memory for conversations
    from agents_framework.memory import SessionMemory, SessionMemoryConfig

    session = SessionMemory(SessionMemoryConfig(max_tokens=4096))
    session.add_message(Message(role="user", content="Hello!"))

    # Use the unified memory manager
    from agents_framework.memory import MemoryManager, MemoryManagerConfig

    config = MemoryManagerConfig(agent_id="agent-1")
    manager = MemoryManager(config)
    await manager.store("Important fact", tier=MemoryTier.LONG_TERM)
"""

from .base import (
    MemoryType,
    MemoryItem,
    MemoryQuery,
    MemoryConfig,
    MemoryStore,
    Message,
)

from .short_term import (
    SlidingWindowBuffer,
    SessionMemoryConfig,
    SessionMemory,
    NamespacedSessionStore,
)

from .manager import (
    TierPriority,
    MemoryTier,
    MemoryManagerConfig,
    MemorySearchResult,
    MemoryManager,
    MemoryManagerRegistry,
)

# Re-export subpackages for convenience
from . import embeddings
from . import long_term

__all__ = [
    # Base types and protocols
    "MemoryType",
    "MemoryItem",
    "MemoryQuery",
    "MemoryConfig",
    "MemoryStore",
    "Message",
    # Short-term memory
    "SlidingWindowBuffer",
    "SessionMemoryConfig",
    "SessionMemory",
    "NamespacedSessionStore",
    # Memory manager
    "TierPriority",
    "MemoryTier",
    "MemoryManagerConfig",
    "MemorySearchResult",
    "MemoryManager",
    "MemoryManagerRegistry",
    # Subpackages
    "embeddings",
    "long_term",
]
