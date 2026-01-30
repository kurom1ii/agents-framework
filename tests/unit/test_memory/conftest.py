"""Local fixtures for memory module tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytest

from agents_framework.memory.base import (
    MemoryConfig,
    MemoryItem,
    MemoryQuery,
    MemoryStore,
    MemoryType,
    Message,
)
from agents_framework.memory.short_term import (
    SessionMemory,
    SessionMemoryConfig,
    SlidingWindowBuffer,
    NamespacedSessionStore,
)
from agents_framework.memory.manager import (
    MemoryManager,
    MemoryManagerConfig,
    MemoryManagerRegistry,
    MemoryTier,
    TierPriority,
)


# ============================================================================
# Message Fixtures
# ============================================================================


@pytest.fixture
def user_message() -> Message:
    """Create a sample user message."""
    return Message(
        role="user",
        content="Hello, how are you?",
        metadata={"source": "test"},
    )


@pytest.fixture
def assistant_message() -> Message:
    """Create a sample assistant message."""
    return Message(
        role="assistant",
        content="I'm doing well, thank you for asking!",
        metadata={"source": "test"},
    )


@pytest.fixture
def system_message() -> Message:
    """Create a sample system message."""
    return Message(
        role="system",
        content="You are a helpful assistant.",
        metadata={"source": "test"},
    )


@pytest.fixture
def conversation_messages() -> List[Message]:
    """Create a list of conversation messages."""
    return [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="What is Python?"),
        Message(
            role="assistant",
            content="Python is a high-level programming language known for its readability.",
        ),
        Message(role="user", content="Can you give me an example?"),
        Message(
            role="assistant",
            content="Sure! Here's a simple example: print('Hello, World!')",
        ),
    ]


@pytest.fixture
def long_message() -> Message:
    """Create a message with a lot of content for token testing."""
    # Create a message with approximately 1000 characters (~250 tokens)
    content = "This is a test message. " * 50
    return Message(role="user", content=content)


# ============================================================================
# Memory Item Fixtures
# ============================================================================


@pytest.fixture
def basic_memory_item() -> MemoryItem:
    """Create a basic memory item."""
    return MemoryItem(
        id="item-001",
        content="This is a basic memory item for testing.",
        metadata={"category": "test", "priority": 1},
        memory_type=MemoryType.SHORT_TERM,
        namespace="test-ns",
    )


@pytest.fixture
def memory_items_batch() -> List[MemoryItem]:
    """Create a batch of memory items with different properties."""
    items = []
    base_time = datetime.utcnow()

    for i in range(10):
        items.append(
            MemoryItem(
                id=f"batch-item-{i:03d}",
                content=f"Memory content number {i} with some searchable text",
                metadata={
                    "index": i,
                    "category": "batch" if i < 5 else "other",
                    "priority": i % 3,
                },
                memory_type=MemoryType.SHORT_TERM,
                namespace="batch-ns",
                timestamp=base_time - timedelta(minutes=i),
            )
        )

    return items


@pytest.fixture
def memory_item_with_embedding() -> MemoryItem:
    """Create a memory item with an embedding vector."""
    return MemoryItem(
        id="embed-item-001",
        content="This item has an embedding vector.",
        metadata={"has_embedding": True},
        embedding=[0.1, 0.2, 0.3, 0.4, 0.5] * 20,  # 100-dim vector
        memory_type=MemoryType.VECTOR,
        namespace="vector-ns",
    )


@pytest.fixture
def memory_item_with_ttl() -> MemoryItem:
    """Create a memory item with TTL."""
    return MemoryItem(
        id="ttl-item-001",
        content="This item has a TTL.",
        metadata={"expiring": True},
        memory_type=MemoryType.SHORT_TERM,
        namespace="ttl-ns",
        ttl=3600,  # 1 hour
    )


# ============================================================================
# Memory Query Fixtures
# ============================================================================


@pytest.fixture
def basic_memory_query() -> MemoryQuery:
    """Create a basic memory query."""
    return MemoryQuery(
        query_text="test query",
        namespace="test-ns",
        limit=10,
    )


@pytest.fixture
def time_range_query() -> MemoryQuery:
    """Create a query with time range filters."""
    now = datetime.utcnow()
    return MemoryQuery(
        query_text="search text",
        start_time=now - timedelta(hours=1),
        end_time=now,
        limit=20,
    )


@pytest.fixture
def metadata_filter_query() -> MemoryQuery:
    """Create a query with metadata filters."""
    return MemoryQuery(
        metadata_filters={"category": "test", "priority": 1},
        limit=5,
    )


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def basic_memory_config() -> MemoryConfig:
    """Create a basic memory configuration."""
    return MemoryConfig(
        namespace="test",
        max_items=100,
        default_ttl=3600,
        enable_embeddings=False,
    )


@pytest.fixture
def session_config() -> SessionMemoryConfig:
    """Create a session memory configuration."""
    return SessionMemoryConfig(
        max_tokens=4096,
        max_messages=100,
        preserve_system_messages=True,
        session_id="test-session-001",
        agent_id="test-agent-001",
    )


@pytest.fixture
def session_config_no_system_preserve() -> SessionMemoryConfig:
    """Create a session config that doesn't preserve system messages."""
    return SessionMemoryConfig(
        max_tokens=2048,
        max_messages=50,
        preserve_system_messages=False,
        session_id="test-session-002",
    )


@pytest.fixture
def small_buffer_config() -> SessionMemoryConfig:
    """Create a session config with a small token buffer for eviction testing."""
    return SessionMemoryConfig(
        max_tokens=100,  # Very small to trigger eviction
        max_messages=None,
        preserve_system_messages=True,
        session_id="small-buffer-session",
    )


@pytest.fixture
def manager_config() -> MemoryManagerConfig:
    """Create a memory manager configuration."""
    return MemoryManagerConfig(
        agent_id="test-agent",
        default_namespace="test-ns",
        short_term_enabled=True,
        long_term_enabled=False,
        vector_enabled=False,
        tier_priority=TierPriority.SHORT_TERM_FIRST,
        auto_promote=True,
        promotion_threshold=3,
        max_short_term_tokens=8192,
    )


@pytest.fixture
def manager_config_all_tiers() -> MemoryManagerConfig:
    """Create a manager config with all tiers enabled."""
    return MemoryManagerConfig(
        agent_id="multi-tier-agent",
        short_term_enabled=True,
        long_term_enabled=True,
        vector_enabled=True,
        tier_priority=TierPriority.SHORT_TERM_FIRST,
    )


# ============================================================================
# Store Fixtures
# ============================================================================


@pytest.fixture
def sliding_window_buffer() -> SlidingWindowBuffer:
    """Create a sliding window buffer."""
    return SlidingWindowBuffer(max_tokens=4096)


@pytest.fixture
def small_sliding_window_buffer() -> SlidingWindowBuffer:
    """Create a small sliding window buffer for eviction testing."""
    return SlidingWindowBuffer(max_tokens=50)


@pytest.fixture
def session_memory(session_config: SessionMemoryConfig) -> SessionMemory:
    """Create a session memory instance."""
    return SessionMemory(session_config)


@pytest.fixture
def session_memory_default() -> SessionMemory:
    """Create a session memory with default config."""
    return SessionMemory()


@pytest.fixture
def namespaced_session_store() -> NamespacedSessionStore:
    """Create a namespaced session store."""
    return NamespacedSessionStore()


@pytest.fixture
def memory_manager(manager_config: MemoryManagerConfig) -> MemoryManager:
    """Create a memory manager."""
    return MemoryManager(manager_config)


@pytest.fixture
def memory_manager_registry() -> MemoryManagerRegistry:
    """Create a memory manager registry."""
    return MemoryManagerRegistry()


# ============================================================================
# Mock Store for Testing
# ============================================================================


class InMemoryStore:
    """Simple in-memory store implementing MemoryStore protocol for testing."""

    def __init__(self):
        self.items: Dict[str, MemoryItem] = {}

    async def store(self, item: MemoryItem) -> str:
        self.items[item.id] = item
        return item.id

    async def retrieve(self, query: MemoryQuery) -> List[MemoryItem]:
        results = list(self.items.values())

        if query.namespace:
            results = [r for r in results if r.namespace == query.namespace]

        if query.query_text:
            results = [
                r for r in results
                if query.query_text.lower() in r.content.lower()
            ]

        if query.metadata_filters:
            results = [
                r for r in results
                if all(r.metadata.get(k) == v for k, v in query.metadata_filters.items())
            ]

        return results[query.offset : query.offset + query.limit]

    async def delete(self, item_id: str) -> bool:
        if item_id in self.items:
            del self.items[item_id]
            return True
        return False

    async def clear(self, namespace: Optional[str] = None) -> None:
        if namespace:
            self.items = {
                k: v for k, v in self.items.items()
                if v.namespace != namespace
            }
        else:
            self.items.clear()

    async def get(self, item_id: str) -> Optional[MemoryItem]:
        return self.items.get(item_id)

    async def count(self, namespace: Optional[str] = None) -> int:
        if namespace:
            return sum(1 for v in self.items.values() if v.namespace == namespace)
        return len(self.items)


@pytest.fixture
def in_memory_store() -> InMemoryStore:
    """Create an in-memory store for testing."""
    return InMemoryStore()


# ============================================================================
# Test Data Generators
# ============================================================================


def create_messages(count: int, role: str = "user") -> List[Message]:
    """Generate a list of test messages."""
    return [
        Message(
            role=role,
            content=f"Test message number {i}",
            metadata={"index": i},
        )
        for i in range(count)
    ]


def create_memory_items(
    count: int,
    namespace: str = "test",
    memory_type: MemoryType = MemoryType.SHORT_TERM,
) -> List[MemoryItem]:
    """Generate a list of test memory items."""
    return [
        MemoryItem(
            id=f"gen-item-{i:03d}",
            content=f"Generated content {i}",
            metadata={"index": i},
            memory_type=memory_type,
            namespace=namespace,
        )
        for i in range(count)
    ]
