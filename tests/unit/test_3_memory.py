"""Test 3: Memory System - Can agents remember?

This test verifies the memory subsystem:
- Session memory (short-term)
- Memory item storage/retrieval
- Message history management
"""

import pytest
from agents_framework.memory.base import MemoryItem, MemoryQuery, MemoryType, Message


class TestMemorySystemCore:
    """Test memory system core functionality."""

    def test_memory_item_creation(self):
        """MemoryItem can be created with required fields."""
        item = MemoryItem(
            content="User asked about the weather",
            metadata={"topic": "weather"},
        )

        assert item.content == "User asked about the weather"
        assert item.metadata["topic"] == "weather"
        assert item.id is not None  # Auto-generated
        assert item.memory_type == MemoryType.SHORT_TERM  # Default

    def test_memory_item_with_namespace(self):
        """MemoryItem supports namespacing."""
        item = MemoryItem(
            content="Important fact",
            namespace="agent_1",
            memory_type=MemoryType.LONG_TERM,
        )

        assert item.namespace == "agent_1"
        assert item.memory_type == MemoryType.LONG_TERM

    def test_memory_query_creation(self):
        """MemoryQuery specifies search criteria."""
        query = MemoryQuery(
            query_text="weather",
            namespace="agent_1",
            limit=5,
        )

        assert query.query_text == "weather"
        assert query.limit == 5
        assert query.namespace == "agent_1"

    def test_message_creation(self):
        """Message represents conversation messages."""
        msg = Message(
            role="user",
            content="What is the capital of France?",
        )

        assert msg.role == "user"
        assert msg.content == "What is the capital of France?"
        assert msg.timestamp is not None

    def test_message_token_estimation(self):
        """Message estimates token count."""
        msg = Message(
            role="assistant",
            content="Paris is the capital of France.",  # ~8 words, ~32 chars
        )

        tokens = msg.estimate_tokens()

        # ~4 chars per token heuristic
        assert tokens > 0
        assert tokens < 20  # Reasonable estimate

    def test_memory_type_enum(self):
        """MemoryType enum has expected values."""
        assert MemoryType.SHORT_TERM.value == "short_term"
        assert MemoryType.LONG_TERM.value == "long_term"
        assert MemoryType.VECTOR.value == "vector"

    def test_memory_item_with_ttl(self):
        """MemoryItem can have TTL for expiry."""
        item = MemoryItem(
            content="Temporary note",
            ttl=3600,  # 1 hour
        )

        assert item.ttl == 3600

    def test_memory_query_with_time_range(self):
        """MemoryQuery supports time-based filtering."""
        from datetime import datetime, timedelta

        now = datetime.utcnow()
        query = MemoryQuery(
            query_text="meeting",
            start_time=now - timedelta(days=7),
            end_time=now,
        )

        assert query.start_time is not None
        assert query.end_time is not None

    def test_memory_query_similarity_threshold(self):
        """MemoryQuery has similarity threshold for vector search."""
        query = MemoryQuery(
            query_text="semantic search",
            similarity_threshold=0.8,
        )

        assert query.similarity_threshold == 0.8
