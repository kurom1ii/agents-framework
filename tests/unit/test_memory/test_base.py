"""Unit tests for memory base types and interfaces.

Tests for:
- MemoryType enum
- MemoryItem model
- MemoryQuery model
- MemoryConfig model
- Message model and estimate_tokens()
"""

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


# ============================================================================
# MemoryType Enum Tests
# ============================================================================


class TestMemoryType:
    """Tests for the MemoryType enum."""

    def test_memory_type_values(self):
        """Test that MemoryType has expected values."""
        assert MemoryType.SHORT_TERM == "short_term"
        assert MemoryType.LONG_TERM == "long_term"
        assert MemoryType.VECTOR == "vector"

    def test_memory_type_from_string(self):
        """Test creating MemoryType from string value."""
        assert MemoryType("short_term") == MemoryType.SHORT_TERM
        assert MemoryType("long_term") == MemoryType.LONG_TERM
        assert MemoryType("vector") == MemoryType.VECTOR

    def test_memory_type_invalid_value(self):
        """Test that invalid value raises ValueError."""
        with pytest.raises(ValueError):
            MemoryType("invalid_type")

    def test_memory_type_is_string_enum(self):
        """Test that MemoryType is a string enum."""
        assert isinstance(MemoryType.SHORT_TERM, str)
        assert isinstance(MemoryType.LONG_TERM, str)
        assert isinstance(MemoryType.VECTOR, str)

    def test_memory_type_iteration(self):
        """Test iterating over MemoryType values."""
        types = list(MemoryType)
        assert len(types) == 3
        assert MemoryType.SHORT_TERM in types
        assert MemoryType.LONG_TERM in types
        assert MemoryType.VECTOR in types


# ============================================================================
# MemoryItem Model Tests
# ============================================================================


class TestMemoryItem:
    """Tests for the MemoryItem model."""

    def test_memory_item_creation_minimal(self):
        """Test creating MemoryItem with only required fields."""
        item = MemoryItem(content="Test content")

        assert item.content == "Test content"
        assert item.id is not None  # Auto-generated UUID
        assert isinstance(item.id, str)
        assert len(item.id) == 36  # UUID format
        assert item.metadata == {}
        assert item.embedding is None
        assert isinstance(item.timestamp, datetime)
        assert item.memory_type == MemoryType.SHORT_TERM
        assert item.namespace is None
        assert item.ttl is None

    def test_memory_item_creation_full(self):
        """Test creating MemoryItem with all fields."""
        timestamp = datetime.utcnow()
        embedding = [0.1, 0.2, 0.3]

        item = MemoryItem(
            id="custom-id-123",
            content="Full test content",
            metadata={"key": "value", "number": 42},
            embedding=embedding,
            timestamp=timestamp,
            memory_type=MemoryType.LONG_TERM,
            namespace="test-namespace",
            ttl=3600,
        )

        assert item.id == "custom-id-123"
        assert item.content == "Full test content"
        assert item.metadata == {"key": "value", "number": 42}
        assert item.embedding == embedding
        assert item.timestamp == timestamp
        assert item.memory_type == MemoryType.LONG_TERM
        assert item.namespace == "test-namespace"
        assert item.ttl == 3600

    def test_memory_item_default_id_uniqueness(self):
        """Test that default IDs are unique."""
        items = [MemoryItem(content=f"Content {i}") for i in range(100)]
        ids = [item.id for item in items]
        assert len(set(ids)) == 100  # All unique

    def test_memory_item_timestamp_default(self):
        """Test that timestamp defaults to current time."""
        before = datetime.utcnow()
        item = MemoryItem(content="Test")
        after = datetime.utcnow()

        assert before <= item.timestamp <= after

    @pytest.mark.parametrize(
        "memory_type",
        [MemoryType.SHORT_TERM, MemoryType.LONG_TERM, MemoryType.VECTOR],
    )
    def test_memory_item_memory_types(self, memory_type: MemoryType):
        """Test creating MemoryItem with different memory types."""
        item = MemoryItem(content="Test", memory_type=memory_type)
        assert item.memory_type == memory_type

    def test_memory_item_metadata_types(self):
        """Test that metadata can contain various types."""
        metadata = {
            "string": "value",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
        }
        item = MemoryItem(content="Test", metadata=metadata)
        assert item.metadata == metadata

    def test_memory_item_embedding_list(self):
        """Test that embedding can be a list of floats."""
        embedding = [float(i) / 10 for i in range(100)]
        item = MemoryItem(content="Test", embedding=embedding)
        assert item.embedding == embedding
        assert len(item.embedding) == 100

    @pytest.mark.parametrize(
        "ttl,expected",
        [
            (None, None),
            (0, 0),
            (3600, 3600),
            (86400, 86400),
        ],
    )
    def test_memory_item_ttl_values(self, ttl: Optional[int], expected: Optional[int]):
        """Test MemoryItem with various TTL values."""
        item = MemoryItem(content="Test", ttl=ttl)
        assert item.ttl == expected

    def test_memory_item_serialization(self):
        """Test that MemoryItem can be serialized to dict."""
        item = MemoryItem(
            id="test-id",
            content="Test content",
            metadata={"key": "value"},
            memory_type=MemoryType.SHORT_TERM,
        )
        data = item.model_dump()

        assert data["id"] == "test-id"
        assert data["content"] == "Test content"
        assert data["metadata"] == {"key": "value"}
        assert data["memory_type"] == "short_term"  # use_enum_values=True

    def test_memory_item_from_dict(self):
        """Test creating MemoryItem from dictionary."""
        data = {
            "id": "from-dict-id",
            "content": "From dict content",
            "metadata": {"source": "dict"},
            "memory_type": "long_term",
        }
        item = MemoryItem(**data)

        assert item.id == "from-dict-id"
        assert item.content == "From dict content"

    def test_memory_item_empty_content(self):
        """Test MemoryItem with empty content."""
        item = MemoryItem(content="")
        assert item.content == ""

    def test_memory_item_long_content(self):
        """Test MemoryItem with long content."""
        long_content = "x" * 100000
        item = MemoryItem(content=long_content)
        assert item.content == long_content
        assert len(item.content) == 100000


# ============================================================================
# MemoryQuery Model Tests
# ============================================================================


class TestMemoryQuery:
    """Tests for the MemoryQuery model."""

    def test_memory_query_defaults(self):
        """Test MemoryQuery default values."""
        query = MemoryQuery()

        assert query.query_text is None
        assert query.namespace is None
        assert query.memory_type is None
        assert query.metadata_filters == {}
        assert query.limit == 10
        assert query.offset == 0
        assert query.start_time is None
        assert query.end_time is None
        assert query.similarity_threshold == 0.7

    def test_memory_query_with_text(self):
        """Test MemoryQuery with query text."""
        query = MemoryQuery(query_text="search term")
        assert query.query_text == "search term"

    def test_memory_query_with_namespace(self):
        """Test MemoryQuery with namespace filter."""
        query = MemoryQuery(namespace="my-namespace")
        assert query.namespace == "my-namespace"

    def test_memory_query_with_memory_type_filter(self):
        """Test MemoryQuery with memory type filter."""
        query = MemoryQuery(memory_type=MemoryType.LONG_TERM)
        assert query.memory_type == MemoryType.LONG_TERM

    def test_memory_query_metadata_filters(self):
        """Test MemoryQuery with metadata filters."""
        filters = {"category": "important", "priority": 1}
        query = MemoryQuery(metadata_filters=filters)
        assert query.metadata_filters == filters

    @pytest.mark.parametrize(
        "limit,offset",
        [
            (1, 0),
            (10, 0),
            (100, 50),
            (5, 10),
        ],
    )
    def test_memory_query_pagination(self, limit: int, offset: int):
        """Test MemoryQuery pagination parameters."""
        query = MemoryQuery(limit=limit, offset=offset)
        assert query.limit == limit
        assert query.offset == offset

    def test_memory_query_time_range(self):
        """Test MemoryQuery with time range."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)

        query = MemoryQuery(start_time=start, end_time=end)
        assert query.start_time == start
        assert query.end_time == end

    @pytest.mark.parametrize(
        "threshold",
        [0.0, 0.5, 0.7, 0.9, 1.0],
    )
    def test_memory_query_similarity_threshold(self, threshold: float):
        """Test MemoryQuery similarity threshold values."""
        query = MemoryQuery(similarity_threshold=threshold)
        assert query.similarity_threshold == threshold

    def test_memory_query_full_parameters(self):
        """Test MemoryQuery with all parameters."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 1)

        query = MemoryQuery(
            query_text="find this",
            namespace="search-ns",
            memory_type=MemoryType.VECTOR,
            metadata_filters={"type": "document"},
            limit=20,
            offset=5,
            start_time=start,
            end_time=end,
            similarity_threshold=0.8,
        )

        assert query.query_text == "find this"
        assert query.namespace == "search-ns"
        assert query.memory_type == MemoryType.VECTOR
        assert query.metadata_filters == {"type": "document"}
        assert query.limit == 20
        assert query.offset == 5
        assert query.start_time == start
        assert query.end_time == end
        assert query.similarity_threshold == 0.8

    def test_memory_query_serialization(self):
        """Test MemoryQuery serialization."""
        query = MemoryQuery(
            query_text="test",
            limit=5,
            similarity_threshold=0.9,
        )
        data = query.model_dump()

        assert data["query_text"] == "test"
        assert data["limit"] == 5
        assert data["similarity_threshold"] == 0.9


# ============================================================================
# MemoryConfig Model Tests
# ============================================================================


class TestMemoryConfig:
    """Tests for the MemoryConfig model."""

    def test_memory_config_defaults(self):
        """Test MemoryConfig default values."""
        config = MemoryConfig()

        assert config.namespace == "default"
        assert config.max_items is None
        assert config.default_ttl is None
        assert config.enable_embeddings is False

    def test_memory_config_custom_namespace(self):
        """Test MemoryConfig with custom namespace."""
        config = MemoryConfig(namespace="custom-ns")
        assert config.namespace == "custom-ns"

    def test_memory_config_max_items(self):
        """Test MemoryConfig with max_items limit."""
        config = MemoryConfig(max_items=1000)
        assert config.max_items == 1000

    @pytest.mark.parametrize(
        "ttl",
        [None, 0, 60, 3600, 86400, 604800],
    )
    def test_memory_config_ttl_values(self, ttl: Optional[int]):
        """Test MemoryConfig with various TTL values."""
        config = MemoryConfig(default_ttl=ttl)
        assert config.default_ttl == ttl

    def test_memory_config_embeddings_enabled(self):
        """Test MemoryConfig with embeddings enabled."""
        config = MemoryConfig(enable_embeddings=True)
        assert config.enable_embeddings is True

    def test_memory_config_full(self):
        """Test MemoryConfig with all parameters."""
        config = MemoryConfig(
            namespace="full-config",
            max_items=500,
            default_ttl=7200,
            enable_embeddings=True,
        )

        assert config.namespace == "full-config"
        assert config.max_items == 500
        assert config.default_ttl == 7200
        assert config.enable_embeddings is True

    def test_memory_config_serialization(self):
        """Test MemoryConfig serialization."""
        config = MemoryConfig(
            namespace="test",
            max_items=100,
            default_ttl=3600,
            enable_embeddings=True,
        )
        data = config.model_dump()

        assert data["namespace"] == "test"
        assert data["max_items"] == 100
        assert data["default_ttl"] == 3600
        assert data["enable_embeddings"] is True


# ============================================================================
# Message Model Tests
# ============================================================================


class TestMessage:
    """Tests for the Message model."""

    def test_message_creation_minimal(self):
        """Test creating Message with required fields only."""
        msg = Message(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.metadata == {}
        assert isinstance(msg.timestamp, datetime)
        assert msg.token_count is None

    def test_message_creation_full(self):
        """Test creating Message with all fields."""
        timestamp = datetime.utcnow()
        msg = Message(
            role="assistant",
            content="How can I help?",
            metadata={"model": "gpt-4", "temperature": 0.7},
            timestamp=timestamp,
            token_count=5,
        )

        assert msg.role == "assistant"
        assert msg.content == "How can I help?"
        assert msg.metadata == {"model": "gpt-4", "temperature": 0.7}
        assert msg.timestamp == timestamp
        assert msg.token_count == 5

    @pytest.mark.parametrize(
        "role",
        ["user", "assistant", "system", "function", "tool"],
    )
    def test_message_roles(self, role: str):
        """Test Message with different roles."""
        msg = Message(role=role, content="Test content")
        assert msg.role == role

    def test_message_timestamp_default(self):
        """Test that Message timestamp defaults to current time."""
        before = datetime.utcnow()
        msg = Message(role="user", content="Test")
        after = datetime.utcnow()

        assert before <= msg.timestamp <= after

    def test_message_metadata_various_types(self):
        """Test Message metadata with various value types."""
        metadata = {
            "string": "value",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }
        msg = Message(role="user", content="Test", metadata=metadata)
        assert msg.metadata == metadata


class TestMessageEstimateTokens:
    """Tests for Message.estimate_tokens() method."""

    def test_estimate_tokens_short_message(self):
        """Test token estimation for short message."""
        msg = Message(role="user", content="Hi")
        # 2 chars / 4 + 1 = 1 token minimum
        tokens = msg.estimate_tokens()
        assert tokens >= 1

    def test_estimate_tokens_empty_content(self):
        """Test token estimation for empty content."""
        msg = Message(role="user", content="")
        # 0 chars / 4 + 1 = 1 token
        assert msg.estimate_tokens() == 1

    @pytest.mark.parametrize(
        "content_length,expected_min,expected_max",
        [
            (4, 1, 3),      # 4 chars -> ~1-2 tokens
            (40, 10, 15),   # 40 chars -> ~10-12 tokens
            (400, 100, 110), # 400 chars -> ~100-102 tokens
            (4000, 1000, 1010), # 4000 chars -> ~1000-1002 tokens
        ],
    )
    def test_estimate_tokens_various_lengths(
        self, content_length: int, expected_min: int, expected_max: int
    ):
        """Test token estimation for various content lengths."""
        content = "x" * content_length
        msg = Message(role="user", content=content)
        tokens = msg.estimate_tokens()

        assert expected_min <= tokens <= expected_max

    def test_estimate_tokens_with_preset_count(self):
        """Test that preset token_count is returned instead of estimation."""
        msg = Message(role="user", content="This is some content", token_count=100)
        assert msg.estimate_tokens() == 100

    def test_estimate_tokens_preset_overrides_calculation(self):
        """Test that preset token count overrides the calculation."""
        # Even with long content, preset value should be returned
        long_content = "x" * 10000
        msg = Message(role="user", content=long_content, token_count=5)
        assert msg.estimate_tokens() == 5

    def test_estimate_tokens_consistency(self):
        """Test that estimate_tokens returns consistent results."""
        msg = Message(role="user", content="This is a test message for consistency")

        # Call multiple times
        results = [msg.estimate_tokens() for _ in range(10)]

        # All results should be the same
        assert all(r == results[0] for r in results)

    def test_estimate_tokens_with_unicode(self):
        """Test token estimation with unicode content."""
        # Unicode characters might have different byte lengths
        msg = Message(role="user", content="Hello, \u4e16\u754c!")
        tokens = msg.estimate_tokens()
        assert tokens >= 1

    def test_estimate_tokens_multiline(self):
        """Test token estimation with multiline content."""
        content = "Line 1\nLine 2\nLine 3\nLine 4"
        msg = Message(role="user", content=content)
        tokens = msg.estimate_tokens()
        # ~28 chars -> ~7-8 tokens
        assert 6 <= tokens <= 10


# ============================================================================
# MemoryStore Protocol Tests
# ============================================================================


class TestMemoryStoreProtocol:
    """Tests for the MemoryStore protocol (interface verification)."""

    def test_memory_store_is_protocol(self):
        """Test that MemoryStore is a Protocol."""
        from typing import runtime_checkable, Protocol

        assert hasattr(MemoryStore, "__protocol_attrs__") or issubclass(
            MemoryStore.__class__, type(Protocol)
        )

    def test_memory_store_runtime_checkable(self):
        """Test that MemoryStore is runtime checkable."""
        # Create a mock class that implements the protocol
        class MockStore:
            async def store(self, item: MemoryItem) -> str:
                return item.id

            async def retrieve(self, query: MemoryQuery) -> List[MemoryItem]:
                return []

            async def delete(self, item_id: str) -> bool:
                return True

            async def clear(self, namespace: Optional[str] = None) -> None:
                pass

            async def get(self, item_id: str) -> Optional[MemoryItem]:
                return None

            async def count(self, namespace: Optional[str] = None) -> int:
                return 0

        mock = MockStore()
        assert isinstance(mock, MemoryStore)

    def test_non_implementing_class_not_memory_store(self):
        """Test that a class not implementing protocol is not a MemoryStore."""
        class NotAStore:
            pass

        obj = NotAStore()
        assert not isinstance(obj, MemoryStore)

    def test_partial_implementation_not_memory_store(self):
        """Test that partial implementation is not a MemoryStore."""
        class PartialStore:
            async def store(self, item: MemoryItem) -> str:
                return item.id
            # Missing other required methods

        obj = PartialStore()
        assert not isinstance(obj, MemoryStore)
