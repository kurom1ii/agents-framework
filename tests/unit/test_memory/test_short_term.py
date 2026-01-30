"""Unit tests for short-term session memory.

Tests for:
- SlidingWindowBuffer class
- SessionMemoryConfig model
- SessionMemory class
- NamespacedSessionStore class
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

import pytest

from agents_framework.memory.base import (
    MemoryItem,
    MemoryQuery,
    MemoryType,
    Message,
)
from agents_framework.memory.short_term import (
    NamespacedSessionStore,
    SessionMemory,
    SessionMemoryConfig,
    SlidingWindowBuffer,
)


# ============================================================================
# SlidingWindowBuffer Tests
# ============================================================================


class TestSlidingWindowBuffer:
    """Tests for the SlidingWindowBuffer class."""

    def test_buffer_init_default(self):
        """Test buffer initialization with default max_tokens."""
        buffer = SlidingWindowBuffer()
        assert buffer.max_tokens == 4096
        assert buffer.total_tokens == 0
        assert len(buffer) == 0
        assert buffer.messages == []

    def test_buffer_init_custom_max_tokens(self):
        """Test buffer initialization with custom max_tokens."""
        buffer = SlidingWindowBuffer(max_tokens=1000)
        assert buffer.max_tokens == 1000

    def test_buffer_add_single_message(self, user_message: Message):
        """Test adding a single message to buffer."""
        buffer = SlidingWindowBuffer(max_tokens=4096)
        evicted = buffer.add(user_message)

        assert evicted == []
        assert len(buffer) == 1
        assert buffer.messages[0] == user_message
        assert buffer.total_tokens > 0

    def test_buffer_add_multiple_messages(self, conversation_messages: List[Message]):
        """Test adding multiple messages to buffer."""
        buffer = SlidingWindowBuffer(max_tokens=4096)

        for msg in conversation_messages:
            buffer.add(msg)

        assert len(buffer) == len(conversation_messages)
        assert buffer.messages == conversation_messages

    def test_buffer_eviction_on_overflow(self):
        """Test that old messages are evicted when buffer overflows."""
        buffer = SlidingWindowBuffer(max_tokens=50)

        # Add messages that will exceed the buffer
        msg1 = Message(role="user", content="A" * 40)  # ~10 tokens
        msg2 = Message(role="user", content="B" * 40)  # ~10 tokens
        msg3 = Message(role="user", content="C" * 40)  # ~10 tokens

        evicted1 = buffer.add(msg1)
        assert evicted1 == []

        evicted2 = buffer.add(msg2)
        assert evicted2 == []

        # Third message should cause eviction
        evicted3 = buffer.add(msg3)
        assert len(evicted3) >= 1

    def test_buffer_eviction_returns_evicted_messages(self):
        """Test that evicted messages are returned correctly."""
        buffer = SlidingWindowBuffer(max_tokens=30)

        msg1 = Message(role="user", content="First message")
        msg2 = Message(role="user", content="Second message that is longer")

        buffer.add(msg1)
        evicted = buffer.add(msg2)

        # msg1 should be evicted to make room for msg2
        if evicted:
            assert msg1 in evicted

    def test_buffer_get_recent_empty(self):
        """Test get_recent on empty buffer."""
        buffer = SlidingWindowBuffer()
        assert buffer.get_recent(5) == []

    def test_buffer_get_recent_less_than_available(self):
        """Test get_recent when requesting more than available."""
        buffer = SlidingWindowBuffer()
        buffer.add(Message(role="user", content="One"))
        buffer.add(Message(role="user", content="Two"))

        result = buffer.get_recent(10)
        assert len(result) == 2

    def test_buffer_get_recent_exact_count(self):
        """Test get_recent with exact count."""
        buffer = SlidingWindowBuffer()
        messages = [
            Message(role="user", content=f"Message {i}")
            for i in range(5)
        ]
        for msg in messages:
            buffer.add(msg)

        result = buffer.get_recent(3)
        assert len(result) == 3
        assert result == messages[-3:]

    @pytest.mark.parametrize("n", [0, -1, -10])
    def test_buffer_get_recent_invalid_n(self, n: int):
        """Test get_recent with zero or negative n."""
        buffer = SlidingWindowBuffer()
        buffer.add(Message(role="user", content="Test"))

        assert buffer.get_recent(n) == []

    def test_buffer_get_within_tokens(self):
        """Test get_within_tokens returns messages within budget."""
        buffer = SlidingWindowBuffer(max_tokens=1000)

        # Add several messages
        for i in range(10):
            buffer.add(Message(role="user", content=f"Message {i} content"))

        # Get messages within a token budget
        result = buffer.get_within_tokens(50)

        # Should return most recent messages that fit
        assert len(result) > 0
        total_tokens = sum(msg.estimate_tokens() for msg in result)
        assert total_tokens <= 50

    def test_buffer_get_within_tokens_empty_buffer(self):
        """Test get_within_tokens on empty buffer."""
        buffer = SlidingWindowBuffer()
        assert buffer.get_within_tokens(100) == []

    def test_buffer_get_within_tokens_zero_budget(self):
        """Test get_within_tokens with zero token budget."""
        buffer = SlidingWindowBuffer()
        buffer.add(Message(role="user", content="Test"))

        result = buffer.get_within_tokens(0)
        assert result == []

    def test_buffer_get_within_tokens_preserves_order(self):
        """Test that get_within_tokens maintains chronological order."""
        buffer = SlidingWindowBuffer(max_tokens=1000)

        messages = [
            Message(role="user", content=f"Msg {i}")
            for i in range(5)
        ]
        for msg in messages:
            buffer.add(msg)

        result = buffer.get_within_tokens(100)

        # Verify order is preserved
        for i in range(len(result) - 1):
            assert result[i].timestamp <= result[i + 1].timestamp

    def test_buffer_clear(self):
        """Test clearing the buffer."""
        buffer = SlidingWindowBuffer()
        buffer.add(Message(role="user", content="Test 1"))
        buffer.add(Message(role="user", content="Test 2"))

        assert len(buffer) == 2
        assert buffer.total_tokens > 0

        buffer.clear()

        assert len(buffer) == 0
        assert buffer.total_tokens == 0
        assert buffer.messages == []

    def test_buffer_len(self):
        """Test __len__ method."""
        buffer = SlidingWindowBuffer()
        assert len(buffer) == 0

        buffer.add(Message(role="user", content="One"))
        assert len(buffer) == 1

        buffer.add(Message(role="user", content="Two"))
        assert len(buffer) == 2

    def test_buffer_iter(self):
        """Test __iter__ method."""
        buffer = SlidingWindowBuffer()
        messages = [
            Message(role="user", content=f"Message {i}")
            for i in range(3)
        ]
        for msg in messages:
            buffer.add(msg)

        iterated = list(buffer)
        assert iterated == messages

    def test_buffer_total_tokens_tracking(self):
        """Test that total_tokens is accurately tracked."""
        buffer = SlidingWindowBuffer(max_tokens=1000)

        msg1 = Message(role="user", content="Short")
        buffer.add(msg1)
        tokens_after_one = buffer.total_tokens

        msg2 = Message(role="user", content="This is a longer message")
        buffer.add(msg2)
        tokens_after_two = buffer.total_tokens

        assert tokens_after_two > tokens_after_one
        expected = msg1.estimate_tokens() + msg2.estimate_tokens()
        assert buffer.total_tokens == expected


# ============================================================================
# SessionMemoryConfig Tests
# ============================================================================


class TestSessionMemoryConfig:
    """Tests for the SessionMemoryConfig model."""

    def test_config_defaults(self):
        """Test SessionMemoryConfig default values."""
        config = SessionMemoryConfig()

        assert config.max_tokens == 8192
        assert config.max_messages is None
        assert config.preserve_system_messages is True
        assert config.session_id is not None  # Auto-generated
        assert len(config.session_id) == 36  # UUID format
        assert config.agent_id is None

    def test_config_custom_values(self):
        """Test SessionMemoryConfig with custom values."""
        config = SessionMemoryConfig(
            max_tokens=4096,
            max_messages=50,
            preserve_system_messages=False,
            session_id="custom-session",
            agent_id="agent-123",
        )

        assert config.max_tokens == 4096
        assert config.max_messages == 50
        assert config.preserve_system_messages is False
        assert config.session_id == "custom-session"
        assert config.agent_id == "agent-123"

    def test_config_session_id_uniqueness(self):
        """Test that default session IDs are unique."""
        configs = [SessionMemoryConfig() for _ in range(100)]
        ids = [c.session_id for c in configs]
        assert len(set(ids)) == 100


# ============================================================================
# SessionMemory Tests
# ============================================================================


class TestSessionMemory:
    """Tests for the SessionMemory class."""

    def test_session_memory_init_default(self):
        """Test SessionMemory initialization with default config."""
        memory = SessionMemory()

        assert memory.config is not None
        assert memory.config.max_tokens == 8192
        assert memory.buffer is not None
        assert memory.namespace is not None

    def test_session_memory_init_custom_config(
        self, session_config: SessionMemoryConfig
    ):
        """Test SessionMemory initialization with custom config."""
        memory = SessionMemory(session_config)

        assert memory.config == session_config
        assert memory.buffer.max_tokens == session_config.max_tokens

    def test_session_memory_namespace_with_agent(self):
        """Test namespace building with agent ID."""
        config = SessionMemoryConfig(
            session_id="sess-001",
            agent_id="agent-001",
        )
        memory = SessionMemory(config)

        assert "agent:agent-001" in memory.namespace
        assert "session:sess-001" in memory.namespace

    def test_session_memory_namespace_without_agent(self):
        """Test namespace building without agent ID."""
        config = SessionMemoryConfig(session_id="sess-002")
        memory = SessionMemory(config)

        assert "agent:" not in memory.namespace
        assert "session:sess-002" in memory.namespace

    def test_session_memory_session_id_property(self):
        """Test session_id property."""
        config = SessionMemoryConfig(session_id="my-session")
        memory = SessionMemory(config)

        assert memory.session_id == "my-session"


class TestSessionMemoryAddMessage:
    """Tests for SessionMemory.add_message method."""

    def test_add_user_message(self, session_memory: SessionMemory, user_message: Message):
        """Test adding a user message."""
        evicted = session_memory.add_message(user_message)

        assert evicted == []
        messages = session_memory.get_messages(include_system=False)
        assert len(messages) == 1
        assert messages[0] == user_message

    def test_add_system_message_preserved(
        self, session_memory: SessionMemory, system_message: Message
    ):
        """Test that system messages are preserved separately."""
        evicted = session_memory.add_message(system_message)

        assert evicted == []
        messages = session_memory.get_messages(include_system=True)
        assert len(messages) == 1
        assert messages[0].role == "system"

    def test_add_system_message_not_in_buffer(
        self, session_memory: SessionMemory, system_message: Message
    ):
        """Test that system messages don't go into the buffer."""
        session_memory.add_message(system_message)

        # Buffer should be empty (system messages stored separately)
        assert len(session_memory.buffer) == 0

    def test_add_message_respects_max_messages(self):
        """Test that max_messages limit is respected."""
        config = SessionMemoryConfig(
            max_tokens=10000,
            max_messages=3,
        )
        memory = SessionMemory(config)

        # Add 4 messages
        for i in range(4):
            memory.add_message(Message(role="user", content=f"Message {i}"))

        # Should only have 3 messages due to limit
        messages = memory.get_messages(include_system=False)
        assert len(messages) == 3

    def test_add_message_eviction_on_overflow(
        self, small_buffer_config: SessionMemoryConfig
    ):
        """Test message eviction when buffer overflows."""
        memory = SessionMemory(small_buffer_config)

        # Add messages that exceed the buffer
        msg1 = Message(role="user", content="A" * 50)
        msg2 = Message(role="user", content="B" * 50)

        memory.add_message(msg1)
        evicted = memory.add_message(msg2)

        # First message should be evicted
        assert len(evicted) > 0 or len(memory.buffer) <= 2


class TestSessionMemoryGetMessages:
    """Tests for SessionMemory.get_messages method."""

    def test_get_messages_empty(self, session_memory: SessionMemory):
        """Test get_messages on empty memory."""
        messages = session_memory.get_messages()
        assert messages == []

    def test_get_messages_include_system(
        self,
        session_memory: SessionMemory,
        system_message: Message,
        user_message: Message,
    ):
        """Test get_messages with include_system=True."""
        session_memory.add_message(system_message)
        session_memory.add_message(user_message)

        messages = session_memory.get_messages(include_system=True)
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"

    def test_get_messages_exclude_system(
        self,
        session_memory: SessionMemory,
        system_message: Message,
        user_message: Message,
    ):
        """Test get_messages with include_system=False."""
        session_memory.add_message(system_message)
        session_memory.add_message(user_message)

        messages = session_memory.get_messages(include_system=False)
        assert len(messages) == 1
        assert messages[0].role == "user"

    def test_get_messages_with_max_tokens(self, session_memory: SessionMemory):
        """Test get_messages with max_tokens limit."""
        # Add several messages
        for i in range(10):
            session_memory.add_message(
                Message(role="user", content=f"Message {i} with some content")
            )

        # Get with token limit
        messages = session_memory.get_messages(max_tokens=50)

        # Should return fewer messages
        total_tokens = sum(msg.estimate_tokens() for msg in messages)
        assert total_tokens <= 50 or len(messages) <= 10


class TestSessionMemoryGetRecentMessages:
    """Tests for SessionMemory.get_recent_messages method."""

    def test_get_recent_messages(self, session_memory: SessionMemory):
        """Test get_recent_messages returns n most recent."""
        for i in range(5):
            session_memory.add_message(
                Message(role="user", content=f"Message {i}")
            )

        messages = session_memory.get_recent_messages(3, include_system=False)
        assert len(messages) == 3

    def test_get_recent_with_system(
        self,
        session_memory: SessionMemory,
        system_message: Message,
    ):
        """Test get_recent_messages includes system messages."""
        session_memory.add_message(system_message)
        for i in range(3):
            session_memory.add_message(
                Message(role="user", content=f"Message {i}")
            )

        messages = session_memory.get_recent_messages(2, include_system=True)

        # Should include system message plus 2 recent
        assert any(m.role == "system" for m in messages)


class TestSessionMemoryStoreProtocol:
    """Tests for SessionMemory implementing MemoryStore protocol."""

    async def test_store(self, session_memory: SessionMemory, basic_memory_item: MemoryItem):
        """Test store method."""
        item_id = await session_memory.store(basic_memory_item)

        assert item_id == basic_memory_item.id
        stored = await session_memory.get(item_id)
        assert stored is not None
        assert stored.content == basic_memory_item.content

    async def test_store_sets_namespace(self, session_memory: SessionMemory):
        """Test that store sets the correct namespace."""
        item = MemoryItem(content="Test content")
        await session_memory.store(item)

        stored = await session_memory.get(item.id)
        assert stored.namespace == session_memory.namespace

    async def test_store_sets_memory_type(self, session_memory: SessionMemory):
        """Test that store sets memory_type to SHORT_TERM."""
        item = MemoryItem(content="Test content", memory_type=MemoryType.LONG_TERM)
        await session_memory.store(item)

        stored = await session_memory.get(item.id)
        assert stored.memory_type == MemoryType.SHORT_TERM

    async def test_store_adds_message(self, session_memory: SessionMemory):
        """Test that store adds content as a message."""
        item = MemoryItem(
            content="Test content",
            metadata={"role": "user"},
        )
        await session_memory.store(item)

        messages = session_memory.get_messages(include_system=False)
        assert len(messages) == 1
        assert messages[0].content == "Test content"

    async def test_retrieve_empty(self, session_memory: SessionMemory):
        """Test retrieve on empty memory."""
        query = MemoryQuery(query_text="test")
        results = await session_memory.retrieve(query)
        assert results == []

    async def test_retrieve_by_text(self, session_memory: SessionMemory):
        """Test retrieve with text search."""
        await session_memory.store(
            MemoryItem(content="Python programming language")
        )
        await session_memory.store(
            MemoryItem(content="JavaScript framework")
        )

        query = MemoryQuery(query_text="Python")
        results = await session_memory.retrieve(query)

        assert len(results) == 1
        assert "Python" in results[0].content

    async def test_retrieve_by_namespace(self, session_memory: SessionMemory):
        """Test retrieve with namespace filter."""
        item = MemoryItem(content="Test")
        await session_memory.store(item)

        # Query with matching namespace
        query = MemoryQuery(namespace=session_memory.namespace)
        results = await session_memory.retrieve(query)
        assert len(results) == 1

        # Query with non-matching namespace
        query = MemoryQuery(namespace="different-namespace")
        results = await session_memory.retrieve(query)
        assert len(results) == 0

    async def test_retrieve_by_time_range(self, session_memory: SessionMemory):
        """Test retrieve with time range filter."""
        now = datetime.utcnow()

        # Create item with specific timestamp
        item = MemoryItem(
            content="Time range test",
            timestamp=now,
        )
        await session_memory.store(item)

        # Query within range
        query = MemoryQuery(
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(hours=1),
        )
        results = await session_memory.retrieve(query)
        assert len(results) == 1

        # Query outside range
        query = MemoryQuery(
            start_time=now + timedelta(hours=1),
            end_time=now + timedelta(hours=2),
        )
        results = await session_memory.retrieve(query)
        assert len(results) == 0

    async def test_retrieve_by_metadata(self, session_memory: SessionMemory):
        """Test retrieve with metadata filters."""
        await session_memory.store(
            MemoryItem(content="Important", metadata={"priority": "high"})
        )
        await session_memory.store(
            MemoryItem(content="Normal", metadata={"priority": "low"})
        )

        query = MemoryQuery(metadata_filters={"priority": "high"})
        results = await session_memory.retrieve(query)

        assert len(results) == 1
        assert results[0].metadata["priority"] == "high"

    async def test_retrieve_pagination(self, session_memory: SessionMemory):
        """Test retrieve with pagination."""
        # Store multiple items
        for i in range(10):
            await session_memory.store(
                MemoryItem(content=f"Item {i}")
            )

        # First page
        query = MemoryQuery(limit=3, offset=0)
        results = await session_memory.retrieve(query)
        assert len(results) == 3

        # Second page
        query = MemoryQuery(limit=3, offset=3)
        results = await session_memory.retrieve(query)
        assert len(results) == 3

    async def test_delete(self, session_memory: SessionMemory):
        """Test delete method."""
        item = MemoryItem(content="To be deleted")
        await session_memory.store(item)

        # Verify stored
        assert await session_memory.get(item.id) is not None

        # Delete
        result = await session_memory.delete(item.id)
        assert result is True

        # Verify deleted
        assert await session_memory.get(item.id) is None

    async def test_delete_nonexistent(self, session_memory: SessionMemory):
        """Test delete on non-existent item."""
        result = await session_memory.delete("nonexistent-id")
        assert result is False

    async def test_clear_all(self, session_memory: SessionMemory):
        """Test clear all items."""
        await session_memory.store(MemoryItem(content="Item 1"))
        await session_memory.store(MemoryItem(content="Item 2"))
        session_memory.add_message(Message(role="system", content="System"))
        session_memory.add_message(Message(role="user", content="User"))

        await session_memory.clear()

        assert await session_memory.count() == 0
        assert len(session_memory.buffer) == 0
        assert session_memory.get_messages() == []

    async def test_clear_by_namespace(self, session_memory: SessionMemory):
        """Test clear by namespace."""
        await session_memory.store(MemoryItem(content="Item in namespace"))

        # Clear with matching namespace
        await session_memory.clear(namespace=session_memory.namespace)
        assert await session_memory.count() == 0

    async def test_get(self, session_memory: SessionMemory):
        """Test get method."""
        item = MemoryItem(id="get-test-id", content="Test content")
        await session_memory.store(item)

        retrieved = await session_memory.get("get-test-id")
        assert retrieved is not None
        assert retrieved.id == "get-test-id"
        assert retrieved.content == "Test content"

    async def test_get_nonexistent(self, session_memory: SessionMemory):
        """Test get on non-existent item."""
        result = await session_memory.get("nonexistent")
        assert result is None

    async def test_count(self, session_memory: SessionMemory):
        """Test count method."""
        assert await session_memory.count() == 0

        await session_memory.store(MemoryItem(content="Item 1"))
        assert await session_memory.count() == 1

        await session_memory.store(MemoryItem(content="Item 2"))
        assert await session_memory.count() == 2

    async def test_count_by_namespace(self, session_memory: SessionMemory):
        """Test count by namespace."""
        await session_memory.store(MemoryItem(content="Item"))

        # Count with matching namespace
        count = await session_memory.count(namespace=session_memory.namespace)
        assert count == 1

        # Count with non-matching namespace
        count = await session_memory.count(namespace="other-namespace")
        assert count == 0


class TestSessionMemoryTokenUsage:
    """Tests for SessionMemory.get_token_usage method."""

    def test_token_usage_empty(self, session_memory: SessionMemory):
        """Test token usage on empty memory."""
        usage = session_memory.get_token_usage()

        assert usage["system_tokens"] == 0
        assert usage["buffer_tokens"] == 0
        assert usage["total_tokens"] == 0
        assert usage["max_tokens"] == session_memory.config.max_tokens
        assert usage["available_tokens"] == session_memory.config.max_tokens

    def test_token_usage_with_messages(
        self,
        session_memory: SessionMemory,
        user_message: Message,
        system_message: Message,
    ):
        """Test token usage with messages."""
        session_memory.add_message(system_message)
        session_memory.add_message(user_message)

        usage = session_memory.get_token_usage()

        assert usage["system_tokens"] > 0
        assert usage["buffer_tokens"] > 0
        assert usage["total_tokens"] == usage["system_tokens"] + usage["buffer_tokens"]
        assert usage["available_tokens"] < usage["max_tokens"]


# ============================================================================
# NamespacedSessionStore Tests
# ============================================================================


class TestNamespacedSessionStore:
    """Tests for the NamespacedSessionStore class."""

    def test_init(self, namespaced_session_store: NamespacedSessionStore):
        """Test NamespacedSessionStore initialization."""
        assert namespaced_session_store._sessions == {}

    def test_get_or_create_new_session(
        self, namespaced_session_store: NamespacedSessionStore
    ):
        """Test creating a new session."""
        session = namespaced_session_store.get_or_create("session-001")

        assert session is not None
        assert session.session_id == "session-001"

    def test_get_or_create_existing_session(
        self, namespaced_session_store: NamespacedSessionStore
    ):
        """Test getting an existing session."""
        session1 = namespaced_session_store.get_or_create("session-001")
        session2 = namespaced_session_store.get_or_create("session-001")

        assert session1 is session2

    def test_get_or_create_with_agent_id(
        self, namespaced_session_store: NamespacedSessionStore
    ):
        """Test creating session with agent ID."""
        session = namespaced_session_store.get_or_create(
            "session-001", agent_id="agent-001"
        )

        assert session.config.agent_id == "agent-001"
        assert "agent:agent-001" in session.namespace

    def test_get_or_create_with_config(
        self, namespaced_session_store: NamespacedSessionStore
    ):
        """Test creating session with custom config."""
        config = SessionMemoryConfig(max_tokens=2048)
        session = namespaced_session_store.get_or_create(
            "session-001", config=config
        )

        assert session.config.max_tokens == 2048

    def test_get_existing_session(
        self, namespaced_session_store: NamespacedSessionStore
    ):
        """Test getting an existing session."""
        namespaced_session_store.get_or_create("session-001")

        session = namespaced_session_store.get("session-001")
        assert session is not None
        assert session.session_id == "session-001"

    def test_get_nonexistent_session(
        self, namespaced_session_store: NamespacedSessionStore
    ):
        """Test getting a non-existent session."""
        session = namespaced_session_store.get("nonexistent")
        assert session is None

    def test_remove_session(
        self, namespaced_session_store: NamespacedSessionStore
    ):
        """Test removing a session."""
        namespaced_session_store.get_or_create("session-001")

        result = namespaced_session_store.remove("session-001")
        assert result is True

        session = namespaced_session_store.get("session-001")
        assert session is None

    def test_remove_nonexistent_session(
        self, namespaced_session_store: NamespacedSessionStore
    ):
        """Test removing a non-existent session."""
        result = namespaced_session_store.remove("nonexistent")
        assert result is False

    def test_list_sessions(
        self, namespaced_session_store: NamespacedSessionStore
    ):
        """Test listing all sessions."""
        namespaced_session_store.get_or_create("session-001")
        namespaced_session_store.get_or_create("session-002")
        namespaced_session_store.get_or_create("session-003")

        sessions = namespaced_session_store.list_sessions()
        assert len(sessions) == 3
        assert "session-001" in sessions
        assert "session-002" in sessions
        assert "session-003" in sessions

    def test_list_sessions_by_agent(
        self, namespaced_session_store: NamespacedSessionStore
    ):
        """Test listing sessions filtered by agent."""
        namespaced_session_store.get_or_create("s1", agent_id="agent-A")
        namespaced_session_store.get_or_create("s2", agent_id="agent-A")
        namespaced_session_store.get_or_create("s3", agent_id="agent-B")

        sessions = namespaced_session_store.list_sessions(agent_id="agent-A")
        assert len(sessions) == 2
        assert "s1" in sessions
        assert "s2" in sessions
        assert "s3" not in sessions

    def test_clear_agent_sessions(
        self, namespaced_session_store: NamespacedSessionStore
    ):
        """Test clearing all sessions for an agent."""
        namespaced_session_store.get_or_create("s1", agent_id="agent-A")
        namespaced_session_store.get_or_create("s2", agent_id="agent-A")
        namespaced_session_store.get_or_create("s3", agent_id="agent-B")

        count = namespaced_session_store.clear_agent_sessions("agent-A")
        assert count == 2

        # agent-A sessions should be gone
        assert namespaced_session_store.get("s1", agent_id="agent-A") is None
        assert namespaced_session_store.get("s2", agent_id="agent-A") is None

        # agent-B session should remain
        assert namespaced_session_store.get("s3", agent_id="agent-B") is not None

    def test_namespace_isolation(
        self, namespaced_session_store: NamespacedSessionStore
    ):
        """Test that sessions with same ID but different agents are isolated."""
        session_a = namespaced_session_store.get_or_create(
            "shared-session", agent_id="agent-A"
        )
        session_b = namespaced_session_store.get_or_create(
            "shared-session", agent_id="agent-B"
        )

        # Should be different sessions
        assert session_a is not session_b
        assert session_a.namespace != session_b.namespace

        # Adding message to one shouldn't affect the other
        session_a.add_message(Message(role="user", content="Message for A"))

        assert len(session_a.get_messages()) == 1
        assert len(session_b.get_messages()) == 0
