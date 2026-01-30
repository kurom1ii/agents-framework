"""Unit tests for the message router module."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import List

import pytest

from agents_framework.teams.router import (
    AgentMessage,
    MessageAcknowledgment,
    MessagePriority,
    MessageQueue,
    MessageRouter,
    MessageStatus,
    RoutingStrategy,
)


# ============================================================================
# MessageQueue Tests
# ============================================================================


class TestMessageQueue:
    """Tests for the MessageQueue class."""

    @pytest.mark.asyncio
    async def test_queue_creation(self):
        """Test creating a message queue."""
        queue = MessageQueue(max_size=100)
        assert queue.size() == 0
        assert queue.is_empty()
        assert not queue.is_full()

    @pytest.mark.asyncio
    async def test_put_and_get_message(self, sample_message: AgentMessage):
        """Test putting and getting a message from the queue."""
        queue = MessageQueue(max_size=100)

        success = await queue.put(sample_message)
        assert success
        assert queue.size() == 1
        assert not queue.is_empty()

        retrieved = await queue.get()
        assert retrieved is not None
        assert retrieved.id == sample_message.id
        assert queue.size() == 0
        assert queue.is_empty()

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Test that messages are retrieved in priority order."""
        queue = MessageQueue(max_size=100)

        # Add messages with different priorities
        low_msg = AgentMessage(
            sender_id="test",
            content="low",
            priority=MessagePriority.LOW,
        )
        normal_msg = AgentMessage(
            sender_id="test",
            content="normal",
            priority=MessagePriority.NORMAL,
        )
        high_msg = AgentMessage(
            sender_id="test",
            content="high",
            priority=MessagePriority.HIGH,
        )
        urgent_msg = AgentMessage(
            sender_id="test",
            content="urgent",
            priority=MessagePriority.URGENT,
        )

        # Add in non-priority order
        await queue.put(low_msg)
        await queue.put(urgent_msg)
        await queue.put(normal_msg)
        await queue.put(high_msg)

        assert queue.size() == 4

        # Should retrieve in priority order (highest first)
        first = await queue.get()
        assert first.priority == MessagePriority.URGENT

        second = await queue.get()
        assert second.priority == MessagePriority.HIGH

        third = await queue.get()
        assert third.priority == MessagePriority.NORMAL

        fourth = await queue.get()
        assert fourth.priority == MessagePriority.LOW

    @pytest.mark.asyncio
    async def test_queue_full(self):
        """Test behavior when queue is full."""
        queue = MessageQueue(max_size=2)

        msg1 = AgentMessage(sender_id="test", content="1")
        msg2 = AgentMessage(sender_id="test", content="2")
        msg3 = AgentMessage(sender_id="test", content="3")

        assert await queue.put(msg1)
        assert await queue.put(msg2)
        assert queue.is_full()

        # Should fail when full
        result = await queue.put(msg3)
        assert not result

    @pytest.mark.asyncio
    async def test_get_with_timeout(self):
        """Test getting message with timeout on empty queue."""
        queue = MessageQueue(max_size=100)

        result = await queue.get(timeout=0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_with_timeout_success(self, sample_message: AgentMessage):
        """Test getting message with timeout when message arrives."""
        queue = MessageQueue(max_size=100)

        async def add_message_later():
            await asyncio.sleep(0.05)
            await queue.put(sample_message)

        # Start adding message in background
        task = asyncio.create_task(add_message_later())

        result = await queue.get(timeout=0.5)
        assert result is not None
        assert result.id == sample_message.id

        await task


# ============================================================================
# AgentMessage Tests
# ============================================================================


class TestAgentMessage:
    """Tests for the AgentMessage dataclass."""

    def test_message_creation(self):
        """Test creating a message."""
        message = AgentMessage(
            sender_id="agent_1",
            recipient_id="agent_2",
            content={"data": "test"},
        )

        assert message.sender_id == "agent_1"
        assert message.recipient_id == "agent_2"
        assert message.content == {"data": "test"}
        assert message.priority == MessagePriority.NORMAL
        assert message.strategy == RoutingStrategy.DIRECT
        assert message.id is not None

    def test_message_with_expiration_not_expired(self):
        """Test message expiration check for non-expired message."""
        future_time = datetime.now() + timedelta(hours=1)
        message = AgentMessage(
            sender_id="test",
            content="test",
            expires_at=future_time,
        )

        assert not message.is_expired

    def test_message_with_expiration_expired(self):
        """Test message expiration check for expired message."""
        past_time = datetime.now() - timedelta(hours=1)
        message = AgentMessage(
            sender_id="test",
            content="test",
            expires_at=past_time,
        )

        assert message.is_expired

    def test_message_without_expiration(self):
        """Test message without expiration is never expired."""
        message = AgentMessage(
            sender_id="test",
            content="test",
            expires_at=None,
        )

        assert not message.is_expired

    def test_message_with_correlation_id(self):
        """Test message with correlation ID for request/response pairs."""
        message = AgentMessage(
            sender_id="test",
            content="test",
            correlation_id="corr_123",
        )

        assert message.correlation_id == "corr_123"


# ============================================================================
# MessageAcknowledgment Tests
# ============================================================================


class TestMessageAcknowledgment:
    """Tests for the MessageAcknowledgment dataclass."""

    def test_acknowledgment_creation(self):
        """Test creating an acknowledgment."""
        ack = MessageAcknowledgment(
            message_id="msg_123",
            agent_id="agent_1",
            success=True,
        )

        assert ack.message_id == "msg_123"
        assert ack.agent_id == "agent_1"
        assert ack.success
        assert ack.error is None

    def test_acknowledgment_with_error(self):
        """Test acknowledgment with error."""
        ack = MessageAcknowledgment(
            message_id="msg_123",
            agent_id="agent_1",
            success=False,
            error="Processing failed",
        )

        assert not ack.success
        assert ack.error == "Processing failed"


# ============================================================================
# MessageRouter Tests
# ============================================================================


class TestMessageRouter:
    """Tests for the MessageRouter class."""

    def test_router_creation(self):
        """Test creating a message router."""
        router = MessageRouter()
        assert router.list_agents() == []
        assert router.list_topics() == []

    def test_router_repr(self):
        """Test router string representation."""
        router = MessageRouter()
        repr_str = repr(router)
        assert "MessageRouter" in repr_str
        assert "agents=0" in repr_str
        assert "topics=0" in repr_str

    def test_register_agent(self, message_router: MessageRouter):
        """Test registering an agent."""
        message_router.register_agent("agent_1")

        assert "agent_1" in message_router.list_agents()
        assert message_router.get_queue_size("agent_1") == 0

    def test_register_agent_with_handler(self, message_router: MessageRouter):
        """Test registering an agent with a message handler."""
        handler_calls = []

        def handler(msg: AgentMessage):
            handler_calls.append(msg)

        message_router.register_agent("agent_1", handler=handler)
        assert "agent_1" in message_router.list_agents()

    def test_register_agent_custom_queue_size(self, message_router: MessageRouter):
        """Test registering agent with custom queue size."""
        message_router.register_agent("agent_1", queue_size=50)
        assert "agent_1" in message_router.list_agents()

    def test_unregister_agent(self, message_router: MessageRouter):
        """Test unregistering an agent."""
        message_router.register_agent("agent_1")
        message_router.unregister_agent("agent_1")

        assert "agent_1" not in message_router.list_agents()

    def test_unregister_agent_removes_from_topics(self, message_router: MessageRouter):
        """Test that unregistering removes agent from topics."""
        message_router.register_agent("agent_1")
        message_router.subscribe("agent_1", "updates")

        assert "agent_1" in message_router.get_subscribers("updates")

        message_router.unregister_agent("agent_1")

        assert "agent_1" not in message_router.get_subscribers("updates")

    def test_subscribe_to_topic(self, message_router: MessageRouter):
        """Test subscribing an agent to a topic."""
        message_router.register_agent("agent_1")
        message_router.subscribe("agent_1", "updates")

        assert "updates" in message_router.list_topics()
        assert "agent_1" in message_router.get_subscribers("updates")

    def test_unsubscribe_from_topic(self, message_router: MessageRouter):
        """Test unsubscribing an agent from a topic."""
        message_router.register_agent("agent_1")
        message_router.subscribe("agent_1", "updates")
        message_router.unsubscribe("agent_1", "updates")

        assert "agent_1" not in message_router.get_subscribers("updates")

    def test_get_subscribers_returns_copy(self, message_router: MessageRouter):
        """Test that get_subscribers returns a copy."""
        message_router.register_agent("agent_1")
        message_router.subscribe("agent_1", "updates")

        subscribers = message_router.get_subscribers("updates")
        subscribers.add("agent_2")  # Modify returned set

        # Original should be unchanged
        assert "agent_2" not in message_router.get_subscribers("updates")

    @pytest.mark.asyncio
    async def test_send_direct_message(
        self,
        message_router: MessageRouter,
        sample_message: AgentMessage,
    ):
        """Test sending a direct message."""
        message_router.register_agent("agent_1")
        message_router.register_agent("agent_2")

        recipients = await message_router.send(sample_message)

        assert recipients == ["agent_2"]
        assert message_router.get_queue_size("agent_2") == 1

    @pytest.mark.asyncio
    async def test_send_direct_message_no_recipient_id(
        self,
        message_router: MessageRouter,
    ):
        """Test sending direct message without recipient_id raises error."""
        message_router.register_agent("agent_1")

        message = AgentMessage(
            sender_id="agent_1",
            content="test",
            strategy=RoutingStrategy.DIRECT,
        )

        with pytest.raises(ValueError, match="recipient_id"):
            await message_router.send(message)

    @pytest.mark.asyncio
    async def test_send_broadcast_message(
        self,
        message_router: MessageRouter,
        broadcast_message: AgentMessage,
    ):
        """Test sending a broadcast message."""
        message_router.register_agent("agent_1")
        message_router.register_agent("agent_2")
        message_router.register_agent("agent_3")

        recipients = await message_router.send(broadcast_message)

        # Should reach all agents except sender
        assert len(recipients) == 2
        assert "agent_2" in recipients
        assert "agent_3" in recipients
        assert "agent_1" not in recipients  # Sender excluded

    @pytest.mark.asyncio
    async def test_send_topic_message(
        self,
        message_router: MessageRouter,
        topic_message: AgentMessage,
    ):
        """Test sending a topic-based message."""
        message_router.register_agent("agent_1")
        message_router.register_agent("agent_2")
        message_router.register_agent("agent_3")

        # Only agent_2 subscribes to the topic
        message_router.subscribe("agent_2", "updates")

        recipients = await message_router.send(topic_message)

        assert recipients == ["agent_2"]

    @pytest.mark.asyncio
    async def test_send_topic_message_no_topic_raises(
        self,
        message_router: MessageRouter,
    ):
        """Test sending topic message without topic raises error."""
        message = AgentMessage(
            sender_id="agent_1",
            content="test",
            strategy=RoutingStrategy.TOPIC,
        )

        with pytest.raises(ValueError, match="topic"):
            await message_router.send(message)

    @pytest.mark.asyncio
    async def test_send_expired_message(
        self,
        message_router: MessageRouter,
    ):
        """Test that expired messages are not sent."""
        message_router.register_agent("agent_1")
        message_router.register_agent("agent_2")

        expired_message = AgentMessage(
            sender_id="agent_1",
            recipient_id="agent_2",
            content="test",
            expires_at=datetime.now() - timedelta(hours=1),
        )

        recipients = await message_router.send(expired_message)
        assert recipients == []

    @pytest.mark.asyncio
    async def test_send_with_handler_invocation(
        self,
        message_router: MessageRouter,
    ):
        """Test that handlers are invoked when messages are delivered."""
        handler_calls = []

        def handler(msg: AgentMessage):
            handler_calls.append(msg)

        message_router.register_agent("agent_1")
        message_router.register_agent("agent_2", handler=handler)

        message = AgentMessage(
            sender_id="agent_1",
            recipient_id="agent_2",
            content="test",
        )

        await message_router.send(message)

        assert len(handler_calls) == 1
        assert handler_calls[0].content == "test"

    @pytest.mark.asyncio
    async def test_send_with_async_handler(
        self,
        message_router: MessageRouter,
    ):
        """Test that async handlers are properly awaited."""
        handler_calls = []

        async def async_handler(msg: AgentMessage):
            await asyncio.sleep(0.01)
            handler_calls.append(msg)

        message_router.register_agent("agent_1")
        message_router.register_agent("agent_2", handler=async_handler)

        message = AgentMessage(
            sender_id="agent_1",
            recipient_id="agent_2",
            content="test",
        )

        await message_router.send(message)

        assert len(handler_calls) == 1

    @pytest.mark.asyncio
    async def test_receive_message(
        self,
        message_router: MessageRouter,
        sample_message: AgentMessage,
    ):
        """Test receiving a message."""
        message_router.register_agent("agent_1")
        message_router.register_agent("agent_2")

        await message_router.send(sample_message)

        received = await message_router.receive("agent_2")
        assert received is not None
        assert received.id == sample_message.id

    @pytest.mark.asyncio
    async def test_receive_from_unregistered_agent(
        self,
        message_router: MessageRouter,
    ):
        """Test receiving from unregistered agent returns None."""
        result = await message_router.receive("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_acknowledge_message(
        self,
        message_router: MessageRouter,
    ):
        """Test acknowledging a message."""
        message_router.register_agent("agent_1")
        message_router.register_agent("agent_2")

        message = AgentMessage(
            sender_id="agent_1",
            recipient_id="agent_2",
            content="test",
            requires_ack=True,
        )

        await message_router.send(message)

        assert message.id in message_router.get_pending_acks()

        ack = MessageAcknowledgment(
            message_id=message.id,
            agent_id="agent_2",
            success=True,
        )

        result = await message_router.acknowledge(ack)
        assert result
        assert message.id not in message_router.get_pending_acks()

    @pytest.mark.asyncio
    async def test_acknowledge_unknown_message(
        self,
        message_router: MessageRouter,
    ):
        """Test acknowledging unknown message returns False."""
        ack = MessageAcknowledgment(
            message_id="nonexistent",
            agent_id="agent_1",
        )

        result = await message_router.acknowledge(ack)
        assert not result

    @pytest.mark.asyncio
    async def test_clear_agent_queue(
        self,
        message_router: MessageRouter,
    ):
        """Test clearing an agent's message queue."""
        message_router.register_agent("agent_1")
        message_router.register_agent("agent_2")

        for i in range(5):
            await message_router.send(
                AgentMessage(
                    sender_id="agent_1",
                    recipient_id="agent_2",
                    content=f"message_{i}",
                )
            )

        assert message_router.get_queue_size("agent_2") == 5

        await message_router.clear_agent_queue("agent_2")

        assert message_router.get_queue_size("agent_2") == 0

    def test_get_queue_size_unregistered(self, message_router: MessageRouter):
        """Test getting queue size for unregistered agent."""
        size = message_router.get_queue_size("nonexistent")
        assert size == 0


class TestRoutingStrategies:
    """Tests for different routing strategies."""

    @pytest.mark.asyncio
    async def test_direct_to_nonexistent_recipient(
        self,
        message_router: MessageRouter,
    ):
        """Test direct routing to non-existent recipient."""
        message_router.register_agent("agent_1")

        message = AgentMessage(
            sender_id="agent_1",
            recipient_id="nonexistent",
            content="test",
        )

        recipients = await message_router.send(message)
        assert recipients == []

    @pytest.mark.asyncio
    async def test_broadcast_with_single_agent(
        self,
        message_router: MessageRouter,
    ):
        """Test broadcast with only one agent (sender)."""
        message_router.register_agent("agent_1")

        message = AgentMessage(
            sender_id="agent_1",
            content="test",
            strategy=RoutingStrategy.BROADCAST,
        )

        recipients = await message_router.send(message)
        assert recipients == []

    @pytest.mark.asyncio
    async def test_topic_sender_excluded(
        self,
        message_router: MessageRouter,
    ):
        """Test that sender is excluded from topic routing."""
        message_router.register_agent("agent_1")
        message_router.subscribe("agent_1", "updates")

        message = AgentMessage(
            sender_id="agent_1",
            topic="updates",
            content="test",
            strategy=RoutingStrategy.TOPIC,
        )

        recipients = await message_router.send(message)
        assert "agent_1" not in recipients

    @pytest.mark.asyncio
    async def test_topic_with_no_subscribers(
        self,
        message_router: MessageRouter,
    ):
        """Test topic routing with no subscribers."""
        message_router.register_agent("agent_1")

        message = AgentMessage(
            sender_id="agent_1",
            topic="empty_topic",
            content="test",
            strategy=RoutingStrategy.TOPIC,
        )

        recipients = await message_router.send(message)
        assert recipients == []

    @pytest.mark.asyncio
    async def test_topic_multiple_subscribers(
        self,
        message_router: MessageRouter,
    ):
        """Test topic routing with multiple subscribers."""
        for i in range(5):
            message_router.register_agent(f"agent_{i}")
            if i > 0:  # Don't subscribe sender
                message_router.subscribe(f"agent_{i}", "news")

        message = AgentMessage(
            sender_id="agent_0",
            topic="news",
            content="Breaking news!",
            strategy=RoutingStrategy.TOPIC,
        )

        recipients = await message_router.send(message)
        assert len(recipients) == 4
        assert "agent_0" not in recipients
