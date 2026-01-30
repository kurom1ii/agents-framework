"""Message router for inter-agent communication.

This module provides the MessageRouter class for routing messages between
agents in a multi-agent team. It supports direct messaging, broadcast,
and topic-based routing strategies.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

if TYPE_CHECKING:
    from agents_framework.agents import BaseAgent


class RoutingStrategy(str, Enum):
    """Strategy for routing messages between agents."""

    DIRECT = "direct"  # Send to a specific agent
    BROADCAST = "broadcast"  # Send to all registered agents
    TOPIC = "topic"  # Send to agents subscribed to a topic


class MessagePriority(int, Enum):
    """Priority levels for messages."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


class MessageStatus(str, Enum):
    """Status of a message in the routing system."""

    PENDING = "pending"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class AgentMessage:
    """A message sent between agents.

    Attributes:
        id: Unique identifier for the message.
        sender_id: ID of the agent sending the message.
        recipient_id: ID of the target agent (for direct routing).
        topic: Topic for topic-based routing.
        content: The message content (can be any serializable data).
        priority: Message priority level.
        strategy: Routing strategy for this message.
        metadata: Additional metadata for the message.
        created_at: Timestamp when the message was created.
        expires_at: Optional expiration timestamp.
        requires_ack: Whether the message requires acknowledgment.
        correlation_id: Optional ID for correlating request/response pairs.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: Optional[str] = None
    topic: Optional[str] = None
    content: Any = None
    priority: MessagePriority = MessagePriority.NORMAL
    strategy: RoutingStrategy = RoutingStrategy.DIRECT
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    requires_ack: bool = False
    correlation_id: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        """Check if the message has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


@dataclass
class MessageAcknowledgment:
    """Acknowledgment for a received message.

    Attributes:
        message_id: ID of the acknowledged message.
        agent_id: ID of the acknowledging agent.
        success: Whether the message was processed successfully.
        error: Error message if processing failed.
        timestamp: When the acknowledgment was created.
    """

    message_id: str
    agent_id: str
    success: bool = True
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class MessageQueue:
    """Per-agent message queue with priority support.

    Messages are stored in priority order and can be retrieved
    asynchronously.
    """

    def __init__(self, max_size: int = 1000):
        """Initialize the message queue.

        Args:
            max_size: Maximum number of messages in the queue.
        """
        self._queue: asyncio.PriorityQueue[tuple[int, datetime, AgentMessage]] = (
            asyncio.PriorityQueue(maxsize=max_size)
        )
        self._max_size = max_size
        self._message_count = 0

    async def put(self, message: AgentMessage) -> bool:
        """Add a message to the queue.

        Args:
            message: The message to add.

        Returns:
            True if added successfully, False if queue is full.
        """
        if self._queue.full():
            return False

        # Priority tuple: higher priority = lower number for queue ordering
        priority_value = -message.priority.value
        await self._queue.put((priority_value, message.created_at, message))
        self._message_count += 1
        return True

    async def get(self, timeout: Optional[float] = None) -> Optional[AgentMessage]:
        """Get the next message from the queue.

        Args:
            timeout: Optional timeout in seconds.

        Returns:
            The next message or None if timeout occurred.
        """
        try:
            if timeout is not None:
                _, _, message = await asyncio.wait_for(
                    self._queue.get(), timeout=timeout
                )
            else:
                _, _, message = await self._queue.get()
            self._message_count -= 1
            return message
        except asyncio.TimeoutError:
            return None

    def size(self) -> int:
        """Get the current queue size."""
        return self._message_count

    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return self._queue.empty()

    def is_full(self) -> bool:
        """Check if the queue is full."""
        return self._queue.full()


MessageHandler = Callable[[AgentMessage], Any]


class MessageRouter:
    """Router for inter-agent communication.

    Manages message queues for registered agents and handles routing
    messages using different strategies (direct, broadcast, topic-based).

    Example:
        router = MessageRouter()

        # Register agents
        router.register_agent(agent1)
        router.register_agent(agent2)

        # Subscribe to topics
        router.subscribe(agent1.id, "task_updates")

        # Send messages
        await router.send(AgentMessage(
            sender_id=agent1.id,
            recipient_id=agent2.id,
            content={"task": "analyze"},
            strategy=RoutingStrategy.DIRECT,
        ))
    """

    def __init__(self, default_queue_size: int = 1000):
        """Initialize the message router.

        Args:
            default_queue_size: Default max size for message queues.
        """
        self._queues: Dict[str, MessageQueue] = {}
        self._topic_subscriptions: Dict[str, Set[str]] = {}  # topic -> agent_ids
        self._pending_acks: Dict[str, AgentMessage] = {}  # message_id -> message
        self._handlers: Dict[str, MessageHandler] = {}  # agent_id -> handler
        self._default_queue_size = default_queue_size
        self._lock = asyncio.Lock()

    def register_agent(
        self,
        agent_id: str,
        handler: Optional[MessageHandler] = None,
        queue_size: Optional[int] = None,
    ) -> None:
        """Register an agent with the router.

        Args:
            agent_id: The agent's unique identifier.
            handler: Optional message handler callback.
            queue_size: Optional custom queue size.
        """
        if agent_id not in self._queues:
            size = queue_size or self._default_queue_size
            self._queues[agent_id] = MessageQueue(max_size=size)
        if handler:
            self._handlers[agent_id] = handler

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the router.

        Args:
            agent_id: The agent's unique identifier.
        """
        self._queues.pop(agent_id, None)
        self._handlers.pop(agent_id, None)

        # Remove from all topic subscriptions
        for subscribers in self._topic_subscriptions.values():
            subscribers.discard(agent_id)

    def subscribe(self, agent_id: str, topic: str) -> None:
        """Subscribe an agent to a topic.

        Args:
            agent_id: The agent's unique identifier.
            topic: The topic to subscribe to.
        """
        if topic not in self._topic_subscriptions:
            self._topic_subscriptions[topic] = set()
        self._topic_subscriptions[topic].add(agent_id)

    def unsubscribe(self, agent_id: str, topic: str) -> None:
        """Unsubscribe an agent from a topic.

        Args:
            agent_id: The agent's unique identifier.
            topic: The topic to unsubscribe from.
        """
        if topic in self._topic_subscriptions:
            self._topic_subscriptions[topic].discard(agent_id)

    def get_subscribers(self, topic: str) -> Set[str]:
        """Get all agents subscribed to a topic.

        Args:
            topic: The topic to query.

        Returns:
            Set of agent IDs subscribed to the topic.
        """
        return self._topic_subscriptions.get(topic, set()).copy()

    async def send(self, message: AgentMessage) -> List[str]:
        """Send a message using its routing strategy.

        Args:
            message: The message to send.

        Returns:
            List of agent IDs that received the message.

        Raises:
            ValueError: If the message has invalid routing parameters.
        """
        if message.is_expired:
            return []

        recipients: List[str] = []

        if message.strategy == RoutingStrategy.DIRECT:
            if not message.recipient_id:
                raise ValueError("Direct routing requires recipient_id")
            recipients = await self._send_direct(message)

        elif message.strategy == RoutingStrategy.BROADCAST:
            recipients = await self._send_broadcast(message)

        elif message.strategy == RoutingStrategy.TOPIC:
            if not message.topic:
                raise ValueError("Topic routing requires topic")
            recipients = await self._send_topic(message)

        # Track messages requiring acknowledgment
        if message.requires_ack and recipients:
            self._pending_acks[message.id] = message

        return recipients

    async def _send_direct(self, message: AgentMessage) -> List[str]:
        """Send a message directly to a specific agent."""
        recipient_id = message.recipient_id
        if recipient_id and recipient_id in self._queues:
            success = await self._queues[recipient_id].put(message)
            if success:
                await self._invoke_handler(recipient_id, message)
                return [recipient_id]
        return []

    async def _send_broadcast(self, message: AgentMessage) -> List[str]:
        """Broadcast a message to all registered agents."""
        recipients: List[str] = []
        for agent_id, queue in self._queues.items():
            if agent_id != message.sender_id:  # Don't send to self
                success = await queue.put(message)
                if success:
                    await self._invoke_handler(agent_id, message)
                    recipients.append(agent_id)
        return recipients

    async def _send_topic(self, message: AgentMessage) -> List[str]:
        """Send a message to all agents subscribed to a topic."""
        recipients: List[str] = []
        topic = message.topic
        if topic and topic in self._topic_subscriptions:
            for agent_id in self._topic_subscriptions[topic]:
                if agent_id != message.sender_id and agent_id in self._queues:
                    success = await self._queues[agent_id].put(message)
                    if success:
                        await self._invoke_handler(agent_id, message)
                        recipients.append(agent_id)
        return recipients

    async def _invoke_handler(self, agent_id: str, message: AgentMessage) -> None:
        """Invoke the message handler for an agent if registered."""
        handler = self._handlers.get(agent_id)
        if handler:
            try:
                result = handler(message)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                # Handlers should not propagate exceptions
                pass

    async def receive(
        self, agent_id: str, timeout: Optional[float] = None
    ) -> Optional[AgentMessage]:
        """Receive the next message for an agent.

        Args:
            agent_id: The agent's unique identifier.
            timeout: Optional timeout in seconds.

        Returns:
            The next message or None if no message available.
        """
        queue = self._queues.get(agent_id)
        if queue:
            return await queue.get(timeout=timeout)
        return None

    async def acknowledge(self, ack: MessageAcknowledgment) -> bool:
        """Acknowledge receipt of a message.

        Args:
            ack: The acknowledgment details.

        Returns:
            True if the message was pending acknowledgment.
        """
        message = self._pending_acks.pop(ack.message_id, None)
        return message is not None

    def get_queue_size(self, agent_id: str) -> int:
        """Get the number of pending messages for an agent.

        Args:
            agent_id: The agent's unique identifier.

        Returns:
            Number of messages in the agent's queue.
        """
        queue = self._queues.get(agent_id)
        return queue.size() if queue else 0

    def get_pending_acks(self) -> List[str]:
        """Get IDs of messages pending acknowledgment.

        Returns:
            List of message IDs awaiting acknowledgment.
        """
        return list(self._pending_acks.keys())

    def list_topics(self) -> List[str]:
        """List all registered topics.

        Returns:
            List of topic names.
        """
        return list(self._topic_subscriptions.keys())

    def list_agents(self) -> List[str]:
        """List all registered agent IDs.

        Returns:
            List of registered agent IDs.
        """
        return list(self._queues.keys())

    async def clear_agent_queue(self, agent_id: str) -> None:
        """Clear all messages from an agent's queue.

        Args:
            agent_id: The agent's unique identifier.
        """
        if agent_id in self._queues:
            self._queues[agent_id] = MessageQueue(max_size=self._default_queue_size)

    def __repr__(self) -> str:
        agents = len(self._queues)
        topics = len(self._topic_subscriptions)
        return f"MessageRouter(agents={agents}, topics={topics})"
