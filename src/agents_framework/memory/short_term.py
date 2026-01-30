"""Short-term memory implementation for agents.

This module provides session-based memory with token-aware buffering,
message history management, and namespaced storage per agent/session.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, Iterator, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from .base import MemoryConfig, MemoryItem, MemoryQuery, MemoryStore, MemoryType, Message


class SlidingWindowBuffer:
    """Token-aware sliding window buffer for message history.

    Maintains a buffer of messages within a maximum token limit,
    automatically removing oldest messages when the limit is exceeded.

    Attributes:
        max_tokens: Maximum number of tokens allowed in the buffer.
        _messages: Internal deque storing messages.
        _total_tokens: Current total token count.
    """

    def __init__(self, max_tokens: int = 4096):
        """Initialize the sliding window buffer.

        Args:
            max_tokens: Maximum token capacity of the buffer.
        """
        self.max_tokens = max_tokens
        self._messages: Deque[Message] = deque()
        self._total_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Get the current total token count."""
        return self._total_tokens

    @property
    def messages(self) -> List[Message]:
        """Get all messages in the buffer."""
        return list(self._messages)

    def add(self, message: Message) -> List[Message]:
        """Add a message to the buffer, evicting old messages if needed.

        Args:
            message: The message to add.

        Returns:
            List of messages that were evicted to make room.
        """
        token_count = message.estimate_tokens()
        evicted: List[Message] = []

        # Evict messages until we have room
        while self._messages and (self._total_tokens + token_count > self.max_tokens):
            evicted_msg = self._messages.popleft()
            self._total_tokens -= evicted_msg.estimate_tokens()
            evicted.append(evicted_msg)

        # Add the new message
        self._messages.append(message)
        self._total_tokens += token_count

        return evicted

    def get_recent(self, n: int) -> List[Message]:
        """Get the most recent n messages.

        Args:
            n: Number of messages to retrieve.

        Returns:
            List of the n most recent messages.
        """
        if n <= 0:
            return []
        if n >= len(self._messages):
            return list(self._messages)
        return list(self._messages)[-n:]

    def get_within_tokens(self, max_tokens: int) -> List[Message]:
        """Get messages that fit within a token budget.

        Args:
            max_tokens: Maximum tokens to include.

        Returns:
            List of messages fitting within the token budget, most recent first.
        """
        result: List[Message] = []
        current_tokens = 0

        for msg in reversed(self._messages):
            msg_tokens = msg.estimate_tokens()
            if current_tokens + msg_tokens > max_tokens:
                break
            result.append(msg)
            current_tokens += msg_tokens

        return list(reversed(result))

    def clear(self) -> None:
        """Clear all messages from the buffer."""
        self._messages.clear()
        self._total_tokens = 0

    def __len__(self) -> int:
        """Get the number of messages in the buffer."""
        return len(self._messages)

    def __iter__(self) -> Iterator[Message]:
        """Iterate over messages in the buffer."""
        return iter(self._messages)


class SessionMemoryConfig(BaseModel):
    """Configuration for session memory.

    Attributes:
        max_tokens: Maximum tokens for the sliding window buffer.
        max_messages: Maximum number of messages to store.
        preserve_system_messages: Whether to keep system messages during eviction.
        session_id: Unique identifier for this session.
        agent_id: Agent ID this session belongs to.
    """

    max_tokens: int = 8192
    max_messages: Optional[int] = None
    preserve_system_messages: bool = True
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: Optional[str] = None


class SessionMemory(MemoryStore):
    """Session-based short-term memory for agents.

    Provides in-memory storage for conversation history with token-aware
    buffering and namespaced storage per agent/session.

    Attributes:
        config: Session memory configuration.
        buffer: The sliding window buffer for messages.
    """

    def __init__(self, config: Optional[SessionMemoryConfig] = None):
        """Initialize session memory.

        Args:
            config: Optional configuration for the session.
        """
        self.config = config or SessionMemoryConfig()
        self.buffer = SlidingWindowBuffer(max_tokens=self.config.max_tokens)
        self._items: Dict[str, MemoryItem] = {}
        self._system_messages: List[Message] = []
        self._namespace = self._build_namespace()

    def _build_namespace(self) -> str:
        """Build the namespace string for this session."""
        parts = []
        if self.config.agent_id:
            parts.append(f"agent:{self.config.agent_id}")
        parts.append(f"session:{self.config.session_id}")
        return ":".join(parts)

    @property
    def namespace(self) -> str:
        """Get the namespace for this session."""
        return self._namespace

    @property
    def session_id(self) -> str:
        """Get the session ID."""
        return self.config.session_id

    def add_message(self, message: Message) -> List[Message]:
        """Add a message to the session history.

        Args:
            message: The message to add.

        Returns:
            List of messages that were evicted.
        """
        # Preserve system messages separately if configured
        if self.config.preserve_system_messages and message.role == "system":
            self._system_messages.append(message)
            return []

        # Check max_messages limit
        if self.config.max_messages and len(self.buffer) >= self.config.max_messages:
            # Evict oldest non-system message
            evicted = []
            if self.buffer.messages:
                oldest = self.buffer.messages[0]
                self.buffer._messages.popleft()
                self.buffer._total_tokens -= oldest.estimate_tokens()
                evicted.append(oldest)
            self.buffer.add(message)
            return evicted

        return self.buffer.add(message)

    def get_messages(
        self,
        include_system: bool = True,
        max_tokens: Optional[int] = None,
    ) -> List[Message]:
        """Get messages from the session.

        Args:
            include_system: Whether to include system messages.
            max_tokens: Optional token limit for returned messages.

        Returns:
            List of messages.
        """
        messages: List[Message] = []

        # Add system messages first
        if include_system and self._system_messages:
            messages.extend(self._system_messages)

        # Get buffer messages
        if max_tokens is not None:
            # Account for system message tokens
            system_tokens = sum(m.estimate_tokens() for m in self._system_messages)
            remaining_tokens = max(0, max_tokens - system_tokens)
            messages.extend(self.buffer.get_within_tokens(remaining_tokens))
        else:
            messages.extend(self.buffer.messages)

        return messages

    def get_recent_messages(self, n: int, include_system: bool = True) -> List[Message]:
        """Get the n most recent messages.

        Args:
            n: Number of messages to retrieve.
            include_system: Whether to include system messages.

        Returns:
            List of messages.
        """
        messages: List[Message] = []

        if include_system and self._system_messages:
            messages.extend(self._system_messages)

        messages.extend(self.buffer.get_recent(n))
        return messages

    async def store(self, item: MemoryItem) -> str:
        """Store a memory item.

        Args:
            item: The MemoryItem to store.

        Returns:
            The ID of the stored item.
        """
        item.namespace = self._namespace
        item.memory_type = MemoryType.SHORT_TERM
        self._items[item.id] = item

        # Also add as a message if it has content
        if item.content:
            message = Message(
                role=item.metadata.get("role", "user"),
                content=item.content,
                metadata=item.metadata,
                timestamp=item.timestamp,
            )
            self.add_message(message)

        return item.id

    async def retrieve(self, query: MemoryQuery) -> List[MemoryItem]:
        """Retrieve memory items matching the query.

        Args:
            query: The MemoryQuery specifying search criteria.

        Returns:
            List of matching MemoryItem objects.
        """
        results: List[MemoryItem] = []

        for item in self._items.values():
            # Filter by namespace
            if query.namespace and item.namespace != query.namespace:
                continue

            # Filter by time range
            if query.start_time and item.timestamp < query.start_time:
                continue
            if query.end_time and item.timestamp > query.end_time:
                continue

            # Filter by metadata
            if query.metadata_filters:
                match = all(
                    item.metadata.get(k) == v
                    for k, v in query.metadata_filters.items()
                )
                if not match:
                    continue

            # Simple text search
            if query.query_text and query.query_text.lower() not in item.content.lower():
                continue

            results.append(item)

        # Apply pagination
        results = results[query.offset : query.offset + query.limit]
        return results

    async def delete(self, item_id: str) -> bool:
        """Delete a memory item by ID.

        Args:
            item_id: The ID of the item to delete.

        Returns:
            True if the item was deleted, False if not found.
        """
        if item_id in self._items:
            del self._items[item_id]
            return True
        return False

    async def clear(self, namespace: Optional[str] = None) -> None:
        """Clear all memory items.

        Args:
            namespace: Optional namespace to clear. If None, clears all items.
        """
        if namespace:
            self._items = {
                k: v for k, v in self._items.items() if v.namespace != namespace
            }
        else:
            self._items.clear()
            self.buffer.clear()
            self._system_messages.clear()

    async def get(self, item_id: str) -> Optional[MemoryItem]:
        """Get a specific memory item by ID.

        Args:
            item_id: The ID of the item to retrieve.

        Returns:
            The MemoryItem if found, None otherwise.
        """
        return self._items.get(item_id)

    async def count(self, namespace: Optional[str] = None) -> int:
        """Count the number of stored items.

        Args:
            namespace: Optional namespace to count items in.

        Returns:
            The number of items.
        """
        if namespace:
            return sum(1 for item in self._items.values() if item.namespace == namespace)
        return len(self._items)

    def get_token_usage(self) -> Dict[str, int]:
        """Get current token usage statistics.

        Returns:
            Dictionary with token usage information.
        """
        system_tokens = sum(m.estimate_tokens() for m in self._system_messages)
        buffer_tokens = self.buffer.total_tokens

        return {
            "system_tokens": system_tokens,
            "buffer_tokens": buffer_tokens,
            "total_tokens": system_tokens + buffer_tokens,
            "max_tokens": self.config.max_tokens,
            "available_tokens": max(0, self.config.max_tokens - buffer_tokens),
        }


class NamespacedSessionStore:
    """Manager for multiple namespaced session memories.

    Provides centralized management of session memories for multiple
    agents and sessions.
    """

    def __init__(self):
        """Initialize the namespaced session store."""
        self._sessions: Dict[str, SessionMemory] = {}

    def get_or_create(
        self,
        session_id: str,
        agent_id: Optional[str] = None,
        config: Optional[SessionMemoryConfig] = None,
    ) -> SessionMemory:
        """Get or create a session memory.

        Args:
            session_id: The session ID.
            agent_id: Optional agent ID.
            config: Optional session configuration.

        Returns:
            The SessionMemory instance.
        """
        namespace = self._build_namespace(session_id, agent_id)

        if namespace not in self._sessions:
            if config is None:
                config = SessionMemoryConfig(session_id=session_id, agent_id=agent_id)
            else:
                config.session_id = session_id
                config.agent_id = agent_id
            self._sessions[namespace] = SessionMemory(config)

        return self._sessions[namespace]

    def _build_namespace(self, session_id: str, agent_id: Optional[str] = None) -> str:
        """Build a namespace key."""
        parts = []
        if agent_id:
            parts.append(f"agent:{agent_id}")
        parts.append(f"session:{session_id}")
        return ":".join(parts)

    def get(
        self,
        session_id: str,
        agent_id: Optional[str] = None,
    ) -> Optional[SessionMemory]:
        """Get a session memory if it exists.

        Args:
            session_id: The session ID.
            agent_id: Optional agent ID.

        Returns:
            The SessionMemory if found, None otherwise.
        """
        namespace = self._build_namespace(session_id, agent_id)
        return self._sessions.get(namespace)

    def remove(
        self,
        session_id: str,
        agent_id: Optional[str] = None,
    ) -> bool:
        """Remove a session memory.

        Args:
            session_id: The session ID.
            agent_id: Optional agent ID.

        Returns:
            True if removed, False if not found.
        """
        namespace = self._build_namespace(session_id, agent_id)
        if namespace in self._sessions:
            del self._sessions[namespace]
            return True
        return False

    def list_sessions(self, agent_id: Optional[str] = None) -> List[str]:
        """List all session IDs.

        Args:
            agent_id: Optional filter by agent ID.

        Returns:
            List of session IDs.
        """
        sessions = []
        for namespace, session in self._sessions.items():
            if agent_id is None or session.config.agent_id == agent_id:
                sessions.append(session.session_id)
        return sessions

    def clear_agent_sessions(self, agent_id: str) -> int:
        """Clear all sessions for an agent.

        Args:
            agent_id: The agent ID.

        Returns:
            Number of sessions cleared.
        """
        to_remove = [
            ns for ns, session in self._sessions.items()
            if session.config.agent_id == agent_id
        ]
        for ns in to_remove:
            del self._sessions[ns]
        return len(to_remove)
