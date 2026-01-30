"""Context management for agents framework.

This module provides context management capabilities including:
- Token counting with tiktoken support
- Message windowing strategies
- Context budget tracking and enforcement

KUR-33: Context Manager Implementation
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Tuple, Union

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agents_framework.llm import Message


class TokenizerType(str, Enum):
    """Supported tokenizer types."""

    TIKTOKEN_CL100K = "cl100k_base"  # GPT-4, GPT-3.5-turbo
    TIKTOKEN_P50K = "p50k_base"  # Codex models
    TIKTOKEN_O200K = "o200k_base"  # GPT-4o
    CHARACTER_ESTIMATE = "char_estimate"  # Fallback: ~4 chars per token


# Model to tokenizer mapping
MODEL_TOKENIZER_MAP: Dict[str, TokenizerType] = {
    # OpenAI models
    "gpt-4o": TokenizerType.TIKTOKEN_O200K,
    "gpt-4o-mini": TokenizerType.TIKTOKEN_O200K,
    "gpt-4-turbo": TokenizerType.TIKTOKEN_CL100K,
    "gpt-4": TokenizerType.TIKTOKEN_CL100K,
    "gpt-3.5-turbo": TokenizerType.TIKTOKEN_CL100K,
    # Anthropic models (use character estimation)
    "claude-3-opus": TokenizerType.CHARACTER_ESTIMATE,
    "claude-3-sonnet": TokenizerType.CHARACTER_ESTIMATE,
    "claude-3-haiku": TokenizerType.CHARACTER_ESTIMATE,
    "claude-3-5-sonnet": TokenizerType.CHARACTER_ESTIMATE,
    # Default
    "default": TokenizerType.CHARACTER_ESTIMATE,
}


class TokenCounter(Protocol):
    """Protocol for token counting implementations."""

    def count(self, text: str) -> int:
        """Count tokens in the given text."""
        ...

    def count_messages(self, messages: List[Any]) -> int:
        """Count tokens in a list of messages."""
        ...


class TiktokenCounter:
    """Token counter using tiktoken library."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        """Initialize tiktoken counter.

        Args:
            encoding_name: The tiktoken encoding to use.

        Raises:
            ImportError: If tiktoken is not installed.
        """
        # Verify tiktoken is available at construction time
        try:
            import tiktoken
            self._tiktoken = tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is required for accurate token counting. "
                "Install with: pip install tiktoken"
            )

        self._encoding_name = encoding_name
        self._encoding: Any = None
        self._tokens_per_message = 3  # Every message has overhead
        self._tokens_per_name = 1  # If there's a name, add tokens

    @property
    def encoding(self) -> Any:
        """Lazy load tiktoken encoding."""
        if self._encoding is None:
            self._encoding = self._tiktoken.get_encoding(self._encoding_name)
        return self._encoding

    def count(self, text: str) -> int:
        """Count tokens in the given text.

        Args:
            text: The text to count tokens for.

        Returns:
            Number of tokens in the text.
        """
        return len(self.encoding.encode(text))

    def count_messages(self, messages: List[Any]) -> int:
        """Count tokens in a list of messages.

        Args:
            messages: List of message objects with role and content.

        Returns:
            Total token count including message overhead.
        """
        total = 0
        for message in messages:
            total += self._tokens_per_message
            # Handle both dict and object-like messages
            if isinstance(message, dict):
                content = message.get("content", "")
                role = message.get("role", "")
                name = message.get("name")
            else:
                content = getattr(message, "content", "")
                role = getattr(message, "role", "")
                if hasattr(role, "value"):  # Handle Enum
                    role = role.value
                name = getattr(message, "name", None)

            total += self.count(str(content))
            total += self.count(str(role))
            if name:
                total += self.count(str(name)) + self._tokens_per_name

        total += 3  # Every reply is primed with assistant
        return total


class CharacterEstimateCounter:
    """Fallback token counter using character estimation."""

    def __init__(self, chars_per_token: float = 4.0):
        """Initialize character estimate counter.

        Args:
            chars_per_token: Average characters per token.
        """
        self._chars_per_token = chars_per_token
        self._tokens_per_message = 3

    def count(self, text: str) -> int:
        """Count tokens using character estimation.

        Args:
            text: The text to count tokens for.

        Returns:
            Estimated number of tokens.
        """
        return max(1, int(len(text) / self._chars_per_token))

    def count_messages(self, messages: List[Any]) -> int:
        """Count tokens in a list of messages using estimation.

        Args:
            messages: List of message objects.

        Returns:
            Estimated total token count.
        """
        total = 0
        for message in messages:
            total += self._tokens_per_message
            if isinstance(message, dict):
                content = message.get("content", "")
            else:
                content = getattr(message, "content", "")
            total += self.count(str(content))
        total += 3
        return total


def get_token_counter(model: str) -> Union[TiktokenCounter, CharacterEstimateCounter]:
    """Get appropriate token counter for a model.

    Args:
        model: The model name.

    Returns:
        A token counter instance.
    """
    # Find matching tokenizer
    tokenizer_type = MODEL_TOKENIZER_MAP.get(model)
    if tokenizer_type is None:
        # Try prefix matching
        for model_prefix, tok_type in MODEL_TOKENIZER_MAP.items():
            if model.startswith(model_prefix):
                tokenizer_type = tok_type
                break

    if tokenizer_type is None:
        tokenizer_type = TokenizerType.CHARACTER_ESTIMATE

    if tokenizer_type == TokenizerType.CHARACTER_ESTIMATE:
        return CharacterEstimateCounter()

    try:
        return TiktokenCounter(tokenizer_type.value)
    except ImportError:
        return CharacterEstimateCounter()


@dataclass
class TokenWindow:
    """Token-based context window management.

    Tracks token usage and provides window management functionality.
    """

    max_tokens: int
    current_tokens: int = 0
    reserved_tokens: int = 0  # Tokens reserved for response
    model: str = "default"

    _counter: Optional[Union[TiktokenCounter, CharacterEstimateCounter]] = field(
        default=None, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize token counter."""
        if self._counter is None:
            self._counter = get_token_counter(self.model)

    @property
    def available_tokens(self) -> int:
        """Get available tokens for new content."""
        return max(0, self.max_tokens - self.current_tokens - self.reserved_tokens)

    @property
    def usage_ratio(self) -> float:
        """Get current usage as a ratio of max tokens."""
        if self.max_tokens == 0:
            return 0.0
        return self.current_tokens / self.max_tokens

    @property
    def is_full(self) -> bool:
        """Check if the window is at capacity."""
        return self.available_tokens <= 0

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: The text to count.

        Returns:
            Token count.
        """
        if self._counter is None:
            self._counter = get_token_counter(self.model)
        return self._counter.count(text)

    def count_messages(self, messages: List[Any]) -> int:
        """Count tokens in messages.

        Args:
            messages: List of messages.

        Returns:
            Total token count.
        """
        if self._counter is None:
            self._counter = get_token_counter(self.model)
        return self._counter.count_messages(messages)

    def can_fit(self, text: str) -> bool:
        """Check if text can fit in available space.

        Args:
            text: The text to check.

        Returns:
            True if text fits, False otherwise.
        """
        return self.count_tokens(text) <= self.available_tokens

    def can_fit_messages(self, messages: List[Any]) -> bool:
        """Check if messages can fit in available space.

        Args:
            messages: The messages to check.

        Returns:
            True if messages fit, False otherwise.
        """
        return self.count_messages(messages) <= self.available_tokens

    def add_tokens(self, count: int) -> None:
        """Add tokens to current usage.

        Args:
            count: Number of tokens to add.
        """
        self.current_tokens += count

    def remove_tokens(self, count: int) -> None:
        """Remove tokens from current usage.

        Args:
            count: Number of tokens to remove.
        """
        self.current_tokens = max(0, self.current_tokens - count)

    def reset(self) -> None:
        """Reset token count to zero."""
        self.current_tokens = 0


@dataclass
class MessageWindow:
    """Message-based context window management.

    Manages messages with a sliding window approach.
    """

    max_messages: int
    messages: List[Any] = field(default_factory=list)
    preserve_system: bool = True  # Always keep system messages

    @property
    def count(self) -> int:
        """Get current message count."""
        return len(self.messages)

    @property
    def is_full(self) -> bool:
        """Check if window is at capacity."""
        return self.count >= self.max_messages

    def add(self, message: Any) -> Optional[Any]:
        """Add a message to the window.

        Args:
            message: The message to add.

        Returns:
            The removed message if window was full, None otherwise.
        """
        removed = None
        if self.is_full:
            removed = self._remove_oldest()
        self.messages.append(message)
        return removed

    def _remove_oldest(self) -> Optional[Any]:
        """Remove the oldest non-system message.

        Returns:
            The removed message, or None if no suitable message found.
        """
        if not self.messages:
            return None

        if self.preserve_system:
            # Find first non-system message
            for i, msg in enumerate(self.messages):
                role = self._get_role(msg)
                if role != "system":
                    return self.messages.pop(i)
            return None
        else:
            return self.messages.pop(0)

    def _get_role(self, message: Any) -> str:
        """Extract role from message."""
        if isinstance(message, dict):
            return message.get("role", "")
        role = getattr(message, "role", "")
        if hasattr(role, "value"):
            return role.value
        return str(role)

    def clear(self, keep_system: bool = True) -> None:
        """Clear messages from window.

        Args:
            keep_system: Whether to keep system messages.
        """
        if keep_system:
            self.messages = [
                msg for msg in self.messages if self._get_role(msg) == "system"
            ]
        else:
            self.messages = []

    def get_messages(self) -> List[Any]:
        """Get all messages in the window."""
        return list(self.messages)


class ContextBudget(BaseModel):
    """Budget allocation for different context components."""

    total_tokens: int = Field(default=8000, description="Total available tokens")
    system_ratio: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Ratio for system prompts"
    )
    conversation_ratio: float = Field(
        default=0.60, ge=0.0, le=1.0, description="Ratio for conversation history"
    )
    tools_ratio: float = Field(
        default=0.10, ge=0.0, le=1.0, description="Ratio for tool definitions"
    )
    response_ratio: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Ratio reserved for response"
    )

    @property
    def system_tokens(self) -> int:
        """Tokens allocated for system prompts."""
        return int(self.total_tokens * self.system_ratio)

    @property
    def conversation_tokens(self) -> int:
        """Tokens allocated for conversation history."""
        return int(self.total_tokens * self.conversation_ratio)

    @property
    def tools_tokens(self) -> int:
        """Tokens allocated for tool definitions."""
        return int(self.total_tokens * self.tools_ratio)

    @property
    def response_tokens(self) -> int:
        """Tokens reserved for model response."""
        return int(self.total_tokens * self.response_ratio)

    def validate_ratios(self) -> bool:
        """Validate that ratios sum to approximately 1.0."""
        total = (
            self.system_ratio
            + self.conversation_ratio
            + self.tools_ratio
            + self.response_ratio
        )
        return 0.99 <= total <= 1.01


@dataclass
class ContextUsage:
    """Tracks current context usage across components."""

    system_tokens: int = 0
    conversation_tokens: int = 0
    tools_tokens: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.system_tokens + self.conversation_tokens + self.tools_tokens


class ContextManager:
    """Main context management class.

    Manages context windows, token budgets, and provides utilities
    for context optimization.
    """

    def __init__(
        self,
        budget: Optional[ContextBudget] = None,
        model: str = "default",
        max_messages: Optional[int] = None,
    ):
        """Initialize context manager.

        Args:
            budget: Token budget configuration.
            model: Model name for tokenization.
            max_messages: Maximum messages in window.
        """
        self.budget = budget or ContextBudget()
        self.model = model
        self._counter = get_token_counter(model)

        # Initialize windows
        self.token_window = TokenWindow(
            max_tokens=self.budget.total_tokens,
            reserved_tokens=self.budget.response_tokens,
            model=model,
        )
        self.message_window = MessageWindow(
            max_messages=max_messages or 100, preserve_system=True
        )

        # Track usage
        self._usage = ContextUsage()
        self._compaction_count = 0

    @property
    def usage(self) -> ContextUsage:
        """Get current context usage."""
        return self._usage

    @property
    def available_tokens(self) -> int:
        """Get available tokens for new content."""
        return self.token_window.available_tokens

    @property
    def needs_compaction(self) -> bool:
        """Check if context needs compaction."""
        return self.token_window.usage_ratio > 0.85

    async def add_message(self, message: Any) -> Tuple[bool, Optional[Any]]:
        """Add a message to context.

        Args:
            message: The message to add.

        Returns:
            Tuple of (success, removed_message).
        """
        token_count = self._counter.count_messages([message])

        # Check if we have space
        if not self.token_window.can_fit_messages([message]):
            return False, None

        # Add to windows
        removed = self.message_window.add(message)
        self.token_window.add_tokens(token_count)
        self._usage.conversation_tokens += token_count

        # If a message was removed, update token count
        if removed:
            removed_tokens = self._counter.count_messages([removed])
            self.token_window.remove_tokens(removed_tokens)
            self._usage.conversation_tokens -= removed_tokens

        return True, removed

    async def add_messages(
        self, messages: List[Any]
    ) -> Tuple[int, List[Any]]:
        """Add multiple messages to context.

        Args:
            messages: List of messages to add.

        Returns:
            Tuple of (added_count, removed_messages).
        """
        added = 0
        removed_messages = []

        for message in messages:
            success, removed = await self.add_message(message)
            if success:
                added += 1
                if removed:
                    removed_messages.append(removed)
            else:
                break  # Stop if we can't fit more

        return added, removed_messages

    async def set_system_prompt(self, system_prompt: str) -> bool:
        """Set the system prompt, checking budget.

        Args:
            system_prompt: The system prompt text.

        Returns:
            True if system prompt fits in budget, False otherwise.
        """
        token_count = self._counter.count(system_prompt)
        if token_count > self.budget.system_tokens:
            return False

        self._usage.system_tokens = token_count
        return True

    async def set_tools(self, tools: List[Any]) -> bool:
        """Set tool definitions, checking budget.

        Args:
            tools: List of tool definitions.

        Returns:
            True if tools fit in budget, False otherwise.
        """
        # Estimate token count for tools
        tool_text = str(tools)  # Simplified - would serialize properly
        token_count = self._counter.count(tool_text)

        if token_count > self.budget.tools_tokens:
            return False

        self._usage.tools_tokens = token_count
        return True

    def get_messages(self) -> List[Any]:
        """Get all messages in context window."""
        return self.message_window.get_messages()

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count.

        Returns:
            Token count.
        """
        return self._counter.count(text)

    def count_message_tokens(self, messages: List[Any]) -> int:
        """Count tokens in messages.

        Args:
            messages: Messages to count.

        Returns:
            Total token count.
        """
        return self._counter.count_messages(messages)

    async def clear(self, keep_system: bool = True) -> None:
        """Clear context.

        Args:
            keep_system: Whether to preserve system messages.
        """
        self.message_window.clear(keep_system=keep_system)
        self.token_window.reset()
        if not keep_system:
            self._usage.system_tokens = 0
        self._usage.conversation_tokens = 0

    def get_context_hash(self) -> str:
        """Get a hash of current context for caching.

        Returns:
            SHA256 hash of context content.
        """
        content = "".join(
            str(getattr(msg, "content", msg.get("content", "")))
            for msg in self.message_window.messages
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get_stats(self) -> Dict[str, Any]:
        """Get context statistics.

        Returns:
            Dictionary with context stats.
        """
        return {
            "total_tokens": self.budget.total_tokens,
            "used_tokens": self._usage.total_tokens,
            "available_tokens": self.available_tokens,
            "usage_ratio": self.token_window.usage_ratio,
            "message_count": self.message_window.count,
            "compaction_count": self._compaction_count,
            "needs_compaction": self.needs_compaction,
            "budget": {
                "system": {
                    "allocated": self.budget.system_tokens,
                    "used": self._usage.system_tokens,
                },
                "conversation": {
                    "allocated": self.budget.conversation_tokens,
                    "used": self._usage.conversation_tokens,
                },
                "tools": {
                    "allocated": self.budget.tools_tokens,
                    "used": self._usage.tools_tokens,
                },
                "response": {
                    "reserved": self.budget.response_tokens,
                },
            },
        }
