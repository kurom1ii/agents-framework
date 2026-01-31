"""Base types and protocols for LLM providers."""

from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)


class MessageRole(str, Enum):
    """Role of a message in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """Represents a tool/function call from the LLM."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolDefinition:
    """Definition of a tool that can be used by the LLM."""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema for parameters


@dataclass
class Message:
    """A message in a conversation."""

    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None  # For tool response messages

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        result: Dict[str, Any] = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.name:
            result["name"] = self.name
        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result


@dataclass
class ThinkingBlock:
    """Represents a thinking block from extended thinking."""

    content: str
    signature: Optional[str] = None  # For thinking signature if provided


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    usage: Optional[Dict[str, int]] = None  # tokens used
    raw_response: Optional[Any] = None  # Original response object
    thinking: Optional[List[ThinkingBlock]] = None  # Extended thinking blocks

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return bool(self.tool_calls)

    @property
    def has_thinking(self) -> bool:
        """Check if response contains thinking blocks."""
        return bool(self.thinking)

    @property
    def thinking_content(self) -> str:
        """Get combined thinking content as a single string."""
        if not self.thinking:
            return ""
        return "\n\n".join(block.content for block in self.thinking)


@dataclass
class RetryConfig:
    """Configuration for retry logic with exponential backoff."""

    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = field(default_factory=lambda: (Exception,))
    retryable_status_codes: tuple = field(default_factory=lambda: (429, 500, 502, 503, 504))

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number."""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        if self.jitter:
            delay = delay * (0.5 + random.random())
        return delay


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""

    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: float = 60.0
    retry_config: Optional[RetryConfig] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.retry_config is None:
            self.retry_config = RetryConfig()


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers.

    All LLM providers must implement this interface to be used
    with the agents framework.
    """

    config: LLMConfig

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            messages: List of messages in the conversation.
            tools: Optional list of tool definitions for function calling.
            **kwargs: Additional provider-specific parameters.

        Returns:
            LLMResponse with the generated content.
        """
        ...

    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a response from the LLM.

        Args:
            messages: List of messages in the conversation.
            tools: Optional list of tool definitions for function calling.
            **kwargs: Additional provider-specific parameters.

        Yields:
            String chunks of the response as they arrive.
        """
        ...

    def supports_tools(self) -> bool:
        """Check if this provider supports tool/function calling.

        Returns:
            True if the provider supports tools, False otherwise.
        """
        ...


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers with common functionality."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: Any = None

    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a response from the LLM."""
        pass

    @abstractmethod
    def supports_tools(self) -> bool:
        """Check if this provider supports tool/function calling."""
        pass

    async def _retry_with_backoff(
        self,
        func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a function with retry logic and exponential backoff.

        Args:
            func: The async function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The result of the function.

        Raises:
            The last exception if all retries are exhausted.
        """
        retry_config = self.config.retry_config or RetryConfig()
        last_exception: Optional[Exception] = None

        for attempt in range(retry_config.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                # Check if we should retry
                should_retry = False

                # Check if it's a retryable exception type
                if isinstance(e, retry_config.retryable_exceptions):
                    should_retry = True

                # Check for HTTP status codes in the exception
                status_code = getattr(e, "status_code", None) or getattr(
                    getattr(e, "response", None), "status_code", None
                )
                if status_code in retry_config.retryable_status_codes:
                    should_retry = True

                if not should_retry or attempt >= retry_config.max_retries:
                    raise

                delay = retry_config.get_delay(attempt)
                await asyncio.sleep(delay)

        # Should not reach here, but raise the last exception just in case
        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected state in retry logic")

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for the provider's API.

        Override this in subclasses for provider-specific formatting.
        """
        return [msg.to_dict() for msg in messages]

    def _format_tools(
        self, tools: Optional[List[ToolDefinition]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Format tool definitions for the provider's API.

        Override this in subclasses for provider-specific formatting.
        """
        if not tools:
            return None
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in tools
        ]


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        response: Optional[Any] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.response = response


class RateLimitError(LLMProviderError):
    """Exception raised when rate limited by the provider."""
    pass


class AuthenticationError(LLMProviderError):
    """Exception raised for authentication failures."""
    pass


class InvalidRequestError(LLMProviderError):
    """Exception raised for invalid requests."""
    pass
