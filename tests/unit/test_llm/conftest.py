"""Local fixtures for LLM module tests."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents_framework.llm.base import (
    BaseLLMProvider,
    LLMConfig,
    LLMProviderError,
    LLMResponse,
    Message,
    MessageRole,
    RateLimitError,
    RetryConfig,
    ToolCall,
    ToolDefinition,
)


# ============================================================================
# Concrete Implementation for Testing Abstract Base Class
# ============================================================================


class ConcreteLLMProvider(BaseLLMProvider):
    """Concrete implementation of BaseLLMProvider for testing."""

    def __init__(
        self,
        config: LLMConfig,
        responses: Optional[List[LLMResponse]] = None,
        should_fail: bool = False,
        fail_times: int = 0,
        failure_exception: Optional[Exception] = None,
    ):
        super().__init__(config)
        self.responses = responses or []
        self.should_fail = should_fail
        self.fail_times = fail_times
        self.failure_exception = failure_exception or Exception("Test failure")
        self.call_count = 0
        self.generate_call_count = 0
        self.stream_call_count = 0
        self._supports_tools = True

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response."""
        self.generate_call_count += 1
        self.call_count += 1

        if self.should_fail:
            raise self.failure_exception

        if self.fail_times > 0 and self.call_count <= self.fail_times:
            raise self.failure_exception

        if self.responses:
            idx = min(self.generate_call_count - 1, len(self.responses) - 1)
            return self.responses[idx]

        return LLMResponse(
            content="Test response",
            model=self.config.model,
            finish_reason="stop",
        )

    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream response chunks."""
        self.stream_call_count += 1

        if self.should_fail:
            raise self.failure_exception

        async def _stream() -> AsyncIterator[str]:
            chunks = ["Test", " ", "stream", " ", "response"]
            for chunk in chunks:
                yield chunk

        async for chunk in _stream():
            yield chunk

    def supports_tools(self) -> bool:
        """Return whether tools are supported."""
        return self._supports_tools


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def basic_llm_config() -> LLMConfig:
    """Create a basic LLM configuration."""
    return LLMConfig(
        model="test-model",
        api_key="test-key",
        temperature=0.5,
    )


@pytest.fixture
def llm_config_with_retry() -> LLMConfig:
    """Create LLM configuration with custom retry settings."""
    return LLMConfig(
        model="test-model",
        api_key="test-key",
        retry_config=RetryConfig(
            max_retries=3,
            base_delay=0.01,  # Fast for testing
            max_delay=0.1,
            exponential_base=2.0,
            jitter=False,
        ),
    )


@pytest.fixture
def retry_config_no_jitter() -> RetryConfig:
    """Create retry config without jitter for deterministic testing."""
    return RetryConfig(
        max_retries=3,
        base_delay=1.0,
        max_delay=10.0,
        exponential_base=2.0,
        jitter=False,
    )


@pytest.fixture
def retry_config_with_jitter() -> RetryConfig:
    """Create retry config with jitter."""
    return RetryConfig(
        max_retries=3,
        base_delay=1.0,
        max_delay=10.0,
        exponential_base=2.0,
        jitter=True,
    )


@pytest.fixture
def concrete_provider(basic_llm_config: LLMConfig) -> ConcreteLLMProvider:
    """Create a concrete LLM provider for testing."""
    return ConcreteLLMProvider(basic_llm_config)


@pytest.fixture
def provider_with_retry(llm_config_with_retry: LLMConfig) -> ConcreteLLMProvider:
    """Create a provider with retry configuration."""
    return ConcreteLLMProvider(llm_config_with_retry)


@pytest.fixture
def failing_provider(basic_llm_config: LLMConfig) -> ConcreteLLMProvider:
    """Create a provider that always fails."""
    return ConcreteLLMProvider(
        basic_llm_config,
        should_fail=True,
        failure_exception=Exception("Always fails"),
    )


@pytest.fixture
def intermittent_provider(llm_config_with_retry: LLMConfig) -> ConcreteLLMProvider:
    """Create a provider that fails a few times then succeeds."""
    return ConcreteLLMProvider(
        llm_config_with_retry,
        fail_times=2,  # Fail first 2 attempts
        failure_exception=Exception("Intermittent failure"),
    )


@pytest.fixture
def message_system() -> Message:
    """Create a system message."""
    return Message(role=MessageRole.SYSTEM, content="You are a helpful assistant.")


@pytest.fixture
def message_user() -> Message:
    """Create a user message."""
    return Message(role=MessageRole.USER, content="Hello!")


@pytest.fixture
def message_assistant() -> Message:
    """Create an assistant message."""
    return Message(role=MessageRole.ASSISTANT, content="Hi there!")


@pytest.fixture
def message_with_tool_calls() -> Message:
    """Create a message with tool calls."""
    return Message(
        role=MessageRole.ASSISTANT,
        content="",
        tool_calls=[
            ToolCall(id="call_1", name="search", arguments={"query": "test"}),
            ToolCall(id="call_2", name="calculate", arguments={"expression": "2+2"}),
        ],
    )


@pytest.fixture
def message_tool_response() -> Message:
    """Create a tool response message."""
    return Message(
        role=MessageRole.TOOL,
        content="Search results...",
        tool_call_id="call_1",
    )


@pytest.fixture
def message_with_name() -> Message:
    """Create a message with a name."""
    return Message(
        role=MessageRole.USER,
        content="Hello!",
        name="test_user",
    )


@pytest.fixture
def tool_definition_search() -> ToolDefinition:
    """Create a search tool definition."""
    return ToolDefinition(
        name="search",
        description="Search for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    )


@pytest.fixture
def tool_definition_calculate() -> ToolDefinition:
    """Create a calculate tool definition."""
    return ToolDefinition(
        name="calculate",
        description="Perform calculations",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
            },
            "required": ["expression"],
        },
    )


@pytest.fixture
def response_with_tool_calls() -> LLMResponse:
    """Create an LLM response with tool calls."""
    return LLMResponse(
        content="",
        model="test-model",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCall(id="call_1", name="search", arguments={"query": "test"}),
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


@pytest.fixture
def response_without_tool_calls() -> LLMResponse:
    """Create an LLM response without tool calls."""
    return LLMResponse(
        content="Hello, I'm here to help!",
        model="test-model",
        finish_reason="stop",
        tool_calls=None,
        usage={"prompt_tokens": 8, "completion_tokens": 10, "total_tokens": 18},
    )


@pytest.fixture
def rate_limit_error() -> RateLimitError:
    """Create a rate limit error."""
    return RateLimitError(
        message="Rate limit exceeded",
        provider="test-provider",
        status_code=429,
    )


@pytest.fixture
def provider_error() -> LLMProviderError:
    """Create a generic provider error."""
    return LLMProviderError(
        message="Something went wrong",
        provider="test-provider",
        status_code=500,
        response={"error": "Internal server error"},
    )
