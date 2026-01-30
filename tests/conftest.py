"""Minimal test fixtures for agents_framework core functionality."""

import pytest
from typing import Any, AsyncIterator, Dict, List, Optional

from agents_framework.llm.base import (
    BaseLLMProvider,
    LLMConfig,
    LLMResponse,
    Message,
    MessageRole,
    ToolCall,
    ToolDefinition,
)


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing."""

    def __init__(
        self,
        config: LLMConfig,
        responses: Optional[List[str]] = None,
        tool_calls: Optional[List[List[ToolCall]]] = None,
    ):
        super().__init__(config)
        self.responses = responses or ["Mock response"]
        self.tool_calls = tool_calls
        self.call_count = 0
        self.messages_history: List[List[Message]] = []

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.messages_history.append(messages)
        response_idx = min(self.call_count, len(self.responses) - 1)

        tool_call_list = None
        if self.tool_calls and self.call_count < len(self.tool_calls):
            tool_call_list = self.tool_calls[self.call_count]

        self.call_count += 1

        return LLMResponse(
            content=self.responses[response_idx],
            model=self.config.model,
            finish_reason="stop" if not tool_call_list else "tool_calls",
            tool_calls=tool_call_list,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        self.messages_history.append(messages)
        self.call_count += 1
        response = self.responses[min(self.call_count - 1, len(self.responses) - 1)]
        for word in response.split():
            yield word + " "

    def supports_tools(self) -> bool:
        return True


@pytest.fixture
def mock_llm_config() -> LLMConfig:
    """Basic LLM config for testing."""
    return LLMConfig(
        model="test-model",
        api_key="test-key",
        temperature=0.7,
    )


@pytest.fixture
def mock_llm_provider(mock_llm_config: LLMConfig) -> MockLLMProvider:
    """Mock LLM provider instance."""
    return MockLLMProvider(mock_llm_config)


@pytest.fixture
def sample_messages() -> List[Message]:
    """Sample conversation messages."""
    return [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="Hello!"),
    ]


@pytest.fixture
def sample_tool_call() -> ToolCall:
    """Sample tool call."""
    return ToolCall(
        id="call_123",
        name="calculator",
        arguments={"expression": "2 + 2"},
    )
