"""Test 1: Core LLM Provider - Can we communicate with LLM?

This test verifies the fundamental LLM communication works:
- Message formatting
- Tool definitions
- Response parsing
- Retry logic
"""

import pytest
from agents_framework.llm.base import (
    LLMConfig,
    LLMResponse,
    Message,
    MessageRole,
    RetryConfig,
    ToolCall,
    ToolDefinition,
)


class TestLLMCoreIntegration:
    """Test LLM provider core functionality."""

    def test_message_creation_and_serialization(self):
        """Messages can be created and converted to dict format."""
        msg = Message(
            role=MessageRole.USER,
            content="Hello, world!",
            name="test_user",
        )

        result = msg.to_dict()

        assert result["role"] == "user"
        assert result["content"] == "Hello, world!"
        assert result["name"] == "test_user"

    def test_message_with_tool_calls(self):
        """Messages can include tool calls."""
        tool_call = ToolCall(
            id="call_abc123",
            name="search",
            arguments={"query": "weather"},
        )
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[tool_call],
        )

        result = msg.to_dict()

        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "search"

    def test_llm_response_has_tool_calls_property(self):
        """LLMResponse correctly detects tool calls."""
        # No tool calls
        response_no_tools = LLMResponse(
            content="Hello",
            model="test",
        )
        assert response_no_tools.has_tool_calls is False

        # With tool calls
        response_with_tools = LLMResponse(
            content="",
            model="test",
            tool_calls=[ToolCall(id="1", name="test", arguments={})],
        )
        assert response_with_tools.has_tool_calls is True

    def test_retry_config_exponential_backoff(self):
        """RetryConfig calculates correct delays."""
        config = RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=False,
        )

        # Delays should be: 1, 2, 4, 8 (capped at 10)
        assert config.get_delay(0) == 1.0
        assert config.get_delay(1) == 2.0
        assert config.get_delay(2) == 4.0
        assert config.get_delay(3) == 8.0
        assert config.get_delay(4) == 10.0  # Capped

    def test_tool_definition_structure(self):
        """Tool definitions have correct structure for LLM."""
        tool = ToolDefinition(
            name="calculator",
            description="Perform math calculations",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
        )

        assert tool.name == "calculator"
        assert "properties" in tool.parameters
        assert "expression" in tool.parameters["properties"]

    @pytest.mark.asyncio
    async def test_mock_llm_provider_generates_response(self, mock_llm_provider, sample_messages):
        """LLM provider generates responses correctly."""
        response = await mock_llm_provider.generate(sample_messages)

        assert response.content == "Mock response"
        assert response.model == "test-model"
        assert mock_llm_provider.call_count == 1

    @pytest.mark.asyncio
    async def test_mock_llm_provider_with_tools(self, mock_llm_config, sample_messages):
        """LLM provider handles tool calls."""
        from tests.conftest import MockLLMProvider

        tool_calls = [[ToolCall(id="1", name="search", arguments={"q": "test"})]]
        provider = MockLLMProvider(mock_llm_config, tool_calls=tool_calls)

        tools = [ToolDefinition(
            name="search",
            description="Search",
            parameters={"type": "object", "properties": {}},
        )]

        response = await provider.generate(sample_messages, tools=tools)

        assert response.has_tool_calls
        assert response.tool_calls[0].name == "search"
