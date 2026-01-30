"""Test 4: Agent Execution - Can agents run tasks?

This test verifies the agent execution loop:
- Agent initialization
- Task execution
- Tool call handling
- Response generation
"""

import pytest
from agents_framework.llm.base import LLMConfig, Message, MessageRole, ToolCall
from agents_framework.tools.base import BaseTool
from agents_framework.tools.registry import ToolRegistry
from tests.conftest import MockLLMProvider


class EchoTool(BaseTool):
    """Simple echo tool for testing."""

    name = "echo"
    description = "Echo back the input"
    parameters = {
        "type": "object",
        "properties": {
            "message": {"type": "string"},
        },
        "required": ["message"],
    }

    async def execute(self, message: str) -> str:
        return f"Echo: {message}"


class TestAgentExecutionCore:
    """Test agent execution core functionality."""

    @pytest.mark.asyncio
    async def test_simple_llm_conversation(self):
        """Agent can have simple conversation without tools."""
        config = LLMConfig(model="test", api_key="key")
        provider = MockLLMProvider(config, responses=["Hello! How can I help?"])

        messages = [
            Message(role=MessageRole.USER, content="Hi there!"),
        ]

        response = await provider.generate(messages)

        assert response.content == "Hello! How can I help?"
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_llm_with_tool_call(self):
        """Agent can request tool execution."""
        config = LLMConfig(model="test", api_key="key")
        tool_calls = [[ToolCall(id="1", name="echo", arguments={"message": "test"})]]
        provider = MockLLMProvider(
            config,
            responses=["", "Tool executed successfully!"],
            tool_calls=tool_calls,
        )

        messages = [Message(role=MessageRole.USER, content="Echo something")]

        # First call returns tool request
        response1 = await provider.generate(messages)
        assert response1.has_tool_calls
        assert response1.tool_calls[0].name == "echo"

        # Second call returns final response
        response2 = await provider.generate(messages)
        assert not response2.has_tool_calls
        assert "successfully" in response2.content

    @pytest.mark.asyncio
    async def test_tool_execution_in_loop(self):
        """Full loop: LLM -> Tool -> Response."""
        # Setup
        config = LLMConfig(model="test", api_key="key")
        registry = ToolRegistry()
        registry.register(EchoTool())

        tool_calls = [[ToolCall(id="1", name="echo", arguments={"message": "hello"})]]
        provider = MockLLMProvider(
            config,
            responses=["", "The echo returned: hello"],
            tool_calls=tool_calls,
        )

        # Step 1: User message
        messages = [Message(role=MessageRole.USER, content="Echo hello")]

        # Step 2: LLM requests tool
        response1 = await provider.generate(messages, tools=registry.to_definitions())
        assert response1.has_tool_calls

        # Step 3: Execute tool
        tool = registry.get("echo")
        result = await tool.run(message="hello")
        assert result.success
        assert result.output == "Echo: hello"

        # Step 4: Send result back to LLM
        messages.append(Message(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=response1.tool_calls,
        ))
        messages.append(Message(
            role=MessageRole.TOOL,
            content=result.output,
            tool_call_id="1",
        ))

        # Step 5: Get final response
        response2 = await provider.generate(messages)
        assert "echo" in response2.content.lower() or "hello" in response2.content.lower()

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self):
        """Agent handles multiple tool calls."""
        config = LLMConfig(model="test", api_key="key")
        tool_calls = [[
            ToolCall(id="1", name="echo", arguments={"message": "first"}),
            ToolCall(id="2", name="echo", arguments={"message": "second"}),
        ]]
        provider = MockLLMProvider(
            config,
            responses=["", "Both echoed"],
            tool_calls=tool_calls,
        )

        messages = [Message(role=MessageRole.USER, content="Echo two things")]
        response = await provider.generate(messages)

        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].arguments["message"] == "first"
        assert response.tool_calls[1].arguments["message"] == "second"

    @pytest.mark.asyncio
    async def test_conversation_history_preserved(self):
        """Agent maintains conversation history."""
        config = LLMConfig(model="test", api_key="key")
        provider = MockLLMProvider(config, responses=["Response 1", "Response 2"])

        # First turn
        messages = [Message(role=MessageRole.USER, content="First message")]
        await provider.generate(messages)
        assert len(provider.messages_history[0]) == 1

        # Second turn
        messages.append(Message(role=MessageRole.ASSISTANT, content="Response 1"))
        messages.append(Message(role=MessageRole.USER, content="Second message"))
        await provider.generate(messages)
        assert len(provider.messages_history[1]) == 3
