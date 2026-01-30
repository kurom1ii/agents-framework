"""Local fixtures for execution module tests."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents_framework.agents.base import (
    AgentConfig,
    AgentRole,
    AgentStatus,
    BaseAgent,
    Task,
    TaskResult,
)
from agents_framework.execution.hooks import (
    Hook,
    HookContext,
    HookEntry,
    HookRegistry,
    HookType,
)
from agents_framework.execution.loop import (
    AgentLoop,
    AgentRunner,
    LoopConfig,
    LoopState,
    LoopStep,
    RunnerConfig,
    StepType,
    TerminationReason,
)
from agents_framework.llm.base import (
    BaseLLMProvider,
    LLMConfig,
    LLMResponse,
    Message,
    MessageRole,
    ToolCall,
    ToolDefinition,
)
from agents_framework.tools.base import BaseTool, ToolResult
from agents_framework.tools.executor import ExecutionConfig, ExecutionResult, ToolExecutor
from agents_framework.tools.registry import ToolRegistry


# ============================================================================
# Mock LLM Provider
# ============================================================================


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for execution tests."""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        responses: Optional[List[LLMResponse]] = None,
    ):
        config = config or LLMConfig(
            model="test-model",
            api_key="test-key",
        )
        super().__init__(config)
        self.responses = responses or []
        self.call_count = 0
        self.last_messages: Optional[List[Message]] = None
        self.last_tools: Optional[List[ToolDefinition]] = None

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a mock response."""
        self.last_messages = messages
        self.last_tools = tools
        self.call_count += 1

        if self.responses:
            idx = min(self.call_count - 1, len(self.responses) - 1)
            return self.responses[idx]

        return LLMResponse(
            content="Mock response",
            model=self.config.model,
            finish_reason="stop",
            usage={"total_tokens": 10},
        )

    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream mock response chunks."""
        self.last_messages = messages
        self.last_tools = tools
        self.call_count += 1

        for chunk in ["Hello", " ", "World", "!"]:
            yield chunk

    def supports_tools(self) -> bool:
        """Mock supports tools."""
        return True


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """Create a mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def mock_llm_with_tool_calls() -> MockLLMProvider:
    """Create a mock LLM that returns tool calls."""
    tool_call = ToolCall(
        id="call_123",
        name="test_tool",
        arguments={"query": "test"},
    )
    responses = [
        LLMResponse(
            content="I'll use the test tool",
            model="test-model",
            finish_reason="tool_calls",
            tool_calls=[tool_call],
            usage={"total_tokens": 15},
        ),
        LLMResponse(
            content="Final Answer: The result is 42",
            model="test-model",
            finish_reason="stop",
            usage={"total_tokens": 10},
        ),
    ]
    return MockLLMProvider(responses=responses)


@pytest.fixture
def mock_llm_final_answer() -> MockLLMProvider:
    """Create a mock LLM that returns a final answer immediately."""
    responses = [
        LLMResponse(
            content="Final Answer: The answer is 42",
            model="test-model",
            finish_reason="stop",
            usage={"total_tokens": 10},
        ),
    ]
    return MockLLMProvider(responses=responses)


@pytest.fixture
def mock_llm_empty() -> MockLLMProvider:
    """Create a mock LLM that returns empty content."""
    responses = [
        LLMResponse(
            content="",
            model="test-model",
            finish_reason="stop",
            usage={"total_tokens": 5},
        ),
    ]
    return MockLLMProvider(responses=responses)


@pytest.fixture
def mock_llm_error() -> MockLLMProvider:
    """Create a mock LLM that raises errors."""
    provider = MockLLMProvider()

    async def error_generate(*args, **kwargs):
        raise RuntimeError("LLM error")

    provider.generate = error_generate
    return provider


# ============================================================================
# Mock Tools
# ============================================================================


class MockTool(BaseTool):
    """Mock tool for testing."""

    name = "test_tool"
    description = "A test tool"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Test query"},
        },
        "required": ["query"],
    }

    def __init__(self, return_value: Any = "tool result"):
        super().__init__()
        self.return_value = return_value
        self.call_count = 0
        self.last_kwargs: Optional[Dict[str, Any]] = None

    async def execute(self, **kwargs: Any) -> Any:
        """Execute the mock tool."""
        self.call_count += 1
        self.last_kwargs = kwargs
        return self.return_value


class FailingTool(BaseTool):
    """Tool that always fails."""

    name = "failing_tool"
    description = "A tool that fails"
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, error_message: str = "Tool failed"):
        super().__init__()
        self.error_message = error_message

    async def execute(self, **kwargs: Any) -> Any:
        """Raise an exception."""
        raise RuntimeError(self.error_message)


@pytest.fixture
def mock_tool() -> MockTool:
    """Create a mock tool."""
    return MockTool()


@pytest.fixture
def failing_tool() -> FailingTool:
    """Create a failing tool."""
    return FailingTool()


@pytest.fixture
def tool_registry(mock_tool: MockTool) -> ToolRegistry:
    """Create a tool registry with a mock tool."""
    registry = ToolRegistry()
    registry.register(mock_tool)
    return registry


@pytest.fixture
def tool_executor(tool_registry: ToolRegistry) -> ToolExecutor:
    """Create a tool executor with a registry."""
    return ToolExecutor(
        registry=tool_registry,
        config=ExecutionConfig(timeout=5.0),
    )


# ============================================================================
# Mock Agent
# ============================================================================


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def __init__(
        self,
        role: Optional[AgentRole] = None,
        result: Optional[TaskResult] = None,
        should_fail: bool = False,
    ):
        role = role or AgentRole(
            name="TestAgent",
            description="A test agent",
            capabilities=["testing"],
        )
        super().__init__(role=role)
        self._result = result
        self._should_fail = should_fail
        self.run_count = 0
        self.last_task: Optional[Task] = None

    async def run(self, task: str | Task) -> TaskResult:
        """Execute the mock agent."""
        self.run_count += 1

        if isinstance(task, str):
            task = Task(description=task)

        self.last_task = task

        if self._should_fail:
            raise RuntimeError("Agent failed")

        if self._result:
            return self._result

        return TaskResult(
            task_id=task.id,
            success=True,
            output="Mock agent result",
        )


@pytest.fixture
def mock_agent() -> MockAgent:
    """Create a mock agent."""
    return MockAgent()


@pytest.fixture
def failing_agent() -> MockAgent:
    """Create a failing agent."""
    return MockAgent(should_fail=True)


# ============================================================================
# Loop and Runner Fixtures
# ============================================================================


@pytest.fixture
def loop_config() -> LoopConfig:
    """Create a loop configuration."""
    return LoopConfig(
        max_iterations=5,
        timeout=30.0,
        continue_on_error=True,
        max_consecutive_errors=2,
    )


@pytest.fixture
def runner_config() -> RunnerConfig:
    """Create a runner configuration."""
    return RunnerConfig(
        max_concurrent=5,
        default_timeout=30.0,
        enable_metrics=True,
    )


@pytest.fixture
def agent_loop(
    mock_llm: MockLLMProvider,
    tool_registry: ToolRegistry,
    loop_config: LoopConfig,
) -> AgentLoop:
    """Create an agent loop."""
    return AgentLoop(
        llm=mock_llm,
        tool_registry=tool_registry,
        config=loop_config,
        system_prompt="You are a test agent.",
    )


@pytest.fixture
def agent_runner(runner_config: RunnerConfig) -> AgentRunner:
    """Create an agent runner."""
    return AgentRunner(config=runner_config)


# ============================================================================
# Hook Fixtures
# ============================================================================


class MockHook(Hook):
    """Mock hook for testing."""

    hook_types = {HookType.PRE_EXECUTE, HookType.POST_EXECUTE}
    priority = 0

    def __init__(self, name: str = "MockHook"):
        super().__init__(name=name)
        self.execute_count = 0
        self.last_context: Optional[HookContext] = None
        self.last_kwargs: Dict[str, Any] = {}

    async def execute(self, context: HookContext, **kwargs: Any) -> None:
        """Record hook execution."""
        self.execute_count += 1
        self.last_context = context
        self.last_kwargs = kwargs


class FailingHook(Hook):
    """Hook that always fails."""

    hook_types = {HookType.PRE_EXECUTE}

    def __init__(self, name: str = "FailingHook"):
        super().__init__(name=name)

    async def execute(self, context: HookContext, **kwargs: Any) -> None:
        """Raise an exception."""
        raise RuntimeError("Hook failed")


@pytest.fixture
def hook_registry() -> HookRegistry:
    """Create a hook registry."""
    return HookRegistry()


@pytest.fixture
def mock_hook() -> MockHook:
    """Create a mock hook."""
    return MockHook()


@pytest.fixture
def failing_hook() -> FailingHook:
    """Create a failing hook."""
    return FailingHook()


# ============================================================================
# Helper Functions
# ============================================================================


def create_llm_response_with_tool_calls(
    tool_name: str = "test_tool",
    tool_id: str = "call_123",
    arguments: Optional[Dict[str, Any]] = None,
    content: str = "",
) -> LLMResponse:
    """Helper to create an LLM response with tool calls."""
    tool_call = ToolCall(
        id=tool_id,
        name=tool_name,
        arguments=arguments or {},
    )
    return LLMResponse(
        content=content,
        model="test-model",
        finish_reason="tool_calls",
        tool_calls=[tool_call],
        usage={"total_tokens": 10},
    )


def create_final_answer_response(answer: str = "42") -> LLMResponse:
    """Helper to create a final answer response."""
    return LLMResponse(
        content=f"Final Answer: {answer}",
        model="test-model",
        finish_reason="stop",
        usage={"total_tokens": 5},
    )
