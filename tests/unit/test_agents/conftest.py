"""Local fixtures for agents module tests."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
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
from agents_framework.agents.supervisor import (
    DelegatedTask,
    DelegationStrategy,
    ExecutionMode,
    SupervisorAgent,
    SupervisorConfig,
)
from agents_framework.agents.worker import (
    WorkerAgent,
    WorkerConfig,
    WorkerProgress,
    WorkerStatus,
)
from agents_framework.agents.router import (
    RouterAgent,
    RouterConfig,
    RoutingDecision,
    RoutingRule,
    RoutingStrategy,
)
from agents_framework.agents.spawner import (
    AgentLifecycleState,
    AgentSpawner,
    AgentTemplate,
    SpawnedAgentInfo,
    SpawnPolicy,
)
from agents_framework.llm.base import (
    LLMConfig,
    LLMResponse,
    Message,
    MessageRole,
    ToolDefinition,
)


# ============================================================================
# Concrete Agent Implementations for Testing
# ============================================================================


class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""

    def __init__(
        self,
        role: AgentRole,
        llm: Any = None,
        config: Optional[AgentConfig] = None,
        return_result: Optional[TaskResult] = None,
        should_fail: bool = False,
        fail_times: int = 0,
        delay: float = 0.0,
    ):
        super().__init__(role=role, llm=llm, config=config)
        self.return_result = return_result
        self.should_fail = should_fail
        self.fail_times = fail_times
        self.delay = delay
        self.run_count = 0
        self.last_task: Optional[Task] = None

    async def run(self, task: str | Task) -> TaskResult:
        """Execute a task."""
        self.run_count += 1

        if isinstance(task, str):
            task = Task(description=task)

        self.last_task = task

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.should_fail:
            return TaskResult(
                task_id=task.id,
                success=False,
                error="Task failed",
            )

        if self.fail_times > 0 and self.run_count <= self.fail_times:
            return TaskResult(
                task_id=task.id,
                success=False,
                error=f"Intermittent failure {self.run_count}",
            )

        if self.return_result:
            return TaskResult(
                task_id=task.id,
                success=self.return_result.success,
                output=self.return_result.output,
                error=self.return_result.error,
                metadata=self.return_result.metadata,
            )

        return TaskResult(
            task_id=task.id,
            success=True,
            output=f"Completed: {task.description}",
        )


class SlowAgent(BaseAgent):
    """Agent that takes a configurable amount of time to execute."""

    def __init__(
        self,
        role: AgentRole,
        llm: Any = None,
        config: Optional[AgentConfig] = None,
        delay: float = 1.0,
    ):
        super().__init__(role=role, llm=llm, config=config)
        self.delay = delay

    async def run(self, task: str | Task) -> TaskResult:
        """Execute a task with delay."""
        if isinstance(task, str):
            task = Task(description=task)

        await asyncio.sleep(self.delay)
        return TaskResult(task_id=task.id, success=True, output="Slow result")


class ExceptionRaisingAgent(BaseAgent):
    """Agent that raises an exception."""

    def __init__(
        self,
        role: AgentRole,
        llm: Any = None,
        config: Optional[AgentConfig] = None,
        exception: Optional[Exception] = None,
    ):
        super().__init__(role=role, llm=llm, config=config)
        self.exception = exception or RuntimeError("Agent error")

    async def run(self, task: str | Task) -> TaskResult:
        """Raise an exception."""
        raise self.exception


# ============================================================================
# Mock LLM Provider
# ============================================================================


class MockLLMProvider:
    """Mock LLM provider for testing agents."""

    def __init__(
        self,
        responses: Optional[List[LLMResponse]] = None,
        should_fail: bool = False,
    ):
        self.responses = responses or []
        self.should_fail = should_fail
        self.call_count = 0
        self.last_messages: Optional[List[Message]] = None

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a mock response."""
        self.call_count += 1
        self.last_messages = messages

        if self.should_fail:
            raise Exception("LLM generation failed")

        if self.responses:
            idx = min(self.call_count - 1, len(self.responses) - 1)
            return self.responses[idx]

        return LLMResponse(
            content='{"agent_id": "test-agent", "confidence": 0.9, "reason": "Best match"}',
            model="test-model",
            finish_reason="stop",
        )

    def supports_tools(self) -> bool:
        """Return True for mock provider."""
        return True


# ============================================================================
# Role Fixtures
# ============================================================================


@pytest.fixture
def basic_role() -> AgentRole:
    """Create a basic agent role."""
    return AgentRole(
        name="test-agent",
        description="A test agent for unit testing",
        capabilities=["test", "mock"],
        instructions="Follow test instructions",
    )


@pytest.fixture
def coding_role() -> AgentRole:
    """Create a coding agent role."""
    return AgentRole(
        name="coder",
        description="Writes and reviews code",
        capabilities=["code", "debug", "review"],
        instructions="Write clean, efficient code",
    )


@pytest.fixture
def research_role() -> AgentRole:
    """Create a research agent role."""
    return AgentRole(
        name="researcher",
        description="Researches and analyzes information",
        capabilities=["search", "analyze", "summarize"],
        instructions="Provide thorough research",
    )


@pytest.fixture
def supervisor_role() -> AgentRole:
    """Create a supervisor agent role."""
    return AgentRole(
        name="supervisor",
        description="Coordinates and delegates tasks",
        capabilities=["delegate", "coordinate", "aggregate"],
        instructions="Efficiently distribute work",
    )


@pytest.fixture
def router_role() -> AgentRole:
    """Create a router agent role."""
    return AgentRole(
        name="router",
        description="Routes tasks to appropriate agents",
        capabilities=["routing", "classification"],
        instructions="Route tasks efficiently",
    )


# ============================================================================
# Config Fixtures
# ============================================================================


@pytest.fixture
def basic_config() -> AgentConfig:
    """Create a basic agent configuration."""
    return AgentConfig(
        name="test-agent",
        max_iterations=10,
        timeout=30.0,
        tools=["test_tool"],
        memory_enabled=True,
    )


@pytest.fixture
def supervisor_config() -> SupervisorConfig:
    """Create a supervisor configuration."""
    return SupervisorConfig(
        name="supervisor",
        delegation_strategy=DelegationStrategy.CAPABILITY_MATCH,
        execution_mode=ExecutionMode.PARALLEL,
        max_workers=5,
        task_timeout=10.0,
        max_retries=2,
        retry_delay=0.1,
        fallback_enabled=True,
    )


@pytest.fixture
def worker_config() -> WorkerConfig:
    """Create a worker configuration."""
    return WorkerConfig(
        name="worker",
        max_iterations=5,
        timeout=30.0,
        allowed_tools=["search", "calculate"],
        denied_tools=["dangerous_tool"],
        report_progress=True,
        progress_interval=0.1,
        allow_cancellation=True,
    )


@pytest.fixture
def router_config() -> RouterConfig:
    """Create a router configuration."""
    return RouterConfig(
        default_strategy=RoutingStrategy.RULE_BASED,
        fallback_enabled=True,
        max_fallback_attempts=3,
        confidence_threshold=0.7,
        timeout_per_agent=10.0,
    )


# ============================================================================
# Task Fixtures
# ============================================================================


@pytest.fixture
def basic_task() -> Task:
    """Create a basic task."""
    return Task(
        description="Test task description",
        context={"key": "value"},
        required_capabilities=["test"],
        priority=1,
    )


@pytest.fixture
def coding_task() -> Task:
    """Create a coding task."""
    return Task(
        description="Write a function to calculate factorial",
        context={"language": "python"},
        required_capabilities=["code"],
        priority=2,
    )


@pytest.fixture
def research_task() -> Task:
    """Create a research task."""
    return Task(
        description="Research the latest AI trends",
        context={"depth": "comprehensive"},
        required_capabilities=["search", "analyze"],
        priority=1,
    )


# ============================================================================
# Agent Fixtures
# ============================================================================


@pytest.fixture
def concrete_agent(basic_role: AgentRole, basic_config: AgentConfig) -> ConcreteAgent:
    """Create a concrete agent for testing."""
    return ConcreteAgent(role=basic_role, config=basic_config)


@pytest.fixture
def failing_agent(basic_role: AgentRole) -> ConcreteAgent:
    """Create an agent that always fails."""
    return ConcreteAgent(role=basic_role, should_fail=True)


@pytest.fixture
def slow_agent(basic_role: AgentRole) -> SlowAgent:
    """Create a slow agent for timeout testing."""
    return SlowAgent(role=basic_role, delay=2.0)


@pytest.fixture
def exception_agent(basic_role: AgentRole) -> ExceptionRaisingAgent:
    """Create an agent that raises exceptions."""
    return ExceptionRaisingAgent(role=basic_role)


@pytest.fixture
def coding_agent(coding_role: AgentRole) -> ConcreteAgent:
    """Create a coding agent."""
    return ConcreteAgent(role=coding_role)


@pytest.fixture
def research_agent(research_role: AgentRole) -> ConcreteAgent:
    """Create a research agent."""
    return ConcreteAgent(role=research_role)


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """Create a mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def mock_llm_with_routing_response() -> MockLLMProvider:
    """Create a mock LLM with routing response."""
    return MockLLMProvider(
        responses=[
            LLMResponse(
                content='{"agent_id": "test-id", "confidence": 0.95, "reason": "Best match for task"}',
                model="test-model",
                finish_reason="stop",
            )
        ]
    )


# ============================================================================
# Supervisor Fixtures
# ============================================================================


@pytest.fixture
def supervisor_agent(
    supervisor_role: AgentRole, supervisor_config: SupervisorConfig
) -> SupervisorAgent:
    """Create a supervisor agent."""
    return SupervisorAgent(role=supervisor_role, config=supervisor_config)


@pytest.fixture
def supervisor_with_workers(
    supervisor_agent: SupervisorAgent,
    coding_agent: ConcreteAgent,
    research_agent: ConcreteAgent,
) -> SupervisorAgent:
    """Create a supervisor with workers."""
    supervisor_agent.add_worker(coding_agent)
    supervisor_agent.add_worker(research_agent)
    return supervisor_agent


# ============================================================================
# Router Fixtures
# ============================================================================


@pytest.fixture
def router_agent(router_config: RouterConfig) -> RouterAgent:
    """Create a router agent."""
    return RouterAgent(router_config=router_config)


@pytest.fixture
def router_with_agents(
    router_agent: RouterAgent,
    coding_agent: ConcreteAgent,
    research_agent: ConcreteAgent,
) -> RouterAgent:
    """Create a router with registered agents."""
    router_agent.register_agent(coding_agent, capabilities=["code", "debug"])
    router_agent.register_agent(research_agent, capabilities=["search", "analyze"])
    return router_agent


# ============================================================================
# Spawner Fixtures
# ============================================================================


@pytest.fixture
def agent_spawner() -> AgentSpawner:
    """Create an agent spawner."""
    return AgentSpawner()


@pytest.fixture
def worker_template(basic_role: AgentRole) -> AgentTemplate:
    """Create a worker template."""
    return AgentTemplate(
        name="worker",
        role=basic_role,
        agent_class=ConcreteAgent,
        spawn_policy=SpawnPolicy.ON_DEMAND,
    )


@pytest.fixture
def pooled_template(basic_role: AgentRole) -> AgentTemplate:
    """Create a pooled template."""
    return AgentTemplate(
        name="pooled_worker",
        role=basic_role,
        agent_class=ConcreteAgent,
        spawn_policy=SpawnPolicy.POOLED,
        pool_size=3,
    )


@pytest.fixture
def singleton_template(basic_role: AgentRole) -> AgentTemplate:
    """Create a singleton template."""
    return AgentTemplate(
        name="singleton_worker",
        role=basic_role,
        agent_class=ConcreteAgent,
        spawn_policy=SpawnPolicy.SINGLETON,
    )


@pytest.fixture
def limited_template(basic_role: AgentRole) -> AgentTemplate:
    """Create a template with max instances limit."""
    return AgentTemplate(
        name="limited_worker",
        role=basic_role,
        agent_class=ConcreteAgent,
        spawn_policy=SpawnPolicy.ON_DEMAND,
        max_instances=2,
    )
