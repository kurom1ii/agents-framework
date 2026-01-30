"""Local fixtures for teams module tests."""

from __future__ import annotations

from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents_framework.agents import (
    AgentConfig,
    AgentRole,
    AgentStatus,
    BaseAgent,
    Task,
    TaskResult,
)
from agents_framework.teams.registry import AgentRegistry
from agents_framework.teams.router import (
    AgentMessage,
    MessagePriority,
    MessageQueue,
    MessageRouter,
    RoutingStrategy,
)
from agents_framework.teams.team import Team, TeamConfig, TeamExecutionStrategy


# ============================================================================
# Mock Agent Classes
# ============================================================================


class MockAgent(BaseAgent):
    """Mock agent for testing team functionality."""

    def __init__(
        self,
        role: AgentRole,
        config: Optional[AgentConfig] = None,
        return_value: Any = None,
        should_fail: bool = False,
        fail_message: str = "Mock agent failure",
        delay: float = 0.0,
    ):
        super().__init__(role=role, config=config)
        self.return_value = return_value
        self.should_fail = should_fail
        self.fail_message = fail_message
        self.delay = delay
        self.run_count = 0
        self.received_tasks: List[Task] = []

    async def run(self, task: str | Task) -> TaskResult:
        """Execute the mock task."""
        import asyncio

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if isinstance(task, str):
            task = Task(description=task)

        self.run_count += 1
        self.received_tasks.append(task)

        if self.should_fail:
            return TaskResult(
                task_id=task.id,
                success=False,
                error=self.fail_message,
            )

        return TaskResult(
            task_id=task.id,
            success=True,
            output=self.return_value or f"Result from {self.role.name}",
        )


# ============================================================================
# Agent Role Fixtures
# ============================================================================


@pytest.fixture
def researcher_role() -> AgentRole:
    """Create a researcher agent role."""
    return AgentRole(
        name="researcher",
        description="Research and gather information",
        capabilities=["search", "analyze", "summarize"],
    )


@pytest.fixture
def analyst_role() -> AgentRole:
    """Create an analyst agent role."""
    return AgentRole(
        name="analyst",
        description="Analyze data and provide insights",
        capabilities=["analyze", "calculate", "report"],
    )


@pytest.fixture
def writer_role() -> AgentRole:
    """Create a writer agent role."""
    return AgentRole(
        name="writer",
        description="Write content and documentation",
        capabilities=["write", "edit", "format"],
    )


@pytest.fixture
def supervisor_role() -> AgentRole:
    """Create a supervisor agent role."""
    return AgentRole(
        name="supervisor",
        description="Supervise and coordinate other agents",
        capabilities=["delegate", "coordinate", "review"],
    )


@pytest.fixture
def coder_role() -> AgentRole:
    """Create a coder agent role."""
    return AgentRole(
        name="coder",
        description="Write and debug code",
        capabilities=["code", "debug", "test"],
    )


# ============================================================================
# Mock Agent Fixtures
# ============================================================================


@pytest.fixture
def mock_researcher(researcher_role: AgentRole) -> MockAgent:
    """Create a mock researcher agent."""
    return MockAgent(
        role=researcher_role,
        config=AgentConfig(name="ResearcherAgent"),
        return_value="Research findings",
    )


@pytest.fixture
def mock_analyst(analyst_role: AgentRole) -> MockAgent:
    """Create a mock analyst agent."""
    return MockAgent(
        role=analyst_role,
        config=AgentConfig(name="AnalystAgent"),
        return_value="Analysis results",
    )


@pytest.fixture
def mock_writer(writer_role: AgentRole) -> MockAgent:
    """Create a mock writer agent."""
    return MockAgent(
        role=writer_role,
        config=AgentConfig(name="WriterAgent"),
        return_value="Written content",
    )


@pytest.fixture
def mock_supervisor(supervisor_role: AgentRole) -> MockAgent:
    """Create a mock supervisor agent."""
    return MockAgent(
        role=supervisor_role,
        config=AgentConfig(name="SupervisorAgent"),
        return_value="Supervision complete",
    )


@pytest.fixture
def mock_coder(coder_role: AgentRole) -> MockAgent:
    """Create a mock coder agent."""
    return MockAgent(
        role=coder_role,
        config=AgentConfig(name="CoderAgent"),
        return_value="Code output",
    )


@pytest.fixture
def failing_agent(researcher_role: AgentRole) -> MockAgent:
    """Create a mock agent that always fails."""
    return MockAgent(
        role=researcher_role,
        config=AgentConfig(name="FailingAgent"),
        should_fail=True,
        fail_message="Intentional test failure",
    )


@pytest.fixture
def slow_agent(researcher_role: AgentRole) -> MockAgent:
    """Create a mock agent with a delay."""
    return MockAgent(
        role=researcher_role,
        config=AgentConfig(name="SlowAgent"),
        return_value="Slow result",
        delay=0.5,
    )


# ============================================================================
# Router Fixtures
# ============================================================================


@pytest.fixture
def message_router() -> MessageRouter:
    """Create a message router."""
    return MessageRouter()


@pytest.fixture
def message_queue() -> MessageQueue:
    """Create a message queue."""
    return MessageQueue(max_size=100)


@pytest.fixture
def sample_message() -> AgentMessage:
    """Create a sample agent message."""
    return AgentMessage(
        sender_id="agent_1",
        recipient_id="agent_2",
        content={"type": "test", "data": "Hello"},
        priority=MessagePriority.NORMAL,
        strategy=RoutingStrategy.DIRECT,
    )


@pytest.fixture
def urgent_message() -> AgentMessage:
    """Create an urgent priority message."""
    return AgentMessage(
        sender_id="agent_1",
        recipient_id="agent_2",
        content={"type": "urgent", "data": "Critical"},
        priority=MessagePriority.URGENT,
        strategy=RoutingStrategy.DIRECT,
    )


@pytest.fixture
def broadcast_message() -> AgentMessage:
    """Create a broadcast message."""
    return AgentMessage(
        sender_id="agent_1",
        content={"type": "broadcast", "data": "Announcement"},
        strategy=RoutingStrategy.BROADCAST,
    )


@pytest.fixture
def topic_message() -> AgentMessage:
    """Create a topic-based message."""
    return AgentMessage(
        sender_id="agent_1",
        topic="updates",
        content={"type": "topic", "data": "Update info"},
        strategy=RoutingStrategy.TOPIC,
    )


# ============================================================================
# Registry Fixtures
# ============================================================================


@pytest.fixture
def agent_registry() -> AgentRegistry:
    """Create an agent registry."""
    return AgentRegistry()


@pytest.fixture
def populated_registry(
    mock_researcher: MockAgent,
    mock_analyst: MockAgent,
    mock_writer: MockAgent,
) -> AgentRegistry:
    """Create a registry with pre-registered agents."""
    registry = AgentRegistry()
    registry.register(mock_researcher)
    registry.register(mock_analyst)
    registry.register(mock_writer)
    return registry


# ============================================================================
# Team Fixtures
# ============================================================================


@pytest.fixture
def team_config() -> TeamConfig:
    """Create a default team configuration."""
    return TeamConfig(
        name="TestTeam",
        strategy=TeamExecutionStrategy.COLLABORATIVE,
        max_concurrent_tasks=5,
        task_timeout=30.0,
        enable_messaging=True,
    )


@pytest.fixture
def team(team_config: TeamConfig) -> Team:
    """Create a team with default configuration."""
    return Team(config=team_config)


@pytest.fixture
def populated_team(
    team_config: TeamConfig,
    mock_researcher: MockAgent,
    mock_analyst: MockAgent,
    mock_writer: MockAgent,
) -> Team:
    """Create a team with pre-registered members."""
    team = Team(config=team_config)
    team.add_member(mock_researcher)
    team.add_member(mock_analyst)
    team.add_member(mock_writer)
    return team


@pytest.fixture
def hierarchical_team_config() -> TeamConfig:
    """Create a hierarchical team configuration."""
    return TeamConfig(
        name="HierarchicalTeam",
        strategy=TeamExecutionStrategy.HIERARCHICAL,
        max_concurrent_tasks=5,
        task_timeout=30.0,
    )


@pytest.fixture
def sequential_team_config() -> TeamConfig:
    """Create a sequential team configuration."""
    return TeamConfig(
        name="SequentialTeam",
        strategy=TeamExecutionStrategy.ROUND_ROBIN,
        max_concurrent_tasks=1,
        task_timeout=30.0,
    )


# ============================================================================
# Task Fixtures
# ============================================================================


@pytest.fixture
def sample_task() -> Task:
    """Create a sample task."""
    return Task(
        description="Perform a test operation",
        context={"test_key": "test_value"},
    )


@pytest.fixture
def complex_task() -> Task:
    """Create a complex task with capabilities."""
    return Task(
        description="Analyze and summarize data",
        context={"data": [1, 2, 3, 4, 5]},
        required_capabilities=["analyze", "summarize"],
        priority=2,
    )
