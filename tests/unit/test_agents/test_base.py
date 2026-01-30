"""Unit tests for the base agent module."""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest

from agents_framework.agents.base import (
    AgentConfig,
    AgentRole,
    AgentStatus,
    BaseAgent,
    Task,
    TaskResult,
)

from .conftest import ConcreteAgent, ExceptionRaisingAgent


class TestAgentRole:
    """Tests for AgentRole dataclass."""

    def test_role_creation_minimal(self):
        """Test creating a role with minimal parameters."""
        role = AgentRole(name="test", description="A test role")
        assert role.name == "test"
        assert role.description == "A test role"
        assert role.capabilities == []
        assert role.instructions == ""

    def test_role_creation_full(self):
        """Test creating a role with all parameters."""
        role = AgentRole(
            name="coder",
            description="Writes code",
            capabilities=["python", "javascript"],
            instructions="Write clean code",
        )
        assert role.name == "coder"
        assert role.description == "Writes code"
        assert role.capabilities == ["python", "javascript"]
        assert role.instructions == "Write clean code"

    def test_role_hash(self):
        """Test that roles are hashable by name."""
        role1 = AgentRole(name="test", description="First")
        role2 = AgentRole(name="test", description="Second")
        role3 = AgentRole(name="other", description="Third")

        assert hash(role1) == hash(role2)
        assert hash(role1) != hash(role3)

    def test_role_equality(self):
        """Test role equality based on name."""
        role1 = AgentRole(name="test", description="First")
        role2 = AgentRole(name="test", description="Second")
        role3 = AgentRole(name="other", description="Third")

        assert role1 == role2
        assert role1 != role3
        assert role1 != "not a role"

    def test_role_in_set(self):
        """Test that roles can be used in sets."""
        role1 = AgentRole(name="test", description="First")
        role2 = AgentRole(name="test", description="Second")
        role3 = AgentRole(name="other", description="Third")

        roles = {role1, role2, role3}
        assert len(roles) == 2  # role1 and role2 are equal


class TestTask:
    """Tests for Task dataclass."""

    def test_task_creation_minimal(self):
        """Test creating a task with minimal parameters."""
        task = Task()
        assert task.id is not None
        assert task.description == ""
        assert task.context == {}
        assert task.required_capabilities == []
        assert task.priority == 0
        assert isinstance(task.created_at, datetime)

    def test_task_creation_full(self):
        """Test creating a task with all parameters."""
        task = Task(
            description="Complete the project",
            context={"deadline": "tomorrow"},
            required_capabilities=["code", "test"],
            priority=5,
        )
        assert task.description == "Complete the project"
        assert task.context == {"deadline": "tomorrow"}
        assert task.required_capabilities == ["code", "test"]
        assert task.priority == 5

    def test_task_id_is_uuid(self):
        """Test that task ID is a valid UUID."""
        task = Task()
        # Should not raise
        uuid.UUID(task.id)

    def test_task_unique_ids(self):
        """Test that each task gets a unique ID."""
        task1 = Task()
        task2 = Task()
        assert task1.id != task2.id


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_result_success(self):
        """Test creating a successful result."""
        result = TaskResult(
            task_id="task-123",
            success=True,
            output="Task completed",
        )
        assert result.task_id == "task-123"
        assert result.success is True
        assert result.output == "Task completed"
        assert result.error is None
        assert result.metadata == {}
        assert isinstance(result.completed_at, datetime)

    def test_result_failure(self):
        """Test creating a failed result."""
        result = TaskResult(
            task_id="task-456",
            success=False,
            error="Something went wrong",
        )
        assert result.task_id == "task-456"
        assert result.success is False
        assert result.output is None
        assert result.error == "Something went wrong"

    def test_result_with_metadata(self):
        """Test creating a result with metadata."""
        result = TaskResult(
            task_id="task-789",
            success=True,
            output="Done",
            metadata={"duration": 10.5, "steps": 5},
        )
        assert result.metadata == {"duration": 10.5, "steps": 5}


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = AgentConfig()
        assert config.name == "Agent"
        assert config.max_iterations == 10
        assert config.timeout == 300.0
        assert config.tools == []
        assert config.memory_enabled is True

    def test_config_custom(self):
        """Test custom configuration values."""
        config = AgentConfig(
            name="CustomAgent",
            max_iterations=5,
            timeout=60.0,
            tools=["search", "calculate"],
            memory_enabled=False,
        )
        assert config.name == "CustomAgent"
        assert config.max_iterations == 5
        assert config.timeout == 60.0
        assert config.tools == ["search", "calculate"]
        assert config.memory_enabled is False


class TestAgentStatus:
    """Tests for AgentStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.BUSY.value == "busy"
        assert AgentStatus.ERROR.value == "error"
        assert AgentStatus.TERMINATED.value == "terminated"

    def test_status_is_string_enum(self):
        """Test that status is a string enum."""
        assert AgentStatus.IDLE == "idle"
        assert str(AgentStatus.BUSY) == "busy"


class TestBaseAgent:
    """Tests for BaseAgent abstract base class."""

    def test_agent_initialization(self, basic_role, basic_config):
        """Test agent initialization."""
        agent = ConcreteAgent(role=basic_role, config=basic_config)

        assert agent.role == basic_role
        assert agent.config == basic_config
        assert agent.llm is None
        assert agent.status == AgentStatus.IDLE
        assert agent.parent_id is None
        assert agent.id is not None

    def test_agent_default_config(self, basic_role):
        """Test agent uses default config when none provided."""
        agent = ConcreteAgent(role=basic_role)
        assert agent.config is not None
        assert agent.config.name == "Agent"

    def test_agent_id_is_uuid(self, basic_role):
        """Test that agent ID is a valid UUID."""
        agent = ConcreteAgent(role=basic_role)
        uuid.UUID(agent.id)  # Should not raise

    def test_agent_unique_ids(self, basic_role):
        """Test that each agent gets a unique ID."""
        agent1 = ConcreteAgent(role=basic_role)
        agent2 = ConcreteAgent(role=basic_role)
        assert agent1.id != agent2.id

    def test_status_property(self, concrete_agent):
        """Test status getter and setter."""
        assert concrete_agent.status == AgentStatus.IDLE

        concrete_agent.status = AgentStatus.BUSY
        assert concrete_agent.status == AgentStatus.BUSY

        concrete_agent.status = AgentStatus.ERROR
        assert concrete_agent.status == AgentStatus.ERROR

    def test_parent_id_property(self, concrete_agent):
        """Test parent_id getter and setter."""
        assert concrete_agent.parent_id is None

        concrete_agent.parent_id = "parent-123"
        assert concrete_agent.parent_id == "parent-123"

        concrete_agent.parent_id = None
        assert concrete_agent.parent_id is None

    def test_has_capability(self, concrete_agent):
        """Test capability checking."""
        # From basic_role fixture: capabilities=["test", "mock"]
        assert concrete_agent.has_capability("test") is True
        assert concrete_agent.has_capability("mock") is True
        assert concrete_agent.has_capability("unknown") is False

    def test_repr(self, concrete_agent):
        """Test string representation."""
        repr_str = repr(concrete_agent)
        assert "ConcreteAgent" in repr_str
        assert concrete_agent.id in repr_str
        assert concrete_agent.role.name in repr_str

    @pytest.mark.asyncio
    async def test_run_with_string_task(self, concrete_agent):
        """Test running with a string task."""
        result = await concrete_agent.run("Do something")
        assert result.success is True
        assert "Do something" in result.output
        assert concrete_agent.last_task.description == "Do something"

    @pytest.mark.asyncio
    async def test_run_with_task_object(self, concrete_agent, basic_task):
        """Test running with a Task object."""
        result = await concrete_agent.run(basic_task)
        assert result.success is True
        assert concrete_agent.last_task == basic_task

    @pytest.mark.asyncio
    async def test_run_increments_count(self, concrete_agent):
        """Test that run increments the call count."""
        assert concrete_agent.run_count == 0
        await concrete_agent.run("Task 1")
        assert concrete_agent.run_count == 1
        await concrete_agent.run("Task 2")
        assert concrete_agent.run_count == 2

    @pytest.mark.asyncio
    async def test_execute_manages_status(self, concrete_agent, basic_task):
        """Test that execute manages agent status."""
        assert concrete_agent.status == AgentStatus.IDLE

        result = await concrete_agent.execute(basic_task)

        assert result.success is True
        assert concrete_agent.status == AgentStatus.IDLE

    @pytest.mark.asyncio
    async def test_execute_sets_error_status_on_exception(
        self, exception_agent, basic_task
    ):
        """Test that execute sets ERROR status on exception."""
        result = await exception_agent.execute(basic_task)

        assert result.success is False
        assert result.error is not None
        assert exception_agent.status == AgentStatus.ERROR

    @pytest.mark.asyncio
    async def test_execute_returns_error_result_on_exception(
        self, exception_agent, basic_task
    ):
        """Test that execute returns proper error result on exception."""
        result = await exception_agent.execute(basic_task)

        assert result.success is False
        assert result.task_id == basic_task.id
        assert "Agent error" in result.error

    @pytest.mark.asyncio
    async def test_failing_agent(self, failing_agent, basic_task):
        """Test agent that always fails."""
        result = await failing_agent.run(basic_task)
        assert result.success is False
        assert result.error == "Task failed"

    @pytest.mark.asyncio
    async def test_agent_with_custom_result(self, basic_role, basic_task):
        """Test agent returning custom result."""
        custom_result = TaskResult(
            task_id="custom",
            success=True,
            output={"data": "custom output"},
            metadata={"key": "value"},
        )
        agent = ConcreteAgent(role=basic_role, return_result=custom_result)

        result = await agent.run(basic_task)
        assert result.success is True
        assert result.output == {"data": "custom output"}
        assert result.metadata == {"key": "value"}

    @pytest.mark.asyncio
    async def test_intermittent_failure(self, basic_role, basic_task):
        """Test agent with intermittent failures."""
        agent = ConcreteAgent(role=basic_role, fail_times=2)

        # First two calls fail
        result1 = await agent.run(basic_task)
        assert result1.success is False

        result2 = await agent.run(basic_task)
        assert result2.success is False

        # Third call succeeds
        result3 = await agent.run(basic_task)
        assert result3.success is True

    @pytest.mark.asyncio
    async def test_agent_with_llm(self, basic_role, mock_llm):
        """Test agent with LLM provider."""
        agent = ConcreteAgent(role=basic_role, llm=mock_llm)
        assert agent.llm == mock_llm
