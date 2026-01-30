"""Unit tests for the base team pattern module."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import pytest

from agents_framework.agents import AgentRole, Task, TaskResult
from agents_framework.teams.patterns.base import (
    BasePattern,
    PatternContext,
    PatternResult,
    PatternStatus,
    StepResult,
    TeamPattern,
)

from ..conftest import MockAgent


# ============================================================================
# PatternContext Tests
# ============================================================================


class TestPatternContext:
    """Tests for the PatternContext class."""

    def test_context_creation(self):
        """Test creating a pattern context."""
        context = PatternContext()

        assert context.id is not None
        assert context.variables == {}
        assert context.history == []
        assert context.metadata == {}
        assert context.created_at is not None

    def test_context_with_initial_values(self):
        """Test context with initial values."""
        context = PatternContext(
            variables={"key": "value"},
            metadata={"source": "test"},
        )

        assert context.variables == {"key": "value"}
        assert context.metadata == {"source": "test"}

    def test_get_variable(self):
        """Test getting a variable."""
        context = PatternContext(variables={"name": "Alice"})

        assert context.get("name") == "Alice"

    def test_get_variable_default(self):
        """Test getting variable with default."""
        context = PatternContext()

        assert context.get("nonexistent", "default") == "default"

    def test_get_variable_none(self):
        """Test getting non-existent variable returns None."""
        context = PatternContext()

        assert context.get("nonexistent") is None

    def test_set_variable(self):
        """Test setting a variable."""
        context = PatternContext()

        context.set("key", "value")

        assert context.get("key") == "value"

    def test_update_variables(self):
        """Test updating multiple variables."""
        context = PatternContext(variables={"existing": "value"})

        context.update({"new1": "val1", "new2": "val2"})

        assert context.get("existing") == "value"
        assert context.get("new1") == "val1"
        assert context.get("new2") == "val2"

    def test_history_tracking(self):
        """Test that history can track step results."""
        context = PatternContext()

        step1 = StepResult(agent_id="agent_1", output="result1")
        step2 = StepResult(agent_id="agent_2", output="result2")

        context.history.append(step1)
        context.history.append(step2)

        assert len(context.history) == 2
        assert context.history[0].agent_id == "agent_1"


# ============================================================================
# StepResult Tests
# ============================================================================


class TestStepResult:
    """Tests for the StepResult dataclass."""

    def test_step_result_creation(self):
        """Test creating a step result."""
        result = StepResult(
            agent_id="agent_123",
            agent_name="researcher",
            success=True,
            output={"data": "output"},
        )

        assert result.step_id is not None
        assert result.agent_id == "agent_123"
        assert result.agent_name == "researcher"
        assert result.success
        assert result.output == {"data": "output"}
        assert result.error is None
        assert result.duration_ms == 0.0
        assert result.timestamp is not None

    def test_step_result_failure(self):
        """Test step result for failure."""
        result = StepResult(
            agent_id="agent_123",
            agent_name="researcher",
            success=False,
            error="Something went wrong",
        )

        assert not result.success
        assert result.error == "Something went wrong"

    def test_step_result_with_duration(self):
        """Test step result with duration."""
        result = StepResult(
            agent_id="agent_123",
            success=True,
            duration_ms=150.5,
        )

        assert result.duration_ms == 150.5

    def test_step_result_with_metadata(self):
        """Test step result with metadata."""
        result = StepResult(
            agent_id="agent_123",
            success=True,
            metadata={"retries": 2, "cached": False},
        )

        assert result.metadata == {"retries": 2, "cached": False}


# ============================================================================
# PatternResult Tests
# ============================================================================


class TestPatternResult:
    """Tests for the PatternResult dataclass."""

    def test_pattern_result_creation(self):
        """Test creating a pattern result."""
        result = PatternResult(
            pattern_name="sequential",
            status=PatternStatus.COMPLETED,
            final_output={"result": "success"},
        )

        assert result.pattern_name == "sequential"
        assert result.status == PatternStatus.COMPLETED
        assert result.final_output == {"result": "success"}
        assert result.success

    def test_pattern_result_success_property(self):
        """Test success property for different statuses."""
        completed = PatternResult(
            pattern_name="test",
            status=PatternStatus.COMPLETED,
        )
        failed = PatternResult(
            pattern_name="test",
            status=PatternStatus.FAILED,
        )
        pending = PatternResult(
            pattern_name="test",
            status=PatternStatus.PENDING,
        )

        assert completed.success
        assert not failed.success
        assert not pending.success

    def test_pattern_result_with_steps(self):
        """Test pattern result with step history."""
        steps = [
            StepResult(agent_id="agent_1", success=True),
            StepResult(agent_id="agent_2", success=True),
        ]

        result = PatternResult(
            pattern_name="sequential",
            status=PatternStatus.COMPLETED,
            steps=steps,
        )

        assert len(result.steps) == 2

    def test_pattern_result_with_error(self):
        """Test pattern result with error."""
        result = PatternResult(
            pattern_name="test",
            status=PatternStatus.FAILED,
            error="Execution failed",
        )

        assert not result.success
        assert result.error == "Execution failed"

    def test_pattern_result_with_context(self):
        """Test pattern result includes context."""
        context = PatternContext(variables={"key": "value"})

        result = PatternResult(
            pattern_name="test",
            status=PatternStatus.COMPLETED,
            context=context,
        )

        assert result.context is context
        assert result.context.get("key") == "value"


# ============================================================================
# PatternStatus Tests
# ============================================================================


class TestPatternStatus:
    """Tests for PatternStatus enum."""

    def test_all_statuses_defined(self):
        """Test that all expected statuses are defined."""
        assert PatternStatus.PENDING.value == "pending"
        assert PatternStatus.RUNNING.value == "running"
        assert PatternStatus.COMPLETED.value == "completed"
        assert PatternStatus.FAILED.value == "failed"
        assert PatternStatus.CANCELLED.value == "cancelled"


# ============================================================================
# BasePattern Tests
# ============================================================================


class TestBasePattern:
    """Tests for the BasePattern class."""

    def test_base_pattern_creation(self):
        """Test creating a base pattern."""
        pattern = BasePattern(name="test_pattern")

        assert pattern.name == "test_pattern"
        assert pattern.status == PatternStatus.PENDING

    def test_base_pattern_default_name(self):
        """Test base pattern default name."""
        pattern = BasePattern()

        assert pattern.name == "base"

    @pytest.mark.asyncio
    async def test_validate_agents_requires_one(self, mock_researcher: MockAgent):
        """Test default validation requires at least one agent."""
        pattern = BasePattern()

        assert await pattern.validate_agents([mock_researcher])
        assert not await pattern.validate_agents([])

    @pytest.mark.asyncio
    async def test_execute_raises_not_implemented(self, mock_researcher: MockAgent):
        """Test that execute raises NotImplementedError."""
        pattern = BasePattern()
        task = Task(description="Test task")

        with pytest.raises(NotImplementedError):
            await pattern.execute(task, [mock_researcher])

    def test_create_context_new(self):
        """Test creating a new context."""
        pattern = BasePattern()

        context = pattern._create_context()

        assert isinstance(context, PatternContext)
        assert context.variables == {}

    def test_create_context_uses_existing(self):
        """Test that existing context is returned."""
        pattern = BasePattern()
        existing = PatternContext(variables={"key": "value"})

        context = pattern._create_context(existing)

        assert context is existing

    def test_create_step_result(self, mock_researcher: MockAgent):
        """Test creating a step result from agent."""
        pattern = BasePattern()

        step = pattern._create_step_result(
            agent=mock_researcher,
            output={"data": "test"},
            success=True,
            duration_ms=100.5,
        )

        assert step.agent_id == mock_researcher.id
        assert step.agent_name == mock_researcher.role.name
        assert step.output == {"data": "test"}
        assert step.success
        assert step.duration_ms == 100.5

    def test_create_step_result_failure(self, mock_researcher: MockAgent):
        """Test creating a failed step result."""
        pattern = BasePattern()

        step = pattern._create_step_result(
            agent=mock_researcher,
            output=None,
            success=False,
            error="Agent failed",
        )

        assert not step.success
        assert step.error == "Agent failed"


# ============================================================================
# TeamPattern Protocol Tests
# ============================================================================


class TestTeamPatternProtocol:
    """Tests for the TeamPattern protocol."""

    def test_base_pattern_implements_protocol(self):
        """Test that BasePattern implements TeamPattern protocol."""
        # This is a runtime check that BasePattern satisfies TeamPattern
        pattern = BasePattern()

        # Check protocol attributes
        assert hasattr(pattern, "name")
        assert hasattr(pattern, "execute")
        assert hasattr(pattern, "validate_agents")


# ============================================================================
# Custom Pattern Tests
# ============================================================================


class ConcretePattern(BasePattern):
    """Concrete implementation for testing."""

    def __init__(self):
        super().__init__(name="concrete")
        self.execute_called = False

    async def execute(
        self,
        task: Task,
        agents: List[MockAgent],
        context: Optional[PatternContext] = None,
    ) -> PatternResult:
        """Execute the pattern."""
        self.execute_called = True
        self._status = PatternStatus.RUNNING

        ctx = self._create_context(context)
        steps = []

        for agent in agents:
            result = await agent.run(task)
            steps.append(
                self._create_step_result(
                    agent=agent,
                    output=result.output,
                    success=result.success,
                )
            )

        self._status = PatternStatus.COMPLETED

        return PatternResult(
            pattern_name=self.name,
            status=PatternStatus.COMPLETED,
            final_output=[s.output for s in steps],
            steps=steps,
            context=ctx,
        )


class TestConcretePattern:
    """Tests for a concrete pattern implementation."""

    @pytest.mark.asyncio
    async def test_execute_pattern(self, mock_researcher: MockAgent):
        """Test executing a concrete pattern."""
        pattern = ConcretePattern()
        task = Task(description="Test task")

        result = await pattern.execute(task, [mock_researcher])

        assert pattern.execute_called
        assert result.success
        assert pattern.status == PatternStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_with_multiple_agents(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test executing pattern with multiple agents."""
        pattern = ConcretePattern()
        task = Task(description="Test task")

        result = await pattern.execute(task, [mock_researcher, mock_analyst])

        assert len(result.steps) == 2
        assert mock_researcher.run_count == 1
        assert mock_analyst.run_count == 1

    @pytest.mark.asyncio
    async def test_execute_with_context(self, mock_researcher: MockAgent):
        """Test executing pattern with initial context."""
        pattern = ConcretePattern()
        task = Task(description="Test task")
        context = PatternContext(variables={"initial": "value"})

        result = await pattern.execute(task, [mock_researcher], context)

        assert result.context is context
        assert result.context.get("initial") == "value"
