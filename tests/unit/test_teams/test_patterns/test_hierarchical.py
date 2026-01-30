"""Unit tests for the hierarchical team pattern module."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import pytest

from agents_framework.agents import AgentRole, Task, TaskResult
from agents_framework.teams.patterns.base import PatternContext, PatternStatus, StepResult
from agents_framework.teams.patterns.hierarchical import (
    EscalationLevel,
    EscalationRequest,
    HierarchicalConfig,
    HierarchicalPattern,
    SubTask,
)

from ..conftest import MockAgent


# ============================================================================
# SubTask Tests
# ============================================================================


class TestSubTask:
    """Tests for the SubTask dataclass."""

    def test_subtask_creation(self):
        """Test creating a subtask."""
        subtask = SubTask(
            id="subtask_1",
            description="Analyze the data",
        )

        assert subtask.id == "subtask_1"
        assert subtask.description == "Analyze the data"
        assert subtask.assigned_agent_id is None
        assert subtask.dependencies == []
        assert subtask.priority == 0
        assert subtask.metadata == {}

    def test_subtask_with_dependencies(self):
        """Test subtask with dependencies."""
        subtask = SubTask(
            id="subtask_2",
            description="Summarize findings",
            dependencies=["subtask_1"],
            priority=1,
        )

        assert subtask.dependencies == ["subtask_1"]
        assert subtask.priority == 1

    def test_subtask_with_assignment(self):
        """Test subtask with agent assignment."""
        subtask = SubTask(
            id="subtask_1",
            description="Research topic",
            assigned_agent_id="agent_123",
        )

        assert subtask.assigned_agent_id == "agent_123"


# ============================================================================
# EscalationRequest Tests
# ============================================================================


class TestEscalationRequest:
    """Tests for the EscalationRequest dataclass."""

    def test_escalation_request_creation(self):
        """Test creating an escalation request."""
        request = EscalationRequest(
            from_agent_id="worker_1",
            level=EscalationLevel.ERROR,
            reason="Unable to complete task",
        )

        assert request.from_agent_id == "worker_1"
        assert request.level == EscalationLevel.ERROR
        assert request.reason == "Unable to complete task"
        assert request.context == {}
        assert request.original_task is None

    def test_escalation_request_with_context(self):
        """Test escalation request with context."""
        subtask = SubTask(id="sub_1", description="Test")
        request = EscalationRequest(
            from_agent_id="worker_1",
            level=EscalationLevel.CRITICAL,
            reason="Critical failure",
            context={"error_code": 500},
            original_task=subtask,
        )

        assert request.context == {"error_code": 500}
        assert request.original_task is subtask


class TestEscalationLevel:
    """Tests for EscalationLevel enum."""

    def test_all_levels_defined(self):
        """Test that all escalation levels are defined."""
        assert EscalationLevel.NONE.value == "none"
        assert EscalationLevel.WARNING.value == "warning"
        assert EscalationLevel.ERROR.value == "error"
        assert EscalationLevel.CRITICAL.value == "critical"


# ============================================================================
# HierarchicalConfig Tests
# ============================================================================


class TestHierarchicalConfig:
    """Tests for the HierarchicalConfig dataclass."""

    def test_default_config(self):
        """Test default hierarchical configuration."""
        config = HierarchicalConfig()

        assert config.max_hierarchy_depth == 3
        assert config.max_parallel_workers == 5
        assert config.escalation_threshold == 2
        assert config.synthesis_strategy == "aggregate"
        assert config.timeout_per_subtask == 60.0

    def test_custom_config(self):
        """Test custom hierarchical configuration."""
        config = HierarchicalConfig(
            max_hierarchy_depth=5,
            max_parallel_workers=10,
            escalation_threshold=3,
            synthesis_strategy="reduce",
            timeout_per_subtask=30.0,
        )

        assert config.max_hierarchy_depth == 5
        assert config.max_parallel_workers == 10
        assert config.escalation_threshold == 3
        assert config.synthesis_strategy == "reduce"
        assert config.timeout_per_subtask == 30.0


# ============================================================================
# HierarchicalPattern Creation Tests
# ============================================================================


class TestHierarchicalPatternCreation:
    """Tests for HierarchicalPattern creation."""

    def test_pattern_creation(self):
        """Test creating a hierarchical pattern."""
        pattern = HierarchicalPattern()

        assert pattern.name == "hierarchical"
        assert pattern.status == PatternStatus.PENDING
        assert pattern.config.max_parallel_workers == 5

    def test_pattern_with_config(self):
        """Test creating pattern with custom config."""
        config = HierarchicalConfig(max_parallel_workers=3)
        pattern = HierarchicalPattern(config=config)

        assert pattern.config.max_parallel_workers == 3

    def test_pattern_with_custom_functions(self):
        """Test creating pattern with custom decompose/synthesize functions."""
        async def custom_decompose(task, context):
            return [SubTask(id="sub_1", description="Custom subtask")]

        async def custom_synthesize(results, context):
            return {"custom": "output"}

        pattern = HierarchicalPattern(
            decompose_fn=custom_decompose,
            synthesize_fn=custom_synthesize,
        )

        assert pattern._decompose_fn is not None
        assert pattern._synthesize_fn is not None


# ============================================================================
# Agent Validation Tests
# ============================================================================


class TestAgentValidation:
    """Tests for agent validation in hierarchical pattern."""

    @pytest.mark.asyncio
    async def test_validate_with_supervisor_only(self, mock_supervisor: MockAgent):
        """Test validation with only supervisor."""
        pattern = HierarchicalPattern()

        # At least one agent (supervisor) is valid
        assert await pattern.validate_agents([mock_supervisor])

    @pytest.mark.asyncio
    async def test_validate_with_supervisor_and_workers(
        self,
        mock_supervisor: MockAgent,
        mock_researcher: MockAgent,
    ):
        """Test validation with supervisor and workers."""
        pattern = HierarchicalPattern()

        assert await pattern.validate_agents([mock_supervisor, mock_researcher])

    @pytest.mark.asyncio
    async def test_validate_empty_list(self):
        """Test validation with empty agent list."""
        pattern = HierarchicalPattern()

        # Need at least one agent
        assert not await pattern.validate_agents([])


# ============================================================================
# Execution Tests
# ============================================================================


class TestHierarchicalExecution:
    """Tests for hierarchical pattern execution."""

    @pytest.mark.asyncio
    async def test_basic_execution(
        self,
        mock_supervisor: MockAgent,
        mock_researcher: MockAgent,
    ):
        """Test basic hierarchical execution."""
        pattern = HierarchicalPattern()
        task = Task(description="Research and analyze topic")

        result = await pattern.execute(task, [mock_supervisor, mock_researcher])

        assert result.pattern_name == "hierarchical"
        assert result.status == PatternStatus.COMPLETED
        assert len(result.steps) > 0

    @pytest.mark.asyncio
    async def test_execution_with_no_workers(self, mock_supervisor: MockAgent):
        """Test execution with only supervisor."""
        pattern = HierarchicalPattern()
        task = Task(description="Simple task")

        result = await pattern.execute(task, [mock_supervisor])

        assert result.status == PatternStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execution_fails_with_no_agents(self):
        """Test execution fails with no agents."""
        pattern = HierarchicalPattern()
        task = Task(description="Test task")

        result = await pattern.execute(task, [])

        assert result.status == PatternStatus.FAILED
        assert "Invalid agents" in result.error

    @pytest.mark.asyncio
    async def test_execution_uses_initial_context(
        self,
        mock_supervisor: MockAgent,
    ):
        """Test that execution uses provided context."""
        pattern = HierarchicalPattern()
        task = Task(description="Test task")
        context = PatternContext(variables={"initial": "value"})

        result = await pattern.execute(task, [mock_supervisor], context)

        assert result.context is context
        assert result.context.get("initial") == "value"


# ============================================================================
# Task Decomposition Tests
# ============================================================================


class TestTaskDecomposition:
    """Tests for task decomposition functionality."""

    @pytest.mark.asyncio
    async def test_custom_decompose_function(self, mock_supervisor: MockAgent):
        """Test using a custom decompose function."""
        expected_subtasks = [
            SubTask(id="sub_1", description="Part 1"),
            SubTask(id="sub_2", description="Part 2"),
        ]

        async def custom_decompose(task, context):
            return expected_subtasks

        pattern = HierarchicalPattern(decompose_fn=custom_decompose)
        task = Task(description="Main task")

        result = await pattern.execute(task, [mock_supervisor])

        # Context should contain the subtasks
        assert result.context.get("subtasks") == expected_subtasks

    def test_parse_subtasks_from_list_of_dicts(self):
        """Test parsing subtasks from list of dictionaries."""
        pattern = HierarchicalPattern()
        output = [
            {"id": "sub_1", "description": "Task 1", "priority": 2},
            {"id": "sub_2", "description": "Task 2", "dependencies": ["sub_1"]},
        ]

        subtasks = pattern._parse_subtasks(output)

        assert len(subtasks) == 2
        assert subtasks[0].id == "sub_1"
        assert subtasks[0].priority == 2
        assert subtasks[1].dependencies == ["sub_1"]

    def test_parse_subtasks_from_list_of_strings(self):
        """Test parsing subtasks from list of strings."""
        pattern = HierarchicalPattern()
        output = ["Research topic", "Analyze data", "Write report"]

        subtasks = pattern._parse_subtasks(output)

        assert len(subtasks) == 3
        assert subtasks[0].description == "Research topic"
        assert subtasks[0].id == "subtask_0"

    def test_parse_subtasks_from_string(self):
        """Test parsing subtasks from single string."""
        pattern = HierarchicalPattern()
        output = "Single task description"

        subtasks = pattern._parse_subtasks(output)

        assert len(subtasks) == 1
        assert subtasks[0].description == "Single task description"

    def test_parse_subtasks_fallback(self):
        """Test parsing subtasks with unsupported format."""
        pattern = HierarchicalPattern()
        output = 12345  # Unsupported type

        subtasks = pattern._parse_subtasks(output)

        assert len(subtasks) == 1
        assert subtasks[0].description == "12345"


# ============================================================================
# Subtask Assignment Tests
# ============================================================================


class TestSubtaskAssignment:
    """Tests for subtask assignment to workers."""

    @pytest.mark.asyncio
    async def test_round_robin_assignment(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test round-robin subtask assignment."""
        pattern = HierarchicalPattern()
        context = PatternContext()

        subtasks = [
            SubTask(id="sub_1", description="Task 1"),
            SubTask(id="sub_2", description="Task 2"),
            SubTask(id="sub_3", description="Task 3"),
        ]

        assignments = await pattern._assign_subtasks(
            subtasks, [mock_researcher, mock_analyst], context
        )

        # Round-robin: 0, 1, 0
        assert assignments["sub_1"].assigned_agent_id == mock_researcher.id
        assert assignments["sub_2"].assigned_agent_id == mock_analyst.id
        assert assignments["sub_3"].assigned_agent_id == mock_researcher.id

    @pytest.mark.asyncio
    async def test_assignment_with_no_workers(self):
        """Test assignment with no workers available."""
        pattern = HierarchicalPattern()
        context = PatternContext()

        subtasks = [SubTask(id="sub_1", description="Task 1")]

        assignments = await pattern._assign_subtasks(subtasks, [], context)

        assert assignments["sub_1"].assigned_agent_id is None


# ============================================================================
# Subtask Execution Tests
# ============================================================================


class TestSubtaskExecution:
    """Tests for subtask execution."""

    @pytest.mark.asyncio
    async def test_parallel_execution(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test that subtasks without dependencies run in parallel."""
        pattern = HierarchicalPattern()
        context = PatternContext()
        steps: List[StepResult] = []

        subtasks = [
            SubTask(id="sub_1", description="Task 1"),
            SubTask(id="sub_2", description="Task 2"),
        ]

        # Assign tasks
        assignments = await pattern._assign_subtasks(
            subtasks, [mock_researcher, mock_analyst], context
        )

        results = await pattern._execute_subtasks(
            assignments, [mock_researcher, mock_analyst], context, steps
        )

        assert len(results) == 2
        assert results["sub_1"].success
        assert results["sub_2"].success

    @pytest.mark.asyncio
    async def test_dependency_ordering(
        self,
        mock_researcher: MockAgent,
    ):
        """Test that dependencies are respected."""
        pattern = HierarchicalPattern()
        context = PatternContext()
        steps: List[StepResult] = []

        subtasks = [
            SubTask(id="sub_1", description="First task"),
            SubTask(id="sub_2", description="Second task", dependencies=["sub_1"]),
        ]

        for st in subtasks:
            st.assigned_agent_id = mock_researcher.id

        assignments = {st.id: st for st in subtasks}

        results = await pattern._execute_subtasks(
            assignments, [mock_researcher], context, steps
        )

        # Both should complete
        assert len(results) == 2
        # sub_2 should have access to sub_1's result
        assert results["sub_1"].success
        assert results["sub_2"].success

    @pytest.mark.asyncio
    async def test_timeout_triggers_escalation(
        self,
        slow_agent: MockAgent,
    ):
        """Test that timeout triggers escalation."""
        config = HierarchicalConfig(timeout_per_subtask=0.1)
        pattern = HierarchicalPattern(config=config)

        task = Task(description="Slow task")
        context = PatternContext()

        # Execute single subtask
        from agents_framework.agents.base import Task as AgentTask

        agent_task = AgentTask(description="Test")
        subtask = SubTask(id="sub_1", description="Test")

        result = await pattern._execute_single_subtask(
            slow_agent, agent_task, subtask, context, {}
        )

        assert not result.success
        assert "timed out" in result.error

        # Should have created escalation
        assert len(pattern._escalations) > 0
        assert pattern._escalations[0].level == EscalationLevel.WARNING


# ============================================================================
# Escalation Tests
# ============================================================================


class TestEscalation:
    """Tests for escalation handling."""

    def test_manual_escalate(self):
        """Test manually triggering escalation."""
        pattern = HierarchicalPattern()

        pattern.escalate(
            from_agent_id="worker_1",
            level=EscalationLevel.ERROR,
            reason="Test escalation",
            context={"detail": "test"},
        )

        assert len(pattern._escalations) == 1
        assert pattern._escalations[0].from_agent_id == "worker_1"
        assert pattern._escalations[0].level == EscalationLevel.ERROR
        assert pattern._escalations[0].context == {"detail": "test"}

    @pytest.mark.asyncio
    async def test_failure_threshold_escalation(
        self,
        failing_agent: MockAgent,
    ):
        """Test that repeated failures trigger escalation."""
        config = HierarchicalConfig(escalation_threshold=2)
        pattern = HierarchicalPattern(config=config)
        context = PatternContext()
        failure_counts: Dict[str, int] = {}

        task = Task(description="Failing task")
        subtask = SubTask(id="sub_1", description="Test")

        # Execute twice to trigger escalation
        await pattern._execute_single_subtask(
            failing_agent, task, subtask, context, failure_counts
        )
        await pattern._execute_single_subtask(
            failing_agent, task, subtask, context, failure_counts
        )

        # Should have escalation after reaching threshold
        assert len(pattern._escalations) >= 1
        escalation = pattern._escalations[-1]
        assert escalation.level == EscalationLevel.ERROR

    @pytest.mark.asyncio
    async def test_handle_escalations(self, mock_supervisor: MockAgent):
        """Test handling escalations through supervisor."""
        pattern = HierarchicalPattern()
        context = PatternContext()

        subtask = SubTask(id="sub_1", description="Failed task")
        pattern._escalations = [
            EscalationRequest(
                from_agent_id="worker_1",
                level=EscalationLevel.ERROR,
                reason="Worker failed",
                original_task=subtask,
            )
        ]

        results = await pattern._handle_escalations(
            mock_supervisor, pattern._escalations, context
        )

        assert "sub_1" in results
        assert mock_supervisor.run_count >= 1


# ============================================================================
# Result Synthesis Tests
# ============================================================================


class TestResultSynthesis:
    """Tests for result synthesis."""

    @pytest.mark.asyncio
    async def test_aggregate_synthesis(self, mock_supervisor: MockAgent):
        """Test aggregate synthesis strategy."""
        pattern = HierarchicalPattern(
            config=HierarchicalConfig(synthesis_strategy="aggregate")
        )
        context = PatternContext()

        results = {
            "sub_1": TaskResult(task_id="sub_1", success=True, output="Output 1"),
            "sub_2": TaskResult(task_id="sub_2", success=True, output="Output 2"),
            "sub_3": TaskResult(task_id="sub_3", success=False, error="Failed"),
        }

        output = await pattern._synthesize_results(mock_supervisor, results, context)

        assert "results" in output
        assert output["results"] == {"sub_1": "Output 1", "sub_2": "Output 2"}
        assert len(output["failed"]) == 1

    @pytest.mark.asyncio
    async def test_custom_synthesis_function(self, mock_supervisor: MockAgent):
        """Test custom synthesis function."""
        async def custom_synthesize(results, context):
            outputs = [r.output for r in results.values() if r.success]
            return {"combined": " | ".join(str(o) for o in outputs)}

        pattern = HierarchicalPattern(synthesize_fn=custom_synthesize)
        context = PatternContext()

        results = {
            "sub_1": TaskResult(task_id="sub_1", success=True, output="A"),
            "sub_2": TaskResult(task_id="sub_2", success=True, output="B"),
        }

        output = await pattern._synthesize_results(mock_supervisor, results, context)

        assert output == {"combined": "A | B"}

    @pytest.mark.asyncio
    async def test_reduce_synthesis(self, mock_supervisor: MockAgent):
        """Test reduce synthesis strategy."""
        pattern = HierarchicalPattern(
            config=HierarchicalConfig(synthesis_strategy="reduce")
        )
        context = PatternContext()

        results = {
            "sub_1": TaskResult(task_id="sub_1", success=True, output="Output 1"),
        }

        # Reduce calls supervisor to synthesize
        output = await pattern._synthesize_results(mock_supervisor, results, context)

        # Supervisor should be called for reduction
        assert mock_supervisor.run_count >= 1


# ============================================================================
# Full Workflow Tests
# ============================================================================


class TestFullWorkflow:
    """Tests for complete hierarchical workflow."""

    @pytest.mark.asyncio
    async def test_complete_workflow(
        self,
        mock_supervisor: MockAgent,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test complete hierarchical workflow."""
        # Supervisor returns subtask list
        mock_supervisor.return_value = [
            {"id": "research", "description": "Research topic"},
            {"id": "analyze", "description": "Analyze findings", "dependencies": ["research"]},
        ]

        pattern = HierarchicalPattern()
        task = Task(description="Complete research project")

        result = await pattern.execute(
            task, [mock_supervisor, mock_researcher, mock_analyst]
        )

        assert result.status == PatternStatus.COMPLETED
        assert result.total_duration_ms > 0
        assert result.context is not None

    @pytest.mark.asyncio
    async def test_workflow_with_failures(
        self,
        mock_supervisor: MockAgent,
        failing_agent: MockAgent,
    ):
        """Test workflow handles failures gracefully."""
        pattern = HierarchicalPattern()
        task = Task(description="Task with failing worker")

        result = await pattern.execute(task, [mock_supervisor, failing_agent])

        # Pattern should complete (possibly with partial failures)
        assert result is not None
        assert result.pattern_name == "hierarchical"

    @pytest.mark.asyncio
    async def test_workflow_exception_handling(self, mock_supervisor: MockAgent):
        """Test that exceptions are caught and returned as failed result."""
        async def failing_decompose(task, context):
            raise RuntimeError("Decomposition failed")

        pattern = HierarchicalPattern(decompose_fn=failing_decompose)
        task = Task(description="Test task")

        result = await pattern.execute(task, [mock_supervisor])

        assert result.status == PatternStatus.FAILED
        assert "Decomposition failed" in result.error
