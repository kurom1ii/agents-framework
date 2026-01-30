"""Unit tests for the sequential team pattern module."""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional

import pytest

from agents_framework.agents import AgentRole, Task, TaskResult
from agents_framework.teams.patterns.base import PatternContext, PatternStatus, StepResult
from agents_framework.teams.patterns.sequential import (
    BranchConfig,
    PipelineStage,
    PipelineState,
    SequentialConfig,
    SequentialPattern,
    StageStatus,
)

from ..conftest import MockAgent


# ============================================================================
# StageStatus Tests
# ============================================================================


class TestStageStatus:
    """Tests for StageStatus enum."""

    def test_all_statuses_defined(self):
        """Test that all stage statuses are defined."""
        assert StageStatus.PENDING.value == "pending"
        assert StageStatus.RUNNING.value == "running"
        assert StageStatus.COMPLETED.value == "completed"
        assert StageStatus.SKIPPED.value == "skipped"
        assert StageStatus.FAILED.value == "failed"


# ============================================================================
# PipelineStage Tests
# ============================================================================


class TestPipelineStage:
    """Tests for the PipelineStage dataclass."""

    def test_stage_creation(self):
        """Test creating a pipeline stage."""
        stage = PipelineStage(
            id="stage_1",
            name="Research",
        )

        assert stage.id == "stage_1"
        assert stage.name == "Research"
        assert stage.agent_id is None
        assert stage.condition is None
        assert stage.transform is None
        assert stage.timeout == 60.0
        assert stage.retry_count == 0
        assert stage.metadata == {}

    def test_stage_with_agent_assignment(self):
        """Test stage with agent assignment."""
        stage = PipelineStage(
            id="stage_1",
            name="Research",
            agent_id="agent_123",
        )

        assert stage.agent_id == "agent_123"

    def test_stage_with_condition(self):
        """Test stage with condition function."""
        def should_run(ctx: PatternContext) -> bool:
            return ctx.get("run_stage", False)

        stage = PipelineStage(
            id="stage_1",
            name="Conditional",
            condition=should_run,
        )

        context = PatternContext(variables={"run_stage": True})
        assert stage.condition(context)

    def test_stage_with_transform(self):
        """Test stage with transform function."""
        def transform_output(output: Any, ctx: PatternContext) -> Any:
            return {"transformed": output}

        stage = PipelineStage(
            id="stage_1",
            name="Transform",
            transform=transform_output,
        )

        context = PatternContext()
        result = stage.transform("original", context)
        assert result == {"transformed": "original"}


# ============================================================================
# PipelineState Tests
# ============================================================================


class TestPipelineState:
    """Tests for the PipelineState dataclass."""

    def test_state_creation(self):
        """Test creating pipeline state."""
        state = PipelineState()

        assert state.current_stage_index == 0
        assert state.current_stage_id is None
        assert state.completed_stages == []
        assert state.skipped_stages == []
        assert state.failed_stages == []
        assert state.stage_outputs == {}
        assert state.is_complete is False

    def test_state_tracking(self):
        """Test state tracking during execution."""
        state = PipelineState()

        state.current_stage_index = 1
        state.current_stage_id = "stage_2"
        state.completed_stages.append("stage_1")
        state.stage_outputs["stage_1"] = "output_1"

        assert state.current_stage_index == 1
        assert len(state.completed_stages) == 1
        assert state.stage_outputs["stage_1"] == "output_1"


# ============================================================================
# BranchConfig Tests
# ============================================================================


class TestBranchConfig:
    """Tests for the BranchConfig dataclass."""

    def test_branch_config_creation(self):
        """Test creating branch configuration."""
        def condition(output: Any, ctx: PatternContext) -> bool:
            return output.get("need_review", False)

        config = BranchConfig(
            condition=condition,
            target_stage_id="review_stage",
            description="Branch to review if needed",
        )

        assert config.target_stage_id == "review_stage"
        assert config.description == "Branch to review if needed"

    def test_branch_condition_evaluation(self):
        """Test branch condition evaluation."""
        def condition(output: Any, ctx: PatternContext) -> bool:
            return output == "needs_review"

        config = BranchConfig(
            condition=condition,
            target_stage_id="review",
        )

        context = PatternContext()
        assert config.condition("needs_review", context)
        assert not config.condition("ok", context)


# ============================================================================
# SequentialConfig Tests
# ============================================================================


class TestSequentialConfig:
    """Tests for the SequentialConfig dataclass."""

    def test_default_config(self):
        """Test default sequential configuration."""
        config = SequentialConfig()

        assert config.stop_on_failure is True
        assert config.pass_output_as_input is True
        assert config.enable_branching is True
        assert config.default_timeout == 60.0
        assert config.max_retries == 1

    def test_custom_config(self):
        """Test custom sequential configuration."""
        config = SequentialConfig(
            stop_on_failure=False,
            pass_output_as_input=False,
            enable_branching=False,
            default_timeout=30.0,
            max_retries=3,
        )

        assert config.stop_on_failure is False
        assert config.pass_output_as_input is False
        assert config.enable_branching is False
        assert config.default_timeout == 30.0
        assert config.max_retries == 3


# ============================================================================
# SequentialPattern Creation Tests
# ============================================================================


class TestSequentialPatternCreation:
    """Tests for SequentialPattern creation."""

    def test_pattern_creation(self):
        """Test creating a sequential pattern."""
        pattern = SequentialPattern()

        assert pattern.name == "sequential"
        assert pattern.status == PatternStatus.PENDING
        assert pattern._stages == []
        assert pattern._branches == {}

    def test_pattern_with_config(self):
        """Test creating pattern with custom config."""
        config = SequentialConfig(stop_on_failure=False)
        pattern = SequentialPattern(config=config)

        assert pattern.config.stop_on_failure is False

    def test_pattern_with_stages(self):
        """Test creating pattern with predefined stages."""
        stages = [
            PipelineStage(id="stage_1", name="Research"),
            PipelineStage(id="stage_2", name="Analyze"),
        ]

        pattern = SequentialPattern(stages=stages)

        assert len(pattern._stages) == 2

    def test_pattern_with_branches(self):
        """Test creating pattern with branch configurations."""
        branches = {
            "stage_1": [
                BranchConfig(
                    condition=lambda o, c: o == "skip",
                    target_stage_id="stage_3",
                )
            ]
        }

        pattern = SequentialPattern(branches=branches)

        assert "stage_1" in pattern._branches


# ============================================================================
# Stage Management Tests
# ============================================================================


class TestStageManagement:
    """Tests for stage management methods."""

    def test_add_stage(self):
        """Test adding a stage."""
        pattern = SequentialPattern()
        stage = PipelineStage(id="stage_1", name="Test")

        pattern.add_stage(stage)

        assert len(pattern._stages) == 1
        assert pattern._stages[0] is stage

    def test_add_stage_at_position(self):
        """Test adding a stage at specific position."""
        pattern = SequentialPattern()
        stage1 = PipelineStage(id="stage_1", name="First")
        stage2 = PipelineStage(id="stage_2", name="Second")
        stage3 = PipelineStage(id="stage_3", name="Middle")

        pattern.add_stage(stage1)
        pattern.add_stage(stage2)
        pattern.add_stage(stage3, position=1)

        assert pattern._stages[0].id == "stage_1"
        assert pattern._stages[1].id == "stage_3"
        assert pattern._stages[2].id == "stage_2"

    def test_add_branch(self):
        """Test adding a branch."""
        pattern = SequentialPattern()

        branch = BranchConfig(
            condition=lambda o, c: True,
            target_stage_id="target",
        )

        pattern.add_branch("source_stage", branch)

        assert "source_stage" in pattern._branches
        assert len(pattern._branches["source_stage"]) == 1

    def test_add_multiple_branches_from_same_stage(self):
        """Test adding multiple branches from the same stage."""
        pattern = SequentialPattern()

        branch1 = BranchConfig(condition=lambda o, c: o == "a", target_stage_id="a")
        branch2 = BranchConfig(condition=lambda o, c: o == "b", target_stage_id="b")

        pattern.add_branch("source", branch1)
        pattern.add_branch("source", branch2)

        assert len(pattern._branches["source"]) == 2

    def test_get_stage(self):
        """Test getting a stage by ID."""
        pattern = SequentialPattern()
        stage = PipelineStage(id="stage_1", name="Test")
        pattern.add_stage(stage)

        found = pattern.get_stage("stage_1")
        assert found is stage

    def test_get_stage_nonexistent(self):
        """Test getting non-existent stage."""
        pattern = SequentialPattern()

        found = pattern.get_stage("nonexistent")
        assert found is None

    def test_get_stage_output(self):
        """Test getting stage output."""
        pattern = SequentialPattern()
        pattern._state.stage_outputs["stage_1"] = "output"

        output = pattern.get_stage_output("stage_1")
        assert output == "output"

    def test_reset_state(self):
        """Test resetting pipeline state."""
        pattern = SequentialPattern()
        pattern._state.current_stage_index = 5
        pattern._state.completed_stages = ["stage_1", "stage_2"]
        pattern._status = PatternStatus.COMPLETED

        pattern.reset_state()

        assert pattern._state.current_stage_index == 0
        assert pattern._state.completed_stages == []
        assert pattern.status == PatternStatus.PENDING


# ============================================================================
# Validation Tests
# ============================================================================


class TestValidation:
    """Tests for agent validation."""

    @pytest.mark.asyncio
    async def test_validate_with_agents(self, mock_researcher: MockAgent):
        """Test validation with agents."""
        pattern = SequentialPattern()

        assert await pattern.validate_agents([mock_researcher])

    @pytest.mark.asyncio
    async def test_validate_empty_list(self):
        """Test validation with empty agent list."""
        pattern = SequentialPattern()

        assert not await pattern.validate_agents([])


# ============================================================================
# Execution Tests
# ============================================================================


class TestSequentialExecution:
    """Tests for sequential pattern execution."""

    @pytest.mark.asyncio
    async def test_basic_execution(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test basic sequential execution."""
        pattern = SequentialPattern()
        task = Task(description="Process data")

        result = await pattern.execute(task, [mock_researcher, mock_analyst])

        assert result.status == PatternStatus.COMPLETED
        assert len(result.steps) == 2
        assert mock_researcher.run_count == 1
        assert mock_analyst.run_count == 1

    @pytest.mark.asyncio
    async def test_execution_creates_stages_from_agents(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test that execution creates stages from agents if not defined."""
        pattern = SequentialPattern()
        task = Task(description="Process data")

        await pattern.execute(task, [mock_researcher, mock_analyst])

        assert len(pattern._stages) == 2
        assert pattern._stages[0].name == mock_researcher.role.name

    @pytest.mark.asyncio
    async def test_execution_fails_with_no_agents(self):
        """Test execution fails with no agents."""
        pattern = SequentialPattern()
        task = Task(description="Test task")

        result = await pattern.execute(task, [])

        assert result.status == PatternStatus.FAILED
        assert "No agents provided" in result.error

    @pytest.mark.asyncio
    async def test_execution_preserves_order(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
        mock_writer: MockAgent,
    ):
        """Test that stages execute in order."""
        execution_order = []

        class OrderTrackingAgent(MockAgent):
            async def run(self, task):
                execution_order.append(self.role.name)
                return await super().run(task)

        agents = [
            OrderTrackingAgent(role=AgentRole(name="first", description="", capabilities=[])),
            OrderTrackingAgent(role=AgentRole(name="second", description="", capabilities=[])),
            OrderTrackingAgent(role=AgentRole(name="third", description="", capabilities=[])),
        ]

        pattern = SequentialPattern()
        await pattern.execute(Task(description="Test"), agents)

        assert execution_order == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_execution_with_predefined_stages(self, mock_researcher: MockAgent):
        """Test execution with predefined stages."""
        stages = [
            PipelineStage(id="stage_1", name="Custom Stage", agent_id=mock_researcher.id),
        ]

        pattern = SequentialPattern(stages=stages)
        task = Task(description="Test")

        result = await pattern.execute(task, [mock_researcher])

        assert result.status == PatternStatus.COMPLETED


# ============================================================================
# Output Passing Tests
# ============================================================================


class TestOutputPassing:
    """Tests for output passing between stages."""

    @pytest.mark.asyncio
    async def test_output_passed_as_input(self):
        """Test that output is passed as input to next stage."""
        received_inputs = []

        class InputCapturingAgent(MockAgent):
            async def run(self, task):
                if isinstance(task, Task):
                    received_inputs.append(task.context.get("input"))
                return TaskResult(task_id="1", success=True, output=f"output_{self.role.name}")

        agents = [
            InputCapturingAgent(role=AgentRole(name="stage_1", description="", capabilities=[])),
            InputCapturingAgent(role=AgentRole(name="stage_2", description="", capabilities=[])),
        ]

        pattern = SequentialPattern()
        await pattern.execute(Task(description="Test"), agents)

        # Second stage should receive first stage's output
        assert received_inputs[1] == "output_stage_1"

    @pytest.mark.asyncio
    async def test_output_not_passed_when_disabled(self):
        """Test that output is not passed when disabled."""
        config = SequentialConfig(pass_output_as_input=False)

        received_inputs = []

        class InputCapturingAgent(MockAgent):
            async def run(self, task):
                if isinstance(task, Task):
                    received_inputs.append(task.context.get("input"))
                return TaskResult(task_id="1", success=True, output="output")

        agents = [
            InputCapturingAgent(role=AgentRole(name="stage_1", description="", capabilities=[])),
            InputCapturingAgent(role=AgentRole(name="stage_2", description="", capabilities=[])),
        ]

        pattern = SequentialPattern(config=config)
        await pattern.execute(Task(description="Test"), agents)

        # Both stages receive initial input
        assert received_inputs[0] == received_inputs[1]


# ============================================================================
# Condition Tests
# ============================================================================


class TestConditionalExecution:
    """Tests for conditional stage execution."""

    @pytest.mark.asyncio
    async def test_stage_skipped_when_condition_false(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test that stage is skipped when condition is false."""
        stages = [
            PipelineStage(id="stage_1", name="Stage 1", agent_id=mock_researcher.id),
            PipelineStage(
                id="stage_2",
                name="Stage 2",
                agent_id=mock_analyst.id,
                condition=lambda ctx: False,  # Always skip
            ),
        ]

        pattern = SequentialPattern(stages=stages)
        result = await pattern.execute(Task(description="Test"), [mock_researcher, mock_analyst])

        assert result.status == PatternStatus.COMPLETED
        assert "stage_2" in pattern.state.skipped_stages
        assert mock_analyst.run_count == 0

    @pytest.mark.asyncio
    async def test_stage_executes_when_condition_true(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test that stage executes when condition is true."""
        stages = [
            PipelineStage(id="stage_1", name="Stage 1", agent_id=mock_researcher.id),
            PipelineStage(
                id="stage_2",
                name="Stage 2",
                agent_id=mock_analyst.id,
                condition=lambda ctx: True,  # Always execute
            ),
        ]

        pattern = SequentialPattern(stages=stages)
        result = await pattern.execute(Task(description="Test"), [mock_researcher, mock_analyst])

        assert result.status == PatternStatus.COMPLETED
        assert "stage_2" in pattern.state.completed_stages
        assert mock_analyst.run_count == 1

    @pytest.mark.asyncio
    async def test_condition_uses_context(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test that condition can use context variables."""
        stages = [
            PipelineStage(id="stage_1", name="Stage 1", agent_id=mock_researcher.id),
            PipelineStage(
                id="stage_2",
                name="Stage 2",
                agent_id=mock_analyst.id,
                condition=lambda ctx: ctx.get("run_second", False),
            ),
        ]

        pattern = SequentialPattern(stages=stages)
        context = PatternContext(variables={"run_second": True})

        result = await pattern.execute(Task(description="Test"), [mock_researcher, mock_analyst], context)

        assert "stage_2" in pattern.state.completed_stages


# ============================================================================
# Transform Tests
# ============================================================================


class TestTransformExecution:
    """Tests for output transformation."""

    @pytest.mark.asyncio
    async def test_transform_applied_to_output(
        self,
        mock_researcher: MockAgent,
    ):
        """Test that transform is applied to stage output."""
        def transform(output: Any, ctx: PatternContext) -> Any:
            return {"transformed": output}

        stages = [
            PipelineStage(
                id="stage_1",
                name="Stage 1",
                agent_id=mock_researcher.id,
                transform=transform,
            ),
        ]

        pattern = SequentialPattern(stages=stages)
        result = await pattern.execute(Task(description="Test"), [mock_researcher])

        assert pattern.state.stage_outputs["stage_1"]["transformed"] == mock_researcher.return_value


# ============================================================================
# Branching Tests
# ============================================================================


class TestBranching:
    """Tests for conditional branching."""

    @pytest.mark.asyncio
    async def test_branch_to_stage(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
        mock_writer: MockAgent,
    ):
        """Test branching to a different stage."""
        # Setup stages
        stages = [
            PipelineStage(id="stage_1", name="Stage 1", agent_id=mock_researcher.id),
            PipelineStage(id="stage_2", name="Stage 2", agent_id=mock_analyst.id),
            PipelineStage(id="stage_3", name="Stage 3", agent_id=mock_writer.id),
        ]

        # Branch from stage_1 directly to stage_3
        branches = {
            "stage_1": [
                BranchConfig(
                    condition=lambda o, c: True,  # Always branch
                    target_stage_id="stage_3",
                )
            ]
        }

        pattern = SequentialPattern(stages=stages, branches=branches)
        result = await pattern.execute(
            Task(description="Test"), [mock_researcher, mock_analyst, mock_writer]
        )

        # Stage 2 should be skipped due to branching
        assert "stage_1" in pattern.state.completed_stages
        assert "stage_3" in pattern.state.completed_stages
        # Stage 2 may or may not be in completed depending on order
        # The branch jumps directly to stage_3

    @pytest.mark.asyncio
    async def test_no_branch_when_condition_false(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test no branching when condition is false."""
        stages = [
            PipelineStage(id="stage_1", name="Stage 1", agent_id=mock_researcher.id),
            PipelineStage(id="stage_2", name="Stage 2", agent_id=mock_analyst.id),
        ]

        branches = {
            "stage_1": [
                BranchConfig(
                    condition=lambda o, c: False,  # Never branch
                    target_stage_id="nonexistent",
                )
            ]
        }

        pattern = SequentialPattern(stages=stages, branches=branches)
        result = await pattern.execute(Task(description="Test"), [mock_researcher, mock_analyst])

        # Both stages should execute in order
        assert pattern.state.completed_stages == ["stage_1", "stage_2"]

    @pytest.mark.asyncio
    async def test_branching_disabled(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test that branching can be disabled."""
        config = SequentialConfig(enable_branching=False)

        stages = [
            PipelineStage(id="stage_1", name="Stage 1", agent_id=mock_researcher.id),
            PipelineStage(id="stage_2", name="Stage 2", agent_id=mock_analyst.id),
        ]

        branches = {
            "stage_1": [
                BranchConfig(
                    condition=lambda o, c: True,  # Would branch if enabled
                    target_stage_id="stage_2",
                )
            ]
        }

        pattern = SequentialPattern(config=config, stages=stages, branches=branches)
        await pattern.execute(Task(description="Test"), [mock_researcher, mock_analyst])

        # Should proceed normally without branching
        assert pattern.state.completed_stages == ["stage_1", "stage_2"]


# ============================================================================
# Failure Handling Tests
# ============================================================================


class TestFailureHandling:
    """Tests for failure handling."""

    @pytest.mark.asyncio
    async def test_stop_on_failure(
        self,
        failing_agent: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test stopping pipeline on failure."""
        config = SequentialConfig(stop_on_failure=True)
        pattern = SequentialPattern(config=config)

        result = await pattern.execute(Task(description="Test"), [failing_agent, mock_analyst])

        assert result.status == PatternStatus.FAILED
        assert mock_analyst.run_count == 0

    @pytest.mark.asyncio
    async def test_continue_on_failure(
        self,
        failing_agent: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test continuing pipeline on failure when configured."""
        config = SequentialConfig(stop_on_failure=False)
        pattern = SequentialPattern(config=config)

        result = await pattern.execute(Task(description="Test"), [failing_agent, mock_analyst])

        assert result.status == PatternStatus.COMPLETED
        assert mock_analyst.run_count == 1
        assert len(pattern.state.failed_stages) == 1

    @pytest.mark.asyncio
    async def test_timeout_handling(self, slow_agent: MockAgent):
        """Test timeout handling."""
        stages = [
            PipelineStage(
                id="stage_1",
                name="Slow Stage",
                agent_id=slow_agent.id,
                timeout=0.1,  # Very short timeout
            ),
        ]

        pattern = SequentialPattern(stages=stages)
        result = await pattern.execute(Task(description="Test"), [slow_agent])

        assert result.status == PatternStatus.FAILED

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, mock_researcher: MockAgent):
        """Test retry mechanism."""
        attempt_count = 0

        class FailOnceAgent(MockAgent):
            async def run(self, task):
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count == 1:
                    return TaskResult(task_id="1", success=False, error="First attempt failed")
                return TaskResult(task_id="1", success=True, output="Success on retry")

        agent = FailOnceAgent(role=AgentRole(name="retry", description="", capabilities=[]))

        stages = [
            PipelineStage(
                id="stage_1",
                name="Retry Stage",
                agent_id=agent.id,
                retry_count=2,
            ),
        ]

        pattern = SequentialPattern(stages=stages)
        result = await pattern.execute(Task(description="Test"), [agent])

        assert result.status == PatternStatus.COMPLETED
        assert attempt_count == 2


# ============================================================================
# Result and Metadata Tests
# ============================================================================


class TestResultsAndMetadata:
    """Tests for result and metadata handling."""

    @pytest.mark.asyncio
    async def test_result_contains_metadata(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test that result contains execution metadata."""
        pattern = SequentialPattern()

        result = await pattern.execute(Task(description="Test"), [mock_researcher, mock_analyst])

        assert result.metadata is not None
        assert "completed_stages" in result.metadata
        assert "skipped_stages" in result.metadata
        assert "failed_stages" in result.metadata
        assert "all_outputs" in result.metadata

    @pytest.mark.asyncio
    async def test_final_output_is_last_stage_output(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test that final output is from the last completed stage."""
        mock_researcher.return_value = "first_output"
        mock_analyst.return_value = "second_output"

        pattern = SequentialPattern()

        result = await pattern.execute(Task(description="Test"), [mock_researcher, mock_analyst])

        assert result.final_output == "second_output"

    @pytest.mark.asyncio
    async def test_state_is_complete_after_execution(self, mock_researcher: MockAgent):
        """Test that state is marked complete after execution."""
        pattern = SequentialPattern()

        await pattern.execute(Task(description="Test"), [mock_researcher])

        assert pattern.state.is_complete

    @pytest.mark.asyncio
    async def test_context_updated_with_stage_outputs(self, mock_researcher: MockAgent):
        """Test that context is updated with stage outputs."""
        pattern = SequentialPattern()

        result = await pattern.execute(Task(description="Test"), [mock_researcher])

        # Context should have stage output stored
        assert result.context.get("stage_stage_0_output") is not None

    @pytest.mark.asyncio
    async def test_state_property(self):
        """Test state property returns current state."""
        pattern = SequentialPattern()

        state = pattern.state
        assert isinstance(state, PipelineState)


# ============================================================================
# Exception Handling Tests
# ============================================================================


class TestExceptionHandling:
    """Tests for exception handling."""

    @pytest.mark.asyncio
    async def test_exception_caught(self):
        """Test that exceptions are caught and returned as failed result."""
        class ExceptionAgent(MockAgent):
            async def run(self, task):
                raise RuntimeError("Unexpected error")

        agent = ExceptionAgent(role=AgentRole(name="error", description="", capabilities=[]))
        pattern = SequentialPattern()

        result = await pattern.execute(Task(description="Test"), [agent])

        assert result.status == PatternStatus.FAILED
        assert "Unexpected error" in result.error

    @pytest.mark.asyncio
    async def test_state_marked_complete_on_exception(self):
        """Test that state is marked complete even on exception."""
        class ExceptionAgent(MockAgent):
            async def run(self, task):
                raise RuntimeError("Error")

        agent = ExceptionAgent(role=AgentRole(name="error", description="", capabilities=[]))
        pattern = SequentialPattern()

        await pattern.execute(Task(description="Test"), [agent])

        assert pattern.state.is_complete
