"""Sequential (pipeline) team pattern implementation.

This module implements a sequential pattern where agents are organized
in a pipeline. Each stage processes the output of the previous stage,
with support for conditional branching and state tracking.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

from .base import (
    BasePattern,
    PatternContext,
    PatternResult,
    PatternStatus,
    StepResult,
)

if TYPE_CHECKING:
    from agents_framework.agents.base import BaseAgent, Task, TaskResult


class StageStatus(str, Enum):
    """Status of a pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class PipelineStage:
    """A stage in the sequential pipeline.

    Attributes:
        id: Unique identifier for the stage.
        name: Human-readable name of the stage.
        agent_id: ID of the agent assigned to this stage.
        condition: Optional condition function to determine if stage runs.
        transform: Optional transform function for stage output.
        timeout: Timeout for this stage in seconds.
        retry_count: Number of retries on failure.
        metadata: Additional metadata for the stage.
    """

    id: str
    name: str
    agent_id: Optional[str] = None
    condition: Optional[Callable[[PatternContext], bool]] = None
    transform: Optional[Callable[[Any, PatternContext], Any]] = None
    timeout: float = 60.0
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineState:
    """State of the pipeline execution.

    Attributes:
        current_stage_index: Index of the currently executing stage.
        current_stage_id: ID of the current stage.
        completed_stages: List of completed stage IDs.
        skipped_stages: List of skipped stage IDs.
        failed_stages: List of failed stage IDs.
        stage_outputs: Outputs from each completed stage.
        is_complete: Whether the pipeline has finished.
    """

    current_stage_index: int = 0
    current_stage_id: Optional[str] = None
    completed_stages: List[str] = field(default_factory=list)
    skipped_stages: List[str] = field(default_factory=list)
    failed_stages: List[str] = field(default_factory=list)
    stage_outputs: Dict[str, Any] = field(default_factory=dict)
    is_complete: bool = False


@dataclass
class BranchConfig:
    """Configuration for conditional branching.

    Attributes:
        condition: Function that evaluates to determine branch.
        target_stage_id: Stage to jump to if condition is true.
        description: Description of the branch condition.
    """

    condition: Callable[[Any, PatternContext], bool]
    target_stage_id: str
    description: str = ""


@dataclass
class SequentialConfig:
    """Configuration for sequential pattern.

    Attributes:
        stop_on_failure: Whether to stop pipeline on any stage failure.
        pass_output_as_input: Whether to pass stage output as next stage input.
        enable_branching: Whether conditional branching is enabled.
        default_timeout: Default timeout for stages without explicit timeout.
        max_retries: Maximum retries per stage.
    """

    stop_on_failure: bool = True
    pass_output_as_input: bool = True
    enable_branching: bool = True
    default_timeout: float = 60.0
    max_retries: int = 1


class SequentialPattern(BasePattern):
    """Sequential pipeline pattern for agent coordination.

    This pattern organizes agents in a linear pipeline where:
    1. Each stage processes data and passes output to the next
    2. Stages can have conditions that determine if they execute
    3. Branching allows jumping to different stages based on conditions
    4. State is tracked throughout the pipeline execution

    Ideal for multi-step processing workflows like:
    - Data transformation pipelines
    - Multi-stage analysis workflows
    - Sequential approval processes
    """

    def __init__(
        self,
        config: Optional[SequentialConfig] = None,
        stages: Optional[List[PipelineStage]] = None,
        branches: Optional[Dict[str, List[BranchConfig]]] = None,
    ):
        """Initialize the sequential pattern.

        Args:
            config: Configuration for the pattern.
            stages: Predefined list of stages.
            branches: Dictionary mapping stage IDs to branch configurations.
        """
        super().__init__(name="sequential")
        self.config = config or SequentialConfig()
        self._stages: List[PipelineStage] = stages or []
        self._branches: Dict[str, List[BranchConfig]] = branches or {}
        self._state = PipelineState()

    @property
    def state(self) -> PipelineState:
        """Get the current pipeline state."""
        return self._state

    def add_stage(
        self,
        stage: PipelineStage,
        position: Optional[int] = None,
    ) -> None:
        """Add a stage to the pipeline.

        Args:
            stage: The stage to add.
            position: Optional position in the pipeline.
        """
        if position is not None:
            self._stages.insert(position, stage)
        else:
            self._stages.append(stage)

    def add_branch(
        self,
        from_stage_id: str,
        branch: BranchConfig,
    ) -> None:
        """Add a conditional branch from a stage.

        Args:
            from_stage_id: ID of the stage to branch from.
            branch: Branch configuration.
        """
        if from_stage_id not in self._branches:
            self._branches[from_stage_id] = []
        self._branches[from_stage_id].append(branch)

    async def execute(
        self,
        task: Task,
        agents: List[BaseAgent],
        context: Optional[PatternContext] = None,
    ) -> PatternResult:
        """Execute the sequential pipeline.

        If stages are not predefined, creates one stage per agent.

        Args:
            task: The initial task/input for the pipeline.
            agents: List of agents for the pipeline stages.
            context: Optional initial context.

        Returns:
            PatternResult with the pipeline output.
        """
        if not await self.validate_agents(agents):
            return PatternResult(
                pattern_name=self.name,
                status=PatternStatus.FAILED,
                error="No agents provided for the pipeline",
            )

        ctx = self._create_context(context)
        self._status = PatternStatus.RUNNING
        self._state = PipelineState()
        start_time = time.time()
        steps: List[StepResult] = []

        # Build stages from agents if not predefined
        if not self._stages:
            self._stages = [
                PipelineStage(
                    id=f"stage_{i}",
                    name=agent.role.name,
                    agent_id=agent.id,
                    timeout=self.config.default_timeout,
                )
                for i, agent in enumerate(agents)
            ]

        agent_map = {a.id: a for a in agents}

        try:
            # Set initial input from task
            current_input = {
                "description": task.description,
                "context": task.context,
            }
            ctx.set("initial_input", current_input)

            stage_index = 0

            while stage_index < len(self._stages):
                stage = self._stages[stage_index]
                self._state.current_stage_index = stage_index
                self._state.current_stage_id = stage.id

                # Check stage condition
                if stage.condition and not stage.condition(ctx):
                    self._state.skipped_stages.append(stage.id)
                    stage_index += 1
                    continue

                # Get the agent for this stage
                agent = agent_map.get(stage.agent_id) if stage.agent_id else None
                if not agent and stage.agent_id:
                    # Try to match by index if ID not found
                    if stage_index < len(agents):
                        agent = agents[stage_index]

                if not agent:
                    if self.config.stop_on_failure:
                        return PatternResult(
                            pattern_name=self.name,
                            status=PatternStatus.FAILED,
                            error=f"No agent found for stage {stage.id}",
                            steps=steps,
                            context=ctx,
                        )
                    self._state.skipped_stages.append(stage.id)
                    stage_index += 1
                    continue

                # Execute the stage
                step_start = time.time()
                result = await self._execute_stage(
                    stage, agent, current_input, ctx
                )
                step_duration = (time.time() - step_start) * 1000

                steps.append(
                    self._create_step_result(
                        agent=agent,
                        output=result.output if result.success else None,
                        success=result.success,
                        error=result.error,
                        duration_ms=step_duration,
                    )
                )

                if result.success:
                    # Apply transform if defined
                    output = result.output
                    if stage.transform:
                        output = stage.transform(output, ctx)

                    self._state.stage_outputs[stage.id] = output
                    self._state.completed_stages.append(stage.id)

                    # Update input for next stage
                    if self.config.pass_output_as_input:
                        current_input = output

                    ctx.set(f"stage_{stage.id}_output", output)
                    ctx.history.append(steps[-1])

                    # Check for branches
                    next_stage_index = await self._check_branches(
                        stage.id, output, ctx
                    )

                    if next_stage_index is not None:
                        stage_index = next_stage_index
                    else:
                        stage_index += 1
                else:
                    self._state.failed_stages.append(stage.id)
                    if self.config.stop_on_failure:
                        self._state.is_complete = True
                        return PatternResult(
                            pattern_name=self.name,
                            status=PatternStatus.FAILED,
                            error=f"Stage {stage.id} failed: {result.error}",
                            steps=steps,
                            context=ctx,
                            total_duration_ms=(time.time() - start_time) * 1000,
                        )
                    stage_index += 1

            self._state.is_complete = True
            self._status = PatternStatus.COMPLETED

            # Final output is the last stage output or aggregated outputs
            final_output = (
                self._state.stage_outputs.get(self._state.completed_stages[-1])
                if self._state.completed_stages
                else None
            )

            return PatternResult(
                pattern_name=self.name,
                status=PatternStatus.COMPLETED,
                final_output=final_output,
                steps=steps,
                context=ctx,
                total_duration_ms=(time.time() - start_time) * 1000,
                metadata={
                    "completed_stages": self._state.completed_stages,
                    "skipped_stages": self._state.skipped_stages,
                    "failed_stages": self._state.failed_stages,
                    "all_outputs": self._state.stage_outputs,
                },
            )

        except Exception as e:
            self._status = PatternStatus.FAILED
            self._state.is_complete = True
            return PatternResult(
                pattern_name=self.name,
                status=PatternStatus.FAILED,
                error=str(e),
                steps=steps,
                context=ctx,
                total_duration_ms=(time.time() - start_time) * 1000,
            )

    async def _execute_stage(
        self,
        stage: PipelineStage,
        agent: BaseAgent,
        input_data: Any,
        context: PatternContext,
    ) -> TaskResult:
        """Execute a single pipeline stage with retries.

        Args:
            stage: The stage configuration.
            agent: The agent to execute.
            input_data: Input data for the stage.
            context: Execution context.

        Returns:
            TaskResult from the stage execution.
        """
        from agents_framework.agents.base import Task as AgentTask, TaskResult

        task = AgentTask(
            description=f"Stage '{stage.name}': Process the input data",
            context={
                "input": input_data,
                "stage_id": stage.id,
                "stage_name": stage.name,
                **context.variables,
            },
        )

        retries = stage.retry_count or self.config.max_retries
        last_error: Optional[str] = None

        for attempt in range(retries + 1):
            try:
                result = await asyncio.wait_for(
                    agent.run(task),
                    timeout=stage.timeout,
                )
                if result.success:
                    return result
                last_error = result.error
            except asyncio.TimeoutError:
                last_error = f"Stage {stage.id} timed out after {stage.timeout}s"
            except Exception as e:
                last_error = str(e)

        return TaskResult(
            task_id=task.id,
            success=False,
            error=last_error,
        )

    async def _check_branches(
        self,
        stage_id: str,
        output: Any,
        context: PatternContext,
    ) -> Optional[int]:
        """Check if any branches should be taken after a stage.

        Args:
            stage_id: ID of the completed stage.
            output: Output from the completed stage.
            context: Execution context.

        Returns:
            Index of the target stage if branching, None otherwise.
        """
        if not self.config.enable_branching:
            return None

        branches = self._branches.get(stage_id, [])

        for branch in branches:
            if branch.condition(output, context):
                # Find the target stage index
                for i, stage in enumerate(self._stages):
                    if stage.id == branch.target_stage_id:
                        return i

        return None

    def get_stage(self, stage_id: str) -> Optional[PipelineStage]:
        """Get a stage by its ID."""
        for stage in self._stages:
            if stage.id == stage_id:
                return stage
        return None

    def get_stage_output(self, stage_id: str) -> Optional[Any]:
        """Get the output from a specific stage."""
        return self._state.stage_outputs.get(stage_id)

    def reset_state(self) -> None:
        """Reset the pipeline state for re-execution."""
        self._state = PipelineState()
        self._status = PatternStatus.PENDING
