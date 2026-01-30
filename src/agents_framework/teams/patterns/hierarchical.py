"""Hierarchical team pattern implementation.

This module implements a hierarchical pattern where a supervisor agent
decomposes tasks, delegates to worker agents, synthesizes results,
and handles escalation when workers cannot complete their tasks.
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


class EscalationLevel(str, Enum):
    """Levels of escalation in the hierarchy."""

    NONE = "none"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SubTask:
    """A subtask decomposed from the main task.

    Attributes:
        id: Unique identifier for the subtask.
        description: Description of what needs to be done.
        assigned_agent_id: ID of the agent assigned to this subtask.
        dependencies: List of subtask IDs that must complete first.
        priority: Priority level (higher = more important).
        metadata: Additional metadata for the subtask.
    """

    id: str
    description: str
    assigned_agent_id: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EscalationRequest:
    """Request to escalate an issue to a higher level.

    Attributes:
        from_agent_id: ID of the agent requesting escalation.
        level: Severity level of the escalation.
        reason: Why the escalation is needed.
        context: Additional context for the escalation.
        original_task: The task that triggered the escalation.
    """

    from_agent_id: str
    level: EscalationLevel
    reason: str
    context: Dict[str, Any] = field(default_factory=dict)
    original_task: Optional[SubTask] = None


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical pattern.

    Attributes:
        max_hierarchy_depth: Maximum levels in the hierarchy.
        max_parallel_workers: Maximum workers running in parallel.
        escalation_threshold: Number of failures before escalation.
        synthesis_strategy: How to combine worker results.
        timeout_per_subtask: Timeout for each subtask in seconds.
    """

    max_hierarchy_depth: int = 3
    max_parallel_workers: int = 5
    escalation_threshold: int = 2
    synthesis_strategy: str = "aggregate"  # aggregate, reduce, chain
    timeout_per_subtask: float = 60.0


class HierarchicalPattern(BasePattern):
    """Hierarchical pattern for supervisor-worker agent teams.

    This pattern implements a tree-like structure where:
    1. A supervisor agent decomposes the main task into subtasks
    2. Worker agents execute the subtasks
    3. Results are synthesized back up the hierarchy
    4. Escalation occurs when workers cannot complete tasks

    The pattern supports multiple levels of hierarchy and
    handles task dependencies between workers.
    """

    def __init__(
        self,
        config: Optional[HierarchicalConfig] = None,
        decompose_fn: Optional[Callable] = None,
        synthesize_fn: Optional[Callable] = None,
    ):
        """Initialize the hierarchical pattern.

        Args:
            config: Configuration for the pattern.
            decompose_fn: Custom function to decompose tasks.
            synthesize_fn: Custom function to synthesize results.
        """
        super().__init__(name="hierarchical")
        self.config = config or HierarchicalConfig()
        self._decompose_fn = decompose_fn
        self._synthesize_fn = synthesize_fn
        self._escalations: List[EscalationRequest] = []
        self._subtask_results: Dict[str, TaskResult] = {}

    async def execute(
        self,
        task: Task,
        agents: List[BaseAgent],
        context: Optional[PatternContext] = None,
    ) -> PatternResult:
        """Execute the hierarchical pattern.

        The first agent in the list is treated as the supervisor,
        and remaining agents are workers.

        Args:
            task: The main task to execute.
            agents: List of agents (first is supervisor, rest are workers).
            context: Optional initial context.

        Returns:
            PatternResult with the synthesized output.
        """
        from agents_framework.agents.base import Task as AgentTask

        if not await self.validate_agents(agents):
            return PatternResult(
                pattern_name=self.name,
                status=PatternStatus.FAILED,
                error="Invalid agents: need at least one supervisor and one worker",
            )

        ctx = self._create_context(context)
        self._status = PatternStatus.RUNNING
        start_time = time.time()
        steps: List[StepResult] = []

        try:
            supervisor = agents[0]
            workers = agents[1:] if len(agents) > 1 else []

            # Step 1: Decompose the task into subtasks
            subtasks = await self._decompose_task(supervisor, task, ctx)
            steps.append(
                self._create_step_result(
                    agent=supervisor,
                    output={"subtasks": [s.description for s in subtasks]},
                    duration_ms=(time.time() - start_time) * 1000,
                )
            )
            ctx.set("subtasks", subtasks)

            # Step 2: Assign subtasks to workers
            assignments = await self._assign_subtasks(subtasks, workers, ctx)

            # Step 3: Execute subtasks respecting dependencies
            worker_results = await self._execute_subtasks(
                assignments, workers, ctx, steps
            )

            # Step 4: Handle any escalations
            if self._escalations:
                escalation_results = await self._handle_escalations(
                    supervisor, self._escalations, ctx
                )
                worker_results.update(escalation_results)

            # Step 5: Synthesize results
            synthesis_start = time.time()
            final_output = await self._synthesize_results(
                supervisor, worker_results, ctx
            )
            steps.append(
                self._create_step_result(
                    agent=supervisor,
                    output={"synthesis": "completed"},
                    duration_ms=(time.time() - synthesis_start) * 1000,
                )
            )

            total_duration = (time.time() - start_time) * 1000
            self._status = PatternStatus.COMPLETED

            return PatternResult(
                pattern_name=self.name,
                status=PatternStatus.COMPLETED,
                final_output=final_output,
                steps=steps,
                context=ctx,
                total_duration_ms=total_duration,
            )

        except Exception as e:
            self._status = PatternStatus.FAILED
            return PatternResult(
                pattern_name=self.name,
                status=PatternStatus.FAILED,
                error=str(e),
                steps=steps,
                context=ctx,
                total_duration_ms=(time.time() - start_time) * 1000,
            )

    async def validate_agents(self, agents: List[BaseAgent]) -> bool:
        """Validate agents for hierarchical pattern.

        Requires at least one supervisor. Workers are optional
        but recommended for actual task decomposition.
        """
        return len(agents) >= 1

    async def _decompose_task(
        self,
        supervisor: BaseAgent,
        task: Task,
        context: PatternContext,
    ) -> List[SubTask]:
        """Decompose the main task into subtasks.

        If a custom decompose function is provided, use it.
        Otherwise, use the supervisor agent to decompose.
        """
        from agents_framework.agents.base import Task as AgentTask

        if self._decompose_fn:
            return await self._decompose_fn(task, context)

        # Default: Ask the supervisor to decompose the task
        decompose_task = AgentTask(
            description=f"Decompose the following task into subtasks: {task.description}",
            context={
                "original_task": task.description,
                "max_subtasks": self.config.max_parallel_workers,
            },
        )

        result = await supervisor.run(decompose_task)

        # Parse the result into subtasks
        if result.success and result.output:
            return self._parse_subtasks(result.output)

        # Fallback: Create a single subtask with the original task
        return [SubTask(id="subtask_0", description=task.description)]

    def _parse_subtasks(self, output: Any) -> List[SubTask]:
        """Parse supervisor output into SubTask objects."""
        subtasks = []

        if isinstance(output, list):
            for i, item in enumerate(output):
                if isinstance(item, dict):
                    subtasks.append(
                        SubTask(
                            id=item.get("id", f"subtask_{i}"),
                            description=item.get("description", str(item)),
                            dependencies=item.get("dependencies", []),
                            priority=item.get("priority", 0),
                        )
                    )
                elif isinstance(item, str):
                    subtasks.append(SubTask(id=f"subtask_{i}", description=item))
        elif isinstance(output, str):
            # Single subtask from string output
            subtasks.append(SubTask(id="subtask_0", description=output))

        return subtasks if subtasks else [SubTask(id="subtask_0", description=str(output))]

    async def _assign_subtasks(
        self,
        subtasks: List[SubTask],
        workers: List[BaseAgent],
        context: PatternContext,
    ) -> Dict[str, SubTask]:
        """Assign subtasks to available workers.

        Uses round-robin assignment by default. Can be extended
        to use capability matching.
        """
        assignments: Dict[str, SubTask] = {}

        if not workers:
            # No workers, assign all to empty (supervisor will handle)
            for subtask in subtasks:
                subtask.assigned_agent_id = None
                assignments[subtask.id] = subtask
            return assignments

        for i, subtask in enumerate(subtasks):
            worker = workers[i % len(workers)]
            subtask.assigned_agent_id = worker.id
            assignments[subtask.id] = subtask

        context.set("assignments", {
            k: v.assigned_agent_id for k, v in assignments.items()
        })

        return assignments

    async def _execute_subtasks(
        self,
        assignments: Dict[str, SubTask],
        workers: List[BaseAgent],
        context: PatternContext,
        steps: List[StepResult],
    ) -> Dict[str, TaskResult]:
        """Execute subtasks respecting dependencies.

        Subtasks without dependencies run in parallel.
        Dependent subtasks wait for their dependencies to complete.
        """
        from agents_framework.agents.base import Task as AgentTask

        worker_map = {w.id: w for w in workers}
        results: Dict[str, TaskResult] = {}
        completed: set = set()
        failure_counts: Dict[str, int] = {}

        # Sort by priority (higher first) then by dependency count
        sorted_subtasks = sorted(
            assignments.values(),
            key=lambda s: (-s.priority, len(s.dependencies)),
        )

        while len(completed) < len(assignments):
            # Find subtasks ready to execute
            ready = [
                s for s in sorted_subtasks
                if s.id not in completed
                and all(dep in completed for dep in s.dependencies)
            ]

            if not ready:
                # No progress possible - circular dependency or all done
                break

            # Execute ready subtasks in parallel (up to limit)
            batch = ready[:self.config.max_parallel_workers]
            batch_tasks = []

            for subtask in batch:
                if subtask.assigned_agent_id and subtask.assigned_agent_id in worker_map:
                    worker = worker_map[subtask.assigned_agent_id]
                    agent_task = AgentTask(
                        description=subtask.description,
                        context={
                            "subtask_id": subtask.id,
                            "dependencies": {
                                dep: results.get(dep)
                                for dep in subtask.dependencies
                                if dep in results
                            },
                            **context.variables,
                        },
                    )
                    batch_tasks.append(
                        self._execute_single_subtask(
                            worker, agent_task, subtask, context, failure_counts
                        )
                    )
                else:
                    # No worker assigned, mark as completed with empty result
                    from agents_framework.agents.base import TaskResult
                    results[subtask.id] = TaskResult(
                        task_id=subtask.id,
                        success=False,
                        error="No worker assigned",
                    )
                    completed.add(subtask.id)

            if batch_tasks:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                for i, result in enumerate(batch_results):
                    subtask = batch[i]
                    if isinstance(result, Exception):
                        from agents_framework.agents.base import TaskResult
                        task_result = TaskResult(
                            task_id=subtask.id,
                            success=False,
                            error=str(result),
                        )
                    else:
                        task_result = result

                    results[subtask.id] = task_result
                    completed.add(subtask.id)

                    # Record step
                    worker = worker_map.get(subtask.assigned_agent_id) if subtask.assigned_agent_id else None
                    if worker:
                        steps.append(
                            self._create_step_result(
                                agent=worker,
                                output=task_result.output if task_result.success else None,
                                success=task_result.success,
                                error=task_result.error,
                            )
                        )

        return results

    async def _execute_single_subtask(
        self,
        worker: BaseAgent,
        task: Task,
        subtask: SubTask,
        context: PatternContext,
        failure_counts: Dict[str, int],
    ) -> TaskResult:
        """Execute a single subtask with timeout and escalation handling."""
        try:
            result = await asyncio.wait_for(
                worker.run(task),
                timeout=self.config.timeout_per_subtask,
            )

            if not result.success:
                failure_counts[worker.id] = failure_counts.get(worker.id, 0) + 1

                if failure_counts[worker.id] >= self.config.escalation_threshold:
                    self._escalations.append(
                        EscalationRequest(
                            from_agent_id=worker.id,
                            level=EscalationLevel.ERROR,
                            reason=f"Worker failed {failure_counts[worker.id]} times",
                            context={"last_error": result.error},
                            original_task=subtask,
                        )
                    )

            return result

        except asyncio.TimeoutError:
            from agents_framework.agents.base import TaskResult
            self._escalations.append(
                EscalationRequest(
                    from_agent_id=worker.id,
                    level=EscalationLevel.WARNING,
                    reason="Subtask timed out",
                    original_task=subtask,
                )
            )
            return TaskResult(
                task_id=task.id,
                success=False,
                error="Subtask execution timed out",
            )

    async def _handle_escalations(
        self,
        supervisor: BaseAgent,
        escalations: List[EscalationRequest],
        context: PatternContext,
    ) -> Dict[str, TaskResult]:
        """Handle escalated issues by involving the supervisor.

        The supervisor can retry the task, reassign to another worker,
        or handle it directly.
        """
        from agents_framework.agents.base import Task as AgentTask

        results: Dict[str, TaskResult] = {}

        for escalation in escalations:
            if escalation.original_task:
                # Ask supervisor to handle the escalated task
                escalation_task = AgentTask(
                    description=f"Handle escalated task: {escalation.original_task.description}. "
                    f"Reason: {escalation.reason}",
                    context={
                        "escalation_level": escalation.level.value,
                        "from_agent": escalation.from_agent_id,
                        **escalation.context,
                    },
                )

                result = await supervisor.run(escalation_task)
                results[escalation.original_task.id] = result

        return results

    async def _synthesize_results(
        self,
        supervisor: BaseAgent,
        results: Dict[str, TaskResult],
        context: PatternContext,
    ) -> Any:
        """Synthesize all subtask results into a final output.

        Uses the configured synthesis strategy or custom function.
        """
        from agents_framework.agents.base import Task as AgentTask

        if self._synthesize_fn:
            return await self._synthesize_fn(results, context)

        # Collect successful outputs
        outputs = {
            task_id: result.output
            for task_id, result in results.items()
            if result.success
        }

        if self.config.synthesis_strategy == "aggregate":
            # Simple aggregation
            return {
                "results": outputs,
                "failed": [
                    {"id": tid, "error": r.error}
                    for tid, r in results.items()
                    if not r.success
                ],
            }

        elif self.config.synthesis_strategy == "reduce":
            # Ask supervisor to reduce/summarize results
            synthesis_task = AgentTask(
                description="Synthesize the following subtask results into a coherent output",
                context={"results": outputs},
            )
            result = await supervisor.run(synthesis_task)
            return result.output if result.success else outputs

        else:
            # Default: return all outputs
            return outputs

    def escalate(
        self,
        from_agent_id: str,
        level: EscalationLevel,
        reason: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Manually trigger an escalation request.

        Args:
            from_agent_id: ID of the agent requesting escalation.
            level: Severity level.
            reason: Reason for escalation.
            context: Additional context.
        """
        self._escalations.append(
            EscalationRequest(
                from_agent_id=from_agent_id,
                level=level,
                reason=reason,
                context=context or {},
            )
        )
