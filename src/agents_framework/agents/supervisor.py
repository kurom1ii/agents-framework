"""Supervisor agent for task delegation and orchestration.

This module provides the SupervisorAgent class which extends BaseAgent
to coordinate work among multiple worker agents, aggregate results,
and handle worker failures.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from .base import AgentConfig, AgentRole, AgentStatus, BaseAgent, Task, TaskResult

if TYPE_CHECKING:
    from agents_framework.teams.registry import AgentRegistry


class DelegationStrategy(str, Enum):
    """Strategy for delegating tasks to workers."""

    ROUND_ROBIN = "round_robin"  # Rotate through workers
    LEAST_BUSY = "least_busy"  # Select worker with fewest pending tasks
    CAPABILITY_MATCH = "capability_match"  # Match by required capabilities
    RANDOM = "random"  # Random selection
    PRIORITY = "priority"  # Select based on worker priority/weight


class ExecutionMode(str, Enum):
    """Mode for executing delegated tasks."""

    PARALLEL = "parallel"  # Execute all tasks in parallel
    SEQUENTIAL = "sequential"  # Execute tasks one by one
    PIPELINE = "pipeline"  # Pass output of one task as input to next


@dataclass
class DelegatedTask:
    """A task delegated to a worker agent.

    Attributes:
        task: The task to execute.
        worker_id: ID of the assigned worker.
        assigned_at: When the task was assigned.
        completed_at: When the task was completed.
        result: The task result when completed.
        retries: Number of retry attempts.
    """

    task: Task
    worker_id: str
    assigned_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    result: Optional[TaskResult] = None
    retries: int = 0


@dataclass
class SupervisorConfig(AgentConfig):
    """Configuration for a supervisor agent.

    Attributes:
        delegation_strategy: Strategy for selecting workers.
        execution_mode: How to execute delegated tasks.
        max_workers: Maximum concurrent workers.
        task_timeout: Timeout for individual worker tasks.
        max_retries: Maximum retries for failed tasks.
        retry_delay: Delay between retries in seconds.
        fallback_enabled: Whether to enable fallback on failures.
    """

    delegation_strategy: DelegationStrategy = DelegationStrategy.CAPABILITY_MATCH
    execution_mode: ExecutionMode = ExecutionMode.PARALLEL
    max_workers: int = 10
    task_timeout: float = 60.0
    max_retries: int = 2
    retry_delay: float = 1.0
    fallback_enabled: bool = True


WorkerSelector = Callable[[List[BaseAgent], Task], Optional[BaseAgent]]
ResultAggregator = Callable[[List[TaskResult]], Any]


class SupervisorAgent(BaseAgent):
    """Supervisor agent that delegates tasks to worker agents.

    The supervisor coordinates work among multiple worker agents,
    delegating tasks based on configurable strategies, aggregating
    results, and handling failures with retries and fallbacks.

    Example:
        from agents_framework.agents import SupervisorAgent, AgentRole
        from agents_framework.teams import AgentRegistry

        role = AgentRole(
            name="supervisor",
            description="Coordinates research tasks",
            capabilities=["delegate", "aggregate"],
        )

        supervisor = SupervisorAgent(
            role=role,
            config=SupervisorConfig(
                execution_mode=ExecutionMode.PARALLEL,
                delegation_strategy=DelegationStrategy.CAPABILITY_MATCH,
            ),
        )

        # Add workers
        supervisor.add_worker(researcher_agent)
        supervisor.add_worker(analyst_agent)

        # Execute a complex task
        result = await supervisor.run(task)
    """

    def __init__(
        self,
        role: AgentRole,
        llm: Any = None,
        config: Optional[SupervisorConfig] = None,
        registry: Optional["AgentRegistry"] = None,
    ) -> None:
        """Initialize the supervisor agent.

        Args:
            role: The agent's role definition.
            llm: Optional LLM provider.
            config: Supervisor configuration.
            registry: Optional agent registry for worker discovery.
        """
        super().__init__(role=role, llm=llm, config=config or SupervisorConfig())
        self._config: SupervisorConfig = config or SupervisorConfig()
        self._workers: Dict[str, BaseAgent] = {}
        self._registry = registry
        self._round_robin_index = 0
        self._custom_selector: Optional[WorkerSelector] = None
        self._custom_aggregator: Optional[ResultAggregator] = None
        self._task_history: List[DelegatedTask] = []

    @property
    def workers(self) -> List[BaseAgent]:
        """Get list of registered workers."""
        return list(self._workers.values())

    def add_worker(self, worker: BaseAgent) -> None:
        """Add a worker agent.

        Args:
            worker: The worker agent to add.
        """
        worker.parent_id = self.id
        self._workers[worker.id] = worker

    def remove_worker(self, worker_id: str) -> Optional[BaseAgent]:
        """Remove a worker agent.

        Args:
            worker_id: ID of the worker to remove.

        Returns:
            The removed worker if found.
        """
        worker = self._workers.pop(worker_id, None)
        if worker:
            worker.parent_id = None
        return worker

    def get_worker(self, worker_id: str) -> Optional[BaseAgent]:
        """Get a worker by ID.

        Args:
            worker_id: The worker's ID.

        Returns:
            The worker if found.
        """
        return self._workers.get(worker_id)

    def set_selector(self, selector: WorkerSelector) -> None:
        """Set a custom worker selection function.

        Args:
            selector: Function that selects a worker for a task.
        """
        self._custom_selector = selector

    def set_aggregator(self, aggregator: ResultAggregator) -> None:
        """Set a custom result aggregation function.

        Args:
            aggregator: Function that aggregates task results.
        """
        self._custom_aggregator = aggregator

    async def run(self, task: str | Task) -> TaskResult:
        """Execute a task by delegating to workers.

        Args:
            task: The task to execute (string description or Task object).

        Returns:
            Aggregated TaskResult from worker executions.
        """
        if isinstance(task, str):
            task = Task(description=task)

        self._status = AgentStatus.BUSY

        try:
            # Break down task if needed
            subtasks = await self._decompose_task(task)

            # Delegate and execute subtasks
            results = await self._execute_subtasks(subtasks)

            # Aggregate results
            aggregated = await self._aggregate_results(task, results)

            self._status = AgentStatus.IDLE
            return aggregated

        except Exception as e:
            self._status = AgentStatus.ERROR
            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e),
            )

    async def _decompose_task(self, task: Task) -> List[Task]:
        """Decompose a task into subtasks.

        Override this method to implement custom task decomposition logic.

        Args:
            task: The task to decompose.

        Returns:
            List of subtasks (by default, just the original task).
        """
        # Default: no decomposition
        return [task]

    async def _execute_subtasks(self, subtasks: List[Task]) -> List[TaskResult]:
        """Execute subtasks according to the execution mode.

        Args:
            subtasks: List of subtasks to execute.

        Returns:
            List of TaskResults from execution.
        """
        if self._config.execution_mode == ExecutionMode.PARALLEL:
            return await self._execute_parallel(subtasks)
        elif self._config.execution_mode == ExecutionMode.SEQUENTIAL:
            return await self._execute_sequential(subtasks)
        elif self._config.execution_mode == ExecutionMode.PIPELINE:
            return await self._execute_pipeline(subtasks)
        else:
            return await self._execute_parallel(subtasks)

    async def _execute_parallel(self, subtasks: List[Task]) -> List[TaskResult]:
        """Execute subtasks in parallel.

        Args:
            subtasks: List of subtasks to execute.

        Returns:
            List of TaskResults.
        """
        async def execute_with_retry(task: Task) -> TaskResult:
            worker = self._select_worker(task)
            if not worker:
                return TaskResult(
                    task_id=task.id,
                    success=False,
                    error="No available worker for task",
                )

            delegated = DelegatedTask(task=task, worker_id=worker.id)

            for attempt in range(self._config.max_retries + 1):
                try:
                    result = await asyncio.wait_for(
                        worker.execute(task),
                        timeout=self._config.task_timeout,
                    )
                    delegated.result = result
                    delegated.completed_at = datetime.now()
                    self._task_history.append(delegated)

                    if result.success:
                        return result

                    # Task failed, try fallback if enabled
                    if self._config.fallback_enabled and attempt < self._config.max_retries:
                        delegated.retries += 1
                        await asyncio.sleep(self._config.retry_delay)
                        worker = self._select_fallback_worker(task, worker.id)
                        if not worker:
                            break
                    else:
                        break

                except asyncio.TimeoutError:
                    delegated.retries += 1
                    if attempt >= self._config.max_retries:
                        return TaskResult(
                            task_id=task.id,
                            success=False,
                            error=f"Task timed out after {self._config.task_timeout}s",
                        )
                    await asyncio.sleep(self._config.retry_delay)
                    worker = self._select_fallback_worker(task, worker.id)
                    if not worker:
                        return TaskResult(
                            task_id=task.id,
                            success=False,
                            error="No fallback worker available",
                        )

            return delegated.result or TaskResult(
                task_id=task.id,
                success=False,
                error="Task failed after all retries",
            )

        # Execute all tasks in parallel with concurrency limit
        semaphore = asyncio.Semaphore(self._config.max_workers)

        async def bounded_execute(task: Task) -> TaskResult:
            async with semaphore:
                return await execute_with_retry(task)

        results = await asyncio.gather(
            *[bounded_execute(task) for task in subtasks],
            return_exceptions=True,
        )

        # Convert exceptions to failed results
        processed_results: List[TaskResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(TaskResult(
                    task_id=subtasks[i].id,
                    success=False,
                    error=str(result),
                ))
            else:
                processed_results.append(result)

        return processed_results

    async def _execute_sequential(self, subtasks: List[Task]) -> List[TaskResult]:
        """Execute subtasks sequentially.

        Args:
            subtasks: List of subtasks to execute.

        Returns:
            List of TaskResults.
        """
        results: List[TaskResult] = []

        for task in subtasks:
            worker = self._select_worker(task)
            if not worker:
                results.append(TaskResult(
                    task_id=task.id,
                    success=False,
                    error="No available worker for task",
                ))
                continue

            delegated = DelegatedTask(task=task, worker_id=worker.id)

            try:
                result = await asyncio.wait_for(
                    worker.execute(task),
                    timeout=self._config.task_timeout,
                )
                delegated.result = result
                delegated.completed_at = datetime.now()
                self._task_history.append(delegated)
                results.append(result)

                # Stop on failure unless fallback succeeds
                if not result.success:
                    if self._config.fallback_enabled:
                        fallback_result = await self._try_fallback(task, worker.id)
                        if fallback_result:
                            results[-1] = fallback_result
                            continue
                    break

            except asyncio.TimeoutError:
                results.append(TaskResult(
                    task_id=task.id,
                    success=False,
                    error=f"Task timed out after {self._config.task_timeout}s",
                ))
                break

        return results

    async def _execute_pipeline(self, subtasks: List[Task]) -> List[TaskResult]:
        """Execute subtasks as a pipeline, passing output to next task.

        Args:
            subtasks: List of subtasks to execute.

        Returns:
            List of TaskResults.
        """
        results: List[TaskResult] = []
        previous_output: Any = None

        for task in subtasks:
            # Inject previous output into task context
            if previous_output is not None:
                task.context["previous_output"] = previous_output

            worker = self._select_worker(task)
            if not worker:
                results.append(TaskResult(
                    task_id=task.id,
                    success=False,
                    error="No available worker for task",
                ))
                break

            delegated = DelegatedTask(task=task, worker_id=worker.id)

            try:
                result = await asyncio.wait_for(
                    worker.execute(task),
                    timeout=self._config.task_timeout,
                )
                delegated.result = result
                delegated.completed_at = datetime.now()
                self._task_history.append(delegated)
                results.append(result)

                if result.success:
                    previous_output = result.output
                else:
                    break

            except asyncio.TimeoutError:
                results.append(TaskResult(
                    task_id=task.id,
                    success=False,
                    error=f"Task timed out after {self._config.task_timeout}s",
                ))
                break

        return results

    def _select_worker(self, task: Task) -> Optional[BaseAgent]:
        """Select a worker for a task using the configured strategy.

        Args:
            task: The task to assign.

        Returns:
            Selected worker or None if no suitable worker found.
        """
        # Use custom selector if provided
        if self._custom_selector:
            return self._custom_selector(self.workers, task)

        available = self._get_available_workers()
        if not available:
            return None

        strategy = self._config.delegation_strategy

        if strategy == DelegationStrategy.ROUND_ROBIN:
            return self._select_round_robin(available)
        elif strategy == DelegationStrategy.CAPABILITY_MATCH:
            return self._select_by_capability(available, task)
        elif strategy == DelegationStrategy.RANDOM:
            return self._select_random(available)
        else:
            return available[0] if available else None

    def _get_available_workers(self) -> List[BaseAgent]:
        """Get list of available (idle or not terminated) workers."""
        available = []

        # Check local workers
        for worker in self._workers.values():
            if worker.status in (AgentStatus.IDLE, AgentStatus.BUSY):
                available.append(worker)

        # Check registry if available
        if self._registry:
            for agent in self._registry.find_idle():
                if agent.id not in self._workers and agent.id != self.id:
                    available.append(agent)

        return available

    def _select_round_robin(self, workers: List[BaseAgent]) -> Optional[BaseAgent]:
        """Select worker using round-robin."""
        if not workers:
            return None
        index = self._round_robin_index % len(workers)
        self._round_robin_index += 1
        return workers[index]

    def _select_by_capability(
        self, workers: List[BaseAgent], task: Task
    ) -> Optional[BaseAgent]:
        """Select worker matching required capabilities."""
        if not task.required_capabilities:
            return workers[0] if workers else None

        for worker in workers:
            if all(
                worker.has_capability(cap) for cap in task.required_capabilities
            ):
                return worker

        # Fallback to first available if no exact match
        return workers[0] if workers else None

    def _select_random(self, workers: List[BaseAgent]) -> Optional[BaseAgent]:
        """Select worker randomly."""
        import random
        return random.choice(workers) if workers else None

    def _select_fallback_worker(
        self, task: Task, exclude_id: str
    ) -> Optional[BaseAgent]:
        """Select a fallback worker, excluding the failed one.

        Args:
            task: The task to assign.
            exclude_id: ID of the worker to exclude.

        Returns:
            Fallback worker or None.
        """
        available = [w for w in self._get_available_workers() if w.id != exclude_id]
        if not available:
            return None

        if self._config.delegation_strategy == DelegationStrategy.CAPABILITY_MATCH:
            return self._select_by_capability(available, task)

        return available[0]

    async def _try_fallback(
        self, task: Task, failed_worker_id: str
    ) -> Optional[TaskResult]:
        """Try executing task with a fallback worker.

        Args:
            task: The task to execute.
            failed_worker_id: ID of the worker that failed.

        Returns:
            TaskResult if fallback succeeded, None otherwise.
        """
        for attempt in range(self._config.max_retries):
            worker = self._select_fallback_worker(task, failed_worker_id)
            if not worker:
                return None

            try:
                result = await asyncio.wait_for(
                    worker.execute(task),
                    timeout=self._config.task_timeout,
                )
                if result.success:
                    return result
                failed_worker_id = worker.id

            except asyncio.TimeoutError:
                failed_worker_id = worker.id
                continue

            await asyncio.sleep(self._config.retry_delay)

        return None

    async def _aggregate_results(
        self, original_task: Task, results: List[TaskResult]
    ) -> TaskResult:
        """Aggregate results from worker executions.

        Args:
            original_task: The original parent task.
            results: List of results from subtask executions.

        Returns:
            Aggregated TaskResult.
        """
        # Use custom aggregator if provided
        if self._custom_aggregator:
            try:
                aggregated_output = self._custom_aggregator(results)
                return TaskResult(
                    task_id=original_task.id,
                    success=all(r.success for r in results),
                    output=aggregated_output,
                    metadata={"subtask_count": len(results)},
                )
            except Exception as e:
                return TaskResult(
                    task_id=original_task.id,
                    success=False,
                    error=f"Aggregation failed: {e}",
                )

        # Default aggregation
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        if not results:
            return TaskResult(
                task_id=original_task.id,
                success=False,
                error="No subtasks were executed",
            )

        all_success = len(failed) == 0

        return TaskResult(
            task_id=original_task.id,
            success=all_success,
            output={
                "successful_results": [r.output for r in successful],
                "failed_count": len(failed),
                "errors": [r.error for r in failed if r.error],
            },
            metadata={
                "total_subtasks": len(results),
                "successful_count": len(successful),
                "failed_count": len(failed),
            },
        )

    def get_task_history(self) -> List[DelegatedTask]:
        """Get history of delegated tasks.

        Returns:
            List of DelegatedTask records.
        """
        return list(self._task_history)

    def clear_task_history(self) -> None:
        """Clear the task history."""
        self._task_history.clear()

    def get_worker_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for each worker.

        Returns:
            Dictionary mapping worker IDs to their statistics.
        """
        stats: Dict[str, Dict[str, Any]] = {}

        for worker_id in self._workers:
            worker_tasks = [t for t in self._task_history if t.worker_id == worker_id]
            successful = sum(1 for t in worker_tasks if t.result and t.result.success)
            failed = sum(1 for t in worker_tasks if t.result and not t.result.success)

            stats[worker_id] = {
                "total_tasks": len(worker_tasks),
                "successful": successful,
                "failed": failed,
                "success_rate": successful / len(worker_tasks) if worker_tasks else 0.0,
                "total_retries": sum(t.retries for t in worker_tasks),
            }

        return stats

    def __repr__(self) -> str:
        return (
            f"SupervisorAgent(id={self.id!r}, role={self.role.name!r}, "
            f"workers={len(self._workers)})"
        )
