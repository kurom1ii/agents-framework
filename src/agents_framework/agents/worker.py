"""Worker agent implementation for focused task execution.

This module provides the WorkerAgent class that extends BaseAgent
for executing specific tasks with:
- Task-specific tool filtering
- Focused execution (single task)
- Progress reporting
- Cancellation support
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Union,
)

from .base import (
    AgentConfig,
    AgentRole,
    AgentStatus,
    BaseAgent,
    Task,
    TaskResult,
)

if TYPE_CHECKING:
    from agents_framework.execution import AgentLoop, HookRegistry, LoopConfig
    from agents_framework.llm import LLMProvider
    from agents_framework.tools import ToolExecutor, ToolRegistry

logger = logging.getLogger(__name__)


class WorkerStatus(str, Enum):
    """Status specific to worker agents."""

    WAITING = "waiting"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class WorkerProgress:
    """Progress information for worker execution.

    Attributes:
        worker_id: ID of the worker agent.
        task_id: ID of the current task.
        status: Current worker status.
        current_step: Current step number.
        total_steps: Estimated total steps (if known).
        message: Current status message.
        started_at: When execution started.
        updated_at: When progress was last updated.
        metadata: Additional progress metadata.
    """

    worker_id: str
    task_id: str
    status: WorkerStatus = WorkerStatus.WAITING
    current_step: int = 0
    total_steps: Optional[int] = None
    message: str = ""
    started_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def progress_percentage(self) -> float:
        """Get progress as percentage (0-100)."""
        if self.total_steps is None or self.total_steps == 0:
            return 0.0
        return min(100.0, (self.current_step / self.total_steps) * 100)

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if not self.started_at:
            return 0.0
        return (self.updated_at - self.started_at).total_seconds()


@dataclass
class WorkerConfig(AgentConfig):
    """Configuration specific to worker agents.

    Attributes:
        allowed_tools: List of tool names this worker can use.
        denied_tools: List of tool names this worker cannot use.
        report_progress: Whether to report progress updates.
        progress_interval: Minimum interval between progress reports (seconds).
        allow_cancellation: Whether cancellation is allowed.
        checkpoint_enabled: Whether to save checkpoints for recovery.
    """

    allowed_tools: List[str] = field(default_factory=list)
    denied_tools: List[str] = field(default_factory=list)
    report_progress: bool = True
    progress_interval: float = 1.0
    allow_cancellation: bool = True
    checkpoint_enabled: bool = False


# Type alias for progress callback
ProgressCallback = Callable[[WorkerProgress], None]


class WorkerAgent(BaseAgent):
    """Worker agent for focused, single-task execution.

    Extends BaseAgent with:
    - Task-specific tool filtering based on configuration
    - Progress reporting with callbacks
    - Cancellation support
    - Focused execution optimized for single tasks

    Example:
        role = AgentRole(
            name="calculator",
            description="Math calculations",
            capabilities=["calculate"],
        )
        config = WorkerConfig(
            allowed_tools=["calculator", "math"],
            report_progress=True,
        )
        worker = WorkerAgent(role=role, llm=my_llm, config=config)

        # With progress callback
        def on_progress(progress):
            print(f"Progress: {progress.progress_percentage:.1f}%")

        result = await worker.run_with_progress(task, on_progress)
    """

    def __init__(
        self,
        role: AgentRole,
        llm: Optional[LLMProvider] = None,
        config: Optional[WorkerConfig] = None,
        tool_registry: Optional[ToolRegistry] = None,
        tool_executor: Optional[ToolExecutor] = None,
        hooks: Optional[HookRegistry] = None,
        system_prompt: str = "",
    ):
        """Initialize the worker agent.

        Args:
            role: The agent's role definition.
            llm: LLM provider for generating responses.
            config: Worker configuration.
            tool_registry: Registry of available tools.
            tool_executor: Executor for running tools.
            hooks: Hook registry for lifecycle events.
            system_prompt: System prompt for the agent.
        """
        super().__init__(role, llm, config or WorkerConfig())
        self.tool_registry = tool_registry
        self.tool_executor = tool_executor
        self.hooks = hooks
        self.system_prompt = system_prompt or self._build_default_system_prompt()

        self._worker_status = WorkerStatus.WAITING
        self._current_progress: Optional[WorkerProgress] = None
        self._cancel_requested = False
        self._progress_callbacks: List[ProgressCallback] = []
        self._last_progress_report = datetime.now()

    @property
    def worker_status(self) -> WorkerStatus:
        """Get the current worker status."""
        return self._worker_status

    @property
    def current_progress(self) -> Optional[WorkerProgress]:
        """Get the current progress information."""
        return self._current_progress

    @property
    def worker_config(self) -> WorkerConfig:
        """Get the worker configuration."""
        if isinstance(self.config, WorkerConfig):
            return self.config
        # Convert base config to worker config
        return WorkerConfig(
            name=self.config.name,
            max_iterations=self.config.max_iterations,
            timeout=self.config.timeout,
            tools=self.config.tools,
            memory_enabled=self.config.memory_enabled,
        )

    def add_progress_callback(self, callback: ProgressCallback) -> None:
        """Add a progress callback.

        Args:
            callback: Function to call on progress updates.
        """
        self._progress_callbacks.append(callback)

    def remove_progress_callback(self, callback: ProgressCallback) -> None:
        """Remove a progress callback.

        Args:
            callback: Function to remove.
        """
        try:
            self._progress_callbacks.remove(callback)
        except ValueError:
            pass

    def cancel(self) -> bool:
        """Request cancellation of current execution.

        Returns:
            True if cancellation was requested, False if not allowed.
        """
        if not self.worker_config.allow_cancellation:
            logger.warning(f"Worker {self.id}: Cancellation not allowed")
            return False

        if self._worker_status not in (WorkerStatus.EXECUTING, WorkerStatus.PAUSED):
            logger.warning(f"Worker {self.id}: Cannot cancel, not executing")
            return False

        self._cancel_requested = True
        self._worker_status = WorkerStatus.CANCELLED
        logger.info(f"Worker {self.id}: Cancellation requested")
        return True

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancel_requested

    async def run(self, task: Union[str, Task]) -> TaskResult:
        """Execute a task.

        Args:
            task: Task string or Task object to execute.

        Returns:
            TaskResult with execution results.
        """
        from agents_framework.execution import AgentLoop, LoopConfig

        # Convert string to Task if needed
        if isinstance(task, str):
            task_obj = Task(description=task)
        else:
            task_obj = task

        # Initialize progress tracking
        self._worker_status = WorkerStatus.EXECUTING
        self._cancel_requested = False
        self._current_progress = WorkerProgress(
            worker_id=self.id,
            task_id=task_obj.id,
            status=WorkerStatus.EXECUTING,
            started_at=datetime.now(),
            message="Starting execution",
        )

        await self._report_progress()

        try:
            # Check if LLM is available
            if not self.llm:
                return TaskResult(
                    task_id=task_obj.id,
                    success=False,
                    error="No LLM provider configured",
                )

            # Create filtered tool registry
            filtered_registry = self._get_filtered_tool_registry()

            # Create the execution loop
            loop_config = LoopConfig(
                max_iterations=self.config.max_iterations,
                timeout=self.config.timeout,
            )

            loop = AgentLoop(
                llm=self.llm,
                tool_registry=filtered_registry,
                tool_executor=self.tool_executor,
                config=loop_config,
                hooks=self.hooks,
                system_prompt=self.system_prompt,
            )

            # Run the loop with progress tracking
            state = await self._run_with_progress_tracking(
                loop, task_obj.description, task_obj
            )

            # Check for cancellation
            if self._cancel_requested:
                self._worker_status = WorkerStatus.CANCELLED
                self._current_progress.status = WorkerStatus.CANCELLED
                self._current_progress.message = "Execution cancelled"
                await self._report_progress()

                return TaskResult(
                    task_id=task_obj.id,
                    success=False,
                    error="Execution cancelled",
                    metadata={"cancellation_requested": True},
                )

            # Build result
            success = state.status == "completed"
            output = None
            error = state.error

            if success and state.steps:
                # Get the last meaningful output
                for step in reversed(state.steps):
                    if step.content:
                        output = step.content
                        break

            self._worker_status = (
                WorkerStatus.COMPLETED if success else WorkerStatus.FAILED
            )
            self._current_progress.status = self._worker_status
            self._current_progress.message = "Completed" if success else f"Failed: {error}"
            await self._report_progress()

            return TaskResult(
                task_id=task_obj.id,
                success=success,
                output=output,
                error=error,
                metadata={
                    "iterations": state.iteration,
                    "total_tokens": state.total_tokens,
                    "duration_seconds": state.duration,
                    "termination_reason": (
                        state.termination_reason.value
                        if state.termination_reason
                        else None
                    ),
                },
            )

        except asyncio.CancelledError:
            self._worker_status = WorkerStatus.CANCELLED
            self._current_progress.status = WorkerStatus.CANCELLED
            self._current_progress.message = "Execution cancelled"
            await self._report_progress()

            return TaskResult(
                task_id=task_obj.id,
                success=False,
                error="Execution cancelled",
            )

        except Exception as e:
            logger.exception(f"Worker {self.id}: Error executing task")
            self._worker_status = WorkerStatus.FAILED
            self._current_progress.status = WorkerStatus.FAILED
            self._current_progress.message = f"Error: {e}"
            await self._report_progress()

            return TaskResult(
                task_id=task_obj.id,
                success=False,
                error=str(e),
            )

    async def run_with_progress(
        self,
        task: Union[str, Task],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> TaskResult:
        """Execute a task with progress reporting.

        Args:
            task: Task to execute.
            progress_callback: Optional callback for progress updates.

        Returns:
            TaskResult with execution results.
        """
        if progress_callback:
            self.add_progress_callback(progress_callback)

        try:
            return await self.run(task)
        finally:
            if progress_callback:
                self.remove_progress_callback(progress_callback)

    async def run_stream(
        self,
        task: Union[str, Task],
    ) -> AsyncIterator[WorkerProgress]:
        """Execute a task and stream progress updates.

        Args:
            task: Task to execute.

        Yields:
            WorkerProgress objects as execution progresses.
        """
        from agents_framework.execution import AgentLoop, LoopConfig

        # Convert string to Task if needed
        if isinstance(task, str):
            task_obj = Task(description=task)
        else:
            task_obj = task

        # Initialize progress tracking
        self._worker_status = WorkerStatus.EXECUTING
        self._cancel_requested = False
        self._current_progress = WorkerProgress(
            worker_id=self.id,
            task_id=task_obj.id,
            status=WorkerStatus.EXECUTING,
            started_at=datetime.now(),
            message="Starting execution",
        )

        yield self._current_progress

        try:
            if not self.llm:
                self._current_progress.status = WorkerStatus.FAILED
                self._current_progress.message = "No LLM provider configured"
                yield self._current_progress
                return

            filtered_registry = self._get_filtered_tool_registry()

            loop_config = LoopConfig(
                max_iterations=self.config.max_iterations,
                timeout=self.config.timeout,
                enable_streaming=True,
            )

            loop = AgentLoop(
                llm=self.llm,
                tool_registry=filtered_registry,
                tool_executor=self.tool_executor,
                config=loop_config,
                hooks=self.hooks,
                system_prompt=self.system_prompt,
            )

            step_count = 0
            async for step in loop.run_stream(task_obj.description):
                if self._cancel_requested:
                    break

                step_count += 1
                self._current_progress.current_step = step_count
                self._current_progress.total_steps = self.config.max_iterations
                self._current_progress.message = f"Step {step_count}: {step.step_type.value}"
                self._current_progress.updated_at = datetime.now()
                self._current_progress.metadata["last_step"] = {
                    "type": step.step_type.value,
                    "content": step.content[:100] if step.content else "",
                }
                yield self._current_progress

            # Final status
            if self._cancel_requested:
                self._current_progress.status = WorkerStatus.CANCELLED
                self._current_progress.message = "Execution cancelled"
            elif loop.state and loop.state.status == "completed":
                self._current_progress.status = WorkerStatus.COMPLETED
                self._current_progress.message = "Completed successfully"
            else:
                self._current_progress.status = WorkerStatus.FAILED
                self._current_progress.message = (
                    loop.state.error if loop.state else "Unknown error"
                )

            yield self._current_progress

        except Exception as e:
            logger.exception(f"Worker {self.id}: Error in stream execution")
            self._current_progress.status = WorkerStatus.FAILED
            self._current_progress.message = f"Error: {e}"
            yield self._current_progress

    def _get_filtered_tool_registry(self) -> Optional[ToolRegistry]:
        """Get a filtered tool registry based on configuration.

        Returns:
            Filtered ToolRegistry or None if no registry.
        """
        if not self.tool_registry:
            return None

        from agents_framework.tools import ToolRegistry

        config = self.worker_config
        all_tools = self.tool_registry.list_tools()

        # Filter tools based on allowed/denied lists
        filtered_tools = []
        for tool in all_tools:
            # Check if tool is explicitly denied
            if config.denied_tools and tool.name in config.denied_tools:
                continue

            # Check if tool is explicitly allowed (if allow list is specified)
            if config.allowed_tools:
                if tool.name not in config.allowed_tools:
                    continue

            # Check if tool matches required capabilities
            if self.role.capabilities:
                # Tool should match at least one capability
                tool_matches = any(
                    cap.lower() in tool.name.lower()
                    or cap.lower() in tool.description.lower()
                    for cap in self.role.capabilities
                )
                # If we have capability requirements and tool doesn't match, skip it
                # unless it's in the allowed list
                if (
                    not tool_matches
                    and config.allowed_tools
                    and tool.name not in config.allowed_tools
                ):
                    continue

            filtered_tools.append(tool)

        # Create new registry with filtered tools
        filtered_registry = ToolRegistry()
        for tool in filtered_tools:
            filtered_registry.register(tool)

        logger.debug(
            f"Worker {self.id}: Filtered tools from {len(all_tools)} to {len(filtered_tools)}"
        )

        return filtered_registry

    async def _run_with_progress_tracking(
        self,
        loop: AgentLoop,
        task_description: str,
        task: Task,
    ) -> Any:
        """Run the loop with progress tracking.

        Args:
            loop: The agent loop to run.
            task_description: Description of the task.
            task: The task object.

        Returns:
            LoopState from execution.
        """
        # Run the loop
        state = await loop.run(task_description)

        # Update progress with final state
        if state.steps:
            self._current_progress.current_step = len(state.steps)
            self._current_progress.total_steps = state.iteration

        return state

    async def _report_progress(self) -> None:
        """Report progress to registered callbacks."""
        if not self._current_progress:
            return

        if not self.worker_config.report_progress:
            return

        # Check progress interval
        now = datetime.now()
        elapsed = (now - self._last_progress_report).total_seconds()
        if elapsed < self.worker_config.progress_interval:
            return

        self._last_progress_report = now
        self._current_progress.updated_at = now

        # Call all callbacks
        for callback in self._progress_callbacks:
            try:
                result = callback(self._current_progress)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")

    def _build_default_system_prompt(self) -> str:
        """Build a default system prompt based on role."""
        return (
            f"You are a {self.role.name} agent.\n"
            f"Description: {self.role.description}\n"
            f"Capabilities: {', '.join(self.role.capabilities) if self.role.capabilities else 'general'}\n"
            f"\n{self.role.instructions}"
        )

    def get_status_info(self) -> Dict[str, Any]:
        """Get detailed status information.

        Returns:
            Dictionary with status information.
        """
        return {
            "id": self.id,
            "role": self.role.name,
            "status": self._status.value,
            "worker_status": self._worker_status.value,
            "cancel_requested": self._cancel_requested,
            "progress": (
                {
                    "current_step": self._current_progress.current_step,
                    "total_steps": self._current_progress.total_steps,
                    "percentage": self._current_progress.progress_percentage,
                    "message": self._current_progress.message,
                    "elapsed_seconds": self._current_progress.elapsed_time,
                }
                if self._current_progress
                else None
            ),
        }

    def __repr__(self) -> str:
        return (
            f"WorkerAgent(id={self.id!r}, role={self.role.name!r}, "
            f"status={self._worker_status.value!r})"
        )
