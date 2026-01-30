"""Agent execution loop implementing the ReAct pattern.

This module provides the core execution engine for agents, implementing
the Thought -> Action -> Observation cycle with support for:
- Iteration limits and termination conditions
- Error handling and recovery
- Streaming support
- Execution management via AgentRunner
"""

from __future__ import annotations

import asyncio
import logging
import time
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
    Union,
)

if TYPE_CHECKING:
    from agents_framework.agents import BaseAgent, Task, TaskResult
    from agents_framework.llm import LLMProvider, Message, ToolCall
    from agents_framework.tools import ToolExecutor, ToolRegistry

    from .hooks import HookRegistry

logger = logging.getLogger(__name__)


class StepType(str, Enum):
    """Type of step in the ReAct loop."""

    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL = "final"
    ERROR = "error"


class TerminationReason(str, Enum):
    """Reason for loop termination."""

    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations"
    TIMEOUT = "timeout"
    ERROR = "error"
    CANCELLED = "cancelled"
    NO_TOOL_CALLS = "no_tool_calls"
    FINAL_ANSWER = "final_answer"


@dataclass
class LoopStep:
    """Represents a single step in the ReAct loop.

    Attributes:
        step_type: The type of this step (thought, action, observation, etc.).
        content: The content produced in this step.
        tool_calls: Optional list of tool calls for action steps.
        tool_results: Optional results from tool execution.
        timestamp: When this step occurred.
        duration: Time taken for this step in seconds.
        metadata: Additional metadata for this step.
    """

    step_type: StepType
    content: str = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoopState:
    """State of the execution loop.

    Attributes:
        id: Unique identifier for this execution.
        iteration: Current iteration number.
        steps: List of steps taken so far.
        status: Current status of the loop.
        termination_reason: Reason for termination if finished.
        started_at: When the loop started.
        finished_at: When the loop finished.
        total_tokens: Total tokens consumed.
        error: Error message if failed.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    iteration: int = 0
    steps: List[LoopStep] = field(default_factory=list)
    status: str = "pending"
    termination_reason: Optional[TerminationReason] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    total_tokens: int = 0
    error: Optional[str] = None

    @property
    def duration(self) -> float:
        """Get total duration in seconds."""
        if not self.started_at:
            return 0.0
        end = self.finished_at or datetime.now()
        return (end - self.started_at).total_seconds()

    @property
    def is_running(self) -> bool:
        """Check if the loop is currently running."""
        return self.status == "running"

    @property
    def is_finished(self) -> bool:
        """Check if the loop has finished."""
        return self.status in ("completed", "failed", "cancelled")


@dataclass
class LoopConfig:
    """Configuration for the agent execution loop.

    Attributes:
        max_iterations: Maximum number of iterations before termination.
        timeout: Maximum execution time in seconds.
        enable_streaming: Whether to enable streaming responses.
        continue_on_error: Whether to continue execution after tool errors.
        max_consecutive_errors: Maximum consecutive errors before termination.
        thought_prefix: Prefix to identify thought output.
        final_answer_prefix: Prefix to identify final answer.
    """

    max_iterations: int = 10
    timeout: float = 300.0
    enable_streaming: bool = False
    continue_on_error: bool = True
    max_consecutive_errors: int = 3
    thought_prefix: str = "Thought:"
    final_answer_prefix: str = "Final Answer:"


class AgentLoop:
    """Execution loop implementing the ReAct pattern.

    The ReAct (Reasoning and Acting) pattern alternates between:
    1. Thought - LLM reasons about the current state
    2. Action - LLM selects and executes a tool
    3. Observation - Results from tool execution

    This cycle continues until a final answer is produced,
    max iterations is reached, or an error occurs.

    Example:
        loop = AgentLoop(
            llm=my_provider,
            tool_registry=my_registry,
            config=LoopConfig(max_iterations=5)
        )
        result = await loop.run("What is 2+2?")
    """

    def __init__(
        self,
        llm: LLMProvider,
        tool_registry: Optional[ToolRegistry] = None,
        tool_executor: Optional[ToolExecutor] = None,
        config: Optional[LoopConfig] = None,
        hooks: Optional[HookRegistry] = None,
        system_prompt: str = "",
    ):
        """Initialize the agent loop.

        Args:
            llm: The LLM provider for generating responses.
            tool_registry: Registry of available tools.
            tool_executor: Executor for running tools.
            config: Loop configuration.
            hooks: Hook registry for lifecycle events.
            system_prompt: System prompt for the agent.
        """
        self.llm = llm
        self.tool_registry = tool_registry
        self.tool_executor = tool_executor
        self.config = config or LoopConfig()
        self.hooks = hooks
        self.system_prompt = system_prompt

        self._state: Optional[LoopState] = None
        self._cancel_requested = False
        self._messages: List[Message] = []

    @property
    def state(self) -> Optional[LoopState]:
        """Get current loop state."""
        return self._state

    async def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> LoopState:
        """Run the agent loop for a given task.

        Args:
            task: The task/query to execute.
            context: Optional additional context.

        Returns:
            LoopState with execution results.
        """
        from agents_framework.llm import Message, MessageRole

        # Initialize state
        self._state = LoopState(status="running", started_at=datetime.now())
        self._cancel_requested = False
        self._messages = []

        # Build initial messages
        if self.system_prompt:
            self._messages.append(
                Message(role=MessageRole.SYSTEM, content=self._build_system_prompt())
            )

        self._messages.append(Message(role=MessageRole.USER, content=task))

        # Fire pre-execute hook
        if self.hooks:
            await self.hooks.fire(
                "pre_execute",
                task=task,
                context=context,
                state=self._state,
            )

        consecutive_errors = 0

        try:
            # Main execution loop
            while self._should_continue():
                self._state.iteration += 1
                step_start = time.monotonic()

                try:
                    # Generate LLM response (Thought + Action)
                    step = await self._execute_step()
                    step.duration = time.monotonic() - step_start
                    self._state.steps.append(step)

                    # Check for final answer
                    if step.step_type == StepType.FINAL:
                        self._state.termination_reason = TerminationReason.FINAL_ANSWER
                        break

                    # Execute tools if any (Action -> Observation)
                    if step.tool_calls:
                        observation_step = await self._execute_tools(step.tool_calls)
                        observation_step.duration = (
                            time.monotonic() - step_start - step.duration
                        )
                        self._state.steps.append(observation_step)
                        consecutive_errors = 0
                    else:
                        # No tool calls and no final answer - might be done
                        if not step.content.strip():
                            self._state.termination_reason = (
                                TerminationReason.NO_TOOL_CALLS
                            )
                            break

                except asyncio.TimeoutError:
                    self._state.termination_reason = TerminationReason.TIMEOUT
                    self._state.error = "Execution timed out"
                    break

                except Exception as e:
                    consecutive_errors += 1
                    logger.error(f"Error in iteration {self._state.iteration}: {e}")

                    error_step = LoopStep(
                        step_type=StepType.ERROR,
                        content=str(e),
                        duration=time.monotonic() - step_start,
                    )
                    self._state.steps.append(error_step)

                    if (
                        not self.config.continue_on_error
                        or consecutive_errors >= self.config.max_consecutive_errors
                    ):
                        self._state.termination_reason = TerminationReason.ERROR
                        self._state.error = str(e)
                        break

            # Check termination reason if not set
            if self._state.termination_reason is None:
                if self._cancel_requested:
                    self._state.termination_reason = TerminationReason.CANCELLED
                elif self._state.iteration >= self.config.max_iterations:
                    self._state.termination_reason = TerminationReason.MAX_ITERATIONS
                else:
                    self._state.termination_reason = TerminationReason.COMPLETED

            # Update final state
            self._state.status = (
                "completed"
                if self._state.termination_reason
                in (
                    TerminationReason.COMPLETED,
                    TerminationReason.FINAL_ANSWER,
                    TerminationReason.NO_TOOL_CALLS,
                )
                else "failed"
            )

        except Exception as e:
            self._state.status = "failed"
            self._state.termination_reason = TerminationReason.ERROR
            self._state.error = str(e)
            logger.exception("Unhandled error in agent loop")

        finally:
            self._state.finished_at = datetime.now()

            # Fire post-execute hook
            if self.hooks:
                await self.hooks.fire(
                    "post_execute",
                    state=self._state,
                    success=self._state.status == "completed",
                )

        return self._state

    async def run_stream(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[LoopStep]:
        """Run the agent loop with streaming output.

        Args:
            task: The task/query to execute.
            context: Optional additional context.

        Yields:
            LoopStep objects as they are produced.
        """
        from agents_framework.llm import Message, MessageRole

        # Initialize state
        self._state = LoopState(status="running", started_at=datetime.now())
        self._cancel_requested = False
        self._messages = []

        # Build initial messages
        if self.system_prompt:
            self._messages.append(
                Message(role=MessageRole.SYSTEM, content=self._build_system_prompt())
            )

        self._messages.append(Message(role=MessageRole.USER, content=task))

        consecutive_errors = 0

        try:
            while self._should_continue():
                self._state.iteration += 1
                step_start = time.monotonic()

                try:
                    # Generate LLM response
                    step = await self._execute_step()
                    step.duration = time.monotonic() - step_start
                    self._state.steps.append(step)
                    yield step

                    if step.step_type == StepType.FINAL:
                        self._state.termination_reason = TerminationReason.FINAL_ANSWER
                        break

                    # Execute tools if any
                    if step.tool_calls:
                        observation_step = await self._execute_tools(step.tool_calls)
                        observation_step.duration = (
                            time.monotonic() - step_start - step.duration
                        )
                        self._state.steps.append(observation_step)
                        yield observation_step
                        consecutive_errors = 0
                    else:
                        if not step.content.strip():
                            self._state.termination_reason = (
                                TerminationReason.NO_TOOL_CALLS
                            )
                            break

                except asyncio.TimeoutError:
                    self._state.termination_reason = TerminationReason.TIMEOUT
                    self._state.error = "Execution timed out"
                    error_step = LoopStep(
                        step_type=StepType.ERROR,
                        content="Execution timed out",
                    )
                    yield error_step
                    break

                except Exception as e:
                    consecutive_errors += 1
                    logger.error(f"Error in iteration {self._state.iteration}: {e}")

                    error_step = LoopStep(
                        step_type=StepType.ERROR,
                        content=str(e),
                        duration=time.monotonic() - step_start,
                    )
                    self._state.steps.append(error_step)
                    yield error_step

                    if (
                        not self.config.continue_on_error
                        or consecutive_errors >= self.config.max_consecutive_errors
                    ):
                        self._state.termination_reason = TerminationReason.ERROR
                        self._state.error = str(e)
                        break

            # Set termination reason if not set
            if self._state.termination_reason is None:
                if self._cancel_requested:
                    self._state.termination_reason = TerminationReason.CANCELLED
                elif self._state.iteration >= self.config.max_iterations:
                    self._state.termination_reason = TerminationReason.MAX_ITERATIONS
                else:
                    self._state.termination_reason = TerminationReason.COMPLETED

            self._state.status = (
                "completed"
                if self._state.termination_reason
                in (
                    TerminationReason.COMPLETED,
                    TerminationReason.FINAL_ANSWER,
                    TerminationReason.NO_TOOL_CALLS,
                )
                else "failed"
            )

        except Exception as e:
            self._state.status = "failed"
            self._state.termination_reason = TerminationReason.ERROR
            self._state.error = str(e)
            logger.exception("Unhandled error in agent loop stream")

        finally:
            self._state.finished_at = datetime.now()

    def cancel(self) -> None:
        """Request cancellation of the running loop."""
        self._cancel_requested = True
        logger.info(f"Cancellation requested for loop {self._state.id if self._state else 'unknown'}")

    def _should_continue(self) -> bool:
        """Check if the loop should continue executing."""
        if self._cancel_requested:
            return False
        if self._state is None:
            return False
        if self._state.iteration >= self.config.max_iterations:
            return False
        if self._state.started_at:
            elapsed = (datetime.now() - self._state.started_at).total_seconds()
            if elapsed >= self.config.timeout:
                return False
        return True

    def _build_system_prompt(self) -> str:
        """Build the system prompt with tool information."""
        prompt = self.system_prompt

        if self.tool_registry:
            tools = self.tool_registry.list_tools()
            if tools:
                tool_descriptions = "\n".join(
                    f"- {t.name}: {t.description}" for t in tools
                )
                prompt += f"\n\nAvailable tools:\n{tool_descriptions}"

        return prompt

    async def _execute_step(self) -> LoopStep:
        """Execute a single step of the loop (Thought + Action).

        Returns:
            LoopStep with the result.
        """
        from agents_framework.llm import Message, MessageRole

        # Get tool definitions if available
        tools = None
        if self.tool_registry:
            tools = self.tool_registry.to_definitions()

        # Generate LLM response
        response = await self.llm.generate(
            messages=self._messages,
            tools=tools,
        )

        # Track token usage
        if response.usage:
            self._state.total_tokens += response.usage.get("total_tokens", 0)

        # Add assistant message to history
        self._messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=response.content,
                tool_calls=response.tool_calls,
            )
        )

        # Determine step type
        if response.has_tool_calls:
            # Action step with tool calls
            tool_call_data = [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
                for tc in response.tool_calls
            ]
            return LoopStep(
                step_type=StepType.ACTION,
                content=response.content,
                tool_calls=tool_call_data,
            )
        elif response.content.startswith(self.config.final_answer_prefix):
            # Final answer
            answer = response.content[len(self.config.final_answer_prefix) :].strip()
            return LoopStep(
                step_type=StepType.FINAL,
                content=answer,
            )
        else:
            # Thought step (response without tool calls might be final)
            return LoopStep(
                step_type=StepType.THOUGHT,
                content=response.content,
            )

    async def _execute_tools(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> LoopStep:
        """Execute tool calls and return observation step.

        Args:
            tool_calls: List of tool calls to execute.

        Returns:
            LoopStep with observations.
        """
        from agents_framework.llm import Message, MessageRole

        results = []

        for call in tool_calls:
            tool_name = call["name"]
            tool_id = call["id"]
            arguments = call.get("arguments", {})

            try:
                if self.tool_executor:
                    # Use executor for managed execution
                    exec_result = await self.tool_executor.execute(
                        tool_name=tool_name,
                        tool_call_id=tool_id,
                        arguments=arguments,
                    )
                    result = {
                        "tool_call_id": tool_id,
                        "name": tool_name,
                        "success": exec_result.result.success,
                        "output": (
                            exec_result.result.output
                            if exec_result.result.success
                            else exec_result.result.error
                        ),
                    }
                elif self.tool_registry:
                    # Direct registry execution
                    tool_result = await self.tool_registry.execute(
                        tool_name, **arguments
                    )
                    result = {
                        "tool_call_id": tool_id,
                        "name": tool_name,
                        "success": tool_result.success,
                        "output": (
                            tool_result.output if tool_result.success else tool_result.error
                        ),
                    }
                else:
                    result = {
                        "tool_call_id": tool_id,
                        "name": tool_name,
                        "success": False,
                        "output": "No tool registry or executor available",
                    }

            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                result = {
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "success": False,
                    "output": str(e),
                }

            results.append(result)

            # Add tool result message to history
            output_str = (
                str(result["output"])
                if result["output"] is not None
                else "No output"
            )
            self._messages.append(
                Message(
                    role=MessageRole.TOOL,
                    content=output_str,
                    tool_call_id=tool_id,
                )
            )

        # Build observation content
        observations = []
        for r in results:
            status = "Success" if r["success"] else "Error"
            observations.append(f"[{r['name']}] {status}: {r['output']}")

        return LoopStep(
            step_type=StepType.OBSERVATION,
            content="\n".join(observations),
            tool_results=results,
        )


@dataclass
class RunnerConfig:
    """Configuration for the AgentRunner.

    Attributes:
        max_concurrent: Maximum concurrent agent executions.
        default_timeout: Default timeout for executions.
        enable_metrics: Whether to collect execution metrics.
    """

    max_concurrent: int = 10
    default_timeout: float = 300.0
    enable_metrics: bool = True


class AgentRunner:
    """Manager for agent execution with lifecycle management.

    Provides a high-level interface for running agents with:
    - Concurrent execution management
    - Hook system integration
    - Error handling and recovery
    - Execution tracking

    Example:
        runner = AgentRunner(hooks=my_hooks)
        result = await runner.run(agent, "What is the weather?")

        # Track running executions
        print(f"Active: {len(runner.active_executions)}")
    """

    def __init__(
        self,
        hooks: Optional[HookRegistry] = None,
        config: Optional[RunnerConfig] = None,
    ):
        """Initialize the runner.

        Args:
            hooks: Hook registry for lifecycle events.
            config: Runner configuration.
        """
        self.hooks = hooks
        self.config = config or RunnerConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._active: Dict[str, LoopState] = {}
        self._completed: List[LoopState] = []

    @property
    def active_executions(self) -> Dict[str, LoopState]:
        """Get currently active executions."""
        return dict(self._active)

    @property
    def completed_executions(self) -> List[LoopState]:
        """Get completed executions."""
        return list(self._completed)

    async def run(
        self,
        agent: BaseAgent,
        task: Union[str, Task],
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> TaskResult:
        """Run an agent on a task.

        Args:
            agent: The agent to run.
            task: Task string or Task object.
            context: Optional execution context.
            timeout: Optional timeout override.

        Returns:
            TaskResult from the agent execution.
        """
        from agents_framework.agents import Task, TaskResult

        # Convert string to Task if needed
        if isinstance(task, str):
            task_obj = Task(description=task, context=context or {})
        else:
            task_obj = task
            if context:
                task_obj.context.update(context)

        execution_id = str(uuid.uuid4())

        async with self._semaphore:
            # Fire pre-run hook
            if self.hooks:
                await self.hooks.fire(
                    "pre_run",
                    agent=agent,
                    task=task_obj,
                    execution_id=execution_id,
                )

            try:
                # Execute with timeout
                effective_timeout = timeout or self.config.default_timeout

                async with asyncio.timeout(effective_timeout):
                    result = await agent.execute(task_obj)

            except asyncio.TimeoutError:
                result = TaskResult(
                    task_id=task_obj.id,
                    success=False,
                    error=f"Execution timed out after {effective_timeout}s",
                )

            except Exception as e:
                logger.exception(f"Error running agent {agent.id}")
                result = TaskResult(
                    task_id=task_obj.id,
                    success=False,
                    error=str(e),
                )

                # Fire error hook
                if self.hooks:
                    await self.hooks.fire(
                        "on_error",
                        agent=agent,
                        task=task_obj,
                        error=e,
                        execution_id=execution_id,
                    )

            finally:
                # Fire post-run hook
                if self.hooks:
                    await self.hooks.fire(
                        "post_run",
                        agent=agent,
                        task=task_obj,
                        result=result,
                        execution_id=execution_id,
                    )

        return result

    async def run_many(
        self,
        agent: BaseAgent,
        tasks: List[Union[str, Task]],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[TaskResult]:
        """Run an agent on multiple tasks concurrently.

        Args:
            agent: The agent to run.
            tasks: List of tasks to execute.
            context: Optional shared context.

        Returns:
            List of TaskResult in the same order as tasks.
        """
        coros = [self.run(agent, task, context) for task in tasks]
        return await asyncio.gather(*coros)

    async def run_with_loop(
        self,
        llm: LLMProvider,
        task: str,
        tool_registry: Optional[ToolRegistry] = None,
        tool_executor: Optional[ToolExecutor] = None,
        loop_config: Optional[LoopConfig] = None,
        system_prompt: str = "",
    ) -> LoopState:
        """Run a task using AgentLoop directly.

        Convenience method for running without a full agent.

        Args:
            llm: LLM provider.
            task: Task to execute.
            tool_registry: Optional tool registry.
            tool_executor: Optional tool executor.
            loop_config: Optional loop configuration.
            system_prompt: Optional system prompt.

        Returns:
            LoopState with execution results.
        """
        loop = AgentLoop(
            llm=llm,
            tool_registry=tool_registry,
            tool_executor=tool_executor,
            config=loop_config,
            hooks=self.hooks,
            system_prompt=system_prompt,
        )

        execution_id = loop._state.id if loop._state else str(uuid.uuid4())

        async with self._semaphore:
            state = await loop.run(task)
            self._active.pop(execution_id, None)
            self._completed.append(state)
            return state

    def cancel(self, execution_id: str) -> bool:
        """Cancel a running execution.

        Args:
            execution_id: ID of the execution to cancel.

        Returns:
            True if cancellation was requested, False if not found.
        """
        if execution_id in self._active:
            # In a real implementation, we would track the loop instance
            # and call loop.cancel()
            logger.info(f"Cancellation requested for {execution_id}")
            return True
        return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics.

        Returns:
            Dictionary with metrics.
        """
        if not self.config.enable_metrics:
            return {}

        total_executions = len(self._completed)
        successful = sum(1 for s in self._completed if s.status == "completed")
        failed = sum(1 for s in self._completed if s.status == "failed")

        total_duration = sum(s.duration for s in self._completed)
        avg_duration = total_duration / total_executions if total_executions > 0 else 0

        total_iterations = sum(s.iteration for s in self._completed)
        avg_iterations = (
            total_iterations / total_executions if total_executions > 0 else 0
        )

        return {
            "total_executions": total_executions,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total_executions if total_executions > 0 else 0,
            "avg_duration_seconds": avg_duration,
            "avg_iterations": avg_iterations,
            "total_tokens": sum(s.total_tokens for s in self._completed),
            "active_count": len(self._active),
        }
