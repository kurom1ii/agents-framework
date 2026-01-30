"""Team orchestration for multi-agent collaboration.

This module provides the Team class for orchestrating multiple agents
to work together on complex tasks, with shared context, team-level
memory, and configurable execution strategies.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from agents_framework.memory import MemoryStore

from agents_framework.agents import AgentConfig, AgentRole, AgentStatus, BaseAgent, Task, TaskResult
from .registry import AgentRegistry
from .router import AgentMessage, MessageRouter, RoutingStrategy


class TeamState(str, Enum):
    """State of a team."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class TeamExecutionStrategy(str, Enum):
    """Strategy for executing team tasks."""

    COLLABORATIVE = "collaborative"  # Agents work together on the same task
    DIVIDE_CONQUER = "divide_conquer"  # Split task among agents
    HIERARCHICAL = "hierarchical"  # Supervisor delegates to workers
    ROUND_ROBIN = "round_robin"  # Rotate task execution among agents
    BROADCAST = "broadcast"  # All agents process the task independently


@dataclass
class TeamConfig:
    """Configuration for a team.

    Attributes:
        name: Team name.
        strategy: Execution strategy for the team.
        max_concurrent_tasks: Maximum concurrent tasks.
        task_timeout: Default timeout for tasks.
        enable_shared_memory: Whether to enable shared team memory.
        enable_messaging: Whether to enable inter-agent messaging.
        auto_cleanup: Whether to clean up on stop.
    """

    name: str = "Team"
    strategy: TeamExecutionStrategy = TeamExecutionStrategy.COLLABORATIVE
    max_concurrent_tasks: int = 10
    task_timeout: float = 300.0
    enable_shared_memory: bool = True
    enable_messaging: bool = True
    auto_cleanup: bool = True


@dataclass
class SharedContext:
    """Shared context for team members.

    Provides a shared data store that all team members can read and write to.

    Attributes:
        data: Dictionary of shared data.
        metadata: Metadata about the context.
        created_at: When the context was created.
        updated_at: When the context was last updated.
    """

    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value from shared context.

        Args:
            key: The key to retrieve.
            default: Default value if key not found.

        Returns:
            The value or default.
        """
        async with self._lock:
            return self.data.get(key, default)

    async def set(self, key: str, value: Any) -> None:
        """Set a value in shared context.

        Args:
            key: The key to set.
            value: The value to store.
        """
        async with self._lock:
            self.data[key] = value
            self.updated_at = datetime.now()

    async def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple values.

        Args:
            updates: Dictionary of key-value pairs to update.
        """
        async with self._lock:
            self.data.update(updates)
            self.updated_at = datetime.now()

    async def delete(self, key: str) -> bool:
        """Delete a value from shared context.

        Args:
            key: The key to delete.

        Returns:
            True if the key was deleted.
        """
        async with self._lock:
            if key in self.data:
                del self.data[key]
                self.updated_at = datetime.now()
                return True
            return False

    async def clear(self) -> None:
        """Clear all shared data."""
        async with self._lock:
            self.data.clear()
            self.updated_at = datetime.now()

    async def keys(self) -> List[str]:
        """Get all keys in shared context.

        Returns:
            List of keys.
        """
        async with self._lock:
            return list(self.data.keys())


@dataclass
class TeamTaskResult:
    """Result of a team task execution.

    Attributes:
        task_id: ID of the executed task.
        success: Whether the task succeeded.
        results: Results from individual agents.
        aggregated_output: Combined output from all agents.
        errors: List of errors that occurred.
        metadata: Additional result metadata.
        started_at: When execution started.
        completed_at: When execution completed.
    """

    task_id: str
    success: bool
    results: List[TaskResult] = field(default_factory=list)
    aggregated_output: Any = None
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


TaskDivider = Callable[[Task, List[BaseAgent]], List[tuple[Task, BaseAgent]]]
ResultMerger = Callable[[List[TaskResult]], Any]


class Team:
    """Multi-agent team for collaborative task execution.

    Teams orchestrate multiple agents working together, providing
    shared context, inter-agent communication, and configurable
    execution strategies.

    Example:
        from agents_framework.teams import Team, TeamConfig

        # Create a team
        team = Team(TeamConfig(
            name="research_team",
            strategy=TeamExecutionStrategy.COLLABORATIVE,
        ))

        # Add agents
        team.add_member(researcher_agent)
        team.add_member(analyst_agent)
        team.add_member(writer_agent)

        # Start the team
        await team.start()

        # Execute a task
        result = await team.run(task)

        # Stop the team
        await team.stop()
    """

    def __init__(
        self,
        config: Optional[TeamConfig] = None,
        memory: Optional["MemoryStore"] = None,
    ):
        """Initialize the team.

        Args:
            config: Team configuration.
            memory: Optional shared memory store.
        """
        self.id = str(uuid.uuid4())
        self.config = config or TeamConfig()
        self._state = TeamState.IDLE
        self._members: Dict[str, BaseAgent] = {}
        self._registry = AgentRegistry()
        self._router = MessageRouter() if config and config.enable_messaging else None
        self._shared_context = SharedContext()
        self._memory = memory
        self._task_history: List[TeamTaskResult] = []
        self._custom_divider: Optional[TaskDivider] = None
        self._custom_merger: Optional[ResultMerger] = None
        self._leader_id: Optional[str] = None
        self._lifecycle_callbacks: List[Callable[[TeamState], Any]] = []
        self._created_at = datetime.now()
        self._started_at: Optional[datetime] = None
        self._stopped_at: Optional[datetime] = None

    @property
    def name(self) -> str:
        """Get team name."""
        return self.config.name

    @property
    def state(self) -> TeamState:
        """Get current team state."""
        return self._state

    @property
    def members(self) -> List[BaseAgent]:
        """Get list of team members."""
        return list(self._members.values())

    @property
    def shared_context(self) -> SharedContext:
        """Get shared context."""
        return self._shared_context

    @property
    def memory(self) -> Optional["MemoryStore"]:
        """Get team memory store."""
        return self._memory

    @property
    def router(self) -> Optional[MessageRouter]:
        """Get message router."""
        return self._router

    @property
    def registry(self) -> AgentRegistry:
        """Get agent registry."""
        return self._registry

    def add_member(
        self,
        agent: BaseAgent,
        is_leader: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an agent to the team.

        Args:
            agent: The agent to add.
            is_leader: Whether this agent is the team leader.
            metadata: Optional agent metadata.
        """
        self._members[agent.id] = agent
        self._registry.register(agent, metadata=metadata)

        if self._router:
            self._router.register_agent(agent.id)

        if is_leader:
            self._leader_id = agent.id

    def remove_member(self, agent_id: str) -> Optional[BaseAgent]:
        """Remove an agent from the team.

        Args:
            agent_id: ID of the agent to remove.

        Returns:
            The removed agent if found.
        """
        agent = self._members.pop(agent_id, None)

        if agent:
            self._registry.unregister(agent_id)

            if self._router:
                self._router.unregister_agent(agent_id)

            if self._leader_id == agent_id:
                self._leader_id = None

        return agent

    def get_member(self, agent_id: str) -> Optional[BaseAgent]:
        """Get a team member by ID.

        Args:
            agent_id: The agent's ID.

        Returns:
            The agent if found.
        """
        return self._members.get(agent_id)

    def get_leader(self) -> Optional[BaseAgent]:
        """Get the team leader.

        Returns:
            The leader agent if set.
        """
        if self._leader_id:
            return self._members.get(self._leader_id)
        return None

    def set_leader(self, agent_id: str) -> bool:
        """Set the team leader.

        Args:
            agent_id: ID of the agent to make leader.

        Returns:
            True if the leader was set.
        """
        if agent_id in self._members:
            self._leader_id = agent_id
            return True
        return False

    def set_task_divider(self, divider: TaskDivider) -> None:
        """Set a custom task divider function.

        Args:
            divider: Function that divides a task among agents.
        """
        self._custom_divider = divider

    def set_result_merger(self, merger: ResultMerger) -> None:
        """Set a custom result merger function.

        Args:
            merger: Function that merges results from agents.
        """
        self._custom_merger = merger

    async def start(self) -> None:
        """Start the team.

        Initializes team resources and prepares members for task execution.
        """
        if self._state == TeamState.RUNNING:
            return

        self._state = TeamState.RUNNING
        self._started_at = datetime.now()
        self._stopped_at = None

        # Initialize shared context
        await self._shared_context.set("team_id", self.id)
        await self._shared_context.set("team_name", self.config.name)
        await self._shared_context.set("started_at", self._started_at.isoformat())

        await self._notify_state_change(TeamState.RUNNING)

    async def stop(self) -> None:
        """Stop the team.

        Cleans up team resources and stops all members.
        """
        if self._state == TeamState.STOPPED:
            return

        self._state = TeamState.STOPPED
        self._stopped_at = datetime.now()

        if self.config.auto_cleanup:
            await self._cleanup()

        await self._notify_state_change(TeamState.STOPPED)

    async def pause(self) -> None:
        """Pause team execution."""
        if self._state == TeamState.RUNNING:
            self._state = TeamState.PAUSED
            await self._notify_state_change(TeamState.PAUSED)

    async def resume(self) -> None:
        """Resume team execution."""
        if self._state == TeamState.PAUSED:
            self._state = TeamState.RUNNING
            await self._notify_state_change(TeamState.RUNNING)

    async def run(self, task: str | Task) -> TeamTaskResult:
        """Execute a task with the team.

        Args:
            task: The task to execute (string or Task object).

        Returns:
            TeamTaskResult with execution results.
        """
        if isinstance(task, str):
            task = Task(description=task)

        if self._state != TeamState.RUNNING:
            await self.start()

        started_at = datetime.now()

        try:
            strategy = self.config.strategy

            if strategy == TeamExecutionStrategy.COLLABORATIVE:
                results = await self._execute_collaborative(task)
            elif strategy == TeamExecutionStrategy.DIVIDE_CONQUER:
                results = await self._execute_divide_conquer(task)
            elif strategy == TeamExecutionStrategy.HIERARCHICAL:
                results = await self._execute_hierarchical(task)
            elif strategy == TeamExecutionStrategy.ROUND_ROBIN:
                results = await self._execute_round_robin(task)
            elif strategy == TeamExecutionStrategy.BROADCAST:
                results = await self._execute_broadcast(task)
            else:
                results = await self._execute_collaborative(task)

            # Aggregate results
            aggregated = await self._aggregate_results(results)
            success = all(r.success for r in results)

            team_result = TeamTaskResult(
                task_id=task.id,
                success=success,
                results=results,
                aggregated_output=aggregated,
                errors=[r.error for r in results if r.error],
                started_at=started_at,
                completed_at=datetime.now(),
                metadata={
                    "strategy": strategy.value,
                    "member_count": len(self._members),
                },
            )

        except Exception as e:
            team_result = TeamTaskResult(
                task_id=task.id,
                success=False,
                errors=[str(e)],
                started_at=started_at,
                completed_at=datetime.now(),
            )

        self._task_history.append(team_result)
        return team_result

    async def _execute_collaborative(self, task: Task) -> List[TaskResult]:
        """Execute task collaboratively - all agents work on the same task.

        In this mode, agents can communicate and share context while
        working on the task.
        """
        results: List[TaskResult] = []

        # Share task in context
        await self._shared_context.set("current_task", task.description)

        # Execute with all agents (they share context)
        semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)

        async def execute_member(agent: BaseAgent) -> TaskResult:
            async with semaphore:
                # Inject shared context into task
                task_copy = Task(
                    id=task.id,
                    description=task.description,
                    context={
                        **task.context,
                        "shared_context": await self._get_context_snapshot(),
                        "team_members": [m.role.name for m in self.members],
                    },
                    required_capabilities=task.required_capabilities,
                    priority=task.priority,
                )

                try:
                    result = await asyncio.wait_for(
                        agent.execute(task_copy),
                        timeout=self.config.task_timeout,
                    )
                    return result
                except asyncio.TimeoutError:
                    return TaskResult(
                        task_id=task.id,
                        success=False,
                        error=f"Task timed out for agent {agent.id}",
                    )

        results = await asyncio.gather(
            *[execute_member(agent) for agent in self.members],
            return_exceptions=True,
        )

        # Convert exceptions to failed results
        processed: List[TaskResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed.append(TaskResult(
                    task_id=task.id,
                    success=False,
                    error=str(result),
                ))
            else:
                processed.append(result)

        return processed

    async def _execute_divide_conquer(self, task: Task) -> List[TaskResult]:
        """Execute task by dividing it among agents."""
        members = list(self._members.values())

        if self._custom_divider:
            # Use custom divider
            assignments = self._custom_divider(task, members)
        else:
            # Default: create subtasks for each agent
            assignments = [
                (
                    Task(
                        description=f"{task.description} (part {i+1}/{len(members)})",
                        context={
                            **task.context,
                            "part_number": i + 1,
                            "total_parts": len(members),
                        },
                        required_capabilities=task.required_capabilities,
                        priority=task.priority,
                    ),
                    agent,
                )
                for i, agent in enumerate(members)
            ]

        semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)

        async def execute_assignment(
            subtask: Task, agent: BaseAgent
        ) -> TaskResult:
            async with semaphore:
                try:
                    return await asyncio.wait_for(
                        agent.execute(subtask),
                        timeout=self.config.task_timeout,
                    )
                except asyncio.TimeoutError:
                    return TaskResult(
                        task_id=subtask.id,
                        success=False,
                        error=f"Task timed out for agent {agent.id}",
                    )

        results = await asyncio.gather(
            *[execute_assignment(t, a) for t, a in assignments],
            return_exceptions=True,
        )

        # Convert exceptions
        processed: List[TaskResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed.append(TaskResult(
                    task_id=assignments[i][0].id,
                    success=False,
                    error=str(result),
                ))
            else:
                processed.append(result)

        return processed

    async def _execute_hierarchical(self, task: Task) -> List[TaskResult]:
        """Execute task hierarchically through the leader."""
        leader = self.get_leader()
        if not leader:
            # No leader, fall back to first member or fail
            if self._members:
                leader = list(self._members.values())[0]
            else:
                return [TaskResult(
                    task_id=task.id,
                    success=False,
                    error="No team members available",
                )]

        # Leader executes and may delegate to others
        try:
            result = await asyncio.wait_for(
                leader.execute(task),
                timeout=self.config.task_timeout,
            )
            return [result]
        except asyncio.TimeoutError:
            return [TaskResult(
                task_id=task.id,
                success=False,
                error="Leader task timed out",
            )]

    async def _execute_round_robin(self, task: Task) -> List[TaskResult]:
        """Execute task with rotating agent selection."""
        if not self._members:
            return [TaskResult(
                task_id=task.id,
                success=False,
                error="No team members available",
            )]

        # Select next agent in rotation
        members = list(self._members.values())
        agent_index = len(self._task_history) % len(members)
        agent = members[agent_index]

        try:
            result = await asyncio.wait_for(
                agent.execute(task),
                timeout=self.config.task_timeout,
            )
            return [result]
        except asyncio.TimeoutError:
            return [TaskResult(
                task_id=task.id,
                success=False,
                error=f"Task timed out for agent {agent.id}",
            )]

    async def _execute_broadcast(self, task: Task) -> List[TaskResult]:
        """Execute task independently on all agents."""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)

        async def execute_member(agent: BaseAgent) -> TaskResult:
            async with semaphore:
                try:
                    return await asyncio.wait_for(
                        agent.execute(task),
                        timeout=self.config.task_timeout,
                    )
                except asyncio.TimeoutError:
                    return TaskResult(
                        task_id=task.id,
                        success=False,
                        error=f"Task timed out for agent {agent.id}",
                    )

        results = await asyncio.gather(
            *[execute_member(agent) for agent in self.members],
            return_exceptions=True,
        )

        processed: List[TaskResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed.append(TaskResult(
                    task_id=task.id,
                    success=False,
                    error=str(result),
                ))
            else:
                processed.append(result)

        return processed

    async def _aggregate_results(self, results: List[TaskResult]) -> Any:
        """Aggregate results from team execution."""
        if self._custom_merger:
            try:
                return self._custom_merger(results)
            except Exception as e:
                return {"error": f"Merger failed: {e}"}

        # Default aggregation
        successful = [r.output for r in results if r.success]
        return {
            "successful_outputs": successful,
            "total_results": len(results),
            "successful_count": len(successful),
            "failed_count": len(results) - len(successful),
        }

    async def _get_context_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the shared context."""
        keys = await self._shared_context.keys()
        snapshot = {}
        for key in keys:
            snapshot[key] = await self._shared_context.get(key)
        return snapshot

    async def _cleanup(self) -> None:
        """Clean up team resources."""
        await self._shared_context.clear()
        # Clear router queues
        if self._router:
            for agent_id in list(self._members.keys()):
                await self._router.clear_agent_queue(agent_id)

    async def send_message(
        self,
        sender_id: str,
        recipient_id: Optional[str] = None,
        content: Any = None,
        topic: Optional[str] = None,
        strategy: RoutingStrategy = RoutingStrategy.DIRECT,
    ) -> List[str]:
        """Send a message between team members.

        Args:
            sender_id: ID of the sending agent.
            recipient_id: ID of recipient (for direct routing).
            content: Message content.
            topic: Topic (for topic-based routing).
            strategy: Routing strategy.

        Returns:
            List of recipient IDs that received the message.
        """
        if not self._router:
            return []

        message = AgentMessage(
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=content,
            topic=topic,
            strategy=strategy,
        )

        return await self._router.send(message)

    def on_state_change(self, callback: Callable[[TeamState], Any]) -> None:
        """Register a callback for state changes.

        Args:
            callback: Function called with the new state.
        """
        self._lifecycle_callbacks.append(callback)

    async def _notify_state_change(self, state: TeamState) -> None:
        """Notify callbacks of a state change."""
        for callback in self._lifecycle_callbacks:
            try:
                result = callback(state)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass

    def get_task_history(self) -> List[TeamTaskResult]:
        """Get history of team task executions.

        Returns:
            List of TeamTaskResult records.
        """
        return list(self._task_history)

    def get_statistics(self) -> Dict[str, Any]:
        """Get team statistics.

        Returns:
            Dictionary with team statistics.
        """
        successful_tasks = sum(1 for t in self._task_history if t.success)
        total_tasks = len(self._task_history)

        return {
            "team_id": self.id,
            "team_name": self.config.name,
            "state": self._state.value,
            "member_count": len(self._members),
            "leader_id": self._leader_id,
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0.0,
            "strategy": self.config.strategy.value,
            "created_at": self._created_at.isoformat(),
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "stopped_at": self._stopped_at.isoformat() if self._stopped_at else None,
        }

    def __len__(self) -> int:
        """Return the number of team members."""
        return len(self._members)

    def __contains__(self, agent_id: str) -> bool:
        """Check if an agent is a team member."""
        return agent_id in self._members

    def __iter__(self):
        """Iterate over team members."""
        return iter(self._members.values())

    def __repr__(self) -> str:
        return (
            f"Team(id={self.id!r}, name={self.config.name!r}, "
            f"members={len(self._members)}, state={self._state.value!r})"
        )
