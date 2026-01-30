"""Base team pattern protocol and shared types.

This module defines the TeamPattern protocol that all team patterns
must implement, along with shared types for pattern execution.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)
import uuid

if TYPE_CHECKING:
    from agents_framework.agents.base import BaseAgent, Task, TaskResult


class PatternStatus(str, Enum):
    """Status of a pattern execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PatternContext:
    """Context passed between agents in a pattern.

    Attributes:
        id: Unique identifier for this context.
        variables: Shared variables accessible by all agents.
        history: List of previous step results.
        metadata: Additional metadata for the pattern execution.
        created_at: When this context was created.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    variables: Dict[str, Any] = field(default_factory=dict)
    history: List[StepResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a variable from context."""
        return self.variables.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a variable in context."""
        self.variables[key] = value

    def update(self, variables: Dict[str, Any]) -> None:
        """Update multiple variables at once."""
        self.variables.update(variables)


@dataclass
class StepResult:
    """Result from a single step in a pattern.

    Attributes:
        step_id: Unique identifier for this step.
        agent_id: ID of the agent that executed this step.
        agent_name: Name of the agent.
        success: Whether the step completed successfully.
        output: Output data from the step.
        error: Error message if the step failed.
        duration_ms: Duration of the step in milliseconds.
        metadata: Additional metadata about the step execution.
        timestamp: When this step completed.
    """

    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    agent_name: str = ""
    success: bool = True
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PatternResult:
    """Result of a complete pattern execution.

    Attributes:
        pattern_name: Name of the pattern that was executed.
        status: Final status of the pattern.
        final_output: The final output from the pattern.
        steps: List of all step results.
        context: The final context after execution.
        total_duration_ms: Total duration of the pattern in milliseconds.
        error: Error message if the pattern failed.
        metadata: Additional metadata about the execution.
    """

    pattern_name: str
    status: PatternStatus = PatternStatus.COMPLETED
    final_output: Any = None
    steps: List[StepResult] = field(default_factory=list)
    context: Optional[PatternContext] = None
    total_duration_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if the pattern completed successfully."""
        return self.status == PatternStatus.COMPLETED


@runtime_checkable
class TeamPattern(Protocol):
    """Protocol defining the interface for team patterns.

    All team patterns (hierarchical, sequential, swarm, etc.) must
    implement this interface. The execute method orchestrates the
    execution of agents according to the pattern's logic.
    """

    @property
    def name(self) -> str:
        """Name of this pattern."""
        ...

    @abstractmethod
    async def execute(
        self,
        task: Task,
        agents: List[BaseAgent],
        context: Optional[PatternContext] = None,
    ) -> PatternResult:
        """Execute the pattern with the given agents.

        Args:
            task: The task to execute.
            agents: List of agents participating in the pattern.
            context: Optional initial context for the execution.

        Returns:
            PatternResult with the outcome of the execution.
        """
        ...

    async def validate_agents(self, agents: List[BaseAgent]) -> bool:
        """Validate that the provided agents are suitable for this pattern.

        Args:
            agents: List of agents to validate.

        Returns:
            True if agents are valid for this pattern.
        """
        ...


class BasePattern:
    """Base class for team patterns with common functionality.

    Provides default implementations for pattern lifecycle management
    and agent validation. Subclasses should override execute() to
    implement their specific orchestration logic.
    """

    def __init__(self, name: str = "base"):
        self._name = name
        self._status = PatternStatus.PENDING

    @property
    def name(self) -> str:
        """Name of this pattern."""
        return self._name

    @property
    def status(self) -> PatternStatus:
        """Current status of this pattern."""
        return self._status

    async def validate_agents(self, agents: List[BaseAgent]) -> bool:
        """Default agent validation - requires at least one agent."""
        return len(agents) > 0

    async def execute(
        self,
        task: Task,
        agents: List[BaseAgent],
        context: Optional[PatternContext] = None,
    ) -> PatternResult:
        """Execute the pattern. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement execute()")

    def _create_context(
        self,
        initial_context: Optional[PatternContext] = None
    ) -> PatternContext:
        """Create or return the execution context."""
        if initial_context:
            return initial_context
        return PatternContext()

    def _create_step_result(
        self,
        agent: BaseAgent,
        output: Any,
        success: bool = True,
        error: Optional[str] = None,
        duration_ms: float = 0.0,
    ) -> StepResult:
        """Create a step result for an agent execution."""
        return StepResult(
            agent_id=agent.id,
            agent_name=agent.role.name,
            success=success,
            output=output,
            error=error,
            duration_ms=duration_ms,
        )
