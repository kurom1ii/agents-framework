"""Base agent implementation."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agents_framework.llm import LLMProvider


class AgentStatus(str, Enum):
    """Status of an agent."""

    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class AgentRole:
    """Role definition for an agent."""

    name: str
    description: str
    capabilities: list[str] = field(default_factory=list)
    instructions: str = ""

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AgentRole):
            return self.name == other.name
        return False


@dataclass
class Task:
    """A task to be executed by an agent."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    required_capabilities: list[str] = field(default_factory=list)
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class TaskResult:
    """Result of a task execution."""

    task_id: str
    success: bool
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.now)


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    name: str = "Agent"
    max_iterations: int = 10
    timeout: float = 300.0
    tools: list[str] = field(default_factory=list)
    memory_enabled: bool = True


class BaseAgent(ABC):
    """Base class for all agents."""

    def __init__(
        self,
        role: AgentRole,
        llm: LLMProvider | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.id = str(uuid.uuid4())
        self.role = role
        self.llm = llm
        self.config = config or AgentConfig()
        self._status = AgentStatus.IDLE
        self._parent_id: str | None = None

    @property
    def status(self) -> AgentStatus:
        """Get current agent status."""
        return self._status

    @status.setter
    def status(self, value: AgentStatus) -> None:
        """Set agent status."""
        self._status = value

    @property
    def parent_id(self) -> str | None:
        """Get parent agent ID if any."""
        return self._parent_id

    @parent_id.setter
    def parent_id(self, value: str | None) -> None:
        """Set parent agent ID."""
        self._parent_id = value

    @abstractmethod
    async def run(self, task: str | Task) -> TaskResult:
        """Execute a task."""
        ...

    async def execute(self, task: Task) -> TaskResult:
        """Execute a task with status management."""
        self._status = AgentStatus.BUSY
        try:
            result = await self.run(task)
            self._status = AgentStatus.IDLE
            return result
        except Exception as e:
            self._status = AgentStatus.ERROR
            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e),
            )

    def has_capability(self, capability: str) -> bool:
        """Check if agent has a specific capability."""
        return capability in self.role.capabilities

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id!r}, role={self.role.name!r})"
