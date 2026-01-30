"""Agent implementations for the agents framework."""

from .base import AgentConfig, AgentRole, AgentStatus, BaseAgent, Task, TaskResult
from .router import RouterAgent
from .spawner import AgentSpawner, AgentTemplate
from .supervisor import AggregatedResult, DelegationStrategy, SupervisorAgent

__all__ = [
    # Base
    "AgentConfig",
    "AgentRole",
    "AgentStatus",
    "BaseAgent",
    "Task",
    "TaskResult",
    # Supervisor
    "SupervisorAgent",
    "AggregatedResult",
    "DelegationStrategy",
    # Router
    "RouterAgent",
    # Spawner
    "AgentSpawner",
    "AgentTemplate",
]
