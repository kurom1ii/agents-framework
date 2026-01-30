"""Agent implementations for the agents framework.

This package provides the agent system including base agents,
worker agents, supervisor agents, and routing capabilities.

Example:
    from agents_framework.agents import BaseAgent, AgentRole, Task

    class MyAgent(BaseAgent):
        async def run(self, task: Task) -> TaskResult:
            # Agent implementation
            pass

    role = AgentRole(
        name="researcher",
        description="Research agent for gathering information",
        capabilities=["search", "summarize"],
    )
    agent = MyAgent(role=role)
"""

from .base import (
    AgentConfig,
    AgentRole,
    AgentStatus,
    BaseAgent,
    Task,
    TaskResult,
)

__all__ = [
    # Base
    "AgentConfig",
    "AgentRole",
    "AgentStatus",
    "BaseAgent",
    "Task",
    "TaskResult",
]
