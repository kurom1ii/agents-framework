"""Agent implementations for the agents framework.

This package provides the agent system including base agents,
worker agents, supervisor agents, spawners, and routing capabilities.

Example:
    from agents_framework.agents import BaseAgent, AgentRole, Task, RouterAgent

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

    # Using WorkerAgent for focused task execution
    from agents_framework.agents import WorkerAgent, WorkerConfig

    config = WorkerConfig(
        allowed_tools=["search", "calculator"],
        report_progress=True,
    )
    worker = WorkerAgent(role=role, config=config)
    result = await worker.run("Calculate 2+2")

    # Using RouterAgent for task routing
    router = RouterAgent()
    router.register_agent(worker, capabilities=["calculate", "search"])

    # Using SupervisorAgent for task delegation
    from agents_framework.agents import SupervisorAgent, SupervisorConfig

    supervisor = SupervisorAgent(
        role=AgentRole(name="supervisor", description="Coordinates tasks"),
        config=SupervisorConfig(
            delegation_strategy=DelegationStrategy.CAPABILITY_MATCH,
            execution_mode=ExecutionMode.PARALLEL,
        ),
    )
    supervisor.add_worker(worker)

    # Using AgentSpawner for dynamic agent creation
    from agents_framework.agents import AgentSpawner, AgentTemplate

    spawner = AgentSpawner()
    spawner.register_template(AgentTemplate(
        name="worker",
        role=role,
        agent_class=WorkerAgent,
    ))
    agent = await spawner.spawn("worker")
"""

from .base import (
    AgentConfig,
    AgentRole,
    AgentStatus,
    BaseAgent,
    Task,
    TaskResult,
)
from .worker import (
    WorkerAgent,
    WorkerConfig,
    WorkerProgress,
    WorkerStatus,
    ProgressCallback,
)
from .router import (
    RouterAgent,
    RouterConfig,
    RoutingDecision,
    RoutingRule,
    RoutingStrategy,
)
from .supervisor import (
    DelegatedTask,
    DelegationStrategy,
    ExecutionMode,
    ResultAggregator,
    SupervisorAgent,
    SupervisorConfig,
    WorkerSelector,
)
from .spawner import (
    AgentLifecycleState,
    AgentSpawner,
    AgentTemplate,
    SpawnedAgentInfo,
    SpawnPolicy,
)

__all__ = [
    # Base
    "AgentConfig",
    "AgentRole",
    "AgentStatus",
    "BaseAgent",
    "Task",
    "TaskResult",
    # Worker
    "WorkerAgent",
    "WorkerConfig",
    "WorkerProgress",
    "WorkerStatus",
    "ProgressCallback",
    # Router
    "RouterAgent",
    "RouterConfig",
    "RoutingDecision",
    "RoutingRule",
    "RoutingStrategy",
    # Supervisor
    "DelegatedTask",
    "DelegationStrategy",
    "ExecutionMode",
    "ResultAggregator",
    "SupervisorAgent",
    "SupervisorConfig",
    "WorkerSelector",
    # Spawner
    "AgentLifecycleState",
    "AgentSpawner",
    "AgentTemplate",
    "SpawnedAgentInfo",
    "SpawnPolicy",
]
