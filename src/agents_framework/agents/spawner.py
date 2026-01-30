"""Agent spawner for dynamic agent creation and lifecycle management.

This module provides the AgentSpawner class which implements the factory
pattern for creating agents, managing their resources, and handling
their lifecycle.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, TypeVar

from .base import AgentConfig, AgentRole, AgentStatus, BaseAgent, Task, TaskResult

if TYPE_CHECKING:
    from agents_framework.llm import LLMProvider
    from agents_framework.teams.registry import AgentRegistry


class SpawnPolicy(str, Enum):
    """Policy for spawning new agents."""

    ON_DEMAND = "on_demand"  # Spawn when needed
    POOLED = "pooled"  # Maintain a pool of pre-spawned agents
    SINGLETON = "singleton"  # Only one instance per template


class AgentLifecycleState(str, Enum):
    """Lifecycle state of a spawned agent."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    SUSPENDED = "suspended"
    TERMINATING = "terminating"
    TERMINATED = "terminated"


@dataclass
class AgentTemplate:
    """Template for creating agents.

    Attributes:
        name: Unique template name.
        role: Role definition for agents created from this template.
        agent_class: The agent class to instantiate.
        config: Configuration for the agent.
        llm_factory: Optional factory function for creating LLM providers.
        spawn_policy: Policy for spawning agents.
        pool_size: Number of agents to maintain in pool (for POOLED policy).
        max_instances: Maximum concurrent instances (0 = unlimited).
        metadata: Additional template metadata.
    """

    name: str
    role: AgentRole
    agent_class: Type[BaseAgent]
    config: Optional[AgentConfig] = None
    llm_factory: Optional[Callable[[], "LLMProvider"]] = None
    spawn_policy: SpawnPolicy = SpawnPolicy.ON_DEMAND
    pool_size: int = 3
    max_instances: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpawnedAgentInfo:
    """Information about a spawned agent.

    Attributes:
        agent_id: Unique ID of the spawned agent.
        template_name: Name of the template used.
        lifecycle_state: Current lifecycle state.
        spawned_at: When the agent was spawned.
        last_used: When the agent was last used.
        task_count: Number of tasks executed.
        parent_spawner_id: ID of the spawner that created this agent.
    """

    agent_id: str
    template_name: str
    lifecycle_state: AgentLifecycleState = AgentLifecycleState.INITIALIZING
    spawned_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    task_count: int = 0
    parent_spawner_id: Optional[str] = None


AgentT = TypeVar("AgentT", bound=BaseAgent)


class AgentSpawner:
    """Factory for dynamically creating and managing agent instances.

    The spawner supports different spawn policies (on-demand, pooled, singleton),
    resource management, and agent lifecycle management.

    Example:
        spawner = AgentSpawner()

        # Register a template
        template = AgentTemplate(
            name="researcher",
            role=AgentRole(name="researcher", description="Research agent"),
            agent_class=ResearchAgent,
            spawn_policy=SpawnPolicy.POOLED,
            pool_size=3,
        )
        spawner.register_template(template)

        # Spawn an agent
        agent = await spawner.spawn("researcher")

        # Use the agent
        result = await agent.run(task)

        # Release when done
        await spawner.release(agent.id)
    """

    def __init__(
        self,
        registry: Optional["AgentRegistry"] = None,
        default_llm_factory: Optional[Callable[[], "LLMProvider"]] = None,
    ):
        """Initialize the agent spawner.

        Args:
            registry: Optional agent registry for tracking spawned agents.
            default_llm_factory: Default factory for creating LLM providers.
        """
        self.id = str(uuid.uuid4())
        self._registry = registry
        self._default_llm_factory = default_llm_factory
        self._templates: Dict[str, AgentTemplate] = {}
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_info: Dict[str, SpawnedAgentInfo] = {}
        self._pools: Dict[str, List[str]] = {}  # template_name -> available agent_ids
        self._instance_counts: Dict[str, int] = {}  # template_name -> count
        self._singletons: Dict[str, str] = {}  # template_name -> agent_id
        self._lock = asyncio.Lock()
        self._lifecycle_callbacks: List[Callable[[str, AgentLifecycleState], Any]] = []

    def register_template(self, template: AgentTemplate) -> None:
        """Register an agent template.

        Args:
            template: The template to register.

        Raises:
            ValueError: If a template with the same name exists.
        """
        if template.name in self._templates:
            raise ValueError(
                f"Template '{template.name}' is already registered. "
                "Use update_template or unregister first."
            )

        self._templates[template.name] = template
        self._instance_counts[template.name] = 0

        if template.spawn_policy == SpawnPolicy.POOLED:
            self._pools[template.name] = []

    def update_template(self, template: AgentTemplate) -> None:
        """Update an existing template.

        Args:
            template: The updated template.
        """
        self._templates[template.name] = template

    def unregister_template(self, name: str) -> Optional[AgentTemplate]:
        """Unregister a template.

        Args:
            name: Name of the template to unregister.

        Returns:
            The unregistered template if found.
        """
        template = self._templates.pop(name, None)
        self._pools.pop(name, None)
        self._instance_counts.pop(name, None)
        self._singletons.pop(name, None)
        return template

    def get_template(self, name: str) -> Optional[AgentTemplate]:
        """Get a template by name.

        Args:
            name: The template name.

        Returns:
            The template if found.
        """
        return self._templates.get(name)

    def list_templates(self) -> List[str]:
        """List all registered template names.

        Returns:
            List of template names.
        """
        return list(self._templates.keys())

    async def spawn(
        self,
        template_name: str,
        config_overrides: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BaseAgent:
        """Spawn a new agent from a template.

        Args:
            template_name: Name of the template to use.
            config_overrides: Optional configuration overrides.
            metadata: Optional metadata for the spawned agent.

        Returns:
            The spawned agent instance.

        Raises:
            ValueError: If template not found or max instances reached.
        """
        async with self._lock:
            template = self._templates.get(template_name)
            if not template:
                raise ValueError(f"Template '{template_name}' not found")

            # Handle singleton policy
            if template.spawn_policy == SpawnPolicy.SINGLETON:
                if template_name in self._singletons:
                    agent_id = self._singletons[template_name]
                    return self._agents[agent_id]

            # Handle pooled policy - try to get from pool first
            if template.spawn_policy == SpawnPolicy.POOLED:
                pool = self._pools.get(template_name, [])
                if pool:
                    agent_id = pool.pop(0)
                    agent = self._agents[agent_id]
                    info = self._agent_info[agent_id]
                    info.lifecycle_state = AgentLifecycleState.RUNNING
                    info.last_used = datetime.now()
                    await self._notify_lifecycle(agent_id, AgentLifecycleState.RUNNING)
                    return agent

            # Check max instances
            if template.max_instances > 0:
                current = self._instance_counts.get(template_name, 0)
                if current >= template.max_instances:
                    raise ValueError(
                        f"Maximum instances ({template.max_instances}) reached "
                        f"for template '{template_name}'"
                    )

            # Create new agent
            agent = await self._create_agent(template, config_overrides)

            # Track the agent
            info = SpawnedAgentInfo(
                agent_id=agent.id,
                template_name=template_name,
                lifecycle_state=AgentLifecycleState.RUNNING,
                parent_spawner_id=self.id,
            )
            self._agents[agent.id] = agent
            self._agent_info[agent.id] = info
            self._instance_counts[template_name] = (
                self._instance_counts.get(template_name, 0) + 1
            )

            # Register in registry if available
            if self._registry:
                self._registry.register(agent, metadata=metadata)

            # Track singleton
            if template.spawn_policy == SpawnPolicy.SINGLETON:
                self._singletons[template_name] = agent.id

            await self._notify_lifecycle(agent.id, AgentLifecycleState.RUNNING)
            return agent

    async def _create_agent(
        self,
        template: AgentTemplate,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> BaseAgent:
        """Create an agent instance from a template.

        Args:
            template: The template to use.
            config_overrides: Optional configuration overrides.

        Returns:
            The created agent instance.
        """
        # Create config
        config = template.config or AgentConfig()
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Create LLM provider
        llm = None
        if template.llm_factory:
            llm = template.llm_factory()
        elif self._default_llm_factory:
            llm = self._default_llm_factory()

        # Instantiate agent
        agent = template.agent_class(
            role=template.role,
            llm=llm,
            config=config,
        )

        return agent

    async def spawn_pool(self, template_name: str) -> List[BaseAgent]:
        """Pre-spawn a pool of agents for a template.

        Args:
            template_name: Name of the template.

        Returns:
            List of spawned agents.

        Raises:
            ValueError: If template not found or not pooled.
        """
        template = self._templates.get(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        if template.spawn_policy != SpawnPolicy.POOLED:
            raise ValueError(
                f"Template '{template_name}' is not configured for pooling"
            )

        agents: List[BaseAgent] = []
        pool = self._pools.setdefault(template_name, [])

        for _ in range(template.pool_size - len(pool)):
            agent = await self._create_agent(template)

            info = SpawnedAgentInfo(
                agent_id=agent.id,
                template_name=template_name,
                lifecycle_state=AgentLifecycleState.SUSPENDED,
                parent_spawner_id=self.id,
            )
            self._agents[agent.id] = agent
            self._agent_info[agent.id] = info
            self._instance_counts[template_name] = (
                self._instance_counts.get(template_name, 0) + 1
            )

            pool.append(agent.id)
            agents.append(agent)

            if self._registry:
                self._registry.register(agent)

        return agents

    async def release(self, agent_id: str) -> bool:
        """Release an agent back to the pool or terminate it.

        Args:
            agent_id: ID of the agent to release.

        Returns:
            True if agent was released successfully.
        """
        async with self._lock:
            info = self._agent_info.get(agent_id)
            if not info:
                return False

            template = self._templates.get(info.template_name)
            if not template:
                return await self.terminate(agent_id)

            # For pooled agents, return to pool
            if template.spawn_policy == SpawnPolicy.POOLED:
                pool = self._pools.setdefault(info.template_name, [])
                if len(pool) < template.pool_size:
                    info.lifecycle_state = AgentLifecycleState.SUSPENDED
                    info.last_used = datetime.now()
                    pool.append(agent_id)
                    await self._notify_lifecycle(
                        agent_id, AgentLifecycleState.SUSPENDED
                    )
                    return True

            # Otherwise terminate
            return await self._terminate_unlocked(agent_id)

    async def terminate(self, agent_id: str) -> bool:
        """Terminate a spawned agent.

        Args:
            agent_id: ID of the agent to terminate.

        Returns:
            True if agent was terminated successfully.
        """
        async with self._lock:
            return await self._terminate_unlocked(agent_id)

    async def _terminate_unlocked(self, agent_id: str) -> bool:
        """Terminate an agent (caller must hold lock).

        Args:
            agent_id: ID of the agent to terminate.

        Returns:
            True if agent was terminated successfully.
        """
        info = self._agent_info.get(agent_id)
        if not info:
            return False

        info.lifecycle_state = AgentLifecycleState.TERMINATING
        await self._notify_lifecycle(agent_id, AgentLifecycleState.TERMINATING)

        agent = self._agents.pop(agent_id, None)
        if agent:
            agent.status = AgentStatus.TERMINATED

        # Update counts
        if info.template_name in self._instance_counts:
            self._instance_counts[info.template_name] = max(
                0, self._instance_counts[info.template_name] - 1
            )

        # Remove from singleton tracking
        if self._singletons.get(info.template_name) == agent_id:
            del self._singletons[info.template_name]

        # Remove from pool
        pool = self._pools.get(info.template_name, [])
        if agent_id in pool:
            pool.remove(agent_id)

        # Unregister from registry
        if self._registry:
            self._registry.unregister(agent_id)

        info.lifecycle_state = AgentLifecycleState.TERMINATED
        await self._notify_lifecycle(agent_id, AgentLifecycleState.TERMINATED)

        # Remove info after notification
        self._agent_info.pop(agent_id, None)

        return True

    async def terminate_all(self, template_name: Optional[str] = None) -> int:
        """Terminate all agents, optionally filtered by template.

        Args:
            template_name: Optional template name to filter by.

        Returns:
            Number of agents terminated.
        """
        async with self._lock:
            agents_to_terminate = list(self._agent_info.items())
            if template_name:
                agents_to_terminate = [
                    (aid, info)
                    for aid, info in agents_to_terminate
                    if info.template_name == template_name
                ]

            count = 0
            for agent_id, _ in agents_to_terminate:
                if await self._terminate_unlocked(agent_id):
                    count += 1

            return count

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get a spawned agent by ID.

        Args:
            agent_id: The agent's ID.

        Returns:
            The agent if found.
        """
        return self._agents.get(agent_id)

    def get_agent_info(self, agent_id: str) -> Optional[SpawnedAgentInfo]:
        """Get info about a spawned agent.

        Args:
            agent_id: The agent's ID.

        Returns:
            SpawnedAgentInfo if found.
        """
        return self._agent_info.get(agent_id)

    def list_agents(
        self, template_name: Optional[str] = None
    ) -> List[BaseAgent]:
        """List all spawned agents.

        Args:
            template_name: Optional template name to filter by.

        Returns:
            List of spawned agents.
        """
        if template_name:
            return [
                self._agents[aid]
                for aid, info in self._agent_info.items()
                if info.template_name == template_name and aid in self._agents
            ]
        return list(self._agents.values())

    def get_pool_status(self, template_name: str) -> Dict[str, Any]:
        """Get pool status for a template.

        Args:
            template_name: The template name.

        Returns:
            Dictionary with pool status information.
        """
        template = self._templates.get(template_name)
        if not template:
            return {"error": f"Template '{template_name}' not found"}

        pool = self._pools.get(template_name, [])
        total = self._instance_counts.get(template_name, 0)

        return {
            "template_name": template_name,
            "spawn_policy": template.spawn_policy.value,
            "pool_size": template.pool_size,
            "available_in_pool": len(pool),
            "total_instances": total,
            "in_use": total - len(pool),
            "max_instances": template.max_instances,
        }

    def on_lifecycle_change(
        self, callback: Callable[[str, AgentLifecycleState], Any]
    ) -> None:
        """Register a callback for lifecycle changes.

        Args:
            callback: Function called with (agent_id, new_state).
        """
        self._lifecycle_callbacks.append(callback)

    async def _notify_lifecycle(
        self, agent_id: str, state: AgentLifecycleState
    ) -> None:
        """Notify callbacks of a lifecycle change."""
        for callback in self._lifecycle_callbacks:
            try:
                result = callback(agent_id, state)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get spawner statistics.

        Returns:
            Dictionary with spawner statistics.
        """
        return {
            "total_agents": len(self._agents),
            "templates": list(self._templates.keys()),
            "instance_counts": dict(self._instance_counts),
            "pool_sizes": {
                name: len(pool) for name, pool in self._pools.items()
            },
            "singletons": list(self._singletons.keys()),
        }

    def __repr__(self) -> str:
        return (
            f"AgentSpawner(id={self.id!r}, templates={len(self._templates)}, "
            f"agents={len(self._agents)})"
        )
