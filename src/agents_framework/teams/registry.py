"""Agent registry for managing and discovering agents.

This module provides the AgentRegistry class for registering, tracking,
and looking up agents by various criteria including role, capability, and ID.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

if TYPE_CHECKING:
    from agents_framework.agents import BaseAgent

from agents_framework.agents import AgentStatus


@dataclass
class AgentInfo:
    """Information about a registered agent.

    Attributes:
        id: Unique identifier of the agent.
        name: Human-readable name of the agent.
        role_name: Name of the agent's role.
        capabilities: List of agent capabilities.
        status: Current agent status.
        metadata: Additional metadata about the agent.
        registered_at: When the agent was registered.
        last_activity: Last known activity timestamp.
        error_message: Last error message if status is ERROR.
    """

    id: str
    name: str
    role_name: str
    capabilities: List[str] = field(default_factory=list)
    status: AgentStatus = AgentStatus.IDLE
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.now)
    last_activity: Optional[datetime] = None
    error_message: Optional[str] = None


class AgentLookupError(Exception):
    """Exception raised when agent lookup fails."""

    pass


class AgentRegistry:
    """Registry for managing and discovering agents.

    Provides agent registration, deregistration, and lookup by various
    criteria. Also tracks agent status and health.

    Example:
        registry = AgentRegistry()

        # Register an agent
        registry.register(agent)

        # Find agents by capability
        analyzers = registry.find_by_capability("analyze")

        # Update status
        registry.update_status(agent.id, AgentStatus.BUSY)

        # Get idle agents
        idle_agents = registry.find_by_status(AgentStatus.IDLE)
    """

    def __init__(self):
        """Initialize an empty agent registry."""
        self._agents: Dict[str, "BaseAgent"] = {}
        self._info: Dict[str, AgentInfo] = {}
        self._by_role: Dict[str, Set[str]] = {}  # role_name -> agent_ids
        self._by_capability: Dict[str, Set[str]] = {}  # capability -> agent_ids
        self._status_callbacks: List[Callable[[str, AgentStatus, AgentStatus], Any]] = []
        self._lock = asyncio.Lock()

    def register(
        self,
        agent: "BaseAgent",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentInfo:
        """Register an agent in the registry.

        Args:
            agent: The agent to register.
            metadata: Optional metadata to associate with the agent.

        Returns:
            AgentInfo for the registered agent.

        Raises:
            ValueError: If an agent with the same ID is already registered.
        """
        if agent.id in self._agents:
            raise ValueError(
                f"Agent '{agent.id}' is already registered. "
                "Use update or unregister first."
            )

        # Create agent info
        info = AgentInfo(
            id=agent.id,
            name=agent.config.name,
            role_name=agent.role.name,
            capabilities=list(agent.role.capabilities),
            status=agent.status,
            metadata=metadata or {},
            registered_at=datetime.now(),
        )

        # Store agent and info
        self._agents[agent.id] = agent
        self._info[agent.id] = info

        # Index by role
        if agent.role.name not in self._by_role:
            self._by_role[agent.role.name] = set()
        self._by_role[agent.role.name].add(agent.id)

        # Index by capabilities
        for capability in agent.role.capabilities:
            if capability not in self._by_capability:
                self._by_capability[capability] = set()
            self._by_capability[capability].add(agent.id)

        return info

    def unregister(self, agent_id: str) -> Optional["BaseAgent"]:
        """Unregister an agent from the registry.

        Args:
            agent_id: The ID of the agent to unregister.

        Returns:
            The unregistered agent if found, None otherwise.
        """
        agent = self._agents.pop(agent_id, None)
        info = self._info.pop(agent_id, None)

        if info:
            # Remove from role index
            if info.role_name in self._by_role:
                self._by_role[info.role_name].discard(agent_id)
                if not self._by_role[info.role_name]:
                    del self._by_role[info.role_name]

            # Remove from capability index
            for capability in info.capabilities:
                if capability in self._by_capability:
                    self._by_capability[capability].discard(agent_id)
                    if not self._by_capability[capability]:
                        del self._by_capability[capability]

        return agent

    def get(self, agent_id: str) -> Optional["BaseAgent"]:
        """Get an agent by ID.

        Args:
            agent_id: The agent's unique identifier.

        Returns:
            The agent if found, None otherwise.
        """
        return self._agents.get(agent_id)

    def get_or_raise(self, agent_id: str) -> "BaseAgent":
        """Get an agent by ID, raising an error if not found.

        Args:
            agent_id: The agent's unique identifier.

        Returns:
            The agent.

        Raises:
            AgentLookupError: If the agent is not found.
        """
        agent = self._agents.get(agent_id)
        if agent is None:
            available = ", ".join(self._agents.keys()) or "none"
            raise AgentLookupError(
                f"Agent '{agent_id}' not found. Available agents: {available}"
            )
        return agent

    def get_info(self, agent_id: str) -> Optional[AgentInfo]:
        """Get info for an agent.

        Args:
            agent_id: The agent's unique identifier.

        Returns:
            AgentInfo if found, None otherwise.
        """
        return self._info.get(agent_id)

    def has(self, agent_id: str) -> bool:
        """Check if an agent is registered.

        Args:
            agent_id: The agent's unique identifier.

        Returns:
            True if the agent is registered, False otherwise.
        """
        return agent_id in self._agents

    def list_agents(self) -> List["BaseAgent"]:
        """List all registered agents.

        Returns:
            List of all registered agents.
        """
        return list(self._agents.values())

    def list_info(self) -> List[AgentInfo]:
        """List info for all registered agents.

        Returns:
            List of AgentInfo for all registered agents.
        """
        return list(self._info.values())

    def list_ids(self) -> List[str]:
        """List all registered agent IDs.

        Returns:
            List of agent IDs.
        """
        return list(self._agents.keys())

    def find_by_role(self, role_name: str) -> List["BaseAgent"]:
        """Find agents by role name.

        Args:
            role_name: The role name to search for.

        Returns:
            List of agents with the specified role.
        """
        agent_ids = self._by_role.get(role_name, set())
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def find_by_capability(self, capability: str) -> List["BaseAgent"]:
        """Find agents by capability.

        Args:
            capability: The capability to search for.

        Returns:
            List of agents with the specified capability.
        """
        agent_ids = self._by_capability.get(capability, set())
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def find_by_capabilities(
        self, capabilities: List[str], match_all: bool = True
    ) -> List["BaseAgent"]:
        """Find agents by multiple capabilities.

        Args:
            capabilities: List of capabilities to search for.
            match_all: If True, agents must have all capabilities.
                      If False, agents must have at least one.

        Returns:
            List of matching agents.
        """
        if not capabilities:
            return []

        # Get agent IDs for each capability
        capability_sets = [
            self._by_capability.get(cap, set()) for cap in capabilities
        ]

        if match_all:
            # Intersection: must have all capabilities
            matching_ids = set.intersection(*capability_sets) if capability_sets else set()
        else:
            # Union: must have at least one capability
            matching_ids = set.union(*capability_sets) if capability_sets else set()

        return [self._agents[aid] for aid in matching_ids if aid in self._agents]

    def find_by_status(self, status: AgentStatus) -> List["BaseAgent"]:
        """Find agents by status.

        Args:
            status: The status to filter by.

        Returns:
            List of agents with the specified status.
        """
        return [
            agent for agent in self._agents.values()
            if self._info[agent.id].status == status
        ]

    def find_idle(self) -> List["BaseAgent"]:
        """Find all idle agents.

        Returns:
            List of idle agents.
        """
        return self.find_by_status(AgentStatus.IDLE)

    def find_available(
        self,
        capability: Optional[str] = None,
        role: Optional[str] = None,
    ) -> List["BaseAgent"]:
        """Find available (idle) agents with optional filtering.

        Args:
            capability: Optional capability to filter by.
            role: Optional role to filter by.

        Returns:
            List of available agents matching the criteria.
        """
        agents = self.find_idle()

        if capability:
            agents = [a for a in agents if a.has_capability(capability)]

        if role:
            agents = [a for a in agents if a.role.name == role]

        return agents

    def update_status(
        self,
        agent_id: str,
        status: AgentStatus,
        error_message: Optional[str] = None,
    ) -> bool:
        """Update an agent's status.

        Args:
            agent_id: The agent's unique identifier.
            status: The new status.
            error_message: Optional error message (for ERROR status).

        Returns:
            True if the status was updated, False if agent not found.
        """
        info = self._info.get(agent_id)
        if info is None:
            return False

        old_status = info.status
        info.status = status
        info.last_activity = datetime.now()

        if status == AgentStatus.ERROR:
            info.error_message = error_message
        else:
            info.error_message = None

        # Also update the agent's status
        agent = self._agents.get(agent_id)
        if agent:
            agent.status = status

        # Notify callbacks
        for callback in self._status_callbacks:
            try:
                result = callback(agent_id, old_status, status)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception:
                pass

        return True

    def on_status_change(
        self, callback: Callable[[str, AgentStatus, AgentStatus], Any]
    ) -> None:
        """Register a callback for status changes.

        Args:
            callback: Function called with (agent_id, old_status, new_status).
        """
        self._status_callbacks.append(callback)

    def clear_status_callbacks(self) -> None:
        """Clear all status change callbacks."""
        self._status_callbacks.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about registered agents.

        Returns:
            Dictionary with agent statistics.
        """
        status_counts = {}
        for info in self._info.values():
            status_name = info.status.value
            status_counts[status_name] = status_counts.get(status_name, 0) + 1

        return {
            "total_agents": len(self._agents),
            "roles": list(self._by_role.keys()),
            "capabilities": list(self._by_capability.keys()),
            "status_counts": status_counts,
        }

    def clear(self) -> None:
        """Clear all registered agents."""
        self._agents.clear()
        self._info.clear()
        self._by_role.clear()
        self._by_capability.clear()

    def __len__(self) -> int:
        """Return the number of registered agents."""
        return len(self._agents)

    def __contains__(self, agent_id: str) -> bool:
        """Check if an agent is registered."""
        return agent_id in self._agents

    def __iter__(self):
        """Iterate over registered agents."""
        return iter(self._agents.values())

    def __repr__(self) -> str:
        return f"AgentRegistry(agents={list(self._agents.keys())})"


# Global default registry
_default_registry = AgentRegistry()


def get_default_registry() -> AgentRegistry:
    """Get the default global agent registry."""
    return _default_registry


def register_agent(
    agent: "BaseAgent",
    metadata: Optional[Dict[str, Any]] = None,
) -> AgentInfo:
    """Register an agent in the default registry.

    Convenience function for registering agents without creating a registry.

    Args:
        agent: The agent to register.
        metadata: Optional metadata to associate with the agent.

    Returns:
        AgentInfo for the registered agent.
    """
    return _default_registry.register(agent, metadata=metadata)
