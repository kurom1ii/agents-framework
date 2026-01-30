"""Unit tests for the agent registry module."""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from agents_framework.agents import AgentConfig, AgentRole, AgentStatus
from agents_framework.teams.registry import (
    AgentInfo,
    AgentLookupError,
    AgentRegistry,
    get_default_registry,
    register_agent,
)

from .conftest import MockAgent


# ============================================================================
# AgentInfo Tests
# ============================================================================


class TestAgentInfo:
    """Tests for the AgentInfo dataclass."""

    def test_agent_info_creation(self):
        """Test creating agent info."""
        info = AgentInfo(
            id="agent_123",
            name="TestAgent",
            role_name="researcher",
            capabilities=["search", "analyze"],
            status=AgentStatus.IDLE,
        )

        assert info.id == "agent_123"
        assert info.name == "TestAgent"
        assert info.role_name == "researcher"
        assert info.capabilities == ["search", "analyze"]
        assert info.status == AgentStatus.IDLE
        assert info.registered_at is not None
        assert info.last_activity is None
        assert info.error_message is None

    def test_agent_info_with_metadata(self):
        """Test agent info with metadata."""
        info = AgentInfo(
            id="agent_123",
            name="TestAgent",
            role_name="researcher",
            metadata={"priority": "high", "team": "alpha"},
        )

        assert info.metadata == {"priority": "high", "team": "alpha"}


# ============================================================================
# AgentRegistry Tests
# ============================================================================


class TestAgentRegistry:
    """Tests for the AgentRegistry class."""

    def test_registry_creation(self, agent_registry: AgentRegistry):
        """Test creating an empty registry."""
        assert len(agent_registry) == 0
        assert agent_registry.list_agents() == []
        assert agent_registry.list_ids() == []

    def test_registry_repr(self, agent_registry: AgentRegistry):
        """Test registry string representation."""
        repr_str = repr(agent_registry)
        assert "AgentRegistry" in repr_str

    def test_register_agent(
        self,
        agent_registry: AgentRegistry,
        mock_researcher: MockAgent,
    ):
        """Test registering an agent."""
        info = agent_registry.register(mock_researcher)

        assert info.id == mock_researcher.id
        assert info.name == mock_researcher.config.name
        assert info.role_name == mock_researcher.role.name
        assert info.capabilities == list(mock_researcher.role.capabilities)
        assert len(agent_registry) == 1

    def test_register_agent_with_metadata(
        self,
        agent_registry: AgentRegistry,
        mock_researcher: MockAgent,
    ):
        """Test registering an agent with metadata."""
        metadata = {"team": "alpha", "priority": 1}
        info = agent_registry.register(mock_researcher, metadata=metadata)

        assert info.metadata == metadata

    def test_register_duplicate_agent_raises(
        self,
        agent_registry: AgentRegistry,
        mock_researcher: MockAgent,
    ):
        """Test that registering duplicate agent raises error."""
        agent_registry.register(mock_researcher)

        with pytest.raises(ValueError, match="already registered"):
            agent_registry.register(mock_researcher)

    def test_unregister_agent(
        self,
        agent_registry: AgentRegistry,
        mock_researcher: MockAgent,
    ):
        """Test unregistering an agent."""
        agent_registry.register(mock_researcher)
        agent = agent_registry.unregister(mock_researcher.id)

        assert agent is mock_researcher
        assert len(agent_registry) == 0
        assert mock_researcher.id not in agent_registry

    def test_unregister_nonexistent_agent(self, agent_registry: AgentRegistry):
        """Test unregistering non-existent agent returns None."""
        agent = agent_registry.unregister("nonexistent")
        assert agent is None

    def test_get_agent(
        self,
        agent_registry: AgentRegistry,
        mock_researcher: MockAgent,
    ):
        """Test getting an agent by ID."""
        agent_registry.register(mock_researcher)

        agent = agent_registry.get(mock_researcher.id)
        assert agent is mock_researcher

    def test_get_nonexistent_agent(self, agent_registry: AgentRegistry):
        """Test getting non-existent agent returns None."""
        agent = agent_registry.get("nonexistent")
        assert agent is None

    def test_get_or_raise(
        self,
        agent_registry: AgentRegistry,
        mock_researcher: MockAgent,
    ):
        """Test get_or_raise for existing agent."""
        agent_registry.register(mock_researcher)

        agent = agent_registry.get_or_raise(mock_researcher.id)
        assert agent is mock_researcher

    def test_get_or_raise_nonexistent(self, agent_registry: AgentRegistry):
        """Test get_or_raise raises for non-existent agent."""
        with pytest.raises(AgentLookupError, match="not found"):
            agent_registry.get_or_raise("nonexistent")

    def test_get_info(
        self,
        agent_registry: AgentRegistry,
        mock_researcher: MockAgent,
    ):
        """Test getting agent info."""
        agent_registry.register(mock_researcher)

        info = agent_registry.get_info(mock_researcher.id)
        assert info is not None
        assert info.id == mock_researcher.id

    def test_get_info_nonexistent(self, agent_registry: AgentRegistry):
        """Test getting info for non-existent agent."""
        info = agent_registry.get_info("nonexistent")
        assert info is None

    def test_has_agent(
        self,
        agent_registry: AgentRegistry,
        mock_researcher: MockAgent,
    ):
        """Test checking if agent is registered."""
        assert not agent_registry.has(mock_researcher.id)

        agent_registry.register(mock_researcher)

        assert agent_registry.has(mock_researcher.id)

    def test_contains_dunder(
        self,
        agent_registry: AgentRegistry,
        mock_researcher: MockAgent,
    ):
        """Test __contains__ protocol."""
        agent_registry.register(mock_researcher)

        assert mock_researcher.id in agent_registry
        assert "nonexistent" not in agent_registry

    def test_list_agents(self, populated_registry: AgentRegistry):
        """Test listing all agents."""
        agents = populated_registry.list_agents()
        assert len(agents) == 3

    def test_list_info(self, populated_registry: AgentRegistry):
        """Test listing all agent info."""
        info_list = populated_registry.list_info()
        assert len(info_list) == 3
        assert all(isinstance(info, AgentInfo) for info in info_list)

    def test_list_ids(self, populated_registry: AgentRegistry):
        """Test listing all agent IDs."""
        ids = populated_registry.list_ids()
        assert len(ids) == 3
        assert all(isinstance(id_, str) for id_ in ids)

    def test_iter_protocol(self, populated_registry: AgentRegistry):
        """Test __iter__ protocol."""
        agents = list(populated_registry)
        assert len(agents) == 3


# ============================================================================
# Find by Role Tests
# ============================================================================


class TestFindByRole:
    """Tests for finding agents by role."""

    def test_find_by_role(self, populated_registry: AgentRegistry):
        """Test finding agents by role name."""
        researchers = populated_registry.find_by_role("researcher")
        assert len(researchers) == 1
        assert researchers[0].role.name == "researcher"

    def test_find_by_role_multiple(
        self,
        agent_registry: AgentRegistry,
        researcher_role: AgentRole,
    ):
        """Test finding multiple agents with same role."""
        agent1 = MockAgent(role=researcher_role, config=AgentConfig(name="Agent1"))
        agent2 = MockAgent(role=researcher_role, config=AgentConfig(name="Agent2"))

        agent_registry.register(agent1)
        agent_registry.register(agent2)

        researchers = agent_registry.find_by_role("researcher")
        assert len(researchers) == 2

    def test_find_by_role_none(self, populated_registry: AgentRegistry):
        """Test finding agents with non-existent role."""
        agents = populated_registry.find_by_role("nonexistent")
        assert agents == []


# ============================================================================
# Find by Capability Tests
# ============================================================================


class TestFindByCapability:
    """Tests for finding agents by capability."""

    def test_find_by_capability(self, populated_registry: AgentRegistry):
        """Test finding agents by single capability."""
        analyzers = populated_registry.find_by_capability("analyze")
        # Both researcher and analyst have "analyze" capability
        assert len(analyzers) == 2

    def test_find_by_capability_none(self, populated_registry: AgentRegistry):
        """Test finding agents with non-existent capability."""
        agents = populated_registry.find_by_capability("nonexistent")
        assert agents == []

    def test_find_by_capabilities_match_all(self, populated_registry: AgentRegistry):
        """Test finding agents with all specified capabilities."""
        agents = populated_registry.find_by_capabilities(
            ["analyze", "summarize"],
            match_all=True,
        )
        # Only researcher has both
        assert len(agents) == 1
        assert agents[0].role.name == "researcher"

    def test_find_by_capabilities_match_any(self, populated_registry: AgentRegistry):
        """Test finding agents with any of specified capabilities."""
        agents = populated_registry.find_by_capabilities(
            ["write", "calculate"],
            match_all=False,
        )
        # Writer has "write", analyst has "calculate"
        assert len(agents) == 2

    def test_find_by_capabilities_empty_list(self, populated_registry: AgentRegistry):
        """Test finding with empty capability list returns empty."""
        agents = populated_registry.find_by_capabilities([])
        assert agents == []


# ============================================================================
# Find by Status Tests
# ============================================================================


class TestFindByStatus:
    """Tests for finding agents by status."""

    def test_find_by_status_idle(self, populated_registry: AgentRegistry):
        """Test finding idle agents."""
        agents = populated_registry.find_by_status(AgentStatus.IDLE)
        assert len(agents) == 3  # All start as idle

    def test_find_by_status_busy(
        self,
        agent_registry: AgentRegistry,
        mock_researcher: MockAgent,
    ):
        """Test finding busy agents."""
        agent_registry.register(mock_researcher)
        agent_registry.update_status(mock_researcher.id, AgentStatus.BUSY)

        busy_agents = agent_registry.find_by_status(AgentStatus.BUSY)
        assert len(busy_agents) == 1
        assert busy_agents[0].id == mock_researcher.id

    def test_find_idle(self, populated_registry: AgentRegistry):
        """Test find_idle convenience method."""
        agents = populated_registry.find_idle()
        assert len(agents) == 3

    def test_find_available_no_filters(self, populated_registry: AgentRegistry):
        """Test find_available without filters."""
        agents = populated_registry.find_available()
        assert len(agents) == 3

    def test_find_available_with_capability(self, populated_registry: AgentRegistry):
        """Test find_available with capability filter."""
        agents = populated_registry.find_available(capability="write")
        assert len(agents) == 1
        assert agents[0].role.name == "writer"

    def test_find_available_with_role(self, populated_registry: AgentRegistry):
        """Test find_available with role filter."""
        agents = populated_registry.find_available(role="analyst")
        assert len(agents) == 1
        assert agents[0].role.name == "analyst"

    def test_find_available_excludes_busy(
        self,
        agent_registry: AgentRegistry,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test that find_available excludes busy agents."""
        agent_registry.register(mock_researcher)
        agent_registry.register(mock_analyst)

        agent_registry.update_status(mock_researcher.id, AgentStatus.BUSY)

        available = agent_registry.find_available()
        assert len(available) == 1
        assert available[0].id == mock_analyst.id


# ============================================================================
# Status Update Tests
# ============================================================================


class TestStatusUpdates:
    """Tests for status update functionality."""

    def test_update_status(
        self,
        agent_registry: AgentRegistry,
        mock_researcher: MockAgent,
    ):
        """Test updating agent status."""
        agent_registry.register(mock_researcher)

        result = agent_registry.update_status(mock_researcher.id, AgentStatus.BUSY)
        assert result

        info = agent_registry.get_info(mock_researcher.id)
        assert info.status == AgentStatus.BUSY
        assert info.last_activity is not None

    def test_update_status_with_error(
        self,
        agent_registry: AgentRegistry,
        mock_researcher: MockAgent,
    ):
        """Test updating status to error with message."""
        agent_registry.register(mock_researcher)

        agent_registry.update_status(
            mock_researcher.id,
            AgentStatus.ERROR,
            error_message="Something went wrong",
        )

        info = agent_registry.get_info(mock_researcher.id)
        assert info.status == AgentStatus.ERROR
        assert info.error_message == "Something went wrong"

    def test_update_status_clears_error_on_recovery(
        self,
        agent_registry: AgentRegistry,
        mock_researcher: MockAgent,
    ):
        """Test that error message is cleared when status changes."""
        agent_registry.register(mock_researcher)

        agent_registry.update_status(
            mock_researcher.id,
            AgentStatus.ERROR,
            error_message="Error",
        )
        agent_registry.update_status(mock_researcher.id, AgentStatus.IDLE)

        info = agent_registry.get_info(mock_researcher.id)
        assert info.status == AgentStatus.IDLE
        assert info.error_message is None

    def test_update_status_updates_agent(
        self,
        agent_registry: AgentRegistry,
        mock_researcher: MockAgent,
    ):
        """Test that status update also updates agent object."""
        agent_registry.register(mock_researcher)

        agent_registry.update_status(mock_researcher.id, AgentStatus.BUSY)

        assert mock_researcher.status == AgentStatus.BUSY

    def test_update_status_nonexistent(self, agent_registry: AgentRegistry):
        """Test updating status of non-existent agent."""
        result = agent_registry.update_status("nonexistent", AgentStatus.BUSY)
        assert not result

    def test_on_status_change_callback(
        self,
        agent_registry: AgentRegistry,
        mock_researcher: MockAgent,
    ):
        """Test status change callback is invoked."""
        callbacks = []

        def callback(agent_id: str, old_status: AgentStatus, new_status: AgentStatus):
            callbacks.append((agent_id, old_status, new_status))

        agent_registry.on_status_change(callback)
        agent_registry.register(mock_researcher)
        agent_registry.update_status(mock_researcher.id, AgentStatus.BUSY)

        assert len(callbacks) == 1
        assert callbacks[0] == (mock_researcher.id, AgentStatus.IDLE, AgentStatus.BUSY)

    def test_clear_status_callbacks(
        self,
        agent_registry: AgentRegistry,
        mock_researcher: MockAgent,
    ):
        """Test clearing status callbacks."""
        callbacks = []

        def callback(agent_id: str, old_status: AgentStatus, new_status: AgentStatus):
            callbacks.append(1)

        agent_registry.on_status_change(callback)
        agent_registry.clear_status_callbacks()
        agent_registry.register(mock_researcher)
        agent_registry.update_status(mock_researcher.id, AgentStatus.BUSY)

        assert len(callbacks) == 0


# ============================================================================
# Index Management Tests
# ============================================================================


class TestIndexManagement:
    """Tests for role and capability index management."""

    def test_role_index_cleanup_on_unregister(
        self,
        agent_registry: AgentRegistry,
        mock_researcher: MockAgent,
    ):
        """Test that role index is cleaned up on unregister."""
        agent_registry.register(mock_researcher)

        # Should find by role
        assert len(agent_registry.find_by_role("researcher")) == 1

        agent_registry.unregister(mock_researcher.id)

        # Role index should be cleaned
        assert len(agent_registry.find_by_role("researcher")) == 0

    def test_capability_index_cleanup_on_unregister(
        self,
        agent_registry: AgentRegistry,
        mock_researcher: MockAgent,
    ):
        """Test that capability index is cleaned up on unregister."""
        agent_registry.register(mock_researcher)

        assert len(agent_registry.find_by_capability("search")) == 1

        agent_registry.unregister(mock_researcher.id)

        assert len(agent_registry.find_by_capability("search")) == 0


# ============================================================================
# Statistics Tests
# ============================================================================


class TestStatistics:
    """Tests for registry statistics."""

    def test_get_statistics_empty(self, agent_registry: AgentRegistry):
        """Test statistics for empty registry."""
        stats = agent_registry.get_statistics()

        assert stats["total_agents"] == 0
        assert stats["roles"] == []
        assert stats["capabilities"] == []
        assert stats["status_counts"] == {}

    def test_get_statistics_populated(self, populated_registry: AgentRegistry):
        """Test statistics for populated registry."""
        stats = populated_registry.get_statistics()

        assert stats["total_agents"] == 3
        assert len(stats["roles"]) == 3
        assert len(stats["capabilities"]) > 0
        assert stats["status_counts"]["idle"] == 3

    def test_get_statistics_with_mixed_status(
        self,
        agent_registry: AgentRegistry,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test statistics with mixed agent statuses."""
        agent_registry.register(mock_researcher)
        agent_registry.register(mock_analyst)

        agent_registry.update_status(mock_researcher.id, AgentStatus.BUSY)

        stats = agent_registry.get_statistics()

        assert stats["status_counts"]["idle"] == 1
        assert stats["status_counts"]["busy"] == 1


# ============================================================================
# Clear and Utility Tests
# ============================================================================


class TestClearAndUtilities:
    """Tests for clear and utility methods."""

    def test_clear(self, populated_registry: AgentRegistry):
        """Test clearing all agents from registry."""
        assert len(populated_registry) == 3

        populated_registry.clear()

        assert len(populated_registry) == 0
        assert populated_registry.list_agents() == []
        assert populated_registry.list_ids() == []

    def test_len_protocol(self, populated_registry: AgentRegistry):
        """Test __len__ protocol."""
        assert len(populated_registry) == 3


# ============================================================================
# Global Registry Tests
# ============================================================================


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_default_registry(self):
        """Test getting default registry."""
        registry = get_default_registry()
        assert isinstance(registry, AgentRegistry)

    def test_get_default_registry_returns_same_instance(self):
        """Test that default registry is a singleton."""
        registry1 = get_default_registry()
        registry2 = get_default_registry()
        assert registry1 is registry2
