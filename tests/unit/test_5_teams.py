"""Test 5: Multi-Agent Teams - Can agents collaborate?

This test verifies multi-agent orchestration:
- Message routing between agents
- Agent registry
- Team patterns (hierarchical, swarm)
"""

import pytest
from dataclasses import dataclass, field
from typing import List
from agents_framework.teams.router import MessageRouter, AgentMessage, MessagePriority, RoutingStrategy
from agents_framework.teams.registry import AgentRegistry
from agents_framework.agents import AgentStatus


@dataclass
class MockRole:
    """Mock role for testing."""
    name: str = "worker"
    capabilities: List[str] = field(default_factory=list)


@dataclass
class MockConfig:
    """Mock config for testing."""
    name: str = "test-agent"


class MockAgent:
    """Simple mock agent for testing teams."""

    def __init__(self, agent_id: str, role: str = "worker"):
        self.id = agent_id
        self.config = MockConfig(name=f"Agent-{agent_id}")
        self.role = MockRole(name=role, capabilities=["process"])
        self.status = AgentStatus.IDLE
        self.received_messages = []

    async def handle_message(self, message):
        self.received_messages.append(message)
        return f"Handled by {self.id}"


class TestMultiAgentTeamsCore:
    """Test multi-agent team functionality."""

    def test_agent_registry_register(self):
        """AgentRegistry can register agents."""
        registry = AgentRegistry()
        agent = MockAgent("agent_1", role="researcher")

        registry.register(agent)

        assert registry.has("agent_1")
        assert registry.get("agent_1") is agent

    def test_agent_registry_by_role(self):
        """AgentRegistry can find agents by role."""
        registry = AgentRegistry()
        researcher = MockAgent("r1", role="researcher")
        writer = MockAgent("w1", role="writer")

        registry.register(researcher)
        registry.register(writer)

        researchers = registry.find_by_role("researcher")
        assert len(researchers) == 1
        assert researchers[0] is researcher

    def test_agent_registry_status_tracking(self):
        """AgentRegistry tracks agent status."""
        registry = AgentRegistry()
        agent = MockAgent("agent_1")

        registry.register(agent)
        registry.update_status("agent_1", AgentStatus.BUSY)

        info = registry.get_info("agent_1")
        assert info.status == AgentStatus.BUSY

    def test_agent_message_creation(self):
        """AgentMessage carries inter-agent communication."""
        message = AgentMessage(
            sender_id="supervisor",
            recipient_id="worker_1",
            content="Please research topic X",
            priority=MessagePriority.HIGH,
        )

        assert message.sender_id == "supervisor"
        assert message.recipient_id == "worker_1"
        assert message.priority == MessagePriority.HIGH

    @pytest.mark.asyncio
    async def test_message_router_direct_routing(self):
        """MessageRouter delivers messages to specific agent."""
        router = MessageRouter()
        agent = MockAgent("worker_1")

        router.register_agent("worker_1", handler=agent.handle_message)

        message = AgentMessage(
            sender_id="supervisor",
            recipient_id="worker_1",
            content="Do task",
        )

        recipients = await router.send(message)

        # Check agent received the message
        assert len(agent.received_messages) == 1
        assert "worker_1" in recipients

    @pytest.mark.asyncio
    async def test_message_router_broadcast(self):
        """MessageRouter can broadcast to topic subscribers."""
        router = MessageRouter()
        agent1 = MockAgent("agent_1")
        agent2 = MockAgent("agent_2")

        router.register_agent("agent_1", handler=agent1.handle_message)
        router.register_agent("agent_2", handler=agent2.handle_message)

        # Subscribe both to a topic
        router.subscribe("agent_1", "announcements")
        router.subscribe("agent_2", "announcements")

        message = AgentMessage(
            sender_id="supervisor",
            topic="announcements",  # Topic-based routing
            content="Announcement",
            strategy=RoutingStrategy.TOPIC,
        )

        recipients = await router.send(message)

        assert len(agent1.received_messages) == 1
        assert len(agent2.received_messages) == 1
        assert len(recipients) == 2

    def test_agent_registry_list_all(self):
        """AgentRegistry lists all registered agents."""
        registry = AgentRegistry()
        registry.register(MockAgent("a1"))
        registry.register(MockAgent("a2"))
        registry.register(MockAgent("a3"))

        all_agent_ids = registry.list_ids()

        assert len(all_agent_ids) == 3
        assert "a1" in all_agent_ids

    def test_agent_registry_unregister(self):
        """AgentRegistry can remove agents."""
        registry = AgentRegistry()
        registry.register(MockAgent("temp"))

        assert registry.has("temp")
        registry.unregister("temp")
        assert not registry.has("temp")

    def test_agent_message_with_metadata(self):
        """AgentMessage can carry metadata."""
        message = AgentMessage(
            sender_id="sender",
            recipient_id="receiver",
            content="test",
            metadata={"key": "value", "count": 42},
        )

        assert message.metadata["key"] == "value"
        assert message.metadata["count"] == 42
