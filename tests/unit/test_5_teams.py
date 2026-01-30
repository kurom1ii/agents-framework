"""Test 5: Multi-Agent Teams - Can agents collaborate?

This test verifies multi-agent orchestration:
- Message routing between agents
- Agent registry
- Team patterns (hierarchical, swarm)
"""

import pytest
from agents_framework.llm.base import LLMConfig
from agents_framework.teams.router import MessageRouter, AgentMessage, MessagePriority
from agents_framework.teams.registry import AgentRegistry, AgentStatus


class MockAgent:
    """Simple mock agent for testing teams."""

    def __init__(self, agent_id: str, role: str = "worker"):
        self.agent_id = agent_id
        self.role = role
        self.received_messages = []

    async def handle_message(self, message):
        self.received_messages.append(message)
        return f"Handled by {self.agent_id}"


class TestMultiAgentTeamsCore:
    """Test multi-agent team functionality."""

    def test_agent_registry_register(self):
        """AgentRegistry can register agents."""
        registry = AgentRegistry()
        agent = MockAgent("agent_1", role="researcher")

        registry.register(agent, agent_id="agent_1", role="researcher")

        assert registry.has("agent_1")
        assert registry.get("agent_1") is agent

    def test_agent_registry_by_role(self):
        """AgentRegistry can find agents by role."""
        registry = AgentRegistry()
        researcher = MockAgent("r1", role="researcher")
        writer = MockAgent("w1", role="writer")

        registry.register(researcher, agent_id="r1", role="researcher")
        registry.register(writer, agent_id="w1", role="writer")

        researchers = registry.get_by_role("researcher")
        assert len(researchers) == 1
        assert researchers[0] is researcher

    def test_agent_registry_status_tracking(self):
        """AgentRegistry tracks agent status."""
        registry = AgentRegistry()
        agent = MockAgent("agent_1")

        registry.register(agent, agent_id="agent_1")
        registry.set_status("agent_1", AgentStatus.BUSY)

        status = registry.get_status("agent_1")
        assert status == AgentStatus.BUSY

    def test_agent_message_creation(self):
        """AgentMessage carries inter-agent communication."""
        message = AgentMessage(
            sender_id="supervisor",
            receiver_id="worker_1",
            content="Please research topic X",
            priority=MessagePriority.HIGH,
        )

        assert message.sender_id == "supervisor"
        assert message.receiver_id == "worker_1"
        assert message.priority == MessagePriority.HIGH

    @pytest.mark.asyncio
    async def test_message_router_direct_routing(self):
        """MessageRouter delivers messages to specific agent."""
        router = MessageRouter()
        agent = MockAgent("worker_1")

        router.register_agent("worker_1", agent.handle_message)

        message = AgentMessage(
            sender_id="supervisor",
            receiver_id="worker_1",
            content="Do task",
        )

        await router.route(message)

        # Check agent received the message
        assert len(agent.received_messages) == 1

    @pytest.mark.asyncio
    async def test_message_router_broadcast(self):
        """MessageRouter can broadcast to all agents."""
        router = MessageRouter()
        agent1 = MockAgent("agent_1")
        agent2 = MockAgent("agent_2")

        router.register_agent("agent_1", agent1.handle_message)
        router.register_agent("agent_2", agent2.handle_message)

        message = AgentMessage(
            sender_id="supervisor",
            receiver_id="*",  # Broadcast
            content="Announcement",
        )

        await router.broadcast(message)

        assert len(agent1.received_messages) == 1
        assert len(agent2.received_messages) == 1

    def test_agent_registry_list_all(self):
        """AgentRegistry lists all registered agents."""
        registry = AgentRegistry()
        registry.register(MockAgent("a1"), agent_id="a1")
        registry.register(MockAgent("a2"), agent_id="a2")
        registry.register(MockAgent("a3"), agent_id="a3")

        all_agents = registry.list_all()

        assert len(all_agents) == 3
        assert "a1" in [a[0] for a in all_agents]

    def test_agent_registry_unregister(self):
        """AgentRegistry can remove agents."""
        registry = AgentRegistry()
        registry.register(MockAgent("temp"), agent_id="temp")

        assert registry.has("temp")
        registry.unregister("temp")
        assert not registry.has("temp")

    @pytest.mark.asyncio
    async def test_priority_message_ordering(self):
        """Messages with higher priority are processed first."""
        router = MessageRouter()
        received_order = []

        async def handler(msg):
            received_order.append(msg.priority.value)

        router.register_agent("worker", handler)

        # Queue messages with different priorities
        low = AgentMessage(sender_id="s", receiver_id="worker", content="low", priority=MessagePriority.LOW)
        high = AgentMessage(sender_id="s", receiver_id="worker", content="high", priority=MessagePriority.HIGH)
        normal = AgentMessage(sender_id="s", receiver_id="worker", content="normal", priority=MessagePriority.NORMAL)

        # Add to priority queue and process
        router.enqueue(high)
        router.enqueue(low)
        router.enqueue(normal)

        await router.process_queue()

        # High priority should be first
        assert received_order[0] >= received_order[1] >= received_order[2]
