"""Unit tests for the swarm team pattern module."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import pytest

from agents_framework.agents import AgentRole, Task, TaskResult
from agents_framework.teams.patterns.base import PatternContext, PatternStatus, StepResult
from agents_framework.teams.patterns.swarm import (
    HandoffReason,
    HandoffRequest,
    SwarmConfig,
    SwarmMessage,
    SwarmPattern,
    SwarmState,
)

from ..conftest import MockAgent


# ============================================================================
# HandoffReason Tests
# ============================================================================


class TestHandoffReason:
    """Tests for HandoffReason enum."""

    def test_all_reasons_defined(self):
        """Test that all handoff reasons are defined."""
        assert HandoffReason.CAPABILITY_MATCH.value == "capability_match"
        assert HandoffReason.EXPLICIT_REQUEST.value == "explicit_request"
        assert HandoffReason.TASK_COMPLETE.value == "task_complete"
        assert HandoffReason.UNABLE_TO_HELP.value == "unable_to_help"
        assert HandoffReason.ESCALATION.value == "escalation"


# ============================================================================
# HandoffRequest Tests
# ============================================================================


class TestHandoffRequest:
    """Tests for the HandoffRequest dataclass."""

    def test_handoff_request_creation(self):
        """Test creating a handoff request."""
        request = HandoffRequest(
            target_agent_id="agent_2",
        )

        assert request.target_agent_id == "agent_2"
        assert request.reason == HandoffReason.EXPLICIT_REQUEST
        assert request.message == ""
        assert request.context_updates == {}
        assert request.preserve_history is True

    def test_handoff_request_with_details(self):
        """Test handoff request with full details."""
        request = HandoffRequest(
            target_agent_id="specialist",
            reason=HandoffReason.CAPABILITY_MATCH,
            message="Need specialist for this task",
            context_updates={"priority": "high"},
            preserve_history=False,
        )

        assert request.target_agent_id == "specialist"
        assert request.reason == HandoffReason.CAPABILITY_MATCH
        assert request.message == "Need specialist for this task"
        assert request.context_updates == {"priority": "high"}
        assert request.preserve_history is False


# ============================================================================
# SwarmMessage Tests
# ============================================================================


class TestSwarmMessage:
    """Tests for the SwarmMessage dataclass."""

    def test_message_creation(self):
        """Test creating a swarm message."""
        message = SwarmMessage(
            sender_id="agent_1",
            sender_name="researcher",
            content="Hello, I need help",
        )

        assert message.sender_id == "agent_1"
        assert message.sender_name == "researcher"
        assert message.content == "Hello, I need help"
        assert message.role == "assistant"
        assert message.timestamp > 0
        assert message.metadata == {}

    def test_message_with_role(self):
        """Test message with specific role."""
        message = SwarmMessage(
            sender_id="user",
            sender_name="user",
            content="Initial query",
            role="user",
        )

        assert message.role == "user"

    def test_message_with_metadata(self):
        """Test message with metadata."""
        message = SwarmMessage(
            sender_id="agent_1",
            sender_name="test",
            content="Test",
            metadata={"tool_used": "search"},
        )

        assert message.metadata == {"tool_used": "search"}


# ============================================================================
# SwarmState Tests
# ============================================================================


class TestSwarmState:
    """Tests for the SwarmState dataclass."""

    def test_state_creation(self):
        """Test creating swarm state."""
        state = SwarmState()

        assert state.active_agent_id is None
        assert state.conversation_history == []
        assert state.handoff_history == []
        assert state.agent_call_counts == {}
        assert state.context_variables == {}

    def test_state_tracking(self):
        """Test state tracking during execution."""
        state = SwarmState()

        state.active_agent_id = "agent_1"
        state.conversation_history.append(
            SwarmMessage(sender_id="user", sender_name="user", content="Hello")
        )
        state.handoff_history.append({"from": "agent_1", "to": "agent_2", "reason": "explicit"})
        state.agent_call_counts["agent_1"] = 2
        state.context_variables["key"] = "value"

        assert state.active_agent_id == "agent_1"
        assert len(state.conversation_history) == 1
        assert len(state.handoff_history) == 1
        assert state.agent_call_counts["agent_1"] == 2
        assert state.context_variables["key"] == "value"


# ============================================================================
# SwarmConfig Tests
# ============================================================================


class TestSwarmConfig:
    """Tests for the SwarmConfig dataclass."""

    def test_default_config(self):
        """Test default swarm configuration."""
        config = SwarmConfig()

        assert config.max_handoffs == 10
        assert config.max_turns == 50
        assert config.allow_self_handoff is False
        assert config.default_agent_id is None
        assert config.handoff_timeout == 30.0
        assert config.require_explicit_handoff is False

    def test_custom_config(self):
        """Test custom swarm configuration."""
        config = SwarmConfig(
            max_handoffs=5,
            max_turns=20,
            allow_self_handoff=True,
            default_agent_id="default_agent",
            handoff_timeout=10.0,
            require_explicit_handoff=True,
        )

        assert config.max_handoffs == 5
        assert config.max_turns == 20
        assert config.allow_self_handoff is True
        assert config.default_agent_id == "default_agent"
        assert config.handoff_timeout == 10.0
        assert config.require_explicit_handoff is True


# ============================================================================
# SwarmPattern Creation Tests
# ============================================================================


class TestSwarmPatternCreation:
    """Tests for SwarmPattern creation."""

    def test_pattern_creation(self):
        """Test creating a swarm pattern."""
        pattern = SwarmPattern()

        assert pattern.name == "swarm"
        assert pattern.status == PatternStatus.PENDING
        assert pattern.config.max_handoffs == 10

    def test_pattern_with_config(self):
        """Test creating pattern with custom config."""
        config = SwarmConfig(max_handoffs=5)
        pattern = SwarmPattern(config=config)

        assert pattern.config.max_handoffs == 5

    def test_pattern_with_custom_handoff_function(self):
        """Test creating pattern with custom handoff function."""
        def custom_handoff(result, agent, state):
            return None

        pattern = SwarmPattern(handoff_fn=custom_handoff)

        assert pattern._handoff_fn is not None


# ============================================================================
# Agent Registration Tests
# ============================================================================


class TestAgentRegistration:
    """Tests for agent registration."""

    def test_register_agent(self, mock_researcher: MockAgent):
        """Test registering an agent."""
        pattern = SwarmPattern()

        pattern.register_agent(mock_researcher)

        assert mock_researcher.id in pattern._agent_registry
        assert mock_researcher.id in pattern._agent_capabilities

    def test_register_agent_with_capabilities(self, mock_researcher: MockAgent):
        """Test registering agent with explicit capabilities."""
        pattern = SwarmPattern()

        pattern.register_agent(mock_researcher, capabilities=["custom", "skills"])

        assert pattern._agent_capabilities[mock_researcher.id] == ["custom", "skills"]

    def test_register_agent_uses_role_capabilities(self, mock_researcher: MockAgent):
        """Test that agent uses role capabilities by default."""
        pattern = SwarmPattern()

        pattern.register_agent(mock_researcher)

        assert pattern._agent_capabilities[mock_researcher.id] == list(mock_researcher.role.capabilities)

    def test_get_agent_capabilities(self, mock_researcher: MockAgent):
        """Test getting agent capabilities."""
        pattern = SwarmPattern()
        pattern.register_agent(mock_researcher, capabilities=["cap1", "cap2"])

        capabilities = pattern.get_agent_capabilities(mock_researcher.id)

        assert capabilities == ["cap1", "cap2"]

    def test_get_agent_capabilities_nonexistent(self):
        """Test getting capabilities for non-existent agent."""
        pattern = SwarmPattern()

        capabilities = pattern.get_agent_capabilities("nonexistent")

        assert capabilities == []


# ============================================================================
# Validation Tests
# ============================================================================


class TestValidation:
    """Tests for agent validation."""

    @pytest.mark.asyncio
    async def test_validate_with_agents(self, mock_researcher: MockAgent):
        """Test validation with agents."""
        pattern = SwarmPattern()

        assert await pattern.validate_agents([mock_researcher])

    @pytest.mark.asyncio
    async def test_validate_empty_list(self):
        """Test validation with empty agent list."""
        pattern = SwarmPattern()

        assert not await pattern.validate_agents([])


# ============================================================================
# Execution Tests
# ============================================================================


class TestSwarmExecution:
    """Tests for swarm pattern execution."""

    @pytest.mark.asyncio
    async def test_basic_execution(
        self,
        mock_researcher: MockAgent,
    ):
        """Test basic swarm execution."""
        pattern = SwarmPattern()
        task = Task(description="Initial query")

        result = await pattern.execute(task, [mock_researcher])

        assert result.status == PatternStatus.COMPLETED
        assert len(result.steps) >= 1
        assert mock_researcher.run_count >= 1

    @pytest.mark.asyncio
    async def test_execution_fails_with_no_agents(self):
        """Test execution fails with no agents."""
        pattern = SwarmPattern()
        task = Task(description="Test query")

        result = await pattern.execute(task, [])

        assert result.status == PatternStatus.FAILED
        assert "No agents provided" in result.error

    @pytest.mark.asyncio
    async def test_execution_sets_initial_active_agent(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test that first agent is set as initial active agent."""
        pattern = SwarmPattern()
        task = Task(description="Test")

        await pattern.execute(task, [mock_researcher, mock_analyst])

        # First agent should have been the starting point
        assert pattern.state.active_agent_id == mock_researcher.id or \
               mock_researcher.id in [h["from"] for h in pattern.state.handoff_history]

    @pytest.mark.asyncio
    async def test_initial_message_added_to_history(self, mock_researcher: MockAgent):
        """Test that initial task is added as user message."""
        pattern = SwarmPattern()
        task = Task(description="Initial question")

        await pattern.execute(task, [mock_researcher])

        # First message should be from user
        assert len(pattern.state.conversation_history) >= 1
        assert pattern.state.conversation_history[0].sender_name == "user"
        assert pattern.state.conversation_history[0].content == "Initial question"

    @pytest.mark.asyncio
    async def test_agent_response_added_to_history(self, mock_researcher: MockAgent):
        """Test that agent response is added to conversation history."""
        mock_researcher.return_value = "Research response"
        pattern = SwarmPattern()
        task = Task(description="Query")

        await pattern.execute(task, [mock_researcher])

        # Should have user message and at least one agent response
        agent_messages = [
            msg for msg in pattern.state.conversation_history
            if msg.role == "assistant"
        ]
        assert len(agent_messages) >= 1


# ============================================================================
# Handoff Tests
# ============================================================================


class TestHandoffs:
    """Tests for handoff functionality."""

    def test_handoff_method(self):
        """Test the convenience handoff method."""
        pattern = SwarmPattern()

        request = pattern.handoff(
            target_agent_id="agent_2",
            reason=HandoffReason.ESCALATION,
            message="Need help",
            context_updates={"urgent": True},
        )

        assert request.target_agent_id == "agent_2"
        assert request.reason == HandoffReason.ESCALATION
        assert request.message == "Need help"
        assert request.context_updates == {"urgent": True}

    @pytest.mark.asyncio
    async def test_handoff_via_dict_output(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test handoff triggered by dict output with handoff_to."""
        mock_researcher.return_value = {
            "handoff_to": mock_analyst.id,
            "reason": "explicit_request",
            "message": "Need analysis",
        }

        pattern = SwarmPattern()
        task = Task(description="Query")

        result = await pattern.execute(task, [mock_researcher, mock_analyst])

        # Should have handoff in history
        assert len(pattern.state.handoff_history) >= 1

    @pytest.mark.asyncio
    async def test_handoff_via_handoff_request(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test handoff triggered by HandoffRequest object."""
        mock_researcher.return_value = HandoffRequest(
            target_agent_id=mock_analyst.id,
            reason=HandoffReason.CAPABILITY_MATCH,
        )

        pattern = SwarmPattern()
        task = Task(description="Query")

        result = await pattern.execute(task, [mock_researcher, mock_analyst])

        assert len(pattern.state.handoff_history) >= 1

    @pytest.mark.asyncio
    async def test_capability_based_handoff(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test handoff based on required capability."""
        mock_researcher.return_value = {
            "required_capability": "calculate",
        }

        pattern = SwarmPattern()
        pattern.register_agent(mock_researcher, capabilities=["search"])
        pattern.register_agent(mock_analyst, capabilities=["calculate"])

        task = Task(description="Query")
        result = await pattern.execute(task, [mock_researcher, mock_analyst])

        # Should find analyst via capability match
        assert result.status == PatternStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_handoff_to_agent_by_name(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test handoff to agent by role name."""
        mock_researcher.return_value = {
            "handoff_to": "analyst",  # Using role name
        }

        pattern = SwarmPattern()
        task = Task(description="Query")

        result = await pattern.execute(task, [mock_researcher, mock_analyst])

        # Should resolve name to agent
        assert result.status == PatternStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_self_handoff_prevented(self, mock_researcher: MockAgent):
        """Test that self-handoff is prevented by default."""
        mock_researcher.return_value = {
            "handoff_to": mock_researcher.id,  # Self-handoff
        }

        config = SwarmConfig(allow_self_handoff=False, max_turns=2)
        pattern = SwarmPattern(config=config)
        task = Task(description="Query")

        result = await pattern.execute(task, [mock_researcher])

        # Self-handoff should be ignored
        assert result.status == PatternStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_self_handoff_allowed_when_configured(self, mock_researcher: MockAgent):
        """Test that self-handoff is allowed when configured."""
        call_count = 0

        class SelfHandoffAgent(MockAgent):
            async def run(self, task):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    return TaskResult(
                        task_id="test",
                        success=True,
                        output={"handoff_to": self.id}
                    )
                return TaskResult(
                    task_id="test",
                    success=True,
                    output={"complete": True}
                )

        agent = SelfHandoffAgent(role=AgentRole(name="test", description="", capabilities=[]))

        config = SwarmConfig(allow_self_handoff=True, max_handoffs=5)
        pattern = SwarmPattern(config=config)
        task = Task(description="Query")

        result = await pattern.execute(task, [agent])

        # Should have allowed multiple self-handoffs
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_context_updates_on_handoff(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test that context is updated on handoff."""
        mock_researcher.return_value = {
            "handoff_to": mock_analyst.id,
            "context_updates": {"new_key": "new_value"},
        }

        pattern = SwarmPattern()
        task = Task(description="Query")

        result = await pattern.execute(task, [mock_researcher, mock_analyst])

        assert pattern.state.context_variables.get("new_key") == "new_value"

    @pytest.mark.asyncio
    async def test_handoff_to_default_agent(self, mock_researcher: MockAgent):
        """Test fallback to default agent when target not found."""
        mock_researcher.return_value = {
            "handoff_to": "nonexistent_agent",
        }

        config = SwarmConfig(default_agent_id=mock_researcher.id)
        pattern = SwarmPattern(config=config)
        task = Task(description="Query")

        result = await pattern.execute(task, [mock_researcher])

        # Should fall back to default
        assert result.status == PatternStatus.COMPLETED


# ============================================================================
# Limits Tests
# ============================================================================


class TestLimits:
    """Tests for handoff and turn limits."""

    @pytest.mark.asyncio
    async def test_max_handoffs_limit(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test that max handoffs limit is enforced."""
        # Create agents that constantly handoff to each other
        class HandoffAgent(MockAgent):
            async def run(self, task):
                target = mock_analyst.id if self.id == mock_researcher.id else mock_researcher.id
                return {"handoff_to": target}

        agent1 = HandoffAgent(role=AgentRole(name="a1", description="", capabilities=[]))
        agent2 = HandoffAgent(role=AgentRole(name="a2", description="", capabilities=[]))

        config = SwarmConfig(max_handoffs=3, max_turns=100)
        pattern = SwarmPattern(config=config)
        task = Task(description="Query")

        result = await pattern.execute(task, [agent1, agent2])

        # Should stop after max handoffs
        assert len(pattern.state.handoff_history) <= 3

    @pytest.mark.asyncio
    async def test_max_turns_limit(self, mock_researcher: MockAgent):
        """Test that max turns limit is enforced."""
        call_count = 0

        class InfiniteAgent(MockAgent):
            async def run(self, task):
                nonlocal call_count
                call_count += 1
                return "Response"  # No handoff, no completion signal

        agent = InfiniteAgent(role=AgentRole(name="test", description="", capabilities=[]))

        config = SwarmConfig(max_turns=5)
        pattern = SwarmPattern(config=config)
        task = Task(description="Query")

        result = await pattern.execute(task, [agent])

        # Should stop after max turns
        assert call_count <= 5


# ============================================================================
# Completion Detection Tests
# ============================================================================


class TestCompletionDetection:
    """Tests for conversation completion detection."""

    @pytest.mark.asyncio
    async def test_completion_via_complete_flag(self, mock_researcher: MockAgent):
        """Test completion via complete flag in output."""
        mock_researcher.return_value = {"complete": True, "result": "done"}

        pattern = SwarmPattern()
        task = Task(description="Query")

        result = await pattern.execute(task, [mock_researcher])

        assert result.status == PatternStatus.COMPLETED
        assert mock_researcher.run_count == 1

    @pytest.mark.asyncio
    async def test_completion_via_end_conversation(self, mock_researcher: MockAgent):
        """Test completion via end_conversation action."""
        mock_researcher.return_value = {"action": "end_conversation"}

        pattern = SwarmPattern()
        task = Task(description="Query")

        result = await pattern.execute(task, [mock_researcher])

        assert result.status == PatternStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_completion_via_context_flag(self, mock_researcher: MockAgent):
        """Test completion via context flag."""
        class ContextCompleteAgent(MockAgent):
            async def run(self, task):
                return TaskResult(task_id="test", success=True, output="Done")

        agent = ContextCompleteAgent(role=AgentRole(name="test", description="", capabilities=[]))

        pattern = SwarmPattern()
        context = PatternContext(variables={"conversation_complete": True})
        task = Task(description="Query")

        result = await pattern.execute(task, [agent], context)

        assert result.status == PatternStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_completion_on_failure(self, failing_agent: MockAgent):
        """Test that failure triggers completion."""
        pattern = SwarmPattern()
        task = Task(description="Query")

        result = await pattern.execute(task, [failing_agent])

        # Should complete (not hang) on failure
        assert result.status == PatternStatus.COMPLETED


# ============================================================================
# Custom Handoff Function Tests
# ============================================================================


class TestCustomHandoffFunction:
    """Tests for custom handoff function."""

    @pytest.mark.asyncio
    async def test_custom_handoff_function_used(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test that custom handoff function is used."""
        handoff_called = False

        def custom_handoff(result, agent, state):
            nonlocal handoff_called
            handoff_called = True
            return HandoffRequest(target_agent_id=mock_analyst.id)

        pattern = SwarmPattern(handoff_fn=custom_handoff)
        task = Task(description="Query")

        await pattern.execute(task, [mock_researcher, mock_analyst])

        assert handoff_called

    @pytest.mark.asyncio
    async def test_custom_handoff_returns_none(self, mock_researcher: MockAgent):
        """Test custom handoff returning None (no handoff)."""
        def custom_handoff(result, agent, state):
            return None  # No handoff

        pattern = SwarmPattern(handoff_fn=custom_handoff)
        config = SwarmConfig(max_turns=2)
        pattern = SwarmPattern(config=config, handoff_fn=custom_handoff)
        task = Task(description="Query")

        result = await pattern.execute(task, [mock_researcher])

        assert result.status == PatternStatus.COMPLETED


# ============================================================================
# Output and History Tests
# ============================================================================


class TestOutputAndHistory:
    """Tests for output building and history."""

    @pytest.mark.asyncio
    async def test_final_output_structure(self, mock_researcher: MockAgent):
        """Test structure of final output."""
        mock_researcher.return_value = "Final answer"

        pattern = SwarmPattern()
        task = Task(description="Query")

        result = await pattern.execute(task, [mock_researcher])

        assert "messages" in result.final_output
        assert "final_response" in result.final_output
        assert "context_variables" in result.final_output

    @pytest.mark.asyncio
    async def test_get_conversation_history(self, mock_researcher: MockAgent):
        """Test getting conversation history."""
        pattern = SwarmPattern()
        task = Task(description="Query")

        await pattern.execute(task, [mock_researcher])

        history = pattern.get_conversation_history()

        assert len(history) >= 1
        assert isinstance(history[0], SwarmMessage)

    @pytest.mark.asyncio
    async def test_conversation_history_is_copy(self, mock_researcher: MockAgent):
        """Test that get_conversation_history returns a copy."""
        pattern = SwarmPattern()
        task = Task(description="Query")

        await pattern.execute(task, [mock_researcher])

        history1 = pattern.get_conversation_history()
        history1.append(SwarmMessage(sender_id="x", sender_name="x", content="x"))

        history2 = pattern.get_conversation_history()

        # Original should be unchanged
        assert len(history2) < len(history1)

    @pytest.mark.asyncio
    async def test_result_metadata(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test result contains execution metadata."""
        mock_researcher.return_value = {"handoff_to": mock_analyst.id}
        mock_analyst.return_value = {"complete": True}

        pattern = SwarmPattern()
        task = Task(description="Query")

        result = await pattern.execute(task, [mock_researcher, mock_analyst])

        assert "handoff_count" in result.metadata
        assert "turn_count" in result.metadata
        assert "agent_calls" in result.metadata
        assert "handoff_history" in result.metadata


# ============================================================================
# State Management Tests
# ============================================================================


class TestStateManagement:
    """Tests for state management."""

    def test_state_property(self):
        """Test state property returns current state."""
        pattern = SwarmPattern()

        state = pattern.state
        assert isinstance(state, SwarmState)

    def test_reset_state(self, mock_researcher: MockAgent):
        """Test resetting swarm state."""
        pattern = SwarmPattern()
        pattern._state.active_agent_id = "agent_1"
        pattern._state.conversation_history.append(
            SwarmMessage(sender_id="x", sender_name="x", content="x")
        )
        pattern._status = PatternStatus.COMPLETED

        pattern.reset_state()

        assert pattern._state.active_agent_id is None
        assert pattern._state.conversation_history == []
        assert pattern.status == PatternStatus.PENDING


# ============================================================================
# Timeout Tests
# ============================================================================


class TestTimeoutHandling:
    """Tests for timeout handling."""

    @pytest.mark.asyncio
    async def test_agent_timeout(self, slow_agent: MockAgent):
        """Test that slow agents timeout."""
        config = SwarmConfig(handoff_timeout=0.1)
        pattern = SwarmPattern(config=config)
        task = Task(description="Query")

        result = await pattern.execute(task, [slow_agent])

        # Should handle timeout gracefully
        assert result.status == PatternStatus.COMPLETED
        # Should have error in steps or result
        assert any(not step.success for step in result.steps)


# ============================================================================
# Exception Handling Tests
# ============================================================================


class TestExceptionHandling:
    """Tests for exception handling."""

    @pytest.mark.asyncio
    async def test_exception_caught(self):
        """Test that exceptions are caught and returned as failed result."""
        class ExceptionAgent(MockAgent):
            async def run(self, task):
                raise RuntimeError("Unexpected error")

        agent = ExceptionAgent(role=AgentRole(name="error", description="", capabilities=[]))
        pattern = SwarmPattern()
        task = Task(description="Query")

        result = await pattern.execute(task, [agent])

        assert result.status == PatternStatus.FAILED
        assert "Unexpected error" in result.error


# ============================================================================
# Full Workflow Tests
# ============================================================================


class TestFullWorkflow:
    """Tests for complete swarm workflow."""

    @pytest.mark.asyncio
    async def test_multi_agent_conversation(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
        mock_writer: MockAgent,
    ):
        """Test multi-agent conversation flow."""
        mock_researcher.return_value = {"handoff_to": mock_analyst.id}
        mock_analyst.return_value = {"handoff_to": mock_writer.id}
        mock_writer.return_value = {"complete": True, "result": "Final document"}

        pattern = SwarmPattern()
        task = Task(description="Create a research document")

        result = await pattern.execute(task, [mock_researcher, mock_analyst, mock_writer])

        assert result.status == PatternStatus.COMPLETED
        assert mock_researcher.run_count >= 1
        assert mock_analyst.run_count >= 1
        assert mock_writer.run_count >= 1

    @pytest.mark.asyncio
    async def test_complex_handoff_chain(self):
        """Test complex chain of handoffs with context updates."""
        class ChainAgent(MockAgent):
            def __init__(self, role, next_agent_id=None):
                super().__init__(role=role)
                self.next_agent_id = next_agent_id

            async def run(self, task):
                if self.next_agent_id:
                    return TaskResult(
                        task_id="test",
                        success=True,
                        output={
                            "handoff_to": self.next_agent_id,
                            "context_updates": {f"step_{self.role.name}": "complete"},
                        }
                    )
                return TaskResult(task_id="test", success=True, output={"complete": True})

        agent1 = ChainAgent(role=AgentRole(name="step1", description="", capabilities=[]))
        agent2 = ChainAgent(role=AgentRole(name="step2", description="", capabilities=[]))
        agent3 = ChainAgent(role=AgentRole(name="step3", description="", capabilities=[]))

        agent1.next_agent_id = agent2.id
        agent2.next_agent_id = agent3.id

        pattern = SwarmPattern()
        task = Task(description="Process through chain")

        result = await pattern.execute(task, [agent1, agent2, agent3])

        assert result.status == PatternStatus.COMPLETED
        assert pattern.state.context_variables.get("step_step1") == "complete"
        assert pattern.state.context_variables.get("step_step2") == "complete"
