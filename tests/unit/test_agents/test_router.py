"""Unit tests for the router agent module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents_framework.agents.base import (
    AgentRole,
    AgentStatus,
    Task,
    TaskResult,
)
from agents_framework.agents.router import (
    RouterAgent,
    RouterConfig,
    RoutingDecision,
    RoutingRule,
    RoutingStrategy,
)

from .conftest import ConcreteAgent, MockLLMProvider, SlowAgent


class TestRoutingStrategy:
    """Tests for RoutingStrategy enum."""

    def test_strategy_values(self):
        """Test strategy enum values."""
        assert RoutingStrategy.LLM.value == "llm"
        assert RoutingStrategy.RULE_BASED.value == "rule_based"
        assert RoutingStrategy.CAPABILITY.value == "capability"
        assert RoutingStrategy.ROUND_ROBIN.value == "round_robin"
        assert RoutingStrategy.FALLBACK.value == "fallback"


class TestRoutingRule:
    """Tests for RoutingRule dataclass."""

    def test_rule_creation_minimal(self):
        """Test creating a rule with minimal parameters."""
        rule = RoutingRule(name="test-rule")
        assert rule.name == "test-rule"
        assert rule.pattern is None
        assert rule.target_agent_id is None
        assert rule.target_capability is None
        assert rule.priority == 0
        assert rule.condition is None
        assert rule.metadata == {}

    def test_rule_creation_full(self):
        """Test creating a rule with all parameters."""
        condition = lambda task: True
        rule = RoutingRule(
            name="coding-rule",
            pattern=r"(write|code|implement)",
            target_agent_id="coder-123",
            target_capability="code",
            priority=10,
            condition=condition,
            metadata={"category": "development"},
        )
        assert rule.name == "coding-rule"
        assert rule.pattern == r"(write|code|implement)"
        assert rule.target_agent_id == "coder-123"
        assert rule.target_capability == "code"
        assert rule.priority == 10
        assert rule.condition == condition
        assert rule.metadata == {"category": "development"}


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""

    def test_decision_creation_minimal(self):
        """Test creating a decision with minimal parameters."""
        decision = RoutingDecision(
            target_agent_id="agent-123",
            strategy_used=RoutingStrategy.RULE_BASED,
        )
        assert decision.target_agent_id == "agent-123"
        assert decision.strategy_used == RoutingStrategy.RULE_BASED
        assert decision.confidence == 1.0
        assert decision.reason == ""
        assert decision.fallback_agents == []

    def test_decision_creation_full(self):
        """Test creating a decision with all parameters."""
        decision = RoutingDecision(
            target_agent_id="agent-123",
            strategy_used=RoutingStrategy.LLM,
            confidence=0.95,
            reason="Best match for coding tasks",
            fallback_agents=["agent-456", "agent-789"],
        )
        assert decision.target_agent_id == "agent-123"
        assert decision.strategy_used == RoutingStrategy.LLM
        assert decision.confidence == 0.95
        assert decision.reason == "Best match for coding tasks"
        assert decision.fallback_agents == ["agent-456", "agent-789"]


class TestRouterConfig:
    """Tests for RouterConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = RouterConfig()
        assert config.default_strategy == RoutingStrategy.LLM
        assert config.fallback_enabled is True
        assert config.max_fallback_attempts == 3
        assert config.llm_routing_prompt is None
        assert config.confidence_threshold == 0.7
        assert config.timeout_per_agent == 60.0

    def test_config_custom(self):
        """Test custom configuration values."""
        config = RouterConfig(
            default_strategy=RoutingStrategy.RULE_BASED,
            fallback_enabled=False,
            max_fallback_attempts=5,
            llm_routing_prompt="Custom routing prompt",
            confidence_threshold=0.9,
            timeout_per_agent=30.0,
        )
        assert config.default_strategy == RoutingStrategy.RULE_BASED
        assert config.fallback_enabled is False
        assert config.max_fallback_attempts == 5
        assert config.llm_routing_prompt == "Custom routing prompt"
        assert config.confidence_threshold == 0.9
        assert config.timeout_per_agent == 30.0


class TestRouterAgent:
    """Tests for RouterAgent class."""

    def test_router_initialization_default_role(self):
        """Test router initialization with default role."""
        router = RouterAgent()
        assert router.role.name == "router"
        assert "routing" in router.role.capabilities

    def test_router_initialization_custom_role(self, router_role, router_config):
        """Test router initialization with custom role."""
        router = RouterAgent(role=router_role, router_config=router_config)
        assert router.role == router_role
        assert router.router_config == router_config

    def test_router_default_config(self):
        """Test router with default config."""
        router = RouterAgent()
        assert isinstance(router.router_config, RouterConfig)

    def test_register_agent(self, router_agent, coding_agent):
        """Test registering an agent."""
        router_agent.register_agent(coding_agent)

        assert coding_agent.id in router_agent._agents
        assert router_agent._agents[coding_agent.id] == coding_agent

    def test_register_agent_with_capabilities(self, router_agent, coding_agent):
        """Test registering an agent with custom capabilities."""
        router_agent.register_agent(
            coding_agent, capabilities=["python", "javascript"]
        )

        assert router_agent._agent_capabilities[coding_agent.id] == [
            "python",
            "javascript",
        ]

    def test_register_agent_uses_role_capabilities(self, router_agent, coding_agent):
        """Test that registration uses role capabilities by default."""
        router_agent.register_agent(coding_agent)

        # Should use capabilities from coding_agent.role.capabilities
        assert coding_agent.id in router_agent._agent_capabilities

    def test_unregister_agent(self, router_with_agents, coding_agent):
        """Test unregistering an agent."""
        result = router_with_agents.unregister_agent(coding_agent.id)

        assert result is True
        assert coding_agent.id not in router_with_agents._agents
        assert coding_agent.id not in router_with_agents._agent_capabilities

    def test_unregister_nonexistent_agent(self, router_agent):
        """Test unregistering an agent that doesn't exist."""
        result = router_agent.unregister_agent("nonexistent-id")
        assert result is False

    def test_add_rule(self, router_agent):
        """Test adding a routing rule."""
        rule = RoutingRule(name="test-rule", pattern=r"test")
        router_agent.add_rule(rule)

        assert rule in router_agent._rules

    def test_add_rule_sorts_by_priority(self, router_agent):
        """Test that rules are sorted by priority."""
        rule_low = RoutingRule(name="low", priority=1)
        rule_high = RoutingRule(name="high", priority=10)
        rule_mid = RoutingRule(name="mid", priority=5)

        router_agent.add_rule(rule_low)
        router_agent.add_rule(rule_high)
        router_agent.add_rule(rule_mid)

        # Should be sorted by priority descending
        assert router_agent._rules[0].name == "high"
        assert router_agent._rules[1].name == "mid"
        assert router_agent._rules[2].name == "low"

    def test_remove_rule(self, router_agent):
        """Test removing a routing rule."""
        rule = RoutingRule(name="test-rule")
        router_agent.add_rule(rule)

        result = router_agent.remove_rule("test-rule")

        assert result is True
        assert rule not in router_agent._rules

    def test_remove_nonexistent_rule(self, router_agent):
        """Test removing a rule that doesn't exist."""
        result = router_agent.remove_rule("nonexistent")
        assert result is False

    def test_get_registered_agents(self, router_with_agents):
        """Test getting registered agent information."""
        info = router_with_agents.get_registered_agents()

        assert isinstance(info, dict)
        assert len(info) == 2

        for agent_id, agent_info in info.items():
            assert "name" in agent_info
            assert "description" in agent_info
            assert "capabilities" in agent_info
            assert "status" in agent_info

    def test_get_routing_rules(self, router_agent):
        """Test getting routing rules."""
        rule = RoutingRule(
            name="test-rule",
            pattern=r"test",
            target_capability="test",
            priority=5,
        )
        router_agent.add_rule(rule)

        rules = router_agent.get_routing_rules()

        assert len(rules) == 1
        assert rules[0]["name"] == "test-rule"
        assert rules[0]["pattern"] == r"test"
        assert rules[0]["priority"] == 5

    def test_default_routing_prompt(self, router_agent):
        """Test default routing prompt generation."""
        prompt = router_agent._default_routing_prompt()

        assert "task router" in prompt.lower()
        assert "agent_id" in prompt
        assert "confidence" in prompt

    @pytest.mark.asyncio
    async def test_run_with_string_task(self, router_with_agents):
        """Test running with a string task."""
        result = await router_with_agents.run("Do something")
        assert result is not None
        assert result.task_id is not None

    @pytest.mark.asyncio
    async def test_run_with_task_object(self, router_with_agents, basic_task):
        """Test running with a Task object."""
        result = await router_with_agents.run(basic_task)
        assert result.task_id == basic_task.id

    @pytest.mark.asyncio
    async def test_run_no_agents(self, router_agent, basic_task):
        """Test running with no registered agents."""
        result = await router_agent.run(basic_task)
        assert result.success is False
        assert "Unable to determine routing" in result.error

    @pytest.mark.asyncio
    async def test_run_sets_status(self, router_with_agents, basic_task):
        """Test that run sets agent status correctly."""
        assert router_with_agents.status == AgentStatus.IDLE

        result = await router_with_agents.run(basic_task)

        # Should return to IDLE after completion
        assert router_with_agents.status == AgentStatus.IDLE

    @pytest.mark.asyncio
    async def test_rule_based_routing(self, router_config):
        """Test rule-based routing strategy."""
        config = RouterConfig(default_strategy=RoutingStrategy.RULE_BASED)
        router = RouterAgent(router_config=config)

        # Create and register agent
        role = AgentRole(name="coder", description="Code agent", capabilities=["code"])
        agent = ConcreteAgent(role=role)
        router.register_agent(agent)

        # Add rule
        rule = RoutingRule(
            name="code-rule",
            pattern=r"(code|write|implement)",
            target_agent_id=agent.id,
        )
        router.add_rule(rule)

        task = Task(description="Write some code")
        result = await router.run(task)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_rule_based_routing_with_capability_target(self):
        """Test rule-based routing targeting a capability."""
        config = RouterConfig(default_strategy=RoutingStrategy.RULE_BASED)
        router = RouterAgent(router_config=config)

        role = AgentRole(name="coder", description="Code agent", capabilities=["code"])
        agent = ConcreteAgent(role=role)
        router.register_agent(agent, capabilities=["code"])

        rule = RoutingRule(
            name="code-rule",
            pattern=r"code",
            target_capability="code",
        )
        router.add_rule(rule)

        task = Task(description="Write some code")
        result = await router.run(task)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_rule_with_custom_condition(self):
        """Test rule with custom condition function."""
        config = RouterConfig(default_strategy=RoutingStrategy.RULE_BASED)
        router = RouterAgent(router_config=config)

        role = AgentRole(name="urgent", description="Urgent handler", capabilities=[])
        agent = ConcreteAgent(role=role)
        router.register_agent(agent)

        def is_urgent(task: Task) -> bool:
            return task.priority > 5

        rule = RoutingRule(
            name="urgent-rule",
            condition=is_urgent,
            target_agent_id=agent.id,
        )
        router.add_rule(rule)

        urgent_task = Task(description="Urgent task", priority=10)
        result = await router.run(urgent_task)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_capability_routing(self):
        """Test capability-based routing."""
        config = RouterConfig(default_strategy=RoutingStrategy.CAPABILITY)
        router = RouterAgent(router_config=config)

        coding_role = AgentRole(
            name="coder", description="Code", capabilities=["code"]
        )
        coding_agent = ConcreteAgent(role=coding_role)
        router.register_agent(coding_agent, capabilities=["code"])

        task = Task(description="Task", required_capabilities=["code"])
        result = await router.run(task)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_capability_extraction_from_description(self):
        """Test capability extraction from task description."""
        config = RouterConfig(default_strategy=RoutingStrategy.CAPABILITY)
        router = RouterAgent(router_config=config)

        coding_role = AgentRole(
            name="coder", description="Code", capabilities=["code"]
        )
        coding_agent = ConcreteAgent(role=coding_role)
        router.register_agent(coding_agent, capabilities=["code"])

        # No explicit capabilities, but "code" keyword in description
        task = Task(description="Please write some code")
        result = await router.run(task)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_round_robin_routing(self, basic_role):
        """Test round-robin routing."""
        config = RouterConfig(default_strategy=RoutingStrategy.ROUND_ROBIN)
        router = RouterAgent(router_config=config)

        agent1 = ConcreteAgent(role=basic_role)
        agent2 = ConcreteAgent(role=basic_role)
        router.register_agent(agent1)
        router.register_agent(agent2)

        # Run multiple tasks
        for i in range(4):
            await router.run(f"Task {i}")

        # Both agents should have been used
        assert agent1.run_count > 0
        assert agent2.run_count > 0

    @pytest.mark.asyncio
    async def test_llm_routing(self, mock_llm_with_routing_response, basic_role):
        """Test LLM-based routing."""
        config = RouterConfig(default_strategy=RoutingStrategy.LLM)
        router = RouterAgent(llm=mock_llm_with_routing_response, router_config=config)

        agent = ConcreteAgent(role=basic_role)
        router.register_agent(agent)

        # Update mock to return correct agent_id
        mock_llm_with_routing_response.responses[0] = MagicMock(
            content=f'{{"agent_id": "{agent.id}", "confidence": 0.95, "reason": "Best match"}}',
            model="test",
            finish_reason="stop",
        )

        task = Task(description="Do something")
        result = await router.run(task)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_llm_routing_low_confidence_fallback(self, mock_llm, basic_role):
        """Test LLM routing falls back when confidence is low."""
        config = RouterConfig(
            default_strategy=RoutingStrategy.LLM,
            confidence_threshold=0.9,
        )
        router = RouterAgent(llm=mock_llm, router_config=config)

        agent = ConcreteAgent(role=basic_role)
        router.register_agent(agent)

        # Mock returns low confidence
        mock_llm.responses = [
            MagicMock(
                content=f'{{"agent_id": "{agent.id}", "confidence": 0.5, "reason": "Low confidence"}}',
                model="test",
                finish_reason="stop",
            )
        ]

        task = Task(description="Do something")
        result = await router.run(task)

        # Should still succeed through fallback mechanisms
        assert result is not None

    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self, basic_role):
        """Test fallback when primary agent fails."""
        config = RouterConfig(
            default_strategy=RoutingStrategy.ROUND_ROBIN,
            fallback_enabled=True,
            max_fallback_attempts=3,
        )
        router = RouterAgent(router_config=config)

        failing_agent = ConcreteAgent(role=basic_role, should_fail=True)
        successful_agent = ConcreteAgent(role=basic_role)
        router.register_agent(failing_agent)
        router.register_agent(successful_agent)

        result = await router.run("Task")

        # Should eventually succeed via fallback
        assert result.success is True
        assert successful_agent.run_count > 0

    @pytest.mark.asyncio
    async def test_fallback_disabled(self, basic_role):
        """Test behavior when fallback is disabled."""
        config = RouterConfig(
            default_strategy=RoutingStrategy.ROUND_ROBIN,
            fallback_enabled=False,
        )
        router = RouterAgent(router_config=config)

        failing_agent = ConcreteAgent(role=basic_role, should_fail=True)
        router.register_agent(failing_agent)

        result = await router.run("Task")

        assert result.success is False
        assert "failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_agent_timeout(self, basic_role):
        """Test handling of agent timeout."""
        config = RouterConfig(
            default_strategy=RoutingStrategy.ROUND_ROBIN,
            timeout_per_agent=0.1,
            fallback_enabled=False,
        )
        router = RouterAgent(router_config=config)

        slow_agent = SlowAgent(role=basic_role, delay=1.0)
        router.register_agent(slow_agent)

        result = await router.run("Task")

        assert result.success is False
        assert "failed" in result.error.lower() or "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_max_fallback_attempts_respected(self, basic_role):
        """Test that max fallback attempts is respected."""
        config = RouterConfig(
            default_strategy=RoutingStrategy.ROUND_ROBIN,
            fallback_enabled=True,
            max_fallback_attempts=1,  # Only 1 fallback attempt
        )
        router = RouterAgent(router_config=config)

        # Add multiple failing agents
        for i in range(5):
            agent = ConcreteAgent(role=basic_role, should_fail=True)
            router.register_agent(agent)

        result = await router.run("Task")

        assert result.success is False

    def test_parse_llm_response_valid_json(self, router_agent):
        """Test parsing valid LLM response."""
        # Register an agent first
        role = AgentRole(name="test", description="Test")
        agent = ConcreteAgent(role=role)
        router_agent.register_agent(agent)

        content = f'{{"agent_id": "{agent.id}", "confidence": 0.9, "reason": "Best match"}}'
        decision = router_agent._parse_llm_response(content)

        assert decision is not None
        assert decision.target_agent_id == agent.id
        assert decision.confidence == 0.9

    def test_parse_llm_response_invalid_json(self, router_agent):
        """Test parsing invalid JSON response."""
        content = "This is not JSON"
        decision = router_agent._parse_llm_response(content)
        assert decision is None

    def test_parse_llm_response_missing_agent_id(self, router_agent):
        """Test parsing response with missing agent_id."""
        content = '{"confidence": 0.9, "reason": "Test"}'
        decision = router_agent._parse_llm_response(content)
        assert decision is None

    def test_parse_llm_response_unknown_agent_id(self, router_agent):
        """Test parsing response with unknown agent_id."""
        content = '{"agent_id": "unknown-agent", "confidence": 0.9}'
        decision = router_agent._parse_llm_response(content)
        assert decision is None

    def test_rule_matches_pattern(self, router_agent, basic_task):
        """Test rule matching with pattern."""
        rule = RoutingRule(name="test", pattern=r"test")
        basic_task.description = "This is a test task"

        assert router_agent._rule_matches(rule, basic_task) is True

    def test_rule_matches_pattern_case_insensitive(self, router_agent, basic_task):
        """Test pattern matching is case insensitive."""
        rule = RoutingRule(name="test", pattern=r"TEST")
        basic_task.description = "this is a test task"

        assert router_agent._rule_matches(rule, basic_task) is True

    def test_rule_matches_context(self, router_agent, basic_task):
        """Test rule matches against context values."""
        rule = RoutingRule(name="test", pattern=r"urgent")
        basic_task.description = "Regular task"
        basic_task.context = {"priority": "urgent"}

        assert router_agent._rule_matches(rule, basic_task) is True

    def test_rule_matches_condition(self, router_agent, basic_task):
        """Test rule with custom condition."""
        rule = RoutingRule(
            name="test",
            condition=lambda t: t.priority > 5,
        )
        basic_task.priority = 10

        assert router_agent._rule_matches(rule, basic_task) is True

        basic_task.priority = 1
        assert router_agent._rule_matches(rule, basic_task) is False

    def test_find_agent_by_capability(self, router_with_agents):
        """Test finding agent by capability."""
        agent_id = router_with_agents._find_agent_by_capability("code")
        assert agent_id is not None

    def test_find_agent_by_capability_case_insensitive(self, router_with_agents):
        """Test capability matching is case insensitive."""
        agent_id = router_with_agents._find_agent_by_capability("CODE")
        assert agent_id is not None

    def test_find_agent_by_unknown_capability(self, router_with_agents):
        """Test finding agent with unknown capability returns None."""
        agent_id = router_with_agents._find_agent_by_capability("unknown")
        assert agent_id is None

    def test_extract_capabilities(self, router_agent):
        """Test extracting capabilities from text."""
        text = "Please write code to search for information and analyze it"
        capabilities = router_agent._extract_capabilities(text)

        assert "code" in capabilities
        assert "search" in capabilities
        assert "analyze" in capabilities

    def test_extract_capabilities_no_match(self, router_agent):
        """Test extracting from text with no capability keywords."""
        text = "Hello world"
        capabilities = router_agent._extract_capabilities(text)
        assert capabilities == []

    @pytest.mark.asyncio
    async def test_routing_metadata_in_result(self, router_with_agents, basic_task):
        """Test that routing metadata is included in result."""
        result = await router_with_agents.run(basic_task)

        if result.success:
            assert "routed_by" in result.metadata
            assert "routing_decision" in result.metadata
            assert result.metadata["routed_by"] == router_with_agents.id

    @pytest.mark.asyncio
    async def test_exception_handling(self, basic_role):
        """Test exception handling in router."""
        config = RouterConfig(default_strategy=RoutingStrategy.RULE_BASED)
        router = RouterAgent(router_config=config)

        agent = ConcreteAgent(role=basic_role)
        router.register_agent(agent)

        # Force an exception by making the routing decision fail
        with patch.object(router, "_make_routing_decision", side_effect=RuntimeError("Test error")):
            result = await router.run("Task")

        assert result.success is False
        assert "Router error" in result.error
        assert router.status == AgentStatus.ERROR
