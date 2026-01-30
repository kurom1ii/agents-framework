"""Unit tests for the agent spawner module."""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents_framework.agents.base import (
    AgentConfig,
    AgentRole,
    AgentStatus,
    BaseAgent,
    Task,
    TaskResult,
)
from agents_framework.agents.spawner import (
    AgentLifecycleState,
    AgentSpawner,
    AgentTemplate,
    SpawnedAgentInfo,
    SpawnPolicy,
)

from .conftest import ConcreteAgent


class TestSpawnPolicy:
    """Tests for SpawnPolicy enum."""

    def test_policy_values(self):
        """Test policy enum values."""
        assert SpawnPolicy.ON_DEMAND.value == "on_demand"
        assert SpawnPolicy.POOLED.value == "pooled"
        assert SpawnPolicy.SINGLETON.value == "singleton"


class TestAgentLifecycleState:
    """Tests for AgentLifecycleState enum."""

    def test_state_values(self):
        """Test lifecycle state enum values."""
        assert AgentLifecycleState.INITIALIZING.value == "initializing"
        assert AgentLifecycleState.RUNNING.value == "running"
        assert AgentLifecycleState.SUSPENDED.value == "suspended"
        assert AgentLifecycleState.TERMINATING.value == "terminating"
        assert AgentLifecycleState.TERMINATED.value == "terminated"


class TestAgentTemplate:
    """Tests for AgentTemplate dataclass."""

    def test_template_creation_minimal(self, basic_role):
        """Test creating a template with minimal parameters."""
        template = AgentTemplate(
            name="test-template",
            role=basic_role,
            agent_class=ConcreteAgent,
        )
        assert template.name == "test-template"
        assert template.role == basic_role
        assert template.agent_class == ConcreteAgent
        assert template.config is None
        assert template.llm_factory is None
        assert template.spawn_policy == SpawnPolicy.ON_DEMAND
        assert template.pool_size == 3
        assert template.max_instances == 0
        assert template.metadata == {}

    def test_template_creation_full(self, basic_role, basic_config):
        """Test creating a template with all parameters."""
        llm_factory = lambda: MagicMock()
        template = AgentTemplate(
            name="full-template",
            role=basic_role,
            agent_class=ConcreteAgent,
            config=basic_config,
            llm_factory=llm_factory,
            spawn_policy=SpawnPolicy.POOLED,
            pool_size=5,
            max_instances=10,
            metadata={"version": "1.0"},
        )
        assert template.name == "full-template"
        assert template.config == basic_config
        assert template.llm_factory == llm_factory
        assert template.spawn_policy == SpawnPolicy.POOLED
        assert template.pool_size == 5
        assert template.max_instances == 10
        assert template.metadata == {"version": "1.0"}


class TestSpawnedAgentInfo:
    """Tests for SpawnedAgentInfo dataclass."""

    def test_info_creation_minimal(self):
        """Test creating info with minimal parameters."""
        info = SpawnedAgentInfo(
            agent_id="agent-123",
            template_name="test-template",
        )
        assert info.agent_id == "agent-123"
        assert info.template_name == "test-template"
        assert info.lifecycle_state == AgentLifecycleState.INITIALIZING
        assert isinstance(info.spawned_at, datetime)
        assert info.last_used is None
        assert info.task_count == 0
        assert info.parent_spawner_id is None

    def test_info_creation_full(self):
        """Test creating info with all parameters."""
        spawned_at = datetime.now()
        last_used = datetime.now()
        info = SpawnedAgentInfo(
            agent_id="agent-123",
            template_name="test-template",
            lifecycle_state=AgentLifecycleState.RUNNING,
            spawned_at=spawned_at,
            last_used=last_used,
            task_count=5,
            parent_spawner_id="spawner-456",
        )
        assert info.lifecycle_state == AgentLifecycleState.RUNNING
        assert info.spawned_at == spawned_at
        assert info.last_used == last_used
        assert info.task_count == 5
        assert info.parent_spawner_id == "spawner-456"


class TestAgentSpawner:
    """Tests for AgentSpawner class."""

    def test_spawner_initialization(self):
        """Test spawner initialization."""
        spawner = AgentSpawner()
        assert spawner.id is not None
        assert spawner._registry is None
        assert spawner._default_llm_factory is None
        assert spawner._templates == {}
        assert spawner._agents == {}
        assert spawner._agent_info == {}
        assert spawner._pools == {}
        assert spawner._instance_counts == {}
        assert spawner._singletons == {}

    def test_spawner_with_registry(self):
        """Test spawner with registry."""
        registry = MagicMock()
        spawner = AgentSpawner(registry=registry)
        assert spawner._registry == registry

    def test_spawner_with_llm_factory(self):
        """Test spawner with default LLM factory."""
        factory = lambda: MagicMock()
        spawner = AgentSpawner(default_llm_factory=factory)
        assert spawner._default_llm_factory == factory

    def test_register_template(self, agent_spawner, worker_template):
        """Test registering a template."""
        agent_spawner.register_template(worker_template)

        assert worker_template.name in agent_spawner._templates
        assert agent_spawner._templates[worker_template.name] == worker_template
        assert agent_spawner._instance_counts[worker_template.name] == 0

    def test_register_pooled_template(self, agent_spawner, pooled_template):
        """Test registering a pooled template creates pool."""
        agent_spawner.register_template(pooled_template)

        assert pooled_template.name in agent_spawner._pools
        assert agent_spawner._pools[pooled_template.name] == []

    def test_register_duplicate_template_raises(
        self, agent_spawner, worker_template
    ):
        """Test registering a duplicate template raises error."""
        agent_spawner.register_template(worker_template)

        with pytest.raises(ValueError) as exc:
            agent_spawner.register_template(worker_template)

        assert "already registered" in str(exc.value)

    def test_update_template(self, agent_spawner, worker_template, basic_role):
        """Test updating a template."""
        agent_spawner.register_template(worker_template)

        updated_template = AgentTemplate(
            name=worker_template.name,
            role=basic_role,
            agent_class=ConcreteAgent,
            max_instances=5,
        )
        agent_spawner.update_template(updated_template)

        assert agent_spawner._templates[worker_template.name].max_instances == 5

    def test_unregister_template(self, agent_spawner, worker_template):
        """Test unregistering a template."""
        agent_spawner.register_template(worker_template)

        result = agent_spawner.unregister_template(worker_template.name)

        assert result == worker_template
        assert worker_template.name not in agent_spawner._templates

    def test_unregister_nonexistent_template(self, agent_spawner):
        """Test unregistering a template that doesn't exist."""
        result = agent_spawner.unregister_template("nonexistent")
        assert result is None

    def test_get_template(self, agent_spawner, worker_template):
        """Test getting a template by name."""
        agent_spawner.register_template(worker_template)

        result = agent_spawner.get_template(worker_template.name)
        assert result == worker_template

    def test_get_nonexistent_template(self, agent_spawner):
        """Test getting a template that doesn't exist."""
        result = agent_spawner.get_template("nonexistent")
        assert result is None

    def test_list_templates(self, agent_spawner, worker_template, pooled_template):
        """Test listing all templates."""
        agent_spawner.register_template(worker_template)
        agent_spawner.register_template(pooled_template)

        templates = agent_spawner.list_templates()

        assert worker_template.name in templates
        assert pooled_template.name in templates

    @pytest.mark.asyncio
    async def test_spawn_on_demand(self, agent_spawner, worker_template):
        """Test spawning an agent on demand."""
        agent_spawner.register_template(worker_template)

        agent = await agent_spawner.spawn(worker_template.name)

        assert agent is not None
        assert isinstance(agent, ConcreteAgent)
        assert agent.id in agent_spawner._agents
        assert agent.id in agent_spawner._agent_info
        assert agent_spawner._instance_counts[worker_template.name] == 1

    @pytest.mark.asyncio
    async def test_spawn_unknown_template_raises(self, agent_spawner):
        """Test spawning from unknown template raises error."""
        with pytest.raises(ValueError) as exc:
            await agent_spawner.spawn("unknown")

        assert "not found" in str(exc.value)

    @pytest.mark.asyncio
    async def test_spawn_singleton(self, agent_spawner, singleton_template):
        """Test singleton spawn policy."""
        agent_spawner.register_template(singleton_template)

        agent1 = await agent_spawner.spawn(singleton_template.name)
        agent2 = await agent_spawner.spawn(singleton_template.name)

        assert agent1 is agent2
        assert agent_spawner._instance_counts[singleton_template.name] == 1

    @pytest.mark.asyncio
    async def test_spawn_pooled_creates_from_pool(
        self, agent_spawner, pooled_template
    ):
        """Test pooled spawning uses pool when available."""
        agent_spawner.register_template(pooled_template)

        # Pre-spawn pool
        await agent_spawner.spawn_pool(pooled_template.name)
        pool_size = len(agent_spawner._pools[pooled_template.name])

        # Spawn should take from pool
        agent = await agent_spawner.spawn(pooled_template.name)

        assert agent is not None
        assert len(agent_spawner._pools[pooled_template.name]) == pool_size - 1

    @pytest.mark.asyncio
    async def test_spawn_pooled_creates_new_when_empty(
        self, agent_spawner, pooled_template
    ):
        """Test pooled spawning creates new agent when pool is empty."""
        agent_spawner.register_template(pooled_template)

        # Pool is empty initially
        agent = await agent_spawner.spawn(pooled_template.name)

        assert agent is not None
        assert agent_spawner._instance_counts[pooled_template.name] == 1

    @pytest.mark.asyncio
    async def test_spawn_max_instances_limit(self, agent_spawner, limited_template):
        """Test max instances limit is enforced."""
        agent_spawner.register_template(limited_template)

        # Spawn up to the limit
        await agent_spawner.spawn(limited_template.name)
        await agent_spawner.spawn(limited_template.name)

        # Third spawn should fail
        with pytest.raises(ValueError) as exc:
            await agent_spawner.spawn(limited_template.name)

        assert "Maximum instances" in str(exc.value)

    @pytest.mark.asyncio
    async def test_spawn_with_config_overrides(
        self, agent_spawner, worker_template
    ):
        """Test spawning with configuration overrides."""
        agent_spawner.register_template(worker_template)

        agent = await agent_spawner.spawn(
            worker_template.name,
            config_overrides={"max_iterations": 20},
        )

        assert agent.config.max_iterations == 20

    @pytest.mark.asyncio
    async def test_spawn_with_llm_factory(self, basic_role):
        """Test spawning uses LLM factory."""
        mock_llm = MagicMock()
        template = AgentTemplate(
            name="test",
            role=basic_role,
            agent_class=ConcreteAgent,
            llm_factory=lambda: mock_llm,
        )

        spawner = AgentSpawner()
        spawner.register_template(template)

        agent = await spawner.spawn("test")
        assert agent.llm == mock_llm

    @pytest.mark.asyncio
    async def test_spawn_with_default_llm_factory(self, basic_role):
        """Test spawning uses default LLM factory."""
        mock_llm = MagicMock()
        template = AgentTemplate(
            name="test",
            role=basic_role,
            agent_class=ConcreteAgent,
        )

        spawner = AgentSpawner(default_llm_factory=lambda: mock_llm)
        spawner.register_template(template)

        agent = await spawner.spawn("test")
        assert agent.llm == mock_llm

    @pytest.mark.asyncio
    async def test_spawn_updates_agent_info(self, agent_spawner, worker_template):
        """Test spawning creates agent info."""
        agent_spawner.register_template(worker_template)

        agent = await agent_spawner.spawn(worker_template.name)
        info = agent_spawner.get_agent_info(agent.id)

        assert info is not None
        assert info.agent_id == agent.id
        assert info.template_name == worker_template.name
        assert info.lifecycle_state == AgentLifecycleState.RUNNING
        assert info.parent_spawner_id == agent_spawner.id

    @pytest.mark.asyncio
    async def test_spawn_with_registry(self, worker_template, basic_role):
        """Test spawning registers agent with registry."""
        registry = MagicMock()
        spawner = AgentSpawner(registry=registry)
        spawner.register_template(worker_template)

        agent = await spawner.spawn(worker_template.name)

        registry.register.assert_called_once()

    @pytest.mark.asyncio
    async def test_spawn_pool(self, agent_spawner, pooled_template):
        """Test pre-spawning a pool of agents."""
        agent_spawner.register_template(pooled_template)

        agents = await agent_spawner.spawn_pool(pooled_template.name)

        assert len(agents) == pooled_template.pool_size
        assert len(agent_spawner._pools[pooled_template.name]) == pooled_template.pool_size

    @pytest.mark.asyncio
    async def test_spawn_pool_partial_fill(self, agent_spawner, pooled_template):
        """Test spawning pool fills remaining slots."""
        agent_spawner.register_template(pooled_template)

        # Pre-fill part of the pool
        first_agents = await agent_spawner.spawn_pool(pooled_template.name)
        initial_count = len(first_agents)

        # Take one agent from the pool
        agent = await agent_spawner.spawn(pooled_template.name)

        # Spawn pool again should only fill the gap
        more_agents = await agent_spawner.spawn_pool(pooled_template.name)

        assert len(more_agents) == 1  # Only filled the one gap

    @pytest.mark.asyncio
    async def test_spawn_pool_not_pooled_raises(
        self, agent_spawner, worker_template
    ):
        """Test spawn_pool with non-pooled template raises error."""
        agent_spawner.register_template(worker_template)

        with pytest.raises(ValueError) as exc:
            await agent_spawner.spawn_pool(worker_template.name)

        assert "not configured for pooling" in str(exc.value)

    @pytest.mark.asyncio
    async def test_release_pooled_agent(self, agent_spawner, pooled_template):
        """Test releasing a pooled agent returns it to pool."""
        agent_spawner.register_template(pooled_template)

        agent = await agent_spawner.spawn(pooled_template.name)
        pool_before = len(agent_spawner._pools[pooled_template.name])

        result = await agent_spawner.release(agent.id)

        assert result is True
        assert len(agent_spawner._pools[pooled_template.name]) == pool_before + 1
        assert agent_spawner._agent_info[agent.id].lifecycle_state == AgentLifecycleState.SUSPENDED

    @pytest.mark.asyncio
    async def test_release_on_demand_agent(self, agent_spawner, worker_template):
        """Test releasing an on-demand agent terminates it."""
        agent_spawner.register_template(worker_template)

        agent = await agent_spawner.spawn(worker_template.name)

        result = await agent_spawner.release(agent.id)

        assert result is True
        assert agent.id not in agent_spawner._agents

    @pytest.mark.asyncio
    async def test_release_unknown_agent(self, agent_spawner):
        """Test releasing an unknown agent returns False."""
        result = await agent_spawner.release("unknown-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_terminate(self, agent_spawner, worker_template):
        """Test terminating an agent."""
        agent_spawner.register_template(worker_template)

        agent = await agent_spawner.spawn(worker_template.name)

        result = await agent_spawner.terminate(agent.id)

        assert result is True
        assert agent.id not in agent_spawner._agents
        assert agent.status == AgentStatus.TERMINATED
        assert agent_spawner._instance_counts[worker_template.name] == 0

    @pytest.mark.asyncio
    async def test_terminate_unknown_agent(self, agent_spawner):
        """Test terminating an unknown agent returns False."""
        result = await agent_spawner.terminate("unknown-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_terminate_singleton(self, agent_spawner, singleton_template):
        """Test terminating a singleton removes tracking."""
        agent_spawner.register_template(singleton_template)

        agent = await agent_spawner.spawn(singleton_template.name)

        await agent_spawner.terminate(agent.id)

        assert singleton_template.name not in agent_spawner._singletons

    @pytest.mark.asyncio
    async def test_terminate_pooled_agent(self, agent_spawner, pooled_template):
        """Test terminating a pooled agent removes from pool."""
        agent_spawner.register_template(pooled_template)
        await agent_spawner.spawn_pool(pooled_template.name)

        pool = agent_spawner._pools[pooled_template.name]
        agent_id = pool[0]

        await agent_spawner.terminate(agent_id)

        assert agent_id not in agent_spawner._pools[pooled_template.name]

    @pytest.mark.asyncio
    async def test_terminate_all(self, agent_spawner, worker_template):
        """Test terminating all agents."""
        agent_spawner.register_template(worker_template)

        # Spawn multiple agents
        await agent_spawner.spawn(worker_template.name)
        await agent_spawner.spawn(worker_template.name)
        await agent_spawner.spawn(worker_template.name)

        count = await agent_spawner.terminate_all()

        assert count == 3
        assert len(agent_spawner._agents) == 0

    @pytest.mark.asyncio
    async def test_terminate_all_by_template(
        self, agent_spawner, worker_template, pooled_template
    ):
        """Test terminating all agents by template."""
        agent_spawner.register_template(worker_template)
        agent_spawner.register_template(pooled_template)

        # Spawn agents from different templates
        await agent_spawner.spawn(worker_template.name)
        await agent_spawner.spawn(worker_template.name)
        await agent_spawner.spawn(pooled_template.name)

        count = await agent_spawner.terminate_all(worker_template.name)

        assert count == 2
        assert len(agent_spawner._agents) == 1

    def test_get_agent(self, agent_spawner, worker_template):
        """Test getting an agent by ID."""
        agent_spawner.register_template(worker_template)

        async def test():
            agent = await agent_spawner.spawn(worker_template.name)
            return agent

        agent = asyncio.get_event_loop().run_until_complete(test())
        result = agent_spawner.get_agent(agent.id)

        assert result == agent

    def test_get_unknown_agent(self, agent_spawner):
        """Test getting an unknown agent returns None."""
        result = agent_spawner.get_agent("unknown-id")
        assert result is None

    def test_get_agent_info(self, agent_spawner, worker_template):
        """Test getting agent info by ID."""
        agent_spawner.register_template(worker_template)

        async def test():
            return await agent_spawner.spawn(worker_template.name)

        agent = asyncio.get_event_loop().run_until_complete(test())
        info = agent_spawner.get_agent_info(agent.id)

        assert info is not None
        assert info.agent_id == agent.id

    def test_get_unknown_agent_info(self, agent_spawner):
        """Test getting info for unknown agent returns None."""
        result = agent_spawner.get_agent_info("unknown-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_agents(self, agent_spawner, worker_template):
        """Test listing all agents."""
        agent_spawner.register_template(worker_template)

        await agent_spawner.spawn(worker_template.name)
        await agent_spawner.spawn(worker_template.name)

        agents = agent_spawner.list_agents()
        assert len(agents) == 2

    @pytest.mark.asyncio
    async def test_list_agents_by_template(
        self, agent_spawner, worker_template, pooled_template
    ):
        """Test listing agents by template."""
        agent_spawner.register_template(worker_template)
        agent_spawner.register_template(pooled_template)

        await agent_spawner.spawn(worker_template.name)
        await agent_spawner.spawn(worker_template.name)
        await agent_spawner.spawn(pooled_template.name)

        agents = agent_spawner.list_agents(worker_template.name)
        assert len(agents) == 2

    def test_get_pool_status(self, agent_spawner, pooled_template):
        """Test getting pool status."""
        agent_spawner.register_template(pooled_template)

        async def test():
            await agent_spawner.spawn_pool(pooled_template.name)
            await agent_spawner.spawn(pooled_template.name)  # Take one

        asyncio.get_event_loop().run_until_complete(test())

        status = agent_spawner.get_pool_status(pooled_template.name)

        assert status["template_name"] == pooled_template.name
        assert status["spawn_policy"] == SpawnPolicy.POOLED.value
        assert status["pool_size"] == pooled_template.pool_size
        assert "available_in_pool" in status
        assert "total_instances" in status
        assert "in_use" in status

    def test_get_pool_status_unknown_template(self, agent_spawner):
        """Test getting pool status for unknown template."""
        status = agent_spawner.get_pool_status("unknown")
        assert "error" in status

    def test_on_lifecycle_change(self, agent_spawner, worker_template):
        """Test lifecycle change callback registration."""
        callback = MagicMock()
        agent_spawner.on_lifecycle_change(callback)

        assert callback in agent_spawner._lifecycle_callbacks

    @pytest.mark.asyncio
    async def test_lifecycle_callbacks_called(
        self, agent_spawner, worker_template
    ):
        """Test lifecycle callbacks are called."""
        callback_states = []

        def callback(agent_id, state):
            callback_states.append((agent_id, state))

        agent_spawner.on_lifecycle_change(callback)
        agent_spawner.register_template(worker_template)

        agent = await agent_spawner.spawn(worker_template.name)
        await agent_spawner.terminate(agent.id)

        # Should have RUNNING, TERMINATING, TERMINATED states
        assert len(callback_states) >= 2
        assert any(s == AgentLifecycleState.RUNNING for _, s in callback_states)
        assert any(s == AgentLifecycleState.TERMINATED for _, s in callback_states)

    @pytest.mark.asyncio
    async def test_async_lifecycle_callback(
        self, agent_spawner, worker_template
    ):
        """Test async lifecycle callbacks are awaited."""
        callback_called = False

        async def async_callback(agent_id, state):
            nonlocal callback_called
            callback_called = True

        agent_spawner.on_lifecycle_change(async_callback)
        agent_spawner.register_template(worker_template)

        await agent_spawner.spawn(worker_template.name)

        assert callback_called is True

    @pytest.mark.asyncio
    async def test_lifecycle_callback_error_handled(
        self, agent_spawner, worker_template
    ):
        """Test lifecycle callback errors are handled gracefully."""

        def failing_callback(agent_id, state):
            raise ValueError("Callback error")

        agent_spawner.on_lifecycle_change(failing_callback)
        agent_spawner.register_template(worker_template)

        # Should not raise
        agent = await agent_spawner.spawn(worker_template.name)
        assert agent is not None

    def test_get_statistics(self, agent_spawner, worker_template, pooled_template):
        """Test getting spawner statistics."""
        agent_spawner.register_template(worker_template)
        agent_spawner.register_template(pooled_template)

        async def test():
            await agent_spawner.spawn(worker_template.name)
            await agent_spawner.spawn(worker_template.name)

        asyncio.get_event_loop().run_until_complete(test())

        stats = agent_spawner.get_statistics()

        assert stats["total_agents"] == 2
        assert worker_template.name in stats["templates"]
        assert pooled_template.name in stats["templates"]
        assert worker_template.name in stats["instance_counts"]
        assert stats["instance_counts"][worker_template.name] == 2

    def test_repr(self, agent_spawner, worker_template):
        """Test string representation."""
        agent_spawner.register_template(worker_template)

        repr_str = repr(agent_spawner)

        assert "AgentSpawner" in repr_str
        assert agent_spawner.id in repr_str
        assert "templates=1" in repr_str

    @pytest.mark.asyncio
    async def test_concurrent_spawn(self, agent_spawner, worker_template):
        """Test concurrent spawning is thread-safe."""
        agent_spawner.register_template(worker_template)

        # Spawn multiple agents concurrently
        tasks = [agent_spawner.spawn(worker_template.name) for _ in range(10)]
        agents = await asyncio.gather(*tasks)

        assert len(agents) == 10
        assert len(agent_spawner._agents) == 10
        assert agent_spawner._instance_counts[worker_template.name] == 10

    @pytest.mark.asyncio
    async def test_release_with_template_unregistered(
        self, agent_spawner, worker_template
    ):
        """Test releasing agent after template is unregistered."""
        agent_spawner.register_template(worker_template)
        agent = await agent_spawner.spawn(worker_template.name)

        agent_spawner.unregister_template(worker_template.name)

        result = await agent_spawner.release(agent.id)

        # Should terminate since template is gone
        assert result is True
        assert agent.id not in agent_spawner._agents

    @pytest.mark.asyncio
    async def test_pool_full_terminates_on_release(
        self, agent_spawner, pooled_template
    ):
        """Test releasing when pool is full terminates the agent."""
        agent_spawner.register_template(pooled_template)

        # Fill the pool
        await agent_spawner.spawn_pool(pooled_template.name)

        # Spawn one more (not from pool)
        agent = await agent_spawner.spawn(pooled_template.name)

        # Release - pool is full, should terminate
        result = await agent_spawner.release(agent.id)

        assert result is True
        # Agent should be terminated since pool was already full
        assert agent.id not in agent_spawner._agents
