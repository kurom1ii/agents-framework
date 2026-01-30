"""Unit tests for the supervisor agent module."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import List
from unittest.mock import AsyncMock, patch

import pytest

from agents_framework.agents.base import (
    AgentRole,
    AgentStatus,
    Task,
    TaskResult,
)
from agents_framework.agents.supervisor import (
    DelegatedTask,
    DelegationStrategy,
    ExecutionMode,
    SupervisorAgent,
    SupervisorConfig,
)

from .conftest import ConcreteAgent, ExceptionRaisingAgent, SlowAgent


class TestSupervisorConfig:
    """Tests for SupervisorConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = SupervisorConfig()
        assert config.delegation_strategy == DelegationStrategy.CAPABILITY_MATCH
        assert config.execution_mode == ExecutionMode.PARALLEL
        assert config.max_workers == 10
        assert config.task_timeout == 60.0
        assert config.max_retries == 2
        assert config.retry_delay == 1.0
        assert config.fallback_enabled is True

    def test_config_custom(self):
        """Test custom configuration values."""
        config = SupervisorConfig(
            delegation_strategy=DelegationStrategy.ROUND_ROBIN,
            execution_mode=ExecutionMode.SEQUENTIAL,
            max_workers=5,
            task_timeout=30.0,
            max_retries=3,
            retry_delay=0.5,
            fallback_enabled=False,
        )
        assert config.delegation_strategy == DelegationStrategy.ROUND_ROBIN
        assert config.execution_mode == ExecutionMode.SEQUENTIAL
        assert config.max_workers == 5
        assert config.task_timeout == 30.0
        assert config.max_retries == 3
        assert config.retry_delay == 0.5
        assert config.fallback_enabled is False


class TestDelegationStrategy:
    """Tests for DelegationStrategy enum."""

    def test_strategy_values(self):
        """Test strategy enum values."""
        assert DelegationStrategy.ROUND_ROBIN.value == "round_robin"
        assert DelegationStrategy.LEAST_BUSY.value == "least_busy"
        assert DelegationStrategy.CAPABILITY_MATCH.value == "capability_match"
        assert DelegationStrategy.RANDOM.value == "random"
        assert DelegationStrategy.PRIORITY.value == "priority"


class TestExecutionMode:
    """Tests for ExecutionMode enum."""

    def test_mode_values(self):
        """Test execution mode enum values."""
        assert ExecutionMode.PARALLEL.value == "parallel"
        assert ExecutionMode.SEQUENTIAL.value == "sequential"
        assert ExecutionMode.PIPELINE.value == "pipeline"


class TestDelegatedTask:
    """Tests for DelegatedTask dataclass."""

    def test_delegated_task_creation(self, basic_task):
        """Test creating a delegated task."""
        delegated = DelegatedTask(
            task=basic_task,
            worker_id="worker-123",
        )
        assert delegated.task == basic_task
        assert delegated.worker_id == "worker-123"
        assert delegated.completed_at is None
        assert delegated.result is None
        assert delegated.retries == 0
        assert isinstance(delegated.assigned_at, datetime)


class TestSupervisorAgent:
    """Tests for SupervisorAgent class."""

    def test_supervisor_initialization(self, supervisor_role, supervisor_config):
        """Test supervisor initialization."""
        supervisor = SupervisorAgent(
            role=supervisor_role,
            config=supervisor_config,
        )
        assert supervisor.role == supervisor_role
        assert supervisor._config == supervisor_config
        assert len(supervisor.workers) == 0
        assert supervisor._round_robin_index == 0

    def test_supervisor_default_config(self, supervisor_role):
        """Test supervisor with default config."""
        supervisor = SupervisorAgent(role=supervisor_role)
        assert isinstance(supervisor._config, SupervisorConfig)

    def test_add_worker(self, supervisor_agent, coding_agent):
        """Test adding a worker."""
        assert len(supervisor_agent.workers) == 0

        supervisor_agent.add_worker(coding_agent)

        assert len(supervisor_agent.workers) == 1
        assert coding_agent in supervisor_agent.workers
        assert coding_agent.parent_id == supervisor_agent.id

    def test_add_multiple_workers(
        self, supervisor_agent, coding_agent, research_agent
    ):
        """Test adding multiple workers."""
        supervisor_agent.add_worker(coding_agent)
        supervisor_agent.add_worker(research_agent)

        assert len(supervisor_agent.workers) == 2
        assert coding_agent in supervisor_agent.workers
        assert research_agent in supervisor_agent.workers

    def test_remove_worker(self, supervisor_with_workers, coding_agent):
        """Test removing a worker."""
        initial_count = len(supervisor_with_workers.workers)
        worker_id = coding_agent.id

        removed = supervisor_with_workers.remove_worker(worker_id)

        assert removed == coding_agent
        assert len(supervisor_with_workers.workers) == initial_count - 1
        assert coding_agent.parent_id is None

    def test_remove_nonexistent_worker(self, supervisor_agent):
        """Test removing a worker that doesn't exist."""
        removed = supervisor_agent.remove_worker("nonexistent-id")
        assert removed is None

    def test_get_worker(self, supervisor_with_workers, coding_agent):
        """Test getting a worker by ID."""
        worker = supervisor_with_workers.get_worker(coding_agent.id)
        assert worker == coding_agent

    def test_get_nonexistent_worker(self, supervisor_agent):
        """Test getting a worker that doesn't exist."""
        worker = supervisor_agent.get_worker("nonexistent-id")
        assert worker is None

    def test_set_custom_selector(self, supervisor_agent):
        """Test setting a custom worker selector."""

        def custom_selector(workers, task):
            return workers[0] if workers else None

        supervisor_agent.set_selector(custom_selector)
        assert supervisor_agent._custom_selector == custom_selector

    def test_set_custom_aggregator(self, supervisor_agent):
        """Test setting a custom result aggregator."""

        def custom_aggregator(results):
            return {"combined": True}

        supervisor_agent.set_aggregator(custom_aggregator)
        assert supervisor_agent._custom_aggregator == custom_aggregator

    def test_workers_property(self, supervisor_with_workers):
        """Test workers property returns list."""
        workers = supervisor_with_workers.workers
        assert isinstance(workers, list)
        assert len(workers) == 2

    def test_repr(self, supervisor_with_workers):
        """Test string representation."""
        repr_str = repr(supervisor_with_workers)
        assert "SupervisorAgent" in repr_str
        assert supervisor_with_workers.id in repr_str
        assert "workers=2" in repr_str

    @pytest.mark.asyncio
    async def test_run_with_string_task(self, supervisor_with_workers):
        """Test running with a string task."""
        result = await supervisor_with_workers.run("Complete the project")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_run_with_task_object(self, supervisor_with_workers, basic_task):
        """Test running with a Task object."""
        result = await supervisor_with_workers.run(basic_task)
        assert result.task_id == basic_task.id

    @pytest.mark.asyncio
    async def test_run_no_workers(self, supervisor_agent, basic_task):
        """Test running with no workers available."""
        result = await supervisor_agent.run(basic_task)
        assert result.success is False
        assert "No available worker" in result.error or "No subtasks" in result.error

    @pytest.mark.asyncio
    async def test_run_parallel_execution(
        self, supervisor_role, coding_agent, research_agent
    ):
        """Test parallel execution mode."""
        config = SupervisorConfig(execution_mode=ExecutionMode.PARALLEL)
        supervisor = SupervisorAgent(role=supervisor_role, config=config)
        supervisor.add_worker(coding_agent)
        supervisor.add_worker(research_agent)

        result = await supervisor.run("Do tasks")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_run_sequential_execution(
        self, supervisor_role, coding_agent, research_agent
    ):
        """Test sequential execution mode."""
        config = SupervisorConfig(
            execution_mode=ExecutionMode.SEQUENTIAL,
            task_timeout=10.0,
        )
        supervisor = SupervisorAgent(role=supervisor_role, config=config)
        supervisor.add_worker(coding_agent)
        supervisor.add_worker(research_agent)

        result = await supervisor.run("Do sequential tasks")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_run_pipeline_execution(self, supervisor_role, coding_agent):
        """Test pipeline execution mode."""
        config = SupervisorConfig(
            execution_mode=ExecutionMode.PIPELINE,
            task_timeout=10.0,
        )
        supervisor = SupervisorAgent(role=supervisor_role, config=config)
        supervisor.add_worker(coding_agent)

        result = await supervisor.run("Do pipeline task")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_round_robin_delegation(
        self, supervisor_role, coding_agent, research_agent
    ):
        """Test round-robin delegation strategy."""
        config = SupervisorConfig(
            delegation_strategy=DelegationStrategy.ROUND_ROBIN,
            task_timeout=10.0,
        )
        supervisor = SupervisorAgent(role=supervisor_role, config=config)
        supervisor.add_worker(coding_agent)
        supervisor.add_worker(research_agent)

        # Run multiple tasks
        for i in range(4):
            await supervisor.run(f"Task {i}")

        # Both agents should have been used
        assert coding_agent.run_count > 0
        assert research_agent.run_count > 0

    @pytest.mark.asyncio
    async def test_capability_match_delegation(
        self, supervisor_role, coding_agent, research_agent
    ):
        """Test capability-based delegation."""
        config = SupervisorConfig(
            delegation_strategy=DelegationStrategy.CAPABILITY_MATCH,
            task_timeout=10.0,
        )
        supervisor = SupervisorAgent(role=supervisor_role, config=config)
        supervisor.add_worker(coding_agent)
        supervisor.add_worker(research_agent)

        # Create task requiring coding capability
        task = Task(
            description="Write code",
            required_capabilities=["code"],
        )
        result = await supervisor.run(task)

        assert result.success is True
        # Coding agent should have been selected
        assert coding_agent.run_count > 0

    @pytest.mark.asyncio
    async def test_random_delegation(self, supervisor_role, coding_agent):
        """Test random delegation strategy."""
        config = SupervisorConfig(
            delegation_strategy=DelegationStrategy.RANDOM,
            task_timeout=10.0,
        )
        supervisor = SupervisorAgent(role=supervisor_role, config=config)
        supervisor.add_worker(coding_agent)

        result = await supervisor.run("Random task")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_worker_failure_with_fallback(self, supervisor_role, basic_role):
        """Test fallback when primary worker fails."""
        config = SupervisorConfig(
            fallback_enabled=True,
            max_retries=1,
            retry_delay=0.01,
            task_timeout=10.0,
        )
        supervisor = SupervisorAgent(role=supervisor_role, config=config)

        # Add a failing worker and a successful one
        failing_worker = ConcreteAgent(role=basic_role, should_fail=True)
        successful_worker = ConcreteAgent(role=basic_role)

        supervisor.add_worker(failing_worker)
        supervisor.add_worker(successful_worker)

        result = await supervisor.run("Fallback task")
        # Should eventually succeed via fallback
        assert result is not None

    @pytest.mark.asyncio
    async def test_worker_timeout(self, supervisor_role, basic_role):
        """Test handling of worker timeout."""
        config = SupervisorConfig(
            task_timeout=0.1,  # Very short timeout
            max_retries=0,
            fallback_enabled=False,
        )
        supervisor = SupervisorAgent(role=supervisor_role, config=config)

        # Add a slow worker
        slow_worker = SlowAgent(role=basic_role, delay=1.0)
        supervisor.add_worker(slow_worker)

        result = await supervisor.run("Timeout task")
        assert result.success is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_custom_selector(self, supervisor_with_workers, basic_task):
        """Test using a custom worker selector."""
        selected_worker = None

        def custom_selector(workers, task):
            nonlocal selected_worker
            selected_worker = workers[0] if workers else None
            return selected_worker

        supervisor_with_workers.set_selector(custom_selector)
        await supervisor_with_workers.run(basic_task)

        assert selected_worker is not None

    @pytest.mark.asyncio
    async def test_custom_aggregator(self, supervisor_with_workers, basic_task):
        """Test using a custom result aggregator."""

        def custom_aggregator(results: List[TaskResult]) -> dict:
            return {"aggregated": True, "count": len(results)}

        supervisor_with_workers.set_aggregator(custom_aggregator)
        result = await supervisor_with_workers.run(basic_task)

        assert result.output == {"aggregated": True, "count": 1}

    @pytest.mark.asyncio
    async def test_custom_aggregator_failure(
        self, supervisor_with_workers, basic_task
    ):
        """Test handling of custom aggregator failure."""

        def failing_aggregator(results):
            raise ValueError("Aggregation error")

        supervisor_with_workers.set_aggregator(failing_aggregator)
        result = await supervisor_with_workers.run(basic_task)

        assert result.success is False
        assert "Aggregation failed" in result.error

    @pytest.mark.asyncio
    async def test_task_history(self, supervisor_with_workers, basic_task):
        """Test that task history is recorded."""
        assert len(supervisor_with_workers.get_task_history()) == 0

        await supervisor_with_workers.run(basic_task)

        history = supervisor_with_workers.get_task_history()
        assert len(history) >= 1
        assert isinstance(history[0], DelegatedTask)

    def test_clear_task_history(self, supervisor_with_workers):
        """Test clearing task history."""
        supervisor_with_workers._task_history.append(
            DelegatedTask(task=Task(), worker_id="test")
        )
        assert len(supervisor_with_workers.get_task_history()) > 0

        supervisor_with_workers.clear_task_history()

        assert len(supervisor_with_workers.get_task_history()) == 0

    @pytest.mark.asyncio
    async def test_worker_stats(self, supervisor_with_workers, basic_task):
        """Test getting worker statistics."""
        await supervisor_with_workers.run(basic_task)

        stats = supervisor_with_workers.get_worker_stats()
        assert isinstance(stats, dict)

        for worker_id, worker_stats in stats.items():
            assert "total_tasks" in worker_stats
            assert "successful" in worker_stats
            assert "failed" in worker_stats
            assert "success_rate" in worker_stats
            assert "total_retries" in worker_stats

    @pytest.mark.asyncio
    async def test_run_exception_handling(self, supervisor_agent, basic_role):
        """Test handling of exceptions during run."""
        exception_worker = ExceptionRaisingAgent(role=basic_role)
        supervisor_agent.add_worker(exception_worker)

        result = await supervisor_agent.run("Exception task")

        # Should handle exception gracefully
        assert result is not None
        assert result.success is False

    @pytest.mark.asyncio
    async def test_status_management(self, supervisor_with_workers, basic_task):
        """Test that supervisor manages its status correctly."""
        assert supervisor_with_workers.status == AgentStatus.IDLE

        result = await supervisor_with_workers.run(basic_task)

        # Should return to IDLE after completion
        assert supervisor_with_workers.status == AgentStatus.IDLE

    @pytest.mark.asyncio
    async def test_parallel_with_semaphore(self, supervisor_role, basic_role):
        """Test parallel execution respects max_workers semaphore."""
        config = SupervisorConfig(
            execution_mode=ExecutionMode.PARALLEL,
            max_workers=2,  # Only 2 concurrent
            task_timeout=10.0,
        )
        supervisor = SupervisorAgent(role=supervisor_role, config=config)

        # Add more workers than max_workers
        for i in range(5):
            worker = ConcreteAgent(role=basic_role, delay=0.01)
            supervisor.add_worker(worker)

        result = await supervisor.run("Parallel task")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_sequential_stops_on_failure(self, supervisor_role, basic_role):
        """Test that sequential execution stops on failure."""
        config = SupervisorConfig(
            execution_mode=ExecutionMode.SEQUENTIAL,
            fallback_enabled=False,
            task_timeout=10.0,
        )
        supervisor = SupervisorAgent(role=supervisor_role, config=config)

        # Add failing worker first
        failing_worker = ConcreteAgent(role=basic_role, should_fail=True)
        successful_worker = ConcreteAgent(role=basic_role)

        supervisor.add_worker(failing_worker)
        supervisor.add_worker(successful_worker)

        result = await supervisor.run("Sequential task")

        # First worker failed, should stop
        assert result.success is False

    @pytest.mark.asyncio
    async def test_pipeline_passes_output(self, supervisor_role, basic_role):
        """Test that pipeline mode passes output between tasks."""
        config = SupervisorConfig(
            execution_mode=ExecutionMode.PIPELINE,
            task_timeout=10.0,
        )
        supervisor = SupervisorAgent(role=supervisor_role, config=config)

        # Use custom agent that checks for previous_output
        worker = ConcreteAgent(role=basic_role)
        supervisor.add_worker(worker)

        result = await supervisor.run("Pipeline task")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_get_available_workers_excludes_terminated(
        self, supervisor_agent, coding_agent
    ):
        """Test that terminated workers are not available."""
        supervisor_agent.add_worker(coding_agent)

        # Mark worker as terminated
        coding_agent.status = AgentStatus.TERMINATED

        available = supervisor_agent._get_available_workers()
        assert len(available) == 0

    @pytest.mark.asyncio
    async def test_get_available_workers_includes_busy(
        self, supervisor_agent, coding_agent
    ):
        """Test that busy workers are still available (for queueing)."""
        supervisor_agent.add_worker(coding_agent)

        # Mark worker as busy
        coding_agent.status = AgentStatus.BUSY

        available = supervisor_agent._get_available_workers()
        assert len(available) == 1
