"""Unit tests for the team orchestration module."""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from agents_framework.agents import AgentRole, AgentStatus, Task, TaskResult
from agents_framework.teams.router import RoutingStrategy
from agents_framework.teams.team import (
    SharedContext,
    Team,
    TeamConfig,
    TeamExecutionStrategy,
    TeamState,
    TeamTaskResult,
)

from .conftest import MockAgent


# ============================================================================
# SharedContext Tests
# ============================================================================


class TestSharedContext:
    """Tests for the SharedContext class."""

    @pytest.mark.asyncio
    async def test_context_creation(self):
        """Test creating a shared context."""
        context = SharedContext()
        assert context.data == {}
        assert context.metadata == {}
        assert context.created_at is not None

    @pytest.mark.asyncio
    async def test_get_and_set(self):
        """Test getting and setting values."""
        context = SharedContext()

        await context.set("key", "value")
        result = await context.get("key")

        assert result == "value"

    @pytest.mark.asyncio
    async def test_get_default(self):
        """Test getting with default value."""
        context = SharedContext()

        result = await context.get("nonexistent", "default")
        assert result == "default"

    @pytest.mark.asyncio
    async def test_get_none_default(self):
        """Test getting non-existent key returns None by default."""
        context = SharedContext()

        result = await context.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_update(self):
        """Test updating multiple values."""
        context = SharedContext()

        await context.update({"key1": "value1", "key2": "value2"})

        assert await context.get("key1") == "value1"
        assert await context.get("key2") == "value2"

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test deleting a value."""
        context = SharedContext()

        await context.set("key", "value")
        result = await context.delete("key")

        assert result is True
        assert await context.get("key") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        """Test deleting non-existent key returns False."""
        context = SharedContext()

        result = await context.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing all data."""
        context = SharedContext()

        await context.update({"key1": "value1", "key2": "value2"})
        await context.clear()

        assert await context.get("key1") is None
        assert await context.get("key2") is None

    @pytest.mark.asyncio
    async def test_keys(self):
        """Test getting all keys."""
        context = SharedContext()

        await context.update({"key1": "value1", "key2": "value2"})
        keys = await context.keys()

        assert sorted(keys) == ["key1", "key2"]

    @pytest.mark.asyncio
    async def test_updated_at_changes(self):
        """Test that updated_at changes on modifications."""
        context = SharedContext()
        original_time = context.updated_at

        await asyncio.sleep(0.01)
        await context.set("key", "value")

        assert context.updated_at > original_time


# ============================================================================
# TeamTaskResult Tests
# ============================================================================


class TestTeamTaskResult:
    """Tests for the TeamTaskResult dataclass."""

    def test_result_creation(self):
        """Test creating a team task result."""
        result = TeamTaskResult(
            task_id="task_123",
            success=True,
            aggregated_output={"data": "output"},
        )

        assert result.task_id == "task_123"
        assert result.success
        assert result.aggregated_output == {"data": "output"}
        assert result.results == []
        assert result.errors == []

    def test_duration_calculation(self):
        """Test duration calculation."""
        from datetime import datetime, timedelta

        started = datetime.now()
        completed = started + timedelta(seconds=5)

        result = TeamTaskResult(
            task_id="task_123",
            success=True,
            started_at=started,
            completed_at=completed,
        )

        assert result.duration == 5.0

    def test_duration_none_when_incomplete(self):
        """Test duration is None when times not set."""
        result = TeamTaskResult(
            task_id="task_123",
            success=True,
        )

        assert result.duration is None


# ============================================================================
# TeamConfig Tests
# ============================================================================


class TestTeamConfig:
    """Tests for the TeamConfig dataclass."""

    def test_default_config(self):
        """Test default team configuration."""
        config = TeamConfig()

        assert config.name == "Team"
        assert config.strategy == TeamExecutionStrategy.COLLABORATIVE
        assert config.max_concurrent_tasks == 10
        assert config.task_timeout == 300.0
        assert config.enable_shared_memory is True
        assert config.enable_messaging is True
        assert config.auto_cleanup is True

    def test_custom_config(self):
        """Test custom team configuration."""
        config = TeamConfig(
            name="CustomTeam",
            strategy=TeamExecutionStrategy.HIERARCHICAL,
            max_concurrent_tasks=5,
            task_timeout=60.0,
        )

        assert config.name == "CustomTeam"
        assert config.strategy == TeamExecutionStrategy.HIERARCHICAL
        assert config.max_concurrent_tasks == 5
        assert config.task_timeout == 60.0


# ============================================================================
# Team Creation Tests
# ============================================================================


class TestTeamCreation:
    """Tests for Team creation and initialization."""

    def test_team_creation(self, team_config: TeamConfig):
        """Test creating a team."""
        team = Team(config=team_config)

        assert team.id is not None
        assert team.name == team_config.name
        assert team.state == TeamState.IDLE
        assert len(team.members) == 0

    def test_team_default_config(self):
        """Test team with default configuration."""
        team = Team()

        assert team.config.name == "Team"
        assert team.state == TeamState.IDLE

    def test_team_repr(self, team: Team):
        """Test team string representation."""
        repr_str = repr(team)

        assert "Team" in repr_str
        assert team.id in repr_str
        assert "idle" in repr_str

    def test_team_has_router_when_messaging_enabled(self, team_config: TeamConfig):
        """Test that team has router when messaging is enabled."""
        team = Team(config=team_config)
        assert team.router is not None

    def test_team_no_router_when_messaging_disabled(self):
        """Test that team has no router when messaging is disabled."""
        config = TeamConfig(enable_messaging=False)
        team = Team(config=config)
        assert team.router is None


# ============================================================================
# Member Management Tests
# ============================================================================


class TestMemberManagement:
    """Tests for team member management."""

    def test_add_member(
        self,
        team: Team,
        mock_researcher: MockAgent,
    ):
        """Test adding a member to the team."""
        team.add_member(mock_researcher)

        assert len(team) == 1
        assert mock_researcher.id in team
        assert mock_researcher in team.members

    def test_add_member_as_leader(
        self,
        team: Team,
        mock_researcher: MockAgent,
    ):
        """Test adding a member as leader."""
        team.add_member(mock_researcher, is_leader=True)

        assert team.get_leader() is mock_researcher

    def test_add_member_with_metadata(
        self,
        team: Team,
        mock_researcher: MockAgent,
    ):
        """Test adding member with metadata."""
        metadata = {"role": "primary"}
        team.add_member(mock_researcher, metadata=metadata)

        info = team.registry.get_info(mock_researcher.id)
        assert info.metadata == metadata

    def test_add_member_registers_with_router(
        self,
        team: Team,
        mock_researcher: MockAgent,
    ):
        """Test that adding member registers with router."""
        team.add_member(mock_researcher)

        assert mock_researcher.id in team.router.list_agents()

    def test_remove_member(
        self,
        team: Team,
        mock_researcher: MockAgent,
    ):
        """Test removing a member from the team."""
        team.add_member(mock_researcher)
        removed = team.remove_member(mock_researcher.id)

        assert removed is mock_researcher
        assert len(team) == 0
        assert mock_researcher.id not in team

    def test_remove_leader_clears_leader_id(
        self,
        team: Team,
        mock_researcher: MockAgent,
    ):
        """Test that removing leader clears leader ID."""
        team.add_member(mock_researcher, is_leader=True)
        team.remove_member(mock_researcher.id)

        assert team.get_leader() is None

    def test_remove_member_unregisters_from_router(
        self,
        team: Team,
        mock_researcher: MockAgent,
    ):
        """Test that removing member unregisters from router."""
        team.add_member(mock_researcher)
        team.remove_member(mock_researcher.id)

        assert mock_researcher.id not in team.router.list_agents()

    def test_get_member(
        self,
        team: Team,
        mock_researcher: MockAgent,
    ):
        """Test getting a member by ID."""
        team.add_member(mock_researcher)

        member = team.get_member(mock_researcher.id)
        assert member is mock_researcher

    def test_get_member_nonexistent(self, team: Team):
        """Test getting non-existent member returns None."""
        member = team.get_member("nonexistent")
        assert member is None

    def test_set_leader(
        self,
        team: Team,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test setting the team leader."""
        team.add_member(mock_researcher)
        team.add_member(mock_analyst)

        result = team.set_leader(mock_analyst.id)

        assert result
        assert team.get_leader() is mock_analyst

    def test_set_leader_nonexistent(self, team: Team):
        """Test setting non-existent member as leader fails."""
        result = team.set_leader("nonexistent")
        assert not result

    def test_contains_protocol(
        self,
        team: Team,
        mock_researcher: MockAgent,
    ):
        """Test __contains__ protocol."""
        team.add_member(mock_researcher)

        assert mock_researcher.id in team
        assert "nonexistent" not in team

    def test_iter_protocol(self, populated_team: Team):
        """Test __iter__ protocol."""
        members = list(populated_team)
        assert len(members) == 3


# ============================================================================
# Team Lifecycle Tests
# ============================================================================


class TestTeamLifecycle:
    """Tests for team lifecycle management."""

    @pytest.mark.asyncio
    async def test_start(self, team: Team):
        """Test starting a team."""
        await team.start()

        assert team.state == TeamState.RUNNING

    @pytest.mark.asyncio
    async def test_start_already_running(self, team: Team):
        """Test that starting already running team is idempotent."""
        await team.start()
        await team.start()

        assert team.state == TeamState.RUNNING

    @pytest.mark.asyncio
    async def test_start_sets_shared_context(self, team: Team):
        """Test that start sets up shared context."""
        await team.start()

        team_id = await team.shared_context.get("team_id")
        team_name = await team.shared_context.get("team_name")

        assert team_id == team.id
        assert team_name == team.config.name

    @pytest.mark.asyncio
    async def test_stop(self, team: Team):
        """Test stopping a team."""
        await team.start()
        await team.stop()

        assert team.state == TeamState.STOPPED

    @pytest.mark.asyncio
    async def test_stop_already_stopped(self, team: Team):
        """Test that stopping already stopped team is idempotent."""
        await team.start()
        await team.stop()
        await team.stop()

        assert team.state == TeamState.STOPPED

    @pytest.mark.asyncio
    async def test_pause(self, team: Team):
        """Test pausing a team."""
        await team.start()
        await team.pause()

        assert team.state == TeamState.PAUSED

    @pytest.mark.asyncio
    async def test_pause_not_running(self, team: Team):
        """Test pausing when not running has no effect."""
        original_state = team.state
        await team.pause()

        assert team.state == original_state

    @pytest.mark.asyncio
    async def test_resume(self, team: Team):
        """Test resuming a paused team."""
        await team.start()
        await team.pause()
        await team.resume()

        assert team.state == TeamState.RUNNING

    @pytest.mark.asyncio
    async def test_resume_not_paused(self, team: Team):
        """Test resuming when not paused has no effect."""
        await team.start()
        original_state = team.state
        await team.resume()

        assert team.state == original_state

    @pytest.mark.asyncio
    async def test_on_state_change_callback(self, team: Team):
        """Test state change callback."""
        states = []

        def callback(state: TeamState):
            states.append(state)

        team.on_state_change(callback)
        await team.start()
        await team.pause()
        await team.resume()
        await team.stop()

        assert TeamState.RUNNING in states
        assert TeamState.PAUSED in states
        assert TeamState.STOPPED in states

    @pytest.mark.asyncio
    async def test_async_state_change_callback(self, team: Team):
        """Test async state change callback."""
        states = []

        async def async_callback(state: TeamState):
            await asyncio.sleep(0.01)
            states.append(state)

        team.on_state_change(async_callback)
        await team.start()

        # Give async callback time to complete
        await asyncio.sleep(0.05)

        assert TeamState.RUNNING in states


# ============================================================================
# Task Execution Tests - Collaborative
# ============================================================================


class TestCollaborativeExecution:
    """Tests for collaborative execution strategy."""

    @pytest.mark.asyncio
    async def test_run_string_task(self, populated_team: Team):
        """Test running a string task."""
        result = await populated_team.run("Perform analysis")

        assert result.success
        assert result.task_id is not None
        assert len(result.results) == 3  # One per member

    @pytest.mark.asyncio
    async def test_run_task_object(self, populated_team: Team, sample_task: Task):
        """Test running a Task object."""
        result = await populated_team.run(sample_task)

        assert result.success
        assert result.task_id == sample_task.id

    @pytest.mark.asyncio
    async def test_run_starts_team_if_idle(
        self,
        team: Team,
        mock_researcher: MockAgent,
    ):
        """Test that run starts team if not running."""
        team.add_member(mock_researcher)

        assert team.state == TeamState.IDLE

        await team.run("Test task")

        assert team.state == TeamState.RUNNING

    @pytest.mark.asyncio
    async def test_collaborative_all_agents_execute(self, populated_team: Team):
        """Test that all agents execute in collaborative mode."""
        await populated_team.run("Test task")

        for member in populated_team.members:
            assert member.run_count == 1

    @pytest.mark.asyncio
    async def test_collaborative_aggregates_results(self, populated_team: Team):
        """Test result aggregation in collaborative mode."""
        result = await populated_team.run("Test task")

        assert result.aggregated_output is not None
        assert "successful_outputs" in result.aggregated_output
        assert result.aggregated_output["total_results"] == 3

    @pytest.mark.asyncio
    async def test_collaborative_handles_failure(
        self,
        team: Team,
        mock_researcher: MockAgent,
        failing_agent: MockAgent,
    ):
        """Test handling of agent failure in collaborative mode."""
        team.add_member(mock_researcher)
        team.add_member(failing_agent)

        result = await team.run("Test task")

        # Overall success depends on all results
        assert not result.success  # One agent failed
        assert len(result.errors) == 1


# ============================================================================
# Task Execution Tests - Other Strategies
# ============================================================================


class TestDivideConquerExecution:
    """Tests for divide and conquer execution strategy."""

    @pytest.mark.asyncio
    async def test_divide_conquer_execution(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test divide and conquer execution."""
        config = TeamConfig(
            name="DivideTeam",
            strategy=TeamExecutionStrategy.DIVIDE_CONQUER,
        )
        team = Team(config=config)
        team.add_member(mock_researcher)
        team.add_member(mock_analyst)

        result = await team.run("Analyze data")

        assert result.success
        assert len(result.results) == 2

    @pytest.mark.asyncio
    async def test_divide_conquer_custom_divider(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test divide and conquer with custom divider."""
        config = TeamConfig(
            name="DivideTeam",
            strategy=TeamExecutionStrategy.DIVIDE_CONQUER,
        )
        team = Team(config=config)
        team.add_member(mock_researcher)
        team.add_member(mock_analyst)

        def custom_divider(task: Task, agents: List[MockAgent]):
            return [
                (Task(description=f"Part for {a.role.name}"), a)
                for a in agents
            ]

        team.set_task_divider(custom_divider)

        result = await team.run("Main task")
        assert result.success


class TestHierarchicalExecution:
    """Tests for hierarchical execution strategy."""

    @pytest.mark.asyncio
    async def test_hierarchical_execution_with_leader(
        self,
        mock_supervisor: MockAgent,
        mock_researcher: MockAgent,
    ):
        """Test hierarchical execution with a leader."""
        config = TeamConfig(
            name="HierarchicalTeam",
            strategy=TeamExecutionStrategy.HIERARCHICAL,
        )
        team = Team(config=config)
        team.add_member(mock_supervisor, is_leader=True)
        team.add_member(mock_researcher)

        result = await team.run("Coordinate task")

        assert result.success
        # Leader executes the task
        assert mock_supervisor.run_count == 1

    @pytest.mark.asyncio
    async def test_hierarchical_fallback_to_first_member(
        self,
        mock_researcher: MockAgent,
    ):
        """Test hierarchical falls back to first member without leader."""
        config = TeamConfig(
            name="HierarchicalTeam",
            strategy=TeamExecutionStrategy.HIERARCHICAL,
        )
        team = Team(config=config)
        team.add_member(mock_researcher)

        result = await team.run("Task without leader")

        assert result.success
        assert mock_researcher.run_count == 1

    @pytest.mark.asyncio
    async def test_hierarchical_no_members(self):
        """Test hierarchical with no members fails gracefully."""
        config = TeamConfig(
            name="HierarchicalTeam",
            strategy=TeamExecutionStrategy.HIERARCHICAL,
        )
        team = Team(config=config)

        result = await team.run("Task")

        assert not result.success


class TestRoundRobinExecution:
    """Tests for round robin execution strategy."""

    @pytest.mark.asyncio
    async def test_round_robin_execution(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
        mock_writer: MockAgent,
    ):
        """Test round robin execution rotates agents."""
        config = TeamConfig(
            name="RoundRobinTeam",
            strategy=TeamExecutionStrategy.ROUND_ROBIN,
        )
        team = Team(config=config)
        team.add_member(mock_researcher)
        team.add_member(mock_analyst)
        team.add_member(mock_writer)

        # First task goes to first agent
        await team.run("Task 1")
        assert mock_researcher.run_count == 1
        assert mock_analyst.run_count == 0

        # Second task goes to second agent
        await team.run("Task 2")
        assert mock_analyst.run_count == 1

        # Third task goes to third agent
        await team.run("Task 3")
        assert mock_writer.run_count == 1

        # Fourth task rotates back to first
        await team.run("Task 4")
        assert mock_researcher.run_count == 2


class TestBroadcastExecution:
    """Tests for broadcast execution strategy."""

    @pytest.mark.asyncio
    async def test_broadcast_execution(
        self,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
        mock_writer: MockAgent,
    ):
        """Test broadcast execution sends to all agents."""
        config = TeamConfig(
            name="BroadcastTeam",
            strategy=TeamExecutionStrategy.BROADCAST,
        )
        team = Team(config=config)
        team.add_member(mock_researcher)
        team.add_member(mock_analyst)
        team.add_member(mock_writer)

        result = await team.run("Broadcast message")

        assert result.success
        # All agents should execute
        assert mock_researcher.run_count == 1
        assert mock_analyst.run_count == 1
        assert mock_writer.run_count == 1


# ============================================================================
# Result Aggregation Tests
# ============================================================================


class TestResultAggregation:
    """Tests for result aggregation."""

    @pytest.mark.asyncio
    async def test_custom_merger(
        self,
        team: Team,
        mock_researcher: MockAgent,
        mock_analyst: MockAgent,
    ):
        """Test custom result merger."""
        team.add_member(mock_researcher)
        team.add_member(mock_analyst)

        def custom_merger(results: List[TaskResult]):
            return {"merged": [r.output for r in results if r.success]}

        team.set_result_merger(custom_merger)

        result = await team.run("Test task")

        assert result.aggregated_output == {
            "merged": ["Research findings", "Analysis results"]
        }


# ============================================================================
# Messaging Tests
# ============================================================================


class TestTeamMessaging:
    """Tests for team messaging functionality."""

    @pytest.mark.asyncio
    async def test_send_message(self, populated_team: Team):
        """Test sending a message between team members."""
        await populated_team.start()

        members = populated_team.members
        sender = members[0]
        recipient = members[1]

        recipients = await populated_team.send_message(
            sender_id=sender.id,
            recipient_id=recipient.id,
            content={"data": "test"},
        )

        assert recipient.id in recipients

    @pytest.mark.asyncio
    async def test_send_message_no_router(
        self,
        mock_researcher: MockAgent,
    ):
        """Test sending message when router disabled."""
        config = TeamConfig(enable_messaging=False)
        team = Team(config=config)
        team.add_member(mock_researcher)

        await team.start()

        recipients = await team.send_message(
            sender_id=mock_researcher.id,
            content={"data": "test"},
            strategy=RoutingStrategy.BROADCAST,
        )

        assert recipients == []


# ============================================================================
# Task History and Statistics Tests
# ============================================================================


class TestTaskHistoryAndStatistics:
    """Tests for task history and statistics."""

    @pytest.mark.asyncio
    async def test_task_history(self, populated_team: Team):
        """Test that task history is recorded."""
        await populated_team.run("Task 1")
        await populated_team.run("Task 2")

        history = populated_team.get_task_history()

        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_get_statistics(self, populated_team: Team):
        """Test getting team statistics."""
        await populated_team.run("Task 1")
        await populated_team.run("Task 2")

        stats = populated_team.get_statistics()

        assert stats["team_name"] == populated_team.config.name
        assert stats["member_count"] == 3
        assert stats["total_tasks"] == 2
        assert stats["successful_tasks"] == 2
        assert stats["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_statistics_with_failures(
        self,
        team: Team,
        mock_researcher: MockAgent,
        failing_agent: MockAgent,
    ):
        """Test statistics with failed tasks."""
        team.add_member(mock_researcher)
        team.add_member(failing_agent)

        await team.run("Task")

        stats = team.get_statistics()

        assert stats["total_tasks"] == 1
        assert stats["successful_tasks"] == 0


# ============================================================================
# Timeout and Error Handling Tests
# ============================================================================


class TestTimeoutAndErrorHandling:
    """Tests for timeout and error handling."""

    @pytest.mark.asyncio
    async def test_task_timeout(
        self,
        slow_agent: MockAgent,
    ):
        """Test that slow tasks timeout."""
        config = TeamConfig(
            name="TimeoutTeam",
            task_timeout=0.1,  # Very short timeout
        )
        team = Team(config=config)
        team.add_member(slow_agent)

        result = await team.run("Slow task")

        assert not result.success
        assert any("timed out" in error for error in result.errors)

    @pytest.mark.asyncio
    async def test_exception_handling(
        self,
        team: Team,
        mock_researcher: MockAgent,
    ):
        """Test that exceptions are handled gracefully."""
        # Create agent that raises exception
        class ExceptionAgent(MockAgent):
            async def run(self, task):
                raise RuntimeError("Unexpected error")

        error_agent = ExceptionAgent(
            role=AgentRole(name="error", description="test", capabilities=[]),
        )
        team.add_member(error_agent)

        result = await team.run("Test task")

        # Should complete with failure, not crash
        assert result is not None
        assert not result.success


# ============================================================================
# Cleanup Tests
# ============================================================================


class TestCleanup:
    """Tests for team cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_on_stop(self, populated_team: Team):
        """Test that cleanup happens on stop."""
        await populated_team.start()
        await populated_team.shared_context.set("key", "value")

        await populated_team.stop()

        # Context should be cleared
        assert await populated_team.shared_context.get("key") is None

    @pytest.mark.asyncio
    async def test_no_cleanup_when_disabled(
        self,
        mock_researcher: MockAgent,
    ):
        """Test no cleanup when auto_cleanup disabled."""
        config = TeamConfig(auto_cleanup=False)
        team = Team(config=config)
        team.add_member(mock_researcher)

        await team.start()
        await team.shared_context.set("key", "value")

        await team.stop()

        # Context should not be cleared
        assert await team.shared_context.get("key") == "value"
