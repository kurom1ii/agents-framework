"""Unit tests for the worker agent module."""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents_framework.agents.base import (
    AgentRole,
    AgentStatus,
    Task,
    TaskResult,
)
from agents_framework.agents.worker import (
    ProgressCallback,
    WorkerAgent,
    WorkerConfig,
    WorkerProgress,
    WorkerStatus,
)


class TestWorkerStatus:
    """Tests for WorkerStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert WorkerStatus.WAITING.value == "waiting"
        assert WorkerStatus.EXECUTING.value == "executing"
        assert WorkerStatus.PAUSED.value == "paused"
        assert WorkerStatus.COMPLETED.value == "completed"
        assert WorkerStatus.CANCELLED.value == "cancelled"
        assert WorkerStatus.FAILED.value == "failed"


class TestWorkerProgress:
    """Tests for WorkerProgress dataclass."""

    def test_progress_creation_minimal(self):
        """Test creating progress with minimal parameters."""
        progress = WorkerProgress(
            worker_id="worker-123",
            task_id="task-456",
        )
        assert progress.worker_id == "worker-123"
        assert progress.task_id == "task-456"
        assert progress.status == WorkerStatus.WAITING
        assert progress.current_step == 0
        assert progress.total_steps is None
        assert progress.message == ""
        assert progress.started_at is None
        assert progress.metadata == {}

    def test_progress_creation_full(self):
        """Test creating progress with all parameters."""
        started = datetime.now()
        progress = WorkerProgress(
            worker_id="worker-123",
            task_id="task-456",
            status=WorkerStatus.EXECUTING,
            current_step=3,
            total_steps=10,
            message="Processing step 3",
            started_at=started,
            metadata={"key": "value"},
        )
        assert progress.status == WorkerStatus.EXECUTING
        assert progress.current_step == 3
        assert progress.total_steps == 10
        assert progress.message == "Processing step 3"
        assert progress.started_at == started

    def test_progress_percentage_with_steps(self):
        """Test progress percentage calculation with defined steps."""
        progress = WorkerProgress(
            worker_id="w",
            task_id="t",
            current_step=5,
            total_steps=10,
        )
        assert progress.progress_percentage == 50.0

    def test_progress_percentage_no_total(self):
        """Test progress percentage when total is not set."""
        progress = WorkerProgress(
            worker_id="w",
            task_id="t",
            current_step=5,
            total_steps=None,
        )
        assert progress.progress_percentage == 0.0

    def test_progress_percentage_zero_total(self):
        """Test progress percentage when total is zero."""
        progress = WorkerProgress(
            worker_id="w",
            task_id="t",
            current_step=5,
            total_steps=0,
        )
        assert progress.progress_percentage == 0.0

    def test_progress_percentage_capped_at_100(self):
        """Test progress percentage is capped at 100."""
        progress = WorkerProgress(
            worker_id="w",
            task_id="t",
            current_step=15,
            total_steps=10,
        )
        assert progress.progress_percentage == 100.0

    def test_elapsed_time_not_started(self):
        """Test elapsed time when not started."""
        progress = WorkerProgress(
            worker_id="w",
            task_id="t",
            started_at=None,
        )
        assert progress.elapsed_time == 0.0

    def test_elapsed_time_with_start(self):
        """Test elapsed time calculation."""
        started = datetime.now()
        progress = WorkerProgress(
            worker_id="w",
            task_id="t",
            started_at=started,
            updated_at=started,
        )
        # Should be very close to 0
        assert progress.elapsed_time >= 0.0


class TestWorkerConfig:
    """Tests for WorkerConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = WorkerConfig()
        assert config.allowed_tools == []
        assert config.denied_tools == []
        assert config.report_progress is True
        assert config.progress_interval == 1.0
        assert config.allow_cancellation is True
        assert config.checkpoint_enabled is False

    def test_config_custom(self):
        """Test custom configuration values."""
        config = WorkerConfig(
            name="custom-worker",
            allowed_tools=["search", "calculate"],
            denied_tools=["dangerous"],
            report_progress=False,
            progress_interval=0.5,
            allow_cancellation=False,
            checkpoint_enabled=True,
        )
        assert config.name == "custom-worker"
        assert config.allowed_tools == ["search", "calculate"]
        assert config.denied_tools == ["dangerous"]
        assert config.report_progress is False
        assert config.progress_interval == 0.5
        assert config.allow_cancellation is False
        assert config.checkpoint_enabled is True


class TestWorkerAgent:
    """Tests for WorkerAgent class."""

    def test_worker_initialization(self, basic_role, worker_config):
        """Test worker agent initialization."""
        worker = WorkerAgent(role=basic_role, config=worker_config)

        assert worker.role == basic_role
        assert worker.config == worker_config
        assert worker.worker_status == WorkerStatus.WAITING
        assert worker.current_progress is None
        assert worker.is_cancelled() is False
        assert worker._progress_callbacks == []

    def test_worker_default_config(self, basic_role):
        """Test worker with default config."""
        worker = WorkerAgent(role=basic_role)
        assert isinstance(worker.config, WorkerConfig)

    def test_worker_config_property(self, basic_role, worker_config):
        """Test worker_config property returns WorkerConfig."""
        worker = WorkerAgent(role=basic_role, config=worker_config)
        assert isinstance(worker.worker_config, WorkerConfig)
        assert worker.worker_config == worker_config

    def test_worker_config_property_with_base_config(self, basic_role, basic_config):
        """Test worker_config property converts base config."""
        worker = WorkerAgent(role=basic_role, config=basic_config)
        worker_cfg = worker.worker_config
        assert isinstance(worker_cfg, WorkerConfig)
        assert worker_cfg.name == basic_config.name
        assert worker_cfg.max_iterations == basic_config.max_iterations

    def test_add_progress_callback(self, basic_role):
        """Test adding a progress callback."""
        worker = WorkerAgent(role=basic_role)
        callback = MagicMock()

        worker.add_progress_callback(callback)

        assert callback in worker._progress_callbacks

    def test_remove_progress_callback(self, basic_role):
        """Test removing a progress callback."""
        worker = WorkerAgent(role=basic_role)
        callback = MagicMock()
        worker.add_progress_callback(callback)

        worker.remove_progress_callback(callback)

        assert callback not in worker._progress_callbacks

    def test_remove_nonexistent_callback(self, basic_role):
        """Test removing a callback that doesn't exist."""
        worker = WorkerAgent(role=basic_role)
        callback = MagicMock()

        # Should not raise
        worker.remove_progress_callback(callback)

    def test_cancel_when_not_executing(self, basic_role, worker_config):
        """Test cancellation when not executing."""
        worker = WorkerAgent(role=basic_role, config=worker_config)

        result = worker.cancel()

        assert result is False

    def test_cancel_when_cancellation_disabled(self, basic_role):
        """Test cancellation when disabled in config."""
        config = WorkerConfig(allow_cancellation=False)
        worker = WorkerAgent(role=basic_role, config=config)
        worker._worker_status = WorkerStatus.EXECUTING

        result = worker.cancel()

        assert result is False

    def test_cancel_when_executing(self, basic_role, worker_config):
        """Test successful cancellation request."""
        worker = WorkerAgent(role=basic_role, config=worker_config)
        worker._worker_status = WorkerStatus.EXECUTING

        result = worker.cancel()

        assert result is True
        assert worker.is_cancelled() is True
        assert worker.worker_status == WorkerStatus.CANCELLED

    def test_cancel_when_paused(self, basic_role, worker_config):
        """Test cancellation from paused state."""
        worker = WorkerAgent(role=basic_role, config=worker_config)
        worker._worker_status = WorkerStatus.PAUSED

        result = worker.cancel()

        assert result is True
        assert worker.worker_status == WorkerStatus.CANCELLED

    def test_is_cancelled(self, basic_role):
        """Test is_cancelled method."""
        worker = WorkerAgent(role=basic_role)

        assert worker.is_cancelled() is False

        worker._cancel_requested = True
        assert worker.is_cancelled() is True

    def test_worker_status_property(self, basic_role):
        """Test worker_status property."""
        worker = WorkerAgent(role=basic_role)
        assert worker.worker_status == WorkerStatus.WAITING

        worker._worker_status = WorkerStatus.EXECUTING
        assert worker.worker_status == WorkerStatus.EXECUTING

    def test_current_progress_property(self, basic_role):
        """Test current_progress property."""
        worker = WorkerAgent(role=basic_role)
        assert worker.current_progress is None

        progress = WorkerProgress(worker_id=worker.id, task_id="task-123")
        worker._current_progress = progress
        assert worker.current_progress == progress

    def test_build_default_system_prompt(self, basic_role):
        """Test default system prompt generation."""
        worker = WorkerAgent(role=basic_role)
        prompt = worker._build_default_system_prompt()

        assert basic_role.name in prompt
        assert basic_role.description in prompt
        assert "test" in prompt  # From capabilities

    def test_custom_system_prompt(self, basic_role):
        """Test custom system prompt."""
        custom_prompt = "You are a specialized worker."
        worker = WorkerAgent(role=basic_role, system_prompt=custom_prompt)

        assert worker.system_prompt == custom_prompt

    def test_get_status_info(self, basic_role):
        """Test getting status information."""
        worker = WorkerAgent(role=basic_role)
        info = worker.get_status_info()

        assert info["id"] == worker.id
        assert info["role"] == basic_role.name
        assert info["status"] == AgentStatus.IDLE.value
        assert info["worker_status"] == WorkerStatus.WAITING.value
        assert info["cancel_requested"] is False
        assert info["progress"] is None

    def test_get_status_info_with_progress(self, basic_role):
        """Test status info includes progress when available."""
        worker = WorkerAgent(role=basic_role)
        worker._current_progress = WorkerProgress(
            worker_id=worker.id,
            task_id="task-123",
            current_step=5,
            total_steps=10,
            message="Processing",
            started_at=datetime.now(),
        )

        info = worker.get_status_info()

        assert info["progress"] is not None
        assert info["progress"]["current_step"] == 5
        assert info["progress"]["total_steps"] == 10
        assert info["progress"]["percentage"] == 50.0
        assert info["progress"]["message"] == "Processing"

    def test_repr(self, basic_role):
        """Test string representation."""
        worker = WorkerAgent(role=basic_role)
        repr_str = repr(worker)

        assert "WorkerAgent" in repr_str
        assert worker.id in repr_str
        assert basic_role.name in repr_str
        assert WorkerStatus.WAITING.value in repr_str

    @pytest.mark.asyncio
    async def test_run_without_llm(self, basic_role, basic_task):
        """Test run fails without LLM provider."""
        worker = WorkerAgent(role=basic_role)

        result = await worker.run(basic_task)

        assert result.success is False
        assert "No LLM provider" in result.error

    @pytest.mark.asyncio
    async def test_run_with_string_task(self, basic_role, mock_llm):
        """Test run with string task converts to Task."""
        worker = WorkerAgent(role=basic_role, llm=mock_llm)

        # Mock the AgentLoop to avoid complex execution
        with patch(
            "agents_framework.agents.worker.AgentLoop"
        ) as MockLoop:
            mock_loop_instance = MagicMock()
            mock_state = MagicMock()
            mock_state.status = "completed"
            mock_state.error = None
            mock_state.iteration = 1
            mock_state.total_tokens = 100
            mock_state.duration = 0.5
            mock_state.termination_reason = None
            mock_state.steps = []
            mock_loop_instance.run = AsyncMock(return_value=mock_state)
            MockLoop.return_value = mock_loop_instance

            result = await worker.run("Do something")

        assert worker.worker_status == WorkerStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_updates_status(self, basic_role, mock_llm):
        """Test that run updates worker status."""
        worker = WorkerAgent(role=basic_role, llm=mock_llm)

        with patch(
            "agents_framework.agents.worker.AgentLoop"
        ) as MockLoop:
            mock_loop_instance = MagicMock()
            mock_state = MagicMock()
            mock_state.status = "completed"
            mock_state.error = None
            mock_state.iteration = 1
            mock_state.total_tokens = 100
            mock_state.duration = 0.5
            mock_state.termination_reason = None
            mock_state.steps = []
            mock_loop_instance.run = AsyncMock(return_value=mock_state)
            MockLoop.return_value = mock_loop_instance

            await worker.run("Task")

        assert worker.worker_status == WorkerStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_initializes_progress(self, basic_role, mock_llm):
        """Test that run initializes progress tracking."""
        worker = WorkerAgent(role=basic_role, llm=mock_llm)

        with patch(
            "agents_framework.agents.worker.AgentLoop"
        ) as MockLoop:
            mock_loop_instance = MagicMock()
            mock_state = MagicMock()
            mock_state.status = "completed"
            mock_state.error = None
            mock_state.iteration = 1
            mock_state.total_tokens = 100
            mock_state.duration = 0.5
            mock_state.termination_reason = None
            mock_state.steps = []
            mock_loop_instance.run = AsyncMock(return_value=mock_state)
            MockLoop.return_value = mock_loop_instance

            await worker.run("Task")

        assert worker.current_progress is not None
        assert worker.current_progress.worker_id == worker.id

    @pytest.mark.asyncio
    async def test_run_handles_cancellation(self, basic_role, mock_llm):
        """Test that run handles cancellation."""
        worker = WorkerAgent(role=basic_role, llm=mock_llm)

        with patch(
            "agents_framework.agents.worker.AgentLoop"
        ) as MockLoop:
            mock_loop_instance = MagicMock()

            async def slow_run(*args, **kwargs):
                # Simulate cancellation during run
                worker._cancel_requested = True
                mock_state = MagicMock()
                mock_state.status = "completed"
                mock_state.error = None
                mock_state.iteration = 1
                mock_state.total_tokens = 100
                mock_state.duration = 0.5
                mock_state.termination_reason = None
                mock_state.steps = []
                return mock_state

            mock_loop_instance.run = slow_run
            MockLoop.return_value = mock_loop_instance

            result = await worker.run("Task")

        assert result.success is False
        assert "cancelled" in result.error.lower()
        assert worker.worker_status == WorkerStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_run_handles_exception(self, basic_role, mock_llm):
        """Test that run handles exceptions gracefully."""
        worker = WorkerAgent(role=basic_role, llm=mock_llm)

        with patch(
            "agents_framework.agents.worker.AgentLoop"
        ) as MockLoop:
            mock_loop_instance = MagicMock()
            mock_loop_instance.run = AsyncMock(
                side_effect=RuntimeError("Test error")
            )
            MockLoop.return_value = mock_loop_instance

            result = await worker.run("Task")

        assert result.success is False
        assert "Test error" in result.error
        assert worker.worker_status == WorkerStatus.FAILED

    @pytest.mark.asyncio
    async def test_run_with_progress_callback(self, basic_role, mock_llm):
        """Test run with progress callback."""
        worker = WorkerAgent(role=basic_role, llm=mock_llm)
        callback_called = False

        def on_progress(progress):
            nonlocal callback_called
            callback_called = True

        with patch(
            "agents_framework.agents.worker.AgentLoop"
        ) as MockLoop:
            mock_loop_instance = MagicMock()
            mock_state = MagicMock()
            mock_state.status = "completed"
            mock_state.error = None
            mock_state.iteration = 1
            mock_state.total_tokens = 100
            mock_state.duration = 0.5
            mock_state.termination_reason = None
            mock_state.steps = []
            mock_loop_instance.run = AsyncMock(return_value=mock_state)
            MockLoop.return_value = mock_loop_instance

            result = await worker.run_with_progress("Task", on_progress)

        assert result is not None
        # Callback should have been added and removed
        assert on_progress not in worker._progress_callbacks

    @pytest.mark.asyncio
    async def test_run_with_progress_removes_callback_on_exception(
        self, basic_role, mock_llm
    ):
        """Test that progress callback is removed even on exception."""
        worker = WorkerAgent(role=basic_role, llm=mock_llm)
        callback = MagicMock()

        with patch(
            "agents_framework.agents.worker.AgentLoop"
        ) as MockLoop:
            mock_loop_instance = MagicMock()
            mock_loop_instance.run = AsyncMock(
                side_effect=RuntimeError("Test error")
            )
            MockLoop.return_value = mock_loop_instance

            await worker.run_with_progress("Task", callback)

        # Callback should still be removed
        assert callback not in worker._progress_callbacks

    @pytest.mark.asyncio
    async def test_report_progress_respects_interval(self, basic_role):
        """Test that progress reporting respects interval."""
        config = WorkerConfig(report_progress=True, progress_interval=1.0)
        worker = WorkerAgent(role=basic_role, config=config)
        worker._current_progress = WorkerProgress(
            worker_id=worker.id, task_id="task-123"
        )

        callback = MagicMock()
        worker.add_progress_callback(callback)

        # First call should work
        await worker._report_progress()
        first_call_count = callback.call_count

        # Immediate second call should be skipped (interval not elapsed)
        await worker._report_progress()
        assert callback.call_count == first_call_count

    @pytest.mark.asyncio
    async def test_report_progress_disabled(self, basic_role):
        """Test that progress reporting can be disabled."""
        config = WorkerConfig(report_progress=False)
        worker = WorkerAgent(role=basic_role, config=config)
        worker._current_progress = WorkerProgress(
            worker_id=worker.id, task_id="task-123"
        )

        callback = MagicMock()
        worker.add_progress_callback(callback)

        await worker._report_progress()

        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_report_progress_no_current_progress(self, basic_role):
        """Test reporting with no current progress."""
        worker = WorkerAgent(role=basic_role)
        callback = MagicMock()
        worker.add_progress_callback(callback)

        await worker._report_progress()

        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_report_progress_async_callback(self, basic_role):
        """Test reporting with async callback."""
        config = WorkerConfig(report_progress=True, progress_interval=0.0)
        worker = WorkerAgent(role=basic_role, config=config)
        worker._current_progress = WorkerProgress(
            worker_id=worker.id, task_id="task-123"
        )

        callback_called = False

        async def async_callback(progress):
            nonlocal callback_called
            callback_called = True

        worker.add_progress_callback(async_callback)
        await worker._report_progress()

        assert callback_called is True

    @pytest.mark.asyncio
    async def test_report_progress_handles_callback_error(self, basic_role):
        """Test that callback errors are handled gracefully."""
        config = WorkerConfig(report_progress=True, progress_interval=0.0)
        worker = WorkerAgent(role=basic_role, config=config)
        worker._current_progress = WorkerProgress(
            worker_id=worker.id, task_id="task-123"
        )

        def failing_callback(progress):
            raise ValueError("Callback error")

        worker.add_progress_callback(failing_callback)

        # Should not raise
        await worker._report_progress()

    def test_get_filtered_tool_registry_no_registry(self, basic_role):
        """Test filtered registry when no registry is set."""
        worker = WorkerAgent(role=basic_role)
        result = worker._get_filtered_tool_registry()
        assert result is None

    @pytest.mark.asyncio
    async def test_run_result_contains_metadata(self, basic_role, mock_llm):
        """Test that run result contains execution metadata."""
        worker = WorkerAgent(role=basic_role, llm=mock_llm)

        with patch(
            "agents_framework.agents.worker.AgentLoop"
        ) as MockLoop:
            mock_loop_instance = MagicMock()
            mock_state = MagicMock()
            mock_state.status = "completed"
            mock_state.error = None
            mock_state.iteration = 5
            mock_state.total_tokens = 500
            mock_state.duration = 2.5
            mock_state.termination_reason = None
            mock_state.steps = []
            mock_loop_instance.run = AsyncMock(return_value=mock_state)
            MockLoop.return_value = mock_loop_instance

            result = await worker.run("Task")

        assert result.metadata["iterations"] == 5
        assert result.metadata["total_tokens"] == 500
        assert result.metadata["duration_seconds"] == 2.5

    @pytest.mark.asyncio
    async def test_run_extracts_output_from_steps(self, basic_role, mock_llm):
        """Test that run extracts output from last step."""
        worker = WorkerAgent(role=basic_role, llm=mock_llm)

        with patch(
            "agents_framework.agents.worker.AgentLoop"
        ) as MockLoop:
            mock_loop_instance = MagicMock()
            mock_state = MagicMock()
            mock_state.status = "completed"
            mock_state.error = None
            mock_state.iteration = 2
            mock_state.total_tokens = 200
            mock_state.duration = 1.0
            mock_state.termination_reason = None

            # Create mock steps with content
            mock_step1 = MagicMock()
            mock_step1.content = "First step output"
            mock_step2 = MagicMock()
            mock_step2.content = "Final step output"
            mock_state.steps = [mock_step1, mock_step2]

            mock_loop_instance.run = AsyncMock(return_value=mock_state)
            MockLoop.return_value = mock_loop_instance

            result = await worker.run("Task")

        assert result.output == "Final step output"
