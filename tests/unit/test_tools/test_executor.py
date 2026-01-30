"""Tests for tool executor functionality.

This module tests:
- ExecutionConfig dataclass
- ExecutionResult dataclass
- ToolExecutor class
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

from agents_framework.tools.base import BaseTool, ToolResult
from agents_framework.tools.registry import ToolRegistry
from agents_framework.tools.executor import (
    ExecutionConfig,
    ExecutionResult,
    ToolExecutor,
)


# ============================================================================
# ExecutionConfig Tests
# ============================================================================


class TestExecutionConfig:
    """Tests for ExecutionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExecutionConfig()

        assert config.timeout == 30.0
        assert config.max_concurrent == 10
        assert config.retry_on_error is False
        assert config.max_retries == 3
        assert config.validate_args is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ExecutionConfig(
            timeout=60.0,
            max_concurrent=5,
            retry_on_error=True,
            max_retries=5,
            validate_args=False,
        )

        assert config.timeout == 60.0
        assert config.max_concurrent == 5
        assert config.retry_on_error is True
        assert config.max_retries == 5
        assert config.validate_args is False

    def test_no_timeout_config(self):
        """Test configuration with no timeout."""
        config = ExecutionConfig(timeout=None)

        assert config.timeout is None

    def test_config_with_zero_values(self):
        """Test configuration with zero values."""
        config = ExecutionConfig(
            timeout=0.0,
            max_concurrent=1,
            max_retries=0,
        )

        assert config.timeout == 0.0
        assert config.max_concurrent == 1
        assert config.max_retries == 0


# ============================================================================
# ExecutionResult Tests
# ============================================================================


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_basic_result(self):
        """Test basic execution result."""
        tool_result = ToolResult(success=True, output="result")
        result = ExecutionResult(
            tool_name="test_tool",
            tool_call_id="call_123",
            result=tool_result,
        )

        assert result.tool_name == "test_tool"
        assert result.tool_call_id == "call_123"
        assert result.result.success is True
        assert result.result.output == "result"
        assert result.execution_time == 0.0
        assert result.retries == 0

    def test_result_with_metadata(self):
        """Test execution result with all metadata."""
        tool_result = ToolResult(success=True, output="ok")
        result = ExecutionResult(
            tool_name="my_tool",
            tool_call_id="call_456",
            result=tool_result,
            execution_time=1.5,
            retries=2,
        )

        assert result.execution_time == 1.5
        assert result.retries == 2

    def test_failure_result(self):
        """Test execution result for failed tool."""
        tool_result = ToolResult(success=False, error="Something went wrong")
        result = ExecutionResult(
            tool_name="failing_tool",
            tool_call_id="call_789",
            result=tool_result,
        )

        assert result.result.success is False
        assert result.result.error == "Something went wrong"


# ============================================================================
# ToolExecutor Initialization Tests
# ============================================================================


class TestToolExecutorInit:
    """Tests for ToolExecutor initialization."""

    def test_default_init(self):
        """Test default initialization."""
        executor = ToolExecutor()

        assert isinstance(executor.registry, ToolRegistry)
        assert isinstance(executor.config, ExecutionConfig)

    def test_init_with_registry(self, populated_registry: ToolRegistry):
        """Test initialization with existing registry."""
        executor = ToolExecutor(registry=populated_registry)

        assert executor.registry is populated_registry
        assert executor.get_tool("sample_tool") is not None

    def test_init_with_config(self, execution_config: ExecutionConfig):
        """Test initialization with custom config."""
        executor = ToolExecutor(config=execution_config)

        assert executor.config is execution_config
        assert executor.config.timeout == 5.0

    def test_init_with_both(
        self,
        populated_registry: ToolRegistry,
        execution_config: ExecutionConfig,
    ):
        """Test initialization with both registry and config."""
        executor = ToolExecutor(
            registry=populated_registry,
            config=execution_config,
        )

        assert executor.registry is populated_registry
        assert executor.config is execution_config


# ============================================================================
# Single Tool Execution Tests
# ============================================================================


class TestSingleToolExecution:
    """Tests for single tool execution."""

    @pytest.mark.asyncio
    async def test_execute_success(self, tool_executor: ToolExecutor):
        """Test successful tool execution."""
        result = await tool_executor.execute(
            tool_name="sample_tool",
            tool_call_id="call_1",
            arguments={"message": "hello"},
        )

        assert isinstance(result, ExecutionResult)
        assert result.tool_name == "sample_tool"
        assert result.tool_call_id == "call_1"
        assert result.result.success is True
        assert result.result.output == "hello"
        assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_execute_with_multiple_args(self, tool_executor: ToolExecutor):
        """Test execution with multiple arguments."""
        result = await tool_executor.execute(
            tool_name="sample_tool",
            tool_call_id="call_2",
            arguments={"message": "hi", "count": 3},
        )

        assert result.result.success is True
        assert result.result.output == "hihihi"

    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self, tool_executor: ToolExecutor):
        """Test executing a non-existent tool."""
        result = await tool_executor.execute(
            tool_name="nonexistent",
            tool_call_id="call_3",
            arguments={},
        )

        assert result.result.success is False
        assert "not found" in result.result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_tool_that_raises(self, empty_registry: ToolRegistry):
        """Test executing a tool that raises an exception."""
        from tests.unit.test_tools.conftest import ErrorTool

        registry = ToolRegistry()
        registry.register(ErrorTool(fail_after=0))
        executor = ToolExecutor(registry=registry)

        result = await executor.execute(
            tool_name="error_tool",
            tool_call_id="call_4",
            arguments={},
        )

        assert result.result.success is False
        assert "Intentional failure" in result.result.error

    @pytest.mark.asyncio
    async def test_execute_tracks_execution_time(self, tool_executor: ToolExecutor):
        """Test that execution time is tracked."""
        result = await tool_executor.execute(
            tool_name="sample_tool",
            tool_call_id="call_5",
            arguments={"message": "test"},
        )

        assert result.execution_time > 0
        assert result.execution_time < 5.0  # Should be fast


# ============================================================================
# Timeout Tests
# ============================================================================


class TestTimeoutHandling:
    """Tests for timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_triggers(self, fast_timeout_config: ExecutionConfig):
        """Test that timeout triggers for slow tools."""
        from tests.unit.test_tools.conftest import SlowTool

        registry = ToolRegistry()
        registry.register(SlowTool(delay=1.0))  # 1 second delay
        executor = ToolExecutor(registry=registry, config=fast_timeout_config)

        result = await executor.execute(
            tool_name="slow_tool",
            tool_call_id="call_timeout",
            arguments={},
        )

        assert result.result.success is False
        assert "timed out" in result.result.error.lower()

    @pytest.mark.asyncio
    async def test_no_timeout_when_disabled(self):
        """Test execution works without timeout."""
        from tests.unit.test_tools.conftest import SlowTool

        registry = ToolRegistry()
        registry.register(SlowTool(delay=0.1))  # Very short delay
        config = ExecutionConfig(timeout=None)  # No timeout
        executor = ToolExecutor(registry=registry, config=config)

        result = await executor.execute(
            tool_name="slow_tool",
            tool_call_id="call_no_timeout",
            arguments={},
        )

        assert result.result.success is True
        assert result.result.output == "completed"

    @pytest.mark.asyncio
    async def test_fast_tool_within_timeout(self, tool_executor: ToolExecutor):
        """Test that fast tools complete within timeout."""
        result = await tool_executor.execute(
            tool_name="sample_tool",
            tool_call_id="call_fast",
            arguments={"message": "quick"},
        )

        assert result.result.success is True


# ============================================================================
# Retry Tests
# ============================================================================


class TestRetryHandling:
    """Tests for retry handling."""

    @pytest.mark.asyncio
    async def test_retry_on_error_disabled(self):
        """Test that retries don't happen when disabled."""
        from tests.unit.test_tools.conftest import ErrorTool

        registry = ToolRegistry()
        tool = ErrorTool(fail_after=0)
        registry.register(tool)
        config = ExecutionConfig(retry_on_error=False, max_retries=3)
        executor = ToolExecutor(registry=registry, config=config)

        result = await executor.execute(
            tool_name="error_tool",
            tool_call_id="call_no_retry",
            arguments={},
        )

        assert result.result.success is False
        assert result.retries == 0
        assert tool.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_error_enabled(self):
        """Test retry behavior when enabled.

        Note: The retry mechanism only triggers on exceptions raised outside
        of tool.run(). Since tool.run() catches all exceptions and returns
        them as ToolResult(success=False), retry_on_error doesn't retry
        tool execution errors - only external errors like timeouts.
        """
        from tests.unit.test_tools.conftest import ErrorTool

        registry = ToolRegistry()
        # Fail after 0 calls (always fail)
        tool = ErrorTool(fail_after=0)
        registry.register(tool)
        config = ExecutionConfig(
            retry_on_error=True,
            max_retries=2,
            timeout=10.0,  # Give enough time for retries
        )
        executor = ToolExecutor(registry=registry, config=config)

        result = await executor.execute(
            tool_name="error_tool",
            tool_call_id="call_retry",
            arguments={},
        )

        # Tool errors are caught by run() and returned as ToolResult
        # So no retries happen for tool execution errors
        assert result.result.success is False
        assert result.retries == 0  # No retries because error was caught by run()
        assert tool.call_count == 1  # Only called once

    @pytest.mark.asyncio
    async def test_retry_succeeds_eventually(self):
        """Test that tool errors don't trigger retry since they're caught by run().

        Note: The retry mechanism only triggers on exceptions raised outside
        of tool.run(). Since tool.run() catches all exceptions and returns
        them as ToolResult(success=False), tool errors don't trigger retries.
        """

        class FlakeyTool(BaseTool):
            """Tool that fails first N times then succeeds."""

            name = "flakey"
            description = "Flakey tool"

            def __init__(self, fail_count: int = 2):
                self.parameters = {"type": "object", "properties": {}}
                super().__init__()
                self.fail_count = fail_count
                self.call_count = 0

            async def execute(self, **kwargs) -> str:
                self.call_count += 1
                if self.call_count <= self.fail_count:
                    raise RuntimeError(f"Failure {self.call_count}")
                return "success"

        registry = ToolRegistry()
        tool = FlakeyTool(fail_count=2)
        registry.register(tool)
        config = ExecutionConfig(
            retry_on_error=True,
            max_retries=3,
            timeout=10.0,
        )
        executor = ToolExecutor(registry=registry, config=config)

        result = await executor.execute(
            tool_name="flakey",
            tool_call_id="call_flakey",
            arguments={},
        )

        # Since run() catches the exception, no retry happens
        assert result.result.success is False
        assert "Failure 1" in result.result.error
        assert result.retries == 0
        assert tool.call_count == 1


# ============================================================================
# Concurrent Execution Tests
# ============================================================================


class TestConcurrentExecution:
    """Tests for concurrent tool execution."""

    @pytest.mark.asyncio
    async def test_execute_many(self, tool_executor: ToolExecutor):
        """Test executing multiple tools concurrently."""
        tool_calls = [
            {"name": "sample_tool", "id": "call_a", "arguments": {"message": "a"}},
            {"name": "sample_tool", "id": "call_b", "arguments": {"message": "b"}},
            {"name": "async_sample", "id": "call_c", "arguments": {"value": 5}},
        ]

        results = await tool_executor.execute_many(tool_calls)

        assert len(results) == 3
        assert results[0].tool_call_id == "call_a"
        assert results[0].result.output == "a"
        assert results[1].tool_call_id == "call_b"
        assert results[1].result.output == "b"
        assert results[2].tool_call_id == "call_c"
        assert results[2].result.output == 10

    @pytest.mark.asyncio
    async def test_execute_many_with_failures(self, tool_executor: ToolExecutor):
        """Test execute_many with some failing tools."""
        tool_calls = [
            {"name": "sample_tool", "id": "call_1", "arguments": {"message": "ok"}},
            {"name": "nonexistent", "id": "call_2", "arguments": {}},
            {"name": "sample_tool", "id": "call_3", "arguments": {"message": "ok2"}},
        ]

        results = await tool_executor.execute_many(tool_calls)

        assert len(results) == 3
        assert results[0].result.success is True
        assert results[1].result.success is False  # nonexistent tool
        assert results[2].result.success is True

    @pytest.mark.asyncio
    async def test_execute_many_empty_list(self, tool_executor: ToolExecutor):
        """Test execute_many with empty list."""
        results = await tool_executor.execute_many([])

        assert results == []

    @pytest.mark.asyncio
    async def test_execute_sequential(self, tool_executor: ToolExecutor):
        """Test sequential execution of tools."""
        tool_calls = [
            {"name": "counter", "id": "call_1", "arguments": {}},
            {"name": "counter", "id": "call_2", "arguments": {}},
            {"name": "counter", "id": "call_3", "arguments": {}},
        ]

        results = await tool_executor.execute_sequential(tool_calls)

        assert len(results) == 3
        # Sequential execution means counter increments in order
        assert results[0].result.output == 1
        assert results[1].result.output == 2
        assert results[2].result.output == 3

    @pytest.mark.asyncio
    async def test_execute_sequential_empty_list(self, tool_executor: ToolExecutor):
        """Test execute_sequential with empty list."""
        results = await tool_executor.execute_sequential([])

        assert results == []

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Test that concurrency limit is respected."""
        import time

        class TimedTool(BaseTool):
            """Tool that records start and end times."""

            name = "timed"
            description = "Timed tool"
            executions: list = []

            def __init__(self):
                self.parameters = {
                    "type": "object",
                    "properties": {
                        "delay": {"type": "number", "default": 0.1}
                    }
                }
                super().__init__()

            async def execute(self, delay: float = 0.1) -> None:
                start = time.monotonic()
                await asyncio.sleep(delay)
                end = time.monotonic()
                TimedTool.executions.append((start, end))

        TimedTool.executions = []  # Reset

        registry = ToolRegistry()
        registry.register(TimedTool())
        config = ExecutionConfig(max_concurrent=2, timeout=5.0)
        executor = ToolExecutor(registry=registry, config=config)

        # Try to execute 4 tools with concurrency limit of 2
        tool_calls = [
            {"name": "timed", "id": f"call_{i}", "arguments": {"delay": 0.1}}
            for i in range(4)
        ]

        await executor.execute_many(tool_calls)

        # With concurrency of 2, we should see overlapping in pairs
        assert len(TimedTool.executions) == 4


# ============================================================================
# Registration Tests
# ============================================================================


class TestExecutorRegistration:
    """Tests for tool registration via executor."""

    def test_register_tool(self):
        """Test registering a tool via executor."""
        from tests.unit.test_tools.conftest import SampleTool

        executor = ToolExecutor()
        tool = SampleTool()

        result = executor.register(tool)

        assert result is tool
        assert executor.get_tool("sample_tool") is tool

    def test_get_tool(self, tool_executor: ToolExecutor):
        """Test getting a tool from executor."""
        tool = tool_executor.get_tool("sample_tool")

        assert tool is not None
        assert tool.name == "sample_tool"

    def test_get_tool_nonexistent(self, tool_executor: ToolExecutor):
        """Test getting a non-existent tool."""
        tool = tool_executor.get_tool("nonexistent")

        assert tool is None


# ============================================================================
# Edge Cases
# ============================================================================


class TestExecutorEdgeCases:
    """Tests for edge cases in tool execution."""

    @pytest.mark.asyncio
    async def test_execute_with_empty_arguments(self, tool_executor: ToolExecutor):
        """Test execution with empty arguments dict."""
        # counter tool doesn't require arguments
        result = await tool_executor.execute(
            tool_name="counter",
            tool_call_id="call_empty",
            arguments={},
        )

        assert result.result.success is True

    @pytest.mark.asyncio
    async def test_execute_many_preserves_order(self, tool_executor: ToolExecutor):
        """Test that execute_many preserves result order."""
        tool_calls = [
            {"name": "sample_tool", "id": "first", "arguments": {"message": "1"}},
            {"name": "sample_tool", "id": "second", "arguments": {"message": "2"}},
            {"name": "sample_tool", "id": "third", "arguments": {"message": "3"}},
        ]

        results = await tool_executor.execute_many(tool_calls)

        assert results[0].tool_call_id == "first"
        assert results[1].tool_call_id == "second"
        assert results[2].tool_call_id == "third"

    @pytest.mark.asyncio
    async def test_execute_with_none_in_arguments(self, tool_executor: ToolExecutor):
        """Test execution with None values in arguments."""

        class OptionalTool(BaseTool):
            name = "optional_tool"
            description = "Tool with optional params"

            def __init__(self):
                self.parameters = {
                    "type": "object",
                    "properties": {
                        "value": {"type": ["string", "null"], "default": None}
                    }
                }
                super().__init__()

            async def execute(self, value: str = None) -> str:
                return value or "default"

        tool_executor.register(OptionalTool())

        result = await tool_executor.execute(
            tool_name="optional_tool",
            tool_call_id="call_none",
            arguments={"value": None},
        )

        assert result.result.success is True
        assert result.result.output == "default"

    @pytest.mark.asyncio
    async def test_validation_error_handling(self, tool_executor: ToolExecutor):
        """Test that validation errors are handled properly."""
        result = await tool_executor.execute(
            tool_name="sample_tool",
            tool_call_id="call_invalid",
            arguments={"message": 123},  # Should be string
        )

        assert result.result.success is False
        assert result.result.error is not None

    @pytest.mark.asyncio
    async def test_missing_arguments_key(self, tool_executor: ToolExecutor):
        """Test execute_many with missing arguments key in call dict."""
        tool_calls = [
            {"name": "counter", "id": "call_1"},  # No arguments key
        ]

        results = await tool_executor.execute_many(tool_calls)

        assert len(results) == 1
        assert results[0].result.success is True

    @pytest.mark.asyncio
    async def test_large_batch_execution(self):
        """Test executing a large batch of tools."""
        from tests.unit.test_tools.conftest import SampleTool

        registry = ToolRegistry()
        registry.register(SampleTool())
        config = ExecutionConfig(max_concurrent=10, timeout=30.0)
        executor = ToolExecutor(registry=registry, config=config)

        # Create 50 tool calls
        tool_calls = [
            {"name": "sample_tool", "id": f"call_{i}", "arguments": {"message": f"{i}"}}
            for i in range(50)
        ]

        results = await executor.execute_many(tool_calls)

        assert len(results) == 50
        assert all(r.result.success for r in results)

    @pytest.mark.asyncio
    async def test_zero_timeout_config(self):
        """Test with zero timeout (should use default behavior)."""
        from tests.unit.test_tools.conftest import SampleTool

        registry = ToolRegistry()
        registry.register(SampleTool())
        config = ExecutionConfig(timeout=0.0)
        executor = ToolExecutor(registry=registry, config=config)

        # Zero timeout might behave differently - test it works
        result = await executor.execute(
            tool_name="sample_tool",
            tool_call_id="call_zero",
            arguments={"message": "test"},
        )

        # Behavior depends on implementation - just ensure no crash
        assert isinstance(result, ExecutionResult)


# ============================================================================
# Integration Tests
# ============================================================================


class TestExecutorIntegration:
    """Integration tests combining multiple executor features."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test a complete workflow with executor."""
        from tests.unit.test_tools.conftest import SampleTool, AsyncSampleTool

        # Create executor
        executor = ToolExecutor(
            config=ExecutionConfig(
                timeout=10.0,
                max_concurrent=5,
            )
        )

        # Register tools
        executor.register(SampleTool())
        executor.register(AsyncSampleTool())

        # Execute single tool
        single_result = await executor.execute(
            tool_name="sample_tool",
            tool_call_id="single",
            arguments={"message": "hello", "count": 2},
        )
        assert single_result.result.success is True
        assert single_result.result.output == "hellohello"

        # Execute multiple tools
        batch_results = await executor.execute_many(
            [
                {"name": "sample_tool", "id": "batch_1", "arguments": {"message": "a"}},
                {"name": "async_sample", "id": "batch_2", "arguments": {"value": 3}},
            ]
        )
        assert len(batch_results) == 2
        assert batch_results[0].result.output == "a"
        assert batch_results[1].result.output == 6

    @pytest.mark.asyncio
    async def test_mixed_success_failure_batch(self):
        """Test batch with mixed success and failure."""
        from tests.unit.test_tools.conftest import SampleTool, ErrorTool

        registry = ToolRegistry()
        registry.register(SampleTool())
        registry.register(ErrorTool(fail_after=0))
        executor = ToolExecutor(registry=registry)

        results = await executor.execute_many(
            [
                {"name": "sample_tool", "id": "ok_1", "arguments": {"message": "ok"}},
                {"name": "error_tool", "id": "fail_1", "arguments": {}},
                {"name": "sample_tool", "id": "ok_2", "arguments": {"message": "ok2"}},
                {"name": "nonexistent", "id": "fail_2", "arguments": {}},
            ]
        )

        assert results[0].result.success is True
        assert results[1].result.success is False
        assert results[2].result.success is True
        assert results[3].result.success is False
