"""Tests for the execution hooks system."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents_framework.execution.hooks import (
    Hook,
    HookContext,
    HookEntry,
    HookRegistry,
    HookType,
    LoggingHook,
    MetricsHook,
    ValidationHook,
)

from .conftest import FailingHook, MockHook


# ============================================================================
# HookType Tests
# ============================================================================


class TestHookType:
    """Tests for HookType enum."""

    def test_hook_type_values(self):
        """Test that HookType has the expected values."""
        assert HookType.PRE_EXECUTE == "pre_execute"
        assert HookType.POST_EXECUTE == "post_execute"
        assert HookType.PRE_RUN == "pre_run"
        assert HookType.POST_RUN == "post_run"
        assert HookType.ON_STEP == "on_step"
        assert HookType.PRE_STEP == "pre_step"
        assert HookType.POST_STEP == "post_step"
        assert HookType.PRE_TOOL_CALL == "pre_tool_call"
        assert HookType.POST_TOOL_CALL == "post_tool_call"
        assert HookType.ON_ERROR == "on_error"
        assert HookType.ON_STREAM_CHUNK == "on_stream_chunk"
        assert HookType.ON_STATE_CHANGE == "on_state_change"

    def test_hook_type_is_string_enum(self):
        """Test that HookType is a string enum."""
        assert isinstance(HookType.PRE_EXECUTE, str)
        assert HookType.POST_EXECUTE.value == "post_execute"


# ============================================================================
# HookContext Tests
# ============================================================================


class TestHookContext:
    """Tests for HookContext dataclass."""

    def test_hook_context_creation(self):
        """Test creating a HookContext."""
        context = HookContext(
            hook_type=HookType.PRE_EXECUTE,
            execution_id="exec-123",
        )
        assert context.hook_type == HookType.PRE_EXECUTE
        assert context.execution_id == "exec-123"
        assert isinstance(context.timestamp, datetime)
        assert context.data == {}
        assert context.metadata == {}

    def test_hook_context_with_data(self):
        """Test HookContext with data."""
        context = HookContext(
            hook_type=HookType.ON_STEP,
            execution_id="exec-123",
            data={"step": 1, "content": "test"},
            metadata={"source": "test"},
        )
        assert context.data == {"step": 1, "content": "test"}
        assert context.metadata == {"source": "test"}

    def test_hook_context_get(self):
        """Test HookContext.get method."""
        context = HookContext(
            hook_type=HookType.PRE_EXECUTE,
            execution_id="exec-123",
            data={"key": "value"},
        )
        assert context.get("key") == "value"
        assert context.get("missing") is None
        assert context.get("missing", "default") == "default"


# ============================================================================
# HookEntry Tests
# ============================================================================


class TestHookEntry:
    """Tests for HookEntry dataclass."""

    def test_hook_entry_creation(self):
        """Test creating a HookEntry."""

        async def my_callback(**kwargs):
            pass

        entry = HookEntry(
            hook_type=HookType.PRE_EXECUTE,
            callback=my_callback,
        )
        assert entry.hook_type == HookType.PRE_EXECUTE
        assert entry.callback == my_callback
        assert entry.priority == 0
        assert entry.name is None
        assert entry.enabled is True

    def test_hook_entry_with_options(self):
        """Test HookEntry with all options."""

        async def my_callback(**kwargs):
            pass

        entry = HookEntry(
            hook_type=HookType.POST_EXECUTE,
            callback=my_callback,
            priority=10,
            name="my_hook",
            enabled=False,
        )
        assert entry.priority == 10
        assert entry.name == "my_hook"
        assert entry.enabled is False

    def test_hook_entry_hash(self):
        """Test HookEntry hashing."""

        async def callback1(**kwargs):
            pass

        async def callback2(**kwargs):
            pass

        entry1 = HookEntry(
            hook_type=HookType.PRE_EXECUTE,
            callback=callback1,
            name="hook1",
        )
        entry2 = HookEntry(
            hook_type=HookType.PRE_EXECUTE,
            callback=callback2,
            name="hook1",
        )
        entry3 = HookEntry(
            hook_type=HookType.PRE_EXECUTE,
            callback=callback1,
            name="hook2",
        )

        # Same name + type should have same hash
        assert hash(entry1) == hash(entry2)
        # Different name should have different hash
        assert hash(entry1) != hash(entry3)


# ============================================================================
# Hook Base Class Tests
# ============================================================================


class TestHookBaseClass:
    """Tests for Hook abstract base class."""

    def test_hook_subclass_creation(self):
        """Test creating a Hook subclass."""
        hook = MockHook()
        assert hook.name == "MockHook"
        assert hook.priority == 0
        assert hook.hook_types == {HookType.PRE_EXECUTE, HookType.POST_EXECUTE}

    def test_hook_custom_name_and_priority(self):
        """Test Hook with custom name and priority."""
        hook = MockHook(name="CustomName")
        hook.priority = 10
        assert hook.name == "CustomName"
        assert hook.priority == 10

    def test_hook_should_run_matching_type(self):
        """Test should_run returns True for matching types."""
        hook = MockHook()
        assert hook.should_run(HookType.PRE_EXECUTE) is True
        assert hook.should_run(HookType.POST_EXECUTE) is True

    def test_hook_should_run_non_matching_type(self):
        """Test should_run returns False for non-matching types."""
        hook = MockHook()
        assert hook.should_run(HookType.ON_ERROR) is False
        assert hook.should_run(HookType.ON_STEP) is False

    def test_hook_should_run_empty_types(self):
        """Test should_run with empty hook_types (runs for all)."""

        class AllTypesHook(Hook):
            hook_types = set()

            async def execute(self, context, **kwargs):
                pass

        hook = AllTypesHook()
        assert hook.should_run(HookType.PRE_EXECUTE) is True
        assert hook.should_run(HookType.ON_ERROR) is True

    def test_hook_repr(self):
        """Test Hook repr."""
        hook = MockHook(name="TestHook")
        repr_str = repr(hook)
        assert "MockHook" in repr_str
        assert "TestHook" in repr_str

    @pytest.mark.asyncio
    async def test_hook_execute(self):
        """Test Hook execute method."""
        hook = MockHook()
        context = HookContext(
            hook_type=HookType.PRE_EXECUTE,
            execution_id="exec-123",
        )

        await hook.execute(context, extra="data")

        assert hook.execute_count == 1
        assert hook.last_context == context
        assert hook.last_kwargs == {"extra": "data"}


# ============================================================================
# HookRegistry Tests
# ============================================================================


class TestHookRegistryInit:
    """Tests for HookRegistry initialization."""

    def test_registry_creation(self):
        """Test creating a HookRegistry."""
        registry = HookRegistry()
        assert len(registry) == 0
        for hook_type in HookType:
            assert registry._hooks[hook_type] == []
        assert registry._class_hooks == []


class TestHookRegistryRegister:
    """Tests for HookRegistry.register method."""

    def test_register_with_callback(self, hook_registry: HookRegistry):
        """Test registering a callback function."""

        async def my_hook(**kwargs):
            pass

        entry = hook_registry.register(HookType.PRE_EXECUTE, my_hook)

        assert isinstance(entry, HookEntry)
        assert entry.callback == my_hook
        assert entry.name == "my_hook"
        assert len(hook_registry) == 1

    def test_register_as_decorator(self, hook_registry: HookRegistry):
        """Test registering using decorator syntax."""

        @hook_registry.register(HookType.POST_EXECUTE)
        async def my_post_hook(**kwargs):
            pass

        hooks = hook_registry.list_hooks(HookType.POST_EXECUTE)
        assert len(hooks) == 1
        assert hooks[0].name == "my_post_hook"

    def test_register_with_priority(self, hook_registry: HookRegistry):
        """Test registering with priority."""

        async def hook1(**kwargs):
            pass

        async def hook2(**kwargs):
            pass

        hook_registry.register(HookType.PRE_EXECUTE, hook1, priority=10)
        hook_registry.register(HookType.PRE_EXECUTE, hook2, priority=5)

        hooks = hook_registry.list_hooks(HookType.PRE_EXECUTE)
        assert len(hooks) == 2
        # Lower priority should come first
        assert hooks[0].priority == 5
        assert hooks[1].priority == 10

    def test_register_with_name(self, hook_registry: HookRegistry):
        """Test registering with custom name."""

        async def my_hook(**kwargs):
            pass

        entry = hook_registry.register(
            HookType.PRE_EXECUTE,
            my_hook,
            name="custom_name",
        )

        assert entry.name == "custom_name"

    def test_register_string_hook_type(self, hook_registry: HookRegistry):
        """Test registering with string hook type."""

        async def my_hook(**kwargs):
            pass

        entry = hook_registry.register("pre_execute", my_hook)

        assert entry.hook_type == HookType.PRE_EXECUTE


class TestHookRegistryRegisterHook:
    """Tests for HookRegistry.register_hook method."""

    def test_register_hook_class(self, hook_registry: HookRegistry, mock_hook: MockHook):
        """Test registering a Hook class instance."""
        hook_registry.register_hook(mock_hook)

        assert len(hook_registry._class_hooks) == 1
        assert mock_hook in hook_registry._class_hooks

    def test_register_multiple_hooks(self, hook_registry: HookRegistry):
        """Test registering multiple Hook instances."""
        hook1 = MockHook(name="Hook1")
        hook2 = MockHook(name="Hook2")

        hook_registry.register_hook(hook1)
        hook_registry.register_hook(hook2)

        assert len(hook_registry._class_hooks) == 2


class TestHookRegistryUnregister:
    """Tests for HookRegistry.unregister method."""

    def test_unregister_by_name(self, hook_registry: HookRegistry):
        """Test unregistering by name."""

        async def my_hook(**kwargs):
            pass

        hook_registry.register(HookType.PRE_EXECUTE, my_hook, name="my_hook")

        count = hook_registry.unregister(name="my_hook")

        assert count == 1
        assert len(hook_registry.list_hooks(HookType.PRE_EXECUTE)) == 0

    def test_unregister_by_hook_type(self, hook_registry: HookRegistry):
        """Test unregistering by hook type."""

        async def hook1(**kwargs):
            pass

        async def hook2(**kwargs):
            pass

        hook_registry.register(HookType.PRE_EXECUTE, hook1, name="hook1")
        hook_registry.register(HookType.PRE_EXECUTE, hook2, name="hook2")
        hook_registry.register(HookType.POST_EXECUTE, hook1, name="hook1")

        count = hook_registry.unregister(hook_type=HookType.PRE_EXECUTE, name="hook1")

        assert count == 1
        assert len(hook_registry.list_hooks(HookType.PRE_EXECUTE)) == 1

    def test_unregister_by_callback(self, hook_registry: HookRegistry):
        """Test unregistering by callback."""

        async def my_hook(**kwargs):
            pass

        hook_registry.register(HookType.PRE_EXECUTE, my_hook)

        count = hook_registry.unregister(callback=my_hook)

        assert count == 1

    def test_unregister_class_hook_by_name(
        self,
        hook_registry: HookRegistry,
        mock_hook: MockHook,
    ):
        """Test unregistering a Hook class by name."""
        hook_registry.register_hook(mock_hook)

        count = hook_registry.unregister(name="MockHook")

        assert count == 1
        assert len(hook_registry._class_hooks) == 0


class TestHookRegistryUnregisterHook:
    """Tests for HookRegistry.unregister_hook method."""

    def test_unregister_hook_success(
        self,
        hook_registry: HookRegistry,
        mock_hook: MockHook,
    ):
        """Test unregistering a Hook instance."""
        hook_registry.register_hook(mock_hook)

        result = hook_registry.unregister_hook(mock_hook)

        assert result is True
        assert mock_hook not in hook_registry._class_hooks

    def test_unregister_hook_not_found(self, hook_registry: HookRegistry):
        """Test unregistering a Hook that was not registered."""
        hook = MockHook()

        result = hook_registry.unregister_hook(hook)

        assert result is False


class TestHookRegistryFire:
    """Tests for HookRegistry.fire method."""

    @pytest.mark.asyncio
    async def test_fire_function_hooks(self, hook_registry: HookRegistry):
        """Test firing function hooks."""
        calls = []

        async def hook1(context, **kwargs):
            calls.append(("hook1", kwargs))

        async def hook2(context, **kwargs):
            calls.append(("hook2", kwargs))

        hook_registry.register(HookType.PRE_EXECUTE, hook1)
        hook_registry.register(HookType.PRE_EXECUTE, hook2)

        await hook_registry.fire(HookType.PRE_EXECUTE, task="test")

        assert len(calls) == 2
        assert calls[0][0] == "hook1"
        assert calls[1][0] == "hook2"
        assert calls[0][1].get("task") == "test"

    @pytest.mark.asyncio
    async def test_fire_class_hooks(
        self,
        hook_registry: HookRegistry,
        mock_hook: MockHook,
    ):
        """Test firing class hooks."""
        hook_registry.register_hook(mock_hook)

        await hook_registry.fire(HookType.PRE_EXECUTE, task="test")

        assert mock_hook.execute_count == 1
        assert mock_hook.last_context.hook_type == HookType.PRE_EXECUTE
        assert mock_hook.last_kwargs.get("task") == "test"

    @pytest.mark.asyncio
    async def test_fire_with_string_hook_type(self, hook_registry: HookRegistry):
        """Test firing with string hook type."""
        calls = []

        async def my_hook(context, **kwargs):
            calls.append(context.hook_type)

        hook_registry.register(HookType.POST_EXECUTE, my_hook)

        await hook_registry.fire("post_execute")

        assert len(calls) == 1
        assert calls[0] == HookType.POST_EXECUTE

    @pytest.mark.asyncio
    async def test_fire_skips_disabled_hooks(self, hook_registry: HookRegistry):
        """Test that disabled hooks are skipped."""
        calls = []

        async def my_hook(context, **kwargs):
            calls.append("called")

        entry = hook_registry.register(HookType.PRE_EXECUTE, my_hook)
        entry.enabled = False

        await hook_registry.fire(HookType.PRE_EXECUTE)

        assert len(calls) == 0

    @pytest.mark.asyncio
    async def test_fire_skips_non_matching_class_hooks(
        self,
        hook_registry: HookRegistry,
        mock_hook: MockHook,
    ):
        """Test that class hooks are skipped for non-matching types."""
        hook_registry.register_hook(mock_hook)

        await hook_registry.fire(HookType.ON_ERROR)

        assert mock_hook.execute_count == 0

    @pytest.mark.asyncio
    async def test_fire_handles_hook_errors(
        self,
        hook_registry: HookRegistry,
        failing_hook: FailingHook,
    ):
        """Test that hook errors are logged but don't stop execution."""
        calls = []

        async def after_failing(context, **kwargs):
            calls.append("after")

        hook_registry.register_hook(failing_hook)
        hook_registry.register(HookType.PRE_EXECUTE, after_failing)

        # Should not raise
        await hook_registry.fire(HookType.PRE_EXECUTE)

        # Other hooks should still run
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_fire_handles_sync_callbacks(self, hook_registry: HookRegistry):
        """Test that sync callbacks are handled correctly."""
        calls = []

        def sync_hook(context, **kwargs):
            calls.append("sync")

        hook_registry.register(HookType.PRE_EXECUTE, sync_hook)

        await hook_registry.fire(HookType.PRE_EXECUTE)

        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_fire_with_execution_id(self, hook_registry: HookRegistry):
        """Test that execution_id is passed in context."""
        captured_context = None

        async def capture_context(context, **kwargs):
            nonlocal captured_context
            captured_context = context

        hook_registry.register(HookType.PRE_EXECUTE, capture_context)

        await hook_registry.fire(
            HookType.PRE_EXECUTE,
            execution_id="exec-123",
            task="test",
        )

        assert captured_context is not None
        assert captured_context.execution_id == "exec-123"


class TestHookRegistryFireParallel:
    """Tests for HookRegistry.fire_parallel method."""

    @pytest.mark.asyncio
    async def test_fire_parallel_executes_all(self, hook_registry: HookRegistry):
        """Test that fire_parallel executes all hooks."""
        calls = []

        async def hook1(context, **kwargs):
            await asyncio.sleep(0.01)
            calls.append("hook1")

        async def hook2(context, **kwargs):
            await asyncio.sleep(0.01)
            calls.append("hook2")

        hook_registry.register(HookType.PRE_EXECUTE, hook1)
        hook_registry.register(HookType.PRE_EXECUTE, hook2)

        await hook_registry.fire_parallel(HookType.PRE_EXECUTE)

        assert set(calls) == {"hook1", "hook2"}

    @pytest.mark.asyncio
    async def test_fire_parallel_handles_errors(
        self,
        hook_registry: HookRegistry,
        failing_hook: FailingHook,
    ):
        """Test that errors in parallel execution are handled."""
        calls = []

        async def successful_hook(context, **kwargs):
            calls.append("success")

        hook_registry.register_hook(failing_hook)
        hook_registry.register(HookType.PRE_EXECUTE, successful_hook)

        # Should not raise
        await hook_registry.fire_parallel(HookType.PRE_EXECUTE)

        # Other hooks should still complete
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_fire_parallel_faster_than_sequential(
        self,
        hook_registry: HookRegistry,
    ):
        """Test that parallel execution is faster than sequential would be."""
        import time

        async def slow_hook(context, **kwargs):
            await asyncio.sleep(0.05)

        hook_registry.register(HookType.PRE_EXECUTE, slow_hook, name="hook1")
        hook_registry.register(HookType.PRE_EXECUTE, slow_hook, name="hook2")
        hook_registry.register(HookType.PRE_EXECUTE, slow_hook, name="hook3")

        start = time.monotonic()
        await hook_registry.fire_parallel(HookType.PRE_EXECUTE)
        duration = time.monotonic() - start

        # 3 hooks * 0.05s = 0.15s sequential, but parallel should be ~0.05s
        assert duration < 0.15


class TestHookRegistryEnableDisable:
    """Tests for HookRegistry.enable and disable methods."""

    def test_disable_by_name(self, hook_registry: HookRegistry):
        """Test disabling hooks by name."""

        async def my_hook(**kwargs):
            pass

        hook_registry.register(HookType.PRE_EXECUTE, my_hook, name="my_hook")

        count = hook_registry.disable(name="my_hook")

        assert count == 1
        hooks = hook_registry.list_hooks(HookType.PRE_EXECUTE)
        assert hooks[0].enabled is False

    def test_disable_by_hook_type(self, hook_registry: HookRegistry):
        """Test disabling hooks by type."""

        async def hook1(**kwargs):
            pass

        async def hook2(**kwargs):
            pass

        hook_registry.register(HookType.PRE_EXECUTE, hook1)
        hook_registry.register(HookType.PRE_EXECUTE, hook2)

        count = hook_registry.disable(hook_type=HookType.PRE_EXECUTE)

        assert count == 2

    def test_enable_by_name(self, hook_registry: HookRegistry):
        """Test enabling hooks by name."""

        async def my_hook(**kwargs):
            pass

        entry = hook_registry.register(HookType.PRE_EXECUTE, my_hook, name="my_hook")
        entry.enabled = False

        count = hook_registry.enable(name="my_hook")

        assert count == 1
        assert entry.enabled is True

    def test_enable_all(self, hook_registry: HookRegistry):
        """Test enabling all hooks."""

        async def hook1(**kwargs):
            pass

        async def hook2(**kwargs):
            pass

        e1 = hook_registry.register(HookType.PRE_EXECUTE, hook1)
        e2 = hook_registry.register(HookType.POST_EXECUTE, hook2)
        e1.enabled = False
        e2.enabled = False

        count = hook_registry.enable()

        assert count == 2
        assert e1.enabled is True
        assert e2.enabled is True


class TestHookRegistryListHooks:
    """Tests for HookRegistry.list_hooks method."""

    def test_list_hooks_by_type(self, hook_registry: HookRegistry):
        """Test listing hooks by type."""

        async def hook1(**kwargs):
            pass

        async def hook2(**kwargs):
            pass

        hook_registry.register(HookType.PRE_EXECUTE, hook1)
        hook_registry.register(HookType.POST_EXECUTE, hook2)

        pre_hooks = hook_registry.list_hooks(HookType.PRE_EXECUTE)
        post_hooks = hook_registry.list_hooks(HookType.POST_EXECUTE)

        assert len(pre_hooks) == 1
        assert len(post_hooks) == 1

    def test_list_all_hooks(self, hook_registry: HookRegistry, mock_hook: MockHook):
        """Test listing all hooks."""

        async def func_hook(**kwargs):
            pass

        hook_registry.register(HookType.PRE_EXECUTE, func_hook)
        hook_registry.register_hook(mock_hook)

        all_hooks = hook_registry.list_hooks()

        assert len(all_hooks) == 2

    def test_list_hooks_includes_class_hooks(
        self,
        hook_registry: HookRegistry,
        mock_hook: MockHook,
    ):
        """Test that listing includes matching class hooks."""
        hook_registry.register_hook(mock_hook)

        hooks = hook_registry.list_hooks(HookType.PRE_EXECUTE)

        assert len(hooks) == 1
        assert hooks[0] == mock_hook


class TestHookRegistryClear:
    """Tests for HookRegistry.clear method."""

    def test_clear_all(self, hook_registry: HookRegistry, mock_hook: MockHook):
        """Test clearing all hooks."""

        async def func_hook(**kwargs):
            pass

        hook_registry.register(HookType.PRE_EXECUTE, func_hook)
        hook_registry.register_hook(mock_hook)

        hook_registry.clear()

        assert len(hook_registry) == 0
        assert len(hook_registry._class_hooks) == 0

    def test_clear_by_type(self, hook_registry: HookRegistry):
        """Test clearing hooks by type."""

        async def hook1(**kwargs):
            pass

        async def hook2(**kwargs):
            pass

        hook_registry.register(HookType.PRE_EXECUTE, hook1)
        hook_registry.register(HookType.POST_EXECUTE, hook2)

        hook_registry.clear(HookType.PRE_EXECUTE)

        assert len(hook_registry.list_hooks(HookType.PRE_EXECUTE)) == 0
        assert len(hook_registry.list_hooks(HookType.POST_EXECUTE)) == 1


class TestHookRegistryLen:
    """Tests for HookRegistry.__len__ method."""

    def test_len_empty(self, hook_registry: HookRegistry):
        """Test length of empty registry."""
        assert len(hook_registry) == 0

    def test_len_with_function_hooks(self, hook_registry: HookRegistry):
        """Test length with function hooks."""

        async def hook1(**kwargs):
            pass

        async def hook2(**kwargs):
            pass

        hook_registry.register(HookType.PRE_EXECUTE, hook1)
        hook_registry.register(HookType.POST_EXECUTE, hook2)

        assert len(hook_registry) == 2

    def test_len_with_class_hooks(
        self,
        hook_registry: HookRegistry,
        mock_hook: MockHook,
    ):
        """Test length includes class hooks."""
        hook_registry.register_hook(mock_hook)

        assert len(hook_registry) == 1


# ============================================================================
# LoggingHook Tests
# ============================================================================


class TestLoggingHook:
    """Tests for LoggingHook built-in hook."""

    def test_logging_hook_creation(self):
        """Test creating a LoggingHook."""
        hook = LoggingHook()
        assert hook.name == "LoggingHook"
        assert hook.log_level == logging.INFO
        assert hook.include_data is False
        assert hook.priority == -100

    def test_logging_hook_custom_options(self):
        """Test LoggingHook with custom options."""
        hook = LoggingHook(
            name="CustomLogger",
            log_level=logging.DEBUG,
            include_data=True,
        )
        assert hook.name == "CustomLogger"
        assert hook.log_level == logging.DEBUG
        assert hook.include_data is True

    def test_logging_hook_types(self):
        """Test that LoggingHook has correct hook types."""
        hook = LoggingHook()
        assert HookType.PRE_EXECUTE in hook.hook_types
        assert HookType.POST_EXECUTE in hook.hook_types
        assert HookType.ON_ERROR in hook.hook_types
        assert HookType.ON_STEP in hook.hook_types

    @pytest.mark.asyncio
    async def test_logging_hook_execute(self):
        """Test LoggingHook execution."""
        hook = LoggingHook()
        context = HookContext(
            hook_type=HookType.PRE_EXECUTE,
            execution_id="exec-123",
        )

        # Should not raise
        await hook.execute(context, task="test")

    @pytest.mark.asyncio
    async def test_logging_hook_error_handling(self):
        """Test LoggingHook with error context."""
        hook = LoggingHook()
        context = HookContext(
            hook_type=HookType.ON_ERROR,
            execution_id="exec-123",
        )

        # Should not raise
        await hook.execute(context, error=RuntimeError("test error"))


# ============================================================================
# ValidationHook Tests
# ============================================================================


class TestValidationHook:
    """Tests for ValidationHook built-in hook."""

    def test_validation_hook_creation(self):
        """Test creating a ValidationHook."""
        hook = ValidationHook()
        assert hook.name == "ValidationHook"
        assert hook.validators == {}
        assert hook.raise_on_failure is False
        assert hook.priority == -50

    def test_validation_hook_with_validators(self):
        """Test ValidationHook with validators."""
        validators = {
            "task": lambda x: len(x) > 0,
            "value": lambda x: x > 0,
        }
        hook = ValidationHook(validators=validators)
        assert len(hook.validators) == 2

    def test_validation_hook_add_validator(self):
        """Test adding validators."""
        hook = ValidationHook()
        hook.add_validator("task", lambda x: len(x) > 0)

        assert "task" in hook.validators

    @pytest.mark.asyncio
    async def test_validation_hook_passes(self):
        """Test validation passes for valid input."""
        hook = ValidationHook(
            validators={"task": lambda x: len(x) > 0},
        )
        context = HookContext(
            hook_type=HookType.PRE_EXECUTE,
            execution_id="exec-123",
        )

        # Should not raise
        await hook.execute(context, task="valid task")

    @pytest.mark.asyncio
    async def test_validation_hook_fails_silently(self):
        """Test validation fails silently by default."""
        hook = ValidationHook(
            validators={"task": lambda x: len(x) > 10},
        )
        context = HookContext(
            hook_type=HookType.PRE_EXECUTE,
            execution_id="exec-123",
        )

        # Should not raise (just logs warning)
        await hook.execute(context, task="short")

    @pytest.mark.asyncio
    async def test_validation_hook_raises_on_failure(self):
        """Test validation raises when configured."""
        hook = ValidationHook(
            validators={"task": lambda x: len(x) > 10},
            raise_on_failure=True,
        )
        context = HookContext(
            hook_type=HookType.PRE_EXECUTE,
            execution_id="exec-123",
        )

        with pytest.raises(ValueError, match="Validation failed"):
            await hook.execute(context, task="short")

    @pytest.mark.asyncio
    async def test_validation_hook_handles_validator_errors(self):
        """Test validation handles validator exceptions."""

        def bad_validator(x):
            raise RuntimeError("Validator error")

        hook = ValidationHook(
            validators={"task": bad_validator},
        )
        context = HookContext(
            hook_type=HookType.PRE_EXECUTE,
            execution_id="exec-123",
        )

        # Should not raise (logs warning)
        await hook.execute(context, task="test")


# ============================================================================
# MetricsHook Tests
# ============================================================================


class TestMetricsHook:
    """Tests for MetricsHook built-in hook."""

    def test_metrics_hook_creation(self):
        """Test creating a MetricsHook."""
        hook = MetricsHook()
        assert hook.name == "MetricsHook"
        assert hook.priority == 100

    def test_metrics_hook_types(self):
        """Test that MetricsHook has correct hook types."""
        hook = MetricsHook()
        assert HookType.PRE_EXECUTE in hook.hook_types
        assert HookType.POST_EXECUTE in hook.hook_types
        assert HookType.ON_ERROR in hook.hook_types
        assert HookType.ON_STEP in hook.hook_types

    def test_metrics_hook_initial_metrics(self):
        """Test initial metrics are zeroed."""
        hook = MetricsHook()
        metrics = hook.get_metrics()

        assert metrics["total_executions"] == 0
        assert metrics["successful"] == 0
        assert metrics["failed"] == 0
        assert metrics["total_steps"] == 0
        assert metrics["success_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_metrics_hook_pre_execute(self):
        """Test metrics on pre_execute."""
        hook = MetricsHook()
        context = HookContext(
            hook_type=HookType.PRE_EXECUTE,
            execution_id="exec-123",
        )

        await hook.execute(context)

        metrics = hook.get_metrics()
        assert metrics["total_executions"] == 1

    @pytest.mark.asyncio
    async def test_metrics_hook_post_execute_success(self):
        """Test metrics on successful post_execute."""
        hook = MetricsHook()

        # Pre-execute
        pre_context = HookContext(
            hook_type=HookType.PRE_EXECUTE,
            execution_id="exec-123",
        )
        await hook.execute(pre_context)

        # Post-execute
        post_context = HookContext(
            hook_type=HookType.POST_EXECUTE,
            execution_id="exec-123",
        )
        await hook.execute(post_context, success=True)

        metrics = hook.get_metrics()
        assert metrics["successful"] == 1
        assert metrics["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_metrics_hook_post_execute_failure(self):
        """Test metrics on failed post_execute."""
        hook = MetricsHook()

        pre_context = HookContext(
            hook_type=HookType.PRE_EXECUTE,
            execution_id="exec-123",
        )
        await hook.execute(pre_context)

        post_context = HookContext(
            hook_type=HookType.POST_EXECUTE,
            execution_id="exec-123",
        )
        await hook.execute(post_context, success=False)

        metrics = hook.get_metrics()
        assert metrics["failed"] == 1
        assert metrics["success_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_metrics_hook_on_step(self):
        """Test metrics on step execution."""
        hook = MetricsHook()
        context = HookContext(
            hook_type=HookType.ON_STEP,
            execution_id="exec-123",
        )

        await hook.execute(context)
        await hook.execute(context)
        await hook.execute(context)

        metrics = hook.get_metrics()
        assert metrics["total_steps"] == 3

    @pytest.mark.asyncio
    async def test_metrics_hook_on_error(self):
        """Test metrics on error."""
        hook = MetricsHook()
        context = HookContext(
            hook_type=HookType.ON_ERROR,
            execution_id="exec-123",
        )

        await hook.execute(context, error=RuntimeError("test error"))

        metrics = hook.get_metrics()
        assert len(metrics["errors"]) == 1
        assert "test error" in metrics["errors"][0]

    @pytest.mark.asyncio
    async def test_metrics_hook_duration_tracking(self):
        """Test that duration is tracked."""
        hook = MetricsHook()

        pre_context = HookContext(
            hook_type=HookType.PRE_EXECUTE,
            execution_id="exec-123",
        )
        await hook.execute(pre_context)

        # Small delay
        await asyncio.sleep(0.01)

        post_context = HookContext(
            hook_type=HookType.POST_EXECUTE,
            execution_id="exec-123",
        )
        await hook.execute(post_context, success=True)

        metrics = hook.get_metrics()
        assert metrics["total_duration_seconds"] > 0
        assert metrics["avg_duration_seconds"] > 0

    def test_metrics_hook_reset(self):
        """Test resetting metrics."""
        hook = MetricsHook()
        hook._metrics["total_executions"] = 10
        hook._metrics["successful"] = 8

        hook.reset()

        metrics = hook.get_metrics()
        assert metrics["total_executions"] == 0
        assert metrics["successful"] == 0

    @pytest.mark.asyncio
    async def test_metrics_hook_avg_steps_per_execution(self):
        """Test average steps per execution calculation."""
        hook = MetricsHook()

        # Simulate two executions with different step counts
        # Execution 1: 3 steps
        pre1 = HookContext(hook_type=HookType.PRE_EXECUTE, execution_id="exec-1")
        await hook.execute(pre1)
        for _ in range(3):
            step = HookContext(hook_type=HookType.ON_STEP, execution_id="exec-1")
            await hook.execute(step)
        post1 = HookContext(hook_type=HookType.POST_EXECUTE, execution_id="exec-1")
        await hook.execute(post1, success=True)

        # Execution 2: 5 steps
        pre2 = HookContext(hook_type=HookType.PRE_EXECUTE, execution_id="exec-2")
        await hook.execute(pre2)
        for _ in range(5):
            step = HookContext(hook_type=HookType.ON_STEP, execution_id="exec-2")
            await hook.execute(step)
        post2 = HookContext(hook_type=HookType.POST_EXECUTE, execution_id="exec-2")
        await hook.execute(post2, success=True)

        metrics = hook.get_metrics()
        assert metrics["total_executions"] == 2
        assert metrics["total_steps"] == 8
        assert metrics["avg_steps_per_execution"] == 4.0
