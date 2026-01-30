"""Lifecycle hooks for agent execution.

This module provides a hook system for intercepting and reacting to
agent lifecycle events, including:
- pre_execute / post_execute: Before and after task execution
- pre_run / post_run: Before and after agent runs
- on_error: When errors occur
- on_step: For each step in the execution loop
- on_tool_call: Before and after tool execution
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class HookType(str, Enum):
    """Types of lifecycle hooks."""

    # Execution lifecycle
    PRE_EXECUTE = "pre_execute"
    POST_EXECUTE = "post_execute"

    # Run lifecycle
    PRE_RUN = "pre_run"
    POST_RUN = "post_run"

    # Step lifecycle
    ON_STEP = "on_step"
    PRE_STEP = "pre_step"
    POST_STEP = "post_step"

    # Tool lifecycle
    PRE_TOOL_CALL = "pre_tool_call"
    POST_TOOL_CALL = "post_tool_call"

    # Error handling
    ON_ERROR = "on_error"

    # Streaming
    ON_STREAM_CHUNK = "on_stream_chunk"

    # State changes
    ON_STATE_CHANGE = "on_state_change"


@dataclass
class HookContext:
    """Context passed to hooks during execution.

    Attributes:
        hook_type: The type of hook being fired.
        execution_id: Unique ID of the current execution.
        timestamp: When this hook was triggered.
        data: Additional data passed to the hook.
        metadata: Optional metadata.
    """

    hook_type: HookType
    execution_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from data."""
        return self.data.get(key, default)


# Type alias for hook functions
HookFunction = Callable[..., Union[None, Coroutine[Any, Any, None]]]


@dataclass
class HookEntry:
    """An entry in the hook registry.

    Attributes:
        hook_type: The type of hook.
        callback: The callback function.
        priority: Hook priority (lower runs first).
        name: Optional name for identification.
        enabled: Whether the hook is enabled.
    """

    hook_type: HookType
    callback: HookFunction
    priority: int = 0
    name: Optional[str] = None
    enabled: bool = True

    def __hash__(self) -> int:
        return hash((self.hook_type, self.name or id(self.callback)))


class Hook(ABC):
    """Abstract base class for hook implementations.

    Provides a structured way to implement hooks with
    consistent interface and optional filtering.

    Example:
        class MyLoggingHook(Hook):
            hook_types = {HookType.PRE_EXECUTE, HookType.POST_EXECUTE}

            async def execute(self, context: HookContext, **kwargs) -> None:
                print(f"Hook fired: {context.hook_type}")
    """

    hook_types: Set[HookType] = set()
    priority: int = 0
    name: str = ""

    def __init__(self, name: Optional[str] = None, priority: Optional[int] = None):
        """Initialize the hook.

        Args:
            name: Optional name override.
            priority: Optional priority override.
        """
        if name:
            self.name = name
        elif not self.name:
            self.name = self.__class__.__name__

        if priority is not None:
            self.priority = priority

    @abstractmethod
    async def execute(self, context: HookContext, **kwargs: Any) -> None:
        """Execute the hook.

        Args:
            context: The hook context.
            **kwargs: Additional arguments.
        """
        pass

    def should_run(self, hook_type: HookType) -> bool:
        """Check if this hook should run for a given type.

        Args:
            hook_type: The hook type being fired.

        Returns:
            True if the hook should run.
        """
        if not self.hook_types:
            return True  # Run for all types if none specified
        return hook_type in self.hook_types

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, priority={self.priority})"


class HookRegistry:
    """Registry for managing and firing lifecycle hooks.

    Provides registration, unregistration, and execution of hooks
    with support for priorities and async execution.

    Example:
        registry = HookRegistry()

        # Register a function
        @registry.register(HookType.PRE_EXECUTE)
        async def my_hook(context, **kwargs):
            print(f"Pre-execute: {kwargs}")

        # Register a Hook class
        registry.register_hook(MyLoggingHook())

        # Fire hooks
        await registry.fire(HookType.PRE_EXECUTE, task="my task")
    """

    def __init__(self):
        """Initialize an empty hook registry."""
        self._hooks: Dict[HookType, List[HookEntry]] = {ht: [] for ht in HookType}
        self._class_hooks: List[Hook] = []

    def register(
        self,
        hook_type: Union[HookType, str],
        callback: Optional[HookFunction] = None,
        priority: int = 0,
        name: Optional[str] = None,
    ) -> Union[HookEntry, Callable[[HookFunction], HookEntry]]:
        """Register a hook function.

        Can be used as a decorator or called directly.

        Args:
            hook_type: The type of hook to register.
            callback: The callback function (optional if used as decorator).
            priority: Hook priority (lower runs first).
            name: Optional name for the hook.

        Returns:
            HookEntry if callback provided, decorator otherwise.

        Example:
            # As decorator
            @registry.register(HookType.PRE_EXECUTE)
            async def my_hook(**kwargs):
                pass

            # Direct registration
            registry.register(HookType.PRE_EXECUTE, my_callback)
        """
        if isinstance(hook_type, str):
            hook_type = HookType(hook_type)

        def decorator(fn: HookFunction) -> HookEntry:
            entry = HookEntry(
                hook_type=hook_type,
                callback=fn,
                priority=priority,
                name=name or fn.__name__,
            )
            self._hooks[hook_type].append(entry)
            self._hooks[hook_type].sort(key=lambda x: x.priority)
            return entry

        if callback is not None:
            return decorator(callback)
        return decorator

    def register_hook(self, hook: Hook) -> None:
        """Register a Hook class instance.

        Args:
            hook: The Hook instance to register.
        """
        self._class_hooks.append(hook)
        logger.debug(f"Registered hook: {hook.name}")

    def unregister(
        self,
        hook_type: Optional[HookType] = None,
        name: Optional[str] = None,
        callback: Optional[HookFunction] = None,
    ) -> int:
        """Unregister hooks matching the criteria.

        Args:
            hook_type: Optional hook type filter.
            name: Optional name filter.
            callback: Optional callback filter.

        Returns:
            Number of hooks unregistered.
        """
        count = 0

        hook_types = [hook_type] if hook_type else list(HookType)

        for ht in hook_types:
            original_len = len(self._hooks[ht])
            self._hooks[ht] = [
                entry
                for entry in self._hooks[ht]
                if not (
                    (name is None or entry.name == name)
                    and (callback is None or entry.callback == callback)
                )
            ]
            count += original_len - len(self._hooks[ht])

        # Also unregister class hooks by name
        if name:
            original_len = len(self._class_hooks)
            self._class_hooks = [h for h in self._class_hooks if h.name != name]
            count += original_len - len(self._class_hooks)

        return count

    def unregister_hook(self, hook: Hook) -> bool:
        """Unregister a Hook class instance.

        Args:
            hook: The Hook instance to unregister.

        Returns:
            True if the hook was found and removed.
        """
        try:
            self._class_hooks.remove(hook)
            return True
        except ValueError:
            return False

    async def fire(
        self,
        hook_type: Union[HookType, str],
        **kwargs: Any,
    ) -> None:
        """Fire all hooks of a given type.

        Args:
            hook_type: The type of hook to fire.
            **kwargs: Arguments to pass to hook callbacks.
        """
        if isinstance(hook_type, str):
            hook_type = HookType(hook_type)

        context = HookContext(
            hook_type=hook_type,
            execution_id=kwargs.pop("execution_id", ""),
            data=dict(kwargs),
        )

        # Fire function hooks
        for entry in self._hooks[hook_type]:
            if not entry.enabled:
                continue

            try:
                result = entry.callback(context=context, **kwargs)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(
                    f"Error in hook {entry.name} for {hook_type}: {e}",
                    exc_info=True,
                )

        # Fire class hooks
        for hook in self._class_hooks:
            if not hook.should_run(hook_type):
                continue

            try:
                await hook.execute(context, **kwargs)
            except Exception as e:
                logger.error(
                    f"Error in hook {hook.name} for {hook_type}: {e}",
                    exc_info=True,
                )

    async def fire_parallel(
        self,
        hook_type: Union[HookType, str],
        **kwargs: Any,
    ) -> None:
        """Fire all hooks of a given type in parallel.

        Args:
            hook_type: The type of hook to fire.
            **kwargs: Arguments to pass to hook callbacks.
        """
        if isinstance(hook_type, str):
            hook_type = HookType(hook_type)

        context = HookContext(
            hook_type=hook_type,
            execution_id=kwargs.pop("execution_id", ""),
            data=dict(kwargs),
        )

        async def run_entry(entry: HookEntry) -> None:
            if not entry.enabled:
                return
            try:
                result = entry.callback(context=context, **kwargs)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in hook {entry.name}: {e}")

        async def run_hook(hook: Hook) -> None:
            if not hook.should_run(hook_type):
                return
            try:
                await hook.execute(context, **kwargs)
            except Exception as e:
                logger.error(f"Error in hook {hook.name}: {e}")

        tasks = [run_entry(e) for e in self._hooks[hook_type]]
        tasks.extend([run_hook(h) for h in self._class_hooks])

        await asyncio.gather(*tasks, return_exceptions=True)

    def enable(
        self,
        hook_type: Optional[HookType] = None,
        name: Optional[str] = None,
    ) -> int:
        """Enable hooks matching the criteria.

        Args:
            hook_type: Optional hook type filter.
            name: Optional name filter.

        Returns:
            Number of hooks enabled.
        """
        return self._set_enabled(True, hook_type, name)

    def disable(
        self,
        hook_type: Optional[HookType] = None,
        name: Optional[str] = None,
    ) -> int:
        """Disable hooks matching the criteria.

        Args:
            hook_type: Optional hook type filter.
            name: Optional name filter.

        Returns:
            Number of hooks disabled.
        """
        return self._set_enabled(False, hook_type, name)

    def _set_enabled(
        self,
        enabled: bool,
        hook_type: Optional[HookType],
        name: Optional[str],
    ) -> int:
        """Set enabled state for matching hooks."""
        count = 0
        hook_types = [hook_type] if hook_type else list(HookType)

        for ht in hook_types:
            for entry in self._hooks[ht]:
                if name is None or entry.name == name:
                    entry.enabled = enabled
                    count += 1

        return count

    def list_hooks(
        self,
        hook_type: Optional[HookType] = None,
    ) -> List[Union[HookEntry, Hook]]:
        """List registered hooks.

        Args:
            hook_type: Optional filter by hook type.

        Returns:
            List of registered hooks.
        """
        result: List[Union[HookEntry, Hook]] = []

        if hook_type:
            result.extend(self._hooks[hook_type])
        else:
            for entries in self._hooks.values():
                result.extend(entries)

        # Add class hooks
        if hook_type:
            result.extend([h for h in self._class_hooks if h.should_run(hook_type)])
        else:
            result.extend(self._class_hooks)

        return result

    def clear(self, hook_type: Optional[HookType] = None) -> None:
        """Clear all hooks.

        Args:
            hook_type: Optional filter to clear only specific type.
        """
        if hook_type:
            self._hooks[hook_type] = []
        else:
            for ht in HookType:
                self._hooks[ht] = []
            self._class_hooks = []

    def __len__(self) -> int:
        """Return total number of registered hooks."""
        count = sum(len(entries) for entries in self._hooks.values())
        count += len(self._class_hooks)
        return count


# Built-in Hooks


class LoggingHook(Hook):
    """Built-in hook for logging execution events.

    Logs key lifecycle events with configurable verbosity.

    Example:
        registry = HookRegistry()
        registry.register_hook(LoggingHook(log_level=logging.DEBUG))
    """

    hook_types = {
        HookType.PRE_EXECUTE,
        HookType.POST_EXECUTE,
        HookType.ON_ERROR,
        HookType.ON_STEP,
    }
    priority = -100  # Run early

    def __init__(
        self,
        name: str = "LoggingHook",
        log_level: int = logging.INFO,
        include_data: bool = False,
    ):
        """Initialize the logging hook.

        Args:
            name: Hook name.
            log_level: Logging level to use.
            include_data: Whether to include data in logs.
        """
        super().__init__(name)
        self.log_level = log_level
        self.include_data = include_data
        self._logger = logging.getLogger(f"{__name__}.{name}")

    async def execute(self, context: HookContext, **kwargs: Any) -> None:
        """Log the hook event."""
        message = f"[{context.hook_type.value}] execution_id={context.execution_id}"

        if self.include_data and context.data:
            message += f" data={context.data}"

        # Add specific info based on hook type
        if context.hook_type == HookType.ON_ERROR:
            error = kwargs.get("error")
            if error:
                message += f" error={error}"
                self._logger.log(logging.ERROR, message)
                return

        if context.hook_type == HookType.POST_EXECUTE:
            success = kwargs.get("success")
            message += f" success={success}"

        self._logger.log(self.log_level, message)


class ValidationHook(Hook):
    """Built-in hook for validating inputs and outputs.

    Validates task inputs before execution and results after.

    Example:
        registry = HookRegistry()
        registry.register_hook(ValidationHook(
            validators={"task": lambda x: len(x) > 0}
        ))
    """

    hook_types = {HookType.PRE_EXECUTE, HookType.POST_EXECUTE}
    priority = -50  # Run before most hooks

    def __init__(
        self,
        name: str = "ValidationHook",
        validators: Optional[Dict[str, Callable[[Any], bool]]] = None,
        raise_on_failure: bool = False,
    ):
        """Initialize the validation hook.

        Args:
            name: Hook name.
            validators: Dict mapping field names to validator functions.
            raise_on_failure: Whether to raise exceptions on validation failure.
        """
        super().__init__(name)
        self.validators = validators or {}
        self.raise_on_failure = raise_on_failure

    def add_validator(
        self,
        field: str,
        validator: Callable[[Any], bool],
    ) -> None:
        """Add a validator for a field.

        Args:
            field: Field name to validate.
            validator: Validation function.
        """
        self.validators[field] = validator

    async def execute(self, context: HookContext, **kwargs: Any) -> None:
        """Validate the context data."""
        failures = []

        for field, validator in self.validators.items():
            value = kwargs.get(field) or context.get(field)
            if value is not None:
                try:
                    if not validator(value):
                        failures.append(f"Validation failed for {field}")
                except Exception as e:
                    failures.append(f"Validation error for {field}: {e}")

        if failures:
            message = "; ".join(failures)
            logger.warning(f"Validation failures: {message}")

            if self.raise_on_failure:
                raise ValueError(message)


class MetricsHook(Hook):
    """Built-in hook for collecting execution metrics.

    Collects timing, success rates, and other metrics.

    Example:
        metrics_hook = MetricsHook()
        registry.register_hook(metrics_hook)

        # After some executions
        print(metrics_hook.get_metrics())
    """

    hook_types = {
        HookType.PRE_EXECUTE,
        HookType.POST_EXECUTE,
        HookType.ON_ERROR,
        HookType.ON_STEP,
    }
    priority = 100  # Run late to capture accurate timing

    def __init__(self, name: str = "MetricsHook"):
        """Initialize the metrics hook."""
        super().__init__(name)
        self._execution_starts: Dict[str, datetime] = {}
        self._metrics = {
            "total_executions": 0,
            "successful": 0,
            "failed": 0,
            "total_steps": 0,
            "total_duration_seconds": 0.0,
            "errors": [],
        }

    async def execute(self, context: HookContext, **kwargs: Any) -> None:
        """Collect metrics based on hook type."""
        execution_id = context.execution_id

        if context.hook_type == HookType.PRE_EXECUTE:
            self._execution_starts[execution_id] = context.timestamp
            self._metrics["total_executions"] += 1

        elif context.hook_type == HookType.POST_EXECUTE:
            if execution_id in self._execution_starts:
                start = self._execution_starts.pop(execution_id)
                duration = (context.timestamp - start).total_seconds()
                self._metrics["total_duration_seconds"] += duration

            if kwargs.get("success"):
                self._metrics["successful"] += 1
            else:
                self._metrics["failed"] += 1

        elif context.hook_type == HookType.ON_ERROR:
            error = kwargs.get("error")
            if error:
                self._metrics["errors"].append(str(error))

        elif context.hook_type == HookType.ON_STEP:
            self._metrics["total_steps"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics.

        Returns:
            Dictionary with metrics.
        """
        total = self._metrics["total_executions"]
        return {
            **self._metrics,
            "success_rate": (
                self._metrics["successful"] / total if total > 0 else 0.0
            ),
            "avg_duration_seconds": (
                self._metrics["total_duration_seconds"] / total if total > 0 else 0.0
            ),
            "avg_steps_per_execution": (
                self._metrics["total_steps"] / total if total > 0 else 0.0
            ),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._execution_starts.clear()
        self._metrics = {
            "total_executions": 0,
            "successful": 0,
            "failed": 0,
            "total_steps": 0,
            "total_duration_seconds": 0.0,
            "errors": [],
        }
