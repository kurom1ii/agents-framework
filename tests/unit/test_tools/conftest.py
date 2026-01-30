"""Local fixtures for tool tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest

from agents_framework.tools.base import BaseTool, FunctionTool, ToolResult
from agents_framework.tools.registry import ToolRegistry
from agents_framework.tools.executor import ExecutionConfig, ToolExecutor


# ============================================================================
# Sample Types for Schema Testing
# ============================================================================


class Priority(Enum):
    """Priority level enumeration for testing."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Address:
    """Sample dataclass for nested schema testing."""

    street: str
    city: str
    zip_code: Optional[str] = None


@dataclass
class Person:
    """Sample dataclass for schema testing."""

    name: str
    age: int
    email: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Sample Tools for Testing
# ============================================================================


class SampleTool(BaseTool):
    """A sample tool implementation for testing."""

    name = "sample_tool"
    description = "A sample tool for testing purposes"

    def __init__(self, name: Optional[str] = None, description: Optional[str] = None):
        # Override parameters before calling super().__init__ to trigger auto-generation
        self.parameters = {}  # Clear to trigger auto-generation
        super().__init__(name=name, description=description)

    async def execute(self, message: str, count: int = 1) -> str:
        """Execute the sample tool.

        Args:
            message: The message to process.
            count: Number of times to repeat.

        Returns:
            The processed message.
        """
        return message * count


class AsyncSampleTool(BaseTool):
    """Sample async tool for testing async execution."""

    name = "async_sample"
    description = "An async sample tool"

    def __init__(self, name: Optional[str] = None, description: Optional[str] = None):
        self.parameters = {}  # Clear to trigger auto-generation
        super().__init__(name=name, description=description)

    async def execute(self, value: int) -> int:
        """Double the input value.

        Args:
            value: The value to double.

        Returns:
            The doubled value.
        """
        return value * 2


class SlowTool(BaseTool):
    """A tool that takes time to execute for timeout testing."""

    name = "slow_tool"
    description = "A slow tool for timeout testing"

    def __init__(self, delay: float = 5.0):
        # Set explicit empty parameters (no kwargs required)
        self.parameters = {"type": "object", "properties": {}}
        super().__init__()
        self.delay = delay

    async def execute(self, **kwargs: Any) -> str:
        """Execute slowly.

        Returns:
            Success message.
        """
        import asyncio

        await asyncio.sleep(self.delay)
        return "completed"


class CounterTool(BaseTool):
    """A tool that counts executions for testing."""

    name = "counter"
    description = "Counts how many times it was called"

    def __init__(self):
        # Set explicit empty parameters (no kwargs required)
        self.parameters = {"type": "object", "properties": {}}
        super().__init__()
        self.count = 0

    async def execute(self, **kwargs: Any) -> int:
        """Increment and return count.

        Returns:
            Current call count.
        """
        self.count += 1
        return self.count


class ErrorTool(BaseTool):
    """A tool that raises an error after n calls."""

    name = "error_tool"
    description = "Raises an error for testing error handling"

    def __init__(self, fail_after: int = 0):
        # Set explicit empty parameters (no kwargs required)
        self.parameters = {"type": "object", "properties": {}}
        super().__init__()
        self.fail_after = fail_after
        self.call_count = 0

    async def execute(self, **kwargs: Any) -> str:
        """Execute and possibly fail.

        Returns:
            Success if not failing.

        Raises:
            RuntimeError: When call count exceeds fail_after.
        """
        self.call_count += 1
        if self.call_count > self.fail_after:
            raise RuntimeError(f"Intentional failure at call {self.call_count}")
        return "success"


# ============================================================================
# Sample Functions for FunctionTool Testing
# ============================================================================


def sync_add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.

    Returns:
        Sum of a and b.
    """
    return a + b


async def async_multiply(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
        x: First number.
        y: Second number.

    Returns:
        Product of x and y.
    """
    return x * y


def greet(name: str, greeting: str = "Hello") -> str:
    """Create a greeting message.

    Args:
        name: The name to greet.
        greeting: The greeting word.

    Returns:
        The greeting message.
    """
    return f"{greeting}, {name}!"


async def process_items(items: List[str], uppercase: bool = False) -> List[str]:
    """Process a list of items.

    Args:
        items: Items to process.
        uppercase: Whether to uppercase items.

    Returns:
        Processed items.
    """
    if uppercase:
        return [item.upper() for item in items]
    return items


def no_docstring_func(x: int) -> int:
    return x * 2


def no_type_hints(a, b):
    """Add two values without type hints.

    Args:
        a: First value.
        b: Second value.

    Returns:
        Sum of values.
    """
    return a + b


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_tool() -> SampleTool:
    """Create a sample tool instance."""
    return SampleTool()


@pytest.fixture
def async_sample_tool() -> AsyncSampleTool:
    """Create an async sample tool instance."""
    return AsyncSampleTool()


@pytest.fixture
def slow_tool() -> SlowTool:
    """Create a slow tool with default delay."""
    return SlowTool(delay=0.5)


@pytest.fixture
def counter_tool() -> CounterTool:
    """Create a counter tool instance."""
    return CounterTool()


@pytest.fixture
def error_tool() -> ErrorTool:
    """Create an error tool that fails immediately."""
    return ErrorTool(fail_after=0)


@pytest.fixture
def empty_registry() -> ToolRegistry:
    """Create an empty tool registry."""
    return ToolRegistry()


@pytest.fixture
def populated_registry() -> ToolRegistry:
    """Create a registry with sample tools."""
    registry = ToolRegistry()
    registry.register(SampleTool())
    registry.register(AsyncSampleTool())
    registry.register(CounterTool())
    return registry


@pytest.fixture
def execution_config() -> ExecutionConfig:
    """Create default execution configuration."""
    return ExecutionConfig(
        timeout=5.0,
        max_concurrent=5,
        retry_on_error=False,
        max_retries=3,
        validate_args=True,
    )


@pytest.fixture
def retry_config() -> ExecutionConfig:
    """Create execution config with retries enabled."""
    return ExecutionConfig(
        timeout=5.0,
        max_concurrent=5,
        retry_on_error=True,
        max_retries=3,
        validate_args=True,
    )


@pytest.fixture
def fast_timeout_config() -> ExecutionConfig:
    """Create execution config with short timeout."""
    return ExecutionConfig(
        timeout=0.1,
        max_concurrent=5,
        retry_on_error=False,
        max_retries=0,
        validate_args=True,
    )


@pytest.fixture
def tool_executor(populated_registry: ToolRegistry, execution_config: ExecutionConfig) -> ToolExecutor:
    """Create a tool executor with populated registry."""
    return ToolExecutor(registry=populated_registry, config=execution_config)


@pytest.fixture
def sync_function_tool() -> FunctionTool:
    """Create a FunctionTool from a sync function."""
    return FunctionTool(sync_add)


@pytest.fixture
def async_function_tool() -> FunctionTool:
    """Create a FunctionTool from an async function."""
    return FunctionTool(async_multiply)


@pytest.fixture
def sample_json_schema() -> Dict[str, Any]:
    """Create a sample JSON schema for validation testing."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "active": {"type": "boolean"},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
            },
            "metadata": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["name", "age"],
    }


@pytest.fixture
def valid_schema_data() -> Dict[str, Any]:
    """Create valid data matching the sample schema."""
    return {
        "name": "Test",
        "age": 25,
        "active": True,
        "tags": ["a", "b"],
        "metadata": {"key": "value"},
    }


@pytest.fixture
def invalid_schema_data() -> Dict[str, Any]:
    """Create invalid data not matching the sample schema."""
    return {
        "name": 123,  # Should be string
        "age": "twenty-five",  # Should be integer
    }
