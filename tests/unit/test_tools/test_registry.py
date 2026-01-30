"""Tests for tool registry functionality.

This module tests:
- ToolRegistry class methods
- Tool registration and lookup
- Default registry functions
"""

from __future__ import annotations

from typing import Any

import pytest

from agents_framework.tools.base import BaseTool, FunctionTool, ToolDefinition
from agents_framework.tools.registry import (
    ToolRegistry,
    get_default_registry,
    register_tool,
    _default_registry,
)


# ============================================================================
# ToolRegistry Initialization Tests
# ============================================================================


class TestToolRegistryInit:
    """Tests for ToolRegistry initialization."""

    def test_empty_registry(self, empty_registry: ToolRegistry):
        """Test that new registry is empty."""
        assert len(empty_registry) == 0
        assert empty_registry.list_names() == []
        assert empty_registry.list_tools() == []

    def test_registry_repr(self, empty_registry: ToolRegistry):
        """Test registry string representation."""
        repr_str = repr(empty_registry)

        assert "ToolRegistry" in repr_str
        assert "tools=" in repr_str


# ============================================================================
# Registration Tests
# ============================================================================


class TestToolRegistration:
    """Tests for tool registration functionality."""

    def test_register_base_tool(self, empty_registry: ToolRegistry, sample_tool):
        """Test registering a BaseTool instance."""
        result = empty_registry.register(sample_tool)

        assert result is sample_tool
        assert empty_registry.has("sample_tool")
        assert len(empty_registry) == 1

    def test_register_callable(self, empty_registry: ToolRegistry):
        """Test registering a callable function."""

        def my_function(x: int) -> int:
            """Double x.

            Args:
                x: Value to double.

            Returns:
                Doubled value.
            """
            return x * 2

        result = empty_registry.register(my_function)

        assert isinstance(result, FunctionTool)
        assert result.name == "my_function"
        assert empty_registry.has("my_function")

    def test_register_with_name_override(self, empty_registry: ToolRegistry, sample_tool):
        """Test registering with custom name."""
        empty_registry.register(sample_tool, name="custom_name")

        assert sample_tool.name == "custom_name"
        assert empty_registry.has("custom_name")
        assert not empty_registry.has("sample_tool")

    def test_register_with_description_override(
        self, empty_registry: ToolRegistry, sample_tool
    ):
        """Test registering with custom description."""
        empty_registry.register(sample_tool, description="Custom description")

        assert sample_tool.description == "Custom description"

    def test_register_duplicate_raises(
        self, empty_registry: ToolRegistry, sample_tool
    ):
        """Test that registering duplicate name raises error."""
        empty_registry.register(sample_tool)

        with pytest.raises(ValueError) as exc_info:
            empty_registry.register(sample_tool)

        assert "already registered" in str(exc_info.value)
        assert "sample_tool" in str(exc_info.value)

    def test_register_all(self, empty_registry: ToolRegistry):
        """Test registering multiple tools at once."""
        from tests.unit.test_tools.conftest import SampleTool, AsyncSampleTool, CounterTool

        tools = [SampleTool(), AsyncSampleTool(), CounterTool()]
        results = empty_registry.register_all(tools)

        assert len(results) == 3
        assert len(empty_registry) == 3
        assert empty_registry.has("sample_tool")
        assert empty_registry.has("async_sample")
        assert empty_registry.has("counter")

    def test_register_all_with_callables(self, empty_registry: ToolRegistry):
        """Test registering mix of tools and callables."""
        from tests.unit.test_tools.conftest import SampleTool

        def helper(x: int) -> int:
            """Help function."""
            return x

        items = [SampleTool(), helper]
        results = empty_registry.register_all(items)

        assert len(results) == 2
        assert empty_registry.has("sample_tool")
        assert empty_registry.has("helper")


# ============================================================================
# Unregistration Tests
# ============================================================================


class TestToolUnregistration:
    """Tests for tool unregistration."""

    def test_unregister_existing(self, populated_registry: ToolRegistry):
        """Test unregistering an existing tool."""
        initial_count = len(populated_registry)
        removed = populated_registry.unregister("sample_tool")

        assert removed is not None
        assert removed.name == "sample_tool"
        assert len(populated_registry) == initial_count - 1
        assert not populated_registry.has("sample_tool")

    def test_unregister_nonexistent(self, populated_registry: ToolRegistry):
        """Test unregistering a non-existent tool."""
        result = populated_registry.unregister("nonexistent_tool")

        assert result is None

    def test_clear_registry(self, populated_registry: ToolRegistry):
        """Test clearing all tools from registry."""
        assert len(populated_registry) > 0

        populated_registry.clear()

        assert len(populated_registry) == 0
        assert populated_registry.list_names() == []


# ============================================================================
# Lookup Tests
# ============================================================================


class TestToolLookup:
    """Tests for tool lookup functionality."""

    def test_get_existing_tool(self, populated_registry: ToolRegistry):
        """Test getting an existing tool."""
        tool = populated_registry.get("sample_tool")

        assert tool is not None
        assert tool.name == "sample_tool"

    def test_get_nonexistent_tool(self, populated_registry: ToolRegistry):
        """Test getting a non-existent tool."""
        result = populated_registry.get("nonexistent")

        assert result is None

    def test_get_or_raise_existing(self, populated_registry: ToolRegistry):
        """Test get_or_raise with existing tool."""
        tool = populated_registry.get_or_raise("sample_tool")

        assert tool is not None
        assert tool.name == "sample_tool"

    def test_get_or_raise_nonexistent(self, populated_registry: ToolRegistry):
        """Test get_or_raise with non-existent tool."""
        with pytest.raises(KeyError) as exc_info:
            populated_registry.get_or_raise("nonexistent")

        assert "nonexistent" in str(exc_info.value)
        assert "not found" in str(exc_info.value).lower()
        # Should list available tools
        assert "sample_tool" in str(exc_info.value) or "Available" in str(exc_info.value)

    def test_get_or_raise_empty_registry(self, empty_registry: ToolRegistry):
        """Test get_or_raise with empty registry."""
        with pytest.raises(KeyError) as exc_info:
            empty_registry.get_or_raise("any_tool")

        assert "none" in str(exc_info.value).lower() or "any_tool" in str(exc_info.value)

    def test_has_existing(self, populated_registry: ToolRegistry):
        """Test has() with existing tool."""
        assert populated_registry.has("sample_tool") is True

    def test_has_nonexistent(self, populated_registry: ToolRegistry):
        """Test has() with non-existent tool."""
        assert populated_registry.has("nonexistent") is False

    def test_contains_operator(self, populated_registry: ToolRegistry):
        """Test 'in' operator."""
        assert "sample_tool" in populated_registry
        assert "nonexistent" not in populated_registry


# ============================================================================
# Listing Tests
# ============================================================================


class TestToolListing:
    """Tests for tool listing functionality."""

    def test_list_tools(self, populated_registry: ToolRegistry):
        """Test listing all tools."""
        tools = populated_registry.list_tools()

        assert isinstance(tools, list)
        assert len(tools) == len(populated_registry)
        assert all(isinstance(t, BaseTool) for t in tools)

    def test_list_names(self, populated_registry: ToolRegistry):
        """Test listing all tool names."""
        names = populated_registry.list_names()

        assert isinstance(names, list)
        assert len(names) == len(populated_registry)
        assert all(isinstance(n, str) for n in names)
        assert "sample_tool" in names

    def test_to_definitions(self, populated_registry: ToolRegistry):
        """Test converting tools to definitions."""
        definitions = populated_registry.to_definitions()

        assert isinstance(definitions, list)
        assert len(definitions) == len(populated_registry)
        assert all(isinstance(d, ToolDefinition) for d in definitions)

        # Check that definitions have correct structure
        names = [d.name for d in definitions]
        assert "sample_tool" in names

    def test_iteration(self, populated_registry: ToolRegistry):
        """Test iterating over registry."""
        tools = list(populated_registry)

        assert len(tools) == len(populated_registry)
        assert all(isinstance(t, BaseTool) for t in tools)


# ============================================================================
# Execution Tests
# ============================================================================


class TestRegistryExecution:
    """Tests for tool execution via registry."""

    @pytest.mark.asyncio
    async def test_execute_existing_tool(self, populated_registry: ToolRegistry):
        """Test executing an existing tool."""
        result = await populated_registry.execute("sample_tool", message="hello")

        assert result.success is True
        assert result.output == "hello"

    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self, populated_registry: ToolRegistry):
        """Test executing a non-existent tool."""
        result = await populated_registry.execute("nonexistent")

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_with_arguments(self, populated_registry: ToolRegistry):
        """Test executing tool with multiple arguments."""
        result = await populated_registry.execute(
            "sample_tool", message="abc", count=3
        )

        assert result.success is True
        assert result.output == "abcabcabc"

    @pytest.mark.asyncio
    async def test_execute_validation_error(self, populated_registry: ToolRegistry):
        """Test executing with invalid arguments."""
        result = await populated_registry.execute("sample_tool", message=123)

        assert result.success is False
        assert result.error is not None


# ============================================================================
# Default Registry Tests
# ============================================================================


class TestDefaultRegistry:
    """Tests for default global registry functions."""

    def test_get_default_registry(self):
        """Test getting the default registry."""
        registry = get_default_registry()

        assert isinstance(registry, ToolRegistry)

    def test_default_registry_is_singleton(self):
        """Test that default registry returns same instance."""
        registry1 = get_default_registry()
        registry2 = get_default_registry()

        assert registry1 is registry2

    def test_register_tool_function(self):
        """Test register_tool convenience function."""
        # Clear default registry first
        default = get_default_registry()
        if "test_global_tool" in default:
            default.unregister("test_global_tool")

        def test_global_tool(x: int) -> int:
            """A test tool.

            Args:
                x: Input value.

            Returns:
                Output value.
            """
            return x

        result = register_tool(test_global_tool)

        assert isinstance(result, BaseTool)
        assert default.has("test_global_tool")

        # Cleanup
        default.unregister("test_global_tool")

    def test_register_tool_with_overrides(self):
        """Test register_tool with name/description overrides."""
        default = get_default_registry()

        # Cleanup if exists
        if "custom_global" in default:
            default.unregister("custom_global")

        def some_func(x: str) -> str:
            """Original description."""
            return x

        register_tool(some_func, name="custom_global", description="Custom desc")

        tool = default.get("custom_global")
        assert tool is not None
        assert tool.description == "Custom desc"

        # Cleanup
        default.unregister("custom_global")


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestRegistryEdgeCases:
    """Tests for edge cases and error handling."""

    def test_register_after_unregister(self, empty_registry: ToolRegistry):
        """Test re-registering after unregistering."""
        from tests.unit.test_tools.conftest import SampleTool

        tool1 = SampleTool()
        empty_registry.register(tool1)
        empty_registry.unregister("sample_tool")

        tool2 = SampleTool()
        empty_registry.register(tool2)

        assert empty_registry.has("sample_tool")

    def test_register_multiple_different_tools(self, empty_registry: ToolRegistry):
        """Test registering multiple different tools."""
        from tests.unit.test_tools.conftest import SampleTool, AsyncSampleTool

        empty_registry.register(SampleTool())
        empty_registry.register(AsyncSampleTool())

        assert len(empty_registry) == 2
        assert empty_registry.has("sample_tool")
        assert empty_registry.has("async_sample")

    def test_modify_tool_after_registration(self, empty_registry: ToolRegistry):
        """Test that modifying tool name after registration works."""
        from tests.unit.test_tools.conftest import SampleTool

        tool = SampleTool()
        empty_registry.register(tool)

        # Modifying the tool's name shouldn't affect registry lookup
        # The registry still uses the original registered name
        original_name = tool.name
        tool.name = "modified_name"

        # Registry should still find it under original name
        found = empty_registry.get(original_name)
        assert found is tool

    def test_len_after_operations(self, empty_registry: ToolRegistry):
        """Test len() reflects actual count after operations."""
        from tests.unit.test_tools.conftest import SampleTool, AsyncSampleTool

        assert len(empty_registry) == 0

        empty_registry.register(SampleTool())
        assert len(empty_registry) == 1

        empty_registry.register(AsyncSampleTool())
        assert len(empty_registry) == 2

        empty_registry.unregister("sample_tool")
        assert len(empty_registry) == 1

        empty_registry.clear()
        assert len(empty_registry) == 0

    @pytest.mark.asyncio
    async def test_execute_tool_that_raises(self, empty_registry: ToolRegistry):
        """Test executing tool that raises an exception."""
        from tests.unit.test_tools.conftest import ErrorTool

        tool = ErrorTool(fail_after=0)
        empty_registry.register(tool)

        result = await empty_registry.execute("error_tool")

        assert result.success is False
        assert "Intentional failure" in result.error

    def test_repr_with_tools(self, populated_registry: ToolRegistry):
        """Test repr shows tool names."""
        repr_str = repr(populated_registry)

        assert "ToolRegistry" in repr_str
        # Should show some tool names
        assert "sample_tool" in repr_str or "tools=" in repr_str

    @pytest.mark.asyncio
    async def test_execute_counter_tool_multiple_times(
        self, populated_registry: ToolRegistry
    ):
        """Test that tool state persists across executions."""
        result1 = await populated_registry.execute("counter")
        result2 = await populated_registry.execute("counter")
        result3 = await populated_registry.execute("counter")

        assert result1.output == 1
        assert result2.output == 2
        assert result3.output == 3


# ============================================================================
# Async Function Registration Tests
# ============================================================================


class TestAsyncFunctionRegistration:
    """Tests for registering async functions."""

    def test_register_async_function(self, empty_registry: ToolRegistry):
        """Test registering an async function."""

        async def async_helper(value: str) -> str:
            """Async helper.

            Args:
                value: The value.

            Returns:
                Processed value.
            """
            return value.upper()

        result = empty_registry.register(async_helper)

        assert isinstance(result, FunctionTool)
        assert result._is_async is True
        assert empty_registry.has("async_helper")

    @pytest.mark.asyncio
    async def test_execute_registered_async_function(
        self, empty_registry: ToolRegistry
    ):
        """Test executing a registered async function."""

        async def double_async(n: int) -> int:
            """Double n async.

            Args:
                n: Number to double.

            Returns:
                Doubled number.
            """
            import asyncio

            await asyncio.sleep(0.01)  # Simulate async work
            return n * 2

        empty_registry.register(double_async)
        result = await empty_registry.execute("double_async", n=5)

        assert result.success is True
        assert result.output == 10
