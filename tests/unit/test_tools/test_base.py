"""Tests for base tool classes and decorators.

This module tests:
- ToolResult dataclass
- ToolDefinition dataclass
- BaseTool class
- FunctionTool class
- @tool decorator
- @sync_tool decorator
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import pytest

from agents_framework.tools.base import (
    BaseTool,
    FunctionTool,
    ToolDefinition,
    ToolResult,
    tool,
    sync_tool,
)


# ============================================================================
# ToolResult Tests
# ============================================================================


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_success_result(self):
        """Test creating a successful result."""
        result = ToolResult(success=True, output="test output")

        assert result.success is True
        assert result.output == "test output"
        assert result.error is None
        assert result.metadata == {}

    def test_failure_result(self):
        """Test creating a failure result."""
        result = ToolResult(success=False, error="Something went wrong")

        assert result.success is False
        assert result.output is None
        assert result.error == "Something went wrong"

    def test_result_with_metadata(self):
        """Test result with metadata."""
        metadata = {"execution_time": 1.5, "tokens_used": 100}
        result = ToolResult(success=True, output="ok", metadata=metadata)

        assert result.metadata == metadata
        assert result.metadata["execution_time"] == 1.5

    def test_result_with_complex_output(self):
        """Test result with complex output types."""
        output = {"data": [1, 2, 3], "nested": {"key": "value"}}
        result = ToolResult(success=True, output=output)

        assert result.output == output
        assert result.output["data"] == [1, 2, 3]

    def test_result_defaults(self):
        """Test default values for ToolResult."""
        result = ToolResult(success=True)

        assert result.output is None
        assert result.error is None
        assert result.metadata == {}

    @pytest.mark.parametrize(
        "success,output,error",
        [
            (True, "success", None),
            (False, None, "error"),
            (True, 42, None),
            (True, ["list", "output"], None),
            (False, None, ""),
        ],
    )
    def test_various_result_combinations(self, success, output, error):
        """Test various combinations of result parameters."""
        result = ToolResult(success=success, output=output, error=error)

        assert result.success == success
        assert result.output == output
        assert result.error == error


# ============================================================================
# ToolDefinition Tests
# ============================================================================


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    def test_basic_definition(self):
        """Test creating a basic tool definition."""
        definition = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        )

        assert definition.name == "test_tool"
        assert definition.description == "A test tool"
        assert definition.parameters == {"type": "object", "properties": {}}

    def test_definition_with_parameters(self):
        """Test definition with full parameter schema."""
        params = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        }
        definition = ToolDefinition(
            name="search",
            description="Search the web",
            parameters=params,
        )

        assert definition.parameters == params
        assert "query" in definition.parameters["properties"]
        assert definition.parameters["required"] == ["query"]

    def test_definition_equality(self):
        """Test that two equivalent definitions are equal."""
        params = {"type": "object", "properties": {}}
        def1 = ToolDefinition(name="tool", description="desc", parameters=params)
        def2 = ToolDefinition(name="tool", description="desc", parameters=params)

        assert def1 == def2


# ============================================================================
# BaseTool Tests
# ============================================================================


class TestBaseTool:
    """Tests for BaseTool abstract class."""

    def test_init_with_defaults(self, sample_tool):
        """Test initialization with default values."""
        assert sample_tool.name == "sample_tool"
        assert sample_tool.description == "A sample tool for testing purposes"
        assert isinstance(sample_tool.parameters, dict)

    def test_init_with_overrides(self):
        """Test initialization with name/description overrides."""
        from tests.unit.test_tools.conftest import SampleTool

        tool = SampleTool(name="custom_name", description="Custom description")

        assert tool.name == "custom_name"
        assert tool.description == "Custom description"

    def test_auto_name_from_class(self):
        """Test auto-generation of name from class name."""

        class MyCustomTool(BaseTool):
            async def execute(self, **kwargs):
                return "ok"

        tool = MyCustomTool()
        assert tool.name == "MyCustomTool"

    def test_auto_description_from_docstring(self):
        """Test auto-generation of description from docstring."""

        class DocumentedTool(BaseTool):
            """This is a documented tool."""

            async def execute(self, **kwargs):
                return "ok"

        tool = DocumentedTool()
        assert tool.description == "This is a documented tool."

    def test_generate_parameters(self, sample_tool):
        """Test automatic parameter generation."""
        params = sample_tool.parameters

        assert params["type"] == "object"
        assert "properties" in params
        assert "message" in params["properties"]
        assert "count" in params["properties"]
        assert params["properties"]["message"]["type"] == "string"
        assert params["properties"]["count"]["type"] == "integer"

    def test_generate_parameters_with_required(self, sample_tool):
        """Test that required parameters are marked correctly."""
        params = sample_tool.parameters

        # message is required (no default), count has default
        assert "required" in params
        assert "message" in params["required"]
        assert "count" not in params["required"]

    @pytest.mark.asyncio
    async def test_run_success(self, sample_tool):
        """Test successful tool execution via run()."""
        result = await sample_tool.run(message="test", count=2)

        assert result.success is True
        assert result.output == "testtest"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_run_with_defaults(self, sample_tool):
        """Test tool execution using default parameter values."""
        result = await sample_tool.run(message="hello")

        assert result.success is True
        assert result.output == "hello"

    @pytest.mark.asyncio
    async def test_run_validation_failure(self, sample_tool):
        """Test that run() returns error for invalid arguments."""
        # Passing wrong type for message (should be string)
        result = await sample_tool.run(message=123)

        assert result.success is False
        assert result.error is not None
        assert "Validation" in result.error or "type" in result.error.lower()

    @pytest.mark.asyncio
    async def test_run_with_missing_required(self, sample_tool):
        """Test run() with missing required parameter."""
        result = await sample_tool.run()  # missing 'message'

        assert result.success is False
        assert result.error is not None
        assert "required" in result.error.lower() or "message" in result.error.lower()

    @pytest.mark.asyncio
    async def test_run_exception_handling(self, error_tool):
        """Test that run() catches exceptions."""
        result = await error_tool.run()

        assert result.success is False
        assert result.error is not None
        assert "Intentional failure" in result.error

    def test_to_definition(self, sample_tool):
        """Test conversion to ToolDefinition."""
        definition = sample_tool.to_definition()

        assert isinstance(definition, ToolDefinition)
        assert definition.name == sample_tool.name
        assert definition.description == sample_tool.description
        assert definition.parameters == sample_tool.parameters

    def test_repr(self, sample_tool):
        """Test string representation."""
        repr_str = repr(sample_tool)

        assert "SampleTool" in repr_str
        assert "sample_tool" in repr_str


# ============================================================================
# FunctionTool Tests
# ============================================================================


class TestFunctionTool:
    """Tests for FunctionTool class."""

    def test_from_sync_function(self, sync_function_tool):
        """Test creating FunctionTool from sync function."""
        assert sync_function_tool.name == "sync_add"
        assert "Add two numbers" in sync_function_tool.description
        assert sync_function_tool._is_async is False

    def test_from_async_function(self, async_function_tool):
        """Test creating FunctionTool from async function."""
        assert async_function_tool.name == "async_multiply"
        assert "Multiply two numbers" in async_function_tool.description
        assert async_function_tool._is_async is True

    def test_with_name_override(self):
        """Test FunctionTool with custom name."""
        from tests.unit.test_tools.conftest import sync_add

        tool = FunctionTool(sync_add, name="custom_add")

        assert tool.name == "custom_add"

    def test_with_description_override(self):
        """Test FunctionTool with custom description."""
        from tests.unit.test_tools.conftest import sync_add

        tool = FunctionTool(sync_add, description="Custom description")

        assert tool.description == "Custom description"

    def test_parameters_from_sync_function(self, sync_function_tool):
        """Test parameter extraction from sync function."""
        params = sync_function_tool.parameters

        assert params["type"] == "object"
        assert "a" in params["properties"]
        assert "b" in params["properties"]
        assert params["properties"]["a"]["type"] == "integer"
        assert params["properties"]["b"]["type"] == "integer"
        assert set(params["required"]) == {"a", "b"}

    def test_parameters_with_defaults(self):
        """Test parameter extraction for function with defaults."""
        from tests.unit.test_tools.conftest import greet

        tool = FunctionTool(greet)
        params = tool.parameters

        assert "name" in params["properties"]
        assert "greeting" in params["properties"]
        assert params["required"] == ["name"]
        assert params["properties"]["greeting"]["default"] == "Hello"

    def test_parameters_with_complex_types(self):
        """Test parameter extraction with complex types."""
        from tests.unit.test_tools.conftest import process_items

        tool = FunctionTool(process_items)
        params = tool.parameters

        assert params["properties"]["items"]["type"] == "array"
        assert params["properties"]["uppercase"]["type"] == "boolean"

    @pytest.mark.asyncio
    async def test_execute_sync_function(self, sync_function_tool):
        """Test executing a wrapped sync function."""
        result = await sync_function_tool.execute(a=3, b=5)

        assert result == 8

    @pytest.mark.asyncio
    async def test_execute_async_function(self, async_function_tool):
        """Test executing a wrapped async function."""
        result = await async_function_tool.execute(x=2.5, y=4.0)

        assert result == 10.0

    @pytest.mark.asyncio
    async def test_run_sync_function(self, sync_function_tool):
        """Test run() with sync function."""
        result = await sync_function_tool.run(a=10, b=20)

        assert result.success is True
        assert result.output == 30

    @pytest.mark.asyncio
    async def test_run_async_function(self, async_function_tool):
        """Test run() with async function."""
        result = await async_function_tool.run(x=3.0, y=3.0)

        assert result.success is True
        assert result.output == 9.0

    def test_from_function_without_docstring(self):
        """Test FunctionTool from function without docstring."""
        from tests.unit.test_tools.conftest import no_docstring_func

        tool = FunctionTool(no_docstring_func)

        assert tool.name == "no_docstring_func"
        assert "no_docstring_func" in tool.description or "Execute" in tool.description

    def test_from_function_without_type_hints(self):
        """Test FunctionTool from function without type hints."""
        from tests.unit.test_tools.conftest import no_type_hints

        tool = FunctionTool(no_type_hints)

        assert tool.name == "no_type_hints"
        assert "a" in tool.parameters["properties"]
        assert "b" in tool.parameters["properties"]


# ============================================================================
# @tool Decorator Tests
# ============================================================================


class TestToolDecorator:
    """Tests for @tool decorator."""

    def test_basic_decorator(self):
        """Test basic @tool decorator usage."""

        @tool()
        async def my_tool(value: int) -> int:
            """Double a value.

            Args:
                value: The value to double.

            Returns:
                Doubled value.
            """
            return value * 2

        assert isinstance(my_tool, FunctionTool)
        assert my_tool.name == "my_tool"
        assert "Double a value" in my_tool.description

    def test_decorator_with_name(self):
        """Test @tool decorator with custom name."""

        @tool(name="custom_tool")
        async def some_function(x: str) -> str:
            """Process a string."""
            return x.upper()

        assert some_function.name == "custom_tool"

    def test_decorator_with_description(self):
        """Test @tool decorator with custom description."""

        @tool(description="A custom description")
        async def another_function(x: int) -> int:
            """Original docstring."""
            return x + 1

        assert another_function.description == "A custom description"

    def test_decorator_with_both_overrides(self):
        """Test @tool with both name and description."""

        @tool(name="my_custom_tool", description="My custom description")
        async def internal_function(data: str) -> str:
            """Internal description."""
            return data

        assert internal_function.name == "my_custom_tool"
        assert internal_function.description == "My custom description"

    @pytest.mark.asyncio
    async def test_decorated_function_execution(self):
        """Test executing a decorated function."""

        @tool()
        async def concat(a: str, b: str) -> str:
            """Concatenate strings.

            Args:
                a: First string.
                b: Second string.

            Returns:
                Concatenated string.
            """
            return a + b

        result = await concat.run(a="hello", b=" world")

        assert result.success is True
        assert result.output == "hello world"

    def test_decorator_preserves_parameters(self):
        """Test that decorator preserves function parameters."""

        @tool()
        async def complex_func(
            required: str,
            optional: int = 10,
            flag: bool = False,
        ) -> Dict[str, Any]:
            """Complex function.

            Args:
                required: A required parameter.
                optional: An optional parameter.
                flag: A boolean flag.

            Returns:
                Result dict.
            """
            return {"required": required, "optional": optional, "flag": flag}

        params = complex_func.parameters
        assert params["required"] == ["required"]
        assert "optional" in params["properties"]
        assert "flag" in params["properties"]
        assert params["properties"]["optional"]["default"] == 10
        assert params["properties"]["flag"]["default"] is False


# ============================================================================
# @sync_tool Decorator Tests
# ============================================================================


class TestSyncToolDecorator:
    """Tests for @sync_tool decorator."""

    def test_basic_sync_decorator(self):
        """Test basic @sync_tool decorator usage."""

        @sync_tool()
        def my_sync_tool(value: int) -> int:
            """Square a value.

            Args:
                value: The value to square.

            Returns:
                Squared value.
            """
            return value ** 2

        assert isinstance(my_sync_tool, FunctionTool)
        assert my_sync_tool.name == "my_sync_tool"
        assert my_sync_tool._is_async is False

    def test_sync_decorator_with_overrides(self):
        """Test @sync_tool with name and description."""

        @sync_tool(name="square", description="Calculate the square")
        def calc_square(n: int) -> int:
            """Internal docstring."""
            return n * n

        assert calc_square.name == "square"
        assert calc_square.description == "Calculate the square"

    @pytest.mark.asyncio
    async def test_sync_decorated_execution(self):
        """Test executing a @sync_tool decorated function."""

        @sync_tool()
        def add_one(x: int) -> int:
            """Add one to a number.

            Args:
                x: The number.

            Returns:
                Number plus one.
            """
            return x + 1

        result = await add_one.run(x=5)

        assert result.success is True
        assert result.output == 6

    def test_sync_tool_is_same_as_tool(self):
        """Test that @sync_tool behaves like @tool."""

        @tool()
        def with_tool(x: int) -> int:
            """Double x."""
            return x * 2

        @sync_tool()
        def with_sync_tool(x: int) -> int:
            """Double x."""
            return x * 2

        # Both should be FunctionTool instances
        assert type(with_tool) == type(with_sync_tool)
        # Both should not be async
        assert with_tool._is_async is False
        assert with_sync_tool._is_async is False


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_tool_with_no_parameters(self):
        """Test tool with no parameters (except self)."""

        class NoParamTool(BaseTool):
            def __init__(self):
                self.parameters = {}  # Clear to trigger auto-generation
                super().__init__()

            async def execute(self) -> str:
                """Return a constant.

                Returns:
                    A constant string.
                """
                return "constant"

        tool = NoParamTool()
        params = tool.parameters

        assert params["type"] == "object"
        assert params["properties"] == {}

    @pytest.mark.asyncio
    async def test_tool_returning_none(self):
        """Test tool that returns None."""

        class NoneReturnTool(BaseTool):
            def __init__(self):
                self.parameters = {}  # Clear to trigger auto-generation
                super().__init__()

            async def execute(self) -> None:
                """Do nothing."""
                pass

        tool = NoneReturnTool()
        result = await tool.run()

        assert result.success is True
        assert result.output is None

    def test_tool_with_optional_parameters(self):
        """Test tool with all optional parameters."""

        @tool()
        async def all_optional(
            a: Optional[str] = None,
            b: Optional[int] = None,
        ) -> Dict[str, Any]:
            """All params optional.

            Args:
                a: Optional string.
                b: Optional int.

            Returns:
                Dict of values.
            """
            return {"a": a, "b": b}

        params = all_optional.parameters
        assert "required" not in params or params["required"] == []

    def test_tool_with_list_parameters(self):
        """Test tool with list type parameters."""

        @tool()
        async def process_list(items: List[str]) -> int:
            """Process list of items.

            Args:
                items: List of items.

            Returns:
                Count of items.
            """
            return len(items)

        params = process_list.parameters
        assert params["properties"]["items"]["type"] == "array"
        assert params["properties"]["items"]["items"]["type"] == "string"

    def test_tool_with_dict_parameters(self):
        """Test tool with dict type parameters."""

        @tool()
        async def process_dict(data: Dict[str, int]) -> int:
            """Process dict of data.

            Args:
                data: Dict of data.

            Returns:
                Sum of values.
            """
            return sum(data.values())

        params = process_dict.parameters
        assert params["properties"]["data"]["type"] == "object"

    @pytest.mark.asyncio
    async def test_tool_with_empty_string_result(self):
        """Test tool returning empty string."""

        @tool()
        async def empty_result() -> str:
            """Return empty string."""
            return ""

        result = await empty_result.run()

        assert result.success is True
        assert result.output == ""

    def test_class_based_tool_with_class_attributes(self):
        """Test class-based tool with predefined class attributes."""

        class PreDefinedTool(BaseTool):
            name = "predefined"
            description = "A predefined tool"
            parameters = {
                "type": "object",
                "properties": {
                    "custom": {"type": "string"},
                },
                "required": ["custom"],
            }

            async def execute(self, custom: str) -> str:
                return f"custom: {custom}"

        tool = PreDefinedTool()

        assert tool.name == "predefined"
        assert tool.description == "A predefined tool"
        assert tool.parameters["required"] == ["custom"]

    @pytest.mark.asyncio
    async def test_nested_async_calls(self):
        """Test tool that makes nested async calls."""

        async def helper(x: int) -> int:
            return x + 1

        @tool()
        async def nested_async(value: int) -> int:
            """Call helper multiple times.

            Args:
                value: Starting value.

            Returns:
                Final value.
            """
            result = value
            for _ in range(3):
                result = await helper(result)
            return result

        result = await nested_async.run(value=0)

        assert result.success is True
        assert result.output == 3
