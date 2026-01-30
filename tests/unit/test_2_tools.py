"""Test 2: Tool System - Can agents use tools?

This test verifies the tool execution pipeline:
- Tool registration
- Tool schema generation
- Tool execution
- Result handling
"""

import pytest
from agents_framework.tools.base import BaseTool, FunctionTool, ToolResult, tool
from agents_framework.tools.registry import ToolRegistry
from agents_framework.tools.executor import ToolExecutor


class CalculatorTool(BaseTool):
    """Simple calculator tool for testing."""

    name = "calculator"
    description = "Perform basic math operations"
    parameters = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression"},
        },
        "required": ["expression"],
    }

    async def execute(self, expression: str) -> str:
        try:
            result = eval(expression)  # Safe for testing only
            return str(result)
        except Exception as e:
            return f"Error: {e}"


class TestToolSystemCore:
    """Test tool system core functionality."""

    def test_tool_result_success(self):
        """ToolResult represents successful execution."""
        result = ToolResult(success=True, output="42")

        assert result.success is True
        assert result.output == "42"
        assert result.error is None

    def test_tool_result_failure(self):
        """ToolResult represents failed execution."""
        result = ToolResult(success=False, error="Division by zero")

        assert result.success is False
        assert result.error == "Division by zero"

    @pytest.mark.asyncio
    async def test_base_tool_execution(self):
        """BaseTool executes and returns result."""
        tool = CalculatorTool()

        result = await tool.run(expression="2 + 2")

        assert result.success is True
        assert result.output == "4"

    def test_tool_to_definition(self):
        """Tool converts to LLM-compatible definition."""
        tool = CalculatorTool()

        definition = tool.to_definition()

        assert definition.name == "calculator"
        assert definition.description == "Perform basic math operations"
        assert "expression" in definition.parameters["properties"]

    @pytest.mark.asyncio
    async def test_function_tool_from_sync_function(self):
        """FunctionTool wraps sync functions."""
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        func_tool = FunctionTool(add)

        assert func_tool.name == "add"
        result = await func_tool.execute(a=2, b=3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_function_tool_from_async_function(self):
        """FunctionTool wraps async functions."""
        async def multiply(x: int, y: int) -> int:
            """Multiply two numbers."""
            return x * y

        func_tool = FunctionTool(multiply)

        result = await func_tool.execute(x=3, y=4)
        assert result == 12

    @pytest.mark.asyncio
    async def test_tool_decorator(self):
        """@tool decorator creates FunctionTool."""
        @tool(name="greet", description="Greet someone")
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        assert greet.name == "greet"
        result = await greet.execute(name="World")
        assert result == "Hello, World!"

    def test_tool_registry_register_and_get(self):
        """ToolRegistry manages tool collection."""
        registry = ToolRegistry()
        calc_tool = CalculatorTool()

        registry.register(calc_tool)

        assert registry.has("calculator")
        retrieved = registry.get("calculator")
        assert retrieved is calc_tool

    def test_tool_registry_to_definitions(self):
        """ToolRegistry exports all tool definitions."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())

        definitions = registry.to_definitions()

        assert len(definitions) == 1
        assert definitions[0].name == "calculator"

    @pytest.mark.asyncio
    async def test_tool_executor_runs_tool(self):
        """ToolExecutor executes tools from registry."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        executor = ToolExecutor(registry)

        execution_result = await executor.execute("calculator", "call_1", {"expression": "10 * 5"})

        assert execution_result.result.success is True
        assert execution_result.result.output == "50"

    @pytest.mark.asyncio
    async def test_tool_executor_handles_missing_tool(self):
        """ToolExecutor handles non-existent tools gracefully."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)

        execution_result = await executor.execute("nonexistent", "call_1", {"arg": "value"})

        assert execution_result.result.success is False
        assert "not found" in execution_result.result.error.lower()
