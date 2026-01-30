"""Tools package for the agents framework.

This package provides the tool system for defining, registering, and
executing tools that agents can use to interact with external systems.

Example:
    # Using the @tool decorator
    from agents_framework.tools import tool

    @tool(name="search", description="Search the web")
    async def search_web(query: str) -> str:
        '''Search the web for information.'''
        return f"Results for: {query}"

    # Using class-based tools
    from agents_framework.tools import BaseTool

    class CalculatorTool(BaseTool):
        name = "calculator"
        description = "Perform mathematical calculations"

        async def execute(self, expression: str) -> float:
            return eval(expression)

    # Using the registry
    from agents_framework.tools import ToolRegistry

    registry = ToolRegistry()
    registry.register(search_web)
    registry.register(CalculatorTool())

    # Execute tools
    result = await registry.execute("search", query="AI news")

    # Using spawn_agent for sub-agent creation
    from agents_framework.tools import spawn_agent

    result = await spawn_agent(
        agent_id="researcher",
        purpose="Research task",
        task="Find AI frameworks"
    )
"""

from .base import (
    BaseTool,
    FunctionTool,
    ToolDefinition,
    ToolResult,
    tool,
    sync_tool,
)
from .registry import (
    ToolRegistry,
    get_default_registry,
    register_tool,
)
from .executor import (
    ExecutionConfig,
    ExecutionResult,
    ToolExecutor,
)
from .schema import (
    generate_function_schema,
    generate_dataclass_schema,
    get_json_type,
    validate_against_schema,
)
from .spawn_agent import (
    spawn_agent,
    spawn_parallel,
    create_spawn_agent_tool,
    create_spawn_parallel_tool,
    SPAWN_AGENT_TOOL_DEFINITION,
)

__all__ = [
    # Base classes
    "BaseTool",
    "FunctionTool",
    "ToolDefinition",
    "ToolResult",
    # Decorators
    "tool",
    "sync_tool",
    # Registry
    "ToolRegistry",
    "get_default_registry",
    "register_tool",
    # Executor
    "ExecutionConfig",
    "ExecutionResult",
    "ToolExecutor",
    # Schema utilities
    "generate_function_schema",
    "generate_dataclass_schema",
    "get_json_type",
    "validate_against_schema",
    # Spawn agent
    "spawn_agent",
    "spawn_parallel",
    "create_spawn_agent_tool",
    "create_spawn_parallel_tool",
    "SPAWN_AGENT_TOOL_DEFINITION",
]
