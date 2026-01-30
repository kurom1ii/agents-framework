"""Base tool classes and decorators for the agents framework."""

from __future__ import annotations

import functools
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from .schema import generate_function_schema, validate_against_schema


@dataclass
class ToolResult:
    """Result of a tool execution."""

    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolDefinition:
    """Definition of a tool for LLM consumption."""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema


class BaseTool(ABC):
    """Abstract base class for tools.

    All tools should inherit from this class and implement the execute method.

    Attributes:
        name: The unique name of the tool.
        description: A description of what the tool does.
        parameters: JSON Schema defining the tool's parameters.
    """

    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Initialize the tool.

        Args:
            name: Optional name override.
            description: Optional description override.
        """
        if name:
            self.name = name
        elif not self.name:
            self.name = self.__class__.__name__

        if description:
            self.description = description
        elif not self.description:
            self.description = self.__doc__ or f"Execute {self.name}"

        # Auto-generate parameters from execute method if not defined
        if not hasattr(self, "parameters") or not self.parameters:
            self.parameters = self._generate_parameters()

    def _generate_parameters(self) -> Dict[str, Any]:
        """Generate JSON Schema parameters from execute method signature."""
        schema = generate_function_schema(self.execute)
        return schema.get("parameters", {"type": "object", "properties": {}})

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with the given arguments.

        Args:
            **kwargs: Tool-specific arguments.

        Returns:
            The result of the tool execution.
        """
        pass

    async def run(self, **kwargs: Any) -> ToolResult:
        """Run the tool and wrap the result.

        Args:
            **kwargs: Tool-specific arguments.

        Returns:
            ToolResult with success status and output or error.
        """
        try:
            # Validate arguments against schema
            errors = validate_against_schema(kwargs, self.parameters)
            if errors:
                return ToolResult(
                    success=False,
                    error=f"Validation errors: {', '.join(errors)}",
                )

            result = await self.execute(**kwargs)
            return ToolResult(success=True, output=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def to_definition(self) -> ToolDefinition:
        """Convert tool to ToolDefinition for LLM consumption."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


F = TypeVar("F", bound=Callable[..., Any])


class FunctionTool(BaseTool):
    """Tool wrapper for regular functions."""

    def __init__(
        self,
        func: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self._func = func
        self._is_async = inspect.iscoroutinefunction(func)

        # Generate schema from function
        schema = generate_function_schema(func, description=description, name=name)

        self.name = schema["name"]
        self.description = schema["description"]
        self.parameters = schema["parameters"]

    async def execute(self, **kwargs: Any) -> Any:
        """Execute the wrapped function."""
        if self._is_async:
            return await self._func(**kwargs)
        else:
            return self._func(**kwargs)


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[[F], FunctionTool]:
    """Decorator to convert a function into a tool.

    Args:
        name: Optional name for the tool. Defaults to function name.
        description: Optional description. Defaults to function docstring.

    Returns:
        A FunctionTool instance wrapping the function.

    Example:
        @tool(name="search", description="Search the web")
        async def search_web(query: str) -> str:
            '''Search the web for information.

            Args:
                query: The search query.

            Returns:
                Search results as text.
            '''
            # Implementation
            return results
    """
    def decorator(func: F) -> FunctionTool:
        return FunctionTool(func, name=name, description=description)

    return decorator


def sync_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[[F], FunctionTool]:
    """Decorator for synchronous tool functions.

    Same as @tool but explicitly for sync functions.
    """
    return tool(name=name, description=description)
