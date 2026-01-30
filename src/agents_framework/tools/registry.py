"""Tool registry for managing and discovering tools."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Type, Union

from .base import BaseTool, FunctionTool, ToolDefinition, ToolResult


class ToolRegistry:
    """Registry for managing tools.

    Provides tool registration, lookup, and execution capabilities.
    Supports both class-based tools (BaseTool) and function-based tools.
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: Dict[str, BaseTool] = {}

    def register(
        self,
        tool: Union[BaseTool, Callable[..., Any]],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> BaseTool:
        """Register a tool in the registry.

        Args:
            tool: A BaseTool instance or a callable to wrap as a tool.
            name: Optional name override for the tool.
            description: Optional description override.

        Returns:
            The registered BaseTool instance.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if isinstance(tool, BaseTool):
            registered_tool = tool
            if name:
                registered_tool.name = name
            if description:
                registered_tool.description = description
        else:
            # Wrap callable as FunctionTool
            registered_tool = FunctionTool(tool, name=name, description=description)

        if registered_tool.name in self._tools:
            raise ValueError(
                f"Tool '{registered_tool.name}' is already registered. "
                "Use a different name or unregister the existing tool first."
            )

        self._tools[registered_tool.name] = registered_tool
        return registered_tool

    def register_all(
        self,
        tools: Iterable[Union[BaseTool, Callable[..., Any]]],
    ) -> List[BaseTool]:
        """Register multiple tools at once.

        Args:
            tools: An iterable of tools to register.

        Returns:
            List of registered BaseTool instances.
        """
        return [self.register(tool) for tool in tools]

    def unregister(self, name: str) -> Optional[BaseTool]:
        """Unregister a tool by name.

        Args:
            name: The name of the tool to unregister.

        Returns:
            The unregistered tool if found, None otherwise.
        """
        return self._tools.pop(name, None)

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name.

        Args:
            name: The name of the tool to retrieve.

        Returns:
            The tool if found, None otherwise.
        """
        return self._tools.get(name)

    def get_or_raise(self, name: str) -> BaseTool:
        """Get a tool by name, raising an error if not found.

        Args:
            name: The name of the tool to retrieve.

        Returns:
            The tool.

        Raises:
            KeyError: If the tool is not found.
        """
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self._tools.keys()) or "none"
            raise KeyError(
                f"Tool '{name}' not found. Available tools: {available}"
            )
        return tool

    def has(self, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: The name of the tool to check.

        Returns:
            True if the tool is registered, False otherwise.
        """
        return name in self._tools

    def list_tools(self) -> List[BaseTool]:
        """List all registered tools.

        Returns:
            List of all registered tools.
        """
        return list(self._tools.values())

    def list_names(self) -> List[str]:
        """List all registered tool names.

        Returns:
            List of tool names.
        """
        return list(self._tools.keys())

    def to_definitions(self) -> List[ToolDefinition]:
        """Convert all tools to ToolDefinition format for LLM consumption.

        Returns:
            List of ToolDefinition objects.
        """
        return [tool.to_definition() for tool in self._tools.values()]

    async def execute(self, name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name.

        Args:
            name: The name of the tool to execute.
            **kwargs: Arguments to pass to the tool.

        Returns:
            ToolResult with success status and output or error.
        """
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                success=False,
                error=f"Tool '{name}' not found",
            )
        return await tool.run(**kwargs)

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def __iter__(self):
        """Iterate over registered tools."""
        return iter(self._tools.values())

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={list(self._tools.keys())})"


# Global default registry
_default_registry = ToolRegistry()


def get_default_registry() -> ToolRegistry:
    """Get the default global tool registry."""
    return _default_registry


def register_tool(
    tool: Union[BaseTool, Callable[..., Any]],
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> BaseTool:
    """Register a tool in the default registry.

    Convenience function for registering tools without creating a registry.

    Args:
        tool: A BaseTool instance or a callable to wrap.
        name: Optional name override.
        description: Optional description override.

    Returns:
        The registered BaseTool instance.
    """
    return _default_registry.register(tool, name=name, description=description)
