"""MCP tools adapter for the agents framework.

This module provides adapters to integrate MCP tools with the
framework's tool system, allowing MCP tools to be used alongside
native tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from agents_framework.tools import BaseTool, ToolDefinition, ToolRegistry, ToolResult

from .client import MCPClient, MCPConnectionManager, MCPTool
from .transport import MCPError


class MCPToolAdapter(BaseTool):
    """Adapter to use MCP tools as framework tools.

    Wraps an MCP tool so it can be used with the framework's
    tool registry and executor.
    """

    def __init__(
        self,
        mcp_tool: MCPTool,
        client: MCPClient,
        name_prefix: Optional[str] = None,
    ):
        """Initialize the adapter.

        Args:
            mcp_tool: The MCP tool to wrap.
            client: The MCP client for calling the tool.
            name_prefix: Optional prefix for the tool name.
        """
        self.mcp_tool = mcp_tool
        self.client = client
        self.name_prefix = name_prefix

        # Set tool attributes
        if name_prefix:
            self.name = f"{name_prefix}/{mcp_tool.name}"
        else:
            self.name = mcp_tool.name

        self.description = mcp_tool.description
        self.parameters = mcp_tool.input_schema

    async def execute(self, **kwargs: Any) -> Any:
        """Execute the MCP tool.

        Args:
            **kwargs: Tool arguments.

        Returns:
            Tool execution result.
        """
        return await self.client.call_tool(self.mcp_tool.name, kwargs)

    def to_definition(self) -> ToolDefinition:
        """Convert to ToolDefinition for LLM consumption."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


class MCPToolRegistry:
    """Registry for managing MCP tools.

    Provides a way to discover and use tools from MCP servers
    through the framework's tool interface.
    """

    def __init__(self, connection_manager: MCPConnectionManager):
        """Initialize the registry.

        Args:
            connection_manager: MCP connection manager.
        """
        self.connection_manager = connection_manager
        self._tools: Dict[str, MCPToolAdapter] = {}
        self._framework_registry: Optional[ToolRegistry] = None

    async def refresh_tools(self) -> None:
        """Refresh the list of available tools from all servers."""
        self._tools.clear()

        for server_name, client in self.connection_manager._clients.items():
            for mcp_tool in client.list_tools():
                # Create adapter with server prefix
                adapter = MCPToolAdapter(
                    mcp_tool=mcp_tool,
                    client=client,
                    name_prefix=server_name,
                )
                self._tools[adapter.name] = adapter

                # Also register without prefix if unique
                if mcp_tool.name not in self._tools:
                    unprefixed_adapter = MCPToolAdapter(
                        mcp_tool=mcp_tool,
                        client=client,
                    )
                    self._tools[mcp_tool.name] = unprefixed_adapter

    def get_tool(self, name: str) -> Optional[MCPToolAdapter]:
        """Get a tool by name.

        Args:
            name: Tool name (may be prefixed or unprefixed).

        Returns:
            MCPToolAdapter if found, None otherwise.
        """
        return self._tools.get(name)

    def list_tools(self) -> List[MCPToolAdapter]:
        """List all available tools.

        Returns:
            List of MCPToolAdapter objects.
        """
        return list(self._tools.values())

    def to_definitions(self) -> List[ToolDefinition]:
        """Convert all tools to ToolDefinition format.

        Returns:
            List of ToolDefinition objects.
        """
        return [tool.to_definition() for tool in self._tools.values()]

    async def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Call a tool by name.

        Args:
            name: Tool name.
            arguments: Tool arguments.

        Returns:
            ToolResult with success status and output.
        """
        tool = self.get_tool(name)
        if tool is None:
            return ToolResult(
                success=False,
                error=f"Tool '{name}' not found",
            )

        try:
            result = await tool.execute(**(arguments or {}))
            return ToolResult(success=True, output=result)
        except MCPError as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(success=False, error=f"Tool execution failed: {e}")

    def register_with_framework(self, registry: ToolRegistry) -> None:
        """Register all MCP tools with a framework tool registry.

        Args:
            registry: The framework's ToolRegistry.
        """
        self._framework_registry = registry
        for tool in self._tools.values():
            if not registry.has(tool.name):
                registry.register(tool)


@dataclass
class MCPToolsConfig:
    """Configuration for MCP tools integration.

    Attributes:
        auto_refresh: Whether to auto-refresh tools on connection.
        prefix_with_server: Whether to prefix tool names with server name.
        register_to_framework: Whether to register with framework registry.
    """

    auto_refresh: bool = True
    prefix_with_server: bool = True
    register_to_framework: bool = True


def create_mcp_tools_from_client(
    client: MCPClient,
    name_prefix: Optional[str] = None,
) -> List[MCPToolAdapter]:
    """Create tool adapters from an MCP client.

    Args:
        client: Connected MCP client.
        name_prefix: Optional prefix for tool names.

    Returns:
        List of MCPToolAdapter objects.
    """
    adapters = []
    for mcp_tool in client.list_tools():
        adapter = MCPToolAdapter(
            mcp_tool=mcp_tool,
            client=client,
            name_prefix=name_prefix,
        )
        adapters.append(adapter)
    return adapters


def register_mcp_tools(
    connection_manager: MCPConnectionManager,
    registry: ToolRegistry,
    use_prefixes: bool = True,
) -> int:
    """Register all MCP tools with a framework tool registry.

    Args:
        connection_manager: MCP connection manager.
        registry: Framework tool registry.
        use_prefixes: Whether to use server name prefixes.

    Returns:
        Number of tools registered.
    """
    count = 0
    for server_name, client in connection_manager._clients.items():
        prefix = server_name if use_prefixes else None
        adapters = create_mcp_tools_from_client(client, name_prefix=prefix)
        for adapter in adapters:
            if not registry.has(adapter.name):
                registry.register(adapter)
                count += 1
    return count
