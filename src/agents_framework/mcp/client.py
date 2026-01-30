"""MCP (Model Context Protocol) client implementation.

This module provides the core MCP client for:
- Connecting to MCP servers
- Managing multiple server connections
- Discovering and invoking tools
- Resource and prompt management
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from .transport import (
    ConnectionError,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPError,
    SSETransportConfig,
    StdioTransportConfig,
    Transport,
    TransportType,
    create_transport,
)


class MCPCapability(str, Enum):
    """MCP server capabilities."""

    TOOLS = "tools"
    RESOURCES = "resources"
    PROMPTS = "prompts"
    LOGGING = "logging"
    SAMPLING = "sampling"


@dataclass
class MCPTool:
    """Represents an MCP tool.

    Attributes:
        name: Tool name.
        description: Tool description.
        input_schema: JSON Schema for tool input.
    """

    name: str
    description: str
    input_schema: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPTool":
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            input_schema=data.get("inputSchema", {}),
        )


@dataclass
class MCPResource:
    """Represents an MCP resource.

    Attributes:
        uri: Resource URI.
        name: Resource name.
        description: Resource description.
        mime_type: MIME type of the resource.
    """

    uri: str
    name: str
    description: str = ""
    mime_type: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPResource":
        """Create from dictionary."""
        return cls(
            uri=data.get("uri", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            mime_type=data.get("mimeType"),
        )


@dataclass
class MCPPrompt:
    """Represents an MCP prompt template.

    Attributes:
        name: Prompt name.
        description: Prompt description.
        arguments: List of argument definitions.
    """

    name: str
    description: str = ""
    arguments: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPPrompt":
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            arguments=data.get("arguments", []),
        )


@dataclass
class MCPServerInfo:
    """Information about an MCP server.

    Attributes:
        name: Server name.
        version: Server version.
        capabilities: List of server capabilities.
    """

    name: str
    version: str
    capabilities: List[MCPCapability] = field(default_factory=list)


@dataclass
class MCPClientConfig:
    """Configuration for MCP client.

    Attributes:
        name: Client name sent to server.
        version: Client version sent to server.
        capabilities: Client capabilities to advertise.
    """

    name: str = "agents_framework"
    version: str = "0.1.0"
    capabilities: List[MCPCapability] = field(default_factory=list)


class MCPClient:
    """Client for communicating with MCP servers.

    Provides a high-level interface for:
    - Connecting to MCP servers
    - Discovering available tools, resources, and prompts
    - Invoking tools and reading resources
    """

    def __init__(
        self,
        transport: Transport,
        config: Optional[MCPClientConfig] = None,
    ):
        """Initialize the MCP client.

        Args:
            transport: Transport for server communication.
            config: Optional client configuration.
        """
        self.transport = transport
        self.config = config or MCPClientConfig()
        self._server_info: Optional[MCPServerInfo] = None
        self._tools: Dict[str, MCPTool] = {}
        self._resources: Dict[str, MCPResource] = {}
        self._prompts: Dict[str, MCPPrompt] = {}
        self._initialized = False

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self.transport.is_connected

    @property
    def is_initialized(self) -> bool:
        """Check if client is initialized."""
        return self._initialized

    @property
    def server_info(self) -> Optional[MCPServerInfo]:
        """Get server information."""
        return self._server_info

    async def connect(self) -> None:
        """Connect to the MCP server and initialize."""
        await self.transport.connect()
        await self._initialize()

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        self._initialized = False
        self._server_info = None
        self._tools.clear()
        self._resources.clear()
        self._prompts.clear()
        await self.transport.disconnect()

    async def _initialize(self) -> None:
        """Perform MCP initialization handshake."""
        # Send initialize request
        request = JSONRPCRequest(
            method="initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    cap.value: {} for cap in self.config.capabilities
                },
                "clientInfo": {
                    "name": self.config.name,
                    "version": self.config.version,
                },
            },
        )

        response = await self.transport.send_request(request)

        if response.is_error:
            raise MCPError(
                f"Initialize failed: {response.get_error_message()}",
                code=response.get_error_code(),
            )

        # Parse server info
        result = response.result or {}
        server_info = result.get("serverInfo", {})
        capabilities = result.get("capabilities", {})

        self._server_info = MCPServerInfo(
            name=server_info.get("name", "unknown"),
            version=server_info.get("version", "unknown"),
            capabilities=[
                MCPCapability(cap) for cap in capabilities.keys()
                if cap in [c.value for c in MCPCapability]
            ],
        )

        # Send initialized notification
        await self.transport.send_notification(
            JSONRPCNotification(method="notifications/initialized")
        )

        self._initialized = True

        # Discover available tools, resources, prompts
        if MCPCapability.TOOLS in self._server_info.capabilities:
            await self._discover_tools()
        if MCPCapability.RESOURCES in self._server_info.capabilities:
            await self._discover_resources()
        if MCPCapability.PROMPTS in self._server_info.capabilities:
            await self._discover_prompts()

    async def _discover_tools(self) -> None:
        """Discover available tools from the server."""
        response = await self.transport.send_request(
            JSONRPCRequest(method="tools/list")
        )

        if response.is_error:
            return

        result = response.result or {}
        tools_data = result.get("tools", [])

        self._tools = {
            tool["name"]: MCPTool.from_dict(tool)
            for tool in tools_data
        }

    async def _discover_resources(self) -> None:
        """Discover available resources from the server."""
        response = await self.transport.send_request(
            JSONRPCRequest(method="resources/list")
        )

        if response.is_error:
            return

        result = response.result or {}
        resources_data = result.get("resources", [])

        self._resources = {
            res["uri"]: MCPResource.from_dict(res)
            for res in resources_data
        }

    async def _discover_prompts(self) -> None:
        """Discover available prompts from the server."""
        response = await self.transport.send_request(
            JSONRPCRequest(method="prompts/list")
        )

        if response.is_error:
            return

        result = response.result or {}
        prompts_data = result.get("prompts", [])

        self._prompts = {
            prompt["name"]: MCPPrompt.from_dict(prompt)
            for prompt in prompts_data
        }

    def list_tools(self) -> List[MCPTool]:
        """List all available tools.

        Returns:
            List of MCPTool objects.
        """
        return list(self._tools.values())

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a tool by name.

        Args:
            name: Tool name.

        Returns:
            MCPTool if found, None otherwise.
        """
        return self._tools.get(name)

    async def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Call a tool on the MCP server.

        Args:
            name: Tool name.
            arguments: Tool arguments.

        Returns:
            Tool execution result.

        Raises:
            MCPError: If tool call fails.
        """
        if not self._initialized:
            raise MCPError("Client not initialized")

        response = await self.transport.send_request(
            JSONRPCRequest(
                method="tools/call",
                params={
                    "name": name,
                    "arguments": arguments or {},
                },
            )
        )

        if response.is_error:
            raise MCPError(
                f"Tool call failed: {response.get_error_message()}",
                code=response.get_error_code(),
            )

        result = response.result or {}

        # Handle isError in tool result
        if result.get("isError", False):
            content = result.get("content", [])
            error_text = ""
            for item in content:
                if item.get("type") == "text":
                    error_text = item.get("text", "")
                    break
            raise MCPError(f"Tool returned error: {error_text}")

        # Extract content from result
        content = result.get("content", [])
        if len(content) == 1:
            item = content[0]
            if item.get("type") == "text":
                return item.get("text", "")
            return item
        return content

    def list_resources(self) -> List[MCPResource]:
        """List all available resources.

        Returns:
            List of MCPResource objects.
        """
        return list(self._resources.values())

    def get_resource(self, uri: str) -> Optional[MCPResource]:
        """Get a resource by URI.

        Args:
            uri: Resource URI.

        Returns:
            MCPResource if found, None otherwise.
        """
        return self._resources.get(uri)

    async def read_resource(self, uri: str) -> Any:
        """Read a resource from the MCP server.

        Args:
            uri: Resource URI.

        Returns:
            Resource content.

        Raises:
            MCPError: If resource read fails.
        """
        if not self._initialized:
            raise MCPError("Client not initialized")

        response = await self.transport.send_request(
            JSONRPCRequest(
                method="resources/read",
                params={"uri": uri},
            )
        )

        if response.is_error:
            raise MCPError(
                f"Resource read failed: {response.get_error_message()}",
                code=response.get_error_code(),
            )

        result = response.result or {}
        contents = result.get("contents", [])

        if len(contents) == 1:
            content = contents[0]
            if "text" in content:
                return content["text"]
            elif "blob" in content:
                return content["blob"]
        return contents

    def list_prompts(self) -> List[MCPPrompt]:
        """List all available prompts.

        Returns:
            List of MCPPrompt objects.
        """
        return list(self._prompts.values())

    def get_prompt(self, name: str) -> Optional[MCPPrompt]:
        """Get a prompt by name.

        Args:
            name: Prompt name.

        Returns:
            MCPPrompt if found, None otherwise.
        """
        return self._prompts.get(name)

    async def get_prompt_messages(
        self,
        name: str,
        arguments: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get prompt messages from the server.

        Args:
            name: Prompt name.
            arguments: Prompt arguments.

        Returns:
            List of message dictionaries.

        Raises:
            MCPError: If prompt retrieval fails.
        """
        if not self._initialized:
            raise MCPError("Client not initialized")

        response = await self.transport.send_request(
            JSONRPCRequest(
                method="prompts/get",
                params={
                    "name": name,
                    "arguments": arguments or {},
                },
            )
        )

        if response.is_error:
            raise MCPError(
                f"Prompt get failed: {response.get_error_message()}",
                code=response.get_error_code(),
            )

        result = response.result or {}
        return result.get("messages", [])


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection.

    Attributes:
        name: Unique name for this server.
        transport_type: Type of transport to use.
        transport_config: Transport-specific configuration.
    """

    name: str
    transport_type: TransportType
    transport_config: Union[StdioTransportConfig, SSETransportConfig]


class MCPConnectionManager:
    """Manages connections to multiple MCP servers.

    Provides a unified interface for:
    - Connecting to multiple MCP servers
    - Aggregating tools from all servers
    - Routing tool calls to the appropriate server
    """

    def __init__(self, client_config: Optional[MCPClientConfig] = None):
        """Initialize the connection manager.

        Args:
            client_config: Optional client configuration.
        """
        self.client_config = client_config or MCPClientConfig()
        self._clients: Dict[str, MCPClient] = {}
        self._tool_to_server: Dict[str, str] = {}

    async def add_server(self, config: MCPServerConfig) -> MCPClient:
        """Add and connect to an MCP server.

        Args:
            config: Server configuration.

        Returns:
            The connected MCPClient.
        """
        transport = create_transport(
            config.transport_type,
            config.transport_config,
        )

        client = MCPClient(transport, self.client_config)
        await client.connect()

        self._clients[config.name] = client

        # Map tools to this server
        for tool in client.list_tools():
            # Use prefixed name to avoid collisions
            full_name = f"{config.name}/{tool.name}"
            self._tool_to_server[full_name] = config.name
            # Also map unprefixed name if unique
            if tool.name not in self._tool_to_server:
                self._tool_to_server[tool.name] = config.name

        return client

    async def remove_server(self, name: str) -> None:
        """Remove and disconnect from an MCP server.

        Args:
            name: Server name.
        """
        client = self._clients.pop(name, None)
        if client:
            # Remove tool mappings
            tools_to_remove = [
                tool_name for tool_name, server_name in self._tool_to_server.items()
                if server_name == name
            ]
            for tool_name in tools_to_remove:
                del self._tool_to_server[tool_name]

            await client.disconnect()

    def get_client(self, name: str) -> Optional[MCPClient]:
        """Get a client by server name.

        Args:
            name: Server name.

        Returns:
            MCPClient if found, None otherwise.
        """
        return self._clients.get(name)

    def list_all_tools(self) -> List[MCPTool]:
        """List all tools from all connected servers.

        Returns:
            Combined list of tools.
        """
        tools = []
        for client in self._clients.values():
            tools.extend(client.list_tools())
        return tools

    def get_tool_server(self, tool_name: str) -> Optional[str]:
        """Get the server name for a tool.

        Args:
            tool_name: Tool name (may be prefixed or unprefixed).

        Returns:
            Server name if found, None otherwise.
        """
        return self._tool_to_server.get(tool_name)

    async def call_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Call a tool, routing to the appropriate server.

        Args:
            tool_name: Tool name.
            arguments: Tool arguments.

        Returns:
            Tool execution result.

        Raises:
            MCPError: If tool not found or call fails.
        """
        # Check for prefixed name first
        server_name = self._tool_to_server.get(tool_name)

        if server_name is None:
            raise MCPError(f"Tool '{tool_name}' not found")

        client = self._clients.get(server_name)
        if client is None:
            raise MCPError(f"Server '{server_name}' not connected")

        # Extract actual tool name if prefixed
        if "/" in tool_name:
            actual_name = tool_name.split("/", 1)[1]
        else:
            actual_name = tool_name

        return await client.call_tool(actual_name, arguments)

    async def close_all(self) -> None:
        """Close all server connections."""
        for name in list(self._clients.keys()):
            await self.remove_server(name)

    def __len__(self) -> int:
        """Return number of connected servers."""
        return len(self._clients)

    async def __aenter__(self) -> "MCPConnectionManager":
        """Enter async context."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        """Exit async context, closing all connections."""
        await self.close_all()
