"""MCP (Model Context Protocol) integration for the agents framework.

This module provides MCP client functionality for connecting to
MCP servers and using their tools, resources, and prompts.

Example:
    # Connect to a single MCP server via stdio
    from agents_framework.mcp import (
        MCPClient,
        StdioTransport,
        StdioTransportConfig,
    )

    config = StdioTransportConfig(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem"],
    )
    transport = StdioTransport(config)
    client = MCPClient(transport)

    await client.connect()
    tools = client.list_tools()
    result = await client.call_tool("read_file", {"path": "/tmp/test.txt"})
    await client.disconnect()

    # Connect to multiple servers
    from agents_framework.mcp import MCPConnectionManager, MCPServerConfig

    async with MCPConnectionManager() as manager:
        await manager.add_server(MCPServerConfig(
            name="filesystem",
            transport_type=TransportType.STDIO,
            transport_config=StdioTransportConfig(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem"],
            ),
        ))
        result = await manager.call_tool("read_file", {"path": "/tmp/test.txt"})
"""

from .transport import (
    ConnectionError,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPError,
    SSETransport,
    SSETransportConfig,
    StdioTransport,
    StdioTransportConfig,
    Transport,
    TransportError,
    TransportType,
    create_transport,
)
from .client import (
    MCPCapability,
    MCPClient,
    MCPClientConfig,
    MCPConnectionManager,
    MCPPrompt,
    MCPResource,
    MCPServerConfig,
    MCPServerInfo,
    MCPTool,
)
from .tools import (
    MCPToolAdapter,
    MCPToolRegistry,
    MCPToolsConfig,
    create_mcp_tools_from_client,
    register_mcp_tools,
)

__all__ = [
    # Transport layer
    "ConnectionError",
    "JSONRPCNotification",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "MCPError",
    "SSETransport",
    "SSETransportConfig",
    "StdioTransport",
    "StdioTransportConfig",
    "Transport",
    "TransportError",
    "TransportType",
    "create_transport",
    # Client
    "MCPCapability",
    "MCPClient",
    "MCPClientConfig",
    "MCPConnectionManager",
    "MCPPrompt",
    "MCPResource",
    "MCPServerConfig",
    "MCPServerInfo",
    "MCPTool",
    # Tools integration
    "MCPToolAdapter",
    "MCPToolRegistry",
    "MCPToolsConfig",
    "create_mcp_tools_from_client",
    "register_mcp_tools",
]
