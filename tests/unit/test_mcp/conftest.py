"""Local fixtures for MCP module tests."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents_framework.mcp.transport import (
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    SSETransport,
    SSETransportConfig,
    StdioTransport,
    StdioTransportConfig,
    Transport,
    TransportType,
)
from agents_framework.mcp.client import (
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
from agents_framework.mcp.tools import (
    MCPToolAdapter,
    MCPToolRegistry,
)


# ============================================================================
# Transport Fixtures
# ============================================================================


@pytest.fixture
def stdio_transport_config() -> StdioTransportConfig:
    """Create a sample stdio transport configuration."""
    return StdioTransportConfig(
        command="python",
        args=["-m", "mcp_server"],
        env={"TEST_VAR": "test_value"},
        cwd="/tmp",
        timeout=10.0,
    )


@pytest.fixture
def sse_transport_config() -> SSETransportConfig:
    """Create a sample SSE transport configuration."""
    return SSETransportConfig(
        url="http://localhost:8080/mcp",
        headers={"Authorization": "Bearer test-token"},
        timeout=15.0,
        retry_interval=3.0,
    )


@pytest.fixture
def sample_jsonrpc_request() -> JSONRPCRequest:
    """Create a sample JSON-RPC request."""
    return JSONRPCRequest(
        method="tools/list",
        params={"filter": "all"},
        id=1,
    )


@pytest.fixture
def sample_jsonrpc_response() -> JSONRPCResponse:
    """Create a sample JSON-RPC response."""
    return JSONRPCResponse(
        id=1,
        result={"tools": [{"name": "test_tool"}]},
    )


@pytest.fixture
def sample_jsonrpc_error_response() -> JSONRPCResponse:
    """Create a sample JSON-RPC error response."""
    return JSONRPCResponse(
        id=1,
        error={"code": -32600, "message": "Invalid Request"},
    )


@pytest.fixture
def sample_jsonrpc_notification() -> JSONRPCNotification:
    """Create a sample JSON-RPC notification."""
    return JSONRPCNotification(
        method="notifications/progress",
        params={"progress": 50, "total": 100},
    )


# ============================================================================
# Client Fixtures
# ============================================================================


@pytest.fixture
def mcp_client_config() -> MCPClientConfig:
    """Create a sample MCP client configuration."""
    return MCPClientConfig(
        name="test-agent",
        version="1.0.0",
        capabilities=[MCPCapability.TOOLS, MCPCapability.RESOURCES],
    )


@pytest.fixture
def sample_mcp_tool() -> MCPTool:
    """Create a sample MCP tool."""
    return MCPTool(
        name="read_file",
        description="Read contents of a file",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
            },
            "required": ["path"],
        },
    )


@pytest.fixture
def sample_mcp_tools() -> List[MCPTool]:
    """Create multiple sample MCP tools."""
    return [
        MCPTool(
            name="read_file",
            description="Read contents of a file",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        ),
        MCPTool(
            name="write_file",
            description="Write contents to a file",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        ),
        MCPTool(
            name="list_directory",
            description="List directory contents",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        ),
    ]


@pytest.fixture
def sample_mcp_resource() -> MCPResource:
    """Create a sample MCP resource."""
    return MCPResource(
        uri="file:///tmp/test.txt",
        name="test.txt",
        description="Test file resource",
        mime_type="text/plain",
    )


@pytest.fixture
def sample_mcp_prompt() -> MCPPrompt:
    """Create a sample MCP prompt."""
    return MCPPrompt(
        name="code_review",
        description="Code review prompt template",
        arguments=[
            {"name": "code", "description": "The code to review", "required": True},
            {"name": "language", "description": "Programming language", "required": False},
        ],
    )


@pytest.fixture
def sample_server_info() -> MCPServerInfo:
    """Create a sample MCP server info."""
    return MCPServerInfo(
        name="test-mcp-server",
        version="1.0.0",
        capabilities=[MCPCapability.TOOLS, MCPCapability.RESOURCES, MCPCapability.PROMPTS],
    )


# ============================================================================
# Mock Transport
# ============================================================================


class MockTransport(Transport):
    """Mock transport for testing MCP client without actual connections."""

    def __init__(
        self,
        responses: Optional[Dict[str, JSONRPCResponse]] = None,
        raise_on_connect: bool = False,
        raise_on_send: bool = False,
    ):
        """Initialize mock transport.

        Args:
            responses: Dict mapping method names to responses.
            raise_on_connect: Whether to raise an error on connect.
            raise_on_send: Whether to raise an error on send.
        """
        self.responses = responses or {}
        self.raise_on_connect = raise_on_connect
        self.raise_on_send = raise_on_send
        self._connected = False
        self.requests_sent: List[JSONRPCRequest] = []
        self.notifications_sent: List[JSONRPCNotification] = []
        self._notification_queue: asyncio.Queue[JSONRPCNotification] = asyncio.Queue()

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        if self.raise_on_connect:
            from agents_framework.mcp.transport import ConnectionError
            raise ConnectionError("Mock connection error")
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def send_request(self, request: JSONRPCRequest) -> JSONRPCResponse:
        from agents_framework.mcp.transport import TransportError

        if self.raise_on_send:
            raise TransportError("Mock send error")

        self.requests_sent.append(request)

        # Return matching response or default success
        if request.method in self.responses:
            return self.responses[request.method]

        return JSONRPCResponse(id=request.id, result={})

    async def send_notification(self, notification: JSONRPCNotification) -> None:
        from agents_framework.mcp.transport import TransportError

        if self.raise_on_send:
            raise TransportError("Mock send error")

        self.notifications_sent.append(notification)

    async def receive_notifications(self) -> AsyncIterator[JSONRPCNotification]:
        while self._connected:
            try:
                notification = await asyncio.wait_for(
                    self._notification_queue.get(),
                    timeout=0.1,
                )
                yield notification
            except asyncio.TimeoutError:
                break

    def add_notification(self, notification: JSONRPCNotification) -> None:
        """Add a notification to be received."""
        self._notification_queue.put_nowait(notification)


@pytest.fixture
def mock_transport() -> MockTransport:
    """Create a basic mock transport."""
    return MockTransport()


@pytest.fixture
def mock_transport_with_tools(sample_mcp_tools: List[MCPTool]) -> MockTransport:
    """Create a mock transport that responds with tools."""
    # Build the initialize response
    init_response = JSONRPCResponse(
        id=1,
        result={
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": "test-server",
                "version": "1.0.0",
            },
            "capabilities": {
                "tools": {},
            },
        },
    )

    # Build the tools/list response
    tools_response = JSONRPCResponse(
        id=2,
        result={
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema,
                }
                for tool in sample_mcp_tools
            ]
        },
    )

    return MockTransport(
        responses={
            "initialize": init_response,
            "tools/list": tools_response,
        }
    )


@pytest.fixture
def mock_transport_full_capabilities(
    sample_mcp_tools: List[MCPTool],
    sample_mcp_resource: MCPResource,
    sample_mcp_prompt: MCPPrompt,
) -> MockTransport:
    """Create a mock transport with full MCP capabilities."""
    init_response = JSONRPCResponse(
        id=1,
        result={
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": "test-server",
                "version": "1.0.0",
            },
            "capabilities": {
                "tools": {},
                "resources": {},
                "prompts": {},
            },
        },
    )

    tools_response = JSONRPCResponse(
        id=2,
        result={
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema,
                }
                for tool in sample_mcp_tools
            ]
        },
    )

    resources_response = JSONRPCResponse(
        id=3,
        result={
            "resources": [
                {
                    "uri": sample_mcp_resource.uri,
                    "name": sample_mcp_resource.name,
                    "description": sample_mcp_resource.description,
                    "mimeType": sample_mcp_resource.mime_type,
                }
            ]
        },
    )

    prompts_response = JSONRPCResponse(
        id=4,
        result={
            "prompts": [
                {
                    "name": sample_mcp_prompt.name,
                    "description": sample_mcp_prompt.description,
                    "arguments": sample_mcp_prompt.arguments,
                }
            ]
        },
    )

    return MockTransport(
        responses={
            "initialize": init_response,
            "tools/list": tools_response,
            "resources/list": resources_response,
            "prompts/list": prompts_response,
        }
    )


# ============================================================================
# MCP Client Fixtures
# ============================================================================


@pytest.fixture
def mcp_client(mock_transport: MockTransport, mcp_client_config: MCPClientConfig) -> MCPClient:
    """Create an MCP client with mock transport (not connected)."""
    return MCPClient(mock_transport, mcp_client_config)


@pytest.fixture
async def connected_mcp_client(
    mock_transport_with_tools: MockTransport,
    mcp_client_config: MCPClientConfig,
) -> MCPClient:
    """Create a connected MCP client with mock transport."""
    client = MCPClient(mock_transport_with_tools, mcp_client_config)
    await client.connect()
    return client


# ============================================================================
# MCP Server Config Fixtures
# ============================================================================


@pytest.fixture
def mcp_server_config_stdio(stdio_transport_config: StdioTransportConfig) -> MCPServerConfig:
    """Create a sample MCP server config with stdio transport."""
    return MCPServerConfig(
        name="test-stdio-server",
        transport_type=TransportType.STDIO,
        transport_config=stdio_transport_config,
    )


@pytest.fixture
def mcp_server_config_sse(sse_transport_config: SSETransportConfig) -> MCPServerConfig:
    """Create a sample MCP server config with SSE transport."""
    return MCPServerConfig(
        name="test-sse-server",
        transport_type=TransportType.SSE,
        transport_config=sse_transport_config,
    )
