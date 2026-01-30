"""Unit tests for MCP client core class.

Tests cover:
- MCPClient initialization and configuration
- Connection lifecycle (connect, disconnect)
- Server capability discovery
- Tool discovery and invocation
- Resource discovery and reading
- Prompt discovery and retrieval
- MCPConnectionManager for multi-server management
- Error handling
"""

from __future__ import annotations

import asyncio
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents_framework.mcp.transport import (
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPError,
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
from agents_framework.mcp.transport import (
    StdioTransportConfig,
    SSETransportConfig,
    TransportType,
)

from .conftest import MockTransport


# ============================================================================
# MCPCapability Tests
# ============================================================================


class TestMCPCapability:
    """Tests for MCPCapability enum."""

    def test_capability_values(self):
        """Test capability enum values."""
        assert MCPCapability.TOOLS.value == "tools"
        assert MCPCapability.RESOURCES.value == "resources"
        assert MCPCapability.PROMPTS.value == "prompts"
        assert MCPCapability.LOGGING.value == "logging"
        assert MCPCapability.SAMPLING.value == "sampling"

    def test_capability_is_string_enum(self):
        """Test capability is a string enum."""
        assert isinstance(MCPCapability.TOOLS, str)
        assert MCPCapability.TOOLS == "tools"


# ============================================================================
# MCPTool Tests
# ============================================================================


class TestMCPTool:
    """Tests for MCPTool dataclass."""

    def test_tool_basic(self, sample_mcp_tool: MCPTool):
        """Test basic tool creation."""
        assert sample_mcp_tool.name == "read_file"
        assert sample_mcp_tool.description == "Read contents of a file"
        assert "path" in sample_mcp_tool.input_schema["properties"]

    def test_tool_from_dict(self):
        """Test creating tool from dictionary."""
        data = {
            "name": "write_file",
            "description": "Write to file",
            "inputSchema": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
            },
        }
        tool = MCPTool.from_dict(data)
        assert tool.name == "write_file"
        assert tool.description == "Write to file"
        assert tool.input_schema["properties"]["content"]["type"] == "string"

    def test_tool_from_dict_missing_fields(self):
        """Test creating tool with missing fields."""
        data = {}
        tool = MCPTool.from_dict(data)
        assert tool.name == ""
        assert tool.description == ""
        assert tool.input_schema == {}


# ============================================================================
# MCPResource Tests
# ============================================================================


class TestMCPResource:
    """Tests for MCPResource dataclass."""

    def test_resource_basic(self, sample_mcp_resource: MCPResource):
        """Test basic resource creation."""
        assert sample_mcp_resource.uri == "file:///tmp/test.txt"
        assert sample_mcp_resource.name == "test.txt"
        assert sample_mcp_resource.description == "Test file resource"
        assert sample_mcp_resource.mime_type == "text/plain"

    def test_resource_from_dict(self):
        """Test creating resource from dictionary."""
        data = {
            "uri": "http://example.com/data",
            "name": "data.json",
            "description": "Data file",
            "mimeType": "application/json",
        }
        resource = MCPResource.from_dict(data)
        assert resource.uri == "http://example.com/data"
        assert resource.name == "data.json"
        assert resource.mime_type == "application/json"

    def test_resource_from_dict_minimal(self):
        """Test creating resource with minimal fields."""
        data = {"uri": "file:///test", "name": "test"}
        resource = MCPResource.from_dict(data)
        assert resource.uri == "file:///test"
        assert resource.description == ""
        assert resource.mime_type is None


# ============================================================================
# MCPPrompt Tests
# ============================================================================


class TestMCPPrompt:
    """Tests for MCPPrompt dataclass."""

    def test_prompt_basic(self, sample_mcp_prompt: MCPPrompt):
        """Test basic prompt creation."""
        assert sample_mcp_prompt.name == "code_review"
        assert sample_mcp_prompt.description == "Code review prompt template"
        assert len(sample_mcp_prompt.arguments) == 2

    def test_prompt_from_dict(self):
        """Test creating prompt from dictionary."""
        data = {
            "name": "summarize",
            "description": "Summarize text",
            "arguments": [
                {"name": "text", "required": True},
            ],
        }
        prompt = MCPPrompt.from_dict(data)
        assert prompt.name == "summarize"
        assert len(prompt.arguments) == 1

    def test_prompt_from_dict_minimal(self):
        """Test creating prompt with minimal fields."""
        data = {"name": "simple"}
        prompt = MCPPrompt.from_dict(data)
        assert prompt.name == "simple"
        assert prompt.description == ""
        assert prompt.arguments == []


# ============================================================================
# MCPServerInfo Tests
# ============================================================================


class TestMCPServerInfo:
    """Tests for MCPServerInfo dataclass."""

    def test_server_info_basic(self, sample_server_info: MCPServerInfo):
        """Test basic server info creation."""
        assert sample_server_info.name == "test-mcp-server"
        assert sample_server_info.version == "1.0.0"
        assert MCPCapability.TOOLS in sample_server_info.capabilities

    def test_server_info_defaults(self):
        """Test server info with defaults."""
        info = MCPServerInfo(name="test", version="1.0")
        assert info.capabilities == []


# ============================================================================
# MCPClientConfig Tests
# ============================================================================


class TestMCPClientConfig:
    """Tests for MCPClientConfig dataclass."""

    def test_config_basic(self, mcp_client_config: MCPClientConfig):
        """Test basic config creation."""
        assert mcp_client_config.name == "test-agent"
        assert mcp_client_config.version == "1.0.0"
        assert MCPCapability.TOOLS in mcp_client_config.capabilities

    def test_config_defaults(self):
        """Test config with defaults."""
        config = MCPClientConfig()
        assert config.name == "agents_framework"
        assert config.version == "0.1.0"
        assert config.capabilities == []


# ============================================================================
# MCPClient Initialization Tests
# ============================================================================


class TestMCPClientInit:
    """Tests for MCPClient initialization."""

    def test_client_initialization(
        self, mock_transport: MockTransport, mcp_client_config: MCPClientConfig
    ):
        """Test client initialization."""
        client = MCPClient(mock_transport, mcp_client_config)
        assert client.transport == mock_transport
        assert client.config == mcp_client_config
        assert client.is_connected is False
        assert client.is_initialized is False
        assert client.server_info is None

    def test_client_initialization_default_config(self, mock_transport: MockTransport):
        """Test client initialization with default config."""
        client = MCPClient(mock_transport)
        assert client.config.name == "agents_framework"


# ============================================================================
# MCPClient Connection Tests
# ============================================================================


class TestMCPClientConnection:
    """Tests for MCPClient connection lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_success(
        self, mock_transport_with_tools: MockTransport, mcp_client_config: MCPClientConfig
    ):
        """Test successful connection."""
        client = MCPClient(mock_transport_with_tools, mcp_client_config)
        await client.connect()

        assert client.is_connected is True
        assert client.is_initialized is True
        assert client.server_info is not None
        assert client.server_info.name == "test-server"

    @pytest.mark.asyncio
    async def test_connect_sends_initialize(
        self, mock_transport_with_tools: MockTransport, mcp_client_config: MCPClientConfig
    ):
        """Test that connect sends initialize request."""
        client = MCPClient(mock_transport_with_tools, mcp_client_config)
        await client.connect()

        # Check initialize request was sent
        init_requests = [
            r for r in mock_transport_with_tools.requests_sent if r.method == "initialize"
        ]
        assert len(init_requests) == 1
        assert init_requests[0].params["clientInfo"]["name"] == "test-agent"

    @pytest.mark.asyncio
    async def test_connect_sends_initialized_notification(
        self, mock_transport_with_tools: MockTransport, mcp_client_config: MCPClientConfig
    ):
        """Test that connect sends initialized notification."""
        client = MCPClient(mock_transport_with_tools, mcp_client_config)
        await client.connect()

        # Check initialized notification was sent
        init_notifications = [
            n
            for n in mock_transport_with_tools.notifications_sent
            if n.method == "notifications/initialized"
        ]
        assert len(init_notifications) == 1

    @pytest.mark.asyncio
    async def test_connect_failure(self, mcp_client_config: MCPClientConfig):
        """Test connection failure."""
        transport = MockTransport(raise_on_connect=True)
        client = MCPClient(transport, mcp_client_config)

        from agents_framework.mcp.transport import ConnectionError

        with pytest.raises(ConnectionError):
            await client.connect()

    @pytest.mark.asyncio
    async def test_connect_initialize_error(self, mcp_client_config: MCPClientConfig):
        """Test initialization error during connect."""
        error_response = JSONRPCResponse(
            id=1,
            error={"code": -32600, "message": "Protocol version not supported"},
        )
        transport = MockTransport(responses={"initialize": error_response})
        client = MCPClient(transport, mcp_client_config)

        with pytest.raises(MCPError, match="Initialize failed"):
            await client.connect()

    @pytest.mark.asyncio
    async def test_disconnect(
        self, mock_transport_with_tools: MockTransport, mcp_client_config: MCPClientConfig
    ):
        """Test disconnection."""
        client = MCPClient(mock_transport_with_tools, mcp_client_config)
        await client.connect()
        await client.disconnect()

        assert client.is_connected is False
        assert client.is_initialized is False
        assert client.server_info is None
        assert len(client.list_tools()) == 0


# ============================================================================
# MCPClient Tool Discovery Tests
# ============================================================================


class TestMCPClientToolDiscovery:
    """Tests for MCPClient tool discovery."""

    @pytest.mark.asyncio
    async def test_list_tools(
        self,
        mock_transport_with_tools: MockTransport,
        mcp_client_config: MCPClientConfig,
        sample_mcp_tools: List[MCPTool],
    ):
        """Test listing tools after connection."""
        client = MCPClient(mock_transport_with_tools, mcp_client_config)
        await client.connect()

        tools = client.list_tools()
        assert len(tools) == len(sample_mcp_tools)

    @pytest.mark.asyncio
    async def test_get_tool(
        self, mock_transport_with_tools: MockTransport, mcp_client_config: MCPClientConfig
    ):
        """Test getting a specific tool."""
        client = MCPClient(mock_transport_with_tools, mcp_client_config)
        await client.connect()

        tool = client.get_tool("read_file")
        assert tool is not None
        assert tool.name == "read_file"

    @pytest.mark.asyncio
    async def test_get_tool_not_found(
        self, mock_transport_with_tools: MockTransport, mcp_client_config: MCPClientConfig
    ):
        """Test getting a non-existent tool."""
        client = MCPClient(mock_transport_with_tools, mcp_client_config)
        await client.connect()

        tool = client.get_tool("nonexistent")
        assert tool is None


# ============================================================================
# MCPClient Tool Invocation Tests
# ============================================================================


class TestMCPClientToolInvocation:
    """Tests for MCPClient tool invocation."""

    @pytest.mark.asyncio
    async def test_call_tool_success(
        self, mock_transport_with_tools: MockTransport, mcp_client_config: MCPClientConfig
    ):
        """Test successful tool call."""
        # Add tool call response
        mock_transport_with_tools.responses["tools/call"] = JSONRPCResponse(
            id=3,
            result={
                "content": [{"type": "text", "text": "File contents here"}],
            },
        )

        client = MCPClient(mock_transport_with_tools, mcp_client_config)
        await client.connect()

        result = await client.call_tool("read_file", {"path": "/tmp/test.txt"})
        assert result == "File contents here"

    @pytest.mark.asyncio
    async def test_call_tool_not_initialized(
        self, mock_transport: MockTransport, mcp_client_config: MCPClientConfig
    ):
        """Test tool call when not initialized."""
        client = MCPClient(mock_transport, mcp_client_config)

        with pytest.raises(MCPError, match="Client not initialized"):
            await client.call_tool("read_file", {"path": "/test"})

    @pytest.mark.asyncio
    async def test_call_tool_error_response(
        self, mock_transport_with_tools: MockTransport, mcp_client_config: MCPClientConfig
    ):
        """Test tool call with error response."""
        mock_transport_with_tools.responses["tools/call"] = JSONRPCResponse(
            id=3,
            error={"code": -32000, "message": "Tool execution failed"},
        )

        client = MCPClient(mock_transport_with_tools, mcp_client_config)
        await client.connect()

        with pytest.raises(MCPError, match="Tool call failed"):
            await client.call_tool("read_file", {"path": "/test"})

    @pytest.mark.asyncio
    async def test_call_tool_is_error_in_result(
        self, mock_transport_with_tools: MockTransport, mcp_client_config: MCPClientConfig
    ):
        """Test tool call with isError in result."""
        mock_transport_with_tools.responses["tools/call"] = JSONRPCResponse(
            id=3,
            result={
                "isError": True,
                "content": [{"type": "text", "text": "File not found"}],
            },
        )

        client = MCPClient(mock_transport_with_tools, mcp_client_config)
        await client.connect()

        with pytest.raises(MCPError, match="Tool returned error"):
            await client.call_tool("read_file", {"path": "/nonexistent"})

    @pytest.mark.asyncio
    async def test_call_tool_multiple_content_items(
        self, mock_transport_with_tools: MockTransport, mcp_client_config: MCPClientConfig
    ):
        """Test tool call with multiple content items."""
        mock_transport_with_tools.responses["tools/call"] = JSONRPCResponse(
            id=3,
            result={
                "content": [
                    {"type": "text", "text": "Part 1"},
                    {"type": "text", "text": "Part 2"},
                ],
            },
        )

        client = MCPClient(mock_transport_with_tools, mcp_client_config)
        await client.connect()

        result = await client.call_tool("test", {})
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_call_tool_non_text_content(
        self, mock_transport_with_tools: MockTransport, mcp_client_config: MCPClientConfig
    ):
        """Test tool call with non-text content."""
        mock_transport_with_tools.responses["tools/call"] = JSONRPCResponse(
            id=3,
            result={
                "content": [{"type": "image", "data": "base64data"}],
            },
        )

        client = MCPClient(mock_transport_with_tools, mcp_client_config)
        await client.connect()

        result = await client.call_tool("screenshot", {})
        assert result["type"] == "image"


# ============================================================================
# MCPClient Resource Tests
# ============================================================================


class TestMCPClientResources:
    """Tests for MCPClient resource operations."""

    @pytest.mark.asyncio
    async def test_list_resources(
        self,
        mock_transport_full_capabilities: MockTransport,
        mcp_client_config: MCPClientConfig,
    ):
        """Test listing resources."""
        client = MCPClient(mock_transport_full_capabilities, mcp_client_config)
        await client.connect()

        resources = client.list_resources()
        assert len(resources) == 1
        assert resources[0].name == "test.txt"

    @pytest.mark.asyncio
    async def test_get_resource(
        self,
        mock_transport_full_capabilities: MockTransport,
        mcp_client_config: MCPClientConfig,
    ):
        """Test getting a specific resource."""
        client = MCPClient(mock_transport_full_capabilities, mcp_client_config)
        await client.connect()

        resource = client.get_resource("file:///tmp/test.txt")
        assert resource is not None
        assert resource.name == "test.txt"

    @pytest.mark.asyncio
    async def test_read_resource_text(
        self,
        mock_transport_full_capabilities: MockTransport,
        mcp_client_config: MCPClientConfig,
    ):
        """Test reading text resource."""
        mock_transport_full_capabilities.responses["resources/read"] = JSONRPCResponse(
            id=5,
            result={
                "contents": [{"uri": "file:///tmp/test.txt", "text": "Hello World"}],
            },
        )

        client = MCPClient(mock_transport_full_capabilities, mcp_client_config)
        await client.connect()

        content = await client.read_resource("file:///tmp/test.txt")
        assert content == "Hello World"

    @pytest.mark.asyncio
    async def test_read_resource_blob(
        self,
        mock_transport_full_capabilities: MockTransport,
        mcp_client_config: MCPClientConfig,
    ):
        """Test reading binary resource."""
        mock_transport_full_capabilities.responses["resources/read"] = JSONRPCResponse(
            id=5,
            result={
                "contents": [{"uri": "file:///tmp/image.png", "blob": "base64data"}],
            },
        )

        client = MCPClient(mock_transport_full_capabilities, mcp_client_config)
        await client.connect()

        content = await client.read_resource("file:///tmp/image.png")
        assert content == "base64data"

    @pytest.mark.asyncio
    async def test_read_resource_not_initialized(
        self, mock_transport: MockTransport, mcp_client_config: MCPClientConfig
    ):
        """Test reading resource when not initialized."""
        client = MCPClient(mock_transport, mcp_client_config)

        with pytest.raises(MCPError, match="Client not initialized"):
            await client.read_resource("file:///test")

    @pytest.mark.asyncio
    async def test_read_resource_error(
        self,
        mock_transport_full_capabilities: MockTransport,
        mcp_client_config: MCPClientConfig,
    ):
        """Test reading resource with error."""
        mock_transport_full_capabilities.responses["resources/read"] = JSONRPCResponse(
            id=5,
            error={"code": -32000, "message": "Resource not found"},
        )

        client = MCPClient(mock_transport_full_capabilities, mcp_client_config)
        await client.connect()

        with pytest.raises(MCPError, match="Resource read failed"):
            await client.read_resource("file:///nonexistent")


# ============================================================================
# MCPClient Prompt Tests
# ============================================================================


class TestMCPClientPrompts:
    """Tests for MCPClient prompt operations."""

    @pytest.mark.asyncio
    async def test_list_prompts(
        self,
        mock_transport_full_capabilities: MockTransport,
        mcp_client_config: MCPClientConfig,
    ):
        """Test listing prompts."""
        client = MCPClient(mock_transport_full_capabilities, mcp_client_config)
        await client.connect()

        prompts = client.list_prompts()
        assert len(prompts) == 1
        assert prompts[0].name == "code_review"

    @pytest.mark.asyncio
    async def test_get_prompt(
        self,
        mock_transport_full_capabilities: MockTransport,
        mcp_client_config: MCPClientConfig,
    ):
        """Test getting a specific prompt."""
        client = MCPClient(mock_transport_full_capabilities, mcp_client_config)
        await client.connect()

        prompt = client.get_prompt("code_review")
        assert prompt is not None
        assert prompt.description == "Code review prompt template"

    @pytest.mark.asyncio
    async def test_get_prompt_messages(
        self,
        mock_transport_full_capabilities: MockTransport,
        mcp_client_config: MCPClientConfig,
    ):
        """Test getting prompt messages."""
        mock_transport_full_capabilities.responses["prompts/get"] = JSONRPCResponse(
            id=5,
            result={
                "messages": [
                    {"role": "user", "content": {"type": "text", "text": "Review: code"}},
                ],
            },
        )

        client = MCPClient(mock_transport_full_capabilities, mcp_client_config)
        await client.connect()

        messages = await client.get_prompt_messages("code_review", {"code": "test"})
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_get_prompt_messages_not_initialized(
        self, mock_transport: MockTransport, mcp_client_config: MCPClientConfig
    ):
        """Test getting prompt messages when not initialized."""
        client = MCPClient(mock_transport, mcp_client_config)

        with pytest.raises(MCPError, match="Client not initialized"):
            await client.get_prompt_messages("test", {})

    @pytest.mark.asyncio
    async def test_get_prompt_messages_error(
        self,
        mock_transport_full_capabilities: MockTransport,
        mcp_client_config: MCPClientConfig,
    ):
        """Test getting prompt messages with error."""
        mock_transport_full_capabilities.responses["prompts/get"] = JSONRPCResponse(
            id=5,
            error={"code": -32000, "message": "Prompt not found"},
        )

        client = MCPClient(mock_transport_full_capabilities, mcp_client_config)
        await client.connect()

        with pytest.raises(MCPError, match="Prompt get failed"):
            await client.get_prompt_messages("nonexistent", {})


# ============================================================================
# MCPServerConfig Tests
# ============================================================================


class TestMCPServerConfig:
    """Tests for MCPServerConfig dataclass."""

    def test_server_config_stdio(
        self, mcp_server_config_stdio: MCPServerConfig, stdio_transport_config: StdioTransportConfig
    ):
        """Test server config with stdio transport."""
        assert mcp_server_config_stdio.name == "test-stdio-server"
        assert mcp_server_config_stdio.transport_type == TransportType.STDIO
        assert mcp_server_config_stdio.transport_config == stdio_transport_config

    def test_server_config_sse(
        self, mcp_server_config_sse: MCPServerConfig, sse_transport_config: SSETransportConfig
    ):
        """Test server config with SSE transport."""
        assert mcp_server_config_sse.name == "test-sse-server"
        assert mcp_server_config_sse.transport_type == TransportType.SSE


# ============================================================================
# MCPConnectionManager Tests
# ============================================================================


class TestMCPConnectionManager:
    """Tests for MCPConnectionManager class."""

    def test_manager_initialization(self, mcp_client_config: MCPClientConfig):
        """Test manager initialization."""
        manager = MCPConnectionManager(mcp_client_config)
        assert manager.client_config == mcp_client_config
        assert len(manager) == 0

    def test_manager_initialization_default_config(self):
        """Test manager initialization with default config."""
        manager = MCPConnectionManager()
        assert manager.client_config.name == "agents_framework"

    @pytest.mark.asyncio
    async def test_add_server(self, mcp_client_config: MCPClientConfig):
        """Test adding a server."""
        manager = MCPConnectionManager(mcp_client_config)

        # Create a mock that returns our mock transport
        mock_transport = MockTransport(
            responses={
                "initialize": JSONRPCResponse(
                    id=1,
                    result={
                        "serverInfo": {"name": "test", "version": "1.0"},
                        "capabilities": {"tools": {}},
                    },
                ),
                "tools/list": JSONRPCResponse(
                    id=2,
                    result={"tools": [{"name": "tool1", "description": "Test", "inputSchema": {}}]},
                ),
            }
        )

        with patch(
            "agents_framework.mcp.client.create_transport", return_value=mock_transport
        ):
            config = MCPServerConfig(
                name="test-server",
                transport_type=TransportType.STDIO,
                transport_config=StdioTransportConfig(command="test"),
            )
            client = await manager.add_server(config)

            assert len(manager) == 1
            assert client.is_connected is True

    @pytest.mark.asyncio
    async def test_remove_server(self, mcp_client_config: MCPClientConfig):
        """Test removing a server."""
        manager = MCPConnectionManager(mcp_client_config)

        mock_transport = MockTransport(
            responses={
                "initialize": JSONRPCResponse(
                    id=1,
                    result={
                        "serverInfo": {"name": "test", "version": "1.0"},
                        "capabilities": {"tools": {}},
                    },
                ),
                "tools/list": JSONRPCResponse(id=2, result={"tools": []}),
            }
        )

        with patch(
            "agents_framework.mcp.client.create_transport", return_value=mock_transport
        ):
            config = MCPServerConfig(
                name="test-server",
                transport_type=TransportType.STDIO,
                transport_config=StdioTransportConfig(command="test"),
            )
            await manager.add_server(config)
            await manager.remove_server("test-server")

            assert len(manager) == 0

    @pytest.mark.asyncio
    async def test_get_client(self, mcp_client_config: MCPClientConfig):
        """Test getting a client by name."""
        manager = MCPConnectionManager(mcp_client_config)

        mock_transport = MockTransport(
            responses={
                "initialize": JSONRPCResponse(
                    id=1,
                    result={
                        "serverInfo": {"name": "test", "version": "1.0"},
                        "capabilities": {"tools": {}},
                    },
                ),
                "tools/list": JSONRPCResponse(id=2, result={"tools": []}),
            }
        )

        with patch(
            "agents_framework.mcp.client.create_transport", return_value=mock_transport
        ):
            config = MCPServerConfig(
                name="my-server",
                transport_type=TransportType.STDIO,
                transport_config=StdioTransportConfig(command="test"),
            )
            await manager.add_server(config)

            client = manager.get_client("my-server")
            assert client is not None

            client = manager.get_client("nonexistent")
            assert client is None

    @pytest.mark.asyncio
    async def test_list_all_tools(self, mcp_client_config: MCPClientConfig):
        """Test listing tools from all servers."""
        manager = MCPConnectionManager(mcp_client_config)

        # Add two servers with different tools
        for i, server_name in enumerate(["server1", "server2"]):
            mock_transport = MockTransport(
                responses={
                    "initialize": JSONRPCResponse(
                        id=1,
                        result={
                            "serverInfo": {"name": server_name, "version": "1.0"},
                            "capabilities": {"tools": {}},
                        },
                    ),
                    "tools/list": JSONRPCResponse(
                        id=2,
                        result={
                            "tools": [
                                {
                                    "name": f"tool_{server_name}",
                                    "description": f"Tool from {server_name}",
                                    "inputSchema": {},
                                }
                            ]
                        },
                    ),
                }
            )

            with patch(
                "agents_framework.mcp.client.create_transport", return_value=mock_transport
            ):
                config = MCPServerConfig(
                    name=server_name,
                    transport_type=TransportType.STDIO,
                    transport_config=StdioTransportConfig(command="test"),
                )
                await manager.add_server(config)

        tools = manager.list_all_tools()
        assert len(tools) == 2

    @pytest.mark.asyncio
    async def test_call_tool_routing(self, mcp_client_config: MCPClientConfig):
        """Test that tool calls are routed to correct server."""
        manager = MCPConnectionManager(mcp_client_config)

        mock_transport = MockTransport(
            responses={
                "initialize": JSONRPCResponse(
                    id=1,
                    result={
                        "serverInfo": {"name": "test", "version": "1.0"},
                        "capabilities": {"tools": {}},
                    },
                ),
                "tools/list": JSONRPCResponse(
                    id=2,
                    result={
                        "tools": [{"name": "my_tool", "description": "Test", "inputSchema": {}}]
                    },
                ),
                "tools/call": JSONRPCResponse(
                    id=3,
                    result={"content": [{"type": "text", "text": "result"}]},
                ),
            }
        )

        with patch(
            "agents_framework.mcp.client.create_transport", return_value=mock_transport
        ):
            config = MCPServerConfig(
                name="test-server",
                transport_type=TransportType.STDIO,
                transport_config=StdioTransportConfig(command="test"),
            )
            await manager.add_server(config)

            result = await manager.call_tool("my_tool", {"arg": "value"})
            assert result == "result"

    @pytest.mark.asyncio
    async def test_call_tool_with_prefix(self, mcp_client_config: MCPClientConfig):
        """Test calling tool with server prefix."""
        manager = MCPConnectionManager(mcp_client_config)

        mock_transport = MockTransport(
            responses={
                "initialize": JSONRPCResponse(
                    id=1,
                    result={
                        "serverInfo": {"name": "test", "version": "1.0"},
                        "capabilities": {"tools": {}},
                    },
                ),
                "tools/list": JSONRPCResponse(
                    id=2,
                    result={
                        "tools": [{"name": "my_tool", "description": "Test", "inputSchema": {}}]
                    },
                ),
                "tools/call": JSONRPCResponse(
                    id=3,
                    result={"content": [{"type": "text", "text": "prefixed result"}]},
                ),
            }
        )

        with patch(
            "agents_framework.mcp.client.create_transport", return_value=mock_transport
        ):
            config = MCPServerConfig(
                name="myserver",
                transport_type=TransportType.STDIO,
                transport_config=StdioTransportConfig(command="test"),
            )
            await manager.add_server(config)

            result = await manager.call_tool("myserver/my_tool", {})
            assert result == "prefixed result"

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self, mcp_client_config: MCPClientConfig):
        """Test calling non-existent tool."""
        manager = MCPConnectionManager(mcp_client_config)

        with pytest.raises(MCPError, match="Tool 'nonexistent' not found"):
            await manager.call_tool("nonexistent", {})

    @pytest.mark.asyncio
    async def test_get_tool_server(self, mcp_client_config: MCPClientConfig):
        """Test getting server name for a tool."""
        manager = MCPConnectionManager(mcp_client_config)

        mock_transport = MockTransport(
            responses={
                "initialize": JSONRPCResponse(
                    id=1,
                    result={
                        "serverInfo": {"name": "test", "version": "1.0"},
                        "capabilities": {"tools": {}},
                    },
                ),
                "tools/list": JSONRPCResponse(
                    id=2,
                    result={
                        "tools": [{"name": "my_tool", "description": "Test", "inputSchema": {}}]
                    },
                ),
            }
        )

        with patch(
            "agents_framework.mcp.client.create_transport", return_value=mock_transport
        ):
            config = MCPServerConfig(
                name="test-server",
                transport_type=TransportType.STDIO,
                transport_config=StdioTransportConfig(command="test"),
            )
            await manager.add_server(config)

            server = manager.get_tool_server("my_tool")
            assert server == "test-server"

            server = manager.get_tool_server("test-server/my_tool")
            assert server == "test-server"

            server = manager.get_tool_server("nonexistent")
            assert server is None

    @pytest.mark.asyncio
    async def test_close_all(self, mcp_client_config: MCPClientConfig):
        """Test closing all connections."""
        manager = MCPConnectionManager(mcp_client_config)

        for server_name in ["server1", "server2"]:
            mock_transport = MockTransport(
                responses={
                    "initialize": JSONRPCResponse(
                        id=1,
                        result={
                            "serverInfo": {"name": server_name, "version": "1.0"},
                            "capabilities": {"tools": {}},
                        },
                    ),
                    "tools/list": JSONRPCResponse(id=2, result={"tools": []}),
                }
            )

            with patch(
                "agents_framework.mcp.client.create_transport", return_value=mock_transport
            ):
                config = MCPServerConfig(
                    name=server_name,
                    transport_type=TransportType.STDIO,
                    transport_config=StdioTransportConfig(command="test"),
                )
                await manager.add_server(config)

        assert len(manager) == 2
        await manager.close_all()
        assert len(manager) == 0

    @pytest.mark.asyncio
    async def test_context_manager(self, mcp_client_config: MCPClientConfig):
        """Test async context manager."""
        async with MCPConnectionManager(mcp_client_config) as manager:
            assert isinstance(manager, MCPConnectionManager)
        # Manager should be closed after exiting context

    @pytest.mark.asyncio
    async def test_remove_nonexistent_server(self, mcp_client_config: MCPClientConfig):
        """Test removing a non-existent server (should be no-op)."""
        manager = MCPConnectionManager(mcp_client_config)
        await manager.remove_server("nonexistent")  # Should not raise
        assert len(manager) == 0
