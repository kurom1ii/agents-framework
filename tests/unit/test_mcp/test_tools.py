"""Unit tests for MCP tools discovery and invocation.

Tests cover:
- MCPToolAdapter wrapping MCP tools
- MCPToolRegistry for managing MCP tools
- Tool discovery and refresh
- Tool execution and error handling
- Integration with framework ToolRegistry
- Helper functions for creating MCP tool adapters
"""

from __future__ import annotations

import asyncio
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents_framework.mcp.transport import (
    JSONRPCResponse,
    MCPError,
)
from agents_framework.mcp.client import (
    MCPClient,
    MCPClientConfig,
    MCPConnectionManager,
    MCPServerConfig,
    MCPTool,
)
from agents_framework.mcp.transport import (
    StdioTransportConfig,
    TransportType,
)
from agents_framework.mcp.tools import (
    MCPToolAdapter,
    MCPToolRegistry,
    MCPToolsConfig,
    create_mcp_tools_from_client,
    register_mcp_tools,
)
from agents_framework.tools import (
    ToolDefinition,
    ToolRegistry,
    ToolResult,
)

from .conftest import MockTransport


# ============================================================================
# MCPToolAdapter Tests
# ============================================================================


class TestMCPToolAdapter:
    """Tests for MCPToolAdapter class."""

    @pytest.mark.asyncio
    async def test_adapter_initialization(
        self,
        sample_mcp_tool: MCPTool,
        mock_transport_with_tools: MockTransport,
        mcp_client_config: MCPClientConfig,
    ):
        """Test adapter initialization."""
        client = MCPClient(mock_transport_with_tools, mcp_client_config)
        await client.connect()

        adapter = MCPToolAdapter(sample_mcp_tool, client)

        assert adapter.name == "read_file"
        assert adapter.description == "Read contents of a file"
        assert adapter.parameters == sample_mcp_tool.input_schema
        assert adapter.mcp_tool == sample_mcp_tool
        assert adapter.client == client

    @pytest.mark.asyncio
    async def test_adapter_with_prefix(
        self,
        sample_mcp_tool: MCPTool,
        mock_transport_with_tools: MockTransport,
        mcp_client_config: MCPClientConfig,
    ):
        """Test adapter with name prefix."""
        client = MCPClient(mock_transport_with_tools, mcp_client_config)
        await client.connect()

        adapter = MCPToolAdapter(sample_mcp_tool, client, name_prefix="filesystem")

        assert adapter.name == "filesystem/read_file"
        assert adapter.name_prefix == "filesystem"

    @pytest.mark.asyncio
    async def test_adapter_execute(
        self,
        sample_mcp_tool: MCPTool,
        mock_transport_with_tools: MockTransport,
        mcp_client_config: MCPClientConfig,
    ):
        """Test adapter execute method."""
        # Add tool call response
        mock_transport_with_tools.responses["tools/call"] = JSONRPCResponse(
            id=3,
            result={
                "content": [{"type": "text", "text": "file contents here"}],
            },
        )

        client = MCPClient(mock_transport_with_tools, mcp_client_config)
        await client.connect()

        adapter = MCPToolAdapter(sample_mcp_tool, client)
        result = await adapter.execute(path="/tmp/test.txt")

        assert result == "file contents here"

    @pytest.mark.asyncio
    async def test_adapter_to_definition(
        self,
        sample_mcp_tool: MCPTool,
        mock_transport_with_tools: MockTransport,
        mcp_client_config: MCPClientConfig,
    ):
        """Test conversion to ToolDefinition."""
        client = MCPClient(mock_transport_with_tools, mcp_client_config)
        await client.connect()

        adapter = MCPToolAdapter(sample_mcp_tool, client)
        definition = adapter.to_definition()

        assert isinstance(definition, ToolDefinition)
        assert definition.name == "read_file"
        assert definition.description == "Read contents of a file"
        assert definition.parameters == sample_mcp_tool.input_schema

    @pytest.mark.asyncio
    async def test_adapter_to_definition_with_prefix(
        self,
        sample_mcp_tool: MCPTool,
        mock_transport_with_tools: MockTransport,
        mcp_client_config: MCPClientConfig,
    ):
        """Test conversion to ToolDefinition with prefix."""
        client = MCPClient(mock_transport_with_tools, mcp_client_config)
        await client.connect()

        adapter = MCPToolAdapter(sample_mcp_tool, client, name_prefix="fs")
        definition = adapter.to_definition()

        assert definition.name == "fs/read_file"


# ============================================================================
# MCPToolRegistry Tests
# ============================================================================


class TestMCPToolRegistry:
    """Tests for MCPToolRegistry class."""

    @pytest.mark.asyncio
    async def test_registry_initialization(self, mcp_client_config: MCPClientConfig):
        """Test registry initialization."""
        manager = MCPConnectionManager(mcp_client_config)
        registry = MCPToolRegistry(manager)

        assert registry.connection_manager == manager
        assert len(registry.list_tools()) == 0

    @pytest.mark.asyncio
    async def test_refresh_tools(self, mcp_client_config: MCPClientConfig):
        """Test refreshing tools from servers."""
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
                        "tools": [
                            {"name": "tool1", "description": "Tool 1", "inputSchema": {}},
                            {"name": "tool2", "description": "Tool 2", "inputSchema": {}},
                        ]
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

        registry = MCPToolRegistry(manager)
        await registry.refresh_tools()

        tools = registry.list_tools()
        # 2 tools with prefixes + 2 without = 4 total
        assert len(tools) >= 2

    @pytest.mark.asyncio
    async def test_get_tool(self, mcp_client_config: MCPClientConfig):
        """Test getting a tool by name."""
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
                        "tools": [{"name": "my_tool", "description": "My Tool", "inputSchema": {}}]
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

        registry = MCPToolRegistry(manager)
        await registry.refresh_tools()

        # Get by prefixed name
        tool = registry.get_tool("test-server/my_tool")
        assert tool is not None
        assert tool.name == "test-server/my_tool"

        # Get by unprefixed name (if unique)
        tool = registry.get_tool("my_tool")
        assert tool is not None

        # Get non-existent
        tool = registry.get_tool("nonexistent")
        assert tool is None

    @pytest.mark.asyncio
    async def test_to_definitions(self, mcp_client_config: MCPClientConfig):
        """Test converting all tools to definitions."""
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
                        "tools": [
                            {"name": "tool1", "description": "Tool 1", "inputSchema": {}},
                        ]
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

        registry = MCPToolRegistry(manager)
        await registry.refresh_tools()

        definitions = registry.to_definitions()
        assert all(isinstance(d, ToolDefinition) for d in definitions)

    @pytest.mark.asyncio
    async def test_call_tool_success(self, mcp_client_config: MCPClientConfig):
        """Test calling a tool through the registry."""
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
                    result={"content": [{"type": "text", "text": "success"}]},
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

        registry = MCPToolRegistry(manager)
        await registry.refresh_tools()

        result = await registry.call_tool("my_tool", {"arg": "value"})
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.output == "success"

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self, mcp_client_config: MCPClientConfig):
        """Test calling a non-existent tool."""
        manager = MCPConnectionManager(mcp_client_config)
        registry = MCPToolRegistry(manager)

        result = await registry.call_tool("nonexistent", {})
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_call_tool_mcp_error(self, mcp_client_config: MCPClientConfig):
        """Test calling a tool that raises MCPError."""
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
                        "tools": [{"name": "failing_tool", "description": "Fails", "inputSchema": {}}]
                    },
                ),
                "tools/call": JSONRPCResponse(
                    id=3,
                    error={"code": -32000, "message": "Tool execution failed"},
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

        registry = MCPToolRegistry(manager)
        await registry.refresh_tools()

        result = await registry.call_tool("failing_tool", {})
        assert result.success is False
        assert "Tool call failed" in result.error

    @pytest.mark.asyncio
    async def test_register_with_framework(self, mcp_client_config: MCPClientConfig):
        """Test registering MCP tools with framework registry."""
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
                        "tools": [
                            {"name": "mcp_tool", "description": "MCP Tool", "inputSchema": {}},
                        ]
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

        mcp_registry = MCPToolRegistry(manager)
        await mcp_registry.refresh_tools()

        framework_registry = ToolRegistry()
        mcp_registry.register_with_framework(framework_registry)

        # Check that tools are registered
        assert framework_registry.has("mcp_tool") or framework_registry.has("test-server/mcp_tool")

    @pytest.mark.asyncio
    async def test_register_with_framework_no_duplicates(self, mcp_client_config: MCPClientConfig):
        """Test that duplicate tools are not registered."""
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
                        "tools": [
                            {"name": "shared_tool", "description": "Shared", "inputSchema": {}},
                        ]
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

        mcp_registry = MCPToolRegistry(manager)
        await mcp_registry.refresh_tools()

        framework_registry = ToolRegistry()

        # Pre-register a tool with same name
        async def existing_tool(**kwargs):
            return "existing"

        framework_registry.register(existing_tool, name="shared_tool")

        # Register MCP tools - should skip duplicates
        mcp_registry.register_with_framework(framework_registry)

        # Original tool should still be there
        tool = framework_registry.get("shared_tool")
        result = await tool.execute()
        assert result == "existing"


# ============================================================================
# MCPToolsConfig Tests
# ============================================================================


class TestMCPToolsConfig:
    """Tests for MCPToolsConfig dataclass."""

    def test_config_defaults(self):
        """Test config with defaults."""
        config = MCPToolsConfig()
        assert config.auto_refresh is True
        assert config.prefix_with_server is True
        assert config.register_to_framework is True

    def test_config_custom(self):
        """Test config with custom values."""
        config = MCPToolsConfig(
            auto_refresh=False,
            prefix_with_server=False,
            register_to_framework=False,
        )
        assert config.auto_refresh is False
        assert config.prefix_with_server is False
        assert config.register_to_framework is False


# ============================================================================
# create_mcp_tools_from_client Tests
# ============================================================================


class TestCreateMCPToolsFromClient:
    """Tests for create_mcp_tools_from_client function."""

    @pytest.mark.asyncio
    async def test_create_tools_basic(
        self,
        mock_transport_with_tools: MockTransport,
        mcp_client_config: MCPClientConfig,
    ):
        """Test creating tool adapters from client."""
        client = MCPClient(mock_transport_with_tools, mcp_client_config)
        await client.connect()

        adapters = create_mcp_tools_from_client(client)

        assert len(adapters) == 3  # Based on sample_mcp_tools fixture
        assert all(isinstance(a, MCPToolAdapter) for a in adapters)

    @pytest.mark.asyncio
    async def test_create_tools_with_prefix(
        self,
        mock_transport_with_tools: MockTransport,
        mcp_client_config: MCPClientConfig,
    ):
        """Test creating tool adapters with name prefix."""
        client = MCPClient(mock_transport_with_tools, mcp_client_config)
        await client.connect()

        adapters = create_mcp_tools_from_client(client, name_prefix="myserver")

        assert all(a.name.startswith("myserver/") for a in adapters)

    @pytest.mark.asyncio
    async def test_create_tools_empty_client(
        self,
        mock_transport: MockTransport,
        mcp_client_config: MCPClientConfig,
    ):
        """Test creating tools from client with no tools."""
        # Setup transport with no tools
        mock_transport.responses["initialize"] = JSONRPCResponse(
            id=1,
            result={
                "serverInfo": {"name": "empty", "version": "1.0"},
                "capabilities": {"tools": {}},
            },
        )
        mock_transport.responses["tools/list"] = JSONRPCResponse(
            id=2, result={"tools": []}
        )

        client = MCPClient(mock_transport, mcp_client_config)
        await client.connect()

        adapters = create_mcp_tools_from_client(client)

        assert len(adapters) == 0


# ============================================================================
# register_mcp_tools Tests
# ============================================================================


class TestRegisterMCPTools:
    """Tests for register_mcp_tools function."""

    @pytest.mark.asyncio
    async def test_register_tools_with_prefixes(self, mcp_client_config: MCPClientConfig):
        """Test registering tools with server name prefixes."""
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
                        "tools": [
                            {"name": "tool1", "description": "Tool 1", "inputSchema": {}},
                            {"name": "tool2", "description": "Tool 2", "inputSchema": {}},
                        ]
                    },
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

        registry = ToolRegistry()
        count = register_mcp_tools(manager, registry, use_prefixes=True)

        assert count == 2
        assert registry.has("myserver/tool1")
        assert registry.has("myserver/tool2")

    @pytest.mark.asyncio
    async def test_register_tools_without_prefixes(self, mcp_client_config: MCPClientConfig):
        """Test registering tools without prefixes."""
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
                        "tools": [
                            {"name": "tool1", "description": "Tool 1", "inputSchema": {}},
                        ]
                    },
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

        registry = ToolRegistry()
        count = register_mcp_tools(manager, registry, use_prefixes=False)

        assert count == 1
        assert registry.has("tool1")
        assert not registry.has("myserver/tool1")

    @pytest.mark.asyncio
    async def test_register_tools_skip_existing(self, mcp_client_config: MCPClientConfig):
        """Test that existing tools are not overwritten."""
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
                        "tools": [
                            {"name": "existing", "description": "MCP version", "inputSchema": {}},
                        ]
                    },
                ),
            }
        )

        with patch(
            "agents_framework.mcp.client.create_transport", return_value=mock_transport
        ):
            config = MCPServerConfig(
                name="server",
                transport_type=TransportType.STDIO,
                transport_config=StdioTransportConfig(command="test"),
            )
            await manager.add_server(config)

        registry = ToolRegistry()

        # Pre-register a tool
        async def existing_func(**kwargs):
            return "original"

        registry.register(existing_func, name="server/existing")

        count = register_mcp_tools(manager, registry, use_prefixes=True)

        # Should not have registered the duplicate
        assert count == 0

    @pytest.mark.asyncio
    async def test_register_tools_multiple_servers(self, mcp_client_config: MCPClientConfig):
        """Test registering tools from multiple servers."""
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

        registry = ToolRegistry()
        count = register_mcp_tools(manager, registry, use_prefixes=True)

        assert count == 2
        assert registry.has("server1/tool_server1")
        assert registry.has("server2/tool_server2")


# ============================================================================
# Integration Tests
# ============================================================================


class TestMCPToolsIntegration:
    """Integration tests for MCP tools with framework."""

    @pytest.mark.asyncio
    async def test_full_tool_execution_flow(self, mcp_client_config: MCPClientConfig):
        """Test complete flow from discovery to execution."""
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
                        "tools": [
                            {
                                "name": "greet",
                                "description": "Greet someone",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "person": {"type": "string"},
                                    },
                                    "required": ["person"],
                                },
                            }
                        ]
                    },
                ),
                "tools/call": JSONRPCResponse(
                    id=3,
                    result={"content": [{"type": "text", "text": "Hello, World!"}]},
                ),
            }
        )

        with patch(
            "agents_framework.mcp.client.create_transport", return_value=mock_transport
        ):
            config = MCPServerConfig(
                name="greeter",
                transport_type=TransportType.STDIO,
                transport_config=StdioTransportConfig(command="test"),
            )
            await manager.add_server(config)

        # Create MCP registry and refresh
        mcp_registry = MCPToolRegistry(manager)
        await mcp_registry.refresh_tools()

        # Register with framework
        framework_registry = ToolRegistry()
        mcp_registry.register_with_framework(framework_registry)

        # Execute via framework registry
        result = await framework_registry.execute("greet", person="World")

        assert result.success is True
        assert result.output == "Hello, World!"

    @pytest.mark.asyncio
    async def test_tool_schema_preserved(self, mcp_client_config: MCPClientConfig):
        """Test that tool schema is preserved through adapter."""
        manager = MCPConnectionManager(mcp_client_config)

        expected_schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        }

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
                        "tools": [
                            {
                                "name": "search",
                                "description": "Search tool",
                                "inputSchema": expected_schema,
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
                name="search-server",
                transport_type=TransportType.STDIO,
                transport_config=StdioTransportConfig(command="test"),
            )
            await manager.add_server(config)

        mcp_registry = MCPToolRegistry(manager)
        await mcp_registry.refresh_tools()

        tool = mcp_registry.get_tool("search")
        definition = tool.to_definition()

        assert definition.parameters == expected_schema

    @pytest.mark.asyncio
    async def test_multiple_content_types(self, mcp_client_config: MCPClientConfig):
        """Test handling different content types in tool results."""
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
                        "tools": [
                            {"name": "mixed_tool", "description": "Returns mixed content", "inputSchema": {}},
                        ]
                    },
                ),
                "tools/call": JSONRPCResponse(
                    id=3,
                    result={
                        "content": [
                            {"type": "text", "text": "Description"},
                            {"type": "image", "data": "base64imagedata", "mimeType": "image/png"},
                        ]
                    },
                ),
            }
        )

        with patch(
            "agents_framework.mcp.client.create_transport", return_value=mock_transport
        ):
            config = MCPServerConfig(
                name="mixed",
                transport_type=TransportType.STDIO,
                transport_config=StdioTransportConfig(command="test"),
            )
            await manager.add_server(config)

        mcp_registry = MCPToolRegistry(manager)
        await mcp_registry.refresh_tools()

        result = await mcp_registry.call_tool("mixed_tool", {})

        assert result.success is True
        assert isinstance(result.output, list)
        assert len(result.output) == 2
