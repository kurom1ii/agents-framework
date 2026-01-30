# MCP API Reference

API reference cho Model Context Protocol (MCP) integration.

## MCPClient

Client kết nối đến MCP server.

### Class Definition

```python
from agents_framework.mcp import MCPClient

class MCPClient:
    def __init__(
        self,
        transport: Transport,
        client_info: Optional[MCPClientConfig] = None,
    ):
        ...
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `transport` | `Transport` | Transport layer (Stdio, SSE, WebSocket) |
| `client_info` | `MCPClientConfig` | Client configuration |

### Methods

#### connect()

Kết nối đến MCP server.

```python
from agents_framework.mcp import MCPClient, StdioTransport, StdioTransportConfig

config = StdioTransportConfig(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem"],
)
transport = StdioTransport(config)
client = MCPClient(transport)

await client.connect()
```

#### disconnect()

Ngắt kết nối.

```python
await client.disconnect()
```

#### list_tools() -> List[MCPTool]

Liệt kê tools từ server.

```python
tools = client.list_tools()
for tool in tools:
    print(f"{tool.name}: {tool.description}")
```

#### call_tool(name: str, arguments: Dict) -> Any

Gọi tool.

```python
result = await client.call_tool("read_file", {"path": "/tmp/test.txt"})
print(result)
```

#### list_resources() -> List[MCPResource]

Liệt kê resources.

```python
resources = await client.list_resources()
```

#### read_resource(uri: str) -> str

Đọc resource.

```python
content = await client.read_resource("file:///path/to/file")
```

#### list_prompts() -> List[MCPPrompt]

Liệt kê prompts.

```python
prompts = await client.list_prompts()
```

#### get_prompt(name: str, arguments: Dict = None) -> str

Lấy prompt.

```python
prompt = await client.get_prompt("summarize", {"text": "..."})
```

## Transport Types

### StdioTransport

Giao tiếp qua stdin/stdout.

```python
from agents_framework.mcp import StdioTransport, StdioTransportConfig

config = StdioTransportConfig(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem"],
    env={"HOME": "/tmp"},
)
transport = StdioTransport(config)
```

### SSETransport

Giao tiếp qua Server-Sent Events.

```python
from agents_framework.mcp import SSETransport, SSETransportConfig

config = SSETransportConfig(
    url="http://localhost:8000/sse",
    headers={"Authorization": "Bearer token"},
)
transport = SSETransport(config)
```

## MCPConnectionManager

Quản lý nhiều MCP servers.

### Class Definition

```python
from agents_framework.mcp import MCPConnectionManager, MCPServerConfig

class MCPConnectionManager:
    def __init__(self):
        ...
```

### Methods

#### add_server(config: MCPServerConfig)

Thêm server.

```python
manager = MCPConnectionManager()

await manager.add_server(MCPServerConfig(
    name="filesystem",
    transport_type=TransportType.STDIO,
    transport_config=StdioTransportConfig(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem"],
    ),
))

await manager.add_server(MCPServerConfig(
    name="database",
    transport_type=TransportType.SSE,
    transport_config=SSETransportConfig(
        url="http://localhost:8080/sse",
    ),
))
```

#### call_tool(name: str, arguments: Dict) -> Any

Gọi tool (tự động route đến server phù hợp).

```python
result = await manager.call_tool("read_file", {"path": "/tmp/test.txt"})
```

#### list_all_tools() -> List[MCPTool]

Liệt kê tools từ tất cả servers.

```python
all_tools = manager.list_all_tools()
```

#### Context Manager

```python
async with MCPConnectionManager() as manager:
    await manager.add_server(...)
    result = await manager.call_tool(...)
# Tự động disconnect khi exit
```

## MCPToolAdapter

Wrap MCP tools thành framework tools.

### Usage

```python
from agents_framework.mcp import MCPToolAdapter, create_mcp_tools_from_client
from agents_framework.tools.registry import ToolRegistry

# Từ single client
tools = create_mcp_tools_from_client(mcp_client)

# Đăng ký vào registry
registry = ToolRegistry()
for tool in tools:
    registry.register(tool)

# Hoặc dùng adapter trực tiếp
adapter = MCPToolAdapter(mcp_client)
mcp_tools = adapter.get_tools()
```

## MCPToolRegistry

Registry chuyên cho MCP tools.

```python
from agents_framework.mcp import MCPToolRegistry

mcp_registry = MCPToolRegistry()

# Thêm từ manager
await mcp_registry.register_from_manager(manager)

# Export definitions
definitions = mcp_registry.to_definitions()
```

## Data Classes

### MCPTool

```python
@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: Dict[str, Any]
```

### MCPResource

```python
@dataclass
class MCPResource:
    uri: str
    name: str
    description: Optional[str]
    mime_type: Optional[str]
```

### MCPPrompt

```python
@dataclass
class MCPPrompt:
    name: str
    description: Optional[str]
    arguments: List[Dict[str, Any]]
```

### MCPServerInfo

```python
@dataclass
class MCPServerInfo:
    name: str
    version: str
    capabilities: MCPCapability
```

## Example: Complete MCP Setup

```python
import asyncio
from agents_framework.mcp import (
    MCPClient,
    MCPConnectionManager,
    MCPServerConfig,
    StdioTransport,
    StdioTransportConfig,
    TransportType,
    create_mcp_tools_from_client,
)
from agents_framework.tools.registry import ToolRegistry
from agents_framework.llm.base import LLMConfig, Message, MessageRole
from agents_framework.llm.providers.openai import OpenAIProvider

async def main():
    # Setup MCP connection
    async with MCPConnectionManager() as manager:
        # Add filesystem server
        await manager.add_server(MCPServerConfig(
            name="filesystem",
            transport_type=TransportType.STDIO,
            transport_config=StdioTransportConfig(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            ),
        ))

        # Get all tools
        mcp_tools = manager.list_all_tools()
        print(f"Available tools: {[t.name for t in mcp_tools]}")

        # Create tool registry với MCP tools
        registry = ToolRegistry()
        for tool in create_mcp_tools_from_client(manager):
            registry.register(tool)

        # Setup LLM
        config = LLMConfig(model="gpt-4", api_key="...")
        provider = OpenAIProvider(config)

        # Use tools in conversation
        messages = [
            Message(role=MessageRole.USER, content="List files in /tmp"),
        ]

        response = await provider.generate(
            messages,
            tools=registry.to_definitions(),
        )

        if response.has_tool_calls:
            for tc in response.tool_calls:
                result = await manager.call_tool(tc.name, tc.arguments)
                print(f"Tool result: {result}")

asyncio.run(main())
```
