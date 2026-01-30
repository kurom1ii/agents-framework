# MCP Integration Guide

Hướng dẫn tích hợp Model Context Protocol (MCP).

## Giới Thiệu

MCP là protocol chuẩn để kết nối AI agents với external tools và data sources. Framework hỗ trợ kết nối đến bất kỳ MCP server nào.

## Quick Start

### 1. Connect to MCP Server

```python
import asyncio
from agents_framework.mcp import (
    MCPClient,
    StdioTransport,
    StdioTransportConfig,
)

async def main():
    # Configure transport
    config = StdioTransportConfig(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    )
    transport = StdioTransport(config)

    # Create client
    client = MCPClient(transport)

    # Connect
    await client.connect()

    # List available tools
    tools = client.list_tools()
    for tool in tools:
        print(f"Tool: {tool.name} - {tool.description}")

    # Call a tool
    result = await client.call_tool("read_file", {"path": "/tmp/test.txt"})
    print(f"Result: {result}")

    # Disconnect
    await client.disconnect()

asyncio.run(main())
```

### 2. Use with LLM Agent

```python
from agents_framework.mcp import create_mcp_tools_from_client
from agents_framework.tools.registry import ToolRegistry
from agents_framework.llm.providers.openai import OpenAIProvider

async def agent_with_mcp():
    # Setup MCP
    client = await setup_mcp_client()

    # Convert MCP tools to framework tools
    mcp_tools = create_mcp_tools_from_client(client)

    # Add to registry
    registry = ToolRegistry()
    for tool in mcp_tools:
        registry.register(tool)

    # Use with LLM
    provider = OpenAIProvider(config)
    response = await provider.generate(
        messages,
        tools=registry.to_definitions(),
    )

    # Handle tool calls
    if response.has_tool_calls:
        for tc in response.tool_calls:
            result = await client.call_tool(tc.name, tc.arguments)
            print(f"Tool result: {result}")
```

## Transport Types

### Stdio Transport

Giao tiếp qua standard input/output. Phổ biến nhất.

```python
from agents_framework.mcp import StdioTransport, StdioTransportConfig

config = StdioTransportConfig(
    command="npx",                          # Command to run
    args=["-y", "@mcp/server-name"],        # Arguments
    env={"API_KEY": "..."},                 # Environment variables
    cwd="/path/to/working/dir",             # Working directory
)
transport = StdioTransport(config)
```

### SSE Transport

Server-Sent Events cho remote servers.

```python
from agents_framework.mcp import SSETransport, SSETransportConfig

config = SSETransportConfig(
    url="http://localhost:8000/sse",
    headers={"Authorization": "Bearer token"},
    timeout=30.0,
)
transport = SSETransport(config)
```

## Multiple MCP Servers

Kết nối nhiều servers cùng lúc.

```python
from agents_framework.mcp import (
    MCPConnectionManager,
    MCPServerConfig,
    TransportType,
    StdioTransportConfig,
)

async def multi_server_setup():
    manager = MCPConnectionManager()

    # Add filesystem server
    await manager.add_server(MCPServerConfig(
        name="filesystem",
        transport_type=TransportType.STDIO,
        transport_config=StdioTransportConfig(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/home"],
        ),
    ))

    # Add database server
    await manager.add_server(MCPServerConfig(
        name="database",
        transport_type=TransportType.STDIO,
        transport_config=StdioTransportConfig(
            command="npx",
            args=["-y", "@mcp/sqlite-server", "database.db"],
        ),
    ))

    # Add web search server
    await manager.add_server(MCPServerConfig(
        name="search",
        transport_type=TransportType.STDIO,
        transport_config=StdioTransportConfig(
            command="npx",
            args=["-y", "@mcp/brave-search"],
            env={"BRAVE_API_KEY": "..."},
        ),
    ))

    # List all tools from all servers
    all_tools = manager.list_all_tools()
    print(f"Total tools: {len(all_tools)}")

    # Call tool - manager routes to correct server
    result = await manager.call_tool("read_file", {"path": "/home/test.txt"})

    return manager
```

## Context Manager Usage

Tự động cleanup connections.

```python
async with MCPConnectionManager() as manager:
    await manager.add_server(config1)
    await manager.add_server(config2)

    result = await manager.call_tool("some_tool", {"arg": "value"})

# Connections automatically closed
```

## Popular MCP Servers

### Filesystem Server

```python
# Access filesystem
config = StdioTransportConfig(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/allowed/path"],
)

# Tools: read_file, write_file, list_directory, etc.
```

### GitHub Server

```python
# Access GitHub
config = StdioTransportConfig(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-github"],
    env={"GITHUB_TOKEN": "ghp_..."},
)

# Tools: create_issue, search_code, get_file_contents, etc.
```

### Brave Search Server

```python
# Web search
config = StdioTransportConfig(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-brave-search"],
    env={"BRAVE_API_KEY": "..."},
)

# Tools: brave_web_search, brave_local_search
```

### SQLite Server

```python
# Database access
config = StdioTransportConfig(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-sqlite", "path/to/db.sqlite"],
)

# Tools: query, execute, describe_table
```

### Slack Server

```python
# Slack integration
config = StdioTransportConfig(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-slack"],
    env={"SLACK_TOKEN": "xoxb-..."},
)

# Tools: send_message, list_channels, search_messages
```

## Resources & Prompts

MCP servers có thể expose resources và prompts.

### Using Resources

```python
# List resources
resources = await client.list_resources()
for r in resources:
    print(f"{r.uri}: {r.name}")

# Read resource
content = await client.read_resource("file:///path/to/file.txt")
print(content)
```

### Using Prompts

```python
# List prompts
prompts = await client.list_prompts()
for p in prompts:
    print(f"{p.name}: {p.description}")

# Get prompt
prompt_text = await client.get_prompt(
    "summarize",
    arguments={"text": "Long text to summarize..."}
)
print(prompt_text)
```

## Error Handling

```python
from agents_framework.mcp import MCPError, TransportError, ConnectionError

try:
    result = await client.call_tool("some_tool", {"arg": "value"})
except ConnectionError as e:
    print(f"Connection failed: {e}")
    # Retry or fallback
except TransportError as e:
    print(f"Transport error: {e}")
    # Check server status
except MCPError as e:
    print(f"MCP error: {e}")
    # Handle MCP-specific error
```

## Complete Example: Research Agent

```python
import asyncio
from agents_framework.mcp import (
    MCPConnectionManager,
    MCPServerConfig,
    TransportType,
    StdioTransportConfig,
    create_mcp_tools_from_client,
)
from agents_framework.llm.base import LLMConfig, Message, MessageRole
from agents_framework.llm.providers.openai import OpenAIProvider
from agents_framework.tools.registry import ToolRegistry

async def research_agent():
    # Setup MCP servers
    async with MCPConnectionManager() as mcp:
        # Web search
        await mcp.add_server(MCPServerConfig(
            name="search",
            transport_type=TransportType.STDIO,
            transport_config=StdioTransportConfig(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-brave-search"],
                env={"BRAVE_API_KEY": "..."},
            ),
        ))

        # Filesystem for saving results
        await mcp.add_server(MCPServerConfig(
            name="fs",
            transport_type=TransportType.STDIO,
            transport_config=StdioTransportConfig(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "./research"],
            ),
        ))

        # Create tool registry
        registry = ToolRegistry()
        for tool in create_mcp_tools_from_client(mcp):
            registry.register(tool)

        # Setup LLM
        llm_config = LLMConfig(
            model="gpt-4",
            api_key="...",
        )
        provider = OpenAIProvider(llm_config)

        # Research conversation
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content="You are a research assistant. Use search to find information and save results to files."
            ),
            Message(
                role=MessageRole.USER,
                content="Research the latest developments in quantum computing and save a summary."
            ),
        ]

        # Agent loop
        max_turns = 10
        for _ in range(max_turns):
            response = await provider.generate(
                messages,
                tools=registry.to_definitions(),
            )

            if response.has_tool_calls:
                for tc in response.tool_calls:
                    print(f"Calling tool: {tc.name}")
                    result = await mcp.call_tool(tc.name, tc.arguments)
                    messages.append(Message(
                        role=MessageRole.TOOL,
                        content=str(result),
                        tool_call_id=tc.id,
                    ))
            else:
                print(f"Final response: {response.content}")
                break

asyncio.run(research_agent())
```

## Custom MCP Server

Tạo MCP server riêng cho tools của bạn:

```python
# my_mcp_server.py
from mcp.server import Server
from mcp.types import Tool

app = Server("my-custom-server")

@app.tool()
async def my_tool(arg1: str, arg2: int) -> str:
    """My custom tool description."""
    return f"Result: {arg1} - {arg2}"

if __name__ == "__main__":
    app.run()
```

Sử dụng:

```python
config = StdioTransportConfig(
    command="python",
    args=["my_mcp_server.py"],
)
```
