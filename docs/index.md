# Agents Framework

Framework Python nhẹ cho multi-agent orchestration với tích hợp MCP.

## Tính năng chính

- 🤖 **Multi-Agent Teams** - Supervisor, Worker, Router patterns
- 🔧 **Tool System** - Decorator-based tool definition với schema auto-generation
- 🧠 **Memory System** - Short-term, Long-term, Vector storage
- 🔌 **MCP Integration** - Kết nối với MCP servers
- 📊 **Observability** - Logging, Tracing, Metrics
- ⚡ **Async-first** - Thiết kế cho hiệu suất cao

## Cài đặt

```bash
pip install agents-framework

# Với tất cả dependencies
pip install agents-framework[all]

# Chỉ Anthropic
pip install agents-framework[anthropic]

# Chỉ OpenAI
pip install agents-framework[openai]
```

## Quick Start

### 1. Simple Agent

```python
import asyncio
from agents_framework.llm.base import LLMConfig, Message, MessageRole
from agents_framework.llm.providers.openai import OpenAIProvider
from agents_framework.tools.base import tool
from agents_framework.tools.registry import ToolRegistry

# Định nghĩa tool
@tool(name="calculator", description="Tính toán")
def calculator(expression: str) -> str:
    return str(eval(expression))

# Cấu hình LLM
config = LLMConfig(
    model="claude-opus-4.5",
    api_key="your-key",
    base_url="http://localhost:4141/v1",
)

async def main():
    provider = OpenAIProvider(config)
    registry = ToolRegistry()
    registry.register(calculator)

    messages = [
        Message(role=MessageRole.USER, content="Tính 5 + 3")
    ]

    response = await provider.generate(messages, tools=registry.to_definitions())
    print(response.content)

asyncio.run(main())
```

### 2. Multi-Agent Team

```python
from agents_framework.teams.router import MessageRouter, AgentMessage
from agents_framework.teams.registry import AgentRegistry

# Tạo registry
registry = AgentRegistry()
router = MessageRouter()

# Đăng ký agents
registry.register(researcher_agent, agent_id="researcher", role="researcher")
registry.register(writer_agent, agent_id="writer", role="writer")

# Route messages
message = AgentMessage(
    sender_id="supervisor",
    receiver_id="researcher",
    content="Tìm kiếm về AI"
)
await router.route(message)
```

### 3. MCP Integration

```python
from agents_framework.mcp.client import MCPClient, MCPClientConfig

config = MCPClientConfig(
    name="filesystem",
    command="npx",
    args=["-y", "@anthropic/mcp-server-filesystem", "/tmp"]
)

client = MCPClient(config)
await client.connect()

# List available tools
tools = await client.list_tools()

# Call a tool
result = await client.call_tool("read_file", path="/tmp/test.txt")
```

## Cấu trúc Project

```
agents_framework/
├── llm/              # LLM providers (OpenAI, Anthropic, Ollama)
├── tools/            # Tool system
├── memory/           # Memory backends
├── teams/            # Multi-agent orchestration
├── mcp/              # MCP client
├── context/          # Context management
├── execution/        # Agent execution loop
├── observability/    # Logging, tracing, metrics
└── skills/           # Reusable workflows
```

## Examples

Xem thư mục `examples/` để biết thêm ví dụ:

- `01_simple_agent.py` - Agent đơn giản với tools
- `02_research_team.py` - Team đa agent (Supervisor pattern)
- `03_mcp_integration.py` - Tích hợp MCP servers

## Configuration

### OpenAI-Compatible Endpoint

```python
config = LLMConfig(
    model="claude-opus-4.5",
    api_key="your-key",
    base_url="http://localhost:4141/v1",
    temperature=0.7,
    max_tokens=4096,
    extra_params={
        "thinking": {"type": "enabled", "budget_tokens": 10000},
    },
)
```

## License

MIT
