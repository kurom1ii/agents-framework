# Getting Started with Agents Framework

Hướng dẫn bắt đầu nhanh với Agents Framework.

## Cài đặt

```bash
# Cài đặt cơ bản
pip install agents-framework

# Với Anthropic Claude
pip install agents-framework[anthropic]

# Với OpenAI
pip install agents-framework[openai]

# Với tất cả providers
pip install agents-framework[all]
```

## Agent Đầu Tiên

### 1. Simple Agent với Tools

```python
import asyncio
from agents_framework.llm.base import LLMConfig, Message, MessageRole
from agents_framework.llm.providers.openai import OpenAIProvider
from agents_framework.tools.base import tool
from agents_framework.tools.registry import ToolRegistry

# Định nghĩa tool với decorator
@tool(name="calculator", description="Tính toán biểu thức số học")
def calculator(expression: str) -> str:
    """Thực hiện tính toán."""
    return str(eval(expression))

@tool(name="greet", description="Chào người dùng")
def greet(name: str) -> str:
    """Chào một người."""
    return f"Xin chào {name}!"

async def main():
    # Cấu hình LLM
    config = LLMConfig(
        model="gpt-4",
        api_key="your-api-key",
        temperature=0.7,
    )

    # Tạo provider và registry
    provider = OpenAIProvider(config)
    registry = ToolRegistry()
    registry.register(calculator)
    registry.register(greet)

    # Tạo conversation
    messages = [
        Message(role=MessageRole.SYSTEM, content="Bạn là trợ lý hữu ích."),
        Message(role=MessageRole.USER, content="Tính 25 * 4 cho tôi"),
    ]

    # Generate với tools
    response = await provider.generate(
        messages,
        tools=registry.to_definitions()
    )

    # Xử lý tool calls nếu có
    if response.has_tool_calls:
        for tc in response.tool_calls:
            tool_obj = registry.get(tc.name)
            result = await tool_obj.run(**tc.arguments)
            print(f"Tool {tc.name}: {result.output}")
    else:
        print(response.content)

asyncio.run(main())
```

### 2. Agent với Memory

```python
from agents_framework.memory.short_term import SessionMemory

# Tạo memory với giới hạn token
memory = SessionMemory(max_tokens=4000)

# Thêm messages vào memory
memory.add_message("user", "Tên tôi là Minh")
memory.add_message("assistant", "Xin chào Minh!")
memory.add_message("user", "Tôi làm developer Python")

# Lấy context string để đưa vào prompt
context = memory.get_context_string()
print(context)

# Lấy danh sách messages
messages = memory.get_messages()
```

### 3. Multi-Agent Team

```python
from agents_framework.teams.router import MessageRouter, AgentMessage
from agents_framework.teams.registry import AgentRegistry

# Tạo registry cho agents
registry = AgentRegistry()

# Đăng ký agents với roles
registry.register(researcher_agent, agent_id="researcher", role="researcher")
registry.register(writer_agent, agent_id="writer", role="writer")
registry.register(supervisor_agent, agent_id="supervisor", role="supervisor")

# Tạo router để gửi messages
router = MessageRouter()

# Route message đến agent cụ thể
message = AgentMessage(
    sender_id="supervisor",
    receiver_id="researcher",
    content="Tìm kiếm thông tin về AI",
)
await router.route(message)

# Lấy agents theo role
researchers = registry.get_by_role("researcher")
```

## Các Patterns Phổ Biến

### Hierarchical Pattern (Supervisor)

```python
from agents_framework.agents.supervisor import SupervisorAgent

supervisor = SupervisorAgent(
    llm_provider=provider,
    workers=["researcher", "writer", "analyst"],
    strategy="parallel",  # hoặc "sequential"
)

result = await supervisor.run("Viết báo cáo về thị trường AI")
```

### Swarm Pattern (Handoff)

```python
from agents_framework.teams.patterns.swarm import SwarmPattern, HandoffResult

swarm = SwarmPattern(
    agents={"triage": triage_agent, "tech": tech_agent, "billing": billing_agent},
    entry_point="triage",
)

# Agent có thể handoff cho agent khác
result = await swarm.run("Tôi cần hỗ trợ kỹ thuật")
```

## MCP Integration

```python
from agents_framework.mcp import MCPClient, StdioTransport, StdioTransportConfig

# Kết nối đến MCP server
config = StdioTransportConfig(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem"],
)
transport = StdioTransport(config)
client = MCPClient(transport)

await client.connect()

# List và sử dụng tools từ MCP server
tools = client.list_tools()
result = await client.call_tool("read_file", {"path": "/tmp/test.txt"})

await client.disconnect()
```

## Bước Tiếp Theo

- [Architecture Overview](architecture.md) - Hiểu cấu trúc framework
- [API Reference](api/agents.md) - Chi tiết các APIs
- [Patterns Guide](guides/patterns.md) - Các patterns multi-agent
- [Memory Guide](guides/memory.md) - Hệ thống memory chi tiết
