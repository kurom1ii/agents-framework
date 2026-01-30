#!/usr/bin/env python3
"""Example 3: MCP Integration - Connect to external tools via MCP.

Ví dụ về tích hợp MCP (Model Context Protocol):
- Kết nối với MCP server
- Sử dụng tools từ MCP
- Kết hợp MCP tools với local tools

Base URL: http://localhost:4141 (OpenAI-compatible)
Model: claude-opus-4.5
Thinking: Enabled
"""

import asyncio
from typing import Optional

from agents_framework.llm.base import LLMConfig, Message, MessageRole
from agents_framework.llm.providers.openai import OpenAIProvider
from agents_framework.tools.base import tool
from agents_framework.tools.registry import ToolRegistry
from agents_framework.mcp.client import MCPClient, MCPClientConfig
from agents_framework.mcp.tools import MCPToolAdapter


# ============================================================================
# Cấu hình
# ============================================================================

LLM_CONFIG = LLMConfig(
    model="claude-opus-4.5",
    api_key="your-api-key",
    base_url="http://localhost:4141/v1",
    temperature=0.7,
    max_tokens=4096,
    extra_params={
        "thinking": {"type": "enabled", "budget_tokens": 10000},
    },
)

# MCP Server configs (ví dụ)
MCP_SERVERS = [
    {
        "name": "filesystem",
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-server-filesystem", "/tmp"],
    },
    {
        "name": "memory",
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-server-memory"],
    },
]


# ============================================================================
# Local Tools
# ============================================================================

@tool(name="get_current_time", description="Lấy thời gian hiện tại")
def get_current_time() -> str:
    """Trả về thời gian hiện tại."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool(name="calculate", description="Tính toán biểu thức")
def calculate(expression: str) -> str:
    """Tính toán biểu thức toán học."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Lỗi: {e}"


# ============================================================================
# MCP Agent Class
# ============================================================================

class MCPAgent:
    """Agent với khả năng sử dụng MCP tools."""

    def __init__(self):
        self.provider = OpenAIProvider(LLM_CONFIG)
        self.registry = ToolRegistry()
        self.mcp_clients: dict[str, MCPClient] = {}

        # Đăng ký local tools
        self.registry.register(get_current_time)
        self.registry.register(calculate)

    async def connect_mcp_server(self, name: str, command: str, args: list) -> bool:
        """Kết nối với một MCP server."""
        try:
            config = MCPClientConfig(
                name=name,
                command=command,
                args=args,
            )
            client = MCPClient(config)
            await client.connect()

            # Lấy tools từ MCP server
            mcp_tools = await client.list_tools()
            for mcp_tool in mcp_tools:
                adapter = MCPToolAdapter(mcp_tool, client)
                self.registry.register(adapter)
                print(f"  - Đăng ký tool: {mcp_tool.name}")

            self.mcp_clients[name] = client
            return True

        except Exception as e:
            print(f"  ! Lỗi kết nối {name}: {e}")
            return False

    async def connect_all_mcp_servers(self):
        """Kết nối tất cả MCP servers đã cấu hình."""
        print("\n[MCP] Đang kết nối các MCP servers...")

        for server in MCP_SERVERS:
            print(f"\n  Connecting to {server['name']}...")
            success = await self.connect_mcp_server(
                name=server["name"],
                command=server["command"],
                args=server["args"],
            )
            if success:
                print(f"  ✓ {server['name']} connected")
            else:
                print(f"  ✗ {server['name']} failed")

    async def disconnect_all(self):
        """Ngắt kết nối tất cả MCP servers."""
        for name, client in self.mcp_clients.items():
            try:
                await client.disconnect()
                print(f"  ✓ Disconnected {name}")
            except Exception:
                pass

    async def chat(self, user_message: str, history: list[Message]) -> str:
        """Xử lý một tin nhắn từ user."""

        # Thêm user message
        history.append(Message(role=MessageRole.USER, content=user_message))

        # Gọi LLM với tools
        response = await self.provider.generate(
            messages=history,
            tools=self.registry.to_definitions(),
        )

        # Xử lý tool calls
        while response.has_tool_calls:
            history.append(Message(
                role=MessageRole.ASSISTANT,
                content=response.content or "",
                tool_calls=response.tool_calls,
            ))

            for tool_call in response.tool_calls:
                print(f"  [Calling: {tool_call.name}]")
                tool_obj = self.registry.get(tool_call.name)

                if tool_obj:
                    result = await tool_obj.run(**tool_call.arguments)
                    output = result.output if result.success else f"Error: {result.error}"
                else:
                    output = f"Tool not found: {tool_call.name}"

                history.append(Message(
                    role=MessageRole.TOOL,
                    content=str(output),
                    tool_call_id=tool_call.id,
                ))
                print(f"  [Result: {str(output)[:100]}...]")

            response = await self.provider.generate(
                messages=history,
                tools=self.registry.to_definitions(),
            )

        # Thêm response vào history
        history.append(Message(
            role=MessageRole.ASSISTANT,
            content=response.content,
        ))

        return response.content


# ============================================================================
# Main
# ============================================================================

async def main():
    """Chạy MCP Agent demo."""

    print("=" * 60)
    print("MCP AGENT - Demo với MCP Integration")
    print("=" * 60)

    agent = MCPAgent()

    # Thử kết nối MCP servers (có thể fail nếu chưa cài)
    try:
        await agent.connect_all_mcp_servers()
    except Exception as e:
        print(f"\n[Warning] Không thể kết nối MCP servers: {e}")
        print("Tiếp tục với local tools only...\n")

    # Hiển thị available tools
    print("\n[Available Tools]")
    for tool_def in agent.registry.to_definitions():
        print(f"  - {tool_def.name}: {tool_def.description}")

    # System prompt
    history = [
        Message(
            role=MessageRole.SYSTEM,
            content="""Bạn là AI assistant với khả năng sử dụng nhiều tools.
Bạn có thể:
- Đọc/ghi files (nếu MCP filesystem connected)
- Lưu trữ memories (nếu MCP memory connected)
- Tính toán và xem thời gian (local tools)

Trả lời bằng tiếng Việt.""",
        )
    ]

    print("\n" + "-" * 60)
    print("Chat (nhập 'quit' để thoát)")
    print("-" * 60)

    while True:
        user_input = input("\nBạn: ").strip()
        if user_input.lower() == "quit":
            break

        try:
            response = await agent.chat(user_input, history)
            print(f"\nAgent: {response}")
        except Exception as e:
            print(f"\n[Error] {e}")

    # Cleanup
    await agent.disconnect_all()
    print("\nTạm biệt!")


if __name__ == "__main__":
    asyncio.run(main())
