#!/usr/bin/env python3
"""Example 3: MCP Integration - Connect to Writing Board MCP Server.

Vi du ve tich hop MCP (Model Context Protocol) voi Writing Board:
- Ket noi voi Writing Board MCP server
- Quan ly bai viet (list, create, update, delete)

Provider: Anthropic (native)
Model: claude-opus-4.5
MCP Server: writing-board
"""

import asyncio
from typing import Optional

from agents_framework.llm.base import LLMConfig, Message, MessageRole
from agents_framework.llm.providers.anthropic import AnthropicProvider
from agents_framework.tools.registry import ToolRegistry
from agents_framework.mcp.client import MCPClient, MCPClientConfig, MCPTool
from agents_framework.mcp.transport import (
    StdioTransportConfig,
    TransportType,
    create_transport,
)
from agents_framework.mcp.tools import MCPToolAdapter


# ============================================================================
# Cau hinh LLM
# ============================================================================

LLM_CONFIG = LLMConfig(
    model="claude-haiku-4.5",
    api_key="test",  # Khong su dung trong Anthropic native
    temperature=0.1,
    base_url="http://localhost:4141",
    max_tokens=16000,
    # extra_params={
    #     # Extended thinking configuration (Anthropic native)
    #     "thinking": {
    #         "type": "enabled",
    #         "budget_tokens": 32000,
    #     }
    # },
)

# MCP Server config - Writing Board
MCP_SERVER_CONFIG = StdioTransportConfig(
    command="npm",
    args=["run", "mcp"],
    cwd="/home/kuromi/project/mydir/board",
)


# ============================================================================
# MCP Agent Class
# ============================================================================

class WritingBoardAgent:
    """Agent voi kha nang quan ly bai viet qua Writing Board MCP."""

    def __init__(self):
        self.provider = AnthropicProvider(LLM_CONFIG)
        self.registry = ToolRegistry()
        self.mcp_client: Optional[MCPClient] = None

    async def connect_mcp_server(self) -> bool:
        """Ket noi voi Writing Board MCP server."""
        try:
            # Tao transport
            transport = create_transport(TransportType.STDIO, MCP_SERVER_CONFIG)

            # Tao MCP client
            client_config = MCPClientConfig(
                name="writing-board-agent",
                version="1.0.0",
            )
            self.mcp_client = MCPClient(transport, client_config)
            await self.mcp_client.connect()

            # Lay tools tu MCP server
            mcp_tools = self.mcp_client.list_tools()
            print(f"\n[Writing Board MCP] Connected! Available tools:")
            for mcp_tool in mcp_tools:
                adapter = MCPToolAdapter(mcp_tool, self.mcp_client)
                self.registry.register(adapter)
                desc = mcp_tool.description[:50] if mcp_tool.description else "No description"
                print(f"  - {mcp_tool.name}: {desc}...")
            return True

        except Exception as e:
            print(f"\n[Error] Khong the ket noi Writing Board MCP: {e}")
            return False

    async def disconnect(self):
        """Ngat ket noi MCP server."""
        if self.mcp_client:
            try:
                await self.mcp_client.disconnect()
                print("[Writing Board MCP] Disconnected")
            except Exception:
                pass

    async def chat(self, user_message: str, history: list[Message]) -> str:
        """Xu ly mot tin nhan tu user."""

        # Them user message
        history.append(Message(role=MessageRole.USER, content=user_message))

        # Goi LLM voi tools
        response = await self.provider.generate(
            messages=history,
            tools=self.registry.to_definitions(),
        )
        #in day du request body gom ca tools
        print(f"\n[LLM Request] Called with tools: {[tool.name for tool in self.registry]}")
        
        # Hien thi thinking neu co
        if response.has_thinking:
            print(f"  [Thinking] {response.thinking_content[:150]}...")

        # Xu ly tool calls
        while response.has_tool_calls:
            history.append(Message(
                role=MessageRole.ASSISTANT,
                content=response.content or "",
                tool_calls=response.tool_calls,
            ))

            for tool_call in response.tool_calls:
                print(f"  [Tool: {tool_call.name}]")
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
                # Truncate output for display
                display_output = str(output)[:200] + "..." if len(str(output)) > 200 else str(output)
                print(f"  [Result] {display_output}")

            response = await self.provider.generate(
                messages=history,
                tools=self.registry.to_definitions(),
            )

            if response.has_thinking:
                print(f"  [Thinking] {response.thinking_content[:150]}...")

        # Them response vao history
        history.append(Message(
            role=MessageRole.ASSISTANT,
            content=response.content,
        ))

        return response.content


# ============================================================================
# Main
# ============================================================================

async def main():
    """Chay Writing Board Agent demo."""

    print("=" * 60)
    print("WRITING BOARD AGENT")
    print("MCP Integration Demo")
    print("=" * 60)

    agent = WritingBoardAgent()

    # Ket noi Writing Board MCP server
    connected = await agent.connect_mcp_server()
    if not connected:
        print("\nKhong the ket noi MCP server. Thoat...")
        return

    # System prompt
    history = [
        Message(
            role=MessageRole.SYSTEM,
            content="""Ban la AI assistant quan ly bai viet qua Writing Board.
        Categories: blog, report, note
        Difficulties: beginner, intermediate, advanced
        Tra loi bang tieng Viet. Khi tao/cap nhat bai viet, hay xac nhan voi user truoc.""",
        )
    ]
    print("\n" + "-" * 60)
    print("Chat voi Writing Board Agent")
    print("Nhap 'quit' de thoat")
    print("-" * 60)

    while True:
        user_input = input("\nBan: ").strip()
        if user_input.lower() == "quit":
            break

        if not user_input:
            continue

        try:
            response = await agent.chat(user_input, history)
            print(f"\nAgent: {response}")
        except Exception as e:
            print(f"\n[Error] {e}")

    # Cleanup
    await agent.disconnect()
    print("\nTam biet!")


if __name__ == "__main__":
    asyncio.run(main())
