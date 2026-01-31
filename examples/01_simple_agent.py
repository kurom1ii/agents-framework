#!/usr/bin/env python3
"""Example 1: Simple Agent - Basic conversation with tools.

Vi du don gian nhat ve cach tao agent voi tools.

Provider: Anthropic (native)
Model: claude-sonnet-4-20250514
Thinking: Extended thinking enabled
"""

import asyncio
from agents_framework.llm.base import LLMConfig, Message, MessageRole
from agents_framework.llm.providers.anthropic import AnthropicProvider
from agents_framework.tools.base import tool
from agents_framework.tools.registry import ToolRegistry


# ============================================================================
# Cau hinh LLM - Anthropic native API
# ============================================================================

LLM_CONFIG = LLMConfig(
    model="claude-haiku-4.5",
    api_key="test" ,  # Khong su dung trong Anthropic native
    temperature=0.1,  # Required for extended thinking
    base_url="http://localhost:4141",
    max_tokens=16000,
    extra_params={
        # Extended thinking configuration (Anthropic native)
        # "thinking": {
        #     "type": "false",
        #     "budget_tokens": 32000,
        # },
    }
)


# ============================================================================
# Dinh nghia Tools
# ============================================================================

@tool(name="calculator", description="Tinh toan bieu thuc toan hoc")
def calculator(expression: str) -> str:
    """Tinh toan mot bieu thuc toan hoc.

    Args:
        expression: Bieu thuc toan hoc, vi du: "2 + 2", "10 * 5"
    """
    try:
        # Safe evaluation - only allow basic math
        allowed_chars = set("0123456789+-*/().% ")
        if not all(c in allowed_chars for c in expression):
            return "Loi: Bieu thuc khong hop le"
        result = eval(expression)
        return f"Ket qua: {result}"
    except Exception as e:
        return f"Loi: {e}"


@tool(name="get_weather", description="Lay thong tin thoi tiet")
def get_weather(city: str) -> str:
    """Lay thong tin thoi tiet cho mot thanh pho.

    Args:
        city: Ten thanh pho
    """
    # Mock data cho vi du
    weather_data = {
        "hanoi": "Ha Noi: 28C, Nang nhe",
        "hochiminh": "TP.HCM: 32C, Co may",
        "danang": "Da Nang: 30C, Mua rao",
    }
    city_key = city.lower().replace(" ", "")
    return weather_data.get(city_key, f"Khong co du lieu cho {city}")


# ============================================================================
# Ham chinh chay Agent
# ============================================================================

async def run_simple_agent():
    """Chay agent don gian voi conversation loop."""

    # Khoi tao LLM Provider - Anthropic native
    provider = AnthropicProvider(LLM_CONFIG)

    # Dang ky tools
    registry = ToolRegistry()
    registry.register(calculator)
    registry.register(get_weather)

    # System prompt
    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="""Ban la tro ly AI thong minh. Ban co the:
- Tinh toan bieu thuc toan hoc bang tool "calculator"
- Tra cuu thoi tiet bang tool "get_weather"

Hay tra loi bang tieng Viet va su dung tools khi can thiet.""",
        )
    ]

    print("=" * 60)
    print("Simple Agent (Anthropic + Extended Thinking)")
    print("Nhap 'quit' de thoat")
    print("=" * 60)

    while True:
        # Nhan input tu user
        user_input = input("\nBan: ").strip()
        if user_input.lower() == "quit":
            print("Tam biet!")
            break

        # Them message cua user
        messages.append(Message(role=MessageRole.USER, content=user_input))

        # Goi LLM
        response = await provider.generate(
            messages=messages,
            tools=registry.to_definitions(),
        )

        # Hien thi thinking neu co
        if response.has_thinking:
            print(f"\n[Thinking] {response.thinking_content[:200]}...")

        # Xu ly tool calls neu co
        while response.has_tool_calls:
            # Them assistant message voi tool calls
            messages.append(Message(
                role=MessageRole.ASSISTANT,
                content=response.content or "",
                tool_calls=response.tool_calls,
            ))

            # Thuc thi tung tool
            for tool_call in response.tool_calls:
                tool_obj = registry.get(tool_call.name)
                if tool_obj:
                    result = await tool_obj.run(**tool_call.arguments)
                    output = result.output if result.success else result.error

                    # Them tool response
                    messages.append(Message(
                        role=MessageRole.TOOL,
                        content=str(output),
                        tool_call_id=tool_call.id,
                    ))
                    print(f"[Tool: {tool_call.name}] -> {output}")

            # Goi LLM lai voi tool results
            response = await provider.generate(
                messages=messages,
                tools=registry.to_definitions(),
            )

            # Hien thi thinking neu co
            if response.has_thinking:
                print(f"\n[Thinking] {response.thinking_content[:200]}...")

        # In response cuoi cung
        print(f"\nAgent: {response.content}")

        # Them assistant response vao history
        messages.append(Message(
            role=MessageRole.ASSISTANT,
            content=response.content,
        ))


if __name__ == "__main__":
    asyncio.run(run_simple_agent())
