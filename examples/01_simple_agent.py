#!/usr/bin/env python3
"""Example 1: Simple Agent - Basic conversation with tools.

Ví dụ đơn giản nhất về cách tạo agent với tools.

Base URL: http://localhost:4141 (OpenAI-compatible)
Model: claude-opus-4.5
Thinking: Enabled
"""

import asyncio
from agents_framework.llm.base import LLMConfig, Message, MessageRole
from agents_framework.llm.providers.openai import OpenAIProvider
from agents_framework.tools.base import tool
from agents_framework.tools.registry import ToolRegistry


# ============================================================================
# Cấu hình LLM - OpenAI-compatible endpoint
# ============================================================================

LLM_CONFIG = LLMConfig(
    model="claude-opus-4.5",
    api_key="test",  # Thay bằng API key thực
    base_url="http://localhost:4141/v1",  # OpenAI-compatible endpoint
    temperature=0.7,
    max_tokens=16000,
    extra_params={
        "thinking": {"type": "enabled", "budget_tokens": 32000},
    },
)


# ============================================================================
# Định nghĩa Tools
# ============================================================================

@tool(name="calculator", description="Tính toán biểu thức toán học")
def calculator(expression: str) -> str:
    """Tính toán một biểu thức toán học.

    Args:
        expression: Biểu thức toán học, ví dụ: "2 + 2", "10 * 5"
    """
    try:
        result = eval(expression)
        return f"Kết quả: {result}"
    except Exception as e:
        return f"Lỗi: {e}"


@tool(name="get_weather", description="Lấy thông tin thời tiết")
def get_weather(city: str) -> str:
    """Lấy thông tin thời tiết cho một thành phố.

    Args:
        city: Tên thành phố
    """
    # Mock data cho ví dụ
    weather_data = {
        "hanoi": "Hà Nội: 28°C, Nắng nhẹ",
        "hochiminh": "TP.HCM: 32°C, Có mây",
        "danang": "Đà Nẵng: 30°C, Mưa rào",
    }
    city_key = city.lower().replace(" ", "")
    return weather_data.get(city_key, f"Không có dữ liệu cho {city}")


# ============================================================================
# Hàm chính chạy Agent
# ============================================================================

async def run_simple_agent():
    """Chạy agent đơn giản với conversation loop."""

    # Khởi tạo LLM Provider
    provider = OpenAIProvider(LLM_CONFIG)

    # Đăng ký tools
    registry = ToolRegistry()
    registry.register(calculator)
    registry.register(get_weather)

    # System prompt
    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="""Bạn là trợ lý AI thông minh. Bạn có thể:
- Tính toán biểu thức toán học bằng tool "calculator"
- Tra cứu thời tiết bằng tool "get_weather"

Hãy trả lời bằng tiếng Việt và sử dụng tools khi cần thiết.""",
        )
    ]

    print("=" * 60)
    print("Simple Agent - Nhập 'quit' để thoát")
    print("=" * 60)

    while True:
        # Nhận input từ user
        user_input = input("\nBạn: ").strip()
        if user_input.lower() == "quit":
            print("Tạm biệt!")
            break

        # Thêm message của user
        messages.append(Message(role=MessageRole.USER, content=user_input))

        # Gọi LLM
        response = await provider.generate(
            messages=messages,
            tools=registry.to_definitions(),
        )

        # Xử lý tool calls nếu có
        while response.has_tool_calls:
            # Thêm assistant message với tool calls
            messages.append(Message(
                role=MessageRole.ASSISTANT,
                content=response.content or "",
                tool_calls=response.tool_calls,
            ))

            # Thực thi từng tool
            for tool_call in response.tool_calls:
                tool_obj = registry.get(tool_call.name)
                if tool_obj:
                    result = await tool_obj.run(**tool_call.arguments)
                    output = result.output if result.success else result.error

                    # Thêm tool response
                    messages.append(Message(
                        role=MessageRole.TOOL,
                        content=str(output),
                        tool_call_id=tool_call.id,
                    ))
                    print(f"[Tool: {tool_call.name}] -> {output}")

            # Gọi LLM lại với tool results
            response = await provider.generate(
                messages=messages,
                tools=registry.to_definitions(),
            )

        # In response cuối cùng
        print(f"\nAgent: {response.content}")

        # Thêm assistant response vào history
        messages.append(Message(
            role=MessageRole.ASSISTANT,
            content=response.content,
        ))


if __name__ == "__main__":
    asyncio.run(run_simple_agent())
