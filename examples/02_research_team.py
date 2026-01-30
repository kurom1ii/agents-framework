#!/usr/bin/env python3
"""Example 2: Research Team - Multi-agent collaboration.

Ví dụ về team đa agent với supervisor pattern:
- Supervisor: Điều phối và tổng hợp
- Researcher: Tìm kiếm thông tin
- Writer: Viết nội dung

Base URL: http://localhost:4141 (OpenAI-compatible)
Model: claude-opus-4.5
Thinking: Enabled
"""

import asyncio
from dataclasses import dataclass
from typing import List, Optional

from agents_framework.llm.base import LLMConfig, Message, MessageRole
from agents_framework.llm.providers.openai import OpenAIProvider
from agents_framework.tools.base import tool


# ============================================================================
# Cấu hình chung
# ============================================================================

def create_llm_config(system_role: str) -> LLMConfig:
    """Tạo LLM config cho từng agent role."""
    return LLMConfig(
        model="claude-opus-4.5",
        api_key="your-api-key",
        base_url="http://localhost:4141/v1",
        temperature=0.7,
        max_tokens=4096,
        extra_params={
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            "metadata": {"role": system_role},
        },
    )


# ============================================================================
# Agent Classes
# ============================================================================

@dataclass
class AgentResult:
    """Kết quả từ một agent."""
    agent_id: str
    content: str
    success: bool
    metadata: dict = None


class BaseAgent:
    """Base class cho tất cả agents."""

    def __init__(self, agent_id: str, role: str, system_prompt: str):
        self.agent_id = agent_id
        self.role = role
        self.system_prompt = system_prompt
        self.provider = OpenAIProvider(create_llm_config(role))

    async def execute(self, task: str, context: Optional[str] = None) -> AgentResult:
        """Thực thi task và trả về kết quả."""
        messages = [
            Message(role=MessageRole.SYSTEM, content=self.system_prompt),
        ]

        if context:
            messages.append(Message(
                role=MessageRole.USER,
                content=f"Context từ các agents khác:\n{context}\n\nTask: {task}",
            ))
        else:
            messages.append(Message(role=MessageRole.USER, content=task))

        try:
            response = await self.provider.generate(messages)
            return AgentResult(
                agent_id=self.agent_id,
                content=response.content,
                success=True,
            )
        except Exception as e:
            return AgentResult(
                agent_id=self.agent_id,
                content=str(e),
                success=False,
            )


class ResearcherAgent(BaseAgent):
    """Agent chuyên tìm kiếm và phân tích thông tin."""

    def __init__(self):
        super().__init__(
            agent_id="researcher",
            role="researcher",
            system_prompt="""Bạn là Researcher Agent - chuyên gia tìm kiếm và phân tích thông tin.

Nhiệm vụ của bạn:
1. Tìm kiếm thông tin liên quan đến topic
2. Phân tích và tổng hợp dữ liệu
3. Trình bày các điểm chính (bullet points)
4. Đánh giá độ tin cậy của thông tin

Trả lời ngắn gọn, có cấu trúc, bằng tiếng Việt.""",
        )


class WriterAgent(BaseAgent):
    """Agent chuyên viết nội dung."""

    def __init__(self):
        super().__init__(
            agent_id="writer",
            role="writer",
            system_prompt="""Bạn là Writer Agent - chuyên gia viết nội dung.

Nhiệm vụ của bạn:
1. Nhận thông tin từ Researcher
2. Viết nội dung hấp dẫn, dễ đọc
3. Sử dụng cấu trúc rõ ràng (heading, paragraphs)
4. Tối ưu cho độc giả Việt Nam

Viết bằng tiếng Việt, văn phong chuyên nghiệp.""",
        )


class SupervisorAgent(BaseAgent):
    """Agent điều phối team."""

    def __init__(self):
        super().__init__(
            agent_id="supervisor",
            role="supervisor",
            system_prompt="""Bạn là Supervisor Agent - điều phối team.

Nhiệm vụ của bạn:
1. Nhận yêu cầu từ user
2. Phân chia task cho Researcher và Writer
3. Tổng hợp kết quả cuối cùng
4. Đảm bảo chất lượng output

Trả lời bằng tiếng Việt.""",
        )


# ============================================================================
# Research Team
# ============================================================================

class ResearchTeam:
    """Team gồm Supervisor, Researcher, và Writer."""

    def __init__(self):
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent()
        self.writer = WriterAgent()

    async def execute_task(self, user_request: str) -> str:
        """Thực thi task với workflow: Supervisor -> Researcher -> Writer -> Supervisor."""

        print("\n" + "=" * 60)
        print("RESEARCH TEAM - Bắt đầu xử lý")
        print("=" * 60)

        # Step 1: Supervisor phân tích yêu cầu
        print("\n[1/4] Supervisor đang phân tích yêu cầu...")
        supervisor_plan = await self.supervisor.execute(
            f"Phân tích yêu cầu sau và tạo research plan:\n{user_request}"
        )
        print(f"✓ Plan: {supervisor_plan.content[:200]}...")

        # Step 2: Researcher tìm kiếm thông tin
        print("\n[2/4] Researcher đang nghiên cứu...")
        research_result = await self.researcher.execute(
            task=user_request,
            context=f"Plan từ Supervisor:\n{supervisor_plan.content}",
        )
        print(f"✓ Research: {research_result.content[:200]}...")

        # Step 3: Writer viết nội dung
        print("\n[3/4] Writer đang viết nội dung...")
        writer_result = await self.writer.execute(
            task=f"Viết bài dựa trên research về: {user_request}",
            context=f"Kết quả research:\n{research_result.content}",
        )
        print(f"✓ Draft: {writer_result.content[:200]}...")

        # Step 4: Supervisor review và hoàn thiện
        print("\n[4/4] Supervisor đang review...")
        final_result = await self.supervisor.execute(
            task="Review và hoàn thiện bài viết",
            context=f"Draft từ Writer:\n{writer_result.content}",
        )

        print("\n" + "=" * 60)
        print("HOÀN THÀNH")
        print("=" * 60)

        return final_result.content


# ============================================================================
# Main
# ============================================================================

async def main():
    """Chạy Research Team với một task mẫu."""

    team = ResearchTeam()

    # Task mẫu
    user_request = """
    Viết một bài blog ngắn về "Tương lai của AI Agents trong năm 2025"
    - Độ dài: 500-800 từ
    - Đối tượng: Developer Việt Nam
    - Focus: Ứng dụng thực tế
    """

    result = await team.execute_task(user_request)

    print("\n" + "=" * 60)
    print("KẾT QUẢ CUỐI CÙNG")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
