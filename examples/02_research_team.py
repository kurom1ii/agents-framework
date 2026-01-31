#!/usr/bin/env python3
"""Example 2: Research Team - Multi-agent collaboration.

Vi du ve team da agent voi supervisor pattern:
- Supervisor: Dieu phoi va tong hop
- Researcher: Tim kiem thong tin
- Writer: Viet noi dung

Provider: Anthropic (native)
Model: claude-sonnet-4-20250514
Thinking: Extended thinking enabled
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

from agents_framework.llm.base import LLMConfig, Message, MessageRole
from agents_framework.llm.providers.anthropic import AnthropicProvider


# ============================================================================
# Cau hinh chung
# ============================================================================

def create_llm_config(system_role: str) -> LLMConfig:
    """Tao LLM config cho tung agent role."""
    return LLMConfig(
        model="claude-opus-4.5",
        api_key="test",  # Khong su dung trong Anthropic native
        temperature=0.1,
        base_url="http://localhost:4141",
        max_tokens=16000,
        extra_params={
            # Extended thinking configuration (Anthropic native)
            # "thinking": {
            #     "type": "enabled",
            #     "budget_tokens": 8000,
            # },
            "metadata": {"role": system_role},
        },
    )


# ============================================================================
# Agent Classes
# ============================================================================

@dataclass
class AgentResult:
    """Ket qua tu mot agent."""
    agent_id: str
    content: str
    success: bool
    thinking: Optional[str] = None
    metadata: dict = None


class BaseAgent:
    """Base class cho tat ca agents."""

    def __init__(self, agent_id: str, role: str, system_prompt: str):
        self.agent_id = agent_id
        self.role = role
        self.system_prompt = system_prompt
        self.provider = AnthropicProvider(create_llm_config(role))

    async def execute(self, task: str, context: Optional[str] = None) -> AgentResult:
        """Thuc thi task va tra ve ket qua."""
        messages = [
            Message(role=MessageRole.SYSTEM, content=self.system_prompt),
        ]

        if context:
            messages.append(Message(
                role=MessageRole.USER,
                content=f"Context tu cac agents khac:\n{context}\n\nTask: {task}",
            ))
        else:
            messages.append(Message(role=MessageRole.USER, content=task))

        try:
            response = await self.provider.generate(messages)
            return AgentResult(
                agent_id=self.agent_id,
                content=response.content,
                success=True,
                thinking=response.thinking_content if response.has_thinking else None,
            )
        except Exception as e:
            return AgentResult(
                agent_id=self.agent_id,
                content=str(e),
                success=False,
            )


class ResearcherAgent(BaseAgent):
    """Agent chuyen tim kiem va phan tich thong tin."""

    def __init__(self):
        super().__init__(
            agent_id="researcher",
            role="researcher",
            system_prompt="""Ban la Researcher Agent - chuyen gia tim kiem va phan tich thong tin.

Nhiem vu cua ban:
1. Tim kiem thong tin lien quan den topic
2. Phan tich va tong hop du lieu
3. Trinh bay cac diem chinh (bullet points)
4. Danh gia do tin cay cua thong tin

Tra loi ngan gon, co cau truc, bang tieng Viet.""",
        )


class WriterAgent(BaseAgent):
    """Agent chuyen viet noi dung."""

    def __init__(self):
        super().__init__(
            agent_id="writer",
            role="writer",
            system_prompt="""Ban la Writer Agent - chuyen gia viet noi dung.

Nhiem vu cua ban:
1. Nhan thong tin tu Researcher
2. Viet noi dung hap dan, de doc
3. Su dung cau truc ro rang (heading, paragraphs)
4. Toi uu cho doc gia Viet Nam

Viet bang tieng Viet, van phong chuyen nghiep.""",
        )


class SupervisorAgent(BaseAgent):
    """Agent dieu phoi team."""

    def __init__(self):
        super().__init__(
            agent_id="supervisor",
            role="supervisor",
            system_prompt="""Ban la Supervisor Agent - dieu phoi team.

Nhiem vu cua ban:
1. Nhan yeu cau tu user
2. Phan chia task cho Researcher va Writer
3. Tong hop ket qua cuoi cung
4. Dam bao chat luong output

Tra loi bang tieng Viet.""",
        )


# ============================================================================
# Research Team
# ============================================================================

class ResearchTeam:
    """Team gom Supervisor, Researcher, va Writer."""

    def __init__(self):
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent()
        self.writer = WriterAgent()

    async def execute_task(self, user_request: str) -> str:
        """Thuc thi task voi workflow: Supervisor -> Researcher -> Writer -> Supervisor."""

        print("\n" + "=" * 60)
        print("RESEARCH TEAM (Anthropic + Extended Thinking)")
        print("=" * 60)

        # Step 1: Supervisor phan tich yeu cau
        print("\n[1/4] Supervisor dang phan tich yeu cau...")
        supervisor_plan = await self.supervisor.execute(
            f"Phan tich yeu cau sau va tao research plan:\n{user_request}"
        )
        if supervisor_plan.thinking:
            print(f"  [Thinking] {supervisor_plan.thinking[:150]}...")
        print(f"  Plan: {supervisor_plan.content[:200]}...")

        # Step 2: Researcher tim kiem thong tin
        print("\n[2/4] Researcher dang nghien cuu...")
        research_result = await self.researcher.execute(
            task=user_request,
            context=f"Plan tu Supervisor:\n{supervisor_plan.content}",
        )
        if research_result.thinking:
            print(f"  [Thinking] {research_result.thinking[:150]}...")
        print(f"  Research: {research_result.content[:200]}...")

        # Step 3: Writer viet noi dung
        print("\n[3/4] Writer dang viet noi dung...")
        writer_result = await self.writer.execute(
            task=f"Viet bai dua tren research ve: {user_request}",
            context=f"Ket qua research:\n{research_result.content}",
        )
        if writer_result.thinking:
            print(f"  [Thinking] {writer_result.thinking[:150]}...")
        print(f"  Draft: {writer_result.content[:200]}...")

        # Step 4: Supervisor review va hoan thien
        print("\n[4/4] Supervisor dang review...")
        final_result = await self.supervisor.execute(
            task="Review va hoan thien bai viet",
            context=f"Draft tu Writer:\n{writer_result.content}",
        )
        if final_result.thinking:
            print(f"  [Thinking] {final_result.thinking[:150]}...")

        print("\n" + "=" * 60)
        print("HOAN THANH")
        print("=" * 60)

        return final_result.content


# ============================================================================
# Main
# ============================================================================

async def main():
    """Chay Research Team voi mot task mau."""

    team = ResearchTeam()

    # Task mau
    user_request = """
    Viet mot bai blog ngan ve "Tuong lai cua AI Agents trong nam 2025"
    - Do dai: 500-800 tu
    - Doi tuong: Developer Viet Nam
    - Focus: Ung dung thuc te
    """

    result = await team.execute_task(user_request)

    print("\n" + "=" * 60)
    print("KET QUA CUOI CUNG")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
