#!/usr/bin/env python3
"""Example 4: Spawn Sub-Agent - Delegate tasks to specialized sub-agents.

Vi du ve cach spawn sub-agents de xu ly cac tac vu con.
Coordinator agent se spawn researcher agent de thu thap thong tin.

Key concepts:
- SubAgentSpawner: Quan ly lifecycle cua spawned agents
- SpawnConfig: Cau hinh resource limits cho sub-agents
- spawn_agent(): Tool wrapper de spawn sub-agents
- Spawn depth limits: Ngan infinite spawning (max 2 levels)
"""

import asyncio
from typing import Any, Dict

from agents_framework.llm.base import LLMConfig, Message, MessageRole
from agents_framework.llm.providers.anthropic import AnthropicProvider
from agents_framework.tools.base import tool
from agents_framework.tools.registry import ToolRegistry
from agents_framework.tools.spawn_agent import (
    spawn_agent,
    create_spawn_agent_tool,
)
from agents_framework.a2a.spawner import (
    SubAgentSpawner,
    SpawnConfig,
    SpawnLimits,
    set_current_spawner,
    set_spawn_context,
)
from agents_framework.a2a.tools.sessions_spawn import set_current_parent_session


# ============================================================================
# Cau hinh LLM
# ============================================================================

LLM_CONFIG = LLMConfig(
    model="claude-haiku-4.5",
    api_key="test",
    temperature=0.1,
    base_url="http://localhost:4141",
    max_tokens=16000,
)


# ============================================================================
# Tools cho Sub-Agents
# ============================================================================

@tool(name="web_search", description="Tim kiem thong tin tren web")
def web_search(query: str) -> str:
    """Tim kiem thong tin tren web (mock).

    Args:
        query: Tu khoa tim kiem
    """
    # Mock search results
    results = {
        "ai frameworks": [
            "1. PyTorch - Deep learning framework by Meta",
            "2. TensorFlow - ML platform by Google",
            "3. LangChain - LLM application framework",
            "4. Hugging Face - Transformers library",
            "5. OpenAI SDK - GPT integration toolkit",
        ],
        "python async": [
            "1. asyncio - Built-in async library",
            "2. aiohttp - Async HTTP client/server",
            "3. trio - Structured concurrency",
        ],
    }

    for key, items in results.items():
        if key in query.lower():
            return "\n".join(items)

    return f"Tim kiem '{query}': Khong tim thay ket qua cu the"


@tool(name="summarize", description="Tom tat noi dung")
def summarize(content: str) -> str:
    """Tom tat noi dung dai thanh ngan gon.

    Args:
        content: Noi dung can tom tat
    """
    lines = content.strip().split("\n")
    if len(lines) <= 3:
        return content
    return f"Tom tat ({len(lines)} items): " + "; ".join(
        line.split(" - ")[0] for line in lines[:5]
    )


# ============================================================================
# Mock AgentFactory cho SubAgentSpawner
# ============================================================================

class MockAgentExecutor:
    """Mock executor cho sub-agent tasks."""

    def __init__(self, tools: ToolRegistry, provider: AnthropicProvider):
        self.tools = tools
        self.provider = provider

    async def execute_task(
        self,
        task: str,
        purpose: str,
        max_turns: int = 10,
    ) -> Dict[str, Any]:
        """Execute task voi tool loop don gian."""
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=f"""Ban la sub-agent chuyen mon: {purpose}

Nhiem vu: Hoan thanh task sau roi tra ve ket qua cuoi cung.
Su dung tools khi can thiet. Tra loi ngan gon, chi tap trung vao ket qua.""",
            ),
            Message(role=MessageRole.USER, content=task),
        ]

        tokens_used = 0
        turns = 0

        for _ in range(max_turns):
            turns += 1
            response = await self.provider.generate(
                messages=messages,
                tools=self.tools.to_definitions(),
            )
            tokens_used += response.usage.get("total_tokens", 0) if response.usage else 0

            # Xu ly tool calls
            if response.has_tool_calls:
                messages.append(Message(
                    role=MessageRole.ASSISTANT,
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                ))

                for tool_call in response.tool_calls:
                    tool_obj = self.tools.get(tool_call.name)
                    if tool_obj:
                        result = await tool_obj.run(**tool_call.arguments)
                        output = result.output if result.success else result.error
                        messages.append(Message(
                            role=MessageRole.TOOL,
                            content=str(output),
                            tool_call_id=tool_call.id,
                        ))
            else:
                # Khong co tool call -> da hoan thanh
                return {
                    "output": response.content,
                    "tokens_used": tokens_used,
                    "turns_used": turns,
                }

        return {
            "output": "Max turns reached",
            "tokens_used": tokens_used,
            "turns_used": turns,
        }


# ============================================================================
# Custom SubAgentSpawner voi Mock Execution
# ============================================================================

class DemoSubAgentSpawner(SubAgentSpawner):
    """SubAgentSpawner demo voi mock execution."""

    def __init__(self, executor: MockAgentExecutor, **kwargs):
        super().__init__(**kwargs)
        self.executor = executor

    async def _run_sub_agent_task(
        self,
        config: SpawnConfig,
        task: str,
        lifecycle,
    ) -> Dict[str, Any]:
        """Override de su dung mock executor."""
        print(f"\n  [Sub-Agent: {config.agent_id}] Bat dau: {config.purpose}")
        print(f"  [Sub-Agent: {config.agent_id}] Task: {task[:80]}...")

        result = await self.executor.execute_task(
            task=task,
            purpose=config.purpose,
            max_turns=config.max_turns,
        )

        print(f"  [Sub-Agent: {config.agent_id}] Hoan thanh!")
        return result


# ============================================================================
# Ham chinh
# ============================================================================

async def run_spawn_example():
    """Chay vi du spawn sub-agent."""

    print("=" * 70)
    print("Example: Spawn Sub-Agent")
    print("Coordinator Agent se spawn Researcher Agent de nghien cuu")
    print("=" * 70)

    # Khoi tao provider
    provider = AnthropicProvider(LLM_CONFIG)

    # Dang ky tools cho sub-agents
    sub_agent_tools = ToolRegistry()
    sub_agent_tools.register(web_search)
    sub_agent_tools.register(summarize)

    # Tao mock executor
    executor = MockAgentExecutor(sub_agent_tools, provider)

    # Khoi tao spawner voi limits
    spawner = DemoSubAgentSpawner(
        executor=executor,
        limits=SpawnLimits(
            max_spawn_depth=2,           # Toi da 2 cap nested
            max_concurrent_per_parent=3,  # 3 sub-agents dong thoi per parent
            max_global_spawns=10,         # Tong 10 spawns
        ),
    )

    # Set spawner vao context de spawn_agent() tool co the su dung
    set_current_spawner(spawner)

    # Set spawn context (root level) va parent session
    set_spawn_context(depth=0, parent=None)
    set_current_parent_session("coordinator:main:session")

    # Coordinator tools (bao gom spawn_agent)
    coordinator_tools = ToolRegistry()
    coordinator_tools.register(create_spawn_agent_tool())

    # System prompt cho Coordinator
    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="""Ban la Coordinator Agent. Nhiem vu:
- Phan tich yeu cau cua user
- Spawn sub-agents de xu ly cac tac vu chuyen mon
- Tong hop ket qua va tra loi user

Hay su dung tool spawn_agent khi can delegate task.""",
        ),
    ]

    print("\nNhap yeu cau de Coordinator spawn sub-agents.")
    print("Vi du: 'Tim hieu ve cac AI framework pho bien'")
    print("Nhap 'quit' de thoat.\n")

    while True:
        user_input = input("Ban: ").strip()
        if user_input.lower() == "quit":
            print("\nTam biet!")
            break

        messages.append(Message(role=MessageRole.USER, content=user_input))

        # Goi Coordinator
        response = await provider.generate(
            messages=messages,
            tools=coordinator_tools.to_definitions(),
        )

        # Xu ly tool calls (spawn_agent)
        while response.has_tool_calls:
            messages.append(Message(
                role=MessageRole.ASSISTANT,
                content=response.content or "",
                tool_calls=response.tool_calls,
            ))

            for tool_call in response.tool_calls:
                if tool_call.name == "spawn_agent":
                    print(f"\n[Coordinator] Spawning sub-agent...")

                    # Goi spawn_agent tool
                    result = await spawn_agent(**tool_call.arguments)

                    status = result.get("status", "unknown")
                    if status == "completed":
                        output = result.get("result", "No result")
                        tokens = result.get("tokens_used", 0)
                        duration = result.get("duration_ms", 0)
                        print(f"[Spawn Result] Status: {status}")
                        print(f"[Spawn Result] Tokens: {tokens}, Duration: {duration}ms")
                    else:
                        output = f"Spawn failed: {result.get('error', 'Unknown error')}"
                        print(f"[Spawn Result] Error: {output}")

                    messages.append(Message(
                        role=MessageRole.TOOL,
                        content=str(output),
                        tool_call_id=tool_call.id,
                    ))

            # Goi lai Coordinator voi spawn results
            response = await provider.generate(
                messages=messages,
                tools=coordinator_tools.to_definitions(),
            )

        # Response cuoi cung
        print(f"\nCoordinator: {response.content}")
        messages.append(Message(
            role=MessageRole.ASSISTANT,
            content=response.content,
        ))

        # Hien thi statistics
        stats = spawner.get_statistics()
        print(f"\n[Stats] Total spawns: {stats['total_spawns']}, "
              f"Status: {stats['status_counts']}")


if __name__ == "__main__":
    asyncio.run(run_spawn_example())
