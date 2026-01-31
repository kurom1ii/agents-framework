#!/usr/bin/env python3
"""Example 5: Parallel Sub-Agents - Spawn multiple agents concurrently.

Vi du ve cach spawn nhieu sub-agents song song de xu ly cac tac vu doc lap.
Coordinator spawn nhieu researchers cung luc va tong hop ket qua.

Key concepts:
- spawn_parallel(): Spawn nhieu sub-agents dong thoi
- asyncio.gather(): Chay cac spawns song song
- Resource limits: max_concurrent_per_parent gioi han so luong dong thoi
- Error handling: Cac spawn that bai khong anh huong den spawn khac
"""

import asyncio
from typing import Any, Dict, List

from agents_framework.llm.base import LLMConfig, Message, MessageRole
from agents_framework.llm.providers.anthropic import AnthropicProvider
from agents_framework.tools.base import tool
from agents_framework.tools.registry import ToolRegistry
from agents_framework.tools.spawn_agent import (
    spawn_agent,
    spawn_parallel,
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
# Mock Tools cho cac Sub-Agents khac nhau
# ============================================================================

# Researcher tools
@tool(name="search_papers", description="Tim kiem bai bao nghien cuu")
def search_papers(topic: str) -> str:
    """Tim kiem bai bao ve chu de.

    Args:
        topic: Chu de can tim
    """
    papers = {
        "machine learning": [
            "Attention Is All You Need (2017) - Transformer architecture",
            "BERT: Pre-training of Deep Bidirectional (2018)",
            "GPT-4 Technical Report (2023)",
        ],
        "computer vision": [
            "ResNet: Deep Residual Learning (2015)",
            "Vision Transformer (ViT) (2020)",
            "CLIP: Connecting Text and Images (2021)",
        ],
        "nlp": [
            "Word2Vec: Distributed Representations (2013)",
            "ELMo: Deep contextualized word representations (2018)",
            "T5: Text-to-Text Transfer Transformer (2019)",
        ],
    }

    for key, items in papers.items():
        if key in topic.lower():
            return "Papers found:\n" + "\n".join(f"- {p}" for p in items)
    return f"No papers found for '{topic}'"


# Coder tools
@tool(name="write_code", description="Viet code mau")
def write_code(description: str, language: str = "python") -> str:
    """Viet code mau theo mo ta.

    Args:
        description: Mo ta chuc nang can viet
        language: Ngon ngu lap trinh
    """
    if "fibonacci" in description.lower():
        return """```python
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```"""
    if "async" in description.lower():
        return """```python
async def fetch_data(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```"""
    return f"# Code for: {description}\n# TODO: Implement"


# Analyst tools
@tool(name="analyze_data", description="Phan tich du lieu")
def analyze_data(data_type: str) -> str:
    """Phan tich du lieu theo loai.

    Args:
        data_type: Loai du lieu can phan tich
    """
    analyses = {
        "performance": "Performance analysis: Avg response time 120ms, P99 450ms",
        "usage": "Usage analysis: 10K daily active users, 85% retention",
        "cost": "Cost analysis: $2.5K/month infrastructure, $0.002 per request",
    }
    return analyses.get(data_type, f"Analysis for {data_type}: No specific data")


# ============================================================================
# Mock Executor cho Parallel Spawns
# ============================================================================

class ParallelMockExecutor:
    """Executor cho parallel sub-agent tasks."""

    def __init__(self, provider: AnthropicProvider):
        self.provider = provider
        self._tools_by_role = {
            "researcher": [search_papers],
            "coder": [write_code],
            "analyst": [analyze_data],
        }

    def get_tools_for_role(self, agent_id: str) -> ToolRegistry:
        """Lay tools phu hop cho tung role."""
        registry = ToolRegistry()
        role = agent_id.split("-")[0] if "-" in agent_id else agent_id
        tools = self._tools_by_role.get(role, [])
        for t in tools:
            registry.register(t)
        return registry

    async def execute_task(
        self,
        agent_id: str,
        task: str,
        purpose: str,
        max_turns: int = 5,
    ) -> Dict[str, Any]:
        """Execute task voi role-specific tools."""
        tools = self.get_tools_for_role(agent_id)

        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=f"Ban la {agent_id}. Muc dich: {purpose}. "
                        "Hoan thanh task ngan gon, chi tra ve ket qua chinh.",
            ),
            Message(role=MessageRole.USER, content=task),
        ]

        tokens_used = 0
        for turn in range(max_turns):
            response = await self.provider.generate(
                messages=messages,
                tools=tools.to_definitions() if tools.list_tools() else None,  # type: ignore
            )
            tokens_used += response.usage.get("total_tokens", 0) if response.usage else 0

            if response.has_tool_calls:
                messages.append(Message(
                    role=MessageRole.ASSISTANT,
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                ))

                for tool_call in response.tool_calls:
                    tool_obj = tools.get(tool_call.name)
                    if tool_obj:
                        result = await tool_obj.run(**tool_call.arguments)
                        messages.append(Message(
                            role=MessageRole.TOOL,
                            content=result.output if result.success else result.error or "",
                            tool_call_id=tool_call.id,
                        ))
            else:
                return {
                    "output": response.content,
                    "tokens_used": tokens_used,
                    "turns_used": turn + 1,
                }

        return {"output": "Max turns", "tokens_used": tokens_used, "turns_used": max_turns}


# ============================================================================
# Custom Spawner cho Parallel Execution
# ============================================================================

class ParallelDemoSpawner(SubAgentSpawner):
    """Spawner demo ho tro parallel execution."""

    def __init__(self, executor: ParallelMockExecutor, **kwargs):
        super().__init__(**kwargs)
        self.executor = executor

    async def _run_sub_agent_task(
        self,
        config: SpawnConfig,
        task: str,
        lifecycle,
    ) -> Dict[str, Any]:
        """Execute sub-agent task."""
        print(f"    [{config.agent_id}] Starting: {task[:50]}...")
        result = await self.executor.execute_task(
            agent_id=config.agent_id,
            task=task,
            purpose=config.purpose,
            max_turns=config.max_turns,
        )
        print(f"    [{config.agent_id}] Done!")
        return result


# ============================================================================
# Demo Functions
# ============================================================================

async def demo_parallel_spawn():
    """Demo spawn nhieu sub-agents song song."""

    print("\n" + "=" * 70)
    print("Demo: Parallel Sub-Agent Spawning")
    print("Spawn 3 sub-agents dong thoi: researcher, coder, analyst")
    print("=" * 70)

    # Setup
    provider = AnthropicProvider(LLM_CONFIG)
    executor = ParallelMockExecutor(provider)

    spawner = ParallelDemoSpawner(
        executor=executor,
        limits=SpawnLimits(
            max_spawn_depth=5,
            max_concurrent_per_parent=5,  # Cho phep 5 concurrent spawns
            max_global_spawns=20,
        ),
    )

    set_current_spawner(spawner)
    set_spawn_context(depth=0, parent=None)
    set_current_parent_session("demo:parallel:session")

    # Dinh nghia 3 tasks song song
    parallel_tasks = [
        {
            "agent_id": "researcher-1",
            "purpose": "Nghien cuu machine learning papers",
            "task": "Tim 3 papers quan trong nhat ve machine learning",
            "max_turns": 3,
        },
        {
            "agent_id": "coder-1",
            "purpose": "Viet code samples",
            "task": "Viet ham fibonacci bang Python",
            "max_turns": 3,
        },
        {
            "agent_id": "analyst-1",
            "purpose": "Phan tich performance",
            "task": "Phan tich performance cua he thong",
            "max_turns": 3,
        },
    ]

    print("\n[Coordinator] Spawning 3 sub-agents in parallel...")
    start = asyncio.get_event_loop().time()

    # Spawn song song
    results = await spawn_parallel(parallel_tasks)

    duration = (asyncio.get_event_loop().time() - start) * 1000
    print(f"\n[Coordinator] All spawns completed in {duration:.0f}ms")

    # Hien thi ket qua
    print("\n" + "-" * 50)
    print("RESULTS:")
    print("-" * 50)

    total_tokens = 0
    for i, result in enumerate(results):
        task_info = parallel_tasks[i]
        print(f"\n[{task_info['agent_id']}]")
        print(f"  Status: {result.get('status', 'unknown')}")

        if result.get("status") == "completed":
            output = result.get("result", "")
            print(f"  Result: {output[:150]}..." if len(str(output)) > 150 else f"  Result: {output}")
            total_tokens += result.get("tokens_used", 0)
        else:
            print(f"  Error: {result.get('error', 'Unknown')}")

    print(f"\n[Summary] Total tokens used: {total_tokens}")

    # Statistics
    stats = spawner.get_statistics()
    print(f"[Stats] {stats}")


async def demo_sequential_vs_parallel():
    """So sanh hieu nang giua sequential va parallel spawning."""

    print("\n" + "=" * 70)
    print("Demo: Sequential vs Parallel Comparison")
    print("=" * 70)

    provider = AnthropicProvider(LLM_CONFIG)
    executor = ParallelMockExecutor(provider)

    spawner = ParallelDemoSpawner(
        executor=executor,
        limits=SpawnLimits(max_concurrent_per_parent=5),
    )

    set_current_spawner(spawner)
    set_spawn_context(depth=0, parent=None)
    set_current_parent_session("demo:parallel:session")

    tasks = [
        {"agent_id": f"worker-{i}", "purpose": f"Task {i}", "task": f"Analyze data type: performance"}
        for i in range(3)
    ]

    # Sequential spawn
    print("\n1. Sequential spawning (one by one)...")
    start_seq = asyncio.get_event_loop().time()
    seq_results = []
    for task in tasks:
        result = await spawn_agent(**task, max_turns=2)
        seq_results.append(result)
    seq_duration = (asyncio.get_event_loop().time() - start_seq) * 1000

    # Reset spawner
    spawner = ParallelDemoSpawner(
        executor=executor,
        limits=SpawnLimits(max_concurrent_per_parent=5),
    )
    set_current_spawner(spawner)

    # Parallel spawn
    print("\n2. Parallel spawning (all at once)...")
    start_par = asyncio.get_event_loop().time()
    par_results = await spawn_parallel(tasks)
    par_duration = (asyncio.get_event_loop().time() - start_par) * 1000

    # Comparison
    print("\n" + "-" * 50)
    print("COMPARISON:")
    print("-" * 50)
    print(f"  Sequential: {seq_duration:.0f}ms ({len(seq_results)} spawns)")
    print(f"  Parallel:   {par_duration:.0f}ms ({len(par_results)} spawns)")
    if seq_duration > 0:
        speedup = seq_duration / par_duration if par_duration > 0 else float('inf')
        print(f"  Speedup:    {speedup:.1f}x faster with parallel")


async def demo_error_handling():
    """Demo xu ly loi khi mot so spawns that bai."""

    print("\n" + "=" * 70)
    print("Demo: Error Handling in Parallel Spawns")
    print("=" * 70)

    provider = AnthropicProvider(LLM_CONFIG)
    executor = ParallelMockExecutor(provider)

    spawner = ParallelDemoSpawner(
        executor=executor,
        limits=SpawnLimits(
            max_spawn_depth=1,  # Chi cho phep 1 level
            max_concurrent_per_parent=2,  # Chi 2 concurrent
        ),
    )

    set_current_spawner(spawner)
    set_spawn_context(depth=0, parent=None)
    set_current_parent_session("demo:parallel:session")

    # Mix of valid and tasks that might hit limits
    tasks = [
        {"agent_id": "worker-1", "purpose": "Task 1", "task": "Do work 1"},
        {"agent_id": "worker-2", "purpose": "Task 2", "task": "Do work 2"},
        {"agent_id": "worker-3", "purpose": "Task 3", "task": "Do work 3"},  # May hit limit
    ]

    print(f"\nSpawning {len(tasks)} tasks with limit of 2 concurrent...")

    results = await spawn_parallel(tasks)

    print("\nResults:")
    for i, result in enumerate(results):
        status = result.get("status", "unknown")
        print(f"  Task {i+1}: {status}")
        if status == "failed":
            print(f"    Error: {result.get('error', 'Unknown')}")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all demos."""
    print("=" * 70)
    print("PARALLEL SUB-AGENT SPAWNING EXAMPLES")
    print("=" * 70)

    await demo_parallel_spawn()
    await demo_sequential_vs_parallel()
    await demo_error_handling()

    print("\n" + "=" * 70)
    print("All demos completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
