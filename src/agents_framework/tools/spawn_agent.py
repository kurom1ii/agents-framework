"""
Spawn Agent Tool - Công cụ để agents spawn sub-agents.

Module này cung cấp spawn_agent tool, một wrapper tiện lợi cho
sessions_spawn từ A2A package. Tool này cho phép agents dễ dàng
tạo sub-agents để xử lý các tác vụ con.

Các tính năng:
- Spawn sub-agents với cấu hình đơn giản
- Resource limits (tokens, time, tools)
- Isolated sessions
- Auto-cleanup và report back

Example:
    from agents_framework.tools.spawn_agent import spawn_agent

    # Spawn một sub-agent
    result = await spawn_agent(
        agent_id="researcher",
        purpose="Nghiên cứu AI frameworks",
        task="Tìm 5 AI frameworks phổ biến nhất",
        tools=["web_search", "web_fetch"],
        max_turns=10
    )

    if result["status"] == "completed":
        print(f"Kết quả: {result['result']}")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..a2a.spawner import (
    SpawnConfig,
    SpawnResult,
    get_current_spawner,
)
from ..a2a.tools.sessions_spawn import (
    get_current_parent_session_key,
)


# Tool definition metadata
SPAWN_AGENT_TOOL_DEFINITION = {
    "name": "spawn_agent",
    "description": """Tạo sub-agent để xử lý tác vụ.

Tool này là wrapper đơn giản cho việc spawn sub-agents.
Sub-agent sẽ có isolated session và bị giới hạn bởi resource limits.

Sử dụng tool này khi cần:
- Ủy quyền tác vụ phụ cho sub-agent chuyên biệt
- Song song hóa công việc với nhiều sub-agents
- Tách biệt context cho các tác vụ khác nhau

Lưu ý:
- Có giới hạn spawn depth (mặc định 2 levels)
- Có giới hạn concurrent spawns (mặc định 5 per parent)
- Sub-agent sẽ tự cleanup sau khi hoàn thành""",
    "parameters": {
        "type": "object",
        "properties": {
            "agent_id": {
                "type": "string",
                "description": "ID duy nhất cho sub-agent",
            },
            "purpose": {
                "type": "string",
                "description": "Mục đích/mô tả của sub-agent",
            },
            "task": {
                "type": "string",
                "description": "Tác vụ cần sub-agent thực hiện",
            },
            "tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Danh sách tools được phép (optional)",
            },
            "model": {
                "type": "string",
                "description": "Model LLM sử dụng (optional)",
            },
            "max_turns": {
                "type": "integer",
                "default": 10,
                "minimum": 1,
                "maximum": 50,
                "description": "Số turns tối đa cho sub-agent",
            },
            "timeout_ms": {
                "type": "integer",
                "default": 300000,
                "minimum": 1000,
                "maximum": 600000,
                "description": "Timeout thực thi (milliseconds)",
            },
            "max_tokens": {
                "type": "integer",
                "default": 50000,
                "minimum": 1000,
                "maximum": 200000,
                "description": "Token budget cho sub-agent",
            },
        },
        "required": ["agent_id", "purpose", "task"],
    },
}


async def spawn_agent(
    agent_id: str,
    purpose: str,
    task: str,
    tools: Optional[List[str]] = None,
    model: Optional[str] = None,
    max_turns: int = 10,
    timeout_ms: int = 300000,
    max_tokens: int = 50000,
) -> Dict[str, Any]:
    """
    Tạo sub-agent để xử lý tác vụ.

    Tool này là wrapper đơn giản cho việc spawn sub-agents,
    cung cấp interface trực tiếp và dễ sử dụng.

    Args:
        agent_id: ID duy nhất cho sub-agent
        purpose: Mục đích/mô tả của sub-agent
        task: Tác vụ cần thực hiện
        tools: Danh sách tools được phép (None để dùng tất cả)
        model: Model LLM sử dụng (None để dùng mặc định)
        max_turns: Số turns tối đa (1-50, default: 10)
        timeout_ms: Timeout thực thi in milliseconds (default: 300000)
        max_tokens: Token budget (default: 50000)

    Returns:
        Dictionary chứa kết quả spawn:
        - session_id: ID của session được tạo
        - session_key: Key để truy cập session
        - status: Trạng thái thực thi (completed/failed/timeout/cancelled)
        - result: Kết quả từ sub-agent (nếu thành công)
        - tokens_used: Số tokens đã sử dụng
        - duration_ms: Thời gian thực thi
        - error: Thông báo lỗi (nếu có)
        - spawn_depth: Độ sâu spawn chain

    Raises:
        RuntimeError: Nếu SubAgentSpawner chưa được khởi tạo
        ValueError: Nếu vượt quá giới hạn spawn

    Example:
        # Spawn một researcher agent
        result = await spawn_agent(
            agent_id="researcher",
            purpose="Nghiên cứu về AI frameworks",
            task="Tìm và tóm tắt 5 AI frameworks phổ biến nhất",
            tools=["web_search", "web_fetch"],
            max_turns=10,
            timeout_ms=300000
        )

        if result["status"] == "completed":
            summary = result["result"]
            print(f"Nghiên cứu hoàn thành: {summary}")
        else:
            print(f"Lỗi: {result['error']}")

        # Spawn một code writer
        result = await spawn_agent(
            agent_id="coder",
            purpose="Viết code Python",
            task="Viết function tính fibonacci",
            tools=["read_file", "write_file"],
            max_tokens=30000
        )

    Ghi chú:
        - Có giới hạn spawn depth (mặc định 2 levels)
        - Có giới hạn concurrent spawns per parent (mặc định 5)
        - Sub-agent sẽ tự cleanup sau khi hoàn thành
    """
    # Validate inputs
    max_turns = max(1, min(50, max_turns))
    timeout_ms = max(1000, min(600000, timeout_ms))
    max_tokens = max(1000, min(200000, max_tokens))

    # Lấy spawner từ context
    spawner = get_current_spawner()

    # Lấy parent session
    parent_session = get_current_parent_session_key()

    # Tạo SpawnConfig
    config = SpawnConfig(
        agent_id=agent_id,
        purpose=purpose,
        model=model,
        tools=tools,
        max_turns=max_turns,
        timeout_ms=timeout_ms,
        max_tokens=max_tokens,
        isolated=True,
        report_back=True,
    )

    # Spawn sub-agent
    result = await spawner.spawn(
        parent_session=parent_session,
        config=config,
        task=task,
    )

    # Convert to dict for return
    return result.to_dict()


async def spawn_parallel(
    tasks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Spawn nhiều sub-agents song song.

    Tiện ích để spawn nhiều sub-agents cùng lúc và chờ
    tất cả hoàn thành.

    Args:
        tasks: Danh sách các spawn configs, mỗi config có:
            - agent_id: ID cho sub-agent
            - purpose: Mục đích
            - task: Tác vụ cần thực hiện
            - tools, model, max_turns, timeout_ms, max_tokens (optional)

    Returns:
        Danh sách spawn results theo thứ tự

    Example:
        results = await spawn_parallel([
            {
                "agent_id": "researcher-1",
                "purpose": "Research A",
                "task": "Research topic A"
            },
            {
                "agent_id": "researcher-2",
                "purpose": "Research B",
                "task": "Research topic B"
            }
        ])

        for result in results:
            print(f"{result['session_id']}: {result['status']}")
    """
    import asyncio

    async def spawn_one(config: Dict[str, Any]) -> Dict[str, Any]:
        return await spawn_agent(
            agent_id=config["agent_id"],
            purpose=config["purpose"],
            task=config["task"],
            tools=config.get("tools"),
            model=config.get("model"),
            max_turns=config.get("max_turns", 10),
            timeout_ms=config.get("timeout_ms", 300000),
            max_tokens=config.get("max_tokens", 50000),
        )

    results = await asyncio.gather(
        *[spawn_one(config) for config in tasks],
        return_exceptions=True
    )

    # Convert exceptions to error results
    processed_results: List[Dict[str, Any]] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "session_id": "",
                "session_key": "",
                "status": "failed",
                "error": str(result),
                "tokens_used": 0,
                "duration_ms": 0,
            })
        else:
            processed_results.append(result)

    return processed_results


def create_spawn_agent_tool():
    """
    Tạo FunctionTool instance cho spawn_agent.

    Returns:
        FunctionTool instance đã được cấu hình

    Example:
        from agents_framework.tools.spawn_agent import create_spawn_agent_tool

        tool = create_spawn_agent_tool()
        registry.register(tool)
    """
    from .base import FunctionTool

    return FunctionTool(
        func=spawn_agent,
        name="spawn_agent",
        description=SPAWN_AGENT_TOOL_DEFINITION["description"],
    )


def create_spawn_parallel_tool():
    """
    Tạo FunctionTool instance cho spawn_parallel.

    Returns:
        FunctionTool instance
    """
    from .base import FunctionTool

    return FunctionTool(
        func=spawn_parallel,
        name="spawn_parallel",
        description="Spawn nhiều sub-agents song song và chờ tất cả hoàn thành.",
    )
