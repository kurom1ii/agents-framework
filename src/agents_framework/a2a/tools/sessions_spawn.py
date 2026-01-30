"""
Tool sessions_spawn cho phép agents spawn sub-agents.

Tool này là một phần của hệ thống A2A (Agent-to-Agent), cho phép agents
tạo sub-agents động để xử lý các tác vụ con với isolated sessions.

Các tính năng:
- Spawn sub-agents với cấu hình tùy chỉnh
- Resource limits (tokens, time, tools)
- Isolated sessions
- Auto-cleanup và report back
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Dict, List, Optional

from ..spawner import (
    SpawnConfig,
    SpawnResult,
    SpawnStatus,
    SubAgentSpawner,
    get_current_spawner,
    get_parent_session,
    get_spawn_depth,
)


# Context variable để lưu parent session cho tool
_current_parent_session: ContextVar[Optional[str]] = ContextVar(
    "current_parent_session", default=None
)


def set_current_parent_session(session_key: str) -> None:
    """
    Đặt parent session cho context hiện tại.

    Hàm này được gọi bởi runtime để inject parent session
    trước khi tool được thực thi.

    Args:
        session_key: Session key của parent agent
    """
    _current_parent_session.set(session_key)


def get_current_parent_session_key() -> str:
    """
    Lấy parent session key từ context.

    Returns:
        Parent session key

    Raises:
        RuntimeError: Nếu chưa có parent session trong context
    """
    session_key = _current_parent_session.get()
    if session_key is None:
        # Fallback to spawn context
        fallback = get_parent_session()
        if fallback:
            return fallback
        raise RuntimeError(
            "Parent session chưa được set trong context. "
            "Hãy đảm bảo set_current_parent_session() đã được gọi."
        )
    return session_key


# Tool definition metadata
SESSIONS_SPAWN_TOOL_DEFINITION = {
    "name": "sessions_spawn",
    "description": """Tạo sub-agent để xử lý tác vụ.

Tool này cho phép agent tạo sub-agents động để xử lý các tác vụ con.
Sub-agent sẽ có isolated session và bị giới hạn bởi resource limits.

Các trường hợp sử dụng:
- Ủy quyền tác vụ phụ cho sub-agent chuyên biệt
- Song song hóa công việc với nhiều sub-agents
- Tách biệt context cho các tác vụ khác nhau
- Giới hạn resources cho tác vụ con

Lưu ý:
- Có giới hạn spawn depth (mặc định 2 levels)
- Có giới hạn concurrent spawns (mặc định 5 per parent)
- Sub-agent sẽ tự cleanup sau khi hoàn thành""",
    "parameters": {
        "type": "object",
        "properties": {
            "agent_config": {
                "type": "object",
                "description": """Cấu hình cho sub-agent:
- id: ID duy nhất cho sub-agent (required)
- purpose: Mục đích/mô tả của sub-agent (required)
- tools: Danh sách tools được phép (optional)
- model: Model LLM sử dụng (optional)""",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "ID cho sub-agent",
                    },
                    "purpose": {
                        "type": "string",
                        "description": "Mục đích của sub-agent",
                    },
                    "tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Danh sách tools được phép",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model sử dụng (optional)",
                    },
                },
                "required": ["id", "purpose"],
            },
            "task": {
                "type": "string",
                "description": "Tác vụ cần sub-agent thực hiện",
            },
            "isolated": {
                "type": "boolean",
                "default": True,
                "description": "Tạo session riêng cho sub-agent",
            },
            "report_back": {
                "type": "boolean",
                "default": True,
                "description": "Sub-agent báo cáo kết quả về parent",
            },
            "max_turns": {
                "type": "integer",
                "default": 10,
                "minimum": 1,
                "maximum": 50,
                "description": "Số turns tối đa cho sub-agent",
            },
            "timeout": {
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
        "required": ["agent_config", "task"],
    },
}


async def sessions_spawn(
    agent_config: Dict[str, Any],
    task: str,
    isolated: bool = True,
    report_back: bool = True,
    max_turns: int = 10,
    timeout: int = 300000,
    max_tokens: int = 50000,
) -> Dict[str, Any]:
    """
    Tạo sub-agent để xử lý tác vụ.

    Tool này cho phép agent tạo sub-agents động để xử lý
    các tác vụ con với isolated sessions và resource limits.

    Args:
        agent_config: Cấu hình cho sub-agent
            - id: ID cho sub-agent (required)
            - purpose: Mục đích của sub-agent (required)
            - tools: Danh sách tools được phép (optional)
            - model: Model sử dụng (optional)
        task: Tác vụ cần thực hiện
        isolated: Tạo session riêng cho sub-agent (default: True)
        report_back: Báo cáo kết quả về parent (default: True)
        max_turns: Số turns tối đa (1-50, default: 10)
        timeout: Timeout thực thi in milliseconds (1000-600000, default: 300000)
        max_tokens: Token budget (1000-200000, default: 50000)

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
        result = await sessions_spawn(
            agent_config={
                "id": "researcher",
                "purpose": "Nghiên cứu về AI frameworks",
                "tools": ["web_search", "web_fetch"]
            },
            task="Tìm và tóm tắt 5 AI frameworks phổ biến nhất",
            max_turns=10,
            timeout=300000
        )

        if result["status"] == "completed":
            summary = result["result"]
            print(f"Nghiên cứu hoàn thành: {summary}")
        else:
            print(f"Lỗi: {result['error']}")

        # Spawn nhiều agents song song
        import asyncio

        tasks = [
            sessions_spawn(
                agent_config={"id": "researcher-1", "purpose": "Research A"},
                task="Research topic A"
            ),
            sessions_spawn(
                agent_config={"id": "researcher-2", "purpose": "Research B"},
                task="Research topic B"
            )
        ]
        results = await asyncio.gather(*tasks)

    Ghi chú:
        - Có giới hạn spawn depth (mặc định 2 levels) để ngăn infinite spawning
        - Có giới hạn concurrent spawns per parent (mặc định 5)
        - Sub-agent sẽ tự cleanup sau khi hoàn thành
        - Sử dụng isolated=False để share session với parent (cẩn thận với context)
    """
    # Validate inputs
    max_turns = max(1, min(50, max_turns))
    timeout = max(1000, min(600000, timeout))
    max_tokens = max(1000, min(200000, max_tokens))

    # Validate agent_config
    if "id" not in agent_config:
        raise ValueError("agent_config phải có 'id'")
    if "purpose" not in agent_config:
        raise ValueError("agent_config phải có 'purpose'")

    # Lấy spawner từ context
    spawner = get_current_spawner()

    # Lấy parent session
    parent_session = get_current_parent_session_key()

    # Lấy spawn depth hiện tại
    current_depth = get_spawn_depth()

    # Tạo SpawnConfig
    config = SpawnConfig(
        agent_id=agent_config["id"],
        purpose=agent_config["purpose"],
        model=agent_config.get("model"),
        tools=agent_config.get("tools"),
        max_turns=max_turns,
        timeout_ms=timeout,
        max_tokens=max_tokens,
        isolated=isolated,
        report_back=report_back,
    )

    # Spawn sub-agent
    result = await spawner.spawn(
        parent_session=parent_session,
        config=config,
        task=task,
    )

    # Convert to dict for return
    return result.to_dict()


async def sessions_spawn_status(
    session_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Lấy trạng thái của một spawn.

    Args:
        session_id: ID của session spawn

    Returns:
        Dictionary chứa trạng thái hoặc None nếu không tìm thấy

    Example:
        status = await sessions_spawn_status("abc123")
        if status and status["status"] == "running":
            print("Spawn đang chạy...")
    """
    spawner = get_current_spawner()
    result = await spawner.get_spawn_status(session_id)

    if result:
        return result.to_dict()
    return None


async def sessions_spawn_cancel(
    session_id: str,
) -> bool:
    """
    Hủy một spawn đang chạy.

    Args:
        session_id: ID của session spawn cần hủy

    Returns:
        True nếu hủy thành công

    Example:
        success = await sessions_spawn_cancel("abc123")
        if success:
            print("Đã hủy spawn")
    """
    spawner = get_current_spawner()
    return await spawner.cancel_spawn(session_id)


async def sessions_spawn_list(
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Liệt kê các spawns của agent hiện tại.

    Args:
        status: Lọc theo status (running/completed/failed/etc)

    Returns:
        Danh sách spawn results

    Example:
        # Lấy tất cả spawns đang chạy
        running = await sessions_spawn_list(status="running")
        for spawn in running:
            print(f"{spawn['session_id']}: {spawn['status']}")
    """
    spawner = get_current_spawner()
    parent_session = get_current_parent_session_key()

    # Convert status string to enum
    spawn_status = None
    if status:
        try:
            spawn_status = SpawnStatus(status)
        except ValueError:
            pass

    results = await spawner.list_spawned(
        parent_session=parent_session,
        status=spawn_status,
    )

    return [r.to_dict() for r in results]


# Additional tool definitions for status and cancel
SESSIONS_SPAWN_STATUS_TOOL_DEFINITION = {
    "name": "sessions_spawn_status",
    "description": """Lấy trạng thái của một spawn đang chạy.

Sử dụng tool này để kiểm tra tiến độ của sub-agent đã spawn.""",
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "ID của session spawn cần kiểm tra",
            },
        },
        "required": ["session_id"],
    },
}

SESSIONS_SPAWN_CANCEL_TOOL_DEFINITION = {
    "name": "sessions_spawn_cancel",
    "description": """Hủy một spawn đang chạy.

Sử dụng tool này để dừng sub-agent nếu không còn cần thiết.""",
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "ID của session spawn cần hủy",
            },
        },
        "required": ["session_id"],
    },
}

SESSIONS_SPAWN_LIST_TOOL_DEFINITION = {
    "name": "sessions_spawn_list",
    "description": """Liệt kê các spawns của agent hiện tại.

Sử dụng tool này để xem tất cả sub-agents đã spawn.""",
    "parameters": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["pending", "running", "completed", "failed", "timeout", "cancelled"],
                "description": "Lọc theo status (optional)",
            },
        },
        "required": [],
    },
}


def create_sessions_spawn_tool():
    """
    Tạo FunctionTool instance cho sessions_spawn.

    Returns:
        FunctionTool instance đã được cấu hình

    Example:
        from agents_framework.a2a.tools import create_sessions_spawn_tool

        tool = create_sessions_spawn_tool()
        registry.register(tool)
    """
    from ...tools.base import FunctionTool

    return FunctionTool(
        func=sessions_spawn,
        name="sessions_spawn",
        description=SESSIONS_SPAWN_TOOL_DEFINITION["description"],
    )


def create_sessions_spawn_status_tool():
    """
    Tạo FunctionTool instance cho sessions_spawn_status.

    Returns:
        FunctionTool instance
    """
    from ...tools.base import FunctionTool

    return FunctionTool(
        func=sessions_spawn_status,
        name="sessions_spawn_status",
        description=SESSIONS_SPAWN_STATUS_TOOL_DEFINITION["description"],
    )


def create_sessions_spawn_cancel_tool():
    """
    Tạo FunctionTool instance cho sessions_spawn_cancel.

    Returns:
        FunctionTool instance
    """
    from ...tools.base import FunctionTool

    return FunctionTool(
        func=sessions_spawn_cancel,
        name="sessions_spawn_cancel",
        description=SESSIONS_SPAWN_CANCEL_TOOL_DEFINITION["description"],
    )


def create_sessions_spawn_list_tool():
    """
    Tạo FunctionTool instance cho sessions_spawn_list.

    Returns:
        FunctionTool instance
    """
    from ...tools.base import FunctionTool

    return FunctionTool(
        func=sessions_spawn_list,
        name="sessions_spawn_list",
        description=SESSIONS_SPAWN_LIST_TOOL_DEFINITION["description"],
    )


def create_all_spawn_tools() -> List:
    """
    Tạo tất cả spawn-related tools.

    Returns:
        List of FunctionTool instances

    Example:
        tools = create_all_spawn_tools()
        for tool in tools:
            registry.register(tool)
    """
    return [
        create_sessions_spawn_tool(),
        create_sessions_spawn_status_tool(),
        create_sessions_spawn_cancel_tool(),
        create_sessions_spawn_list_tool(),
    ]
