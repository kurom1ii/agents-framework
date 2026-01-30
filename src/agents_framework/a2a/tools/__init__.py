"""
A2A Tools Package.

Package này chứa các tools cho phép agents tương tác với hệ thống A2A,
bao gồm khám phá sessions, agents, spawn sub-agents, và inter-session messaging.

Các tools có sẵn:
- sessions_list: Liệt kê các sessions đang hoạt động
- sessions_spawn: Spawn sub-agent để xử lý tác vụ
- sessions_spawn_status: Kiểm tra trạng thái spawn
- sessions_spawn_cancel: Hủy spawn đang chạy
- sessions_spawn_list: Liệt kê các spawns của agent
- sessions_send: Gửi tin nhắn đến session khác
- sessions_history: Truy cập lịch sử session khác

Example:
    from agents_framework.a2a.tools import (
        sessions_list,
        sessions_spawn,
        sessions_send,
        sessions_history,
        create_all_spawn_tools,
    )

    # Liệt kê sessions
    sessions = await sessions_list(filter="idle")

    # Spawn sub-agent
    result = await sessions_spawn(
        agent_config={"id": "researcher", "purpose": "Research task"},
        task="Research AI frameworks"
    )

    # Gửi tin nhắn đến session khác
    response = await sessions_send(
        session_key="agent:coder:main",
        message="Review PR #123",
        wait_for_response=True
    )

    # Xem lịch sử session
    history = await sessions_history(
        session_key="agent:researcher:main",
        limit=10,
        format="summary"
    )
"""

from .sessions_list import (
    sessions_list,
    create_sessions_list_tool,
    SESSIONS_LIST_TOOL_DEFINITION,
    get_current_discovery,
    set_current_discovery,
)
from .sessions_spawn import (
    sessions_spawn,
    sessions_spawn_status,
    sessions_spawn_cancel,
    sessions_spawn_list,
    create_sessions_spawn_tool,
    create_sessions_spawn_status_tool,
    create_sessions_spawn_cancel_tool,
    create_sessions_spawn_list_tool,
    create_all_spawn_tools,
    set_current_parent_session,
    get_current_parent_session_key,
    SESSIONS_SPAWN_TOOL_DEFINITION,
    SESSIONS_SPAWN_STATUS_TOOL_DEFINITION,
    SESSIONS_SPAWN_CANCEL_TOOL_DEFINITION,
    SESSIONS_SPAWN_LIST_TOOL_DEFINITION,
)
from .sessions_send import (
    sessions_send,
    create_sessions_send_tool,
    SESSIONS_SEND_TOOL_DEFINITION,
    get_current_messaging,
    set_current_messaging,
    get_current_session_key,
    set_current_session_key,
)
from .sessions_history import (
    sessions_history,
    create_sessions_history_tool,
    SESSIONS_HISTORY_TOOL_DEFINITION,
    get_current_transcript_store,
    set_current_transcript_store,
    get_history_discovery,
    set_history_discovery,
    get_history_session_key,
    set_history_session_key,
)

__all__ = [
    # sessions_list
    "sessions_list",
    "create_sessions_list_tool",
    "SESSIONS_LIST_TOOL_DEFINITION",
    "get_current_discovery",
    "set_current_discovery",
    # sessions_spawn
    "sessions_spawn",
    "sessions_spawn_status",
    "sessions_spawn_cancel",
    "sessions_spawn_list",
    "create_sessions_spawn_tool",
    "create_sessions_spawn_status_tool",
    "create_sessions_spawn_cancel_tool",
    "create_sessions_spawn_list_tool",
    "create_all_spawn_tools",
    "set_current_parent_session",
    "get_current_parent_session_key",
    "SESSIONS_SPAWN_TOOL_DEFINITION",
    "SESSIONS_SPAWN_STATUS_TOOL_DEFINITION",
    "SESSIONS_SPAWN_CANCEL_TOOL_DEFINITION",
    "SESSIONS_SPAWN_LIST_TOOL_DEFINITION",
    # sessions_send
    "sessions_send",
    "create_sessions_send_tool",
    "SESSIONS_SEND_TOOL_DEFINITION",
    "get_current_messaging",
    "set_current_messaging",
    "get_current_session_key",
    "set_current_session_key",
    # sessions_history
    "sessions_history",
    "create_sessions_history_tool",
    "SESSIONS_HISTORY_TOOL_DEFINITION",
    "get_current_transcript_store",
    "set_current_transcript_store",
    "get_history_discovery",
    "set_history_discovery",
    "get_history_session_key",
    "set_history_session_key",
]
