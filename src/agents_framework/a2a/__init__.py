"""
A2A (Agent-to-Agent) Package.

Package này cung cấp hệ thống giao tiếp Agent-to-Agent, cho phép các agents
tương tác và phối hợp với nhau trong hệ thống.

Các thành phần chính:
- SessionInfo, AgentInfo: Data classes chứa thông tin về sessions và agents
- SessionStatus, AgentStatus: Enums định nghĩa trạng thái
- SessionDiscovery: Khám phá và liệt kê sessions đang hoạt động
- AgentDiscovery: Khám phá agents có sẵn trong hệ thống
- sessions_list: Tool cho phép agents khám phá sessions

Example:
    # Khám phá sessions
    from agents_framework.a2a import SessionDiscovery, SessionStatus

    discovery = SessionDiscovery(session_manager)
    active_sessions = await discovery.list_active_sessions(minutes=60)
    idle_sessions = await discovery.list_sessions(filter_status=SessionStatus.IDLE)

    # Khám phá agents
    from agents_framework.a2a import AgentDiscovery

    agent_discovery = AgentDiscovery(agent_registry)
    coders = agent_discovery.find_by_capability("coding")
    available = agent_discovery.find_available()

    # Sử dụng tool trong agent
    from agents_framework.a2a import sessions_list, set_current_discovery

    set_current_discovery(discovery)
    sessions = await sessions_list(filter="idle", agent_filter="coder")
"""

from .base import (
    AgentInfo,
    AgentStatus,
    SessionInfo,
    SessionStatus,
)
from .discovery import (
    AgentDiscovery,
    AgentRegistryProtocol,
    SessionDiscovery,
    SessionManagerProtocol,
)
from .tools.sessions_list import (
    create_sessions_list_tool,
    get_current_discovery,
    sessions_list,
    set_current_discovery,
    SESSIONS_LIST_TOOL_DEFINITION,
)

__all__ = [
    # Base classes
    "SessionInfo",
    "AgentInfo",
    "SessionStatus",
    "AgentStatus",
    # Discovery classes
    "SessionDiscovery",
    "AgentDiscovery",
    # Protocols
    "SessionManagerProtocol",
    "AgentRegistryProtocol",
    # Tools
    "sessions_list",
    "create_sessions_list_tool",
    "SESSIONS_LIST_TOOL_DEFINITION",
    # Context utilities
    "get_current_discovery",
    "set_current_discovery",
]
