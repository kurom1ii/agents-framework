"""
A2A (Agent-to-Agent) Package.

Package này cung cấp hệ thống giao tiếp Agent-to-Agent, cho phép các agents
tương tác, phối hợp, spawn sub-agents, giao tiếp inter-session và bảo mật trong hệ thống.

Các thành phần chính:
- SessionInfo, AgentInfo: Data classes chứa thông tin về sessions và agents
- SessionStatus, AgentStatus: Enums định nghĩa trạng thái
- SessionDiscovery: Khám phá và liệt kê sessions đang hoạt động
- AgentDiscovery: Khám phá agents có sẵn trong hệ thống
- SubAgentSpawner: Spawn sub-agents động cho tác vụ con
- SpawnedAgentLifecycle: Quản lý lifecycle của spawned agents
- InterSessionMessaging: Giao tiếp giữa các sessions (sync/async)
- MessageRouter: Định tuyến messages (direct, broadcast, topic)
- MessageQueue: Hàng đợi messages cho offline sessions
- sessions_list: Tool cho phép agents khám phá sessions
- sessions_spawn: Tool cho phép agents spawn sub-agents
- sessions_send: Tool gửi tin nhắn đến session khác
- sessions_history: Tool xem lịch sử session khác
- Security: Hệ thống bảo mật và access control cho A2A

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

    # Spawn sub-agents
    from agents_framework.a2a import SubAgentSpawner, SpawnConfig

    spawner = SubAgentSpawner(agent_factory, session_manager)
    config = SpawnConfig(
        agent_id="researcher",
        purpose="Nghiên cứu AI frameworks",
        max_turns=10
    )
    result = await spawner.spawn(
        parent_session="agent:coordinator:main",
        config=config,
        task="Tìm 5 AI frameworks phổ biến"
    )

    # Giao tiếp giữa sessions
    from agents_framework.a2a import InterSessionMessaging, A2AMessage

    messaging = InterSessionMessaging(session_manager, message_queue)
    response = await messaging.send(
        from_session="agent:orchestrator:main",
        to_session="agent:coder:main",
        message="Review PR #123",
        wait_for_response=True
    )

    # Sử dụng tool trong agent
    from agents_framework.a2a import sessions_list, set_current_discovery

    set_current_discovery(discovery)
    sessions = await sessions_list(filter="idle", agent_filter="coder")

    # Sử dụng security và access control
    from agents_framework.a2a import (
        A2AAccessControl,
        A2ASecurityConfig,
        PermissionLevel,
        SpawnSandboxConfig,
    )

    # Tạo cấu hình bảo mật
    config = A2ASecurityConfig(
        allow_incoming=["helper", "researcher"],
        allow_outgoing=["*"],
        history_access={"helper": PermissionLevel.HISTORY},
        spawn_sandbox=SpawnSandboxConfig(max_spawn_depth=2)
    )

    access_control = A2AAccessControl(configs={"main-agent": config})
    if access_control.can_send("helper", "main-agent"):
        # Gửi message
        pass
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
from .messaging import (
    A2AMessage,
    A2AResponse,
    MessagePriority,
    MessageState,
    MessageType,
    SessionHistory,
)
from .queue import (
    BaseMessageQueue,
    FileMessageQueue,
    InMemoryMessageQueue,
    MessageQueue,
    MessageQueueProtocol,
    QueueStats,
)
from .router import (
    InterSessionMessaging,
    MessageHandlerProtocol,
    MessageRouter,
    RoutingResult,
    TopicRegistry,
)
from .lifecycle import (
    LifecycleEvent,
    LifecycleManager,
    LifecycleMetrics,
    SpawnedAgentLifecycle,
    SpawnLifecycleState,
)
from .spawner import (
    SpawnConfig,
    SpawnLimits,
    SpawnResult,
    SpawnStatus,
    SubAgentSpawner,
    get_current_spawner,
    get_parent_session,
    get_spawn_depth,
    set_current_spawner,
    set_spawn_context,
)
from .tools.sessions_list import (
    create_sessions_list_tool,
    get_current_discovery,
    sessions_list,
    set_current_discovery,
    SESSIONS_LIST_TOOL_DEFINITION,
)
from .tools.sessions_spawn import (
    create_all_spawn_tools,
    create_sessions_spawn_cancel_tool,
    create_sessions_spawn_list_tool,
    create_sessions_spawn_status_tool,
    create_sessions_spawn_tool,
    get_current_parent_session_key,
    sessions_spawn,
    sessions_spawn_cancel,
    sessions_spawn_list,
    sessions_spawn_status,
    set_current_parent_session,
    SESSIONS_SPAWN_CANCEL_TOOL_DEFINITION,
    SESSIONS_SPAWN_LIST_TOOL_DEFINITION,
    SESSIONS_SPAWN_STATUS_TOOL_DEFINITION,
    SESSIONS_SPAWN_TOOL_DEFINITION,
)
from .tools.sessions_send import (
    create_sessions_send_tool,
    get_current_messaging,
    get_current_session_key,
    sessions_send,
    set_current_messaging,
    set_current_session_key,
    SESSIONS_SEND_TOOL_DEFINITION,
)
from .tools.sessions_history import (
    create_sessions_history_tool,
    get_current_transcript_store,
    get_history_discovery,
    get_history_session_key,
    sessions_history,
    set_current_transcript_store,
    set_history_discovery,
    set_history_session_key,
    SESSIONS_HISTORY_TOOL_DEFINITION,
)
from .security import (
    # Permission system
    A2APermission,
    PermissionLevel,
    PermissionManager,
    # Security config
    A2ASecurityConfig,
    # Access control
    A2AAccessControl,
    # Sandbox
    NetworkAccess,
    SandboxMode,
    SpawnSandbox,
    SpawnSandboxConfig,
    WorkspaceAccess,
    # Audit logging
    A2AAuditLog,
    AuditEntry,
    AuditEventType,
    AuditSeverity,
    AuditStorageProtocol,
)
from .policies import (
    PolicyAction,
    PolicyManager,
    PolicyPriority,
    PolicyRule,
    RolePermission,
    SecurityPolicy,
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
    # Messaging - Message types
    "A2AMessage",
    "A2AResponse",
    "MessageType",
    "MessagePriority",
    "MessageState",
    "SessionHistory",
    # Messaging - Queue
    "MessageQueueProtocol",
    "BaseMessageQueue",
    "InMemoryMessageQueue",
    "FileMessageQueue",
    "MessageQueue",
    "QueueStats",
    # Messaging - Router
    "MessageHandlerProtocol",
    "MessageRouter",
    "InterSessionMessaging",
    "RoutingResult",
    "TopicRegistry",
    # Spawner classes
    "SubAgentSpawner",
    "SpawnConfig",
    "SpawnResult",
    "SpawnStatus",
    "SpawnLimits",
    # Lifecycle classes
    "SpawnedAgentLifecycle",
    "SpawnLifecycleState",
    "LifecycleEvent",
    "LifecycleMetrics",
    "LifecycleManager",
    # Tools - sessions_list
    "sessions_list",
    "create_sessions_list_tool",
    "SESSIONS_LIST_TOOL_DEFINITION",
    # Tools - sessions_spawn
    "sessions_spawn",
    "sessions_spawn_status",
    "sessions_spawn_cancel",
    "sessions_spawn_list",
    "create_sessions_spawn_tool",
    "create_sessions_spawn_status_tool",
    "create_sessions_spawn_cancel_tool",
    "create_sessions_spawn_list_tool",
    "create_all_spawn_tools",
    "SESSIONS_SPAWN_TOOL_DEFINITION",
    "SESSIONS_SPAWN_STATUS_TOOL_DEFINITION",
    "SESSIONS_SPAWN_CANCEL_TOOL_DEFINITION",
    "SESSIONS_SPAWN_LIST_TOOL_DEFINITION",
    # Tools - sessions_send
    "sessions_send",
    "create_sessions_send_tool",
    "SESSIONS_SEND_TOOL_DEFINITION",
    # Tools - sessions_history
    "sessions_history",
    "create_sessions_history_tool",
    "SESSIONS_HISTORY_TOOL_DEFINITION",
    # Context utilities
    "get_current_discovery",
    "set_current_discovery",
    "get_current_spawner",
    "set_current_spawner",
    "get_spawn_depth",
    "get_parent_session",
    "set_spawn_context",
    "get_current_parent_session_key",
    "set_current_parent_session",
    "get_current_messaging",
    "set_current_messaging",
    "get_current_session_key",
    "set_current_session_key",
    "get_current_transcript_store",
    "set_current_transcript_store",
    "get_history_discovery",
    "set_history_discovery",
    "get_history_session_key",
    "set_history_session_key",
    # Security - Permission system
    "PermissionLevel",
    "A2APermission",
    "PermissionManager",
    # Security - Config
    "A2ASecurityConfig",
    # Security - Access control
    "A2AAccessControl",
    # Security - Sandbox
    "SandboxMode",
    "WorkspaceAccess",
    "NetworkAccess",
    "SpawnSandboxConfig",
    "SpawnSandbox",
    # Security - Audit logging
    "AuditEventType",
    "AuditSeverity",
    "AuditEntry",
    "AuditStorageProtocol",
    "A2AAuditLog",
    # Security - Policies
    "PolicyAction",
    "PolicyPriority",
    "PolicyRule",
    "RolePermission",
    "SecurityPolicy",
    "PolicyManager",
]
