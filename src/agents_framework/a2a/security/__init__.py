"""
A2A Security Package.

Package này cung cấp hệ thống bảo mật và kiểm soát quyền truy cập
cho giao tiếp Agent-to-Agent (A2A) trong agents framework.

Các thành phần chính:
- PermissionLevel: Các cấp độ quyền (none, notify, request, history, spawn, full)
- A2APermission: Định nghĩa permission giữa hai agents
- PermissionManager: Quản lý và kiểm tra permissions
- A2ASecurityConfig: Cấu hình bảo mật cho từng agent
- A2AAccessControl: Hệ thống kiểm soát truy cập tổng thể
- SpawnSandboxConfig: Cấu hình sandbox cho spawned agents
- SpawnSandbox: Runtime sandbox để enforce restrictions
- A2AAuditLog: Audit logging cho A2A communications

Example:
    # Cấu hình access control
    from agents_framework.a2a.security import (
        A2AAccessControl,
        A2ASecurityConfig,
        PermissionLevel,
        SpawnSandboxConfig,
    )

    # Tạo cấu hình cho main-agent
    main_config = A2ASecurityConfig(
        allow_incoming=["helper", "researcher"],
        allow_outgoing=["*"],
        history_access={
            "helper": PermissionLevel.HISTORY,
            "*": PermissionLevel.NONE
        },
        spawn_sandbox=SpawnSandboxConfig(
            allowed_tools=["read", "write"],
            max_spawn_depth=2
        )
    )

    # Tạo access control
    access_control = A2AAccessControl(
        configs={"main-agent": main_config}
    )

    # Kiểm tra quyền
    if access_control.can_send("helper", "main-agent"):
        print("Helper có thể gửi message đến main-agent")

    # Tạo sandbox cho spawned agent
    sandbox = access_control.create_sandbox("main-agent", spawn_depth=1)
    if sandbox and sandbox.can_use_tool("read"):
        print("Sub-agent có thể sử dụng tool read")
"""

from .access_control import (
    A2AAccessControl,
    A2ASecurityConfig,
)
from .audit import (
    A2AAuditLog,
    AuditEntry,
    AuditEventType,
    AuditSeverity,
    AuditStorageProtocol,
)
from .permissions import (
    A2APermission,
    PermissionLevel,
    PermissionManager,
)
from .sandbox import (
    NetworkAccess,
    SandboxMode,
    SpawnSandbox,
    SpawnSandboxConfig,
    WorkspaceAccess,
)

__all__ = [
    # Permission system
    "PermissionLevel",
    "A2APermission",
    "PermissionManager",
    # Security config
    "A2ASecurityConfig",
    # Access control
    "A2AAccessControl",
    # Sandbox
    "SandboxMode",
    "WorkspaceAccess",
    "NetworkAccess",
    "SpawnSandboxConfig",
    "SpawnSandbox",
    # Audit logging
    "AuditEventType",
    "AuditSeverity",
    "AuditEntry",
    "AuditStorageProtocol",
    "A2AAuditLog",
]
