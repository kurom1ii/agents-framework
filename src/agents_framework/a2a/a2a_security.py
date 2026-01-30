"""
A2A Security Module - Main Entry Point.

Module này cung cấp các classes và interfaces chính cho hệ thống bảo mật A2A.
Đây là điểm entry point đơn giản để sử dụng security features.

Re-exports các components từ security/ package.
"""

from __future__ import annotations

# Re-export tất cả từ security package
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
