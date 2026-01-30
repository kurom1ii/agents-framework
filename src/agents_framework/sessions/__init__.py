"""
Sessions Package - Quản lý Sessions.

Package này cung cấp hệ thống quản lý sessions cho agents,
bao gồm tạo, lưu trữ và theo dõi trạng thái của sessions.

Các thành phần chính:
- Session: Data class đại diện cho một session
- SessionScope: Phạm vi session (main, per_peer, per_context)
- SessionState: Trạng thái session (active, idle, expired, archived)
- SessionConfig: Cấu hình session
- SessionManager: Quản lý lifecycle của sessions
- SessionStore: Interface lưu trữ sessions
- SessionResolver: Giải quyết session keys

Example:
    from agents_framework.sessions import (
        SessionManager,
        SessionConfig,
        SessionScope,
        InMemorySessionStore,
    )

    # Tạo session manager
    store = InMemorySessionStore()
    config = SessionConfig(reset_policy="daily")
    manager = SessionManager(store=store, config=config)

    # Tạo hoặc lấy session
    session = await manager.get_or_create_session(
        agent_id="assistant",
        scope=SessionScope.PER_PEER,
        identifier="user123"
    )
"""

from .base import (
    Session,
    SessionConfig,
    SessionScope,
    SessionState,
)
from .store import (
    CompositeSessionStore,
    FileSessionStore,
    InMemorySessionStore,
    SessionStore,
)
from .resolver import (
    SessionKeyComponents,
    SessionResolver,
    get_resolver,
)
from .manager import SessionManager

__all__ = [
    # Base
    "Session",
    "SessionConfig",
    "SessionScope",
    "SessionState",
    # Store
    "SessionStore",
    "InMemorySessionStore",
    "FileSessionStore",
    "CompositeSessionStore",
    # Resolver
    "SessionResolver",
    "SessionKeyComponents",
    "get_resolver",
    # Manager
    "SessionManager",
]
