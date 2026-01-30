"""
Session Management Core.

Module này cung cấp hệ thống quản lý Session theo mô hình OpenClaw,
bao gồm:

- Session và SessionConfig: Models cơ bản
- SessionStore: Protocol và implementations cho lưu trữ
- SessionManager: Quản lý vòng đời session
- SessionResolver: Giải quyết session keys
- TranscriptStore: Lưu trữ bản ghi hội thoại
- Reset Policies: Các chính sách reset session tự động và thủ công

Session Stores có sẵn:
- InMemorySessionStore: Lưu trữ trong bộ nhớ (development, testing)
- FileSessionStore: Lưu trữ file JSON (simple persistence)
- SQLiteSessionStore: Lưu trữ SQLite (production, với connection pooling)
- CompositeSessionStore: Kết hợp cache và persistence

Reset Policies có sẵn:
- DailyResetPolicy: Reset session vào giờ cố định mỗi ngày
- IdleResetPolicy: Reset session sau thời gian không hoạt động
- CombinedResetPolicy: Kết hợp daily + idle reset
- MultiResetPolicy: Kết hợp nhiều policies tùy ý
- ManualResetPolicy: Reset thủ công qua commands
- EventBasedResetPolicy: Reset dựa trên events (lỗi, completion)

Ví dụ sử dụng InMemorySessionStore:
    ```python
    from agents_framework.sessions import (
        SessionManager,
        SessionConfig,
        SessionScope,
        InMemorySessionStore,
    )

    # Tạo session manager với in-memory store
    store = InMemorySessionStore()
    config = SessionConfig(
        scope=SessionScope.PER_PEER,
        reset_mode="combined",
        idle_minutes=60
    )
    manager = SessionManager(store, config)
    ```

Ví dụ sử dụng SQLiteSessionStore (production):
    ```python
    from pathlib import Path
    from agents_framework.sessions import (
        SessionManager,
        SessionConfig,
        SQLiteSessionStore,
    )

    # Tạo session manager với SQLite store
    async def setup():
        store = SQLiteSessionStore(
            db_path=Path("./data/sessions.db"),
            pool_size=5,
            timeout=30.0
        )
        await store.initialize()

        # Kiểm tra health
        if await store.health_check():
            print("Database healthy!")

        config = SessionConfig(reset_mode="combined")
        manager = SessionManager(store, config)

        # Sử dụng context manager
        async with SQLiteSessionStore(Path("./sessions.db")) as store:
            sessions = await store.list_all()
    ```

Ví dụ sử dụng Reset Policies:
    ```python
    from datetime import datetime
    from agents_framework.sessions import (
        DailyResetPolicy,
        IdleResetPolicy,
        CombinedResetPolicy,
        ManualResetPolicy,
    )

    # Sử dụng combined policy (khuyến nghị)
    policy = CombinedResetPolicy(
        daily=DailyResetPolicy(reset_hour=4),
        idle=IdleResetPolicy(idle_minutes=120)
    )

    # Kiểm tra session có cần reset không
    result = policy.check(session, datetime.now())
    if result.should_reset:
        new_session = await manager.reset(session.session_key)

    # Xử lý manual reset triggers
    manual_policy = ManualResetPolicy(triggers=["/new", "/reset"])
    if manual_policy.is_reset_trigger(user_message):
        await manager.reset(session.session_key)
    ```
"""

from .base import (
    Session,
    SessionConfig,
    SessionScope,
    SessionState,
)

from .store import (
    SessionStore,
    InMemorySessionStore,
    FileSessionStore,
    CompositeSessionStore,
    SQLiteSessionStore,
)

from .manager import (
    SessionManager,
    create_in_memory_manager,
)

from .resolver import (
    SessionResolver,
    SessionKeyComponents,
    get_resolver,
)

from .transcript import (
    TranscriptEntry,
    TranscriptStore,
    TranscriptStoreProtocol,
    InMemoryTranscriptStore,
    FileTranscriptStore,
)

from .policies import (
    # Base - Enums
    ResetReason,
    # Base - Data classes
    ResetResult,
    ResetEvent,
    # Base - Protocols
    ResetPolicy,
    ResetPolicyBase,
    # Base - Types
    ResetHookCallback,
    ResetTrigger,
    DEFAULT_RESET_TRIGGERS,
    # Daily
    DailyResetPolicy,
    # Idle
    IdleResetPolicy,
    # Combined
    CombinedResetPolicy,
    MultiResetPolicy,
    # Manual
    ManualResetPolicy,
    EventBasedResetPolicy,
)

__all__ = [
    # Base models
    "Session",
    "SessionConfig",
    "SessionScope",
    "SessionState",
    # Store
    "SessionStore",
    "InMemorySessionStore",
    "FileSessionStore",
    "CompositeSessionStore",
    "SQLiteSessionStore",
    # Manager
    "SessionManager",
    "create_in_memory_manager",
    # Resolver
    "SessionResolver",
    "SessionKeyComponents",
    "get_resolver",
    # Transcript
    "TranscriptEntry",
    "TranscriptStore",
    "TranscriptStoreProtocol",
    "InMemoryTranscriptStore",
    "FileTranscriptStore",
    # Policies - Base
    "ResetReason",
    "ResetResult",
    "ResetEvent",
    "ResetPolicy",
    "ResetPolicyBase",
    "ResetHookCallback",
    "ResetTrigger",
    "DEFAULT_RESET_TRIGGERS",
    # Policies - Implementations
    "DailyResetPolicy",
    "IdleResetPolicy",
    "CombinedResetPolicy",
    "MultiResetPolicy",
    "ManualResetPolicy",
    "EventBasedResetPolicy",
]
