"""
Session Management Core.

Module này cung cấp hệ thống quản lý Session theo mô hình OpenClaw,
bao gồm:

- Session và SessionConfig: Models cơ bản
- SessionStore: Protocol và implementations cho lưu trữ
- SessionManager: Quản lý vòng đời session
- SessionResolver: Giải quyết session keys
- TranscriptStore: Lưu trữ bản ghi hội thoại

Ví dụ sử dụng:
    ```python
    from agents_framework.sessions import (
        SessionManager,
        SessionConfig,
        SessionScope,
        InMemorySessionStore,
        TranscriptStore,
        TranscriptEntry,
    )

    # Tạo session manager
    store = InMemorySessionStore()
    config = SessionConfig(
        scope=SessionScope.PER_PEER,
        reset_mode="combined",
        idle_minutes=60
    )
    manager = SessionManager(store, config)

    # Tạo hoặc lấy session
    session = await manager.get_or_create(
        key="agent:assistant:dm:user123",
        agent_id="assistant"
    )

    # Cập nhật token usage
    session.update_tokens(input_tokens=500, output_tokens=200)
    await manager.update(session)

    # Lưu transcript
    transcript_store = TranscriptStore()
    await transcript_store.append(
        session.session_id,
        TranscriptEntry.user("Hello, how are you?")
    )
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
]
