"""
Session Reset Policies.

Module này cung cấp các chính sách reset session tự động và thủ công.

Các loại policies:
- DailyResetPolicy: Reset session vào giờ cố định mỗi ngày
- IdleResetPolicy: Reset session sau thời gian không hoạt động
- CombinedResetPolicy: Kết hợp daily + idle reset
- MultiResetPolicy: Kết hợp nhiều policies tùy ý
- ManualResetPolicy: Reset thủ công qua commands
- EventBasedResetPolicy: Reset dựa trên events (lỗi, completion)

Ví dụ sử dụng:
    ```python
    from agents_framework.sessions.policies import (
        DailyResetPolicy,
        IdleResetPolicy,
        CombinedResetPolicy,
        ManualResetPolicy,
        ResetReason,
    )

    # Sử dụng combined policy (khuyến nghị cho production)
    policy = CombinedResetPolicy(
        daily=DailyResetPolicy(reset_hour=4),
        idle=IdleResetPolicy(idle_minutes=120)
    )

    # Kiểm tra session có cần reset không
    result = policy.check(session, datetime.now())
    if result.should_reset:
        print(f"Cần reset vì: {result.reason.value}")
        new_session = await manager.reset(session.session_key)

    # Xử lý manual reset triggers
    manual_policy = ManualResetPolicy(triggers=["/new", "/reset"])
    if manual_policy.is_reset_trigger(user_message):
        await manager.reset(session.session_key)
    ```
"""

from .base import (
    # Enums
    ResetReason,
    # Data classes
    ResetResult,
    ResetEvent,
    # Protocols
    ResetPolicy,
    ResetPolicyBase,
    # Types
    ResetHookCallback,
    ResetTrigger,
    DEFAULT_RESET_TRIGGERS,
)

from .daily import (
    DailyResetPolicy,
)

from .idle import (
    IdleResetPolicy,
)

from .combined import (
    CombinedResetPolicy,
    MultiResetPolicy,
)

from .manual import (
    ManualResetPolicy,
    EventBasedResetPolicy,
)

__all__ = [
    # Base - Enums
    "ResetReason",
    # Base - Data classes
    "ResetResult",
    "ResetEvent",
    # Base - Protocols
    "ResetPolicy",
    "ResetPolicyBase",
    # Base - Types
    "ResetHookCallback",
    "ResetTrigger",
    "DEFAULT_RESET_TRIGGERS",
    # Daily
    "DailyResetPolicy",
    # Idle
    "IdleResetPolicy",
    # Combined
    "CombinedResetPolicy",
    "MultiResetPolicy",
    # Manual
    "ManualResetPolicy",
    "EventBasedResetPolicy",
]
