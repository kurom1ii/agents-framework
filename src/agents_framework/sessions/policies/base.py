"""
Reset Policy Base - Protocol và types cơ bản.

Module này định nghĩa ResetPolicy protocol - giao diện chuẩn cho các
chính sách reset session. Tất cả các policy cụ thể phải implement
protocol này.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from ..base import Session


class ResetReason(Enum):
    """
    Lý do reset session.

    Enum này xác định nguyên nhân dẫn đến việc reset session,
    hữu ích cho logging và analytics.

    Attributes:
        DAILY: Reset theo lịch hàng ngày
        IDLE: Reset do session không hoạt động quá lâu
        MANUAL: Reset thủ công bởi người dùng hoặc API
        TOKEN_LIMIT: Reset do vượt quá giới hạn token
        ERROR: Reset do lỗi xảy ra
        COMPLETION: Reset khi hoàn thành một task
        POLICY: Reset theo policy tùy chỉnh khác
    """
    DAILY = "daily"
    IDLE = "idle"
    MANUAL = "manual"
    TOKEN_LIMIT = "token_limit"
    ERROR = "error"
    COMPLETION = "completion"
    POLICY = "policy"


@dataclass
class ResetResult:
    """
    Kết quả kiểm tra reset.

    Dataclass này chứa thông tin chi tiết về việc có cần reset hay không
    và lý do tại sao.

    Attributes:
        should_reset: True nếu cần reset session
        reason: Lý do reset (None nếu không cần reset)
        details: Thông tin chi tiết bổ sung
        next_reset_at: Thời điểm reset tiếp theo (ước tính)
    """
    should_reset: bool
    reason: Optional[ResetReason] = None
    details: Optional[str] = None
    next_reset_at: Optional[datetime] = None

    @classmethod
    def no_reset(cls, next_reset_at: Optional[datetime] = None) -> "ResetResult":
        """
        Tạo kết quả không cần reset.

        Args:
            next_reset_at: Thời điểm reset tiếp theo (ước tính)

        Returns:
            ResetResult với should_reset=False
        """
        return cls(should_reset=False, next_reset_at=next_reset_at)

    @classmethod
    def needs_reset(
        cls,
        reason: ResetReason,
        details: Optional[str] = None
    ) -> "ResetResult":
        """
        Tạo kết quả cần reset.

        Args:
            reason: Lý do cần reset
            details: Chi tiết bổ sung

        Returns:
            ResetResult với should_reset=True
        """
        return cls(should_reset=True, reason=reason, details=details)


@dataclass
class ResetEvent:
    """
    Event phát sinh khi reset session.

    Dataclass này chứa thông tin về sự kiện reset để các hooks
    có thể xử lý.

    Attributes:
        session_key: Key của session được reset
        old_session_id: ID của session cũ
        new_session_id: ID của session mới
        reason: Lý do reset
        reset_at: Thời điểm reset
        metadata: Metadata bổ sung
    """
    session_key: str
    old_session_id: str
    new_session_id: str
    reason: ResetReason
    reset_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển event thành dictionary.

        Returns:
            Dictionary chứa thông tin event
        """
        return {
            "session_key": self.session_key,
            "old_session_id": self.old_session_id,
            "new_session_id": self.new_session_id,
            "reason": self.reason.value,
            "reset_at": self.reset_at.isoformat(),
            "metadata": self.metadata
        }


# Type alias cho reset hook callback
ResetHookCallback = Callable[[ResetEvent], None]


@runtime_checkable
class ResetPolicy(Protocol):
    """
    Protocol định nghĩa giao diện cho các chính sách reset session.

    Tất cả các implementation của ResetPolicy phải triển khai
    các phương thức này để đảm bảo tính nhất quán.

    Ví dụ sử dụng:
        ```python
        policy = DailyResetPolicy(reset_hour=4)

        # Kiểm tra xem session có cần reset không
        if policy.should_reset(session, datetime.now()):
            new_session = await manager.reset(session.session_key)

        # Lấy thời điểm reset tiếp theo
        next_reset = policy.get_next_reset(session, datetime.now())
        ```
    """

    @abstractmethod
    def should_reset(self, session: "Session", now: datetime) -> bool:
        """
        Kiểm tra xem session có cần reset không.

        Phương thức này được gọi mỗi khi session được truy cập
        để xác định xem có cần reset session hay không.

        Args:
            session: Session cần kiểm tra
            now: Thời điểm hiện tại

        Returns:
            True nếu cần reset session
        """
        ...

    @abstractmethod
    def get_next_reset(
        self,
        session: "Session",
        now: datetime
    ) -> Optional[datetime]:
        """
        Lấy thời điểm reset tiếp theo (nếu có).

        Phương thức này trả về thời điểm dự kiến reset tiếp theo
        dựa trên policy hiện tại.

        Args:
            session: Session cần kiểm tra
            now: Thời điểm hiện tại

        Returns:
            Thời điểm reset tiếp theo hoặc None nếu không xác định được
        """
        ...


class ResetPolicyBase:
    """
    Base class cho các ResetPolicy implementations.

    Class này cung cấp các utility methods chung mà các policy
    cụ thể có thể sử dụng.
    """

    def check(self, session: "Session", now: datetime) -> ResetResult:
        """
        Kiểm tra chi tiết xem session có cần reset không.

        Phương thức này trả về ResetResult với đầy đủ thông tin,
        khác với should_reset() chỉ trả về bool.

        Args:
            session: Session cần kiểm tra
            now: Thời điểm hiện tại

        Returns:
            ResetResult với đầy đủ thông tin
        """
        if self.should_reset(session, now):
            return ResetResult.needs_reset(
                reason=self._get_reset_reason(),
                details=self._get_reset_details(session, now)
            )
        return ResetResult.no_reset(
            next_reset_at=self.get_next_reset(session, now)
        )

    def should_reset(self, session: "Session", now: datetime) -> bool:
        """
        Kiểm tra xem session có cần reset không.

        Các subclass phải override phương thức này.

        Args:
            session: Session cần kiểm tra
            now: Thời điểm hiện tại

        Returns:
            True nếu cần reset session
        """
        raise NotImplementedError("Subclass phải implement should_reset()")

    def get_next_reset(
        self,
        session: "Session",
        now: datetime
    ) -> Optional[datetime]:
        """
        Lấy thời điểm reset tiếp theo.

        Các subclass phải override phương thức này.

        Args:
            session: Session cần kiểm tra
            now: Thời điểm hiện tại

        Returns:
            Thời điểm reset tiếp theo hoặc None
        """
        raise NotImplementedError("Subclass phải implement get_next_reset()")

    def _get_reset_reason(self) -> ResetReason:
        """
        Lấy lý do reset mặc định.

        Các subclass nên override để trả về lý do phù hợp.

        Returns:
            ResetReason mặc định
        """
        return ResetReason.POLICY

    def _get_reset_details(self, session: "Session", now: datetime) -> str:
        """
        Lấy chi tiết về lý do reset.

        Các subclass có thể override để cung cấp thông tin chi tiết hơn.

        Args:
            session: Session đang được kiểm tra
            now: Thời điểm hiện tại

        Returns:
            Chuỗi mô tả chi tiết lý do reset
        """
        return f"Session {session.session_id} cần reset theo policy"


# Type alias cho danh sách reset triggers
ResetTrigger = str
DEFAULT_RESET_TRIGGERS: List[ResetTrigger] = ["/new", "/reset", "/clear"]
