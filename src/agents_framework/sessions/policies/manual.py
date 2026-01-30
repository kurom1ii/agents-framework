"""
Manual Reset Policy - Chính sách reset thủ công.

Module này cung cấp ManualResetPolicy để xử lý các trigger
reset thủ công như commands /new, /reset, /clear.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Set, TYPE_CHECKING

from .base import ResetPolicyBase, ResetReason, ResetResult, DEFAULT_RESET_TRIGGERS

if TYPE_CHECKING:
    from ..base import Session


@dataclass
class ManualResetPolicy(ResetPolicyBase):
    """
    Chính sách reset thủ công qua commands hoặc API.

    Policy này không tự động reset session mà chỉ cung cấp các
    utilities để xử lý reset triggers từ user hoặc programmatic calls.

    Attributes:
        triggers: Danh sách các commands kích hoạt reset
        case_sensitive: Có phân biệt hoa thường không
        allow_api_reset: Cho phép reset qua API

    Ví dụ sử dụng:
        ```python
        policy = ManualResetPolicy(
            triggers=["/new", "/reset", "/clear"],
            case_sensitive=False
        )

        # Kiểm tra message có phải là reset trigger không
        if policy.is_reset_trigger("/reset"):
            print("User yêu cầu reset session!")

        # Reset qua API
        if policy.allow_api_reset:
            await manager.reset(session.session_key)
        ```
    """
    triggers: List[str] = field(default_factory=lambda: list(DEFAULT_RESET_TRIGGERS))
    case_sensitive: bool = False
    allow_api_reset: bool = True

    def __post_init__(self) -> None:
        """Normalize triggers nếu cần."""
        if not self.case_sensitive:
            self._normalized_triggers: Set[str] = {
                t.lower() for t in self.triggers
            }
        else:
            self._normalized_triggers = set(self.triggers)

    def should_reset(self, session: "Session", now: datetime) -> bool:
        """
        ManualResetPolicy không tự động reset.

        Luôn trả về False vì reset thủ công được xử lý thông qua
        is_reset_trigger() và check_message().

        Args:
            session: Session cần kiểm tra
            now: Thời điểm hiện tại

        Returns:
            Luôn trả về False
        """
        return False

    def get_next_reset(
        self,
        session: "Session",
        now: datetime
    ) -> Optional[datetime]:
        """
        Không có thời điểm reset tiếp theo với manual policy.

        Args:
            session: Session cần kiểm tra
            now: Thời điểm hiện tại

        Returns:
            Luôn trả về None
        """
        return None

    def is_reset_trigger(self, message: str) -> bool:
        """
        Kiểm tra message có phải là reset trigger không.

        Args:
            message: Message cần kiểm tra

        Returns:
            True nếu message là một reset trigger
        """
        text = message.strip()
        if not self.case_sensitive:
            text = text.lower()
        return text in self._normalized_triggers

    def check_message(self, message: str) -> ResetResult:
        """
        Kiểm tra message và trả về ResetResult.

        Args:
            message: Message cần kiểm tra

        Returns:
            ResetResult với should_reset=True nếu là reset trigger
        """
        if self.is_reset_trigger(message):
            return ResetResult.needs_reset(
                reason=ResetReason.MANUAL,
                details=f"Reset trigger: {message.strip()}"
            )
        return ResetResult.no_reset()

    def add_trigger(self, trigger: str) -> "ManualResetPolicy":
        """
        Thêm trigger mới.

        Args:
            trigger: Trigger cần thêm

        Returns:
            Self để hỗ trợ method chaining
        """
        self.triggers.append(trigger)
        if self.case_sensitive:
            self._normalized_triggers.add(trigger)
        else:
            self._normalized_triggers.add(trigger.lower())
        return self

    def remove_trigger(self, trigger: str) -> "ManualResetPolicy":
        """
        Xóa trigger.

        Args:
            trigger: Trigger cần xóa

        Returns:
            Self để hỗ trợ method chaining
        """
        if trigger in self.triggers:
            self.triggers.remove(trigger)
        normalized = trigger if self.case_sensitive else trigger.lower()
        self._normalized_triggers.discard(normalized)
        return self

    def get_triggers(self) -> List[str]:
        """
        Lấy danh sách triggers hiện tại.

        Returns:
            Danh sách triggers
        """
        return self.triggers.copy()

    def _get_reset_reason(self) -> ResetReason:
        """Trả về lý do reset là MANUAL."""
        return ResetReason.MANUAL


@dataclass
class EventBasedResetPolicy(ResetPolicyBase):
    """
    Chính sách reset dựa trên events.

    Policy này reset session khi có các events nhất định xảy ra
    như lỗi, hoàn thành task, v.v.

    Attributes:
        reset_on_error: Reset khi có lỗi xảy ra
        reset_on_completion: Reset khi hoàn thành task
        error_threshold: Số lỗi liên tiếp trước khi reset
        completion_keywords: Các từ khóa đánh dấu hoàn thành

    Ví dụ sử dụng:
        ```python
        policy = EventBasedResetPolicy(
            reset_on_error=True,
            error_threshold=3,
            reset_on_completion=True,
            completion_keywords=["done", "completed", "finished"]
        )

        # Kiểm tra có nên reset sau lỗi không
        if policy.should_reset_on_error(error_count=3):
            await manager.reset(session.session_key)

        # Kiểm tra message có đánh dấu completion không
        if policy.is_completion_message("Task completed successfully"):
            await manager.reset(session.session_key)
        ```
    """
    reset_on_error: bool = True
    reset_on_completion: bool = False
    error_threshold: int = 3
    completion_keywords: List[str] = field(
        default_factory=lambda: ["done", "completed", "finished", "goodbye"]
    )

    def __post_init__(self) -> None:
        """Validate cấu hình."""
        if self.error_threshold < 1:
            raise ValueError(
                f"error_threshold phải >= 1, nhận được: {self.error_threshold}"
            )

    def should_reset(self, session: "Session", now: datetime) -> bool:
        """
        EventBasedResetPolicy không tự động kiểm tra reset.

        Reset được xác định thông qua các phương thức kiểm tra events.

        Args:
            session: Session cần kiểm tra
            now: Thời điểm hiện tại

        Returns:
            Luôn trả về False
        """
        return False

    def get_next_reset(
        self,
        session: "Session",
        now: datetime
    ) -> Optional[datetime]:
        """
        Không có thời điểm reset tiếp theo với event-based policy.

        Args:
            session: Session cần kiểm tra
            now: Thời điểm hiện tại

        Returns:
            Luôn trả về None
        """
        return None

    def should_reset_on_error(self, error_count: int) -> bool:
        """
        Kiểm tra có nên reset sau một số lỗi.

        Args:
            error_count: Số lỗi liên tiếp đã xảy ra

        Returns:
            True nếu nên reset
        """
        if not self.reset_on_error:
            return False
        return error_count >= self.error_threshold

    def is_completion_message(self, message: str) -> bool:
        """
        Kiểm tra message có đánh dấu completion không.

        Args:
            message: Message cần kiểm tra

        Returns:
            True nếu message chứa completion keyword
        """
        if not self.reset_on_completion:
            return False

        lower_message = message.lower()
        return any(
            keyword.lower() in lower_message
            for keyword in self.completion_keywords
        )

    def check_error(self, error_count: int) -> ResetResult:
        """
        Kiểm tra có cần reset do lỗi không.

        Args:
            error_count: Số lỗi liên tiếp

        Returns:
            ResetResult
        """
        if self.should_reset_on_error(error_count):
            return ResetResult.needs_reset(
                reason=ResetReason.ERROR,
                details=f"Đã có {error_count} lỗi liên tiếp (ngưỡng: {self.error_threshold})"
            )
        return ResetResult.no_reset()

    def check_completion(self, message: str) -> ResetResult:
        """
        Kiểm tra có cần reset do completion không.

        Args:
            message: Message cần kiểm tra

        Returns:
            ResetResult
        """
        if self.is_completion_message(message):
            return ResetResult.needs_reset(
                reason=ResetReason.COMPLETION,
                details=f"Phát hiện completion trong message"
            )
        return ResetResult.no_reset()

    def _get_reset_reason(self) -> ResetReason:
        """Trả về lý do reset mặc định."""
        return ResetReason.ERROR
