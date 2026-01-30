"""
Idle Reset Policy - Chính sách reset session khi không hoạt động.

Module này cung cấp IdleResetPolicy để reset session sau một
khoảng thời gian không hoạt động.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, TYPE_CHECKING

from .base import ResetPolicyBase, ResetReason

if TYPE_CHECKING:
    from ..base import Session


@dataclass
class IdleResetPolicy(ResetPolicyBase):
    """
    Chính sách reset session khi không hoạt động quá lâu.

    Policy này sẽ reset session nếu thời gian từ lần cập nhật
    cuối cùng đến hiện tại vượt quá ngưỡng idle_minutes.

    Attributes:
        idle_minutes: Số phút không hoạt động trước khi reset, mặc định 120 (2 giờ)
        grace_period_minutes: Thời gian gia hạn sau khi đạt ngưỡng idle, mặc định 0

    Ví dụ sử dụng:
        ```python
        # Reset sau 2 giờ không hoạt động
        policy = IdleResetPolicy(idle_minutes=120)

        # Reset sau 30 phút không hoạt động
        policy = IdleResetPolicy(idle_minutes=30)

        # Kiểm tra session
        if policy.should_reset(session, datetime.now()):
            print("Session đã idle quá lâu, cần reset!")

        # Lấy thời gian còn lại trước khi reset
        remaining = policy.get_remaining_time(session, datetime.now())
        print(f"Còn {remaining.total_seconds() / 60:.0f} phút trước khi reset")
        ```
    """
    idle_minutes: int = 120
    grace_period_minutes: int = 0

    def __post_init__(self) -> None:
        """Validate cấu hình sau khi khởi tạo."""
        if self.idle_minutes <= 0:
            raise ValueError(
                f"idle_minutes phải > 0, nhận được: {self.idle_minutes}"
            )
        if self.grace_period_minutes < 0:
            raise ValueError(
                f"grace_period_minutes phải >= 0, nhận được: {self.grace_period_minutes}"
            )

    @property
    def idle_threshold(self) -> timedelta:
        """
        Tính ngưỡng idle tổng cộng (bao gồm grace period).

        Returns:
            Timedelta đại diện cho ngưỡng idle
        """
        total_minutes = self.idle_minutes + self.grace_period_minutes
        return timedelta(minutes=total_minutes)

    def get_idle_duration(self, session: "Session", now: datetime) -> timedelta:
        """
        Tính thời gian session đã idle.

        Args:
            session: Session cần kiểm tra
            now: Thời điểm hiện tại

        Returns:
            Thời gian đã idle
        """
        return now - session.updated_at

    def get_remaining_time(
        self,
        session: "Session",
        now: datetime
    ) -> timedelta:
        """
        Tính thời gian còn lại trước khi session bị reset.

        Args:
            session: Session cần kiểm tra
            now: Thời điểm hiện tại

        Returns:
            Thời gian còn lại (có thể âm nếu đã quá ngưỡng)
        """
        idle_duration = self.get_idle_duration(session, now)
        return self.idle_threshold - idle_duration

    def should_reset(self, session: "Session", now: datetime) -> bool:
        """
        Kiểm tra xem session có cần reset do idle không.

        Session cần reset nếu thời gian không hoạt động vượt quá
        ngưỡng idle_minutes + grace_period_minutes.

        Args:
            session: Session cần kiểm tra
            now: Thời điểm hiện tại

        Returns:
            True nếu session đã idle quá lâu và cần reset
        """
        idle_duration = self.get_idle_duration(session, now)
        return idle_duration > self.idle_threshold

    def get_next_reset(
        self,
        session: "Session",
        now: datetime
    ) -> Optional[datetime]:
        """
        Lấy thời điểm reset tiếp theo nếu không có hoạt động.

        Args:
            session: Session cần kiểm tra
            now: Thời điểm hiện tại

        Returns:
            Thời điểm reset tiếp theo dựa trên thời điểm cập nhật cuối
        """
        # Nếu đã cần reset, trả về ngay bây giờ
        if self.should_reset(session, now):
            return now

        # Tính thời điểm reset = updated_at + idle_threshold
        return session.updated_at + self.idle_threshold

    def _get_reset_reason(self) -> ResetReason:
        """Trả về lý do reset là IDLE."""
        return ResetReason.IDLE

    def _get_reset_details(self, session: "Session", now: datetime) -> str:
        """
        Tạo chi tiết về lý do reset.

        Args:
            session: Session đang được kiểm tra
            now: Thời điểm hiện tại

        Returns:
            Chuỗi mô tả chi tiết
        """
        idle_duration = self.get_idle_duration(session, now)
        idle_minutes = idle_duration.total_seconds() / 60
        return (
            f"Session không hoạt động {idle_minutes:.0f} phút "
            f"(ngưỡng: {self.idle_minutes} phút)"
        )

    def is_approaching_reset(
        self,
        session: "Session",
        now: datetime,
        warning_minutes: int = 15
    ) -> bool:
        """
        Kiểm tra xem session có sắp bị reset không.

        Phương thức utility để cảnh báo trước khi reset.

        Args:
            session: Session cần kiểm tra
            now: Thời điểm hiện tại
            warning_minutes: Số phút cảnh báo trước

        Returns:
            True nếu session sẽ bị reset trong warning_minutes tới
        """
        remaining = self.get_remaining_time(session, now)
        return timedelta(0) < remaining <= timedelta(minutes=warning_minutes)

    def extend_idle(self, additional_minutes: int) -> "IdleResetPolicy":
        """
        Tạo policy mới với thời gian idle được mở rộng.

        Args:
            additional_minutes: Số phút bổ sung

        Returns:
            IdleResetPolicy mới với idle_minutes tăng thêm
        """
        return IdleResetPolicy(
            idle_minutes=self.idle_minutes + additional_minutes,
            grace_period_minutes=self.grace_period_minutes
        )
