"""
Daily Reset Policy - Chính sách reset session hàng ngày.

Module này cung cấp DailyResetPolicy để reset session vào một
giờ cố định mỗi ngày.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, TYPE_CHECKING

from .base import ResetPolicyBase, ResetReason

if TYPE_CHECKING:
    from ..base import Session


@dataclass
class DailyResetPolicy(ResetPolicyBase):
    """
    Chính sách reset session hàng ngày vào giờ cố định.

    Policy này sẽ reset session nếu thời điểm cập nhật cuối cùng
    của session nằm trước thời điểm reset gần nhất trong ngày.

    Ví dụ với reset_hour=4:
    - Session được cập nhật lúc 3:00 AM hôm nay -> Cần reset sau 4:00 AM
    - Session được cập nhật lúc 5:00 AM hôm nay -> Không cần reset

    Attributes:
        reset_hour: Giờ reset trong ngày (0-23), mặc định 4 (4:00 AM)
        reset_minute: Phút reset (0-59), mặc định 0
        timezone_name: Tên timezone (mặc định "local" sử dụng UTC)

    Ví dụ sử dụng:
        ```python
        # Reset lúc 4:00 AM mỗi ngày
        policy = DailyResetPolicy(reset_hour=4)

        # Reset lúc 2:30 AM mỗi ngày
        policy = DailyResetPolicy(reset_hour=2, reset_minute=30)

        # Kiểm tra session
        if policy.should_reset(session, datetime.now()):
            print("Session cần được reset!")
        ```
    """
    reset_hour: int = 4
    reset_minute: int = 0
    timezone_name: str = "local"

    def __post_init__(self) -> None:
        """Validate cấu hình sau khi khởi tạo."""
        if not 0 <= self.reset_hour <= 23:
            raise ValueError(
                f"reset_hour phải trong khoảng 0-23, nhận được: {self.reset_hour}"
            )
        if not 0 <= self.reset_minute <= 59:
            raise ValueError(
                f"reset_minute phải trong khoảng 0-59, nhận được: {self.reset_minute}"
            )

    def _get_reset_time_today(self, now: datetime) -> datetime:
        """
        Tính thời điểm reset trong ngày hiện tại.

        Args:
            now: Thời điểm hiện tại

        Returns:
            Thời điểm reset trong ngày hôm nay
        """
        return now.replace(
            hour=self.reset_hour,
            minute=self.reset_minute,
            second=0,
            microsecond=0
        )

    def _get_last_reset_time(self, now: datetime) -> datetime:
        """
        Tính thời điểm reset gần nhất đã qua.

        Args:
            now: Thời điểm hiện tại

        Returns:
            Thời điểm reset gần nhất (có thể là hôm nay hoặc hôm qua)
        """
        reset_today = self._get_reset_time_today(now)

        # Nếu chưa đến giờ reset hôm nay, lấy reset hôm qua
        if now < reset_today:
            return reset_today - timedelta(days=1)
        return reset_today

    def should_reset(self, session: "Session", now: datetime) -> bool:
        """
        Kiểm tra xem session có cần reset theo daily policy không.

        Session cần reset nếu thời điểm cập nhật cuối cùng của nó
        nằm trước thời điểm reset gần nhất.

        Args:
            session: Session cần kiểm tra
            now: Thời điểm hiện tại

        Returns:
            True nếu session cần reset
        """
        last_reset_time = self._get_last_reset_time(now)

        # Session cần reset nếu được cập nhật trước thời điểm reset gần nhất
        return session.updated_at < last_reset_time

    def get_next_reset(
        self,
        session: "Session",
        now: datetime
    ) -> Optional[datetime]:
        """
        Lấy thời điểm reset tiếp theo.

        Args:
            session: Session cần kiểm tra
            now: Thời điểm hiện tại

        Returns:
            Thời điểm reset tiếp theo
        """
        reset_today = self._get_reset_time_today(now)

        # Nếu chưa đến giờ reset hôm nay, reset tiếp theo là hôm nay
        if now < reset_today:
            return reset_today

        # Đã qua giờ reset hôm nay, reset tiếp theo là ngày mai
        return reset_today + timedelta(days=1)

    def _get_reset_reason(self) -> ResetReason:
        """Trả về lý do reset là DAILY."""
        return ResetReason.DAILY

    def _get_reset_details(self, session: "Session", now: datetime) -> str:
        """
        Tạo chi tiết về lý do reset.

        Args:
            session: Session đang được kiểm tra
            now: Thời điểm hiện tại

        Returns:
            Chuỗi mô tả chi tiết
        """
        last_reset = self._get_last_reset_time(now)
        return (
            f"Session cập nhật lúc {session.updated_at.strftime('%H:%M %d/%m')} "
            f"trước thời điểm reset {last_reset.strftime('%H:%M %d/%m')}"
        )

    def get_reset_schedule(self, now: datetime, days: int = 7) -> list[datetime]:
        """
        Lấy lịch reset trong số ngày tới.

        Phương thức utility để lấy danh sách các thời điểm reset
        trong tương lai.

        Args:
            now: Thời điểm hiện tại
            days: Số ngày cần lấy lịch

        Returns:
            Danh sách các thời điểm reset
        """
        schedule = []
        next_reset = self.get_next_reset(None, now)  # type: ignore

        for _ in range(days):
            if next_reset:
                schedule.append(next_reset)
                next_reset = next_reset + timedelta(days=1)

        return schedule
