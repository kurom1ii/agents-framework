"""
Combined Reset Policy - Chính sách kết hợp nhiều policies.

Module này cung cấp CombinedResetPolicy để kết hợp nhiều chính sách
reset khác nhau. Session sẽ được reset khi đạt một trong các điều kiện.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

from .base import ResetPolicyBase, ResetPolicy, ResetReason, ResetResult
from .daily import DailyResetPolicy
from .idle import IdleResetPolicy

if TYPE_CHECKING:
    from ..base import Session


@dataclass
class CombinedResetPolicy(ResetPolicyBase):
    """
    Chính sách kết hợp Daily và Idle reset.

    Policy này sẽ reset session khi đạt một trong hai điều kiện:
    1. Đến giờ reset hàng ngày (daily policy)
    2. Session không hoạt động quá lâu (idle policy)

    Đây là policy được khuyến nghị sử dụng trong production vì nó
    đảm bảo session luôn được refresh định kỳ và không giữ lại
    sessions không hoạt động.

    Attributes:
        daily: DailyResetPolicy để kiểm tra reset theo lịch
        idle: IdleResetPolicy để kiểm tra reset theo idle time

    Ví dụ sử dụng:
        ```python
        # Kết hợp daily reset lúc 4:00 AM và idle reset sau 2 giờ
        policy = CombinedResetPolicy(
            daily=DailyResetPolicy(reset_hour=4),
            idle=IdleResetPolicy(idle_minutes=120)
        )

        # Kiểm tra session
        result = policy.check(session, datetime.now())
        if result.should_reset:
            print(f"Cần reset vì: {result.reason.value}")
        ```
    """
    daily: DailyResetPolicy = field(default_factory=lambda: DailyResetPolicy())
    idle: IdleResetPolicy = field(default_factory=lambda: IdleResetPolicy())

    def should_reset(self, session: "Session", now: datetime) -> bool:
        """
        Kiểm tra xem session có cần reset theo một trong hai policy không.

        Session cần reset nếu một trong hai điều kiện sau được đáp ứng:
        - Daily policy: đã qua thời điểm reset hàng ngày
        - Idle policy: session không hoạt động quá ngưỡng

        Args:
            session: Session cần kiểm tra
            now: Thời điểm hiện tại

        Returns:
            True nếu session cần reset
        """
        return (
            self.daily.should_reset(session, now) or
            self.idle.should_reset(session, now)
        )

    def get_next_reset(
        self,
        session: "Session",
        now: datetime
    ) -> Optional[datetime]:
        """
        Lấy thời điểm reset tiếp theo (sớm nhất giữa daily và idle).

        Args:
            session: Session cần kiểm tra
            now: Thời điểm hiện tại

        Returns:
            Thời điểm reset tiếp theo (sớm nhất)
        """
        daily_next = self.daily.get_next_reset(session, now)
        idle_next = self.idle.get_next_reset(session, now)

        if daily_next is None:
            return idle_next
        if idle_next is None:
            return daily_next

        return min(daily_next, idle_next)

    def check(self, session: "Session", now: datetime) -> ResetResult:
        """
        Kiểm tra chi tiết với thông tin về policy nào trigger reset.

        Args:
            session: Session cần kiểm tra
            now: Thời điểm hiện tại

        Returns:
            ResetResult với đầy đủ thông tin
        """
        # Kiểm tra daily trước (ưu tiên cao hơn)
        if self.daily.should_reset(session, now):
            return ResetResult.needs_reset(
                reason=ResetReason.DAILY,
                details=self.daily._get_reset_details(session, now)
            )

        # Kiểm tra idle
        if self.idle.should_reset(session, now):
            return ResetResult.needs_reset(
                reason=ResetReason.IDLE,
                details=self.idle._get_reset_details(session, now)
            )

        # Không cần reset, trả về thời điểm reset tiếp theo
        return ResetResult.no_reset(
            next_reset_at=self.get_next_reset(session, now)
        )

    def _get_reset_reason(self) -> ResetReason:
        """Trả về lý do reset mặc định (DAILY có ưu tiên cao hơn)."""
        return ResetReason.DAILY

    def get_status(self, session: "Session", now: datetime) -> dict:
        """
        Lấy trạng thái chi tiết của tất cả policies.

        Phương thức utility để debug và monitoring.

        Args:
            session: Session cần kiểm tra
            now: Thời điểm hiện tại

        Returns:
            Dictionary chứa trạng thái của từng policy
        """
        return {
            "daily": {
                "should_reset": self.daily.should_reset(session, now),
                "next_reset": (
                    self.daily.get_next_reset(session, now).isoformat()
                    if self.daily.get_next_reset(session, now) else None
                ),
                "reset_hour": self.daily.reset_hour,
            },
            "idle": {
                "should_reset": self.idle.should_reset(session, now),
                "next_reset": (
                    self.idle.get_next_reset(session, now).isoformat()
                    if self.idle.get_next_reset(session, now) else None
                ),
                "idle_minutes": self.idle.idle_minutes,
                "idle_duration": self.idle.get_idle_duration(session, now).total_seconds() / 60,
            },
            "combined": {
                "should_reset": self.should_reset(session, now),
                "next_reset": (
                    self.get_next_reset(session, now).isoformat()
                    if self.get_next_reset(session, now) else None
                ),
            }
        }


@dataclass
class MultiResetPolicy(ResetPolicyBase):
    """
    Chính sách kết hợp nhiều policies tùy ý.

    Khác với CombinedResetPolicy chỉ hỗ trợ daily + idle,
    MultiResetPolicy cho phép kết hợp bất kỳ số lượng policies nào.

    Attributes:
        policies: Danh sách các policies cần kết hợp
        mode: Chế độ kết hợp ("any" hoặc "all")
            - "any": Reset khi một trong các policies trigger (OR)
            - "all": Reset khi tất cả policies trigger (AND)

    Ví dụ sử dụng:
        ```python
        # Reset khi một trong các điều kiện được đáp ứng
        policy = MultiResetPolicy(
            policies=[
                DailyResetPolicy(reset_hour=4),
                IdleResetPolicy(idle_minutes=60),
            ],
            mode="any"
        )
        ```
    """
    policies: List[ResetPolicy] = field(default_factory=list)
    mode: str = "any"

    def __post_init__(self) -> None:
        """Validate cấu hình sau khi khởi tạo."""
        if self.mode not in ("any", "all"):
            raise ValueError(
                f"mode phải là 'any' hoặc 'all', nhận được: {self.mode}"
            )
        if not self.policies:
            raise ValueError("Phải có ít nhất một policy")

    def should_reset(self, session: "Session", now: datetime) -> bool:
        """
        Kiểm tra xem session có cần reset không dựa trên mode.

        Args:
            session: Session cần kiểm tra
            now: Thời điểm hiện tại

        Returns:
            True nếu session cần reset
        """
        results = [p.should_reset(session, now) for p in self.policies]

        if self.mode == "any":
            return any(results)
        else:  # mode == "all"
            return all(results)

    def get_next_reset(
        self,
        session: "Session",
        now: datetime
    ) -> Optional[datetime]:
        """
        Lấy thời điểm reset tiếp theo.

        - Mode "any": Trả về thời điểm sớm nhất
        - Mode "all": Trả về thời điểm muộn nhất

        Args:
            session: Session cần kiểm tra
            now: Thời điểm hiện tại

        Returns:
            Thời điểm reset tiếp theo
        """
        times = [
            p.get_next_reset(session, now)
            for p in self.policies
        ]
        times = [t for t in times if t is not None]

        if not times:
            return None

        if self.mode == "any":
            return min(times)
        else:  # mode == "all"
            return max(times)

    def check(self, session: "Session", now: datetime) -> ResetResult:
        """
        Kiểm tra chi tiết với thông tin về policies trigger reset.

        Args:
            session: Session cần kiểm tra
            now: Thời điểm hiện tại

        Returns:
            ResetResult với đầy đủ thông tin
        """
        triggered_policies = []
        for policy in self.policies:
            if policy.should_reset(session, now):
                triggered_policies.append(policy.__class__.__name__)

        should_reset = (
            len(triggered_policies) > 0 if self.mode == "any"
            else len(triggered_policies) == len(self.policies)
        )

        if should_reset:
            return ResetResult.needs_reset(
                reason=ResetReason.POLICY,
                details=f"Triggered by: {', '.join(triggered_policies)}"
            )

        return ResetResult.no_reset(
            next_reset_at=self.get_next_reset(session, now)
        )

    def add_policy(self, policy: ResetPolicy) -> "MultiResetPolicy":
        """
        Thêm policy mới vào danh sách.

        Args:
            policy: Policy cần thêm

        Returns:
            Self để hỗ trợ method chaining
        """
        self.policies.append(policy)
        return self

    def remove_policy(self, policy_type: type) -> "MultiResetPolicy":
        """
        Xóa tất cả policies thuộc một type.

        Args:
            policy_type: Type của policy cần xóa

        Returns:
            Self để hỗ trợ method chaining
        """
        self.policies = [
            p for p in self.policies
            if not isinstance(p, policy_type)
        ]
        return self
