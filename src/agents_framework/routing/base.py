"""Base protocols và models cho Agent Routing Engine.

Module này định nghĩa các protocols và data classes cơ bản cho hệ thống
định tuyến tin nhắn/request đến các agent khác nhau.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class RoutingRequest:
    """Request cần được định tuyến đến agent phù hợp.

    Attributes:
        message: Nội dung tin nhắn/request cần xử lý.
        sender: ID của người gửi (optional).
        context: Ngữ cảnh bổ sung cho request.
        metadata: Metadata tùy chỉnh.
    """

    message: str
    sender: Optional[str] = None
    context: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Khởi tạo các giá trị mặc định."""
        if self.context is None:
            self.context = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RoutingRule:
    """Quy tắc định tuyến request đến agent cụ thể.

    Attributes:
        agent_id: ID của agent sẽ xử lý request khớp rule này.
        pattern: Regex pattern để match với message content.
        sender: Filter theo sender ID.
        context_key: Filter theo key trong context.
        time_range: Khoảng thời gian áp dụng rule (format: "HH:MM-HH:MM").
        priority: Độ ưu tiên của rule (cao hơn = đánh giá trước).
    """

    agent_id: str
    pattern: Optional[str] = None
    sender: Optional[str] = None
    context_key: Optional[str] = None
    time_range: Optional[str] = None
    priority: int = 0

    def __post_init__(self) -> None:
        """Validate các fields sau khi khởi tạo."""
        if self.time_range:
            self._validate_time_range(self.time_range)

    def _validate_time_range(self, time_range: str) -> None:
        """Validate format của time_range.

        Args:
            time_range: Chuỗi time range cần validate.

        Raises:
            ValueError: Nếu format không hợp lệ.
        """
        time_pattern = re.compile(r"^\d{2}:\d{2}-\d{2}:\d{2}$")
        if not time_pattern.match(time_range):
            raise ValueError(
                f"Invalid time_range format: {time_range}. "
                "Expected format: HH:MM-HH:MM"
            )


@dataclass
class RoutingResult:
    """Kết quả của quá trình routing.

    Attributes:
        agent_id: ID của agent được chọn.
        matched_rule: Rule đã match (nếu có).
        confidence: Độ tin cậy của kết quả routing (0.0-1.0).
        metadata: Metadata bổ sung về quá trình routing.
    """

    agent_id: str
    matched_rule: Optional[RoutingRule] = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Router(Protocol):
    """Protocol định nghĩa interface cho các Router implementations.

    Tất cả các router cần implement method route() để xác định
    agent_id phù hợp cho một RoutingRequest.
    """

    async def route(self, request: RoutingRequest) -> str:
        """Định tuyến request đến agent phù hợp.

        Args:
            request: RoutingRequest cần được định tuyến.

        Returns:
            ID của agent sẽ xử lý request.
        """
        ...


class TimeRangeChecker:
    """Utility class để kiểm tra thời gian hiện tại có nằm trong time range.

    Class này cung cấp method để parse và kiểm tra time range
    theo format "HH:MM-HH:MM".
    """

    @staticmethod
    def parse_time_range(time_range: str) -> tuple[tuple[int, int], tuple[int, int]]:
        """Parse time range string thành tuple của start và end time.

        Args:
            time_range: Chuỗi time range format "HH:MM-HH:MM".

        Returns:
            Tuple chứa (start_hour, start_minute), (end_hour, end_minute).
        """
        start_str, end_str = time_range.split("-")
        start_hour, start_minute = map(int, start_str.split(":"))
        end_hour, end_minute = map(int, end_str.split(":"))
        return (start_hour, start_minute), (end_hour, end_minute)

    @staticmethod
    def is_in_range(
        time_range: str,
        current_time: Optional[datetime] = None
    ) -> bool:
        """Kiểm tra thời gian hiện tại có nằm trong time range.

        Args:
            time_range: Chuỗi time range format "HH:MM-HH:MM".
            current_time: Thời gian cần kiểm tra (mặc định: now).

        Returns:
            True nếu thời gian hiện tại nằm trong range.
        """
        if current_time is None:
            current_time = datetime.now()

        (start_h, start_m), (end_h, end_m) = TimeRangeChecker.parse_time_range(
            time_range
        )

        # Chuyển đổi thành phút từ đầu ngày để so sánh
        current_minutes = current_time.hour * 60 + current_time.minute
        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m

        # Xử lý trường hợp time range qua đêm (ví dụ: 22:00-06:00)
        if start_minutes <= end_minutes:
            return start_minutes <= current_minutes <= end_minutes
        else:
            return current_minutes >= start_minutes or current_minutes <= end_minutes
