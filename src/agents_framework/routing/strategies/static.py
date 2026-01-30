"""Static Router implementation.

Module này triển khai StaticRouter - router định tuyến dựa trên
các rules cố định với các điều kiện như sender, context, time range.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from ..base import RoutingRequest, RoutingResult, RoutingRule, TimeRangeChecker


class StaticRouter:
    """Router định tuyến dựa trên rules cố định.

    StaticRouter đánh giá các rules theo thứ tự priority (cao đến thấp)
    và trả về agent_id của rule đầu tiên match với request.

    Các điều kiện có thể match:
    - sender: Match theo sender ID
    - context_key: Yêu cầu key tồn tại trong context
    - time_range: Áp dụng trong khoảng thời gian cụ thể

    Attributes:
        rules: Danh sách rules đã sắp xếp theo priority.
        default_agent: Agent mặc định khi không có rule match.
    """

    def __init__(
        self,
        rules: list[RoutingRule],
        default_agent: str
    ) -> None:
        """Khởi tạo StaticRouter.

        Args:
            rules: Danh sách các RoutingRule.
            default_agent: ID của agent mặc định.
        """
        self.rules = sorted(rules, key=lambda r: -r.priority)
        self.default_agent = default_agent
        self._time_checker = TimeRangeChecker()

    async def route(self, request: RoutingRequest) -> str:
        """Định tuyến request đến agent phù hợp.

        Đánh giá từng rule theo thứ tự priority và trả về
        agent_id của rule đầu tiên match.

        Args:
            request: RoutingRequest cần định tuyến.

        Returns:
            ID của agent sẽ xử lý request.
        """
        for rule in self.rules:
            if self._matches(rule, request):
                return rule.agent_id
        return self.default_agent

    async def route_with_result(
        self,
        request: RoutingRequest
    ) -> RoutingResult:
        """Định tuyến request và trả về kết quả chi tiết.

        Args:
            request: RoutingRequest cần định tuyến.

        Returns:
            RoutingResult với thông tin chi tiết về routing.
        """
        for rule in self.rules:
            if self._matches(rule, request):
                return RoutingResult(
                    agent_id=rule.agent_id,
                    matched_rule=rule,
                    confidence=1.0,
                    metadata={"router_type": "static"}
                )

        return RoutingResult(
            agent_id=self.default_agent,
            matched_rule=None,
            confidence=1.0,
            metadata={"router_type": "static", "is_default": True}
        )

    def _matches(
        self,
        rule: RoutingRule,
        request: RoutingRequest,
        current_time: Optional[datetime] = None
    ) -> bool:
        """Kiểm tra xem rule có match với request không.

        Tất cả các điều kiện được định nghĩa trong rule phải
        đồng thời thỏa mãn (AND logic).

        Args:
            rule: RoutingRule cần kiểm tra.
            request: RoutingRequest để so sánh.
            current_time: Thời gian hiện tại (cho testing).

        Returns:
            True nếu tất cả điều kiện của rule đều match.
        """
        # Check sender filter
        if rule.sender and request.sender != rule.sender:
            return False

        # Check context key filter
        if rule.context_key and request.context:
            if rule.context_key not in request.context:
                return False
        elif rule.context_key and not request.context:
            # Rule yêu cầu context_key nhưng request không có context
            return False

        # Check time range filter
        if rule.time_range:
            if not self._in_time_range(rule.time_range, current_time):
                return False

        return True

    def _in_time_range(
        self,
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
        return self._time_checker.is_in_range(time_range, current_time)

    def add_rule(self, rule: RoutingRule) -> None:
        """Thêm rule mới và sắp xếp lại theo priority.

        Args:
            rule: RoutingRule cần thêm.
        """
        self.rules.append(rule)
        self.rules.sort(key=lambda r: -r.priority)

    def remove_rule_by_agent(self, agent_id: str) -> int:
        """Xóa tất cả rules cho một agent_id.

        Args:
            agent_id: ID của agent cần xóa rules.

        Returns:
            Số lượng rules đã xóa.
        """
        original_count = len(self.rules)
        self.rules = [r for r in self.rules if r.agent_id != agent_id]
        return original_count - len(self.rules)
