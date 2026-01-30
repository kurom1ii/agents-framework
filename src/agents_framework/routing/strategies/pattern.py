"""Pattern Router implementation.

Module này triển khai PatternRouter - router định tuyến dựa trên
regex patterns match với nội dung message.
"""

from __future__ import annotations

import re
from typing import Optional, Pattern

from ..base import RoutingRequest, RoutingResult, RoutingRule


class PatternRouter:
    """Router định tuyến dựa trên regex patterns.

    PatternRouter sử dụng regex patterns để match với nội dung
    message và định tuyến đến agent tương ứng.

    Các patterns được compile sẵn để tối ưu hiệu suất.

    Attributes:
        rules: Danh sách rules đã sắp xếp theo priority.
        default_agent: Agent mặc định khi không có pattern match.
    """

    def __init__(
        self,
        rules: list[RoutingRule],
        default_agent: str,
        case_sensitive: bool = False
    ) -> None:
        """Khởi tạo PatternRouter.

        Args:
            rules: Danh sách các RoutingRule với patterns.
            default_agent: ID của agent mặc định.
            case_sensitive: Có phân biệt hoa/thường không (mặc định: False).
        """
        self.rules = sorted(rules, key=lambda r: -r.priority)
        self.default_agent = default_agent
        self.case_sensitive = case_sensitive
        self._compiled: dict[str, Pattern[str]] = {}

        # Pre-compile tất cả patterns
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile tất cả regex patterns từ rules.

        Patterns được cache trong _compiled dict để tránh
        compile lại mỗi lần route.
        """
        flags = 0 if self.case_sensitive else re.IGNORECASE

        for rule in self.rules:
            if rule.pattern and rule.pattern not in self._compiled:
                try:
                    self._compiled[rule.pattern] = re.compile(rule.pattern, flags)
                except re.error as e:
                    raise ValueError(
                        f"Invalid regex pattern '{rule.pattern}': {e}"
                    ) from e

    async def route(self, request: RoutingRequest) -> str:
        """Định tuyến request dựa trên pattern matching.

        Đánh giá từng rule theo thứ tự priority và trả về
        agent_id của rule có pattern match đầu tiên.

        Args:
            request: RoutingRequest cần định tuyến.

        Returns:
            ID của agent sẽ xử lý request.
        """
        for rule in self.rules:
            if rule.pattern:
                compiled_pattern = self._compiled.get(rule.pattern)
                if compiled_pattern and compiled_pattern.search(request.message):
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
            if rule.pattern:
                compiled_pattern = self._compiled.get(rule.pattern)
                if compiled_pattern:
                    match = compiled_pattern.search(request.message)
                    if match:
                        return RoutingResult(
                            agent_id=rule.agent_id,
                            matched_rule=rule,
                            confidence=1.0,
                            metadata={
                                "router_type": "pattern",
                                "matched_text": match.group(),
                                "match_start": match.start(),
                                "match_end": match.end()
                            }
                        )

        return RoutingResult(
            agent_id=self.default_agent,
            matched_rule=None,
            confidence=1.0,
            metadata={"router_type": "pattern", "is_default": True}
        )

    def add_rule(self, rule: RoutingRule) -> None:
        """Thêm rule mới với pattern.

        Args:
            rule: RoutingRule cần thêm (phải có pattern).

        Raises:
            ValueError: Nếu rule không có pattern.
        """
        if not rule.pattern:
            raise ValueError("PatternRouter requires rules with patterns")

        # Compile pattern mới
        flags = 0 if self.case_sensitive else re.IGNORECASE
        try:
            self._compiled[rule.pattern] = re.compile(rule.pattern, flags)
        except re.error as e:
            raise ValueError(
                f"Invalid regex pattern '{rule.pattern}': {e}"
            ) from e

        self.rules.append(rule)
        self.rules.sort(key=lambda r: -r.priority)

    def remove_rule_by_pattern(self, pattern: str) -> bool:
        """Xóa rule theo pattern.

        Args:
            pattern: Pattern của rule cần xóa.

        Returns:
            True nếu rule được xóa thành công.
        """
        original_count = len(self.rules)
        self.rules = [r for r in self.rules if r.pattern != pattern]

        # Cleanup compiled pattern
        if pattern in self._compiled:
            del self._compiled[pattern]

        return len(self.rules) < original_count

    def get_matching_patterns(self, message: str) -> list[tuple[str, str]]:
        """Lấy tất cả patterns match với message.

        Hữu ích cho debugging và logging.

        Args:
            message: Nội dung message cần kiểm tra.

        Returns:
            Danh sách tuple (pattern, agent_id) của các patterns match.
        """
        matches: list[tuple[str, str]] = []

        for rule in self.rules:
            if rule.pattern:
                compiled_pattern = self._compiled.get(rule.pattern)
                if compiled_pattern and compiled_pattern.search(message):
                    matches.append((rule.pattern, rule.agent_id))

        return matches
