"""Configuration cho Agent Routing Engine.

Module này định nghĩa các cấu hình cho hệ thống routing,
bao gồm các strategy và rule definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .base import RoutingRule


RoutingStrategy = Literal["static", "pattern", "combined"]


@dataclass
class RoutingConfig:
    """Cấu hình cho Routing Engine.

    Attributes:
        default_agent: ID của agent mặc định khi không có rule nào match.
        rules: Danh sách các routing rules.
        strategy: Chiến lược routing (static, pattern, hoặc combined).
        fallback_enabled: Cho phép fallback đến default_agent.
        max_routing_time_ms: Thời gian tối đa cho quá trình routing (ms).
    """

    default_agent: str
    rules: list[RoutingRule] = field(default_factory=list)
    strategy: RoutingStrategy = "static"
    fallback_enabled: bool = True
    max_routing_time_ms: int = 1000

    def __post_init__(self) -> None:
        """Validate config sau khi khởi tạo."""
        if not self.default_agent:
            raise ValueError("default_agent is required")

        if self.strategy not in ("static", "pattern", "combined"):
            raise ValueError(
                f"Invalid strategy: {self.strategy}. "
                "Must be one of: static, pattern, combined"
            )

    def add_rule(self, rule: RoutingRule) -> None:
        """Thêm một routing rule mới.

        Args:
            rule: RoutingRule cần thêm vào config.
        """
        self.rules.append(rule)

    def remove_rule(self, agent_id: str) -> bool:
        """Xóa tất cả rules cho một agent_id.

        Args:
            agent_id: ID của agent cần xóa rules.

        Returns:
            True nếu có ít nhất một rule được xóa.
        """
        original_count = len(self.rules)
        self.rules = [r for r in self.rules if r.agent_id != agent_id]
        return len(self.rules) < original_count

    def get_rules_for_agent(self, agent_id: str) -> list[RoutingRule]:
        """Lấy tất cả rules cho một agent_id cụ thể.

        Args:
            agent_id: ID của agent cần lấy rules.

        Returns:
            Danh sách các RoutingRule cho agent đó.
        """
        return [r for r in self.rules if r.agent_id == agent_id]

    def get_sorted_rules(self) -> list[RoutingRule]:
        """Lấy danh sách rules đã sắp xếp theo priority (cao đến thấp).

        Returns:
            Danh sách RoutingRule đã sắp xếp.
        """
        return sorted(self.rules, key=lambda r: -r.priority)


@dataclass
class AgentRouteMapping:
    """Mapping trực tiếp từ identifier đến agent.

    Sử dụng cho các trường hợp routing đơn giản như
    mapping theo sender ID hoặc channel ID.

    Attributes:
        identifier: Identifier cần map (sender_id, channel_id, etc.).
        agent_id: ID của agent sẽ xử lý.
        description: Mô tả cho mapping này.
    """

    identifier: str
    agent_id: str
    description: str = ""


@dataclass
class ChannelRoutingConfig:
    """Cấu hình routing theo channel.

    Attributes:
        channel_id: ID của channel.
        default_agent: Agent mặc định cho channel này.
        mappings: Danh sách các mappings cụ thể.
        require_mention: Yêu cầu mention để trigger agent.
    """

    channel_id: str
    default_agent: str
    mappings: list[AgentRouteMapping] = field(default_factory=list)
    require_mention: bool = False
