"""Routing Engine - Core logic cho Agent Routing.

Module này triển khai RoutingEngine - engine điều phối routing
requests đến các agents dựa trên cấu hình và strategies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .base import Router, RoutingRequest, RoutingResult, RoutingRule
from .config import RoutingConfig
from .strategies.pattern import PatternRouter
from .strategies.static import StaticRouter

if TYPE_CHECKING:
    from ..agents.base import BaseAgent


class CombinedRouter:
    """Router kết hợp pattern matching và static rules.

    CombinedRouter thử pattern matching trước, nếu không match
    thì fallback sang static rules.

    Attributes:
        pattern_router: PatternRouter cho pattern matching.
        static_router: StaticRouter cho static rules.
        default_agent: Agent mặc định.
    """

    def __init__(
        self,
        rules: list[RoutingRule],
        default_agent: str
    ) -> None:
        """Khởi tạo CombinedRouter.

        Args:
            rules: Danh sách tất cả routing rules.
            default_agent: ID của agent mặc định.
        """
        self.default_agent = default_agent

        # Phân loại rules
        pattern_rules = [r for r in rules if r.pattern]
        static_rules = [r for r in rules if not r.pattern]

        self.pattern_router = PatternRouter(pattern_rules, default_agent)
        self.static_router = StaticRouter(static_rules, default_agent)

    async def route(self, request: RoutingRequest) -> str:
        """Định tuyến request sử dụng combined strategy.

        Thử pattern matching trước, nếu trả về default thì
        tiếp tục với static routing.

        Args:
            request: RoutingRequest cần định tuyến.

        Returns:
            ID của agent sẽ xử lý request.
        """
        # Thử pattern matching trước
        pattern_result = await self.pattern_router.route_with_result(request)

        # Nếu pattern match được agent (không phải default), sử dụng kết quả đó
        if pattern_result.matched_rule is not None:
            return pattern_result.agent_id

        # Fallback sang static routing
        static_result = await self.static_router.route_with_result(request)

        return static_result.agent_id

    async def route_with_result(
        self,
        request: RoutingRequest
    ) -> RoutingResult:
        """Định tuyến request và trả về kết quả chi tiết.

        Args:
            request: RoutingRequest cần định tuyến.

        Returns:
            RoutingResult với thông tin chi tiết.
        """
        # Thử pattern matching trước
        pattern_result = await self.pattern_router.route_with_result(request)

        if pattern_result.matched_rule is not None:
            pattern_result.metadata["router_type"] = "combined:pattern"
            return pattern_result

        # Fallback sang static routing
        static_result = await self.static_router.route_with_result(request)
        static_result.metadata["router_type"] = "combined:static"

        return static_result


class RoutingEngine:
    """Engine điều phối routing requests đến agents.

    RoutingEngine là thành phần trung tâm của hệ thống routing,
    quản lý agents và điều phối requests dựa trên configuration.

    Attributes:
        config: Cấu hình routing.
        agents: Dict mapping agent_id -> BaseAgent.
    """

    def __init__(self, config: RoutingConfig) -> None:
        """Khởi tạo RoutingEngine.

        Args:
            config: RoutingConfig chứa rules và strategy.
        """
        self.config = config
        self.agents: dict[str, BaseAgent] = {}
        self._router: Router = self._create_router()

    def _create_router(self) -> Router:
        """Tạo router dựa trên strategy trong config.

        Returns:
            Router instance phù hợp với strategy.
        """
        if self.config.strategy == "static":
            return StaticRouter(
                self.config.rules,
                self.config.default_agent
            )
        elif self.config.strategy == "pattern":
            return PatternRouter(
                self.config.rules,
                self.config.default_agent
            )
        else:  # combined
            return CombinedRouter(
                self.config.rules,
                self.config.default_agent
            )

    def add_agent(self, agent_id: str, agent: BaseAgent) -> None:
        """Đăng ký agent với engine.

        Args:
            agent_id: ID unique cho agent.
            agent: BaseAgent instance.
        """
        self.agents[agent_id] = agent

    def remove_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Xóa agent khỏi engine.

        Args:
            agent_id: ID của agent cần xóa.

        Returns:
            Agent đã xóa hoặc None nếu không tìm thấy.
        """
        return self.agents.pop(agent_id, None)

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Lấy agent theo ID.

        Args:
            agent_id: ID của agent.

        Returns:
            BaseAgent hoặc None nếu không tìm thấy.
        """
        return self.agents.get(agent_id)

    def has_agent(self, agent_id: str) -> bool:
        """Kiểm tra agent có được đăng ký không.

        Args:
            agent_id: ID của agent cần kiểm tra.

        Returns:
            True nếu agent đã được đăng ký.
        """
        return agent_id in self.agents

    def list_agents(self) -> list[str]:
        """Lấy danh sách tất cả agent IDs đã đăng ký.

        Returns:
            Danh sách agent IDs.
        """
        return list(self.agents.keys())

    async def route(self, request: RoutingRequest) -> str:
        """Định tuyến request đến agent phù hợp.

        Args:
            request: RoutingRequest cần định tuyến.

        Returns:
            ID của agent sẽ xử lý request.
        """
        return await self._router.route(request)

    async def route_with_result(
        self,
        request: RoutingRequest
    ) -> RoutingResult:
        """Định tuyến request và trả về kết quả chi tiết.

        Args:
            request: RoutingRequest cần định tuyến.

        Returns:
            RoutingResult với thông tin chi tiết.
        """
        # Kiểm tra xem router có method route_with_result không
        if hasattr(self._router, 'route_with_result'):
            return await self._router.route_with_result(request)  # type: ignore

        # Fallback nếu router không có method này
        agent_id = await self._router.route(request)
        return RoutingResult(agent_id=agent_id)

    async def route_and_execute(
        self,
        request: RoutingRequest
    ) -> Any:
        """Định tuyến request và thực thi với agent được chọn.

        Method này kết hợp routing và execution trong một bước:
        1. Định tuyến request để tìm agent phù hợp
        2. Thực thi request với agent đó

        Args:
            request: RoutingRequest cần xử lý.

        Returns:
            Kết quả từ agent.run().

        Raises:
            ValueError: Nếu không tìm thấy agent.
        """
        agent_id = await self.route(request)
        agent = self.agents.get(agent_id)

        if not agent:
            if self.config.fallback_enabled:
                # Thử fallback đến default agent
                agent = self.agents.get(self.config.default_agent)

            if not agent:
                raise ValueError(
                    f"Agent not found: {agent_id}. "
                    f"Available agents: {list(self.agents.keys())}"
                )

        # Thực thi với agent
        return await agent.run(request.message)

    async def route_and_execute_with_context(
        self,
        request: RoutingRequest,
        additional_context: Optional[dict[str, Any]] = None
    ) -> tuple[str, Any]:
        """Định tuyến và thực thi, trả về cả agent_id và kết quả.

        Args:
            request: RoutingRequest cần xử lý.
            additional_context: Context bổ sung cho agent (reserved for future use).

        Returns:
            Tuple (agent_id, result) từ quá trình thực thi.

        Raises:
            ValueError: Nếu không tìm thấy agent.
        """
        agent_id = await self.route(request)
        agent = self.agents.get(agent_id)

        if not agent:
            raise ValueError(f"Agent not found: {agent_id}")

        # TODO: Truyền merged context khi BaseAgent hỗ trợ
        # Hiện tại chỉ thực thi với message
        _ = additional_context  # Reserved for future use

        result = await agent.run(request.message)

        return agent_id, result

    def add_rule(self, rule: RoutingRule) -> None:
        """Thêm routing rule mới và rebuild router.

        Args:
            rule: RoutingRule cần thêm.
        """
        self.config.add_rule(rule)
        self._router = self._create_router()

    def remove_rules_for_agent(self, agent_id: str) -> bool:
        """Xóa tất cả rules cho một agent và rebuild router.

        Args:
            agent_id: ID của agent cần xóa rules.

        Returns:
            True nếu có rules được xóa.
        """
        removed = self.config.remove_rule(agent_id)
        if removed:
            self._router = self._create_router()
        return removed

    def update_config(self, config: RoutingConfig) -> None:
        """Cập nhật config và rebuild router.

        Args:
            config: RoutingConfig mới.
        """
        self.config = config
        self._router = self._create_router()

    def get_stats(self) -> dict[str, Any]:
        """Lấy thống kê về engine.

        Returns:
            Dict chứa các thông tin thống kê.
        """
        return {
            "strategy": self.config.strategy,
            "default_agent": self.config.default_agent,
            "total_rules": len(self.config.rules),
            "registered_agents": len(self.agents),
            "agent_ids": list(self.agents.keys()),
            "fallback_enabled": self.config.fallback_enabled
        }
