"""Dynamic Router implementation.

Module này triển khai DynamicRouter - router định tuyến động dựa trên
phân tích nội dung, context, và agent capabilities.

DynamicRouter kết hợp nhiều chiến lược routing:
- Content-based routing (commands, hashtags, keywords)
- Context-based routing (user preferences, session history)
- Capability-based agent matching
- Load balancing across similar agents

Ví dụ sử dụng:
    from agents_framework.routing.dynamic import DynamicRouter

    router = DynamicRouter(
        agents={
            "work": work_agent,
            "personal": personal_agent,
            "coder": code_agent,
        },
        command_mapping={
            "/code": "coder",
            "/research": "researcher",
        },
        hashtag_mapping={
            "#work": "work",
            "#personal": "personal",
        },
    )

    # Request với hashtag
    request = RoutingRequest(message="#work Schedule meeting")
    agent_id = await router.route(request)  # -> "work"
"""

from __future__ import annotations

import random
import uuid
from typing import TYPE_CHECKING, Any, Optional

from .base import RoutingRequest, RoutingResult
from .hooks import RoutingHookRegistry, RoutingHookType
from .strategies.content import ContentAnalyzer, ContentHints, ContentRouter

if TYPE_CHECKING:
    from ..agents.base import BaseAgent
    from ..teams.registry import AgentRegistry


class DynamicRouter:
    """Router định tuyến động dựa trên nội dung và context.

    DynamicRouter cung cấp khả năng định tuyến linh hoạt dựa trên:
    - Commands (/code, /research, etc.)
    - Hashtags (#work, #personal, etc.)
    - Keywords và capabilities
    - Context metadata (user preferences, session history)
    - Load balancing

    Attributes:
        agents: Dict mapping agent_id -> BaseAgent.
        analyzer: ContentAnalyzer để phân tích tin nhắn.
        content_router: ContentRouter cho content-based routing.
        hook_registry: RoutingHookRegistry cho pre/post hooks.
        default_agent: Agent mặc định.
        enable_load_balancing: Bật load balancing không.
    """

    def __init__(
        self,
        agents: Optional[dict[str, BaseAgent]] = None,
        registry: Optional[AgentRegistry] = None,
        analyzer: Optional[ContentAnalyzer] = None,
        command_mapping: Optional[dict[str, str]] = None,
        hashtag_mapping: Optional[dict[str, str]] = None,
        capability_mapping: Optional[dict[str, str]] = None,
        intent_mapping: Optional[dict[str, str]] = None,
        context_routing_rules: Optional[list[dict[str, Any]]] = None,
        default_agent: str = "default",
        hook_registry: Optional[RoutingHookRegistry] = None,
        enable_load_balancing: bool = False,
        keyword_patterns: Optional[dict[str, list[str]]] = None,
        intent_patterns: Optional[dict[str, list[str]]] = None,
    ) -> None:
        """Khởi tạo DynamicRouter.

        Args:
            agents: Dict mapping agent_id -> BaseAgent.
            registry: AgentRegistry để tự động discover agents.
            analyzer: ContentAnalyzer tùy chỉnh.
            command_mapping: Dict mapping command -> agent_id.
            hashtag_mapping: Dict mapping hashtag -> agent_id.
            capability_mapping: Dict mapping capability -> agent_id.
            intent_mapping: Dict mapping intent -> agent_id.
            context_routing_rules: Danh sách rules cho context-based routing.
            default_agent: ID của agent mặc định.
            hook_registry: RoutingHookRegistry cho hooks.
            enable_load_balancing: Bật load balancing cho similar agents.
            keyword_patterns: Patterns để map keywords sang capabilities.
            intent_patterns: Patterns để phát hiện intents.
        """
        self.agents = agents or {}
        self._registry = registry
        self.default_agent = default_agent
        self.enable_load_balancing = enable_load_balancing

        # Nếu có registry, đồng bộ agents
        if registry:
            self._sync_from_registry()

        # Tạo analyzer
        self.analyzer = analyzer or ContentAnalyzer(
            keyword_patterns=keyword_patterns,
            intent_patterns=intent_patterns,
        )

        # Tạo content router
        self.content_router = ContentRouter(
            agents=self.agents,
            command_mapping=command_mapping,
            hashtag_mapping=hashtag_mapping,
            capability_mapping=capability_mapping,
            intent_mapping=intent_mapping,
            default_agent=default_agent,
            analyzer=self.analyzer,
        )

        # Hook registry
        self.hook_registry = hook_registry or RoutingHookRegistry()

        # Context routing rules
        self._context_rules = context_routing_rules or []

        # Load balancing state
        self._load_balancer_state: dict[str, int] = {}  # capability -> last index

    def _sync_from_registry(self) -> None:
        """Đồng bộ agents từ registry."""
        if self._registry:
            for agent in self._registry.list_agents():
                self.agents[agent.id] = agent

    async def route(self, request: RoutingRequest) -> str:
        """Định tuyến request đến agent phù hợp.

        Args:
            request: RoutingRequest cần định tuyến.

        Returns:
            ID của agent sẽ xử lý request.
        """
        result = await self.route_with_result(request)
        return result.agent_id

    async def route_with_result(
        self,
        request: RoutingRequest
    ) -> RoutingResult:
        """Định tuyến request và trả về kết quả chi tiết.

        Quy trình routing:
        1. Fire PRE_ROUTE hooks (có thể modify request)
        2. Phân tích nội dung để lấy hints
        3. Kiểm tra context-based rules
        4. Sử dụng content router cho content-based routing
        5. Apply load balancing nếu cần
        6. Fire POST_ROUTE hooks

        Args:
            request: RoutingRequest cần định tuyến.

        Returns:
            RoutingResult với thông tin chi tiết.
        """
        routing_id = str(uuid.uuid4())[:8]

        try:
            # 1. Fire PRE_ROUTE hooks
            modified_request = await self.hook_registry.fire_pre_route(
                request,
                routing_id=routing_id,
            )
            request = modified_request

            # 2. Phân tích nội dung
            hints = self.analyzer.analyze(request.message)

            # 3. Kiểm tra context-based routing trước
            context_result = await self._route_by_context(request, hints, routing_id)
            if context_result:
                await self._fire_post_hooks(request, context_result, routing_id)
                return context_result

            # 4. Sử dụng content router
            content_result = await self.content_router.route_with_result(request)

            # 5. Apply load balancing nếu cần
            if self.enable_load_balancing and hints.required_capabilities:
                balanced_result = await self._apply_load_balancing(
                    content_result,
                    hints,
                    routing_id,
                )
                if balanced_result:
                    content_result = balanced_result

            # 6. Fire POST_ROUTE hooks
            await self._fire_post_hooks(request, content_result, routing_id)

            # Check for fallback
            if content_result.metadata.get("is_default"):
                await self.hook_registry.fire_on_fallback(
                    request,
                    content_result,
                    routing_id=routing_id,
                )

            return content_result

        except Exception as e:
            # Fire error hooks
            await self.hook_registry.fire_on_error(
                request,
                e,
                routing_id=routing_id,
            )
            raise

    async def _route_by_context(
        self,
        request: RoutingRequest,
        hints: ContentHints,
        routing_id: str,
    ) -> Optional[RoutingResult]:
        """Định tuyến dựa trên context.

        Kiểm tra các context rules và trả về agent phù hợp.

        Args:
            request: RoutingRequest.
            hints: ContentHints từ phân tích.
            routing_id: ID của routing operation.

        Returns:
            RoutingResult nếu có rule match, None nếu không.
        """
        if not request.context or not self._context_rules:
            return None

        for rule in self._context_rules:
            if self._matches_context_rule(request.context, rule):
                agent_id = rule.get("agent_id")
                if agent_id and self._is_valid_agent(agent_id):
                    return RoutingResult(
                        agent_id=agent_id,
                        confidence=rule.get("confidence", 0.9),
                        metadata={
                            "router_type": "dynamic:context",
                            "matched_rule": rule.get("name", "unnamed"),
                            "routing_id": routing_id,
                        },
                    )

        return None

    def _matches_context_rule(
        self,
        context: dict[str, Any],
        rule: dict[str, Any],
    ) -> bool:
        """Kiểm tra context có match với rule không.

        Args:
            context: Context từ request.
            rule: Rule cần kiểm tra.

        Returns:
            True nếu context match với rule.
        """
        conditions = rule.get("conditions", {})

        for key, expected in conditions.items():
            # Support nested keys với dot notation
            value = self._get_nested_value(context, key)

            if isinstance(expected, dict):
                # Support operators: $eq, $ne, $in, $contains, $exists
                if "$eq" in expected and value != expected["$eq"]:
                    return False
                if "$ne" in expected and value == expected["$ne"]:
                    return False
                if "$in" in expected and value not in expected["$in"]:
                    return False
                if "$contains" in expected:
                    if not isinstance(value, (list, str)):
                        return False
                    if expected["$contains"] not in value:
                        return False
                if "$exists" in expected:
                    exists = value is not None
                    if expected["$exists"] != exists:
                        return False
            else:
                # Simple equality check
                if value != expected:
                    return False

        return True

    def _get_nested_value(
        self,
        data: dict[str, Any],
        key: str,
    ) -> Any:
        """Lấy giá trị nested từ dict với dot notation.

        Args:
            data: Dict chứa dữ liệu.
            key: Key với dot notation (ví dụ: "user.preferences.theme").

        Returns:
            Giá trị tương ứng hoặc None.
        """
        keys = key.split(".")
        value = data

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return None

        return value

    async def _apply_load_balancing(
        self,
        result: RoutingResult,
        hints: ContentHints,
        routing_id: str,
    ) -> Optional[RoutingResult]:
        """Apply load balancing cho similar agents.

        Nếu có nhiều agents với cùng capability, chọn agent
        theo round-robin để phân tải.

        Args:
            result: RoutingResult hiện tại.
            hints: ContentHints chứa capabilities.
            routing_id: ID của routing operation.

        Returns:
            RoutingResult mới nếu có load balancing, None nếu không.
        """
        if not hints.required_capabilities:
            return None

        # Tìm agents có capability đầu tiên
        capability = hints.required_capabilities[0]
        capable_agents = self._find_agents_by_capability(capability)

        if len(capable_agents) <= 1:
            return None

        # Round-robin selection
        if capability not in self._load_balancer_state:
            self._load_balancer_state[capability] = 0

        idx = self._load_balancer_state[capability] % len(capable_agents)
        selected_agent = capable_agents[idx]
        self._load_balancer_state[capability] = idx + 1

        return RoutingResult(
            agent_id=selected_agent,
            confidence=result.confidence,
            metadata={
                **result.metadata,
                "router_type": "dynamic:load_balanced",
                "load_balance_index": idx,
                "capable_agents": capable_agents,
            },
        )

    def _find_agents_by_capability(self, capability: str) -> list[str]:
        """Tìm các agents có capability cụ thể.

        Args:
            capability: Capability cần tìm.

        Returns:
            Danh sách agent IDs có capability đó.
        """
        capable = []

        for agent_id, agent in self.agents.items():
            if hasattr(agent, "has_capability") and agent.has_capability(capability):
                capable.append(agent_id)

        return capable

    async def _fire_post_hooks(
        self,
        request: RoutingRequest,
        result: RoutingResult,
        routing_id: str,
    ) -> None:
        """Fire POST_ROUTE hooks.

        Args:
            request: RoutingRequest đã xử lý.
            result: RoutingResult.
            routing_id: ID của routing operation.
        """
        await self.hook_registry.fire_post_route(
            request,
            result,
            routing_id=routing_id,
        )

    def _is_valid_agent(self, agent_id: str) -> bool:
        """Kiểm tra agent_id có hợp lệ không.

        Args:
            agent_id: ID của agent.

        Returns:
            True nếu agent_id hợp lệ.
        """
        if not self.agents:
            return True
        return agent_id in self.agents

    # Agent management methods

    def register_agent(self, agent_id: str, agent: BaseAgent) -> None:
        """Đăng ký agent với router.

        Args:
            agent_id: ID của agent.
            agent: BaseAgent instance.
        """
        self.agents[agent_id] = agent
        self.content_router.register_agent(agent_id, agent)

    def unregister_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Hủy đăng ký agent.

        Args:
            agent_id: ID của agent.

        Returns:
            Agent đã hủy đăng ký hoặc None.
        """
        agent = self.agents.pop(agent_id, None)
        self.content_router.unregister_agent(agent_id)
        return agent

    def list_agents(self) -> list[str]:
        """Lấy danh sách agent IDs.

        Returns:
            Danh sách agent IDs đã đăng ký.
        """
        return list(self.agents.keys())

    def has_agent(self, agent_id: str) -> bool:
        """Kiểm tra agent có được đăng ký không.

        Args:
            agent_id: ID của agent.

        Returns:
            True nếu agent đã được đăng ký.
        """
        return agent_id in self.agents

    # Mapping management methods

    def add_command_mapping(self, command: str, agent_id: str) -> None:
        """Thêm command mapping.

        Args:
            command: Command (với hoặc không có /).
            agent_id: ID của agent.
        """
        self.content_router.add_command_mapping(command, agent_id)

    def add_hashtag_mapping(self, hashtag: str, agent_id: str) -> None:
        """Thêm hashtag mapping.

        Args:
            hashtag: Hashtag (với hoặc không có #).
            agent_id: ID của agent.
        """
        self.content_router.add_hashtag_mapping(hashtag, agent_id)

    def add_capability_mapping(self, capability: str, agent_id: str) -> None:
        """Thêm capability mapping.

        Args:
            capability: Capability.
            agent_id: ID của agent.
        """
        self.content_router.add_capability_mapping(capability, agent_id)

    def add_intent_mapping(self, intent: str, agent_id: str) -> None:
        """Thêm intent mapping.

        Args:
            intent: Intent.
            agent_id: ID của agent.
        """
        self.content_router.add_intent_mapping(intent, agent_id)

    def add_context_rule(
        self,
        name: str,
        conditions: dict[str, Any],
        agent_id: str,
        confidence: float = 0.9,
    ) -> None:
        """Thêm context routing rule.

        Args:
            name: Tên của rule.
            conditions: Dict chứa các conditions.
            agent_id: ID của agent khi rule match.
            confidence: Độ tin cậy của rule.

        Ví dụ:
            router.add_context_rule(
                name="premium_user",
                conditions={"user.tier": "premium"},
                agent_id="premium_agent",
            )
        """
        self._context_rules.append({
            "name": name,
            "conditions": conditions,
            "agent_id": agent_id,
            "confidence": confidence,
        })

    def remove_context_rule(self, name: str) -> bool:
        """Xóa context routing rule.

        Args:
            name: Tên của rule.

        Returns:
            True nếu rule được xóa.
        """
        original_len = len(self._context_rules)
        self._context_rules = [r for r in self._context_rules if r.get("name") != name]
        return len(self._context_rules) < original_len

    # Hook management methods

    def add_hook(self, hook: Any) -> None:
        """Thêm routing hook.

        Args:
            hook: RoutingHook instance.
        """
        self.hook_registry.register_hook(hook)

    def remove_hook(self, hook: Any) -> bool:
        """Xóa routing hook.

        Args:
            hook: RoutingHook instance.

        Returns:
            True nếu hook được xóa.
        """
        return self.hook_registry.unregister_hook(hook)

    # Statistics and info

    def get_stats(self) -> dict[str, Any]:
        """Lấy thống kê về router.

        Returns:
            Dict chứa các thông tin thống kê.
        """
        return {
            "registered_agents": len(self.agents),
            "agent_ids": list(self.agents.keys()),
            "default_agent": self.default_agent,
            "load_balancing_enabled": self.enable_load_balancing,
            "context_rules_count": len(self._context_rules),
            "hooks_count": len(self.hook_registry),
            "mappings": self.content_router.get_all_mappings(),
        }

    def get_all_mappings(self) -> dict[str, dict[str, str]]:
        """Lấy tất cả mappings.

        Returns:
            Dict chứa tất cả mappings.
        """
        return self.content_router.get_all_mappings()


class DynamicRouterBuilder:
    """Builder pattern để tạo DynamicRouter.

    Cung cấp fluent API để cấu hình và tạo DynamicRouter.

    Ví dụ:
        router = (
            DynamicRouterBuilder()
            .with_default_agent("general")
            .add_command("/code", "coder")
            .add_hashtag("#work", "work_agent")
            .enable_load_balancing()
            .build()
        )
    """

    def __init__(self) -> None:
        """Khởi tạo builder."""
        self._agents: dict[str, BaseAgent] = {}
        self._registry: Optional[AgentRegistry] = None
        self._command_mapping: dict[str, str] = {}
        self._hashtag_mapping: dict[str, str] = {}
        self._capability_mapping: dict[str, str] = {}
        self._intent_mapping: dict[str, str] = {}
        self._context_rules: list[dict[str, Any]] = []
        self._default_agent = "default"
        self._hook_registry: Optional[RoutingHookRegistry] = None
        self._enable_load_balancing = False
        self._keyword_patterns: dict[str, list[str]] = {}
        self._intent_patterns: dict[str, list[str]] = {}

    def with_agents(self, agents: dict[str, BaseAgent]) -> DynamicRouterBuilder:
        """Đặt agents.

        Args:
            agents: Dict mapping agent_id -> BaseAgent.

        Returns:
            Self để chain method calls.
        """
        self._agents = agents
        return self

    def with_registry(self, registry: AgentRegistry) -> DynamicRouterBuilder:
        """Đặt agent registry.

        Args:
            registry: AgentRegistry.

        Returns:
            Self để chain method calls.
        """
        self._registry = registry
        return self

    def with_default_agent(self, agent_id: str) -> DynamicRouterBuilder:
        """Đặt default agent.

        Args:
            agent_id: ID của default agent.

        Returns:
            Self để chain method calls.
        """
        self._default_agent = agent_id
        return self

    def add_command(self, command: str, agent_id: str) -> DynamicRouterBuilder:
        """Thêm command mapping.

        Args:
            command: Command (với hoặc không có /).
            agent_id: ID của agent.

        Returns:
            Self để chain method calls.
        """
        if not command.startswith("/"):
            command = f"/{command}"
        self._command_mapping[command.lower()] = agent_id
        return self

    def add_hashtag(self, hashtag: str, agent_id: str) -> DynamicRouterBuilder:
        """Thêm hashtag mapping.

        Args:
            hashtag: Hashtag (với hoặc không có #).
            agent_id: ID của agent.

        Returns:
            Self để chain method calls.
        """
        if not hashtag.startswith("#"):
            hashtag = f"#{hashtag}"
        self._hashtag_mapping[hashtag.lower()] = agent_id
        return self

    def add_capability(
        self,
        capability: str,
        agent_id: str
    ) -> DynamicRouterBuilder:
        """Thêm capability mapping.

        Args:
            capability: Capability.
            agent_id: ID của agent.

        Returns:
            Self để chain method calls.
        """
        self._capability_mapping[capability] = agent_id
        return self

    def add_intent(self, intent: str, agent_id: str) -> DynamicRouterBuilder:
        """Thêm intent mapping.

        Args:
            intent: Intent.
            agent_id: ID của agent.

        Returns:
            Self để chain method calls.
        """
        self._intent_mapping[intent] = agent_id
        return self

    def add_context_rule(
        self,
        name: str,
        conditions: dict[str, Any],
        agent_id: str,
        confidence: float = 0.9,
    ) -> DynamicRouterBuilder:
        """Thêm context routing rule.

        Args:
            name: Tên của rule.
            conditions: Dict chứa conditions.
            agent_id: ID của agent.
            confidence: Độ tin cậy.

        Returns:
            Self để chain method calls.
        """
        self._context_rules.append({
            "name": name,
            "conditions": conditions,
            "agent_id": agent_id,
            "confidence": confidence,
        })
        return self

    def with_hooks(self, registry: RoutingHookRegistry) -> DynamicRouterBuilder:
        """Đặt hook registry.

        Args:
            registry: RoutingHookRegistry.

        Returns:
            Self để chain method calls.
        """
        self._hook_registry = registry
        return self

    def enable_load_balancing(self) -> DynamicRouterBuilder:
        """Bật load balancing.

        Returns:
            Self để chain method calls.
        """
        self._enable_load_balancing = True
        return self

    def add_keyword_pattern(
        self,
        capability: str,
        keywords: list[str]
    ) -> DynamicRouterBuilder:
        """Thêm keyword patterns cho capability.

        Args:
            capability: Capability cần map.
            keywords: Danh sách keywords.

        Returns:
            Self để chain method calls.
        """
        if capability not in self._keyword_patterns:
            self._keyword_patterns[capability] = []
        self._keyword_patterns[capability].extend(keywords)
        return self

    def add_intent_pattern(
        self,
        intent: str,
        patterns: list[str]
    ) -> DynamicRouterBuilder:
        """Thêm intent patterns.

        Args:
            intent: Intent.
            patterns: Danh sách regex patterns.

        Returns:
            Self để chain method calls.
        """
        if intent not in self._intent_patterns:
            self._intent_patterns[intent] = []
        self._intent_patterns[intent].extend(patterns)
        return self

    def build(self) -> DynamicRouter:
        """Tạo DynamicRouter từ configuration.

        Returns:
            DynamicRouter đã được cấu hình.
        """
        return DynamicRouter(
            agents=self._agents,
            registry=self._registry,
            command_mapping=self._command_mapping,
            hashtag_mapping=self._hashtag_mapping,
            capability_mapping=self._capability_mapping,
            intent_mapping=self._intent_mapping,
            context_routing_rules=self._context_rules,
            default_agent=self._default_agent,
            hook_registry=self._hook_registry,
            enable_load_balancing=self._enable_load_balancing,
            keyword_patterns=self._keyword_patterns,
            intent_patterns=self._intent_patterns,
        )
