"""Agent Routing Engine.

Module này cung cấp hệ thống định tuyến tin nhắn/request đến các agent
khác nhau dựa trên các quy tắc và chiến lược routing.

Các thành phần chính:
- RoutingEngine: Engine điều phối routing requests
- RoutingRequest: Request cần được định tuyến
- RoutingRule: Quy tắc định tuyến
- RoutingConfig: Cấu hình cho routing engine
- StaticRouter: Router dựa trên rules cố định
- PatternRouter: Router dựa trên regex patterns
- ContentRouter: Router dựa trên nội dung (commands, hashtags, keywords)
- DynamicRouter: Router động kết hợp nhiều chiến lược
- AgentDiscovery: Tự động phát hiện và quản lý agents

Ví dụ sử dụng:

    from agents_framework.routing import (
        RoutingEngine,
        RoutingConfig,
        RoutingRequest,
        RoutingRule,
    )

    # Tạo config với rules
    config = RoutingConfig(
        default_agent="general",
        strategy="pattern",
        rules=[
            RoutingRule(
                agent_id="code_agent",
                pattern=r"(code|lập trình|python|java)",
                priority=10
            ),
            RoutingRule(
                agent_id="support_agent",
                pattern=r"(help|hỗ trợ|support)",
                priority=5
            ),
        ]
    )

    # Khởi tạo engine
    engine = RoutingEngine(config)

    # Đăng ký agents
    engine.add_agent("general", general_agent)
    engine.add_agent("code_agent", code_agent)
    engine.add_agent("support_agent", support_agent)

    # Routing request
    request = RoutingRequest(
        message="Tôi cần help với Python code",
        sender="user123"
    )

    agent_id = await engine.route(request)  # -> "code_agent"

    # Hoặc route và execute trực tiếp
    result = await engine.route_and_execute(request)

    # Sử dụng DynamicRouter với commands và hashtags
    from agents_framework.routing import DynamicRouter

    router = DynamicRouter(
        command_mapping={"/code": "coder", "/research": "researcher"},
        hashtag_mapping={"#work": "work_agent", "#personal": "personal_agent"},
    )

    request = RoutingRequest(message="#work Schedule meeting tomorrow")
    agent_id = await router.route(request)  # -> "work_agent"
"""

from .base import (
    Router,
    RoutingRequest,
    RoutingResult,
    RoutingRule,
    TimeRangeChecker,
)
from .config import (
    AgentRouteMapping,
    ChannelRoutingConfig,
    RoutingConfig,
    RoutingStrategy,
)
from .discovery import (
    AgentDiscovery,
    CapabilityMatcher,
    DiscoveredAgent,
    LoadBalancingStrategy,
)
from .dynamic import DynamicRouter, DynamicRouterBuilder
from .engine import CombinedRouter, RoutingEngine
from .hooks import (
    AuditRoutingHook,
    LoggingRoutingHook,
    MetricsRoutingHook,
    RequestModifierHook,
    RoutingHook,
    RoutingHookContext,
    RoutingHookEntry,
    RoutingHookRegistry,
    RoutingHookType,
)
from .strategies import (
    ContentAnalyzer,
    ContentHints,
    ContentRouter,
    IntentClassifier,
    KeywordMatcher,
    PatternRouter,
    StaticRouter,
)

__all__ = [
    # Core
    "RoutingEngine",
    "RoutingRequest",
    "RoutingResult",
    "RoutingRule",
    "RoutingConfig",
    # Protocol
    "Router",
    # Strategies
    "StaticRouter",
    "PatternRouter",
    "CombinedRouter",
    # Content-based routing
    "ContentRouter",
    "ContentAnalyzer",
    "ContentHints",
    "KeywordMatcher",
    "IntentClassifier",
    # Dynamic routing
    "DynamicRouter",
    "DynamicRouterBuilder",
    # Discovery
    "AgentDiscovery",
    "DiscoveredAgent",
    "CapabilityMatcher",
    "LoadBalancingStrategy",
    # Hooks
    "RoutingHook",
    "RoutingHookType",
    "RoutingHookContext",
    "RoutingHookEntry",
    "RoutingHookRegistry",
    "LoggingRoutingHook",
    "MetricsRoutingHook",
    "AuditRoutingHook",
    "RequestModifierHook",
    # Config types
    "RoutingStrategy",
    "AgentRouteMapping",
    "ChannelRoutingConfig",
    # Utilities
    "TimeRangeChecker",
]
