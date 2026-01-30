"""Routing Hooks cho Agent Routing Engine.

Module này triển khai hệ thống hooks cho routing lifecycle,
bao gồm pre-routing và post-routing hooks để can thiệp và
giám sát quá trình định tuyến.

Các thành phần chính:
- RoutingHookType: Enum định nghĩa các loại hooks
- RoutingHookContext: Context truyền cho hooks
- RoutingHook: Protocol cho hook implementations
- RoutingHookRegistry: Registry quản lý hooks
- Built-in hooks: LoggingHook, MetricsHook, AuditHook

Ví dụ sử dụng:
    from agents_framework.routing.hooks import (
        RoutingHookRegistry,
        RoutingHookType,
        LoggingRoutingHook,
    )

    # Tạo registry
    registry = RoutingHookRegistry()

    # Đăng ký hook
    registry.register_hook(LoggingRoutingHook())

    # Sử dụng với router
    router = DynamicRouter(
        agents=agents,
        hook_registry=registry,
    )

    # Hooks sẽ được gọi tự động khi route
    agent_id = await router.route(request)
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Optional, Union

from .base import RoutingRequest, RoutingResult

logger = logging.getLogger(__name__)


class RoutingHookType(str, Enum):
    """Các loại routing hooks.

    Attributes:
        PRE_ROUTE: Hook chạy trước khi routing.
        POST_ROUTE: Hook chạy sau khi routing hoàn thành.
        ON_ROUTE_ERROR: Hook chạy khi có lỗi trong quá trình routing.
        ON_FALLBACK: Hook chạy khi fallback sang default agent.
        ON_AGENT_NOT_FOUND: Hook chạy khi không tìm thấy agent.
    """

    PRE_ROUTE = "pre_route"
    POST_ROUTE = "post_route"
    ON_ROUTE_ERROR = "on_route_error"
    ON_FALLBACK = "on_fallback"
    ON_AGENT_NOT_FOUND = "on_agent_not_found"


@dataclass
class RoutingHookContext:
    """Context được truyền cho routing hooks.

    Attributes:
        hook_type: Loại hook đang được thực thi.
        request: RoutingRequest gốc.
        result: RoutingResult (chỉ có trong POST_ROUTE).
        timestamp: Thời điểm hook được trigger.
        routing_id: ID duy nhất cho routing operation này.
        metadata: Metadata bổ sung.
        error: Exception nếu có lỗi.
    """

    hook_type: RoutingHookType
    request: RoutingRequest
    result: Optional[RoutingResult] = None
    timestamp: datetime = field(default_factory=datetime.now)
    routing_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None

    def get(self, key: str, default: Any = None) -> Any:
        """Lấy giá trị từ metadata.

        Args:
            key: Key cần lấy.
            default: Giá trị mặc định nếu key không tồn tại.

        Returns:
            Giá trị tương ứng hoặc default.
        """
        return self.metadata.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Đặt giá trị trong metadata.

        Args:
            key: Key cần đặt.
            value: Giá trị cần đặt.
        """
        self.metadata[key] = value


class RoutingHook(ABC):
    """Abstract base class cho routing hooks.

    Subclass để tạo custom hooks với logic xử lý riêng.

    Attributes:
        hook_types: Set các loại hooks mà hook này xử lý.
        priority: Độ ưu tiên (thấp hơn chạy trước).
        name: Tên của hook để nhận diện.
        enabled: Hook có được bật hay không.
    """

    hook_types: set[RoutingHookType] = set()
    priority: int = 0
    name: str = ""
    enabled: bool = True

    def __init__(
        self,
        name: Optional[str] = None,
        priority: Optional[int] = None,
        enabled: bool = True,
    ) -> None:
        """Khởi tạo hook.

        Args:
            name: Tên tùy chỉnh cho hook.
            priority: Độ ưu tiên tùy chỉnh.
            enabled: Hook có được bật hay không.
        """
        if name:
            self.name = name
        elif not self.name:
            self.name = self.__class__.__name__

        if priority is not None:
            self.priority = priority

        self.enabled = enabled

    @abstractmethod
    async def execute(
        self,
        context: RoutingHookContext,
        **kwargs: Any
    ) -> Optional[RoutingRequest]:
        """Thực thi hook.

        Args:
            context: RoutingHookContext chứa thông tin về routing.
            **kwargs: Arguments bổ sung.

        Returns:
            RoutingRequest đã được modify (cho PRE_ROUTE hooks) hoặc None.
        """
        pass

    def should_run(self, hook_type: RoutingHookType) -> bool:
        """Kiểm tra hook có nên chạy cho loại hook này không.

        Args:
            hook_type: Loại hook đang được trigger.

        Returns:
            True nếu hook nên chạy.
        """
        if not self.enabled:
            return False
        if not self.hook_types:
            return True  # Chạy cho tất cả nếu không specify
        return hook_type in self.hook_types

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, priority={self.priority})"


# Type alias cho hook functions
RoutingHookFunction = Callable[
    [RoutingHookContext],
    Union[Optional[RoutingRequest], Coroutine[Any, Any, Optional[RoutingRequest]]]
]


@dataclass
class RoutingHookEntry:
    """Entry trong hook registry.

    Attributes:
        hook_type: Loại hook.
        callback: Callback function.
        priority: Độ ưu tiên.
        name: Tên hook.
        enabled: Hook có được bật không.
    """

    hook_type: RoutingHookType
    callback: RoutingHookFunction
    priority: int = 0
    name: Optional[str] = None
    enabled: bool = True

    def __hash__(self) -> int:
        return hash((self.hook_type, self.name or id(self.callback)))


class RoutingHookRegistry:
    """Registry quản lý và thực thi routing hooks.

    Cung cấp các method để đăng ký, hủy đăng ký, và thực thi hooks
    cho routing lifecycle.

    Ví dụ:
        registry = RoutingHookRegistry()

        # Đăng ký function hook
        @registry.register(RoutingHookType.PRE_ROUTE)
        async def my_pre_hook(context):
            print(f"Pre-route: {context.request.message}")
            return context.request

        # Đăng ký class hook
        registry.register_hook(LoggingRoutingHook())

        # Fire hooks
        modified_request = await registry.fire_pre_route(request, routing_id="123")
    """

    def __init__(self) -> None:
        """Khởi tạo empty hook registry."""
        self._hooks: dict[RoutingHookType, list[RoutingHookEntry]] = {
            ht: [] for ht in RoutingHookType
        }
        self._class_hooks: list[RoutingHook] = []

    def register(
        self,
        hook_type: Union[RoutingHookType, str],
        callback: Optional[RoutingHookFunction] = None,
        priority: int = 0,
        name: Optional[str] = None,
    ) -> Union[RoutingHookEntry, Callable[[RoutingHookFunction], RoutingHookEntry]]:
        """Đăng ký hook function.

        Có thể sử dụng như decorator hoặc gọi trực tiếp.

        Args:
            hook_type: Loại hook.
            callback: Callback function (optional nếu dùng như decorator).
            priority: Độ ưu tiên (thấp hơn chạy trước).
            name: Tên cho hook.

        Returns:
            RoutingHookEntry nếu có callback, decorator nếu không.

        Ví dụ:
            # Dùng như decorator
            @registry.register(RoutingHookType.PRE_ROUTE)
            async def my_hook(context):
                return context.request

            # Gọi trực tiếp
            registry.register(RoutingHookType.PRE_ROUTE, my_callback)
        """
        if isinstance(hook_type, str):
            hook_type = RoutingHookType(hook_type)

        def decorator(fn: RoutingHookFunction) -> RoutingHookEntry:
            entry = RoutingHookEntry(
                hook_type=hook_type,
                callback=fn,
                priority=priority,
                name=name or getattr(fn, "__name__", "anonymous"),
            )
            self._hooks[hook_type].append(entry)
            self._hooks[hook_type].sort(key=lambda x: x.priority)
            return entry

        if callback is not None:
            return decorator(callback)
        return decorator

    def register_hook(self, hook: RoutingHook) -> None:
        """Đăng ký class-based hook.

        Args:
            hook: RoutingHook instance.
        """
        self._class_hooks.append(hook)
        logger.debug(f"Registered routing hook: {hook.name}")

    def unregister(
        self,
        hook_type: Optional[RoutingHookType] = None,
        name: Optional[str] = None,
        callback: Optional[RoutingHookFunction] = None,
    ) -> int:
        """Hủy đăng ký hooks khớp với criteria.

        Args:
            hook_type: Filter theo loại hook.
            name: Filter theo tên.
            callback: Filter theo callback function.

        Returns:
            Số lượng hooks đã hủy đăng ký.
        """
        count = 0
        hook_types = [hook_type] if hook_type else list(RoutingHookType)

        for ht in hook_types:
            original_len = len(self._hooks[ht])
            self._hooks[ht] = [
                entry
                for entry in self._hooks[ht]
                if not (
                    (name is None or entry.name == name)
                    and (callback is None or entry.callback == callback)
                )
            ]
            count += original_len - len(self._hooks[ht])

        # Hủy đăng ký class hooks theo tên
        if name:
            original_len = len(self._class_hooks)
            self._class_hooks = [h for h in self._class_hooks if h.name != name]
            count += original_len - len(self._class_hooks)

        return count

    def unregister_hook(self, hook: RoutingHook) -> bool:
        """Hủy đăng ký class-based hook.

        Args:
            hook: RoutingHook instance cần hủy.

        Returns:
            True nếu hook được tìm thấy và xóa.
        """
        try:
            self._class_hooks.remove(hook)
            return True
        except ValueError:
            return False

    async def fire(
        self,
        hook_type: Union[RoutingHookType, str],
        context: RoutingHookContext,
        **kwargs: Any,
    ) -> Optional[RoutingRequest]:
        """Fire tất cả hooks của một loại.

        Args:
            hook_type: Loại hook cần fire.
            context: RoutingHookContext.
            **kwargs: Arguments bổ sung.

        Returns:
            RoutingRequest đã được modify (từ PRE_ROUTE hooks) hoặc None.
        """
        if isinstance(hook_type, str):
            hook_type = RoutingHookType(hook_type)

        modified_request: Optional[RoutingRequest] = None

        # Fire function hooks
        for entry in self._hooks[hook_type]:
            if not entry.enabled:
                continue

            try:
                result = entry.callback(context)
                if asyncio.iscoroutine(result):
                    result = await result

                # Cập nhật request nếu hook trả về modified request
                if result is not None and isinstance(result, RoutingRequest):
                    modified_request = result
                    context.request = result

            except Exception as e:
                logger.error(
                    f"Error in routing hook {entry.name} for {hook_type}: {e}",
                    exc_info=True,
                )

        # Fire class hooks
        for hook in self._class_hooks:
            if not hook.should_run(hook_type):
                continue

            try:
                result = await hook.execute(context, **kwargs)

                if result is not None and isinstance(result, RoutingRequest):
                    modified_request = result
                    context.request = result

            except Exception as e:
                logger.error(
                    f"Error in routing hook {hook.name} for {hook_type}: {e}",
                    exc_info=True,
                )

        return modified_request

    async def fire_pre_route(
        self,
        request: RoutingRequest,
        routing_id: str = "",
        **kwargs: Any,
    ) -> RoutingRequest:
        """Fire PRE_ROUTE hooks.

        Args:
            request: RoutingRequest gốc.
            routing_id: ID của routing operation.
            **kwargs: Arguments bổ sung.

        Returns:
            RoutingRequest (có thể đã được modify bởi hooks).
        """
        context = RoutingHookContext(
            hook_type=RoutingHookType.PRE_ROUTE,
            request=request,
            routing_id=routing_id,
            metadata=dict(kwargs),
        )

        modified = await self.fire(RoutingHookType.PRE_ROUTE, context, **kwargs)
        return modified if modified is not None else request

    async def fire_post_route(
        self,
        request: RoutingRequest,
        result: RoutingResult,
        routing_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Fire POST_ROUTE hooks.

        Args:
            request: RoutingRequest đã xử lý.
            result: RoutingResult từ routing.
            routing_id: ID của routing operation.
            **kwargs: Arguments bổ sung.
        """
        context = RoutingHookContext(
            hook_type=RoutingHookType.POST_ROUTE,
            request=request,
            result=result,
            routing_id=routing_id,
            metadata=dict(kwargs),
        )

        await self.fire(RoutingHookType.POST_ROUTE, context, **kwargs)

    async def fire_on_error(
        self,
        request: RoutingRequest,
        error: Exception,
        routing_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Fire ON_ROUTE_ERROR hooks.

        Args:
            request: RoutingRequest gây lỗi.
            error: Exception đã xảy ra.
            routing_id: ID của routing operation.
            **kwargs: Arguments bổ sung.
        """
        context = RoutingHookContext(
            hook_type=RoutingHookType.ON_ROUTE_ERROR,
            request=request,
            error=error,
            routing_id=routing_id,
            metadata=dict(kwargs),
        )

        await self.fire(RoutingHookType.ON_ROUTE_ERROR, context, **kwargs)

    async def fire_on_fallback(
        self,
        request: RoutingRequest,
        result: RoutingResult,
        routing_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Fire ON_FALLBACK hooks.

        Args:
            request: RoutingRequest đã fallback.
            result: RoutingResult với default agent.
            routing_id: ID của routing operation.
            **kwargs: Arguments bổ sung.
        """
        context = RoutingHookContext(
            hook_type=RoutingHookType.ON_FALLBACK,
            request=request,
            result=result,
            routing_id=routing_id,
            metadata=dict(kwargs),
        )

        await self.fire(RoutingHookType.ON_FALLBACK, context, **kwargs)

    def enable(
        self,
        hook_type: Optional[RoutingHookType] = None,
        name: Optional[str] = None,
    ) -> int:
        """Bật hooks khớp với criteria.

        Args:
            hook_type: Filter theo loại hook.
            name: Filter theo tên.

        Returns:
            Số lượng hooks đã bật.
        """
        return self._set_enabled(True, hook_type, name)

    def disable(
        self,
        hook_type: Optional[RoutingHookType] = None,
        name: Optional[str] = None,
    ) -> int:
        """Tắt hooks khớp với criteria.

        Args:
            hook_type: Filter theo loại hook.
            name: Filter theo tên.

        Returns:
            Số lượng hooks đã tắt.
        """
        return self._set_enabled(False, hook_type, name)

    def _set_enabled(
        self,
        enabled: bool,
        hook_type: Optional[RoutingHookType],
        name: Optional[str],
    ) -> int:
        """Đặt trạng thái enabled cho hooks."""
        count = 0
        hook_types = [hook_type] if hook_type else list(RoutingHookType)

        for ht in hook_types:
            for entry in self._hooks[ht]:
                if name is None or entry.name == name:
                    entry.enabled = enabled
                    count += 1

        # Cập nhật class hooks
        for hook in self._class_hooks:
            if name is None or hook.name == name:
                hook.enabled = enabled
                count += 1

        return count

    def list_hooks(
        self,
        hook_type: Optional[RoutingHookType] = None,
    ) -> list[Union[RoutingHookEntry, RoutingHook]]:
        """Liệt kê các hooks đã đăng ký.

        Args:
            hook_type: Filter theo loại hook (optional).

        Returns:
            Danh sách hooks.
        """
        result: list[Union[RoutingHookEntry, RoutingHook]] = []

        if hook_type:
            result.extend(self._hooks[hook_type])
        else:
            for entries in self._hooks.values():
                result.extend(entries)

        # Thêm class hooks
        if hook_type:
            result.extend([h for h in self._class_hooks if h.should_run(hook_type)])
        else:
            result.extend(self._class_hooks)

        return result

    def clear(self, hook_type: Optional[RoutingHookType] = None) -> None:
        """Xóa tất cả hooks.

        Args:
            hook_type: Chỉ xóa loại hook cụ thể (optional).
        """
        if hook_type:
            self._hooks[hook_type] = []
        else:
            for ht in RoutingHookType:
                self._hooks[ht] = []
            self._class_hooks = []

    def __len__(self) -> int:
        """Trả về tổng số hooks đã đăng ký."""
        count = sum(len(entries) for entries in self._hooks.values())
        count += len(self._class_hooks)
        return count


# Built-in Hooks


class LoggingRoutingHook(RoutingHook):
    """Hook để log routing events.

    Logs các sự kiện routing với log level có thể cấu hình.

    Ví dụ:
        registry = RoutingHookRegistry()
        registry.register_hook(LoggingRoutingHook(log_level=logging.DEBUG))
    """

    hook_types = {
        RoutingHookType.PRE_ROUTE,
        RoutingHookType.POST_ROUTE,
        RoutingHookType.ON_ROUTE_ERROR,
        RoutingHookType.ON_FALLBACK,
    }
    priority = -100  # Chạy sớm

    def __init__(
        self,
        name: str = "LoggingRoutingHook",
        log_level: int = logging.INFO,
        include_message: bool = True,
    ) -> None:
        """Khởi tạo logging hook.

        Args:
            name: Tên hook.
            log_level: Log level để sử dụng.
            include_message: Có include message content trong log không.
        """
        super().__init__(name)
        self.log_level = log_level
        self.include_message = include_message
        self._logger = logging.getLogger(f"{__name__}.{name}")

    async def execute(
        self,
        context: RoutingHookContext,
        **kwargs: Any
    ) -> Optional[RoutingRequest]:
        """Log routing event."""
        message_preview = ""
        if self.include_message:
            msg = context.request.message
            message_preview = f" msg={msg[:50]}..." if len(msg) > 50 else f" msg={msg}"

        if context.hook_type == RoutingHookType.PRE_ROUTE:
            self._logger.log(
                self.log_level,
                f"[PRE_ROUTE] routing_id={context.routing_id}{message_preview}"
            )

        elif context.hook_type == RoutingHookType.POST_ROUTE:
            result = context.result
            agent_id = result.agent_id if result else "unknown"
            confidence = result.confidence if result else 0.0
            self._logger.log(
                self.log_level,
                f"[POST_ROUTE] routing_id={context.routing_id} "
                f"agent={agent_id} confidence={confidence:.2f}"
            )

        elif context.hook_type == RoutingHookType.ON_ROUTE_ERROR:
            self._logger.error(
                f"[ON_ROUTE_ERROR] routing_id={context.routing_id} "
                f"error={context.error}"
            )

        elif context.hook_type == RoutingHookType.ON_FALLBACK:
            result = context.result
            agent_id = result.agent_id if result else "unknown"
            self._logger.warning(
                f"[ON_FALLBACK] routing_id={context.routing_id} "
                f"fallback_agent={agent_id}"
            )

        return None


class MetricsRoutingHook(RoutingHook):
    """Hook để thu thập metrics về routing.

    Thu thập timing, success rates, và các metrics khác.

    Ví dụ:
        metrics_hook = MetricsRoutingHook()
        registry.register_hook(metrics_hook)

        # Sau một số routing operations
        print(metrics_hook.get_metrics())
    """

    hook_types = {
        RoutingHookType.PRE_ROUTE,
        RoutingHookType.POST_ROUTE,
        RoutingHookType.ON_ROUTE_ERROR,
        RoutingHookType.ON_FALLBACK,
    }
    priority = 100  # Chạy muộn để đo timing chính xác

    def __init__(self, name: str = "MetricsRoutingHook") -> None:
        """Khởi tạo metrics hook."""
        super().__init__(name)
        self._route_starts: dict[str, datetime] = {}
        self._metrics: dict[str, Any] = {
            "total_routes": 0,
            "successful": 0,
            "fallbacks": 0,
            "errors": 0,
            "total_duration_ms": 0.0,
            "agent_counts": {},  # agent_id -> count
            "error_messages": [],
        }

    async def execute(
        self,
        context: RoutingHookContext,
        **kwargs: Any
    ) -> Optional[RoutingRequest]:
        """Thu thập metrics."""
        routing_id = context.routing_id

        if context.hook_type == RoutingHookType.PRE_ROUTE:
            self._route_starts[routing_id] = context.timestamp
            self._metrics["total_routes"] += 1

        elif context.hook_type == RoutingHookType.POST_ROUTE:
            # Tính duration
            if routing_id in self._route_starts:
                start = self._route_starts.pop(routing_id)
                duration_ms = (context.timestamp - start).total_seconds() * 1000
                self._metrics["total_duration_ms"] += duration_ms

            self._metrics["successful"] += 1

            # Đếm theo agent
            if context.result:
                agent_id = context.result.agent_id
                if agent_id not in self._metrics["agent_counts"]:
                    self._metrics["agent_counts"][agent_id] = 0
                self._metrics["agent_counts"][agent_id] += 1

        elif context.hook_type == RoutingHookType.ON_ROUTE_ERROR:
            self._metrics["errors"] += 1
            if context.error:
                self._metrics["error_messages"].append(str(context.error))

        elif context.hook_type == RoutingHookType.ON_FALLBACK:
            self._metrics["fallbacks"] += 1

        return None

    def get_metrics(self) -> dict[str, Any]:
        """Lấy metrics đã thu thập.

        Returns:
            Dict chứa các metrics.
        """
        total = self._metrics["total_routes"]
        return {
            **self._metrics,
            "success_rate": (
                self._metrics["successful"] / total if total > 0 else 0.0
            ),
            "fallback_rate": (
                self._metrics["fallbacks"] / total if total > 0 else 0.0
            ),
            "error_rate": (
                self._metrics["errors"] / total if total > 0 else 0.0
            ),
            "avg_duration_ms": (
                self._metrics["total_duration_ms"] / total if total > 0 else 0.0
            ),
        }

    def reset(self) -> None:
        """Reset tất cả metrics."""
        self._route_starts.clear()
        self._metrics = {
            "total_routes": 0,
            "successful": 0,
            "fallbacks": 0,
            "errors": 0,
            "total_duration_ms": 0.0,
            "agent_counts": {},
            "error_messages": [],
        }


class AuditRoutingHook(RoutingHook):
    """Hook để audit trail cho routing decisions.

    Ghi lại tất cả routing decisions để review và debugging.

    Ví dụ:
        audit_hook = AuditRoutingHook(max_entries=1000)
        registry.register_hook(audit_hook)

        # Lấy audit trail
        entries = audit_hook.get_entries()
    """

    hook_types = {
        RoutingHookType.POST_ROUTE,
        RoutingHookType.ON_ROUTE_ERROR,
        RoutingHookType.ON_FALLBACK,
    }
    priority = 50

    def __init__(
        self,
        name: str = "AuditRoutingHook",
        max_entries: int = 1000,
    ) -> None:
        """Khởi tạo audit hook.

        Args:
            name: Tên hook.
            max_entries: Số entries tối đa để giữ.
        """
        super().__init__(name)
        self.max_entries = max_entries
        self._entries: list[dict[str, Any]] = []

    async def execute(
        self,
        context: RoutingHookContext,
        **kwargs: Any
    ) -> Optional[RoutingRequest]:
        """Ghi audit entry."""
        entry = {
            "timestamp": context.timestamp.isoformat(),
            "routing_id": context.routing_id,
            "hook_type": context.hook_type.value,
            "message": context.request.message[:100],  # Truncate
            "sender": context.request.sender,
        }

        if context.result:
            entry["agent_id"] = context.result.agent_id
            entry["confidence"] = context.result.confidence
            entry["matched_by"] = context.result.metadata.get("matched_by")

        if context.error:
            entry["error"] = str(context.error)

        self._entries.append(entry)

        # Trim nếu vượt quá max
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]

        return None

    def get_entries(
        self,
        limit: Optional[int] = None,
        agent_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Lấy audit entries.

        Args:
            limit: Số entries tối đa để trả về.
            agent_id: Filter theo agent_id.

        Returns:
            Danh sách audit entries.
        """
        entries = self._entries

        if agent_id:
            entries = [e for e in entries if e.get("agent_id") == agent_id]

        if limit:
            entries = entries[-limit:]

        return entries

    def clear(self) -> None:
        """Xóa tất cả audit entries."""
        self._entries = []


class RequestModifierHook(RoutingHook):
    """Hook để modify request trước khi routing.

    Cho phép transform message, thêm context, hoặc metadata
    trước khi routing decision được thực hiện.

    Ví dụ:
        def add_context(request):
            request.context["timestamp"] = datetime.now().isoformat()
            return request

        modifier = RequestModifierHook(modifier_fn=add_context)
        registry.register_hook(modifier)
    """

    hook_types = {RoutingHookType.PRE_ROUTE}
    priority = -50  # Chạy sớm

    def __init__(
        self,
        name: str = "RequestModifierHook",
        modifier_fn: Optional[Callable[[RoutingRequest], RoutingRequest]] = None,
    ) -> None:
        """Khởi tạo request modifier hook.

        Args:
            name: Tên hook.
            modifier_fn: Function để modify request.
        """
        super().__init__(name)
        self._modifier_fn = modifier_fn

    async def execute(
        self,
        context: RoutingHookContext,
        **kwargs: Any
    ) -> Optional[RoutingRequest]:
        """Modify request nếu có modifier function."""
        if self._modifier_fn:
            return self._modifier_fn(context.request)
        return None

    def set_modifier(
        self,
        modifier_fn: Callable[[RoutingRequest], RoutingRequest]
    ) -> None:
        """Đặt modifier function.

        Args:
            modifier_fn: Function để modify request.
        """
        self._modifier_fn = modifier_fn
