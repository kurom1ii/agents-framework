"""
Lifecycle Management cho Spawned Agents.

Module này quản lý vòng đời của các sub-agents được spawn,
bao gồm tracking trạng thái, cleanup, và error recovery.

Các trạng thái lifecycle:
- INITIALIZING: Đang khởi tạo
- RUNNING: Đang thực thi
- COMPLETED: Hoàn thành thành công
- FAILED: Thất bại
- TIMEOUT: Hết thời gian
- CANCELLED: Bị hủy
- CLEANED_UP: Đã dọn dẹp
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
)

if TYPE_CHECKING:
    from .spawner import SpawnConfig


class SpawnLifecycleState(str, Enum):
    """
    Trạng thái lifecycle của spawned agent.

    Các trạng thái chuyển đổi:
    - INITIALIZING -> RUNNING (khi bắt đầu thực thi)
    - RUNNING -> COMPLETED (khi hoàn thành thành công)
    - RUNNING -> FAILED (khi gặp lỗi)
    - RUNNING -> TIMEOUT (khi hết thời gian)
    - RUNNING -> CANCELLED (khi bị hủy)
    - COMPLETED/FAILED/TIMEOUT/CANCELLED -> CLEANED_UP (sau khi dọn dẹp)
    """
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    CLEANED_UP = "cleaned_up"


@dataclass
class LifecycleEvent:
    """
    Sự kiện trong lifecycle của spawned agent.

    Mỗi sự kiện ghi lại một thay đổi trạng thái hoặc action
    quan trọng trong vòng đời của agent.

    Attributes:
        event_type: Loại sự kiện
        timestamp: Thời điểm xảy ra
        from_state: Trạng thái trước
        to_state: Trạng thái sau
        message: Thông điệp mô tả
        metadata: Dữ liệu bổ sung
    """
    event_type: str
    timestamp: datetime
    from_state: Optional[SpawnLifecycleState] = None
    to_state: Optional[SpawnLifecycleState] = None
    message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển event thành dictionary."""
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "from_state": self.from_state.value if self.from_state else None,
            "to_state": self.to_state.value if self.to_state else None,
            "message": self.message,
            "metadata": self.metadata,
        }


@dataclass
class LifecycleMetrics:
    """
    Metrics cho lifecycle của spawned agent.

    Theo dõi các thông số hiệu suất và resource usage
    trong suốt vòng đời của agent.

    Attributes:
        start_time_ms: Thời điểm bắt đầu (milliseconds since epoch)
        end_time_ms: Thời điểm kết thúc
        duration_ms: Thời gian thực thi (milliseconds)
        tokens_used: Tổng tokens đã sử dụng
        turns_executed: Số turns đã thực thi
        errors_count: Số lỗi đã gặp
        retries_count: Số lần retry
    """
    start_time_ms: int = 0
    end_time_ms: int = 0
    duration_ms: int = 0
    tokens_used: int = 0
    turns_executed: int = 0
    errors_count: int = 0
    retries_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển metrics thành dictionary."""
        return {
            "start_time_ms": self.start_time_ms,
            "end_time_ms": self.end_time_ms,
            "duration_ms": self.duration_ms,
            "tokens_used": self.tokens_used,
            "turns_executed": self.turns_executed,
            "errors_count": self.errors_count,
            "retries_count": self.retries_count,
        }


class SpawnedAgentLifecycle:
    """
    Quản lý lifecycle của một spawned agent.

    Class này theo dõi và quản lý toàn bộ vòng đời của
    một sub-agent được spawn, từ khởi tạo đến cleanup.

    Tính năng:
    - Theo dõi trạng thái lifecycle
    - Ghi lại các events quan trọng
    - Thu thập metrics
    - Xử lý cleanup
    - Error recovery

    Ví dụ sử dụng:
        ```python
        lifecycle = SpawnedAgentLifecycle(
            session_id="abc123",
            session_key="agent:researcher:spawn:xyz",
            parent_session="agent:coordinator:main",
            config=spawn_config,
            spawn_depth=1
        )

        lifecycle.mark_running()
        # ... agent thực thi ...
        lifecycle.mark_completed("Kết quả thành công")

        # Hoặc nếu lỗi:
        lifecycle.mark_failed("Lỗi: Connection timeout")
        ```
    """

    def __init__(
        self,
        session_id: str,
        session_key: str,
        parent_session: Optional[str] = None,
        config: Optional["SpawnConfig"] = None,
        spawn_depth: int = 0,
    ) -> None:
        """
        Khởi tạo lifecycle tracker.

        Args:
            session_id: ID của session
            session_key: Key của session
            parent_session: Session key của parent
            config: Cấu hình spawn
            spawn_depth: Độ sâu spawn chain
        """
        self.session_id = session_id
        self.session_key = session_key
        self.parent_session = parent_session
        self.config = config
        self.spawn_depth = spawn_depth

        # State tracking
        self._state = SpawnLifecycleState.INITIALIZING
        self._events: List[LifecycleEvent] = []
        self._metrics = LifecycleMetrics()

        # Result tracking
        self.result: Optional[str] = None
        self.error: Optional[str] = None
        self.duration_ms: int = 0

        # Timestamps
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

        # Callbacks
        self._state_callbacks: List[Callable[[SpawnLifecycleState], Any]] = []

        # Log initialization event
        self._add_event(
            event_type="lifecycle_created",
            to_state=SpawnLifecycleState.INITIALIZING,
            message=f"Lifecycle created for spawn at depth {spawn_depth}",
        )

    @property
    def state(self) -> SpawnLifecycleState:
        """Trạng thái hiện tại."""
        return self._state

    @property
    def events(self) -> List[LifecycleEvent]:
        """Danh sách các events."""
        return list(self._events)

    @property
    def metrics(self) -> LifecycleMetrics:
        """Metrics hiện tại."""
        return self._metrics

    def is_running(self) -> bool:
        """Kiểm tra agent có đang chạy không."""
        return self._state == SpawnLifecycleState.RUNNING

    def is_completed(self) -> bool:
        """Kiểm tra agent đã hoàn thành chưa."""
        return self._state == SpawnLifecycleState.COMPLETED

    def is_failed(self) -> bool:
        """Kiểm tra agent có thất bại không."""
        return self._state in (
            SpawnLifecycleState.FAILED,
            SpawnLifecycleState.TIMEOUT,
            SpawnLifecycleState.CANCELLED,
        )

    def is_terminal(self) -> bool:
        """Kiểm tra agent đã ở trạng thái kết thúc chưa."""
        return self._state in (
            SpawnLifecycleState.COMPLETED,
            SpawnLifecycleState.FAILED,
            SpawnLifecycleState.TIMEOUT,
            SpawnLifecycleState.CANCELLED,
            SpawnLifecycleState.CLEANED_UP,
        )

    def mark_running(self) -> None:
        """
        Đánh dấu agent bắt đầu chạy.

        Ghi lại thời điểm bắt đầu và chuyển sang trạng thái RUNNING.
        """
        if self._state != SpawnLifecycleState.INITIALIZING:
            return

        old_state = self._state
        self._state = SpawnLifecycleState.RUNNING
        self.started_at = datetime.utcnow()
        self._metrics.start_time_ms = int(time.time() * 1000)

        self._add_event(
            event_type="started",
            from_state=old_state,
            to_state=SpawnLifecycleState.RUNNING,
            message="Agent started execution",
        )

        self._notify_state_change()

    def mark_completed(self, result: Optional[str] = None) -> None:
        """
        Đánh dấu agent hoàn thành thành công.

        Args:
            result: Kết quả của agent
        """
        if not self.is_running():
            return

        old_state = self._state
        self._state = SpawnLifecycleState.COMPLETED
        self.result = result
        self.completed_at = datetime.utcnow()
        self._finalize_metrics()

        self._add_event(
            event_type="completed",
            from_state=old_state,
            to_state=SpawnLifecycleState.COMPLETED,
            message=f"Agent completed successfully",
            metadata={"result_length": len(result) if result else 0},
        )

        self._notify_state_change()

    def mark_failed(self, error: str) -> None:
        """
        Đánh dấu agent thất bại.

        Args:
            error: Thông báo lỗi
        """
        if self.is_terminal():
            return

        old_state = self._state
        self._state = SpawnLifecycleState.FAILED
        self.error = error
        self.completed_at = datetime.utcnow()
        self._metrics.errors_count += 1
        self._finalize_metrics()

        self._add_event(
            event_type="failed",
            from_state=old_state,
            to_state=SpawnLifecycleState.FAILED,
            message=f"Agent failed: {error}",
            metadata={"error": error},
        )

        self._notify_state_change()

    def mark_timeout(self) -> None:
        """Đánh dấu agent timeout."""
        if self.is_terminal():
            return

        old_state = self._state
        self._state = SpawnLifecycleState.TIMEOUT
        self.error = "Execution timed out"
        self.completed_at = datetime.utcnow()
        self._finalize_metrics()

        self._add_event(
            event_type="timeout",
            from_state=old_state,
            to_state=SpawnLifecycleState.TIMEOUT,
            message="Agent execution timed out",
            metadata={
                "timeout_ms": self.config.timeout_ms if self.config else 0,
            },
        )

        self._notify_state_change()

    def mark_cancelled(self) -> None:
        """Đánh dấu agent bị hủy."""
        if self.is_terminal():
            return

        old_state = self._state
        self._state = SpawnLifecycleState.CANCELLED
        self.error = "Execution cancelled"
        self.completed_at = datetime.utcnow()
        self._finalize_metrics()

        self._add_event(
            event_type="cancelled",
            from_state=old_state,
            to_state=SpawnLifecycleState.CANCELLED,
            message="Agent execution was cancelled",
        )

        self._notify_state_change()

    def mark_cleaned_up(self) -> None:
        """Đánh dấu đã dọn dẹp xong."""
        if self._state == SpawnLifecycleState.CLEANED_UP:
            return

        old_state = self._state
        self._state = SpawnLifecycleState.CLEANED_UP

        self._add_event(
            event_type="cleaned_up",
            from_state=old_state,
            to_state=SpawnLifecycleState.CLEANED_UP,
            message="Agent resources cleaned up",
        )

        self._notify_state_change()

    def update_metrics(
        self,
        tokens_used: Optional[int] = None,
        turns_executed: Optional[int] = None,
    ) -> None:
        """
        Cập nhật metrics.

        Args:
            tokens_used: Số tokens đã sử dụng
            turns_executed: Số turns đã thực thi
        """
        if tokens_used is not None:
            self._metrics.tokens_used = tokens_used
        if turns_executed is not None:
            self._metrics.turns_executed = turns_executed

    def record_error(self, error: str) -> None:
        """
        Ghi lại một lỗi (không làm fail agent).

        Args:
            error: Thông báo lỗi
        """
        self._metrics.errors_count += 1
        self._add_event(
            event_type="error_recorded",
            message=f"Error: {error}",
            metadata={"error": error},
        )

    def record_retry(self) -> None:
        """Ghi lại một lần retry."""
        self._metrics.retries_count += 1
        self._add_event(
            event_type="retry",
            message=f"Retry attempt #{self._metrics.retries_count}",
        )

    def on_state_change(
        self,
        callback: Callable[[SpawnLifecycleState], Any]
    ) -> None:
        """
        Đăng ký callback cho state changes.

        Args:
            callback: Function được gọi với new state
        """
        self._state_callbacks.append(callback)

    def _add_event(
        self,
        event_type: str,
        from_state: Optional[SpawnLifecycleState] = None,
        to_state: Optional[SpawnLifecycleState] = None,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Thêm event vào history."""
        event = LifecycleEvent(
            event_type=event_type,
            timestamp=datetime.utcnow(),
            from_state=from_state,
            to_state=to_state,
            message=message,
            metadata=metadata or {},
        )
        self._events.append(event)

    def _finalize_metrics(self) -> None:
        """Hoàn tất metrics khi lifecycle kết thúc."""
        self._metrics.end_time_ms = int(time.time() * 1000)
        if self._metrics.start_time_ms > 0:
            self._metrics.duration_ms = (
                self._metrics.end_time_ms - self._metrics.start_time_ms
            )
        self.duration_ms = self._metrics.duration_ms

    def _notify_state_change(self) -> None:
        """Notify callbacks về state change."""
        for callback in self._state_callbacks:
            try:
                result = callback(self._state)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception:
                pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển lifecycle thành dictionary.

        Returns:
            Dictionary chứa thông tin lifecycle
        """
        return {
            "session_id": self.session_id,
            "session_key": self.session_key,
            "parent_session": self.parent_session,
            "spawn_depth": self.spawn_depth,
            "state": self._state.value,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metrics": self._metrics.to_dict(),
            "events": [e.to_dict() for e in self._events],
        }

    def __repr__(self) -> str:
        return (
            f"SpawnedAgentLifecycle(session_id={self.session_id!r}, "
            f"state={self._state.value!r})"
        )


class LifecycleManager:
    """
    Quản lý tập trung các lifecycles của spawned agents.

    LifecycleManager theo dõi và quản lý tất cả các lifecycles
    trong hệ thống, cung cấp các chức năng:
    - Tracking tất cả lifecycles
    - Cleanup tự động
    - Thống kê và reporting
    - Error recovery coordination

    Ví dụ sử dụng:
        ```python
        manager = LifecycleManager()

        # Register lifecycle
        manager.register(lifecycle)

        # Query lifecycles
        running = manager.get_by_state(SpawnLifecycleState.RUNNING)
        parent_lifecycles = manager.get_by_parent("agent:coordinator:main")

        # Cleanup
        cleaned = await manager.cleanup_terminal()
        ```
    """

    def __init__(self) -> None:
        """Khởi tạo LifecycleManager."""
        self._lifecycles: Dict[str, SpawnedAgentLifecycle] = {}
        self._by_parent: Dict[str, List[str]] = {}
        self._lock = asyncio.Lock()

    @property
    def count(self) -> int:
        """Số lượng lifecycles đang quản lý."""
        return len(self._lifecycles)

    def register(self, lifecycle: SpawnedAgentLifecycle) -> None:
        """
        Đăng ký lifecycle để quản lý.

        Args:
            lifecycle: Lifecycle cần đăng ký
        """
        self._lifecycles[lifecycle.session_id] = lifecycle

        # Index by parent
        parent = lifecycle.parent_session
        if parent:
            if parent not in self._by_parent:
                self._by_parent[parent] = []
            self._by_parent[parent].append(lifecycle.session_id)

    def unregister(self, session_id: str) -> Optional[SpawnedAgentLifecycle]:
        """
        Hủy đăng ký lifecycle.

        Args:
            session_id: ID của session

        Returns:
            Lifecycle đã hủy nếu tìm thấy
        """
        lifecycle = self._lifecycles.pop(session_id, None)
        if lifecycle:
            # Remove from parent index
            parent = lifecycle.parent_session
            if parent and parent in self._by_parent:
                if session_id in self._by_parent[parent]:
                    self._by_parent[parent].remove(session_id)

        return lifecycle

    def get(self, session_id: str) -> Optional[SpawnedAgentLifecycle]:
        """
        Lấy lifecycle theo session ID.

        Args:
            session_id: ID của session

        Returns:
            Lifecycle nếu tìm thấy
        """
        return self._lifecycles.get(session_id)

    def get_by_state(
        self,
        state: SpawnLifecycleState
    ) -> List[SpawnedAgentLifecycle]:
        """
        Lấy lifecycles theo trạng thái.

        Args:
            state: Trạng thái cần lọc

        Returns:
            Danh sách lifecycles
        """
        return [
            lc for lc in self._lifecycles.values()
            if lc.state == state
        ]

    def get_by_parent(
        self,
        parent_session: str
    ) -> List[SpawnedAgentLifecycle]:
        """
        Lấy lifecycles theo parent session.

        Args:
            parent_session: Session key của parent

        Returns:
            Danh sách lifecycles
        """
        session_ids = self._by_parent.get(parent_session, [])
        return [
            self._lifecycles[sid] for sid in session_ids
            if sid in self._lifecycles
        ]

    def get_running(self) -> List[SpawnedAgentLifecycle]:
        """Lấy tất cả lifecycles đang chạy."""
        return self.get_by_state(SpawnLifecycleState.RUNNING)

    def get_terminal(self) -> List[SpawnedAgentLifecycle]:
        """Lấy tất cả lifecycles đã kết thúc."""
        return [
            lc for lc in self._lifecycles.values()
            if lc.is_terminal()
        ]

    async def cleanup_terminal(
        self,
        max_age_seconds: int = 3600
    ) -> int:
        """
        Dọn dẹp các lifecycles đã kết thúc.

        Args:
            max_age_seconds: Tuổi tối đa trước khi dọn

        Returns:
            Số lượng đã dọn
        """
        async with self._lock:
            now = datetime.utcnow()
            to_remove: List[str] = []

            for session_id, lifecycle in self._lifecycles.items():
                if lifecycle.is_terminal():
                    age = (now - lifecycle.created_at).total_seconds()
                    if age > max_age_seconds:
                        to_remove.append(session_id)

            for session_id in to_remove:
                lifecycle = self._lifecycles.pop(session_id, None)
                if lifecycle:
                    lifecycle.mark_cleaned_up()

                    # Remove from parent index
                    parent = lifecycle.parent_session
                    if parent and parent in self._by_parent:
                        if session_id in self._by_parent[parent]:
                            self._by_parent[parent].remove(session_id)

            return len(to_remove)

    async def cancel_all_running(
        self,
        parent_session: Optional[str] = None
    ) -> int:
        """
        Hủy tất cả lifecycles đang chạy.

        Args:
            parent_session: Lọc theo parent (None để hủy tất cả)

        Returns:
            Số lượng đã hủy
        """
        async with self._lock:
            count = 0
            lifecycles = (
                self.get_by_parent(parent_session)
                if parent_session
                else list(self._lifecycles.values())
            )

            for lifecycle in lifecycles:
                if lifecycle.is_running():
                    lifecycle.mark_cancelled()
                    count += 1

            return count

    def get_statistics(self) -> Dict[str, Any]:
        """
        Lấy thống kê về lifecycles.

        Returns:
            Dictionary với các thống kê
        """
        state_counts: Dict[str, int] = {}
        total_tokens = 0
        total_duration = 0

        for lifecycle in self._lifecycles.values():
            state = lifecycle.state.value
            state_counts[state] = state_counts.get(state, 0) + 1
            total_tokens += lifecycle.metrics.tokens_used
            total_duration += lifecycle.metrics.duration_ms

        return {
            "total_lifecycles": len(self._lifecycles),
            "state_counts": state_counts,
            "parent_count": len(self._by_parent),
            "total_tokens_used": total_tokens,
            "total_duration_ms": total_duration,
        }

    def __repr__(self) -> str:
        return f"LifecycleManager(count={self.count})"
