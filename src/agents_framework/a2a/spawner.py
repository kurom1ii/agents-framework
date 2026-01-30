"""
Sub-Agent Spawner - Hệ thống spawn sub-agents động.

Module này cung cấp SubAgentSpawner class để parent agents có thể
tạo sub-agents động cho các tác vụ cụ thể với isolated sessions.

Các tính năng chính:
- Spawn sub-agents với cấu hình tùy chỉnh
- Resource limits (tokens, time, tools)
- Spawn chain limits (max depth, max concurrent)
- Auto-cleanup khi hoàn thành
- Report back to parent
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Set,
)

from .lifecycle import SpawnedAgentLifecycle, SpawnLifecycleState

if TYPE_CHECKING:
    from agents_framework.agents import BaseAgent, AgentSpawner as AgentFactory
    from agents_framework.sessions import SessionManager


class SpawnStatus(str, Enum):
    """
    Trạng thái của một spawned sub-agent.

    - PENDING: Đang chờ spawn
    - RUNNING: Đang thực thi tác vụ
    - COMPLETED: Hoàn thành thành công
    - FAILED: Thực thi thất bại
    - TIMEOUT: Timeout khi thực thi
    - CANCELLED: Bị hủy bỏ
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class SpawnConfig:
    """
    Cấu hình cho việc spawn một sub-agent.

    Định nghĩa các tham số và giới hạn cho sub-agent được tạo,
    bao gồm model, tools được phép, và resource limits.

    Attributes:
        agent_id: ID duy nhất cho sub-agent
        purpose: Mục đích/mô tả của sub-agent
        model: Model LLM sử dụng (None để dùng model mặc định)
        tools: Danh sách tools được phép (None để dùng tất cả)
        max_turns: Số turns tối đa cho sub-agent
        timeout_ms: Timeout thực thi (milliseconds)
        max_tokens: Token budget cho sub-agent
        isolated: Tạo session riêng biệt cho sub-agent
        report_back: Gửi kết quả về parent agent

    Example:
        config = SpawnConfig(
            agent_id="researcher-1",
            purpose="Nghiên cứu về AI frameworks",
            tools=["web_search", "web_fetch"],
            max_turns=10,
            timeout_ms=300000,
            max_tokens=50000
        )
    """
    agent_id: str
    purpose: str
    model: Optional[str] = None
    tools: Optional[List[str]] = None
    max_turns: int = 10
    timeout_ms: int = 300000  # 5 minutes default
    max_tokens: int = 50000
    isolated: bool = True
    report_back: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate cấu hình sau khi khởi tạo."""
        if self.max_turns <= 0:
            raise ValueError(f"max_turns phải > 0, nhận được: {self.max_turns}")
        if self.timeout_ms <= 0:
            raise ValueError(f"timeout_ms phải > 0, nhận được: {self.timeout_ms}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens phải > 0, nhận được: {self.max_tokens}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpawnConfig":
        """
        Tạo SpawnConfig từ dictionary.

        Args:
            data: Dictionary chứa cấu hình

        Returns:
            SpawnConfig instance
        """
        return cls(
            agent_id=data.get("id", data.get("agent_id", str(uuid.uuid4())[:8])),
            purpose=data.get("purpose", "Sub-agent task"),
            model=data.get("model"),
            tools=data.get("tools"),
            max_turns=data.get("max_turns", 10),
            timeout_ms=data.get("timeout_ms", data.get("timeout", 300000)),
            max_tokens=data.get("max_tokens", 50000),
            isolated=data.get("isolated", True),
            report_back=data.get("report_back", True),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển SpawnConfig thành dictionary.

        Returns:
            Dictionary chứa cấu hình
        """
        return {
            "agent_id": self.agent_id,
            "purpose": self.purpose,
            "model": self.model,
            "tools": self.tools,
            "max_turns": self.max_turns,
            "timeout_ms": self.timeout_ms,
            "max_tokens": self.max_tokens,
            "isolated": self.isolated,
            "report_back": self.report_back,
            "metadata": self.metadata,
        }


@dataclass
class SpawnResult:
    """
    Kết quả của việc spawn và thực thi sub-agent.

    Chứa thông tin về session, trạng thái thực thi, kết quả
    và các metrics liên quan.

    Attributes:
        session_id: ID của session được tạo
        session_key: Key để truy cập session
        status: Trạng thái thực thi cuối cùng
        result: Kết quả trả về từ sub-agent (nếu thành công)
        tokens_used: Tổng số tokens đã sử dụng
        duration_ms: Thời gian thực thi (milliseconds)
        error: Thông báo lỗi (nếu có)
        turns_used: Số turns đã sử dụng
        spawn_depth: Độ sâu spawn chain hiện tại
        parent_session: Session key của parent (nếu có)
        metadata: Dữ liệu bổ sung

    Example:
        result = SpawnResult(
            session_id="abc123",
            session_key="agent:researcher-1:spawn:xyz",
            status=SpawnStatus.COMPLETED,
            result="Kết quả nghiên cứu...",
            tokens_used=15000,
            duration_ms=45000
        )
    """
    session_id: str
    session_key: str
    status: SpawnStatus
    result: Optional[str] = None
    tokens_used: int = 0
    duration_ms: int = 0
    error: Optional[str] = None
    turns_used: int = 0
    spawn_depth: int = 0
    parent_session: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_success(self) -> bool:
        """Kiểm tra kết quả có thành công không."""
        return self.status == SpawnStatus.COMPLETED

    def is_failed(self) -> bool:
        """Kiểm tra kết quả có thất bại không."""
        return self.status in (SpawnStatus.FAILED, SpawnStatus.TIMEOUT, SpawnStatus.CANCELLED)

    def is_running(self) -> bool:
        """Kiểm tra sub-agent có đang chạy không."""
        return self.status == SpawnStatus.RUNNING

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển SpawnResult thành dictionary.

        Returns:
            Dictionary chứa kết quả
        """
        return {
            "session_id": self.session_id,
            "session_key": self.session_key,
            "status": self.status.value,
            "result": self.result,
            "tokens_used": self.tokens_used,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "turns_used": self.turns_used,
            "spawn_depth": self.spawn_depth,
            "parent_session": self.parent_session,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpawnResult":
        """
        Tạo SpawnResult từ dictionary.

        Args:
            data: Dictionary chứa kết quả

        Returns:
            SpawnResult instance
        """
        return cls(
            session_id=data["session_id"],
            session_key=data["session_key"],
            status=SpawnStatus(data["status"]),
            result=data.get("result"),
            tokens_used=data.get("tokens_used", 0),
            duration_ms=data.get("duration_ms", 0),
            error=data.get("error"),
            turns_used=data.get("turns_used", 0),
            spawn_depth=data.get("spawn_depth", 0),
            parent_session=data.get("parent_session"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SpawnLimits:
    """
    Giới hạn cho hệ thống spawn.

    Định nghĩa các giới hạn để ngăn infinite spawning
    và kiểm soát resource usage.

    Attributes:
        max_spawn_depth: Độ sâu tối đa của spawn chain
        max_concurrent_per_parent: Số sub-agents đồng thời tối đa per parent
        max_global_spawns: Tổng số spawns tối đa trong hệ thống
        default_timeout_ms: Timeout mặc định cho spawns
        default_max_tokens: Token budget mặc định
    """
    max_spawn_depth: int = 2
    max_concurrent_per_parent: int = 5
    max_global_spawns: int = 50
    default_timeout_ms: int = 300000
    default_max_tokens: int = 50000


# Context variable để theo dõi spawn depth trong execution chain
_spawn_depth: ContextVar[int] = ContextVar("spawn_depth", default=0)
_parent_session: ContextVar[Optional[str]] = ContextVar("parent_session", default=None)

# Context variable để lưu spawner instance
_current_spawner: ContextVar[Optional["SubAgentSpawner"]] = ContextVar(
    "current_spawner", default=None
)


def get_spawn_depth() -> int:
    """
    Lấy spawn depth hiện tại từ context.

    Returns:
        Spawn depth hiện tại (0 nếu là root)
    """
    return _spawn_depth.get()


def get_parent_session() -> Optional[str]:
    """
    Lấy parent session key từ context.

    Returns:
        Parent session key hoặc None nếu là root
    """
    return _parent_session.get()


def set_spawn_context(depth: int, parent: Optional[str]) -> None:
    """
    Đặt spawn context cho execution hiện tại.

    Args:
        depth: Spawn depth hiện tại
        parent: Parent session key
    """
    _spawn_depth.set(depth)
    _parent_session.set(parent)


def get_current_spawner() -> "SubAgentSpawner":
    """
    Lấy SubAgentSpawner instance từ context hiện tại.

    Returns:
        SubAgentSpawner instance

    Raises:
        RuntimeError: Nếu chưa có spawner trong context
    """
    spawner = _current_spawner.get()
    if spawner is None:
        raise RuntimeError(
            "SubAgentSpawner chưa được khởi tạo trong context. "
            "Hãy đảm bảo set_current_spawner() đã được gọi."
        )
    return spawner


def set_current_spawner(spawner: "SubAgentSpawner") -> None:
    """
    Đặt SubAgentSpawner instance cho context hiện tại.

    Args:
        spawner: SubAgentSpawner instance
    """
    _current_spawner.set(spawner)


class SubAgentSpawner:
    """
    Hệ thống spawn sub-agents động.

    SubAgentSpawner cho phép parent agents tạo sub-agents
    để xử lý các tác vụ con với isolated sessions và resource limits.

    Tính năng chính:
    - Spawn sub-agents với cấu hình tùy chỉnh
    - Quản lý lifecycle của spawned agents
    - Resource limits (tokens, time, tools)
    - Spawn chain limits (max depth, max concurrent)
    - Auto-cleanup khi hoàn thành
    - Progress tracking và cancellation

    Ví dụ sử dụng:
        ```python
        spawner = SubAgentSpawner(
            agent_factory=agent_factory,
            session_manager=session_manager,
            limits=SpawnLimits(max_spawn_depth=2, max_concurrent_per_parent=5)
        )

        config = SpawnConfig(
            agent_id="researcher",
            purpose="Nghiên cứu AI frameworks",
            tools=["web_search", "web_fetch"],
            max_turns=10,
            timeout_ms=300000
        )

        result = await spawner.spawn(
            parent_session="agent:coordinator:main",
            config=config,
            task="Tìm 5 AI frameworks phổ biến nhất"
        )

        if result.is_success():
            print(f"Kết quả: {result.result}")
        ```
    """

    def __init__(
        self,
        agent_factory: Optional["AgentFactory"] = None,
        session_manager: Optional["SessionManager"] = None,
        limits: Optional[SpawnLimits] = None,
        on_spawn_complete: Optional[Callable[[SpawnResult], Any]] = None,
    ) -> None:
        """
        Khởi tạo SubAgentSpawner.

        Args:
            agent_factory: Factory để tạo agents (AgentSpawner)
            session_manager: Manager để quản lý sessions
            limits: Giới hạn spawn (mặc định nếu None)
            on_spawn_complete: Callback khi spawn hoàn thành
        """
        self._agent_factory = agent_factory
        self._session_manager = session_manager
        self._limits = limits or SpawnLimits()
        self._on_spawn_complete = on_spawn_complete

        # Tracking structures
        self._active_spawns: Dict[str, SpawnResult] = {}
        self._parent_spawns: Dict[str, Set[str]] = {}  # parent_session -> set of spawn session_ids
        self._lifecycles: Dict[str, SpawnedAgentLifecycle] = {}
        self._spawn_tasks: Dict[str, asyncio.Task] = {}

        # Lock cho thread safety
        self._lock = asyncio.Lock()

        # Callbacks cho lifecycle events
        self._lifecycle_callbacks: List[Callable[[str, SpawnLifecycleState], Any]] = []

    @property
    def limits(self) -> SpawnLimits:
        """Trả về giới hạn spawn hiện tại."""
        return self._limits

    @property
    def active_spawn_count(self) -> int:
        """Số lượng spawns đang active."""
        return len(self._active_spawns)

    async def spawn(
        self,
        parent_session: str,
        config: SpawnConfig,
        task: str,
    ) -> SpawnResult:
        """
        Spawn một sub-agent để thực thi tác vụ.

        Tạo sub-agent với cấu hình đã cho và thực thi tác vụ.
        Sub-agent sẽ có isolated session và bị giới hạn bởi
        resource limits đã định nghĩa.

        Args:
            parent_session: Session key của parent agent
            config: Cấu hình cho sub-agent
            task: Tác vụ cần thực thi

        Returns:
            SpawnResult với kết quả thực thi

        Raises:
            ValueError: Nếu vượt quá giới hạn spawn
            RuntimeError: Nếu không thể spawn sub-agent

        Example:
            result = await spawner.spawn(
                parent_session="agent:coordinator:main",
                config=SpawnConfig(
                    agent_id="researcher",
                    purpose="Research task",
                    max_turns=10
                ),
                task="Research AI frameworks"
            )
        """
        async with self._lock:
            # Validate spawn limits
            await self._validate_spawn_limits(parent_session)

            # Lấy spawn depth hiện tại
            current_depth = get_spawn_depth()

            # Tạo session key cho sub-agent
            spawn_id = str(uuid.uuid4())[:8]
            session_key = f"agent:{config.agent_id}:spawn:{spawn_id}"
            session_id = str(uuid.uuid4())

            # Tạo SpawnResult ban đầu
            result = SpawnResult(
                session_id=session_id,
                session_key=session_key,
                status=SpawnStatus.PENDING,
                spawn_depth=current_depth + 1,
                parent_session=parent_session,
                metadata={
                    "config": config.to_dict(),
                    "task": task,
                    "created_at": datetime.utcnow().isoformat(),
                }
            )

            # Track spawn
            self._active_spawns[session_id] = result
            if parent_session not in self._parent_spawns:
                self._parent_spawns[parent_session] = set()
            self._parent_spawns[parent_session].add(session_id)

            # Tạo lifecycle tracker
            lifecycle = SpawnedAgentLifecycle(
                session_id=session_id,
                session_key=session_key,
                parent_session=parent_session,
                config=config,
                spawn_depth=current_depth + 1,
            )
            self._lifecycles[session_id] = lifecycle

        # Thực thi spawn (không giữ lock)
        try:
            result = await self._execute_spawn(
                result=result,
                config=config,
                task=task,
                lifecycle=lifecycle,
            )
        except Exception as e:
            result.status = SpawnStatus.FAILED
            result.error = str(e)
            lifecycle.mark_failed(str(e))

        # Cleanup và callback
        await self._handle_spawn_complete(result)

        return result

    async def _validate_spawn_limits(self, parent_session: str) -> None:
        """
        Validate các giới hạn spawn.

        Args:
            parent_session: Session key của parent

        Raises:
            ValueError: Nếu vượt quá giới hạn
        """
        # Check spawn depth
        current_depth = get_spawn_depth()
        if current_depth >= self._limits.max_spawn_depth:
            raise ValueError(
                f"Đã đạt giới hạn spawn depth tối đa ({self._limits.max_spawn_depth}). "
                f"Depth hiện tại: {current_depth}"
            )

        # Check concurrent spawns per parent
        parent_spawns = self._parent_spawns.get(parent_session, set())
        active_count = sum(
            1 for sid in parent_spawns
            if sid in self._active_spawns
            and self._active_spawns[sid].status == SpawnStatus.RUNNING
        )
        if active_count >= self._limits.max_concurrent_per_parent:
            raise ValueError(
                f"Đã đạt giới hạn concurrent spawns per parent "
                f"({self._limits.max_concurrent_per_parent})"
            )

        # Check global spawn limit
        global_active = sum(
            1 for r in self._active_spawns.values()
            if r.status == SpawnStatus.RUNNING
        )
        if global_active >= self._limits.max_global_spawns:
            raise ValueError(
                f"Đã đạt giới hạn global spawns tối đa "
                f"({self._limits.max_global_spawns})"
            )

    async def _execute_spawn(
        self,
        result: SpawnResult,
        config: SpawnConfig,
        task: str,
        lifecycle: SpawnedAgentLifecycle,
    ) -> SpawnResult:
        """
        Thực thi spawn và chạy tác vụ.

        Args:
            result: SpawnResult để cập nhật
            config: Cấu hình spawn
            task: Tác vụ cần thực thi
            lifecycle: Lifecycle tracker

        Returns:
            SpawnResult đã cập nhật
        """
        start_time = time.time()
        result.status = SpawnStatus.RUNNING
        lifecycle.mark_running()

        # Notify lifecycle callbacks
        await self._notify_lifecycle(result.session_id, SpawnLifecycleState.RUNNING)

        try:
            # Set spawn context cho execution
            set_spawn_context(result.spawn_depth, result.parent_session)

            # Timeout wrapper
            timeout_seconds = config.timeout_ms / 1000

            # Simulate sub-agent execution (trong thực tế sẽ gọi agent factory)
            # Đây là placeholder - implementation thực sẽ tạo agent và run task
            task_result = await asyncio.wait_for(
                self._run_sub_agent_task(config, task, lifecycle),
                timeout=timeout_seconds,
            )

            # Cập nhật result
            result.status = SpawnStatus.COMPLETED
            result.result = task_result.get("output")
            result.tokens_used = task_result.get("tokens_used", 0)
            result.turns_used = task_result.get("turns_used", 0)
            lifecycle.mark_completed(result.result)

        except asyncio.TimeoutError:
            result.status = SpawnStatus.TIMEOUT
            result.error = f"Timeout sau {config.timeout_ms}ms"
            lifecycle.mark_timeout()

        except asyncio.CancelledError:
            result.status = SpawnStatus.CANCELLED
            result.error = "Spawn bị hủy"
            lifecycle.mark_cancelled()
            raise

        except Exception as e:
            result.status = SpawnStatus.FAILED
            result.error = str(e)
            lifecycle.mark_failed(str(e))

        finally:
            # Tính duration
            result.duration_ms = int((time.time() - start_time) * 1000)
            lifecycle.duration_ms = result.duration_ms

        return result

    async def _run_sub_agent_task(
        self,
        config: SpawnConfig,
        task: str,
        lifecycle: SpawnedAgentLifecycle,
    ) -> Dict[str, Any]:
        """
        Chạy tác vụ với sub-agent.

        Đây là method được override trong implementation thực tế
        để tích hợp với agent factory và execution loop.

        Args:
            config: Cấu hình spawn
            task: Tác vụ cần thực thi
            lifecycle: Lifecycle tracker

        Returns:
            Dictionary với output và metrics
        """
        # Placeholder implementation
        # Trong thực tế sẽ:
        # 1. Tạo agent từ agent_factory với config
        # 2. Tạo session từ session_manager
        # 3. Run agent với task
        # 4. Collect metrics và return

        if self._agent_factory is None or self._session_manager is None:
            # Mock result khi không có factory/manager
            await asyncio.sleep(0.1)  # Simulate some work
            return {
                "output": f"[Simulated] Task completed: {task[:100]}...",
                "tokens_used": 1000,
                "turns_used": 1,
            }

        # Real implementation với agent factory
        from agents_framework.agents import Task as AgentTask

        # Tạo session cho sub-agent
        session = await self._session_manager.get_or_create(
            key=lifecycle.session_key,
            agent_id=config.agent_id,
            metadata={
                "spawn_depth": lifecycle.spawn_depth,
                "parent_session": lifecycle.parent_session,
                "purpose": config.purpose,
            }
        )

        # Spawn agent
        agent = await self._agent_factory.spawn(
            template_name=config.agent_id,
            config_overrides={
                "max_iterations": config.max_turns,
                "timeout": config.timeout_ms / 1000,
                "tools": config.tools,
            },
            metadata={
                "session_key": lifecycle.session_key,
                "spawn_depth": lifecycle.spawn_depth,
            }
        )

        # Run task
        agent_task = AgentTask(
            description=task,
            context={
                "purpose": config.purpose,
                "max_tokens": config.max_tokens,
            }
        )

        task_result = await agent.run(agent_task)

        # Collect metrics từ session
        session = await self._session_manager.get(lifecycle.session_key)
        tokens_used = session.total_tokens if session else 0

        # Release agent
        await self._agent_factory.release(agent.id)

        return {
            "output": task_result.output if task_result.success else task_result.error,
            "tokens_used": tokens_used,
            "turns_used": config.max_turns,  # Simplified
        }

    async def _handle_spawn_complete(self, result: SpawnResult) -> None:
        """
        Xử lý sau khi spawn hoàn thành.

        Args:
            result: SpawnResult của spawn đã hoàn thành
        """
        async with self._lock:
            # Cập nhật lifecycle state
            if result.session_id in self._lifecycles:
                lifecycle = self._lifecycles[result.session_id]

                if result.status == SpawnStatus.COMPLETED:
                    await self._notify_lifecycle(
                        result.session_id, SpawnLifecycleState.COMPLETED
                    )
                elif result.status == SpawnStatus.FAILED:
                    await self._notify_lifecycle(
                        result.session_id, SpawnLifecycleState.FAILED
                    )
                elif result.status == SpawnStatus.TIMEOUT:
                    await self._notify_lifecycle(
                        result.session_id, SpawnLifecycleState.TIMEOUT
                    )
                elif result.status == SpawnStatus.CANCELLED:
                    await self._notify_lifecycle(
                        result.session_id, SpawnLifecycleState.CANCELLED
                    )

        # Gọi callback nếu có
        if self._on_spawn_complete:
            try:
                callback_result = self._on_spawn_complete(result)
                if asyncio.iscoroutine(callback_result):
                    await callback_result
            except Exception:
                pass  # Ignore callback errors

    async def get_spawn_status(self, session_id: str) -> Optional[SpawnResult]:
        """
        Lấy trạng thái của một spawn.

        Args:
            session_id: ID của session spawn

        Returns:
            SpawnResult nếu tìm thấy, None nếu không
        """
        return self._active_spawns.get(session_id)

    async def cancel_spawn(self, session_id: str) -> bool:
        """
        Hủy một spawn đang chạy.

        Args:
            session_id: ID của session spawn cần hủy

        Returns:
            True nếu hủy thành công
        """
        async with self._lock:
            result = self._active_spawns.get(session_id)
            if not result or result.status != SpawnStatus.RUNNING:
                return False

            # Cancel task nếu có
            task = self._spawn_tasks.get(session_id)
            if task and not task.done():
                task.cancel()

            # Update status
            result.status = SpawnStatus.CANCELLED
            result.error = "Spawn bị hủy bởi parent"

            # Update lifecycle
            if session_id in self._lifecycles:
                self._lifecycles[session_id].mark_cancelled()

            return True

    async def list_spawned(
        self,
        parent_session: Optional[str] = None,
        status: Optional[SpawnStatus] = None,
    ) -> List[SpawnResult]:
        """
        Liệt kê các spawns.

        Args:
            parent_session: Lọc theo parent session (None để lấy tất cả)
            status: Lọc theo status (None để lấy tất cả)

        Returns:
            Danh sách SpawnResults
        """
        results: List[SpawnResult] = []

        for session_id, result in self._active_spawns.items():
            # Filter by parent
            if parent_session and result.parent_session != parent_session:
                continue

            # Filter by status
            if status and result.status != status:
                continue

            results.append(result)

        return results

    async def cleanup_completed(self, max_age_seconds: int = 3600) -> int:
        """
        Dọn dẹp các spawns đã hoàn thành.

        Args:
            max_age_seconds: Tuổi tối đa của spawn trước khi dọn (giây)

        Returns:
            Số lượng spawns đã dọn
        """
        async with self._lock:
            now = datetime.utcnow()
            to_remove: List[str] = []

            for session_id, result in self._active_spawns.items():
                if result.status in (
                    SpawnStatus.COMPLETED,
                    SpawnStatus.FAILED,
                    SpawnStatus.TIMEOUT,
                    SpawnStatus.CANCELLED,
                ):
                    # Check age
                    created_at = result.metadata.get("created_at")
                    if created_at:
                        created = datetime.fromisoformat(created_at)
                        age = (now - created).total_seconds()
                        if age > max_age_seconds:
                            to_remove.append(session_id)

            # Remove old spawns
            for session_id in to_remove:
                del self._active_spawns[session_id]
                self._lifecycles.pop(session_id, None)

                # Remove from parent tracking
                for parent_spawns in self._parent_spawns.values():
                    parent_spawns.discard(session_id)

            return len(to_remove)

    def on_lifecycle_change(
        self,
        callback: Callable[[str, SpawnLifecycleState], Any]
    ) -> None:
        """
        Đăng ký callback cho lifecycle changes.

        Args:
            callback: Function được gọi với (session_id, new_state)
        """
        self._lifecycle_callbacks.append(callback)

    async def _notify_lifecycle(
        self,
        session_id: str,
        state: SpawnLifecycleState
    ) -> None:
        """Notify callbacks về lifecycle change."""
        for callback in self._lifecycle_callbacks:
            try:
                result = callback(session_id, state)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass

    def get_statistics(self) -> Dict[str, Any]:
        """
        Lấy thống kê về spawner.

        Returns:
            Dictionary với các thống kê
        """
        status_counts: Dict[str, int] = {}
        for result in self._active_spawns.values():
            status = result.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total_spawns": len(self._active_spawns),
            "status_counts": status_counts,
            "parent_count": len(self._parent_spawns),
            "limits": {
                "max_spawn_depth": self._limits.max_spawn_depth,
                "max_concurrent_per_parent": self._limits.max_concurrent_per_parent,
                "max_global_spawns": self._limits.max_global_spawns,
            }
        }

    def __repr__(self) -> str:
        return (
            f"SubAgentSpawner(active={self.active_spawn_count}, "
            f"limits={self._limits})"
        )
