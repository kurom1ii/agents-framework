"""
Session Management Base Models và Protocols.

Module này định nghĩa các models cơ bản cho hệ thống quản lý Session,
bao gồm SessionScope, SessionConfig và Session.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Literal, Optional
import uuid


class SessionScope(Enum):
    """
    Phạm vi session xác định cách session được tạo và chia sẻ.

    - MAIN: Session chính duy nhất cho agent
    - PER_PEER: Session riêng cho mỗi người dùng/peer
    - PER_CONTEXT: Session riêng cho mỗi context (channel, group, thread)
    """
    MAIN = "main"
    PER_PEER = "per-peer"
    PER_CONTEXT = "per-context"


class SessionState(Enum):
    """
    Trạng thái của session.

    - ACTIVE: Session đang hoạt động
    - IDLE: Session không hoạt động nhưng chưa hết hạn
    - EXPIRED: Session đã hết hạn, cần reset
    - ARCHIVED: Session đã được lưu trữ
    """
    ACTIVE = "active"
    IDLE = "idle"
    EXPIRED = "expired"
    ARCHIVED = "archived"


@dataclass
class SessionConfig:
    """
    Cấu hình cho Session Manager.

    Attributes:
        scope: Phạm vi session (main, per-peer, per-context)
        reset_mode: Chế độ reset session
            - "daily": Reset theo giờ cố định mỗi ngày
            - "idle": Reset sau thời gian idle
            - "combined": Kết hợp cả daily và idle
        reset_hour: Giờ reset hàng ngày (0-23), mặc định 4h sáng
        idle_minutes: Số phút idle trước khi reset, mặc định 120 phút
        max_context_tokens: Số token tối đa trong context, mặc định 100000
        auto_archive: Tự động lưu trữ session khi reset
        preserve_metadata: Giữ lại metadata khi reset session
    """
    scope: SessionScope = SessionScope.MAIN
    reset_mode: Literal["daily", "idle", "combined"] = "daily"
    reset_hour: int = 4
    idle_minutes: int = 120
    max_context_tokens: int = 100000
    auto_archive: bool = True
    preserve_metadata: bool = True

    def __post_init__(self) -> None:
        """Validate cấu hình sau khi khởi tạo."""
        if not 0 <= self.reset_hour <= 23:
            raise ValueError(f"reset_hour phải trong khoảng 0-23, nhận được: {self.reset_hour}")
        if self.idle_minutes <= 0:
            raise ValueError(f"idle_minutes phải > 0, nhận được: {self.idle_minutes}")
        if self.max_context_tokens <= 0:
            raise ValueError(f"max_context_tokens phải > 0, nhận được: {self.max_context_tokens}")


@dataclass
class Session:
    """
    Đại diện cho một session trong hệ thống.

    Session là đơn vị quản lý ngữ cảnh hội thoại giữa agent và người dùng.
    Mỗi session có một ID duy nhất và một key để định danh.

    Attributes:
        session_id: ID duy nhất của session (UUID)
        session_key: Key để tra cứu session (format: agent:<agentId>:<scope>:<identifier>)
        agent_id: ID của agent sở hữu session
        created_at: Thời điểm tạo session
        updated_at: Thời điểm cập nhật cuối cùng
        input_tokens: Tổng số token đầu vào đã sử dụng
        output_tokens: Tổng số token đầu ra đã sử dụng
        context_tokens: Số token hiện tại trong context
        state: Trạng thái của session
        metadata: Dữ liệu bổ sung tùy chỉnh
    """
    session_id: str
    session_key: str
    agent_id: str
    created_at: datetime
    updated_at: datetime
    input_tokens: int = 0
    output_tokens: int = 0
    context_tokens: int = 0
    state: SessionState = SessionState.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        session_key: str,
        agent_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "Session":
        """
        Tạo session mới với ID tự động.

        Args:
            session_key: Key định danh session
            agent_id: ID của agent
            metadata: Metadata tùy chọn

        Returns:
            Session mới được tạo
        """
        now = datetime.utcnow()
        return cls(
            session_id=str(uuid.uuid4()),
            session_key=session_key,
            agent_id=agent_id,
            created_at=now,
            updated_at=now,
            metadata=metadata or {}
        )

    def update_tokens(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        context_tokens: Optional[int] = None
    ) -> None:
        """
        Cập nhật số lượng token đã sử dụng.

        Args:
            input_tokens: Số token đầu vào mới
            output_tokens: Số token đầu ra mới
            context_tokens: Số token context mới (None để giữ nguyên)
        """
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if context_tokens is not None:
            self.context_tokens = context_tokens
        self.updated_at = datetime.utcnow()

    def touch(self) -> None:
        """Cập nhật thời gian hoạt động cuối cùng."""
        self.updated_at = datetime.utcnow()
        if self.state == SessionState.IDLE:
            self.state = SessionState.ACTIVE

    def mark_idle(self) -> None:
        """Đánh dấu session là idle."""
        self.state = SessionState.IDLE

    def mark_expired(self) -> None:
        """Đánh dấu session đã hết hạn."""
        self.state = SessionState.EXPIRED

    def is_active(self) -> bool:
        """Kiểm tra session có đang hoạt động không."""
        return self.state == SessionState.ACTIVE

    def is_expired(self) -> bool:
        """Kiểm tra session đã hết hạn chưa."""
        return self.state == SessionState.EXPIRED

    @property
    def total_tokens(self) -> int:
        """Tổng số token đã sử dụng (input + output)."""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển session thành dictionary để lưu trữ.

        Returns:
            Dictionary chứa thông tin session
        """
        return {
            "session_id": self.session_id,
            "session_key": self.session_key,
            "agent_id": self.agent_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "context_tokens": self.context_tokens,
            "state": self.state.value,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """
        Tạo session từ dictionary.

        Args:
            data: Dictionary chứa thông tin session

        Returns:
            Session được khôi phục
        """
        return cls(
            session_id=data["session_id"],
            session_key=data["session_key"],
            agent_id=data["agent_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            context_tokens=data.get("context_tokens", 0),
            state=SessionState(data.get("state", "active")),
            metadata=data.get("metadata", {})
        )
