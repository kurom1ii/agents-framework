"""
A2A (Agent-to-Agent) Base Classes và Protocols.

Module này định nghĩa các models cơ bản cho hệ thống giao tiếp Agent-to-Agent,
bao gồm SessionInfo, AgentInfo và các enums trạng thái.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class SessionStatus(Enum):
    """
    Trạng thái của một session trong hệ thống A2A.

    - IDLE: Session đang rảnh, không xử lý tác vụ
    - BUSY: Session đang bận xử lý tác vụ
    - ERROR: Session gặp lỗi
    """

    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"


class AgentStatus(Enum):
    """
    Trạng thái của một agent trong hệ thống A2A.

    - AVAILABLE: Agent sẵn sàng nhận tác vụ mới
    - BUSY: Agent đang bận xử lý tác vụ
    - OFFLINE: Agent không hoạt động
    """

    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"


@dataclass
class SessionInfo:
    """
    Thông tin về một session trong hệ thống.

    SessionInfo chứa các thông tin cần thiết để agents có thể khám phá
    và giao tiếp với các sessions khác trong hệ thống.

    Attributes:
        session_key: Key duy nhất để định danh session (format: agent:<agentId>:<scope>:<identifier>)
        agent_id: ID của agent sở hữu session
        status: Trạng thái hiện tại của session (idle, busy, error)
        last_activity: Thời điểm hoạt động cuối cùng
        context_tokens: Số tokens hiện tại trong context của session
        display_name: Tên hiển thị thân thiện của session (tùy chọn)
        metadata: Dữ liệu bổ sung tùy chỉnh

    Example:
        session = SessionInfo(
            session_key="agent:coder:main",
            agent_id="coder",
            status=SessionStatus.IDLE,
            last_activity=datetime.now(),
            context_tokens=4500,
            display_name="Coder Assistant"
        )
    """

    session_key: str
    agent_id: str
    status: SessionStatus
    last_activity: datetime
    context_tokens: int
    display_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_idle(self) -> bool:
        """Kiểm tra session có đang rảnh không."""
        return self.status == SessionStatus.IDLE

    def is_busy(self) -> bool:
        """Kiểm tra session có đang bận không."""
        return self.status == SessionStatus.BUSY

    def is_error(self) -> bool:
        """Kiểm tra session có đang lỗi không."""
        return self.status == SessionStatus.ERROR

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển SessionInfo thành dictionary.

        Returns:
            Dictionary chứa thông tin session
        """
        return {
            "session_key": self.session_key,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "last_activity": self.last_activity.isoformat(),
            "context_tokens": self.context_tokens,
            "display_name": self.display_name,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionInfo":
        """
        Tạo SessionInfo từ dictionary.

        Args:
            data: Dictionary chứa thông tin session

        Returns:
            SessionInfo được khôi phục
        """
        return cls(
            session_key=data["session_key"],
            agent_id=data["agent_id"],
            status=SessionStatus(data["status"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            context_tokens=data["context_tokens"],
            display_name=data.get("display_name"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AgentInfo:
    """
    Thông tin về một agent trong hệ thống.

    AgentInfo chứa các thông tin cần thiết để khám phá và tương tác
    với các agents khác trong hệ thống A2A.

    Attributes:
        agent_id: ID duy nhất của agent
        role: Vai trò của agent (ví dụ: "coder", "researcher", "writer")
        capabilities: Danh sách các khả năng của agent
        status: Trạng thái hiện tại của agent
        current_session: Session key hiện tại của agent (nếu có)
        description: Mô tả về agent
        metadata: Dữ liệu bổ sung tùy chỉnh

    Example:
        agent = AgentInfo(
            agent_id="coder",
            role="coder",
            capabilities=["code_review", "refactoring", "testing"],
            status=AgentStatus.AVAILABLE,
            description="Agent chuyên về lập trình và review code"
        )
    """

    agent_id: str
    role: str
    capabilities: List[str] = field(default_factory=list)
    status: AgentStatus = AgentStatus.AVAILABLE
    current_session: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_available(self) -> bool:
        """Kiểm tra agent có sẵn sàng không."""
        return self.status == AgentStatus.AVAILABLE

    def is_busy(self) -> bool:
        """Kiểm tra agent có đang bận không."""
        return self.status == AgentStatus.BUSY

    def is_offline(self) -> bool:
        """Kiểm tra agent có offline không."""
        return self.status == AgentStatus.OFFLINE

    def has_capability(self, capability: str) -> bool:
        """
        Kiểm tra agent có khả năng cụ thể không.

        Args:
            capability: Khả năng cần kiểm tra

        Returns:
            True nếu agent có khả năng này
        """
        return capability in self.capabilities

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển AgentInfo thành dictionary.

        Returns:
            Dictionary chứa thông tin agent
        """
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "capabilities": self.capabilities,
            "status": self.status.value,
            "current_session": self.current_session,
            "description": self.description,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentInfo":
        """
        Tạo AgentInfo từ dictionary.

        Args:
            data: Dictionary chứa thông tin agent

        Returns:
            AgentInfo được khôi phục
        """
        return cls(
            agent_id=data["agent_id"],
            role=data["role"],
            capabilities=data.get("capabilities", []),
            status=AgentStatus(data.get("status", "available")),
            current_session=data.get("current_session"),
            description=data.get("description"),
            metadata=data.get("metadata", {}),
        )
