"""
A2A Inter-Session Messaging - Message Types và Data Classes.

Module này định nghĩa các message types cho hệ thống giao tiếp giữa các sessions,
bao gồm A2AMessage, A2AResponse và các enums liên quan.

Các loại message được hỗ trợ:
- Request: Yêu cầu xử lý và trả về kết quả (synchronous)
- Notification: Thông báo một chiều (asynchronous)
- Event: Broadcast đến nhiều sessions

Ví dụ sử dụng:
    ```python
    from agents_framework.a2a.messaging import A2AMessage, MessageType, MessagePriority

    # Tạo request message
    message = A2AMessage.create_request(
        from_session="agent:orchestrator:main",
        to_session="agent:coder:main",
        content="Hãy review PR #123",
        timeout_ms=30000
    )

    # Tạo notification
    notification = A2AMessage.create_notification(
        from_session="agent:monitor:main",
        to_session="agent:admin:main",
        content="Có lỗi xảy ra trong hệ thống"
    )
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Literal, Optional
import uuid


class MessageType(Enum):
    """
    Loại tin nhắn trong hệ thống A2A.

    - REQUEST: Yêu cầu xử lý và đợi phản hồi
    - NOTIFICATION: Thông báo một chiều, không đợi phản hồi
    - EVENT: Broadcast đến nhiều sessions
    """

    REQUEST = "request"
    NOTIFICATION = "notification"
    EVENT = "event"


class MessagePriority(Enum):
    """
    Độ ưu tiên của tin nhắn.

    - LOW: Độ ưu tiên thấp, xử lý khi rảnh
    - NORMAL: Độ ưu tiên bình thường
    - HIGH: Độ ưu tiên cao, xử lý ngay lập tức
    """

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"

    @classmethod
    def from_string(cls, value: str) -> "MessagePriority":
        """
        Chuyển string thành MessagePriority.

        Args:
            value: Giá trị string (low, normal, high)

        Returns:
            MessagePriority tương ứng

        Raises:
            ValueError: Nếu giá trị không hợp lệ
        """
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Độ ưu tiên không hợp lệ: {value}. "
                f"Các giá trị hợp lệ: {[p.value for p in cls]}"
            )


class MessageState(Enum):
    """
    Trạng thái của tin nhắn trong queue.

    - PENDING: Đang chờ gửi
    - DELIVERED: Đã gửi đến session đích
    - PROCESSING: Đang được xử lý
    - COMPLETED: Đã xử lý xong
    - FAILED: Xử lý thất bại
    - EXPIRED: Đã hết hạn (timeout)
    """

    PENDING = "pending"
    DELIVERED = "delivered"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class A2AMessage:
    """
    Tin nhắn trong hệ thống giao tiếp Agent-to-Agent.

    A2AMessage đại diện cho một tin nhắn được gửi từ một session đến
    một hoặc nhiều sessions khác trong hệ thống.

    Attributes:
        id: ID duy nhất của tin nhắn (UUID)
        from_session: Session key của người gửi
        to_session: Session key của người nhận (hoặc pattern cho broadcast)
        message_type: Loại tin nhắn (request, notification, event)
        content: Nội dung tin nhắn
        priority: Độ ưu tiên (low, normal, high)
        timeout_ms: Thời gian timeout tính bằng milliseconds
        created_at: Thời điểm tạo tin nhắn
        state: Trạng thái hiện tại của tin nhắn
        metadata: Dữ liệu bổ sung tùy chỉnh
        topic: Topic cho event messages (optional)
        correlation_id: ID để liên kết request-response (optional)
        reply_to: Session để gửi response về (optional)

    Ví dụ:
        message = A2AMessage(
            id=str(uuid.uuid4()),
            from_session="agent:orchestrator:main",
            to_session="agent:coder:main",
            message_type=MessageType.REQUEST,
            content="Review PR #123",
            priority=MessagePriority.HIGH,
            timeout_ms=60000
        )
    """

    id: str
    from_session: str
    to_session: str
    message_type: MessageType
    content: str
    priority: MessagePriority = MessagePriority.NORMAL
    timeout_ms: int = 30000
    created_at: datetime = field(default_factory=datetime.utcnow)
    state: MessageState = MessageState.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    topic: Optional[str] = None
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate và khởi tạo giá trị mặc định sau khi tạo."""
        if not self.id:
            self.id = str(uuid.uuid4())

        if self.timeout_ms <= 0:
            raise ValueError(f"timeout_ms phải > 0, nhận được: {self.timeout_ms}")

        # Validate session keys
        if not self.from_session:
            raise ValueError("from_session không được để trống")

        # to_session có thể rỗng cho broadcast events
        if self.message_type != MessageType.EVENT and not self.to_session:
            raise ValueError("to_session không được để trống cho request/notification")

    @classmethod
    def create_request(
        cls,
        from_session: str,
        to_session: str,
        content: str,
        priority: MessagePriority = MessagePriority.NORMAL,
        timeout_ms: int = 30000,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "A2AMessage":
        """
        Tạo tin nhắn request đợi phản hồi.

        Request message yêu cầu session đích xử lý và trả về kết quả.
        Sender sẽ đợi response hoặc timeout.

        Args:
            from_session: Session key người gửi
            to_session: Session key người nhận
            content: Nội dung yêu cầu
            priority: Độ ưu tiên (mặc định normal)
            timeout_ms: Timeout milliseconds (mặc định 30s)
            metadata: Dữ liệu bổ sung

        Returns:
            A2AMessage với message_type=REQUEST

        Ví dụ:
            request = A2AMessage.create_request(
                from_session="agent:orchestrator:main",
                to_session="agent:coder:main",
                content="Implement feature X",
                timeout_ms=60000
            )
        """
        message_id = str(uuid.uuid4())
        return cls(
            id=message_id,
            from_session=from_session,
            to_session=to_session,
            message_type=MessageType.REQUEST,
            content=content,
            priority=priority,
            timeout_ms=timeout_ms,
            metadata=metadata or {},
            correlation_id=message_id,
            reply_to=from_session,
        )

    @classmethod
    def create_notification(
        cls,
        from_session: str,
        to_session: str,
        content: str,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "A2AMessage":
        """
        Tạo tin nhắn notification một chiều.

        Notification message thông báo đến session đích mà không đợi phản hồi.
        Phù hợp cho các thông báo trạng thái, cảnh báo.

        Args:
            from_session: Session key người gửi
            to_session: Session key người nhận
            content: Nội dung thông báo
            priority: Độ ưu tiên (mặc định normal)
            metadata: Dữ liệu bổ sung

        Returns:
            A2AMessage với message_type=NOTIFICATION

        Ví dụ:
            notification = A2AMessage.create_notification(
                from_session="agent:monitor:main",
                to_session="agent:admin:main",
                content="Task completed successfully"
            )
        """
        return cls(
            id=str(uuid.uuid4()),
            from_session=from_session,
            to_session=to_session,
            message_type=MessageType.NOTIFICATION,
            content=content,
            priority=priority,
            timeout_ms=5000,  # Notification có timeout ngắn
            metadata=metadata or {},
        )

    @classmethod
    def create_event(
        cls,
        from_session: str,
        topic: str,
        content: str,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "A2AMessage":
        """
        Tạo tin nhắn event để broadcast.

        Event message được gửi đến tất cả sessions đã subscribe topic.
        Phù hợp cho các sự kiện hệ thống, updates.

        Args:
            from_session: Session key người gửi
            topic: Topic để broadcast
            content: Nội dung sự kiện
            priority: Độ ưu tiên (mặc định normal)
            metadata: Dữ liệu bổ sung

        Returns:
            A2AMessage với message_type=EVENT

        Ví dụ:
            event = A2AMessage.create_event(
                from_session="agent:orchestrator:main",
                topic="task.completed",
                content="Task #123 has been completed"
            )
        """
        return cls(
            id=str(uuid.uuid4()),
            from_session=from_session,
            to_session="",  # Event không có to_session cụ thể
            message_type=MessageType.EVENT,
            content=content,
            priority=priority,
            timeout_ms=5000,
            metadata=metadata or {},
            topic=topic,
        )

    def is_request(self) -> bool:
        """Kiểm tra có phải request message không."""
        return self.message_type == MessageType.REQUEST

    def is_notification(self) -> bool:
        """Kiểm tra có phải notification message không."""
        return self.message_type == MessageType.NOTIFICATION

    def is_event(self) -> bool:
        """Kiểm tra có phải event message không."""
        return self.message_type == MessageType.EVENT

    def is_expired(self) -> bool:
        """
        Kiểm tra message đã hết hạn chưa.

        Returns:
            True nếu đã vượt quá timeout
        """
        if self.state == MessageState.EXPIRED:
            return True

        elapsed_ms = (datetime.utcnow() - self.created_at).total_seconds() * 1000
        return elapsed_ms > self.timeout_ms

    def is_high_priority(self) -> bool:
        """Kiểm tra có phải message ưu tiên cao không."""
        return self.priority == MessagePriority.HIGH

    def mark_delivered(self) -> None:
        """Đánh dấu message đã được gửi đến đích."""
        self.state = MessageState.DELIVERED

    def mark_processing(self) -> None:
        """Đánh dấu message đang được xử lý."""
        self.state = MessageState.PROCESSING

    def mark_completed(self) -> None:
        """Đánh dấu message đã xử lý xong."""
        self.state = MessageState.COMPLETED

    def mark_failed(self, error: Optional[str] = None) -> None:
        """
        Đánh dấu message xử lý thất bại.

        Args:
            error: Thông tin lỗi (optional)
        """
        self.state = MessageState.FAILED
        if error:
            self.metadata["error"] = error

    def mark_expired(self) -> None:
        """Đánh dấu message đã hết hạn."""
        self.state = MessageState.EXPIRED

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển message thành dictionary để lưu trữ.

        Returns:
            Dictionary chứa thông tin message
        """
        return {
            "id": self.id,
            "from_session": self.from_session,
            "to_session": self.to_session,
            "message_type": self.message_type.value,
            "content": self.content,
            "priority": self.priority.value,
            "timeout_ms": self.timeout_ms,
            "created_at": self.created_at.isoformat(),
            "state": self.state.value,
            "metadata": self.metadata,
            "topic": self.topic,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AMessage":
        """
        Tạo message từ dictionary.

        Args:
            data: Dictionary chứa thông tin message

        Returns:
            A2AMessage được khôi phục
        """
        return cls(
            id=data["id"],
            from_session=data["from_session"],
            to_session=data["to_session"],
            message_type=MessageType(data["message_type"]),
            content=data["content"],
            priority=MessagePriority(data["priority"]),
            timeout_ms=data["timeout_ms"],
            created_at=datetime.fromisoformat(data["created_at"]),
            state=MessageState(data["state"]),
            metadata=data.get("metadata", {}),
            topic=data.get("topic"),
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to"),
        )


@dataclass
class A2AResponse:
    """
    Phản hồi cho A2AMessage request.

    A2AResponse chứa kết quả xử lý của session đích cho một request message.

    Attributes:
        message_id: ID của message gốc
        success: Trạng thái xử lý (thành công/thất bại)
        response: Nội dung phản hồi (nếu thành công)
        error: Thông tin lỗi (nếu thất bại)
        response_time_ms: Thời gian xử lý tính bằng milliseconds
        from_session: Session đã xử lý request
        metadata: Dữ liệu bổ sung

    Ví dụ:
        response = A2AResponse(
            message_id="msg-123",
            success=True,
            response="PR #123 đã được review, có 2 comments",
            response_time_ms=1500
        )
    """

    message_id: str
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    response_time_ms: int = 0
    from_session: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success_response(
        cls,
        message_id: str,
        response: str,
        response_time_ms: int = 0,
        from_session: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "A2AResponse":
        """
        Tạo response thành công.

        Args:
            message_id: ID của message gốc
            response: Nội dung phản hồi
            response_time_ms: Thời gian xử lý
            from_session: Session đã xử lý
            metadata: Dữ liệu bổ sung

        Returns:
            A2AResponse với success=True
        """
        return cls(
            message_id=message_id,
            success=True,
            response=response,
            response_time_ms=response_time_ms,
            from_session=from_session,
            metadata=metadata or {},
        )

    @classmethod
    def error_response(
        cls,
        message_id: str,
        error: str,
        response_time_ms: int = 0,
        from_session: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "A2AResponse":
        """
        Tạo response lỗi.

        Args:
            message_id: ID của message gốc
            error: Thông tin lỗi
            response_time_ms: Thời gian xử lý
            from_session: Session đã xử lý
            metadata: Dữ liệu bổ sung

        Returns:
            A2AResponse với success=False
        """
        return cls(
            message_id=message_id,
            success=False,
            error=error,
            response_time_ms=response_time_ms,
            from_session=from_session,
            metadata=metadata or {},
        )

    @classmethod
    def timeout_response(
        cls,
        message_id: str,
        timeout_ms: int,
    ) -> "A2AResponse":
        """
        Tạo response timeout.

        Args:
            message_id: ID của message gốc
            timeout_ms: Thời gian timeout đã cấu hình

        Returns:
            A2AResponse với error timeout
        """
        return cls(
            message_id=message_id,
            success=False,
            error=f"Request timeout sau {timeout_ms}ms",
            response_time_ms=timeout_ms,
            metadata={"timeout": True},
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển response thành dictionary.

        Returns:
            Dictionary chứa thông tin response
        """
        return {
            "message_id": self.message_id,
            "success": self.success,
            "response": self.response,
            "error": self.error,
            "response_time_ms": self.response_time_ms,
            "from_session": self.from_session,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AResponse":
        """
        Tạo response từ dictionary.

        Args:
            data: Dictionary chứa thông tin response

        Returns:
            A2AResponse được khôi phục
        """
        return cls(
            message_id=data["message_id"],
            success=data["success"],
            response=data.get("response"),
            error=data.get("error"),
            response_time_ms=data.get("response_time_ms", 0),
            from_session=data.get("from_session"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SessionHistory:
    """
    Lịch sử tin nhắn của một session.

    SessionHistory chứa các entries trong lịch sử hội thoại của session,
    được sử dụng bởi sessions_history tool.

    Attributes:
        session_key: Key của session
        entries: Danh sách các entry trong lịch sử
        total_count: Tổng số entries
        has_more: Còn entries khác không
        metadata: Thông tin bổ sung về session
    """

    session_key: str
    entries: list = field(default_factory=list)
    total_count: int = 0
    has_more: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển history thành dictionary.

        Returns:
            Dictionary chứa thông tin history
        """
        return {
            "session_key": self.session_key,
            "entries": self.entries,
            "total_count": self.total_count,
            "has_more": self.has_more,
            "metadata": self.metadata,
        }
