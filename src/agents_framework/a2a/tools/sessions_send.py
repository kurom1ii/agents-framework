"""
Tool sessions_send cho phép agents gửi tin nhắn đến sessions khác.

Tool này là một phần của hệ thống A2A (Agent-to-Agent), cho phép agents
giao tiếp với nhau thông qua inter-session messaging. Hỗ trợ cả sync
(đợi response) và async (fire-and-forget) modes.

Ví dụ sử dụng trong agent:
    # Gửi request và đợi response
    response = await agent.call_tool(
        "sessions_send",
        session_key="agent:coder:main",
        message="Hãy review PR #123",
        wait_for_response=True,
        timeout=30000
    )

    # Gửi notification (không đợi response)
    await agent.call_tool(
        "sessions_send",
        session_key="agent:monitor:main",
        message="Task completed",
        wait_for_response=False
    )
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Dict, Literal, Optional

from ..messaging import A2AResponse, MessagePriority
from ..router import InterSessionMessaging


# Context variable để lưu trữ messaging instance và current session
_current_messaging: ContextVar[Optional[InterSessionMessaging]] = ContextVar(
    "current_messaging", default=None
)

_current_session_key: ContextVar[Optional[str]] = ContextVar(
    "current_session_key", default=None
)


def set_current_messaging(messaging: InterSessionMessaging) -> None:
    """
    Đặt InterSessionMessaging instance cho context hiện tại.

    Hàm này được gọi bởi runtime để inject messaging vào context
    trước khi tool được thực thi.

    Args:
        messaging: InterSessionMessaging instance để sử dụng
    """
    _current_messaging.set(messaging)


def get_current_messaging() -> InterSessionMessaging:
    """
    Lấy InterSessionMessaging instance từ context hiện tại.

    Returns:
        InterSessionMessaging instance

    Raises:
        RuntimeError: Nếu chưa có messaging được set trong context
    """
    messaging = _current_messaging.get()
    if messaging is None:
        raise RuntimeError(
            "InterSessionMessaging chưa được khởi tạo. "
            "Hãy đảm bảo runtime đã gọi set_current_messaging() "
            "trước khi sử dụng tool sessions_send."
        )
    return messaging


def set_current_session_key(session_key: str) -> None:
    """
    Đặt session key của session hiện tại.

    Args:
        session_key: Key của session đang thực thi
    """
    _current_session_key.set(session_key)


def get_current_session_key() -> str:
    """
    Lấy session key của session hiện tại.

    Returns:
        Session key

    Raises:
        RuntimeError: Nếu chưa có session key được set
    """
    session_key = _current_session_key.get()
    if session_key is None:
        raise RuntimeError(
            "Session key chưa được set. "
            "Hãy đảm bảo runtime đã gọi set_current_session_key() "
            "trước khi sử dụng tool sessions_send."
        )
    return session_key


# Tool definition metadata
SESSIONS_SEND_TOOL_DEFINITION = {
    "name": "sessions_send",
    "description": """Gửi tin nhắn đến session khác trong hệ thống A2A.

Tool này cho phép agent gửi tin nhắn đến sessions khác để:
- Yêu cầu xử lý tác vụ (sync mode với wait_for_response=True)
- Gửi thông báo một chiều (async mode với wait_for_response=False)
- Ủy quyền công việc cho agents khác

Các trường hợp sử dụng:
- Yêu cầu agent khác review code, viết test
- Thông báo hoàn thành tác vụ
- Phối hợp công việc giữa nhiều agents""",
    "parameters": {
        "type": "object",
        "properties": {
            "session_key": {
                "type": "string",
                "description": """Key của session đích.
Format: agent:<agentId>:<scope>:<identifier>
Ví dụ: "agent:coder:main", "agent:researcher:dm:user123"
Có thể lấy từ tool sessions_list.""",
            },
            "message": {
                "type": "string",
                "description": "Nội dung tin nhắn gửi đến session đích.",
            },
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high"],
                "default": "normal",
                "description": """Độ ưu tiên của tin nhắn:
- "low": Xử lý khi rảnh
- "normal": Độ ưu tiên bình thường (mặc định)
- "high": Xử lý ngay lập tức""",
            },
            "wait_for_response": {
                "type": "boolean",
                "default": True,
                "description": """Có đợi phản hồi từ session đích không:
- True: Đợi session đích xử lý và trả về kết quả (mặc định)
- False: Gửi xong trả về ngay, không đợi phản hồi""",
            },
            "timeout": {
                "type": "integer",
                "default": 30000,
                "minimum": 1000,
                "maximum": 300000,
                "description": "Thời gian chờ phản hồi tối đa (ms). Chỉ áp dụng khi wait_for_response=True. Mặc định 30000ms (30s).",
            },
        },
        "required": ["session_key", "message"],
    },
}


async def sessions_send(
    session_key: str,
    message: str,
    priority: Literal["low", "normal", "high"] = "normal",
    wait_for_response: bool = True,
    timeout: int = 30000,
) -> Dict[str, Any]:
    """
    Gửi tin nhắn đến session khác trong hệ thống A2A.

    Tool này cho phép agent gửi tin nhắn đến sessions khác để yêu cầu
    xử lý tác vụ hoặc gửi thông báo. Hỗ trợ cả sync (đợi response)
    và async (fire-and-forget) modes.

    Args:
        session_key: Key của session đích.
            Format: agent:<agentId>:<scope>:<identifier>
            Ví dụ: "agent:coder:main", "agent:researcher:dm:user123"
        message: Nội dung tin nhắn gửi đến session đích.
        priority: Độ ưu tiên của tin nhắn
            - "low": Xử lý khi rảnh
            - "normal": Độ ưu tiên bình thường (mặc định)
            - "high": Xử lý ngay lập tức
        wait_for_response: Có đợi phản hồi không
            - True: Đợi session đích xử lý và trả về kết quả (mặc định)
            - False: Gửi xong trả về ngay, không đợi phản hồi
        timeout: Thời gian chờ phản hồi tối đa (ms).
            Chỉ áp dụng khi wait_for_response=True. Mặc định 30000ms.

    Returns:
        Dictionary chứa:
        - message_id: ID của tin nhắn
        - success: True/False
        - response: Nội dung phản hồi (nếu wait_for_response=True và success=True)
        - error: Thông tin lỗi (nếu success=False)
        - response_time_ms: Thời gian xử lý (ms)

    Raises:
        RuntimeError: Nếu messaging hoặc session_key chưa được khởi tạo

    Ví dụ:
        # Sync mode: Gửi yêu cầu và đợi phản hồi
        result = await sessions_send(
            session_key="agent:coder:main",
            message="Review PR #123 và cho nhận xét",
            wait_for_response=True,
            timeout=60000
        )
        if result["success"]:
            print(f"Coder phản hồi: {result['response']}")

        # Async mode: Gửi thông báo không đợi
        await sessions_send(
            session_key="agent:monitor:main",
            message="Task đã hoàn thành",
            wait_for_response=False
        )

    Ghi chú:
        - Nếu session đích đang busy, message sẽ được queue
        - Nếu session đích offline, message được lưu để gửi sau
        - timeout chỉ có tác dụng khi wait_for_response=True
        - Messages với priority="high" được xử lý trước
    """
    # Validate inputs
    if not session_key or not session_key.strip():
        return {
            "message_id": "",
            "success": False,
            "error": "session_key không được để trống",
            "response_time_ms": 0,
        }

    if not message or not message.strip():
        return {
            "message_id": "",
            "success": False,
            "error": "message không được để trống",
            "response_time_ms": 0,
        }

    # Validate timeout
    timeout = max(1000, min(300000, timeout))

    # Lấy messaging và session key từ context
    messaging = get_current_messaging()
    from_session = get_current_session_key()

    # Kiểm tra không gửi cho chính mình
    if session_key == from_session:
        return {
            "message_id": "",
            "success": False,
            "error": "Không thể gửi tin nhắn cho chính mình",
            "response_time_ms": 0,
        }

    # Gửi message
    response = await messaging.send(
        from_session=from_session,
        to_session=session_key,
        message=message,
        wait_for_response=wait_for_response,
        timeout_ms=timeout,
        priority=priority,
    )

    # Chuyển A2AResponse thành dict
    return response.to_dict()


def create_sessions_send_tool():
    """
    Tạo FunctionTool instance cho sessions_send.

    Returns:
        FunctionTool instance đã được cấu hình

    Ví dụ:
        from agents_framework.a2a.tools import create_sessions_send_tool

        tool = create_sessions_send_tool()
        registry.register(tool)
    """
    from ...tools.base import FunctionTool

    return FunctionTool(
        func=sessions_send,
        name="sessions_send",
        description=SESSIONS_SEND_TOOL_DEFINITION["description"],
    )
