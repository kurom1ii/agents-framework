"""
Tool sessions_history cho phép agents truy cập lịch sử của sessions khác.

Tool này là một phần của hệ thống A2A (Agent-to-Agent), cho phép agents
xem lịch sử hội thoại của các sessions khác để hiểu context hoặc
theo dõi tiến độ công việc.

Ví dụ sử dụng trong agent:
    # Lấy lịch sử gần nhất của session
    history = await agent.call_tool(
        "sessions_history",
        session_key="agent:researcher:main",
        limit=10,
        format="summary"
    )

    # Lấy lịch sử từ thời điểm cụ thể
    history = await agent.call_tool(
        "sessions_history",
        session_key="agent:coder:main",
        since="2024-01-15T10:00:00",
        format="full"
    )
"""

from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Protocol

from ..messaging import SessionHistory
from ..discovery import SessionDiscovery


class TranscriptStoreProtocol(Protocol):
    """
    Protocol định nghĩa interface cho TranscriptStore.

    Đây là interface mà sessions_history tool yêu cầu để truy cập
    lịch sử hội thoại của sessions.
    """

    async def get_recent(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[Any]:
        """Lấy các entries gần nhất của session."""
        ...

    async def get_entries(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Any]:
        """Lấy các entries của session."""
        ...

    async def count(self, session_id: str) -> int:
        """Đếm số entries của session."""
        ...


# Context variables để lưu trữ dependencies
_current_transcript_store: ContextVar[Optional[TranscriptStoreProtocol]] = ContextVar(
    "current_transcript_store", default=None
)

_current_discovery: ContextVar[Optional[SessionDiscovery]] = ContextVar(
    "current_history_discovery", default=None
)

_current_session_key: ContextVar[Optional[str]] = ContextVar(
    "current_history_session_key", default=None
)


def set_current_transcript_store(store: TranscriptStoreProtocol) -> None:
    """
    Đặt TranscriptStore instance cho context hiện tại.

    Args:
        store: TranscriptStore instance để sử dụng
    """
    _current_transcript_store.set(store)


def get_current_transcript_store() -> TranscriptStoreProtocol:
    """
    Lấy TranscriptStore instance từ context hiện tại.

    Returns:
        TranscriptStore instance

    Raises:
        RuntimeError: Nếu chưa có store được set trong context
    """
    store = _current_transcript_store.get()
    if store is None:
        raise RuntimeError(
            "TranscriptStore chưa được khởi tạo. "
            "Hãy đảm bảo runtime đã gọi set_current_transcript_store() "
            "trước khi sử dụng tool sessions_history."
        )
    return store


def set_history_discovery(discovery: SessionDiscovery) -> None:
    """
    Đặt SessionDiscovery instance cho sessions_history.

    Args:
        discovery: SessionDiscovery instance
    """
    _current_discovery.set(discovery)


def get_history_discovery() -> SessionDiscovery:
    """
    Lấy SessionDiscovery instance.

    Returns:
        SessionDiscovery instance

    Raises:
        RuntimeError: Nếu chưa có discovery được set
    """
    discovery = _current_discovery.get()
    if discovery is None:
        raise RuntimeError(
            "SessionDiscovery chưa được khởi tạo cho sessions_history."
        )
    return discovery


def set_history_session_key(session_key: str) -> None:
    """
    Đặt session key của session hiện tại.

    Args:
        session_key: Key của session đang thực thi
    """
    _current_session_key.set(session_key)


def get_history_session_key() -> str:
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
            "Session key chưa được set cho sessions_history."
        )
    return session_key


# Tool definition metadata
SESSIONS_HISTORY_TOOL_DEFINITION = {
    "name": "sessions_history",
    "description": """Truy cập lịch sử hội thoại của session khác.

Tool này cho phép agent xem lịch sử hội thoại của các sessions khác
trong hệ thống để:
- Hiểu context của session đó
- Theo dõi tiến độ công việc
- Tham khảo kết quả đã xử lý trước đó

Các trường hợp sử dụng:
- Xem conversation history của session để tiếp tục công việc
- Kiểm tra kết quả xử lý của agent khác
- Tổng hợp thông tin từ nhiều sessions""",
    "parameters": {
        "type": "object",
        "properties": {
            "session_key": {
                "type": "string",
                "description": """Key của session cần xem lịch sử.
Format: agent:<agentId>:<scope>:<identifier>
Ví dụ: "agent:coder:main", "agent:researcher:dm:user123"
Có thể lấy từ tool sessions_list.""",
            },
            "limit": {
                "type": "integer",
                "default": 20,
                "minimum": 1,
                "maximum": 100,
                "description": "Số lượng entries tối đa trả về. Mặc định 20.",
            },
            "since": {
                "type": "string",
                "description": """Lấy entries từ thời điểm này (ISO 8601 format).
Ví dụ: "2024-01-15T10:00:00", "2024-01-15"
Để trống để lấy entries gần nhất.""",
            },
            "format": {
                "type": "string",
                "enum": ["full", "summary", "messages_only"],
                "default": "summary",
                "description": """Định dạng output:
- "full": Tất cả thông tin bao gồm metadata
- "summary": Tóm tắt các messages chính (mặc định)
- "messages_only": Chỉ nội dung messages""",
            },
        },
        "required": ["session_key"],
    },
}


async def sessions_history(
    session_key: str,
    limit: int = 20,
    since: Optional[str] = None,
    format: Literal["full", "summary", "messages_only"] = "summary",
) -> Dict[str, Any]:
    """
    Truy cập lịch sử hội thoại của session khác.

    Tool này cho phép agent xem lịch sử hội thoại của các sessions khác
    trong hệ thống để hiểu context hoặc theo dõi tiến độ công việc.

    Args:
        session_key: Key của session cần xem lịch sử.
            Format: agent:<agentId>:<scope>:<identifier>
            Ví dụ: "agent:coder:main", "agent:researcher:dm:user123"
        limit: Số lượng entries tối đa trả về (1-100). Mặc định 20.
        since: Lấy entries từ thời điểm này (ISO 8601 format).
            Ví dụ: "2024-01-15T10:00:00"
            None để lấy entries gần nhất.
        format: Định dạng output
            - "full": Tất cả thông tin bao gồm metadata
            - "summary": Tóm tắt các messages chính (mặc định)
            - "messages_only": Chỉ nội dung messages

    Returns:
        SessionHistory dictionary chứa:
        - session_key: Key của session
        - entries: Danh sách entries với định dạng theo format
        - total_count: Tổng số entries trong session
        - has_more: Còn entries khác không
        - metadata: Thông tin bổ sung về session

    Raises:
        RuntimeError: Nếu các dependencies chưa được khởi tạo

    Ví dụ:
        # Lấy 10 entries gần nhất, định dạng summary
        history = await sessions_history(
            session_key="agent:researcher:main",
            limit=10,
            format="summary"
        )
        for entry in history["entries"]:
            print(f"{entry['role']}: {entry['content'][:100]}...")

        # Lấy entries từ thời điểm cụ thể, định dạng đầy đủ
        history = await sessions_history(
            session_key="agent:coder:main",
            since="2024-01-15T10:00:00",
            format="full"
        )

    Ghi chú:
        - Session đích phải tồn tại và có transcript
        - Entries được trả về theo thứ tự thời gian (cũ -> mới)
        - Format "summary" lọc bỏ system messages và tool calls chi tiết
        - has_more=True nghĩa là còn entries cũ hơn không được trả về
    """
    # Validate inputs
    if not session_key or not session_key.strip():
        return {
            "session_key": "",
            "entries": [],
            "total_count": 0,
            "has_more": False,
            "metadata": {"error": "session_key không được để trống"},
        }

    # Validate limit
    limit = max(1, min(100, limit))

    # Parse since datetime
    since_datetime: Optional[datetime] = None
    if since:
        try:
            since_datetime = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            return {
                "session_key": session_key,
                "entries": [],
                "total_count": 0,
                "has_more": False,
                "metadata": {"error": f"since không hợp lệ: {since}. Dùng ISO 8601 format."},
            }

    # Lấy dependencies từ context
    transcript_store = get_current_transcript_store()
    discovery = get_history_discovery()

    # Kiểm tra session có tồn tại không
    session_info = await discovery.get_session_info(session_key)
    if session_info is None:
        return {
            "session_key": session_key,
            "entries": [],
            "total_count": 0,
            "has_more": False,
            "metadata": {"error": f"Session không tồn tại: {session_key}"},
        }

    # Lấy session_id từ session_key (có thể cùng hoặc khác)
    # Thường session_id được lấy từ Session object
    session_id = session_key  # Simplify: dùng session_key làm session_id

    # Lấy entries từ transcript store
    if since_datetime:
        # Lấy tất cả và filter theo thời gian
        all_entries = await transcript_store.get_entries(session_id, limit=1000)
        entries = [
            e for e in all_entries
            if hasattr(e, 'timestamp') and e.timestamp >= since_datetime
        ][:limit]
    else:
        # Lấy entries gần nhất
        entries = await transcript_store.get_recent(session_id, limit=limit)

    # Đếm tổng số entries
    total_count = await transcript_store.count(session_id)
    has_more = total_count > len(entries)

    # Format entries theo yêu cầu
    formatted_entries = _format_entries(entries, format)

    # Tạo SessionHistory response
    return {
        "session_key": session_key,
        "entries": formatted_entries,
        "total_count": total_count,
        "has_more": has_more,
        "metadata": {
            "agent_id": session_info.agent_id,
            "status": session_info.status.value,
            "last_activity": session_info.last_activity.isoformat(),
            "context_tokens": session_info.context_tokens,
            "format": format,
        },
    }


def _format_entries(
    entries: List[Any],
    format_type: str
) -> List[Dict[str, Any]]:
    """
    Format entries theo định dạng yêu cầu.

    Args:
        entries: Danh sách TranscriptEntry
        format_type: Loại format (full, summary, messages_only)

    Returns:
        Danh sách entries đã format
    """
    result = []

    for entry in entries:
        # Xử lý entry là dict hoặc object
        if isinstance(entry, dict):
            entry_dict = entry
        elif hasattr(entry, 'to_dict'):
            entry_dict = entry.to_dict()
        else:
            # Tạo dict từ attributes
            entry_dict = {
                "role": getattr(entry, 'role', 'unknown'),
                "content": getattr(entry, 'content', ''),
                "timestamp": getattr(entry, 'timestamp', datetime.utcnow()).isoformat()
                    if hasattr(getattr(entry, 'timestamp', None), 'isoformat')
                    else str(getattr(entry, 'timestamp', '')),
                "entry_id": getattr(entry, 'entry_id', ''),
                "metadata": getattr(entry, 'metadata', {}),
            }

        # Format theo loại
        if format_type == "full":
            # Trả về tất cả thông tin
            result.append(entry_dict)

        elif format_type == "messages_only":
            # Chỉ trả về role và content
            result.append({
                "role": entry_dict.get("role", "unknown"),
                "content": entry_dict.get("content", ""),
            })

        else:  # summary
            # Trả về thông tin chính, bỏ metadata chi tiết
            role = entry_dict.get("role", "unknown")

            # Bỏ qua system messages trong summary
            if role == "system":
                continue

            # Tóm tắt tool results
            content = entry_dict.get("content", "")
            if role == "tool":
                tool_name = entry_dict.get("metadata", {}).get("tool_name", "tool")
                # Cắt ngắn content nếu quá dài
                if len(content) > 200:
                    content = content[:200] + "..."
                content = f"[{tool_name}] {content}"

            result.append({
                "role": role,
                "content": content,
                "timestamp": entry_dict.get("timestamp", ""),
            })

    return result


def create_sessions_history_tool():
    """
    Tạo FunctionTool instance cho sessions_history.

    Returns:
        FunctionTool instance đã được cấu hình

    Ví dụ:
        from agents_framework.a2a.tools import create_sessions_history_tool

        tool = create_sessions_history_tool()
        registry.register(tool)
    """
    from ...tools.base import FunctionTool

    return FunctionTool(
        func=sessions_history,
        name="sessions_history",
        description=SESSIONS_HISTORY_TOOL_DEFINITION["description"],
    )
