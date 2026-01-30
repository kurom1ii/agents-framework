"""
Tool sessions_list cho phép agents khám phá các sessions đang hoạt động.

Tool này là một phần của hệ thống A2A (Agent-to-Agent), cho phép agents
tìm kiếm và liệt kê các sessions khác để có thể giao tiếp hoặc ủy quyền tác vụ.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Dict, List, Literal, Optional

from ..base import SessionStatus
from ..discovery import SessionDiscovery

# Context variable để lưu trữ discovery instance hiện tại
_current_discovery: ContextVar[Optional[SessionDiscovery]] = ContextVar(
    "current_discovery", default=None
)


def set_current_discovery(discovery: SessionDiscovery) -> None:
    """
    Đặt SessionDiscovery instance cho context hiện tại.

    Hàm này được gọi bởi runtime để inject SessionDiscovery vào context
    trước khi tool được thực thi.

    Args:
        discovery: SessionDiscovery instance để sử dụng
    """
    _current_discovery.set(discovery)


def get_current_discovery() -> SessionDiscovery:
    """
    Lấy SessionDiscovery instance từ context hiện tại.

    Returns:
        SessionDiscovery instance

    Raises:
        RuntimeError: Nếu chưa có discovery được set trong context
    """
    discovery = _current_discovery.get()
    if discovery is None:
        raise RuntimeError(
            "SessionDiscovery chưa được khởi tạo. "
            "Hãy đảm bảo runtime đã gọi set_current_discovery() "
            "trước khi sử dụng tool sessions_list."
        )
    return discovery


# Tool definition metadata
SESSIONS_LIST_TOOL_DEFINITION = {
    "name": "sessions_list",
    "description": """Khám phá các sessions đang hoạt động trong hệ thống.

Tool này cho phép agent khám phá các sessions khác trong hệ thống
để có thể giao tiếp hoặc ủy quyền tác vụ thông qua hệ thống A2A.

Các trường hợp sử dụng:
- Tìm sessions đang rảnh để ủy quyền tác vụ
- Kiểm tra sessions của một agent cụ thể
- Theo dõi hoạt động của các sessions trong hệ thống
- Chuẩn bị cho việc giao tiếp A2A với sessions khác""",
    "parameters": {
        "type": "object",
        "properties": {
            "filter": {
                "type": "string",
                "enum": ["active", "all", "idle"],
                "default": "active",
                "description": """Lọc sessions theo trạng thái:
- "active": Sessions hoạt động gần đây (mặc định)
- "all": Tất cả sessions
- "idle": Sessions đang idle, sẵn sàng nhận tác vụ mới""",
            },
            "limit": {
                "type": "integer",
                "default": 10,
                "minimum": 1,
                "maximum": 100,
                "description": "Số lượng sessions tối đa trả về (1-100)",
            },
            "agent_filter": {
                "type": "string",
                "description": "Lọc theo agent_id cụ thể. Để trống để lấy tất cả agents.",
            },
        },
        "required": [],
    },
}


async def sessions_list(
    filter: Literal["active", "all", "idle"] = "active",
    limit: int = 10,
    agent_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Khám phá các sessions đang hoạt động trong hệ thống.

    Tool này cho phép agent khám phá các sessions khác trong hệ thống
    để có thể giao tiếp hoặc ủy quyền tác vụ thông qua hệ thống A2A.

    Args:
        filter: Lọc sessions theo trạng thái
            - "active": Sessions hoạt động gần đây trong 60 phút (mặc định)
            - "all": Tất cả sessions không phân biệt trạng thái
            - "idle": Sessions đang idle, sẵn sàng nhận tác vụ mới
        limit: Số lượng sessions tối đa trả về (1-100). Mặc định 10.
        agent_filter: Lọc theo agent_id cụ thể. None để lấy tất cả.

    Returns:
        Danh sách sessions với các thông tin:
        - session_key: Key duy nhất để gửi message đến session
        - agent_id: ID của agent sở hữu session
        - status: Trạng thái hiện tại (idle/busy/error)
        - last_activity: Thời gian hoạt động cuối (ISO format)
        - context_tokens: Số tokens trong context hiện tại
        - display_name: Tên hiển thị thân thiện (nếu có)

    Raises:
        RuntimeError: Nếu SessionDiscovery chưa được khởi tạo

    Example:
        # Tìm tất cả sessions đang hoạt động
        sessions = await sessions_list()
        for s in sessions:
            print(f"{s['session_key']}: {s['status']}")

        # Tìm sessions đang idle của coder agent
        idle_sessions = await sessions_list(
            filter="idle",
            agent_filter="coder"
        )

        # Lấy tất cả sessions, giới hạn 20
        all_sessions = await sessions_list(filter="all", limit=20)

    Ghi chú:
        - Sessions đang busy có thể không phản hồi ngay lập tức
        - Sessions đang idle là ứng cử viên tốt để ủy quyền tác vụ
        - context_tokens cho biết session đã sử dụng bao nhiêu tokens,
          hữu ích để đánh giá độ phức tạp của context hiện tại
    """
    # Validate limit
    limit = max(1, min(100, limit))

    # Lấy discovery từ context
    discovery = get_current_discovery()

    # Lấy sessions theo filter
    if filter == "active":
        sessions = await discovery.list_active_sessions(minutes=60)
    elif filter == "idle":
        sessions = await discovery.list_sessions(
            filter_status=SessionStatus.IDLE, limit=limit * 2
        )
    else:  # all
        sessions = await discovery.list_sessions(limit=limit * 2)

    # Lọc theo agent nếu có
    if agent_filter:
        sessions = [s for s in sessions if s.agent_id == agent_filter]

    # Giới hạn kết quả và chuyển đổi sang dict
    result: List[Dict[str, Any]] = []
    for session in sessions[:limit]:
        result.append(
            {
                "session_key": session.session_key,
                "agent_id": session.agent_id,
                "status": session.status.value,
                "last_activity": session.last_activity.isoformat(),
                "context_tokens": session.context_tokens,
                "display_name": session.display_name,
            }
        )

    return result


# Tạo FunctionTool từ function sessions_list
# Import lazy để tránh circular import
def create_sessions_list_tool():
    """
    Tạo FunctionTool instance cho sessions_list.

    Returns:
        FunctionTool instance đã được cấu hình

    Example:
        from agents_framework.a2a.tools import create_sessions_list_tool

        tool = create_sessions_list_tool()
        registry.register(tool)
    """
    from ...tools.base import FunctionTool

    return FunctionTool(
        func=sessions_list,
        name="sessions_list",
        description=SESSIONS_LIST_TOOL_DEFINITION["description"],
    )
