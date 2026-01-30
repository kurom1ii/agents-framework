"""
Permission definitions và PermissionManager cho A2A Security.

Module này định nghĩa các permission levels và quản lý quyền truy cập
giữa các agents trong hệ thống A2A.

Permission Levels:
- NONE: Không có quyền truy cập
- NOTIFY: Chỉ được gửi notifications
- REQUEST: Được gửi request và nhận response
- HISTORY: Được đọc lịch sử session
- SPAWN: Được tạo sub-agents
- FULL: Tất cả quyền
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class PermissionLevel(Enum):
    """
    Các cấp độ quyền truy cập trong hệ thống A2A.

    Các cấp độ được sắp xếp theo thứ tự tăng dần của quyền hạn:
    - NONE: Không có quyền, bị từ chối hoàn toàn
    - NOTIFY: Chỉ được gửi thông báo một chiều (fire-and-forget)
    - REQUEST: Được gửi yêu cầu và chờ phản hồi
    - HISTORY: Được đọc lịch sử giao tiếp và session
    - SPAWN: Được tạo và quản lý sub-agents
    - FULL: Có tất cả các quyền trên

    Example:
        level = PermissionLevel.REQUEST
        if level.has_access(PermissionLevel.NOTIFY):
            # Có thể gửi notification
            pass
    """

    NONE = "none"
    NOTIFY = "notify"
    REQUEST = "request"
    HISTORY = "history"
    SPAWN = "spawn"
    FULL = "full"

    @classmethod
    def from_string(cls, value: str) -> "PermissionLevel":
        """
        Tạo PermissionLevel từ string.

        Args:
            value: String biểu diễn permission level

        Returns:
            PermissionLevel tương ứng

        Raises:
            ValueError: Nếu value không hợp lệ
        """
        for level in cls:
            if level.value == value.lower():
                return level
        raise ValueError(f"Permission level không hợp lệ: {value}")

    def has_access(self, required: "PermissionLevel") -> bool:
        """
        Kiểm tra permission level này có đủ quyền cho required level không.

        Permission hierarchy:
        FULL > SPAWN > HISTORY > REQUEST > NOTIFY > NONE

        Args:
            required: Permission level yêu cầu

        Returns:
            True nếu có đủ quyền

        Example:
            PermissionLevel.FULL.has_access(PermissionLevel.SPAWN)  # True
            PermissionLevel.NOTIFY.has_access(PermissionLevel.REQUEST)  # False
        """
        hierarchy = {
            PermissionLevel.NONE: 0,
            PermissionLevel.NOTIFY: 1,
            PermissionLevel.REQUEST: 2,
            PermissionLevel.HISTORY: 3,
            PermissionLevel.SPAWN: 4,
            PermissionLevel.FULL: 5,
        }
        return hierarchy.get(self, 0) >= hierarchy.get(required, 0)

    @property
    def can_notify(self) -> bool:
        """Kiểm tra có quyền gửi notification không."""
        return self.has_access(PermissionLevel.NOTIFY)

    @property
    def can_request(self) -> bool:
        """Kiểm tra có quyền gửi request không."""
        return self.has_access(PermissionLevel.REQUEST)

    @property
    def can_read_history(self) -> bool:
        """Kiểm tra có quyền đọc history không."""
        return self.has_access(PermissionLevel.HISTORY)

    @property
    def can_spawn(self) -> bool:
        """Kiểm tra có quyền tạo sub-agents không."""
        return self.has_access(PermissionLevel.SPAWN)


@dataclass
class A2APermission:
    """
    Định nghĩa một permission giữa hai agents.

    A2APermission mô tả quyền truy cập cụ thể của một agent (from_agent)
    đến một agent khác (to_agent) với một mức độ quyền (level) nhất định.

    Attributes:
        from_agent: ID của agent nguồn (agent yêu cầu quyền)
        to_agent: ID của agent đích (agent được truy cập)
        level: Mức độ quyền được cấp
        expires_at: Thời điểm hết hạn của permission (None = không hết hạn)
        granted_at: Thời điểm cấp permission
        granted_by: Agent hoặc system đã cấp permission này
        metadata: Dữ liệu bổ sung về permission

    Example:
        permission = A2APermission(
            from_agent="helper",
            to_agent="main-agent",
            level=PermissionLevel.REQUEST,
            expires_at=datetime.now() + timedelta(hours=24),
            granted_by="admin"
        )
    """

    from_agent: str
    to_agent: str
    level: PermissionLevel
    expires_at: Optional[datetime] = None
    granted_at: datetime = field(default_factory=datetime.now)
    granted_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """
        Kiểm tra permission đã hết hạn chưa.

        Returns:
            True nếu đã hết hạn, False nếu còn hiệu lực hoặc không có hạn
        """
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def is_valid(self) -> bool:
        """
        Kiểm tra permission có còn hiệu lực không.

        Returns:
            True nếu permission còn hiệu lực và không phải NONE
        """
        return not self.is_expired() and self.level != PermissionLevel.NONE

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển permission thành dictionary.

        Returns:
            Dictionary chứa thông tin permission
        """
        return {
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "level": self.level.value,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "granted_at": self.granted_at.isoformat(),
            "granted_by": self.granted_by,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2APermission":
        """
        Tạo A2APermission từ dictionary.

        Args:
            data: Dictionary chứa thông tin permission

        Returns:
            A2APermission instance
        """
        expires_at = None
        if data.get("expires_at"):
            expires_at = datetime.fromisoformat(data["expires_at"])

        return cls(
            from_agent=data["from_agent"],
            to_agent=data["to_agent"],
            level=PermissionLevel.from_string(data["level"]),
            expires_at=expires_at,
            granted_at=datetime.fromisoformat(data.get("granted_at", datetime.now().isoformat())),
            granted_by=data.get("granted_by"),
            metadata=data.get("metadata", {}),
        )


class PermissionManager:
    """
    Quản lý permissions giữa các agents trong hệ thống A2A.

    PermissionManager theo dõi và đánh giá các quyền truy cập giữa
    các agents, hỗ trợ cấp phép, thu hồi và kiểm tra permissions.

    Attributes:
        permissions: Dictionary lưu trữ permissions theo cặp (from_agent, to_agent)
        default_level: Permission level mặc định khi không có permission cụ thể

    Example:
        manager = PermissionManager()

        # Cấp quyền REQUEST cho helper -> main-agent
        manager.grant_permission(
            from_agent="helper",
            to_agent="main-agent",
            level=PermissionLevel.REQUEST
        )

        # Kiểm tra quyền
        if manager.check_permission("helper", "main-agent", PermissionLevel.REQUEST):
            # Được phép gửi request
            pass
    """

    def __init__(self, default_level: PermissionLevel = PermissionLevel.NONE) -> None:
        """
        Khởi tạo PermissionManager.

        Args:
            default_level: Permission level mặc định cho các cặp agent
                          chưa được cấu hình. Mặc định là NONE.
        """
        self.default_level = default_level
        self._permissions: Dict[tuple[str, str], A2APermission] = {}
        self._group_permissions: Dict[str, PermissionLevel] = {}

    def grant_permission(
        self,
        from_agent: str,
        to_agent: str,
        level: PermissionLevel,
        expires_at: Optional[datetime] = None,
        granted_by: Optional[str] = None,
    ) -> A2APermission:
        """
        Cấp permission cho một agent.

        Args:
            from_agent: ID agent nguồn (agent được cấp quyền)
            to_agent: ID agent đích (agent được truy cập)
            level: Mức độ quyền cấp cho
            expires_at: Thời điểm hết hạn (None = vĩnh viễn)
            granted_by: Agent hoặc system cấp permission

        Returns:
            A2APermission đã được tạo

        Example:
            permission = manager.grant_permission(
                "helper",
                "main-agent",
                PermissionLevel.REQUEST,
                expires_at=datetime.now() + timedelta(hours=1)
            )
        """
        permission = A2APermission(
            from_agent=from_agent,
            to_agent=to_agent,
            level=level,
            expires_at=expires_at,
            granted_by=granted_by,
        )
        self._permissions[(from_agent, to_agent)] = permission
        return permission

    def revoke_permission(self, from_agent: str, to_agent: str) -> bool:
        """
        Thu hồi permission giữa hai agents.

        Args:
            from_agent: ID agent nguồn
            to_agent: ID agent đích

        Returns:
            True nếu permission đã tồn tại và được thu hồi,
            False nếu không tồn tại permission
        """
        key = (from_agent, to_agent)
        if key in self._permissions:
            del self._permissions[key]
            return True
        return False

    def get_permission(self, from_agent: str, to_agent: str) -> Optional[A2APermission]:
        """
        Lấy permission giữa hai agents.

        Args:
            from_agent: ID agent nguồn
            to_agent: ID agent đích

        Returns:
            A2APermission nếu tồn tại, None nếu không
        """
        return self._permissions.get((from_agent, to_agent))

    def get_permission_level(self, from_agent: str, to_agent: str) -> PermissionLevel:
        """
        Lấy permission level hiện tại giữa hai agents.

        Phương thức này kiểm tra cả permissions cụ thể và group permissions,
        cũng như xử lý wildcard (*).

        Args:
            from_agent: ID agent nguồn
            to_agent: ID agent đích

        Returns:
            PermissionLevel hiện tại

        Example:
            level = manager.get_permission_level("helper", "main-agent")
            if level.can_request:
                # Có thể gửi request
                pass
        """
        # Kiểm tra permission cụ thể
        permission = self._permissions.get((from_agent, to_agent))
        if permission and permission.is_valid():
            return permission.level

        # Kiểm tra wildcard permission (* -> to_agent)
        wildcard_permission = self._permissions.get(("*", to_agent))
        if wildcard_permission and wildcard_permission.is_valid():
            return wildcard_permission.level

        # Kiểm tra wildcard permission (from_agent -> *)
        wildcard_permission = self._permissions.get((from_agent, "*"))
        if wildcard_permission and wildcard_permission.is_valid():
            return wildcard_permission.level

        # Kiểm tra global wildcard (* -> *)
        global_wildcard = self._permissions.get(("*", "*"))
        if global_wildcard and global_wildcard.is_valid():
            return global_wildcard.level

        # Trả về mức mặc định
        return self.default_level

    def check_permission(
        self,
        from_agent: str,
        to_agent: str,
        required_level: PermissionLevel,
    ) -> bool:
        """
        Kiểm tra agent có đủ quyền truy cập không.

        Args:
            from_agent: ID agent nguồn (agent yêu cầu)
            to_agent: ID agent đích (agent được truy cập)
            required_level: Mức độ quyền yêu cầu

        Returns:
            True nếu có đủ quyền, False nếu không

        Example:
            if manager.check_permission("helper", "main-agent", PermissionLevel.REQUEST):
                # Được phép gửi request
                await send_request(to="main-agent", message="...")
        """
        current_level = self.get_permission_level(from_agent, to_agent)
        return current_level.has_access(required_level)

    def set_group_permission(self, group_name: str, level: PermissionLevel) -> None:
        """
        Đặt permission level cho một nhóm agents.

        Args:
            group_name: Tên nhóm (ví dụ: "admins", "workers")
            level: Permission level cho nhóm
        """
        self._group_permissions[group_name] = level

    def get_all_permissions(self, agent_id: Optional[str] = None) -> List[A2APermission]:
        """
        Lấy tất cả permissions, có thể lọc theo agent.

        Args:
            agent_id: Lọc permissions liên quan đến agent này.
                     None để lấy tất cả.

        Returns:
            Danh sách A2APermission
        """
        if agent_id is None:
            return list(self._permissions.values())

        result = []
        for key, permission in self._permissions.items():
            if key[0] == agent_id or key[1] == agent_id:
                result.append(permission)
        return result

    def cleanup_expired(self) -> int:
        """
        Xóa các permissions đã hết hạn.

        Returns:
            Số lượng permissions đã xóa
        """
        expired_keys = [
            key for key, perm in self._permissions.items()
            if perm.is_expired()
        ]
        for key in expired_keys:
            del self._permissions[key]
        return len(expired_keys)

    def get_agents_with_access(
        self,
        to_agent: str,
        minimum_level: PermissionLevel = PermissionLevel.NOTIFY,
    ) -> Set[str]:
        """
        Lấy danh sách agents có quyền truy cập đến một agent.

        Args:
            to_agent: ID agent đích
            minimum_level: Mức quyền tối thiểu

        Returns:
            Set các agent_id có đủ quyền
        """
        result = set()
        for (from_agent, target), permission in self._permissions.items():
            if target == to_agent and permission.is_valid():
                if permission.level.has_access(minimum_level):
                    result.add(from_agent)
        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize PermissionManager thành dictionary.

        Returns:
            Dictionary chứa state của manager
        """
        return {
            "default_level": self.default_level.value,
            "permissions": [
                perm.to_dict() for perm in self._permissions.values()
            ],
            "group_permissions": {
                group: level.value
                for group, level in self._group_permissions.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PermissionManager":
        """
        Khôi phục PermissionManager từ dictionary.

        Args:
            data: Dictionary chứa state của manager

        Returns:
            PermissionManager instance
        """
        default_level = PermissionLevel.from_string(
            data.get("default_level", "none")
        )
        manager = cls(default_level=default_level)

        # Khôi phục permissions
        for perm_data in data.get("permissions", []):
            permission = A2APermission.from_dict(perm_data)
            manager._permissions[(permission.from_agent, permission.to_agent)] = permission

        # Khôi phục group permissions
        for group, level_str in data.get("group_permissions", {}).items():
            manager._group_permissions[group] = PermissionLevel.from_string(level_str)

        return manager
