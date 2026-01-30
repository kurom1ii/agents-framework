"""
Audit Logging cho A2A Communications.

Module này cung cấp hệ thống audit logging để theo dõi và ghi nhận
tất cả hoạt động A2A trong hệ thống, bao gồm messages, spawn events,
và history access.

Features:
- Log tất cả A2A communications
- Track who accessed what
- Alert on suspicious patterns
- Retention policies
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol


class AuditEventType(Enum):
    """
    Các loại audit events trong hệ thống A2A.

    - MESSAGE_SENT: Agent gửi message
    - MESSAGE_RECEIVED: Agent nhận message
    - SPAWN_REQUEST: Yêu cầu spawn sub-agent
    - SPAWN_CREATED: Sub-agent được tạo thành công
    - SPAWN_DENIED: Yêu cầu spawn bị từ chối
    - HISTORY_ACCESS: Truy cập lịch sử session
    - PERMISSION_GRANTED: Cấp permission mới
    - PERMISSION_REVOKED: Thu hồi permission
    - PERMISSION_DENIED: Yêu cầu bị từ chối do thiếu quyền
    - TOOL_EXECUTED: Tool được thực thi (trong sandbox)
    - SECURITY_VIOLATION: Vi phạm security policy
    """

    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    SPAWN_REQUEST = "spawn_request"
    SPAWN_CREATED = "spawn_created"
    SPAWN_DENIED = "spawn_denied"
    HISTORY_ACCESS = "history_access"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    PERMISSION_DENIED = "permission_denied"
    TOOL_EXECUTED = "tool_executed"
    SECURITY_VIOLATION = "security_violation"


class AuditSeverity(Enum):
    """
    Mức độ nghiêm trọng của audit events.

    - INFO: Thông tin thường
    - WARNING: Cảnh báo, cần chú ý
    - ERROR: Lỗi hoặc vi phạm
    - CRITICAL: Vi phạm nghiêm trọng, cần xử lý ngay
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEntry:
    """
    Một entry trong audit log.

    AuditEntry chứa tất cả thông tin về một sự kiện trong hệ thống A2A,
    bao gồm who, what, when, và kết quả của sự kiện.

    Attributes:
        id: ID duy nhất của entry
        timestamp: Thời điểm xảy ra sự kiện
        event_type: Loại sự kiện
        severity: Mức độ nghiêm trọng
        agent_id: ID của agent liên quan chính
        target_agent_id: ID của agent đích (nếu có)
        session_key: Key của session liên quan (nếu có)
        action: Mô tả hành động
        result: Kết quả (success, denied, error)
        details: Chi tiết bổ sung
        metadata: Dữ liệu tùy chỉnh

    Example:
        entry = AuditEntry(
            id="audit-001",
            timestamp=datetime.now(),
            event_type=AuditEventType.MESSAGE_SENT,
            severity=AuditSeverity.INFO,
            agent_id="helper",
            target_agent_id="main-agent",
            action="Send request message",
            result="success",
            details={"message_type": "request", "topic": "code_review"}
        )
    """

    id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity
    agent_id: str
    target_agent_id: Optional[str] = None
    session_key: Optional[str] = None
    action: str = ""
    result: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize AuditEntry thành dictionary.

        Returns:
            Dictionary chứa thông tin entry
        """
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "agent_id": self.agent_id,
            "target_agent_id": self.target_agent_id,
            "session_key": self.session_key,
            "action": self.action,
            "result": self.result,
            "details": self.details,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEntry":
        """
        Tạo AuditEntry từ dictionary.

        Args:
            data: Dictionary chứa thông tin entry

        Returns:
            AuditEntry instance
        """
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_type=AuditEventType(data["event_type"]),
            severity=AuditSeverity(data["severity"]),
            agent_id=data["agent_id"],
            target_agent_id=data.get("target_agent_id"),
            session_key=data.get("session_key"),
            action=data.get("action", ""),
            result=data.get("result", ""),
            details=data.get("details", {}),
            metadata=data.get("metadata", {}),
        )


class AuditStorageProtocol(Protocol):
    """
    Protocol cho audit storage backend.

    Định nghĩa interface mà các storage backends cần implement
    để lưu trữ audit logs.
    """

    async def store(self, entry: AuditEntry) -> None:
        """Lưu một audit entry."""
        ...

    async def query(
        self,
        agent_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """Truy vấn audit entries."""
        ...

    async def delete_before(self, before: datetime) -> int:
        """Xóa entries trước một thời điểm."""
        ...


# Type alias cho alert handlers
AlertHandler = Callable[[AuditEntry], None]


class A2AAuditLog:
    """
    Hệ thống Audit Logging cho A2A Communications.

    A2AAuditLog theo dõi và ghi nhận tất cả hoạt động A2A trong hệ thống,
    hỗ trợ querying, alerting, và retention policies.

    Attributes:
        storage: Backend lưu trữ audit logs
        alert_handlers: Danh sách handlers cho các alerts
        retention_days: Số ngày giữ logs (0 = vĩnh viễn)

    Example:
        audit_log = A2AAuditLog()

        # Log một message event
        await audit_log.log_message(
            from_agent="helper",
            to_agent="main-agent",
            message_type="request",
            success=True
        )

        # Query audit trail
        entries = await audit_log.get_audit_trail(
            agent="helper",
            since=datetime.now() - timedelta(hours=24)
        )

        # Đăng ký alert handler
        audit_log.add_alert_handler(
            lambda entry: print(f"Alert: {entry.action}")
        )
    """

    def __init__(
        self,
        storage: Optional[AuditStorageProtocol] = None,
        retention_days: int = 30,
    ) -> None:
        """
        Khởi tạo A2AAuditLog.

        Args:
            storage: Backend lưu trữ. Nếu None, sử dụng in-memory storage.
            retention_days: Số ngày giữ logs (0 = vĩnh viễn)
        """
        self._storage = storage
        self._retention_days = retention_days
        self._alert_handlers: List[AlertHandler] = []
        self._entries: List[AuditEntry] = []  # In-memory storage fallback
        self._entry_counter: int = 0

    def _generate_id(self) -> str:
        """Tạo ID duy nhất cho audit entry."""
        self._entry_counter += 1
        return f"audit-{self._entry_counter:08d}"

    async def _store_entry(self, entry: AuditEntry) -> None:
        """
        Lưu entry vào storage.

        Args:
            entry: AuditEntry cần lưu
        """
        if self._storage:
            await self._storage.store(entry)
        else:
            self._entries.append(entry)

    def _trigger_alerts(self, entry: AuditEntry) -> None:
        """
        Trigger alerts cho entry nếu cần.

        Args:
            entry: AuditEntry để kiểm tra
        """
        # Chỉ trigger alert cho WARNING trở lên
        if entry.severity in (AuditSeverity.WARNING, AuditSeverity.ERROR, AuditSeverity.CRITICAL):
            for handler in self._alert_handlers:
                try:
                    handler(entry)
                except Exception:
                    # Ignore alert handler errors
                    pass

    async def log_message(
        self,
        from_agent: str,
        to_agent: str,
        message_type: str = "request",
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log một A2A message event.

        Args:
            from_agent: ID agent gửi
            to_agent: ID agent nhận
            message_type: Loại message (request, response, notification)
            success: Message có được gửi thành công không
            details: Chi tiết bổ sung

        Example:
            await audit_log.log_message(
                from_agent="helper",
                to_agent="main-agent",
                message_type="request",
                success=True,
                details={"topic": "code_review"}
            )
        """
        entry = AuditEntry(
            id=self._generate_id(),
            timestamp=datetime.now(),
            event_type=AuditEventType.MESSAGE_SENT,
            severity=AuditSeverity.INFO if success else AuditSeverity.WARNING,
            agent_id=from_agent,
            target_agent_id=to_agent,
            action=f"Send {message_type} message to {to_agent}",
            result="success" if success else "failed",
            details=details or {},
        )
        await self._store_entry(entry)
        self._trigger_alerts(entry)

    async def log_spawn(
        self,
        parent_agent: str,
        child_agent_id: str,
        success: bool = True,
        denied_reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log một spawn event.

        Args:
            parent_agent: ID agent cha
            child_agent_id: ID của child agent được spawn
            success: Spawn có thành công không
            denied_reason: Lý do bị từ chối (nếu không thành công)
            details: Chi tiết bổ sung

        Example:
            await audit_log.log_spawn(
                parent_agent="main-agent",
                child_agent_id="sub-agent-1",
                success=True,
                details={"task": "research"}
            )
        """
        event_type = AuditEventType.SPAWN_CREATED if success else AuditEventType.SPAWN_DENIED
        severity = AuditSeverity.INFO if success else AuditSeverity.WARNING

        entry = AuditEntry(
            id=self._generate_id(),
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            agent_id=parent_agent,
            target_agent_id=child_agent_id,
            action=f"Spawn child agent {child_agent_id}",
            result="success" if success else f"denied: {denied_reason}",
            details=details or {},
        )
        await self._store_entry(entry)
        self._trigger_alerts(entry)

    async def log_history_access(
        self,
        agent: str,
        target_session: str,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log một history access event.

        Args:
            agent: ID agent truy cập
            target_session: Session key được truy cập
            success: Truy cập có thành công không
            details: Chi tiết bổ sung

        Example:
            await audit_log.log_history_access(
                agent="helper",
                target_session="agent:main:session-1",
                success=True
            )
        """
        entry = AuditEntry(
            id=self._generate_id(),
            timestamp=datetime.now(),
            event_type=AuditEventType.HISTORY_ACCESS,
            severity=AuditSeverity.INFO if success else AuditSeverity.WARNING,
            agent_id=agent,
            session_key=target_session,
            action=f"Access history of session {target_session}",
            result="success" if success else "denied",
            details=details or {},
        )
        await self._store_entry(entry)
        self._trigger_alerts(entry)

    async def log_permission_change(
        self,
        from_agent: str,
        to_agent: str,
        action: str,
        new_level: Optional[str] = None,
        changed_by: Optional[str] = None,
    ) -> None:
        """
        Log thay đổi permission.

        Args:
            from_agent: Agent nguồn
            to_agent: Agent đích
            action: granted, revoked, denied
            new_level: Permission level mới (nếu granted)
            changed_by: Agent/system thực hiện thay đổi
        """
        event_type = {
            "granted": AuditEventType.PERMISSION_GRANTED,
            "revoked": AuditEventType.PERMISSION_REVOKED,
            "denied": AuditEventType.PERMISSION_DENIED,
        }.get(action, AuditEventType.PERMISSION_DENIED)

        severity = AuditSeverity.WARNING if action == "denied" else AuditSeverity.INFO

        entry = AuditEntry(
            id=self._generate_id(),
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            agent_id=from_agent,
            target_agent_id=to_agent,
            action=f"Permission {action}: {from_agent} -> {to_agent}",
            result=new_level or action,
            details={"changed_by": changed_by} if changed_by else {},
        )
        await self._store_entry(entry)
        self._trigger_alerts(entry)

    async def log_security_violation(
        self,
        agent: str,
        violation_type: str,
        details: Dict[str, Any],
        target: Optional[str] = None,
    ) -> None:
        """
        Log một security violation.

        Args:
            agent: Agent vi phạm
            violation_type: Loại vi phạm
            details: Chi tiết về vi phạm
            target: Đối tượng bị vi phạm (nếu có)

        Example:
            await audit_log.log_security_violation(
                agent="malicious-agent",
                violation_type="unauthorized_tool_access",
                details={"tool": "bash", "sandbox": "restrictive"},
                target="main-agent"
            )
        """
        entry = AuditEntry(
            id=self._generate_id(),
            timestamp=datetime.now(),
            event_type=AuditEventType.SECURITY_VIOLATION,
            severity=AuditSeverity.CRITICAL,
            agent_id=agent,
            target_agent_id=target,
            action=f"Security violation: {violation_type}",
            result="violation",
            details=details,
        )
        await self._store_entry(entry)
        self._trigger_alerts(entry)

    async def log_tool_execution(
        self,
        agent: str,
        tool_name: str,
        success: bool = True,
        sandbox_enforced: bool = False,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log tool execution trong sandbox.

        Args:
            agent: Agent thực thi tool
            tool_name: Tên tool
            success: Thực thi có thành công không
            sandbox_enforced: Có đang chạy trong sandbox không
            details: Chi tiết bổ sung
        """
        entry = AuditEntry(
            id=self._generate_id(),
            timestamp=datetime.now(),
            event_type=AuditEventType.TOOL_EXECUTED,
            severity=AuditSeverity.INFO,
            agent_id=agent,
            action=f"Execute tool: {tool_name}",
            result="success" if success else "failed",
            details={
                "tool_name": tool_name,
                "sandbox_enforced": sandbox_enforced,
                **(details or {}),
            },
        )
        await self._store_entry(entry)

    async def get_audit_trail(
        self,
        agent: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """
        Lấy audit trail theo các điều kiện.

        Args:
            agent: Lọc theo agent ID
            since: Lấy entries từ thời điểm này
            until: Lấy entries đến thời điểm này
            event_types: Lọc theo loại events
            limit: Số lượng tối đa trả về

        Returns:
            Danh sách AuditEntry thỏa mãn điều kiện

        Example:
            # Lấy tất cả events của helper trong 24h qua
            entries = await audit_log.get_audit_trail(
                agent="helper",
                since=datetime.now() - timedelta(hours=24)
            )

            # Lấy chỉ security violations
            violations = await audit_log.get_audit_trail(
                event_types=[AuditEventType.SECURITY_VIOLATION]
            )
        """
        if self._storage:
            # Sử dụng storage backend
            entries = await self._storage.query(
                agent_id=agent,
                since=since,
                until=until,
                limit=limit,
            )
        else:
            # In-memory filtering
            entries = self._entries.copy()

        # Áp dụng filters
        filtered = []
        for entry in entries:
            # Filter by agent
            if agent and entry.agent_id != agent and entry.target_agent_id != agent:
                continue

            # Filter by time
            if since and entry.timestamp < since:
                continue
            if until and entry.timestamp > until:
                continue

            # Filter by event types
            if event_types and entry.event_type not in event_types:
                continue

            filtered.append(entry)

            if len(filtered) >= limit:
                break

        # Sắp xếp theo thời gian mới nhất
        return sorted(filtered, key=lambda e: e.timestamp, reverse=True)

    async def get_agent_activity(
        self,
        agent: str,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Lấy tóm tắt hoạt động của một agent.

        Args:
            agent: ID agent
            hours: Số giờ gần đây để tính

        Returns:
            Dictionary chứa thống kê hoạt động

        Example:
            activity = await audit_log.get_agent_activity("helper", hours=24)
            # {
            #     "messages_sent": 15,
            #     "messages_received": 10,
            #     "spawns": 2,
            #     "violations": 0,
            #     ...
            # }
        """
        since = datetime.now() - timedelta(hours=hours)
        entries = await self.get_audit_trail(agent=agent, since=since, limit=1000)

        stats = {
            "agent": agent,
            "period_hours": hours,
            "total_events": len(entries),
            "messages_sent": 0,
            "messages_received": 0,
            "spawns": 0,
            "history_accesses": 0,
            "permission_changes": 0,
            "violations": 0,
            "tools_executed": 0,
        }

        for entry in entries:
            if entry.event_type == AuditEventType.MESSAGE_SENT:
                if entry.agent_id == agent:
                    stats["messages_sent"] += 1
                else:
                    stats["messages_received"] += 1
            elif entry.event_type in (AuditEventType.SPAWN_CREATED, AuditEventType.SPAWN_DENIED):
                stats["spawns"] += 1
            elif entry.event_type == AuditEventType.HISTORY_ACCESS:
                stats["history_accesses"] += 1
            elif entry.event_type in (AuditEventType.PERMISSION_GRANTED, AuditEventType.PERMISSION_REVOKED):
                stats["permission_changes"] += 1
            elif entry.event_type == AuditEventType.SECURITY_VIOLATION:
                stats["violations"] += 1
            elif entry.event_type == AuditEventType.TOOL_EXECUTED:
                stats["tools_executed"] += 1

        return stats

    def add_alert_handler(self, handler: AlertHandler) -> None:
        """
        Đăng ký handler để nhận alerts.

        Args:
            handler: Callback function nhận AuditEntry

        Example:
            def on_alert(entry: AuditEntry):
                if entry.severity == AuditSeverity.CRITICAL:
                    send_notification(entry)

            audit_log.add_alert_handler(on_alert)
        """
        self._alert_handlers.append(handler)

    def remove_alert_handler(self, handler: AlertHandler) -> None:
        """
        Gỡ bỏ một alert handler.

        Args:
            handler: Handler cần gỡ
        """
        if handler in self._alert_handlers:
            self._alert_handlers.remove(handler)

    async def cleanup_old_entries(self) -> int:
        """
        Xóa các entries cũ theo retention policy.

        Returns:
            Số entries đã xóa
        """
        if self._retention_days <= 0:
            return 0

        cutoff = datetime.now() - timedelta(days=self._retention_days)

        if self._storage:
            return await self._storage.delete_before(cutoff)
        else:
            original_count = len(self._entries)
            self._entries = [e for e in self._entries if e.timestamp >= cutoff]
            return original_count - len(self._entries)

    def get_entry_count(self) -> int:
        """
        Lấy tổng số entries hiện tại.

        Returns:
            Số lượng entries
        """
        return len(self._entries)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize audit log state.

        Returns:
            Dictionary chứa state
        """
        return {
            "retention_days": self._retention_days,
            "entry_count": len(self._entries),
            "entries": [e.to_dict() for e in self._entries[-100:]],  # Last 100
        }
