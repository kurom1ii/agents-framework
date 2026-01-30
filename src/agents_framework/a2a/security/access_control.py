"""
A2A Access Control System.

Module này cung cấp hệ thống kiểm soát quyền truy cập cho giao tiếp A2A,
bao gồm allow/deny lists, permission checking, và security configuration.

Features:
- Định nghĩa ai có thể gửi message đến ai
- Allowlist và denylist cho mỗi agent
- Role-based access control
- Sandbox configuration cho spawned agents
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .audit import A2AAuditLog
from .permissions import A2APermission, PermissionLevel, PermissionManager
from .sandbox import SpawnSandbox, SpawnSandboxConfig


@dataclass
class A2ASecurityConfig:
    """
    Cấu hình bảo mật cho một agent trong hệ thống A2A.

    A2ASecurityConfig định nghĩa các quy tắc truy cập cho một agent cụ thể,
    bao gồm ai được phép gửi message đến agent, agent được gửi đến ai,
    và các quyền đọc history.

    Attributes:
        allow_incoming: Danh sách agent IDs được phép gửi message đến agent này.
                       Sử dụng ["*"] để cho phép tất cả.
        allow_outgoing: Danh sách agent IDs mà agent này được phép gửi đến.
                       Sử dụng ["*"] để cho phép tất cả.
        deny_incoming: Danh sách agent IDs bị cấm gửi message đến agent này.
                      Có ưu tiên cao hơn allow_incoming.
        deny_outgoing: Danh sách agent IDs mà agent này bị cấm gửi đến.
                      Có ưu tiên cao hơn allow_outgoing.
        history_access: Dict mapping agent_id -> PermissionLevel cho việc đọc history.
        spawn_sandbox: Cấu hình sandbox khi agent spawn sub-agents.
        require_authentication: Yêu cầu authentication cho incoming messages.
        max_requests_per_minute: Giới hạn số requests từ mỗi agent mỗi phút.
        metadata: Dữ liệu bổ sung.

    Example:
        config = A2ASecurityConfig(
            allow_incoming=["helper", "researcher"],
            allow_outgoing=["*"],
            deny_incoming=["untrusted-agent"],
            history_access={
                "helper": PermissionLevel.HISTORY,
                "researcher": PermissionLevel.FULL,
                "*": PermissionLevel.NONE
            },
            spawn_sandbox=SpawnSandboxConfig(
                mode="always",
                allowed_tools=["read", "write", "web_search"],
                max_spawn_depth=2
            )
        )
    """

    allow_incoming: List[str] = field(default_factory=lambda: ["*"])
    allow_outgoing: List[str] = field(default_factory=lambda: ["*"])
    deny_incoming: List[str] = field(default_factory=list)
    deny_outgoing: List[str] = field(default_factory=list)
    history_access: Dict[str, PermissionLevel] = field(default_factory=dict)
    spawn_sandbox: SpawnSandboxConfig = field(default_factory=SpawnSandboxConfig)
    require_authentication: bool = False
    max_requests_per_minute: int = 0  # 0 = không giới hạn
    metadata: Dict[str, Any] = field(default_factory=dict)

    def can_receive_from(self, from_agent: str) -> bool:
        """
        Kiểm tra agent có được phép nhận message từ from_agent không.

        Args:
            from_agent: ID của agent gửi

        Returns:
            True nếu được phép nhận

        Example:
            if config.can_receive_from("helper"):
                # Xử lý message
                pass
        """
        # Kiểm tra deny list trước (ưu tiên cao hơn)
        if from_agent in self.deny_incoming:
            return False

        # Kiểm tra allow list
        if "*" in self.allow_incoming:
            return True

        return from_agent in self.allow_incoming

    def can_send_to(self, to_agent: str) -> bool:
        """
        Kiểm tra agent có được phép gửi message đến to_agent không.

        Args:
            to_agent: ID của agent nhận

        Returns:
            True nếu được phép gửi

        Example:
            if config.can_send_to("main-agent"):
                # Gửi message
                pass
        """
        # Kiểm tra deny list trước
        if to_agent in self.deny_outgoing:
            return False

        # Kiểm tra allow list
        if "*" in self.allow_outgoing:
            return True

        return to_agent in self.allow_outgoing

    def get_history_access(self, agent_id: str) -> PermissionLevel:
        """
        Lấy permission level cho việc đọc history của agent_id.

        Args:
            agent_id: ID của agent muốn đọc history

        Returns:
            PermissionLevel cho agent đó

        Example:
            level = config.get_history_access("helper")
            if level.can_read_history:
                # Cho phép đọc history
                pass
        """
        # Kiểm tra permission cụ thể
        if agent_id in self.history_access:
            return self.history_access[agent_id]

        # Kiểm tra wildcard
        if "*" in self.history_access:
            return self.history_access["*"]

        # Mặc định không có quyền
        return PermissionLevel.NONE

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize A2ASecurityConfig thành dictionary.

        Returns:
            Dictionary chứa cấu hình
        """
        return {
            "allow_incoming": self.allow_incoming,
            "allow_outgoing": self.allow_outgoing,
            "deny_incoming": self.deny_incoming,
            "deny_outgoing": self.deny_outgoing,
            "history_access": {
                k: v.value for k, v in self.history_access.items()
            },
            "spawn_sandbox": self.spawn_sandbox.to_dict(),
            "require_authentication": self.require_authentication,
            "max_requests_per_minute": self.max_requests_per_minute,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2ASecurityConfig":
        """
        Tạo A2ASecurityConfig từ dictionary.

        Args:
            data: Dictionary chứa cấu hình

        Returns:
            A2ASecurityConfig instance
        """
        history_access = {}
        for agent_id, level_str in data.get("history_access", {}).items():
            if isinstance(level_str, str):
                history_access[agent_id] = PermissionLevel.from_string(level_str)
            else:
                history_access[agent_id] = level_str

        spawn_sandbox_data = data.get("spawn_sandbox", {})
        spawn_sandbox = (
            SpawnSandboxConfig.from_dict(spawn_sandbox_data)
            if spawn_sandbox_data
            else SpawnSandboxConfig()
        )

        return cls(
            allow_incoming=data.get("allow_incoming", ["*"]),
            allow_outgoing=data.get("allow_outgoing", ["*"]),
            deny_incoming=data.get("deny_incoming", []),
            deny_outgoing=data.get("deny_outgoing", []),
            history_access=history_access,
            spawn_sandbox=spawn_sandbox,
            require_authentication=data.get("require_authentication", False),
            max_requests_per_minute=data.get("max_requests_per_minute", 0),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def open_policy(cls) -> "A2ASecurityConfig":
        """
        Tạo cấu hình mở (cho phép tất cả).

        Returns:
            A2ASecurityConfig với tất cả quyền

        Example:
            config = A2ASecurityConfig.open_policy()
        """
        return cls(
            allow_incoming=["*"],
            allow_outgoing=["*"],
            history_access={"*": PermissionLevel.HISTORY},
            spawn_sandbox=SpawnSandboxConfig.permissive(),
        )

    @classmethod
    def closed_policy(cls) -> "A2ASecurityConfig":
        """
        Tạo cấu hình đóng (từ chối tất cả).

        Returns:
            A2ASecurityConfig không có quyền

        Example:
            config = A2ASecurityConfig.closed_policy()
        """
        return cls(
            allow_incoming=[],
            allow_outgoing=[],
            history_access={"*": PermissionLevel.NONE},
            spawn_sandbox=SpawnSandboxConfig.restrictive(),
        )


class A2AAccessControl:
    """
    Hệ thống kiểm soát quyền truy cập cho A2A communications.

    A2AAccessControl quản lý và enforce các quy tắc bảo mật cho việc
    giao tiếp giữa các agents, bao gồm message sending, history access,
    và spawning.

    Attributes:
        configs: Dictionary mapping agent_id -> A2ASecurityConfig
        permission_manager: PermissionManager để quản lý permissions động
        audit_log: A2AAuditLog để ghi nhận các access attempts
        default_config: Cấu hình mặc định cho agents chưa được cấu hình

    Example:
        # Khởi tạo với cấu hình
        access_control = A2AAccessControl(
            configs={
                "main-agent": A2ASecurityConfig(
                    allow_incoming=["helper", "researcher"],
                    allow_outgoing=["*"]
                ),
                "helper": A2ASecurityConfig(
                    allow_incoming=["main-agent"],
                    allow_outgoing=["main-agent"]
                )
            }
        )

        # Kiểm tra quyền gửi message
        if access_control.can_send("helper", "main-agent"):
            # Cho phép gửi
            pass

        # Kiểm tra quyền đọc history
        level = access_control.can_read_history("helper", "session-123")
    """

    def __init__(
        self,
        configs: Optional[Dict[str, A2ASecurityConfig]] = None,
        permission_manager: Optional[PermissionManager] = None,
        audit_log: Optional[A2AAuditLog] = None,
        default_config: Optional[A2ASecurityConfig] = None,
    ) -> None:
        """
        Khởi tạo A2AAccessControl.

        Args:
            configs: Dictionary mapping agent_id -> A2ASecurityConfig
            permission_manager: PermissionManager để quản lý permissions
            audit_log: A2AAuditLog để ghi logs
            default_config: Cấu hình mặc định cho agents chưa được cấu hình
        """
        self._configs: Dict[str, A2ASecurityConfig] = configs or {}
        self._permission_manager = permission_manager or PermissionManager()
        self._audit_log = audit_log
        self._default_config = default_config or A2ASecurityConfig()
        self._session_agent_map: Dict[str, str] = {}  # session_key -> agent_id

    def get_config(self, agent_id: str) -> A2ASecurityConfig:
        """
        Lấy cấu hình bảo mật của một agent.

        Args:
            agent_id: ID của agent

        Returns:
            A2ASecurityConfig của agent, hoặc default config nếu không tìm thấy
        """
        return self._configs.get(agent_id, self._default_config)

    def set_config(self, agent_id: str, config: A2ASecurityConfig) -> None:
        """
        Đặt cấu hình bảo mật cho một agent.

        Args:
            agent_id: ID của agent
            config: A2ASecurityConfig mới
        """
        self._configs[agent_id] = config

    def register_session(self, session_key: str, agent_id: str) -> None:
        """
        Đăng ký mapping session_key -> agent_id.

        Args:
            session_key: Key của session
            agent_id: ID của agent sở hữu session
        """
        self._session_agent_map[session_key] = agent_id

    def can_send(self, from_agent: str, to_agent: str) -> bool:
        """
        Kiểm tra agent có thể gửi message đến agent khác không.

        Phương thức này kiểm tra cả hai phía:
        1. from_agent có được phép gửi đến to_agent không
        2. to_agent có cho phép nhận từ from_agent không

        Args:
            from_agent: ID agent gửi
            to_agent: ID agent nhận

        Returns:
            True nếu được phép gửi

        Example:
            if access_control.can_send("helper", "main-agent"):
                await send_message(to="main-agent", message="...")
        """
        # Kiểm tra cấu hình của from_agent
        from_config = self.get_config(from_agent)
        if not from_config.can_send_to(to_agent):
            return False

        # Kiểm tra cấu hình của to_agent
        to_config = self.get_config(to_agent)
        if not to_config.can_receive_from(from_agent):
            return False

        # Kiểm tra dynamic permissions
        permission_level = self._permission_manager.get_permission_level(from_agent, to_agent)
        if not permission_level.can_notify:
            return False

        return True

    def can_request(self, from_agent: str, to_agent: str) -> bool:
        """
        Kiểm tra agent có thể gửi request (và nhận response) không.

        Args:
            from_agent: ID agent gửi request
            to_agent: ID agent nhận request

        Returns:
            True nếu được phép gửi request

        Example:
            if access_control.can_request("helper", "main-agent"):
                response = await send_request(to="main-agent", request="...")
        """
        if not self.can_send(from_agent, to_agent):
            return False

        # Kiểm tra permission level
        permission_level = self._permission_manager.get_permission_level(from_agent, to_agent)
        return permission_level.can_request

    def can_read_history(
        self,
        agent: str,
        target_session: str,
    ) -> PermissionLevel:
        """
        Kiểm tra quyền đọc history của một session.

        Args:
            agent: ID agent muốn đọc history
            target_session: Session key cần đọc

        Returns:
            PermissionLevel cho việc đọc history

        Example:
            level = access_control.can_read_history("helper", "agent:main:session-1")
            if level.can_read_history:
                history = await get_session_history("agent:main:session-1")
        """
        # Tìm agent sở hữu session
        target_agent = self._session_agent_map.get(target_session)
        if target_agent is None:
            # Extract agent_id từ session_key nếu có thể
            # Format: agent:<agentId>:<scope>:<identifier>
            parts = target_session.split(":")
            if len(parts) >= 2 and parts[0] == "agent":
                target_agent = parts[1]
            else:
                return PermissionLevel.NONE

        # Lấy config của target agent
        config = self.get_config(target_agent)
        return config.get_history_access(agent)

    def can_spawn(self, parent_agent: str) -> bool:
        """
        Kiểm tra agent có được phép spawn sub-agents không.

        Args:
            parent_agent: ID agent muốn spawn

        Returns:
            True nếu được phép spawn

        Example:
            if access_control.can_spawn("main-agent"):
                child = await spawn_agent(parent="main-agent", ...)
        """
        config = self.get_config(parent_agent)
        sandbox_config = config.spawn_sandbox

        # Kiểm tra sandbox mode
        if sandbox_config.mode.value == "never":
            return False

        # Kiểm tra spawn depth
        return sandbox_config.max_spawn_depth > 0

    def get_sandbox_config(self, parent_agent: str) -> SpawnSandboxConfig:
        """
        Lấy cấu hình sandbox cho spawned agents.

        Args:
            parent_agent: ID agent cha

        Returns:
            SpawnSandboxConfig cho sub-agents

        Example:
            sandbox_config = access_control.get_sandbox_config("main-agent")
            sandbox = SpawnSandbox(
                config=sandbox_config,
                parent_agent_id="main-agent",
                spawn_depth=1
            )
        """
        config = self.get_config(parent_agent)
        return config.spawn_sandbox

    def create_sandbox(
        self,
        parent_agent: str,
        spawn_depth: int = 1,
    ) -> Optional[SpawnSandbox]:
        """
        Tạo sandbox cho spawned agent.

        Args:
            parent_agent: ID agent cha
            spawn_depth: Độ sâu spawn hiện tại

        Returns:
            SpawnSandbox nếu được phép spawn, None nếu không

        Example:
            sandbox = access_control.create_sandbox("main-agent", spawn_depth=1)
            if sandbox:
                # spawn child với sandbox
                pass
        """
        if not self.can_spawn(parent_agent):
            return None

        sandbox_config = self.get_sandbox_config(parent_agent)

        if not sandbox_config.can_spawn_more(spawn_depth):
            return None

        return SpawnSandbox(
            config=sandbox_config,
            parent_agent_id=parent_agent,
            spawn_depth=spawn_depth,
        )

    def grant_permission(
        self,
        from_agent: str,
        to_agent: str,
        level: PermissionLevel,
        granted_by: Optional[str] = None,
    ) -> A2APermission:
        """
        Cấp dynamic permission cho một agent.

        Args:
            from_agent: Agent được cấp quyền
            to_agent: Agent đích
            level: Mức độ quyền
            granted_by: Agent/system cấp quyền

        Returns:
            A2APermission đã tạo
        """
        return self._permission_manager.grant_permission(
            from_agent=from_agent,
            to_agent=to_agent,
            level=level,
            granted_by=granted_by,
        )

    def revoke_permission(self, from_agent: str, to_agent: str) -> bool:
        """
        Thu hồi permission giữa hai agents.

        Args:
            from_agent: Agent nguồn
            to_agent: Agent đích

        Returns:
            True nếu đã thu hồi thành công
        """
        return self._permission_manager.revoke_permission(from_agent, to_agent)

    def get_allowed_senders(self, agent_id: str) -> Set[str]:
        """
        Lấy danh sách agents được phép gửi message đến agent.

        Args:
            agent_id: ID của agent đích

        Returns:
            Set các agent_id được phép gửi
        """
        config = self.get_config(agent_id)

        if "*" in config.allow_incoming:
            # Tất cả được phép (trừ deny list)
            return {"*"} - set(config.deny_incoming)

        return set(config.allow_incoming) - set(config.deny_incoming)

    def get_allowed_receivers(self, agent_id: str) -> Set[str]:
        """
        Lấy danh sách agents mà agent được phép gửi đến.

        Args:
            agent_id: ID của agent nguồn

        Returns:
            Set các agent_id được phép nhận
        """
        config = self.get_config(agent_id)

        if "*" in config.allow_outgoing:
            return {"*"} - set(config.deny_outgoing)

        return set(config.allow_outgoing) - set(config.deny_outgoing)

    async def validate_message(
        self,
        from_agent: str,
        to_agent: str,
        message_type: str = "request",
    ) -> tuple[bool, Optional[str]]:
        """
        Validate một message trước khi gửi.

        Args:
            from_agent: Agent gửi
            to_agent: Agent nhận
            message_type: Loại message

        Returns:
            Tuple (allowed, reason) - True nếu được phép, reason nếu bị từ chối

        Example:
            allowed, reason = await access_control.validate_message(
                "helper", "main-agent", "request"
            )
            if not allowed:
                raise PermissionError(reason)
        """
        # Kiểm tra quyền gửi cơ bản
        if not self.can_send(from_agent, to_agent):
            reason = f"Agent '{from_agent}' không được phép gửi message đến '{to_agent}'"
            if self._audit_log:
                await self._audit_log.log_permission_change(
                    from_agent=from_agent,
                    to_agent=to_agent,
                    action="denied",
                )
            return False, reason

        # Kiểm tra quyền request nếu cần
        if message_type == "request" and not self.can_request(from_agent, to_agent):
            reason = f"Agent '{from_agent}' không có quyền REQUEST đến '{to_agent}'"
            return False, reason

        return True, None

    async def validate_history_access(
        self,
        agent: str,
        target_session: str,
    ) -> tuple[bool, Optional[str]]:
        """
        Validate quyền đọc history.

        Args:
            agent: Agent muốn đọc
            target_session: Session cần đọc

        Returns:
            Tuple (allowed, reason)
        """
        level = self.can_read_history(agent, target_session)

        if not level.can_read_history:
            reason = f"Agent '{agent}' không có quyền đọc history của session '{target_session}'"
            if self._audit_log:
                await self._audit_log.log_history_access(
                    agent=agent,
                    target_session=target_session,
                    success=False,
                )
            return False, reason

        if self._audit_log:
            await self._audit_log.log_history_access(
                agent=agent,
                target_session=target_session,
                success=True,
            )

        return True, None

    async def validate_spawn(
        self,
        parent_agent: str,
        spawn_depth: int = 1,
    ) -> tuple[bool, Optional[str], Optional[SpawnSandbox]]:
        """
        Validate quyền spawn và tạo sandbox.

        Args:
            parent_agent: Agent cha
            spawn_depth: Độ sâu spawn

        Returns:
            Tuple (allowed, reason, sandbox)
        """
        sandbox = self.create_sandbox(parent_agent, spawn_depth)

        if sandbox is None:
            reason = f"Agent '{parent_agent}' không được phép spawn sub-agents"
            if self._audit_log:
                await self._audit_log.log_spawn(
                    parent_agent=parent_agent,
                    child_agent_id="(không tạo)",
                    success=False,
                    denied_reason=reason,
                )
            return False, reason, None

        return True, None, sandbox

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize A2AAccessControl thành dictionary.

        Returns:
            Dictionary chứa state
        """
        return {
            "configs": {
                agent_id: config.to_dict()
                for agent_id, config in self._configs.items()
            },
            "default_config": self._default_config.to_dict(),
            "permissions": self._permission_manager.to_dict(),
            "session_agent_map": self._session_agent_map,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AAccessControl":
        """
        Khôi phục A2AAccessControl từ dictionary.

        Args:
            data: Dictionary chứa state

        Returns:
            A2AAccessControl instance
        """
        configs = {
            agent_id: A2ASecurityConfig.from_dict(config_data)
            for agent_id, config_data in data.get("configs", {}).items()
        }

        default_config = A2ASecurityConfig.from_dict(data.get("default_config", {}))

        permission_manager = PermissionManager.from_dict(data.get("permissions", {}))

        access_control = cls(
            configs=configs,
            default_config=default_config,
            permission_manager=permission_manager,
        )
        access_control._session_agent_map = data.get("session_agent_map", {})

        return access_control
