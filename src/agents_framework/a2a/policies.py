"""
A2A Security Policies.

Module này định nghĩa các security policies cho A2A communication,
bao gồm allowlist, denylist, và role-based access control.

Policies cung cấp cách cấu hình bảo mật cao cấp dựa trên roles
và groups thay vì chỉ agent IDs cụ thể.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from .security import (
    A2AAccessControl,
    A2ASecurityConfig,
    PermissionLevel,
    SpawnSandboxConfig,
)


class PolicyAction(Enum):
    """
    Hành động khi policy được trigger.

    - ALLOW: Cho phép hành động
    - DENY: Từ chối hành động
    - REQUIRE_APPROVAL: Yêu cầu approval trước khi thực hiện
    - LOG_ONLY: Chỉ log, không can thiệp
    """

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    LOG_ONLY = "log_only"


class PolicyPriority(Enum):
    """
    Mức độ ưu tiên của policy.

    Policies với priority cao hơn sẽ được xét trước.
    """

    CRITICAL = 100
    HIGH = 75
    NORMAL = 50
    LOW = 25
    DEFAULT = 0


@dataclass
class PolicyRule:
    """
    Một rule trong security policy.

    PolicyRule định nghĩa điều kiện và hành động cho một tình huống
    cụ thể trong A2A communication.

    Attributes:
        name: Tên của rule
        description: Mô tả rule
        condition: Function kiểm tra điều kiện (trả về True nếu rule applies)
        action: Hành động khi rule applies
        priority: Mức độ ưu tiên
        enabled: Rule có đang active không
        metadata: Dữ liệu bổ sung

    Example:
        rule = PolicyRule(
            name="deny_untrusted",
            description="Từ chối messages từ untrusted agents",
            condition=lambda ctx: ctx.get("from_agent") in untrusted_agents,
            action=PolicyAction.DENY,
            priority=PolicyPriority.HIGH
        )
    """

    name: str
    description: str = ""
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    action: PolicyAction = PolicyAction.DENY
    priority: PolicyPriority = PolicyPriority.NORMAL
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        Đánh giá rule với context cho trước.

        Args:
            context: Dictionary chứa thông tin context

        Returns:
            True nếu rule applies
        """
        if not self.enabled:
            return False

        if self.condition is None:
            return True

        return self.condition(context)


@dataclass
class RolePermission:
    """
    Permission cho một role cụ thể.

    Attributes:
        role: Tên role
        permission_level: Permission level cho role này
        can_send_to_roles: Danh sách roles mà role này có thể gửi message đến
        can_receive_from_roles: Danh sách roles có thể gửi message đến role này
        spawn_allowed: Có được phép spawn sub-agents không
        max_spawn_depth: Độ sâu spawn tối đa (nếu được phép)
    """

    role: str
    permission_level: PermissionLevel = PermissionLevel.REQUEST
    can_send_to_roles: List[str] = field(default_factory=lambda: ["*"])
    can_receive_from_roles: List[str] = field(default_factory=lambda: ["*"])
    spawn_allowed: bool = False
    max_spawn_depth: int = 0


class SecurityPolicy:
    """
    Security policy tổng thể cho A2A communication.

    SecurityPolicy quản lý tập hợp các rules và role-based permissions
    để kiểm soát giao tiếp giữa các agents.

    Attributes:
        name: Tên của policy
        rules: Danh sách các PolicyRules
        role_permissions: Dict mapping role -> RolePermission
        default_action: Hành động mặc định khi không có rule nào applies

    Example:
        policy = SecurityPolicy(name="production")

        # Thêm rules
        policy.add_rule(PolicyRule(
            name="deny_external",
            condition=lambda ctx: ctx.get("is_external", False),
            action=PolicyAction.DENY
        ))

        # Cấu hình role permissions
        policy.set_role_permission(RolePermission(
            role="admin",
            permission_level=PermissionLevel.FULL,
            spawn_allowed=True,
            max_spawn_depth=3
        ))

        # Evaluate
        result = policy.evaluate({
            "from_agent": "helper",
            "to_agent": "main",
            "is_external": False
        })
    """

    def __init__(
        self,
        name: str = "default",
        default_action: PolicyAction = PolicyAction.ALLOW,
    ) -> None:
        """
        Khởi tạo SecurityPolicy.

        Args:
            name: Tên của policy
            default_action: Hành động mặc định
        """
        self.name = name
        self.default_action = default_action
        self._rules: List[PolicyRule] = []
        self._role_permissions: Dict[str, RolePermission] = {}
        self._agent_roles: Dict[str, str] = {}  # agent_id -> role
        self._allowlist: Set[str] = set()
        self._denylist: Set[str] = set()

    def add_rule(self, rule: PolicyRule) -> None:
        """
        Thêm một rule vào policy.

        Args:
            rule: PolicyRule cần thêm
        """
        self._rules.append(rule)
        # Sắp xếp theo priority (cao trước)
        self._rules.sort(key=lambda r: r.priority.value, reverse=True)

    def remove_rule(self, rule_name: str) -> bool:
        """
        Xóa một rule theo tên.

        Args:
            rule_name: Tên rule cần xóa

        Returns:
            True nếu rule đã được xóa
        """
        for i, rule in enumerate(self._rules):
            if rule.name == rule_name:
                del self._rules[i]
                return True
        return False

    def get_rule(self, rule_name: str) -> Optional[PolicyRule]:
        """
        Lấy rule theo tên.

        Args:
            rule_name: Tên rule

        Returns:
            PolicyRule nếu tìm thấy, None nếu không
        """
        for rule in self._rules:
            if rule.name == rule_name:
                return rule
        return None

    def set_role_permission(self, permission: RolePermission) -> None:
        """
        Đặt permission cho một role.

        Args:
            permission: RolePermission cho role
        """
        self._role_permissions[permission.role] = permission

    def get_role_permission(self, role: str) -> Optional[RolePermission]:
        """
        Lấy permission của một role.

        Args:
            role: Tên role

        Returns:
            RolePermission nếu tìm thấy, None nếu không
        """
        return self._role_permissions.get(role)

    def assign_role(self, agent_id: str, role: str) -> None:
        """
        Gán role cho một agent.

        Args:
            agent_id: ID của agent
            role: Role cần gán
        """
        self._agent_roles[agent_id] = role

    def get_agent_role(self, agent_id: str) -> Optional[str]:
        """
        Lấy role của một agent.

        Args:
            agent_id: ID của agent

        Returns:
            Role của agent, None nếu chưa được gán
        """
        return self._agent_roles.get(agent_id)

    def add_to_allowlist(self, agent_id: str) -> None:
        """
        Thêm agent vào allowlist.

        Args:
            agent_id: ID agent cần thêm
        """
        self._allowlist.add(agent_id)
        # Xóa khỏi denylist nếu có
        self._denylist.discard(agent_id)

    def add_to_denylist(self, agent_id: str) -> None:
        """
        Thêm agent vào denylist.

        Args:
            agent_id: ID agent cần thêm
        """
        self._denylist.add(agent_id)
        # Xóa khỏi allowlist nếu có
        self._allowlist.discard(agent_id)

    def remove_from_allowlist(self, agent_id: str) -> None:
        """Xóa agent khỏi allowlist."""
        self._allowlist.discard(agent_id)

    def remove_from_denylist(self, agent_id: str) -> None:
        """Xóa agent khỏi denylist."""
        self._denylist.discard(agent_id)

    def is_allowed(self, agent_id: str) -> bool:
        """
        Kiểm tra agent có trong allowlist không.

        Args:
            agent_id: ID agent

        Returns:
            True nếu trong allowlist
        """
        return agent_id in self._allowlist

    def is_denied(self, agent_id: str) -> bool:
        """
        Kiểm tra agent có trong denylist không.

        Args:
            agent_id: ID agent

        Returns:
            True nếu trong denylist
        """
        return agent_id in self._denylist

    def evaluate(self, context: Dict[str, Any]) -> tuple[PolicyAction, Optional[PolicyRule]]:
        """
        Đánh giá policy với context cho trước.

        Args:
            context: Dictionary chứa thông tin cần đánh giá.
                    Các keys thường dùng:
                    - from_agent: ID agent gửi
                    - to_agent: ID agent nhận
                    - action: Loại action (send, request, spawn, etc.)

        Returns:
            Tuple (PolicyAction, PolicyRule hoặc None nếu dùng default)

        Example:
            action, rule = policy.evaluate({
                "from_agent": "helper",
                "to_agent": "main-agent",
                "action": "request"
            })
            if action == PolicyAction.DENY:
                raise PermissionError(f"Denied by rule: {rule.name}")
        """
        from_agent = context.get("from_agent")
        to_agent = context.get("to_agent")

        # Kiểm tra denylist trước (ưu tiên cao nhất)
        if from_agent and self.is_denied(from_agent):
            return PolicyAction.DENY, PolicyRule(
                name="_denylist",
                description=f"Agent {from_agent} trong denylist"
            )

        # Kiểm tra allowlist
        if from_agent and self.is_allowed(from_agent):
            return PolicyAction.ALLOW, PolicyRule(
                name="_allowlist",
                description=f"Agent {from_agent} trong allowlist"
            )

        # Kiểm tra role-based permissions
        if from_agent and to_agent:
            from_role = self.get_agent_role(from_agent)
            to_role = self.get_agent_role(to_agent)

            if from_role:
                from_perm = self.get_role_permission(from_role)
                if from_perm:
                    # Kiểm tra có thể gửi đến role của to_agent không
                    if to_role and to_role not in from_perm.can_send_to_roles:
                        if "*" not in from_perm.can_send_to_roles:
                            return PolicyAction.DENY, PolicyRule(
                                name="_role_restriction",
                                description=f"Role {from_role} không thể gửi đến role {to_role}"
                            )

        # Evaluate rules theo thứ tự priority
        for rule in self._rules:
            if rule.evaluate(context):
                return rule.action, rule

        # Trả về default action
        return self.default_action, None

    def can_communicate(
        self,
        from_agent: str,
        to_agent: str,
        action: str = "request",
    ) -> bool:
        """
        Kiểm tra hai agents có thể giao tiếp không.

        Args:
            from_agent: Agent gửi
            to_agent: Agent nhận
            action: Loại action

        Returns:
            True nếu được phép
        """
        policy_action, _ = self.evaluate({
            "from_agent": from_agent,
            "to_agent": to_agent,
            "action": action,
        })
        return policy_action == PolicyAction.ALLOW

    def generate_security_config(self, agent_id: str) -> A2ASecurityConfig:
        """
        Tạo A2ASecurityConfig từ policy cho một agent.

        Args:
            agent_id: ID của agent

        Returns:
            A2ASecurityConfig được tạo từ policy

        Example:
            config = policy.generate_security_config("helper")
            access_control.set_config("helper", config)
        """
        role = self.get_agent_role(agent_id)
        role_perm = self.get_role_permission(role) if role else None

        # Xác định allow_incoming
        if self.is_denied(agent_id):
            allow_incoming = []
        else:
            allow_incoming = ["*"]

        # Xác định allow_outgoing
        if role_perm:
            if "*" in role_perm.can_send_to_roles:
                allow_outgoing = ["*"]
            else:
                # Tìm các agents thuộc các roles được phép
                allow_outgoing = []
                for aid, r in self._agent_roles.items():
                    if r in role_perm.can_send_to_roles:
                        allow_outgoing.append(aid)
        else:
            allow_outgoing = ["*"]

        # Xác định spawn config
        if role_perm and role_perm.spawn_allowed:
            spawn_sandbox = SpawnSandboxConfig(
                max_spawn_depth=role_perm.max_spawn_depth
            )
        else:
            spawn_sandbox = SpawnSandboxConfig(max_spawn_depth=0)

        # Xác định history access
        history_access = {}
        if role_perm:
            history_access["*"] = role_perm.permission_level

        return A2ASecurityConfig(
            allow_incoming=allow_incoming,
            allow_outgoing=allow_outgoing,
            deny_incoming=list(self._denylist),
            history_access=history_access,
            spawn_sandbox=spawn_sandbox,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize SecurityPolicy thành dictionary.

        Returns:
            Dictionary chứa policy data
        """
        return {
            "name": self.name,
            "default_action": self.default_action.value,
            "rules": [
                {
                    "name": r.name,
                    "description": r.description,
                    "action": r.action.value,
                    "priority": r.priority.value,
                    "enabled": r.enabled,
                    "metadata": r.metadata,
                }
                for r in self._rules
            ],
            "role_permissions": {
                role: {
                    "role": perm.role,
                    "permission_level": perm.permission_level.value,
                    "can_send_to_roles": perm.can_send_to_roles,
                    "can_receive_from_roles": perm.can_receive_from_roles,
                    "spawn_allowed": perm.spawn_allowed,
                    "max_spawn_depth": perm.max_spawn_depth,
                }
                for role, perm in self._role_permissions.items()
            },
            "agent_roles": self._agent_roles,
            "allowlist": list(self._allowlist),
            "denylist": list(self._denylist),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecurityPolicy":
        """
        Khôi phục SecurityPolicy từ dictionary.

        Args:
            data: Dictionary chứa policy data

        Returns:
            SecurityPolicy instance
        """
        policy = cls(
            name=data.get("name", "default"),
            default_action=PolicyAction(data.get("default_action", "allow")),
        )

        # Khôi phục rules (không có condition vì không serialize được)
        for rule_data in data.get("rules", []):
            policy.add_rule(PolicyRule(
                name=rule_data["name"],
                description=rule_data.get("description", ""),
                action=PolicyAction(rule_data.get("action", "deny")),
                priority=PolicyPriority(rule_data.get("priority", 50)),
                enabled=rule_data.get("enabled", True),
                metadata=rule_data.get("metadata", {}),
            ))

        # Khôi phục role permissions
        for role, perm_data in data.get("role_permissions", {}).items():
            policy.set_role_permission(RolePermission(
                role=perm_data["role"],
                permission_level=PermissionLevel.from_string(
                    perm_data.get("permission_level", "request")
                ),
                can_send_to_roles=perm_data.get("can_send_to_roles", ["*"]),
                can_receive_from_roles=perm_data.get("can_receive_from_roles", ["*"]),
                spawn_allowed=perm_data.get("spawn_allowed", False),
                max_spawn_depth=perm_data.get("max_spawn_depth", 0),
            ))

        # Khôi phục agent roles
        policy._agent_roles = data.get("agent_roles", {})

        # Khôi phục allow/deny lists
        policy._allowlist = set(data.get("allowlist", []))
        policy._denylist = set(data.get("denylist", []))

        return policy


class PolicyManager:
    """
    Quản lý nhiều security policies.

    PolicyManager cho phép quản lý và chuyển đổi giữa nhiều policies
    khác nhau, ví dụ giữa development và production policies.

    Example:
        manager = PolicyManager()

        # Thêm policies
        manager.add_policy(dev_policy)
        manager.add_policy(prod_policy)

        # Kích hoạt policy
        manager.activate("production")

        # Evaluate với active policy
        action, rule = manager.evaluate(context)
    """

    def __init__(self) -> None:
        """Khởi tạo PolicyManager."""
        self._policies: Dict[str, SecurityPolicy] = {}
        self._active_policy: Optional[str] = None

    def add_policy(self, policy: SecurityPolicy) -> None:
        """
        Thêm một policy.

        Args:
            policy: SecurityPolicy cần thêm
        """
        self._policies[policy.name] = policy

    def remove_policy(self, name: str) -> bool:
        """
        Xóa một policy.

        Args:
            name: Tên policy cần xóa

        Returns:
            True nếu đã xóa thành công
        """
        if name in self._policies:
            del self._policies[name]
            if self._active_policy == name:
                self._active_policy = None
            return True
        return False

    def get_policy(self, name: str) -> Optional[SecurityPolicy]:
        """
        Lấy policy theo tên.

        Args:
            name: Tên policy

        Returns:
            SecurityPolicy nếu tìm thấy
        """
        return self._policies.get(name)

    def activate(self, name: str) -> bool:
        """
        Kích hoạt một policy.

        Args:
            name: Tên policy cần kích hoạt

        Returns:
            True nếu kích hoạt thành công
        """
        if name in self._policies:
            self._active_policy = name
            return True
        return False

    def get_active_policy(self) -> Optional[SecurityPolicy]:
        """
        Lấy policy đang active.

        Returns:
            SecurityPolicy đang active, None nếu không có
        """
        if self._active_policy:
            return self._policies.get(self._active_policy)
        return None

    def evaluate(self, context: Dict[str, Any]) -> tuple[PolicyAction, Optional[PolicyRule]]:
        """
        Evaluate context với active policy.

        Args:
            context: Context cần evaluate

        Returns:
            Tuple (PolicyAction, PolicyRule hoặc None)

        Raises:
            RuntimeError: Nếu không có active policy
        """
        policy = self.get_active_policy()
        if policy is None:
            raise RuntimeError("Không có active policy. Hãy gọi activate() trước.")
        return policy.evaluate(context)

    def list_policies(self) -> List[str]:
        """
        Liệt kê tất cả policy names.

        Returns:
            Danh sách tên policies
        """
        return list(self._policies.keys())
