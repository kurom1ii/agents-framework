"""
Sandbox configuration cho Spawned Agents.

Module này định nghĩa các cấu hình sandbox để giới hạn quyền hạn
của các sub-agents được spawn bởi parent agents trong hệ thống A2A.

Sandbox features:
- Tool restrictions (whitelist/blacklist)
- Workspace access control
- Network access control
- Resource limits (tokens, timeout)
- Spawn depth limits
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set


class SandboxMode(Enum):
    """
    Chế độ sandbox cho spawned agents.

    - ALWAYS: Luôn áp dụng sandbox cho tất cả spawned agents
    - NON_MAIN: Chỉ áp dụng cho agents không phải main (sub-agents)
    - NEVER: Không áp dụng sandbox
    """

    ALWAYS = "always"
    NON_MAIN = "non-main"
    NEVER = "never"


class WorkspaceAccess(Enum):
    """
    Mức độ truy cập workspace cho spawned agents.

    - NONE: Không được truy cập workspace
    - READ_ONLY: Chỉ được đọc
    - READ_WRITE: Được đọc và ghi
    """

    NONE = "none"
    READ_ONLY = "ro"
    READ_WRITE = "rw"


class NetworkAccess(Enum):
    """
    Mức độ truy cập network cho spawned agents.

    - NONE: Không được truy cập network
    - LOCAL: Chỉ được truy cập localhost
    - ALLOWED_HOSTS: Chỉ được truy cập các hosts được phép
    - FULL: Được truy cập tất cả
    """

    NONE = "none"
    LOCAL = "local"
    ALLOWED_HOSTS = "allowed_hosts"
    FULL = "full"


@dataclass
class SpawnSandboxConfig:
    """
    Cấu hình sandbox cho spawned agents.

    SpawnSandboxConfig định nghĩa các giới hạn và restrictions được áp dụng
    khi một agent spawn sub-agents. Đây là cơ chế quan trọng để đảm bảo
    security trong hệ thống A2A.

    Attributes:
        mode: Chế độ sandbox (always, non-main, never)
        workspace_access: Mức truy cập workspace (none, ro, rw)
        allowed_tools: Danh sách tools được phép sử dụng
        denied_tools: Danh sách tools bị cấm (ưu tiên cao hơn allowed)
        max_tokens: Số tokens tối đa cho spawned agent
        timeout_ms: Thời gian timeout (milliseconds)
        max_spawn_depth: Độ sâu spawn tối đa (1 = không được spawn thêm)
        network_access: Mức truy cập network
        allowed_hosts: Danh sách hosts được phép (khi network_access = allowed_hosts)
        inherit_context: Có kế thừa context từ parent không
        metadata: Dữ liệu bổ sung

    Example:
        config = SpawnSandboxConfig(
            mode=SandboxMode.ALWAYS,
            workspace_access=WorkspaceAccess.READ_ONLY,
            allowed_tools=["read", "write", "web_search"],
            denied_tools=["bash", "browser"],
            max_tokens=50000,
            timeout_ms=300000,
            max_spawn_depth=1
        )
    """

    mode: SandboxMode = SandboxMode.ALWAYS
    workspace_access: WorkspaceAccess = WorkspaceAccess.READ_ONLY
    allowed_tools: List[str] = field(default_factory=lambda: ["read", "write"])
    denied_tools: List[str] = field(default_factory=lambda: ["bash", "browser"])
    max_tokens: int = 50000
    timeout_ms: int = 300000  # 5 phút
    max_spawn_depth: int = 1
    network_access: NetworkAccess = NetworkAccess.NONE
    allowed_hosts: List[str] = field(default_factory=list)
    inherit_context: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_tool_allowed(self, tool_name: str) -> bool:
        """
        Kiểm tra tool có được phép sử dụng trong sandbox không.

        Denied tools có ưu tiên cao hơn allowed tools.

        Args:
            tool_name: Tên tool cần kiểm tra

        Returns:
            True nếu tool được phép

        Example:
            config = SpawnSandboxConfig(
                allowed_tools=["read", "write"],
                denied_tools=["bash"]
            )
            config.is_tool_allowed("read")  # True
            config.is_tool_allowed("bash")  # False
            config.is_tool_allowed("unknown")  # False
        """
        # Denied tools có ưu tiên cao nhất
        if tool_name in self.denied_tools:
            return False

        # Nếu allowed_tools rỗng hoặc chứa "*", cho phép tất cả
        if not self.allowed_tools or "*" in self.allowed_tools:
            return True

        # Kiểm tra trong allowed list
        return tool_name in self.allowed_tools

    def is_host_allowed(self, host: str) -> bool:
        """
        Kiểm tra host có được phép truy cập không.

        Args:
            host: Hostname hoặc IP cần kiểm tra

        Returns:
            True nếu host được phép truy cập
        """
        if self.network_access == NetworkAccess.NONE:
            return False
        if self.network_access == NetworkAccess.LOCAL:
            return host in ("localhost", "127.0.0.1", "::1")
        if self.network_access == NetworkAccess.FULL:
            return True
        if self.network_access == NetworkAccess.ALLOWED_HOSTS:
            # Kiểm tra exact match hoặc wildcard
            for allowed in self.allowed_hosts:
                if allowed == "*" or allowed == host:
                    return True
                # Hỗ trợ wildcard subdomain (*.example.com)
                if allowed.startswith("*.") and host.endswith(allowed[1:]):
                    return True
            return False
        return False

    def can_spawn_more(self, current_depth: int) -> bool:
        """
        Kiểm tra có thể spawn thêm sub-agents không.

        Args:
            current_depth: Độ sâu spawn hiện tại

        Returns:
            True nếu còn được phép spawn
        """
        return current_depth < self.max_spawn_depth

    def get_filtered_tools(self, available_tools: List[str]) -> List[str]:
        """
        Lọc danh sách tools theo sandbox config.

        Args:
            available_tools: Danh sách tất cả tools có sẵn

        Returns:
            Danh sách tools được phép trong sandbox

        Example:
            available = ["read", "write", "bash", "web_search"]
            allowed = config.get_filtered_tools(available)
            # ["read", "write"] (nếu bash bị cấm)
        """
        return [tool for tool in available_tools if self.is_tool_allowed(tool)]

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize SpawnSandboxConfig thành dictionary.

        Returns:
            Dictionary chứa cấu hình sandbox
        """
        return {
            "mode": self.mode.value,
            "workspace_access": self.workspace_access.value,
            "allowed_tools": self.allowed_tools,
            "denied_tools": self.denied_tools,
            "max_tokens": self.max_tokens,
            "timeout_ms": self.timeout_ms,
            "max_spawn_depth": self.max_spawn_depth,
            "network_access": self.network_access.value,
            "allowed_hosts": self.allowed_hosts,
            "inherit_context": self.inherit_context,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpawnSandboxConfig":
        """
        Tạo SpawnSandboxConfig từ dictionary.

        Args:
            data: Dictionary chứa cấu hình

        Returns:
            SpawnSandboxConfig instance
        """
        return cls(
            mode=SandboxMode(data.get("mode", "always")),
            workspace_access=WorkspaceAccess(data.get("workspace_access", "ro")),
            allowed_tools=data.get("allowed_tools", ["read", "write"]),
            denied_tools=data.get("denied_tools", ["bash", "browser"]),
            max_tokens=data.get("max_tokens", 50000),
            timeout_ms=data.get("timeout_ms", 300000),
            max_spawn_depth=data.get("max_spawn_depth", 1),
            network_access=NetworkAccess(data.get("network_access", "none")),
            allowed_hosts=data.get("allowed_hosts", []),
            inherit_context=data.get("inherit_context", True),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def permissive(cls) -> "SpawnSandboxConfig":
        """
        Tạo cấu hình sandbox cho phép nhiều quyền.

        Returns:
            SpawnSandboxConfig với nhiều quyền

        Example:
            config = SpawnSandboxConfig.permissive()
        """
        return cls(
            mode=SandboxMode.NON_MAIN,
            workspace_access=WorkspaceAccess.READ_WRITE,
            allowed_tools=["*"],
            denied_tools=[],
            max_tokens=100000,
            timeout_ms=600000,
            max_spawn_depth=3,
            network_access=NetworkAccess.FULL,
        )

    @classmethod
    def restrictive(cls) -> "SpawnSandboxConfig":
        """
        Tạo cấu hình sandbox hạn chế cao.

        Returns:
            SpawnSandboxConfig với ít quyền

        Example:
            config = SpawnSandboxConfig.restrictive()
        """
        return cls(
            mode=SandboxMode.ALWAYS,
            workspace_access=WorkspaceAccess.NONE,
            allowed_tools=["read"],
            denied_tools=["bash", "browser", "write", "execute"],
            max_tokens=10000,
            timeout_ms=60000,
            max_spawn_depth=0,
            network_access=NetworkAccess.NONE,
        )


class SpawnSandbox:
    """
    Runtime sandbox cho spawned agents.

    SpawnSandbox cung cấp các phương thức để enforce các restrictions
    được định nghĩa trong SpawnSandboxConfig tại runtime.

    Attributes:
        config: Cấu hình sandbox
        parent_agent_id: ID của parent agent đã spawn
        spawn_depth: Độ sâu spawn hiện tại
        created_at: Thời điểm tạo sandbox

    Example:
        sandbox = SpawnSandbox(
            config=SpawnSandboxConfig.restrictive(),
            parent_agent_id="main-agent",
            spawn_depth=1
        )

        # Kiểm tra tool trước khi thực thi
        if sandbox.can_use_tool("bash"):
            # Thực thi tool
            pass
        else:
            # Từ chối
            pass
    """

    def __init__(
        self,
        config: SpawnSandboxConfig,
        parent_agent_id: str,
        spawn_depth: int = 1,
    ) -> None:
        """
        Khởi tạo SpawnSandbox.

        Args:
            config: Cấu hình sandbox
            parent_agent_id: ID của parent agent
            spawn_depth: Độ sâu spawn hiện tại
        """
        self.config = config
        self.parent_agent_id = parent_agent_id
        self.spawn_depth = spawn_depth
        self._used_tokens: int = 0
        self._start_time: Optional[float] = None

    def can_use_tool(self, tool_name: str) -> bool:
        """
        Kiểm tra tool có được phép sử dụng không.

        Args:
            tool_name: Tên tool

        Returns:
            True nếu được phép
        """
        return self.config.is_tool_allowed(tool_name)

    def can_access_network(self, host: str) -> bool:
        """
        Kiểm tra có được truy cập host không.

        Args:
            host: Hostname hoặc IP

        Returns:
            True nếu được phép
        """
        return self.config.is_host_allowed(host)

    def can_spawn_child(self) -> bool:
        """
        Kiểm tra có được spawn child agents không.

        Returns:
            True nếu được phép
        """
        return self.config.can_spawn_more(self.spawn_depth)

    def can_write_workspace(self) -> bool:
        """
        Kiểm tra có được ghi vào workspace không.

        Returns:
            True nếu được phép
        """
        return self.config.workspace_access == WorkspaceAccess.READ_WRITE

    def can_read_workspace(self) -> bool:
        """
        Kiểm tra có được đọc workspace không.

        Returns:
            True nếu được phép
        """
        return self.config.workspace_access in (
            WorkspaceAccess.READ_ONLY,
            WorkspaceAccess.READ_WRITE,
        )

    def record_tokens(self, tokens: int) -> None:
        """
        Ghi nhận số tokens đã sử dụng.

        Args:
            tokens: Số tokens vừa sử dụng
        """
        self._used_tokens += tokens

    def is_token_limit_exceeded(self) -> bool:
        """
        Kiểm tra đã vượt giới hạn tokens chưa.

        Returns:
            True nếu đã vượt
        """
        return self._used_tokens > self.config.max_tokens

    def get_remaining_tokens(self) -> int:
        """
        Lấy số tokens còn lại.

        Returns:
            Số tokens còn được sử dụng
        """
        return max(0, self.config.max_tokens - self._used_tokens)

    def create_child_sandbox(self) -> Optional["SpawnSandbox"]:
        """
        Tạo sandbox cho child agent (nếu được phép).

        Returns:
            SpawnSandbox mới nếu được phép spawn, None nếu không

        Example:
            if sandbox.can_spawn_child():
                child_sandbox = sandbox.create_child_sandbox()
                # spawn child agent với child_sandbox
        """
        if not self.can_spawn_child():
            return None

        return SpawnSandbox(
            config=self.config,
            parent_agent_id=self.parent_agent_id,
            spawn_depth=self.spawn_depth + 1,
        )

    def validate_operation(
        self,
        operation: Literal["tool", "network", "spawn", "workspace_read", "workspace_write"],
        target: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Validate một operation trong sandbox.

        Args:
            operation: Loại operation
            target: Đối tượng của operation (tool name, host, etc.)

        Returns:
            Tuple (allowed, reason) - True nếu được phép, reason nếu bị từ chối

        Example:
            allowed, reason = sandbox.validate_operation("tool", "bash")
            if not allowed:
                raise PermissionError(reason)
        """
        if operation == "tool":
            if target and not self.can_use_tool(target):
                return False, f"Tool '{target}' không được phép trong sandbox"
            return True, None

        if operation == "network":
            if target and not self.can_access_network(target):
                return False, f"Host '{target}' không được phép truy cập"
            return True, None

        if operation == "spawn":
            if not self.can_spawn_child():
                return False, f"Đã đạt giới hạn spawn depth ({self.spawn_depth}/{self.config.max_spawn_depth})"
            return True, None

        if operation == "workspace_read":
            if not self.can_read_workspace():
                return False, "Không được phép đọc workspace"
            return True, None

        if operation == "workspace_write":
            if not self.can_write_workspace():
                return False, "Không được phép ghi workspace"
            return True, None

        return True, None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize sandbox state thành dictionary.

        Returns:
            Dictionary chứa state của sandbox
        """
        return {
            "config": self.config.to_dict(),
            "parent_agent_id": self.parent_agent_id,
            "spawn_depth": self.spawn_depth,
            "used_tokens": self._used_tokens,
        }
