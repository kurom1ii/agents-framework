"""
Session Discovery và Agent Discovery cho hệ thống A2A.

Module này cung cấp các class để khám phá và liệt kê:
- SessionDiscovery: Khám phá các sessions đang hoạt động
- AgentDiscovery: Khám phá các agents có sẵn trong hệ thống
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, List, Optional, Protocol

from .base import AgentInfo, AgentStatus, SessionInfo, SessionStatus

if TYPE_CHECKING:
    from ..sessions.base import Session


class SessionManagerProtocol(Protocol):
    """
    Protocol định nghĩa interface cho Session Manager.

    Đây là interface mà SessionDiscovery yêu cầu từ Session Manager
    để có thể truy vấn thông tin sessions.
    """

    async def get_all_sessions(self) -> List[Any]:
        """Lấy tất cả sessions trong hệ thống."""
        ...

    async def get_session(self, session_key: str) -> Optional[Any]:
        """Lấy session theo key."""
        ...


class AgentRegistryProtocol(Protocol):
    """
    Protocol định nghĩa interface cho Agent Registry.

    Đây là interface mà AgentDiscovery yêu cầu từ Agent Registry
    để có thể truy vấn thông tin agents.
    """

    def get_all_agents(self) -> List[Any]:
        """Lấy tất cả agents đã đăng ký."""
        ...

    def get_agent(self, agent_id: str) -> Optional[Any]:
        """Lấy agent theo ID."""
        ...


class SessionDiscovery:
    """
    Khám phá và liệt kê các sessions đang hoạt động trong hệ thống.

    SessionDiscovery cho phép agents khám phá các sessions khác để có thể
    giao tiếp hoặc ủy quyền tác vụ. Hỗ trợ các bộ lọc theo trạng thái,
    agent, và thời gian hoạt động.

    Attributes:
        session_manager: Session Manager để truy vấn sessions

    Example:
        discovery = SessionDiscovery(session_manager)

        # Liệt kê tất cả sessions đang hoạt động
        active_sessions = await discovery.list_active_sessions(minutes=60)

        # Lọc sessions theo agent
        coder_sessions = await discovery.list_sessions(filter_agent="coder")

        # Lấy thông tin chi tiết của một session
        info = await discovery.get_session_info("agent:coder:main")
    """

    def __init__(self, session_manager: SessionManagerProtocol) -> None:
        """
        Khởi tạo SessionDiscovery.

        Args:
            session_manager: Session Manager để truy vấn sessions
        """
        self.session_manager = session_manager

    async def list_sessions(
        self,
        filter_status: Optional[SessionStatus] = None,
        filter_agent: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 50,
    ) -> List[SessionInfo]:
        """
        Liệt kê các sessions theo bộ lọc.

        Phương thức này cho phép lọc sessions theo nhiều tiêu chí khác nhau
        để tìm sessions phù hợp với nhu cầu giao tiếp A2A.

        Args:
            filter_status: Lọc theo trạng thái (idle, busy, error).
                          None để lấy tất cả trạng thái.
            filter_agent: Lọc theo agent_id cụ thể.
                         None để lấy sessions của tất cả agents.
            since: Chỉ lấy sessions hoạt động từ thời điểm này.
                  None để không giới hạn thời gian.
            limit: Số lượng sessions tối đa trả về. Mặc định 50.

        Returns:
            Danh sách SessionInfo đã được lọc và giới hạn.

        Example:
            # Lấy sessions idle của coder agent
            sessions = await discovery.list_sessions(
                filter_status=SessionStatus.IDLE,
                filter_agent="coder",
                limit=10
            )
        """
        all_sessions = await self.session_manager.get_all_sessions()
        results: List[SessionInfo] = []

        for session in all_sessions:
            session_info = self._convert_to_session_info(session)

            # Áp dụng bộ lọc theo trạng thái
            if filter_status is not None and session_info.status != filter_status:
                continue

            # Áp dụng bộ lọc theo agent
            if filter_agent is not None and session_info.agent_id != filter_agent:
                continue

            # Áp dụng bộ lọc theo thời gian
            if since is not None and session_info.last_activity < since:
                continue

            results.append(session_info)

            # Kiểm tra giới hạn
            if len(results) >= limit:
                break

        return results

    async def get_session_info(self, session_key: str) -> Optional[SessionInfo]:
        """
        Lấy thông tin chi tiết của một session.

        Args:
            session_key: Key của session cần lấy thông tin

        Returns:
            SessionInfo nếu tìm thấy, None nếu không tồn tại

        Example:
            info = await discovery.get_session_info("agent:coder:main")
            if info:
                print(f"Status: {info.status.value}")
                print(f"Tokens: {info.context_tokens}")
        """
        session = await self.session_manager.get_session(session_key)
        if session is None:
            return None
        return self._convert_to_session_info(session)

    async def list_active_sessions(self, minutes: int = 60) -> List[SessionInfo]:
        """
        Liệt kê sessions hoạt động trong N phút gần đây.

        Phương thức tiện lợi để nhanh chóng tìm các sessions đang hoạt động
        gần đây, phù hợp cho việc khám phá sessions để giao tiếp A2A.

        Args:
            minutes: Số phút tính từ thời điểm hiện tại. Mặc định 60 phút.

        Returns:
            Danh sách SessionInfo của các sessions hoạt động gần đây.

        Example:
            # Lấy sessions hoạt động trong 30 phút qua
            recent = await discovery.list_active_sessions(minutes=30)
        """
        since = datetime.now() - timedelta(minutes=minutes)
        return await self.list_sessions(since=since)

    async def list_idle_sessions(self, limit: int = 50) -> List[SessionInfo]:
        """
        Liệt kê các sessions đang idle.

        Phương thức tiện lợi để tìm sessions đang rảnh, phù hợp cho việc
        ủy quyền tác vụ mới.

        Args:
            limit: Số lượng sessions tối đa trả về

        Returns:
            Danh sách SessionInfo của các sessions đang idle
        """
        return await self.list_sessions(filter_status=SessionStatus.IDLE, limit=limit)

    async def list_busy_sessions(self, limit: int = 50) -> List[SessionInfo]:
        """
        Liệt kê các sessions đang busy.

        Args:
            limit: Số lượng sessions tối đa trả về

        Returns:
            Danh sách SessionInfo của các sessions đang busy
        """
        return await self.list_sessions(filter_status=SessionStatus.BUSY, limit=limit)

    async def count_sessions(
        self,
        filter_status: Optional[SessionStatus] = None,
        filter_agent: Optional[str] = None,
    ) -> int:
        """
        Đếm số lượng sessions theo bộ lọc.

        Args:
            filter_status: Lọc theo trạng thái
            filter_agent: Lọc theo agent

        Returns:
            Số lượng sessions thỏa mãn điều kiện
        """
        sessions = await self.list_sessions(
            filter_status=filter_status,
            filter_agent=filter_agent,
            limit=10000,  # Large limit to count all
        )
        return len(sessions)

    def _convert_to_session_info(self, session: Any) -> SessionInfo:
        """
        Chuyển đổi Session object thành SessionInfo.

        Args:
            session: Session object từ Session Manager

        Returns:
            SessionInfo tương ứng
        """
        # Xử lý trường hợp session đã là SessionInfo
        if isinstance(session, SessionInfo):
            return session

        # Xử lý trường hợp session là dict
        if isinstance(session, dict):
            return SessionInfo.from_dict(session)

        # Xử lý trường hợp session là Session object từ sessions.base
        # Map SessionState -> SessionStatus
        status_mapping = {
            "active": SessionStatus.BUSY,
            "idle": SessionStatus.IDLE,
            "expired": SessionStatus.ERROR,
            "archived": SessionStatus.ERROR,
        }

        session_state = getattr(session, "state", None)
        if session_state is not None:
            state_value = (
                session_state.value
                if hasattr(session_state, "value")
                else str(session_state)
            )
            status = status_mapping.get(state_value, SessionStatus.IDLE)
        else:
            status = SessionStatus.IDLE

        return SessionInfo(
            session_key=getattr(session, "session_key", str(session)),
            agent_id=getattr(session, "agent_id", "unknown"),
            status=status,
            last_activity=getattr(session, "updated_at", datetime.now()),
            context_tokens=getattr(session, "context_tokens", 0),
            display_name=getattr(session, "metadata", {}).get("display_name"),
            metadata=getattr(session, "metadata", {}),
        )


class AgentDiscovery:
    """
    Khám phá các agents có sẵn trong hệ thống.

    AgentDiscovery cho phép tìm kiếm và lấy thông tin về các agents
    dựa trên khả năng, trạng thái, và các tiêu chí khác.

    Attributes:
        registry: Agent Registry để truy vấn agents

    Example:
        discovery = AgentDiscovery(agent_registry)

        # Tìm agents có khả năng "code_review"
        reviewers = discovery.find_by_capability("code_review")

        # Tìm agents đang available
        available = discovery.find_available()

        # Lấy thông tin chi tiết của một agent
        info = discovery.get_agent_info("coder")
    """

    def __init__(self, agent_registry: AgentRegistryProtocol) -> None:
        """
        Khởi tạo AgentDiscovery.

        Args:
            agent_registry: Agent Registry để truy vấn agents
        """
        self.registry = agent_registry

    def list_agents(
        self,
        filter_status: Optional[AgentStatus] = None,
        filter_role: Optional[str] = None,
        limit: int = 50,
    ) -> List[AgentInfo]:
        """
        Liệt kê các agents theo bộ lọc.

        Args:
            filter_status: Lọc theo trạng thái (available, busy, offline)
            filter_role: Lọc theo role của agent
            limit: Số lượng agents tối đa trả về

        Returns:
            Danh sách AgentInfo đã được lọc
        """
        all_agents = self.registry.get_all_agents()
        results: List[AgentInfo] = []

        for agent in all_agents:
            agent_info = self._convert_to_agent_info(agent)

            # Áp dụng bộ lọc theo trạng thái
            if filter_status is not None and agent_info.status != filter_status:
                continue

            # Áp dụng bộ lọc theo role
            if filter_role is not None and agent_info.role != filter_role:
                continue

            results.append(agent_info)

            if len(results) >= limit:
                break

        return results

    def find_by_capability(self, capability: str) -> List[str]:
        """
        Tìm agents có khả năng cụ thể.

        Phương thức này trả về danh sách agent_id của các agents
        có khả năng được chỉ định.

        Args:
            capability: Khả năng cần tìm (ví dụ: "code_review", "research")

        Returns:
            Danh sách agent_id có khả năng này

        Example:
            coders = discovery.find_by_capability("coding")
            # ["coder", "senior-dev", "assistant"]
        """
        all_agents = self.registry.get_all_agents()
        results: List[str] = []

        for agent in all_agents:
            agent_info = self._convert_to_agent_info(agent)
            if agent_info.has_capability(capability):
                results.append(agent_info.agent_id)

        return results

    def find_available(self) -> List[str]:
        """
        Tìm agents đang available.

        Returns:
            Danh sách agent_id đang sẵn sàng nhận tác vụ

        Example:
            available = discovery.find_available()
            if available:
                target = available[0]
        """
        agents = self.list_agents(filter_status=AgentStatus.AVAILABLE)
        return [a.agent_id for a in agents]

    def find_by_role(self, role: str) -> List[str]:
        """
        Tìm agents theo role.

        Args:
            role: Role cần tìm

        Returns:
            Danh sách agent_id có role này
        """
        agents = self.list_agents(filter_role=role)
        return [a.agent_id for a in agents]

    def get_agent_info(self, agent_id: str) -> Optional[AgentInfo]:
        """
        Lấy thông tin chi tiết của một agent.

        Args:
            agent_id: ID của agent cần lấy thông tin

        Returns:
            AgentInfo nếu tìm thấy, None nếu không tồn tại

        Example:
            info = discovery.get_agent_info("coder")
            if info:
                print(f"Role: {info.role}")
                print(f"Capabilities: {info.capabilities}")
        """
        agent = self.registry.get_agent(agent_id)
        if agent is None:
            return None
        return self._convert_to_agent_info(agent)

    def get_agents_with_capabilities(
        self, capabilities: List[str], match_all: bool = True
    ) -> List[AgentInfo]:
        """
        Tìm agents có nhiều khả năng cụ thể.

        Args:
            capabilities: Danh sách khả năng cần tìm
            match_all: True nếu cần tất cả khả năng, False nếu chỉ cần một

        Returns:
            Danh sách AgentInfo thỏa mãn điều kiện

        Example:
            # Tìm agents có cả "coding" và "testing"
            agents = discovery.get_agents_with_capabilities(
                ["coding", "testing"],
                match_all=True
            )
        """
        all_agents = self.registry.get_all_agents()
        results: List[AgentInfo] = []

        for agent in all_agents:
            agent_info = self._convert_to_agent_info(agent)

            if match_all:
                # Cần có tất cả capabilities
                if all(cap in agent_info.capabilities for cap in capabilities):
                    results.append(agent_info)
            else:
                # Chỉ cần có ít nhất một capability
                if any(cap in agent_info.capabilities for cap in capabilities):
                    results.append(agent_info)

        return results

    def count_available(self) -> int:
        """
        Đếm số agents đang available.

        Returns:
            Số lượng agents sẵn sàng
        """
        return len(self.find_available())

    def _convert_to_agent_info(self, agent: Any) -> AgentInfo:
        """
        Chuyển đổi Agent object thành AgentInfo.

        Args:
            agent: Agent object từ Agent Registry

        Returns:
            AgentInfo tương ứng
        """
        # Xử lý trường hợp agent đã là AgentInfo
        if isinstance(agent, AgentInfo):
            return agent

        # Xử lý trường hợp agent là dict
        if isinstance(agent, dict):
            return AgentInfo.from_dict(agent)

        # Xử lý trường hợp agent là BaseAgent object
        # Map AgentStatus từ agents.base -> AgentStatus A2A
        status_mapping = {
            "idle": AgentStatus.AVAILABLE,
            "busy": AgentStatus.BUSY,
            "error": AgentStatus.OFFLINE,
            "terminated": AgentStatus.OFFLINE,
        }

        agent_status = getattr(agent, "status", None)
        if agent_status is not None:
            status_value = (
                agent_status.value
                if hasattr(agent_status, "value")
                else str(agent_status)
            )
            status = status_mapping.get(status_value, AgentStatus.AVAILABLE)
        else:
            status = AgentStatus.AVAILABLE

        # Lấy role từ agent
        role_obj = getattr(agent, "role", None)
        if role_obj is not None:
            role_name = getattr(role_obj, "name", str(role_obj))
            capabilities = getattr(role_obj, "capabilities", [])
            description = getattr(role_obj, "description", None)
        else:
            role_name = "unknown"
            capabilities = []
            description = None

        return AgentInfo(
            agent_id=getattr(agent, "id", str(agent)),
            role=role_name,
            capabilities=list(capabilities),
            status=status,
            current_session=None,
            description=description,
        )
