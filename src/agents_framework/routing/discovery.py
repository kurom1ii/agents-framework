"""Agent Discovery module.

Module này triển khai AgentDiscovery - hệ thống tự động phát hiện
và quản lý các agents có sẵn dựa trên capabilities, status, và metadata.

Các thành phần chính:
- AgentDiscovery: Service để discover và filter agents
- CapabilityMatcher: Utility để match agents theo capabilities
- LoadBalancer: Chiến lược load balancing cho agents

Ví dụ sử dụng:
    from agents_framework.routing.discovery import AgentDiscovery

    # Tạo discovery với registry
    discovery = AgentDiscovery(registry)

    # Tìm agents theo capability
    agents = discovery.find_by_capability("coding")

    # Tìm agents available
    available = discovery.find_available()

    # Lấy thông tin agent
    info = discovery.get_agent_info("agent_id")
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from ..agents.base import BaseAgent
    from ..teams.registry import AgentRegistry

from ..agents.base import AgentStatus


@dataclass
class DiscoveredAgent:
    """Thông tin về agent được discover.

    Attributes:
        id: ID duy nhất của agent.
        name: Tên của agent.
        role_name: Tên role của agent.
        capabilities: Danh sách capabilities.
        status: Trạng thái hiện tại.
        load: Mức tải hiện tại (0.0-1.0).
        metadata: Metadata bổ sung.
        last_activity: Thời điểm hoạt động cuối.
        error_count: Số lỗi gần đây.
        success_rate: Tỷ lệ thành công.
    """

    id: str
    name: str
    role_name: str
    capabilities: list[str] = field(default_factory=list)
    status: AgentStatus = AgentStatus.IDLE
    load: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    last_activity: Optional[datetime] = None
    error_count: int = 0
    success_rate: float = 1.0

    def is_available(self) -> bool:
        """Kiểm tra agent có available không.

        Returns:
            True nếu agent ở trạng thái IDLE.
        """
        return self.status == AgentStatus.IDLE

    def has_capability(self, capability: str) -> bool:
        """Kiểm tra agent có capability không.

        Args:
            capability: Capability cần kiểm tra.

        Returns:
            True nếu agent có capability đó.
        """
        return capability in self.capabilities

    def has_all_capabilities(self, capabilities: list[str]) -> bool:
        """Kiểm tra agent có tất cả capabilities không.

        Args:
            capabilities: Danh sách capabilities cần kiểm tra.

        Returns:
            True nếu agent có tất cả capabilities.
        """
        return all(cap in self.capabilities for cap in capabilities)

    def has_any_capability(self, capabilities: list[str]) -> bool:
        """Kiểm tra agent có ít nhất một capability không.

        Args:
            capabilities: Danh sách capabilities cần kiểm tra.

        Returns:
            True nếu agent có ít nhất một capability.
        """
        return any(cap in self.capabilities for cap in capabilities)


class LoadBalancingStrategy(str, Enum):
    """Các chiến lược load balancing.

    Attributes:
        ROUND_ROBIN: Lần lượt theo thứ tự.
        RANDOM: Chọn ngẫu nhiên.
        LEAST_LOAD: Chọn agent có load thấp nhất.
        LEAST_ERRORS: Chọn agent có ít lỗi nhất.
        HIGHEST_SUCCESS: Chọn agent có tỷ lệ thành công cao nhất.
    """

    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_LOAD = "least_load"
    LEAST_ERRORS = "least_errors"
    HIGHEST_SUCCESS = "highest_success"


class AgentDiscovery:
    """Service để discover và quản lý agents.

    AgentDiscovery cung cấp các method để:
    - Tìm agents theo capability, status, role
    - Load balancing giữa các agents
    - Theo dõi health và metrics của agents

    Attributes:
        registry: AgentRegistry chứa các agents.
        agents: Dict mapping agent_id -> DiscoveredAgent.
    """

    def __init__(
        self,
        registry: Optional[AgentRegistry] = None,
        agents: Optional[dict[str, BaseAgent]] = None,
        load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
    ) -> None:
        """Khởi tạo AgentDiscovery.

        Args:
            registry: AgentRegistry (optional).
            agents: Dict mapping agent_id -> BaseAgent (optional).
            load_balancing_strategy: Chiến lược load balancing.
        """
        self._registry = registry
        self._agents: dict[str, BaseAgent] = agents or {}
        self._discovered: dict[str, DiscoveredAgent] = {}
        self._load_balancing_strategy = load_balancing_strategy
        self._round_robin_state: dict[str, int] = {}  # capability -> last index

        # Nếu có registry, discover agents
        if registry:
            self._discover_from_registry()
        elif agents:
            self._discover_from_dict(agents)

    def _discover_from_registry(self) -> None:
        """Discover agents từ registry."""
        if not self._registry:
            return

        for agent in self._registry.list_agents():
            self._agents[agent.id] = agent
            self._discovered[agent.id] = self._create_discovered_agent(agent)

    def _discover_from_dict(self, agents: dict[str, BaseAgent]) -> None:
        """Discover agents từ dict.

        Args:
            agents: Dict mapping agent_id -> BaseAgent.
        """
        for agent_id, agent in agents.items():
            self._agents[agent_id] = agent
            self._discovered[agent_id] = self._create_discovered_agent(agent)

    def _create_discovered_agent(self, agent: BaseAgent) -> DiscoveredAgent:
        """Tạo DiscoveredAgent từ BaseAgent.

        Args:
            agent: BaseAgent instance.

        Returns:
            DiscoveredAgent với thông tin của agent.
        """
        return DiscoveredAgent(
            id=agent.id,
            name=agent.config.name,
            role_name=agent.role.name,
            capabilities=list(agent.role.capabilities),
            status=agent.status,
        )

    def refresh(self) -> None:
        """Refresh thông tin về các agents.

        Đồng bộ lại với registry và cập nhật status.
        """
        if self._registry:
            self._discover_from_registry()
        else:
            # Cập nhật status từ agents
            for agent_id, agent in self._agents.items():
                if agent_id in self._discovered:
                    self._discovered[agent_id].status = agent.status

    def register_agent(
        self,
        agent_id: str,
        agent: BaseAgent,
    ) -> DiscoveredAgent:
        """Đăng ký agent mới.

        Args:
            agent_id: ID của agent.
            agent: BaseAgent instance.

        Returns:
            DiscoveredAgent đã được tạo.
        """
        self._agents[agent_id] = agent
        discovered = self._create_discovered_agent(agent)
        self._discovered[agent_id] = discovered
        return discovered

    def unregister_agent(self, agent_id: str) -> Optional[DiscoveredAgent]:
        """Hủy đăng ký agent.

        Args:
            agent_id: ID của agent.

        Returns:
            DiscoveredAgent đã xóa hoặc None.
        """
        self._agents.pop(agent_id, None)
        return self._discovered.pop(agent_id, None)

    def find_by_capability(self, capability: str) -> list[DiscoveredAgent]:
        """Tìm agents có capability cụ thể.

        Args:
            capability: Capability cần tìm.

        Returns:
            Danh sách DiscoveredAgent có capability đó.
        """
        return [
            agent for agent in self._discovered.values()
            if agent.has_capability(capability)
        ]

    def find_by_capabilities(
        self,
        capabilities: list[str],
        match_all: bool = True,
    ) -> list[DiscoveredAgent]:
        """Tìm agents có nhiều capabilities.

        Args:
            capabilities: Danh sách capabilities.
            match_all: True = phải có tất cả, False = có ít nhất một.

        Returns:
            Danh sách DiscoveredAgent phù hợp.
        """
        if match_all:
            return [
                agent for agent in self._discovered.values()
                if agent.has_all_capabilities(capabilities)
            ]
        else:
            return [
                agent for agent in self._discovered.values()
                if agent.has_any_capability(capabilities)
            ]

    def find_by_role(self, role_name: str) -> list[DiscoveredAgent]:
        """Tìm agents theo role.

        Args:
            role_name: Tên role.

        Returns:
            Danh sách DiscoveredAgent có role đó.
        """
        return [
            agent for agent in self._discovered.values()
            if agent.role_name == role_name
        ]

    def find_by_status(self, status: AgentStatus) -> list[DiscoveredAgent]:
        """Tìm agents theo status.

        Args:
            status: AgentStatus.

        Returns:
            Danh sách DiscoveredAgent có status đó.
        """
        return [
            agent for agent in self._discovered.values()
            if agent.status == status
        ]

    def find_available(
        self,
        capability: Optional[str] = None,
        role: Optional[str] = None,
    ) -> list[DiscoveredAgent]:
        """Tìm agents available (IDLE) với optional filters.

        Args:
            capability: Filter theo capability (optional).
            role: Filter theo role (optional).

        Returns:
            Danh sách DiscoveredAgent available.
        """
        agents = [a for a in self._discovered.values() if a.is_available()]

        if capability:
            agents = [a for a in agents if a.has_capability(capability)]

        if role:
            agents = [a for a in agents if a.role_name == role]

        return agents

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Lấy BaseAgent theo ID.

        Args:
            agent_id: ID của agent.

        Returns:
            BaseAgent hoặc None.
        """
        return self._agents.get(agent_id)

    def get_agent_info(self, agent_id: str) -> Optional[DiscoveredAgent]:
        """Lấy thông tin DiscoveredAgent theo ID.

        Args:
            agent_id: ID của agent.

        Returns:
            DiscoveredAgent hoặc None.
        """
        return self._discovered.get(agent_id)

    def list_agents(self) -> list[DiscoveredAgent]:
        """Lấy danh sách tất cả agents đã discover.

        Returns:
            Danh sách DiscoveredAgent.
        """
        return list(self._discovered.values())

    def list_agent_ids(self) -> list[str]:
        """Lấy danh sách tất cả agent IDs.

        Returns:
            Danh sách agent IDs.
        """
        return list(self._discovered.keys())

    def list_capabilities(self) -> list[str]:
        """Lấy danh sách tất cả capabilities unique.

        Returns:
            Danh sách capabilities.
        """
        capabilities: set[str] = set()
        for agent in self._discovered.values():
            capabilities.update(agent.capabilities)
        return list(capabilities)

    def list_roles(self) -> list[str]:
        """Lấy danh sách tất cả roles unique.

        Returns:
            Danh sách role names.
        """
        return list(set(agent.role_name for agent in self._discovered.values()))

    def select_agent(
        self,
        capability: Optional[str] = None,
        role: Optional[str] = None,
        strategy: Optional[LoadBalancingStrategy] = None,
    ) -> Optional[str]:
        """Chọn một agent theo strategy load balancing.

        Args:
            capability: Filter theo capability (optional).
            role: Filter theo role (optional).
            strategy: Chiến lược load balancing (default: ROUND_ROBIN).

        Returns:
            Agent ID hoặc None nếu không có agent phù hợp.
        """
        # Tìm agents available
        candidates = self.find_available(capability=capability, role=role)

        if not candidates:
            return None

        strategy = strategy or self._load_balancing_strategy

        if strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(candidates).id

        elif strategy == LoadBalancingStrategy.LEAST_LOAD:
            sorted_agents = sorted(candidates, key=lambda a: a.load)
            return sorted_agents[0].id

        elif strategy == LoadBalancingStrategy.LEAST_ERRORS:
            sorted_agents = sorted(candidates, key=lambda a: a.error_count)
            return sorted_agents[0].id

        elif strategy == LoadBalancingStrategy.HIGHEST_SUCCESS:
            sorted_agents = sorted(candidates, key=lambda a: -a.success_rate)
            return sorted_agents[0].id

        else:  # ROUND_ROBIN
            key = f"{capability or 'any'}:{role or 'any'}"
            if key not in self._round_robin_state:
                self._round_robin_state[key] = 0

            idx = self._round_robin_state[key] % len(candidates)
            self._round_robin_state[key] = idx + 1
            return candidates[idx].id

    def update_agent_status(
        self,
        agent_id: str,
        status: AgentStatus,
    ) -> bool:
        """Cập nhật status của agent.

        Args:
            agent_id: ID của agent.
            status: Status mới.

        Returns:
            True nếu cập nhật thành công.
        """
        if agent_id not in self._discovered:
            return False

        self._discovered[agent_id].status = status
        self._discovered[agent_id].last_activity = datetime.now()

        # Cập nhật agent thực tế nếu có
        if agent_id in self._agents:
            self._agents[agent_id].status = status

        return True

    def update_agent_load(self, agent_id: str, load: float) -> bool:
        """Cập nhật load của agent.

        Args:
            agent_id: ID của agent.
            load: Mức load mới (0.0-1.0).

        Returns:
            True nếu cập nhật thành công.
        """
        if agent_id not in self._discovered:
            return False

        self._discovered[agent_id].load = max(0.0, min(1.0, load))
        return True

    def record_agent_error(self, agent_id: str) -> None:
        """Ghi nhận lỗi cho agent.

        Args:
            agent_id: ID của agent.
        """
        if agent_id in self._discovered:
            self._discovered[agent_id].error_count += 1
            self._update_success_rate(agent_id)

    def record_agent_success(self, agent_id: str) -> None:
        """Ghi nhận success cho agent.

        Args:
            agent_id: ID của agent.
        """
        if agent_id in self._discovered:
            self._update_success_rate(agent_id)

    def _update_success_rate(self, agent_id: str) -> None:
        """Cập nhật success rate cho agent.

        Sử dụng exponential moving average.

        Args:
            agent_id: ID của agent.
        """
        agent = self._discovered.get(agent_id)
        if not agent:
            return

        # Simple calculation based on error count
        # In real implementation, would use EMA with recent history
        total = agent.error_count + 1
        agent.success_rate = 1.0 - (agent.error_count / total)

    def reset_agent_metrics(self, agent_id: str) -> bool:
        """Reset metrics cho agent.

        Args:
            agent_id: ID của agent.

        Returns:
            True nếu reset thành công.
        """
        if agent_id not in self._discovered:
            return False

        self._discovered[agent_id].error_count = 0
        self._discovered[agent_id].success_rate = 1.0
        self._discovered[agent_id].load = 0.0
        return True

    def get_statistics(self) -> dict[str, Any]:
        """Lấy thống kê về agents.

        Returns:
            Dict chứa thống kê.
        """
        agents = list(self._discovered.values())
        status_counts: dict[str, int] = {}

        for agent in agents:
            status_name = agent.status.value
            status_counts[status_name] = status_counts.get(status_name, 0) + 1

        available = len([a for a in agents if a.is_available()])
        avg_load = sum(a.load for a in agents) / len(agents) if agents else 0.0
        avg_success = (
            sum(a.success_rate for a in agents) / len(agents) if agents else 0.0
        )

        return {
            "total_agents": len(agents),
            "available_agents": available,
            "status_counts": status_counts,
            "capabilities": self.list_capabilities(),
            "roles": self.list_roles(),
            "average_load": avg_load,
            "average_success_rate": avg_success,
            "total_errors": sum(a.error_count for a in agents),
        }

    def clear(self) -> None:
        """Xóa tất cả agents đã discover."""
        self._agents.clear()
        self._discovered.clear()
        self._round_robin_state.clear()

    def __len__(self) -> int:
        """Trả về số lượng agents đã discover."""
        return len(self._discovered)

    def __contains__(self, agent_id: str) -> bool:
        """Kiểm tra agent có trong discovery không."""
        return agent_id in self._discovered


class CapabilityMatcher:
    """Utility class để match agents theo capabilities.

    CapabilityMatcher cung cấp các method nâng cao để
    tìm kiếm agents theo các tiêu chí capabilities.

    Attributes:
        discovery: AgentDiscovery instance.
    """

    def __init__(self, discovery: AgentDiscovery) -> None:
        """Khởi tạo CapabilityMatcher.

        Args:
            discovery: AgentDiscovery instance.
        """
        self._discovery = discovery

    def find_best_match(
        self,
        required: list[str],
        preferred: Optional[list[str]] = None,
    ) -> Optional[str]:
        """Tìm agent tốt nhất match với capabilities.

        Ưu tiên:
        1. Agent có tất cả required + nhiều preferred nhất
        2. Agent có tất cả required
        3. Agent có nhiều required nhất

        Args:
            required: Danh sách capabilities bắt buộc.
            preferred: Danh sách capabilities ưu tiên (optional).

        Returns:
            Agent ID hoặc None.
        """
        preferred = preferred or []
        candidates = self._discovery.find_available()

        if not candidates:
            return None

        best_agent: Optional[DiscoveredAgent] = None
        best_score = -1

        for agent in candidates:
            # Đếm số required capabilities match
            required_matches = sum(
                1 for cap in required if agent.has_capability(cap)
            )

            # Nếu không có đủ required, skip
            if required_matches < len(required):
                continue

            # Đếm số preferred capabilities match
            preferred_matches = sum(
                1 for cap in preferred if agent.has_capability(cap)
            )

            score = required_matches * 10 + preferred_matches

            if score > best_score:
                best_score = score
                best_agent = agent

        return best_agent.id if best_agent else None

    def find_with_score(
        self,
        capabilities: list[str],
        min_score: float = 0.5,
    ) -> list[tuple[str, float]]:
        """Tìm agents với score dựa trên capabilities match.

        Score = số capabilities match / tổng số capabilities yêu cầu.

        Args:
            capabilities: Danh sách capabilities.
            min_score: Score tối thiểu (0.0-1.0).

        Returns:
            Danh sách tuple (agent_id, score) sắp xếp theo score giảm dần.
        """
        candidates = self._discovery.find_available()
        results: list[tuple[str, float]] = []

        for agent in candidates:
            matches = sum(1 for cap in capabilities if agent.has_capability(cap))
            score = matches / len(capabilities) if capabilities else 0.0

            if score >= min_score:
                results.append((agent.id, score))

        return sorted(results, key=lambda x: -x[1])

    def group_by_capability(self) -> dict[str, list[str]]:
        """Nhóm agents theo capabilities.

        Returns:
            Dict mapping capability -> list of agent IDs.
        """
        groups: dict[str, list[str]] = {}

        for agent in self._discovery.list_agents():
            for cap in agent.capabilities:
                if cap not in groups:
                    groups[cap] = []
                groups[cap].append(agent.id)

        return groups
