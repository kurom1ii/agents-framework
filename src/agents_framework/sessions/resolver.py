"""
Session Resolver - Giải quyết Session Key.

Module này cung cấp logic để tạo và phân tích session keys
theo format chuẩn: agent:<agentId>:<scope>:<identifier>
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import re

from .base import SessionScope


@dataclass
class SessionKeyComponents:
    """
    Các thành phần của một session key.

    Attributes:
        agent_id: ID của agent
        scope: Phạm vi session
        identifier: Định danh bổ sung (peer_id, group_id, v.v.)
        channel: Kênh giao tiếp (optional)
    """
    agent_id: str
    scope: SessionScope
    identifier: Optional[str] = None
    channel: Optional[str] = None


class SessionResolver:
    """
    Giải quyết và tạo session keys.

    Session key là định danh duy nhất cho mỗi session, được tạo
    từ agent_id, scope và các identifier tùy chọn.

    Format key:
    - Main: agent:<agentId>:main
    - Per-peer: agent:<agentId>:dm:<peerId>
    - Per-context: agent:<agentId>:<channel>:group:<contextId>

    Ví dụ:
        - agent:personal:main
        - agent:work:dm:+1234567890
        - agent:assistant:telegram:group:123456789
    """

    # Regex pattern để parse session key
    KEY_PATTERN = re.compile(
        r"^agent:(?P<agent_id>[^:]+):(?P<rest>.+)$"
    )

    def resolve_key(
        self,
        agent_id: str,
        scope: SessionScope,
        identifier: Optional[str] = None,
        channel: Optional[str] = None
    ) -> str:
        """
        Tạo session key từ các thành phần.

        Args:
            agent_id: ID của agent
            scope: Phạm vi session
            identifier: Định danh bổ sung (peer_id, context_id)
            channel: Kênh giao tiếp

        Returns:
            Session key đã được format

        Raises:
            ValueError: Nếu thiếu identifier khi scope không phải MAIN
        """
        if scope == SessionScope.MAIN:
            return f"agent:{agent_id}:main"

        if scope == SessionScope.PER_PEER:
            if not identifier:
                raise ValueError("identifier là bắt buộc cho scope PER_PEER")
            if channel:
                return f"agent:{agent_id}:{channel}:dm:{identifier}"
            return f"agent:{agent_id}:dm:{identifier}"

        if scope == SessionScope.PER_CONTEXT:
            if not identifier:
                raise ValueError("identifier là bắt buộc cho scope PER_CONTEXT")
            if channel:
                return f"agent:{agent_id}:{channel}:group:{identifier}"
            return f"agent:{agent_id}:group:{identifier}"

        raise ValueError(f"Scope không hợp lệ: {scope}")

    def parse_key(self, session_key: str) -> SessionKeyComponents:
        """
        Phân tích session key thành các thành phần.

        Args:
            session_key: Session key cần phân tích

        Returns:
            SessionKeyComponents chứa các thành phần

        Raises:
            ValueError: Nếu session key không hợp lệ
        """
        match = self.KEY_PATTERN.match(session_key)
        if not match:
            raise ValueError(f"Session key không hợp lệ: {session_key}")

        agent_id = match.group("agent_id")
        rest = match.group("rest")

        # Parse phần còn lại
        return self._parse_rest(agent_id, rest)

    def _parse_rest(self, agent_id: str, rest: str) -> SessionKeyComponents:
        """Phân tích phần còn lại của session key."""
        # Main session
        if rest == "main":
            return SessionKeyComponents(
                agent_id=agent_id,
                scope=SessionScope.MAIN
            )

        parts = rest.split(":")

        # Per-peer without channel: dm:<peerId>
        if len(parts) == 2 and parts[0] == "dm":
            return SessionKeyComponents(
                agent_id=agent_id,
                scope=SessionScope.PER_PEER,
                identifier=parts[1]
            )

        # Per-context without channel: group:<contextId>
        if len(parts) == 2 and parts[0] == "group":
            return SessionKeyComponents(
                agent_id=agent_id,
                scope=SessionScope.PER_CONTEXT,
                identifier=parts[1]
            )

        # Per-peer with channel: <channel>:dm:<peerId>
        if len(parts) == 3 and parts[1] == "dm":
            return SessionKeyComponents(
                agent_id=agent_id,
                scope=SessionScope.PER_PEER,
                identifier=parts[2],
                channel=parts[0]
            )

        # Per-context with channel: <channel>:group:<contextId>
        if len(parts) == 3 and parts[1] == "group":
            return SessionKeyComponents(
                agent_id=agent_id,
                scope=SessionScope.PER_CONTEXT,
                identifier=parts[2],
                channel=parts[0]
            )

        raise ValueError(f"Không thể parse session key: agent:{agent_id}:{rest}")

    def extract_agent_id(self, session_key: str) -> str:
        """
        Trích xuất agent_id từ session key.

        Args:
            session_key: Session key

        Returns:
            Agent ID
        """
        components = self.parse_key(session_key)
        return components.agent_id

    def extract_scope(self, session_key: str) -> SessionScope:
        """
        Trích xuất scope từ session key.

        Args:
            session_key: Session key

        Returns:
            SessionScope
        """
        components = self.parse_key(session_key)
        return components.scope

    def is_main_session(self, session_key: str) -> bool:
        """
        Kiểm tra session key có phải main session không.

        Args:
            session_key: Session key cần kiểm tra

        Returns:
            True nếu là main session
        """
        try:
            components = self.parse_key(session_key)
            return components.scope == SessionScope.MAIN
        except ValueError:
            return False

    def is_valid_key(self, session_key: str) -> bool:
        """
        Kiểm tra session key có hợp lệ không.

        Args:
            session_key: Session key cần kiểm tra

        Returns:
            True nếu hợp lệ
        """
        try:
            self.parse_key(session_key)
            return True
        except ValueError:
            return False

    def get_peer_key(self, agent_id: str, peer_id: str, channel: Optional[str] = None) -> str:
        """
        Tạo session key cho peer chat (DM).

        Args:
            agent_id: ID của agent
            peer_id: ID của peer (người dùng)
            channel: Kênh giao tiếp (optional)

        Returns:
            Session key cho peer
        """
        return self.resolve_key(
            agent_id=agent_id,
            scope=SessionScope.PER_PEER,
            identifier=peer_id,
            channel=channel
        )

    def get_group_key(
        self,
        agent_id: str,
        group_id: str,
        channel: Optional[str] = None
    ) -> str:
        """
        Tạo session key cho group chat.

        Args:
            agent_id: ID của agent
            group_id: ID của group
            channel: Kênh giao tiếp (optional)

        Returns:
            Session key cho group
        """
        return self.resolve_key(
            agent_id=agent_id,
            scope=SessionScope.PER_CONTEXT,
            identifier=group_id,
            channel=channel
        )

    def get_thread_key(
        self,
        agent_id: str,
        group_id: str,
        thread_id: str,
        channel: Optional[str] = None
    ) -> str:
        """
        Tạo session key cho thread trong group.

        Args:
            agent_id: ID của agent
            group_id: ID của group
            thread_id: ID của thread
            channel: Kênh giao tiếp (optional)

        Returns:
            Session key cho thread
        """
        # Format: agent:<agentId>:<channel>:group:<groupId>:topic:<threadId>
        context_id = f"{group_id}:topic:{thread_id}"
        return self.resolve_key(
            agent_id=agent_id,
            scope=SessionScope.PER_CONTEXT,
            identifier=context_id,
            channel=channel
        )


# Singleton instance cho tiện sử dụng
_default_resolver: Optional[SessionResolver] = None


def get_resolver() -> SessionResolver:
    """Lấy instance SessionResolver mặc định."""
    global _default_resolver
    if _default_resolver is None:
        _default_resolver = SessionResolver()
    return _default_resolver
