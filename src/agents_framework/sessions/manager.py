"""
Session Manager - Quản lý vòng đời Session.

Module này cung cấp SessionManager class để quản lý toàn bộ
vòng đời của sessions: tạo, cập nhật, reset và hết hạn.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio

from .base import Session, SessionConfig, SessionScope, SessionState
from .store import SessionStore, InMemorySessionStore
from .resolver import SessionResolver, get_resolver


class SessionManager:
    """
    Quản lý vòng đời của sessions.

    SessionManager là thành phần trung tâm để:
    - Tạo mới và lấy sessions
    - Cập nhật trạng thái session
    - Kiểm tra và reset sessions hết hạn
    - Quản lý token usage

    Ví dụ sử dụng:
        ```python
        store = InMemorySessionStore()
        config = SessionConfig(scope=SessionScope.PER_PEER)
        manager = SessionManager(store, config)

        # Lấy hoặc tạo session
        session = await manager.get_or_create("key", "agent-1")

        # Cập nhật session
        session.update_tokens(input_tokens=100, output_tokens=50)
        await manager.update(session)

        # Reset session
        new_session = await manager.reset("key")
        ```
    """

    def __init__(
        self,
        store: SessionStore,
        config: Optional[SessionConfig] = None,
        resolver: Optional[SessionResolver] = None
    ) -> None:
        """
        Khởi tạo SessionManager.

        Args:
            store: Store để lưu trữ sessions
            config: Cấu hình session (mặc định nếu None)
            resolver: SessionResolver instance (mặc định nếu None)
        """
        self._store = store
        self._config = config or SessionConfig()
        self._resolver = resolver or get_resolver()
        self._locks: Dict[str, asyncio.Lock] = {}

    @property
    def config(self) -> SessionConfig:
        """Trả về cấu hình hiện tại."""
        return self._config

    @property
    def store(self) -> SessionStore:
        """Trả về store hiện tại."""
        return self._store

    def _get_lock(self, key: str) -> asyncio.Lock:
        """Lấy hoặc tạo lock cho session key."""
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    async def get_or_create(
        self,
        key: str,
        agent_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Session:
        """
        Lấy session hiện có hoặc tạo mới nếu chưa tồn tại.

        Phương thức này đảm bảo thread-safe bằng cách sử dụng lock
        cho mỗi session key.

        Args:
            key: Session key
            agent_id: ID của agent
            metadata: Metadata tùy chọn cho session mới

        Returns:
            Session (mới hoặc hiện có)
        """
        lock = self._get_lock(key)
        async with lock:
            # Thử tải session hiện có
            session = await self._store.load(key)

            if session is not None:
                # Kiểm tra xem session có cần reset không
                if self._should_reset(session):
                    session = await self._do_reset(session, metadata)
                else:
                    # Touch session để cập nhật thời gian
                    session.touch()
                    await self._store.save(session)
                return session

            # Tạo session mới
            session = Session.create(
                session_key=key,
                agent_id=agent_id,
                metadata=metadata
            )
            await self._store.save(session)
            return session

    async def get(self, key: str) -> Optional[Session]:
        """
        Lấy session theo key mà không tạo mới.

        Args:
            key: Session key

        Returns:
            Session nếu tìm thấy, None nếu không
        """
        return await self._store.load(key)

    async def create(
        self,
        agent_id: str,
        scope: Optional[SessionScope] = None,
        identifier: Optional[str] = None,
        channel: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Session:
        """
        Tạo session mới với key được tự động generate.

        Args:
            agent_id: ID của agent
            scope: Phạm vi session (mặc định từ config)
            identifier: Định danh bổ sung
            channel: Kênh giao tiếp
            metadata: Metadata tùy chọn

        Returns:
            Session mới được tạo
        """
        scope = scope or self._config.scope
        key = self._resolver.resolve_key(
            agent_id=agent_id,
            scope=scope,
            identifier=identifier,
            channel=channel
        )
        return await self.get_or_create(key, agent_id, metadata)

    async def update(self, session: Session) -> None:
        """
        Cập nhật session vào store.

        Args:
            session: Session cần cập nhật
        """
        session.updated_at = datetime.utcnow()
        await self._store.save(session)

    async def reset(self, key: str) -> Optional[Session]:
        """
        Reset session - tạo session mới với ID mới, giữ lại key.

        Args:
            key: Session key cần reset

        Returns:
            Session mới nếu tồn tại, None nếu không tìm thấy session
        """
        lock = self._get_lock(key)
        async with lock:
            session = await self._store.load(key)
            if session is None:
                return None

            return await self._do_reset(session)

    async def _do_reset(
        self,
        old_session: Session,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Session:
        """Thực hiện reset session thực sự."""
        # Lưu metadata nếu cần
        new_metadata = {}
        if self._config.preserve_metadata and old_session.metadata:
            new_metadata = old_session.metadata.copy()
        if metadata:
            new_metadata.update(metadata)

        # Đánh dấu session cũ là expired
        old_session.mark_expired()
        if self._config.auto_archive:
            old_session.state = SessionState.ARCHIVED

        # Tạo session mới với cùng key
        new_session = Session.create(
            session_key=old_session.session_key,
            agent_id=old_session.agent_id,
            metadata=new_metadata
        )

        # Lưu session mới (ghi đè session cũ trong store)
        await self._store.save(new_session)
        return new_session

    async def delete(self, key: str) -> bool:
        """
        Xóa session.

        Args:
            key: Session key cần xóa

        Returns:
            True nếu xóa thành công
        """
        lock = self._get_lock(key)
        async with lock:
            result = await self._store.delete(key)
            # Cleanup lock
            if key in self._locks:
                del self._locks[key]
            return result

    async def list_active(self, since: Optional[datetime] = None) -> List[Session]:
        """
        Liệt kê các sessions đang hoạt động.

        Args:
            since: Lọc sessions được cập nhật từ thời điểm này

        Returns:
            Danh sách sessions đang hoạt động
        """
        all_sessions = await self._store.list_all()
        active_sessions = [
            s for s in all_sessions
            if s.state in (SessionState.ACTIVE, SessionState.IDLE)
        ]

        if since is not None:
            active_sessions = [
                s for s in active_sessions
                if s.updated_at >= since
            ]

        return active_sessions

    async def list_by_agent(self, agent_id: str) -> List[Session]:
        """
        Liệt kê sessions của một agent.

        Args:
            agent_id: ID của agent

        Returns:
            Danh sách sessions của agent
        """
        return await self._store.list_by_agent(agent_id)

    async def list_idle(self) -> List[Session]:
        """
        Liệt kê các sessions đang idle.

        Returns:
            Danh sách sessions idle
        """
        all_sessions = await self._store.list_all()
        return [s for s in all_sessions if s.state == SessionState.IDLE]

    async def list_expired(self) -> List[Session]:
        """
        Liệt kê các sessions đã hết hạn.

        Returns:
            Danh sách sessions expired
        """
        all_sessions = await self._store.list_all()
        return [s for s in all_sessions if s.state == SessionState.EXPIRED]

    def _should_reset(self, session: Session) -> bool:
        """
        Kiểm tra session có cần reset không dựa trên config.

        Args:
            session: Session cần kiểm tra

        Returns:
            True nếu cần reset
        """
        if session.is_expired():
            return True

        now = datetime.utcnow()
        mode = self._config.reset_mode

        if mode == "daily" or mode == "combined":
            if self._check_daily_reset(session, now):
                return True

        if mode == "idle" or mode == "combined":
            if self._check_idle_reset(session, now):
                return True

        # Kiểm tra context tokens
        if session.context_tokens > self._config.max_context_tokens:
            return True

        return False

    def _check_daily_reset(self, session: Session, now: datetime) -> bool:
        """Kiểm tra cần reset theo daily policy."""
        reset_hour = self._config.reset_hour

        # Tính thời điểm reset hôm nay
        today_reset = now.replace(
            hour=reset_hour,
            minute=0,
            second=0,
            microsecond=0
        )

        # Nếu chưa đến giờ reset hôm nay, dùng reset hôm qua
        if now.hour < reset_hour:
            today_reset = today_reset - timedelta(days=1)

        # Session cần reset nếu được cập nhật trước thời điểm reset
        return session.updated_at < today_reset

    def _check_idle_reset(self, session: Session, now: datetime) -> bool:
        """Kiểm tra cần reset theo idle policy."""
        idle_threshold = timedelta(minutes=self._config.idle_minutes)
        return (now - session.updated_at) > idle_threshold

    async def mark_idle(self, key: str) -> Optional[Session]:
        """
        Đánh dấu session là idle.

        Args:
            key: Session key

        Returns:
            Session đã cập nhật hoặc None
        """
        session = await self._store.load(key)
        if session is None:
            return None

        session.mark_idle()
        await self._store.save(session)
        return session

    async def touch(self, key: str) -> Optional[Session]:
        """
        Cập nhật thời gian hoạt động của session.

        Args:
            key: Session key

        Returns:
            Session đã cập nhật hoặc None
        """
        session = await self._store.load(key)
        if session is None:
            return None

        session.touch()
        await self._store.save(session)
        return session

    async def update_tokens(
        self,
        key: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        context_tokens: Optional[int] = None
    ) -> Optional[Session]:
        """
        Cập nhật số token của session.

        Args:
            key: Session key
            input_tokens: Số token đầu vào mới
            output_tokens: Số token đầu ra mới
            context_tokens: Số token context mới

        Returns:
            Session đã cập nhật hoặc None
        """
        session = await self._store.load(key)
        if session is None:
            return None

        session.update_tokens(input_tokens, output_tokens, context_tokens)
        await self._store.save(session)
        return session

    async def cleanup_expired(self) -> int:
        """
        Xóa các sessions đã hết hạn từ store.

        Returns:
            Số lượng sessions đã xóa
        """
        expired = await self.list_expired()
        count = 0
        for session in expired:
            if await self._store.delete(session.session_key):
                count += 1
        return count

    async def check_and_update_states(self) -> Dict[str, int]:
        """
        Kiểm tra và cập nhật trạng thái của tất cả sessions.

        Returns:
            Dictionary với số lượng sessions theo trạng thái mới
        """
        all_sessions = await self._store.list_all()
        now = datetime.utcnow()
        stats = {"marked_idle": 0, "marked_expired": 0}

        for session in all_sessions:
            if session.state == SessionState.ACTIVE:
                # Kiểm tra idle
                if self._check_idle_reset(session, now):
                    session.mark_idle()
                    stats["marked_idle"] += 1
                    await self._store.save(session)

            elif session.state == SessionState.IDLE:
                # Kiểm tra expired
                if self._should_reset(session):
                    session.mark_expired()
                    stats["marked_expired"] += 1
                    await self._store.save(session)

        return stats


# Factory function để tạo SessionManager với InMemory store
def create_in_memory_manager(config: Optional[SessionConfig] = None) -> SessionManager:
    """
    Tạo SessionManager với InMemorySessionStore.

    Args:
        config: Cấu hình session (optional)

    Returns:
        SessionManager mới
    """
    store = InMemorySessionStore()
    return SessionManager(store, config)
