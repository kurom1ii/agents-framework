"""
Session Store Protocol và Implementations.

Module này định nghĩa SessionStore protocol - giao diện chuẩn cho việc
lưu trữ và truy xuất sessions, cùng với các implementation cơ bản.
"""

from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import json
import aiofiles
import aiofiles.os

from .base import Session, SessionState


@runtime_checkable
class SessionStore(Protocol):
    """
    Protocol định nghĩa giao diện lưu trữ session.

    Tất cả các implementation của SessionStore phải triển khai
    các phương thức này để đảm bảo tính nhất quán.
    """

    @abstractmethod
    async def save(self, session: Session) -> None:
        """
        Lưu hoặc cập nhật session.

        Args:
            session: Session cần lưu
        """
        ...

    @abstractmethod
    async def load(self, session_key: str) -> Optional[Session]:
        """
        Tải session theo key.

        Args:
            session_key: Key của session cần tải

        Returns:
            Session nếu tìm thấy, None nếu không
        """
        ...

    @abstractmethod
    async def delete(self, session_key: str) -> bool:
        """
        Xóa session theo key.

        Args:
            session_key: Key của session cần xóa

        Returns:
            True nếu xóa thành công, False nếu không tìm thấy
        """
        ...

    @abstractmethod
    async def list_all(self) -> List[Session]:
        """
        Liệt kê tất cả sessions.

        Returns:
            Danh sách tất cả sessions
        """
        ...

    @abstractmethod
    async def list_by_agent(self, agent_id: str) -> List[Session]:
        """
        Liệt kê sessions theo agent ID.

        Args:
            agent_id: ID của agent

        Returns:
            Danh sách sessions của agent
        """
        ...


class InMemorySessionStore:
    """
    Session store lưu trữ trong bộ nhớ.

    Phù hợp cho testing và các ứng dụng không cần persistence.
    Dữ liệu sẽ mất khi process kết thúc.
    """

    def __init__(self) -> None:
        """Khởi tạo store với dictionary rỗng."""
        self._sessions: Dict[str, Session] = {}

    async def save(self, session: Session) -> None:
        """Lưu session vào bộ nhớ."""
        self._sessions[session.session_key] = session

    async def load(self, session_key: str) -> Optional[Session]:
        """Tải session từ bộ nhớ."""
        return self._sessions.get(session_key)

    async def delete(self, session_key: str) -> bool:
        """Xóa session khỏi bộ nhớ."""
        if session_key in self._sessions:
            del self._sessions[session_key]
            return True
        return False

    async def list_all(self) -> List[Session]:
        """Liệt kê tất cả sessions trong bộ nhớ."""
        return list(self._sessions.values())

    async def list_by_agent(self, agent_id: str) -> List[Session]:
        """Liệt kê sessions của một agent."""
        return [
            session for session in self._sessions.values()
            if session.agent_id == agent_id
        ]

    async def clear(self) -> None:
        """Xóa tất cả sessions (dùng cho testing)."""
        self._sessions.clear()


class FileSessionStore:
    """
    Session store lưu trữ vào file JSON.

    Lưu trữ sessions vào file sessions.json trong thư mục chỉ định.
    Phù hợp cho các ứng dụng đơn giản cần persistence.

    Cấu trúc thư mục:
        base_path/
        └── sessions.json  # Kho session (key -> metadata)
    """

    def __init__(self, base_path: Path) -> None:
        """
        Khởi tạo FileSessionStore.

        Args:
            base_path: Đường dẫn thư mục lưu trữ
        """
        self._base_path = Path(base_path)
        self._sessions_file = self._base_path / "sessions.json"
        self._cache: Optional[Dict[str, Dict[str, Any]]] = None

    async def _ensure_directory(self) -> None:
        """Đảm bảo thư mục lưu trữ tồn tại."""
        if not self._base_path.exists():
            await aiofiles.os.makedirs(str(self._base_path), exist_ok=True)

    async def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Tải cache từ file nếu chưa có."""
        if self._cache is not None:
            return self._cache

        if not self._sessions_file.exists():
            self._cache = {}
            return self._cache

        try:
            async with aiofiles.open(self._sessions_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                self._cache = json.loads(content) if content.strip() else {}
        except (json.JSONDecodeError, OSError):
            self._cache = {}

        return self._cache

    async def _save_cache(self) -> None:
        """Lưu cache vào file."""
        await self._ensure_directory()
        async with aiofiles.open(self._sessions_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(self._cache, indent=2, ensure_ascii=False))

    async def save(self, session: Session) -> None:
        """Lưu session vào file."""
        cache = await self._load_cache()
        cache[session.session_key] = session.to_dict()
        await self._save_cache()

    async def load(self, session_key: str) -> Optional[Session]:
        """Tải session từ file."""
        cache = await self._load_cache()
        data = cache.get(session_key)
        if data is None:
            return None
        return Session.from_dict(data)

    async def delete(self, session_key: str) -> bool:
        """Xóa session khỏi file."""
        cache = await self._load_cache()
        if session_key in cache:
            del cache[session_key]
            await self._save_cache()
            return True
        return False

    async def list_all(self) -> List[Session]:
        """Liệt kê tất cả sessions."""
        cache = await self._load_cache()
        return [Session.from_dict(data) for data in cache.values()]

    async def list_by_agent(self, agent_id: str) -> List[Session]:
        """Liệt kê sessions của một agent."""
        cache = await self._load_cache()
        return [
            Session.from_dict(data)
            for data in cache.values()
            if data.get("agent_id") == agent_id
        ]

    async def list_by_state(self, state: SessionState) -> List[Session]:
        """
        Liệt kê sessions theo trạng thái.

        Args:
            state: Trạng thái cần lọc

        Returns:
            Danh sách sessions có trạng thái tương ứng
        """
        cache = await self._load_cache()
        return [
            Session.from_dict(data)
            for data in cache.values()
            if data.get("state") == state.value
        ]

    async def cleanup_expired(self) -> int:
        """
        Xóa các sessions đã hết hạn.

        Returns:
            Số lượng sessions đã xóa
        """
        cache = await self._load_cache()
        expired_keys = [
            key for key, data in cache.items()
            if data.get("state") == SessionState.EXPIRED.value
        ]
        for key in expired_keys:
            del cache[key]

        if expired_keys:
            await self._save_cache()

        return len(expired_keys)

    def invalidate_cache(self) -> None:
        """Vô hiệu hóa cache để tải lại từ file."""
        self._cache = None


class CompositeSessionStore:
    """
    Session store kết hợp nhiều stores.

    Sử dụng in-memory store làm cache và file store làm persistence.
    Đọc từ cache trước, ghi vào cả hai.
    """

    def __init__(self, cache_store: InMemorySessionStore, persistent_store: FileSessionStore) -> None:
        """
        Khởi tạo CompositeSessionStore.

        Args:
            cache_store: Store trong bộ nhớ làm cache
            persistent_store: Store lưu trữ persistent
        """
        self._cache = cache_store
        self._persistent = persistent_store

    async def save(self, session: Session) -> None:
        """Lưu session vào cả cache và persistent store."""
        await self._cache.save(session)
        await self._persistent.save(session)

    async def load(self, session_key: str) -> Optional[Session]:
        """Tải session từ cache trước, nếu không có thì từ persistent."""
        session = await self._cache.load(session_key)
        if session is not None:
            return session

        session = await self._persistent.load(session_key)
        if session is not None:
            await self._cache.save(session)
        return session

    async def delete(self, session_key: str) -> bool:
        """Xóa session khỏi cả cache và persistent store."""
        cache_deleted = await self._cache.delete(session_key)
        persistent_deleted = await self._persistent.delete(session_key)
        return cache_deleted or persistent_deleted

    async def list_all(self) -> List[Session]:
        """Liệt kê tất cả sessions từ persistent store."""
        return await self._persistent.list_all()

    async def list_by_agent(self, agent_id: str) -> List[Session]:
        """Liệt kê sessions của một agent từ persistent store."""
        return await self._persistent.list_by_agent(agent_id)

    async def sync_cache(self) -> None:
        """Đồng bộ cache với persistent store."""
        await self._cache.clear()
        sessions = await self._persistent.list_all()
        for session in sessions:
            await self._cache.save(session)
