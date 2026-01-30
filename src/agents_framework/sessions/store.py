"""
Session Store Protocol và Implementations.

Module này định nghĩa SessionStore protocol - giao diện chuẩn cho việc
lưu trữ và truy xuất sessions, cùng với các implementation cơ bản.

Bao gồm:
- SessionStore: Protocol định nghĩa giao diện
- InMemorySessionStore: Lưu trữ trong bộ nhớ (development)
- FileSessionStore: Lưu trữ file JSON (simple persistence)
- SQLiteSessionStore: Lưu trữ SQLite (production)
- CompositeSessionStore: Kết hợp cache và persistence
"""

from abc import abstractmethod
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, runtime_checkable
import json
import sqlite3
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


class SQLiteSessionStore:
    """
    Session store sử dụng SQLite cho production.

    Cung cấp:
    - Persistence đáng tin cậy với SQLite
    - Connection pooling để tối ưu hiệu suất
    - Health checks để theo dõi trạng thái
    - Concurrent access với proper locking
    - Indexing cho fast lookup

    Schema:
        sessions: Bảng chính lưu trữ session metadata
        transcripts: Bảng lưu trữ bản ghi hội thoại (optional)

    Ví dụ:
        ```python
        store = SQLiteSessionStore(Path("./sessions.db"))
        await store.initialize()

        # Kiểm tra health
        if await store.health_check():
            session = Session.create(...)
            await store.save(session)
        ```
    """

    # SQL schema cho việc tạo bảng
    _CREATE_SESSIONS_TABLE = """
        CREATE TABLE IF NOT EXISTS sessions (
            session_key TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            context_tokens INTEGER DEFAULT 0,
            state TEXT DEFAULT 'active',
            metadata TEXT,
            UNIQUE(session_id)
        )
    """

    _CREATE_SESSIONS_AGENT_INDEX = """
        CREATE INDEX IF NOT EXISTS idx_sessions_agent ON sessions(agent_id)
    """

    _CREATE_SESSIONS_UPDATED_INDEX = """
        CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at)
    """

    _CREATE_SESSIONS_STATE_INDEX = """
        CREATE INDEX IF NOT EXISTS idx_sessions_state ON sessions(state)
    """

    _CREATE_TRANSCRIPTS_TABLE = """
        CREATE TABLE IF NOT EXISTS transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            metadata TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """

    _CREATE_TRANSCRIPTS_SESSION_INDEX = """
        CREATE INDEX IF NOT EXISTS idx_transcripts_session ON transcripts(session_id)
    """

    def __init__(
        self,
        db_path: Path,
        pool_size: int = 5,
        timeout: float = 30.0
    ) -> None:
        """
        Khởi tạo SQLiteSessionStore.

        Args:
            db_path: Đường dẫn đến file database SQLite
            pool_size: Số lượng connections trong pool (mặc định: 5)
            timeout: Thời gian timeout cho mỗi connection (giây, mặc định: 30)
        """
        self._db_path = Path(db_path)
        self._pool_size = pool_size
        self._timeout = timeout
        self._pool: List[sqlite3.Connection] = []
        self._pool_lock = asyncio.Lock()
        self._initialized = False
        self._executor_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """
        Khởi tạo database và connection pool.

        Tạo database file nếu chưa tồn tại, tạo các bảng và indexes,
        và khởi tạo connection pool.
        """
        if self._initialized:
            return

        # Đảm bảo thư mục tồn tại
        if not self._db_path.parent.exists():
            await aiofiles.os.makedirs(str(self._db_path.parent), exist_ok=True)

        # Tạo schema
        conn = await self._create_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(self._CREATE_SESSIONS_TABLE)
            cursor.execute(self._CREATE_SESSIONS_AGENT_INDEX)
            cursor.execute(self._CREATE_SESSIONS_UPDATED_INDEX)
            cursor.execute(self._CREATE_SESSIONS_STATE_INDEX)
            cursor.execute(self._CREATE_TRANSCRIPTS_TABLE)
            cursor.execute(self._CREATE_TRANSCRIPTS_SESSION_INDEX)
            conn.commit()
        finally:
            conn.close()

        # Khởi tạo connection pool
        for _ in range(self._pool_size):
            self._pool.append(await self._create_connection())

        self._initialized = True

    async def _create_connection(self) -> sqlite3.Connection:
        """
        Tạo một connection mới đến database.

        Returns:
            Connection mới với cấu hình tối ưu
        """
        loop = asyncio.get_event_loop()

        def _connect() -> sqlite3.Connection:
            conn = sqlite3.connect(
                str(self._db_path),
                timeout=self._timeout,
                check_same_thread=False
            )
            # Cấu hình tối ưu cho SQLite
            conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
            conn.execute("PRAGMA synchronous=NORMAL")  # Cân bằng giữa safety và speed
            conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            conn.execute("PRAGMA busy_timeout=30000")  # 30s busy timeout
            conn.row_factory = sqlite3.Row
            return conn

        return await loop.run_in_executor(None, _connect)

    @asynccontextmanager
    async def _get_connection(self) -> AsyncIterator[sqlite3.Connection]:
        """
        Lấy connection từ pool (context manager).

        Yields:
            Connection từ pool, tự động trả lại khi xong
        """
        async with self._pool_lock:
            if not self._pool:
                # Pool rỗng, tạo connection mới
                conn = await self._create_connection()
            else:
                conn = self._pool.pop()

        try:
            yield conn
        finally:
            async with self._pool_lock:
                if len(self._pool) < self._pool_size:
                    self._pool.append(conn)
                else:
                    conn.close()

    async def _execute(
        self,
        query: str,
        params: tuple = ()
    ) -> List[sqlite3.Row]:
        """
        Thực thi query và trả về kết quả.

        Args:
            query: SQL query cần thực thi
            params: Parameters cho query

        Returns:
            Danh sách các rows kết quả
        """
        loop = asyncio.get_event_loop()

        async with self._get_connection() as conn:
            def _run() -> List[sqlite3.Row]:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                return cursor.fetchall()

            async with self._executor_lock:
                return await loop.run_in_executor(None, _run)

    async def _execute_many(
        self,
        query: str,
        params_list: List[tuple]
    ) -> int:
        """
        Thực thi query với nhiều bộ parameters.

        Args:
            query: SQL query cần thực thi
            params_list: Danh sách parameters

        Returns:
            Số lượng rows bị ảnh hưởng
        """
        loop = asyncio.get_event_loop()

        async with self._get_connection() as conn:
            def _run() -> int:
                cursor = conn.cursor()
                cursor.executemany(query, params_list)
                conn.commit()
                return cursor.rowcount

            async with self._executor_lock:
                return await loop.run_in_executor(None, _run)

    async def save(self, session: Session) -> None:
        """
        Lưu hoặc cập nhật session.

        Sử dụng INSERT OR REPLACE để xử lý cả insert và update.

        Args:
            session: Session cần lưu
        """
        if not self._initialized:
            await self.initialize()

        query = """
            INSERT OR REPLACE INTO sessions
            (session_key, session_id, agent_id, created_at, updated_at,
             input_tokens, output_tokens, context_tokens, state, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            session.session_key,
            session.session_id,
            session.agent_id,
            session.created_at.isoformat(),
            session.updated_at.isoformat(),
            session.input_tokens,
            session.output_tokens,
            session.context_tokens,
            session.state.value,
            json.dumps(session.metadata, ensure_ascii=False)
        )
        await self._execute(query, params)

    async def load(self, session_key: str) -> Optional[Session]:
        """
        Tải session theo key.

        Args:
            session_key: Key của session cần tải

        Returns:
            Session nếu tìm thấy, None nếu không
        """
        if not self._initialized:
            await self.initialize()

        query = "SELECT * FROM sessions WHERE session_key = ?"
        rows = await self._execute(query, (session_key,))

        if not rows:
            return None

        return self._row_to_session(rows[0])

    async def load_by_id(self, session_id: str) -> Optional[Session]:
        """
        Tải session theo session_id.

        Args:
            session_id: ID của session cần tải

        Returns:
            Session nếu tìm thấy, None nếu không
        """
        if not self._initialized:
            await self.initialize()

        query = "SELECT * FROM sessions WHERE session_id = ?"
        rows = await self._execute(query, (session_id,))

        if not rows:
            return None

        return self._row_to_session(rows[0])

    async def delete(self, session_key: str) -> bool:
        """
        Xóa session theo key.

        Xóa cả session và các transcripts liên quan.

        Args:
            session_key: Key của session cần xóa

        Returns:
            True nếu xóa thành công, False nếu không tìm thấy
        """
        if not self._initialized:
            await self.initialize()

        # Lấy session_id để xóa transcripts
        session = await self.load(session_key)
        if session is None:
            return False

        # Xóa transcripts trước
        await self._execute(
            "DELETE FROM transcripts WHERE session_id = ?",
            (session.session_id,)
        )

        # Xóa session
        await self._execute(
            "DELETE FROM sessions WHERE session_key = ?",
            (session_key,)
        )

        return True

    async def list_all(self) -> List[Session]:
        """
        Liệt kê tất cả sessions.

        Returns:
            Danh sách tất cả sessions, sắp xếp theo updated_at giảm dần
        """
        if not self._initialized:
            await self.initialize()

        query = "SELECT * FROM sessions ORDER BY updated_at DESC"
        rows = await self._execute(query)

        return [self._row_to_session(row) for row in rows]

    async def list_by_agent(self, agent_id: str) -> List[Session]:
        """
        Liệt kê sessions theo agent ID.

        Args:
            agent_id: ID của agent

        Returns:
            Danh sách sessions của agent, sắp xếp theo updated_at giảm dần
        """
        if not self._initialized:
            await self.initialize()

        query = """
            SELECT * FROM sessions
            WHERE agent_id = ?
            ORDER BY updated_at DESC
        """
        rows = await self._execute(query, (agent_id,))

        return [self._row_to_session(row) for row in rows]

    async def list_by_state(self, state: SessionState) -> List[Session]:
        """
        Liệt kê sessions theo trạng thái.

        Args:
            state: Trạng thái cần lọc

        Returns:
            Danh sách sessions có trạng thái tương ứng
        """
        if not self._initialized:
            await self.initialize()

        query = """
            SELECT * FROM sessions
            WHERE state = ?
            ORDER BY updated_at DESC
        """
        rows = await self._execute(query, (state.value,))

        return [self._row_to_session(row) for row in rows]

    async def cleanup_expired(self, before: Optional[datetime] = None) -> int:
        """
        Xóa các sessions đã hết hạn hoặc cũ hơn thời điểm chỉ định.

        Args:
            before: Xóa sessions updated trước thời điểm này.
                   Nếu None, xóa tất cả sessions có state=expired

        Returns:
            Số lượng sessions đã xóa
        """
        if not self._initialized:
            await self.initialize()

        # Đếm số sessions sẽ bị xóa trước
        if before is not None:
            count_rows = await self._execute(
                "SELECT COUNT(*) as count FROM sessions WHERE updated_at < ?",
                (before.isoformat(),)
            )
        else:
            count_rows = await self._execute(
                "SELECT COUNT(*) as count FROM sessions WHERE state = 'expired'"
            )

        count = count_rows[0]["count"] if count_rows else 0

        if count == 0:
            return 0

        if before is not None:
            # Xóa transcripts tương ứng trước
            await self._execute(
                """
                DELETE FROM transcripts
                WHERE session_id IN (
                    SELECT session_id FROM sessions WHERE updated_at < ?
                )
                """,
                (before.isoformat(),)
            )
            # Xóa sessions cũ hơn thời điểm chỉ định
            await self._execute(
                "DELETE FROM sessions WHERE updated_at < ?",
                (before.isoformat(),)
            )
        else:
            # Xóa transcripts của sessions expired trước
            await self._execute(
                """
                DELETE FROM transcripts
                WHERE session_id IN (
                    SELECT session_id FROM sessions WHERE state = 'expired'
                )
                """
            )
            # Xóa sessions có state = expired
            await self._execute(
                "DELETE FROM sessions WHERE state = 'expired'"
            )

        return count

    async def count(self) -> int:
        """
        Đếm tổng số sessions.

        Returns:
            Số lượng sessions trong database
        """
        if not self._initialized:
            await self.initialize()

        query = "SELECT COUNT(*) as count FROM sessions"
        rows = await self._execute(query)
        return rows[0]["count"] if rows else 0

    async def count_by_agent(self, agent_id: str) -> int:
        """
        Đếm số sessions của một agent.

        Args:
            agent_id: ID của agent

        Returns:
            Số lượng sessions của agent
        """
        if not self._initialized:
            await self.initialize()

        query = "SELECT COUNT(*) as count FROM sessions WHERE agent_id = ?"
        rows = await self._execute(query, (agent_id,))
        return rows[0]["count"] if rows else 0

    async def health_check(self) -> bool:
        """
        Kiểm tra sức khỏe của database connection.

        Thực hiện một query đơn giản để đảm bảo database hoạt động.

        Returns:
            True nếu database hoạt động bình thường, False nếu có lỗi
        """
        try:
            if not self._initialized:
                await self.initialize()

            query = "SELECT 1 as ping"
            rows = await self._execute(query)
            return len(rows) == 1 and rows[0]["ping"] == 1
        except Exception:
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê về database.

        Returns:
            Dictionary chứa các thống kê:
            - total_sessions: Tổng số sessions
            - active_sessions: Số sessions đang active
            - idle_sessions: Số sessions idle
            - expired_sessions: Số sessions expired
            - pool_size: Kích thước connection pool hiện tại
            - db_path: Đường dẫn database
        """
        if not self._initialized:
            await self.initialize()

        total = await self.count()

        active_rows = await self._execute(
            "SELECT COUNT(*) as count FROM sessions WHERE state = 'active'"
        )
        active = active_rows[0]["count"] if active_rows else 0

        idle_rows = await self._execute(
            "SELECT COUNT(*) as count FROM sessions WHERE state = 'idle'"
        )
        idle = idle_rows[0]["count"] if idle_rows else 0

        expired_rows = await self._execute(
            "SELECT COUNT(*) as count FROM sessions WHERE state = 'expired'"
        )
        expired = expired_rows[0]["count"] if expired_rows else 0

        return {
            "total_sessions": total,
            "active_sessions": active,
            "idle_sessions": idle,
            "expired_sessions": expired,
            "pool_size": len(self._pool),
            "db_path": str(self._db_path)
        }

    async def vacuum(self) -> None:
        """
        Tối ưu hóa database bằng VACUUM.

        Nên chạy định kỳ sau khi xóa nhiều records để thu hồi không gian.
        """
        if not self._initialized:
            await self.initialize()

        await self._execute("VACUUM")

    async def close(self) -> None:
        """
        Đóng tất cả connections trong pool.

        Nên gọi khi không còn sử dụng store nữa.
        """
        async with self._pool_lock:
            for conn in self._pool:
                conn.close()
            self._pool.clear()
            self._initialized = False

    def _row_to_session(self, row: sqlite3.Row) -> Session:
        """
        Chuyển đổi database row thành Session object.

        Args:
            row: Row từ database

        Returns:
            Session object
        """
        metadata = {}
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except json.JSONDecodeError:
                metadata = {}

        return Session(
            session_id=row["session_id"],
            session_key=row["session_key"],
            agent_id=row["agent_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            context_tokens=row["context_tokens"],
            state=SessionState(row["state"]),
            metadata=metadata
        )

    async def __aenter__(self) -> "SQLiteSessionStore":
        """Context manager entry - khởi tạo store."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - đóng connections."""
        await self.close()
