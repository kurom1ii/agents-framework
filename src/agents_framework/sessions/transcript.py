"""
Transcript Store - Lưu trữ bản ghi hội thoại.

Module này cung cấp TranscriptStore để lưu trữ lịch sử hội thoại
của mỗi session theo định dạng JSONL (JSON Lines).
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import json
import aiofiles
import aiofiles.os


@dataclass
class TranscriptEntry:
    """
    Một entry trong bản ghi hội thoại.

    Attributes:
        role: Vai trò của người gửi (user, assistant, system, tool)
        content: Nội dung tin nhắn
        timestamp: Thời điểm gửi
        entry_id: ID duy nhất của entry (tự động tạo)
        metadata: Dữ liệu bổ sung (tool_calls, model info, v.v.)
    """
    role: str
    content: str
    timestamp: datetime
    entry_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Tự động tạo entry_id nếu chưa có."""
        if self.entry_id is None:
            import uuid
            self.entry_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển entry thành dictionary để lưu trữ.

        Returns:
            Dictionary chứa thông tin entry
        """
        return {
            "entry_id": self.entry_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

    def to_jsonl(self) -> str:
        """
        Chuyển entry thành JSON string (một dòng).

        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptEntry":
        """
        Tạo entry từ dictionary.

        Args:
            data: Dictionary chứa thông tin entry

        Returns:
            TranscriptEntry được khôi phục
        """
        return cls(
            entry_id=data.get("entry_id"),
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )

    @classmethod
    def from_jsonl(cls, line: str) -> "TranscriptEntry":
        """
        Tạo entry từ JSON string.

        Args:
            line: JSON string (một dòng)

        Returns:
            TranscriptEntry được khôi phục
        """
        data = json.loads(line)
        return cls.from_dict(data)

    @classmethod
    def user(cls, content: str, metadata: Optional[Dict[str, Any]] = None) -> "TranscriptEntry":
        """
        Tạo entry từ user.

        Args:
            content: Nội dung tin nhắn
            metadata: Metadata tùy chọn

        Returns:
            TranscriptEntry với role='user'
        """
        return cls(
            role="user",
            content=content,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )

    @classmethod
    def assistant(cls, content: str, metadata: Optional[Dict[str, Any]] = None) -> "TranscriptEntry":
        """
        Tạo entry từ assistant.

        Args:
            content: Nội dung tin nhắn
            metadata: Metadata tùy chọn

        Returns:
            TranscriptEntry với role='assistant'
        """
        return cls(
            role="assistant",
            content=content,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )

    @classmethod
    def system(cls, content: str, metadata: Optional[Dict[str, Any]] = None) -> "TranscriptEntry":
        """
        Tạo entry từ system.

        Args:
            content: Nội dung tin nhắn
            metadata: Metadata tùy chọn

        Returns:
            TranscriptEntry với role='system'
        """
        return cls(
            role="system",
            content=content,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )

    @classmethod
    def tool(
        cls,
        content: str,
        tool_name: str,
        tool_call_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "TranscriptEntry":
        """
        Tạo entry từ tool result.

        Args:
            content: Kết quả tool
            tool_name: Tên tool
            tool_call_id: ID của tool call
            metadata: Metadata tùy chọn

        Returns:
            TranscriptEntry với role='tool'
        """
        meta = metadata or {}
        meta["tool_name"] = tool_name
        if tool_call_id:
            meta["tool_call_id"] = tool_call_id

        return cls(
            role="tool",
            content=content,
            timestamp=datetime.utcnow(),
            metadata=meta
        )


@runtime_checkable
class TranscriptStoreProtocol(Protocol):
    """Protocol định nghĩa giao diện TranscriptStore."""

    async def append(self, session_id: str, entry: TranscriptEntry) -> None:
        """Thêm entry vào bản ghi."""
        ...

    async def get_entries(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[TranscriptEntry]:
        """Lấy các entries của session."""
        ...

    async def clear(self, session_id: str) -> None:
        """Xóa tất cả entries của session."""
        ...


class InMemoryTranscriptStore:
    """
    Transcript store lưu trữ trong bộ nhớ.

    Phù hợp cho testing và các ứng dụng không cần persistence.
    """

    def __init__(self) -> None:
        """Khởi tạo store với dictionary rỗng."""
        self._transcripts: Dict[str, List[TranscriptEntry]] = {}

    async def append(self, session_id: str, entry: TranscriptEntry) -> None:
        """
        Thêm entry vào bản ghi của session.

        Args:
            session_id: ID của session
            entry: Entry cần thêm
        """
        if session_id not in self._transcripts:
            self._transcripts[session_id] = []
        self._transcripts[session_id].append(entry)

    async def get_entries(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[TranscriptEntry]:
        """
        Lấy các entries của session.

        Args:
            session_id: ID của session
            limit: Số lượng entries tối đa
            offset: Bỏ qua bao nhiêu entries đầu tiên

        Returns:
            Danh sách entries
        """
        entries = self._transcripts.get(session_id, [])
        return entries[offset:offset + limit]

    async def get_recent(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[TranscriptEntry]:
        """
        Lấy các entries gần nhất của session.

        Args:
            session_id: ID của session
            limit: Số lượng entries tối đa

        Returns:
            Danh sách entries gần nhất
        """
        entries = self._transcripts.get(session_id, [])
        if len(entries) <= limit:
            return entries
        return entries[-limit:]

    async def clear(self, session_id: str) -> None:
        """
        Xóa tất cả entries của session.

        Args:
            session_id: ID của session
        """
        if session_id in self._transcripts:
            del self._transcripts[session_id]

    async def count(self, session_id: str) -> int:
        """
        Đếm số entries của session.

        Args:
            session_id: ID của session

        Returns:
            Số lượng entries
        """
        return len(self._transcripts.get(session_id, []))

    async def clear_all(self) -> None:
        """Xóa tất cả transcripts (dùng cho testing)."""
        self._transcripts.clear()


class FileTranscriptStore:
    """
    Transcript store lưu trữ vào file JSONL.

    Mỗi session được lưu trong một file riêng với format JSONL,
    trong đó mỗi dòng là một JSON object đại diện cho một entry.

    Cấu trúc thư mục:
        base_path/
        ├── <session_id_1>.jsonl
        ├── <session_id_2>.jsonl
        └── ...
    """

    def __init__(self, base_path: Path) -> None:
        """
        Khởi tạo FileTranscriptStore.

        Args:
            base_path: Đường dẫn thư mục lưu trữ
        """
        self._base_path = Path(base_path)

    def _get_file_path(self, session_id: str) -> Path:
        """Lấy đường dẫn file cho session."""
        # Sanitize session_id để dùng làm tên file
        safe_id = session_id.replace(":", "_").replace("/", "_")
        return self._base_path / f"{safe_id}.jsonl"

    async def _ensure_directory(self) -> None:
        """Đảm bảo thư mục lưu trữ tồn tại."""
        if not self._base_path.exists():
            await aiofiles.os.makedirs(str(self._base_path), exist_ok=True)

    async def append(self, session_id: str, entry: TranscriptEntry) -> None:
        """
        Thêm entry vào file JSONL của session.

        Args:
            session_id: ID của session
            entry: Entry cần thêm
        """
        await self._ensure_directory()
        file_path = self._get_file_path(session_id)

        async with aiofiles.open(file_path, 'a', encoding='utf-8') as f:
            await f.write(entry.to_jsonl() + "\n")

    async def append_batch(self, session_id: str, entries: List[TranscriptEntry]) -> None:
        """
        Thêm nhiều entries cùng lúc.

        Args:
            session_id: ID của session
            entries: Danh sách entries cần thêm
        """
        if not entries:
            return

        await self._ensure_directory()
        file_path = self._get_file_path(session_id)

        lines = [entry.to_jsonl() for entry in entries]
        async with aiofiles.open(file_path, 'a', encoding='utf-8') as f:
            await f.write("\n".join(lines) + "\n")

    async def get_entries(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[TranscriptEntry]:
        """
        Lấy các entries của session từ file.

        Args:
            session_id: ID của session
            limit: Số lượng entries tối đa
            offset: Bỏ qua bao nhiêu entries đầu tiên

        Returns:
            Danh sách entries
        """
        file_path = self._get_file_path(session_id)
        if not file_path.exists():
            return []

        entries: List[TranscriptEntry] = []
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                line_num = 0
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    if line_num < offset:
                        line_num += 1
                        continue

                    if len(entries) >= limit:
                        break

                    try:
                        entry = TranscriptEntry.from_jsonl(line)
                        entries.append(entry)
                    except (json.JSONDecodeError, KeyError):
                        # Skip invalid lines
                        pass

                    line_num += 1
        except OSError:
            pass

        return entries

    async def get_recent(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[TranscriptEntry]:
        """
        Lấy các entries gần nhất của session.

        Args:
            session_id: ID của session
            limit: Số lượng entries tối đa

        Returns:
            Danh sách entries gần nhất
        """
        file_path = self._get_file_path(session_id)
        if not file_path.exists():
            return []

        # Đọc tất cả entries rồi lấy limit cuối cùng
        all_entries: List[TranscriptEntry] = []
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = TranscriptEntry.from_jsonl(line)
                        all_entries.append(entry)
                    except (json.JSONDecodeError, KeyError):
                        pass
        except OSError:
            pass

        if len(all_entries) <= limit:
            return all_entries
        return all_entries[-limit:]

    async def get_by_role(
        self,
        session_id: str,
        role: str,
        limit: int = 100
    ) -> List[TranscriptEntry]:
        """
        Lấy các entries theo role.

        Args:
            session_id: ID của session
            role: Role cần lọc (user, assistant, system, tool)
            limit: Số lượng entries tối đa

        Returns:
            Danh sách entries có role tương ứng
        """
        all_entries = await self.get_entries(session_id, limit=10000)
        filtered = [e for e in all_entries if e.role == role]
        return filtered[:limit]

    async def get_since(
        self,
        session_id: str,
        since: datetime,
        limit: int = 100
    ) -> List[TranscriptEntry]:
        """
        Lấy các entries từ một thời điểm.

        Args:
            session_id: ID của session
            since: Thời điểm bắt đầu
            limit: Số lượng entries tối đa

        Returns:
            Danh sách entries từ thời điểm đã chỉ định
        """
        all_entries = await self.get_entries(session_id, limit=10000)
        filtered = [e for e in all_entries if e.timestamp >= since]
        return filtered[:limit]

    async def clear(self, session_id: str) -> None:
        """
        Xóa file transcript của session.

        Args:
            session_id: ID của session
        """
        file_path = self._get_file_path(session_id)
        if file_path.exists():
            await aiofiles.os.remove(str(file_path))

    async def count(self, session_id: str) -> int:
        """
        Đếm số entries của session.

        Args:
            session_id: ID của session

        Returns:
            Số lượng entries
        """
        file_path = self._get_file_path(session_id)
        if not file_path.exists():
            return 0

        count = 0
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                async for line in f:
                    if line.strip():
                        count += 1
        except OSError:
            pass

        return count

    async def list_sessions(self) -> List[str]:
        """
        Liệt kê tất cả session IDs có transcript.

        Returns:
            Danh sách session IDs
        """
        if not self._base_path.exists():
            return []

        sessions = []
        for file_path in self._base_path.glob("*.jsonl"):
            # Chuyển filename về session_id
            session_id = file_path.stem.replace("_", ":")
            sessions.append(session_id)

        return sessions

    async def archive(self, session_id: str, archive_path: Path) -> bool:
        """
        Lưu trữ transcript vào thư mục archive.

        Args:
            session_id: ID của session
            archive_path: Đường dẫn thư mục archive

        Returns:
            True nếu archive thành công
        """
        file_path = self._get_file_path(session_id)
        if not file_path.exists():
            return False

        if not archive_path.exists():
            await aiofiles.os.makedirs(str(archive_path), exist_ok=True)

        # Tạo tên file archive với timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_id = session_id.replace(":", "_").replace("/", "_")
        archive_file = archive_path / f"{safe_id}_{timestamp}.jsonl"

        # Copy file
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as src:
            content = await src.read()
        async with aiofiles.open(archive_file, 'w', encoding='utf-8') as dst:
            await dst.write(content)

        return True


class TranscriptStore:
    """
    Facade class kết hợp InMemory cache và File storage.

    Sử dụng in-memory store làm cache để tăng tốc đọc,
    và file store để đảm bảo persistence.
    """

    def __init__(
        self,
        base_path: Optional[Path] = None,
        cache_enabled: bool = True
    ) -> None:
        """
        Khởi tạo TranscriptStore.

        Args:
            base_path: Đường dẫn thư mục lưu trữ (None để chỉ dùng memory)
            cache_enabled: Bật/tắt cache trong bộ nhớ
        """
        self._cache = InMemoryTranscriptStore() if cache_enabled else None
        self._file_store = FileTranscriptStore(base_path) if base_path else None

    async def append(self, session_id: str, entry: TranscriptEntry) -> None:
        """
        Thêm entry vào transcript.

        Args:
            session_id: ID của session
            entry: Entry cần thêm
        """
        if self._cache:
            await self._cache.append(session_id, entry)
        if self._file_store:
            await self._file_store.append(session_id, entry)

    async def get_entries(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[TranscriptEntry]:
        """
        Lấy các entries của session.

        Args:
            session_id: ID của session
            limit: Số lượng entries tối đa
            offset: Bỏ qua bao nhiêu entries đầu tiên

        Returns:
            Danh sách entries
        """
        # Ưu tiên đọc từ cache
        if self._cache:
            entries = await self._cache.get_entries(session_id, limit, offset)
            if entries:
                return entries

        # Fallback to file store
        if self._file_store:
            return await self._file_store.get_entries(session_id, limit, offset)

        return []

    async def get_recent(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[TranscriptEntry]:
        """
        Lấy các entries gần nhất.

        Args:
            session_id: ID của session
            limit: Số lượng entries tối đa

        Returns:
            Danh sách entries gần nhất
        """
        if self._cache:
            entries = await self._cache.get_recent(session_id, limit)
            if entries:
                return entries

        if self._file_store:
            return await self._file_store.get_recent(session_id, limit)

        return []

    async def clear(self, session_id: str) -> None:
        """
        Xóa tất cả entries của session.

        Args:
            session_id: ID của session
        """
        if self._cache:
            await self._cache.clear(session_id)
        if self._file_store:
            await self._file_store.clear(session_id)

    async def count(self, session_id: str) -> int:
        """
        Đếm số entries của session.

        Args:
            session_id: ID của session

        Returns:
            Số lượng entries
        """
        if self._cache:
            count = await self._cache.count(session_id)
            if count > 0:
                return count

        if self._file_store:
            return await self._file_store.count(session_id)

        return 0
