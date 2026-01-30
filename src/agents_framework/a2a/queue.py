"""
A2A Message Queue - Hàng đợi tin nhắn cho offline sessions.

Module này cung cấp MessageQueue để quản lý tin nhắn khi session đích
không khả dụng, hỗ trợ priority-based ordering và persistence.

Các tính năng chính:
- Priority-based message ordering (high > normal > low)
- Message expiration và cleanup
- Persistence support qua FileMessageQueue
- Thread-safe operations với asyncio locks

Ví dụ sử dụng:
    ```python
    from agents_framework.a2a.queue import InMemoryMessageQueue, MessageQueue

    # Tạo queue
    queue = InMemoryMessageQueue()

    # Thêm message
    await queue.enqueue(message)

    # Lấy messages cho session
    messages = await queue.get_messages_for_session("agent:coder:main")

    # Xử lý và xác nhận
    await queue.acknowledge(message.id)
    ```
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set, runtime_checkable

import aiofiles
import aiofiles.os

from .messaging import A2AMessage, MessagePriority, MessageState


@runtime_checkable
class MessageQueueProtocol(Protocol):
    """
    Protocol định nghĩa interface cho Message Queue.

    Đây là interface mà tất cả message queue implementations phải tuân theo.
    """

    async def enqueue(self, message: A2AMessage) -> bool:
        """Thêm message vào queue."""
        ...

    async def dequeue(self, session_key: str) -> Optional[A2AMessage]:
        """Lấy message tiếp theo cho session."""
        ...

    async def get_messages_for_session(
        self, session_key: str, limit: int = 10
    ) -> List[A2AMessage]:
        """Lấy danh sách messages cho session."""
        ...

    async def acknowledge(self, message_id: str) -> bool:
        """Xác nhận đã xử lý message."""
        ...

    async def reject(self, message_id: str, requeue: bool = False) -> bool:
        """Từ chối message."""
        ...


@dataclass
class QueueStats:
    """
    Thống kê về message queue.

    Attributes:
        total_messages: Tổng số messages trong queue
        pending_messages: Số messages đang chờ
        processing_messages: Số messages đang xử lý
        completed_messages: Số messages đã hoàn thành
        failed_messages: Số messages thất bại
        expired_messages: Số messages đã hết hạn
        messages_by_priority: Số messages theo priority
        oldest_message_age_ms: Tuổi của message cũ nhất (ms)
    """

    total_messages: int = 0
    pending_messages: int = 0
    processing_messages: int = 0
    completed_messages: int = 0
    failed_messages: int = 0
    expired_messages: int = 0
    messages_by_priority: Dict[str, int] = field(default_factory=dict)
    oldest_message_age_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển stats thành dictionary."""
        return {
            "total_messages": self.total_messages,
            "pending_messages": self.pending_messages,
            "processing_messages": self.processing_messages,
            "completed_messages": self.completed_messages,
            "failed_messages": self.failed_messages,
            "expired_messages": self.expired_messages,
            "messages_by_priority": self.messages_by_priority,
            "oldest_message_age_ms": self.oldest_message_age_ms,
        }


class BaseMessageQueue(ABC):
    """
    Abstract base class cho Message Queue.

    Cung cấp các phương thức utility chung cho tất cả queue implementations.
    """

    @abstractmethod
    async def enqueue(self, message: A2AMessage) -> bool:
        """
        Thêm message vào queue.

        Args:
            message: Message cần thêm

        Returns:
            True nếu thêm thành công
        """
        pass

    @abstractmethod
    async def dequeue(self, session_key: str) -> Optional[A2AMessage]:
        """
        Lấy message ưu tiên cao nhất cho session.

        Args:
            session_key: Key của session đích

        Returns:
            Message tiếp theo hoặc None nếu không có
        """
        pass

    @abstractmethod
    async def get_messages_for_session(
        self, session_key: str, limit: int = 10
    ) -> List[A2AMessage]:
        """
        Lấy danh sách messages cho session.

        Args:
            session_key: Key của session đích
            limit: Số lượng tối đa

        Returns:
            Danh sách messages đã sắp xếp theo priority
        """
        pass

    @abstractmethod
    async def acknowledge(self, message_id: str) -> bool:
        """
        Xác nhận đã xử lý message thành công.

        Args:
            message_id: ID của message

        Returns:
            True nếu xác nhận thành công
        """
        pass

    @abstractmethod
    async def reject(self, message_id: str, requeue: bool = False) -> bool:
        """
        Từ chối message.

        Args:
            message_id: ID của message
            requeue: True để đưa lại vào queue

        Returns:
            True nếu từ chối thành công
        """
        pass

    @abstractmethod
    async def get_stats(self) -> QueueStats:
        """
        Lấy thống kê queue.

        Returns:
            QueueStats với các thông tin thống kê
        """
        pass

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """
        Xóa các messages đã hết hạn.

        Returns:
            Số messages đã xóa
        """
        pass


class InMemoryMessageQueue(BaseMessageQueue):
    """
    Message queue lưu trữ trong bộ nhớ.

    Phù hợp cho development và testing. Dữ liệu sẽ mất
    khi process kết thúc.

    Attributes:
        _messages: Dictionary lưu messages theo session_key
        _all_messages: Dictionary lưu tất cả messages theo id
        _processing: Set các message_id đang xử lý
        _lock: Asyncio lock để đảm bảo thread-safety

    Ví dụ:
        queue = InMemoryMessageQueue()

        # Thêm message
        await queue.enqueue(message)

        # Lấy message tiếp theo
        next_msg = await queue.dequeue("agent:coder:main")

        # Xác nhận hoàn thành
        await queue.acknowledge(next_msg.id)
    """

    def __init__(self) -> None:
        """Khởi tạo queue với các containers rỗng."""
        # Messages grouped by destination session
        self._messages: Dict[str, List[A2AMessage]] = defaultdict(list)
        # All messages by ID for quick lookup
        self._all_messages: Dict[str, A2AMessage] = {}
        # Messages currently being processed
        self._processing: Set[str] = set()
        # Lock for thread-safety
        self._lock = asyncio.Lock()

    async def enqueue(self, message: A2AMessage) -> bool:
        """
        Thêm message vào queue.

        Message sẽ được thêm vào queue của session đích và sắp xếp
        theo priority (high > normal > low).

        Args:
            message: Message cần thêm

        Returns:
            True nếu thêm thành công, False nếu message đã tồn tại
        """
        async with self._lock:
            # Kiểm tra duplicate
            if message.id in self._all_messages:
                return False

            # Lưu message
            self._all_messages[message.id] = message

            # Thêm vào queue của session đích
            session_queue = self._messages[message.to_session]
            session_queue.append(message)

            # Sắp xếp theo priority (HIGH=0, NORMAL=1, LOW=2)
            priority_order = {
                MessagePriority.HIGH: 0,
                MessagePriority.NORMAL: 1,
                MessagePriority.LOW: 2,
            }
            session_queue.sort(key=lambda m: priority_order.get(m.priority, 1))

            return True

    async def dequeue(self, session_key: str) -> Optional[A2AMessage]:
        """
        Lấy message ưu tiên cao nhất cho session.

        Message sẽ được đánh dấu là đang xử lý nhưng chưa bị xóa
        khỏi queue cho đến khi được acknowledge.

        Args:
            session_key: Key của session đích

        Returns:
            Message có priority cao nhất, hoặc None nếu queue rỗng
        """
        async with self._lock:
            if session_key not in self._messages:
                return None

            session_queue = self._messages[session_key]

            # Tìm message đầu tiên chưa processing và chưa expired
            for message in session_queue:
                if message.id in self._processing:
                    continue
                if message.is_expired():
                    message.mark_expired()
                    continue

                # Đánh dấu đang xử lý
                self._processing.add(message.id)
                message.mark_processing()
                return message

            return None

    async def peek(self, session_key: str) -> Optional[A2AMessage]:
        """
        Xem message tiếp theo mà không lấy ra.

        Args:
            session_key: Key của session đích

        Returns:
            Message tiếp theo hoặc None
        """
        async with self._lock:
            if session_key not in self._messages:
                return None

            session_queue = self._messages[session_key]
            for message in session_queue:
                if message.id not in self._processing and not message.is_expired():
                    return message
            return None

    async def get_messages_for_session(
        self, session_key: str, limit: int = 10
    ) -> List[A2AMessage]:
        """
        Lấy danh sách messages đang chờ cho session.

        Args:
            session_key: Key của session đích
            limit: Số lượng tối đa

        Returns:
            Danh sách messages đã sắp xếp theo priority
        """
        async with self._lock:
            if session_key not in self._messages:
                return []

            session_queue = self._messages[session_key]
            result = []

            for message in session_queue:
                if message.is_expired():
                    continue
                if message.state in (MessageState.COMPLETED, MessageState.FAILED):
                    continue

                result.append(message)
                if len(result) >= limit:
                    break

            return result

    async def acknowledge(self, message_id: str) -> bool:
        """
        Xác nhận đã xử lý message thành công.

        Message sẽ bị xóa khỏi queue sau khi acknowledge.

        Args:
            message_id: ID của message

        Returns:
            True nếu xác nhận thành công
        """
        async with self._lock:
            message = self._all_messages.get(message_id)
            if message is None:
                return False

            # Cập nhật trạng thái
            message.mark_completed()

            # Xóa khỏi processing
            self._processing.discard(message_id)

            # Xóa khỏi queue của session
            session_queue = self._messages.get(message.to_session, [])
            self._messages[message.to_session] = [
                m for m in session_queue if m.id != message_id
            ]

            return True

    async def reject(self, message_id: str, requeue: bool = False) -> bool:
        """
        Từ chối message.

        Args:
            message_id: ID của message
            requeue: True để đưa lại vào queue

        Returns:
            True nếu từ chối thành công
        """
        async with self._lock:
            message = self._all_messages.get(message_id)
            if message is None:
                return False

            # Xóa khỏi processing
            self._processing.discard(message_id)

            if requeue:
                # Đưa lại vào queue với state PENDING
                message.state = MessageState.PENDING
            else:
                # Đánh dấu failed và xóa khỏi queue
                message.mark_failed("Rejected")
                session_queue = self._messages.get(message.to_session, [])
                self._messages[message.to_session] = [
                    m for m in session_queue if m.id != message_id
                ]

            return True

    async def get_message(self, message_id: str) -> Optional[A2AMessage]:
        """
        Lấy message theo ID.

        Args:
            message_id: ID của message

        Returns:
            Message hoặc None nếu không tìm thấy
        """
        async with self._lock:
            return self._all_messages.get(message_id)

    async def get_stats(self) -> QueueStats:
        """
        Lấy thống kê queue.

        Returns:
            QueueStats với các thông tin thống kê
        """
        async with self._lock:
            stats = QueueStats()
            priority_counts: Dict[str, int] = defaultdict(int)
            oldest_age_ms = 0
            now = datetime.utcnow()

            for message in self._all_messages.values():
                stats.total_messages += 1
                priority_counts[message.priority.value] += 1

                if message.state == MessageState.PENDING:
                    stats.pending_messages += 1
                elif message.state == MessageState.PROCESSING:
                    stats.processing_messages += 1
                elif message.state == MessageState.COMPLETED:
                    stats.completed_messages += 1
                elif message.state == MessageState.FAILED:
                    stats.failed_messages += 1
                elif message.state == MessageState.EXPIRED:
                    stats.expired_messages += 1

                # Tính tuổi message
                age_ms = int((now - message.created_at).total_seconds() * 1000)
                if age_ms > oldest_age_ms:
                    oldest_age_ms = age_ms

            stats.messages_by_priority = dict(priority_counts)
            stats.oldest_message_age_ms = oldest_age_ms

            return stats

    async def cleanup_expired(self) -> int:
        """
        Xóa các messages đã hết hạn.

        Returns:
            Số messages đã xóa
        """
        async with self._lock:
            expired_ids = []

            for message_id, message in self._all_messages.items():
                if message.is_expired():
                    message.mark_expired()
                    expired_ids.append(message_id)

            # Xóa các messages đã expired
            for message_id in expired_ids:
                message = self._all_messages.pop(message_id, None)
                if message:
                    session_queue = self._messages.get(message.to_session, [])
                    self._messages[message.to_session] = [
                        m for m in session_queue if m.id != message_id
                    ]
                self._processing.discard(message_id)

            return len(expired_ids)

    async def count_pending(self, session_key: str) -> int:
        """
        Đếm số messages đang chờ cho session.

        Args:
            session_key: Key của session

        Returns:
            Số messages pending
        """
        async with self._lock:
            if session_key not in self._messages:
                return 0

            return len([
                m for m in self._messages[session_key]
                if m.state == MessageState.PENDING and not m.is_expired()
            ])

    async def clear(self) -> None:
        """Xóa tất cả messages (dùng cho testing)."""
        async with self._lock:
            self._messages.clear()
            self._all_messages.clear()
            self._processing.clear()

    async def clear_session(self, session_key: str) -> int:
        """
        Xóa tất cả messages của một session.

        Args:
            session_key: Key của session

        Returns:
            Số messages đã xóa
        """
        async with self._lock:
            if session_key not in self._messages:
                return 0

            session_queue = self._messages.pop(session_key, [])
            count = len(session_queue)

            for message in session_queue:
                self._all_messages.pop(message.id, None)
                self._processing.discard(message.id)

            return count


class FileMessageQueue(BaseMessageQueue):
    """
    Message queue với persistence vào file.

    Sử dụng file JSON để lưu trữ messages, đảm bảo dữ liệu
    không bị mất khi restart.

    Attributes:
        _base_path: Đường dẫn thư mục lưu trữ
        _cache: InMemoryMessageQueue làm cache
        _lock: Asyncio lock

    Ví dụ:
        queue = FileMessageQueue(Path("/data/queues"))
        await queue.enqueue(message)
    """

    def __init__(self, base_path: Path) -> None:
        """
        Khởi tạo FileMessageQueue.

        Args:
            base_path: Đường dẫn thư mục lưu trữ
        """
        self._base_path = Path(base_path)
        self._cache = InMemoryMessageQueue()
        self._lock = asyncio.Lock()
        self._loaded = False

    def _get_queue_file(self) -> Path:
        """Lấy đường dẫn file queue."""
        return self._base_path / "message_queue.json"

    async def _ensure_directory(self) -> None:
        """Đảm bảo thư mục lưu trữ tồn tại."""
        if not self._base_path.exists():
            await aiofiles.os.makedirs(str(self._base_path), exist_ok=True)

    async def _load_from_file(self) -> None:
        """Tải messages từ file vào cache."""
        if self._loaded:
            return

        queue_file = self._get_queue_file()
        if not queue_file.exists():
            self._loaded = True
            return

        try:
            async with aiofiles.open(queue_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                if not content.strip():
                    self._loaded = True
                    return

                data = json.loads(content)
                for msg_data in data.get("messages", []):
                    message = A2AMessage.from_dict(msg_data)
                    # Chỉ load messages chưa hoàn thành
                    if message.state not in (
                        MessageState.COMPLETED, MessageState.FAILED, MessageState.EXPIRED
                    ):
                        await self._cache.enqueue(message)

        except (json.JSONDecodeError, OSError, KeyError):
            pass

        self._loaded = True

    async def _save_to_file(self) -> None:
        """Lưu messages từ cache vào file."""
        await self._ensure_directory()
        queue_file = self._get_queue_file()

        # Lấy tất cả messages từ cache
        messages_data = []
        stats = await self._cache.get_stats()

        # Duyệt qua cache và lấy messages
        async with self._cache._lock:
            for message in self._cache._all_messages.values():
                messages_data.append(message.to_dict())

        data = {
            "updated_at": datetime.utcnow().isoformat(),
            "stats": stats.to_dict(),
            "messages": messages_data,
        }

        async with aiofiles.open(queue_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(data, indent=2, ensure_ascii=False))

    async def enqueue(self, message: A2AMessage) -> bool:
        """
        Thêm message vào queue và persist.

        Args:
            message: Message cần thêm

        Returns:
            True nếu thêm thành công
        """
        async with self._lock:
            await self._load_from_file()
            result = await self._cache.enqueue(message)
            if result:
                await self._save_to_file()
            return result

    async def dequeue(self, session_key: str) -> Optional[A2AMessage]:
        """
        Lấy message tiếp theo và cập nhật file.

        Args:
            session_key: Key của session đích

        Returns:
            Message hoặc None
        """
        async with self._lock:
            await self._load_from_file()
            message = await self._cache.dequeue(session_key)
            if message:
                await self._save_to_file()
            return message

    async def get_messages_for_session(
        self, session_key: str, limit: int = 10
    ) -> List[A2AMessage]:
        """
        Lấy danh sách messages cho session.

        Args:
            session_key: Key của session đích
            limit: Số lượng tối đa

        Returns:
            Danh sách messages
        """
        async with self._lock:
            await self._load_from_file()
            return await self._cache.get_messages_for_session(session_key, limit)

    async def acknowledge(self, message_id: str) -> bool:
        """
        Xác nhận message và cập nhật file.

        Args:
            message_id: ID của message

        Returns:
            True nếu thành công
        """
        async with self._lock:
            await self._load_from_file()
            result = await self._cache.acknowledge(message_id)
            if result:
                await self._save_to_file()
            return result

    async def reject(self, message_id: str, requeue: bool = False) -> bool:
        """
        Từ chối message và cập nhật file.

        Args:
            message_id: ID của message
            requeue: True để đưa lại vào queue

        Returns:
            True nếu thành công
        """
        async with self._lock:
            await self._load_from_file()
            result = await self._cache.reject(message_id, requeue)
            await self._save_to_file()
            return result

    async def get_stats(self) -> QueueStats:
        """Lấy thống kê queue."""
        async with self._lock:
            await self._load_from_file()
            return await self._cache.get_stats()

    async def cleanup_expired(self) -> int:
        """
        Xóa messages hết hạn và cập nhật file.

        Returns:
            Số messages đã xóa
        """
        async with self._lock:
            await self._load_from_file()
            count = await self._cache.cleanup_expired()
            if count > 0:
                await self._save_to_file()
            return count

    async def reload(self) -> None:
        """Force reload từ file."""
        async with self._lock:
            self._loaded = False
            await self._cache.clear()
            await self._load_from_file()


# Type alias cho convenience
MessageQueue = InMemoryMessageQueue
