"""
A2A Message Router - Định tuyến tin nhắn giữa các sessions.

Module này cung cấp MessageRouter để định tuyến messages đến các sessions đích,
hỗ trợ 3 chế độ routing:
- Direct: Gửi đến session cụ thể
- Broadcast: Gửi đến tất cả sessions của một agent
- Topic-based: Gửi đến các sessions đã subscribe topic

Các tính năng chính:
- Synchronous mode (đợi response) và Asynchronous mode
- Message queuing cho offline sessions
- Topic subscription management
- Response correlation

Ví dụ sử dụng:
    ```python
    from agents_framework.a2a.router import MessageRouter, InterSessionMessaging

    # Tạo router
    messaging = InterSessionMessaging(session_manager, message_queue)

    # Gửi request và đợi response
    response = await messaging.send(
        from_session="agent:orchestrator:main",
        to_session="agent:coder:main",
        message="Review PR #123",
        wait_for_response=True
    )

    # Broadcast đến tất cả sessions của agent
    responses = await messaging.broadcast(
        from_session="agent:admin:main",
        agent_id="worker",
        message="New task available"
    )
    ```
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Protocol, Set

from .base import SessionInfo, SessionStatus
from .discovery import SessionDiscovery, SessionManagerProtocol
from .messaging import (
    A2AMessage,
    A2AResponse,
    MessagePriority,
    MessageState,
    MessageType,
)
from .queue import InMemoryMessageQueue, MessageQueue, MessageQueueProtocol


class MessageHandlerProtocol(Protocol):
    """
    Protocol định nghĩa handler xử lý messages.

    Các components xử lý messages phải implement protocol này.
    """

    async def handle_message(self, message: A2AMessage) -> Optional[A2AResponse]:
        """
        Xử lý message và trả về response (nếu có).

        Args:
            message: Message cần xử lý

        Returns:
            A2AResponse cho request messages, None cho notifications/events
        """
        ...


@dataclass
class RoutingResult:
    """
    Kết quả của quá trình routing message.

    Attributes:
        message_id: ID của message
        routing_type: Loại routing (direct, broadcast, topic)
        target_sessions: Danh sách sessions đích
        delivered_count: Số sessions đã nhận message
        queued_count: Số sessions message được queue (offline)
        failed_count: Số sessions gửi thất bại
        errors: Chi tiết lỗi nếu có
    """

    message_id: str
    routing_type: str
    target_sessions: List[str] = field(default_factory=list)
    delivered_count: int = 0
    queued_count: int = 0
    failed_count: int = 0
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Kiểm tra routing có thành công không."""
        return self.delivered_count > 0 or self.queued_count > 0

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển thành dictionary."""
        return {
            "message_id": self.message_id,
            "routing_type": self.routing_type,
            "target_sessions": self.target_sessions,
            "delivered_count": self.delivered_count,
            "queued_count": self.queued_count,
            "failed_count": self.failed_count,
            "errors": self.errors,
        }


class TopicRegistry:
    """
    Registry quản lý topic subscriptions.

    Cho phép sessions subscribe và unsubscribe từ các topics,
    và tìm kiếm sessions theo topic.

    Attributes:
        _subscriptions: Mapping topic -> set of session_keys
        _session_topics: Mapping session_key -> set of topics

    Ví dụ:
        registry = TopicRegistry()

        # Subscribe
        registry.subscribe("agent:worker:main", "task.new")
        registry.subscribe("agent:worker:main", "task.completed")

        # Lấy subscribers
        subscribers = registry.get_subscribers("task.new")

        # Unsubscribe
        registry.unsubscribe("agent:worker:main", "task.new")
    """

    def __init__(self) -> None:
        """Khởi tạo registry với containers rỗng."""
        # Topic -> Sessions subscribed
        self._subscriptions: Dict[str, Set[str]] = {}
        # Session -> Topics subscribed
        self._session_topics: Dict[str, Set[str]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, session_key: str, topic: str) -> bool:
        """
        Subscribe session vào topic.

        Args:
            session_key: Key của session
            topic: Topic để subscribe

        Returns:
            True nếu subscription mới, False nếu đã subscribe trước đó
        """
        async with self._lock:
            # Thêm vào subscriptions
            if topic not in self._subscriptions:
                self._subscriptions[topic] = set()

            was_subscribed = session_key in self._subscriptions[topic]
            self._subscriptions[topic].add(session_key)

            # Thêm vào session_topics
            if session_key not in self._session_topics:
                self._session_topics[session_key] = set()
            self._session_topics[session_key].add(topic)

            return not was_subscribed

    async def unsubscribe(self, session_key: str, topic: str) -> bool:
        """
        Unsubscribe session khỏi topic.

        Args:
            session_key: Key của session
            topic: Topic để unsubscribe

        Returns:
            True nếu đã unsubscribe, False nếu chưa từng subscribe
        """
        async with self._lock:
            # Xóa khỏi subscriptions
            if topic in self._subscriptions:
                was_subscribed = session_key in self._subscriptions[topic]
                self._subscriptions[topic].discard(session_key)
                if not self._subscriptions[topic]:
                    del self._subscriptions[topic]

                # Xóa khỏi session_topics
                if session_key in self._session_topics:
                    self._session_topics[session_key].discard(topic)
                    if not self._session_topics[session_key]:
                        del self._session_topics[session_key]

                return was_subscribed
            return False

    async def unsubscribe_all(self, session_key: str) -> int:
        """
        Unsubscribe session khỏi tất cả topics.

        Args:
            session_key: Key của session

        Returns:
            Số topics đã unsubscribe
        """
        async with self._lock:
            if session_key not in self._session_topics:
                return 0

            topics = list(self._session_topics[session_key])
            count = len(topics)

            for topic in topics:
                if topic in self._subscriptions:
                    self._subscriptions[topic].discard(session_key)
                    if not self._subscriptions[topic]:
                        del self._subscriptions[topic]

            del self._session_topics[session_key]
            return count

    async def get_subscribers(self, topic: str) -> List[str]:
        """
        Lấy danh sách sessions đã subscribe topic.

        Args:
            topic: Topic cần tìm

        Returns:
            Danh sách session_keys
        """
        async with self._lock:
            return list(self._subscriptions.get(topic, set()))

    async def get_subscribed_topics(self, session_key: str) -> List[str]:
        """
        Lấy danh sách topics mà session đã subscribe.

        Args:
            session_key: Key của session

        Returns:
            Danh sách topics
        """
        async with self._lock:
            return list(self._session_topics.get(session_key, set()))

    async def is_subscribed(self, session_key: str, topic: str) -> bool:
        """
        Kiểm tra session có subscribe topic không.

        Args:
            session_key: Key của session
            topic: Topic cần kiểm tra

        Returns:
            True nếu đã subscribe
        """
        async with self._lock:
            return (
                topic in self._subscriptions
                and session_key in self._subscriptions[topic]
            )

    async def list_topics(self) -> List[str]:
        """
        Liệt kê tất cả topics có subscribers.

        Returns:
            Danh sách topics
        """
        async with self._lock:
            return list(self._subscriptions.keys())

    async def get_subscriber_count(self, topic: str) -> int:
        """
        Đếm số subscribers của topic.

        Args:
            topic: Topic cần đếm

        Returns:
            Số subscribers
        """
        async with self._lock:
            return len(self._subscriptions.get(topic, set()))

    async def clear(self) -> None:
        """Xóa tất cả subscriptions."""
        async with self._lock:
            self._subscriptions.clear()
            self._session_topics.clear()


class MessageRouter:
    """
    Router định tuyến messages đến các sessions đích.

    MessageRouter chịu trách nhiệm:
    - Xác định sessions đích dựa trên routing type
    - Gửi messages đến sessions đang online
    - Queue messages cho sessions offline
    - Quản lý topic subscriptions

    Attributes:
        session_manager: Manager để truy vấn sessions
        message_queue: Queue để lưu messages offline
        topic_registry: Registry quản lý topic subscriptions
        discovery: SessionDiscovery để tìm sessions

    Ví dụ:
        router = MessageRouter(session_manager, message_queue)

        # Direct routing
        result = await router.route_direct(message)

        # Broadcast routing
        result = await router.route_broadcast(message, agent_id="worker")

        # Topic routing
        result = await router.route_topic(message, topic="task.new")
    """

    def __init__(
        self,
        session_manager: SessionManagerProtocol,
        message_queue: Optional[MessageQueueProtocol] = None,
        topic_registry: Optional[TopicRegistry] = None,
    ) -> None:
        """
        Khởi tạo MessageRouter.

        Args:
            session_manager: Manager để truy vấn sessions
            message_queue: Queue cho offline messages (mặc định InMemoryMessageQueue)
            topic_registry: Registry cho topic subscriptions (mặc định TopicRegistry)
        """
        self.session_manager = session_manager
        self.message_queue = message_queue or InMemoryMessageQueue()
        self.topic_registry = topic_registry or TopicRegistry()
        self.discovery = SessionDiscovery(session_manager)
        self._message_handlers: Dict[str, MessageHandlerProtocol] = {}
        self._lock = asyncio.Lock()

    async def register_handler(
        self, session_key: str, handler: MessageHandlerProtocol
    ) -> None:
        """
        Đăng ký handler xử lý messages cho session.

        Args:
            session_key: Key của session
            handler: Handler xử lý messages
        """
        async with self._lock:
            self._message_handlers[session_key] = handler

    async def unregister_handler(self, session_key: str) -> bool:
        """
        Hủy đăng ký handler của session.

        Args:
            session_key: Key của session

        Returns:
            True nếu đã hủy, False nếu không tìm thấy
        """
        async with self._lock:
            if session_key in self._message_handlers:
                del self._message_handlers[session_key]
                return True
            return False

    async def route_direct(self, message: A2AMessage) -> RoutingResult:
        """
        Route message trực tiếp đến session đích.

        Args:
            message: Message cần gửi

        Returns:
            RoutingResult với thông tin routing
        """
        result = RoutingResult(
            message_id=message.id,
            routing_type="direct",
            target_sessions=[message.to_session],
        )

        # Kiểm tra session đích có tồn tại không
        session_info = await self.discovery.get_session_info(message.to_session)

        if session_info is None:
            # Queue message cho offline session
            await self.message_queue.enqueue(message)
            result.queued_count = 1
            return result

        # Kiểm tra session có handler không
        handler = self._message_handlers.get(message.to_session)

        if handler is not None and session_info.is_idle():
            # Gửi trực tiếp đến handler
            message.mark_delivered()
            result.delivered_count = 1
        else:
            # Session busy hoặc không có handler, queue message
            await self.message_queue.enqueue(message)
            result.queued_count = 1

        return result

    async def route_broadcast(
        self, message: A2AMessage, agent_id: str
    ) -> RoutingResult:
        """
        Broadcast message đến tất cả sessions của một agent.

        Args:
            message: Message cần gửi
            agent_id: ID của agent để broadcast

        Returns:
            RoutingResult với thông tin routing
        """
        result = RoutingResult(
            message_id=message.id,
            routing_type="broadcast",
        )

        # Lấy tất cả sessions của agent
        sessions = await self.discovery.list_sessions(filter_agent=agent_id)

        if not sessions:
            result.errors["no_sessions"] = f"Không tìm thấy sessions cho agent {agent_id}"
            return result

        result.target_sessions = [s.session_key for s in sessions]

        for session_info in sessions:
            session_key = session_info.session_key

            # Clone message cho mỗi session
            session_message = A2AMessage(
                id=f"{message.id}:{session_key}",
                from_session=message.from_session,
                to_session=session_key,
                message_type=message.message_type,
                content=message.content,
                priority=message.priority,
                timeout_ms=message.timeout_ms,
                metadata=message.metadata.copy(),
                correlation_id=message.correlation_id,
            )

            # Route message
            handler = self._message_handlers.get(session_key)
            if handler is not None and session_info.is_idle():
                session_message.mark_delivered()
                result.delivered_count += 1
            else:
                await self.message_queue.enqueue(session_message)
                result.queued_count += 1

        return result

    async def route_topic(self, message: A2AMessage, topic: str) -> RoutingResult:
        """
        Route message đến các sessions đã subscribe topic.

        Args:
            message: Message cần gửi
            topic: Topic để publish

        Returns:
            RoutingResult với thông tin routing
        """
        result = RoutingResult(
            message_id=message.id,
            routing_type="topic",
        )

        # Lấy subscribers của topic
        subscribers = await self.topic_registry.get_subscribers(topic)

        if not subscribers:
            result.errors["no_subscribers"] = f"Không có sessions subscribe topic {topic}"
            return result

        result.target_sessions = subscribers

        for session_key in subscribers:
            # Không gửi cho chính mình
            if session_key == message.from_session:
                continue

            # Clone message cho mỗi subscriber
            subscriber_message = A2AMessage(
                id=f"{message.id}:{session_key}",
                from_session=message.from_session,
                to_session=session_key,
                message_type=MessageType.EVENT,
                content=message.content,
                priority=message.priority,
                timeout_ms=message.timeout_ms,
                metadata=message.metadata.copy(),
                topic=topic,
            )

            # Kiểm tra session status
            session_info = await self.discovery.get_session_info(session_key)
            handler = self._message_handlers.get(session_key)

            if handler is not None and session_info and session_info.is_idle():
                subscriber_message.mark_delivered()
                result.delivered_count += 1
            else:
                await self.message_queue.enqueue(subscriber_message)
                result.queued_count += 1

        return result

    async def route(self, message: A2AMessage) -> RoutingResult:
        """
        Route message dựa trên message type và target.

        Tự động xác định routing type dựa trên message properties:
        - Event messages với topic -> topic routing
        - Messages có to_session -> direct routing

        Args:
            message: Message cần route

        Returns:
            RoutingResult với thông tin routing
        """
        if message.is_event() and message.topic:
            return await self.route_topic(message, message.topic)
        else:
            return await self.route_direct(message)


class InterSessionMessaging:
    """
    Giao tiếp giữa các sessions trong hệ thống A2A.

    InterSessionMessaging cung cấp API high-level cho việc gửi messages
    giữa các sessions, hỗ trợ cả sync và async modes.

    Attributes:
        session_manager: Manager để truy vấn sessions
        message_queue: Queue cho offline messages
        router: MessageRouter để định tuyến
        _pending_responses: Dictionary lưu các responses đang chờ

    Ví dụ:
        messaging = InterSessionMessaging(session_manager, message_queue)

        # Sync mode - đợi response
        response = await messaging.send(
            from_session="agent:orchestrator:main",
            to_session="agent:coder:main",
            message="Review PR #123",
            wait_for_response=True,
            timeout_ms=30000
        )

        # Async mode - fire and forget
        await messaging.send(
            from_session="agent:monitor:main",
            to_session="agent:admin:main",
            message="Alert: High CPU usage",
            wait_for_response=False
        )
    """

    def __init__(
        self,
        session_manager: SessionManagerProtocol,
        message_queue: Optional[MessageQueueProtocol] = None,
    ) -> None:
        """
        Khởi tạo InterSessionMessaging.

        Args:
            session_manager: Manager để truy vấn sessions
            message_queue: Queue cho offline messages
        """
        self.session_manager = session_manager
        self.message_queue = message_queue or InMemoryMessageQueue()
        self.router = MessageRouter(session_manager, self.message_queue)

        # Pending responses keyed by correlation_id
        self._pending_responses: Dict[str, asyncio.Future[A2AResponse]] = {}
        self._lock = asyncio.Lock()

    async def send(
        self,
        from_session: str,
        to_session: str,
        message: str,
        wait_for_response: bool = True,
        timeout_ms: int = 30000,
        priority: str = "normal",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> A2AResponse:
        """
        Gửi message đến session đích.

        Hỗ trợ 2 mode:
        - Sync mode (wait_for_response=True): Đợi session đích xử lý và trả response
        - Async mode (wait_for_response=False): Fire and forget

        Args:
            from_session: Session key người gửi
            to_session: Session key người nhận
            message: Nội dung tin nhắn
            wait_for_response: True để đợi response (mặc định)
            timeout_ms: Timeout tính bằng milliseconds (mặc định 30s)
            priority: Độ ưu tiên (low, normal, high)
            metadata: Dữ liệu bổ sung

        Returns:
            A2AResponse với kết quả

        Raises:
            asyncio.TimeoutError: Nếu timeout khi đợi response

        Ví dụ:
            # Sync mode
            response = await messaging.send(
                from_session="agent:orchestrator:main",
                to_session="agent:coder:main",
                message="Implement feature X",
                wait_for_response=True
            )
            if response.success:
                print(f"Response: {response.response}")

            # Async mode
            await messaging.send(
                from_session="agent:admin:main",
                to_session="agent:worker:main",
                message="Start background task",
                wait_for_response=False
            )
        """
        start_time = datetime.utcnow()

        # Parse priority
        try:
            msg_priority = MessagePriority(priority.lower())
        except ValueError:
            msg_priority = MessagePriority.NORMAL

        # Tạo message
        if wait_for_response:
            a2a_message = A2AMessage.create_request(
                from_session=from_session,
                to_session=to_session,
                content=message,
                priority=msg_priority,
                timeout_ms=timeout_ms,
                metadata=metadata,
            )
        else:
            a2a_message = A2AMessage.create_notification(
                from_session=from_session,
                to_session=to_session,
                content=message,
                priority=msg_priority,
                metadata=metadata,
            )

        # Route message
        routing_result = await self.router.route(a2a_message)

        # Nếu không đợi response, trả về success ngay
        if not wait_for_response:
            elapsed_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            if routing_result.is_success:
                return A2AResponse.success_response(
                    message_id=a2a_message.id,
                    response="Message delivered",
                    response_time_ms=elapsed_ms,
                    metadata={
                        "delivered": routing_result.delivered_count,
                        "queued": routing_result.queued_count,
                    },
                )
            else:
                return A2AResponse.error_response(
                    message_id=a2a_message.id,
                    error="Failed to deliver message",
                    response_time_ms=elapsed_ms,
                    metadata={"errors": routing_result.errors},
                )

        # Sync mode: tạo future để đợi response
        response_future: asyncio.Future[A2AResponse] = asyncio.Future()

        async with self._lock:
            self._pending_responses[a2a_message.correlation_id] = response_future

        try:
            # Đợi response với timeout
            timeout_seconds = timeout_ms / 1000.0
            response = await asyncio.wait_for(response_future, timeout=timeout_seconds)
            return response

        except asyncio.TimeoutError:
            return A2AResponse.timeout_response(a2a_message.id, timeout_ms)

        finally:
            # Cleanup pending response
            async with self._lock:
                self._pending_responses.pop(a2a_message.correlation_id, None)

    async def respond(
        self,
        message_id: str,
        response: str,
        success: bool = True,
        from_session: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Gửi response cho một request message.

        Phương thức này được gọi bởi session đích để trả response
        cho request message đã nhận.

        Args:
            message_id: ID của message gốc (correlation_id)
            response: Nội dung response
            success: True nếu xử lý thành công
            from_session: Session đã xử lý request
            metadata: Dữ liệu bổ sung

        Returns:
            True nếu response được gửi thành công

        Ví dụ:
            # Xử lý request và gửi response
            await messaging.respond(
                message_id="msg-123",
                response="PR reviewed, 2 comments added",
                success=True
            )
        """
        async with self._lock:
            future = self._pending_responses.get(message_id)
            if future is None:
                return False

            if success:
                a2a_response = A2AResponse.success_response(
                    message_id=message_id,
                    response=response,
                    from_session=from_session,
                    metadata=metadata,
                )
            else:
                a2a_response = A2AResponse.error_response(
                    message_id=message_id,
                    error=response,
                    from_session=from_session,
                    metadata=metadata,
                )

            future.set_result(a2a_response)
            return True

    async def broadcast(
        self,
        from_session: str,
        agent_id: str,
        message: str,
        priority: str = "normal",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[A2AResponse]:
        """
        Broadcast message đến tất cả sessions của một agent.

        Args:
            from_session: Session key người gửi
            agent_id: ID của agent để broadcast
            message: Nội dung tin nhắn
            priority: Độ ưu tiên
            metadata: Dữ liệu bổ sung

        Returns:
            Danh sách A2AResponse từ mỗi session

        Ví dụ:
            responses = await messaging.broadcast(
                from_session="agent:orchestrator:main",
                agent_id="worker",
                message="New task available"
            )
            for r in responses:
                print(f"{r.from_session}: {r.success}")
        """
        # Parse priority
        try:
            msg_priority = MessagePriority(priority.lower())
        except ValueError:
            msg_priority = MessagePriority.NORMAL

        # Tạo event message
        a2a_message = A2AMessage.create_notification(
            from_session=from_session,
            to_session="",  # Will be set by router
            content=message,
            priority=msg_priority,
            metadata=metadata,
        )

        # Route broadcast
        routing_result = await self.router.route_broadcast(a2a_message, agent_id)

        # Tạo responses
        responses = []
        for session_key in routing_result.target_sessions:
            if session_key in routing_result.errors:
                responses.append(
                    A2AResponse.error_response(
                        message_id=a2a_message.id,
                        error=routing_result.errors[session_key],
                        from_session=session_key,
                    )
                )
            else:
                responses.append(
                    A2AResponse.success_response(
                        message_id=a2a_message.id,
                        response="Message delivered",
                        from_session=session_key,
                    )
                )

        return responses

    async def publish(
        self,
        from_session: str,
        topic: str,
        message: str,
        priority: str = "normal",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RoutingResult:
        """
        Publish message đến topic.

        Args:
            from_session: Session key người gửi
            topic: Topic để publish
            message: Nội dung tin nhắn
            priority: Độ ưu tiên
            metadata: Dữ liệu bổ sung

        Returns:
            RoutingResult với thông tin routing

        Ví dụ:
            result = await messaging.publish(
                from_session="agent:orchestrator:main",
                topic="task.completed",
                message="Task #123 completed"
            )
            print(f"Delivered to {result.delivered_count} sessions")
        """
        # Parse priority
        try:
            msg_priority = MessagePriority(priority.lower())
        except ValueError:
            msg_priority = MessagePriority.NORMAL

        # Tạo event message
        a2a_message = A2AMessage.create_event(
            from_session=from_session,
            topic=topic,
            content=message,
            priority=msg_priority,
            metadata=metadata,
        )

        # Route topic
        return await self.router.route_topic(a2a_message, topic)

    async def subscribe(self, session_key: str, topic: str) -> bool:
        """
        Subscribe session vào topic.

        Args:
            session_key: Key của session
            topic: Topic để subscribe

        Returns:
            True nếu subscription mới
        """
        return await self.router.topic_registry.subscribe(session_key, topic)

    async def unsubscribe(self, session_key: str, topic: str) -> bool:
        """
        Unsubscribe session khỏi topic.

        Args:
            session_key: Key của session
            topic: Topic để unsubscribe

        Returns:
            True nếu đã unsubscribe
        """
        return await self.router.topic_registry.unsubscribe(session_key, topic)

    async def get_pending_messages(
        self, session_key: str, limit: int = 10
    ) -> List[A2AMessage]:
        """
        Lấy danh sách messages đang chờ cho session.

        Args:
            session_key: Key của session
            limit: Số lượng tối đa

        Returns:
            Danh sách messages
        """
        return await self.message_queue.get_messages_for_session(session_key, limit)

    async def process_pending(
        self,
        session_key: str,
        handler: Callable[[A2AMessage], A2AResponse],
    ) -> int:
        """
        Xử lý các messages đang chờ cho session.

        Args:
            session_key: Key của session
            handler: Callback xử lý mỗi message

        Returns:
            Số messages đã xử lý
        """
        processed = 0

        while True:
            message = await self.message_queue.dequeue(session_key)
            if message is None:
                break

            try:
                response = handler(message)

                # Gửi response nếu là request
                if message.is_request() and message.correlation_id:
                    await self.respond(
                        message_id=message.correlation_id,
                        response=response.response or "",
                        success=response.success,
                        from_session=session_key,
                    )

                await self.message_queue.acknowledge(message.id)
                processed += 1

            except Exception as e:
                # Reject và requeue
                await self.message_queue.reject(message.id, requeue=True)

        return processed
