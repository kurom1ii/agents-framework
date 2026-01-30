"""MCP (Model Context Protocol) transport layer.

This module provides transport implementations for MCP communication:
- Stdio transport for subprocess-based MCP servers
- SSE (Server-Sent Events) transport for HTTP-based MCP servers
"""

from __future__ import annotations

import asyncio
import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)


class TransportType(str, Enum):
    """Types of MCP transports."""

    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"


class MCPError(Exception):
    """Base exception for MCP errors."""

    def __init__(
        self,
        message: str,
        code: Optional[int] = None,
        data: Optional[Any] = None,
    ):
        super().__init__(message)
        self.code = code
        self.data = data


class TransportError(MCPError):
    """Transport layer error."""
    pass


class ConnectionError(TransportError):
    """Connection-related error."""
    pass


@dataclass
class JSONRPCRequest:
    """JSON-RPC 2.0 request message.

    Attributes:
        method: The method to call.
        params: Optional parameters for the method.
        id: Request ID for matching responses.
    """

    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-RPC 2.0 format."""
        result: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": self.method,
        }
        if self.params is not None:
            result["params"] = self.params
        if self.id is not None:
            result["id"] = self.id
        return result

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class JSONRPCResponse:
    """JSON-RPC 2.0 response message.

    Attributes:
        result: The result of the method call (if successful).
        error: Error information (if failed).
        id: Request ID matching the original request.
    """

    id: Optional[Union[str, int]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JSONRPCResponse":
        """Create from dictionary."""
        return cls(
            id=data.get("id"),
            result=data.get("result"),
            error=data.get("error"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "JSONRPCResponse":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @property
    def is_error(self) -> bool:
        """Check if response is an error."""
        return self.error is not None

    def get_error_message(self) -> Optional[str]:
        """Get error message if this is an error response."""
        if self.error:
            return self.error.get("message", "Unknown error")
        return None

    def get_error_code(self) -> Optional[int]:
        """Get error code if this is an error response."""
        if self.error:
            return self.error.get("code")
        return None


@dataclass
class JSONRPCNotification:
    """JSON-RPC 2.0 notification message (no response expected).

    Attributes:
        method: The method to call.
        params: Optional parameters for the method.
    """

    method: str
    params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-RPC 2.0 format."""
        result: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": self.method,
        }
        if self.params is not None:
            result["params"] = self.params
        return result

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())


class Transport(ABC):
    """Abstract base class for MCP transports.

    Transports handle the low-level communication with MCP servers,
    providing methods for sending requests and receiving responses.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the MCP server."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the MCP server."""
        pass

    @abstractmethod
    async def send_request(
        self,
        request: JSONRPCRequest,
    ) -> JSONRPCResponse:
        """Send a request and wait for response.

        Args:
            request: The JSON-RPC request to send.

        Returns:
            The JSON-RPC response.
        """
        pass

    @abstractmethod
    async def send_notification(
        self,
        notification: JSONRPCNotification,
    ) -> None:
        """Send a notification (no response expected).

        Args:
            notification: The JSON-RPC notification to send.
        """
        pass

    @abstractmethod
    async def receive_notifications(self) -> AsyncIterator[JSONRPCNotification]:
        """Receive incoming notifications from the server.

        Yields:
            JSON-RPC notifications from the server.
        """
        yield  # pragma: no cover

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if transport is connected."""
        pass


@dataclass
class StdioTransportConfig:
    """Configuration for stdio transport.

    Attributes:
        command: Command to execute to start the MCP server.
        args: Command line arguments.
        env: Environment variables to set.
        cwd: Working directory for the process.
        timeout: Request timeout in seconds.
    """

    command: str
    args: List[str] = field(default_factory=list)
    env: Optional[Dict[str, str]] = None
    cwd: Optional[str] = None
    timeout: float = 30.0


class StdioTransport(Transport):
    """Stdio-based transport for MCP servers.

    Communicates with MCP servers via stdin/stdout of a subprocess.
    Uses newline-delimited JSON for message framing.
    """

    def __init__(self, config: StdioTransportConfig):
        """Initialize the stdio transport.

        Args:
            config: Configuration for the transport.
        """
        self.config = config
        self._process: Optional[asyncio.subprocess.Process] = None
        self._request_id = 0
        self._pending_requests: Dict[
            Union[str, int], asyncio.Future[JSONRPCResponse]
        ] = {}
        self._notification_queue: asyncio.Queue[JSONRPCNotification] = asyncio.Queue()
        self._read_task: Optional[asyncio.Task[None]] = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if transport is connected."""
        return self._connected and self._process is not None

    async def connect(self) -> None:
        """Start the MCP server process and establish connection."""
        if self._connected:
            return

        try:
            # Build the command
            cmd = [self.config.command] + self.config.args

            # Start the process
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self.config.env,
                cwd=self.config.cwd,
            )

            self._connected = True

            # Start reading responses
            self._read_task = asyncio.create_task(self._read_loop())

        except Exception as e:
            raise ConnectionError(f"Failed to start MCP server: {e}") from e

    async def disconnect(self) -> None:
        """Stop the MCP server process."""
        self._connected = False

        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
            self._read_task = None

        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            except Exception:
                pass
            self._process = None

        # Cancel pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

    async def send_request(
        self,
        request: JSONRPCRequest,
    ) -> JSONRPCResponse:
        """Send a request and wait for response."""
        if not self.is_connected or not self._process:
            raise ConnectionError("Transport not connected")

        # Assign request ID if not set
        if request.id is None:
            self._request_id += 1
            request.id = self._request_id

        # Create future for response
        future: asyncio.Future[JSONRPCResponse] = asyncio.get_event_loop().create_future()
        self._pending_requests[request.id] = future

        try:
            # Send request
            message = request.to_json() + "\n"
            assert self._process.stdin is not None
            self._process.stdin.write(message.encode())
            await self._process.stdin.drain()

            # Wait for response with timeout
            response = await asyncio.wait_for(
                future,
                timeout=self.config.timeout,
            )
            return response

        except asyncio.TimeoutError:
            self._pending_requests.pop(request.id, None)
            raise TransportError(f"Request timed out after {self.config.timeout}s")
        except Exception as e:
            self._pending_requests.pop(request.id, None)
            raise TransportError(f"Request failed: {e}") from e

    async def send_notification(
        self,
        notification: JSONRPCNotification,
    ) -> None:
        """Send a notification (no response expected)."""
        if not self.is_connected or not self._process:
            raise ConnectionError("Transport not connected")

        message = notification.to_json() + "\n"
        assert self._process.stdin is not None
        self._process.stdin.write(message.encode())
        await self._process.stdin.drain()

    async def receive_notifications(self) -> AsyncIterator[JSONRPCNotification]:
        """Receive incoming notifications from the server."""
        while self.is_connected:
            try:
                notification = await asyncio.wait_for(
                    self._notification_queue.get(),
                    timeout=1.0,
                )
                yield notification
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def _read_loop(self) -> None:
        """Background task to read responses from the process."""
        if not self._process or not self._process.stdout:
            return

        try:
            while self._connected:
                line = await self._process.stdout.readline()
                if not line:
                    break

                try:
                    data = json.loads(line.decode())
                except json.JSONDecodeError:
                    continue

                # Check if it's a response or notification
                if "id" in data:
                    # It's a response
                    response = JSONRPCResponse.from_dict(data)
                    future = self._pending_requests.pop(response.id, None)
                    if future and not future.done():
                        future.set_result(response)
                elif "method" in data:
                    # It's a notification
                    notification = JSONRPCNotification(
                        method=data["method"],
                        params=data.get("params"),
                    )
                    await self._notification_queue.put(notification)

        except asyncio.CancelledError:
            pass
        except Exception:
            self._connected = False


@dataclass
class SSETransportConfig:
    """Configuration for SSE transport.

    Attributes:
        url: Base URL of the MCP server.
        headers: Additional HTTP headers.
        timeout: Request timeout in seconds.
        retry_interval: Time between connection retries.
    """

    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    retry_interval: float = 5.0


class SSETransport(Transport):
    """SSE-based transport for MCP servers.

    Communicates with MCP servers via HTTP using Server-Sent Events
    for receiving notifications and standard HTTP POST for requests.
    """

    def __init__(self, config: SSETransportConfig):
        """Initialize the SSE transport.

        Args:
            config: Configuration for the transport.
        """
        self.config = config
        self._request_id = 0
        self._connected = False
        self._sse_task: Optional[asyncio.Task[None]] = None
        self._notification_queue: asyncio.Queue[JSONRPCNotification] = asyncio.Queue()

    @property
    def is_connected(self) -> bool:
        """Check if transport is connected."""
        return self._connected

    async def connect(self) -> None:
        """Connect to the SSE endpoint."""
        if self._connected:
            return

        self._connected = True
        # Start SSE listener if the server supports it
        # This is optional - some servers only support request/response

    async def disconnect(self) -> None:
        """Disconnect from the server."""
        self._connected = False

        if self._sse_task:
            self._sse_task.cancel()
            try:
                await self._sse_task
            except asyncio.CancelledError:
                pass
            self._sse_task = None

    async def send_request(
        self,
        request: JSONRPCRequest,
    ) -> JSONRPCResponse:
        """Send a request via HTTP POST."""
        if not self._connected:
            raise ConnectionError("Transport not connected")

        # Import here to avoid dependency issues
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for SSE transport. "
                "Install it with: pip install httpx"
            )

        # Assign request ID if not set
        if request.id is None:
            self._request_id += 1
            request.id = self._request_id

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(
                    self.config.url,
                    json=request.to_dict(),
                    headers={
                        "Content-Type": "application/json",
                        **self.config.headers,
                    },
                )
                response.raise_for_status()
                return JSONRPCResponse.from_dict(response.json())

        except httpx.TimeoutException:
            raise TransportError(f"Request timed out after {self.config.timeout}s")
        except httpx.HTTPStatusError as e:
            raise TransportError(f"HTTP error: {e.response.status_code}")
        except Exception as e:
            raise TransportError(f"Request failed: {e}") from e

    async def send_notification(
        self,
        notification: JSONRPCNotification,
    ) -> None:
        """Send a notification via HTTP POST."""
        if not self._connected:
            raise ConnectionError("Transport not connected")

        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for SSE transport. "
                "Install it with: pip install httpx"
            )

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                await client.post(
                    self.config.url,
                    json=notification.to_dict(),
                    headers={
                        "Content-Type": "application/json",
                        **self.config.headers,
                    },
                )
        except Exception as e:
            raise TransportError(f"Notification failed: {e}") from e

    async def receive_notifications(self) -> AsyncIterator[JSONRPCNotification]:
        """Receive notifications via SSE."""
        while self._connected:
            try:
                notification = await asyncio.wait_for(
                    self._notification_queue.get(),
                    timeout=1.0,
                )
                yield notification
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break


def create_transport(
    transport_type: TransportType,
    config: Union[StdioTransportConfig, SSETransportConfig],
) -> Transport:
    """Factory function to create a transport.

    Args:
        transport_type: Type of transport to create.
        config: Configuration for the transport.

    Returns:
        A Transport instance.
    """
    if transport_type == TransportType.STDIO:
        if not isinstance(config, StdioTransportConfig):
            raise ValueError("StdioTransportConfig required for stdio transport")
        return StdioTransport(config)
    elif transport_type in (TransportType.SSE, TransportType.HTTP):
        if not isinstance(config, SSETransportConfig):
            raise ValueError("SSETransportConfig required for SSE/HTTP transport")
        return SSETransport(config)
    else:
        raise ValueError(f"Unknown transport type: {transport_type}")
