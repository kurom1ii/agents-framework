"""Unit tests for MCP transport layer.

Tests cover:
- JSON-RPC message serialization/deserialization
- Transport type enum
- Stdio transport configuration and connection
- SSE transport configuration and connection
- Transport factory function
- Error handling
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from agents_framework.mcp.transport import (
    ConnectionError,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPError,
    SSETransport,
    SSETransportConfig,
    StdioTransport,
    StdioTransportConfig,
    Transport,
    TransportError,
    TransportType,
    create_transport,
)


# ============================================================================
# TransportType Tests
# ============================================================================


class TestTransportType:
    """Tests for TransportType enum."""

    def test_transport_type_values(self):
        """Test TransportType enum values."""
        assert TransportType.STDIO.value == "stdio"
        assert TransportType.SSE.value == "sse"
        assert TransportType.HTTP.value == "http"

    def test_transport_type_is_string_enum(self):
        """Test TransportType is a string enum."""
        assert isinstance(TransportType.STDIO, str)
        assert TransportType.STDIO == "stdio"


# ============================================================================
# MCPError Tests
# ============================================================================


class TestMCPError:
    """Tests for MCPError exception."""

    def test_mcp_error_basic(self):
        """Test basic MCPError creation."""
        error = MCPError("Test error")
        assert str(error) == "Test error"
        assert error.code is None
        assert error.data is None

    def test_mcp_error_with_code(self):
        """Test MCPError with error code."""
        error = MCPError("Test error", code=-32600)
        assert error.code == -32600

    def test_mcp_error_with_data(self):
        """Test MCPError with additional data."""
        error = MCPError("Test error", data={"details": "more info"})
        assert error.data == {"details": "more info"}

    def test_mcp_error_full(self):
        """Test MCPError with all parameters."""
        error = MCPError("Full error", code=-32600, data={"key": "value"})
        assert str(error) == "Full error"
        assert error.code == -32600
        assert error.data == {"key": "value"}


class TestTransportError:
    """Tests for TransportError exception."""

    def test_transport_error_inherits_mcp_error(self):
        """Test TransportError inherits from MCPError."""
        assert issubclass(TransportError, MCPError)

    def test_transport_error_creation(self):
        """Test TransportError creation."""
        error = TransportError("Transport failed", code=-1)
        assert str(error) == "Transport failed"
        assert error.code == -1


class TestConnectionError:
    """Tests for ConnectionError exception."""

    def test_connection_error_inherits_transport_error(self):
        """Test ConnectionError inherits from TransportError."""
        assert issubclass(ConnectionError, TransportError)


# ============================================================================
# JSONRPCRequest Tests
# ============================================================================


class TestJSONRPCRequest:
    """Tests for JSONRPCRequest dataclass."""

    def test_request_basic(self, sample_jsonrpc_request: JSONRPCRequest):
        """Test basic request creation."""
        assert sample_jsonrpc_request.method == "tools/list"
        assert sample_jsonrpc_request.params == {"filter": "all"}
        assert sample_jsonrpc_request.id == 1

    def test_request_minimal(self):
        """Test minimal request (method only)."""
        request = JSONRPCRequest(method="ping")
        assert request.method == "ping"
        assert request.params is None
        assert request.id is None

    def test_request_to_dict(self, sample_jsonrpc_request: JSONRPCRequest):
        """Test conversion to dictionary."""
        result = sample_jsonrpc_request.to_dict()
        assert result["jsonrpc"] == "2.0"
        assert result["method"] == "tools/list"
        assert result["params"] == {"filter": "all"}
        assert result["id"] == 1

    def test_request_to_dict_minimal(self):
        """Test to_dict with minimal request."""
        request = JSONRPCRequest(method="ping")
        result = request.to_dict()
        assert result["jsonrpc"] == "2.0"
        assert result["method"] == "ping"
        assert "params" not in result
        assert "id" not in result

    def test_request_to_json(self, sample_jsonrpc_request: JSONRPCRequest):
        """Test JSON serialization."""
        json_str = sample_jsonrpc_request.to_json()
        data = json.loads(json_str)
        assert data["jsonrpc"] == "2.0"
        assert data["method"] == "tools/list"

    def test_request_with_string_id(self):
        """Test request with string ID."""
        request = JSONRPCRequest(method="test", id="request-1")
        result = request.to_dict()
        assert result["id"] == "request-1"


# ============================================================================
# JSONRPCResponse Tests
# ============================================================================


class TestJSONRPCResponse:
    """Tests for JSONRPCResponse dataclass."""

    def test_response_success(self, sample_jsonrpc_response: JSONRPCResponse):
        """Test successful response."""
        assert sample_jsonrpc_response.id == 1
        assert sample_jsonrpc_response.result == {"tools": [{"name": "test_tool"}]}
        assert sample_jsonrpc_response.error is None
        assert sample_jsonrpc_response.is_error is False

    def test_response_error(self, sample_jsonrpc_error_response: JSONRPCResponse):
        """Test error response."""
        assert sample_jsonrpc_error_response.is_error is True
        assert sample_jsonrpc_error_response.get_error_message() == "Invalid Request"
        assert sample_jsonrpc_error_response.get_error_code() == -32600

    def test_response_from_dict(self):
        """Test creating response from dictionary."""
        data = {
            "jsonrpc": "2.0",
            "id": 42,
            "result": {"status": "ok"},
        }
        response = JSONRPCResponse.from_dict(data)
        assert response.id == 42
        assert response.result == {"status": "ok"}
        assert response.error is None

    def test_response_from_dict_error(self):
        """Test creating error response from dictionary."""
        data = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32700, "message": "Parse error"},
        }
        response = JSONRPCResponse.from_dict(data)
        assert response.is_error is True
        assert response.get_error_code() == -32700
        assert response.get_error_message() == "Parse error"

    def test_response_from_json(self):
        """Test creating response from JSON string."""
        json_str = '{"jsonrpc": "2.0", "id": 1, "result": "success"}'
        response = JSONRPCResponse.from_json(json_str)
        assert response.id == 1
        assert response.result == "success"

    def test_get_error_message_unknown(self):
        """Test get_error_message with no message field."""
        response = JSONRPCResponse(id=1, error={"code": -1})
        assert response.get_error_message() == "Unknown error"

    def test_get_error_code_none(self):
        """Test get_error_code when no error."""
        response = JSONRPCResponse(id=1, result="ok")
        assert response.get_error_code() is None

    def test_get_error_message_none(self):
        """Test get_error_message when no error."""
        response = JSONRPCResponse(id=1, result="ok")
        assert response.get_error_message() is None


# ============================================================================
# JSONRPCNotification Tests
# ============================================================================


class TestJSONRPCNotification:
    """Tests for JSONRPCNotification dataclass."""

    def test_notification_basic(self, sample_jsonrpc_notification: JSONRPCNotification):
        """Test basic notification creation."""
        assert sample_jsonrpc_notification.method == "notifications/progress"
        assert sample_jsonrpc_notification.params == {"progress": 50, "total": 100}

    def test_notification_minimal(self):
        """Test minimal notification."""
        notification = JSONRPCNotification(method="ping")
        assert notification.method == "ping"
        assert notification.params is None

    def test_notification_to_dict(self, sample_jsonrpc_notification: JSONRPCNotification):
        """Test conversion to dictionary."""
        result = sample_jsonrpc_notification.to_dict()
        assert result["jsonrpc"] == "2.0"
        assert result["method"] == "notifications/progress"
        assert result["params"] == {"progress": 50, "total": 100}
        assert "id" not in result  # Notifications don't have IDs

    def test_notification_to_dict_minimal(self):
        """Test to_dict with minimal notification."""
        notification = JSONRPCNotification(method="heartbeat")
        result = notification.to_dict()
        assert result["jsonrpc"] == "2.0"
        assert result["method"] == "heartbeat"
        assert "params" not in result

    def test_notification_to_json(self, sample_jsonrpc_notification: JSONRPCNotification):
        """Test JSON serialization."""
        json_str = sample_jsonrpc_notification.to_json()
        data = json.loads(json_str)
        assert data["jsonrpc"] == "2.0"
        assert data["method"] == "notifications/progress"


# ============================================================================
# StdioTransportConfig Tests
# ============================================================================


class TestStdioTransportConfig:
    """Tests for StdioTransportConfig dataclass."""

    def test_config_basic(self, stdio_transport_config: StdioTransportConfig):
        """Test basic config creation."""
        assert stdio_transport_config.command == "python"
        assert stdio_transport_config.args == ["-m", "mcp_server"]
        assert stdio_transport_config.env == {"TEST_VAR": "test_value"}
        assert stdio_transport_config.cwd == "/tmp"
        assert stdio_transport_config.timeout == 10.0

    def test_config_defaults(self):
        """Test config with defaults."""
        config = StdioTransportConfig(command="node")
        assert config.command == "node"
        assert config.args == []
        assert config.env is None
        assert config.cwd is None
        assert config.timeout == 30.0


# ============================================================================
# StdioTransport Tests
# ============================================================================


class TestStdioTransport:
    """Tests for StdioTransport class."""

    def test_transport_initialization(self, stdio_transport_config: StdioTransportConfig):
        """Test transport initialization."""
        transport = StdioTransport(stdio_transport_config)
        assert transport.config == stdio_transport_config
        assert transport.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_success(self, stdio_transport_config: StdioTransportConfig):
        """Test successful connection."""
        transport = StdioTransport(stdio_transport_config)

        # Mock subprocess creation
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(return_value=b"")
        mock_process.stderr = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await transport.connect()

        assert transport.is_connected is True

    @pytest.mark.asyncio
    async def test_connect_already_connected(self, stdio_transport_config: StdioTransportConfig):
        """Test connecting when already connected."""
        transport = StdioTransport(stdio_transport_config)

        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(return_value=b"")
        mock_process.stderr = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            await transport.connect()
            await transport.connect()  # Second connect should be no-op

        # Should only create subprocess once
        mock_exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure(self, stdio_transport_config: StdioTransportConfig):
        """Test connection failure."""
        transport = StdioTransport(stdio_transport_config)

        with patch("asyncio.create_subprocess_exec", side_effect=OSError("Command not found")):
            with pytest.raises(ConnectionError, match="Failed to start MCP server"):
                await transport.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self, stdio_transport_config: StdioTransportConfig):
        """Test disconnection."""
        transport = StdioTransport(stdio_transport_config)

        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(return_value=b"")
        mock_process.stderr = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.wait = AsyncMock()
        mock_process.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await transport.connect()
            await transport.disconnect()

        assert transport.is_connected is False
        mock_process.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_timeout(self, stdio_transport_config: StdioTransportConfig):
        """Test disconnection with timeout."""
        transport = StdioTransport(stdio_transport_config)

        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(return_value=b"")
        mock_process.stderr = MagicMock()
        mock_process.terminate = MagicMock()

        # First wait call times out, second (after kill) succeeds
        wait_call_count = 0

        async def mock_wait():
            nonlocal wait_call_count
            wait_call_count += 1
            if wait_call_count == 1:
                raise asyncio.TimeoutError()
            return 0

        mock_process.wait = mock_wait
        mock_process.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await transport.connect()
            await transport.disconnect()

        mock_process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_request_not_connected(self, stdio_transport_config: StdioTransportConfig):
        """Test sending request when not connected."""
        transport = StdioTransport(stdio_transport_config)
        request = JSONRPCRequest(method="test")

        with pytest.raises(ConnectionError, match="Transport not connected"):
            await transport.send_request(request)

    @pytest.mark.asyncio
    async def test_send_request_success(self, stdio_transport_config: StdioTransportConfig):
        """Test successful request sending."""
        # Use a short timeout for this test
        config = StdioTransportConfig(
            command="python",
            timeout=1.0,
        )
        transport = StdioTransport(config)

        # Create a proper async mock for stdin
        mock_stdin = MagicMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()

        # Response to be returned
        response_data = {"jsonrpc": "2.0", "id": 1, "result": "success"}
        response_line = json.dumps(response_data).encode() + b"\n"

        # Use an Event to synchronize the response
        response_ready = asyncio.Event()
        response_consumed = asyncio.Event()

        async def mock_readline():
            # Wait until request is sent, then return response
            await response_ready.wait()
            response_ready.clear()
            response_consumed.set()
            return response_line

        mock_stdout = MagicMock()
        mock_stdout.readline = mock_readline

        mock_process = MagicMock()
        mock_process.stdin = mock_stdin
        mock_process.stdout = mock_stdout
        mock_process.stderr = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await transport.connect()

            # Give read loop time to start
            await asyncio.sleep(0.01)

            # Create the request task
            request = JSONRPCRequest(method="test")

            async def send_request():
                return await transport.send_request(request)

            # Start request in background
            request_task = asyncio.create_task(send_request())

            # Give time for request to be sent
            await asyncio.sleep(0.01)

            # Signal that response is ready
            response_ready.set()

            # Wait for response
            response = await request_task

            assert response.result == "success"

            await transport.disconnect()

    @pytest.mark.asyncio
    async def test_send_request_timeout(self, stdio_transport_config: StdioTransportConfig):
        """Test request timeout."""
        config = StdioTransportConfig(
            command="python",
            timeout=0.1,  # Very short timeout
        )
        transport = StdioTransport(config)

        mock_stdin = MagicMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()

        # Never return a response
        async def mock_readline():
            await asyncio.sleep(10)  # Longer than timeout
            return b""

        mock_stdout = MagicMock()
        mock_stdout.readline = mock_readline

        mock_process = MagicMock()
        mock_process.stdin = mock_stdin
        mock_process.stdout = mock_stdout
        mock_process.stderr = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await transport.connect()
            request = JSONRPCRequest(method="test")

            with pytest.raises(TransportError, match="timed out"):
                await transport.send_request(request)

    @pytest.mark.asyncio
    async def test_send_notification_not_connected(
        self, stdio_transport_config: StdioTransportConfig
    ):
        """Test sending notification when not connected."""
        transport = StdioTransport(stdio_transport_config)
        notification = JSONRPCNotification(method="ping")

        with pytest.raises(ConnectionError, match="Transport not connected"):
            await transport.send_notification(notification)

    @pytest.mark.asyncio
    async def test_send_notification_success(self, stdio_transport_config: StdioTransportConfig):
        """Test successful notification sending."""
        transport = StdioTransport(stdio_transport_config)

        mock_stdin = MagicMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()

        mock_stdout = MagicMock()
        mock_stdout.readline = AsyncMock(return_value=b"")

        mock_process = MagicMock()
        mock_process.stdin = mock_stdin
        mock_process.stdout = mock_stdout
        mock_process.stderr = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await transport.connect()

            notification = JSONRPCNotification(method="ping")
            await transport.send_notification(notification)

            # Verify notification was written
            assert mock_stdin.write.called


# ============================================================================
# SSETransportConfig Tests
# ============================================================================


class TestSSETransportConfig:
    """Tests for SSETransportConfig dataclass."""

    def test_config_basic(self, sse_transport_config: SSETransportConfig):
        """Test basic config creation."""
        assert sse_transport_config.url == "http://localhost:8080/mcp"
        assert sse_transport_config.headers == {"Authorization": "Bearer test-token"}
        assert sse_transport_config.timeout == 15.0
        assert sse_transport_config.retry_interval == 3.0

    def test_config_defaults(self):
        """Test config with defaults."""
        config = SSETransportConfig(url="http://localhost:8080")
        assert config.url == "http://localhost:8080"
        assert config.headers == {}
        assert config.timeout == 30.0
        assert config.retry_interval == 5.0


# ============================================================================
# SSETransport Tests
# ============================================================================


class TestSSETransport:
    """Tests for SSETransport class."""

    def test_transport_initialization(self, sse_transport_config: SSETransportConfig):
        """Test transport initialization."""
        transport = SSETransport(sse_transport_config)
        assert transport.config == sse_transport_config
        assert transport.is_connected is False

    @pytest.mark.asyncio
    async def test_connect(self, sse_transport_config: SSETransportConfig):
        """Test connection."""
        transport = SSETransport(sse_transport_config)
        await transport.connect()
        assert transport.is_connected is True

    @pytest.mark.asyncio
    async def test_connect_already_connected(self, sse_transport_config: SSETransportConfig):
        """Test connecting when already connected."""
        transport = SSETransport(sse_transport_config)
        await transport.connect()
        await transport.connect()  # Should be no-op
        assert transport.is_connected is True

    @pytest.mark.asyncio
    async def test_disconnect(self, sse_transport_config: SSETransportConfig):
        """Test disconnection."""
        transport = SSETransport(sse_transport_config)
        await transport.connect()
        await transport.disconnect()
        assert transport.is_connected is False

    @pytest.mark.asyncio
    async def test_send_request_not_connected(self, sse_transport_config: SSETransportConfig):
        """Test sending request when not connected."""
        transport = SSETransport(sse_transport_config)
        request = JSONRPCRequest(method="test")

        with pytest.raises(ConnectionError, match="Transport not connected"):
            await transport.send_request(request)

    @pytest.mark.asyncio
    async def test_send_request_httpx_not_installed(
        self, sse_transport_config: SSETransportConfig
    ):
        """Test error when httpx is not installed."""
        transport = SSETransport(sse_transport_config)
        await transport.connect()

        request = JSONRPCRequest(method="test")

        with patch.dict("sys.modules", {"httpx": None}):
            # Force reimport to trigger ImportError
            with patch("builtins.__import__", side_effect=ImportError("No module named 'httpx'")):
                with pytest.raises(ImportError, match="httpx is required"):
                    await transport.send_request(request)

    @pytest.mark.asyncio
    async def test_send_request_success(self, sse_transport_config: SSETransportConfig):
        """Test successful request with mocked httpx."""
        transport = SSETransport(sse_transport_config)
        await transport.connect()

        request = JSONRPCRequest(method="test")

        # Mock httpx
        mock_response = MagicMock()
        mock_response.json.return_value = {"jsonrpc": "2.0", "id": 1, "result": "ok"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            response = await transport.send_request(request)

        assert response.result == "ok"

    @pytest.mark.asyncio
    async def test_send_request_timeout(self, sse_transport_config: SSETransportConfig):
        """Test request timeout."""
        transport = SSETransport(sse_transport_config)
        await transport.connect()

        request = JSONRPCRequest(method="test")

        # Import httpx types for mocking
        import httpx

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(TransportError, match="timed out"):
                await transport.send_request(request)

    @pytest.mark.asyncio
    async def test_send_request_http_error(self, sse_transport_config: SSETransportConfig):
        """Test HTTP error handling."""
        transport = SSETransport(sse_transport_config)
        await transport.connect()

        request = JSONRPCRequest(method="test")

        import httpx

        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Server Error", request=mock_request, response=mock_response
            )
        )

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(TransportError, match="HTTP error: 500"):
                await transport.send_request(request)

    @pytest.mark.asyncio
    async def test_send_notification_not_connected(
        self, sse_transport_config: SSETransportConfig
    ):
        """Test sending notification when not connected."""
        transport = SSETransport(sse_transport_config)
        notification = JSONRPCNotification(method="ping")

        with pytest.raises(ConnectionError, match="Transport not connected"):
            await transport.send_notification(notification)

    @pytest.mark.asyncio
    async def test_send_notification_success(self, sse_transport_config: SSETransportConfig):
        """Test successful notification sending."""
        transport = SSETransport(sse_transport_config)
        await transport.connect()

        notification = JSONRPCNotification(method="ping")

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=MagicMock())

        with patch("httpx.AsyncClient", return_value=mock_client):
            await transport.send_notification(notification)

        mock_client.post.assert_called_once()


# ============================================================================
# create_transport Tests
# ============================================================================


class TestCreateTransport:
    """Tests for create_transport factory function."""

    def test_create_stdio_transport(self, stdio_transport_config: StdioTransportConfig):
        """Test creating stdio transport."""
        transport = create_transport(TransportType.STDIO, stdio_transport_config)
        assert isinstance(transport, StdioTransport)

    def test_create_sse_transport(self, sse_transport_config: SSETransportConfig):
        """Test creating SSE transport."""
        transport = create_transport(TransportType.SSE, sse_transport_config)
        assert isinstance(transport, SSETransport)

    def test_create_http_transport(self, sse_transport_config: SSETransportConfig):
        """Test creating HTTP transport (uses SSE)."""
        transport = create_transport(TransportType.HTTP, sse_transport_config)
        assert isinstance(transport, SSETransport)

    def test_create_stdio_wrong_config(self, sse_transport_config: SSETransportConfig):
        """Test error when using wrong config type for stdio."""
        with pytest.raises(ValueError, match="StdioTransportConfig required"):
            create_transport(TransportType.STDIO, sse_transport_config)

    def test_create_sse_wrong_config(self, stdio_transport_config: StdioTransportConfig):
        """Test error when using wrong config type for SSE."""
        with pytest.raises(ValueError, match="SSETransportConfig required"):
            create_transport(TransportType.SSE, stdio_transport_config)


# ============================================================================
# Transport ABC Tests
# ============================================================================


class TestTransportABC:
    """Tests for Transport abstract base class."""

    def test_transport_is_abstract(self):
        """Test that Transport cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Transport()

    def test_transport_abstract_methods(self):
        """Test that Transport defines required abstract methods."""
        abstract_methods = Transport.__abstractmethods__
        assert "connect" in abstract_methods
        assert "disconnect" in abstract_methods
        assert "send_request" in abstract_methods
        assert "send_notification" in abstract_methods
        assert "receive_notifications" in abstract_methods
        assert "is_connected" in abstract_methods


# ============================================================================
# Integration-like Tests
# ============================================================================


class TestRequestIdAssignment:
    """Tests for automatic request ID assignment."""

    @pytest.mark.asyncio
    async def test_stdio_auto_assigns_id(self, stdio_transport_config: StdioTransportConfig):
        """Test that StdioTransport auto-assigns request IDs."""
        # Use a short timeout
        config = StdioTransportConfig(
            command="python",
            timeout=1.0,
        )
        transport = StdioTransport(config)

        # Setup mock process
        mock_stdin = MagicMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()

        response_data = {"jsonrpc": "2.0", "id": 1, "result": "ok"}
        response_line = json.dumps(response_data).encode() + b"\n"

        # Use an Event to synchronize the response
        response_ready = asyncio.Event()

        async def mock_readline():
            await response_ready.wait()
            response_ready.clear()
            return response_line

        mock_stdout = MagicMock()
        mock_stdout.readline = mock_readline

        mock_process = MagicMock()
        mock_process.stdin = mock_stdin
        mock_process.stdout = mock_stdout
        mock_process.stderr = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await transport.connect()
            await asyncio.sleep(0.01)

            # Request without ID
            request = JSONRPCRequest(method="test")
            assert request.id is None

            async def send_request():
                return await transport.send_request(request)

            # Start request in background
            request_task = asyncio.create_task(send_request())

            # Give time for request to be sent
            await asyncio.sleep(0.01)

            # Signal that response is ready
            response_ready.set()

            # Wait for response
            await request_task

            # After sending, ID should be assigned
            assert request.id is not None
            assert request.id == 1

            await transport.disconnect()

    @pytest.mark.asyncio
    async def test_sse_auto_assigns_id(self, sse_transport_config: SSETransportConfig):
        """Test that SSETransport auto-assigns request IDs."""
        transport = SSETransport(sse_transport_config)
        await transport.connect()

        mock_response = MagicMock()
        mock_response.json.return_value = {"jsonrpc": "2.0", "id": 1, "result": "ok"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            request = JSONRPCRequest(method="test")
            assert request.id is None

            await transport.send_request(request)
            assert request.id is not None
            assert request.id == 1
