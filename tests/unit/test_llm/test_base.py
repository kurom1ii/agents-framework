"""Unit tests for the LLM base module."""

from __future__ import annotations

import asyncio
import random
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents_framework.llm.base import (
    AuthenticationError,
    BaseLLMProvider,
    InvalidRequestError,
    LLMConfig,
    LLMProviderError,
    LLMResponse,
    Message,
    MessageRole,
    RateLimitError,
    RetryConfig,
    ToolCall,
    ToolDefinition,
)

from typing import AsyncIterator


# ============================================================================
# Helper Classes for Testing
# ============================================================================


class ConcreteLLMProviderHelper(BaseLLMProvider):
    """Concrete implementation of BaseLLMProvider for testing in test file."""

    def __init__(
        self,
        config: LLMConfig,
        responses: Optional[List[LLMResponse]] = None,
        should_fail: bool = False,
        fail_times: int = 0,
        failure_exception: Optional[Exception] = None,
    ):
        super().__init__(config)
        self.responses = responses or []
        self.should_fail = should_fail
        self.fail_times = fail_times
        self.failure_exception = failure_exception or Exception("Test failure")
        self.call_count = 0
        self.generate_call_count = 0
        self._supports_tools = True

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response."""
        self.generate_call_count += 1
        self.call_count += 1

        if self.should_fail:
            raise self.failure_exception

        if self.fail_times > 0 and self.call_count <= self.fail_times:
            raise self.failure_exception

        if self.responses:
            idx = min(self.generate_call_count - 1, len(self.responses) - 1)
            return self.responses[idx]

        return LLMResponse(
            content="Test response",
            model=self.config.model,
            finish_reason="stop",
        )

    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream response chunks."""
        if self.should_fail:
            raise self.failure_exception

        async def _stream() -> AsyncIterator[str]:
            chunks = ["Test", " ", "stream", " ", "response"]
            for chunk in chunks:
                yield chunk

        async for chunk in _stream():
            yield chunk

    def supports_tools(self) -> bool:
        """Return whether tools are supported."""
        return self._supports_tools


# ============================================================================
# MessageRole Tests
# ============================================================================


class TestMessageRole:
    """Tests for MessageRole enum."""

    def test_message_role_values(self):
        """Test that MessageRole has correct string values."""
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.TOOL.value == "tool"

    def test_message_role_is_string_enum(self):
        """Test that MessageRole inherits from str."""
        assert isinstance(MessageRole.SYSTEM, str)
        assert isinstance(MessageRole.USER, str)
        assert isinstance(MessageRole.ASSISTANT, str)
        assert isinstance(MessageRole.TOOL, str)

    def test_message_role_string_comparison(self):
        """Test that MessageRole can be compared with strings."""
        assert MessageRole.SYSTEM == "system"
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.TOOL == "tool"

    def test_message_role_members(self):
        """Test that all expected members exist."""
        members = list(MessageRole)
        assert len(members) == 4
        assert MessageRole.SYSTEM in members
        assert MessageRole.USER in members
        assert MessageRole.ASSISTANT in members
        assert MessageRole.TOOL in members

    def test_message_role_from_value(self):
        """Test creating MessageRole from string value."""
        assert MessageRole("system") == MessageRole.SYSTEM
        assert MessageRole("user") == MessageRole.USER
        assert MessageRole("assistant") == MessageRole.ASSISTANT
        assert MessageRole("tool") == MessageRole.TOOL

    def test_message_role_invalid_value(self):
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError):
            MessageRole("invalid")


# ============================================================================
# ToolCall Tests
# ============================================================================


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_tool_call_creation(self):
        """Test basic ToolCall creation."""
        tool_call = ToolCall(
            id="call_123",
            name="search",
            arguments={"query": "test"},
        )
        assert tool_call.id == "call_123"
        assert tool_call.name == "search"
        assert tool_call.arguments == {"query": "test"}

    def test_tool_call_with_complex_arguments(self):
        """Test ToolCall with complex nested arguments."""
        arguments = {
            "query": "test",
            "filters": {"date": "2024-01-01", "source": ["web", "news"]},
            "limit": 10,
            "nested": {"deep": {"value": True}},
        }
        tool_call = ToolCall(id="call_456", name="complex_search", arguments=arguments)
        assert tool_call.arguments == arguments
        assert tool_call.arguments["filters"]["source"] == ["web", "news"]

    def test_tool_call_with_empty_arguments(self):
        """Test ToolCall with empty arguments."""
        tool_call = ToolCall(id="call_789", name="no_args", arguments={})
        assert tool_call.arguments == {}

    def test_tool_call_equality(self):
        """Test ToolCall equality comparison."""
        call1 = ToolCall(id="call_1", name="test", arguments={"a": 1})
        call2 = ToolCall(id="call_1", name="test", arguments={"a": 1})
        call3 = ToolCall(id="call_2", name="test", arguments={"a": 1})

        assert call1 == call2
        assert call1 != call3

    def test_tool_call_hash(self):
        """Test that ToolCall is hashable (default dataclass behavior)."""
        # Note: Default dataclass is not hashable when mutable fields exist
        # This test documents that behavior
        tool_call = ToolCall(id="call_1", name="test", arguments={"a": 1})
        with pytest.raises(TypeError):
            hash(tool_call)


# ============================================================================
# ToolDefinition Tests
# ============================================================================


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    def test_tool_definition_creation(self):
        """Test basic ToolDefinition creation."""
        tool_def = ToolDefinition(
            name="search",
            description="Search for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
            },
        )
        assert tool_def.name == "search"
        assert tool_def.description == "Search for information"
        assert tool_def.parameters["type"] == "object"

    def test_tool_definition_with_json_schema(
        self, tool_definition_search: ToolDefinition
    ):
        """Test ToolDefinition with full JSON schema."""
        assert tool_definition_search.parameters["type"] == "object"
        assert "query" in tool_definition_search.parameters["properties"]
        assert tool_definition_search.parameters["required"] == ["query"]

    def test_tool_definition_with_empty_parameters(self):
        """Test ToolDefinition with empty parameters."""
        tool_def = ToolDefinition(
            name="ping",
            description="Simple ping command",
            parameters={},
        )
        assert tool_def.parameters == {}

    def test_tool_definition_equality(self):
        """Test ToolDefinition equality."""
        def1 = ToolDefinition(name="test", description="desc", parameters={"a": 1})
        def2 = ToolDefinition(name="test", description="desc", parameters={"a": 1})
        def3 = ToolDefinition(name="test", description="different", parameters={"a": 1})

        assert def1 == def2
        assert def1 != def3


# ============================================================================
# Message Tests
# ============================================================================


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_basic_creation(self):
        """Test basic Message creation."""
        msg = Message(role=MessageRole.USER, content="Hello!")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello!"
        assert msg.name is None
        assert msg.tool_calls is None
        assert msg.tool_call_id is None

    def test_message_with_all_fields(self, message_with_tool_calls: Message):
        """Test Message with all fields populated."""
        assert message_with_tool_calls.role == MessageRole.ASSISTANT
        assert message_with_tool_calls.content == ""
        assert len(message_with_tool_calls.tool_calls) == 2
        assert message_with_tool_calls.tool_calls[0].name == "search"
        assert message_with_tool_calls.tool_calls[1].name == "calculate"

    def test_message_tool_response(self, message_tool_response: Message):
        """Test tool response message."""
        assert message_tool_response.role == MessageRole.TOOL
        assert message_tool_response.tool_call_id == "call_1"
        assert "results" in message_tool_response.content.lower()

    def test_message_with_name(self, message_with_name: Message):
        """Test message with name field."""
        assert message_with_name.name == "test_user"
        assert message_with_name.role == MessageRole.USER


class TestMessageToDict:
    """Tests for Message.to_dict() method."""

    def test_to_dict_basic(self, message_user: Message):
        """Test to_dict with basic message."""
        result = message_user.to_dict()
        assert result == {
            "role": "user",
            "content": "Hello!",
        }

    def test_to_dict_with_name(self, message_with_name: Message):
        """Test to_dict includes name when present."""
        result = message_with_name.to_dict()
        assert result["name"] == "test_user"
        assert result["role"] == "user"
        assert result["content"] == "Hello!"

    def test_to_dict_with_tool_calls(self, message_with_tool_calls: Message):
        """Test to_dict includes tool_calls when present."""
        result = message_with_tool_calls.to_dict()
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0] == {
            "id": "call_1",
            "name": "search",
            "arguments": {"query": "test"},
        }
        assert result["tool_calls"][1] == {
            "id": "call_2",
            "name": "calculate",
            "arguments": {"expression": "2+2"},
        }

    def test_to_dict_with_tool_call_id(self, message_tool_response: Message):
        """Test to_dict includes tool_call_id when present."""
        result = message_tool_response.to_dict()
        assert result["tool_call_id"] == "call_1"
        assert result["role"] == "tool"

    def test_to_dict_excludes_none_values(self, message_user: Message):
        """Test to_dict excludes None optional fields."""
        result = message_user.to_dict()
        assert "name" not in result
        assert "tool_calls" not in result
        assert "tool_call_id" not in result

    def test_to_dict_all_message_roles(self):
        """Test to_dict with all message roles."""
        for role in MessageRole:
            msg = Message(role=role, content="test")
            result = msg.to_dict()
            assert result["role"] == role.value


# ============================================================================
# LLMResponse Tests
# ============================================================================


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_llm_response_basic_creation(self):
        """Test basic LLMResponse creation."""
        response = LLMResponse(
            content="Hello!",
            model="gpt-4",
        )
        assert response.content == "Hello!"
        assert response.model == "gpt-4"
        assert response.finish_reason is None
        assert response.tool_calls is None
        assert response.usage is None
        assert response.raw_response is None

    def test_llm_response_with_all_fields(self, response_with_tool_calls: LLMResponse):
        """Test LLMResponse with all fields."""
        assert response_with_tool_calls.content == ""
        assert response_with_tool_calls.model == "test-model"
        assert response_with_tool_calls.finish_reason == "tool_calls"
        assert len(response_with_tool_calls.tool_calls) == 1
        assert response_with_tool_calls.usage["total_tokens"] == 15

    def test_llm_response_with_raw_response(self):
        """Test LLMResponse with raw_response field."""
        raw = {"id": "chatcmpl-123", "object": "chat.completion"}
        response = LLMResponse(
            content="Test",
            model="test",
            raw_response=raw,
        )
        assert response.raw_response == raw


class TestLLMResponseHasToolCalls:
    """Tests for LLMResponse.has_tool_calls property."""

    def test_has_tool_calls_true(self, response_with_tool_calls: LLMResponse):
        """Test has_tool_calls returns True when tool_calls present."""
        assert response_with_tool_calls.has_tool_calls is True

    def test_has_tool_calls_false_none(self, response_without_tool_calls: LLMResponse):
        """Test has_tool_calls returns False when tool_calls is None."""
        assert response_without_tool_calls.has_tool_calls is False

    def test_has_tool_calls_false_empty_list(self):
        """Test has_tool_calls returns False when tool_calls is empty list."""
        response = LLMResponse(
            content="Test",
            model="test",
            tool_calls=[],
        )
        assert response.has_tool_calls is False

    def test_has_tool_calls_with_multiple_calls(self):
        """Test has_tool_calls with multiple tool calls."""
        response = LLMResponse(
            content="",
            model="test",
            tool_calls=[
                ToolCall(id="1", name="a", arguments={}),
                ToolCall(id="2", name="b", arguments={}),
                ToolCall(id="3", name="c", arguments={}),
            ],
        )
        assert response.has_tool_calls is True


# ============================================================================
# RetryConfig Tests
# ============================================================================


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_retry_config_defaults(self):
        """Test RetryConfig default values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert Exception in config.retryable_exceptions
        assert 429 in config.retryable_status_codes
        assert 500 in config.retryable_status_codes

    def test_retry_config_custom_values(self):
        """Test RetryConfig with custom values."""
        config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=3.0,
            jitter=False,
            retryable_exceptions=(ValueError, TypeError),
            retryable_status_codes=(429, 503),
        )
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 3.0
        assert config.jitter is False
        assert config.retryable_exceptions == (ValueError, TypeError)
        assert config.retryable_status_codes == (429, 503)


class TestRetryConfigGetDelay:
    """Tests for RetryConfig.get_delay() method."""

    def test_get_delay_without_jitter(self, retry_config_no_jitter: RetryConfig):
        """Test get_delay returns deterministic values without jitter."""
        # Attempt 0: 1.0 * 2^0 = 1.0
        assert retry_config_no_jitter.get_delay(0) == 1.0
        # Attempt 1: 1.0 * 2^1 = 2.0
        assert retry_config_no_jitter.get_delay(1) == 2.0
        # Attempt 2: 1.0 * 2^2 = 4.0
        assert retry_config_no_jitter.get_delay(2) == 4.0
        # Attempt 3: 1.0 * 2^3 = 8.0
        assert retry_config_no_jitter.get_delay(3) == 8.0

    def test_get_delay_respects_max_delay(self, retry_config_no_jitter: RetryConfig):
        """Test get_delay is capped by max_delay."""
        # Attempt 10: 1.0 * 2^10 = 1024, but max is 10.0
        assert retry_config_no_jitter.get_delay(10) == 10.0
        # Very high attempt number
        assert retry_config_no_jitter.get_delay(100) == 10.0

    def test_get_delay_with_jitter(self, retry_config_with_jitter: RetryConfig):
        """Test get_delay adds jitter when enabled."""
        # With jitter, delay is multiplied by (0.5 + random())
        # This gives range [0.5 * base_delay, 1.5 * base_delay]
        random.seed(42)  # Set seed for reproducibility
        delay = retry_config_with_jitter.get_delay(0)
        # Base delay is 1.0, so result should be between 0.5 and 1.5
        assert 0.5 <= delay <= 1.5

    def test_get_delay_jitter_variance(self, retry_config_with_jitter: RetryConfig):
        """Test that jitter produces varied delays."""
        delays = [retry_config_with_jitter.get_delay(0) for _ in range(100)]
        unique_delays = set(delays)
        # Should have many unique values due to jitter
        assert len(unique_delays) > 50

    @pytest.mark.parametrize(
        "attempt,expected",
        [
            (0, 1.0),
            (1, 2.0),
            (2, 4.0),
            (3, 8.0),
            (4, 10.0),  # Capped at max_delay
            (5, 10.0),
        ],
    )
    def test_get_delay_parametrized(
        self, retry_config_no_jitter: RetryConfig, attempt: int, expected: float
    ):
        """Parametrized test for get_delay at various attempts."""
        assert retry_config_no_jitter.get_delay(attempt) == expected

    def test_get_delay_with_different_base(self):
        """Test get_delay with different exponential base."""
        config = RetryConfig(
            base_delay=1.0,
            max_delay=100.0,
            exponential_base=3.0,
            jitter=False,
        )
        assert config.get_delay(0) == 1.0  # 1 * 3^0
        assert config.get_delay(1) == 3.0  # 1 * 3^1
        assert config.get_delay(2) == 9.0  # 1 * 3^2
        assert config.get_delay(3) == 27.0  # 1 * 3^3

    def test_get_delay_with_fractional_base_delay(self):
        """Test get_delay with fractional base delay."""
        config = RetryConfig(
            base_delay=0.1,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=False,
        )
        assert config.get_delay(0) == 0.1
        assert config.get_delay(1) == 0.2
        assert config.get_delay(2) == 0.4


# ============================================================================
# LLMConfig Tests
# ============================================================================


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_llm_config_minimal(self):
        """Test LLMConfig with only required fields."""
        config = LLMConfig(model="gpt-4")
        assert config.model == "gpt-4"
        assert config.api_key is None
        assert config.base_url is None
        assert config.temperature == 0.7
        assert config.max_tokens is None
        assert config.timeout == 60.0
        assert config.extra_params == {}

    def test_llm_config_with_all_fields(self):
        """Test LLMConfig with all fields."""
        retry = RetryConfig(max_retries=5)
        config = LLMConfig(
            model="gpt-4-turbo",
            api_key="sk-xxx",
            base_url="https://api.custom.com",
            temperature=0.9,
            max_tokens=2000,
            timeout=120.0,
            retry_config=retry,
            extra_params={"top_p": 0.95, "presence_penalty": 0.1},
        )
        assert config.model == "gpt-4-turbo"
        assert config.api_key == "sk-xxx"
        assert config.base_url == "https://api.custom.com"
        assert config.temperature == 0.9
        assert config.max_tokens == 2000
        assert config.timeout == 120.0
        assert config.retry_config.max_retries == 5
        assert config.extra_params == {"top_p": 0.95, "presence_penalty": 0.1}

    def test_llm_config_post_init_creates_retry_config(self):
        """Test that __post_init__ creates default RetryConfig if None."""
        config = LLMConfig(model="test")
        assert config.retry_config is not None
        assert isinstance(config.retry_config, RetryConfig)
        assert config.retry_config.max_retries == 3

    def test_llm_config_post_init_preserves_custom_retry_config(self):
        """Test that __post_init__ preserves custom RetryConfig."""
        custom_retry = RetryConfig(max_retries=10)
        config = LLMConfig(model="test", retry_config=custom_retry)
        assert config.retry_config.max_retries == 10

    def test_llm_config_extra_params_mutation(self):
        """Test that extra_params can be mutated after creation."""
        config = LLMConfig(model="test")
        config.extra_params["new_param"] = "value"
        assert config.extra_params["new_param"] == "value"


# ============================================================================
# BaseLLMProvider Tests
# ============================================================================


class TestBaseLLMProviderInit:
    """Tests for BaseLLMProvider initialization."""

    def test_provider_init(self, concrete_provider, basic_llm_config: LLMConfig):
        """Test provider initialization stores config."""
        # The concrete_provider fixture creates a provider with basic_llm_config
        provider = concrete_provider
        assert provider.config == basic_llm_config
        assert provider._client is None

    def test_provider_config_access(self, concrete_provider):
        """Test provider config is accessible."""
        assert concrete_provider.config.model == "test-model"
        assert concrete_provider.config.api_key == "test-key"


class TestBaseLLMProviderFormatMessages:
    """Tests for BaseLLMProvider._format_messages() method."""

    def test_format_messages_basic(
        self, concrete_provider, message_system: Message, message_user: Message
    ):
        """Test formatting basic messages."""
        messages = [message_system, message_user]
        result = concrete_provider._format_messages(messages)

        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "You are a helpful assistant."}
        assert result[1] == {"role": "user", "content": "Hello!"}

    def test_format_messages_with_tool_calls(
        self, concrete_provider, message_with_tool_calls: Message
    ):
        """Test formatting messages with tool calls."""
        result = concrete_provider._format_messages([message_with_tool_calls])

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "tool_calls" in result[0]
        assert len(result[0]["tool_calls"]) == 2

    def test_format_messages_empty_list(self, concrete_provider):
        """Test formatting empty message list."""
        result = concrete_provider._format_messages([])
        assert result == []

    def test_format_messages_preserves_order(self, concrete_provider):
        """Test that message order is preserved."""
        messages = [
            Message(role=MessageRole.SYSTEM, content="1"),
            Message(role=MessageRole.USER, content="2"),
            Message(role=MessageRole.ASSISTANT, content="3"),
            Message(role=MessageRole.USER, content="4"),
        ]
        result = concrete_provider._format_messages(messages)

        assert [m["content"] for m in result] == ["1", "2", "3", "4"]


class TestBaseLLMProviderFormatTools:
    """Tests for BaseLLMProvider._format_tools() method."""

    def test_format_tools_basic(
        self,
        concrete_provider,
        tool_definition_search: ToolDefinition,
        tool_definition_calculate: ToolDefinition,
    ):
        """Test formatting tool definitions."""
        tools = [tool_definition_search, tool_definition_calculate]
        result = concrete_provider._format_tools(tools)

        assert len(result) == 2
        assert result[0]["name"] == "search"
        assert result[0]["description"] == "Search for information"
        assert result[0]["parameters"]["type"] == "object"
        assert result[1]["name"] == "calculate"

    def test_format_tools_none(self, concrete_provider):
        """Test formatting None tools returns None."""
        result = concrete_provider._format_tools(None)
        assert result is None

    def test_format_tools_empty_list(self, concrete_provider):
        """Test formatting empty list returns None."""
        result = concrete_provider._format_tools([])
        assert result is None

    def test_format_tools_single_tool(
        self, concrete_provider, tool_definition_search: ToolDefinition
    ):
        """Test formatting single tool."""
        result = concrete_provider._format_tools([tool_definition_search])

        assert len(result) == 1
        assert result[0]["name"] == "search"


class TestBaseLLMProviderRetryWithBackoff:
    """Tests for BaseLLMProvider._retry_with_backoff() method."""

    @pytest.mark.asyncio
    async def test_retry_success_first_attempt(self, concrete_provider):
        """Test successful execution on first attempt."""

        async def success_func():
            return "success"

        result = await concrete_provider._retry_with_backoff(success_func)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self, intermittent_provider):
        """Test successful execution after initial failures."""
        call_count = 0

        async def intermittent_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"

        result = await intermittent_provider._retry_with_backoff(intermittent_func)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, provider_with_retry):
        """Test that retries are exhausted after max_retries."""
        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise Exception("Always fails")

        with pytest.raises(Exception, match="Always fails"):
            await provider_with_retry._retry_with_backoff(always_fail)

        # max_retries=3, so 4 total attempts (0, 1, 2, 3)
        assert call_count == 4

    @pytest.mark.asyncio
    async def test_retry_with_status_code(self, provider_with_retry):
        """Test retry on retryable status codes."""
        call_count = 0

        class HTTPError(Exception):
            def __init__(self, status_code):
                self.status_code = status_code
                super().__init__(f"HTTP {status_code}")

        async def rate_limited():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise HTTPError(429)
            return "success"

        result = await provider_with_retry._retry_with_backoff(rate_limited)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_with_response_status_code(self, provider_with_retry):
        """Test retry checks response.status_code."""
        call_count = 0

        class ResponseError(Exception):
            def __init__(self, status_code):
                self.response = MagicMock(status_code=status_code)
                super().__init__(f"HTTP {status_code}")

        async def server_error():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ResponseError(503)
            return "success"

        result = await provider_with_retry._retry_with_backoff(server_error)
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_non_retryable_exception(self, provider_with_retry):
        """Test that non-retryable exceptions are raised immediately."""
        # Configure provider to only retry ValueError
        provider_with_retry.config.retry_config = RetryConfig(
            max_retries=3,
            retryable_exceptions=(ValueError,),
            retryable_status_codes=(),
        )
        call_count = 0

        async def type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("Not retryable")

        with pytest.raises(TypeError, match="Not retryable"):
            await provider_with_retry._retry_with_backoff(type_error)

        # Should fail on first attempt since TypeError is not retryable
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_passes_args_and_kwargs(self, concrete_provider):
        """Test that args and kwargs are passed to the function."""
        received_args = []
        received_kwargs = {}

        async def capture_args(*args, **kwargs):
            received_args.extend(args)
            received_kwargs.update(kwargs)
            return "done"

        await concrete_provider._retry_with_backoff(
            capture_args, 1, 2, 3, key1="value1", key2="value2"
        )

        assert received_args == [1, 2, 3]
        assert received_kwargs == {"key1": "value1", "key2": "value2"}

    @pytest.mark.asyncio
    async def test_retry_uses_default_config_if_none(self, basic_llm_config: LLMConfig):
        """Test retry uses default RetryConfig if config is None."""
        # Create config with None retry_config
        config = LLMConfig(model="test")
        config.retry_config = None
        provider = ConcreteLLMProviderHelper(config)

        call_count = 0

        async def intermittent():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary")
            return "ok"

        result = await provider._retry_with_backoff(intermittent)
        assert result == "ok"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_delay_is_applied(self):
        """Test that delays are actually applied between retries."""
        config = LLMConfig(
            model="test",
            retry_config=RetryConfig(
                max_retries=2,
                base_delay=0.05,  # 50ms
                max_delay=1.0,
                jitter=False,
            ),
        )
        provider = ConcreteLLMProviderHelper(config)
        call_count = 0
        call_times = []

        async def timed_fail():
            nonlocal call_count
            call_times.append(asyncio.get_event_loop().time())
            call_count += 1
            if call_count < 3:
                raise Exception("Fail")
            return "ok"

        result = await provider._retry_with_backoff(timed_fail)
        assert result == "ok"
        assert call_count == 3

        # Check delays between calls
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        # First retry: 0.05 * 2^0 = 0.05
        assert delay1 >= 0.04  # Allow small tolerance
        # Second retry: 0.05 * 2^1 = 0.1
        assert delay2 >= 0.08


# ============================================================================
# LLMProviderError Tests
# ============================================================================


class TestLLMProviderError:
    """Tests for LLMProviderError exception class."""

    def test_provider_error_basic(self):
        """Test basic LLMProviderError creation."""
        error = LLMProviderError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.provider is None
        assert error.status_code is None
        assert error.response is None

    def test_provider_error_with_all_fields(self, provider_error: LLMProviderError):
        """Test LLMProviderError with all fields."""
        assert str(provider_error) == "Something went wrong"
        assert provider_error.provider == "test-provider"
        assert provider_error.status_code == 500
        assert provider_error.response == {"error": "Internal server error"}

    def test_provider_error_inheritance(self):
        """Test LLMProviderError inherits from Exception."""
        error = LLMProviderError("test")
        assert isinstance(error, Exception)

    def test_provider_error_can_be_raised(self):
        """Test LLMProviderError can be raised and caught."""
        with pytest.raises(LLMProviderError) as exc_info:
            raise LLMProviderError("Test error", provider="openai")

        assert str(exc_info.value) == "Test error"
        assert exc_info.value.provider == "openai"


class TestRateLimitError:
    """Tests for RateLimitError exception class."""

    def test_rate_limit_error_creation(self, rate_limit_error: RateLimitError):
        """Test RateLimitError creation."""
        assert str(rate_limit_error) == "Rate limit exceeded"
        assert rate_limit_error.provider == "test-provider"
        assert rate_limit_error.status_code == 429

    def test_rate_limit_error_inheritance(self):
        """Test RateLimitError inherits from LLMProviderError."""
        error = RateLimitError("rate limited")
        assert isinstance(error, LLMProviderError)
        assert isinstance(error, Exception)

    def test_rate_limit_error_can_be_caught_as_provider_error(self):
        """Test RateLimitError can be caught as LLMProviderError."""
        with pytest.raises(LLMProviderError):
            raise RateLimitError("Too many requests")


class TestAuthenticationError:
    """Tests for AuthenticationError exception class."""

    def test_authentication_error_creation(self):
        """Test AuthenticationError creation."""
        error = AuthenticationError(
            "Invalid API key",
            provider="openai",
            status_code=401,
        )
        assert str(error) == "Invalid API key"
        assert error.status_code == 401

    def test_authentication_error_inheritance(self):
        """Test AuthenticationError inherits from LLMProviderError."""
        error = AuthenticationError("Unauthorized")
        assert isinstance(error, LLMProviderError)

    def test_authentication_error_can_be_caught_separately(self):
        """Test AuthenticationError can be caught specifically."""
        with pytest.raises(AuthenticationError):
            raise AuthenticationError("Bad credentials")


class TestInvalidRequestError:
    """Tests for InvalidRequestError exception class."""

    def test_invalid_request_error_creation(self):
        """Test InvalidRequestError creation."""
        error = InvalidRequestError(
            "Invalid model name",
            provider="anthropic",
            status_code=400,
            response={"error": "Model not found"},
        )
        assert str(error) == "Invalid model name"
        assert error.status_code == 400
        assert error.response["error"] == "Model not found"

    def test_invalid_request_error_inheritance(self):
        """Test InvalidRequestError inherits from LLMProviderError."""
        error = InvalidRequestError("Bad request")
        assert isinstance(error, LLMProviderError)

    def test_error_hierarchy(self):
        """Test all error types can be caught as base Exception."""
        errors = [
            LLMProviderError("base"),
            RateLimitError("rate"),
            AuthenticationError("auth"),
            InvalidRequestError("invalid"),
        ]

        for error in errors:
            with pytest.raises(Exception):
                raise error


# ============================================================================
# Integration Tests
# ============================================================================


class TestLLMBaseIntegration:
    """Integration tests for LLM base components."""

    @pytest.mark.asyncio
    async def test_full_message_flow(self, concrete_provider):
        """Test complete message creation and formatting flow."""
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are helpful."),
            Message(role=MessageRole.USER, content="Hello!"),
        ]

        formatted = concrete_provider._format_messages(messages)
        assert len(formatted) == 2
        assert formatted[0]["role"] == "system"
        assert formatted[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_tool_call_round_trip(self, concrete_provider):
        """Test tool definition to tool call flow."""
        # Define tools
        tools = [
            ToolDefinition(
                name="get_weather",
                description="Get weather info",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            )
        ]

        formatted_tools = concrete_provider._format_tools(tools)
        assert formatted_tools[0]["name"] == "get_weather"

        # Simulate response with tool call
        response = LLMResponse(
            content="",
            model="test",
            tool_calls=[
                ToolCall(id="call_1", name="get_weather", arguments={"city": "NYC"})
            ],
        )

        assert response.has_tool_calls
        assert response.tool_calls[0].arguments["city"] == "NYC"

        # Create tool response message
        tool_response = Message(
            role=MessageRole.TOOL,
            content='{"temperature": 72}',
            tool_call_id="call_1",
        )

        formatted_response = tool_response.to_dict()
        assert formatted_response["role"] == "tool"
        assert formatted_response["tool_call_id"] == "call_1"

    @pytest.mark.asyncio
    async def test_retry_with_generate(self, intermittent_provider):
        """Test retry logic integrates with generate method."""
        # Intermittent provider fails first 2 times
        intermittent_provider.fail_times = 1

        async def generate_with_retry():
            return await intermittent_provider._retry_with_backoff(
                intermittent_provider.generate,
                messages=[Message(role=MessageRole.USER, content="Hi")],
            )

        # This will retry and eventually succeed
        response = await generate_with_retry()
        assert response.content == "Test response"
        assert intermittent_provider.call_count == 2
