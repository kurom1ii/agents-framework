"""OpenAI LLM Provider implementation."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from ..base import (
    BaseLLMProvider,
    LLMConfig,
    LLMProviderError,
    LLMResponse,
    Message,
    MessageRole,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
    ToolCall,
    ToolDefinition,
)

try:
    from openai import AsyncOpenAI, APIError, RateLimitError as OpenAIRateLimitError
    from openai import AuthenticationError as OpenAIAuthError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider for GPT models.

    Supports:
    - GPT-4, GPT-4 Turbo, GPT-4o
    - GPT-3.5 Turbo
    - Tool/function calling
    - Streaming responses
    """

    def __init__(self, config: LLMConfig):
        """Initialize OpenAI provider.

        Args:
            config: LLM configuration with model, api_key, etc.

        Raises:
            ImportError: If openai package is not installed.
        """
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required for OpenAIProvider. "
                "Install it with: pip install openai"
            )
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )

    def supports_tools(self) -> bool:
        """OpenAI supports tool/function calling."""
        return True

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for OpenAI API."""
        formatted = []
        for msg in messages:
            message_dict: Dict[str, Any] = {
                "role": msg.role.value,
                "content": msg.content,
            }

            if msg.name and msg.role == MessageRole.TOOL:
                message_dict["name"] = msg.name

            if msg.tool_call_id and msg.role == MessageRole.TOOL:
                message_dict["tool_call_id"] = msg.tool_call_id

            if msg.tool_calls and msg.role == MessageRole.ASSISTANT:
                message_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                            if isinstance(tc.arguments, dict)
                            else tc.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]

            formatted.append(message_dict)
        return formatted

    def _format_tools(
        self, tools: Optional[List[ToolDefinition]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Format tools for OpenAI API."""
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def _handle_error(self, error: Exception) -> None:
        """Convert OpenAI errors to our error types."""
        if OPENAI_AVAILABLE:
            if isinstance(error, OpenAIRateLimitError):
                raise RateLimitError(
                    str(error),
                    provider="openai",
                    status_code=429,
                )
            elif isinstance(error, OpenAIAuthError):
                raise AuthenticationError(
                    str(error),
                    provider="openai",
                    status_code=401,
                )
        raise LLMProviderError(str(error), provider="openai")

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response using OpenAI API.

        Args:
            messages: List of messages in the conversation.
            tools: Optional list of tool definitions.
            **kwargs: Additional parameters passed to the API.

        Returns:
            LLMResponse with the generated content.
        """
        async def _make_request() -> LLMResponse:
            request_params: Dict[str, Any] = {
                "model": self.config.model,
                "messages": self._format_messages(messages),
                "temperature": kwargs.get("temperature", self.config.temperature),
            }

            if self.config.max_tokens:
                request_params["max_tokens"] = self.config.max_tokens

            formatted_tools = self._format_tools(tools)
            if formatted_tools:
                request_params["tools"] = formatted_tools

            # Merge extra params
            request_params.update(self.config.extra_params)
            request_params.update(kwargs)

            # Remove non-API params
            request_params.pop("temperature", None)
            request_params["temperature"] = kwargs.get(
                "temperature", self.config.temperature
            )

            try:
                response = await self._client.chat.completions.create(**request_params)
            except Exception as e:
                self._handle_error(e)

            choice = response.choices[0]
            message = choice.message

            # Parse tool calls if present
            tool_calls = None
            if message.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments)
                        if isinstance(tc.function.arguments, str)
                        else tc.function.arguments,
                    )
                    for tc in message.tool_calls
                ]

            return LLMResponse(
                content=message.content or "",
                model=response.model,
                finish_reason=choice.finish_reason,
                tool_calls=tool_calls,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                if response.usage
                else None,
                raw_response=response,
            )

        return await self._retry_with_backoff(_make_request)

    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a response from OpenAI API.

        Args:
            messages: List of messages in the conversation.
            tools: Optional list of tool definitions.
            **kwargs: Additional parameters passed to the API.

        Yields:
            String chunks of the response as they arrive.
        """
        request_params: Dict[str, Any] = {
            "model": self.config.model,
            "messages": self._format_messages(messages),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": True,
        }

        if self.config.max_tokens:
            request_params["max_tokens"] = self.config.max_tokens

        formatted_tools = self._format_tools(tools)
        if formatted_tools:
            request_params["tools"] = formatted_tools

        # Merge extra params
        request_params.update(self.config.extra_params)
        for key in ["temperature", "stream"]:
            request_params.pop(key, None)
        request_params["temperature"] = kwargs.get(
            "temperature", self.config.temperature
        )
        request_params["stream"] = True

        try:
            stream = await self._client.chat.completions.create(**request_params)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            self._handle_error(e)
