"""Anthropic LLM Provider implementation."""

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
    ThinkingBlock,
    ToolCall,
    ToolDefinition,
)

try:
    from anthropic import AsyncAnthropic, APIError
    from anthropic import RateLimitError as AnthropicRateLimitError
    from anthropic import AuthenticationError as AnthropicAuthError
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    AsyncAnthropic = None


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider for Claude models.

    Supports:
    - Claude 3 Opus, Sonnet, Haiku
    - Claude 3.5 Sonnet
    - Tool/function calling
    - Streaming responses
    """

    def __init__(self, config: LLMConfig):
        """Initialize Anthropic provider.

        Args:
            config: LLM configuration with model, api_key, etc.

        Raises:
            ImportError: If anthropic package is not installed.
        """
        super().__init__(config)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package is required for AnthropicProvider. "
                "Install it with: pip install anthropic"
            )
        self._client = AsyncAnthropic(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )

    def supports_tools(self) -> bool:
        """Anthropic Claude 3+ supports tool/function calling."""
        return True

    def _format_messages(self, messages: List[Message]) -> tuple[str, List[Dict[str, Any]]]:
        """Format messages for Anthropic API.

        Anthropic uses a separate system parameter, so we extract it here.

        Returns:
            Tuple of (system_prompt, formatted_messages)
        """
        system_prompt = ""
        formatted = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
                continue

            if msg.role == MessageRole.TOOL:
                # Tool results in Anthropic format
                formatted.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content,
                        }
                    ],
                })
                continue

            message_dict: Dict[str, Any] = {
                "role": "user" if msg.role == MessageRole.USER else "assistant",
            }

            # Handle tool calls in assistant messages
            if msg.tool_calls and msg.role == MessageRole.ASSISTANT:
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                message_dict["content"] = content
            else:
                message_dict["content"] = msg.content

            formatted.append(message_dict)

        return system_prompt, formatted

    def _format_tools(
        self, tools: Optional[List[ToolDefinition]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Format tools for Anthropic API."""
        if not tools:
            return None
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    def _handle_error(self, error: Exception) -> None:
        """Convert Anthropic errors to our error types."""
        if ANTHROPIC_AVAILABLE:
            if isinstance(error, AnthropicRateLimitError):
                raise RateLimitError(
                    str(error),
                    provider="anthropic",
                    status_code=429,
                )
            elif isinstance(error, AnthropicAuthError):
                raise AuthenticationError(
                    str(error),
                    provider="anthropic",
                    status_code=401,
                )
        raise LLMProviderError(str(error), provider="anthropic")

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response using Anthropic API.

        Args:
            messages: List of messages in the conversation.
            tools: Optional list of tool definitions.
            **kwargs: Additional parameters passed to the API.

        Returns:
            LLMResponse with the generated content.
        """
        async def _make_request() -> LLMResponse:
            system_prompt, formatted_messages = self._format_messages(messages)

            request_params: Dict[str, Any] = {
                "model": self.config.model,
                "messages": formatted_messages,
                "max_tokens": self.config.max_tokens or 4096,
            }

            if system_prompt:
                request_params["system"] = system_prompt

            # Set temperature if not default
            if self.config.temperature != 0.7:
                request_params["temperature"] = self.config.temperature

            formatted_tools = self._format_tools(tools)
            if formatted_tools:
                request_params["tools"] = formatted_tools

            # Merge extra params
            request_params.update(self.config.extra_params)
            request_params.update(kwargs)

            try:
                response = await self._client.messages.create(**request_params)
            except Exception as e:
                self._handle_error(e)

            # Parse response content
            content_parts = []
            tool_calls = []
            thinking_blocks = []

            for block in response.content:
                if block.type == "text":
                    content_parts.append(block.text)
                elif block.type == "thinking":
                    # Extended thinking block
                    thinking_blocks.append(
                        ThinkingBlock(
                            content=block.thinking,
                            signature=getattr(block, "signature", None),
                        )
                    )
                elif block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=block.input if isinstance(block.input, dict) else {},
                        )
                    )

            return LLMResponse(
                content="".join(content_parts),
                model=response.model,
                finish_reason=response.stop_reason,
                tool_calls=tool_calls if tool_calls else None,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                raw_response=response,
                thinking=thinking_blocks if thinking_blocks else None,
            )

        return await self._retry_with_backoff(_make_request)

    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a response from Anthropic API.

        Args:
            messages: List of messages in the conversation.
            tools: Optional list of tool definitions.
            **kwargs: Additional parameters passed to the API.

        Yields:
            String chunks of the response as they arrive.
        """
        system_prompt, formatted_messages = self._format_messages(messages)

        request_params: Dict[str, Any] = {
            "model": self.config.model,
            "messages": formatted_messages,
            "max_tokens": self.config.max_tokens or 4096,
        }

        if system_prompt:
            request_params["system"] = system_prompt

        if self.config.temperature != 0.7:
            request_params["temperature"] = self.config.temperature

        formatted_tools = self._format_tools(tools)
        if formatted_tools:
            request_params["tools"] = formatted_tools

        # Merge extra params
        request_params.update(self.config.extra_params)
        request_params.update(kwargs)

        try:
            async with self._client.messages.stream(**request_params) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            self._handle_error(e)
