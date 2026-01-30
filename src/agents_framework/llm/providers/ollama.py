"""Ollama LLM Provider implementation for local models."""

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
    ToolCall,
    ToolDefinition,
)

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class OllamaProvider(BaseLLMProvider):
    """Ollama provider for local LLM models.

    Supports:
    - Llama 2, Llama 3
    - Mistral, Mixtral
    - CodeLlama
    - Other Ollama-supported models
    - Streaming responses
    - Tool calling (for supported models)
    """

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, config: LLMConfig):
        """Initialize Ollama provider.

        Args:
            config: LLM configuration with model name, base_url, etc.

        Raises:
            ImportError: If httpx package is not installed.
        """
        super().__init__(config)
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx package is required for OllamaProvider. "
                "Install it with: pip install httpx"
            )
        self._base_url = config.base_url or self.DEFAULT_BASE_URL
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=config.timeout,
        )

    def supports_tools(self) -> bool:
        """Ollama tool support depends on the model.

        Models like Llama 3.1+ and Mistral support function calling.
        """
        # Check if model name suggests tool support
        tool_capable_models = [
            "llama3.1", "llama3.2", "llama3.3",
            "mistral", "mixtral",
            "qwen2", "qwen2.5",
            "command-r",
        ]
        model_lower = self.config.model.lower()
        return any(model in model_lower for model in tool_capable_models)

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for Ollama API."""
        formatted = []
        for msg in messages:
            message_dict: Dict[str, Any] = {
                "role": msg.role.value,
                "content": msg.content,
            }

            # Handle tool calls
            if msg.tool_calls and msg.role == MessageRole.ASSISTANT:
                message_dict["tool_calls"] = [
                    {
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    }
                    for tc in msg.tool_calls
                ]

            formatted.append(message_dict)
        return formatted

    def _format_tools(
        self, tools: Optional[List[ToolDefinition]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Format tools for Ollama API."""
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

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response using Ollama API.

        Args:
            messages: List of messages in the conversation.
            tools: Optional list of tool definitions.
            **kwargs: Additional parameters passed to the API.

        Returns:
            LLMResponse with the generated content.
        """
        async def _make_request() -> LLMResponse:
            request_data: Dict[str, Any] = {
                "model": self.config.model,
                "messages": self._format_messages(messages),
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.config.temperature),
                },
            }

            if self.config.max_tokens:
                request_data["options"]["num_predict"] = self.config.max_tokens

            formatted_tools = self._format_tools(tools)
            if formatted_tools and self.supports_tools():
                request_data["tools"] = formatted_tools

            # Merge extra params
            request_data.update(self.config.extra_params)

            try:
                response = await self._client.post(
                    "/api/chat",
                    json=request_data,
                )
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as e:
                raise LLMProviderError(
                    f"Ollama API error: {e.response.text}",
                    provider="ollama",
                    status_code=e.response.status_code,
                )
            except Exception as e:
                raise LLMProviderError(str(e), provider="ollama")

            message = data.get("message", {})
            content = message.get("content", "")

            # Parse tool calls if present
            tool_calls = None
            if "tool_calls" in message:
                tool_calls = [
                    ToolCall(
                        id=f"call_{i}",
                        name=tc.get("function", {}).get("name", ""),
                        arguments=tc.get("function", {}).get("arguments", {}),
                    )
                    for i, tc in enumerate(message.get("tool_calls", []))
                ]

            return LLMResponse(
                content=content,
                model=data.get("model", self.config.model),
                finish_reason=data.get("done_reason", "stop"),
                tool_calls=tool_calls,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                },
                raw_response=data,
            )

        return await self._retry_with_backoff(_make_request)

    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a response from Ollama API.

        Args:
            messages: List of messages in the conversation.
            tools: Optional list of tool definitions.
            **kwargs: Additional parameters passed to the API.

        Yields:
            String chunks of the response as they arrive.
        """
        request_data: Dict[str, Any] = {
            "model": self.config.model,
            "messages": self._format_messages(messages),
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
            },
        }

        if self.config.max_tokens:
            request_data["options"]["num_predict"] = self.config.max_tokens

        formatted_tools = self._format_tools(tools)
        if formatted_tools and self.supports_tools():
            request_data["tools"] = formatted_tools

        # Merge extra params
        request_data.update(self.config.extra_params)

        try:
            async with self._client.stream(
                "POST",
                "/api/chat",
                json=request_data,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except httpx.HTTPStatusError as e:
            raise LLMProviderError(
                f"Ollama API error: {e.response.text}",
                provider="ollama",
                status_code=e.response.status_code,
            )
        except Exception as e:
            raise LLMProviderError(str(e), provider="ollama")

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
