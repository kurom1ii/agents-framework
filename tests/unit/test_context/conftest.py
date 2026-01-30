"""Local fixtures for context module tests."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents_framework.context import (
    CharacterEstimateCounter,
    CompactionConfig,
    ContextBudget,
    ContextManager,
    SummaryCache,
    SummaryPrompt,
    SummaryType,
)
from agents_framework.llm.base import LLMConfig, LLMResponse, Message, MessageRole


# ============================================================================
# Message Fixtures
# ============================================================================


@pytest.fixture
def sample_dict_messages() -> List[Dict[str, Any]]:
    """Create sample messages as dictionaries."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "Can you help me with a task?"},
        {"role": "assistant", "content": "Of course! What do you need help with?"},
    ]


@pytest.fixture
def sample_message_objects() -> List[Message]:
    """Create sample messages as Message objects."""
    return [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="Hello, how are you?"),
        Message(role=MessageRole.ASSISTANT, content="I'm doing well, thank you!"),
        Message(role=MessageRole.USER, content="Can you help me with a task?"),
        Message(role=MessageRole.ASSISTANT, content="Of course! What do you need help with?"),
    ]


@pytest.fixture
def large_conversation() -> List[Dict[str, Any]]:
    """Create a larger conversation for compaction tests."""
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(20):
        messages.append({"role": "user", "content": f"User message {i}: " + "x" * 100})
        messages.append(
            {"role": "assistant", "content": f"Assistant response {i}: " + "y" * 150}
        )
    return messages


@pytest.fixture
def messages_with_tool_calls() -> List[Dict[str, Any]]:
    """Create messages including tool calls and results."""
    return [
        {"role": "system", "content": "You are a helpful assistant with tools."},
        {"role": "user", "content": "Search for Python tutorials"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "name": "search", "arguments": {"query": "Python tutorials"}}
            ],
        },
        {
            "role": "tool",
            "content": "Found 10 Python tutorial results...",
            "tool_call_id": "call_1",
        },
        {"role": "assistant", "content": "I found several Python tutorials for you."},
    ]


# ============================================================================
# Context Budget Fixtures
# ============================================================================


@pytest.fixture
def default_budget() -> ContextBudget:
    """Create a default context budget."""
    return ContextBudget()


@pytest.fixture
def small_budget() -> ContextBudget:
    """Create a small context budget for testing limits."""
    return ContextBudget(
        total_tokens=1000,
        system_ratio=0.20,
        conversation_ratio=0.50,
        tools_ratio=0.10,
        response_ratio=0.20,
    )


@pytest.fixture
def large_budget() -> ContextBudget:
    """Create a large context budget."""
    return ContextBudget(
        total_tokens=32000,
        system_ratio=0.10,
        conversation_ratio=0.70,
        tools_ratio=0.10,
        response_ratio=0.10,
    )


# ============================================================================
# Context Manager Fixtures
# ============================================================================


@pytest.fixture
def context_manager() -> ContextManager:
    """Create a context manager with default settings."""
    return ContextManager()


@pytest.fixture
def context_manager_small_budget(small_budget: ContextBudget) -> ContextManager:
    """Create a context manager with a small budget."""
    return ContextManager(budget=small_budget, model="default", max_messages=20)


@pytest.fixture
def context_manager_gpt4() -> ContextManager:
    """Create a context manager configured for GPT-4."""
    budget = ContextBudget(total_tokens=8000)
    return ContextManager(budget=budget, model="gpt-4", max_messages=50)


# ============================================================================
# Compaction Fixtures
# ============================================================================


@pytest.fixture
def default_compaction_config() -> CompactionConfig:
    """Create default compaction configuration."""
    return CompactionConfig()


@pytest.fixture
def aggressive_compaction_config() -> CompactionConfig:
    """Create aggressive compaction configuration."""
    return CompactionConfig(
        token_threshold_ratio=0.70,
        target_token_ratio=0.40,
        max_messages_before_compact=20,
        target_messages_after_compact=10,
        preserve_recent_count=3,
        max_summary_tokens=300,
    )


@pytest.fixture
def lenient_compaction_config() -> CompactionConfig:
    """Create lenient compaction configuration."""
    return CompactionConfig(
        token_threshold_ratio=0.95,
        target_token_ratio=0.80,
        max_messages_before_compact=100,
        target_messages_after_compact=80,
        preserve_recent_count=10,
    )


# ============================================================================
# Summarizer Fixtures
# ============================================================================


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider for summarization tests."""
    provider = MagicMock()
    provider.generate = AsyncMock(
        return_value=LLMResponse(
            content="This is a summary of the conversation discussing user queries and assistant responses.",
            model="test-model",
            finish_reason="stop",
        )
    )
    return provider


@pytest.fixture
def mock_llm_with_key_points():
    """Create a mock LLM that returns key points."""
    provider = MagicMock()
    provider.generate = AsyncMock(
        return_value=LLMResponse(
            content="- First key point about the discussion\n- Second key point\n- Third key point",
            model="test-model",
            finish_reason="stop",
        )
    )
    return provider


@pytest.fixture
def summary_cache() -> SummaryCache:
    """Create a summary cache for testing."""
    return SummaryCache(max_size=50, ttl_seconds=1800)


@pytest.fixture
def default_summary_prompt() -> SummaryPrompt:
    """Create a default summary prompt."""
    return SummaryPrompt()


@pytest.fixture
def technical_summary_prompt() -> SummaryPrompt:
    """Create a technical summary prompt."""
    return SummaryPrompt(
        system_prompt="You are a technical summarizer.",
        user_prompt_template=(
            "Summarize this technical discussion:\n{conversation}\n"
            "Type: {summary_type}, Max tokens: {max_tokens}"
        ),
        summary_type=SummaryType.DETAILED,
    )


# ============================================================================
# Token Counter Fixtures
# ============================================================================


@pytest.fixture
def char_counter() -> CharacterEstimateCounter:
    """Create a character estimate counter."""
    return CharacterEstimateCounter(chars_per_token=4.0)


# ============================================================================
# Helper Functions
# ============================================================================


def create_message(role: str, content: str, **kwargs) -> Dict[str, Any]:
    """Helper to create a message dictionary."""
    msg = {"role": role, "content": content}
    msg.update(kwargs)
    return msg


def create_messages_with_tokens(
    count: int, tokens_per_message: int = 50
) -> List[Dict[str, Any]]:
    """Create messages with approximate token counts.

    Uses ~4 chars per token estimation.
    """
    chars_per_msg = tokens_per_message * 4
    messages = []
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        content = f"Message {i}: " + "x" * (chars_per_msg - 15)
        messages.append({"role": role, "content": content})
    return messages
