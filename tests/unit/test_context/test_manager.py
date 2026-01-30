"""Tests for context manager with token budget allocation.

Tests cover:
- Token counting with different tokenizers
- Token window management
- Message window management
- Context budget allocation and tracking
- ContextManager core functionality
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from agents_framework.context import (
    CharacterEstimateCounter,
    ContextBudget,
    ContextManager,
    ContextUsage,
    MessageWindow,
    TokenizerType,
    TokenWindow,
    get_token_counter,
    MODEL_TOKENIZER_MAP,
)
from agents_framework.llm.base import Message, MessageRole


# ============================================================================
# CharacterEstimateCounter Tests
# ============================================================================


class TestCharacterEstimateCounter:
    """Tests for CharacterEstimateCounter."""

    def test_init_default_chars_per_token(self):
        """Test initialization with default chars per token."""
        counter = CharacterEstimateCounter()
        # Default is 4.0 chars per token
        assert counter.count("test") == 1  # 4 chars / 4 = 1 token

    def test_init_custom_chars_per_token(self):
        """Test initialization with custom chars per token."""
        counter = CharacterEstimateCounter(chars_per_token=2.0)
        assert counter.count("test") == 2  # 4 chars / 2 = 2 tokens

    def test_count_empty_string(self):
        """Test counting tokens in empty string."""
        counter = CharacterEstimateCounter()
        # Minimum is 1 token
        assert counter.count("") == 1

    def test_count_short_text(self):
        """Test counting tokens in short text."""
        counter = CharacterEstimateCounter(chars_per_token=4.0)
        assert counter.count("Hello World!") == 3  # 12 chars / 4 = 3

    def test_count_long_text(self):
        """Test counting tokens in longer text."""
        counter = CharacterEstimateCounter(chars_per_token=4.0)
        text = "a" * 100
        assert counter.count(text) == 25  # 100 / 4 = 25

    def test_count_messages_empty_list(self):
        """Test counting tokens in empty message list."""
        counter = CharacterEstimateCounter()
        result = counter.count_messages([])
        # Just the priming tokens
        assert result == 3

    def test_count_messages_dict_format(self):
        """Test counting tokens in dict-format messages."""
        counter = CharacterEstimateCounter(chars_per_token=4.0)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = counter.count_messages(messages)
        # 2 messages * 3 overhead + content tokens + 3 priming
        assert result > 6  # At least overhead + priming

    def test_count_messages_object_format(self):
        """Test counting tokens in Message object format."""
        counter = CharacterEstimateCounter(chars_per_token=4.0)
        messages = [
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi there"),
        ]
        result = counter.count_messages(messages)
        assert result > 6


# ============================================================================
# get_token_counter Tests
# ============================================================================


class TestGetTokenCounter:
    """Tests for get_token_counter factory function."""

    def test_get_counter_for_unknown_model(self):
        """Test getting counter for unknown model returns character estimate."""
        counter = get_token_counter("unknown-model-xyz")
        assert isinstance(counter, CharacterEstimateCounter)

    def test_get_counter_for_claude_model(self):
        """Test getting counter for Claude model (character estimate)."""
        counter = get_token_counter("claude-3-opus")
        assert isinstance(counter, CharacterEstimateCounter)

    def test_get_counter_for_default(self):
        """Test getting counter with 'default' returns character estimate."""
        counter = get_token_counter("default")
        assert isinstance(counter, CharacterEstimateCounter)

    def test_model_tokenizer_map_has_common_models(self):
        """Test that MODEL_TOKENIZER_MAP contains common models."""
        assert "gpt-4" in MODEL_TOKENIZER_MAP
        assert "gpt-4o" in MODEL_TOKENIZER_MAP
        assert "claude-3-opus" in MODEL_TOKENIZER_MAP

    def test_get_counter_prefix_matching(self):
        """Test that prefix matching works for model variants."""
        # gpt-4-turbo-preview should match gpt-4-turbo
        counter = get_token_counter("gpt-4-turbo-preview")
        # Should either get tiktoken or fallback to character estimate
        assert counter is not None


# ============================================================================
# TokenWindow Tests
# ============================================================================


class TestTokenWindow:
    """Tests for TokenWindow class."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        window = TokenWindow(max_tokens=8000)
        assert window.max_tokens == 8000
        assert window.current_tokens == 0
        assert window.reserved_tokens == 0

    def test_init_with_reserved_tokens(self):
        """Test initialization with reserved tokens."""
        window = TokenWindow(max_tokens=8000, reserved_tokens=1000)
        assert window.reserved_tokens == 1000
        assert window.available_tokens == 7000

    def test_available_tokens_calculation(self):
        """Test available tokens calculation."""
        window = TokenWindow(max_tokens=8000, reserved_tokens=1000)
        window.current_tokens = 2000
        assert window.available_tokens == 5000

    def test_available_tokens_never_negative(self):
        """Test that available tokens is never negative."""
        window = TokenWindow(max_tokens=1000, reserved_tokens=500)
        window.current_tokens = 1000
        assert window.available_tokens == 0

    def test_usage_ratio(self):
        """Test usage ratio calculation."""
        window = TokenWindow(max_tokens=8000)
        window.current_tokens = 4000
        assert window.usage_ratio == 0.5

    def test_usage_ratio_empty(self):
        """Test usage ratio when empty."""
        window = TokenWindow(max_tokens=8000)
        assert window.usage_ratio == 0.0

    def test_usage_ratio_zero_max(self):
        """Test usage ratio with zero max tokens."""
        window = TokenWindow(max_tokens=0)
        assert window.usage_ratio == 0.0

    def test_is_full_when_at_capacity(self):
        """Test is_full property when at capacity."""
        window = TokenWindow(max_tokens=1000, reserved_tokens=200)
        window.current_tokens = 800
        assert window.is_full is True

    def test_is_full_when_has_space(self):
        """Test is_full property when has space."""
        window = TokenWindow(max_tokens=1000, reserved_tokens=200)
        window.current_tokens = 500
        assert window.is_full is False

    def test_count_tokens(self):
        """Test counting tokens in text."""
        window = TokenWindow(max_tokens=8000, model="default")
        count = window.count_tokens("Hello world, this is a test.")
        assert count > 0

    def test_count_messages(self, sample_dict_messages):
        """Test counting tokens in messages."""
        window = TokenWindow(max_tokens=8000, model="default")
        count = window.count_messages(sample_dict_messages)
        assert count > 0

    def test_can_fit_true(self):
        """Test can_fit returns True when text fits."""
        window = TokenWindow(max_tokens=8000, model="default")
        assert window.can_fit("Short text") is True

    def test_can_fit_false(self):
        """Test can_fit returns False when text doesn't fit."""
        window = TokenWindow(max_tokens=10, model="default")
        # Long text won't fit in 10 tokens
        long_text = "a" * 1000
        assert window.can_fit(long_text) is False

    def test_can_fit_messages(self, sample_dict_messages):
        """Test can_fit_messages returns True when messages fit."""
        window = TokenWindow(max_tokens=8000, model="default")
        assert window.can_fit_messages(sample_dict_messages) is True

    def test_add_tokens(self):
        """Test adding tokens to current count."""
        window = TokenWindow(max_tokens=8000)
        window.add_tokens(100)
        assert window.current_tokens == 100
        window.add_tokens(50)
        assert window.current_tokens == 150

    def test_remove_tokens(self):
        """Test removing tokens from current count."""
        window = TokenWindow(max_tokens=8000)
        window.current_tokens = 200
        window.remove_tokens(50)
        assert window.current_tokens == 150

    def test_remove_tokens_never_negative(self):
        """Test that current_tokens never goes negative."""
        window = TokenWindow(max_tokens=8000)
        window.current_tokens = 50
        window.remove_tokens(100)
        assert window.current_tokens == 0

    def test_reset(self):
        """Test resetting token count."""
        window = TokenWindow(max_tokens=8000)
        window.current_tokens = 500
        window.reset()
        assert window.current_tokens == 0


# ============================================================================
# MessageWindow Tests
# ============================================================================


class TestMessageWindow:
    """Tests for MessageWindow class."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        window = MessageWindow(max_messages=100)
        assert window.max_messages == 100
        assert window.count == 0
        assert window.preserve_system is True

    def test_add_message(self):
        """Test adding a message."""
        window = MessageWindow(max_messages=10)
        msg = {"role": "user", "content": "Hello"}
        removed = window.add(msg)
        assert removed is None
        assert window.count == 1

    def test_add_message_when_full_removes_oldest(self, sample_dict_messages):
        """Test that adding to full window removes oldest non-system."""
        window = MessageWindow(max_messages=3, preserve_system=True)
        # Add system, user, assistant
        for msg in sample_dict_messages[:3]:
            window.add(msg)

        assert window.count == 3

        # Add another - should remove oldest non-system (user)
        new_msg = {"role": "user", "content": "New message"}
        removed = window.add(new_msg)

        assert removed is not None
        assert removed["role"] == "user"
        assert removed["content"] == "Hello, how are you?"
        assert window.count == 3

    def test_preserve_system_messages(self):
        """Test that system messages are preserved when removing."""
        window = MessageWindow(max_messages=2, preserve_system=True)
        window.add({"role": "system", "content": "You are helpful"})
        window.add({"role": "user", "content": "Hello"})

        # Window is full, add another
        removed = window.add({"role": "assistant", "content": "Hi"})

        # User message should be removed, not system
        assert removed["role"] == "user"
        # System should still be there
        roles = [window._get_role(m) for m in window.messages]
        assert "system" in roles

    def test_dont_preserve_system_when_disabled(self):
        """Test that system messages can be removed when preserve_system=False."""
        window = MessageWindow(max_messages=2, preserve_system=False)
        window.add({"role": "system", "content": "You are helpful"})
        window.add({"role": "user", "content": "Hello"})

        # Add another
        removed = window.add({"role": "assistant", "content": "Hi"})

        # System message should be removed (it was first)
        assert removed["role"] == "system"

    def test_count_property(self, sample_dict_messages):
        """Test count property."""
        window = MessageWindow(max_messages=100)
        for i, msg in enumerate(sample_dict_messages):
            window.add(msg)
            assert window.count == i + 1

    def test_is_full_property(self):
        """Test is_full property."""
        window = MessageWindow(max_messages=2)
        assert window.is_full is False
        window.add({"role": "user", "content": "1"})
        assert window.is_full is False
        window.add({"role": "user", "content": "2"})
        assert window.is_full is True

    def test_clear_keep_system(self):
        """Test clearing while keeping system messages."""
        window = MessageWindow(max_messages=100)
        window.add({"role": "system", "content": "System"})
        window.add({"role": "user", "content": "User"})
        window.add({"role": "assistant", "content": "Assistant"})

        window.clear(keep_system=True)

        assert window.count == 1
        assert window.messages[0]["role"] == "system"

    def test_clear_remove_all(self):
        """Test clearing all messages including system."""
        window = MessageWindow(max_messages=100)
        window.add({"role": "system", "content": "System"})
        window.add({"role": "user", "content": "User"})

        window.clear(keep_system=False)

        assert window.count == 0

    def test_get_messages(self, sample_dict_messages):
        """Test getting all messages."""
        window = MessageWindow(max_messages=100)
        for msg in sample_dict_messages:
            window.add(msg)

        result = window.get_messages()
        assert len(result) == len(sample_dict_messages)
        # Should return a copy
        assert result is not window.messages

    def test_get_role_from_dict(self):
        """Test extracting role from dict message."""
        window = MessageWindow(max_messages=10)
        msg = {"role": "user", "content": "Test"}
        assert window._get_role(msg) == "user"

    def test_get_role_from_object(self):
        """Test extracting role from Message object."""
        window = MessageWindow(max_messages=10)
        msg = Message(role=MessageRole.ASSISTANT, content="Test")
        assert window._get_role(msg) == "assistant"


# ============================================================================
# ContextBudget Tests
# ============================================================================


class TestContextBudget:
    """Tests for ContextBudget class."""

    def test_default_values(self):
        """Test default budget values."""
        budget = ContextBudget()
        assert budget.total_tokens == 8000
        assert budget.system_ratio == 0.15
        assert budget.conversation_ratio == 0.60
        assert budget.tools_ratio == 0.10
        assert budget.response_ratio == 0.15

    def test_custom_values(self):
        """Test custom budget values."""
        budget = ContextBudget(
            total_tokens=16000,
            system_ratio=0.20,
            conversation_ratio=0.50,
            tools_ratio=0.15,
            response_ratio=0.15,
        )
        assert budget.total_tokens == 16000
        assert budget.system_ratio == 0.20

    def test_system_tokens_property(self):
        """Test system_tokens calculation."""
        budget = ContextBudget(total_tokens=10000, system_ratio=0.20)
        assert budget.system_tokens == 2000

    def test_conversation_tokens_property(self):
        """Test conversation_tokens calculation."""
        budget = ContextBudget(total_tokens=10000, conversation_ratio=0.60)
        assert budget.conversation_tokens == 6000

    def test_tools_tokens_property(self):
        """Test tools_tokens calculation."""
        budget = ContextBudget(total_tokens=10000, tools_ratio=0.10)
        assert budget.tools_tokens == 1000

    def test_response_tokens_property(self):
        """Test response_tokens calculation."""
        budget = ContextBudget(total_tokens=10000, response_ratio=0.10)
        assert budget.response_tokens == 1000

    def test_validate_ratios_valid(self):
        """Test validate_ratios with valid ratios summing to 1.0."""
        budget = ContextBudget(
            system_ratio=0.15,
            conversation_ratio=0.60,
            tools_ratio=0.10,
            response_ratio=0.15,
        )
        assert budget.validate_ratios() is True

    def test_validate_ratios_invalid_low(self):
        """Test validate_ratios with ratios summing to less than 1.0."""
        budget = ContextBudget(
            system_ratio=0.10,
            conversation_ratio=0.40,
            tools_ratio=0.10,
            response_ratio=0.10,
        )
        assert budget.validate_ratios() is False

    def test_validate_ratios_invalid_high(self):
        """Test validate_ratios with ratios summing to more than 1.0."""
        budget = ContextBudget(
            system_ratio=0.30,
            conversation_ratio=0.60,
            tools_ratio=0.20,
            response_ratio=0.20,
        )
        assert budget.validate_ratios() is False


# ============================================================================
# ContextUsage Tests
# ============================================================================


class TestContextUsage:
    """Tests for ContextUsage dataclass."""

    def test_default_values(self):
        """Test default usage values."""
        usage = ContextUsage()
        assert usage.system_tokens == 0
        assert usage.conversation_tokens == 0
        assert usage.tools_tokens == 0

    def test_total_tokens_property(self):
        """Test total_tokens calculation."""
        usage = ContextUsage(
            system_tokens=100,
            conversation_tokens=500,
            tools_tokens=50,
        )
        assert usage.total_tokens == 650

    def test_timestamp_auto_set(self):
        """Test that timestamp is automatically set."""
        usage = ContextUsage()
        assert usage.timestamp is not None


# ============================================================================
# ContextManager Tests
# ============================================================================


class TestContextManager:
    """Tests for ContextManager class."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        manager = ContextManager()
        assert manager.budget is not None
        assert manager.model == "default"
        assert manager.token_window is not None
        assert manager.message_window is not None

    def test_init_with_custom_budget(self, small_budget):
        """Test initialization with custom budget."""
        manager = ContextManager(budget=small_budget)
        assert manager.budget.total_tokens == 1000

    def test_init_with_model(self):
        """Test initialization with specific model."""
        manager = ContextManager(model="gpt-4")
        assert manager.model == "gpt-4"

    def test_init_with_max_messages(self):
        """Test initialization with max messages."""
        manager = ContextManager(max_messages=50)
        assert manager.message_window.max_messages == 50

    def test_usage_property(self):
        """Test usage property."""
        manager = ContextManager()
        assert manager.usage is not None
        assert isinstance(manager.usage, ContextUsage)

    def test_available_tokens_property(self):
        """Test available_tokens property."""
        manager = ContextManager()
        assert manager.available_tokens > 0

    def test_needs_compaction_false_initially(self):
        """Test needs_compaction is False when empty."""
        manager = ContextManager()
        assert manager.needs_compaction is False

    @pytest.mark.asyncio
    async def test_add_message_success(self):
        """Test successfully adding a message."""
        manager = ContextManager()
        msg = {"role": "user", "content": "Hello"}
        success, removed = await manager.add_message(msg)
        assert success is True
        assert removed is None
        assert manager.message_window.count == 1

    @pytest.mark.asyncio
    async def test_add_message_updates_usage(self):
        """Test that adding message updates usage stats."""
        manager = ContextManager()
        initial_tokens = manager.usage.conversation_tokens
        msg = {"role": "user", "content": "Hello world"}
        await manager.add_message(msg)
        assert manager.usage.conversation_tokens > initial_tokens

    @pytest.mark.asyncio
    async def test_add_message_returns_removed_when_full(self):
        """Test add_message returns removed message when window is full."""
        manager = ContextManager(max_messages=2)
        await manager.add_message({"role": "user", "content": "Message 1"})
        await manager.add_message({"role": "assistant", "content": "Message 2"})

        success, removed = await manager.add_message({"role": "user", "content": "Message 3"})
        assert success is True
        assert removed is not None

    @pytest.mark.asyncio
    async def test_add_message_fails_when_too_large(self):
        """Test add_message fails when message is too large."""
        budget = ContextBudget(total_tokens=50)  # Very small budget
        manager = ContextManager(budget=budget)

        # Very long message
        long_content = "x" * 10000
        msg = {"role": "user", "content": long_content}
        success, removed = await manager.add_message(msg)
        assert success is False

    @pytest.mark.asyncio
    async def test_add_messages_multiple(self, sample_dict_messages):
        """Test adding multiple messages."""
        manager = ContextManager()
        added, removed = await manager.add_messages(sample_dict_messages)
        assert added == len(sample_dict_messages)
        assert manager.message_window.count == len(sample_dict_messages)

    @pytest.mark.asyncio
    async def test_add_messages_stops_when_full(self):
        """Test add_messages stops when can't fit more."""
        budget = ContextBudget(total_tokens=200)
        manager = ContextManager(budget=budget)

        messages = [
            {"role": "user", "content": "x" * 100}
            for _ in range(10)
        ]
        added, _ = await manager.add_messages(messages)
        # Should not add all 10 due to token limit
        assert added < 10

    @pytest.mark.asyncio
    async def test_set_system_prompt_success(self):
        """Test setting system prompt within budget."""
        manager = ContextManager()
        result = await manager.set_system_prompt("You are a helpful assistant.")
        assert result is True
        assert manager.usage.system_tokens > 0

    @pytest.mark.asyncio
    async def test_set_system_prompt_too_large(self):
        """Test setting system prompt that exceeds budget."""
        budget = ContextBudget(total_tokens=100, system_ratio=0.10)  # 10 tokens for system
        manager = ContextManager(budget=budget)
        long_prompt = "x" * 1000
        result = await manager.set_system_prompt(long_prompt)
        assert result is False

    @pytest.mark.asyncio
    async def test_set_tools_success(self):
        """Test setting tools within budget."""
        manager = ContextManager()
        tools = [{"name": "search", "description": "Search tool"}]
        result = await manager.set_tools(tools)
        assert result is True
        assert manager.usage.tools_tokens > 0

    @pytest.mark.asyncio
    async def test_set_tools_too_large(self):
        """Test setting tools that exceed budget."""
        budget = ContextBudget(total_tokens=100, tools_ratio=0.05)  # 5 tokens for tools
        manager = ContextManager(budget=budget)
        large_tools = [{"name": "x", "description": "y" * 1000}]
        result = await manager.set_tools(large_tools)
        assert result is False

    def test_get_messages(self, sample_dict_messages):
        """Test getting messages from manager."""
        manager = ContextManager()
        for msg in sample_dict_messages:
            manager.message_window.add(msg)

        result = manager.get_messages()
        assert len(result) == len(sample_dict_messages)

    def test_count_tokens(self):
        """Test counting tokens in text."""
        manager = ContextManager()
        count = manager.count_tokens("Hello world")
        assert count > 0

    def test_count_message_tokens(self, sample_dict_messages):
        """Test counting tokens in messages."""
        manager = ContextManager()
        count = manager.count_message_tokens(sample_dict_messages)
        assert count > 0

    @pytest.mark.asyncio
    async def test_clear_keeps_system(self):
        """Test clearing context while keeping system messages."""
        manager = ContextManager()
        await manager.add_message({"role": "system", "content": "System prompt"})
        await manager.add_message({"role": "user", "content": "User message"})

        await manager.clear(keep_system=True)

        assert manager.message_window.count == 1
        assert manager.usage.conversation_tokens == 0

    @pytest.mark.asyncio
    async def test_clear_removes_all(self):
        """Test clearing all context including system."""
        manager = ContextManager()
        await manager.set_system_prompt("System prompt")
        await manager.add_message({"role": "user", "content": "User message"})

        await manager.clear(keep_system=False)

        assert manager.message_window.count == 0
        assert manager.usage.system_tokens == 0

    def test_get_context_hash(self, sample_dict_messages):
        """Test getting context hash."""
        manager = ContextManager()
        for msg in sample_dict_messages:
            manager.message_window.add(msg)

        hash1 = manager.get_context_hash()
        assert len(hash1) == 16  # SHA256 truncated to 16 chars

        # Same content should produce same hash
        hash2 = manager.get_context_hash()
        assert hash1 == hash2

    def test_get_context_hash_changes_with_content(self):
        """Test that context hash changes when content changes."""
        manager = ContextManager()
        manager.message_window.add({"role": "user", "content": "Hello"})
        hash1 = manager.get_context_hash()

        manager.message_window.add({"role": "assistant", "content": "Hi"})
        hash2 = manager.get_context_hash()

        assert hash1 != hash2

    def test_get_stats(self, sample_dict_messages):
        """Test getting context statistics."""
        manager = ContextManager()
        for msg in sample_dict_messages:
            manager.message_window.add(msg)

        stats = manager.get_stats()

        assert "total_tokens" in stats
        assert "used_tokens" in stats
        assert "available_tokens" in stats
        assert "usage_ratio" in stats
        assert "message_count" in stats
        assert "needs_compaction" in stats
        assert "budget" in stats
        assert "system" in stats["budget"]
        assert "conversation" in stats["budget"]
        assert "tools" in stats["budget"]
        assert "response" in stats["budget"]

    def test_needs_compaction_true_when_high_usage(self):
        """Test needs_compaction is True when usage is high."""
        budget = ContextBudget(total_tokens=1000)
        manager = ContextManager(budget=budget)
        # Manually set high usage
        manager.token_window.current_tokens = 900
        assert manager.needs_compaction is True


# ============================================================================
# Integration Tests
# ============================================================================


class TestContextManagerIntegration:
    """Integration tests for ContextManager with various components."""

    @pytest.mark.asyncio
    async def test_full_conversation_flow(self, sample_dict_messages):
        """Test a full conversation flow with the manager."""
        manager = ContextManager()

        # Set system prompt
        await manager.set_system_prompt("You are a helpful assistant.")

        # Add messages
        for msg in sample_dict_messages:
            if msg["role"] != "system":
                await manager.add_message(msg)

        # Verify state
        assert manager.message_window.count == len(sample_dict_messages) - 1  # Minus system
        assert manager.usage.system_tokens > 0
        assert manager.usage.conversation_tokens > 0

        # Get stats
        stats = manager.get_stats()
        assert stats["message_count"] == len(sample_dict_messages) - 1

    @pytest.mark.asyncio
    async def test_conversation_with_windowing(self):
        """Test conversation that triggers windowing."""
        manager = ContextManager(max_messages=5)

        # Add more messages than max
        for i in range(10):
            msg = {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            await manager.add_message(msg)

        # Should only have 5 messages
        assert manager.message_window.count == 5

        # Most recent messages should be present
        messages = manager.get_messages()
        assert "Message 9" in messages[-1]["content"]

    @pytest.mark.asyncio
    async def test_budget_enforcement(self):
        """Test that budget is enforced properly."""
        budget = ContextBudget(
            total_tokens=500,
            system_ratio=0.20,
            conversation_ratio=0.60,
            tools_ratio=0.10,
            response_ratio=0.10,
        )
        manager = ContextManager(budget=budget)

        # System prompt should fit
        assert await manager.set_system_prompt("Short system prompt")

        # Long system prompt should not fit
        manager2 = ContextManager(budget=budget)
        long_prompt = "x" * 5000
        assert await manager2.set_system_prompt(long_prompt) is False
