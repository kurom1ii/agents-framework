"""Tests for conversation summarization.

Tests cover:
- SimpleSummarizer - Extractive summarization without LLM
- ConversationSummarizer - LLM-powered summarization
- SummaryCache - Caching for summaries
- ProgressiveSummary - Progressive summarization state
- SummaryPrompt - Prompt configuration
- Factory function create_summarizer
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents_framework.context import (
    ConversationSummarizer,
    DEFAULT_PROMPTS,
    ProgressiveSummary,
    SimpleSummarizer,
    SummaryCache,
    SummaryPrompt,
    SummaryType,
    create_summarizer,
)
from agents_framework.llm.base import LLMResponse, Message, MessageRole


# ============================================================================
# SummaryType Tests
# ============================================================================


class TestSummaryType:
    """Tests for SummaryType enum."""

    def test_summary_types_exist(self):
        """Test that all expected summary types exist."""
        assert SummaryType.BRIEF == "brief"
        assert SummaryType.DETAILED == "detailed"
        assert SummaryType.CONTEXTUAL == "contextual"
        assert SummaryType.ACTION_FOCUSED == "action_focused"
        assert SummaryType.TOPIC_BASED == "topic_based"


# ============================================================================
# SummaryPrompt Tests
# ============================================================================


class TestSummaryPrompt:
    """Tests for SummaryPrompt configuration."""

    def test_default_values(self):
        """Test default prompt values."""
        prompt = SummaryPrompt()
        assert "conversation summarizer" in prompt.system_prompt.lower()
        assert "{conversation}" in prompt.user_prompt_template
        assert "{summary_type}" in prompt.user_prompt_template
        assert "{max_tokens}" in prompt.user_prompt_template
        assert prompt.summary_type == SummaryType.CONTEXTUAL

    def test_custom_values(self):
        """Test custom prompt values."""
        prompt = SummaryPrompt(
            system_prompt="Custom system prompt",
            user_prompt_template="Summarize: {conversation}",
            summary_type=SummaryType.BRIEF,
        )
        assert prompt.system_prompt == "Custom system prompt"
        assert prompt.summary_type == SummaryType.BRIEF

    def test_default_prompts_exist(self):
        """Test that default prompts dictionary has expected presets."""
        assert "general" in DEFAULT_PROMPTS
        assert "technical" in DEFAULT_PROMPTS
        assert "task_oriented" in DEFAULT_PROMPTS
        assert "brief" in DEFAULT_PROMPTS

    def test_technical_prompt_has_technical_focus(self):
        """Test that technical prompt focuses on technical content."""
        prompt = DEFAULT_PROMPTS["technical"]
        assert "technical" in prompt.system_prompt.lower()

    def test_brief_prompt_has_brief_type(self):
        """Test that brief prompt uses BRIEF summary type."""
        prompt = DEFAULT_PROMPTS["brief"]
        assert prompt.summary_type == SummaryType.BRIEF


# ============================================================================
# SummaryCache Tests
# ============================================================================


class TestSummaryCache:
    """Tests for SummaryCache."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        cache = SummaryCache()
        assert cache.max_size == 100
        assert cache.ttl_seconds == 3600

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        cache = SummaryCache(max_size=50, ttl_seconds=1800)
        assert cache.max_size == 50
        assert cache.ttl_seconds == 1800

    def test_set_and_get(self, sample_dict_messages):
        """Test setting and getting cached summary."""
        cache = SummaryCache()
        summary = "This is a test summary"

        cache.set(sample_dict_messages, summary)
        result = cache.get(sample_dict_messages)

        assert result == summary

    def test_get_returns_none_for_missing(self, sample_dict_messages):
        """Test get returns None for missing entry."""
        cache = SummaryCache()
        result = cache.get(sample_dict_messages)
        assert result is None

    def test_get_returns_none_for_expired(self, sample_dict_messages):
        """Test get returns None for expired entry."""
        cache = SummaryCache(ttl_seconds=0)  # Immediate expiry

        cache.set(sample_dict_messages, "Summary")

        # Entry should be expired immediately
        result = cache.get(sample_dict_messages)
        assert result is None

    def test_cache_eviction_when_full(self):
        """Test that oldest entry is evicted when cache is full."""
        cache = SummaryCache(max_size=2)

        messages1 = [{"role": "user", "content": "Message 1"}]
        messages2 = [{"role": "user", "content": "Message 2"}]
        messages3 = [{"role": "user", "content": "Message 3"}]

        cache.set(messages1, "Summary 1")
        cache.set(messages2, "Summary 2")
        cache.set(messages3, "Summary 3")  # Should evict messages1

        assert cache.get(messages1) is None
        assert cache.get(messages2) is not None
        assert cache.get(messages3) is not None

    def test_clear(self, sample_dict_messages):
        """Test clearing the cache."""
        cache = SummaryCache()
        cache.set(sample_dict_messages, "Summary")

        cache.clear()

        assert cache.get(sample_dict_messages) is None

    def test_invalidate_existing(self, sample_dict_messages):
        """Test invalidating an existing entry."""
        cache = SummaryCache()
        cache.set(sample_dict_messages, "Summary")

        result = cache.invalidate(sample_dict_messages)

        assert result is True
        assert cache.get(sample_dict_messages) is None

    def test_invalidate_missing(self, sample_dict_messages):
        """Test invalidating a missing entry."""
        cache = SummaryCache()

        result = cache.invalidate(sample_dict_messages)

        assert result is False

    def test_cache_key_deterministic(self, sample_dict_messages):
        """Test that cache key is deterministic for same messages."""
        cache = SummaryCache()

        key1 = cache._generate_key(sample_dict_messages)
        key2 = cache._generate_key(sample_dict_messages)

        assert key1 == key2

    def test_cache_key_different_for_different_messages(self):
        """Test that cache key differs for different messages."""
        cache = SummaryCache()

        messages1 = [{"role": "user", "content": "Hello"}]
        messages2 = [{"role": "user", "content": "World"}]

        key1 = cache._generate_key(messages1)
        key2 = cache._generate_key(messages2)

        assert key1 != key2


# ============================================================================
# ProgressiveSummary Tests
# ============================================================================


class TestProgressiveSummary:
    """Tests for ProgressiveSummary state tracking."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        state = ProgressiveSummary()
        assert state.current_summary == ""
        assert state.messages_summarized == 0
        assert state.summary_chain == []

    def test_add_summary_first(self):
        """Test adding first summary."""
        state = ProgressiveSummary()

        state.add_summary("First summary", 10)

        assert state.current_summary == "First summary"
        assert state.messages_summarized == 10
        assert len(state.summary_chain) == 0  # First summary not in chain

    def test_add_summary_subsequent(self):
        """Test adding subsequent summaries."""
        state = ProgressiveSummary()

        state.add_summary("First summary", 10)
        state.add_summary("Second summary", 5)

        assert state.current_summary == "Second summary"
        assert state.messages_summarized == 15
        assert len(state.summary_chain) == 1
        assert state.summary_chain[0] == "First summary"

    def test_get_full_context_single(self):
        """Test getting full context with single summary."""
        state = ProgressiveSummary()
        state.add_summary("Only summary", 5)

        context = state.get_full_context()

        assert context == "Only summary"

    def test_get_full_context_multiple(self):
        """Test getting full context with multiple summaries."""
        state = ProgressiveSummary()
        state.add_summary("First", 5)
        state.add_summary("Second", 5)
        state.add_summary("Third", 5)

        context = state.get_full_context()

        assert "First" in context
        assert "Second" in context
        assert "Third" in context
        assert "---" in context  # Separator

    def test_last_update_timestamp(self):
        """Test that last_update timestamp is updated."""
        state = ProgressiveSummary()
        initial_time = state.last_update

        state.add_summary("New summary", 5)

        assert state.last_update >= initial_time


# ============================================================================
# SimpleSummarizer Tests
# ============================================================================


class TestSimpleSummarizer:
    """Tests for SimpleSummarizer extractive summarization."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        summarizer = SimpleSummarizer()
        assert summarizer._max_chars == 100
        assert summarizer._include_roles is True

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        summarizer = SimpleSummarizer(
            max_chars_per_message=200,
            include_roles=False,
        )
        assert summarizer._max_chars == 200
        assert summarizer._include_roles is False

    @pytest.mark.asyncio
    async def test_summarize_empty_messages(self):
        """Test summarizing empty message list."""
        summarizer = SimpleSummarizer()
        result = await summarizer.summarize([])
        assert result == "[No content to summarize]"

    @pytest.mark.asyncio
    async def test_summarize_single_message(self):
        """Test summarizing single message."""
        summarizer = SimpleSummarizer()
        messages = [{"role": "user", "content": "Hello world"}]

        result = await summarizer.summarize(messages)

        assert "USER" in result
        assert "Hello world" in result

    @pytest.mark.asyncio
    async def test_summarize_multiple_messages(self, sample_dict_messages):
        """Test summarizing multiple messages."""
        summarizer = SimpleSummarizer()

        result = await summarizer.summarize(sample_dict_messages)

        # Should contain role prefixes
        assert any(role in result for role in ["USER", "ASSISTANT", "SYSTEM"])

    @pytest.mark.asyncio
    async def test_summarize_truncates_long_content(self):
        """Test that long message content is truncated."""
        summarizer = SimpleSummarizer(max_chars_per_message=50)
        messages = [{"role": "user", "content": "x" * 200}]

        result = await summarizer.summarize(messages)

        # Should be truncated with ellipsis
        assert "..." in result
        assert len(result) < 200 + 20  # Content + role prefix

    @pytest.mark.asyncio
    async def test_summarize_respects_max_tokens(self):
        """Test that summary respects max_tokens limit."""
        summarizer = SimpleSummarizer()
        messages = [
            {"role": "user", "content": f"Message {i}: " + "x" * 50}
            for i in range(20)
        ]

        result = await summarizer.summarize(messages, max_tokens=50)

        # Result should be limited (50 tokens * 4 chars = 200 chars approximately)
        assert len(result) <= 250  # Some buffer for overhead

    @pytest.mark.asyncio
    async def test_summarize_without_roles(self):
        """Test summarizing without role prefixes."""
        summarizer = SimpleSummarizer(include_roles=False)
        messages = [{"role": "user", "content": "Hello world"}]

        result = await summarizer.summarize(messages)

        assert "[USER]" not in result
        assert "Hello world" in result

    @pytest.mark.asyncio
    async def test_summarize_skips_empty_messages(self):
        """Test that empty messages are skipped."""
        summarizer = SimpleSummarizer()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},  # Empty
            {"role": "user", "content": "World"},
        ]

        result = await summarizer.summarize(messages)

        assert "Hello" in result
        assert "World" in result

    @pytest.mark.asyncio
    async def test_summarize_message_objects(self, sample_message_objects):
        """Test summarizing Message objects."""
        summarizer = SimpleSummarizer()

        result = await summarizer.summarize(sample_message_objects)

        assert len(result) > 0


# ============================================================================
# ConversationSummarizer Tests
# ============================================================================


class TestConversationSummarizer:
    """Tests for ConversationSummarizer LLM-powered summarization."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        summarizer = ConversationSummarizer()
        assert summarizer._llm is None
        assert summarizer._prompt is not None
        assert summarizer._enable_cache is True

    def test_init_with_llm(self, mock_llm_provider):
        """Test initialization with LLM provider."""
        summarizer = ConversationSummarizer(llm_provider=mock_llm_provider)
        assert summarizer._llm is mock_llm_provider

    def test_init_with_custom_prompt(self, technical_summary_prompt):
        """Test initialization with custom prompt."""
        summarizer = ConversationSummarizer(prompt=technical_summary_prompt)
        assert summarizer._prompt.summary_type == SummaryType.DETAILED

    def test_init_cache_disabled(self):
        """Test initialization with cache disabled."""
        summarizer = ConversationSummarizer(enable_cache=False)
        assert summarizer._cache is None

    def test_prompt_property(self):
        """Test prompt property getter and setter."""
        summarizer = ConversationSummarizer()
        original_prompt = summarizer.prompt

        new_prompt = SummaryPrompt(system_prompt="New prompt")
        summarizer.prompt = new_prompt

        assert summarizer.prompt.system_prompt == "New prompt"

    def test_set_prompt_preset(self):
        """Test setting prompt from preset."""
        summarizer = ConversationSummarizer()

        summarizer.set_prompt_preset("technical")

        assert "technical" in summarizer.prompt.system_prompt.lower()

    def test_set_prompt_preset_invalid(self):
        """Test setting invalid prompt preset raises error."""
        summarizer = ConversationSummarizer()

        with pytest.raises(ValueError, match="Unknown preset"):
            summarizer.set_prompt_preset("nonexistent")

    @pytest.mark.asyncio
    async def test_summarize_empty_messages(self):
        """Test summarizing empty message list."""
        summarizer = ConversationSummarizer()

        result = await summarizer.summarize([])

        assert result == "[Empty conversation]"

    @pytest.mark.asyncio
    async def test_summarize_with_llm(self, mock_llm_provider, sample_dict_messages):
        """Test summarizing with LLM provider."""
        summarizer = ConversationSummarizer(
            llm_provider=mock_llm_provider,
            enable_cache=False,
        )

        result = await summarizer.summarize(sample_dict_messages)

        assert result is not None
        assert len(result) > 0
        mock_llm_provider.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_uses_cache(self, mock_llm_provider, sample_dict_messages):
        """Test that summarize uses cache."""
        # Explicitly pass a cache instance since enable_cache=True without cache is still None
        cache = SummaryCache()
        summarizer = ConversationSummarizer(
            llm_provider=mock_llm_provider,
            cache=cache,
            enable_cache=True,
        )

        # First call
        result1 = await summarizer.summarize(sample_dict_messages)
        # Second call with same messages
        result2 = await summarizer.summarize(sample_dict_messages)

        # LLM should only be called once
        assert mock_llm_provider.generate.call_count == 1
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_summarize_fallback_on_error(self, sample_dict_messages):
        """Test that summarize falls back to simple summarizer on error."""
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=Exception("LLM error"))

        summarizer = ConversationSummarizer(
            llm_provider=mock_llm,
            enable_cache=False,
        )

        result = await summarizer.summarize(sample_dict_messages)

        # Should get a result from fallback summarizer
        assert result is not None
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_summarize_fallback_no_llm(self, sample_dict_messages):
        """Test that summarize uses fallback when no LLM."""
        summarizer = ConversationSummarizer(
            llm_provider=None,
            enable_cache=False,
        )

        result = await summarizer.summarize(sample_dict_messages)

        # Should get a result from fallback summarizer
        assert result is not None

    def test_format_conversation(self, sample_dict_messages):
        """Test formatting conversation for LLM."""
        summarizer = ConversationSummarizer()

        result = summarizer._format_conversation(sample_dict_messages)

        assert "USER:" in result or "ASSISTANT:" in result or "SYSTEM:" in result

    @pytest.mark.asyncio
    async def test_summarize_progressively(self, mock_llm_provider):
        """Test progressive summarization."""
        summarizer = ConversationSummarizer(
            llm_provider=mock_llm_provider,
            enable_cache=False,
        )

        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(25)
        ]

        result = await summarizer.summarize_progressively(
            messages,
            chunk_size=5,
            max_tokens_per_chunk=100,
        )

        assert result is not None
        # Should have called LLM multiple times for chunks + final
        assert mock_llm_provider.generate.call_count > 1

    @pytest.mark.asyncio
    async def test_summarize_progressively_small_list(self, mock_llm_provider):
        """Test progressive summarization with small message list."""
        summarizer = ConversationSummarizer(
            llm_provider=mock_llm_provider,
            enable_cache=False,
        )

        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(5)
        ]

        result = await summarizer.summarize_progressively(
            messages,
            chunk_size=10,  # Larger than message count
        )

        assert result is not None
        # Should do single summarization, not chunked
        assert mock_llm_provider.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_summarize_progressively_empty(self):
        """Test progressive summarization with empty list."""
        summarizer = ConversationSummarizer()

        result = await summarizer.summarize_progressively([])

        assert result == "[Empty conversation]"

    @pytest.mark.asyncio
    async def test_update_progressive_summary(self, mock_llm_provider):
        """Test updating progressive summary with new messages."""
        summarizer = ConversationSummarizer(
            llm_provider=mock_llm_provider,
            enable_cache=False,
        )

        # First batch
        messages1 = [{"role": "user", "content": "First message"}]
        result1 = await summarizer.update_progressive_summary(messages1)

        # Second batch
        messages2 = [{"role": "user", "content": "Second message"}]
        result2 = await summarizer.update_progressive_summary(messages2)

        assert result1 is not None
        assert result2 is not None
        # State should be updated
        assert summarizer._progressive_state is not None
        assert summarizer._progressive_state.messages_summarized > 0

    def test_get_progressive_state(self):
        """Test getting progressive state."""
        summarizer = ConversationSummarizer()
        assert summarizer.get_progressive_state() is None

    def test_reset_progressive_state(self, mock_llm_provider):
        """Test resetting progressive state."""
        summarizer = ConversationSummarizer()
        summarizer._progressive_state = ProgressiveSummary()
        summarizer._progressive_state.add_summary("Test", 5)

        summarizer.reset_progressive_state()

        assert summarizer._progressive_state is None

    def test_clear_cache(self, sample_dict_messages):
        """Test clearing the cache."""
        # Need to explicitly pass a cache instance
        cache = SummaryCache()
        summarizer = ConversationSummarizer(cache=cache, enable_cache=True)
        summarizer._cache.set(sample_dict_messages, "Cached summary")

        summarizer.clear_cache()

        assert summarizer._cache.get(sample_dict_messages) is None

    @pytest.mark.asyncio
    async def test_summarize_by_topics(self, mock_llm_provider, sample_dict_messages):
        """Test topic-based summarization."""
        summarizer = ConversationSummarizer(
            llm_provider=mock_llm_provider,
            enable_cache=False,
        )

        result = await summarizer.summarize_by_topics(sample_dict_messages)

        assert isinstance(result, dict)
        assert "main_discussion" in result

    @pytest.mark.asyncio
    async def test_extract_key_points_with_llm(self, mock_llm_with_key_points, sample_dict_messages):
        """Test key point extraction with LLM."""
        summarizer = ConversationSummarizer(
            llm_provider=mock_llm_with_key_points,
            enable_cache=False,
        )

        result = await summarizer.extract_key_points(sample_dict_messages, max_points=3)

        assert isinstance(result, list)
        assert len(result) <= 3

    @pytest.mark.asyncio
    async def test_extract_key_points_without_llm(self, sample_dict_messages):
        """Test key point extraction without LLM (fallback)."""
        summarizer = ConversationSummarizer(llm_provider=None)

        result = await summarizer.extract_key_points(sample_dict_messages, max_points=3)

        assert isinstance(result, list)
        # Should extract from messages directly
        assert len(result) <= 3


# ============================================================================
# create_summarizer Factory Tests
# ============================================================================


class TestCreateSummarizer:
    """Tests for create_summarizer factory function."""

    def test_create_default(self):
        """Test creating summarizer with defaults."""
        summarizer = create_summarizer()
        assert isinstance(summarizer, ConversationSummarizer)
        assert summarizer._enable_cache is True

    def test_create_with_llm(self, mock_llm_provider):
        """Test creating summarizer with LLM provider."""
        summarizer = create_summarizer(llm_provider=mock_llm_provider)
        assert summarizer._llm is mock_llm_provider

    def test_create_with_preset(self):
        """Test creating summarizer with preset."""
        summarizer = create_summarizer(preset="technical")
        assert "technical" in summarizer._prompt.system_prompt.lower()

    def test_create_with_unknown_preset_uses_default(self):
        """Test that unknown preset falls back to general."""
        summarizer = create_summarizer(preset="unknown")
        # Should use general preset
        assert summarizer._prompt is not None

    def test_create_without_cache(self):
        """Test creating summarizer with cache disabled."""
        summarizer = create_summarizer(enable_cache=False)
        assert summarizer._cache is None


# ============================================================================
# Integration Tests
# ============================================================================


class TestSummarizerIntegration:
    """Integration tests for summarizer components."""

    @pytest.mark.asyncio
    async def test_full_summarization_workflow(self, mock_llm_provider):
        """Test complete summarization workflow."""
        # Create summarizer with all features
        summarizer = create_summarizer(
            llm_provider=mock_llm_provider,
            preset="general",
            enable_cache=True,
        )

        # Build conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Can you help me with Python?"},
            {"role": "assistant", "content": "Of course! What do you need help with?"},
            {"role": "user", "content": "I need to understand decorators."},
            {"role": "assistant", "content": "Decorators are functions that modify other functions."},
        ]

        # Get summary
        summary = await summarizer.summarize(messages)
        assert summary is not None

        # Check caching works
        summary2 = await summarizer.summarize(messages)
        assert summary == summary2
        assert mock_llm_provider.generate.call_count == 1

        # Extract key points
        points = await summarizer.extract_key_points(messages)
        assert isinstance(points, list)

    @pytest.mark.asyncio
    async def test_progressive_summarization_workflow(self, mock_llm_provider):
        """Test progressive summarization workflow."""
        summarizer = create_summarizer(
            llm_provider=mock_llm_provider,
            enable_cache=False,
        )

        # Simulate long conversation
        all_messages = []
        for batch in range(3):
            batch_messages = [
                {"role": "user", "content": f"Batch {batch} question {i}"}
                for i in range(5)
            ]
            all_messages.extend(batch_messages)

            # Update progressive summary
            await summarizer.update_progressive_summary(batch_messages)

        # Get final state
        state = summarizer.get_progressive_state()
        assert state is not None
        assert state.messages_summarized == 15
        assert len(state.summary_chain) >= 1  # Previous summaries in chain

    @pytest.mark.asyncio
    async def test_summarizer_with_message_objects(self, mock_llm_provider, sample_message_objects):
        """Test summarizer works with Message objects."""
        summarizer = create_summarizer(
            llm_provider=mock_llm_provider,
            enable_cache=False,
        )

        summary = await summarizer.summarize(sample_message_objects)
        assert summary is not None

    @pytest.mark.asyncio
    async def test_cache_integration(self):
        """Test cache integration with summarizer."""
        cache = SummaryCache(max_size=10, ttl_seconds=60)

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(
                content="Cached summary result",
                model="test",
                finish_reason="stop",
            )
        )

        summarizer = ConversationSummarizer(
            llm_provider=mock_llm,
            cache=cache,
            enable_cache=True,
        )

        messages = [{"role": "user", "content": "Test message"}]

        # First call - should hit LLM
        result1 = await summarizer.summarize(messages)
        assert mock_llm.generate.call_count == 1

        # Second call - should hit cache
        result2 = await summarizer.summarize(messages)
        assert mock_llm.generate.call_count == 1  # No additional call

        # Different messages - should hit LLM
        different_messages = [{"role": "user", "content": "Different message"}]
        result3 = await summarizer.summarize(different_messages)
        assert mock_llm.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_fallback_chain(self, sample_dict_messages):
        """Test fallback from LLM to simple summarizer."""
        # Create LLM that fails
        failing_llm = MagicMock()
        failing_llm.generate = AsyncMock(side_effect=Exception("Network error"))

        # Create custom fallback
        fallback = SimpleSummarizer(max_chars_per_message=50)

        summarizer = ConversationSummarizer(
            llm_provider=failing_llm,
            fallback_summarizer=fallback,
            enable_cache=False,
        )

        result = await summarizer.summarize(sample_dict_messages)

        # Should have gotten result from fallback
        assert result is not None
        assert "..." in result or len(result) < 500  # Fallback truncates
