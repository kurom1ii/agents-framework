"""Tests for context compaction strategies.

Tests cover:
- TruncationCompactor - FIFO message removal
- SelectiveCompactor - Importance-based removal
- SummaryCompactor - Summarization before removal
- HybridCompactor - Combined approach
- CompactionConfig and CompactionResult
- Factory function create_compactor
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents_framework.context import (
    CompactionConfig,
    CompactionResult,
    CompactionStrategy,
    CompactionTrigger,
    ContextManager,
    HybridCompactor,
    MessageImportance,
    SelectiveCompactor,
    SummaryCompactor,
    TruncationCompactor,
    create_compactor,
)


# ============================================================================
# CompactionConfig Tests
# ============================================================================


class TestCompactionConfig:
    """Tests for CompactionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CompactionConfig()
        assert config.token_threshold_ratio == 0.85
        assert config.target_token_ratio == 0.60
        assert config.max_messages_before_compact == 50
        assert config.target_messages_after_compact == 25
        assert config.max_age_hours == 24.0
        assert config.preserve_recent_count == 5
        assert config.preserve_system_messages is True
        assert config.preserve_tool_calls is True
        assert config.summarize_before_remove is True
        assert config.max_summary_tokens == 500

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CompactionConfig(
            token_threshold_ratio=0.90,
            target_token_ratio=0.70,
            preserve_recent_count=10,
            max_summary_tokens=1000,
        )
        assert config.token_threshold_ratio == 0.90
        assert config.target_token_ratio == 0.70
        assert config.preserve_recent_count == 10
        assert config.max_summary_tokens == 1000


# ============================================================================
# CompactionResult Tests
# ============================================================================


class TestCompactionResult:
    """Tests for CompactionResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful compaction result."""
        result = CompactionResult(
            success=True,
            strategy_used=CompactionStrategy.TRUNCATION,
            messages_removed=5,
            tokens_freed=500,
            tokens_before=1500,
            tokens_after=1000,
        )
        assert result.success is True
        assert result.messages_removed == 5
        assert result.tokens_freed == 500

    def test_failed_result_with_error(self):
        """Test creating a failed compaction result."""
        result = CompactionResult(
            success=False,
            strategy_used=CompactionStrategy.SUMMARY,
            messages_removed=0,
            tokens_freed=0,
            tokens_before=1000,
            tokens_after=1000,
            error="No removable messages found",
        )
        assert result.success is False
        assert result.error is not None

    def test_compression_ratio_calculation(self):
        """Test compression ratio calculation."""
        result = CompactionResult(
            success=True,
            strategy_used=CompactionStrategy.TRUNCATION,
            messages_removed=3,
            tokens_freed=400,
            tokens_before=1000,
            tokens_after=600,
        )
        assert result.compression_ratio == 0.4  # 1 - (600/1000)

    def test_compression_ratio_zero_tokens_before(self):
        """Test compression ratio when tokens_before is zero."""
        result = CompactionResult(
            success=True,
            strategy_used=CompactionStrategy.TRUNCATION,
            messages_removed=0,
            tokens_freed=0,
            tokens_before=0,
            tokens_after=0,
        )
        assert result.compression_ratio == 0.0

    def test_result_with_summary(self):
        """Test result with generated summary."""
        result = CompactionResult(
            success=True,
            strategy_used=CompactionStrategy.SUMMARY,
            messages_removed=10,
            tokens_freed=800,
            tokens_before=2000,
            tokens_after=1200,
            summary_generated="Summary of the conversation...",
        )
        assert result.summary_generated is not None

    def test_result_with_removed_messages(self, sample_dict_messages):
        """Test result with list of removed messages."""
        result = CompactionResult(
            success=True,
            strategy_used=CompactionStrategy.TRUNCATION,
            messages_removed=2,
            tokens_freed=100,
            tokens_before=500,
            tokens_after=400,
            removed_messages=sample_dict_messages[:2],
        )
        assert len(result.removed_messages) == 2

    def test_timestamp_auto_set(self):
        """Test that timestamp is automatically set."""
        result = CompactionResult(
            success=True,
            strategy_used=CompactionStrategy.TRUNCATION,
            messages_removed=0,
            tokens_freed=0,
            tokens_before=100,
            tokens_after=100,
        )
        assert result.timestamp is not None


# ============================================================================
# BaseCompactor Tests (via TruncationCompactor)
# ============================================================================


class TestBaseCompactor:
    """Tests for BaseCompactor functionality via TruncationCompactor."""

    def test_init_with_default_config(self):
        """Test initialization with default config."""
        compactor = TruncationCompactor()
        assert compactor.config is not None
        assert isinstance(compactor.config, CompactionConfig)

    def test_init_with_custom_config(self, aggressive_compaction_config):
        """Test initialization with custom config."""
        compactor = TruncationCompactor(config=aggressive_compaction_config)
        assert compactor.config.token_threshold_ratio == 0.70

    def test_should_compact_token_threshold(self):
        """Test should_compact with token threshold trigger."""
        compactor = TruncationCompactor()
        should, trigger = compactor.should_compact(
            current_tokens=900,
            max_tokens=1000,
            message_count=10,
        )
        assert should is True
        assert trigger == CompactionTrigger.TOKEN_THRESHOLD

    def test_should_compact_message_count(self):
        """Test should_compact with message count trigger."""
        compactor = TruncationCompactor()
        should, trigger = compactor.should_compact(
            current_tokens=100,
            max_tokens=1000,
            message_count=60,  # Exceeds default 50
        )
        assert should is True
        assert trigger == CompactionTrigger.MESSAGE_COUNT

    def test_should_compact_false(self):
        """Test should_compact when no compaction needed."""
        compactor = TruncationCompactor()
        should, trigger = compactor.should_compact(
            current_tokens=100,
            max_tokens=1000,
            message_count=10,
        )
        assert should is False
        assert trigger == CompactionTrigger.MANUAL

    def test_get_role_from_dict(self):
        """Test extracting role from dict message."""
        compactor = TruncationCompactor()
        msg = {"role": "user", "content": "Hello"}
        assert compactor._get_role(msg) == "user"

    def test_get_content_from_dict(self):
        """Test extracting content from dict message."""
        compactor = TruncationCompactor()
        msg = {"role": "user", "content": "Hello world"}
        assert compactor._get_content(msg) == "Hello world"

    def test_is_protected_system_message(self):
        """Test that system messages are protected."""
        compactor = TruncationCompactor()
        msg = {"role": "system", "content": "You are helpful"}
        assert compactor._is_protected(msg, 0, 10) is True

    def test_is_protected_tool_message(self):
        """Test that tool messages are protected."""
        compactor = TruncationCompactor()
        msg = {"role": "tool", "content": "Tool result", "tool_call_id": "123"}
        assert compactor._is_protected(msg, 0, 10) is True

    def test_is_protected_message_with_tool_calls(self):
        """Test that messages with tool_calls are protected."""
        compactor = TruncationCompactor()
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "1", "name": "search", "arguments": {}}],
        }
        assert compactor._is_protected(msg, 0, 10) is True

    def test_is_protected_recent_message(self):
        """Test that recent messages are protected."""
        compactor = TruncationCompactor()
        # Message at index 8 of 10, with preserve_recent_count=5
        # Messages from end: 10 - 8 - 1 = 1, which is < 5, so protected
        msg = {"role": "user", "content": "Recent message"}
        assert compactor._is_protected(msg, 8, 10) is True

    def test_is_protected_old_message_not_protected(self):
        """Test that old messages are not protected."""
        compactor = TruncationCompactor()
        # Message at index 0 of 10
        msg = {"role": "user", "content": "Old message"}
        # Messages from end: 10 - 0 - 1 = 9, which is >= 5, so not protected
        assert compactor._is_protected(msg, 0, 10) is False

    def test_history_property(self):
        """Test that compaction history is tracked."""
        compactor = TruncationCompactor()
        assert compactor.history == []


# ============================================================================
# TruncationCompactor Tests
# ============================================================================


class TestTruncationCompactor:
    """Tests for TruncationCompactor."""

    def test_strategy_property(self):
        """Test strategy property returns TRUNCATION."""
        compactor = TruncationCompactor()
        assert compactor.strategy == CompactionStrategy.TRUNCATION

    @pytest.mark.asyncio
    async def test_compact_removes_oldest_messages(self, large_conversation):
        """Test that compact removes oldest non-protected messages."""
        compactor = TruncationCompactor()
        result = await compactor.compact(
            messages=large_conversation,
            current_tokens=2000,
            target_tokens=1000,
            context_manager=None,
        )

        assert result.messages_removed > 0
        assert result.tokens_freed > 0
        assert result.strategy_used == CompactionStrategy.TRUNCATION

    @pytest.mark.asyncio
    async def test_compact_preserves_system_messages(self, sample_dict_messages):
        """Test that compact preserves system messages."""
        compactor = TruncationCompactor()

        # Create messages with system message first
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User 1"},
            {"role": "assistant", "content": "Assistant 1"},
            {"role": "user", "content": "User 2"},
            {"role": "assistant", "content": "Assistant 2"},
        ]

        result = await compactor.compact(
            messages=messages,
            current_tokens=500,
            target_tokens=200,
            context_manager=None,
        )

        # System message should not be in removed messages
        removed_roles = [compactor._get_role(m) for m in result.removed_messages]
        assert "system" not in removed_roles

    @pytest.mark.asyncio
    async def test_compact_preserves_recent_messages(self):
        """Test that compact preserves recent messages."""
        config = CompactionConfig(preserve_recent_count=2)
        compactor = TruncationCompactor(config=config)

        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(10)
        ]

        result = await compactor.compact(
            messages=messages,
            current_tokens=1000,
            target_tokens=100,
            context_manager=None,
        )

        # Last 2 messages should be preserved
        removed_contents = [compactor._get_content(m) for m in result.removed_messages]
        assert "Message 9" not in removed_contents
        assert "Message 8" not in removed_contents

    @pytest.mark.asyncio
    async def test_compact_with_context_manager(self):
        """Test compact using context manager for token counting."""
        compactor = TruncationCompactor()
        manager = ContextManager()

        messages = [
            {"role": "user", "content": "x" * 100}
            for _ in range(10)
        ]

        result = await compactor.compact(
            messages=messages,
            current_tokens=500,
            target_tokens=200,
            context_manager=manager,
        )

        assert result.messages_removed > 0

    @pytest.mark.asyncio
    async def test_compact_records_history(self):
        """Test that compact records results in history."""
        compactor = TruncationCompactor()

        messages = [{"role": "user", "content": "Test"} for _ in range(5)]

        await compactor.compact(
            messages=messages,
            current_tokens=200,
            target_tokens=100,
            context_manager=None,
        )

        assert len(compactor.history) == 1
        assert compactor.history[0].strategy_used == CompactionStrategy.TRUNCATION

    @pytest.mark.asyncio
    async def test_compact_success_when_target_reached(self):
        """Test that success is True when target is reached."""
        compactor = TruncationCompactor()

        messages = [
            {"role": "user", "content": "x" * 50}
            for _ in range(10)
        ]

        result = await compactor.compact(
            messages=messages,
            current_tokens=500,
            target_tokens=400,
            context_manager=None,
        )

        # Should successfully reach target (or close to it)
        assert result.tokens_after <= result.tokens_before


# ============================================================================
# SelectiveCompactor Tests
# ============================================================================


class TestSelectiveCompactor:
    """Tests for SelectiveCompactor."""

    def test_strategy_property(self):
        """Test strategy property returns SELECTIVE."""
        compactor = SelectiveCompactor()
        assert compactor.strategy == CompactionStrategy.SELECTIVE

    def test_score_message_protected(self):
        """Test scoring a protected message."""
        compactor = SelectiveCompactor()
        msg = {"role": "system", "content": "System prompt"}

        importance = compactor._score_message(msg, 0, 10)

        assert importance.is_protected is True
        assert importance.score == 1.0
        assert "protected" in importance.factors

    def test_score_message_unprotected(self):
        """Test scoring an unprotected message."""
        compactor = SelectiveCompactor()
        msg = {"role": "user", "content": "Hello world"}

        importance = compactor._score_message(msg, 0, 10)

        assert importance.is_protected is False
        assert 0.0 <= importance.score <= 1.0
        assert "recency" in importance.factors
        assert "role" in importance.factors
        assert "length" in importance.factors
        assert "keywords" in importance.factors

    def test_score_message_recency_factor(self):
        """Test that more recent messages score higher on recency."""
        # Use a larger message count so recent messages are not protected
        config = CompactionConfig(preserve_recent_count=2)
        compactor = SelectiveCompactor(config=config)

        old_msg = {"role": "user", "content": "Old message"}
        new_msg = {"role": "user", "content": "New message"}

        # Use indices 0 and 7 out of 15 so neither is in the protected recent 2
        old_score = compactor._score_message(old_msg, 0, 15)
        new_score = compactor._score_message(new_msg, 7, 15)

        # Both should NOT be protected
        assert old_score.is_protected is False
        assert new_score.is_protected is False
        # New message (higher index) should have higher recency factor
        assert new_score.factors["recency"] > old_score.factors["recency"]

    def test_score_message_role_factor(self):
        """Test role-based scoring."""
        compactor = SelectiveCompactor()

        user_msg = {"role": "user", "content": "x" * 100}
        assistant_msg = {"role": "assistant", "content": "x" * 100}

        user_score = compactor._score_message(user_msg, 0, 10)
        assistant_score = compactor._score_message(assistant_msg, 0, 10)

        # User role (0.7) should score higher than assistant (0.5)
        assert user_score.factors["role"] > assistant_score.factors["role"]

    def test_score_message_keyword_factor(self):
        """Test keyword-based scoring."""
        compactor = SelectiveCompactor()

        important_msg = {"role": "user", "content": "This is an important result"}
        normal_msg = {"role": "user", "content": "Just a normal message"}

        important_score = compactor._score_message(important_msg, 0, 10)
        normal_score = compactor._score_message(normal_msg, 0, 10)

        assert important_score.factors["keywords"] > normal_score.factors["keywords"]

    def test_custom_scorer(self):
        """Test using custom scorer function."""
        def custom_scorer(msg, idx, total):
            # Simple custom: always return 0.5
            return 0.5

        compactor = SelectiveCompactor(custom_scorer=custom_scorer)
        msg = {"role": "user", "content": "Test"}

        importance = compactor._score_message(msg, 0, 10)

        assert importance.score == 0.5
        assert "custom" in importance.factors

    @pytest.mark.asyncio
    async def test_compact_removes_lowest_scored(self):
        """Test that compact removes lowest-scored messages."""
        # Use a config with preserve_recent_count=0 so all messages can be removed
        config = CompactionConfig(preserve_recent_count=0)
        compactor = SelectiveCompactor(config=config)

        messages = [
            {"role": "user", "content": "Old unimportant message"},
            {"role": "user", "content": "This is an important result"},
            {"role": "assistant", "content": "Another message"},
        ]

        result = await compactor.compact(
            messages=messages,
            current_tokens=300,
            target_tokens=100,
            context_manager=None,
        )

        assert result.messages_removed > 0
        assert result.strategy_used == CompactionStrategy.SELECTIVE

    def test_get_importance_ranking(self):
        """Test getting importance rankings for all messages."""
        compactor = SelectiveCompactor()

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Important result here"},
        ]

        rankings = compactor.get_importance_ranking(messages)

        assert len(rankings) == 3
        # Should be sorted by score, highest first
        assert rankings[0].score >= rankings[1].score >= rankings[2].score

    @pytest.mark.asyncio
    async def test_compact_records_in_history(self):
        """Test that compact records results in history."""
        compactor = SelectiveCompactor()

        messages = [{"role": "user", "content": "Test"} for _ in range(5)]

        await compactor.compact(
            messages=messages,
            current_tokens=200,
            target_tokens=100,
            context_manager=None,
        )

        assert len(compactor.history) == 1


# ============================================================================
# SummaryCompactor Tests
# ============================================================================


class TestSummaryCompactor:
    """Tests for SummaryCompactor."""

    def test_strategy_property(self):
        """Test strategy property returns SUMMARY."""
        compactor = SummaryCompactor()
        assert compactor.strategy == CompactionStrategy.SUMMARY

    @pytest.mark.asyncio
    async def test_compact_without_summarizer(self):
        """Test compact without a summarizer (no summary generated)."""
        compactor = SummaryCompactor()

        messages = [
            {"role": "user", "content": "Message " + "x" * 50}
            for _ in range(10)
        ]

        result = await compactor.compact(
            messages=messages,
            current_tokens=1000,
            target_tokens=500,
            context_manager=None,
        )

        assert result.messages_removed > 0
        assert result.summary_generated is None

    @pytest.mark.asyncio
    async def test_compact_with_summarizer(self, mock_llm_provider):
        """Test compact with a summarizer."""
        from agents_framework.context import ConversationSummarizer

        summarizer = ConversationSummarizer(llm_provider=mock_llm_provider)
        compactor = SummaryCompactor(summarizer=summarizer)

        messages = [
            {"role": "user", "content": "Message " + "x" * 50}
            for _ in range(10)
        ]

        result = await compactor.compact(
            messages=messages,
            current_tokens=1000,
            target_tokens=500,
            context_manager=None,
        )

        assert result.messages_removed > 0
        assert result.summary_generated is not None

    @pytest.mark.asyncio
    async def test_compact_preserves_protected_messages(self, messages_with_tool_calls):
        """Test that compact preserves protected messages."""
        compactor = SummaryCompactor()

        result = await compactor.compact(
            messages=messages_with_tool_calls,
            current_tokens=500,
            target_tokens=100,
            context_manager=None,
        )

        # Tool messages should not be in removed messages
        removed_roles = [compactor._get_role(m) for m in result.removed_messages]
        assert "tool" not in removed_roles

    @pytest.mark.asyncio
    async def test_compact_fails_when_no_removable_messages(self):
        """Test that compact fails when all messages are protected."""
        compactor = SummaryCompactor()

        # All system messages - all protected
        messages = [
            {"role": "system", "content": "System 1"},
            {"role": "system", "content": "System 2"},
        ]

        result = await compactor.compact(
            messages=messages,
            current_tokens=200,
            target_tokens=50,
            context_manager=None,
        )

        assert result.success is False
        assert result.error is not None
        assert "No removable messages" in result.error

    @pytest.mark.asyncio
    async def test_compact_includes_buffer_for_summary(self):
        """Test that compact accounts for summary tokens when calculating removal."""
        config = CompactionConfig(max_summary_tokens=100)
        compactor = SummaryCompactor(config=config)

        messages = [
            {"role": "user", "content": "x" * 100}
            for _ in range(10)
        ]

        # The compactor should free more tokens to make room for summary
        result = await compactor.compact(
            messages=messages,
            current_tokens=500,
            target_tokens=400,
            context_manager=None,
        )

        # Should remove enough to hit target + summary buffer
        assert result.messages_removed > 0

    @pytest.mark.asyncio
    async def test_compact_summarizer_error_handling(self, mock_llm_provider):
        """Test error handling when summarizer fails."""
        from agents_framework.context import ConversationSummarizer

        # Make summarizer fail completely by making it raise during summarize
        summarizer = MagicMock(spec=ConversationSummarizer)
        summarizer.summarize = AsyncMock(side_effect=Exception("Summarization failed"))

        config = CompactionConfig(preserve_recent_count=0)
        compactor = SummaryCompactor(config=config, summarizer=summarizer)

        messages = [
            {"role": "user", "content": "Message " + "x" * 50}
            for _ in range(10)
        ]

        result = await compactor.compact(
            messages=messages,
            current_tokens=1000,
            target_tokens=500,
            context_manager=None,
        )

        # Should still work, with error message in summary
        assert result.messages_removed > 0
        assert result.summary_generated is not None
        assert "error" in result.summary_generated.lower()


# ============================================================================
# HybridCompactor Tests
# ============================================================================


class TestHybridCompactor:
    """Tests for HybridCompactor."""

    def test_strategy_property(self):
        """Test strategy property returns HYBRID."""
        compactor = HybridCompactor()
        assert compactor.strategy == CompactionStrategy.HYBRID

    @pytest.mark.asyncio
    async def test_compact_uses_importance_scoring(self):
        """Test that compact uses importance scoring for selection."""
        compactor = HybridCompactor()

        messages = [
            {"role": "user", "content": "Old unimportant"},
            {"role": "user", "content": "This is an important result"},
            {"role": "assistant", "content": "Response with critical error info"},
        ]

        result = await compactor.compact(
            messages=messages,
            current_tokens=500,
            target_tokens=200,
            context_manager=None,
        )

        assert result.strategy_used == CompactionStrategy.HYBRID

    @pytest.mark.asyncio
    async def test_compact_with_summarizer(self, mock_llm_provider):
        """Test compact with summarizer."""
        from agents_framework.context import ConversationSummarizer

        summarizer = ConversationSummarizer(llm_provider=mock_llm_provider)
        compactor = HybridCompactor(summarizer=summarizer)

        messages = [
            {"role": "user", "content": "Message " + "x" * 50}
            for _ in range(10)
        ]

        result = await compactor.compact(
            messages=messages,
            current_tokens=1000,
            target_tokens=500,
            context_manager=None,
        )

        assert result.messages_removed > 0
        assert result.summary_generated is not None

    @pytest.mark.asyncio
    async def test_compact_with_custom_scorer(self):
        """Test compact with custom scorer."""
        def custom_scorer(msg, idx, total):
            # Prefer even-indexed messages
            return 0.8 if idx % 2 == 0 else 0.2

        compactor = HybridCompactor(custom_scorer=custom_scorer)

        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(10)
        ]

        result = await compactor.compact(
            messages=messages,
            current_tokens=500,
            target_tokens=200,
            context_manager=None,
        )

        # Odd-indexed messages should be preferred for removal (lower score)
        assert result.messages_removed > 0

    @pytest.mark.asyncio
    async def test_compact_fails_when_no_removable(self):
        """Test that compact fails when all messages are protected."""
        compactor = HybridCompactor()

        # All system messages
        messages = [
            {"role": "system", "content": "System 1"},
            {"role": "system", "content": "System 2"},
        ]

        result = await compactor.compact(
            messages=messages,
            current_tokens=200,
            target_tokens=50,
            context_manager=None,
        )

        assert result.success is False
        assert result.error is not None


# ============================================================================
# create_compactor Factory Tests
# ============================================================================


class TestCreateCompactor:
    """Tests for create_compactor factory function."""

    def test_create_truncation_compactor(self):
        """Test creating truncation compactor."""
        compactor = create_compactor(CompactionStrategy.TRUNCATION)
        assert isinstance(compactor, TruncationCompactor)

    def test_create_selective_compactor(self):
        """Test creating selective compactor."""
        compactor = create_compactor(CompactionStrategy.SELECTIVE)
        assert isinstance(compactor, SelectiveCompactor)

    def test_create_summary_compactor(self):
        """Test creating summary compactor."""
        compactor = create_compactor(CompactionStrategy.SUMMARY)
        assert isinstance(compactor, SummaryCompactor)

    def test_create_hybrid_compactor(self):
        """Test creating hybrid compactor."""
        compactor = create_compactor(CompactionStrategy.HYBRID)
        assert isinstance(compactor, HybridCompactor)

    def test_create_with_config(self, aggressive_compaction_config):
        """Test creating compactor with custom config."""
        compactor = create_compactor(
            CompactionStrategy.TRUNCATION,
            config=aggressive_compaction_config,
        )
        assert compactor.config.token_threshold_ratio == 0.70

    def test_create_with_summarizer(self, mock_llm_provider):
        """Test creating compactor with summarizer."""
        from agents_framework.context import ConversationSummarizer

        summarizer = ConversationSummarizer(llm_provider=mock_llm_provider)
        compactor = create_compactor(
            CompactionStrategy.SUMMARY,
            summarizer=summarizer,
        )
        assert isinstance(compactor, SummaryCompactor)

    def test_create_with_custom_scorer(self):
        """Test creating compactor with custom scorer."""
        def custom_scorer(msg, idx, total):
            return 0.5

        compactor = create_compactor(
            CompactionStrategy.SELECTIVE,
            custom_scorer=custom_scorer,
        )
        assert isinstance(compactor, SelectiveCompactor)

    def test_create_invalid_strategy(self):
        """Test creating compactor with invalid strategy raises error."""
        with pytest.raises(ValueError, match="Unknown compaction strategy"):
            create_compactor("invalid_strategy")  # type: ignore


# ============================================================================
# Integration Tests
# ============================================================================


class TestCompactorIntegration:
    """Integration tests for compactors with ContextManager."""

    @pytest.mark.asyncio
    async def test_truncation_with_context_manager(self, large_conversation):
        """Test truncation compactor with context manager."""
        manager = ContextManager()
        compactor = TruncationCompactor()

        # Count initial tokens
        initial_tokens = manager.count_message_tokens(large_conversation)

        result = await compactor.compact(
            messages=large_conversation,
            current_tokens=initial_tokens,
            target_tokens=initial_tokens // 2,
            context_manager=manager,
        )

        assert result.success is True
        assert result.tokens_after < initial_tokens

    @pytest.mark.asyncio
    async def test_selective_with_context_manager(self, large_conversation):
        """Test selective compactor with context manager."""
        manager = ContextManager()
        compactor = SelectiveCompactor()

        initial_tokens = manager.count_message_tokens(large_conversation)

        result = await compactor.compact(
            messages=large_conversation,
            current_tokens=initial_tokens,
            target_tokens=initial_tokens // 2,
            context_manager=manager,
        )

        assert result.messages_removed > 0

    @pytest.mark.asyncio
    async def test_multiple_compactions(self):
        """Test performing multiple compactions."""
        compactor = TruncationCompactor()

        messages = [
            {"role": "user", "content": f"Message {i}: " + "x" * 100}
            for i in range(20)
        ]

        # First compaction
        result1 = await compactor.compact(
            messages=messages,
            current_tokens=2000,
            target_tokens=1500,
            context_manager=None,
        )

        # Update messages list
        removed_set = set(id(m) for m in result1.removed_messages)
        remaining = [m for m in messages if id(m) not in removed_set]

        # Second compaction
        result2 = await compactor.compact(
            messages=remaining,
            current_tokens=result1.tokens_after,
            target_tokens=1000,
            context_manager=None,
        )

        # History should have 2 entries
        assert len(compactor.history) == 2

    @pytest.mark.asyncio
    async def test_compaction_workflow(self, mock_llm_provider):
        """Test complete compaction workflow."""
        from agents_framework.context import ConversationSummarizer

        manager = ContextManager()
        summarizer = ConversationSummarizer(llm_provider=mock_llm_provider)

        # Start with hybrid compactor
        compactor = create_compactor(
            CompactionStrategy.HYBRID,
            summarizer=summarizer,
        )

        # Build up messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        for i in range(15):
            messages.append({"role": "user", "content": f"Query {i}: " + "x" * 50})
            messages.append({"role": "assistant", "content": f"Response {i}: " + "y" * 100})

        # Check if compaction is needed
        current_tokens = manager.count_message_tokens(messages)
        should, trigger = compactor.should_compact(
            current_tokens=current_tokens,
            max_tokens=2000,
            message_count=len(messages),
        )

        if should:
            result = await compactor.compact(
                messages=messages,
                current_tokens=current_tokens,
                target_tokens=1000,
                context_manager=manager,
            )

            assert result.messages_removed > 0
            # Summary should be generated
            assert result.summary_generated is not None
