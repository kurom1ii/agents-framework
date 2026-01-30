"""Context compaction strategies for agents framework.

This module provides compaction strategies to reduce context size when
approaching token limits:
- TruncationCompactor: Remove oldest messages
- SelectiveCompactor: Remove by importance scoring
- SummaryCompactor: Summarize then truncate

KUR-34: Compactor Implementation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agents_framework.context.manager import ContextManager
    from agents_framework.context.summarizer import ConversationSummarizer


class CompactionStrategy(str, Enum):
    """Available compaction strategies."""

    TRUNCATION = "truncation"
    SELECTIVE = "selective"
    SUMMARY = "summary"
    HYBRID = "hybrid"  # Combines selective + summary


class CompactionTrigger(str, Enum):
    """Triggers for automatic compaction."""

    TOKEN_THRESHOLD = "token_threshold"
    MESSAGE_COUNT = "message_count"
    MANUAL = "manual"
    TIME_BASED = "time_based"


@dataclass
class CompactionConfig:
    """Configuration for compaction behavior."""

    # Token-based triggers
    token_threshold_ratio: float = 0.85  # Trigger when usage exceeds this ratio
    target_token_ratio: float = 0.60  # Target usage after compaction

    # Message-based triggers
    max_messages_before_compact: int = 50
    target_messages_after_compact: int = 25

    # Time-based triggers
    max_age_hours: float = 24.0  # Remove messages older than this

    # Behavior settings
    preserve_recent_count: int = 5  # Always keep N most recent messages
    preserve_system_messages: bool = True
    preserve_tool_calls: bool = True

    # Summary settings
    summarize_before_remove: bool = True
    max_summary_tokens: int = 500


@dataclass
class CompactionResult:
    """Result of a compaction operation."""

    success: bool
    strategy_used: CompactionStrategy
    messages_removed: int
    tokens_freed: int
    tokens_before: int
    tokens_after: int
    summary_generated: Optional[str] = None
    removed_messages: List[Any] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio achieved."""
        if self.tokens_before == 0:
            return 0.0
        return 1.0 - (self.tokens_after / self.tokens_before)


class MessageImportance(BaseModel):
    """Importance scoring for a message."""

    message_index: int = Field(description="Index in message list")
    score: float = Field(ge=0.0, le=1.0, description="Importance score 0-1")
    factors: Dict[str, float] = Field(
        default_factory=dict, description="Individual factor scores"
    )
    is_protected: bool = Field(
        default=False, description="Whether message is protected from removal"
    )


class BaseCompactor(ABC):
    """Abstract base class for context compactors."""

    def __init__(self, config: Optional[CompactionConfig] = None):
        """Initialize compactor.

        Args:
            config: Compaction configuration.
        """
        self.config = config or CompactionConfig()
        self._compaction_history: List[CompactionResult] = []

    @property
    @abstractmethod
    def strategy(self) -> CompactionStrategy:
        """Get the compaction strategy type."""
        ...

    @abstractmethod
    async def compact(
        self,
        messages: List[Any],
        current_tokens: int,
        target_tokens: int,
        context_manager: Optional[ContextManager] = None,
    ) -> CompactionResult:
        """Compact the message list.

        Args:
            messages: List of messages to compact.
            current_tokens: Current token count.
            target_tokens: Target token count after compaction.
            context_manager: Optional context manager for token counting.

        Returns:
            CompactionResult with details of the operation.
        """
        ...

    def should_compact(
        self,
        current_tokens: int,
        max_tokens: int,
        message_count: int,
    ) -> Tuple[bool, CompactionTrigger]:
        """Check if compaction should be triggered.

        Args:
            current_tokens: Current token count.
            max_tokens: Maximum allowed tokens.
            message_count: Current message count.

        Returns:
            Tuple of (should_compact, trigger_reason).
        """
        # Check token threshold
        if current_tokens / max_tokens >= self.config.token_threshold_ratio:
            return True, CompactionTrigger.TOKEN_THRESHOLD

        # Check message count
        if message_count >= self.config.max_messages_before_compact:
            return True, CompactionTrigger.MESSAGE_COUNT

        return False, CompactionTrigger.MANUAL

    def _get_role(self, message: Any) -> str:
        """Extract role from a message."""
        if isinstance(message, dict):
            return message.get("role", "")
        role = getattr(message, "role", "")
        if hasattr(role, "value"):
            return role.value
        return str(role)

    def _get_content(self, message: Any) -> str:
        """Extract content from a message."""
        if isinstance(message, dict):
            return message.get("content", "")
        return getattr(message, "content", "")

    def _is_protected(self, message: Any, index: int, total: int) -> bool:
        """Check if a message should be protected from removal.

        Args:
            message: The message to check.
            index: Index in message list.
            total: Total number of messages.

        Returns:
            True if message should be protected.
        """
        role = self._get_role(message)

        # Protect system messages
        if self.config.preserve_system_messages and role == "system":
            return True

        # Protect tool calls and results
        if self.config.preserve_tool_calls:
            if role == "tool":
                return True
            if isinstance(message, dict):
                if message.get("tool_calls"):
                    return True
            elif hasattr(message, "tool_calls") and message.tool_calls:
                return True

        # Protect recent messages
        messages_from_end = total - index - 1
        if messages_from_end < self.config.preserve_recent_count:
            return True

        return False

    @property
    def history(self) -> List[CompactionResult]:
        """Get compaction history."""
        return self._compaction_history


class TruncationCompactor(BaseCompactor):
    """Compactor that removes oldest messages first.

    Simple FIFO strategy - removes messages from the beginning of the
    conversation until target token count is reached.
    """

    @property
    def strategy(self) -> CompactionStrategy:
        return CompactionStrategy.TRUNCATION

    async def compact(
        self,
        messages: List[Any],
        current_tokens: int,
        target_tokens: int,
        context_manager: Optional[ContextManager] = None,
    ) -> CompactionResult:
        """Remove oldest messages until target is reached.

        Args:
            messages: List of messages to compact.
            current_tokens: Current token count.
            target_tokens: Target token count after compaction.
            context_manager: Optional context manager for token counting.

        Returns:
            CompactionResult with details.
        """
        tokens_before = current_tokens
        removed_messages: List[Any] = []
        remaining_messages = list(messages)
        tokens_freed = 0

        # Sort by index to find removable messages
        total = len(remaining_messages)

        # Iterate from oldest to newest, skip protected messages
        i = 0
        while current_tokens > target_tokens and i < len(remaining_messages):
            msg = remaining_messages[i]

            # Check if message is protected
            if self._is_protected(msg, i, total):
                i += 1
                continue

            # Calculate tokens for this message
            if context_manager:
                msg_tokens = context_manager.count_message_tokens([msg])
            else:
                msg_tokens = len(self._get_content(msg)) // 4 + 3

            # Remove message
            removed_messages.append(remaining_messages.pop(i))
            current_tokens -= msg_tokens
            tokens_freed += msg_tokens
            total -= 1
            # Don't increment i since we popped

        result = CompactionResult(
            success=current_tokens <= target_tokens,
            strategy_used=self.strategy,
            messages_removed=len(removed_messages),
            tokens_freed=tokens_freed,
            tokens_before=tokens_before,
            tokens_after=current_tokens,
            removed_messages=removed_messages,
        )

        self._compaction_history.append(result)
        return result


class SelectiveCompactor(BaseCompactor):
    """Compactor that removes messages by importance scoring.

    Uses multiple factors to determine message importance:
    - Recency: More recent = more important
    - Role: System > Assistant with tools > User > Assistant
    - Content length: Longer messages may be more important
    - References: Messages that reference others are important
    """

    def __init__(
        self,
        config: Optional[CompactionConfig] = None,
        custom_scorer: Optional[Callable[[Any, int, int], float]] = None,
    ):
        """Initialize selective compactor.

        Args:
            config: Compaction configuration.
            custom_scorer: Optional custom scoring function.
        """
        super().__init__(config)
        self._custom_scorer = custom_scorer

    @property
    def strategy(self) -> CompactionStrategy:
        return CompactionStrategy.SELECTIVE

    def _score_message(
        self, message: Any, index: int, total: int
    ) -> MessageImportance:
        """Score a message's importance.

        Args:
            message: The message to score.
            index: Index in message list.
            total: Total number of messages.

        Returns:
            MessageImportance with score and factors.
        """
        factors: Dict[str, float] = {}

        # Check if protected
        if self._is_protected(message, index, total):
            return MessageImportance(
                message_index=index,
                score=1.0,
                factors={"protected": 1.0},
                is_protected=True,
            )

        # Custom scorer takes precedence
        if self._custom_scorer:
            custom_score = self._custom_scorer(message, index, total)
            factors["custom"] = custom_score
            return MessageImportance(
                message_index=index,
                score=custom_score,
                factors=factors,
            )

        role = self._get_role(message)
        content = self._get_content(message)

        # Recency score (0.3 weight) - more recent = higher score
        recency = index / max(total - 1, 1)
        factors["recency"] = recency * 0.3

        # Role score (0.3 weight)
        role_scores = {
            "system": 1.0,
            "user": 0.7,
            "assistant": 0.5,
            "tool": 0.8,
        }
        role_score = role_scores.get(role, 0.5)
        factors["role"] = role_score * 0.3

        # Content length score (0.2 weight) - moderate length preferred
        # Very short or very long messages get lower scores
        length = len(content)
        if 50 <= length <= 500:
            length_score = 1.0
        elif length < 50:
            length_score = length / 50
        else:
            length_score = max(0.5, 1.0 - (length - 500) / 2000)
        factors["length"] = length_score * 0.2

        # Keyword importance (0.2 weight) - check for important keywords
        important_keywords = [
            "important",
            "critical",
            "error",
            "exception",
            "result",
            "answer",
            "conclusion",
            "summary",
        ]
        keyword_found = any(kw in content.lower() for kw in important_keywords)
        factors["keywords"] = (1.0 if keyword_found else 0.5) * 0.2

        total_score = sum(factors.values())

        return MessageImportance(
            message_index=index,
            score=min(1.0, total_score),
            factors=factors,
        )

    async def compact(
        self,
        messages: List[Any],
        current_tokens: int,
        target_tokens: int,
        context_manager: Optional[ContextManager] = None,
    ) -> CompactionResult:
        """Remove lowest-importance messages until target is reached.

        Args:
            messages: List of messages to compact.
            current_tokens: Current token count.
            target_tokens: Target token count after compaction.
            context_manager: Optional context manager for token counting.

        Returns:
            CompactionResult with details.
        """
        tokens_before = current_tokens
        total = len(messages)

        # Score all messages
        scored = [self._score_message(msg, i, total) for i, msg in enumerate(messages)]

        # Sort by score (lowest first) - these will be removed first
        scored_sorted = sorted(
            [(s, i) for i, s in enumerate(scored) if not s.is_protected],
            key=lambda x: x[0].score,
        )

        removed_messages: List[Any] = []
        removed_indices: set[int] = set()
        tokens_freed = 0

        # Remove lowest scored messages until target reached
        for score_info, original_idx in scored_sorted:
            if current_tokens <= target_tokens:
                break

            msg = messages[original_idx]

            # Calculate tokens for this message
            if context_manager:
                msg_tokens = context_manager.count_message_tokens([msg])
            else:
                msg_tokens = len(self._get_content(msg)) // 4 + 3

            removed_messages.append(msg)
            removed_indices.add(original_idx)
            current_tokens -= msg_tokens
            tokens_freed += msg_tokens

        result = CompactionResult(
            success=current_tokens <= target_tokens,
            strategy_used=self.strategy,
            messages_removed=len(removed_messages),
            tokens_freed=tokens_freed,
            tokens_before=tokens_before,
            tokens_after=current_tokens,
            removed_messages=removed_messages,
        )

        self._compaction_history.append(result)
        return result

    def get_importance_ranking(
        self, messages: List[Any]
    ) -> List[MessageImportance]:
        """Get importance rankings for all messages.

        Args:
            messages: List of messages to rank.

        Returns:
            List of MessageImportance sorted by score (highest first).
        """
        total = len(messages)
        scored = [self._score_message(msg, i, total) for i, msg in enumerate(messages)]
        return sorted(scored, key=lambda x: x.score, reverse=True)


class SummaryCompactor(BaseCompactor):
    """Compactor that summarizes messages before removing them.

    Creates a summary of messages being removed and adds it as a
    system message, preserving context while reducing token count.
    """

    def __init__(
        self,
        config: Optional[CompactionConfig] = None,
        summarizer: Optional[ConversationSummarizer] = None,
    ):
        """Initialize summary compactor.

        Args:
            config: Compaction configuration.
            summarizer: Conversation summarizer instance.
        """
        super().__init__(config)
        self._summarizer = summarizer

    @property
    def strategy(self) -> CompactionStrategy:
        return CompactionStrategy.SUMMARY

    async def compact(
        self,
        messages: List[Any],
        current_tokens: int,
        target_tokens: int,
        context_manager: Optional[ContextManager] = None,
    ) -> CompactionResult:
        """Summarize and remove messages until target is reached.

        Args:
            messages: List of messages to compact.
            current_tokens: Current token count.
            target_tokens: Target token count after compaction.
            context_manager: Optional context manager for token counting.

        Returns:
            CompactionResult with details.
        """
        tokens_before = current_tokens
        total = len(messages)

        # Find messages to remove (oldest non-protected)
        to_remove: List[Tuple[int, Any]] = []
        tokens_to_free = current_tokens - target_tokens

        # Add buffer for summary
        tokens_to_free += self.config.max_summary_tokens

        accumulated_tokens = 0
        for i, msg in enumerate(messages):
            if self._is_protected(msg, i, total):
                continue

            if context_manager:
                msg_tokens = context_manager.count_message_tokens([msg])
            else:
                msg_tokens = len(self._get_content(msg)) // 4 + 3

            to_remove.append((i, msg))
            accumulated_tokens += msg_tokens

            if accumulated_tokens >= tokens_to_free:
                break

        if not to_remove:
            return CompactionResult(
                success=False,
                strategy_used=self.strategy,
                messages_removed=0,
                tokens_freed=0,
                tokens_before=tokens_before,
                tokens_after=current_tokens,
                error="No removable messages found",
            )

        # Generate summary if summarizer available
        summary = None
        if self._summarizer and self.config.summarize_before_remove:
            messages_to_summarize = [msg for _, msg in to_remove]
            try:
                summary = await self._summarizer.summarize(
                    messages_to_summarize,
                    max_tokens=self.config.max_summary_tokens,
                )
            except Exception as e:
                # Fall back to no summary on error
                summary = f"[Previous conversation context removed due to error: {e}]"

        # Calculate actual tokens freed
        tokens_freed = accumulated_tokens
        if summary:
            if context_manager:
                summary_tokens = context_manager.count_tokens(summary)
            else:
                summary_tokens = len(summary) // 4 + 3
            tokens_freed -= summary_tokens

        removed_messages = [msg for _, msg in to_remove]
        tokens_after = current_tokens - tokens_freed

        result = CompactionResult(
            success=tokens_after <= target_tokens,
            strategy_used=self.strategy,
            messages_removed=len(removed_messages),
            tokens_freed=tokens_freed,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            summary_generated=summary,
            removed_messages=removed_messages,
        )

        self._compaction_history.append(result)
        return result


class HybridCompactor(BaseCompactor):
    """Compactor combining selective scoring with summarization.

    Uses importance scoring to select messages for removal, then
    summarizes them before removal to preserve context.
    """

    def __init__(
        self,
        config: Optional[CompactionConfig] = None,
        summarizer: Optional[ConversationSummarizer] = None,
        custom_scorer: Optional[Callable[[Any, int, int], float]] = None,
    ):
        """Initialize hybrid compactor.

        Args:
            config: Compaction configuration.
            summarizer: Conversation summarizer instance.
            custom_scorer: Optional custom scoring function.
        """
        super().__init__(config)
        self._selective = SelectiveCompactor(config, custom_scorer)
        self._summary = SummaryCompactor(config, summarizer)
        self._summarizer = summarizer

    @property
    def strategy(self) -> CompactionStrategy:
        return CompactionStrategy.HYBRID

    async def compact(
        self,
        messages: List[Any],
        current_tokens: int,
        target_tokens: int,
        context_manager: Optional[ContextManager] = None,
    ) -> CompactionResult:
        """Selectively remove and summarize messages.

        Args:
            messages: List of messages to compact.
            current_tokens: Current token count.
            target_tokens: Target token count after compaction.
            context_manager: Optional context manager for token counting.

        Returns:
            CompactionResult with details.
        """
        tokens_before = current_tokens

        # Get importance rankings
        rankings = self._selective.get_importance_ranking(messages)

        # Select lowest-importance messages for removal
        to_remove: List[Any] = []
        accumulated_tokens = 0
        tokens_needed = current_tokens - target_tokens + self.config.max_summary_tokens

        for importance in reversed(rankings):  # Start with lowest scores
            if importance.is_protected:
                continue

            msg = messages[importance.message_index]
            if context_manager:
                msg_tokens = context_manager.count_message_tokens([msg])
            else:
                msg_tokens = len(self._get_content(msg)) // 4 + 3

            to_remove.append(msg)
            accumulated_tokens += msg_tokens

            if accumulated_tokens >= tokens_needed:
                break

        if not to_remove:
            return CompactionResult(
                success=False,
                strategy_used=self.strategy,
                messages_removed=0,
                tokens_freed=0,
                tokens_before=tokens_before,
                tokens_after=current_tokens,
                error="No removable messages found",
            )

        # Generate summary
        summary = None
        if self._summarizer and self.config.summarize_before_remove:
            try:
                summary = await self._summarizer.summarize(
                    to_remove,
                    max_tokens=self.config.max_summary_tokens,
                )
            except Exception as e:
                summary = f"[Context summarization failed: {e}]"

        # Calculate tokens
        tokens_freed = accumulated_tokens
        if summary:
            if context_manager:
                summary_tokens = context_manager.count_tokens(summary)
            else:
                summary_tokens = len(summary) // 4 + 3
            tokens_freed -= summary_tokens

        tokens_after = current_tokens - tokens_freed

        result = CompactionResult(
            success=tokens_after <= target_tokens,
            strategy_used=self.strategy,
            messages_removed=len(to_remove),
            tokens_freed=tokens_freed,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            summary_generated=summary,
            removed_messages=to_remove,
        )

        self._compaction_history.append(result)
        return result


def create_compactor(
    strategy: CompactionStrategy,
    config: Optional[CompactionConfig] = None,
    summarizer: Optional[ConversationSummarizer] = None,
    custom_scorer: Optional[Callable[[Any, int, int], float]] = None,
) -> BaseCompactor:
    """Factory function to create a compactor.

    Args:
        strategy: The compaction strategy to use.
        config: Optional compaction configuration.
        summarizer: Optional summarizer for summary-based strategies.
        custom_scorer: Optional custom scoring function.

    Returns:
        A compactor instance.
    """
    if strategy == CompactionStrategy.TRUNCATION:
        return TruncationCompactor(config)
    elif strategy == CompactionStrategy.SELECTIVE:
        return SelectiveCompactor(config, custom_scorer)
    elif strategy == CompactionStrategy.SUMMARY:
        return SummaryCompactor(config, summarizer)
    elif strategy == CompactionStrategy.HYBRID:
        return HybridCompactor(config, summarizer, custom_scorer)
    else:
        raise ValueError(f"Unknown compaction strategy: {strategy}")
