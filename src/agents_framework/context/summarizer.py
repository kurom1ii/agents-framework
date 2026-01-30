"""Conversation summarization for context management.

This module provides summarization capabilities for managing context:
- ConversationSummarizer: Main summarization class
- Progressive summarization for long conversations
- Summary caching for efficiency
- Customizable summary prompts

KUR-35: Summarizer Implementation
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Tuple

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agents_framework.llm import LLMProvider, Message


class SummaryType(str, Enum):
    """Types of summaries that can be generated."""

    BRIEF = "brief"  # Very short, key points only
    DETAILED = "detailed"  # Comprehensive summary
    CONTEXTUAL = "contextual"  # Focus on context for continuation
    ACTION_FOCUSED = "action_focused"  # Focus on actions and decisions
    TOPIC_BASED = "topic_based"  # Organize by topics discussed


class SummaryPrompt(BaseModel):
    """Configuration for summary generation prompts."""

    system_prompt: str = Field(
        default=(
            "You are a conversation summarizer. Create concise, accurate summaries "
            "that preserve key information, decisions made, and context needed for "
            "continuing the conversation."
        ),
        description="System prompt for summarization",
    )
    user_prompt_template: str = Field(
        default=(
            "Please summarize the following conversation. "
            "Focus on: key points, decisions made, actions taken, and important context.\n\n"
            "Conversation:\n{conversation}\n\n"
            "Provide a {summary_type} summary in no more than {max_tokens} tokens."
        ),
        description="Template for user prompt with placeholders",
    )
    summary_type: SummaryType = Field(
        default=SummaryType.CONTEXTUAL,
        description="Type of summary to generate",
    )


# Default prompts for different contexts
DEFAULT_PROMPTS: Dict[str, SummaryPrompt] = {
    "general": SummaryPrompt(),
    "technical": SummaryPrompt(
        system_prompt=(
            "You are a technical conversation summarizer. Focus on technical details, "
            "code snippets, architecture decisions, and implementation specifics."
        ),
        user_prompt_template=(
            "Summarize this technical conversation, preserving:\n"
            "- Technical decisions and rationale\n"
            "- Code or configuration discussed\n"
            "- Problems identified and solutions proposed\n\n"
            "Conversation:\n{conversation}\n\n"
            "Summary ({summary_type}, max {max_tokens} tokens):"
        ),
    ),
    "task_oriented": SummaryPrompt(
        system_prompt=(
            "You are a task-focused summarizer. Extract tasks, assignments, "
            "deadlines, and action items from conversations."
        ),
        user_prompt_template=(
            "Extract and summarize from this conversation:\n"
            "- Tasks assigned or discussed\n"
            "- Decisions made\n"
            "- Action items and next steps\n"
            "- Any blockers or dependencies mentioned\n\n"
            "Conversation:\n{conversation}\n\n"
            "Summary:"
        ),
        summary_type=SummaryType.ACTION_FOCUSED,
    ),
    "brief": SummaryPrompt(
        system_prompt="You are a concise summarizer. Create very brief summaries.",
        user_prompt_template=(
            "Provide a very brief summary (2-3 sentences) of this conversation:\n\n"
            "{conversation}\n\n"
            "Brief summary:"
        ),
        summary_type=SummaryType.BRIEF,
    ),
}


@dataclass
class SummaryCache:
    """Cache for storing generated summaries."""

    max_size: int = 100
    ttl_seconds: int = 3600  # 1 hour default

    _cache: Dict[str, Tuple[str, datetime]] = field(default_factory=dict)

    def _generate_key(self, messages: List[Any]) -> str:
        """Generate cache key from messages."""
        content = "".join(
            self._get_content(msg) for msg in messages
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_content(self, message: Any) -> str:
        """Extract content from message."""
        if isinstance(message, dict):
            return message.get("content", "")
        return getattr(message, "content", "")

    def get(self, messages: List[Any]) -> Optional[str]:
        """Get cached summary if available and not expired.

        Args:
            messages: The messages to look up.

        Returns:
            Cached summary or None.
        """
        key = self._generate_key(messages)
        if key not in self._cache:
            return None

        summary, timestamp = self._cache[key]
        if datetime.utcnow() - timestamp > timedelta(seconds=self.ttl_seconds):
            del self._cache[key]
            return None

        return summary

    def set(self, messages: List[Any], summary: str) -> None:
        """Cache a summary.

        Args:
            messages: The source messages.
            summary: The generated summary.
        """
        # Evict oldest entries if at capacity
        if len(self._cache) >= self.max_size:
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k][1]
            )
            del self._cache[oldest_key]

        key = self._generate_key(messages)
        self._cache[key] = (summary, datetime.utcnow())

    def clear(self) -> None:
        """Clear all cached summaries."""
        self._cache.clear()

    def invalidate(self, messages: List[Any]) -> bool:
        """Invalidate a specific cache entry.

        Args:
            messages: The messages to invalidate.

        Returns:
            True if entry was found and removed.
        """
        key = self._generate_key(messages)
        if key in self._cache:
            del self._cache[key]
            return True
        return False


@dataclass
class ProgressiveSummary:
    """Tracks progressive summarization state."""

    current_summary: str = ""
    messages_summarized: int = 0
    summary_chain: List[str] = field(default_factory=list)  # History of summaries
    last_update: datetime = field(default_factory=datetime.utcnow)

    def add_summary(self, summary: str, messages_count: int) -> None:
        """Add a new summary to the chain.

        Args:
            summary: The new summary.
            messages_count: Number of messages summarized.
        """
        if self.current_summary:
            self.summary_chain.append(self.current_summary)
        self.current_summary = summary
        self.messages_summarized += messages_count
        self.last_update = datetime.utcnow()

    def get_full_context(self) -> str:
        """Get combined context from all summaries."""
        if not self.summary_chain:
            return self.current_summary
        return "\n\n---\n\n".join(self.summary_chain + [self.current_summary])


class SummarizerProtocol(Protocol):
    """Protocol for summarizer implementations."""

    async def summarize(
        self,
        messages: List[Any],
        max_tokens: Optional[int] = None,
    ) -> str:
        """Summarize messages."""
        ...


class BaseSummarizer(ABC):
    """Abstract base class for summarizers."""

    @abstractmethod
    async def summarize(
        self,
        messages: List[Any],
        max_tokens: Optional[int] = None,
    ) -> str:
        """Summarize the given messages.

        Args:
            messages: List of messages to summarize.
            max_tokens: Maximum tokens for the summary.

        Returns:
            Generated summary string.
        """
        ...


class SimpleSummarizer(BaseSummarizer):
    """Simple summarizer that creates extractive summaries.

    Does not require an LLM - creates summaries by extracting
    key content from messages.
    """

    def __init__(
        self,
        max_chars_per_message: int = 100,
        include_roles: bool = True,
    ):
        """Initialize simple summarizer.

        Args:
            max_chars_per_message: Maximum characters per message in summary.
            include_roles: Whether to include role prefixes.
        """
        self._max_chars = max_chars_per_message
        self._include_roles = include_roles

    def _get_content(self, message: Any) -> str:
        """Extract content from message."""
        if isinstance(message, dict):
            return message.get("content", "")
        return getattr(message, "content", "")

    def _get_role(self, message: Any) -> str:
        """Extract role from message."""
        if isinstance(message, dict):
            return message.get("role", "")
        role = getattr(message, "role", "")
        if hasattr(role, "value"):
            return role.value
        return str(role)

    async def summarize(
        self,
        messages: List[Any],
        max_tokens: Optional[int] = None,
    ) -> str:
        """Create extractive summary of messages.

        Args:
            messages: List of messages to summarize.
            max_tokens: Maximum tokens for summary (approximated by chars/4).

        Returns:
            Extractive summary string.
        """
        max_chars = (max_tokens or 500) * 4  # Approximate chars from tokens
        parts: List[str] = []
        current_chars = 0

        for msg in messages:
            role = self._get_role(msg)
            content = self._get_content(msg)

            # Skip empty messages
            if not content.strip():
                continue

            # Truncate content if needed
            if len(content) > self._max_chars:
                content = content[: self._max_chars - 3] + "..."

            if self._include_roles:
                part = f"[{role.upper()}]: {content}"
            else:
                part = content

            if current_chars + len(part) > max_chars:
                remaining = max_chars - current_chars
                if remaining > 20:
                    parts.append(part[:remaining - 3] + "...")
                break

            parts.append(part)
            current_chars += len(part) + 1  # +1 for newline

        if not parts:
            return "[No content to summarize]"

        return "\n".join(parts)


class ConversationSummarizer(BaseSummarizer):
    """LLM-powered conversation summarizer.

    Uses an LLM provider to generate intelligent summaries of
    conversations with support for:
    - Multiple summary types and prompts
    - Progressive summarization
    - Caching for efficiency
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        prompt: Optional[SummaryPrompt] = None,
        cache: Optional[SummaryCache] = None,
        enable_cache: bool = True,
        fallback_summarizer: Optional[BaseSummarizer] = None,
    ):
        """Initialize conversation summarizer.

        Args:
            llm_provider: LLM provider for generating summaries.
            prompt: Summary prompt configuration.
            cache: Summary cache instance.
            enable_cache: Whether to enable caching.
            fallback_summarizer: Fallback for when LLM is unavailable.
        """
        self._llm = llm_provider
        self._prompt = prompt or DEFAULT_PROMPTS["general"]
        self._cache = cache if enable_cache else None
        self._enable_cache = enable_cache
        self._fallback = fallback_summarizer or SimpleSummarizer()

        # Progressive summarization state
        self._progressive_state: Optional[ProgressiveSummary] = None

    @property
    def prompt(self) -> SummaryPrompt:
        """Get current prompt configuration."""
        return self._prompt

    @prompt.setter
    def prompt(self, value: SummaryPrompt) -> None:
        """Set prompt configuration."""
        self._prompt = value

    def set_prompt_preset(self, preset: str) -> None:
        """Set prompt from a preset.

        Args:
            preset: Preset name (general, technical, task_oriented, brief).

        Raises:
            ValueError: If preset not found.
        """
        if preset not in DEFAULT_PROMPTS:
            raise ValueError(
                f"Unknown preset: {preset}. "
                f"Available: {list(DEFAULT_PROMPTS.keys())}"
            )
        self._prompt = DEFAULT_PROMPTS[preset]

    def _format_conversation(self, messages: List[Any]) -> str:
        """Format messages into conversation text.

        Args:
            messages: List of messages.

        Returns:
            Formatted conversation string.
        """
        parts = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", "unknown")
                if hasattr(role, "value"):
                    role = role.value
                content = getattr(msg, "content", "")

            if content.strip():
                parts.append(f"{role.upper()}: {content}")

        return "\n\n".join(parts)

    async def summarize(
        self,
        messages: List[Any],
        max_tokens: Optional[int] = None,
    ) -> str:
        """Summarize the conversation.

        Args:
            messages: List of messages to summarize.
            max_tokens: Maximum tokens for the summary.

        Returns:
            Generated summary string.
        """
        if not messages:
            return "[Empty conversation]"

        max_tokens = max_tokens or 500

        # Check cache first
        if self._cache:
            cached = self._cache.get(messages)
            if cached:
                return cached

        # Use LLM if available
        if self._llm:
            try:
                summary = await self._generate_with_llm(messages, max_tokens)

                # Cache the result
                if self._cache:
                    self._cache.set(messages, summary)

                return summary
            except Exception:
                # Fall back to simple summarizer
                pass

        # Fallback
        return await self._fallback.summarize(messages, max_tokens)

    async def _generate_with_llm(
        self,
        messages: List[Any],
        max_tokens: int,
    ) -> str:
        """Generate summary using LLM.

        Args:
            messages: Messages to summarize.
            max_tokens: Maximum tokens for summary.

        Returns:
            LLM-generated summary.
        """
        if not self._llm:
            raise RuntimeError("No LLM provider configured")

        # Import here to avoid circular imports
        from agents_framework.llm import Message as LLMMessage, MessageRole

        conversation = self._format_conversation(messages)
        user_prompt = self._prompt.user_prompt_template.format(
            conversation=conversation,
            summary_type=self._prompt.summary_type.value,
            max_tokens=max_tokens,
        )

        llm_messages = [
            LLMMessage(role=MessageRole.SYSTEM, content=self._prompt.system_prompt),
            LLMMessage(role=MessageRole.USER, content=user_prompt),
        ]

        response = await self._llm.generate(llm_messages)
        return response.content

    async def summarize_progressively(
        self,
        messages: List[Any],
        chunk_size: int = 10,
        max_tokens_per_chunk: int = 200,
    ) -> str:
        """Progressively summarize a long conversation.

        Breaks conversation into chunks, summarizes each, then
        combines summaries for a final result.

        Args:
            messages: All messages to summarize.
            chunk_size: Number of messages per chunk.
            max_tokens_per_chunk: Max tokens per chunk summary.

        Returns:
            Final combined summary.
        """
        if not messages:
            return "[Empty conversation]"

        if len(messages) <= chunk_size:
            return await self.summarize(messages, max_tokens_per_chunk * 2)

        # Initialize progressive state
        self._progressive_state = ProgressiveSummary()

        # Process in chunks
        chunk_summaries: List[str] = []
        for i in range(0, len(messages), chunk_size):
            chunk = messages[i : i + chunk_size]
            chunk_summary = await self.summarize(chunk, max_tokens_per_chunk)
            chunk_summaries.append(chunk_summary)
            self._progressive_state.add_summary(chunk_summary, len(chunk))

        # If we have multiple chunk summaries, summarize them
        if len(chunk_summaries) > 1:
            # Create pseudo-messages from summaries
            summary_messages = [
                {"role": "summary", "content": s} for s in chunk_summaries
            ]
            final_summary = await self.summarize(
                summary_messages,
                max_tokens_per_chunk * 2,
            )
            self._progressive_state.add_summary(final_summary, 0)
            return final_summary

        return chunk_summaries[0] if chunk_summaries else "[No content]"

    async def update_progressive_summary(
        self,
        new_messages: List[Any],
        max_tokens: int = 300,
    ) -> str:
        """Update progressive summary with new messages.

        Args:
            new_messages: New messages to incorporate.
            max_tokens: Max tokens for updated summary.

        Returns:
            Updated summary.
        """
        if not self._progressive_state:
            self._progressive_state = ProgressiveSummary()

        # Summarize new messages
        new_summary = await self.summarize(new_messages, max_tokens // 2)

        if self._progressive_state.current_summary:
            # Combine with existing summary
            combined = [
                {"role": "previous_summary", "content": self._progressive_state.current_summary},
                {"role": "new_content", "content": new_summary},
            ]
            combined_summary = await self.summarize(combined, max_tokens)
            self._progressive_state.add_summary(combined_summary, len(new_messages))
            return combined_summary
        else:
            self._progressive_state.add_summary(new_summary, len(new_messages))
            return new_summary

    def get_progressive_state(self) -> Optional[ProgressiveSummary]:
        """Get current progressive summarization state."""
        return self._progressive_state

    def reset_progressive_state(self) -> None:
        """Reset progressive summarization state."""
        self._progressive_state = None

    def clear_cache(self) -> None:
        """Clear the summary cache."""
        if self._cache:
            self._cache.clear()

    async def summarize_by_topics(
        self,
        messages: List[Any],
        max_tokens: int = 500,
    ) -> Dict[str, str]:
        """Summarize conversation organized by topics.

        Args:
            messages: Messages to summarize.
            max_tokens: Maximum tokens for summary.

        Returns:
            Dictionary of topic to summary mappings.
        """
        # For now, return single topic - would need topic extraction
        summary = await self.summarize(messages, max_tokens)
        return {"main_discussion": summary}

    async def extract_key_points(
        self,
        messages: List[Any],
        max_points: int = 5,
    ) -> List[str]:
        """Extract key points from conversation.

        Args:
            messages: Messages to analyze.
            max_points: Maximum number of key points.

        Returns:
            List of key point strings.
        """
        if not self._llm:
            # Simple extraction without LLM
            points = []
            for msg in messages[:max_points * 2]:
                content = self._get_content(msg)
                if content and len(content) > 20:
                    points.append(content[:100] + "..." if len(content) > 100 else content)
                if len(points) >= max_points:
                    break
            return points

        # Use LLM for extraction
        from agents_framework.llm import Message as LLMMessage, MessageRole

        conversation = self._format_conversation(messages)
        prompt = (
            f"Extract the {max_points} most important key points from this conversation. "
            f"Return each point on a new line, prefixed with '- '.\n\n"
            f"Conversation:\n{conversation}\n\n"
            f"Key points:"
        )

        llm_messages = [
            LLMMessage(role=MessageRole.USER, content=prompt),
        ]

        response = await self._llm.generate(llm_messages)

        # Parse response
        points = []
        for line in response.content.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                points.append(line[2:])
            elif line.startswith("* "):
                points.append(line[2:])
            elif line and not line.startswith("#"):
                points.append(line)

            if len(points) >= max_points:
                break

        return points

    def _get_content(self, message: Any) -> str:
        """Extract content from message."""
        if isinstance(message, dict):
            return message.get("content", "")
        return getattr(message, "content", "")


def create_summarizer(
    llm_provider: Optional[LLMProvider] = None,
    preset: str = "general",
    enable_cache: bool = True,
) -> ConversationSummarizer:
    """Factory function to create a summarizer.

    Args:
        llm_provider: Optional LLM provider.
        preset: Prompt preset name.
        enable_cache: Whether to enable caching.

    Returns:
        Configured ConversationSummarizer instance.
    """
    prompt = DEFAULT_PROMPTS.get(preset, DEFAULT_PROMPTS["general"])
    cache = SummaryCache() if enable_cache else None

    return ConversationSummarizer(
        llm_provider=llm_provider,
        prompt=prompt,
        cache=cache,
        enable_cache=enable_cache,
    )
