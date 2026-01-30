"""Summarize skill for text summarization.

This skill provides text summarization capabilities using LLMs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agents_framework.skills.base import (
    BaseSkill,
    SkillCategory,
    SkillContext,
    SkillMetadata,
)


@dataclass
class SummarizeOptions:
    """Options for text summarization.

    Attributes:
        max_length: Maximum length of the summary.
        style: Summarization style (brief, detailed, bullet_points).
        focus: Optional focus area for the summary.
        language: Output language (default: same as input).
    """

    max_length: int = 150
    style: str = "brief"
    focus: Optional[str] = None
    language: Optional[str] = None


class SummarizeSkill(BaseSkill):
    """Skill for summarizing text content.

    Uses an LLM to generate concise summaries of input text.
    Supports various summarization styles and options.
    """

    def __init__(self):
        """Initialize the summarize skill."""
        super().__init__(
            metadata=SkillMetadata(
                name="summarize",
                description="Summarize text content into a concise version",
                category=SkillCategory.TEXT,
                tags=["text", "summarization", "nlp"],
                requires_llm=True,
            )
        )

    async def execute(
        self,
        context: SkillContext,
        text: str,
        max_length: int = 150,
        style: str = "brief",
        focus: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        """Summarize the given text.

        Args:
            context: Execution context with LLM provider.
            text: Text content to summarize.
            max_length: Maximum length of the summary in words.
            style: Summarization style (brief, detailed, bullet_points).
            focus: Optional focus area for the summary.
            language: Output language (default: same as input).

        Returns:
            Summarized text.

        Raises:
            ValueError: If LLM is not available.
        """
        if context.llm is None:
            raise ValueError("LLM provider is required for summarization")

        # Build the prompt based on options
        prompt = self._build_prompt(
            text=text,
            max_length=max_length,
            style=style,
            focus=focus,
            language=language,
        )

        # Import LLM types
        from agents_framework.llm import Message, MessageRole

        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content="You are a helpful assistant skilled at summarizing text content.",
            ),
            Message(
                role=MessageRole.USER,
                content=prompt,
            ),
        ]

        response = await context.llm.generate(messages)
        return response.content.strip()

    def _build_prompt(
        self,
        text: str,
        max_length: int,
        style: str,
        focus: Optional[str],
        language: Optional[str],
    ) -> str:
        """Build the summarization prompt."""
        style_instructions = {
            "brief": "Provide a brief, concise summary.",
            "detailed": "Provide a detailed summary covering key points.",
            "bullet_points": "Summarize as a list of bullet points.",
        }

        style_instruction = style_instructions.get(
            style,
            style_instructions["brief"],
        )

        prompt_parts = [
            f"Summarize the following text in approximately {max_length} words.",
            style_instruction,
        ]

        if focus:
            prompt_parts.append(f"Focus on: {focus}")

        if language:
            prompt_parts.append(f"Write the summary in {language}.")

        prompt_parts.extend([
            "",
            "Text to summarize:",
            "---",
            text,
            "---",
            "",
            "Summary:",
        ])

        return "\n".join(prompt_parts)

    async def summarize_multiple(
        self,
        context: SkillContext,
        texts: List[str],
        **options: Any,
    ) -> List[str]:
        """Summarize multiple texts.

        Args:
            context: Execution context.
            texts: List of texts to summarize.
            **options: Summarization options.

        Returns:
            List of summaries.
        """
        summaries = []
        for text in texts:
            summary = await self.execute(context, text=text, **options)
            summaries.append(summary)
        return summaries
