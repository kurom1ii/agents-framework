"""Search skill for information retrieval.

This skill provides search capabilities for finding information
from various sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from agents_framework.skills.base import (
    BaseSkill,
    SkillCategory,
    SkillContext,
    SkillMetadata,
)


class SearchSource(str, Enum):
    """Available search sources."""

    MEMORY = "memory"
    TOOLS = "tools"
    WEB = "web"
    ALL = "all"


@dataclass
class SearchResult:
    """A single search result.

    Attributes:
        content: The result content.
        source: Source of the result.
        score: Relevance score (0-1).
        metadata: Additional metadata.
    """

    content: str
    source: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SearchSkill(BaseSkill):
    """Skill for searching and retrieving information.

    Searches across available sources (memory, tools, web) to find
    relevant information based on a query.
    """

    def __init__(self):
        """Initialize the search skill."""
        super().__init__(
            metadata=SkillMetadata(
                name="search",
                description="Search for information across available sources",
                category=SkillCategory.SEARCH,
                tags=["search", "retrieval", "information"],
                requires_llm=False,
            )
        )

    async def execute(
        self,
        context: SkillContext,
        query: str,
        source: str = "all",
        max_results: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search for information matching the query.

        Args:
            context: Execution context.
            query: Search query string.
            source: Source to search (memory, tools, web, all).
            max_results: Maximum number of results to return.
            min_score: Minimum relevance score threshold.

        Returns:
            List of search results as dictionaries.
        """
        results: List[SearchResult] = []

        # Search based on source
        if source in ("memory", "all") and context.memory is not None:
            memory_results = await self._search_memory(
                context, query, max_results
            )
            results.extend(memory_results)

        if source in ("tools", "all") and context.tools is not None:
            tool_results = self._search_tools(context, query, max_results)
            results.extend(tool_results)

        # Filter by min_score and sort by score
        results = [r for r in results if r.score >= min_score]
        results.sort(key=lambda r: r.score, reverse=True)

        # Limit results
        results = results[:max_results]

        # Convert to dictionaries
        return [
            {
                "content": r.content,
                "source": r.source,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in results
        ]

    async def _search_memory(
        self,
        context: SkillContext,
        query: str,
        limit: int,
    ) -> List[SearchResult]:
        """Search in memory store.

        Args:
            context: Execution context.
            query: Search query.
            limit: Maximum results.

        Returns:
            List of SearchResult from memory.
        """
        if context.memory is None:
            return []

        results = []

        try:
            # Import memory types
            from agents_framework.memory import MemoryQuery

            memory_query = MemoryQuery(
                query_text=query,
                limit=limit,
            )

            items = await context.memory.retrieve(memory_query)

            for item in items:
                results.append(
                    SearchResult(
                        content=item.content,
                        source="memory",
                        score=0.8,  # Default score for memory matches
                        metadata={
                            "id": item.id,
                            "type": item.memory_type.value if hasattr(item.memory_type, 'value') else str(item.memory_type),
                            "timestamp": item.timestamp.isoformat(),
                        },
                    )
                )
        except Exception as e:
            # Log error but continue
            pass

        return results

    def _search_tools(
        self,
        context: SkillContext,
        query: str,
        limit: int,
    ) -> List[SearchResult]:
        """Search available tools by name and description.

        Args:
            context: Execution context.
            query: Search query.
            limit: Maximum results.

        Returns:
            List of SearchResult for matching tools.
        """
        if context.tools is None:
            return []

        results = []
        query_lower = query.lower()

        try:
            tools = context.tools.list_tools()

            for tool in tools:
                # Calculate simple score based on query match
                score = 0.0
                name_lower = tool.name.lower()
                desc_lower = tool.description.lower()

                if query_lower in name_lower:
                    score = 0.9
                elif query_lower in desc_lower:
                    score = 0.7
                elif any(word in name_lower or word in desc_lower
                        for word in query_lower.split()):
                    score = 0.5

                if score > 0:
                    results.append(
                        SearchResult(
                            content=f"{tool.name}: {tool.description}",
                            source="tools",
                            score=score,
                            metadata={
                                "name": tool.name,
                                "description": tool.description,
                            },
                        )
                    )

            # Sort and limit
            results.sort(key=lambda r: r.score, reverse=True)
            results = results[:limit]

        except Exception:
            pass

        return results

    async def search_with_llm(
        self,
        context: SkillContext,
        query: str,
        search_results: List[Dict[str, Any]],
    ) -> str:
        """Use LLM to synthesize search results into an answer.

        Args:
            context: Execution context with LLM.
            query: Original search query.
            search_results: Results from execute().

        Returns:
            Synthesized answer from the search results.
        """
        if context.llm is None:
            raise ValueError("LLM provider is required for synthesis")

        # Format search results for the prompt
        formatted_results = []
        for i, result in enumerate(search_results, 1):
            formatted_results.append(
                f"{i}. [{result['source']}] {result['content']}"
            )

        results_text = "\n".join(formatted_results) if formatted_results else "No results found."

        from agents_framework.llm import Message, MessageRole

        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content="You are a helpful assistant. Use the provided search results to answer the user's question. If the results don't contain relevant information, say so.",
            ),
            Message(
                role=MessageRole.USER,
                content=f"Question: {query}\n\nSearch Results:\n{results_text}\n\nPlease provide a helpful answer based on these results.",
            ),
        ]

        response = await context.llm.generate(messages)
        return response.content.strip()
