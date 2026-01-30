"""Unified memory manager for agents.

This module provides the MemoryManager class that unifies all memory types
(short-term, long-term, and vector) with tiered memory strategy and
namespace isolation per agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel

from .base import MemoryConfig, MemoryItem, MemoryQuery, MemoryStore, MemoryType, Message
from .short_term import SessionMemory, SessionMemoryConfig, NamespacedSessionStore

if TYPE_CHECKING:
    from .embeddings import BaseEmbeddingProvider
    from .long_term import RedisMemoryStore, VectorMemoryStore


class TierPriority(str, Enum):
    """Memory tier priority for retrieval."""

    SHORT_TERM_FIRST = "short_term_first"
    LONG_TERM_FIRST = "long_term_first"
    VECTOR_FIRST = "vector_first"
    PARALLEL = "parallel"


class MemoryTier(str, Enum):
    """Memory storage tiers."""

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    VECTOR = "vector"


class MemoryManagerConfig(BaseModel):
    """Configuration for the memory manager.

    Attributes:
        agent_id: The agent this manager belongs to.
        default_namespace: Default namespace for memory operations.
        short_term_enabled: Whether short-term memory is enabled.
        long_term_enabled: Whether long-term memory is enabled.
        vector_enabled: Whether vector memory is enabled.
        tier_priority: Priority order for memory retrieval.
        auto_promote: Whether to auto-promote important memories to long-term.
        promotion_threshold: Number of accesses before promoting to long-term.
        max_short_term_tokens: Maximum tokens for short-term memory.
        enable_deduplication: Whether to deduplicate memories.
    """

    agent_id: str
    default_namespace: Optional[str] = None
    short_term_enabled: bool = True
    long_term_enabled: bool = False
    vector_enabled: bool = False
    tier_priority: TierPriority = TierPriority.SHORT_TERM_FIRST
    auto_promote: bool = True
    promotion_threshold: int = 3
    max_short_term_tokens: int = 8192
    enable_deduplication: bool = True


@dataclass
class MemorySearchResult:
    """Result from a memory search operation.

    Attributes:
        item: The memory item.
        tier: The tier where the item was found.
        score: Optional relevance/similarity score.
        access_count: Number of times this item has been accessed.
    """

    item: MemoryItem
    tier: MemoryTier
    score: Optional[float] = None
    access_count: int = 0


class MemoryManager:
    """Unified memory manager for agents.

    Provides a unified interface for managing multiple memory tiers
    (short-term, long-term, vector) with automatic tiering, promotion,
    and namespace isolation.

    Attributes:
        config: Memory manager configuration.
    """

    def __init__(
        self,
        config: MemoryManagerConfig,
        short_term_store: Optional[SessionMemory] = None,
        long_term_store: Optional[MemoryStore] = None,
        vector_store: Optional[MemoryStore] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
    ):
        """Initialize the memory manager.

        Args:
            config: Memory manager configuration.
            short_term_store: Optional short-term memory store.
            long_term_store: Optional long-term memory store.
            vector_store: Optional vector memory store.
            embedding_provider: Optional embedding provider for vector operations.
        """
        self.config = config
        self._namespace = config.default_namespace or f"agent:{config.agent_id}"

        # Initialize stores
        self._short_term = short_term_store
        self._long_term = long_term_store
        self._vector = vector_store
        self._embedding_provider = embedding_provider

        # Access tracking for promotion
        self._access_counts: Dict[str, int] = {}

        # Initialize default short-term store if enabled and not provided
        if config.short_term_enabled and self._short_term is None:
            session_config = SessionMemoryConfig(
                max_tokens=config.max_short_term_tokens,
                agent_id=config.agent_id,
            )
            self._short_term = SessionMemory(session_config)

    @property
    def namespace(self) -> str:
        """Get the namespace for this manager."""
        return self._namespace

    @property
    def short_term(self) -> Optional[SessionMemory]:
        """Get the short-term memory store."""
        return self._short_term

    @property
    def long_term(self) -> Optional[MemoryStore]:
        """Get the long-term memory store."""
        return self._long_term

    @property
    def vector(self) -> Optional[MemoryStore]:
        """Get the vector memory store."""
        return self._vector

    def set_short_term(self, store: SessionMemory) -> None:
        """Set the short-term memory store.

        Args:
            store: The SessionMemory instance.
        """
        self._short_term = store

    def set_long_term(self, store: MemoryStore) -> None:
        """Set the long-term memory store.

        Args:
            store: The long-term MemoryStore instance.
        """
        self._long_term = store

    def set_vector(self, store: MemoryStore) -> None:
        """Set the vector memory store.

        Args:
            store: The vector MemoryStore instance.
        """
        self._vector = store

    async def store(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        tier: MemoryTier = MemoryTier.SHORT_TERM,
        embedding: Optional[List[float]] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """Store a memory item.

        Args:
            content: The content to store.
            metadata: Optional metadata.
            tier: Which tier to store in.
            embedding: Optional pre-computed embedding.
            ttl: Optional time-to-live in seconds.

        Returns:
            The ID of the stored item.
        """
        item = MemoryItem(
            content=content,
            metadata=metadata or {},
            namespace=self._namespace,
            embedding=embedding,
            ttl=ttl,
        )

        if tier == MemoryTier.SHORT_TERM and self._short_term:
            item.memory_type = MemoryType.SHORT_TERM
            return await self._short_term.store(item)

        elif tier == MemoryTier.LONG_TERM and self._long_term:
            item.memory_type = MemoryType.LONG_TERM
            return await self._long_term.store(item)

        elif tier == MemoryTier.VECTOR and self._vector:
            item.memory_type = MemoryType.VECTOR
            # Generate embedding if needed
            if item.embedding is None and self._embedding_provider:
                item.embedding = await self._embedding_provider.embed(content)
            return await self._vector.store(item)

        raise ValueError(f"Memory tier {tier} is not available")

    async def store_message(self, message: Message) -> List[Message]:
        """Store a conversation message in short-term memory.

        Args:
            message: The message to store.

        Returns:
            List of evicted messages (if any).
        """
        if self._short_term is None:
            raise ValueError("Short-term memory is not available")

        return self._short_term.add_message(message)

    async def get_messages(
        self,
        include_system: bool = True,
        max_tokens: Optional[int] = None,
    ) -> List[Message]:
        """Get conversation messages from short-term memory.

        Args:
            include_system: Whether to include system messages.
            max_tokens: Optional token limit.

        Returns:
            List of messages.
        """
        if self._short_term is None:
            return []

        return self._short_term.get_messages(
            include_system=include_system,
            max_tokens=max_tokens,
        )

    async def retrieve(
        self,
        query: Optional[str] = None,
        limit: int = 10,
        tiers: Optional[List[MemoryTier]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.7,
    ) -> List[MemorySearchResult]:
        """Retrieve memories from specified tiers.

        Args:
            query: Optional search query.
            limit: Maximum results per tier.
            tiers: Tiers to search (defaults based on priority).
            metadata_filters: Optional metadata filters.
            similarity_threshold: Minimum similarity for vector search.

        Returns:
            List of MemorySearchResult objects.
        """
        if tiers is None:
            tiers = self._get_tier_order()

        results: List[MemorySearchResult] = []

        memory_query = MemoryQuery(
            query_text=query,
            namespace=self._namespace,
            metadata_filters=metadata_filters or {},
            limit=limit,
            similarity_threshold=similarity_threshold,
        )

        for tier in tiers:
            tier_results = await self._retrieve_from_tier(tier, memory_query)
            results.extend(tier_results)

            # Track access for promotion
            for result in tier_results:
                self._track_access(result.item.id)

        # Sort by score if available
        results.sort(key=lambda r: r.score or 0, reverse=True)

        return results[:limit]

    async def _retrieve_from_tier(
        self,
        tier: MemoryTier,
        query: MemoryQuery,
    ) -> List[MemorySearchResult]:
        """Retrieve memories from a specific tier.

        Args:
            tier: The tier to search.
            query: The memory query.

        Returns:
            List of results from this tier.
        """
        results: List[MemorySearchResult] = []

        if tier == MemoryTier.SHORT_TERM and self._short_term:
            items = await self._short_term.retrieve(query)
            for item in items:
                results.append(MemorySearchResult(
                    item=item,
                    tier=MemoryTier.SHORT_TERM,
                    access_count=self._access_counts.get(item.id, 0),
                ))

        elif tier == MemoryTier.LONG_TERM and self._long_term:
            items = await self._long_term.retrieve(query)
            for item in items:
                results.append(MemorySearchResult(
                    item=item,
                    tier=MemoryTier.LONG_TERM,
                    access_count=self._access_counts.get(item.id, 0),
                ))

        elif tier == MemoryTier.VECTOR and self._vector:
            # Use vector search for semantic queries
            items = await self._vector.retrieve(query)
            for item in items:
                # Calculate a basic score based on position
                results.append(MemorySearchResult(
                    item=item,
                    tier=MemoryTier.VECTOR,
                    score=1.0,  # Vector store should set proper score
                    access_count=self._access_counts.get(item.id, 0),
                ))

        return results

    def _get_tier_order(self) -> List[MemoryTier]:
        """Get tier search order based on priority.

        Returns:
            List of tiers in priority order.
        """
        available_tiers = []

        if self.config.short_term_enabled and self._short_term:
            available_tiers.append(MemoryTier.SHORT_TERM)
        if self.config.long_term_enabled and self._long_term:
            available_tiers.append(MemoryTier.LONG_TERM)
        if self.config.vector_enabled and self._vector:
            available_tiers.append(MemoryTier.VECTOR)

        # Reorder based on priority
        if self.config.tier_priority == TierPriority.LONG_TERM_FIRST:
            if MemoryTier.LONG_TERM in available_tiers:
                available_tiers.remove(MemoryTier.LONG_TERM)
                available_tiers.insert(0, MemoryTier.LONG_TERM)

        elif self.config.tier_priority == TierPriority.VECTOR_FIRST:
            if MemoryTier.VECTOR in available_tiers:
                available_tiers.remove(MemoryTier.VECTOR)
                available_tiers.insert(0, MemoryTier.VECTOR)

        return available_tiers

    def _track_access(self, item_id: str) -> None:
        """Track access to a memory item.

        Args:
            item_id: The item ID.
        """
        self._access_counts[item_id] = self._access_counts.get(item_id, 0) + 1

    async def promote(self, item_id: str, from_tier: MemoryTier, to_tier: MemoryTier) -> bool:
        """Promote a memory item to a higher tier.

        Args:
            item_id: The item ID to promote.
            from_tier: Current tier.
            to_tier: Target tier.

        Returns:
            True if promotion was successful.
        """
        # Get item from source tier
        source_store = self._get_store(from_tier)
        target_store = self._get_store(to_tier)

        if source_store is None or target_store is None:
            return False

        item = await source_store.get(item_id)
        if item is None:
            return False

        # Update memory type
        if to_tier == MemoryTier.LONG_TERM:
            item.memory_type = MemoryType.LONG_TERM
        elif to_tier == MemoryTier.VECTOR:
            item.memory_type = MemoryType.VECTOR
            # Generate embedding if needed
            if item.embedding is None and self._embedding_provider:
                item.embedding = await self._embedding_provider.embed(item.content)

        # Store in target tier
        await target_store.store(item)

        return True

    async def check_and_promote(self) -> List[str]:
        """Check items for automatic promotion.

        Returns:
            List of promoted item IDs.
        """
        if not self.config.auto_promote:
            return []

        promoted = []

        for item_id, count in list(self._access_counts.items()):
            if count >= self.config.promotion_threshold:
                # Try to promote from short-term to long-term
                if self._short_term and self._long_term:
                    success = await self.promote(
                        item_id,
                        MemoryTier.SHORT_TERM,
                        MemoryTier.LONG_TERM,
                    )
                    if success:
                        promoted.append(item_id)
                        # Reset access count
                        del self._access_counts[item_id]

        return promoted

    def _get_store(self, tier: MemoryTier) -> Optional[MemoryStore]:
        """Get the store for a specific tier.

        Args:
            tier: The memory tier.

        Returns:
            The MemoryStore for that tier, or None.
        """
        if tier == MemoryTier.SHORT_TERM:
            return self._short_term
        elif tier == MemoryTier.LONG_TERM:
            return self._long_term
        elif tier == MemoryTier.VECTOR:
            return self._vector
        return None

    async def delete(self, item_id: str, tier: Optional[MemoryTier] = None) -> bool:
        """Delete a memory item.

        Args:
            item_id: The item ID to delete.
            tier: Optional specific tier (searches all if None).

        Returns:
            True if item was deleted.
        """
        if tier is not None:
            store = self._get_store(tier)
            if store:
                return await store.delete(item_id)
            return False

        # Search all tiers
        for t in MemoryTier:
            store = self._get_store(t)
            if store:
                if await store.delete(item_id):
                    return True

        return False

    async def clear(self, tier: Optional[MemoryTier] = None) -> None:
        """Clear memories.

        Args:
            tier: Optional specific tier (clears all if None).
        """
        if tier is not None:
            store = self._get_store(tier)
            if store:
                await store.clear(self._namespace)
            return

        # Clear all tiers
        for t in MemoryTier:
            store = self._get_store(t)
            if store:
                await store.clear(self._namespace)

        # Clear access counts
        self._access_counts.clear()

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dictionary with memory statistics.
        """
        stats: Dict[str, Any] = {
            "agent_id": self.config.agent_id,
            "namespace": self._namespace,
            "tiers": {},
        }

        if self._short_term:
            token_usage = self._short_term.get_token_usage()
            count = await self._short_term.count(self._namespace)
            stats["tiers"]["short_term"] = {
                "enabled": self.config.short_term_enabled,
                "count": count,
                "token_usage": token_usage,
            }

        if self._long_term:
            count = await self._long_term.count(self._namespace)
            stats["tiers"]["long_term"] = {
                "enabled": self.config.long_term_enabled,
                "count": count,
            }

        if self._vector:
            count = await self._vector.count(self._namespace)
            stats["tiers"]["vector"] = {
                "enabled": self.config.vector_enabled,
                "count": count,
            }

        stats["access_tracking"] = {
            "tracked_items": len(self._access_counts),
            "promotion_threshold": self.config.promotion_threshold,
        }

        return stats

    async def search_semantic(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> List[MemorySearchResult]:
        """Perform semantic search using vector memory.

        Args:
            query: The search query.
            limit: Maximum results.
            threshold: Minimum similarity threshold.

        Returns:
            List of relevant memories with scores.
        """
        if not self._vector or not self.config.vector_enabled:
            raise ValueError("Vector memory is not available")

        # Use vector store's search capability
        memory_query = MemoryQuery(
            query_text=query,
            namespace=self._namespace,
            limit=limit,
            similarity_threshold=threshold,
        )

        items = await self._vector.retrieve(memory_query)

        return [
            MemorySearchResult(
                item=item,
                tier=MemoryTier.VECTOR,
                score=0.0,  # Score should be set by vector store
            )
            for item in items
        ]

    async def consolidate(self) -> Dict[str, int]:
        """Consolidate short-term memories to long-term.

        Moves frequently accessed short-term memories to long-term storage.

        Returns:
            Statistics about the consolidation.
        """
        stats = {"promoted": 0, "deduplicated": 0}

        # Promote based on access counts
        promoted = await self.check_and_promote()
        stats["promoted"] = len(promoted)

        # TODO: Add deduplication logic if enabled
        if self.config.enable_deduplication:
            pass  # Future enhancement

        return stats


class MemoryManagerRegistry:
    """Registry for managing multiple memory managers.

    Provides centralized management of memory managers for different agents.
    """

    def __init__(self):
        """Initialize the registry."""
        self._managers: Dict[str, MemoryManager] = {}
        self._session_store = NamespacedSessionStore()

    def get_or_create(
        self,
        agent_id: str,
        config: Optional[MemoryManagerConfig] = None,
        **kwargs,
    ) -> MemoryManager:
        """Get or create a memory manager for an agent.

        Args:
            agent_id: The agent ID.
            config: Optional configuration.
            **kwargs: Additional arguments for MemoryManager.

        Returns:
            The MemoryManager instance.
        """
        if agent_id not in self._managers:
            if config is None:
                config = MemoryManagerConfig(agent_id=agent_id)
            else:
                config.agent_id = agent_id

            self._managers[agent_id] = MemoryManager(config, **kwargs)

        return self._managers[agent_id]

    def get(self, agent_id: str) -> Optional[MemoryManager]:
        """Get a memory manager by agent ID.

        Args:
            agent_id: The agent ID.

        Returns:
            The MemoryManager if found, None otherwise.
        """
        return self._managers.get(agent_id)

    def remove(self, agent_id: str) -> bool:
        """Remove a memory manager.

        Args:
            agent_id: The agent ID.

        Returns:
            True if removed, False if not found.
        """
        if agent_id in self._managers:
            del self._managers[agent_id]
            return True
        return False

    async def clear_all(self) -> None:
        """Clear all memory managers and their data."""
        for manager in self._managers.values():
            await manager.clear()
        self._managers.clear()

    def list_agents(self) -> List[str]:
        """List all agent IDs with memory managers.

        Returns:
            List of agent IDs.
        """
        return list(self._managers.keys())
