"""Vector memory store with ChromaDB backend.

This module provides a vector-based memory store implementation using
ChromaDB for semantic similarity search with metadata filtering support.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import uuid4

from ..base import MemoryConfig, MemoryItem, MemoryQuery, MemoryStore, MemoryType

if TYPE_CHECKING:
    from ..embeddings import BaseEmbeddingProvider

# Optional ChromaDB import
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    Settings = None


class VectorMemoryConfig(MemoryConfig):
    """Configuration for vector memory store.

    Attributes:
        collection_name: Name of the ChromaDB collection.
        persist_directory: Directory for persistent storage.
        distance_metric: Distance metric for similarity search.
        host: ChromaDB server host (for client mode).
        port: ChromaDB server port (for client mode).
        use_server: Whether to connect to a ChromaDB server.
    """

    collection_name: str = "agent_memories"
    persist_directory: Optional[str] = None
    distance_metric: str = "cosine"
    host: Optional[str] = None
    port: int = 8000
    use_server: bool = False


class VectorMemoryStoreError(Exception):
    """Exception for vector memory store errors."""
    pass


class VectorMemoryStore(MemoryStore):
    """Vector-based memory store using ChromaDB.

    Provides semantic similarity search for memory items with support
    for metadata filtering and namespace isolation.

    Attributes:
        config: Vector memory configuration.
    """

    def __init__(
        self,
        config: Optional[VectorMemoryConfig] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
    ):
        """Initialize vector memory store.

        Args:
            config: Optional vector memory configuration.
            embedding_provider: Optional embedding provider for generating embeddings.

        Raises:
            ImportError: If chromadb package is not installed.
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb package is required for VectorMemoryStore. "
                "Install it with: pip install chromadb"
            )

        self.config = config or VectorMemoryConfig()
        self._embedding_provider = embedding_provider
        self._client: Optional[chromadb.Client] = None
        self._collection: Optional[chromadb.Collection] = None

    def _init_client(self) -> None:
        """Initialize ChromaDB client."""
        if self._client is not None:
            return

        if self.config.use_server and self.config.host:
            # Connect to ChromaDB server
            self._client = chromadb.HttpClient(
                host=self.config.host,
                port=self.config.port,
            )
        elif self.config.persist_directory:
            # Persistent local storage
            self._client = chromadb.PersistentClient(
                path=self.config.persist_directory,
            )
        else:
            # In-memory storage
            self._client = chromadb.Client()

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.distance_metric},
        )

    @property
    def collection(self) -> chromadb.Collection:
        """Get the ChromaDB collection.

        Returns:
            The ChromaDB collection.
        """
        self._init_client()
        return self._collection  # type: ignore

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        if self._embedding_provider:
            return await self._embedding_provider.embed(text)
        raise VectorMemoryStoreError(
            "No embedding provider configured. "
            "Either provide embeddings or set an embedding provider."
        )

    async def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if self._embedding_provider:
            return await self._embedding_provider.embed_batch(texts)
        raise VectorMemoryStoreError(
            "No embedding provider configured. "
            "Either provide embeddings or set an embedding provider."
        )

    def _prepare_metadata(self, item: MemoryItem) -> Dict[str, Any]:
        """Prepare metadata for ChromaDB storage.

        ChromaDB requires flat metadata with string, int, float, or bool values.

        Args:
            item: The memory item.

        Returns:
            Flattened metadata dictionary.
        """
        metadata: Dict[str, Any] = {
            "namespace": item.namespace or self.config.namespace,
            "memory_type": str(item.memory_type),
            "timestamp": item.timestamp.isoformat(),
        }

        if item.ttl:
            metadata["ttl"] = item.ttl

        # Flatten item metadata
        for key, value in item.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                metadata[f"meta_{key}"] = value
            elif value is not None:
                # Convert other types to string
                metadata[f"meta_{key}"] = str(value)

        return metadata

    def _parse_metadata(self, metadata: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Parse ChromaDB metadata back to namespace and item metadata.

        Args:
            metadata: ChromaDB metadata.

        Returns:
            Tuple of (namespace, item_metadata).
        """
        namespace = metadata.get("namespace", self.config.namespace)
        item_metadata: Dict[str, Any] = {}

        for key, value in metadata.items():
            if key.startswith("meta_"):
                item_metadata[key[5:]] = value

        return namespace, item_metadata

    async def store(self, item: MemoryItem) -> str:
        """Store a memory item with vector embedding.

        Args:
            item: The MemoryItem to store.

        Returns:
            The ID of the stored item.
        """
        self._init_client()

        # Set namespace and type
        item.namespace = item.namespace or self.config.namespace
        item.memory_type = MemoryType.VECTOR

        # Get or generate embedding
        if item.embedding is None:
            item.embedding = await self._get_embedding(item.content)

        # Prepare metadata
        metadata = self._prepare_metadata(item)

        # Store in ChromaDB
        self.collection.add(
            ids=[item.id],
            embeddings=[item.embedding],
            documents=[item.content],
            metadatas=[metadata],
        )

        return item.id

    async def store_batch(self, items: List[MemoryItem]) -> List[str]:
        """Store multiple memory items.

        Args:
            items: List of MemoryItems to store.

        Returns:
            List of stored item IDs.
        """
        if not items:
            return []

        self._init_client()

        # Get embeddings for items without them
        texts_to_embed = []
        embed_indices = []
        for i, item in enumerate(items):
            if item.embedding is None:
                texts_to_embed.append(item.content)
                embed_indices.append(i)

        if texts_to_embed:
            embeddings = await self._get_embeddings_batch(texts_to_embed)
            for i, embedding in zip(embed_indices, embeddings):
                items[i].embedding = embedding

        # Prepare data for batch insert
        ids = []
        embeddings_list = []
        documents = []
        metadatas = []

        for item in items:
            item.namespace = item.namespace or self.config.namespace
            item.memory_type = MemoryType.VECTOR

            ids.append(item.id)
            embeddings_list.append(item.embedding)
            documents.append(item.content)
            metadatas.append(self._prepare_metadata(item))

        # Batch insert
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas,
        )

        return ids

    async def retrieve(self, query: MemoryQuery) -> List[MemoryItem]:
        """Retrieve memory items matching the query.

        Uses semantic similarity search when query_text is provided.

        Args:
            query: The MemoryQuery specifying search criteria.

        Returns:
            List of matching MemoryItem objects.
        """
        self._init_client()

        # Build where clause for filtering
        where: Optional[Dict[str, Any]] = None
        where_clauses = []

        namespace = query.namespace or self.config.namespace
        if namespace:
            where_clauses.append({"namespace": {"$eq": namespace}})

        if query.metadata_filters:
            for key, value in query.metadata_filters.items():
                where_clauses.append({f"meta_{key}": {"$eq": value}})

        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        # Perform query
        if query.query_text:
            # Semantic search
            query_embedding = await self._get_embedding(query.query_text)

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=query.limit + query.offset,
                where=where,
                include=["embeddings", "documents", "metadatas", "distances"],
            )
        else:
            # Get all matching items
            results = self.collection.get(
                where=where,
                limit=query.limit + query.offset,
                include=["embeddings", "documents", "metadatas"],
            )

        # Convert results to MemoryItems
        items: List[MemoryItem] = []

        if "ids" in results and results["ids"]:
            # Handle query results (nested lists)
            if isinstance(results["ids"][0], list):
                ids = results["ids"][0]
                documents = results["documents"][0] if results.get("documents") else []
                embeddings = results["embeddings"][0] if results.get("embeddings") else []
                metadatas = results["metadatas"][0] if results.get("metadatas") else []
                distances = results.get("distances", [[]])[0]
            else:
                ids = results["ids"]
                documents = results.get("documents", [])
                embeddings = results.get("embeddings", [])
                metadatas = results.get("metadatas", [])
                distances = results.get("distances", [])

            for i, item_id in enumerate(ids):
                # Apply similarity threshold
                if distances and i < len(distances):
                    # Convert distance to similarity (1 - distance for cosine)
                    similarity = 1 - distances[i]
                    if similarity < query.similarity_threshold:
                        continue

                namespace, item_metadata = self._parse_metadata(
                    metadatas[i] if i < len(metadatas) else {}
                )

                # Parse timestamp
                timestamp_str = metadatas[i].get("timestamp") if i < len(metadatas) else None
                timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.utcnow()

                item = MemoryItem(
                    id=item_id,
                    content=documents[i] if i < len(documents) else "",
                    embedding=embeddings[i] if i < len(embeddings) else None,
                    metadata=item_metadata,
                    namespace=namespace,
                    memory_type=MemoryType.VECTOR,
                    timestamp=timestamp,
                )
                items.append(item)

        # Apply offset
        return items[query.offset : query.offset + query.limit]

    async def search(
        self,
        query_text: str,
        namespace: Optional[str] = None,
        limit: int = 10,
        threshold: float = 0.7,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[MemoryItem, float]]:
        """Search for similar memories with similarity scores.

        Args:
            query_text: Text to search for.
            namespace: Optional namespace filter.
            limit: Maximum results to return.
            threshold: Minimum similarity threshold.
            metadata_filters: Optional metadata filters.

        Returns:
            List of (MemoryItem, similarity_score) tuples.
        """
        self._init_client()

        # Build where clause
        where: Optional[Dict[str, Any]] = None
        where_clauses = []

        ns = namespace or self.config.namespace
        if ns:
            where_clauses.append({"namespace": {"$eq": ns}})

        if metadata_filters:
            for key, value in metadata_filters.items():
                where_clauses.append({f"meta_{key}": {"$eq": value}})

        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        # Get query embedding
        query_embedding = await self._get_embedding(query_text)

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit * 2,  # Fetch extra for filtering
            where=where,
            include=["embeddings", "documents", "metadatas", "distances"],
        )

        # Convert to items with scores
        scored_items: List[tuple[MemoryItem, float]] = []

        if results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0] if results.get("documents") else []
            embeddings = results["embeddings"][0] if results.get("embeddings") else []
            metadatas = results["metadatas"][0] if results.get("metadatas") else []
            distances = results["distances"][0] if results.get("distances") else []

            for i, item_id in enumerate(ids):
                # Calculate similarity from distance
                similarity = 1 - distances[i] if i < len(distances) else 0

                if similarity < threshold:
                    continue

                namespace_parsed, item_metadata = self._parse_metadata(
                    metadatas[i] if i < len(metadatas) else {}
                )

                timestamp_str = metadatas[i].get("timestamp") if i < len(metadatas) else None
                timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.utcnow()

                item = MemoryItem(
                    id=item_id,
                    content=documents[i] if i < len(documents) else "",
                    embedding=embeddings[i] if i < len(embeddings) else None,
                    metadata=item_metadata,
                    namespace=namespace_parsed,
                    memory_type=MemoryType.VECTOR,
                    timestamp=timestamp,
                )
                scored_items.append((item, similarity))

                if len(scored_items) >= limit:
                    break

        return scored_items

    async def delete(self, item_id: str) -> bool:
        """Delete a memory item by ID.

        Args:
            item_id: The ID of the item to delete.

        Returns:
            True if the item was deleted, False if not found.
        """
        self._init_client()

        try:
            # Check if item exists
            existing = self.collection.get(ids=[item_id])
            if not existing["ids"]:
                return False

            self.collection.delete(ids=[item_id])
            return True
        except Exception:
            return False

    async def clear(self, namespace: Optional[str] = None) -> None:
        """Clear all memory items.

        Args:
            namespace: Optional namespace to clear. If None, clears all items.
        """
        self._init_client()

        ns = namespace or self.config.namespace

        if ns:
            # Delete by namespace filter
            self.collection.delete(
                where={"namespace": {"$eq": ns}}
            )
        else:
            # Delete entire collection and recreate
            if self._client:
                self._client.delete_collection(self.config.collection_name)
                self._collection = self._client.create_collection(
                    name=self.config.collection_name,
                    metadata={"hnsw:space": self.config.distance_metric},
                )

    async def get(self, item_id: str) -> Optional[MemoryItem]:
        """Get a specific memory item by ID.

        Args:
            item_id: The ID of the item to retrieve.

        Returns:
            The MemoryItem if found, None otherwise.
        """
        self._init_client()

        try:
            results = self.collection.get(
                ids=[item_id],
                include=["embeddings", "documents", "metadatas"],
            )

            if not results["ids"]:
                return None

            namespace, item_metadata = self._parse_metadata(
                results["metadatas"][0] if results.get("metadatas") else {}
            )

            timestamp_str = results["metadatas"][0].get("timestamp") if results.get("metadatas") else None
            timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.utcnow()

            return MemoryItem(
                id=results["ids"][0],
                content=results["documents"][0] if results.get("documents") else "",
                embedding=results["embeddings"][0] if results.get("embeddings") else None,
                metadata=item_metadata,
                namespace=namespace,
                memory_type=MemoryType.VECTOR,
                timestamp=timestamp,
            )
        except Exception:
            return None

    async def count(self, namespace: Optional[str] = None) -> int:
        """Count the number of stored items.

        Args:
            namespace: Optional namespace to count items in.

        Returns:
            The number of items.
        """
        self._init_client()

        ns = namespace or self.config.namespace

        if ns:
            results = self.collection.get(
                where={"namespace": {"$eq": ns}},
                include=[],
            )
            return len(results["ids"])
        else:
            return self.collection.count()

    async def update_embedding(
        self,
        item_id: str,
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """Update the embedding for an existing item.

        Args:
            item_id: The item ID.
            embedding: New embedding (or regenerate if None).

        Returns:
            True if updated, False if not found.
        """
        self._init_client()

        item = await self.get(item_id)
        if not item:
            return False

        if embedding is None:
            embedding = await self._get_embedding(item.content)

        self.collection.update(
            ids=[item_id],
            embeddings=[embedding],
        )

        return True
