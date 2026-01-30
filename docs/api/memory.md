# Memory API Reference

API reference cho Memory System.

## SessionMemory

Short-term memory cho conversation.

### Class Definition

```python
from agents_framework.memory.short_term import SessionMemory

class SessionMemory:
    def __init__(
        self,
        max_tokens: int = 4000,
        model: str = "gpt-4",
    ):
        ...
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_tokens` | `int` | Giới hạn tokens trong memory |
| `model` | `str` | Model để tính tokens |

### Methods

#### add_message(role: str, content: str)

Thêm message vào memory.

```python
memory = SessionMemory(max_tokens=4000)
memory.add_message("user", "Xin chào!")
memory.add_message("assistant", "Chào bạn!")
```

#### get_messages() -> List[Dict]

Lấy tất cả messages.

```python
messages = memory.get_messages()
# [{"role": "user", "content": "..."}, ...]
```

#### get_context_string() -> str

Lấy context dạng string.

```python
context = memory.get_context_string()
```

#### clear()

Xóa toàn bộ memory.

```python
memory.clear()
```

## RedisMemory

Long-term persistent memory.

### Class Definition

```python
from agents_framework.memory.long_term.redis import RedisMemory

class RedisMemory:
    def __init__(
        self,
        url: str = "redis://localhost:6379",
        prefix: str = "agents",
        ttl: Optional[int] = None,
    ):
        ...
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `url` | `str` | Redis connection URL |
| `prefix` | `str` | Key prefix cho namespacing |
| `ttl` | `int` | Time-to-live in seconds |

### Methods

#### store(key: str, value: Any, namespace: str = None)

Lưu data vào Redis.

```python
redis_memory = RedisMemory(url="redis://localhost:6379")
await redis_memory.store("user_prefs", {"theme": "dark"}, namespace="user-123")
```

#### retrieve(key: str, namespace: str = None) -> Any

Lấy data từ Redis.

```python
prefs = await redis_memory.retrieve("user_prefs", namespace="user-123")
```

#### delete(key: str, namespace: str = None)

Xóa key.

```python
await redis_memory.delete("user_prefs", namespace="user-123")
```

## VectorMemory

Semantic search với embeddings.

### Class Definition

```python
from agents_framework.memory.long_term.vector import VectorMemory

class VectorMemory:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        collection_name: str = "memories",
        persist_directory: str = None,
    ):
        ...
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `embedding_provider` | `EmbeddingProvider` | Provider cho embeddings |
| `collection_name` | `str` | Tên collection |
| `persist_directory` | `str` | Thư mục lưu data |

### Methods

#### store(content: str, metadata: Dict = None)

Lưu content với embedding.

```python
vector_memory = VectorMemory(embedding_provider=openai_embeddings)
await vector_memory.store(
    "Python là ngôn ngữ lập trình phổ biến",
    metadata={"topic": "programming"}
)
```

#### search(query: str, k: int = 5) -> List[MemoryItem]

Tìm kiếm semantic.

```python
results = await vector_memory.search("ngôn ngữ lập trình", k=3)
for item in results:
    print(f"Score: {item.score}, Content: {item.content}")
```

## MemoryManager

Unified interface cho tất cả memory types.

### Class Definition

```python
from agents_framework.memory.manager import MemoryManager

class MemoryManager:
    def __init__(
        self,
        short_term: Optional[SessionMemory] = None,
        long_term: Optional[RedisMemory] = None,
        vector: Optional[VectorMemory] = None,
    ):
        ...
```

### Methods

#### add_to_session(role: str, content: str)

Thêm vào short-term memory.

```python
manager = MemoryManager(
    short_term=SessionMemory(max_tokens=4000),
    long_term=RedisMemory(),
)
manager.add_to_session("user", "Hello!")
```

#### persist(key: str, value: Any)

Lưu vào long-term memory.

```python
await manager.persist("important_fact", "The answer is 42")
```

#### search_similar(query: str, k: int = 5) -> List

Tìm kiếm trong vector memory.

```python
results = await manager.search_similar("programming languages", k=3)
```

#### get_context(max_tokens: int = 2000) -> str

Lấy context từ tất cả sources.

```python
context = await manager.get_context(max_tokens=2000)
```

## Embedding Providers

### OpenAIEmbeddings

```python
from agents_framework.memory.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    api_key="...",
    model="text-embedding-3-small",
)

vector = await embeddings.embed("Hello world")
vectors = await embeddings.embed_batch(["Hello", "World"])
```

### SentenceTransformerEmbeddings

```python
from agents_framework.memory.embeddings.sentence_transformers import (
    SentenceTransformerEmbeddings
)

embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
```

## Example: Full Memory Setup

```python
from agents_framework.memory.short_term import SessionMemory
from agents_framework.memory.long_term.redis import RedisMemory
from agents_framework.memory.long_term.vector import VectorMemory
from agents_framework.memory.embeddings.openai import OpenAIEmbeddings
from agents_framework.memory.manager import MemoryManager

# Setup embeddings
embeddings = OpenAIEmbeddings(api_key="...")

# Create memory layers
short_term = SessionMemory(max_tokens=4000)
long_term = RedisMemory(url="redis://localhost:6379")
vector = VectorMemory(
    embedding_provider=embeddings,
    collection_name="agent_memories",
    persist_directory="./data/vectors",
)

# Create manager
memory = MemoryManager(
    short_term=short_term,
    long_term=long_term,
    vector=vector,
)

# Use in agent
memory.add_to_session("user", "My name is John")
await memory.persist("user_name", "John")
await vector.store("User prefers dark mode", metadata={"type": "preference"})

# Retrieve context
context = await memory.get_context(max_tokens=2000)
similar = await memory.search_similar("user preferences", k=3)
```
