# Memory System Guide

Hướng dẫn chi tiết về Memory System.

## Tổng Quan

Memory System có 3 layers:

```
┌─────────────────────────────────────────┐
│           Short-term Memory             │
│    (SessionMemory - In conversation)    │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│           Long-term Memory              │
│    (Redis/File - Persistent storage)    │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│            Vector Memory                │
│    (ChromaDB - Semantic search)         │
└─────────────────────────────────────────┘
```

## 1. Short-term Memory

Lưu trữ conversation hiện tại.

### Basic Usage

```python
from agents_framework.memory.short_term import SessionMemory

# Tạo memory với giới hạn
memory = SessionMemory(max_tokens=4000)

# Thêm messages
memory.add_message("user", "Xin chào, tôi là Minh")
memory.add_message("assistant", "Chào Minh! Tôi có thể giúp gì?")
memory.add_message("user", "Tôi đang học Python")

# Lấy messages
messages = memory.get_messages()
# [
#   {"role": "user", "content": "Xin chào..."},
#   {"role": "assistant", "content": "Chào Minh..."},
#   {"role": "user", "content": "Tôi đang..."},
# ]

# Lấy context string
context = memory.get_context_string()
```

### Token Management

```python
# Memory tự động truncate khi vượt limit
memory = SessionMemory(max_tokens=1000)

# Thêm nhiều messages
for i in range(100):
    memory.add_message("user", f"Message {i}: " + "x" * 100)

# Chỉ giữ messages gần nhất trong limit
messages = memory.get_messages()
print(f"Kept {len(messages)} messages")
```

### With System Prompt

```python
memory = SessionMemory(
    max_tokens=4000,
    system_prompt="Bạn là trợ lý lập trình chuyên về Python.",
)

# System prompt được include trong context
context = memory.get_context_string()
assert "trợ lý lập trình" in context
```

## 2. Long-term Memory

Lưu trữ persistent qua sessions.

### Redis Backend

```python
from agents_framework.memory.long_term.redis import RedisMemory

# Connect
redis = RedisMemory(
    url="redis://localhost:6379",
    prefix="myapp",
    ttl=86400,  # 24 hours
)

# Store với namespace
await redis.store("preferences", {"theme": "dark"}, namespace="user-123")
await redis.store("history", ["item1", "item2"], namespace="user-123")

# Retrieve
prefs = await redis.retrieve("preferences", namespace="user-123")
history = await redis.retrieve("history", namespace="user-123")

# Delete
await redis.delete("preferences", namespace="user-123")

# List keys trong namespace
keys = await redis.list_keys(namespace="user-123")
```

### File Backend

```python
from agents_framework.memory.long_term.file import FileMemory

file_memory = FileMemory(
    base_path="./data/memory",
    format="json",  # hoặc "pickle"
)

# Store
await file_memory.store("user_data", {"name": "Minh"}, namespace="user-123")

# Files được lưu tại: ./data/memory/user-123/user_data.json
```

## 3. Vector Memory

Semantic search với embeddings.

### Setup

```python
from agents_framework.memory.long_term.vector import VectorMemory
from agents_framework.memory.embeddings.openai import OpenAIEmbeddings

# Setup embedding provider
embeddings = OpenAIEmbeddings(
    api_key="...",
    model="text-embedding-3-small",
)

# Create vector memory
vector = VectorMemory(
    embedding_provider=embeddings,
    collection_name="agent_memories",
    persist_directory="./data/vectors",
)
```

### Store Documents

```python
# Store single document
await vector.store(
    content="Python là ngôn ngữ lập trình phổ biến cho data science",
    metadata={
        "topic": "programming",
        "source": "article",
        "date": "2024-01-15",
    }
)

# Store multiple
documents = [
    ("Machine learning cần nhiều data", {"topic": "ml"}),
    ("FastAPI là framework Python", {"topic": "web"}),
    ("Docker containers rất hữu ích", {"topic": "devops"}),
]
for content, metadata in documents:
    await vector.store(content, metadata)
```

### Semantic Search

```python
# Basic search
results = await vector.search("ngôn ngữ lập trình", k=3)

for item in results:
    print(f"Score: {item.score:.3f}")
    print(f"Content: {item.content}")
    print(f"Metadata: {item.metadata}")
    print("---")

# Search với metadata filter
results = await vector.search(
    "programming",
    k=5,
    filter={"topic": "ml"}
)
```

## 4. Unified Memory Manager

Combine tất cả memory types.

### Setup

```python
from agents_framework.memory.manager import MemoryManager
from agents_framework.memory.short_term import SessionMemory
from agents_framework.memory.long_term.redis import RedisMemory
from agents_framework.memory.long_term.vector import VectorMemory

manager = MemoryManager(
    short_term=SessionMemory(max_tokens=4000),
    long_term=RedisMemory(url="redis://localhost:6379"),
    vector=VectorMemory(embedding_provider=embeddings),
)
```

### Usage

```python
# Short-term operations
manager.add_to_session("user", "Hello")
manager.add_to_session("assistant", "Hi there!")
session_context = manager.get_session_context()

# Long-term operations
await manager.persist("user_name", "Minh", namespace="user-123")
name = await manager.retrieve("user_name", namespace="user-123")

# Vector operations
await manager.store_memory(
    "User prefers dark mode",
    metadata={"type": "preference"}
)
similar = await manager.search_similar("theme preferences", k=3)

# Combined context
full_context = await manager.get_context(
    max_tokens=2000,
    include_similar=True,
    similar_query="current topic",
)
```

## 5. Memory Migration

Tự động migrate từ short → long term.

```python
from agents_framework.memory.manager import MemoryManager

manager = MemoryManager(
    short_term=SessionMemory(max_tokens=2000),
    long_term=RedisMemory(),
    auto_migrate=True,
    migration_threshold=1500,  # Migrate khi > 1500 tokens
)

# Khi short-term đầy, auto migrate older messages to long-term
for i in range(100):
    manager.add_to_session("user", f"Message {i}")
    manager.add_to_session("assistant", f"Response {i}")

# Short-term chỉ giữ messages gần nhất
# Older messages được migrate sang long-term
```

## 6. Memory Consolidation

Tổng hợp và summarize memories.

```python
from agents_framework.memory.consolidation import MemoryConsolidator

consolidator = MemoryConsolidator(llm_provider=provider)

# Consolidate conversation into summary
summary = await consolidator.consolidate(
    messages=memory.get_messages(),
    strategy="summarize",
)

# Store consolidated memory
await manager.persist("conversation_summary", summary)
```

## Best Practices

### 1. Choose Right Memory Type

| Memory Type | Use Case |
|-------------|----------|
| Short-term | Current conversation |
| Long-term | User preferences, facts |
| Vector | Similar examples, RAG |

### 2. Namespace Strategy

```python
# Per-user namespace
await redis.store("prefs", data, namespace=f"user-{user_id}")

# Per-agent namespace
await redis.store("state", data, namespace=f"agent-{agent_id}")

# Per-session namespace
await redis.store("context", data, namespace=f"session-{session_id}")
```

### 3. TTL Management

```python
# Short TTL cho temporary data
await redis.store("temp", data, ttl=3600)  # 1 hour

# Long TTL cho persistent data
await redis.store("prefs", data, ttl=86400 * 30)  # 30 days

# No TTL cho permanent data
await redis.store("facts", data, ttl=None)
```

### 4. Memory Cleanup

```python
# Cleanup old sessions
await manager.cleanup(older_than_days=7)

# Clear specific namespace
await redis.clear_namespace("user-123")

# Vacuum vector store
await vector.vacuum()
```

## Example: Complete Memory Setup

```python
import asyncio
from agents_framework.memory.manager import MemoryManager
from agents_framework.memory.short_term import SessionMemory
from agents_framework.memory.long_term.redis import RedisMemory
from agents_framework.memory.long_term.vector import VectorMemory
from agents_framework.memory.embeddings.openai import OpenAIEmbeddings

async def main():
    # Setup all components
    embeddings = OpenAIEmbeddings(api_key="...")

    manager = MemoryManager(
        short_term=SessionMemory(
            max_tokens=4000,
            system_prompt="You are a helpful assistant.",
        ),
        long_term=RedisMemory(
            url="redis://localhost:6379",
            prefix="myapp",
        ),
        vector=VectorMemory(
            embedding_provider=embeddings,
            collection_name="memories",
            persist_directory="./data/vectors",
        ),
        auto_migrate=True,
    )

    # Use in conversation
    manager.add_to_session("user", "My name is Minh")
    manager.add_to_session("assistant", "Hello Minh!")

    # Persist important facts
    await manager.persist("user_name", "Minh", namespace="user-123")

    # Store for semantic search
    await manager.store_memory(
        "Minh is a Python developer who likes FastAPI",
        metadata={"user_id": "123", "type": "profile"}
    )

    # Later, retrieve context
    context = await manager.get_context(max_tokens=2000)
    similar = await manager.search_similar("programming skills", k=3)

    print(f"Context: {context[:200]}...")
    print(f"Similar memories: {len(similar)}")

asyncio.run(main())
```
