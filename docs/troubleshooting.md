# Troubleshooting Guide

Hướng dẫn xử lý các vấn đề thường gặp.

## LLM Provider Issues

### Connection Errors

**Triệu chứng:** `ConnectionError: Failed to connect to API`

**Nguyên nhân:**
- API key không hợp lệ
- Network issues
- API endpoint không đúng

**Giải pháp:**

```python
# Kiểm tra API key
import os
print(f"API Key set: {bool(os.getenv('OPENAI_API_KEY'))}")

# Verify config
config = LLMConfig(
    model="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1",  # Kiểm tra URL
)

# Test connection
try:
    provider = OpenAIProvider(config)
    response = await provider.generate([
        Message(role=MessageRole.USER, content="Hello")
    ])
    print("Connection OK!")
except Exception as e:
    print(f"Error: {e}")
```

### Rate Limiting

**Triệu chứng:** `RateLimitError: Too many requests`

**Giải pháp:**

```python
from agents_framework.llm.base import LLMConfig

config = LLMConfig(
    model="gpt-4",
    api_key="...",
    extra_params={
        "max_retries": 3,
        "retry_delay": 1.0,  # seconds
    },
)

# Hoặc implement retry logic
import asyncio

async def generate_with_retry(provider, messages, max_retries=3):
    for i in range(max_retries):
        try:
            return await provider.generate(messages)
        except RateLimitError:
            if i < max_retries - 1:
                await asyncio.sleep(2 ** i)  # Exponential backoff
            else:
                raise
```

### Token Limit Exceeded

**Triệu chứng:** `TokenLimitError: Context too long`

**Giải pháp:**

```python
from agents_framework.memory.short_term import SessionMemory
from agents_framework.context.compactor import ContextCompactor

# Sử dụng memory với token limit
memory = SessionMemory(max_tokens=3000)  # Leave room for response

# Hoặc compact context
compactor = ContextCompactor(llm_provider=provider)
compacted = await compactor.compact(
    messages,
    target_tokens=3000,
    strategy="summarize",
)
```

## Tool Execution Issues

### Tool Not Found

**Triệu chứng:** `ToolNotFoundError: Tool 'xyz' not registered`

**Giải pháp:**

```python
# Kiểm tra registered tools
registry = ToolRegistry()
registry.register(my_tool)

# List all tools
print(f"Registered tools: {registry.list()}")

# Verify tool name
@tool(name="calculator", description="...")  # Name phải match
def calculator(expr: str) -> str:
    ...
```

### Tool Execution Timeout

**Triệu chứng:** `TimeoutError: Tool execution timed out`

**Giải pháp:**

```python
from agents_framework.tools.executor import ToolExecutor

# Tăng timeout
executor = ToolExecutor(
    registry,
    timeout=60.0,  # 60 seconds
)

# Hoặc per-tool timeout
@tool(name="slow_tool", description="...", timeout=120.0)
async def slow_tool() -> str:
    # Long running operation
    ...
```

### Tool Schema Errors

**Triệu chứng:** `SchemaError: Invalid tool arguments`

**Giải pháp:**

```python
# Kiểm tra type hints
@tool(name="search", description="Search")
def search(
    query: str,           # Required
    limit: int = 10,      # Optional với default
    filters: dict = None, # Optional
) -> str:
    ...

# Generated schema sẽ có:
# required: ["query"]
# properties với types đúng
```

## Memory Issues

### Redis Connection Failed

**Triệu chứng:** `ConnectionError: Cannot connect to Redis`

**Giải pháp:**

```python
# Kiểm tra Redis running
# $ redis-cli ping
# PONG

# Verify URL format
from agents_framework.memory.long_term.redis import RedisMemory

redis = RedisMemory(
    url="redis://localhost:6379/0",  # host:port/db
    # Hoặc với auth
    url="redis://:password@localhost:6379/0",
)

# Test connection
try:
    await redis.store("test", "value")
    print("Redis OK!")
except Exception as e:
    print(f"Redis error: {e}")
```

### Vector Store Errors

**Triệu chứng:** `ChromaDB errors`

**Giải pháp:**

```python
# Kiểm tra persist directory
import os
persist_dir = "./data/vectors"
os.makedirs(persist_dir, exist_ok=True)

# Verify embedding provider
embeddings = OpenAIEmbeddings(api_key="...")
test_vector = await embeddings.embed("test")
print(f"Embedding dimension: {len(test_vector)}")

# Clear và recreate collection nếu corrupt
vector = VectorMemory(
    embedding_provider=embeddings,
    collection_name="memories",
    persist_directory=persist_dir,
)
await vector.clear()  # Reset collection
```

## MCP Issues

### MCP Server Not Starting

**Triệu chứng:** `ProcessError: Failed to start MCP server`

**Giải pháp:**

```bash
# Kiểm tra npx/node available
$ npx --version
$ node --version

# Test run MCP server manually
$ npx -y @modelcontextprotocol/server-filesystem /tmp

# Check path trong config
```

```python
config = StdioTransportConfig(
    command="npx",  # Hoặc full path: "/usr/bin/npx"
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
)
```

### MCP Tool Call Failed

**Triệu chứng:** `MCPError: Tool call failed`

**Giải pháp:**

```python
# Kiểm tra tool exists
tools = client.list_tools()
tool_names = [t.name for t in tools]
print(f"Available: {tool_names}")

# Verify arguments match schema
tool = next(t for t in tools if t.name == "read_file")
print(f"Schema: {tool.input_schema}")

# Call với đúng arguments
result = await client.call_tool("read_file", {
    "path": "/tmp/test.txt"  # Đúng theo schema
})
```

### MCP Connection Timeout

**Triệu chứng:** `TimeoutError: MCP connection timed out`

**Giải pháp:**

```python
from agents_framework.mcp import MCPClient, MCPClientConfig

client = MCPClient(
    transport,
    client_info=MCPClientConfig(
        name="my-client",
        version="1.0",
        timeout=60.0,  # Tăng timeout
    ),
)
```

## Team/Multi-Agent Issues

### Agent Not Responding

**Triệu chứng:** Agent trong team không xử lý messages

**Giải pháp:**

```python
# Kiểm tra agent registered
registry = AgentRegistry()
print(f"Registered agents: {list(registry._agents.keys())}")

# Verify handler registered với router
router = MessageRouter()
router.register_agent("agent-1", handler)

# Test routing
msg = AgentMessage(
    sender_id="test",
    receiver_id="agent-1",
    content="test"
)
await router.route(msg)
```

### Infinite Handoff Loop

**Triệu chứng:** Agents handoff qua lại không dừng

**Giải pháp:**

```python
from agents_framework.teams.patterns.swarm import SwarmPattern

pattern = SwarmPattern(
    agents=agents,
    entry_point="triage",
    max_handoffs=5,  # Limit handoffs
    detect_loops=True,  # Detect A->B->A loops
)
```

## Session Issues

### Session Not Persisting

**Triệu chứng:** Session data bị mất sau restart

**Giải pháp:**

```python
from agents_framework.sessions import (
    SessionManager,
    FileSessionStore,  # Hoặc SQLiteSessionStore
)

# Dùng persistent store
store = FileSessionStore(base_path="./data/sessions")
# Hoặc
store = SQLiteSessionStore(db_path="./data/sessions.db")

manager = SessionManager(store=store, config=config)

# Verify persistence
session = await manager.get_or_create("user:123")
print(f"Session ID: {session.session_id}")

# After restart, should get same session
```

### Session Conflict

**Triệu chứng:** `ConflictError: Session already exists`

**Giải pháp:**

```python
# Use get_or_create instead of create
session = await manager.get_or_create("user:123")

# Hoặc force recreate
session = await manager.create("user:123", force=True)
```

## Performance Issues

### Slow Response Times

**Giải pháp:**

```python
# 1. Use streaming
async for chunk in provider.stream(messages):
    print(chunk, end="", flush=True)

# 2. Reduce context size
memory = SessionMemory(max_tokens=2000)  # Smaller context

# 3. Parallel tool execution
executor = ToolExecutor(registry, parallel=True)
results = await executor.execute_batch(tool_calls)

# 4. Cache embeddings
embeddings = OpenAIEmbeddings(
    api_key="...",
    cache_enabled=True,
    cache_ttl=3600,
)
```

### High Memory Usage

**Giải pháp:**

```python
# 1. Clear session memory periodically
memory.clear()

# 2. Use TTL for Redis
redis = RedisMemory(url="...", ttl=3600)

# 3. Limit vector store size
await vector.cleanup(older_than_days=30)

# 4. Use lazy loading
manager = MemoryManager(
    lazy_load=True,  # Only load when needed
)
```

## Debug Mode

Enable debug logging:

```python
import logging

# Enable all logging
logging.basicConfig(level=logging.DEBUG)

# Or specific modules
logging.getLogger("agents_framework.llm").setLevel(logging.DEBUG)
logging.getLogger("agents_framework.mcp").setLevel(logging.DEBUG)
logging.getLogger("agents_framework.tools").setLevel(logging.DEBUG)
```

## Getting Help

1. **Check logs:** Enable DEBUG logging
2. **Verify configs:** Print and check all configurations
3. **Isolate issue:** Test components individually
4. **Check versions:** Ensure compatible package versions
5. **Report issue:** https://github.com/your-org/agents-framework/issues
