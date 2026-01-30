# Architecture Overview

Tổng quan kiến trúc của Agents Framework.

## Kiến Trúc Tổng Thể

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Agents    │  │   Teams     │  │   Skills    │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
└─────────┼────────────────┼────────────────┼────────────────────┘
          │                │                │
┌─────────┼────────────────┼────────────────┼────────────────────┐
│         ▼                ▼                ▼    Core Layer       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Execution Engine                      │   │
│  │  (ReAct Loop, Lifecycle Hooks, Error Handling)          │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
│  ┌──────────┐  ┌──────────┐ │ ┌──────────┐  ┌──────────────┐   │
│  │   LLM    │  │  Tools   │ │ │  Memory  │  │   Context    │   │
│  │ Providers│  │  System  │ │ │  System  │  │  Management  │   │
│  └──────────┘  └──────────┘ │ └──────────┘  └──────────────┘   │
└─────────────────────────────┼──────────────────────────────────┘
                              │
┌─────────────────────────────┼──────────────────────────────────┐
│                             ▼    Infrastructure Layer           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐   │
│  │   MCP    │  │ Sessions │  │  Routing │  │ Observability │   │
│  │  Client  │  │  Store   │  │  Engine  │  │   (Tracing)   │   │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Các Components Chính

### 1. LLM Providers (`llm/`)

Abstraction layer cho các LLM providers:

- **BaseLLMProvider**: Interface chung cho tất cả providers
- **OpenAIProvider**: Hỗ trợ OpenAI và compatible APIs
- **AnthropicProvider**: Claude models
- **OllamaProvider**: Local models qua Ollama

```python
from agents_framework.llm.base import LLMConfig
from agents_framework.llm.providers.openai import OpenAIProvider

config = LLMConfig(model="gpt-4", api_key="...")
provider = OpenAIProvider(config)
```

### 2. Tool System (`tools/`)

Hệ thống định nghĩa và thực thi tools:

- **@tool decorator**: Tự động generate JSON schema
- **ToolRegistry**: Quản lý nhiều tools
- **ToolExecutor**: Thực thi tools với error handling

```python
from agents_framework.tools.base import tool
from agents_framework.tools.registry import ToolRegistry

@tool(name="search", description="Search the web")
def search(query: str) -> str:
    return f"Results for {query}"

registry = ToolRegistry()
registry.register(search)
```

### 3. Memory System (`memory/`)

Quản lý context và lịch sử:

- **SessionMemory**: Short-term, in-memory
- **RedisMemory**: Long-term, persistent
- **VectorMemory**: Semantic search với embeddings
- **MemoryManager**: Unified interface

```python
from agents_framework.memory.short_term import SessionMemory
from agents_framework.memory.manager import MemoryManager

memory = MemoryManager(
    short_term=SessionMemory(max_tokens=4000),
    long_term=RedisMemory(url="redis://localhost"),
)
```

### 4. Context Management (`context/`)

Quản lý token budget và compaction:

- **TokenCounter**: Đếm tokens cho các models
- **ContextCompactor**: Compress context khi cần
- **Strategies**: Different compaction strategies

### 5. Execution Engine (`execution/`)

ReAct loop và agent lifecycle:

- **ExecutionLoop**: Core agent loop
- **LifecycleHooks**: Before/after callbacks
- **ErrorRecovery**: Retry và fallback logic

### 6. Teams (`teams/`)

Multi-agent orchestration:

- **AgentRegistry**: Quản lý agents
- **MessageRouter**: Route messages giữa agents
- **Patterns**: Hierarchical, Sequential, Swarm, Router

### 7. MCP Client (`mcp/`)

Model Context Protocol integration:

- **MCPClient**: Connect đến MCP servers
- **Transport**: Stdio, SSE, WebSocket
- **MCPToolAdapter**: Wrap MCP tools

### 8. Sessions (`sessions/`)

Session management cho multi-turn:

- **SessionManager**: Lifecycle management
- **SessionStore**: In-memory, File, SQLite
- **ResetPolicies**: Daily, Idle, Manual
- **TranscriptStore**: Lưu lịch sử conversation

### 9. Routing (`routing/`)

Request routing engine:

- **RoutingEngine**: Central router
- **StaticRouter**: Rule-based routing
- **PatternRouter**: Regex pattern matching
- **ContentRouter**: Content-aware routing

### 10. Observability (`observability/`)

Logging, tracing, metrics:

- **AgentTracer**: Trace agent execution
- **StructuredLogger**: Contextual logging
- **Metrics**: Token usage, latency, etc.

## Data Flow

### Single Agent Request

```
User Input
    │
    ▼
┌───────────┐
│  Context  │◄──── Memory
│  Builder  │
└─────┬─────┘
      │
      ▼
┌───────────┐
│    LLM    │
│  Provider │
└─────┬─────┘
      │
      ├─── No Tool Calls ───► Response
      │
      ▼
┌───────────┐
│   Tool    │
│ Executor  │
└─────┬─────┘
      │
      ▼
Loop back to LLM with tool results
```

### Multi-Agent Team

```
Task Input
    │
    ▼
┌───────────────┐
│   Supervisor  │
│    Agent      │
└───────┬───────┘
        │
   ┌────┴────┐
   ▼         ▼
┌──────┐  ┌──────┐
│Worker│  │Worker│
│  A   │  │  B   │
└──┬───┘  └──┬───┘
   │         │
   └────┬────┘
        ▼
┌───────────────┐
│   Supervisor  │
│  Aggregates   │
└───────────────┘
        │
        ▼
    Final Result
```

## Design Principles

1. **Async-first**: Tất cả operations đều async
2. **Protocol-based**: Interfaces qua Python protocols
3. **Composable**: Components có thể combine tự do
4. **Observable**: Built-in tracing và metrics
5. **Extensible**: Dễ dàng thêm providers, tools, patterns
