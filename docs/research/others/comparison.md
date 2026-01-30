# AI Agent Frameworks Comparison

## Overview

This document provides a comprehensive comparison of major AI agent frameworks, highlighting their key features, architectures, and best use cases.

## Quick Reference Table

| Feature | OpenAI Swarm/Agents SDK | Semantic Kernel | LlamaIndex | Haystack |
|---------|------------------------|-----------------|------------|----------|
| **Primary Focus** | Multi-agent orchestration | Enterprise AI SDK | Data agents & RAG | Pipeline-based LLM apps |
| **Developer** | OpenAI | Microsoft | LlamaIndex Inc. | deepset |
| **Stars (GitHub)** | ~21,000 | ~27,000 | ~47,000 | ~24,000 |
| **Languages** | Python (JS/TS for SDK) | C#, Python, Java | Python, TypeScript | Python |
| **License** | MIT | MIT | MIT | Apache 2.0 |
| **Production Ready** | Yes (Agents SDK) | Yes | Yes | Yes |

## Detailed Comparison

### Architecture Philosophy

| Framework | Approach | Key Abstraction |
|-----------|----------|-----------------|
| **OpenAI Swarm/Agents SDK** | Minimal, stateless | Agents + Handoffs |
| **Semantic Kernel** | Enterprise DI container | Kernel + Plugins |
| **LlamaIndex** | Data-centric | Query Engines + Tools |
| **Haystack** | Pipeline composition | Components + Pipelines |

### Agent Capabilities

| Capability | OpenAI Agents | Semantic Kernel | LlamaIndex | Haystack |
|------------|---------------|-----------------|------------|----------|
| Single Agent | Yes | Yes | Yes | Yes |
| Multi-Agent | Yes (handoffs) | Yes (group chat) | Yes (workflow) | Yes (pipelines) |
| Tool Calling | Yes | Yes (plugins) | Yes | Yes |
| Handoffs | Native | Via orchestration | Via workflow | Via routing |
| Memory | Sessions (SQL/Redis) | Vector stores | ChatMemoryBuffer | Short/long-term |
| Streaming | Yes | Yes | Yes | Yes |
| Guardrails | Built-in | Via middleware | Custom | Custom |

### Planning & Reasoning

| Feature | OpenAI Agents | Semantic Kernel | LlamaIndex | Haystack |
|---------|---------------|-----------------|------------|----------|
| Planning Method | LLM-driven loop | Function calling | Agent loop | Pipeline routing |
| ReAct Support | No | No | Yes (ReActAgent) | Via agents |
| Custom Planners | Via tools | Deprecated (use FC) | Custom agents | Pipeline composition |
| Step Tracking | Tracing | ChatHistory | Agent events | Pipeline tracing |

### Data & RAG Integration

| Feature | OpenAI Agents | Semantic Kernel | LlamaIndex | Haystack |
|---------|---------------|-----------------|------------|----------|
| Vector Store Support | No (external) | Yes (many) | Yes (many) | Yes (many) |
| Document Loading | No | Limited | Extensive | Extensive |
| Query Engines | No | No | Core feature | Via retrievers |
| RAG Pipelines | No | Via plugins | Core feature | Core feature |
| Knowledge Graphs | No | No | Yes | Limited |

### Enterprise Features

| Feature | OpenAI Agents | Semantic Kernel | LlamaIndex | Haystack |
|---------|---------------|-----------------|------------|----------|
| Multi-language | Python, JS/TS | C#, Python, Java | Python, TS | Python |
| Cloud Integration | OpenAI API | Azure-first | Cloud-agnostic | Cloud-agnostic |
| Observability | Built-in tracing | Events/middleware | LlamaTrace | Telemetry |
| Security | Guardrails | Enterprise controls | Custom | Enterprise |
| DI Support | No | Native | Limited | Limited |

### Model Support

| Provider | OpenAI Agents | Semantic Kernel | LlamaIndex | Haystack |
|----------|---------------|-----------------|------------|----------|
| OpenAI | Native | Yes | Yes | Yes |
| Azure OpenAI | Yes | Native | Yes | Yes |
| Anthropic | Via LiteLLM | Yes | Yes | Yes |
| Google | Via LiteLLM | Yes | Yes | Yes |
| HuggingFace | Via LiteLLM | Yes | Yes | Yes |
| Local (Ollama) | Via LiteLLM | Yes | Yes | Yes |
| 100+ LLMs | Yes (LiteLLM) | Many | Many | Many |

## Use Case Recommendations

### When to Use OpenAI Swarm/Agents SDK

**Best for:**
- Multi-agent routing and handoffs
- Customer service applications
- Conversational workflows
- Rapid prototyping of multi-agent systems
- Teams already using OpenAI

**Example use cases:**
- Customer support triage
- Multi-department request routing
- Interactive assistants with specialized agents
- Educational projects learning multi-agent patterns

### When to Use Semantic Kernel

**Best for:**
- Enterprise applications
- Microsoft/Azure ecosystem projects
- .NET applications
- Complex business process automation
- Projects requiring multiple language support

**Example use cases:**
- Enterprise chatbots with Azure integration
- Business process automation
- Internal knowledge assistants
- Multi-language application development

### When to Use LlamaIndex

**Best for:**
- Data-intensive agent applications
- RAG systems with complex retrieval needs
- Multi-source data synthesis
- Document Q&A systems
- Research and analysis tools

**Example use cases:**
- Document search and Q&A
- Research assistants
- SQL query agents
- Knowledge base chatbots
- Multi-index retrieval systems

### When to Use Haystack

**Best for:**
- Production RAG systems
- Complex multi-step pipelines
- Hybrid search applications
- Customizable processing workflows
- Enterprise deployments

**Example use cases:**
- Enterprise document search
- Question answering systems
- Content extraction pipelines
- Semantic search applications
- Agent + RAG combinations

## Feature Deep Dive

### Multi-Agent Orchestration

| Framework | Pattern | Implementation |
|-----------|---------|----------------|
| **OpenAI Agents** | Handoffs | Function returns Agent object, control transfers |
| **Semantic Kernel** | Group Chat | Multiple agents collaborate in shared context |
| **LlamaIndex** | AgentWorkflow | Explicit handoff definitions in workflow |
| **Haystack** | Pipeline Routing | ConditionalRouter directs to agent components |

### Tool/Plugin Systems

**OpenAI Agents SDK:**
```python
@function_tool
def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny"
```

**Semantic Kernel:**
```python
@kernel_function(description="Get weather")
def get_weather(self, city: str) -> str:
    return f"Weather in {city}: Sunny"
```

**LlamaIndex:**
```python
tool = FunctionTool.from_defaults(fn=get_weather)
```

**Haystack:**
```python
@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny"
```

### Memory/Session Management

| Framework | Short-term | Long-term | Persistence |
|-----------|------------|-----------|-------------|
| OpenAI Agents | Session class | Session stores | SQLite, Redis |
| Semantic Kernel | ChatHistory | Vector stores | Multiple backends |
| LlamaIndex | ChatMemoryBuffer | Storage context | Multiple backends |
| Haystack | Conversation | Memory components | Custom stores |

## Performance Considerations

### Overhead & Complexity

| Framework | Setup Complexity | Runtime Overhead | Learning Curve |
|-----------|-----------------|------------------|----------------|
| OpenAI Agents | Low | Low | Low |
| Semantic Kernel | Medium-High | Medium | Medium-High |
| LlamaIndex | Medium | Medium-High (indexing) | Medium |
| Haystack | Medium | Medium | Medium |

### Scalability Patterns

| Framework | Horizontal Scaling | Stateless Design | Distributed Agents |
|-----------|-------------------|------------------|-------------------|
| OpenAI Agents | Easy (stateless) | Native | Via external state |
| Semantic Kernel | Enterprise patterns | Via DI | Via services |
| LlamaIndex | Via external stores | Configurable | Limited |
| Haystack | Via Hayhooks | Pipeline-based | Via components |

## Integration Ecosystem

### Number of Integrations

| Category | OpenAI Agents | Semantic Kernel | LlamaIndex | Haystack |
|----------|---------------|-----------------|------------|----------|
| LLMs | 100+ (LiteLLM) | ~20 | ~50 | ~30 |
| Vector Stores | External | ~10 | ~30 | ~15 |
| Data Loaders | No | Limited | 300+ | ~50 |
| Tools/APIs | Custom | Plugins | LlamaHub | Integrations |

## Summary Matrix

### Strengths

| Framework | Key Strengths |
|-----------|--------------|
| **OpenAI Agents** | Simple, lightweight, native handoffs, built-in tracing |
| **Semantic Kernel** | Enterprise-ready, multi-language, Azure integration, plugins |
| **LlamaIndex** | Best for RAG, extensive data connectors, query engines as tools |
| **Haystack** | Pipeline flexibility, production-ready, explicit data flow |

### Limitations

| Framework | Key Limitations |
|-----------|----------------|
| **OpenAI Agents** | No RAG built-in, limited memory, OpenAI-centric |
| **Semantic Kernel** | Complex setup, Microsoft-centric, learning curve |
| **LlamaIndex** | Data-focused, complexity for simple cases, heavy for non-RAG |
| **Haystack** | Verbose, agent features newer, pipeline learning curve |

## Decision Guide

```
Need multi-agent handoffs with minimal setup?
  --> OpenAI Agents SDK

Building enterprise app in .NET/Azure?
  --> Semantic Kernel

Need agents over complex data/documents?
  --> LlamaIndex

Want explicit, composable pipelines?
  --> Haystack

Need RAG + agents combined?
  --> LlamaIndex or Haystack

Starting fresh, want simplicity?
  --> OpenAI Agents SDK

Need multi-language support (C#, Python, Java)?
  --> Semantic Kernel
```

## References

- [OpenAI Swarm](https://github.com/openai/swarm)
- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
- [Microsoft Semantic Kernel](https://github.com/microsoft/semantic-kernel)
- [LlamaIndex](https://github.com/run-llama/llama_index)
- [Haystack](https://github.com/deepset-ai/haystack)
