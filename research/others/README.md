# AI Agent Frameworks Research

This directory contains comprehensive research on notable AI agent frameworks. The research covers architecture, key features, use cases, and comparisons to help inform design decisions.

## Contents

| File | Description |
|------|-------------|
| [swarm.md](./swarm.md) | OpenAI Swarm & Agents SDK - Lightweight multi-agent orchestration |
| [semantic-kernel.md](./semantic-kernel.md) | Microsoft Semantic Kernel - Enterprise AI SDK |
| [llamaindex.md](./llamaindex.md) | LlamaIndex - Data agents and RAG framework |
| [haystack.md](./haystack.md) | Haystack - Pipeline-based LLM applications |
| [comparison.md](./comparison.md) | Detailed comparison of all frameworks |

## Framework Overview

### OpenAI Swarm / Agents SDK
- **Focus**: Multi-agent orchestration with handoffs
- **Key Pattern**: Agents + Handoffs
- **Best For**: Customer service, routing, conversational workflows
- **Repository**: https://github.com/openai/openai-agents-python

### Microsoft Semantic Kernel
- **Focus**: Enterprise AI application development
- **Key Pattern**: Kernel + Plugins (DI container)
- **Best For**: Enterprise apps, Azure integration, .NET projects
- **Repository**: https://github.com/microsoft/semantic-kernel

### LlamaIndex
- **Focus**: Data agents and RAG applications
- **Key Pattern**: Query Engines as Agent Tools
- **Best For**: Document Q&A, research assistants, data-intensive agents
- **Repository**: https://github.com/run-llama/llama_index

### Haystack
- **Focus**: Production-ready LLM pipelines
- **Key Pattern**: Composable Pipelines
- **Best For**: Enterprise RAG, complex workflows, hybrid search
- **Repository**: https://github.com/deepset-ai/haystack

## Quick Comparison

| Feature | Swarm/Agents | Semantic Kernel | LlamaIndex | Haystack |
|---------|--------------|-----------------|------------|----------|
| Stars | ~21K | ~27K | ~47K | ~24K |
| Languages | Python, JS/TS | C#, Python, Java | Python, TS | Python |
| Multi-Agent | Handoffs | Group Chat | Workflows | Pipelines |
| RAG Built-in | No | Via plugins | Native | Native |
| Enterprise | Growing | Strong | Moderate | Strong |

## Key Takeaways

### Design Patterns Observed

1. **Agent Loop Pattern**: All frameworks implement a similar loop:
   - Receive input -> LLM decides action -> Execute tools -> Return to LLM

2. **Tool Abstraction**: Every framework wraps functions as tools:
   - Decorator-based (`@function_tool`, `@kernel_function`, `@tool`)
   - Class-based (`FunctionTool`, `Tool`, `ComponentTool`)

3. **Memory Strategies**:
   - Session-based (Agents SDK)
   - Vector store-backed (Semantic Kernel, LlamaIndex, Haystack)
   - Chat history buffer (all)

4. **Multi-Agent Patterns**:
   - Handoff returns (Swarm)
   - Orchestration/Group chat (Semantic Kernel)
   - Workflow definitions (LlamaIndex)
   - Pipeline routing (Haystack)

### Emerging Trends

1. **Function Calling as Planning**: Native LLM function calling replacing prompt-based planners
2. **MCP Integration**: Model Context Protocol support growing (Semantic Kernel)
3. **Provider Agnostic**: All frameworks moving toward multi-provider support
4. **Built-in Observability**: Tracing and monitoring becoming standard
5. **Sessions/Memory**: Better state management across conversation turns

## Choosing a Framework

```
Simple multi-agent routing?      --> OpenAI Agents SDK
Enterprise/.NET application?     --> Semantic Kernel
Complex data/document agents?    --> LlamaIndex
Production RAG pipelines?        --> Haystack
Learning agent patterns?         --> OpenAI Swarm (educational)
```

## Research Methodology

This research was compiled by:
1. Analyzing official GitHub repositories and READMEs
2. Reviewing official documentation
3. Examining source code structure and patterns
4. Comparing feature sets and capabilities
5. Identifying use cases from real-world examples

## Last Updated

January 2026
