# AutoGen Architecture

## Overview

AutoGen is a framework developed by Microsoft for creating multi-agent AI applications that can act autonomously or work alongside humans. The framework has evolved significantly, with version 0.4+ representing a ground-up rewrite adopting an async, event-driven architecture.

## Layered Architecture

AutoGen follows a layered architecture design:

```
+------------------------------------------+
|              AutoGen Studio              |
|         (No-code GUI Interface)          |
+------------------------------------------+
|              AgentChat API               |
|    (High-level, opinionated patterns)    |
+------------------------------------------+
|              Core API                    |
|     (Event-driven actor framework)       |
+------------------------------------------+
|              Extensions                  |
|  (LLM clients, code executors, tools)    |
+------------------------------------------+
```

### Core Layer

The Core API provides:
- **Message passing**: Asynchronous communication between agents
- **Event-driven agents**: Agents respond to events and messages
- **Local/Distributed runtimes**: Support for both local and distributed execution
- **Actor-based model**: Agents as independent actors with their own state

### AgentChat Layer

AgentChat is the high-level API recommended for most users:
- Intuitive defaults for rapid prototyping
- Pre-built agent types (AssistantAgent, UserProxyAgent, CodeExecutorAgent)
- Team patterns (RoundRobinGroupChat, SelectorGroupChat, Swarm)
- Built-in support for common multi-agent workflows

### Extensions Layer

Pluggable components for:
- LLM clients (OpenAI, Azure OpenAI, etc.)
- Code execution (Docker, Local)
- Tools and function calling
- MCP (Model Context Protocol) integration

## Conversable Agents Concept

The core abstraction in AutoGen is the **Conversable Agent** - an entity that can send and receive messages, participate in conversations, and execute actions.

### Key Characteristics

1. **Autonomous Communication**: Agents can independently send and receive messages
2. **Stateful**: Each agent maintains its own conversation history and state
3. **Configurable Behavior**: Agents can be customized with system messages, tools, and reply logic
4. **Interoperable**: Any agent can communicate with any other agent

### Agent Communication Flow

```
┌─────────────────┐         ┌─────────────────┐
│    Agent A      │         │    Agent B      │
│                 │ message │                 │
│  ┌───────────┐  │────────>│  ┌───────────┐  │
│  │ on_message│  │         │  │ on_message│  │
│  └───────────┘  │<────────│  └───────────┘  │
│                 │ response│                 │
└─────────────────┘         └─────────────────┘
```

## Multi-Agent Conversation Patterns

### Two-Agent Conversation

The simplest pattern involves two agents communicating:

```python
# Agent A sends a message to Agent B
# Agent B processes and responds
# Communication continues until termination condition
```

### Group Chat (Round Robin)

Agents take turns in a predetermined order:

```
Agent 1 -> Agent 2 -> Agent 3 -> Agent 1 -> ...
```

### Selector Group Chat

An LLM-based selector dynamically chooses the next speaker:

```
                 ┌──────────────┐
                 │   Selector   │
                 │    (LLM)     │
                 └──────┬───────┘
                        │ selects
        ┌───────────────┼───────────────┐
        v               v               v
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │ Agent 1 │    │ Agent 2 │    │ Agent 3 │
   └─────────┘    └─────────┘    └─────────┘
```

### Swarm Pattern

Agents hand off tasks to each other based on tool-based selection:

```python
agent1 = AssistantAgent(
    "Alice",
    handoffs=["Bob"],  # Can hand off to Bob
    system_message="You are Alice..."
)
```

### Nested Conversations

Teams or agents can contain inner teams, enabling hierarchical workflows:

```
┌───────────────────────────────────────────┐
│              Outer Team                    │
│  ┌─────────────────────────────────────┐  │
│  │           Inner Team                 │  │
│  │   Agent A <-> Agent B <-> Agent C   │  │
│  └─────────────────────────────────────┘  │
│                    ^                       │
│                    │                       │
│            Agent D (Orchestrator)         │
└───────────────────────────────────────────┘
```

## Agent Orchestration

### Termination Conditions

AutoGen provides flexible termination conditions:

```python
from autogen_agentchat.conditions import (
    MaxMessageTermination,    # Stop after N messages
    TextMentionTermination,   # Stop when specific text appears
    ExternalTermination,      # External trigger
)

# Combine conditions with | (OR) or & (AND)
termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(10)
```

### Message Handling

Agents handle messages through the `on_messages` method:

```python
class CustomAgent(BaseChatAgent):
    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken
    ) -> Response:
        # Process messages
        # Return response
        return Response(chat_message=TextMessage(...))
```

### State Management

v0.4 introduces proper state persistence:

```python
# Save agent/team state
state = await team.save_state()

# Resume from saved state
await team.load_state(state)
```

## Runtime Models

### Synchronous Execution

Traditional blocking execution for simple use cases.

### Asynchronous Execution

The recommended approach in v0.4:

```python
async def main():
    result = await team.run(task="Your task here")
```

### Streaming Execution

For real-time output and intermediate results:

```python
async for message in team.run_stream(task="Your task"):
    print(message)
```

## Version Comparison (v0.2 vs v0.4)

| Aspect | v0.2 | v0.4 |
|--------|------|------|
| Architecture | Synchronous | Async, event-driven |
| Model Config | `llm_config` dict | Explicit `model_client` |
| Agent Creation | `ConversableAgent` + `register_reply` | Custom `BaseChatAgent` class |
| Group Chat | `GroupChat` + `GroupChatManager` | `RoundRobinGroupChat`, `SelectorGroupChat` |
| Caching | Default enabled (`cache_seed`) | Opt-in via `ChatCompletionCache` |
| State | Manual history export | `save_state`/`load_state` |
| Streaming | Limited | Full support via `run_stream` |

## Design Principles

1. **Modularity**: Agents are modular and can have different levels of information access
2. **Human Involvement**: Prioritizes human-in-the-loop for oversight
3. **Safety**: Recommends Docker containers for code execution
4. **Flexibility**: Supports application-specific agent topologies
5. **Observability**: Built-in logging and streaming for debugging
