# AutoGen Framework Research

> **AutoGen** is a framework developed by Microsoft for creating multi-agent AI applications that can act autonomously or work alongside humans.

**Repository:** [github.com/microsoft/autogen](https://github.com/microsoft/autogen)

**Documentation:** [microsoft.github.io/autogen](https://microsoft.github.io/autogen/)

**License:** MIT (Code), CC-BY-4.0 (Documentation)

**Requirements:** Python 3.10 or later

---

## Table of Contents

1. [Overview](#overview)
2. [Core Architecture](#core-architecture)
3. [Key Components](#key-components)
4. [Unique Features](#unique-features)
5. [Strengths and Weaknesses](#strengths-and-weaknesses)
6. [Use Cases](#use-cases)
7. [Quick Start](#quick-start)
8. [Related Documentation](#related-documentation)

---

## Overview

AutoGen is Microsoft's open-source framework for building multi-agent AI applications. It enables developers to create systems where multiple AI agents collaborate, communicate, and work together (or with humans) to accomplish complex tasks.

### Key Characteristics

- **Multi-Agent Orchestration**: Agents can call other agents as tools
- **Human-in-the-Loop**: First-class support for human oversight and intervention
- **Code Execution**: Safe execution of generated code via Docker or local environments
- **Flexible Patterns**: Support for various conversation patterns (round-robin, selector-based, swarm)
- **No-Code Option**: AutoGen Studio provides a GUI for building workflows

### Version Information

AutoGen has two major versions with significant architectural differences:

| Version | Architecture | Status |
|---------|-------------|--------|
| v0.2.x | Synchronous, `llm_config` based | Stable, legacy |
| v0.4+ | Async, event-driven, `model_client` based | Current, recommended |

---

## Core Architecture

AutoGen follows a layered architecture:

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

### Conversable Agents

The fundamental abstraction in AutoGen is the **Conversable Agent** - entities that can:
- Send and receive messages autonomously
- Maintain conversation state
- Execute actions and use tools
- Communicate with any other agent

### Multi-Agent Patterns

| Pattern | Description | Use Case |
|---------|-------------|----------|
| Two-Agent | Simple back-and-forth | Code generation, Q&A |
| Round Robin | Fixed turn order | Debate, review cycles |
| Selector | LLM picks next speaker | Complex workflows |
| Swarm | Tool-based handoffs | Customer service |
| Nested | Teams within teams | Hierarchical tasks |

See [architecture.md](./architecture.md) for detailed architecture documentation.

---

## Key Components

### Agent Types

| Component | Description | Version |
|-----------|-------------|---------|
| `AssistantAgent` | LLM-powered general-purpose agent | v0.2, v0.4 |
| `UserProxyAgent` | Human proxy / user input handler | v0.2, v0.4 |
| `CodeExecutorAgent` | Executes code blocks | v0.4 |
| `ConversableAgent` | Base class for custom agents | v0.2 |
| `BaseChatAgent` | Base class for custom agents | v0.4 |

### Team Types (v0.4)

| Component | Description |
|-----------|-------------|
| `RoundRobinGroupChat` | Agents speak in fixed order |
| `SelectorGroupChat` | LLM selects next speaker |
| `Swarm` | Agents hand off via tools |

### Orchestration (v0.2)

| Component | Description |
|-----------|-------------|
| `GroupChat` | Multi-agent conversation container |
| `GroupChatManager` | Orchestrates message flow |

### Code Executors

| Executor | Description | Safety |
|----------|-------------|--------|
| `DockerCommandLineCodeExecutor` | Docker container execution | High |
| `LocalCommandLineCodeExecutor` | Direct host execution | Low |

See [components.md](./components.md) for detailed component documentation.

---

## Unique Features

### 1. Human-in-the-Loop Patterns

AutoGen prioritizes human oversight with multiple intervention patterns:

```python
# User approval before proceeding
user_proxy = UserProxyAgent("user", description="Human for approval")

def selector_with_approval(messages):
    if messages[-1].source == "planner":
        return "user"  # Get human approval
    if "approve" in messages[-1].content.lower():
        return "worker"  # Proceed with work
    return "planner"  # Revise plan
```

### 2. Code Execution Capabilities

Safe code execution with Docker isolation:

```python
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

async with DockerCommandLineCodeExecutor(work_dir="coding") as executor:
    result = await executor.execute_code_blocks(
        code_blocks=[CodeBlock(language="python", code="print('Safe!')")],
        cancellation_token=CancellationToken(),
    )
```

### 3. Nested Conversations

Teams can contain inner teams for hierarchical workflows:

```python
class NestedTeamAgent(BaseChatAgent):
    def __init__(self, name, inner_team):
        self._inner_team = inner_team

    async def on_messages(self, messages, cancellation_token):
        result = await self._inner_team.run(task=messages)
        return Response(chat_message=result.messages[-1])
```

### 4. Agent Customization

Custom agents with full control over behavior:

```python
class CustomAgent(BaseChatAgent):
    async def on_messages(self, messages, cancellation_token):
        # Custom logic
        return Response(chat_message=TextMessage(content="Custom response"))

    async def on_reset(self, cancellation_token):
        # Reset state
        pass
```

### 5. Tool/Function Calling

Agents can use external tools:

```python
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"72F and sunny in {city}"

assistant = AssistantAgent(
    "assistant",
    model_client=model_client,
    tools=[get_weather],
)
```

### 6. MCP Integration

Connect to Model Context Protocol servers:

```python
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

workbench = McpWorkbench(
    StdioServerParams(command="npx", args=["@playwright/mcp"])
)

assistant = AssistantAgent("web_agent", tools=workbench.tools)
```

See [patterns.md](./patterns.md) for detailed pattern documentation.

---

## Strengths and Weaknesses

### Strengths

| Strength | Description |
|----------|-------------|
| **Microsoft Backing** | Strong corporate support and active development |
| **Flexible Architecture** | Supports diverse multi-agent patterns |
| **Human-in-the-Loop** | First-class support for human oversight |
| **Code Execution** | Safe Docker-based execution environment |
| **Rich Documentation** | Extensive docs, examples, and tutorials |
| **AutoGen Studio** | No-code GUI for rapid prototyping |
| **Model Agnostic** | Works with OpenAI, Azure, and compatible APIs |
| **State Persistence** | Save and resume agent/team state |
| **Streaming Support** | Real-time output for long-running tasks |
| **Active Community** | Growing ecosystem and community |

### Weaknesses

| Weakness | Description |
|----------|-------------|
| **Breaking Changes** | v0.2 to v0.4 migration requires significant refactoring |
| **Complexity** | Multiple patterns can be overwhelming for beginners |
| **Learning Curve** | Understanding when to use which pattern |
| **Async-First (v0.4)** | Requires async/await knowledge |
| **Version Confusion** | Two major versions with different APIs |
| **Docker Dependency** | Recommended for safe code execution but adds complexity |
| **Token Costs** | Multi-agent conversations consume more tokens |
| **Debugging** | Multi-agent interactions can be hard to debug |

### Comparison with Other Frameworks

| Aspect | AutoGen | LangChain | CrewAI |
|--------|---------|-----------|--------|
| Focus | Multi-agent conversations | Chains and agents | Role-based crews |
| Architecture | Event-driven actors | Sequential chains | Task-based |
| Human-in-Loop | Excellent | Good | Good |
| Code Execution | Built-in (Docker) | Via tools | Limited |
| Learning Curve | Moderate-High | Moderate | Low |
| Flexibility | Very High | High | Moderate |
| Documentation | Excellent | Excellent | Good |

---

## Use Cases

### Ideal For

1. **Complex Multi-Agent Workflows**
   - Collaborative problem-solving
   - Expert system orchestration
   - Research and analysis tasks

2. **Code Generation and Execution**
   - Automated coding assistants
   - Data analysis pipelines
   - Script generation and testing

3. **Human-Supervised AI Systems**
   - Content review workflows
   - Decision approval systems
   - Guided task completion

4. **Collaborative Debate/Discussion**
   - Multi-perspective analysis
   - Critique and refinement
   - Brainstorming systems

5. **Customer Service**
   - Multi-department routing (Swarm)
   - Escalation handling
   - Specialized agent handoffs

### Less Suitable For

1. **Simple Single-Agent Tasks** - Overhead may not be justified
2. **Real-Time Low-Latency** - Multi-agent adds latency
3. **Resource-Constrained Environments** - Docker requirements
4. **Deterministic Workflows** - LLM-based selection adds variability

---

## Quick Start

### Installation

```bash
# Core packages
pip install -U "autogen-agentchat" "autogen-ext[openai]"

# With Docker support
pip install -U "autogen-ext[docker]"

# AutoGen Studio (GUI)
pip install -U "autogenstudio"
```

### Hello World (v0.4)

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="You are a helpful assistant.",
    )

    result = await assistant.run(task="Say 'Hello World!'")
    print(result.messages[-1].content)

    await model_client.close()

asyncio.run(main())
```

### Two-Agent Chat (v0.2)

```python
import autogen

config_list = [{"model": "gpt-4o", "api_key": "YOUR_KEY"}]
llm_config = {"config_list": config_list}

assistant = autogen.AssistantAgent("assistant", llm_config=llm_config)
user_proxy = autogen.UserProxyAgent("user", human_input_mode="NEVER")

user_proxy.initiate_chat(assistant, message="Tell me a joke")
```

### Group Chat (v0.4)

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    client = OpenAIChatCompletionClient(model="gpt-4o")

    writer = AssistantAgent("writer", model_client=client, system_message="Write content.")
    critic = AssistantAgent("critic", model_client=client, system_message="Critique. Say APPROVE when done.")

    team = RoundRobinGroupChat([writer, critic], termination_condition=TextMentionTermination("APPROVE"))

    await Console(team.run_stream(task="Write a haiku about AI"))

asyncio.run(main())
```

See [examples.md](./examples.md) for more comprehensive examples.

---

## Related Documentation

- [Architecture Details](./architecture.md) - Core architecture and design patterns
- [Components Reference](./components.md) - All agent and team types
- [Conversation Patterns](./patterns.md) - Multi-agent patterns and workflows
- [Code Examples](./examples.md) - Working code samples

---

## External Resources

- [Official Documentation](https://microsoft.github.io/autogen/)
- [GitHub Repository](https://github.com/microsoft/autogen)
- [AutoGen Studio](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/index.html)
- [Migration Guide (v0.2 to v0.4)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/migration-guide.html)
- [PyPI Package](https://pypi.org/project/autogen-agentchat/)

---

*Last Updated: January 2025*
