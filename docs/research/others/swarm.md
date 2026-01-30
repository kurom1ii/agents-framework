# OpenAI Swarm / Agents SDK

## Overview

**OpenAI Swarm** was an experimental, educational framework for exploring lightweight multi-agent orchestration patterns. It has been **replaced by the OpenAI Agents SDK**, which is the production-ready evolution of Swarm and is actively maintained by the OpenAI team.

- **Repository (Legacy Swarm)**: https://github.com/openai/swarm
- **Repository (Agents SDK)**: https://github.com/openai/openai-agents-python
- **Stars**: ~21,000 (Swarm)
- **Language**: Python
- **License**: MIT

## Core Philosophy

Swarm (and its successor) focuses on making agent **coordination** and **execution** lightweight, highly controllable, and easily testable. The framework uses two primitive abstractions:

1. **Agents**: LLMs configured with instructions, tools, guardrails, and handoffs
2. **Handoffs**: Specialized tool calls for transferring control between agents

These primitives are powerful enough to express rich dynamics between tools and networks of agents, allowing developers to build scalable, real-world solutions while avoiding a steep learning curve.

## Key Concepts

### Agents

An Agent encapsulates:
- **Instructions**: System prompt (can be static string or dynamic function)
- **Functions/Tools**: Python functions the agent can call
- **Model**: The LLM to use (default: gpt-4o)
- **Tool Choice**: How the agent should select tools

```python
from agents import Agent, Runner

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant",
    tools=[get_weather],
)

result = Runner.run_sync(agent, "What's the weather?")
```

### Handoffs

Handoffs are a specialized mechanism for transferring control between agents. When a function returns an Agent object, execution transfers to that agent.

```python
spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language.",
    handoffs=[spanish_agent, english_agent],
)
```

### The Agent Loop

When `Runner.run()` is called, an execution loop runs until final output:

1. Call the LLM with model, settings, and message history
2. LLM returns a response (may include tool calls)
3. If response has final output, return it and end the loop
4. If response has a handoff, switch to the new agent and go to step 1
5. Process tool calls and append tool response messages, then go to step 1

### Context Variables

Context variables allow passing state across agents and function calls:

```python
def instructions(context_variables):
    user_name = context_variables["user_name"]
    return f"Help the user, {user_name}, do whatever they want."

agent = Agent(instructions=instructions)
response = client.run(
    agent=agent,
    messages=[{"role": "user", "content": "Hi!"}],
    context_variables={"user_name": "John"}
)
```

### Functions/Tools

Tools are Python functions that agents can call. The SDK automatically converts functions to JSON schemas:

```python
from agents import function_tool

@function_tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"The weather in {city} is sunny."

agent = Agent(
    name="Weather Agent",
    tools=[get_weather],
)
```

## OpenAI Agents SDK Features (Production Version)

### Core Concepts

1. **Agents**: LLMs configured with instructions, tools, guardrails, and handoffs
2. **Handoffs**: Specialized tool calls for transferring control between agents
3. **Guardrails**: Configurable safety checks for input and output validation
4. **Sessions**: Automatic conversation history management across agent runs
5. **Tracing**: Built-in tracking of agent runs for debugging and optimization

### Sessions (Memory Management)

```python
from agents import Agent, Runner, SQLiteSession

agent = Agent(name="Assistant", instructions="Reply concisely.")
session = SQLiteSession("conversation_123")

# First turn
result = await Runner.run(agent, "What city is the Golden Gate Bridge in?", session=session)
print(result.final_output)  # "San Francisco"

# Second turn - agent remembers context
result = await Runner.run(agent, "What state is it in?", session=session)
print(result.final_output)  # "California"
```

### Tracing

The SDK automatically traces agent runs with support for external destinations:
- Logfire
- AgentOps
- Braintrust
- Scorecard
- Keywords AI

### Provider Agnostic

The SDK supports:
- OpenAI Responses and Chat Completions APIs
- 100+ other LLMs via LiteLLM integration

## Architecture

### Swarm Core (Legacy)

```
swarm/
  __init__.py      # Exports: Swarm, Agent
  core.py          # Main Swarm class with run() and handle_tool_calls()
  types.py         # Agent, Result, Response types
  util.py          # function_to_json, debug utilities
  repl/            # Interactive REPL for testing
```

### Agents SDK (Current)

```
agents/
  agent.py         # Agent definition
  runner.py        # Runner for executing agents
  tools/           # Tool definitions and function_tool decorator
  memory/          # Session implementations (SQLite, Redis)
  tracing/         # Tracing infrastructure
  extensions/      # Voice support, Redis sessions, etc.
```

## Example Use Cases

1. **Triage Agent**: Routes requests to specialized agents based on content
2. **Customer Service Bot**: Multi-agent system for handling different request types
3. **Personal Shopper**: Agent with tools for product search and order management
4. **Support Bot**: User interface agent + help center agent with tools
5. **Airline Customer Service**: Complex multi-agent routing

## Key Differentiators

| Feature | Description |
|---------|-------------|
| **Lightweight** | Minimal abstraction, runs on Chat Completions API |
| **Stateless** | No state stored between calls (like Chat Completions) |
| **Handoff Pattern** | Native support for agent-to-agent transfer |
| **Function-First** | Direct Python function calls as tools |
| **Educational** | Designed to teach multi-agent patterns |
| **Provider Agnostic** | Supports 100+ LLMs (Agents SDK) |
| **Built-in Tracing** | Automatic tracking and debugging |
| **Session Management** | SQLite and Redis session support |

## Best Use Cases

- **Multi-agent routing**: When you need specialized agents for different tasks
- **Customer service**: Triage and handoff between departments
- **Conversational workflows**: Linear or branching conversation flows
- **Learning multi-agent patterns**: Educational framework for understanding concepts
- **Lightweight prototypes**: Quick multi-agent proof-of-concepts
- **Production applications**: With the Agents SDK for enterprise deployments

## Limitations

- **OpenAI-centric** (original Swarm): Primarily designed for OpenAI models
- **No built-in planning**: Relies on LLM to decide handoffs
- **Minimal memory**: Context variables for simple state only (original Swarm)
- **No vector store integration**: No built-in RAG capabilities

## Installation

### Swarm (Legacy/Educational)
```bash
pip install git+https://github.com/openai/swarm.git
```

### OpenAI Agents SDK (Production)
```bash
pip install openai-agents

# With voice support
pip install 'openai-agents[voice]'

# With Redis session support
pip install 'openai-agents[redis]'
```

## References

- [OpenAI Swarm GitHub](https://github.com/openai/swarm)
- [OpenAI Agents SDK GitHub](https://github.com/openai/openai-agents-python)
- [Agents SDK Documentation](https://openai.github.io/openai-agents-python/)
- [Agents SDK JS/TS](https://github.com/openai/openai-agents-js)
