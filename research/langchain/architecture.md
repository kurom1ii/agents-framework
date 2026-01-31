# LangChain Architecture

This document provides a comprehensive overview of LangChain's core architecture, including agent structure, agent types, memory systems, and the tool/function calling mechanism.

---

## Table of Contents

1. [Core Architecture Overview](#core-architecture-overview)
2. [Agent Structure](#agent-structure)
3. [Agent Types](#agent-types)
4. [Memory Systems](#memory-systems)
5. [Tool/Function Calling Mechanism](#toolfunction-calling-mechanism)
6. [LangGraph Integration](#langgraph-integration)

---

## Core Architecture Overview

LangChain agents are built on top of **LangGraph**, a low-level orchestration framework that provides:

- **Durable Execution**: Reliable state persistence across agent runs
- **Streaming**: Real-time output streaming during execution
- **Human-in-the-Loop**: Interrupt and resume capabilities for human oversight
- **Persistence**: State checkpointing for recovery and debugging
- **Conditional Routing**: Dynamic workflow decisions based on state

### High-Level Architecture

```
+------------------------------------------------------------------+
|                     LangChain Application                        |
+------------------------------------------------------------------+
|                                                                  |
|  +----------------------+    +---------------------------+       |
|  |     Agent Layer      |    |   Integration Layer       |       |
|  |  - create_agent()    |    |  - Model Providers        |       |
|  |  - Tools/Toolkits    |    |  - Vector Stores          |       |
|  |  - Memory/State      |    |  - Document Loaders       |       |
|  +----------------------+    +---------------------------+       |
|              |                           |                       |
|              v                           v                       |
|  +----------------------------------------------------------+   |
|  |                    LangGraph Runtime                      |   |
|  |  - StateGraph         - Checkpointers                     |   |
|  |  - Nodes/Edges        - Memory Stores                     |   |
|  |  - Conditional Logic  - Streaming                         |   |
|  +----------------------------------------------------------+   |
|              |                                                   |
|              v                                                   |
|  +----------------------------------------------------------+   |
|  |                    LangSmith (Optional)                   |   |
|  |  - Tracing           - Debugging                          |   |
|  |  - Evaluation        - Monitoring                         |   |
|  +----------------------------------------------------------+   |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Agent Structure

### Modern Agent Creation

The modern LangChain API uses a simplified `create_agent()` function:

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72F"

agent = create_agent(
    model="claude-sonnet-4-5-20250929",  # Model specification
    tools=[search, get_weather],          # List of tools
    system_prompt="You are a helpful assistant"  # System instruction
)
```

### Agent Components

| Component | Description |
|-----------|-------------|
| **Model** | The LLM that powers the agent's reasoning (OpenAI, Anthropic, Google, etc.) |
| **Tools** | Functions the agent can call to interact with external systems |
| **System Prompt** | Instructions that define the agent's behavior and personality |
| **State** | Persistent data including message history and custom state |
| **Middleware** | Hooks for customizing agent behavior (error handling, logging, etc.) |
| **Checkpointer** | State persistence mechanism for memory and resumption |

### Agent Execution Flow

```
User Input
    |
    v
+------------------+
|  System Prompt   |
|  + Message       |
|  History         |
+------------------+
    |
    v
+------------------+
|   LLM Call       |
| (Reasoning)      |
+------------------+
    |
    +-------> Tool Calls Present?
    |              |
    |         Yes  |  No
    |              |   |
    |              v   v
    |         +------------------+
    |         |  Execute Tools   |
    |         +------------------+
    |              |
    |              v
    |         +------------------+
    |         | Add Tool Results |
    |         | to Messages      |
    |         +------------------+
    |              |
    +<-------------+
    |
    v
+------------------+
|  Final Response  |
+------------------+
```

---

## Agent Types

### 1. ReAct Agent (Reasoning + Acting)

The ReAct pattern interleaves reasoning steps with tool calls:

```python
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)
agent = create_agent(model, tools=[tool])

# ReAct execution produces: Thought -> Action -> Observation -> Thought -> ...
events = agent.stream(
    {"messages": [("user", "Search in google drive, who is 'Yann LeCun'?")]},
    stream_mode="values",
)
```

**Characteristics:**
- Explicit reasoning traces visible in output
- Iterative refinement based on observations
- Good for complex, multi-step tasks
- Verbose output aids debugging

**When to Use:**
- Tasks requiring iterative retrieval and verification
- Multi-step tool use scenarios
- When transparency of reasoning is important

### 2. OpenAI Functions Agent

Uses OpenAI's native function calling capability:

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_classic import hub

prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(temperature=0)

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

result = agent_executor.invoke({
    "input": "How is the tech sector being affected by fed policy?"
})
```

**Characteristics:**
- Uses model's native function calling
- More structured tool invocation
- Less verbose than ReAct
- Better for models with strong function calling

### 3. Structured Chat Agent

For models that need structured JSON output:

```python
from langchain.agents import StructuredChatAgent, AgentExecutor
from langchain_classic.chains import LLMChain

prompt = StructuredChatAgent.create_prompt(
    prefix=prefix,
    tools=tools,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)

llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = StructuredChatAgent(llm_chain=llm_chain, verbose=True, tools=tools)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, verbose=True, memory=memory, tools=tools
)
```

### 4. Zero-Shot ReAct Agent

Original ReAct implementation without few-shot examples:

```python
from langchain.agents import AgentType, create_pandas_dataframe_agent

agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
    df,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
```

### Agent Type Comparison

| Agent Type | Best For | Model Requirement | Output Style |
|------------|----------|-------------------|--------------|
| ReAct | Multi-step reasoning | Any LLM | Verbose traces |
| OpenAI Functions | Structured tool calls | OpenAI/Compatible | Concise |
| Structured Chat | JSON-based interaction | Any LLM | JSON formatted |
| Zero-Shot ReAct | Simple tasks | Any LLM | ReAct format |

---

## Memory Systems

LangChain provides multiple memory patterns for maintaining context across conversations.

### Memory Architecture

```
+------------------------------------------------------------------+
|                      Memory Layer                                |
+------------------------------------------------------------------+
|                                                                  |
|  +-----------------+  +-----------------+  +-----------------+   |
|  | Short-Term      |  | Long-Term       |  | Semantic        |   |
|  | (Message Buffer)|  | (Checkpointer)  |  | (Vector Store)  |   |
|  +-----------------+  +-----------------+  +-----------------+   |
|          |                    |                    |             |
|          v                    v                    v             |
|  +---------------------------------------------------------+    |
|  |                   State Management                       |    |
|  |  - Message History    - Custom State                     |    |
|  |  - Thread IDs         - User Context                     |    |
|  +---------------------------------------------------------+    |
|                                                                  |
+------------------------------------------------------------------+
```

### 1. Conversation Buffer Memory

Stores the complete conversation history:

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory
)
```

### 2. Message Trimming (Short-Term Memory)

Keeps conversation within context limits:

```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents.middleware import before_model

@before_model
def trim_messages(state: AgentState, runtime: Runtime):
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

agent = create_agent(
    model,
    tools=tools,
    middleware=[trim_messages],
    checkpointer=InMemorySaver(),
)
```

### 3. Summarization Memory

Automatically summarizes long conversations:

```python
from langchain import createAgent, summarizationMiddleware
from langgraph import MemorySaver

checkpointer = MemorySaver()

agent = createAgent({
    model: "gpt-4o",
    tools: [],
    middleware: [
        summarizationMiddleware({
            model: "gpt-4o-mini",
            trigger: { tokens: 4000 },  # Trigger at 4000 tokens
            keep: { messages: 20 },      # Keep last 20 messages
        }),
    ],
    checkpointer,
})
```

### 4. Long-Term Memory (Vector Store)

Semantic search over past interactions:

```python
from langgraph.store.memory import InMemoryStore

def embed(texts: list[str]) -> list[list[float]]:
    # Replace with actual embedding function
    return [[1.0, 2.0] * len(texts)]

store = InMemoryStore(index={"embed": embed, "dims": 2})
user_id = "my-user"
namespace = (user_id, "memories")

# Store memories
store.put(
    namespace,
    "a-memory",
    {
        "rules": [
            "User likes short, direct language",
            "User only speaks English & python",
        ],
    },
)

# Search memories
items = store.search(
    namespace,
    query="language preferences"
)
```

### 5. Checkpointer-Based Persistence

For durable state across sessions:

```python
from langgraph.checkpoint.memory import MemorySaver
# Or for production:
# from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = MemorySaver()
agent = create_agent(
    model,
    tools=tools,
    checkpointer=checkpointer
)

# Each thread maintains separate conversation state
config = {"configurable": {"thread_id": "user-123"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Hello!"}]},
    config
)
```

### Memory Type Comparison

| Memory Type | Use Case | Persistence | Scalability |
|-------------|----------|-------------|-------------|
| Buffer | Simple conversations | In-memory | Limited |
| Trimming | Long conversations | In-memory | Good |
| Summarization | Extended sessions | In-memory + LLM | Good |
| Vector Store | Semantic recall | Database | Excellent |
| Checkpointer | Production apps | Configurable | Excellent |

---

## Tool/Function Calling Mechanism

### Tool Definition

Tools are defined using the `@tool` decorator:

```python
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather for a location.

    Args:
        location: The city and state, e.g. San Francisco, CA
    """
    return f"Weather in {location}: Sunny, 72F"
```

### Advanced Tool Definition with Pydantic

```python
from pydantic import BaseModel, Field
from langchain.tools import tool

class WeatherInput(BaseModel):
    """Input for weather tool."""
    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")
    unit: str = Field(default="fahrenheit", description="Temperature unit")

@tool(args_schema=WeatherInput)
def get_weather(location: str, unit: str = "fahrenheit") -> str:
    """Get the current weather in a given location."""
    return f"Weather in {location}: Sunny, 72{unit[0].upper()}"
```

### Tool Binding to Models

```python
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather for a location."""
    return f"Weather in {location}: Sunny, 72F"

# Bind tools to model
model_with_tools = model.bind_tools([get_weather])
response = model_with_tools.invoke("What's the weather in San Francisco?")
print(response.tool_calls)
# [{'name': 'get_weather', 'args': {'location': 'San Francisco'}, 'id': '...'}]
```

### Tool Execution Flow

```
Model Response with tool_calls
           |
           v
+------------------------+
| Tool Call Extraction   |
| - Name                 |
| - Arguments            |
| - Call ID              |
+------------------------+
           |
           v
+------------------------+
| Tool Function Lookup   |
+------------------------+
           |
           v
+------------------------+
| Argument Validation    |
| (Pydantic Schema)      |
+------------------------+
           |
           v
+------------------------+
| Tool Execution         |
+------------------------+
           |
           v
+------------------------+
| ToolMessage Creation   |
| - Content (result)     |
| - Tool Call ID         |
+------------------------+
           |
           v
Add to Message History
```

### Error Handling for Tools

```python
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model="gpt-4o",
    tools=[search, get_weather],
    middleware=[handle_tool_errors]
)
```

---

## LangGraph Integration

LangChain agents are built on LangGraph for advanced orchestration.

### StateGraph Basics

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    input: str
    output: str

def process_node(state: State):
    return {"output": f"Processed: {state['input']}"}

# Build graph
builder = StateGraph(State)
builder.add_node("process", process_node)
builder.add_edge(START, "process")
builder.add_edge("process", END)

graph = builder.compile()
result = graph.invoke({"input": "Hello"})
```

### Conditional Routing

```python
from langgraph.graph import StateGraph, START, END

def route_decision(state: State):
    if state["decision"] == "story":
        return "story_node"
    elif state["decision"] == "joke":
        return "joke_node"
    return "poem_node"

builder = StateGraph(State)
builder.add_node("router", router_node)
builder.add_node("story_node", story_node)
builder.add_node("joke_node", joke_node)
builder.add_node("poem_node", poem_node)

builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    route_decision,
    {
        "story_node": "story_node",
        "joke_node": "joke_node",
        "poem_node": "poem_node",
    }
)
builder.add_edge("story_node", END)
builder.add_edge("joke_node", END)
builder.add_edge("poem_node", END)

graph = builder.compile()
```

### Human-in-the-Loop with Interrupts

```python
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

def approval_node(state):
    decision = interrupt({
        "question": "Approve this action?",
        "details": state["action_details"],
    })
    return Command(goto="proceed" if decision else "cancel")

builder = StateGraph(State)
builder.add_node("approval", approval_node)
builder.add_node("proceed", proceed_node)
builder.add_node("cancel", cancel_node)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# First invocation pauses at interrupt
config = {"configurable": {"thread_id": "approval-123"}}
initial = graph.invoke({"action_details": "Transfer $500"}, config=config)

# Resume with decision
resumed = graph.invoke(Command(resume=True), config=config)
```

---

## Next Steps

- [Components Reference](./components.md) - Detailed component documentation
- [Agent Patterns](./patterns.md) - Common patterns and workflows
- [Code Examples](./examples.md) - Practical implementation examples
