# LangChain Components Reference

This document provides detailed documentation of LangChain's key components including AgentExecutor, Agent classes, Tools, Callbacks, and Output Parsers.

---

## Table of Contents

1. [AgentExecutor](#agentexecutor)
2. [Agent Classes and Creation](#agent-classes-and-creation)
3. [Tools and Toolkits](#tools-and-toolkits)
4. [Middleware System](#middleware-system)
5. [Callbacks and Handlers](#callbacks-and-handlers)
6. [Output Parsers and Structured Output](#output-parsers-and-structured-output)

---

## AgentExecutor

### Overview

`AgentExecutor` is the legacy runtime that manages agent execution loops. In modern LangChain, agents created with `create_agent()` are built on LangGraph and handle execution internally.

### Legacy AgentExecutor Usage

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_classic import hub

# Create the agent
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)

# Wrap in AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,           # Print execution traces
    max_iterations=15,      # Maximum reasoning loops
    max_execution_time=60,  # Timeout in seconds
    handle_parsing_errors=True,  # Graceful error handling
)

# Execute
result = agent_executor.invoke({"input": "What is the weather in Tokyo?"})
```

### AgentExecutor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent` | Agent | Required | The agent to execute |
| `tools` | List[Tool] | Required | Tools available to the agent |
| `verbose` | bool | False | Print execution traces |
| `max_iterations` | int | 15 | Maximum reasoning loops |
| `max_execution_time` | float | None | Timeout in seconds |
| `handle_parsing_errors` | bool | False | Handle LLM output parsing errors |
| `early_stopping_method` | str | "force" | How to stop if max iterations reached |
| `memory` | Memory | None | Memory component for conversation |
| `return_intermediate_steps` | bool | False | Return all intermediate steps |

### Modern Agent Execution (LangGraph-based)

The modern approach uses `create_agent()` which returns a compiled LangGraph:

```python
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o",
    tools=[search, get_weather],
    system_prompt="You are a helpful assistant"
)

# Direct invocation
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather?"}]
})

# Streaming
for event in agent.stream(
    {"messages": [{"role": "user", "content": "What's the weather?"}]},
    stream_mode="values"
):
    event["messages"][-1].pretty_print()
```

### Execution Loop Comparison

**Legacy AgentExecutor Loop:**
```
Input -> Agent.plan() -> Action -> Tool.run() -> Observation -> Agent.plan() -> ... -> Finish
```

**Modern LangGraph Loop:**
```
Input -> LLM Call -> Tool Calls? -> Execute Tools -> Add Results -> LLM Call -> ... -> Final Response
```

---

## Agent Classes and Creation

### Modern Agent Creation

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

# Initialize model
model = init_chat_model("claude-sonnet-4-5-20250929", model_provider="anthropic")

# Create agent
agent = create_agent(
    model=model,
    tools=[tool1, tool2],
    system_prompt="You are a helpful assistant",
    checkpointer=checkpointer,  # Optional: for state persistence
    middleware=[middleware1],    # Optional: for custom behavior
)
```

### Agent Creation Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | str or ChatModel | Model identifier or initialized model |
| `tools` | List[Tool] | Tools the agent can use |
| `system_prompt` | str | System instructions for the agent |
| `checkpointer` | Checkpointer | State persistence mechanism |
| `middleware` | List[Middleware] | Custom behavior hooks |
| `state_schema` | TypedDict | Custom state schema |
| `response_format` | Schema | Structured output format |

### Legacy Agent Types

For backwards compatibility, legacy agent types are still available:

```python
from langchain.agents import AgentType

# Available types
AgentType.ZERO_SHOT_REACT_DESCRIPTION  # ReAct without examples
AgentType.REACT_DOCSTORE              # ReAct for document stores
AgentType.SELF_ASK_WITH_SEARCH        # Self-ask pattern
AgentType.CONVERSATIONAL_REACT_DESCRIPTION  # Conversational ReAct
AgentType.OPENAI_FUNCTIONS            # OpenAI function calling
AgentType.OPENAI_MULTI_FUNCTIONS      # Multiple function calls
AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT  # Structured chat
```

### Agent Factory Functions

```python
# OpenAI Functions Agent
from langchain.agents import create_openai_functions_agent
agent = create_openai_functions_agent(llm, tools, prompt)

# ReAct Agent
from langchain.agents import create_react_agent
agent = create_react_agent(llm, tools, prompt)

# Structured Chat Agent
from langchain.agents import create_structured_chat_agent
agent = create_structured_chat_agent(llm, tools, prompt)

# Tool Calling Agent (generic)
from langchain.agents import create_tool_calling_agent
agent = create_tool_calling_agent(llm, tools, prompt)
```

---

## Tools and Toolkits

### Tool Definition

Tools are the primary way agents interact with external systems.

#### Basic Tool with Decorator

```python
from langchain.tools import tool

@tool
def search(query: str) -> str:
    """Search for information on the web.

    Args:
        query: The search query string
    """
    # Implementation
    return f"Search results for: {query}"
```

#### Tool with Pydantic Schema

```python
from langchain.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    """Input schema for search tool."""
    query: str = Field(..., description="The search query")
    max_results: int = Field(default=10, description="Maximum results to return")

@tool(args_schema=SearchInput)
def search(query: str, max_results: int = 10) -> str:
    """Search for information with configurable result count."""
    return f"Top {max_results} results for: {query}"
```

#### Tool Class Definition

```python
from langchain.tools import Tool

def search_func(query: str) -> str:
    return f"Results for: {query}"

search_tool = Tool(
    name="search",
    description="Search for information on the web",
    func=search_func,
)
```

#### Structured Tool with Complex Schema

```python
from langchain.tools import StructuredTool
from pydantic import BaseModel

class EmailInput(BaseModel):
    to: list[str]
    subject: str
    body: str
    cc: list[str] = []

def send_email(to: list[str], subject: str, body: str, cc: list[str] = []) -> str:
    return f"Email sent to {', '.join(to)}"

email_tool = StructuredTool.from_function(
    func=send_email,
    name="send_email",
    description="Send an email",
    args_schema=EmailInput,
)
```

### Tool Properties

| Property | Description |
|----------|-------------|
| `name` | Unique identifier for the tool |
| `description` | Describes what the tool does (used by LLM) |
| `args_schema` | Pydantic model defining input schema |
| `return_direct` | If True, return tool result directly without LLM processing |
| `func` | The function to execute |
| `coroutine` | Async function for async execution |

### Toolkits

Toolkits are collections of related tools:

```python
from langchain_community.agent_toolkits import FileManagementToolkit

# Get all tools from toolkit
toolkit = FileManagementToolkit()
tools = toolkit.get_tools()

# Or select specific tools
tools = toolkit.get_tools(
    selected_tools=["read_file", "write_file"]
)
```

### Built-in Tool Categories

| Category | Examples |
|----------|----------|
| **Search** | DuckDuckGo, Google, Bing, Tavily, You.com |
| **Web** | Requests, Wikipedia, ArXiv |
| **Code** | Python REPL, Shell, SQL |
| **File** | File read/write, Directory operations |
| **Math** | Calculator, Wolfram Alpha |
| **APIs** | REST API, GraphQL |
| **Databases** | SQL, Vector stores |

### Dynamic Tools

Tools can be dynamically added or filtered:

```python
from langchain.agents.middleware import before_model

@before_model
def filter_tools_by_permission(state, runtime):
    """Filter tools based on user permissions."""
    user_role = state.get("user_role", "basic")

    if user_role == "admin":
        return None  # All tools available

    # Filter to basic tools only
    basic_tools = [t for t in runtime.tools if t.name in ["search", "weather"]]
    return {"tools": basic_tools}
```

---

## Middleware System

Middleware provides hooks for customizing agent behavior at various stages.

### Middleware Types

| Type | Decorator | Purpose |
|------|-----------|---------|
| `before_model` | `@before_model` | Modify state before LLM call |
| `after_model` | `@after_model` | Process LLM response |
| `wrap_tool_call` | `@wrap_tool_call` | Wrap individual tool executions |

### Before Model Middleware

```python
from langchain.agents.middleware import before_model

@before_model
def add_context(state, runtime):
    """Add context to messages before LLM call."""
    current_time = datetime.now().isoformat()
    return {
        "messages": [
            {"role": "system", "content": f"Current time: {current_time}"},
            *state["messages"]
        ]
    }
```

### After Model Middleware

```python
from langchain.agents.middleware import after_model

@after_model
def log_response(state, response, runtime):
    """Log model responses."""
    print(f"Model response: {response.content}")
    return None  # No state modifications
```

### Wrap Tool Call Middleware

```python
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors."""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Error: {str(e)}. Please try again.",
            tool_call_id=request.tool_call["id"]
        )

@wrap_tool_call
def log_tool_calls(request, handler):
    """Log all tool calls."""
    print(f"Calling tool: {request.tool_call['name']}")
    result = handler(request)
    print(f"Tool result: {result.content}")
    return result
```

### Combining Middleware

```python
agent = create_agent(
    model="gpt-4o",
    tools=[search, weather],
    middleware=[
        add_context,
        handle_tool_errors,
        log_tool_calls,
    ]
)
```

---

## Callbacks and Handlers

Callbacks provide hooks for monitoring and debugging agent execution.

### Callback Handler Interface

```python
from langchain.callbacks.base import BaseCallbackHandler

class CustomHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when LLM starts."""
        print(f"LLM starting with {len(prompts)} prompts")

    def on_llm_end(self, response, **kwargs):
        """Called when LLM ends."""
        print(f"LLM finished: {response}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        """Called when tool starts."""
        print(f"Tool starting: {serialized['name']}")

    def on_tool_end(self, output, **kwargs):
        """Called when tool ends."""
        print(f"Tool output: {output}")

    def on_agent_action(self, action, **kwargs):
        """Called when agent takes action."""
        print(f"Agent action: {action}")

    def on_agent_finish(self, finish, **kwargs):
        """Called when agent finishes."""
        print(f"Agent finished: {finish}")
```

### Using Callbacks

```python
# With agent invocation
result = agent.invoke(
    {"messages": [...]},
    config={"callbacks": [CustomHandler()]}
)

# Global callbacks
from langchain.globals import set_verbose, set_debug

set_verbose(True)  # Print basic info
set_debug(True)    # Print detailed debug info
```

### Built-in Handlers

| Handler | Description |
|---------|-------------|
| `StdOutCallbackHandler` | Print to stdout |
| `FileCallbackHandler` | Write to file |
| `StreamingStdOutCallbackHandler` | Stream tokens to stdout |
| `LangChainTracer` | Send traces to LangSmith |

### LangSmith Integration

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"

# Traces are automatically sent to LangSmith
agent = create_agent(model="gpt-4o", tools=tools)
result = agent.invoke({"messages": [...]})
```

---

## Output Parsers and Structured Output

### Structured Output Strategies

LangChain provides two main strategies for structured output:

#### 1. ToolStrategy

Uses synthetic tool calls to force structured responses:

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field
from typing import Literal

class ProductReview(BaseModel):
    """Analysis of a product review."""
    rating: int | None = Field(description="Rating 1-5", ge=1, le=5)
    sentiment: Literal["positive", "negative"]
    key_points: list[str] = Field(description="Key points, 1-3 words each")

agent = create_agent(
    model="gpt-4o",
    tools=[],
    response_format=ToolStrategy(ProductReview)
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Analyze: 'Great product, 5 stars. Fast shipping, expensive'"
    }]
})

print(result["structured_response"])
# ProductReview(rating=5, sentiment='positive', key_points=['fast shipping', 'expensive'])
```

#### 2. ProviderStrategy

Uses model provider's native structured output:

```python
from langchain.agents.structured_output import ProviderStrategy

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model="gpt-4o",
    tools=[],
    response_format=ProviderStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Extract: John Doe, john@example.com, (555) 123-4567"
    }]
})

print(result["structured_response"])
# {'name': 'John Doe', 'email': 'john@example.com', 'phone': '(555) 123-4567'}
```

### Schema Definition Options

```python
# 1. Pydantic BaseModel
from pydantic import BaseModel

class Response(BaseModel):
    answer: str
    confidence: float

# 2. TypedDict
from typing_extensions import TypedDict

class Response(TypedDict):
    answer: str
    confidence: float

# 3. Dataclass
from dataclasses import dataclass

@dataclass
class Response:
    answer: str
    confidence: float

# 4. JSON Schema (dict)
response_schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"}
    },
    "required": ["answer", "confidence"]
}
```

### Union Types for Multiple Schemas

```python
from pydantic import BaseModel
from typing import Literal

class ProductReview(BaseModel):
    rating: int
    sentiment: Literal["positive", "negative"]

class CustomerComplaint(BaseModel):
    issue_type: Literal["product", "service", "shipping"]
    severity: Literal["low", "medium", "high"]

# Agent can return either type
agent = create_agent(
    model="gpt-4o",
    tools=[],
    response_format=ToolStrategy([ProductReview, CustomerComplaint])
)
```

### Legacy Output Parsers

For legacy chains, output parsers are still available:

```python
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

parser = PydanticOutputParser(pydantic_object=ProductReview)

prompt = PromptTemplate(
    template="Analyze this review:\n{review}\n\n{format_instructions}",
    input_variables=["review"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
```

---

## Component Integration Example

Here's how all components work together:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, wrap_tool_call
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

# Define tools
@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

# Define middleware
@wrap_tool_call
def log_tools(request, handler):
    print(f"Tool: {request.tool_call['name']}")
    return handler(request)

# Define output schema
class Response(BaseModel):
    answer: str
    sources: list[str]

# Create agent with all components
agent = create_agent(
    model="gpt-4o",
    tools=[search],
    system_prompt="You are a research assistant.",
    middleware=[log_tools],
    checkpointer=MemorySaver(),
    response_format=ToolStrategy(Response),
)

# Execute with callbacks
from langchain.callbacks import StdOutCallbackHandler

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is LangChain?"}]},
    config={
        "configurable": {"thread_id": "research-1"},
        "callbacks": [StdOutCallbackHandler()]
    }
)

print(result["structured_response"])
```

---

## Next Steps

- [Agent Patterns](./patterns.md) - Common patterns and workflows
- [Code Examples](./examples.md) - Practical implementation examples
