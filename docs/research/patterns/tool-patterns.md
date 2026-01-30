# Tool Integration Patterns

This document covers patterns for integrating tools with AI agents, including function calling, tool schemas, dynamic selection, and error handling.

## 1. Function Calling Patterns

### Overview

Function calling allows LLMs to invoke external functions/tools in a structured way. The model outputs structured data (function name + arguments) that can be parsed and executed.

### Basic Tool Definition

#### Using Decorators (LangChain)

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    """Input schema for search tool."""
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, ge=1, le=20, description="Maximum results to return")

@tool("web_search", args_schema=SearchInput)
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for information.

    Args:
        query: The search query
        max_results: Maximum number of results

    Returns:
        Search results as formatted string
    """
    # Implementation
    results = perform_search(query, max_results)
    return format_results(results)
```

#### Using BaseTool Class (CrewAI)

```python
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
from typing import Type, List, Any
import os

class WeatherToolInput(BaseModel):
    """Input schema for weather tool."""
    city: str = Field(..., description="City name")
    units: str = Field(default="celsius", description="Temperature units")

class WeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Get current weather for a city"
    args_schema: Type[BaseModel] = WeatherToolInput

    # Environment variables needed
    env_vars: List[EnvVar] = [
        EnvVar(name="WEATHER_API_KEY", description="API key", required=True)
    ]

    # Package dependencies
    package_dependencies: List[str] = ["requests"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "WEATHER_API_KEY" not in os.environ:
            raise ValueError("WEATHER_API_KEY required")

    def _run(self, city: str, units: str = "celsius") -> str:
        """Synchronous execution."""
        # Implementation
        return f"Weather in {city}: 22 degrees {units}"

    async def _arun(self, city: str, units: str = "celsius") -> str:
        """Async execution."""
        return self._run(city, units)
```

#### Raw Tool Schema

```python
# Manual tool schema definition
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g., 'San Francisco'"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Bind tools to LLM
llm_with_tools = llm.bind(tools=tools)
response = llm_with_tools.invoke(messages)
```

---

## 2. Tool Schema Patterns

### Pydantic Schema with Validation

```python
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class CreateTaskInput(BaseModel):
    """Schema for task creation with validation."""

    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Task title"
    )

    description: Optional[str] = Field(
        None,
        max_length=1000,
        description="Detailed task description"
    )

    priority: Priority = Field(
        default=Priority.MEDIUM,
        description="Task priority level"
    )

    due_date: Optional[str] = Field(
        None,
        description="Due date in YYYY-MM-DD format"
    )

    tags: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Task tags"
    )

    @field_validator("due_date")
    @classmethod
    def validate_date_format(cls, v):
        if v is None:
            return v
        from datetime import datetime
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")
        return v

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v):
        return [tag.lower().strip() for tag in v]
```

### Nested Schemas

```python
from pydantic import BaseModel, Field
from typing import List

class Address(BaseModel):
    street: str = Field(..., description="Street address")
    city: str = Field(..., description="City name")
    country: str = Field(..., description="Country code")
    postal_code: str = Field(..., description="Postal/ZIP code")

class ContactInfo(BaseModel):
    email: str = Field(..., description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")

class CreateCustomerInput(BaseModel):
    """Create a new customer record."""
    name: str = Field(..., description="Customer full name")
    contact: ContactInfo = Field(..., description="Contact information")
    addresses: List[Address] = Field(
        default_factory=list,
        description="Customer addresses"
    )

@tool("create_customer", args_schema=CreateCustomerInput)
def create_customer(name: str, contact: ContactInfo, addresses: List[Address]) -> str:
    """Create a new customer with contact info and addresses."""
    # Implementation
    return f"Created customer: {name}"
```

---

## 3. Dynamic Tool Selection

### Overview

For agents with many tools, dynamic selection filters relevant tools at runtime to reduce prompt size and improve accuracy.

### Decorator-Based Selection

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

def get_relevant_tools(state: dict, all_tools: list) -> list:
    """Select relevant tools based on current state."""
    # Category-based selection
    category = state.get("category", "general")
    category_tools = {
        "finance": ["calculator", "stock_lookup", "currency_convert"],
        "research": ["web_search", "wiki_lookup", "arxiv_search"],
        "communication": ["send_email", "send_slack", "create_ticket"]
    }
    tool_names = category_tools.get(category, [])

    # Filter tools
    return [t for t in all_tools if t.name in tool_names]

@wrap_model_call
def select_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Middleware to dynamically select tools."""
    relevant_tools = get_relevant_tools(request.state, request.tools)
    return handler(request.override(tools=relevant_tools))

agent = create_agent(
    model="gpt-4o",
    tools=all_tools,  # All tools registered
    middleware=[select_tools]
)
```

### Semantic Tool Selection

```python
from langchain_openai import OpenAIEmbeddings
import numpy as np

class SemanticToolSelector:
    def __init__(self, tools: List):
        self.tools = tools
        self.embeddings = OpenAIEmbeddings()

        # Pre-compute tool embeddings from descriptions
        self.tool_embeddings = {}
        for tool in tools:
            desc = f"{tool.name}: {tool.description}"
            self.tool_embeddings[tool.name] = self.embeddings.embed_query(desc)

    def select_tools(self, query: str, k: int = 5) -> List:
        """Select k most relevant tools for the query."""
        query_embedding = self.embeddings.embed_query(query)

        # Calculate similarities
        scores = []
        for tool in self.tools:
            tool_emb = self.tool_embeddings[tool.name]
            similarity = np.dot(query_embedding, tool_emb)
            scores.append((tool, similarity))

        # Sort by similarity and return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [tool for tool, _ in scores[:k]]

# Usage
selector = SemanticToolSelector(all_tools)
relevant_tools = selector.select_tools("I need to search the web for information")
agent = create_agent(model="gpt-4o", tools=relevant_tools)
```

### Hierarchical Tool Organization

```python
class ToolCategory:
    """Organize tools into categories for hierarchical selection."""

    def __init__(self, name: str, description: str, tools: List):
        self.name = name
        self.description = description
        self.tools = tools

# Define categories
categories = [
    ToolCategory(
        name="search",
        description="Tools for searching and retrieving information",
        tools=[web_search, wiki_search, arxiv_search]
    ),
    ToolCategory(
        name="calculation",
        description="Tools for mathematical operations",
        tools=[calculator, unit_converter, statistics]
    ),
    ToolCategory(
        name="communication",
        description="Tools for sending messages and notifications",
        tools=[email, slack, sms]
    )
]

# Two-step selection: first category, then tools
@tool("select_category")
def select_category(task_description: str) -> str:
    """Select the most appropriate tool category for a task."""
    # LLM selects category
    return llm.invoke(f"Which category for: {task_description}")

def get_tools_for_category(category_name: str) -> List:
    """Get tools for a specific category."""
    for cat in categories:
        if cat.name == category_name:
            return cat.tools
    return []
```

---

## 4. Tool Chaining

### Overview

Tool chaining connects multiple tool calls where the output of one becomes input to another.

### Sequential Chaining

```python
from langgraph.graph import StateGraph, END

class ChainState(TypedDict):
    query: str
    search_results: str
    analysis: str
    summary: str

def search_step(state: ChainState):
    """Step 1: Search for information."""
    results = web_search.invoke({"query": state["query"]})
    return {"search_results": results}

def analyze_step(state: ChainState):
    """Step 2: Analyze search results."""
    analysis = analyzer.invoke({"data": state["search_results"]})
    return {"analysis": analysis}

def summarize_step(state: ChainState):
    """Step 3: Summarize analysis."""
    summary = summarizer.invoke({"content": state["analysis"]})
    return {"summary": summary}

workflow = StateGraph(ChainState)
workflow.add_node("search", search_step)
workflow.add_node("analyze", analyze_step)
workflow.add_node("summarize", summarize_step)

workflow.add_edge("search", "analyze")
workflow.add_edge("analyze", "summarize")
workflow.add_edge("summarize", END)

workflow.set_entry_point("search")
chain = workflow.compile()
```

### Conditional Chaining

```python
def route_after_search(state: ChainState) -> str:
    """Decide next step based on search results."""
    if not state["search_results"]:
        return "retry_search"
    if needs_deep_analysis(state["search_results"]):
        return "deep_analyze"
    return "quick_summarize"

workflow = StateGraph(ChainState)
workflow.add_node("search", search_step)
workflow.add_node("retry_search", retry_search_step)
workflow.add_node("deep_analyze", deep_analyze_step)
workflow.add_node("quick_summarize", quick_summarize_step)

workflow.add_conditional_edges("search", route_after_search, {
    "retry_search": "retry_search",
    "deep_analyze": "deep_analyze",
    "quick_summarize": "quick_summarize"
})
```

### Parallel Tool Execution

```python
import asyncio
from typing import List

async def parallel_tool_execution(tools: List, inputs: List[dict]) -> List:
    """Execute multiple tools in parallel."""
    async def execute_tool(tool, input_data):
        return await tool.ainvoke(input_data)

    tasks = [execute_tool(t, i) for t, i in zip(tools, inputs)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "tool": tools[i].name,
                "error": str(result)
            })
        else:
            processed_results.append({
                "tool": tools[i].name,
                "result": result
            })

    return processed_results

# Usage
results = await parallel_tool_execution(
    [search_tool, wiki_tool, news_tool],
    [{"query": q} for q in ["AI agents", "LLM agents", "autonomous AI"]]
)
```

---

## 5. Error Handling and Recovery

### Retry Patterns

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

class ToolWithRetry(BaseTool):
    name = "api_tool"
    description = "Tool with built-in retry logic"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutError, httpx.HTTPStatusError))
    )
    def _run(self, input: str) -> str:
        """Execute with automatic retries on failure."""
        response = httpx.get(f"https://api.example.com/{input}", timeout=10)
        response.raise_for_status()
        return response.json()
```

### CrewAI Guardrails with Retry

```python
from crewai import Task, TaskOutput
from typing import Tuple, Any
import json

def validate_json_output(result: TaskOutput) -> Tuple[bool, Any]:
    """Validate output is valid JSON."""
    try:
        data = json.loads(result.raw)
        # Additional validation
        if "required_field" not in data:
            return (False, "Missing required_field in response")
        return (True, data)
    except json.JSONDecodeError as e:
        return (False, f"Invalid JSON: {e}")

task = Task(
    description="Generate a JSON report",
    expected_output="Valid JSON with required_field",
    agent=analyst,
    guardrail=validate_json_output,
    guardrail_max_retries=3  # Agent retries up to 3 times
)
```

### Fallback Strategies

```python
class ToolWithFallback:
    """Tool that falls back to alternative on failure."""

    def __init__(self, primary_tool, fallback_tool):
        self.primary = primary_tool
        self.fallback = fallback_tool

    async def invoke(self, input: dict) -> str:
        try:
            return await self.primary.ainvoke(input)
        except Exception as e:
            print(f"Primary tool failed: {e}, trying fallback")
            try:
                return await self.fallback.ainvoke(input)
            except Exception as e2:
                return f"Both tools failed. Primary: {e}, Fallback: {e2}"

# Usage
search_with_fallback = ToolWithFallback(
    primary_tool=google_search,
    fallback_tool=bing_search
)
```

### Human Escalation

```python
from langgraph.graph import StateGraph, END

class EscalationState(TypedDict):
    input: str
    tool_result: str
    error: Optional[str]
    requires_human: bool
    human_response: Optional[str]

def execute_tool(state: EscalationState):
    """Execute tool with error handling."""
    try:
        result = tool.invoke(state["input"])
        return {"tool_result": result, "error": None, "requires_human": False}
    except Exception as e:
        # Determine if human escalation needed
        if is_critical_error(e):
            return {
                "error": str(e),
                "requires_human": True,
                "tool_result": None
            }
        return {"error": str(e), "requires_human": False, "tool_result": None}

def human_escalation(state: EscalationState):
    """Request human intervention."""
    # In production, this would trigger notification/workflow
    print(f"Human intervention required for: {state['error']}")
    # Block until human responds (or timeout)
    human_response = wait_for_human_response(state)
    return {"human_response": human_response}

def route_after_tool(state: EscalationState) -> str:
    if state["requires_human"]:
        return "human"
    if state["error"]:
        return "retry"
    return "complete"

workflow = StateGraph(EscalationState)
workflow.add_node("execute", execute_tool)
workflow.add_node("human", human_escalation)
workflow.add_node("retry", retry_step)
workflow.add_node("complete", complete_step)

workflow.add_conditional_edges("execute", route_after_tool)
workflow.add_edge("human", "complete")
workflow.add_edge("retry", "execute")
workflow.add_edge("complete", END)
```

### Graceful Degradation

```python
class GracefulTool(BaseTool):
    """Tool that degrades gracefully on partial failures."""

    name = "resilient_search"
    description = "Search across multiple sources with graceful degradation"

    def _run(self, query: str) -> str:
        sources = [
            ("google", self._search_google),
            ("bing", self._search_bing),
            ("duckduckgo", self._search_ddg)
        ]

        results = []
        errors = []

        for source_name, search_fn in sources:
            try:
                result = search_fn(query)
                results.append(f"[{source_name}] {result}")
            except Exception as e:
                errors.append(f"{source_name}: {e}")

        # Return whatever we got
        if results:
            output = "Search results:\n" + "\n".join(results)
            if errors:
                output += f"\n\nNote: Some sources failed: {errors}"
            return output
        else:
            return f"All search sources failed: {errors}"
```

---

## 6. Tool Best Practices

### 1. Tool Design

```python
# Good: Clear, focused tool
@tool("get_user_email")
def get_user_email(user_id: str) -> str:
    """Get the email address for a specific user.

    Args:
        user_id: The unique identifier for the user

    Returns:
        The user's email address or error message
    """
    pass

# Bad: Overly broad tool
@tool("manage_user")
def manage_user(action: str, user_id: str, data: dict) -> str:
    """Manage user - create, update, delete, or fetch."""
    pass  # Too many responsibilities
```

### 2. Clear Descriptions

```python
# Good description
@tool
def calculate_compound_interest(
    principal: float,
    rate: float,
    time: int,
    compounds_per_year: int = 12
) -> float:
    """Calculate compound interest on an investment.

    Use this tool when you need to calculate how much an investment
    will grow over time with compound interest.

    Args:
        principal: Initial investment amount in dollars
        rate: Annual interest rate as decimal (e.g., 0.05 for 5%)
        time: Number of years
        compounds_per_year: How often interest compounds (default: 12 for monthly)

    Returns:
        Final amount after compound interest

    Example:
        calculate_compound_interest(1000, 0.05, 10, 12)
        Returns: 1647.01 (for $1000 at 5% for 10 years, monthly compounding)
    """
    pass
```

### 3. Input Validation

```python
from pydantic import BaseModel, Field, field_validator

class TransferFundsInput(BaseModel):
    from_account: str = Field(..., description="Source account ID")
    to_account: str = Field(..., description="Destination account ID")
    amount: float = Field(..., gt=0, description="Amount to transfer")
    currency: str = Field(default="USD", description="Currency code")

    @field_validator("from_account", "to_account")
    @classmethod
    def validate_account_format(cls, v):
        if not v.startswith("ACC-"):
            raise ValueError("Account ID must start with 'ACC-'")
        return v

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, v):
        valid_currencies = ["USD", "EUR", "GBP", "JPY"]
        if v not in valid_currencies:
            raise ValueError(f"Currency must be one of {valid_currencies}")
        return v
```

### 4. Error Messages

```python
@tool
def fetch_stock_price(symbol: str) -> str:
    """Fetch current stock price for a ticker symbol."""
    try:
        price = stock_api.get_price(symbol)
        return f"Current price of {symbol}: ${price:.2f}"
    except SymbolNotFoundError:
        return f"Error: Stock symbol '{symbol}' not found. Please verify the ticker symbol."
    except APIRateLimitError:
        return "Error: Rate limit exceeded. Please try again in a few seconds."
    except Exception as e:
        return f"Error fetching stock price: {str(e)}. Please try again."
```

### 5. Idempotency

```python
import hashlib
from functools import lru_cache

class IdempotentTool(BaseTool):
    """Tool that ensures idempotent operations."""

    def __init__(self):
        super().__init__()
        self._executed_operations = {}

    def _generate_operation_key(self, **kwargs) -> str:
        """Generate unique key for operation."""
        content = json.dumps(kwargs, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def _run(self, **kwargs) -> str:
        op_key = self._generate_operation_key(**kwargs)

        # Check if already executed
        if op_key in self._executed_operations:
            return f"Operation already executed. Result: {self._executed_operations[op_key]}"

        # Execute and store
        result = self._execute(**kwargs)
        self._executed_operations[op_key] = result
        return result
```

---

## Framework Comparison

| Feature | LangChain | CrewAI | AutoGen |
|---------|-----------|--------|---------|
| Tool Definition | @tool decorator | BaseTool class | Function def |
| Schema | Pydantic | Pydantic | Dict/Pydantic |
| Async Support | Native | Native | Native |
| Dynamic Selection | Middleware | Custom | Custom |
| Error Handling | Try/catch | Guardrails | Message handling |
| Tool Chaining | LangGraph | Task context | Conversation |
