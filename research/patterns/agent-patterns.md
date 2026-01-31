# Agent Design Patterns

This document covers core reasoning and acting patterns used by individual AI agents across major frameworks.

## 1. ReAct (Reasoning + Acting)

### Overview

The ReAct pattern is the foundational agent pattern that alternates between reasoning steps and actions. The agent:
1. **Reasons** about the current state and what information/action is needed
2. **Acts** by calling a tool or producing output
3. **Observes** the result
4. Repeats until the task is complete

### How It Works

```
User Query -> Reason -> Act (Tool Call) -> Observe -> Reason -> Act (Final Answer)
```

The agent maintains a scratchpad of intermediate thoughts and observations, using them to make informed decisions about next steps.

### Implementation Examples

#### LangChain Zero-Shot ReAct Agent

```python
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

agent = initialize_agent(
    tools=toolkit.get_tools(),
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    return_intermediate_steps=True
)
```

#### LangGraph ReAct Agent

```python
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for {query}"

class AgentState(TypedDict):
    input: str
    agent_outcome: Optional[AgentOutcome]
    intermediate_steps: List[Tuple[AgentAction, str]]

def should_continue(state: AgentState) -> str:
    if isinstance(state["agent_outcome"], AgentFinish):
        return "end"
    return "continue"

workflow = StateGraph(AgentState)
workflow.add_node("agent", run_agent)
workflow.add_conditional_edges("agent", should_continue, {
    "end": END,
    "continue": "tools"
})
workflow.add_node("tools", execute_tools)
workflow.add_edge("tools", "agent")
```

### Best Practices

- **Clear Tool Descriptions**: The agent relies on tool descriptions to decide which tool to use
- **Verbose Mode**: Enable verbose logging during development to understand reasoning
- **Limit Iterations**: Set max_iterations to prevent infinite loops
- **Temperature 0**: Use low temperature for consistent reasoning

---

## 2. Plan-and-Execute

### Overview

The Plan-and-Execute pattern separates planning from execution:
1. **Planning Phase**: Create a high-level plan with steps
2. **Execution Phase**: Execute each step, potentially re-planning if needed

This is more suitable for complex, multi-step tasks where upfront planning improves outcomes.

### How It Works

```
User Query -> Planner Agent -> [Step 1, Step 2, Step 3, ...]
           -> Executor Agent -> Execute Step 1 -> Result 1
           -> Executor Agent -> Execute Step 2 -> Result 2
           -> ... -> Final Answer
```

### Implementation Example

```python
from langgraph.graph import StateGraph, END

class PlanExecuteState(TypedDict):
    input: str
    plan: List[str]
    current_step: int
    results: List[str]
    final_answer: str

def planner(state: PlanExecuteState):
    """Create a plan for the task."""
    plan = llm.invoke(f"Create a step-by-step plan for: {state['input']}")
    return {"plan": parse_plan(plan), "current_step": 0}

def executor(state: PlanExecuteState):
    """Execute the current step."""
    step = state["plan"][state["current_step"]]
    result = execute_step(step)
    return {
        "results": state["results"] + [result],
        "current_step": state["current_step"] + 1
    }

def should_continue(state: PlanExecuteState):
    if state["current_step"] >= len(state["plan"]):
        return "synthesize"
    return "executor"

workflow = StateGraph(PlanExecuteState)
workflow.add_node("planner", planner)
workflow.add_node("executor", executor)
workflow.add_node("synthesize", synthesize_results)
workflow.add_edge("planner", "executor")
workflow.add_conditional_edges("executor", should_continue)
workflow.add_edge("synthesize", END)
```

### When to Use

- Complex tasks with multiple independent steps
- Tasks requiring resource allocation before execution
- When task decomposition improves accuracy
- Long-running tasks that benefit from checkpointing

---

## 3. Reflection Patterns

### Overview

Reflection patterns enable agents to evaluate and improve their own outputs:
1. Generate initial response
2. Critique the response
3. Revise based on critique
4. Repeat until satisfactory

### Self-Reflection Implementation

```python
from langgraph.graph import StateGraph, END

class ReflectionState(TypedDict):
    input: str
    draft: str
    critique: str
    revision_count: int
    final_output: str

def generate_draft(state: ReflectionState):
    """Generate initial response."""
    draft = llm.invoke(f"Answer this question: {state['input']}")
    return {"draft": draft, "revision_count": 0}

def critique(state: ReflectionState):
    """Critique the current draft."""
    critique = llm.invoke(f"""
    Critique this response for accuracy, completeness, and clarity:

    Question: {state['input']}
    Response: {state['draft']}

    Provide specific suggestions for improvement.
    """)
    return {"critique": critique}

def revise(state: ReflectionState):
    """Revise based on critique."""
    revised = llm.invoke(f"""
    Improve this response based on the critique:

    Original: {state['draft']}
    Critique: {state['critique']}

    Provide an improved response.
    """)
    return {"draft": revised, "revision_count": state["revision_count"] + 1}

def should_continue(state: ReflectionState):
    if state["revision_count"] >= 3:
        return "finalize"
    if "no improvements needed" in state["critique"].lower():
        return "finalize"
    return "revise"

workflow = StateGraph(ReflectionState)
workflow.add_node("generate", generate_draft)
workflow.add_node("critique", critique)
workflow.add_node("revise", revise)
workflow.add_node("finalize", lambda s: {"final_output": s["draft"]})

workflow.add_edge("generate", "critique")
workflow.add_conditional_edges("critique", should_continue)
workflow.add_edge("revise", "critique")
workflow.add_edge("finalize", END)
```

### Types of Reflection

| Type | Description | Use Case |
|------|-------------|----------|
| Self-Critique | Agent evaluates its own output | Quality improvement |
| Peer Review | Another agent evaluates output | Multi-agent systems |
| Execution Reflection | Reflect on action outcomes | Learning from mistakes |
| Meta-Cognitive | Reflect on reasoning process | Complex reasoning tasks |

---

## 4. Self-Critique Patterns

### Overview

Self-critique is a specialized form of reflection where the agent explicitly validates its outputs against criteria.

### Implementation with Guardrails

```python
from crewai import Task, TaskOutput
import json

def validate_json_output(result: TaskOutput) -> Tuple[bool, Any]:
    """Validate and parse JSON output."""
    try:
        data = json.loads(result)
        # Additional validation logic
        if "required_field" not in data:
            return (False, "Missing required_field")
        return (True, data)
    except json.JSONDecodeError as e:
        return (False, f"Invalid JSON: {e}")

task = Task(
    description="Generate a JSON report",
    expected_output="A valid JSON object with required_field",
    agent=analyst,
    guardrail=validate_json_output,
    guardrail_max_retries=3  # Retry up to 3 times
)
```

### Critique Dimensions

Common dimensions for self-critique:
- **Accuracy**: Is the information correct?
- **Completeness**: Are all aspects addressed?
- **Relevance**: Does it answer the question?
- **Clarity**: Is it well-organized and understandable?
- **Safety**: Does it avoid harmful content?

---

## 5. Tool Use Patterns

### Overview

Tool use is fundamental to agent capabilities. Agents use tools to:
- Access external information (search, databases)
- Perform actions (send emails, create files)
- Execute code
- Interact with APIs

### Basic Tool Definition

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, description="Maximum results")

@tool("web_search", args_schema=SearchInput)
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for information."""
    # Implementation
    return f"Search results for: {query}"
```

### Tool Selection Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| Zero-Shot | Select based on description | Simple tool sets |
| Few-Shot | Include examples | Complex tools |
| Dynamic | Filter tools at runtime | Large tool sets |
| Hierarchical | Organize tools in categories | Many related tools |

---

## 6. Agentic Loops

### Overview

Agentic loops are iterative patterns where agents repeatedly take actions until a goal is met.

### Implementation

```python
from langgraph.graph import StateGraph, END

class LoopState(TypedDict):
    input: str
    iterations: int
    max_iterations: int
    output: str
    is_complete: bool

def agent_step(state: LoopState):
    """Execute one agent step."""
    result = agent.invoke(state["input"])
    is_complete = check_completion(result)
    return {
        "output": result,
        "iterations": state["iterations"] + 1,
        "is_complete": is_complete
    }

def should_continue(state: LoopState):
    if state["is_complete"]:
        return END
    if state["iterations"] >= state["max_iterations"]:
        return END
    return "agent"

workflow = StateGraph(LoopState)
workflow.add_node("agent", agent_step)
workflow.add_conditional_edges("agent", should_continue)
workflow.set_entry_point("agent")
```

### Loop Control Best Practices

1. **Set Maximum Iterations**: Prevent infinite loops
2. **Clear Exit Conditions**: Define when the task is complete
3. **Progress Tracking**: Log iteration progress
4. **Timeout Handling**: Set overall time limits
5. **Checkpoint State**: Save state for recovery

---

## Pattern Comparison

| Pattern | Reasoning | Planning | Iteration | Best For |
|---------|-----------|----------|-----------|----------|
| ReAct | Per-step | None | Tool loop | General tasks |
| Plan-Execute | Minimal | Upfront | Plan steps | Complex tasks |
| Reflection | Critique | None | Improvement | Quality-critical |
| Self-Critique | Validation | None | Retry | Structured output |
| Agentic Loop | Continuous | None | Goal-driven | Iterative tasks |

## Framework Support

| Pattern | LangChain | LangGraph | CrewAI | AutoGen |
|---------|-----------|-----------|--------|---------|
| ReAct | Native | Native | Via process | Via agents |
| Plan-Execute | Template | Custom | Hierarchical | Custom |
| Reflection | Custom | Native | Via tasks | Custom |
| Self-Critique | Custom | Native | Guardrails | Custom |
| Agentic Loop | Limited | Native | Sequential | Native |
