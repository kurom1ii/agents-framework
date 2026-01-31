# LangChain Agent Patterns

This document covers common architectural patterns used in LangChain agent development, including single vs. multi-agent systems, sequential chains, router patterns, and human-in-the-loop workflows.

---

## Table of Contents

1. [Single Agent Pattern](#single-agent-pattern)
2. [Multi-Agent Patterns](#multi-agent-patterns)
3. [Sequential Chains](#sequential-chains)
4. [Router Patterns](#router-patterns)
5. [Plan-and-Execute Pattern](#plan-and-execute-pattern)
6. [Human-in-the-Loop Pattern](#human-in-the-loop-pattern)
7. [Workflow Patterns with LangGraph](#workflow-patterns-with-langgraph)

---

## Single Agent Pattern

The simplest pattern where one agent handles all tasks.

### Basic Single Agent

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

agent = create_agent(
    model="gpt-4o",
    tools=[search, calculator],
    system_prompt="""You are a helpful assistant that can search the web
    and perform calculations. Use the appropriate tool for each task."""
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is the population of Tokyo times 2?"}]
})
```

### Single Agent with Memory

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

agent = create_agent(
    model="gpt-4o",
    tools=[search, calculator],
    system_prompt="You are a helpful assistant.",
    checkpointer=checkpointer
)

# Conversation persists across calls
config = {"configurable": {"thread_id": "user-123"}}

agent.invoke({"messages": [{"role": "user", "content": "My name is Alice"}]}, config)
agent.invoke({"messages": [{"role": "user", "content": "What's my name?"}]}, config)
# Agent remembers: "Your name is Alice"
```

### When to Use Single Agent

| Use Case | Recommendation |
|----------|----------------|
| Simple tasks | Ideal |
| < 5 tools | Ideal |
| Homogeneous tasks | Ideal |
| Complex multi-domain | Consider multi-agent |
| Many specialized tools | Consider multi-agent |

---

## Multi-Agent Patterns

Multi-agent systems coordinate specialized agents to handle complex tasks.

### Pattern 1: Supervisor Pattern

A central supervisor delegates tasks to specialized sub-agents.

```
                    +------------------+
                    |   Supervisor     |
                    |     Agent        |
                    +------------------+
                           |
           +---------------+---------------+
           |               |               |
           v               v               v
    +------------+  +------------+  +------------+
    |  Calendar  |  |   Email    |  |  Research  |
    |   Agent    |  |   Agent    |  |   Agent    |
    +------------+  +------------+  +------------+
```

```python
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

model = init_chat_model("claude-haiku-4-5-20251001")

# Step 1: Define low-level tools
@tool
def create_calendar_event(title: str, start_time: str, end_time: str) -> str:
    """Create a calendar event."""
    return f"Event created: {title} from {start_time} to {end_time}"

@tool
def send_email(to: list[str], subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {', '.join(to)} - Subject: {subject}"

# Step 2: Create specialized sub-agents
calendar_agent = create_agent(
    model,
    tools=[create_calendar_event],
    system_prompt="""You are a calendar scheduling assistant.
    Parse natural language requests into proper datetime formats.
    Always confirm what was scheduled."""
)

email_agent = create_agent(
    model,
    tools=[send_email],
    system_prompt="""You are an email assistant.
    Compose professional emails and confirm what was sent."""
)

# Step 3: Wrap sub-agents as tools for supervisor
@tool
def schedule_event(request: str) -> str:
    """Schedule calendar events using natural language."""
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text

@tool
def manage_email(request: str) -> str:
    """Send emails using natural language."""
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text

# Step 4: Create supervisor agent
supervisor = create_agent(
    model,
    tools=[schedule_event, manage_email],
    system_prompt="""You are a personal assistant.
    You can schedule events and send emails.
    Break down complex requests into appropriate tool calls."""
)

# Execute multi-step request
result = supervisor.invoke({
    "messages": [{
        "role": "user",
        "content": "Schedule a meeting with design team Tuesday 2pm and email them a reminder"
    }]
})
```

### Pattern 2: Parallel Agents

Multiple agents work independently on different aspects of a task.

```python
import asyncio
from langchain.agents import create_agent

# Create specialized agents
research_agent = create_agent(
    model="gpt-4o",
    tools=[web_search],
    system_prompt="Research and summarize topics."
)

analysis_agent = create_agent(
    model="gpt-4o",
    tools=[data_analyzer],
    system_prompt="Analyze data and provide insights."
)

async def parallel_analysis(topic: str):
    """Run multiple agents in parallel."""
    research_task = asyncio.create_task(
        research_agent.ainvoke({
            "messages": [{"role": "user", "content": f"Research: {topic}"}]
        })
    )
    analysis_task = asyncio.create_task(
        analysis_agent.ainvoke({
            "messages": [{"role": "user", "content": f"Analyze trends in: {topic}"}]
        })
    )

    research_result, analysis_result = await asyncio.gather(
        research_task, analysis_task
    )

    return {
        "research": research_result["messages"][-1].content,
        "analysis": analysis_result["messages"][-1].content
    }
```

### Pattern 3: Pipeline Agents

Agents process data sequentially, each building on the previous output.

```python
# Agent 1: Research
research_agent = create_agent(
    model="gpt-4o",
    tools=[web_search],
    system_prompt="Research topics and gather information."
)

# Agent 2: Summarize
summary_agent = create_agent(
    model="gpt-4o",
    tools=[],
    system_prompt="Summarize research into key points."
)

# Agent 3: Write
writer_agent = create_agent(
    model="gpt-4o",
    tools=[],
    system_prompt="Write polished content from summaries."
)

def pipeline(topic: str):
    # Step 1: Research
    research = research_agent.invoke({
        "messages": [{"role": "user", "content": f"Research: {topic}"}]
    })

    # Step 2: Summarize
    summary = summary_agent.invoke({
        "messages": [{
            "role": "user",
            "content": f"Summarize this research: {research['messages'][-1].content}"
        }]
    })

    # Step 3: Write
    final = writer_agent.invoke({
        "messages": [{
            "role": "user",
            "content": f"Write an article from: {summary['messages'][-1].content}"
        }]
    })

    return final["messages"][-1].content
```

---

## Sequential Chains

Link multiple processing steps where output flows to input.

### Basic Sequential Chain

```python
from langchain_classic.chains import SequentialChain
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# Chain 1: Generate outline
outline_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Create an outline for an article about: {topic}"
)
outline_chain = LLMChain(llm=llm, prompt=outline_prompt, output_key="outline")

# Chain 2: Write from outline
write_prompt = PromptTemplate(
    input_variables=["outline"],
    template="Write an article based on this outline:\n{outline}"
)
write_chain = LLMChain(llm=llm, prompt=write_prompt, output_key="article")

# Combine chains
sequential_chain = SequentialChain(
    chains=[outline_chain, write_chain],
    input_variables=["topic"],
    output_variables=["outline", "article"]
)

result = sequential_chain.invoke({"topic": "AI Agents"})
print(result["article"])
```

### LangGraph Sequential Pattern

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    topic: str
    outline: str
    article: str

def generate_outline(state: State):
    result = llm.invoke(f"Create outline for: {state['topic']}")
    return {"outline": result.content}

def write_article(state: State):
    result = llm.invoke(f"Write article from: {state['outline']}")
    return {"article": result.content}

# Build graph
builder = StateGraph(State)
builder.add_node("outline", generate_outline)
builder.add_node("write", write_article)
builder.add_edge(START, "outline")
builder.add_edge("outline", "write")
builder.add_edge("write", END)

chain = builder.compile()
result = chain.invoke({"topic": "AI Agents"})
```

---

## Router Patterns

Route requests to different handlers based on content or context.

### LLM-Based Router

```python
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from typing import Literal

# Schema for routing decision
class Route(BaseModel):
    destination: Literal["story", "joke", "poem"]

# Router LLM with structured output
router = llm.with_structured_output(Route)

class State(TypedDict):
    input: str
    decision: str
    output: str

def route_input(state: State):
    """Determine which handler to use."""
    decision = router.invoke([
        {"role": "system", "content": "Route to story, joke, or poem based on request."},
        {"role": "user", "content": state["input"]}
    ])
    return {"decision": decision.destination}

def write_story(state: State):
    result = llm.invoke(f"Write a story about: {state['input']}")
    return {"output": result.content}

def write_joke(state: State):
    result = llm.invoke(f"Write a joke about: {state['input']}")
    return {"output": result.content}

def write_poem(state: State):
    result = llm.invoke(f"Write a poem about: {state['input']}")
    return {"output": result.content}

def route_decision(state: State):
    """Return the node to visit based on decision."""
    return f"write_{state['decision']}"

# Build router graph
builder = StateGraph(State)
builder.add_node("router", route_input)
builder.add_node("write_story", write_story)
builder.add_node("write_joke", write_joke)
builder.add_node("write_poem", write_poem)

builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    route_decision,
    {
        "write_story": "write_story",
        "write_joke": "write_joke",
        "write_poem": "write_poem"
    }
)
builder.add_edge("write_story", END)
builder.add_edge("write_joke", END)
builder.add_edge("write_poem", END)

router_graph = builder.compile()
```

### Knowledge Base Router

Route questions to appropriate knowledge sources:

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def search_api_docs(query: str) -> str:
    """Search API documentation."""
    # Search API docs vector store
    return "API documentation results..."

@tool
def search_tutorials(query: str) -> str:
    """Search tutorial content."""
    return "Tutorial results..."

@tool
def search_faq(query: str) -> str:
    """Search frequently asked questions."""
    return "FAQ results..."

# Router agent selects appropriate source
router_agent = create_agent(
    model="gpt-4o",
    tools=[search_api_docs, search_tutorials, search_faq],
    system_prompt="""You are a documentation assistant.
    Route questions to the most appropriate knowledge source:
    - API docs: for technical API questions
    - Tutorials: for how-to guides and examples
    - FAQ: for common questions and troubleshooting"""
)
```

---

## Plan-and-Execute Pattern

Break complex tasks into subtasks, then execute each.

### LangGraph Plan-Execute

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from pydantic import BaseModel

class Step(BaseModel):
    description: str
    completed: bool = False

class PlanExecuteState(TypedDict):
    objective: str
    plan: List[Step]
    current_step: int
    results: List[str]
    final_answer: str

def create_plan(state: PlanExecuteState):
    """Generate execution plan."""
    planner = llm.with_structured_output(List[Step])
    plan = planner.invoke([
        {"role": "system", "content": "Break this objective into 3-5 steps."},
        {"role": "user", "content": state["objective"]}
    ])
    return {"plan": plan, "current_step": 0, "results": []}

def execute_step(state: PlanExecuteState):
    """Execute current step."""
    step = state["plan"][state["current_step"]]
    context = "\n".join(state["results"])

    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": f"Context: {context}\n\nExecute: {step.description}"
        }]
    })

    new_results = state["results"] + [result["messages"][-1].content]
    return {
        "results": new_results,
        "current_step": state["current_step"] + 1
    }

def synthesize(state: PlanExecuteState):
    """Synthesize final answer from results."""
    result = llm.invoke([
        {"role": "system", "content": "Synthesize these results into a final answer."},
        {"role": "user", "content": "\n\n".join(state["results"])}
    ])
    return {"final_answer": result.content}

def should_continue(state: PlanExecuteState):
    """Check if more steps remain."""
    if state["current_step"] >= len(state["plan"]):
        return "synthesize"
    return "execute"

# Build plan-execute graph
builder = StateGraph(PlanExecuteState)
builder.add_node("plan", create_plan)
builder.add_node("execute", execute_step)
builder.add_node("synthesize", synthesize)

builder.add_edge(START, "plan")
builder.add_edge("plan", "execute")
builder.add_conditional_edges(
    "execute",
    should_continue,
    {"execute": "execute", "synthesize": "synthesize"}
)
builder.add_edge("synthesize", END)

plan_execute = builder.compile()
```

---

## Human-in-the-Loop Pattern

Pause execution for human review, approval, or input.

### Basic Approval Flow

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal, Optional, TypedDict

class ApprovalState(TypedDict):
    action_details: str
    status: Optional[Literal["pending", "approved", "rejected"]]

def approval_node(state: ApprovalState) -> Command[Literal["proceed", "cancel"]]:
    """Pause for human approval."""
    decision = interrupt({
        "question": "Approve this action?",
        "details": state["action_details"],
    })

    # Route based on human decision
    return Command(goto="proceed" if decision else "cancel")

def proceed_node(state: ApprovalState):
    return {"status": "approved"}

def cancel_node(state: ApprovalState):
    return {"status": "rejected"}

# Build approval graph
builder = StateGraph(ApprovalState)
builder.add_node("approval", approval_node)
builder.add_node("proceed", proceed_node)
builder.add_node("cancel", cancel_node)
builder.add_edge(START, "approval")
builder.add_edge("proceed", END)
builder.add_edge("cancel", END)

checkpointer = MemorySaver()
approval_graph = builder.compile(checkpointer=checkpointer)

# Execute - pauses at approval
config = {"configurable": {"thread_id": "approval-123"}}
initial = approval_graph.invoke(
    {"action_details": "Transfer $500", "status": "pending"},
    config=config
)
print(initial["__interrupt__"])  # Shows approval request

# Resume with decision
resumed = approval_graph.invoke(Command(resume=True), config=config)
print(resumed["status"])  # "approved"
```

### Review and Edit Flow

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

class ReviewState(TypedDict):
    generated_text: str

def review_node(state: ReviewState):
    """Pause for human review and editing."""
    updated = interrupt({
        "instruction": "Review and edit this content",
        "content": state["generated_text"]
    })
    return {"generated_text": updated}

builder = StateGraph(ReviewState)
builder.add_node("review", review_node)
builder.add_edge(START, "review")
builder.add_edge("review", END)

checkpointer = MemorySaver()
review_graph = builder.compile(checkpointer=checkpointer)

# Start with draft
config = {"configurable": {"thread_id": "review-42"}}
initial = review_graph.invoke({"generated_text": "Initial draft"}, config)
# Shows interrupt with content for review

# Resume with edited text
final = review_graph.invoke(
    Command(resume="Improved draft after review"),
    config
)
print(final["generated_text"])  # "Improved draft after review"
```

### Agent with Approval Before Actions

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

agent = create_agent(
    model="gpt-4o",
    tools=[dangerous_action, safe_action],
    checkpointer=checkpointer,
    interrupt_before=["dangerous_action"]  # Pause before this tool
)

config = {"configurable": {"thread_id": "agent-123"}}

# Agent pauses before calling dangerous_action
result = agent.invoke({
    "messages": [{"role": "user", "content": "Delete all files"}]
}, config)

# Review the pending action
print(result["__interrupt__"])  # Shows tool call for review

# Approve and continue
result = agent.invoke(Command(resume=True), config)
```

---

## Workflow Patterns with LangGraph

### State Graph with Retry

```python
from langgraph.types import RetryPolicy
from langgraph.graph import StateGraph

workflow = StateGraph(State)

# Add node with retry policy
workflow.add_node(
    "unreliable_api",
    call_api,
    retry_policy=RetryPolicy(max_attempts=3)
)
```

### Error Recovery Pattern

```python
from langgraph.types import Command

def execute_with_recovery(state: State) -> Command[Literal["agent", "recover"]]:
    try:
        result = run_tool(state['tool_call'])
        return Command(update={"result": result}, goto="agent")
    except ToolError as e:
        # Let agent see error and adjust
        return Command(
            update={"result": f"Error: {str(e)}"},
            goto="agent"
        )
```

### Parallel Execution Pattern

```python
from langgraph.graph import StateGraph
import asyncio

class ParallelState(TypedDict):
    query: str
    results: dict

async def parallel_search(state: ParallelState):
    """Run multiple searches in parallel."""
    async def search_source(name, query):
        # Simulate async search
        return f"{name} results for: {query}"

    results = await asyncio.gather(
        search_source("web", state["query"]),
        search_source("docs", state["query"]),
        search_source("code", state["query"])
    )

    return {"results": dict(zip(["web", "docs", "code"], results))}
```

---

## Pattern Selection Guide

| Pattern | Use Case | Complexity | Scalability |
|---------|----------|------------|-------------|
| Single Agent | Simple tasks, few tools | Low | Limited |
| Supervisor | Multi-domain tasks | Medium | Good |
| Parallel | Independent subtasks | Medium | Excellent |
| Pipeline | Sequential processing | Low | Good |
| Router | Content-based dispatch | Low | Excellent |
| Plan-Execute | Complex multi-step | High | Good |
| Human-in-Loop | High-stakes decisions | Medium | Limited |

---

## Next Steps

- [Code Examples](./examples.md) - Complete implementation examples
- [Architecture](./architecture.md) - Core architecture details
- [Components](./components.md) - Component reference
