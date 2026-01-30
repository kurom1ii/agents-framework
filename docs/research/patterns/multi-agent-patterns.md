# Multi-Agent Design Patterns

This document covers coordination patterns for systems with multiple AI agents working together.

## 1. Hierarchical Patterns (Supervisor/Worker)

### Overview

In hierarchical patterns, a supervisor agent coordinates and delegates tasks to worker agents. The supervisor:
- Receives the initial task
- Breaks it down into subtasks
- Delegates to specialized workers
- Aggregates results

### Architecture

```
                    ┌──────────────┐
                    │  Supervisor  │
                    │    Agent     │
                    └──────┬───────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Worker 1 │    │ Worker 2 │    │ Worker 3 │
    │(Research)│    │ (Writer) │    │(Analyst) │
    └──────────┘    └──────────┘    └──────────┘
```

### LangGraph Implementation

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph_supervisor import create_supervisor

# Define worker agents
research_agent = create_react_agent(
    model=model,
    tools=[search_tool, wiki_tool],
    name="researcher"
)

math_agent = create_react_agent(
    model=model,
    tools=[calculator_tool],
    name="mathematician"
)

# Create supervisor workflow
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt="You are a team supervisor managing a research expert and a math expert."
)

# Add memory for persistence
checkpointer = InMemorySaver()
store = InMemoryStore()

app = workflow.compile(
    checkpointer=checkpointer,
    store=store
)
```

### CrewAI Implementation

```python
from crewai import Agent, Crew, Task, Process

# Manager agent
manager = Agent(
    role="Project Manager",
    goal="Coordinate team efforts and ensure project success",
    backstory="Experienced project manager skilled at delegation",
    allow_delegation=True,
    verbose=True
)

# Specialist agents
researcher = Agent(
    role="Researcher",
    goal="Provide accurate research and analysis",
    backstory="Expert researcher with deep analytical skills",
    allow_delegation=False
)

writer = Agent(
    role="Writer",
    goal="Create compelling content",
    backstory="Skilled writer who creates engaging content",
    allow_delegation=False
)

# Create hierarchical crew
crew = Crew(
    agents=[manager, researcher, writer],
    tasks=[project_task],
    process=Process.hierarchical,
    manager_llm="gpt-4o",
    verbose=True
)
```

### Multi-Level Hierarchies

```python
from langgraph_supervisor import create_supervisor

# Create team-level supervisors
research_team = create_supervisor(
    [research_agent, math_agent],
    model=model,
    supervisor_name="research_supervisor"
).compile(name="research_team")

writing_team = create_supervisor(
    [writing_agent, publishing_agent],
    model=model,
    supervisor_name="writing_supervisor"
).compile(name="writing_team")

# Create top-level supervisor
top_level_supervisor = create_supervisor(
    [research_team, writing_team],
    model=model,
    supervisor_name="top_level_supervisor"
).compile(name="top_level_supervisor")
```

### Best Practices

1. **Clear Role Definitions**: Each worker should have a specific, well-defined role
2. **Limited Scope**: Workers should focus on their expertise
3. **Explicit Delegation**: Supervisor should explicitly state which worker to use
4. **Result Aggregation**: Supervisor should synthesize worker outputs

---

## 2. Collaborative Patterns (Peer-to-Peer)

### Overview

Collaborative patterns enable agents to work together as peers, sharing information and building on each other's work without a central coordinator.

### Swarm Architecture

```
    ┌──────────┐     ┌──────────┐
    │  Agent A │◄───►│  Agent B │
    └────┬─────┘     └────┬─────┘
         │                │
         │    ┌──────────┐│
         └───►│  Agent C │◄┘
              └──────────┘
```

### LangGraph Swarm Implementation

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph_swarm import create_swarm

# Create agents with handoff capabilities
alice = create_react_agent(
    model=model,
    tools=[search_tool, create_handoff_tool(bob)],
    name="Alice"
)

bob = create_react_agent(
    model=model,
    tools=[analysis_tool, create_handoff_tool(alice)],
    name="Bob"
)

# Create swarm
workflow = create_swarm(
    [alice, bob],
    default_active_agent="Alice"
)

# Compile with memory
checkpointer = InMemorySaver()
store = InMemoryStore()

app = workflow.compile(
    checkpointer=checkpointer,
    store=store
)
```

### AutoGen Conversation Pattern

```python
from autogen_core import (
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler
)

@dataclass
class Message:
    content: str
    sender: str

class CollaborativeAgent(RoutedAgent):
    def __init__(self, model_client, neighbors):
        super().__init__("Collaborative agent")
        self._model_client = model_client
        self._neighbors = neighbors
        self._history = []

    @message_handler
    async def handle_message(self, message: Message, ctx):
        # Process message
        self._history.append(message)

        # Generate response
        response = await self._model_client.create(
            self._system_messages + self._history
        )

        # Share with neighbors if needed
        for neighbor in self._neighbors:
            await self.send_message(
                Message(content=response, sender=self.id),
                neighbor
            )

        return response
```

### Collaboration Patterns

| Pattern | Description | Use Case |
|---------|-------------|----------|
| Round-Robin | Agents take turns | Sequential refinement |
| Broadcast | All agents receive all messages | Consensus building |
| Handoff | Explicit transfer of control | Specialized processing |
| Debate | Agents argue positions | Decision making |

---

## 3. Competitive Patterns

### Overview

Competitive patterns have multiple agents working on the same problem, with their outputs compared or aggregated.

### Multi-Agent Debate

```python
from autogen_core import RoutedAgent, default_subscription

@dataclass
class SolverRequest:
    content: str
    question: str

@dataclass
class IntermediateSolverResponse:
    content: str
    question: str
    answer: str
    round: int

@default_subscription
class DebateSolver(RoutedAgent):
    def __init__(self, model_client, num_neighbors, max_rounds):
        super().__init__("Debate solver")
        self._model_client = model_client
        self._num_neighbors = num_neighbors
        self._max_rounds = max_rounds
        self._history = []
        self._buffer = {}
        self._round = 0

    @message_handler
    async def handle_request(self, message: SolverRequest, ctx):
        # Generate solution
        self._history.append(UserMessage(content=message.content))
        response = await self._model_client.create(
            self._system_messages + self._history
        )

        # Extract answer and share with neighbors
        answer = extract_answer(response)
        self._round += 1

        if self._round < self._max_rounds:
            # Share intermediate response
            await self.publish_message(
                IntermediateSolverResponse(
                    content=response,
                    question=message.question,
                    answer=answer,
                    round=self._round
                )
            )
        else:
            # Publish final answer
            await self.publish_message(FinalResponse(answer=answer))

    @message_handler
    async def handle_neighbor_response(self, message: IntermediateSolverResponse, ctx):
        # Aggregate neighbor solutions
        self._buffer.setdefault(message.round, []).append(message)

        if len(self._buffer[message.round]) == self._num_neighbors:
            # Prepare refined prompt with neighbor solutions
            prompt = "Consider these solutions from other agents:\n"
            for resp in self._buffer[message.round]:
                prompt += f"Solution: {resp.content}\n"
            prompt += "Provide your refined answer."

            await self.send_message(
                SolverRequest(content=prompt, question=message.question),
                self.id
            )
```

### Ensemble Voting

```python
class EnsembleCoordinator:
    def __init__(self, agents: List[Agent]):
        self.agents = agents

    async def solve(self, question: str) -> str:
        # Get answers from all agents
        answers = []
        for agent in self.agents:
            answer = await agent.invoke(question)
            answers.append(answer)

        # Aggregate by voting
        return self.majority_vote(answers)

    def majority_vote(self, answers: List[str]) -> str:
        # Count occurrences and return most common
        from collections import Counter
        counts = Counter(answers)
        return counts.most_common(1)[0][0]
```

---

## 4. Sequential Pipeline

### Overview

Sequential pipelines pass work through a chain of agents, each performing a specific transformation or analysis.

### Architecture

```
Input -> Agent 1 -> Agent 2 -> Agent 3 -> Output
         (Extract)   (Analyze)  (Summarize)
```

### CrewAI Sequential Process

```python
from crewai import Agent, Crew, Task, Process

# Define pipeline agents
extractor = Agent(
    role="Data Extractor",
    goal="Extract relevant information from raw data",
    backstory="Specialist in data extraction"
)

analyst = Agent(
    role="Data Analyst",
    goal="Analyze extracted data for insights",
    backstory="Expert data analyst"
)

reporter = Agent(
    role="Report Writer",
    goal="Create clear, actionable reports",
    backstory="Technical writer with data expertise"
)

# Define sequential tasks
extraction_task = Task(
    description="Extract key metrics from the data",
    agent=extractor,
    expected_output="Structured data with key metrics"
)

analysis_task = Task(
    description="Analyze metrics and identify trends",
    agent=analyst,
    expected_output="Analysis with identified trends",
    context=[extraction_task]  # Depends on extraction
)

report_task = Task(
    description="Create executive summary report",
    agent=reporter,
    expected_output="Executive summary with recommendations",
    context=[analysis_task]  # Depends on analysis
)

# Create sequential crew
crew = Crew(
    agents=[extractor, analyst, reporter],
    tasks=[extraction_task, analysis_task, report_task],
    process=Process.sequential
)
```

### LangGraph Pipeline

```python
from langgraph.graph import StateGraph, END

class PipelineState(TypedDict):
    input: str
    extracted_data: str
    analysis: str
    report: str

def extract(state: PipelineState):
    result = extractor_agent.invoke(state["input"])
    return {"extracted_data": result}

def analyze(state: PipelineState):
    result = analyst_agent.invoke(state["extracted_data"])
    return {"analysis": result}

def report(state: PipelineState):
    result = reporter_agent.invoke(state["analysis"])
    return {"report": result}

workflow = StateGraph(PipelineState)
workflow.add_node("extract", extract)
workflow.add_node("analyze", analyze)
workflow.add_node("report", report)

workflow.add_edge("extract", "analyze")
workflow.add_edge("analyze", "report")
workflow.add_edge("report", END)

workflow.set_entry_point("extract")
app = workflow.compile()
```

---

## 5. Router/Dispatcher Patterns

### Overview

Router patterns use a central dispatcher to route requests to the appropriate specialized agent based on the request type.

### Architecture

```
                    ┌──────────────┐
                    │    Router    │
                    │   (Classify) │
                    └──────┬───────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │   Tech   │    │  Sales   │    │ Support  │
    │  Agent   │    │  Agent   │    │  Agent   │
    └──────────┘    └──────────┘    └──────────┘
```

### Implementation

```python
from langgraph.graph import StateGraph, END

class RouterState(TypedDict):
    input: str
    category: str
    response: str

def router(state: RouterState):
    """Classify the request and determine routing."""
    classification = llm.invoke(f"""
    Classify this request into one of: tech, sales, support

    Request: {state['input']}

    Return only the category name.
    """)
    return {"category": classification.strip().lower()}

def route_to_agent(state: RouterState) -> str:
    """Route to appropriate agent based on category."""
    category = state["category"]
    if category == "tech":
        return "tech_agent"
    elif category == "sales":
        return "sales_agent"
    else:
        return "support_agent"

def tech_agent(state: RouterState):
    response = tech_specialist.invoke(state["input"])
    return {"response": response}

def sales_agent(state: RouterState):
    response = sales_specialist.invoke(state["input"])
    return {"response": response}

def support_agent(state: RouterState):
    response = support_specialist.invoke(state["input"])
    return {"response": response}

workflow = StateGraph(RouterState)
workflow.add_node("router", router)
workflow.add_node("tech_agent", tech_agent)
workflow.add_node("sales_agent", sales_agent)
workflow.add_node("support_agent", support_agent)

workflow.add_conditional_edges("router", route_to_agent)
workflow.add_edge("tech_agent", END)
workflow.add_edge("sales_agent", END)
workflow.add_edge("support_agent", END)

workflow.set_entry_point("router")
app = workflow.compile()
```

### Semantic Routing

```python
from langchain_core.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

class SemanticRouter:
    def __init__(self, routes: Dict[str, Agent]):
        self.routes = routes
        self.embeddings = OpenAIEmbeddings()
        self.route_embeddings = {}

        # Pre-compute route embeddings
        for name, agent in routes.items():
            description = f"{agent.role}: {agent.goal}"
            self.route_embeddings[name] = self.embeddings.embed_query(description)

    def route(self, query: str) -> str:
        query_embedding = self.embeddings.embed_query(query)

        best_route = None
        best_score = -1

        for name, route_embedding in self.route_embeddings.items():
            score = cosine_similarity([query_embedding], [route_embedding])[0][0]
            if score > best_score:
                best_score = score
                best_route = name

        return best_route
```

---

## Pattern Comparison

| Pattern | Coordination | Scalability | Complexity | Use Case |
|---------|--------------|-------------|------------|----------|
| Hierarchical | Centralized | Medium | Medium | Structured teams |
| Collaborative | Distributed | High | High | Creative tasks |
| Competitive | Parallel | High | Low | Accuracy-critical |
| Sequential | Linear | Low | Low | Pipeline processing |
| Router | Centralized | High | Medium | Request routing |

## Framework Comparison

| Pattern | LangGraph | CrewAI | AutoGen |
|---------|-----------|--------|---------|
| Hierarchical | Supervisor lib | Process.hierarchical | Custom |
| Collaborative | Swarm lib | allow_delegation | Conversation |
| Competitive | Custom | Custom | Multi-agent debate |
| Sequential | Graph edges | Process.sequential | Custom |
| Router | Conditional edges | Custom | Topic routing |

## Best Practices

### 1. Agent Design
- Give each agent a clear, focused role
- Limit the number of tools per agent
- Use descriptive names and backstories

### 2. Communication
- Define clear message protocols
- Limit message size to reduce context
- Use structured formats (JSON) for data exchange

### 3. Coordination
- Set clear termination conditions
- Implement timeouts for long-running tasks
- Add fallback handlers for agent failures

### 4. Scalability
- Use async patterns for parallel execution
- Implement agent pools for high-load scenarios
- Consider message queuing for large systems
