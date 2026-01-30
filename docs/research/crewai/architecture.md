# CrewAI Architecture

## Overview

CrewAI implements a multi-agent architecture centered around the concept of "crews" - collaborative groups of AI agents working together to accomplish complex tasks. The architecture emphasizes role-based design, goal-oriented execution, and flexible orchestration.

## Core Architecture Components

### 1. Agent Layer

Agents are the fundamental building blocks of CrewAI. Each agent is an autonomous AI entity with:

```
+------------------------------------------+
|                 AGENT                     |
+------------------------------------------+
| Identity                                  |
|   - role: str (function/expertise)        |
|   - goal: str (objective)                 |
|   - backstory: str (context/personality)  |
+------------------------------------------+
| Capabilities                              |
|   - tools: List[BaseTool]                 |
|   - knowledge_sources: List[...]          |
|   - llm: LLM                              |
+------------------------------------------+
| Behavior                                  |
|   - allow_delegation: bool                |
|   - allow_code_execution: bool            |
|   - memory: bool                          |
|   - verbose: bool                         |
+------------------------------------------+
| Constraints                               |
|   - max_iter: int                         |
|   - max_rpm: int                          |
|   - max_execution_time: int               |
+------------------------------------------+
```

### 2. Task Layer

Tasks represent units of work to be completed by agents:

```
+------------------------------------------+
|                  TASK                     |
+------------------------------------------+
| Definition                                |
|   - description: str                      |
|   - expected_output: str                  |
|   - agent: Agent (optional)               |
+------------------------------------------+
| I/O Configuration                         |
|   - context: List[Task]                   |
|   - output_file: str                      |
|   - output_json: Type[BaseModel]          |
|   - output_pydantic: Type[BaseModel]      |
+------------------------------------------+
| Execution Control                         |
|   - async_execution: bool                 |
|   - human_input: bool                     |
|   - guardrails: List[Callable]            |
|   - callback: Callable                    |
+------------------------------------------+
```

### 3. Crew Layer

Crews orchestrate agent collaboration:

```
+------------------------------------------+
|                  CREW                     |
+------------------------------------------+
| Composition                               |
|   - agents: List[Agent]                   |
|   - tasks: List[Task]                     |
+------------------------------------------+
| Orchestration                             |
|   - process: Process (seq/hierarchical)   |
|   - manager_llm: LLM (for hierarchical)   |
|   - manager_agent: Agent (optional)       |
+------------------------------------------+
| Shared Resources                          |
|   - memory: bool                          |
|   - cache: bool                           |
|   - knowledge_sources: List[...]          |
|   - embedder: Dict                        |
+------------------------------------------+
| Callbacks & Logging                       |
|   - step_callback: Callable               |
|   - task_callback: Callable               |
|   - output_log_file: str                  |
+------------------------------------------+
```

### 4. Flow Layer

Flows provide higher-level orchestration with event-driven patterns:

```
+------------------------------------------+
|                  FLOW                     |
+------------------------------------------+
| Control Primitives                        |
|   - @start(): Entry points                |
|   - @listen(): Event handlers             |
|   - @router(): Conditional routing        |
|   - or_(): Any-of triggers                |
|   - and_(): All-of triggers               |
+------------------------------------------+
| State Management                          |
|   - Unstructured: dict-like state         |
|   - Structured: Pydantic BaseModel        |
|   - Auto-generated unique ID              |
+------------------------------------------+
| Persistence                               |
|   - @persist: State persistence           |
|   - SQLiteFlowPersistence (default)       |
|   - Custom FlowPersistence implementations|
+------------------------------------------+
| Human-in-Loop                             |
|   - @human_feedback: Review gates         |
|   - Routed outcomes via LLM               |
|   - last_human_feedback access            |
|   - human_feedback_history tracking       |
+------------------------------------------+
| Execution                                 |
|   - kickoff(): Sync execution             |
|   - kickoff_async(): Async execution      |
|   - plot(): Flow visualization            |
|   - stream: Streaming output support      |
+------------------------------------------+
```

#### Flow Decorators Detail

| Decorator | Description | Usage |
|-----------|-------------|-------|
| `@start()` | Entry point, can be gated by label or condition | `@start()` or `@start("label")` |
| `@listen(method)` | Triggers when method completes | `@listen(method_name)` |
| `@router(method)` | Returns label for conditional routing | Returns string label |
| `@persist` | Enable state persistence | Class or method level |
| `@human_feedback` | Collect human review/decisions | With outcomes list |

#### Conditional Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `or_(a, b)` | Trigger when ANY method emits | `@listen(or_(method1, method2))` |
| `and_(a, b)` | Trigger when ALL methods emit | `@listen(and_(method1, method2))` |

## System Architecture Diagram

```
                    +------------------+
                    |   User/Client    |
                    +--------+---------+
                             |
                             v
+-----------------------------------------------------------+
|                       FLOW LAYER                           |
|  +-------+   +-------+   +--------+   +----------------+  |
|  |@start |-->|@listen|-->|@router |-->|@human_feedback |  |
|  +-------+   +-------+   +--------+   +----------------+  |
|                    |                                       |
|              State Management                              |
|              (Structured/Unstructured)                     |
+-----------------------------------------------------------+
                             |
                             v
+-----------------------------------------------------------+
|                       CREW LAYER                           |
|  +--------------------------------------------------+     |
|  |                  CREW                             |     |
|  |  Process: Sequential | Hierarchical               |     |
|  |  Manager: LLM or Agent (hierarchical only)        |     |
|  +--------------------------------------------------+     |
|         |                                                  |
|         v                                                  |
|  +------+-------+-------+-------+                         |
|  |Task 1|Task 2 |Task 3 |Task N |  (Ordered execution)    |
|  +------+-------+-------+-------+                         |
+-----------------------------------------------------------+
                             |
                             v
+-----------------------------------------------------------+
|                      AGENT LAYER                           |
|  +------------+  +------------+  +------------+           |
|  |  Agent 1   |  |  Agent 2   |  |  Agent N   |           |
|  |  (Role)    |  |  (Role)    |  |  (Role)    |           |
|  +------------+  +------------+  +------------+           |
|        |               |               |                   |
|        v               v               v                   |
|  +---------+     +---------+     +---------+              |
|  | Tools   |     | Tools   |     | Tools   |              |
|  +---------+     +---------+     +---------+              |
+-----------------------------------------------------------+
                             |
                             v
+-----------------------------------------------------------+
|                    FOUNDATION LAYER                        |
|  +----------+  +----------+  +-----------+  +----------+  |
|  |   LLM    |  |  Memory  |  | Knowledge |  | Embedder |  |
|  | Provider |  |  System  |  |  Sources  |  |          |  |
|  +----------+  +----------+  +-----------+  +----------+  |
+-----------------------------------------------------------+
```

## Process Types

### Sequential Process

Tasks execute in the order defined. Each task's output can be passed as context to subsequent tasks.

```
Task 1 --> Task 2 --> Task 3 --> Task N --> Final Output
   |          ^
   +----------+ (Output flows as context)
```

**Characteristics**:
- Deterministic execution order
- Output of one task becomes context for the next
- Simple and predictable workflow
- Best for pipeline-style workflows

### Hierarchical Process

A manager agent coordinates task delegation and validates work completion.

```
                 +------------------+
                 |  Manager Agent   |
                 | (Planning/Coord) |
                 +--------+---------+
                          |
        +-----------------+-----------------+
        |                 |                 |
        v                 v                 v
  +-----------+     +-----------+     +-----------+
  |  Agent 1  |     |  Agent 2  |     |  Agent N  |
  | (Delegate)|     | (Delegate)|     | (Delegate)|
  +-----------+     +-----------+     +-----------+
```

**Characteristics**:
- Manager plans, delegates, and validates
- Dynamic task assignment based on agent capabilities
- More flexible but requires manager LLM
- Best for complex, adaptive workflows

## Memory Architecture

CrewAI implements a multi-tier memory system:

```
+-----------------------------------------------------------+
|                    MEMORY SYSTEM                           |
+-----------------------------------------------------------+
|                                                            |
|  +------------------+     +------------------+             |
|  | Short-Term Memory|     | Long-Term Memory |             |
|  | (Current session)|     | (Across sessions)|             |
|  | ChromaDB + RAG   |     | SQLite3          |             |
|  +------------------+     +------------------+             |
|                                                            |
|  +------------------+     +------------------+             |
|  |  Entity Memory   |     | External Memory  |             |
|  | (People, places) |     | (Mem0, custom)   |             |
|  | ChromaDB + RAG   |     |                  |             |
|  +------------------+     +------------------+             |
|                                                            |
|  +--------------------------------------------------+     |
|  |              Contextual Memory                    |     |
|  | (Combines all memory types for coherent context)  |     |
|  +--------------------------------------------------+     |
+-----------------------------------------------------------+
```

### Memory Types

| Type | Storage | Purpose |
|------|---------|---------|
| Short-Term | ChromaDB | Recent interactions in current execution |
| Long-Term | SQLite3 | Valuable insights across sessions |
| Entity | ChromaDB | Information about entities (people, places, concepts) |
| External | Pluggable | Integration with external memory providers |
| Contextual | Combined | Unified context from all memory types |

## Knowledge Architecture

Knowledge sources provide agents with domain-specific information:

```
+-----------------------------------------------------------+
|                  KNOWLEDGE SYSTEM                          |
+-----------------------------------------------------------+
|                                                            |
|  Built-in Sources:                                         |
|  +--------+ +--------+ +--------+ +--------+ +--------+   |
|  |  Text  | |  PDF   | |  CSV   | | Excel  | |  JSON  |   |
|  +--------+ +--------+ +--------+ +--------+ +--------+   |
|                                                            |
|  Web Sources:                                              |
|  +--------------------------------------------------+     |
|  |            CrewDoclingSource (URLs)               |     |
|  +--------------------------------------------------+     |
|                                                            |
|  Custom Sources:                                           |
|  +--------------------------------------------------+     |
|  |       BaseKnowledgeSource (subclass)              |     |
|  +--------------------------------------------------+     |
|                                                            |
|  Storage:                                                  |
|  +--------------------------------------------------+     |
|  |  ChromaDB (default) | Qdrant (optional)           |     |
|  +--------------------------------------------------+     |
|                                                            |
+-----------------------------------------------------------+
```

## Tool Integration Architecture

```
+-----------------------------------------------------------+
|                    TOOL SYSTEM                             |
+-----------------------------------------------------------+
|                                                            |
|  Built-in Tools:                                           |
|  +----------+ +----------+ +----------+ +----------+      |
|  |  Search  | |   RAG    | |  Scrape  | |  Code    |      |
|  | (Serper) | | (Vector) | |  (Web)   | | (Interp) |      |
|  +----------+ +----------+ +----------+ +----------+      |
|                                                            |
|  Custom Tools:                                             |
|  +--------------------------------------------------+     |
|  |  BaseTool Subclass  |  @tool Decorator            |     |
|  +--------------------------------------------------+     |
|                                                            |
|  Features:                                                 |
|  - Caching (cache_function)                               |
|  - Error handling                                          |
|  - Async support                                           |
|  - Pydantic input schemas                                  |
|                                                            |
+-----------------------------------------------------------+
```

## LLM Integration

CrewAI is LLM-agnostic and supports multiple providers:

| Provider | Model Prefix | Key Features |
|----------|--------------|--------------|
| OpenAI | `openai/` | Responses API, streaming |
| Anthropic | `anthropic/` | Extended thinking, Claude models |
| Google | `gemini/` | Multimodal, Vertex AI |
| Azure | `azure/` | Enterprise, deployments |
| AWS Bedrock | `bedrock/` | Converse API, guardrails |
| Ollama | `ollama/` | Local models |
| Groq | `groq/` | Fast inference |
| And more... | | |

## Event System

CrewAI emits events for monitoring and observability:

```
+-----------------------------------------------------------+
|                    EVENT SYSTEM                            |
+-----------------------------------------------------------+
|                                                            |
|  Memory Events:                                            |
|  - MemoryQueryCompletedEvent                              |
|  - MemorySaveCompletedEvent                               |
|  - MemorySaveFailedEvent                                  |
|                                                            |
|  Knowledge Events:                                         |
|  - KnowledgeRetrievalStartedEvent                         |
|  - KnowledgeRetrievalCompletedEvent                       |
|  - KnowledgeQueryFailedEvent                              |
|                                                            |
|  LLM Events:                                               |
|  - LLMStreamChunkEvent (streaming)                        |
|                                                            |
|  Subscribe via BaseEventListener subclass                  |
+-----------------------------------------------------------+
```

## Execution Flow

### Standard Crew Execution

```
1. Crew.kickoff(inputs)
       |
2. Initialize agents with config
       |
3. For each task (based on process):
       |
   3a. Assign to agent (sequential) or delegate (hierarchical)
       |
   3b. Agent executes with tools/knowledge/memory
       |
   3c. Apply guardrails (validate/transform)
       |
   3d. Execute callbacks
       |
   3e. Pass output as context to next task
       |
4. Return CrewOutput (raw, pydantic, json_dict)
```

### Flow Execution

```
1. Flow.kickoff()
       |
2. Execute @start methods (entry points)
       |
3. For each method output:
       |
   3a. Update state
       |
   3b. Trigger @listen methods matching output
       |
   3c. Execute @router for conditional branching
       |
   3d. Handle @human_feedback if present
       |
4. Persist state if @persist enabled
       |
5. Return final method output
```

## Deployment Architecture

### Local Development

```
Developer Machine
    |
    +-- CrewAI Project
    |       |
    |       +-- crew.py (agents, tasks)
    |       +-- main.py (entry)
    |       +-- config/ (YAML)
    |       +-- tools/ (custom)
    |
    +-- .env (API keys)
    |
    +-- Local LLM (Ollama) or Remote API
```

### Production (Enterprise)

```
+-----------------------------------------------------------+
|                 CrewAI Enterprise                          |
+-----------------------------------------------------------+
|                                                            |
|  +------------------+    +------------------+              |
|  | Environment Mgmt |    |    Monitoring    |              |
|  | (Safe redeploys) |    | (Live run view)  |              |
|  +------------------+    +------------------+              |
|                                                            |
|  +------------------+    +------------------+              |
|  |    Triggers      |    |   Team Mgmt      |              |
|  | (Gmail, Slack,   |    |   (RBAC)         |              |
|  |  Salesforce...)  |    |                  |              |
|  +------------------+    +------------------+              |
|                                                            |
|  +--------------------------------------------------+     |
|  |           Crew Control Plane                      |     |
|  | (Tracing, unified management, security, analytics)|     |
|  +--------------------------------------------------+     |
|                                                            |
+-----------------------------------------------------------+
```

## Design Principles

1. **Role-Based Design**: Agents have clear roles, goals, and backstories
2. **Composability**: Agents, tasks, and tools are modular and reusable
3. **Flexibility**: Support for multiple process types and execution patterns
4. **Observability**: Built-in logging, events, and monitoring
5. **Production-Ready**: Memory, persistence, guardrails, and enterprise features
6. **LLM Agnostic**: Works with any major LLM provider
7. **Developer Experience**: Simple API, CLI tools, YAML configuration
