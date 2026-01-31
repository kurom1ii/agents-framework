# Memory Patterns

This document covers memory architectures and state management patterns for AI agents.

## Overview

Memory is crucial for agents to maintain context, learn from interactions, and make informed decisions. Different memory types serve different purposes:

| Memory Type | Duration | Purpose | Implementation |
|-------------|----------|---------|----------------|
| Short-term | Session | Conversation context | Buffer, Checkpointer |
| Long-term | Permanent | Knowledge retention | Vector stores, databases |
| Episodic | Variable | Experience recall | Indexed events |
| Semantic | Permanent | Concept relationships | Embeddings, knowledge graphs |
| Working | Temporary | Active task state | State management |

---

## 1. Short-term Memory (Conversation Buffer)

### Overview

Short-term memory maintains the immediate context of a conversation or task. It typically stores:
- Recent messages
- Current task state
- Intermediate results

### Implementation with Checkpointers

```python
from langgraph.checkpoint.memory import InMemorySaver

# Create checkpointer for short-term memory
checkpointer = InMemorySaver()

# Compile workflow with checkpointer
app = workflow.compile(checkpointer=checkpointer)

# Invoke with thread_id to maintain conversation
config = {"configurable": {"thread_id": "conversation-123"}}
result = app.invoke({"input": "Hello"}, config)
```

### Conversation Buffer Patterns

#### Sliding Window Buffer

```python
class SlidingWindowBuffer:
    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self.messages = []

    def add_message(self, message: dict):
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)

    def get_context(self) -> List[dict]:
        return self.messages.copy()
```

#### Token-Limited Buffer

```python
from tiktoken import encoding_for_model

class TokenLimitedBuffer:
    def __init__(self, max_tokens: int = 4000, model: str = "gpt-4"):
        self.max_tokens = max_tokens
        self.encoder = encoding_for_model(model)
        self.messages = []

    def add_message(self, message: dict):
        self.messages.append(message)
        self._trim_to_limit()

    def _trim_to_limit(self):
        while self._total_tokens() > self.max_tokens and len(self.messages) > 1:
            self.messages.pop(0)

    def _total_tokens(self) -> int:
        text = " ".join([m["content"] for m in self.messages])
        return len(self.encoder.encode(text))
```

#### Summary Buffer

```python
class SummaryBuffer:
    def __init__(self, llm, max_messages: int = 5):
        self.llm = llm
        self.max_messages = max_messages
        self.messages = []
        self.summary = ""

    def add_message(self, message: dict):
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self._summarize_oldest()

    def _summarize_oldest(self):
        # Take oldest messages and summarize
        to_summarize = self.messages[:3]
        self.messages = self.messages[3:]

        # Create summary
        summary_prompt = f"""
        Summarize this conversation segment:
        {to_summarize}

        Previous summary: {self.summary}
        """
        self.summary = self.llm.invoke(summary_prompt)

    def get_context(self) -> str:
        return f"Summary: {self.summary}\n\nRecent: {self.messages}"
```

---

## 2. Long-term Memory (Vector Stores)

### Overview

Long-term memory stores information persistently, typically using vector embeddings for semantic retrieval.

### LangGraph Store Implementation

```python
from langgraph.store.memory import InMemoryStore
from langchain.embeddings import init_embeddings

# Initialize store with embeddings for semantic search
store = InMemoryStore(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "dims": 1536,
        "fields": ["content", "$"]  # Fields to embed
    }
)

# Store a memory
import uuid

namespace = ("user", "user-123", "memories")
store.put(
    namespace,
    str(uuid.uuid4()),
    {
        "content": "User prefers dark mode",
        "category": "preferences",
        "timestamp": "2024-01-15"
    },
    index=["content"]  # Which fields to embed
)

# Retrieve by semantic search
memories = store.search(
    namespace,
    query="What are the user's UI preferences?",
    limit=5
)
```

### Vector Store Integration

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class LongTermMemory:
    def __init__(self, collection_name: str = "agent_memory"):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory="./memory_db"
        )

    def store(self, content: str, metadata: dict = None):
        """Store a memory with optional metadata."""
        self.vectorstore.add_texts(
            texts=[content],
            metadatas=[metadata] if metadata else None
        )

    def recall(self, query: str, k: int = 5) -> List[str]:
        """Recall relevant memories."""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def recall_with_score(self, query: str, k: int = 5, threshold: float = 0.7):
        """Recall memories above relevance threshold."""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [(doc.page_content, score) for doc, score in results if score >= threshold]
```

### CrewAI Memory Integration

```python
from crewai import Crew, Agent

# Enable memory at crew level
crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    memory=True,  # Enables short-term, long-term, and entity memory
    verbose=True
)

# Memory is automatically used for:
# 1. Store conversation history (short-term memory)
# 2. Remember important facts (long-term memory)
# 3. Track entities and relationships (entity memory)

# Agent-level memory
agent = Agent(
    role="Customer Support",
    goal="Provide helpful support",
    backstory="Experienced support agent",
    memory=True  # Agent remembers past interactions
)
```

---

## 3. Episodic Memory

### Overview

Episodic memory stores specific experiences or events, allowing the agent to recall and learn from past situations.

### Implementation

```python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import json

@dataclass
class Episode:
    id: str
    timestamp: datetime
    context: str
    action: str
    result: str
    outcome: str  # success, failure, partial
    metadata: dict

class EpisodicMemory:
    def __init__(self, store: InMemoryStore, embeddings):
        self.store = store
        self.embeddings = embeddings
        self.namespace = ("episodes",)

    def record_episode(self, episode: Episode):
        """Record a new episode."""
        self.store.put(
            self.namespace,
            episode.id,
            {
                "timestamp": episode.timestamp.isoformat(),
                "context": episode.context,
                "action": episode.action,
                "result": episode.result,
                "outcome": episode.outcome,
                "metadata": json.dumps(episode.metadata)
            },
            index=["context", "action", "result"]
        )

    def recall_similar(self, current_context: str, k: int = 5) -> List[Episode]:
        """Recall episodes similar to current context."""
        results = self.store.search(
            self.namespace,
            query=current_context,
            limit=k
        )
        return [self._to_episode(r) for r in results]

    def recall_by_outcome(self, outcome: str) -> List[Episode]:
        """Recall episodes with specific outcome."""
        results = self.store.list(self.namespace)
        return [
            self._to_episode(r) for r in results
            if r.value.get("outcome") == outcome
        ]

    def learn_from_failure(self, current_context: str) -> Optional[str]:
        """Learn from past failures in similar contexts."""
        failures = self.recall_similar(current_context, k=3)
        failures = [e for e in failures if e.outcome == "failure"]

        if not failures:
            return None

        # Generate advice based on past failures
        advice = f"Based on past experience, avoid these actions:\n"
        for episode in failures:
            advice += f"- {episode.action} (resulted in: {episode.result})\n"
        return advice
```

### Episode Recording in Workflows

```python
from langgraph.graph import StateGraph

class EpisodicState(TypedDict):
    input: str
    context: str
    action: str
    result: str
    outcome: str

def record_and_execute(state: EpisodicState, episodic_memory: EpisodicMemory):
    """Execute action and record episode."""
    # Check for similar past experiences
    similar_episodes = episodic_memory.recall_similar(state["context"])

    # Get advice from failures
    failure_advice = episodic_memory.learn_from_failure(state["context"])
    if failure_advice:
        # Inject advice into context
        state["context"] += f"\n\nNote: {failure_advice}"

    # Execute action
    result = execute_action(state["action"])
    outcome = evaluate_outcome(result)

    # Record episode
    episode = Episode(
        id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        context=state["context"],
        action=state["action"],
        result=result,
        outcome=outcome,
        metadata={"similar_episodes": len(similar_episodes)}
    )
    episodic_memory.record_episode(episode)

    return {"result": result, "outcome": outcome}
```

---

## 4. Semantic Memory

### Overview

Semantic memory stores concepts, facts, and relationships, organized for efficient retrieval based on meaning.

### Knowledge Graph Integration

```python
from langchain_community.graphs import Neo4jGraph

class SemanticMemory:
    def __init__(self, uri: str, user: str, password: str):
        self.graph = Neo4jGraph(url=uri, username=user, password=password)

    def store_fact(self, subject: str, predicate: str, object: str):
        """Store a fact as a relationship."""
        query = """
        MERGE (s:Entity {name: $subject})
        MERGE (o:Entity {name: $object})
        MERGE (s)-[r:RELATES {type: $predicate}]->(o)
        """
        self.graph.query(query, {
            "subject": subject,
            "predicate": predicate,
            "object": object
        })

    def query_facts(self, query: str) -> List[dict]:
        """Query the knowledge graph."""
        return self.graph.query(query)

    def get_related_concepts(self, concept: str, depth: int = 2) -> List[str]:
        """Get concepts related to the given concept."""
        query = f"""
        MATCH (e:Entity {{name: $concept}})-[*1..{depth}]-(related:Entity)
        RETURN DISTINCT related.name as name
        """
        results = self.graph.query(query, {"concept": concept})
        return [r["name"] for r in results]
```

### Embedding-Based Semantic Memory

```python
from langchain.embeddings import OpenAIEmbeddings
import numpy as np

class SemanticIndex:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.concepts = {}  # name -> embedding
        self.relationships = []  # (concept1, relation, concept2)

    def add_concept(self, name: str, description: str):
        """Add a concept with its embedding."""
        embedding = self.embeddings.embed_query(description)
        self.concepts[name] = {
            "description": description,
            "embedding": embedding
        }

    def add_relationship(self, concept1: str, relation: str, concept2: str):
        """Add a relationship between concepts."""
        self.relationships.append((concept1, relation, concept2))

    def find_similar_concepts(self, query: str, k: int = 5) -> List[str]:
        """Find concepts similar to query."""
        query_embedding = self.embeddings.embed_query(query)

        similarities = []
        for name, data in self.concepts.items():
            sim = np.dot(query_embedding, data["embedding"])
            similarities.append((name, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in similarities[:k]]

    def get_concept_context(self, concept: str) -> str:
        """Get full context for a concept including relationships."""
        if concept not in self.concepts:
            return ""

        context = f"Concept: {concept}\n"
        context += f"Description: {self.concepts[concept]['description']}\n"
        context += "Relationships:\n"

        for c1, rel, c2 in self.relationships:
            if c1 == concept:
                context += f"  - {rel} {c2}\n"
            elif c2 == concept:
                context += f"  - {c1} {rel} this\n"

        return context
```

---

## 5. Working Memory

### Overview

Working memory is temporary storage for the current task, holding intermediate results, active goals, and relevant context.

### LangGraph State as Working Memory

```python
from langgraph.graph import StateGraph
from typing import TypedDict, List, Optional

class WorkingMemory(TypedDict):
    # Current task
    current_goal: str
    sub_goals: List[str]
    completed_goals: List[str]

    # Active context
    relevant_facts: List[str]
    active_entities: List[str]

    # Intermediate results
    scratch_pad: str
    calculations: dict

    # Meta-cognitive
    confidence: float
    uncertainties: List[str]

def update_working_memory(state: WorkingMemory, llm) -> WorkingMemory:
    """Update working memory based on current state."""
    # Identify active entities from recent context
    entities = extract_entities(state["scratch_pad"])

    # Update confidence based on uncertainties
    confidence = 1.0 - (len(state["uncertainties"]) * 0.1)

    return {
        **state,
        "active_entities": entities,
        "confidence": max(0.0, confidence)
    }
```

### Scratchpad Pattern

```python
class ScratchPad:
    """Working memory scratchpad for intermediate reasoning."""

    def __init__(self):
        self.thoughts = []
        self.observations = []
        self.plans = []
        self.calculations = {}

    def add_thought(self, thought: str):
        self.thoughts.append({
            "content": thought,
            "timestamp": datetime.now()
        })

    def add_observation(self, observation: str, source: str):
        self.observations.append({
            "content": observation,
            "source": source,
            "timestamp": datetime.now()
        })

    def set_plan(self, steps: List[str]):
        self.plans = steps

    def mark_step_complete(self, step_index: int):
        if step_index < len(self.plans):
            self.plans[step_index] = f"[DONE] {self.plans[step_index]}"

    def store_calculation(self, key: str, value: any):
        self.calculations[key] = value

    def get_calculation(self, key: str) -> any:
        return self.calculations.get(key)

    def to_context(self) -> str:
        """Convert scratchpad to context string for LLM."""
        context = "## Working Memory\n\n"

        if self.plans:
            context += "### Current Plan:\n"
            for i, step in enumerate(self.plans):
                context += f"{i+1}. {step}\n"

        if self.thoughts:
            context += "\n### Recent Thoughts:\n"
            for thought in self.thoughts[-5:]:
                context += f"- {thought['content']}\n"

        if self.observations:
            context += "\n### Observations:\n"
            for obs in self.observations[-5:]:
                context += f"- [{obs['source']}] {obs['content']}\n"

        if self.calculations:
            context += "\n### Stored Values:\n"
            for key, value in self.calculations.items():
                context += f"- {key}: {value}\n"

        return context
```

---

## Memory Integration Patterns

### Combined Memory Architecture

```python
class AgentMemorySystem:
    """Unified memory system combining all memory types."""

    def __init__(self, store, embeddings, graph_db=None):
        # Short-term
        self.conversation_buffer = SlidingWindowBuffer(max_messages=20)

        # Long-term
        self.long_term = LongTermMemory(store, embeddings)

        # Episodic
        self.episodic = EpisodicMemory(store, embeddings)

        # Semantic
        self.semantic = SemanticMemory(graph_db) if graph_db else None

        # Working
        self.scratchpad = ScratchPad()

    def get_context(self, query: str) -> str:
        """Get combined context from all memory systems."""
        context = ""

        # Recent conversation
        context += "## Recent Conversation\n"
        context += str(self.conversation_buffer.get_context())

        # Relevant long-term memories
        context += "\n## Relevant Memories\n"
        memories = self.long_term.recall(query, k=3)
        for mem in memories:
            context += f"- {mem}\n"

        # Similar past episodes
        context += "\n## Past Experiences\n"
        episodes = self.episodic.recall_similar(query, k=2)
        for ep in episodes:
            context += f"- {ep.action} -> {ep.outcome}\n"

        # Related concepts
        if self.semantic:
            context += "\n## Related Concepts\n"
            concepts = self.semantic.find_similar_concepts(query, k=3)
            for concept in concepts:
                context += f"- {concept}\n"

        # Current working memory
        context += self.scratchpad.to_context()

        return context

    def update_from_interaction(self, input: str, output: str, outcome: str):
        """Update all memory systems from an interaction."""
        # Update conversation buffer
        self.conversation_buffer.add_message({"role": "user", "content": input})
        self.conversation_buffer.add_message({"role": "assistant", "content": output})

        # Store in long-term if significant
        if self._is_significant(output):
            self.long_term.store(output, metadata={"type": "response"})

        # Record episode
        episode = Episode(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            context=input,
            action=output,
            result=output,
            outcome=outcome,
            metadata={}
        )
        self.episodic.record_episode(episode)

    def _is_significant(self, content: str) -> bool:
        """Determine if content is significant enough for long-term storage."""
        # Simple heuristics - could be more sophisticated
        return len(content) > 100 or any(keyword in content.lower()
            for keyword in ["important", "remember", "note", "key point"])
```

---

## Best Practices

### 1. Memory Selection
- Use short-term for conversation continuity
- Use long-term for persistent knowledge
- Use episodic for learning from experience
- Use semantic for concept relationships
- Use working memory for complex reasoning

### 2. Memory Hygiene
- Regularly prune irrelevant memories
- Implement importance scoring
- Set TTL for temporary memories
- Compress old memories into summaries

### 3. Retrieval Optimization
- Use hybrid search (keyword + semantic)
- Implement relevance thresholds
- Limit context size for LLM calls
- Cache frequently accessed memories

### 4. Privacy and Security
- Implement access controls
- Encrypt sensitive memories
- Provide memory deletion capabilities
- Audit memory access

### 5. Performance
- Use async operations for memory access
- Batch memory writes
- Index frequently queried fields
- Consider memory tiers (hot/warm/cold)
