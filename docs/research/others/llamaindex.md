# LlamaIndex Agents

## Overview

**LlamaIndex** (formerly GPT Index) is the leading framework for building LLM-powered agents over your data. While originally focused on RAG (Retrieval-Augmented Generation), LlamaIndex has evolved into a comprehensive data framework with powerful agent capabilities.

- **Repository**: https://github.com/run-llama/llama_index
- **Stars**: ~47,000
- **Language**: Python (also TypeScript via LlamaIndexTS)
- **License**: MIT
- **Documentation**: https://docs.llamaindex.ai/

## Core Philosophy

LlamaIndex defines an "agent" as a specific system that uses an LLM, memory, and tools to handle inputs from outside users. The framework excels at turning data into a form that LLMs can effectively use, making it ideal for data-intensive agent applications.

## Key Concepts

### Agent Definition

An agent combines:
- **LLM**: The language model "brain"
- **Memory**: Conversation history and context
- **Tools**: Functions the agent can call to interact with data and systems

### The Agent Loop

1. Receive latest message + chat history
2. Send tool schemas + history to LLM
3. LLM returns either a direct response or tool-call(s)
4. Execute tool(s) and append results to history
5. Re-invoke LLM until final response

```python
from llama_index.core.agent import FunctionAgent
from llama_index.core.tools import FunctionTool

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 72F"

weather_tool = FunctionTool.from_defaults(fn=get_weather)

agent = FunctionAgent(
    tools=[weather_tool],
    llm=llm,
    system_prompt="You are a helpful weather assistant."
)

response = await agent.chat("What's the weather in Tokyo?")
```

## Agent Types

### FunctionAgent
Uses LLM function/tool-calling to invoke Python tools. The most common agent type for general use.

```python
from llama_index.core.agent import FunctionAgent

agent = FunctionAgent(
    tools=[tool1, tool2],
    llm=llm,
    system_prompt="You are a helpful assistant."
)
```

### ReActAgent
Uses prompting strategies (not function-calling) to drive tool use. Based on the ReAct (Reasoning + Acting) paradigm.

```python
from llama_index.core.agent import ReActAgent

agent = ReActAgent.from_tools(
    tools=[tool1, tool2],
    llm=llm,
    verbose=True
)
```

### CodeActAgent
Uses code generation and execution for complex reasoning and actions.

### AgentWorkflow / Multi-Agent
Coordinates multiple agents with handoff capabilities.

```python
from llama_index.core.agent import AgentWorkflow

workflow = AgentWorkflow(
    agents=[agent1, agent2, agent3],
    root_agent=triage_agent
)
```

## Tools

### FunctionTool
Wrap Python functions as tools:

```python
from llama_index.core.tools import FunctionTool

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

tool = FunctionTool.from_defaults(fn=multiply)
```

### QueryEngineTool
Use query engines (RAG pipelines) as agent tools:

```python
from llama_index.core.tools import QueryEngineTool

# Create index from documents
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Wrap as tool
query_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="document_search",
    description="Search through company documents"
)

agent = FunctionAgent(tools=[query_tool], llm=llm)
```

### Tool Specs
Pre-built tool collections for common APIs:

```python
from llama_index.tools.google import GmailToolSpec

gmail_tools = GmailToolSpec().to_tool_list()
```

Available tool specs include:
- Google Suite (Gmail, Calendar, Drive)
- Slack
- Notion
- GitHub
- SQL databases
- And many more via LlamaHub

## Data Agents

Data agents are a key differentiator for LlamaIndex. They combine:
- **Data ingestion**: Load documents, APIs, databases
- **Indexing**: Vector stores, knowledge graphs, keyword indices
- **Retrieval**: Semantic search, hybrid search
- **Agent capabilities**: Tools, memory, reasoning

### Query Engines as Agents

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load and index documents
documents = SimpleDirectoryReader("data/").load_data()
index = VectorStoreIndex.from_documents(documents)

# Create query engine
query_engine = index.as_query_engine()

# Use directly or wrap as tool for agent
response = query_engine.query("What are the key findings?")
```

### Multi-Index Agents

Agents can orchestrate across multiple data sources:

```python
from llama_index.core.tools import QueryEngineTool

# Multiple query engines for different data sources
sales_tool = QueryEngineTool.from_defaults(
    query_engine=sales_index.as_query_engine(),
    name="sales_data",
    description="Query sales data"
)

support_tool = QueryEngineTool.from_defaults(
    query_engine=support_index.as_query_engine(),
    name="support_tickets",
    description="Query support tickets"
)

agent = FunctionAgent(tools=[sales_tool, support_tool], llm=llm)
```

## Memory

Agents default to ChatMemoryBuffer for conversation history:

```python
from llama_index.core.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

agent = FunctionAgent(
    tools=tools,
    llm=llm,
    memory=memory
)
```

## Multi-Modal Agents

LlamaIndex agents support multi-modal inputs (images + text):

```python
from llama_index.core.schema import ChatMessage, ImageBlock, TextBlock

message = ChatMessage(
    role="user",
    blocks=[
        TextBlock(text="What's in this image?"),
        ImageBlock(image_url="path/to/image.jpg")
    ]
)

response = await agent.chat(message)
```

## Streaming

Streaming is enabled by default:

```python
agent = FunctionAgent(tools=tools, llm=llm, streaming=True)

async for chunk in agent.stream_chat("Tell me a story"):
    print(chunk.delta, end="")
```

## Architecture

```
llama_index/
├── core/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── react/           # ReAct agent implementation
│   │   │   ├── formatter.py # ReActChatFormatter
│   │   │   └── output_parser.py
│   │   └── workflow/        # Workflow-based agents
│   │       ├── function_agent.py
│   │       ├── react_agent.py
│   │       ├── codeact_agent.py
│   │       └── multi_agent_workflow.py
│   ├── tools/               # Tool definitions
│   ├── memory/              # Memory implementations
│   ├── indices/             # Index types (vector, keyword, etc.)
│   ├── query_engine/        # Query engine implementations
│   └── storage/             # Storage backends
├── llms/                    # LLM integrations
├── embeddings/              # Embedding model integrations
└── readers/                 # Data loaders
```

## Integration Ecosystem

LlamaIndex has 300+ integrations on LlamaHub:

### LLM Providers
- OpenAI, Azure OpenAI
- Anthropic Claude
- Google (Gemini, PaLM)
- Cohere, AI21
- HuggingFace
- Ollama, LMStudio (local)

### Vector Stores
- Pinecone, Weaviate, Qdrant
- Chroma, Milvus, FAISS
- Azure AI Search, Elasticsearch

### Data Loaders
- PDFs, Word docs, PowerPoint
- Databases (SQL, MongoDB)
- APIs (Notion, Slack, GitHub)
- Web scraping

## Key Differentiators

| Feature | Description |
|---------|-------------|
| **Data-First** | Built for RAG and data-intensive applications |
| **Query Engines as Tools** | Unique ability to use RAG pipelines as agent tools |
| **Extensive Integrations** | 300+ integrations on LlamaHub |
| **Flexible Indexing** | Vector, keyword, knowledge graph, tree indices |
| **Composable** | Build complex pipelines from simple components |
| **Production Ready** | Observability, evaluation, and deployment tools |

## Best Use Cases

- **RAG Applications**: Document Q&A, knowledge bases
- **Data Agents**: Agents that need to query structured/unstructured data
- **Multi-Source Retrieval**: Agents spanning multiple data sources
- **Enterprise Search**: Semantic search over company documents
- **Research Assistants**: Agents that synthesize information from many sources
- **SQL Agents**: Natural language to SQL queries

## Limitations

- **Data-centric**: Less suited for pure conversation or action-oriented agents
- **Complexity**: Many components and options can be overwhelming
- **Memory overhead**: Large indices can be resource-intensive
- **LLM dependency**: Heavy reliance on LLM capabilities for complex reasoning

## Installation

```bash
# Starter package with common integrations
pip install llama-index

# Or core only with specific integrations
pip install llama-index-core
pip install llama-index-llms-openai
pip install llama-index-embeddings-huggingface
```

## Example: Complete Data Agent

```python
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent import FunctionAgent
from llama_index.core.tools import QueryEngineTool, FunctionTool

# Load documents
documents = SimpleDirectoryReader("data/").load_data()

# Create vector index
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Create query tool
doc_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="document_search",
    description="Search through documents to find relevant information"
)

# Create additional tools
def get_current_date() -> str:
    """Get the current date."""
    from datetime import date
    return str(date.today())

date_tool = FunctionTool.from_defaults(fn=get_current_date)

# Create agent
agent = FunctionAgent(
    tools=[doc_tool, date_tool],
    system_prompt="You are a helpful research assistant."
)

# Query the agent
response = await agent.chat("What are the main findings in the documents?")
print(response)
```

## References

- [LlamaIndex GitHub](https://github.com/run-llama/llama_index)
- [Documentation](https://docs.llamaindex.ai/)
- [LlamaHub (Integrations)](https://llamahub.ai/)
- [LlamaIndex TypeScript](https://github.com/run-llama/LlamaIndexTS)
- [Discord Community](https://discord.gg/dGcwcsnxhU)
- [X (Twitter)](https://x.com/llama_index)
