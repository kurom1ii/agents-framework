# Haystack Agents

## Overview

**Haystack** by deepset is an end-to-end AI orchestration framework for building production-ready LLM applications. Originally focused on search and question-answering, Haystack has evolved into a comprehensive framework supporting RAG, agents, and complex pipelines.

- **Repository**: https://github.com/deepset-ai/haystack
- **Stars**: ~24,000
- **Language**: Python
- **License**: Apache 2.0
- **Documentation**: https://docs.haystack.deepset.ai/

## Core Philosophy

Haystack emphasizes a **pipeline-based approach** where components are connected to form complex workflows. This makes it highly flexible, explicit, and extensible. The framework is technology-agnostic, allowing developers to choose and switch vendors easily.

## Key Concepts

### Agents in Haystack

An AI agent in Haystack:
- Handles queries (text, image, audio)
- Retrieves information
- Generates responses
- Takes actions using tools
- Plans and adapts using memory and reasoning

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.tools import Tool

agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4"),
    tools=[web_search_tool, calculator_tool],
    system_prompt="You are a helpful assistant."
)

result = agent.run(messages=[{"role": "user", "content": "What's the weather?"}])
```

### Core Components

#### LLM as the Brain
The language model handles context and natural-language reasoning:

```python
from haystack.components.generators.chat import OpenAIChatGenerator

generator = OpenAIChatGenerator(model="gpt-4o")
```

#### Tools
Interfaces for interacting with external systems, APIs, pipelines, or components:

```python
from haystack.tools import Tool

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny"

weather_tool = Tool(
    name="get_weather",
    description="Get weather for a city",
    function=get_weather
)
```

#### Memory
- **Short-term**: Conversation state within a session
- **Long-term**: Optional persistence for future interactions

### Tool Types

#### Tool Class
Explicit tool representation with name, description, and behavior:

```python
from haystack.tools import Tool

tool = Tool(
    name="calculator",
    description="Perform mathematical calculations",
    function=calculate
)
```

#### ComponentTool
Wrap Haystack components as callable tools:

```python
from haystack.tools import ComponentTool
from haystack.components.websearch import SerperDevWebSearch

web_search = SerperDevWebSearch()
search_tool = ComponentTool(
    component=web_search,
    name="web_search",
    description="Search the web for information"
)
```

#### @tool Decorator
Create tools from Python functions:

```python
from haystack.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b
```

#### Toolset
Group multiple tools together:

```python
from haystack.tools import Toolset

toolset = Toolset(tools=[tool1, tool2, tool3])
agent = Agent(chat_generator=generator, tools=toolset)
```

## Pipeline-Based Architecture

Haystack's strength lies in its pipeline architecture, where components are connected to form workflows.

### Pipeline Components

```
haystack/components/
├── agents/         # Agent components
├── builders/       # Prompt builders
├── converters/     # Document converters
├── embedders/      # Embedding models
├── generators/     # Text generators
├── readers/        # Document readers
├── retrievers/     # Document retrievers
├── routers/        # Conditional routing
├── rankers/        # Re-ranking components
├── preprocessors/  # Text preprocessing
├── writers/        # Document writers
└── tools/          # Tool components
```

### Building Pipelines

```python
from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.routers import ConditionalRouter
from haystack.components.tools import ToolInvoker

pipeline = Pipeline()

# Add components
pipeline.add_component("generator", OpenAIChatGenerator())
pipeline.add_component("router", ConditionalRouter(routes=routes))
pipeline.add_component("tool_invoker", ToolInvoker(tools=tools))

# Connect components
pipeline.connect("generator", "router")
pipeline.connect("router.tool_calls", "tool_invoker")
pipeline.connect("tool_invoker", "generator")
```

### Agent Pipeline Flow

1. **Initial prompt/system message** defines role and objectives
2. **Chat Generator** analyzes user input
3. Returns either assistant reply or tool-call instruction
4. **Router** (ConditionalRouter) directs output:
   - Tool calls present: branch to tool-invocation flow
   - No tool calls: return final reply
5. **ToolInvoker** executes selected tools
6. **MessageCollector** stores question and tool responses
7. Feed back into generator for final response

## Component Examples

### Chat Generator

```python
from haystack.components.generators.chat import OpenAIChatGenerator

generator = OpenAIChatGenerator(
    model="gpt-4o",
    system_prompt="You are a helpful assistant."
)
```

### Document Retriever

```python
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

document_store = InMemoryDocumentStore()
retriever = InMemoryBM25Retriever(document_store=document_store)
```

### Conditional Router

```python
from haystack.components.routers import ConditionalRouter

routes = [
    {
        "condition": "{{replies[0].tool_calls | length > 0}}",
        "output": "tool_calls",
        "output_name": "tool_calls"
    },
    {
        "condition": "{{replies[0].tool_calls | length == 0}}",
        "output": "replies",
        "output_name": "final_reply"
    }
]

router = ConditionalRouter(routes=routes)
```

### Tool Invoker

```python
from haystack.components.tools import ToolInvoker

invoker = ToolInvoker(tools=[tool1, tool2])
```

## RAG Pipeline Example

```python
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers import InMemoryBM25Retriever

# Create pipeline
rag_pipeline = Pipeline()

# Add components
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", PromptBuilder(template=template))
rag_pipeline.add_component("llm", OpenAIGenerator())

# Connect
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

# Run
result = rag_pipeline.run({
    "retriever": {"query": "What is machine learning?"},
    "prompt_builder": {"query": "What is machine learning?"}
})
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Technology Agnostic** | Easily switch between vendors and models |
| **Explicit** | Clear component connections and data flow |
| **Flexible** | Database access, file conversion, cleaning, training, eval, inference |
| **Extensible** | Easy to create custom components |
| **Pipeline-First** | Composable, debuggable workflows |
| **Production Ready** | Built for enterprise deployment |

### Supported Integrations

#### LLM Providers
- OpenAI, Azure OpenAI
- Anthropic Claude
- Google Gemini
- Cohere
- HuggingFace models
- Ollama (local)

#### Vector Stores
- Elasticsearch
- Pinecone, Weaviate, Qdrant
- Chroma, Milvus
- Azure AI Search
- PostgreSQL (pgvector)

#### Document Converters
- PDF, Word, PowerPoint
- HTML, Markdown
- Audio/Video transcription

## Use Cases

Haystack examples from production:
- **RAG**: Question answering over documents
- **Semantic Search**: Find documents by meaning
- **Agents**: Complex decision-making systems
- **Conversational AI**: Chatbots with tool use
- **Document Processing**: Ingestion and indexing pipelines

### Companies Using Haystack
- Apple, Meta, Netflix
- NVIDIA, Intel, Databricks
- Airbus, LEGO
- German Federal Ministry
- Zeit Online, Rakuten

## Hayhooks (REST API Deployment)

Deploy Haystack pipelines as REST APIs:

```python
# Wrap pipelines with custom logic
# Expose via HTTP endpoints
# OpenAI-compatible chat completion endpoints
# Compatible with open-webui
```

## Best Use Cases

- **Production RAG systems**: Enterprise-grade document Q&A
- **Complex pipelines**: Multi-step processing workflows
- **Hybrid search**: Combine keyword and semantic search
- **Enterprise applications**: Where reliability and observability matter
- **Custom workflows**: Unique processing requirements
- **Agent + RAG combinations**: Agents that need retrieval

## Limitations

- **Learning curve**: Pipeline concepts take time to master
- **Verbosity**: More code for simple use cases
- **Agent focus**: Newer compared to other agent frameworks
- **Documentation**: Can be overwhelming with many options

## Installation

```bash
pip install haystack-ai

# From main branch (latest features)
pip install git+https://github.com/deepset-ai/haystack.git@main
```

## Complete Agent Example

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.tools import Tool, tool, ComponentTool
from haystack.components.websearch import SerperDevWebSearch

# Define tools
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

# Component as tool
web_search = SerperDevWebSearch()
search_tool = ComponentTool(
    component=web_search,
    name="web_search",
    description="Search the web for current information"
)

# Create agent
agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o"),
    tools=[calculate, search_tool],
    system_prompt="""You are a helpful assistant.
    Use tools to find information and perform calculations."""
)

# Run agent
result = agent.run(
    messages=[{"role": "user", "content": "What is 15% of the population of France?"}]
)
print(result["replies"][0].text)
```

## References

- [Haystack GitHub](https://github.com/deepset-ai/haystack)
- [Documentation](https://docs.haystack.deepset.ai/)
- [Tutorials](https://haystack.deepset.ai/tutorials)
- [Cookbook](https://haystack.deepset.ai/cookbook)
- [Integrations](https://github.com/deepset-ai/haystack-integrations)
- [Hayhooks (REST API)](https://github.com/deepset-ai/hayhooks)
- [Discord Community](https://discord.com/invite/xYvH6drSmA)
