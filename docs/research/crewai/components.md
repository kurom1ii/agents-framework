# CrewAI Components Reference

This document provides a detailed reference for all core CrewAI components.

## Table of Contents

1. [Agent Class](#agent-class)
2. [Task Class](#task-class)
3. [Crew Class](#crew-class)
4. [Tools](#tools)
5. [LLM Configuration](#llm-configuration)
6. [Memory](#memory)
7. [Knowledge](#knowledge)
8. [Flows](#flows)

---

## Agent Class

The `Agent` class represents an autonomous AI entity that performs tasks within a crew.

### Import

```python
from crewai import Agent
```

### Core Identity Parameters (Required)

| Parameter | Type | Description |
|-----------|------|-------------|
| `role` | `str` | The agent's function or expertise (e.g., "Senior Research Analyst") |
| `goal` | `str` | The agent's objective (e.g., "Find accurate data on AI trends") |
| `backstory` | `str` | Context and personality that shapes behavior |

### LLM Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `Union[str, LLM, Any]` | `"gpt-4"` | Model to use (string or LLM instance) |
| `function_calling_llm` | `Optional[Any]` | `None` | Override LLM for tool/function calls |

### Capabilities

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tools` | `List[BaseTool]` | `[]` | List of tools the agent can use |
| `knowledge_sources` | `Optional[List[BaseKnowledgeSource]]` | `None` | External knowledge bases |
| `embedder` | `Optional[Dict[str, Any]]` | `None` | Embedder configuration |

### Behavior Flags

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | `bool` | `False` | Enable detailed logging |
| `cache` | `bool` | `True` | Cache tool results |
| `allow_delegation` | `bool` | `False` | Allow delegating tasks to other agents |
| `allow_code_execution` | `Optional[bool]` | `False` | Permit running code |
| `code_execution_mode` | `Literal["safe","unsafe"]` | `"safe"` | "safe" uses Docker |
| `multimodal` | `bool` | `False` | Enable text+image processing |
| `inject_date` | `bool` | `False` | Auto-insert current date |
| `date_format` | `str` | `"%Y-%m-%d"` | Format for injected date |
| `reasoning` | `bool` | `False` | Enable reflect/plan behavior |
| `respect_context_window` | `bool` | `True` | Auto context-window management |
| `use_system_prompt` | `Optional[bool]` | `True` | Include system prompt |
| `memory` | `bool` | `False` | Enable conversation memory |

### Execution Limits

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_iter` | `int` | `20` | Maximum iterations before returning best answer |
| `max_rpm` | `Optional[int]` | `None` | Rate-limit requests per minute |
| `max_execution_time` | `Optional[int]` | `None` | Overall timeout in seconds |
| `max_retry_limit` | `int` | `2` | Retry attempts on error |
| `max_reasoning_attempts` | `Optional[int]` | `None` | Limit for planning/reflection loops |

### Templates

| Parameter | Type | Description |
|-----------|------|-------------|
| `system_template` | `Optional[str]` | Custom system prompt format |
| `prompt_template` | `Optional[str]` | Custom prompt format |
| `response_template` | `Optional[str]` | Output formatting template |

### Callbacks

| Parameter | Type | Description |
|-----------|------|-------------|
| `step_callback` | `Optional[Any]` | Function invoked after each agent step |

### Example

```python
from crewai import Agent
from crewai_tools import SerperDevTool, WebsiteSearchTool

researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI and data science",
    backstory="""You work at a leading tech think tank. Your expertise
    lies in identifying emerging trends and analyzing complex data.""",
    tools=[SerperDevTool(), WebsiteSearchTool()],
    llm="openai/gpt-4o",
    verbose=True,
    allow_delegation=False,
    memory=True,
    max_iter=15,
    max_rpm=10
)
```

### Direct Agent Execution

Agents can be run directly without a crew:

```python
from crewai import Agent

agent = Agent(
    role="Poet",
    goal="Write beautiful poetry",
    backstory="You are a renowned poet"
)

result = agent.kickoff("Write a haiku about programming")
print(result.raw)
```

---

## Task Class

The `Task` class represents a unit of work assigned to an agent.

### Import

```python
from crewai import Task
```

### Core Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `description` | `str` | Yes | What the task entails (the prompt) |
| `expected_output` | `str` | Yes | What completion should look like |
| `name` | `Optional[str]` | No | Optional identifier |
| `agent` | `Optional[BaseAgent]` | No | Responsible agent (can be assigned by crew) |

### Tools & Context

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tools` | `List[BaseTool]` | `[]` | Override agent's default tools for this task |
| `context` | `Optional[List[Task]]` | `None` | Tasks whose outputs provide context |

### Output Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_file` | `Optional[str]` | `None` | Path to write output to disk |
| `create_directory` | `Optional[bool]` | `True` | Auto-create directories for output_file |
| `output_json` | `Optional[Type[BaseModel]]` | `None` | Pydantic model for JSON output |
| `output_pydantic` | `Optional[Type[BaseModel]]` | `None` | Pydantic model for structured output |
| `markdown` | `Optional[bool]` | `None` | Format output in Markdown |

### Execution Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `async_execution` | `Optional[bool]` | `None` | Run asynchronously |
| `human_input` | `Optional[bool]` | `None` | Require human review |
| `config` | `Optional[Dict[str, Any]]` | `None` | Arbitrary task parameters |
| `callback` | `Optional[Any]` | `None` | Function executed after completion |

### Guardrails

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `guardrail` | `Optional[Callable \| str]` | `None` | Single validation function or LLM rule |
| `guardrails` | `Optional[List[Callable \| str]]` | `None` | List of guardrails (overrides guardrail) |
| `guardrail_max_retries` | `Optional[int]` | `3` | Retry attempts when guardrails fail |

### TaskOutput Object

Task outputs are wrapped in `TaskOutput`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `raw` | `str` | Raw textual output |
| `pydantic` | `Optional[BaseModel]` | Populated if output_pydantic given |
| `json_dict` | `Optional[Dict[str, Any]]` | Populated if output_json given |
| `description` | `str` | Task description |
| `summary` | `str` | Output summary |
| `agent` | `str` | Agent that executed task |

### Example

```python
from crewai import Task
from pydantic import BaseModel

class ResearchReport(BaseModel):
    topic: str
    findings: list[str]
    conclusion: str

research_task = Task(
    description="""Research the latest developments in quantum computing.
    Focus on practical applications and industry adoption.""",
    expected_output="A comprehensive research report with key findings",
    agent=researcher,
    output_pydantic=ResearchReport,
    output_file="research_report.md"
)

# With context from another task
writing_task = Task(
    description="Write an article based on the research findings",
    expected_output="A well-structured article",
    agent=writer,
    context=[research_task]  # Will receive research_task output
)
```

### Guardrails Example

```python
def validate_length(output):
    """Function-based guardrail"""
    if len(output.raw) < 100:
        return (False, "Output too short, please elaborate")
    return (True, output)

task = Task(
    description="Write a detailed analysis",
    expected_output="Comprehensive analysis",
    agent=analyst,
    guardrails=[
        validate_length,
        "Ensure the output is professional and factual"  # LLM guardrail
    ],
    guardrail_max_retries=3
)
```

---

## Crew Class

The `Crew` class orchestrates agents working together on tasks.

### Import

```python
from crewai import Crew, Process
```

### Core Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agents` | `List[Agent]` | Yes | Participating agents |
| `tasks` | `List[Task]` | Yes | Tasks to complete |
| `process` | `Process` | No | Execution flow (default: `Process.sequential`) |

### Orchestration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `manager_llm` | `Optional[str \| LLM]` | `None` | LLM for manager in hierarchical process |
| `manager_agent` | `Optional[Agent]` | `None` | Custom manager agent |
| `planning` | `Optional[bool]` | `None` | Enable planning before each iteration |
| `planning_llm` | `Optional[str \| LLM]` | `None` | LLM for planning |

### Resources

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `memory` | `Optional[bool]` | `None` | Enable memory (short/long/entity) |
| `cache` | `Optional[bool]` | `True` | Cache tool results |
| `knowledge_sources` | `Optional[List]` | `None` | Crew-level knowledge sources |
| `embedder` | `Optional[Dict]` | `{"provider": "openai"}` | Embedder configuration |
| `function_calling_llm` | `Optional[LLM]` | `None` | LLM for function calling |

### Callbacks & Logging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | `bool` | `False` | Logging verbosity |
| `step_callback` | `Optional[Callable]` | `None` | Called after each agent step |
| `task_callback` | `Optional[Callable]` | `None` | Called after each task |
| `output_log_file` | `Optional[str \| bool]` | `None` | Save logs (True = logs.txt) |

### Execution Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_rpm` | `Optional[int]` | `None` | Crew-level rate limit (overrides agents) |
| `stream` | `Optional[bool]` | `False` | Enable streaming output |

### CrewOutput Object

| Attribute | Type | Description |
|-----------|------|-------------|
| `raw` | `str` | Raw string output |
| `pydantic` | `Optional[BaseModel]` | Structured output |
| `json_dict` | `Optional[Dict]` | JSON output |
| `tasks_output` | `List[TaskOutput]` | All task outputs |
| `token_usage` | `Dict` | Token usage summary |

### Kickoff Methods

| Method | Description |
|--------|-------------|
| `kickoff(inputs)` | Synchronous execution |
| `akickoff(inputs)` | Native async execution |
| `kickoff_async(inputs)` | Thread-wrapped async |
| `kickoff_for_each(inputs)` | Execute for each input in list |
| `akickoff_for_each(inputs)` | Async for each |

### Example

```python
from crewai import Crew, Process, Agent, Task

# Sequential process
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,
    memory=True,
    verbose=True,
    output_log_file="crew_log.json"
)

result = crew.kickoff(inputs={"topic": "AI Trends 2025"})
print(result.raw)

# Hierarchical process
hierarchical_crew = Crew(
    agents=[specialist_1, specialist_2, specialist_3],
    tasks=[complex_task_1, complex_task_2],
    process=Process.hierarchical,
    manager_llm="openai/gpt-4o",
    verbose=True
)
```

### Streaming Example

```python
crew = Crew(
    agents=[...],
    tasks=[...],
    stream=True
)

streaming = crew.kickoff(inputs={...})

for chunk in streaming:
    print(chunk.content, end="", flush=True)

final_result = streaming.result
```

---

## Tools

Tools extend agent capabilities with external functions.

### Import

```python
from crewai.tools import BaseTool, tool
from crewai_tools import SerperDevTool, WebsiteSearchTool
```

### Built-in Tools

CrewAI provides 100+ built-in tools organized by category:

**Search Tools**
| Tool | Description |
|------|-------------|
| `SerperDevTool` | Web search via Serper API |
| `BraveSearchTool` | Web search via Brave |
| `EXASearchTool` | Semantic web search |
| `TavilySearchTool` | AI-powered search |
| `LinkupSearchTool` | Link discovery |

**Web Scraping Tools**
| Tool | Description |
|------|-------------|
| `ScrapeWebsiteTool` | Basic web scraping |
| `WebsiteSearchTool` | Search within websites |
| `FirecrawlScrapeWebsiteTool` | Advanced scraping via Firecrawl |
| `FirecrawlCrawlWebsiteTool` | Website crawling |
| `SeleniumScrapingTool` | Browser-based scraping |
| `ScrapflyScrapeWebsiteTool` | Scrapfly integration |
| `BrowserbaseLoadTool` | Browser automation |
| `HyperbrowserLoadTool` | Hyperbrowser integration |
| `JinaScrapeWebsiteTool` | Jina AI scraping |

**Document Search Tools**
| Tool | Description |
|------|-------------|
| `PDFSearchTool` | Search PDFs |
| `DOCXSearchTool` | Search Word documents |
| `CSVSearchTool` | Search CSV files |
| `JSONSearchTool` | Search JSON files |
| `TXTSearchTool` | Search text files |
| `XMLSearchTool` | Search XML files |
| `MDXSearchTool` | Search MDX files |

**File System Tools**
| Tool | Description |
|------|-------------|
| `FileReadTool` | Read files |
| `FileWriterTool` | Write files |
| `DirectoryReadTool` | Read directories |
| `DirectorySearchTool` | Search directories |
| `FileCompressorTool` | Compress files |

**Code Tools**
| Tool | Description |
|------|-------------|
| `CodeInterpreterTool` | Execute Python code |
| `CodeDocsSearchTool` | Search code documentation |
| `GithubSearchTool` | Search GitHub |

**RAG & Vector Tools**
| Tool | Description |
|------|-------------|
| `RagTool` | RAG retrieval |
| `QdrantVectorSearchTool` | Qdrant vector search |
| `MongoDBVectorSearchTool` | MongoDB vector search |
| `WeaviateVectorSearchTool` | Weaviate vector search |
| `CouchbaseFTSVectorSearchTool` | Couchbase search |
| `SingleStoreSearchTool` | SingleStore search |

**Vision & Image Tools**
| Tool | Description |
|------|-------------|
| `VisionTool` | Image analysis |
| `DallETool` | DALL-E image generation |
| `OCRTool` | Optical character recognition |

**AWS Tools**
| Tool | Description |
|------|-------------|
| `S3ReaderTool` | Read from S3 |
| `S3WriterTool` | Write to S3 |
| `BedrockInvokeAgentTool` | Invoke Bedrock agents |
| `BedrockKBRetrieverTool` | Bedrock knowledge base |

**Database Tools**
| Tool | Description |
|------|-------------|
| `MySQLSearchTool` | MySQL queries |
| `DatabricksQueryTool` | Databricks queries |
| `SnowflakeSearchTool` | Snowflake queries |
| `NL2SQLTool` | Natural language to SQL |

**YouTube Tools**
| Tool | Description |
|------|-------------|
| `YoutubeVideoSearchTool` | Search YouTube videos |
| `YoutubeChannelSearchTool` | Search YouTube channels |

**Integration Tools**
| Tool | Description |
|------|-------------|
| `ComposioTool` | Composio integrations |
| `ZapierActionTool` | Zapier automations |
| `MCPServerAdapter` | MCP server adapter |
| `LlamaIndexTool` | LlamaIndex integration |
| `ApifyActorsTool` | Apify actors |
| `MultiOnTool` | MultiOn browser automation |

**Other Specialized Tools**
| Tool | Description |
|------|-------------|
| `ArxivPaperTool` | Search arXiv papers |
| `AIMindTool` | AI Mind integration |
| `PatronusEvalTool` | Patronus evaluation |
| `ContextualAIQueryTool` | Contextual AI |
| `ScrapeElementFromWebsiteTool` | Scrape specific elements |
| `ParallelSearchTool` | Parallel search operations |

### Creating Custom Tools

#### Method 1: Subclass BaseTool

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class CalculatorInput(BaseModel):
    operation: str = Field(..., description="Math operation to perform")
    numbers: list[float] = Field(..., description="Numbers to calculate")

class CalculatorTool(BaseTool):
    name: str = "Calculator"
    description: str = "Perform mathematical calculations"
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, operation: str, numbers: list[float]) -> str:
        if operation == "add":
            return str(sum(numbers))
        elif operation == "multiply":
            result = 1
            for n in numbers:
                result *= n
            return str(result)
        return "Unknown operation"

calculator = CalculatorTool()
```

#### Method 2: Use @tool Decorator

```python
from crewai.tools import tool

@tool("Web Scraper")
def scrape_website(url: str) -> str:
    """Scrape content from a website URL"""
    import requests
    response = requests.get(url)
    return response.text[:5000]
```

### Async Tools

```python
from crewai.tools import tool

@tool("Async Fetcher")
async def fetch_data(url: str) -> str:
    """Fetch data asynchronously"""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
```

### Tool Caching

```python
def my_cache_function(args, result):
    """Return True to cache, False to skip"""
    return len(result) > 100

tool = MyTool()
tool.cache_function = my_cache_function
```

---

## LLM Configuration

Configure different LLM providers.

### Import

```python
from crewai import LLM
```

### Common Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str` | Provider-prefixed model (e.g., "openai/gpt-4o") |
| `api_key` | `str` | Provider API key |
| `temperature` | `float` | Sampling temperature |
| `max_tokens` | `int` | Maximum tokens |
| `timeout` | `int` | Request timeout |
| `stream` | `bool` | Enable streaming |
| `response_format` | `Type[BaseModel]` | Structured output schema |

### Provider Examples

```python
from crewai import LLM

# OpenAI
openai_llm = LLM(
    model="openai/gpt-4o",
    temperature=0.7,
    max_tokens=4000
)

# Anthropic (max_tokens required!)
anthropic_llm = LLM(
    model="anthropic/claude-3-5-sonnet-20241022",
    max_tokens=4096,
    temperature=0.5
)

# Google Gemini
gemini_llm = LLM(
    model="gemini/gemini-1.5-pro",
    temperature=0.7
)

# Azure OpenAI
azure_llm = LLM(
    model="azure/my-deployment",
    api_key="...",
    base_url="https://my-resource.openai.azure.com"
)

# Ollama (local)
ollama_llm = LLM(
    model="ollama/llama2",
    base_url="http://localhost:11434"
)
```

### Environment Variables

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=...

# Google
GOOGLE_API_KEY=...

# Azure
AZURE_API_KEY=...
AZURE_ENDPOINT=...
```

---

## Memory

Enable agents to remember across interactions.

### Enable Memory

```python
crew = Crew(
    agents=[...],
    tasks=[...],
    memory=True  # Enables short-term, long-term, entity memory
)
```

### Custom Storage

```python
import os
os.environ["CREWAI_STORAGE_DIR"] = "./my_storage"

# Or custom SQLite location
from crewai.memory import LongTermMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage

crew = Crew(
    agents=[...],
    tasks=[...],
    memory=True,
    long_term_memory=LongTermMemory(
        storage=LTMSQLiteStorage(db_path="./storage/memory.db")
    )
)
```

### Custom Embedder

```python
crew = Crew(
    agents=[...],
    tasks=[...],
    memory=True,
    embedder={
        "provider": "ollama",
        "config": {
            "model": "mxbai-embed-large",
            "url": "http://localhost:11434/api/embeddings"
        }
    }
)
```

---

## Knowledge

Provide agents with domain-specific information.

### Built-in Sources

```python
from crewai.knowledge.source import (
    StringKnowledgeSource,
    TextFileKnowledgeSource,
    PDFKnowledgeSource,
    CSVKnowledgeSource,
    ExcelKnowledgeSource,
    JSONKnowledgeSource
)
```

### Example

```python
from crewai import Agent, Crew
from crewai.knowledge.source import StringKnowledgeSource

# Create knowledge source
company_info = StringKnowledgeSource(
    content="""
    Our company was founded in 2020. We specialize in AI solutions.
    Our main products are: AI Assistant, Data Analyzer, and AutoML Platform.
    """
)

# Agent with knowledge
support_agent = Agent(
    role="Customer Support",
    goal="Help customers with product questions",
    backstory="Expert on company products",
    knowledge_sources=[company_info]
)

# Or crew-level knowledge
crew = Crew(
    agents=[support_agent],
    tasks=[...],
    knowledge_sources=[company_info]  # Shared by all agents
)
```

---

## Flows

Event-driven workflow orchestration.

### Import

```python
from crewai.flow.flow import Flow, start, listen, router, or_, and_
```

### Decorators

| Decorator | Description |
|-----------|-------------|
| `@start()` | Entry point method |
| `@listen(method)` | Triggers when method emits output |
| `@router()` | Returns routing labels for conditional flow |
| `or_(m1, m2)` | Triggers when any method emits |
| `and_(m1, m2)` | Triggers when all methods emit |
| `@human_feedback` | Pause for human review |
| `@persist` | Enable state persistence |

### Basic Flow Example

```python
from crewai.flow.flow import Flow, start, listen
from pydantic import BaseModel

class ContentState(BaseModel):
    topic: str = ""
    research: str = ""
    article: str = ""

class ContentFlow(Flow[ContentState]):

    @start()
    def get_topic(self):
        self.state.topic = "AI Trends"
        return self.state.topic

    @listen(get_topic)
    def research_topic(self, topic):
        # Could kick off a research crew here
        self.state.research = f"Research on {topic}..."
        return self.state.research

    @listen(research_topic)
    def write_article(self, research):
        self.state.article = f"Article based on: {research}"
        return self.state.article

flow = ContentFlow()
result = flow.kickoff()
```

### Router Example

```python
from crewai.flow.flow import Flow, start, listen, router

class ReviewFlow(Flow):

    @start()
    def analyze_content(self):
        # Analyze content quality
        quality_score = 0.8
        return quality_score

    @router(analyze_content)
    def route_by_quality(self, score):
        if score > 0.7:
            return "high_quality"
        else:
            return "needs_improvement"

    @listen("high_quality")
    def publish(self):
        return "Published!"

    @listen("needs_improvement")
    def revise(self):
        return "Sent for revision"
```

### Flow with Crews

```python
from crewai.flow.flow import Flow, start, listen
from crewai import Crew

class MultiCrewFlow(Flow):

    @start()
    def research_phase(self):
        research_crew = Crew(agents=[...], tasks=[...])
        return research_crew.kickoff(inputs={...})

    @listen(research_phase)
    def writing_phase(self, research_result):
        writing_crew = Crew(agents=[...], tasks=[...])
        return writing_crew.kickoff(inputs={"research": research_result.raw})
```

### Persistence

```python
from crewai.flow.flow import Flow, start, persist
from crewai.flow.persistence import SQLiteFlowPersistence

@persist  # Class-level persistence
class PersistentFlow(Flow):

    @start()
    def step_one(self):
        return "Step one complete"
```

---

## Process Types

### Process.sequential

Tasks execute in defined order. Output flows as context.

```python
from crewai import Crew, Process

crew = Crew(
    agents=[agent1, agent2, agent3],
    tasks=[task1, task2, task3],
    process=Process.sequential
)
```

### Process.hierarchical

Manager delegates and validates. Requires manager_llm or manager_agent.

```python
from crewai import Crew, Process

crew = Crew(
    agents=[specialist1, specialist2],
    tasks=[task1, task2],
    process=Process.hierarchical,
    manager_llm="openai/gpt-4o"
)
```
