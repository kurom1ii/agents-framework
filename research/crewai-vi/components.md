# Tham chiếu các thành phần CrewAI

Tài liệu này cung cấp tham chiếu chi tiết cho tất cả các thành phần cốt lõi của CrewAI.

## Mục lục

1. [Lớp Agent](#lớp-agent)
2. [Lớp Task](#lớp-task)
3. [Lớp Crew](#lớp-crew)
4. [Tools](#tools)
5. [Cấu hình LLM](#cấu-hình-llm)
6. [Memory](#memory)
7. [Knowledge](#knowledge)
8. [Flows](#flows)

---

## Lớp Agent

Lớp `Agent` đại diện cho một thực thể AI tự chủ thực hiện các task trong crew.

### Import

```python
from crewai import Agent
```

### Tham số danh tính cốt lõi (Bắt buộc)

| Tham số | Kiểu | Mô tả |
|---------|------|-------|
| `role` | `str` | Chức năng hoặc chuyên môn của agent (vd: "Senior Research Analyst") |
| `goal` | `str` | Mục tiêu của agent (vd: "Find accurate data on AI trends") |
| `backstory` | `str` | Bối cảnh và tính cách định hình hành vi |

### Cấu hình LLM

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `llm` | `Union[str, LLM, Any]` | `"gpt-4"` | Model sử dụng (string hoặc LLM instance) |
| `function_calling_llm` | `Optional[Any]` | `None` | Override LLM cho tool/function calls |

### Khả năng

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `tools` | `List[BaseTool]` | `[]` | Danh sách tools agent có thể sử dụng |
| `knowledge_sources` | `Optional[List[BaseKnowledgeSource]]` | `None` | Các cơ sở knowledge bên ngoài |
| `embedder` | `Optional[Dict[str, Any]]` | `None` | Cấu hình embedder |

### Cờ hành vi

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `verbose` | `bool` | `False` | Bật logging chi tiết |
| `cache` | `bool` | `True` | Cache kết quả tool |
| `allow_delegation` | `bool` | `False` | Cho phép ủy thác task cho agent khác |
| `allow_code_execution` | `Optional[bool]` | `False` | Cho phép chạy code |
| `code_execution_mode` | `Literal["safe","unsafe"]` | `"safe"` | "safe" sử dụng Docker |
| `multimodal` | `bool` | `False` | Bật xử lý text+image |
| `inject_date` | `bool` | `False` | Tự động chèn ngày hiện tại |
| `date_format` | `str` | `"%Y-%m-%d"` | Định dạng cho ngày được chèn |
| `reasoning` | `bool` | `False` | Bật hành vi reflect/plan |
| `respect_context_window` | `bool` | `True` | Quản lý context-window tự động |
| `use_system_prompt` | `Optional[bool]` | `True` | Bao gồm system prompt |
| `memory` | `bool` | `False` | Bật conversation memory |

### Giới hạn thực thi

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `max_iter` | `int` | `20` | Số lần lặp tối đa trước khi trả về câu trả lời tốt nhất |
| `max_rpm` | `Optional[int]` | `None` | Giới hạn số request mỗi phút |
| `max_execution_time` | `Optional[int]` | `None` | Timeout tổng thể tính bằng giây |
| `max_retry_limit` | `int` | `2` | Số lần thử lại khi lỗi |
| `max_reasoning_attempts` | `Optional[int]` | `None` | Giới hạn cho vòng lặp planning/reflection |

### Templates

| Tham số | Kiểu | Mô tả |
|---------|------|-------|
| `system_template` | `Optional[str]` | Định dạng system prompt tùy chỉnh |
| `prompt_template` | `Optional[str]` | Định dạng prompt tùy chỉnh |
| `response_template` | `Optional[str]` | Template định dạng output |

### Callbacks

| Tham số | Kiểu | Mô tả |
|---------|------|-------|
| `step_callback` | `Optional[Any]` | Hàm được gọi sau mỗi bước agent |

### Ví dụ

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

### Thực thi Agent trực tiếp

Agent có thể được chạy trực tiếp mà không cần crew:

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

## Lớp Task

Lớp `Task` đại diện cho một đơn vị công việc được giao cho agent.

### Import

```python
from crewai import Task
```

### Tham số cốt lõi

| Tham số | Kiểu | Bắt buộc | Mô tả |
|---------|------|----------|-------|
| `description` | `str` | Có | Task bao gồm những gì (prompt) |
| `expected_output` | `str` | Có | Hoàn thành nên trông như thế nào |
| `name` | `Optional[str]` | Không | Định danh tùy chọn |
| `agent` | `Optional[BaseAgent]` | Không | Agent chịu trách nhiệm (có thể được giao bởi crew) |

### Tools & Context

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `tools` | `List[BaseTool]` | `[]` | Override tools mặc định của agent cho task này |
| `context` | `Optional[List[Task]]` | `None` | Các task có output cung cấp context |

### Cấu hình Output

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `output_file` | `Optional[str]` | `None` | Đường dẫn để ghi output ra đĩa |
| `create_directory` | `Optional[bool]` | `True` | Tự động tạo thư mục cho output_file |
| `output_json` | `Optional[Type[BaseModel]]` | `None` | Pydantic model cho output JSON |
| `output_pydantic` | `Optional[Type[BaseModel]]` | `None` | Pydantic model cho structured output |
| `markdown` | `Optional[bool]` | `None` | Định dạng output trong Markdown |

### Điều khiển thực thi

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `async_execution` | `Optional[bool]` | `None` | Chạy bất đồng bộ |
| `human_input` | `Optional[bool]` | `None` | Yêu cầu đánh giá của con người |
| `config` | `Optional[Dict[str, Any]]` | `None` | Tham số task tùy ý |
| `callback` | `Optional[Any]` | `None` | Hàm thực thi sau khi hoàn thành |

### Guardrails

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `guardrail` | `Optional[Callable \| str]` | `None` | Hàm validation đơn hoặc quy tắc LLM |
| `guardrails` | `Optional[List[Callable \| str]]` | `None` | Danh sách guardrails (override guardrail) |
| `guardrail_max_retries` | `Optional[int]` | `3` | Số lần thử lại khi guardrails thất bại |

### Đối tượng TaskOutput

Output của task được bọc trong `TaskOutput`:

| Thuộc tính | Kiểu | Mô tả |
|------------|------|-------|
| `raw` | `str` | Output văn bản thô |
| `pydantic` | `Optional[BaseModel]` | Được điền nếu output_pydantic được cung cấp |
| `json_dict` | `Optional[Dict[str, Any]]` | Được điền nếu output_json được cung cấp |
| `description` | `str` | Mô tả task |
| `summary` | `str` | Tóm tắt output |
| `agent` | `str` | Agent đã thực thi task |

### Ví dụ

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

# Với context từ task khác
writing_task = Task(
    description="Write an article based on the research findings",
    expected_output="A well-structured article",
    agent=writer,
    context=[research_task]  # Sẽ nhận output của research_task
)
```

### Ví dụ Guardrails

```python
def validate_length(output):
    """Guardrail dựa trên hàm"""
    if len(output.raw) < 100:
        return (False, "Output too short, please elaborate")
    return (True, output)

task = Task(
    description="Write a detailed analysis",
    expected_output="Comprehensive analysis",
    agent=analyst,
    guardrails=[
        validate_length,
        "Ensure the output is professional and factual"  # Guardrail LLM
    ],
    guardrail_max_retries=3
)
```

---

## Lớp Crew

Lớp `Crew` điều phối các agent làm việc cùng nhau trên các task.

### Import

```python
from crewai import Crew, Process
```

### Tham số cốt lõi

| Tham số | Kiểu | Bắt buộc | Mô tả |
|---------|------|----------|-------|
| `agents` | `List[Agent]` | Có | Các agent tham gia |
| `tasks` | `List[Task]` | Có | Các task cần hoàn thành |
| `process` | `Process` | Không | Luồng thực thi (mặc định: `Process.sequential`) |

### Điều phối

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `manager_llm` | `Optional[str \| LLM]` | `None` | LLM cho manager trong hierarchical process |
| `manager_agent` | `Optional[Agent]` | `None` | Manager agent tùy chỉnh |
| `planning` | `Optional[bool]` | `None` | Bật planning trước mỗi iteration |
| `planning_llm` | `Optional[str \| LLM]` | `None` | LLM cho planning |

### Tài nguyên

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `memory` | `Optional[bool]` | `None` | Bật memory (short/long/entity) |
| `cache` | `Optional[bool]` | `True` | Cache kết quả tool |
| `knowledge_sources` | `Optional[List]` | `None` | Nguồn knowledge cấp crew |
| `embedder` | `Optional[Dict]` | `{"provider": "openai"}` | Cấu hình embedder |
| `function_calling_llm` | `Optional[LLM]` | `None` | LLM cho function calling |

### Callbacks & Logging

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `verbose` | `bool` | `False` | Độ chi tiết logging |
| `step_callback` | `Optional[Callable]` | `None` | Được gọi sau mỗi bước agent |
| `task_callback` | `Optional[Callable]` | `None` | Được gọi sau mỗi task |
| `output_log_file` | `Optional[str \| bool]` | `None` | Lưu logs (True = logs.txt) |

### Điều khiển thực thi

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `max_rpm` | `Optional[int]` | `None` | Giới hạn tốc độ cấp crew (override agents) |
| `stream` | `Optional[bool]` | `False` | Bật streaming output |

### Đối tượng CrewOutput

| Thuộc tính | Kiểu | Mô tả |
|------------|------|-------|
| `raw` | `str` | Output string thô |
| `pydantic` | `Optional[BaseModel]` | Structured output |
| `json_dict` | `Optional[Dict]` | Output JSON |
| `tasks_output` | `List[TaskOutput]` | Tất cả output của task |
| `token_usage` | `Dict` | Tóm tắt sử dụng token |

### Phương thức Kickoff

| Phương thức | Mô tả |
|-------------|-------|
| `kickoff(inputs)` | Thực thi đồng bộ |
| `akickoff(inputs)` | Thực thi async native |
| `kickoff_async(inputs)` | Async bọc thread |
| `kickoff_for_each(inputs)` | Thực thi cho mỗi input trong list |
| `akickoff_for_each(inputs)` | Async cho mỗi |

### Ví dụ

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

### Ví dụ Streaming

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

Tools mở rộng khả năng của agent với các hàm bên ngoài.

### Import

```python
from crewai.tools import BaseTool, tool
from crewai_tools import SerperDevTool, WebsiteSearchTool
```

### Tools tích hợp

CrewAI cung cấp hơn 100 tools tích hợp được tổ chức theo danh mục:

**Tools tìm kiếm**
| Tool | Mô tả |
|------|-------|
| `SerperDevTool` | Tìm kiếm web qua Serper API |
| `BraveSearchTool` | Tìm kiếm web qua Brave |
| `EXASearchTool` | Tìm kiếm web ngữ nghĩa |
| `TavilySearchTool` | Tìm kiếm hỗ trợ AI |
| `LinkupSearchTool` | Khám phá liên kết |

**Tools Web Scraping**
| Tool | Mô tả |
|------|-------|
| `ScrapeWebsiteTool` | Scraping web cơ bản |
| `WebsiteSearchTool` | Tìm kiếm trong website |
| `FirecrawlScrapeWebsiteTool` | Scraping nâng cao qua Firecrawl |
| `FirecrawlCrawlWebsiteTool` | Crawl website |
| `SeleniumScrapingTool` | Scraping dựa trên browser |
| `ScrapflyScrapeWebsiteTool` | Tích hợp Scrapfly |
| `BrowserbaseLoadTool` | Browser automation |
| `HyperbrowserLoadTool` | Tích hợp Hyperbrowser |
| `JinaScrapeWebsiteTool` | Scraping Jina AI |

**Tools tìm kiếm tài liệu**
| Tool | Mô tả |
|------|-------|
| `PDFSearchTool` | Tìm kiếm PDF |
| `DOCXSearchTool` | Tìm kiếm tài liệu Word |
| `CSVSearchTool` | Tìm kiếm file CSV |
| `JSONSearchTool` | Tìm kiếm file JSON |
| `TXTSearchTool` | Tìm kiếm file text |
| `XMLSearchTool` | Tìm kiếm file XML |
| `MDXSearchTool` | Tìm kiếm file MDX |

**Tools hệ thống File**
| Tool | Mô tả |
|------|-------|
| `FileReadTool` | Đọc file |
| `FileWriterTool` | Ghi file |
| `DirectoryReadTool` | Đọc thư mục |
| `DirectorySearchTool` | Tìm kiếm thư mục |
| `FileCompressorTool` | Nén file |

**Tools Code**
| Tool | Mô tả |
|------|-------|
| `CodeInterpreterTool` | Thực thi code Python |
| `CodeDocsSearchTool` | Tìm kiếm tài liệu code |
| `GithubSearchTool` | Tìm kiếm GitHub |

**Tools RAG & Vector**
| Tool | Mô tả |
|------|-------|
| `RagTool` | Truy xuất RAG |
| `QdrantVectorSearchTool` | Tìm kiếm vector Qdrant |
| `MongoDBVectorSearchTool` | Tìm kiếm vector MongoDB |
| `WeaviateVectorSearchTool` | Tìm kiếm vector Weaviate |
| `CouchbaseFTSVectorSearchTool` | Tìm kiếm Couchbase |
| `SingleStoreSearchTool` | Tìm kiếm SingleStore |

**Tools Vision & Image**
| Tool | Mô tả |
|------|-------|
| `VisionTool` | Phân tích hình ảnh |
| `DallETool` | Tạo ảnh DALL-E |
| `OCRTool` | Nhận dạng ký tự quang học |

**Tools AWS**
| Tool | Mô tả |
|------|-------|
| `S3ReaderTool` | Đọc từ S3 |
| `S3WriterTool` | Ghi vào S3 |
| `BedrockInvokeAgentTool` | Gọi agent Bedrock |
| `BedrockKBRetrieverTool` | Knowledge base Bedrock |

**Tools Database**
| Tool | Mô tả |
|------|-------|
| `MySQLSearchTool` | Truy vấn MySQL |
| `DatabricksQueryTool` | Truy vấn Databricks |
| `SnowflakeSearchTool` | Truy vấn Snowflake |
| `NL2SQLTool` | Ngôn ngữ tự nhiên sang SQL |

**Tools YouTube**
| Tool | Mô tả |
|------|-------|
| `YoutubeVideoSearchTool` | Tìm kiếm video YouTube |
| `YoutubeChannelSearchTool` | Tìm kiếm kênh YouTube |

**Tools tích hợp**
| Tool | Mô tả |
|------|-------|
| `ComposioTool` | Tích hợp Composio |
| `ZapierActionTool` | Automation Zapier |
| `MCPServerAdapter` | MCP server adapter |
| `LlamaIndexTool` | Tích hợp LlamaIndex |
| `ApifyActorsTool` | Apify actors |
| `MultiOnTool` | Browser automation MultiOn |

**Tools chuyên biệt khác**
| Tool | Mô tả |
|------|-------|
| `ArxivPaperTool` | Tìm kiếm bài báo arXiv |
| `AIMindTool` | Tích hợp AI Mind |
| `PatronusEvalTool` | Đánh giá Patronus |
| `ContextualAIQueryTool` | AI bối cảnh |
| `ScrapeElementFromWebsiteTool` | Scrape phần tử cụ thể |
| `ParallelSearchTool` | Thao tác tìm kiếm song song |

### Tạo Tools tùy chỉnh

#### Phương pháp 1: Kế thừa BaseTool

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

#### Phương pháp 2: Sử dụng @tool Decorator

```python
from crewai.tools import tool

@tool("Web Scraper")
def scrape_website(url: str) -> str:
    """Scrape content from a website URL"""
    import requests
    response = requests.get(url)
    return response.text[:5000]
```

### Tools Async

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

### Caching Tool

```python
def my_cache_function(args, result):
    """Trả về True để cache, False để bỏ qua"""
    return len(result) > 100

tool = MyTool()
tool.cache_function = my_cache_function
```

---

## Cấu hình LLM

Cấu hình các nhà cung cấp LLM khác nhau.

### Import

```python
from crewai import LLM
```

### Tham số chung

| Tham số | Kiểu | Mô tả |
|---------|------|-------|
| `model` | `str` | Model có tiền tố nhà cung cấp (vd: "openai/gpt-4o") |
| `api_key` | `str` | API key của nhà cung cấp |
| `temperature` | `float` | Nhiệt độ sampling |
| `max_tokens` | `int` | Số token tối đa |
| `timeout` | `int` | Timeout request |
| `stream` | `bool` | Bật streaming |
| `response_format` | `Type[BaseModel]` | Schema structured output |

### Ví dụ các nhà cung cấp

```python
from crewai import LLM

# OpenAI
openai_llm = LLM(
    model="openai/gpt-4o",
    temperature=0.7,
    max_tokens=4000
)

# Anthropic (bắt buộc max_tokens!)
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

### Biến môi trường

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

Cho phép agent nhớ qua các tương tác.

### Bật Memory

```python
crew = Crew(
    agents=[...],
    tasks=[...],
    memory=True  # Bật memory ngắn hạn, dài hạn, entity
)
```

### Lưu trữ tùy chỉnh

```python
import os
os.environ["CREWAI_STORAGE_DIR"] = "./my_storage"

# Hoặc vị trí SQLite tùy chỉnh
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

### Embedder tùy chỉnh

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

Cung cấp cho agent thông tin chuyên ngành.

### Nguồn tích hợp

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

### Ví dụ

```python
from crewai import Agent, Crew
from crewai.knowledge.source import StringKnowledgeSource

# Tạo nguồn knowledge
company_info = StringKnowledgeSource(
    content="""
    Our company was founded in 2020. We specialize in AI solutions.
    Our main products are: AI Assistant, Data Analyzer, and AutoML Platform.
    """
)

# Agent với knowledge
support_agent = Agent(
    role="Customer Support",
    goal="Help customers with product questions",
    backstory="Expert on company products",
    knowledge_sources=[company_info]
)

# Hoặc knowledge cấp crew
crew = Crew(
    agents=[support_agent],
    tasks=[...],
    knowledge_sources=[company_info]  # Chia sẻ cho tất cả agents
)
```

---

## Flows

Điều phối workflow hướng sự kiện.

### Import

```python
from crewai.flow.flow import Flow, start, listen, router, or_, and_
```

### Decorators

| Decorator | Mô tả |
|-----------|-------|
| `@start()` | Phương thức điểm vào |
| `@listen(method)` | Kích hoạt khi method phát ra output |
| `@router()` | Trả về routing labels cho luồng có điều kiện |
| `or_(m1, m2)` | Kích hoạt khi bất kỳ method nào phát ra |
| `and_(m1, m2)` | Kích hoạt khi tất cả method phát ra |
| `@human_feedback` | Tạm dừng để đánh giá của con người |
| `@persist` | Bật lưu trữ trạng thái |

### Ví dụ Flow cơ bản

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
        # Có thể khởi chạy research crew ở đây
        self.state.research = f"Research on {topic}..."
        return self.state.research

    @listen(research_topic)
    def write_article(self, research):
        self.state.article = f"Article based on: {research}"
        return self.state.article

flow = ContentFlow()
result = flow.kickoff()
```

### Ví dụ Router

```python
from crewai.flow.flow import Flow, start, listen, router

class ReviewFlow(Flow):

    @start()
    def analyze_content(self):
        # Phân tích chất lượng nội dung
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

### Flow với Crews

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

@persist  # Persistence cấp class
class PersistentFlow(Flow):

    @start()
    def step_one(self):
        return "Step one complete"
```

---

## Các loại Process

### Process.sequential

Các task thực thi theo thứ tự được định nghĩa. Output chảy như context.

```python
from crewai import Crew, Process

crew = Crew(
    agents=[agent1, agent2, agent3],
    tasks=[task1, task2, task3],
    process=Process.sequential
)
```

### Process.hierarchical

Manager phân công và xác nhận. Yêu cầu manager_llm hoặc manager_agent.

```python
from crewai import Crew, Process

crew = Crew(
    agents=[specialist1, specialist2],
    tasks=[task1, task2],
    process=Process.hierarchical,
    manager_llm="openai/gpt-4o"
)
```
