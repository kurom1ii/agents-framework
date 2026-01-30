# Haystack Agents

## Tổng Quan

**Haystack** của deepset là framework điều phối AI end-to-end để xây dựng các ứng dụng LLM sẵn sàng production. Ban đầu tập trung vào tìm kiếm và trả lời câu hỏi, Haystack đã phát triển thành một framework toàn diện hỗ trợ RAG, agents và pipelines phức tạp.

- **Repository**: https://github.com/deepset-ai/haystack
- **Stars**: ~24,000
- **Ngôn ngữ**: Python
- **License**: Apache 2.0
- **Tài liệu**: https://docs.haystack.deepset.ai/

## Triết Lý Cốt Lõi

Haystack nhấn mạnh **cách tiếp cận dựa trên pipeline** trong đó các thành phần được kết nối để tạo thành các workflow phức tạp. Điều này khiến nó rất linh hoạt, rõ ràng và có thể mở rộng. Framework không phụ thuộc công nghệ, cho phép developer dễ dàng chọn và thay đổi vendors.

## Các Khái Niệm Chính

### Agents trong Haystack

Một AI agent trong Haystack:
- Xử lý queries (text, hình ảnh, audio)
- Truy xuất thông tin
- Tạo phản hồi
- Thực hiện hành động sử dụng tools
- Lập kế hoạch và thích ứng sử dụng memory và lập luận

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

### Các Thành Phần Cốt Lõi

#### LLM như Bộ Não
Model ngôn ngữ xử lý ngữ cảnh và lập luận ngôn ngữ tự nhiên:

```python
from haystack.components.generators.chat import OpenAIChatGenerator

generator = OpenAIChatGenerator(model="gpt-4o")
```

#### Tools
Giao diện để tương tác với hệ thống bên ngoài, APIs, pipelines hoặc components:

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
- **Ngắn hạn**: Trạng thái hội thoại trong một session
- **Dài hạn**: Lưu trữ tùy chọn cho các tương tác trong tương lai

### Các Loại Tool

#### Tool Class
Biểu diễn tool rõ ràng với tên, mô tả và hành vi:

```python
from haystack.tools import Tool

tool = Tool(
    name="calculator",
    description="Perform mathematical calculations",
    function=calculate
)
```

#### ComponentTool
Wrap các Haystack components thành callable tools:

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

#### Decorator @tool
Tạo tools từ Python functions:

```python
from haystack.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b
```

#### Toolset
Nhóm nhiều tools lại với nhau:

```python
from haystack.tools import Toolset

toolset = Toolset(tools=[tool1, tool2, tool3])
agent = Agent(chat_generator=generator, tools=toolset)
```

## Kiến Trúc Dựa Trên Pipeline

Điểm mạnh của Haystack nằm ở kiến trúc pipeline, nơi các components được kết nối để tạo thành workflows.

### Các Pipeline Components

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

### Xây Dựng Pipelines

```python
from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.routers import ConditionalRouter
from haystack.components.tools import ToolInvoker

pipeline = Pipeline()

# Thêm components
pipeline.add_component("generator", OpenAIChatGenerator())
pipeline.add_component("router", ConditionalRouter(routes=routes))
pipeline.add_component("tool_invoker", ToolInvoker(tools=tools))

# Kết nối components
pipeline.connect("generator", "router")
pipeline.connect("router.tool_calls", "tool_invoker")
pipeline.connect("tool_invoker", "generator")
```

### Luồng Agent Pipeline

1. **Prompt/system message khởi tạo** định nghĩa vai trò và mục tiêu
2. **Chat Generator** phân tích input người dùng
3. Trả về hoặc phản hồi assistant hoặc chỉ thị tool-call
4. **Router** (ConditionalRouter) điều hướng output:
   - Có tool calls: nhánh đến luồng gọi tool
   - Không có tool calls: trả về phản hồi cuối cùng
5. **ToolInvoker** thực thi các tools được chọn
6. **MessageCollector** lưu câu hỏi và phản hồi tool
7. Đưa trở lại generator cho phản hồi cuối cùng

## Ví Dụ Components

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

## Ví Dụ RAG Pipeline

```python
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers import InMemoryBM25Retriever

# Tạo pipeline
rag_pipeline = Pipeline()

# Thêm components
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", PromptBuilder(template=template))
rag_pipeline.add_component("llm", OpenAIGenerator())

# Kết nối
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

# Chạy
result = rag_pipeline.run({
    "retriever": {"query": "What is machine learning?"},
    "prompt_builder": {"query": "What is machine learning?"}
})
```

## Tính Năng Chính

| Tính năng | Mô tả |
|-----------|-------|
| **Không Phụ Thuộc Công Nghệ** | Dễ dàng chuyển đổi giữa vendors và models |
| **Rõ Ràng** | Kết nối component và luồng dữ liệu rõ ràng |
| **Linh Hoạt** | Truy cập database, chuyển đổi file, làm sạch, training, eval, inference |
| **Có Thể Mở Rộng** | Dễ tạo custom components |
| **Ưu Tiên Pipeline** | Workflows có thể kết hợp, dễ debug |
| **Sẵn Sàng Production** | Được xây dựng cho triển khai doanh nghiệp |

### Các Tích Hợp Được Hỗ Trợ

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
- Chuyển đổi Audio/Video

## Trường Hợp Sử Dụng

Các ví dụ Haystack từ production:
- **RAG**: Trả lời câu hỏi trên tài liệu
- **Tìm kiếm Semantic**: Tìm tài liệu theo ý nghĩa
- **Agents**: Hệ thống ra quyết định phức tạp
- **AI Hội Thoại**: Chatbots với sử dụng tool
- **Xử Lý Tài Liệu**: Pipelines nhập và indexing

### Các Công Ty Sử Dụng Haystack
- Apple, Meta, Netflix
- NVIDIA, Intel, Databricks
- Airbus, LEGO
- Bộ Liên Bang Đức
- Zeit Online, Rakuten

## Hayhooks (Triển Khai REST API)

Triển khai Haystack pipelines như REST APIs:

```python
# Wrap pipelines với logic tùy chỉnh
# Expose qua HTTP endpoints
# Endpoints chat completion tương thích OpenAI
# Tương thích với open-webui
```

## Trường Hợp Sử Dụng Tốt Nhất

- **Hệ thống RAG production**: Hỏi đáp tài liệu cấp doanh nghiệp
- **Pipelines phức tạp**: Workflows xử lý nhiều bước
- **Tìm kiếm lai**: Kết hợp tìm kiếm keyword và semantic
- **Ứng dụng doanh nghiệp**: Nơi độ tin cậy và observability quan trọng
- **Workflows tùy chỉnh**: Yêu cầu xử lý độc đáo
- **Kết hợp Agent + RAG**: Agents cần truy xuất

## Hạn Chế

- **Đường cong học tập**: Khái niệm pipeline cần thời gian để thành thạo
- **Dài dòng**: Nhiều code hơn cho use cases đơn giản
- **Tập trung Agent**: Mới hơn so với các framework agent khác
- **Tài liệu**: Có thể choáng ngợp với nhiều tùy chọn

## Cài Đặt

```bash
pip install haystack-ai

# Từ main branch (tính năng mới nhất)
pip install git+https://github.com/deepset-ai/haystack.git@main
```

## Ví Dụ Agent Hoàn Chỉnh

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.tools import Tool, tool, ComponentTool
from haystack.components.websearch import SerperDevWebSearch

# Định nghĩa tools
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

# Component như tool
web_search = SerperDevWebSearch()
search_tool = ComponentTool(
    component=web_search,
    name="web_search",
    description="Search the web for current information"
)

# Tạo agent
agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o"),
    tools=[calculate, search_tool],
    system_prompt="""You are a helpful assistant.
    Use tools to find information and perform calculations."""
)

# Chạy agent
result = agent.run(
    messages=[{"role": "user", "content": "What is 15% of the population of France?"}]
)
print(result["replies"][0].text)
```

## Tài Liệu Tham Khảo

- [Haystack GitHub](https://github.com/deepset-ai/haystack)
- [Tài liệu](https://docs.haystack.deepset.ai/)
- [Tutorials](https://haystack.deepset.ai/tutorials)
- [Cookbook](https://haystack.deepset.ai/cookbook)
- [Tích hợp](https://github.com/deepset-ai/haystack-integrations)
- [Hayhooks (REST API)](https://github.com/deepset-ai/hayhooks)
- [Cộng đồng Discord](https://discord.com/invite/xYvH6drSmA)
