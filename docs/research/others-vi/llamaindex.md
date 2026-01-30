# LlamaIndex Agents

## Tổng Quan

**LlamaIndex** (trước đây là GPT Index) là framework hàng đầu để xây dựng các agent được hỗ trợ bởi LLM trên dữ liệu của bạn. Ban đầu tập trung vào RAG (Retrieval-Augmented Generation), LlamaIndex đã phát triển thành một data framework toàn diện với khả năng agent mạnh mẽ.

- **Repository**: https://github.com/run-llama/llama_index
- **Stars**: ~47,000
- **Ngôn ngữ**: Python (cũng có TypeScript qua LlamaIndexTS)
- **License**: MIT
- **Tài liệu**: https://docs.llamaindex.ai/

## Triết Lý Cốt Lõi

LlamaIndex định nghĩa "agent" là một hệ thống cụ thể sử dụng LLM, memory và tools để xử lý input từ người dùng bên ngoài. Framework này xuất sắc trong việc biến dữ liệu thành dạng mà LLM có thể sử dụng hiệu quả, khiến nó trở nên lý tưởng cho các ứng dụng agent xử lý dữ liệu nặng.

## Các Khái Niệm Chính

### Định Nghĩa Agent

Một agent kết hợp:
- **LLM**: "Bộ não" model ngôn ngữ
- **Memory**: Lịch sử hội thoại và ngữ cảnh
- **Tools**: Các function mà agent có thể gọi để tương tác với dữ liệu và hệ thống

### Vòng Lặp Agent

1. Nhận tin nhắn mới nhất + lịch sử chat
2. Gửi tool schemas + lịch sử đến LLM
3. LLM trả về phản hồi trực tiếp hoặc tool-call(s)
4. Thực thi tool(s) và thêm kết quả vào lịch sử
5. Gọi lại LLM cho đến khi có phản hồi cuối cùng

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

## Các Loại Agent

### FunctionAgent
Sử dụng function/tool-calling của LLM để gọi Python tools. Loại agent phổ biến nhất cho sử dụng chung.

```python
from llama_index.core.agent import FunctionAgent

agent = FunctionAgent(
    tools=[tool1, tool2],
    llm=llm,
    system_prompt="You are a helpful assistant."
)
```

### ReActAgent
Sử dụng chiến lược prompting (không phải function-calling) để điều khiển việc sử dụng tool. Dựa trên paradigm ReAct (Reasoning + Acting).

```python
from llama_index.core.agent import ReActAgent

agent = ReActAgent.from_tools(
    tools=[tool1, tool2],
    llm=llm,
    verbose=True
)
```

### CodeActAgent
Sử dụng sinh code và thực thi cho lập luận và hành động phức tạp.

### AgentWorkflow / Multi-Agent
Phối hợp nhiều agent với khả năng handoff.

```python
from llama_index.core.agent import AgentWorkflow

workflow = AgentWorkflow(
    agents=[agent1, agent2, agent3],
    root_agent=triage_agent
)
```

## Tools

### FunctionTool
Wrap các hàm Python thành tools:

```python
from llama_index.core.tools import FunctionTool

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

tool = FunctionTool.from_defaults(fn=multiply)
```

### QueryEngineTool
Sử dụng query engines (RAG pipelines) như agent tools:

```python
from llama_index.core.tools import QueryEngineTool

# Tạo index từ documents
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Wrap thành tool
query_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="document_search",
    description="Search through company documents"
)

agent = FunctionAgent(tools=[query_tool], llm=llm)
```

### Tool Specs
Bộ sưu tập tool được xây dựng sẵn cho các API phổ biến:

```python
from llama_index.tools.google import GmailToolSpec

gmail_tools = GmailToolSpec().to_tool_list()
```

Các tool specs có sẵn bao gồm:
- Google Suite (Gmail, Calendar, Drive)
- Slack
- Notion
- GitHub
- SQL databases
- Và nhiều hơn nữa qua LlamaHub

## Data Agents

Data agents là điểm khác biệt chính của LlamaIndex. Chúng kết hợp:
- **Nhập dữ liệu**: Tải documents, APIs, databases
- **Indexing**: Vector stores, knowledge graphs, keyword indices
- **Truy xuất**: Tìm kiếm semantic, tìm kiếm lai
- **Khả năng agent**: Tools, memory, lập luận

### Query Engines như Agents

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Tải và index documents
documents = SimpleDirectoryReader("data/").load_data()
index = VectorStoreIndex.from_documents(documents)

# Tạo query engine
query_engine = index.as_query_engine()

# Sử dụng trực tiếp hoặc wrap thành tool cho agent
response = query_engine.query("What are the key findings?")
```

### Agents Đa Index

Agents có thể điều phối qua nhiều nguồn dữ liệu:

```python
from llama_index.core.tools import QueryEngineTool

# Nhiều query engines cho các nguồn dữ liệu khác nhau
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

Agents mặc định sử dụng ChatMemoryBuffer cho lịch sử hội thoại:

```python
from llama_index.core.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

agent = FunctionAgent(
    tools=tools,
    llm=llm,
    memory=memory
)
```

## Agents Đa Phương Thức

LlamaIndex agents hỗ trợ input đa phương thức (hình ảnh + text):

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

Streaming được bật mặc định:

```python
agent = FunctionAgent(tools=tools, llm=llm, streaming=True)

async for chunk in agent.stream_chat("Tell me a story"):
    print(chunk.delta, end="")
```

## Kiến Trúc

```
llama_index/
├── core/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── react/           # Triển khai ReAct agent
│   │   │   ├── formatter.py # ReActChatFormatter
│   │   │   └── output_parser.py
│   │   └── workflow/        # Agents dựa trên workflow
│   │       ├── function_agent.py
│   │       ├── react_agent.py
│   │       ├── codeact_agent.py
│   │       └── multi_agent_workflow.py
│   ├── tools/               # Định nghĩa Tool
│   ├── memory/              # Triển khai Memory
│   ├── indices/             # Các loại Index (vector, keyword, v.v.)
│   ├── query_engine/        # Triển khai Query engine
│   └── storage/             # Storage backends
├── llms/                    # Tích hợp LLM
├── embeddings/              # Tích hợp Embedding model
└── readers/                 # Data loaders
```

## Hệ Sinh Thái Tích Hợp

LlamaIndex có 300+ tích hợp trên LlamaHub:

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

## Điểm Khác Biệt Chính

| Tính năng | Mô tả |
|-----------|-------|
| **Ưu tiên Dữ liệu** | Được xây dựng cho RAG và ứng dụng xử lý dữ liệu nặng |
| **Query Engines như Tools** | Khả năng độc đáo sử dụng RAG pipelines làm agent tools |
| **Tích hợp Mở rộng** | 300+ tích hợp trên LlamaHub |
| **Indexing Linh hoạt** | Vector, keyword, knowledge graph, tree indices |
| **Có thể Kết hợp** | Xây dựng pipelines phức tạp từ các thành phần đơn giản |
| **Sẵn sàng Production** | Công cụ observability, evaluation và deployment |

## Trường Hợp Sử Dụng Tốt Nhất

- **Ứng dụng RAG**: Hỏi đáp tài liệu, knowledge bases
- **Data Agents**: Agents cần truy vấn dữ liệu có cấu trúc/không cấu trúc
- **Truy xuất Đa Nguồn**: Agents trải rộng nhiều nguồn dữ liệu
- **Tìm kiếm Doanh nghiệp**: Tìm kiếm semantic trên tài liệu công ty
- **Trợ lý Nghiên cứu**: Agents tổng hợp thông tin từ nhiều nguồn
- **SQL Agents**: Truy vấn SQL ngôn ngữ tự nhiên

## Hạn Chế

- **Tập trung dữ liệu**: Ít phù hợp cho agents hội thoại thuần túy hoặc hướng hành động
- **Phức tạp**: Nhiều thành phần và tùy chọn có thể choáng ngợp
- **Chi phí memory**: Các index lớn có thể tốn tài nguyên
- **Phụ thuộc LLM**: Phụ thuộc nhiều vào khả năng LLM cho lập luận phức tạp

## Cài Đặt

```bash
# Package starter với các tích hợp phổ biến
pip install llama-index

# Hoặc chỉ core với các tích hợp cụ thể
pip install llama-index-core
pip install llama-index-llms-openai
pip install llama-index-embeddings-huggingface
```

## Ví Dụ: Data Agent Hoàn Chỉnh

```python
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent import FunctionAgent
from llama_index.core.tools import QueryEngineTool, FunctionTool

# Tải documents
documents = SimpleDirectoryReader("data/").load_data()

# Tạo vector index
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Tạo query tool
doc_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="document_search",
    description="Search through documents to find relevant information"
)

# Tạo tools bổ sung
def get_current_date() -> str:
    """Get the current date."""
    from datetime import date
    return str(date.today())

date_tool = FunctionTool.from_defaults(fn=get_current_date)

# Tạo agent
agent = FunctionAgent(
    tools=[doc_tool, date_tool],
    system_prompt="You are a helpful research assistant."
)

# Truy vấn agent
response = await agent.chat("What are the main findings in the documents?")
print(response)
```

## Tài Liệu Tham Khảo

- [LlamaIndex GitHub](https://github.com/run-llama/llama_index)
- [Tài liệu](https://docs.llamaindex.ai/)
- [LlamaHub (Tích hợp)](https://llamahub.ai/)
- [LlamaIndex TypeScript](https://github.com/run-llama/LlamaIndexTS)
- [Cộng đồng Discord](https://discord.gg/dGcwcsnxhU)
- [X (Twitter)](https://x.com/llama_index)
