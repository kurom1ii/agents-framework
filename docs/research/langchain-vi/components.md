# Tham Khảo Thành Phần LangChain

Tài liệu này cung cấp tài liệu chi tiết về các thành phần chính của LangChain bao gồm AgentExecutor, các lớp Agent, Tools, Callbacks và Output Parsers.

---

## Mục Lục

1. [AgentExecutor](#agentexecutor)
2. [Các Lớp Agent và Tạo Agent](#cac-lop-agent-va-tao-agent)
3. [Tools và Toolkits](#tools-va-toolkits)
4. [Hệ Thống Middleware](#he-thong-middleware)
5. [Callbacks và Handlers](#callbacks-va-handlers)
6. [Output Parsers và Structured Output](#output-parsers-va-structured-output)

---

## AgentExecutor

### Tổng Quan

`AgentExecutor` là runtime legacy quản lý các vòng lặp thực thi agent. Trong LangChain hiện đại, các agents được tạo với `create_agent()` được xây dựng trên LangGraph và xử lý thực thi nội bộ.

### Sử Dụng AgentExecutor Legacy

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_classic import hub

# Tạo agent
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)

# Bọc trong AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,           # In dấu vết thực thi
    max_iterations=15,      # Số vòng lặp suy luận tối đa
    max_execution_time=60,  # Timeout tính bằng giây
    handle_parsing_errors=True,  # Xử lý lỗi một cách duyên dáng
)

# Thực thi
result = agent_executor.invoke({"input": "What is the weather in Tokyo?"})
```

### Các Tham Số AgentExecutor

| Tham số | Kiểu | Mặc định | Mô tả |
|-----------|------|---------|-------------|
| `agent` | Agent | Bắt buộc | Agent để thực thi |
| `tools` | List[Tool] | Bắt buộc | Tools có sẵn cho agent |
| `verbose` | bool | False | In dấu vết thực thi |
| `max_iterations` | int | 15 | Số vòng lặp suy luận tối đa |
| `max_execution_time` | float | None | Timeout tính bằng giây |
| `handle_parsing_errors` | bool | False | Xử lý lỗi phân tích đầu ra LLM |
| `early_stopping_method` | str | "force" | Cách dừng nếu đạt max iterations |
| `memory` | Memory | None | Thành phần bộ nhớ cho hội thoại |
| `return_intermediate_steps` | bool | False | Trả về tất cả các bước trung gian |

### Thực Thi Agent Hiện Đại (Dựa trên LangGraph)

Cách tiếp cận hiện đại sử dụng `create_agent()` trả về một LangGraph đã biên dịch:

```python
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o",
    tools=[search, get_weather],
    system_prompt="You are a helpful assistant"
)

# Gọi trực tiếp
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather?"}]
})

# Streaming
for event in agent.stream(
    {"messages": [{"role": "user", "content": "What's the weather?"}]},
    stream_mode="values"
):
    event["messages"][-1].pretty_print()
```

### So Sánh Vòng Lặp Thực Thi

**Vòng lặp AgentExecutor Legacy:**
```
Input -> Agent.plan() -> Action -> Tool.run() -> Observation -> Agent.plan() -> ... -> Finish
```

**Vòng lặp LangGraph Hiện đại:**
```
Input -> LLM Call -> Tool Calls? -> Execute Tools -> Add Results -> LLM Call -> ... -> Final Response
```

---

## Các Lớp Agent và Tạo Agent

### Tạo Agent Hiện Đại

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

# Khởi tạo model
model = init_chat_model("claude-sonnet-4-5-20250929", model_provider="anthropic")

# Tạo agent
agent = create_agent(
    model=model,
    tools=[tool1, tool2],
    system_prompt="You are a helpful assistant",
    checkpointer=checkpointer,  # Tùy chọn: cho lưu trữ trạng thái
    middleware=[middleware1],    # Tùy chọn: cho hành vi tùy chỉnh
)
```

### Các Tham Số Tạo Agent

| Tham số | Kiểu | Mô tả |
|-----------|------|-------------|
| `model` | str hoặc ChatModel | Định danh mô hình hoặc mô hình đã khởi tạo |
| `tools` | List[Tool] | Tools agent có thể sử dụng |
| `system_prompt` | str | Hướng dẫn hệ thống cho agent |
| `checkpointer` | Checkpointer | Cơ chế lưu trữ trạng thái |
| `middleware` | List[Middleware] | Các hooks hành vi tùy chỉnh |
| `state_schema` | TypedDict | Schema trạng thái tùy chỉnh |
| `response_format` | Schema | Định dạng đầu ra có cấu trúc |

### Các Loại Agent Legacy

Để tương thích ngược, các loại agent legacy vẫn có sẵn:

```python
from langchain.agents import AgentType

# Các loại có sẵn
AgentType.ZERO_SHOT_REACT_DESCRIPTION  # ReAct không có ví dụ
AgentType.REACT_DOCSTORE              # ReAct cho document stores
AgentType.SELF_ASK_WITH_SEARCH        # Mẫu self-ask
AgentType.CONVERSATIONAL_REACT_DESCRIPTION  # ReAct hội thoại
AgentType.OPENAI_FUNCTIONS            # OpenAI function calling
AgentType.OPENAI_MULTI_FUNCTIONS      # Nhiều lần gọi function
AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT  # Structured chat
```

### Các Hàm Factory Agent

```python
# OpenAI Functions Agent
from langchain.agents import create_openai_functions_agent
agent = create_openai_functions_agent(llm, tools, prompt)

# ReAct Agent
from langchain.agents import create_react_agent
agent = create_react_agent(llm, tools, prompt)

# Structured Chat Agent
from langchain.agents import create_structured_chat_agent
agent = create_structured_chat_agent(llm, tools, prompt)

# Tool Calling Agent (generic)
from langchain.agents import create_tool_calling_agent
agent = create_tool_calling_agent(llm, tools, prompt)
```

---

## Tools và Toolkits

### Định Nghĩa Tool

Tools là cách chính để agents tương tác với các hệ thống bên ngoài.

#### Tool Cơ bản với Decorator

```python
from langchain.tools import tool

@tool
def search(query: str) -> str:
    """Search for information on the web.

    Args:
        query: The search query string
    """
    # Triển khai
    return f"Search results for: {query}"
```

#### Tool với Pydantic Schema

```python
from langchain.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    """Input schema for search tool."""
    query: str = Field(..., description="The search query")
    max_results: int = Field(default=10, description="Maximum results to return")

@tool(args_schema=SearchInput)
def search(query: str, max_results: int = 10) -> str:
    """Search for information with configurable result count."""
    return f"Top {max_results} results for: {query}"
```

#### Định Nghĩa Tool Class

```python
from langchain.tools import Tool

def search_func(query: str) -> str:
    return f"Results for: {query}"

search_tool = Tool(
    name="search",
    description="Search for information on the web",
    func=search_func,
)
```

#### Structured Tool với Schema Phức Tạp

```python
from langchain.tools import StructuredTool
from pydantic import BaseModel

class EmailInput(BaseModel):
    to: list[str]
    subject: str
    body: str
    cc: list[str] = []

def send_email(to: list[str], subject: str, body: str, cc: list[str] = []) -> str:
    return f"Email sent to {', '.join(to)}"

email_tool = StructuredTool.from_function(
    func=send_email,
    name="send_email",
    description="Send an email",
    args_schema=EmailInput,
)
```

### Các Thuộc Tính Tool

| Thuộc tính | Mô tả |
|----------|-------------|
| `name` | Định danh duy nhất cho tool |
| `description` | Mô tả tool làm gì (được LLM sử dụng) |
| `args_schema` | Mô hình Pydantic định nghĩa input schema |
| `return_direct` | Nếu True, trả về kết quả tool trực tiếp mà không qua xử lý LLM |
| `func` | Hàm để thực thi |
| `coroutine` | Hàm async cho thực thi async |

### Toolkits

Toolkits là các bộ sưu tập tools liên quan:

```python
from langchain_community.agent_toolkits import FileManagementToolkit

# Lấy tất cả tools từ toolkit
toolkit = FileManagementToolkit()
tools = toolkit.get_tools()

# Hoặc chọn tools cụ thể
tools = toolkit.get_tools(
    selected_tools=["read_file", "write_file"]
)
```

### Các Danh Mục Tool Tích Hợp

| Danh mục | Ví dụ |
|----------|----------|
| **Tìm kiếm** | DuckDuckGo, Google, Bing, Tavily, You.com |
| **Web** | Requests, Wikipedia, ArXiv |
| **Code** | Python REPL, Shell, SQL |
| **File** | Đọc/ghi file, Thao tác thư mục |
| **Toán học** | Calculator, Wolfram Alpha |
| **APIs** | REST API, GraphQL |
| **Databases** | SQL, Vector stores |

### Dynamic Tools

Tools có thể được thêm hoặc lọc động:

```python
from langchain.agents.middleware import before_model

@before_model
def filter_tools_by_permission(state, runtime):
    """Lọc tools dựa trên quyền của người dùng."""
    user_role = state.get("user_role", "basic")

    if user_role == "admin":
        return None  # Tất cả tools có sẵn

    # Lọc chỉ còn basic tools
    basic_tools = [t for t in runtime.tools if t.name in ["search", "weather"]]
    return {"tools": basic_tools}
```

---

## Hệ Thống Middleware

Middleware cung cấp các hooks để tùy chỉnh hành vi agent ở các giai đoạn khác nhau.

### Các Loại Middleware

| Loại | Decorator | Mục đích |
|------|-----------|---------|
| `before_model` | `@before_model` | Sửa đổi trạng thái trước lời gọi LLM |
| `after_model` | `@after_model` | Xử lý phản hồi LLM |
| `wrap_tool_call` | `@wrap_tool_call` | Bọc các thực thi tool riêng lẻ |

### Before Model Middleware

```python
from langchain.agents.middleware import before_model

@before_model
def add_context(state, runtime):
    """Thêm ngữ cảnh vào messages trước lời gọi LLM."""
    current_time = datetime.now().isoformat()
    return {
        "messages": [
            {"role": "system", "content": f"Current time: {current_time}"},
            *state["messages"]
        ]
    }
```

### After Model Middleware

```python
from langchain.agents.middleware import after_model

@after_model
def log_response(state, response, runtime):
    """Ghi log các phản hồi mô hình."""
    print(f"Model response: {response.content}")
    return None  # Không sửa đổi trạng thái
```

### Wrap Tool Call Middleware

```python
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Xử lý lỗi thực thi tool."""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Error: {str(e)}. Please try again.",
            tool_call_id=request.tool_call["id"]
        )

@wrap_tool_call
def log_tool_calls(request, handler):
    """Ghi log tất cả các lời gọi tool."""
    print(f"Calling tool: {request.tool_call['name']}")
    result = handler(request)
    print(f"Tool result: {result.content}")
    return result
```

### Kết Hợp Middleware

```python
agent = create_agent(
    model="gpt-4o",
    tools=[search, weather],
    middleware=[
        add_context,
        handle_tool_errors,
        log_tool_calls,
    ]
)
```

---

## Callbacks và Handlers

Callbacks cung cấp các hooks để giám sát và gỡ lỗi thực thi agent.

### Giao Diện Callback Handler

```python
from langchain.callbacks.base import BaseCallbackHandler

class CustomHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Được gọi khi LLM bắt đầu."""
        print(f"LLM starting with {len(prompts)} prompts")

    def on_llm_end(self, response, **kwargs):
        """Được gọi khi LLM kết thúc."""
        print(f"LLM finished: {response}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        """Được gọi khi tool bắt đầu."""
        print(f"Tool starting: {serialized['name']}")

    def on_tool_end(self, output, **kwargs):
        """Được gọi khi tool kết thúc."""
        print(f"Tool output: {output}")

    def on_agent_action(self, action, **kwargs):
        """Được gọi khi agent thực hiện hành động."""
        print(f"Agent action: {action}")

    def on_agent_finish(self, finish, **kwargs):
        """Được gọi khi agent hoàn thành."""
        print(f"Agent finished: {finish}")
```

### Sử Dụng Callbacks

```python
# Với lời gọi agent
result = agent.invoke(
    {"messages": [...]},
    config={"callbacks": [CustomHandler()]}
)

# Callbacks toàn cục
from langchain.globals import set_verbose, set_debug

set_verbose(True)  # In thông tin cơ bản
set_debug(True)    # In thông tin debug chi tiết
```

### Các Handlers Tích Hợp

| Handler | Mô tả |
|---------|-------------|
| `StdOutCallbackHandler` | In ra stdout |
| `FileCallbackHandler` | Ghi vào file |
| `StreamingStdOutCallbackHandler` | Stream tokens ra stdout |
| `LangChainTracer` | Gửi traces đến LangSmith |

### Tích Hợp LangSmith

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"

# Traces được tự động gửi đến LangSmith
agent = create_agent(model="gpt-4o", tools=tools)
result = agent.invoke({"messages": [...]})
```

---

## Output Parsers và Structured Output

### Chiến Lược Structured Output

LangChain cung cấp hai chiến lược chính cho structured output:

#### 1. ToolStrategy

Sử dụng các lời gọi tool tổng hợp để ép phản hồi có cấu trúc:

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field
from typing import Literal

class ProductReview(BaseModel):
    """Analysis of a product review."""
    rating: int | None = Field(description="Rating 1-5", ge=1, le=5)
    sentiment: Literal["positive", "negative"]
    key_points: list[str] = Field(description="Key points, 1-3 words each")

agent = create_agent(
    model="gpt-4o",
    tools=[],
    response_format=ToolStrategy(ProductReview)
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Analyze: 'Great product, 5 stars. Fast shipping, expensive'"
    }]
})

print(result["structured_response"])
# ProductReview(rating=5, sentiment='positive', key_points=['fast shipping', 'expensive'])
```

#### 2. ProviderStrategy

Sử dụng structured output native của nhà cung cấp mô hình:

```python
from langchain.agents.structured_output import ProviderStrategy

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model="gpt-4o",
    tools=[],
    response_format=ProviderStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Extract: John Doe, john@example.com, (555) 123-4567"
    }]
})

print(result["structured_response"])
# {'name': 'John Doe', 'email': 'john@example.com', 'phone': '(555) 123-4567'}
```

### Các Tùy Chọn Định Nghĩa Schema

```python
# 1. Pydantic BaseModel
from pydantic import BaseModel

class Response(BaseModel):
    answer: str
    confidence: float

# 2. TypedDict
from typing_extensions import TypedDict

class Response(TypedDict):
    answer: str
    confidence: float

# 3. Dataclass
from dataclasses import dataclass

@dataclass
class Response:
    answer: str
    confidence: float

# 4. JSON Schema (dict)
response_schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"}
    },
    "required": ["answer", "confidence"]
}
```

### Union Types cho Nhiều Schemas

```python
from pydantic import BaseModel
from typing import Literal

class ProductReview(BaseModel):
    rating: int
    sentiment: Literal["positive", "negative"]

class CustomerComplaint(BaseModel):
    issue_type: Literal["product", "service", "shipping"]
    severity: Literal["low", "medium", "high"]

# Agent có thể trả về một trong hai loại
agent = create_agent(
    model="gpt-4o",
    tools=[],
    response_format=ToolStrategy([ProductReview, CustomerComplaint])
)
```

### Output Parsers Legacy

Cho các chains legacy, output parsers vẫn có sẵn:

```python
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

parser = PydanticOutputParser(pydantic_object=ProductReview)

prompt = PromptTemplate(
    template="Analyze this review:\n{review}\n\n{format_instructions}",
    input_variables=["review"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
```

---

## Ví Dụ Tích Hợp Thành Phần

Đây là cách tất cả các thành phần hoạt động cùng nhau:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, wrap_tool_call
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

# Định nghĩa tools
@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

# Định nghĩa middleware
@wrap_tool_call
def log_tools(request, handler):
    print(f"Tool: {request.tool_call['name']}")
    return handler(request)

# Định nghĩa output schema
class Response(BaseModel):
    answer: str
    sources: list[str]

# Tạo agent với tất cả các thành phần
agent = create_agent(
    model="gpt-4o",
    tools=[search],
    system_prompt="You are a research assistant.",
    middleware=[log_tools],
    checkpointer=MemorySaver(),
    response_format=ToolStrategy(Response),
)

# Thực thi với callbacks
from langchain.callbacks import StdOutCallbackHandler

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is LangChain?"}]},
    config={
        "configurable": {"thread_id": "research-1"},
        "callbacks": [StdOutCallbackHandler()]
    }
)

print(result["structured_response"])
```

---

## Bước Tiếp Theo

- [Mẫu Agent](./patterns.md) - Các mẫu và luồng công việc phổ biến
- [Ví dụ Code](./examples.md) - Các ví dụ triển khai thực tế
