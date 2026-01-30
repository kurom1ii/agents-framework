# Kiến Trúc LangChain

Tài liệu này cung cấp tổng quan toàn diện về kiến trúc cốt lõi của LangChain, bao gồm cấu trúc agent, các loại agent, hệ thống bộ nhớ và cơ chế gọi tool/function.

---

## Mục Lục

1. [Tổng Quan Kiến Trúc Cốt Lõi](#tong-quan-kien-truc-cot-loi)
2. [Cấu Trúc Agent](#cau-truc-agent)
3. [Các Loại Agent](#cac-loai-agent)
4. [Hệ Thống Bộ Nhớ](#he-thong-bo-nho)
5. [Cơ Chế Gọi Tool/Function](#co-che-goi-toolfunction)
6. [Tích Hợp LangGraph](#tich-hop-langgraph)

---

## Tổng Quan Kiến Trúc Cốt Lõi

Các agents của LangChain được xây dựng trên **LangGraph**, một framework điều phối cấp thấp cung cấp:

- **Thực thi Bền vững**: Lưu trữ trạng thái đáng tin cậy giữa các lần chạy agent
- **Streaming**: Streaming đầu ra thời gian thực trong quá trình thực thi
- **Human-in-the-Loop**: Khả năng ngắt và tiếp tục cho sự giám sát của con người
- **Persistence**: Checkpointing trạng thái để phục hồi và gỡ lỗi
- **Định tuyến Có điều kiện**: Quyết định luồng công việc động dựa trên trạng thái

### Kiến Trúc Cấp Cao

```
+------------------------------------------------------------------+
|                     Ứng dụng LangChain                           |
+------------------------------------------------------------------+
|                                                                  |
|  +----------------------+    +---------------------------+       |
|  |     Lớp Agent        |    |   Lớp Tích hợp            |       |
|  |  - create_agent()    |    |  - Nhà cung cấp Model     |       |
|  |  - Tools/Toolkits    |    |  - Vector Stores          |       |
|  |  - Memory/State      |    |  - Document Loaders       |       |
|  +----------------------+    +---------------------------+       |
|              |                           |                       |
|              v                           v                       |
|  +----------------------------------------------------------+   |
|  |                    LangGraph Runtime                      |   |
|  |  - StateGraph         - Checkpointers                     |   |
|  |  - Nodes/Edges        - Memory Stores                     |   |
|  |  - Conditional Logic  - Streaming                         |   |
|  +----------------------------------------------------------+   |
|              |                                                   |
|              v                                                   |
|  +----------------------------------------------------------+   |
|  |                    LangSmith (Tùy chọn)                   |   |
|  |  - Tracing           - Debugging                          |   |
|  |  - Evaluation        - Monitoring                         |   |
|  +----------------------------------------------------------+   |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Cấu Trúc Agent

### Tạo Agent Hiện Đại

API LangChain hiện đại sử dụng hàm `create_agent()` đơn giản hóa:

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72F"

agent = create_agent(
    model="claude-sonnet-4-5-20250929",  # Đặc tả mô hình
    tools=[search, get_weather],          # Danh sách tools
    system_prompt="You are a helpful assistant"  # Hướng dẫn hệ thống
)
```

### Các Thành Phần Agent

| Thành phần | Mô tả |
|-----------|-------------|
| **Model** | LLM cung cấp năng lực suy luận cho agent (OpenAI, Anthropic, Google, v.v.) |
| **Tools** | Các hàm agent có thể gọi để tương tác với hệ thống bên ngoài |
| **System Prompt** | Hướng dẫn định nghĩa hành vi và tính cách của agent |
| **State** | Dữ liệu bền vững bao gồm lịch sử tin nhắn và trạng thái tùy chỉnh |
| **Middleware** | Các hooks để tùy chỉnh hành vi agent (xử lý lỗi, logging, v.v.) |
| **Checkpointer** | Cơ chế lưu trữ trạng thái cho bộ nhớ và tiếp tục |

### Luồng Thực Thi Agent

```
User Input
    |
    v
+------------------+
|  System Prompt   |
|  + Message       |
|  History         |
+------------------+
    |
    v
+------------------+
|   LLM Call       |
| (Reasoning)      |
+------------------+
    |
    +-------> Tool Calls Có?
    |              |
    |         Có   |  Không
    |              |   |
    |              v   v
    |         +------------------+
    |         |  Thực thi Tools  |
    |         +------------------+
    |              |
    |              v
    |         +------------------+
    |         | Thêm Kết quả Tool|
    |         | vào Messages     |
    |         +------------------+
    |              |
    +<-------------+
    |
    v
+------------------+
|  Phản hồi Cuối   |
+------------------+
```

---

## Các Loại Agent

### 1. ReAct Agent (Reasoning + Acting)

Mẫu ReAct xen kẽ các bước suy luận với các lời gọi tool:

```python
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)
agent = create_agent(model, tools=[tool])

# Thực thi ReAct tạo ra: Thought -> Action -> Observation -> Thought -> ...
events = agent.stream(
    {"messages": [("user", "Search in google drive, who is 'Yann LeCun'?")]},
    stream_mode="values",
)
```

**Đặc điểm:**
- Dấu vết suy luận rõ ràng hiển thị trong đầu ra
- Tinh chỉnh lặp đi lặp lại dựa trên quan sát
- Tốt cho các nhiệm vụ phức tạp, nhiều bước
- Đầu ra chi tiết hỗ trợ gỡ lỗi

**Khi nào Sử dụng:**
- Các nhiệm vụ yêu cầu truy xuất và xác minh lặp đi lặp lại
- Các tình huống sử dụng tool nhiều bước
- Khi tính minh bạch của suy luận là quan trọng

### 2. OpenAI Functions Agent

Sử dụng khả năng gọi function native của OpenAI:

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_classic import hub

prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(temperature=0)

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

result = agent_executor.invoke({
    "input": "How is the tech sector being affected by fed policy?"
})
```

**Đặc điểm:**
- Sử dụng gọi function native của mô hình
- Gọi tool có cấu trúc hơn
- Ít chi tiết hơn ReAct
- Tốt hơn cho các mô hình có khả năng gọi function mạnh

### 3. Structured Chat Agent

Cho các mô hình cần đầu ra JSON có cấu trúc:

```python
from langchain.agents import StructuredChatAgent, AgentExecutor
from langchain_classic.chains import LLMChain

prompt = StructuredChatAgent.create_prompt(
    prefix=prefix,
    tools=tools,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)

llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = StructuredChatAgent(llm_chain=llm_chain, verbose=True, tools=tools)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, verbose=True, memory=memory, tools=tools
)
```

### 4. Zero-Shot ReAct Agent

Triển khai ReAct gốc không có ví dụ few-shot:

```python
from langchain.agents import AgentType, create_pandas_dataframe_agent

agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
    df,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
```

### So Sánh Các Loại Agent

| Loại Agent | Phù hợp nhất cho | Yêu cầu Mô hình | Kiểu Đầu ra |
|------------|----------|-------------------|--------------|
| ReAct | Suy luận nhiều bước | Bất kỳ LLM | Dấu vết chi tiết |
| OpenAI Functions | Gọi tool có cấu trúc | OpenAI/Tương thích | Ngắn gọn |
| Structured Chat | Tương tác dựa trên JSON | Bất kỳ LLM | Định dạng JSON |
| Zero-Shot ReAct | Nhiệm vụ đơn giản | Bất kỳ LLM | Định dạng ReAct |

---

## Hệ Thống Bộ Nhớ

LangChain cung cấp nhiều mẫu bộ nhớ để duy trì ngữ cảnh qua các cuộc hội thoại.

### Kiến Trúc Bộ Nhớ

```
+------------------------------------------------------------------+
|                      Lớp Bộ Nhớ                                  |
+------------------------------------------------------------------+
|                                                                  |
|  +-----------------+  +-----------------+  +-----------------+   |
|  | Ngắn hạn        |  | Dài hạn         |  | Ngữ nghĩa       |   |
|  | (Message Buffer)|  | (Checkpointer)  |  | (Vector Store)  |   |
|  +-----------------+  +-----------------+  +-----------------+   |
|          |                    |                    |             |
|          v                    v                    v             |
|  +---------------------------------------------------------+    |
|  |                   Quản lý Trạng thái                     |    |
|  |  - Message History    - Custom State                     |    |
|  |  - Thread IDs         - User Context                     |    |
|  +---------------------------------------------------------+    |
|                                                                  |
+------------------------------------------------------------------+
```

### 1. Conversation Buffer Memory

Lưu trữ toàn bộ lịch sử hội thoại:

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory
)
```

### 2. Message Trimming (Bộ nhớ Ngắn hạn)

Giữ cuộc hội thoại trong giới hạn ngữ cảnh:

```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents.middleware import before_model

@before_model
def trim_messages(state: AgentState, runtime: Runtime):
    """Chỉ giữ một vài tin nhắn gần nhất để phù hợp với cửa sổ ngữ cảnh."""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # Không cần thay đổi

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

agent = create_agent(
    model,
    tools=tools,
    middleware=[trim_messages],
    checkpointer=InMemorySaver(),
)
```

### 3. Summarization Memory

Tự động tóm tắt các cuộc hội thoại dài:

```python
from langchain import createAgent, summarizationMiddleware
from langgraph import MemorySaver

checkpointer = MemorySaver()

agent = createAgent({
    model: "gpt-4o",
    tools: [],
    middleware: [
        summarizationMiddleware({
            model: "gpt-4o-mini",
            trigger: { tokens: 4000 },  # Kích hoạt ở 4000 tokens
            keep: { messages: 20 },      # Giữ 20 tin nhắn cuối
        }),
    ],
    checkpointer,
})
```

### 4. Long-Term Memory (Vector Store)

Tìm kiếm ngữ nghĩa qua các tương tác trước đó:

```python
from langgraph.store.memory import InMemoryStore

def embed(texts: list[str]) -> list[list[float]]:
    # Thay thế bằng hàm embedding thực tế
    return [[1.0, 2.0] * len(texts)]

store = InMemoryStore(index={"embed": embed, "dims": 2})
user_id = "my-user"
namespace = (user_id, "memories")

# Lưu trữ memories
store.put(
    namespace,
    "a-memory",
    {
        "rules": [
            "User likes short, direct language",
            "User only speaks English & python",
        ],
    },
)

# Tìm kiếm memories
items = store.search(
    namespace,
    query="language preferences"
)
```

### 5. Checkpointer-Based Persistence

Cho trạng thái bền vững qua các phiên:

```python
from langgraph.checkpoint.memory import MemorySaver
# Hoặc cho production:
# from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = MemorySaver()
agent = create_agent(
    model,
    tools=tools,
    checkpointer=checkpointer
)

# Mỗi thread duy trì trạng thái hội thoại riêng biệt
config = {"configurable": {"thread_id": "user-123"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Hello!"}]},
    config
)
```

### So Sánh Loại Bộ Nhớ

| Loại Bộ nhớ | Trường hợp Sử dụng | Persistence | Khả năng Mở rộng |
|-------------|----------|-------------|-------------|
| Buffer | Cuộc hội thoại đơn giản | Trong bộ nhớ | Hạn chế |
| Trimming | Cuộc hội thoại dài | Trong bộ nhớ | Tốt |
| Summarization | Phiên mở rộng | Trong bộ nhớ + LLM | Tốt |
| Vector Store | Nhớ lại ngữ nghĩa | Database | Xuất sắc |
| Checkpointer | Ứng dụng production | Có thể cấu hình | Xuất sắc |

---

## Cơ Chế Gọi Tool/Function

### Định Nghĩa Tool

Tools được định nghĩa sử dụng decorator `@tool`:

```python
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather for a location.

    Args:
        location: The city and state, e.g. San Francisco, CA
    """
    return f"Weather in {location}: Sunny, 72F"
```

### Định Nghĩa Tool Nâng Cao với Pydantic

```python
from pydantic import BaseModel, Field
from langchain.tools import tool

class WeatherInput(BaseModel):
    """Input for weather tool."""
    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")
    unit: str = Field(default="fahrenheit", description="Temperature unit")

@tool(args_schema=WeatherInput)
def get_weather(location: str, unit: str = "fahrenheit") -> str:
    """Get the current weather in a given location."""
    return f"Weather in {location}: Sunny, 72{unit[0].upper()}"
```

### Binding Tools với Models

```python
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather for a location."""
    return f"Weather in {location}: Sunny, 72F"

# Bind tools với model
model_with_tools = model.bind_tools([get_weather])
response = model_with_tools.invoke("What's the weather in San Francisco?")
print(response.tool_calls)
# [{'name': 'get_weather', 'args': {'location': 'San Francisco'}, 'id': '...'}]
```

### Luồng Thực Thi Tool

```
Model Response với tool_calls
           |
           v
+------------------------+
| Trích xuất Tool Call   |
| - Name                 |
| - Arguments            |
| - Call ID              |
+------------------------+
           |
           v
+------------------------+
| Tra cứu Tool Function  |
+------------------------+
           |
           v
+------------------------+
| Xác thực Argument      |
| (Pydantic Schema)      |
+------------------------+
           |
           v
+------------------------+
| Thực thi Tool          |
+------------------------+
           |
           v
+------------------------+
| Tạo ToolMessage        |
| - Content (kết quả)    |
| - Tool Call ID         |
+------------------------+
           |
           v
Thêm vào Message History
```

### Xử Lý Lỗi cho Tools

```python
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Xử lý lỗi thực thi tool với thông báo tùy chỉnh."""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model="gpt-4o",
    tools=[search, get_weather],
    middleware=[handle_tool_errors]
)
```

---

## Tích Hợp LangGraph

LangChain agents được xây dựng trên LangGraph cho điều phối nâng cao.

### Cơ Bản StateGraph

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    input: str
    output: str

def process_node(state: State):
    return {"output": f"Processed: {state['input']}"}

# Xây dựng graph
builder = StateGraph(State)
builder.add_node("process", process_node)
builder.add_edge(START, "process")
builder.add_edge("process", END)

graph = builder.compile()
result = graph.invoke({"input": "Hello"})
```

### Định Tuyến Có Điều Kiện

```python
from langgraph.graph import StateGraph, START, END

def route_decision(state: State):
    if state["decision"] == "story":
        return "story_node"
    elif state["decision"] == "joke":
        return "joke_node"
    return "poem_node"

builder = StateGraph(State)
builder.add_node("router", router_node)
builder.add_node("story_node", story_node)
builder.add_node("joke_node", joke_node)
builder.add_node("poem_node", poem_node)

builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    route_decision,
    {
        "story_node": "story_node",
        "joke_node": "joke_node",
        "poem_node": "poem_node",
    }
)
builder.add_edge("story_node", END)
builder.add_edge("joke_node", END)
builder.add_edge("poem_node", END)

graph = builder.compile()
```

### Human-in-the-Loop với Interrupts

```python
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

def approval_node(state):
    decision = interrupt({
        "question": "Approve this action?",
        "details": state["action_details"],
    })
    return Command(goto="proceed" if decision else "cancel")

builder = StateGraph(State)
builder.add_node("approval", approval_node)
builder.add_node("proceed", proceed_node)
builder.add_node("cancel", cancel_node)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Lần gọi đầu tiên tạm dừng tại interrupt
config = {"configurable": {"thread_id": "approval-123"}}
initial = graph.invoke({"action_details": "Transfer $500"}, config=config)

# Tiếp tục với quyết định
resumed = graph.invoke(Command(resume=True), config=config)
```

---

## Bước Tiếp Theo

- [Tham khảo Thành phần](./components.md) - Tài liệu thành phần chi tiết
- [Mẫu Agent](./patterns.md) - Các mẫu và luồng công việc phổ biến
- [Ví dụ Code](./examples.md) - Các ví dụ triển khai thực tế
