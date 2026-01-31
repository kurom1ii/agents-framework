# Ví Dụ Code LangChain

Tài liệu này cung cấp các ví dụ code thực tế, có thể chạy được cho các kịch bản agent LangChain phổ biến.

---

## Mục Lục

1. [Thiết Lập Agent Cơ Bản](#thiet-lap-agent-co-ban)
2. [Tạo Tool Tùy Chỉnh](#tao-tool-tuy-chinh)
3. [Tích Hợp Bộ Nhớ](#tich-hop-bo-nho)
4. [Hệ Thống Multi-Agent](#he-thong-multi-agent)
5. [Structured Output](#structured-output)
6. [Xử Lý Lỗi](#xu-ly-loi)
7. [Streaming](#streaming)
8. [Mẫu Nâng Cao](#mau-nang-cao)

---

## Thiết Lập Agent Cơ Bản

### Agent Tối Thiểu (5 Dòng)

```python
from langchain.agents import create_agent

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[],
    system_prompt="You are a helpful assistant"
)

result = agent.invoke({"messages": [{"role": "user", "content": "Hello!"}]})
print(result["messages"][-1].content)
```

### Agent với Tools

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Triển khai mock
    weather_data = {
        "tokyo": "Sunny, 22C",
        "london": "Cloudy, 15C",
        "new york": "Rainy, 18C"
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

agent = create_agent(
    model="gpt-4o",
    tools=[get_weather, calculate],
    system_prompt="""You are a helpful assistant that can check weather
    and perform calculations. Use the appropriate tool when needed."""
)

# Kiểm tra agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]
})
print(result["messages"][-1].content)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is 15 * 23 + 42?"}]
})
print(result["messages"][-1].content)
```

### Agent với Các Nhà Cung Cấp Mô Hình Khác Nhau

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

# OpenAI
openai_model = init_chat_model("gpt-4o", model_provider="openai", temperature=0)

# Anthropic
anthropic_model = init_chat_model(
    "claude-sonnet-4-5-20250929",
    model_provider="anthropic",
    temperature=0
)

# Google
google_model = init_chat_model(
    "gemini-1.5-pro",
    model_provider="google_genai",
    temperature=0
)

# Tạo agents với các mô hình khác nhau
openai_agent = create_agent(model=openai_model, tools=tools)
anthropic_agent = create_agent(model=anthropic_model, tools=tools)
google_agent = create_agent(model=google_model, tools=tools)
```

---

## Tạo Tool Tùy Chỉnh

### Tool Cơ Bản với Decorator

```python
from langchain.tools import tool

@tool
def search_database(query: str) -> str:
    """Search the company database for information.

    Args:
        query: The search query to find relevant information
    """
    # Triển khai
    return f"Database results for: {query}"
```

### Tool với Pydantic Schema

```python
from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Optional

class EmailInput(BaseModel):
    """Input schema for sending emails."""
    to: str = Field(..., description="Recipient email address")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body content")
    cc: Optional[str] = Field(None, description="CC email address")
    priority: str = Field(default="normal", description="Priority: low, normal, high")

@tool(args_schema=EmailInput)
def send_email(to: str, subject: str, body: str, cc: Optional[str] = None, priority: str = "normal") -> str:
    """Send an email to the specified recipient."""
    cc_info = f" (CC: {cc})" if cc else ""
    return f"Email sent to {to}{cc_info}\nSubject: {subject}\nPriority: {priority}"

# Sử dụng
agent = create_agent(
    model="gpt-4o",
    tools=[send_email],
    system_prompt="You are an email assistant. Help users compose and send emails."
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Send an email to john@example.com about the meeting tomorrow at 2pm"
    }]
})
```

### Tool với Hỗ Trợ Async

```python
from langchain.tools import tool
import aiohttp

@tool
async def fetch_url(url: str) -> str:
    """Fetch content from a URL asynchronously."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.text()
                return content[:1000]  # Trả về 1000 ký tự đầu
            return f"Error: HTTP {response.status}"

# Sử dụng với async agent
result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "Fetch the content from example.com"}]
})
```

### Tool với Tích Hợp API

```python
from langchain.tools import tool
from pydantic import BaseModel, Field
import requests

class StockInput(BaseModel):
    """Input for stock lookup."""
    symbol: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")

@tool(args_schema=StockInput)
def get_stock_price(symbol: str) -> str:
    """Get the current stock price for a given ticker symbol."""
    # Ví dụ với dữ liệu mock (thay thế bằng API thực)
    mock_prices = {
        "AAPL": 178.50,
        "GOOGL": 142.30,
        "MSFT": 378.90,
        "AMZN": 178.25
    }

    symbol = symbol.upper()
    if symbol in mock_prices:
        return f"{symbol}: ${mock_prices[symbol]:.2f}"
    return f"Stock symbol {symbol} not found"

@tool
def get_stock_history(symbol: str, days: int = 7) -> str:
    """Get stock price history for the past N days."""
    # Triển khai mock
    return f"Price history for {symbol.upper()} (last {days} days): [Mock data]"

# Tạo trợ lý tài chính
finance_agent = create_agent(
    model="gpt-4o",
    tools=[get_stock_price, get_stock_history],
    system_prompt="""You are a financial assistant.
    Help users check stock prices and analyze market data.
    Always verify stock symbols before looking up prices."""
)
```

### Tool dạng Class (StructuredTool)

```python
from langchain.tools import StructuredTool
from pydantic import BaseModel
from typing import List

class DatabaseQueryInput(BaseModel):
    table: str
    columns: List[str]
    where: str = ""
    limit: int = 10

def execute_query(table: str, columns: List[str], where: str = "", limit: int = 10) -> str:
    """Execute a database query."""
    cols = ", ".join(columns)
    query = f"SELECT {cols} FROM {table}"
    if where:
        query += f" WHERE {where}"
    query += f" LIMIT {limit}"

    # Thực thi mock
    return f"Query executed: {query}\nResults: [Mock data]"

database_tool = StructuredTool.from_function(
    func=execute_query,
    name="query_database",
    description="Execute SQL-like queries against the database",
    args_schema=DatabaseQueryInput
)
```

---

## Tích Hợp Bộ Nhớ

### Bộ Nhớ Ngắn Hạn với Checkpointer

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

# Tạo checkpointer cho lưu trữ trạng thái
checkpointer = MemorySaver()

agent = create_agent(
    model="gpt-4o",
    tools=[],
    system_prompt="You are a helpful assistant with memory.",
    checkpointer=checkpointer
)

# Cuộc hội thoại với bộ nhớ
config = {"configurable": {"thread_id": "user-alice-123"}}

# Lượt 1
result = agent.invoke({
    "messages": [{"role": "user", "content": "My name is Alice and I work at Acme Corp."}]
}, config)
print(result["messages"][-1].content)

# Lượt 2 - Agent nhớ ngữ cảnh trước đó
result = agent.invoke({
    "messages": [{"role": "user", "content": "Where do I work?"}]
}, config)
print(result["messages"][-1].content)  # "You work at Acme Corp."

# Lượt 3
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's my name?"}]
}, config)
print(result["messages"][-1].content)  # "Your name is Alice."
```

### Cắt Bớt Tin Nhắn cho Cuộc Hội Thoại Dài

```python
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import MemorySaver

@before_model
def trim_messages(state: AgentState, runtime):
    """Chỉ giữ các tin nhắn gần đây để phù hợp với cửa sổ ngữ cảnh."""
    messages = state["messages"]

    # Giữ tất cả tin nhắn nếu cuộc hội thoại ngắn
    if len(messages) <= 5:
        return None

    # Giữ tin nhắn đầu (system) và 4 tin nhắn cuối
    first_msg = messages[0]
    recent = messages[-4:]

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            first_msg,
            *recent
        ]
    }

agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[trim_messages],
    checkpointer=MemorySaver()
)
```

### Bộ Nhớ Tóm Tắt

```python
from langchain.agents import create_agent
from langchain.agents.middleware import summarizationMiddleware
from langgraph.checkpoint.memory import MemorySaver

agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        summarizationMiddleware({
            "model": "gpt-4o-mini",  # Sử dụng mô hình nhỏ hơn cho tóm tắt
            "trigger": {"tokens": 4000},  # Kích hoạt ở 4000 tokens
            "keep": {"messages": 10}  # Giữ 10 tin nhắn cuối
        })
    ],
    checkpointer=MemorySaver()
)
```

### Bộ Nhớ Dài Hạn với Vector Store

```python
from langchain.agents import create_agent
from langgraph.store.memory import InMemoryStore
from langchain_openai import OpenAIEmbeddings

# Tạo hàm embedding
embeddings = OpenAIEmbeddings()

def embed(texts: list[str]) -> list[list[float]]:
    return embeddings.embed_documents(texts)

# Tạo memory store
store = InMemoryStore(index={"embed": embed, "dims": 1536})

# Lưu trữ preferences của người dùng
user_id = "user-123"
namespace = (user_id, "preferences")

store.put(namespace, "lang-pref", {"text": "User prefers concise responses"})
store.put(namespace, "tech-level", {"text": "User is an advanced Python developer"})
store.put(namespace, "interests", {"text": "User is interested in AI and machine learning"})

# Node agent sử dụng bộ nhớ dài hạn
async def call_model(state, config):
    # Tìm kiếm các memories liên quan
    memories = await config.store.search(
        (config.configurable["user_id"], "preferences"),
        query=state["messages"][-1].content,
        limit=3
    )

    memory_context = "\n".join([m.value["text"] for m in memories])

    system_msg = f"""You are a helpful assistant.
    User context: {memory_context}"""

    response = await model.invoke([
        {"role": "system", "content": system_msg},
        *state["messages"]
    ])

    return {"messages": [response]}
```

### PostgreSQL Persistence (Production)

```python
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

DB_URI = "postgresql://user:pass@localhost:5432/mydb"

# Tạo checkpointer và store bền vững
checkpointer = PostgresSaver.from_conn_string(DB_URI)
store = PostgresStore.from_conn_string(DB_URI)

# Setup tables (chạy một lần)
# await checkpointer.setup()
# await store.setup()

agent = create_agent(
    model="gpt-4o",
    tools=tools,
    checkpointer=checkpointer,
    # store=store  # Cho bộ nhớ dài hạn
)

# Các cuộc hội thoại được duy trì qua các lần khởi động lại
config = {"configurable": {"thread_id": "persistent-thread-1"}}
result = agent.invoke({"messages": [...]}, config)
```

---

## Hệ Thống Multi-Agent

### Supervisor với Sub-Agents

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o")

# Sub-agent 1: Nghiên cứu
@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

research_agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt="""You are a research assistant.
    Search for information and provide detailed summaries."""
)

# Sub-agent 2: Viết
writer_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="""You are a professional writer.
    Create polished content based on provided information."""
)

# Bọc sub-agents thành tools
@tool
def research(topic: str) -> str:
    """Research a topic thoroughly."""
    result = research_agent.invoke({
        "messages": [{"role": "user", "content": f"Research: {topic}"}]
    })
    return result["messages"][-1].content

@tool
def write_content(brief: str) -> str:
    """Write content based on a brief."""
    result = writer_agent.invoke({
        "messages": [{"role": "user", "content": brief}]
    })
    return result["messages"][-1].content

# Supervisor agent
supervisor = create_agent(
    model=model,
    tools=[research, write_content],
    system_prompt="""You are a content production supervisor.
    Coordinate research and writing to produce high-quality content.

    For content requests:
    1. First use the research tool to gather information
    2. Then use the write_content tool to create the final piece
    """
)

# Thực thi
result = supervisor.invoke({
    "messages": [{
        "role": "user",
        "content": "Create a blog post about the latest AI developments"
    }]
})
```

### Thực Thi Nhiệm Vụ Song Song

```python
import asyncio
from langchain.agents import create_agent
from langchain.tools import tool

# Tạo các agents chuyên biệt
@tool
def analyze_sentiment(text: str) -> str:
    """Analyze sentiment of text."""
    # Triển khai mock
    return "Positive sentiment (0.85 confidence)"

@tool
def extract_entities(text: str) -> str:
    """Extract named entities from text."""
    return "Entities: [Person: John, Organization: Acme, Location: NYC]"

@tool
def summarize(text: str) -> str:
    """Summarize text."""
    return "Summary: [Condensed version of the text]"

# Hàm phân tích song song
async def parallel_analysis(text: str):
    """Chạy nhiều phân tích song song."""

    sentiment_agent = create_agent(model="gpt-4o-mini", tools=[analyze_sentiment])
    entity_agent = create_agent(model="gpt-4o-mini", tools=[extract_entities])
    summary_agent = create_agent(model="gpt-4o-mini", tools=[summarize])

    tasks = [
        sentiment_agent.ainvoke({"messages": [{"role": "user", "content": f"Analyze: {text}"}]}),
        entity_agent.ainvoke({"messages": [{"role": "user", "content": f"Extract from: {text}"}]}),
        summary_agent.ainvoke({"messages": [{"role": "user", "content": f"Summarize: {text}"}]})
    ]

    results = await asyncio.gather(*tasks)

    return {
        "sentiment": results[0]["messages"][-1].content,
        "entities": results[1]["messages"][-1].content,
        "summary": results[2]["messages"][-1].content
    }

# Sử dụng
result = asyncio.run(parallel_analysis("Your text to analyze..."))
```

---

## Structured Output

### Output với Pydantic Model

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field
from typing import Literal, List

class MovieReview(BaseModel):
    """Structured movie review analysis."""
    title: str = Field(description="Movie title")
    rating: int = Field(ge=1, le=10, description="Rating from 1-10")
    sentiment: Literal["positive", "negative", "mixed"]
    pros: List[str] = Field(description="Positive aspects")
    cons: List[str] = Field(description="Negative aspects")
    recommendation: bool = Field(description="Would recommend")

agent = create_agent(
    model="gpt-4o",
    tools=[],
    response_format=ToolStrategy(MovieReview)
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": """Analyze this review:
        'The new sci-fi movie was visually stunning with incredible special effects.
        The plot was somewhat predictable but the acting was superb.
        Overall, a fun ride worth watching in theaters.'"""
    }]
})

review = result["structured_response"]
print(f"Title: {review.title}")
print(f"Rating: {review.rating}/10")
print(f"Sentiment: {review.sentiment}")
print(f"Recommend: {review.recommendation}")
```

### Nhiều Output Schemas

```python
from pydantic import BaseModel
from typing import Literal, List

class ProductReview(BaseModel):
    """Product review analysis."""
    product_name: str
    rating: int
    sentiment: Literal["positive", "negative"]
    key_points: List[str]

class ServiceComplaint(BaseModel):
    """Service complaint analysis."""
    issue_type: Literal["delivery", "quality", "support", "billing"]
    severity: Literal["low", "medium", "high"]
    description: str
    suggested_resolution: str

# Agent có thể xuất ra một trong hai schema
agent = create_agent(
    model="gpt-4o",
    tools=[],
    response_format=ToolStrategy([ProductReview, ServiceComplaint])
)

# Mô hình chọn schema phù hợp dựa trên đầu vào
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "My order arrived 2 weeks late and customer support was unhelpful"
    }]
})
# Trả về ServiceComplaint

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Great headphones! Amazing sound quality, 5 stars"
    }]
})
# Trả về ProductReview
```

---

## Xử Lý Lỗi

### Middleware Xử Lý Lỗi Tool

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

@wrap_tool_call
def handle_errors(request, handler):
    """Bắt và xử lý lỗi tool một cách duyên dáng."""
    try:
        return handler(request)
    except ValueError as e:
        return ToolMessage(
            content=f"Invalid input: {str(e)}. Please provide valid parameters.",
            tool_call_id=request.tool_call["id"]
        )
    except ConnectionError as e:
        return ToolMessage(
            content=f"Connection failed: {str(e)}. Please try again.",
            tool_call_id=request.tool_call["id"]
        )
    except Exception as e:
        return ToolMessage(
            content=f"Unexpected error: {str(e)}. Please try a different approach.",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model="gpt-4o",
    tools=[risky_tool],
    middleware=[handle_errors]
)
```

### Middleware Thử Lại

```python
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
import time

@wrap_tool_call
def retry_on_failure(request, handler, max_retries=3):
    """Thử lại các lời gọi tool thất bại với exponential backoff."""
    last_error = None

    for attempt in range(max_retries):
        try:
            return handler(request)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

    return ToolMessage(
        content=f"Failed after {max_retries} attempts: {str(last_error)}",
        tool_call_id=request.tool_call["id"]
    )
```

### Middleware Xác Thực

```python
from langchain.agents.middleware import before_model

@before_model
def validate_input(state, runtime):
    """Xác thực đầu vào trước khi xử lý."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    if not last_message:
        raise ValueError("No input message provided")

    content = last_message.get("content", "")
    if len(content) > 10000:
        raise ValueError("Input too long. Maximum 10,000 characters.")

    # Kiểm tra nội dung bị cấm
    prohibited = ["password", "api_key", "secret"]
    if any(word in content.lower() for word in prohibited):
        return {
            "messages": [
                *messages,
                {"role": "assistant", "content": "I cannot process requests containing sensitive information."}
            ]
        }

    return None  # Tiếp tục bình thường
```

---

## Streaming

### Stream Output của Agent

```python
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o",
    tools=[search, calculator],
    system_prompt="You are a helpful assistant."
)

# Stream values (trạng thái đầy đủ ở mỗi bước)
for event in agent.stream(
    {"messages": [{"role": "user", "content": "Search for AI news and summarize"}]},
    stream_mode="values"
):
    last_message = event["messages"][-1]
    print(f"Step: {last_message.type}")
    if hasattr(last_message, 'content') and last_message.content:
        print(f"Content: {last_message.content[:100]}...")

# Stream updates (chỉ những thay đổi)
for event in agent.stream(
    {"messages": [{"role": "user", "content": "What is 2+2?"}]},
    stream_mode="updates"
):
    for node, update in event.items():
        print(f"Node: {node}")
        print(f"Update: {update}")
```

### Stream Tokens

```python
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o",
    tools=[],
    system_prompt="You are a helpful assistant."
)

# Stream từng token
async for token in agent.astream(
    {"messages": [{"role": "user", "content": "Write a haiku about coding"}]},
    stream_mode="tokens"
):
    print(token, end="", flush=True)
```

### Stream với Progress Callback

```python
def stream_with_progress(agent, message):
    """Stream output của agent với theo dõi tiến trình."""
    step_count = 0

    for event in agent.stream(
        {"messages": [{"role": "user", "content": message}]},
        stream_mode="values"
    ):
        step_count += 1
        last_msg = event["messages"][-1]

        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                print(f"[Step {step_count}] Calling tool: {tc['name']}")

        elif last_msg.type == "tool":
            print(f"[Step {step_count}] Tool result received")

        elif last_msg.type == "ai" and last_msg.content:
            print(f"[Step {step_count}] Final response:")
            print(last_msg.content)

    return step_count
```

---

## Mẫu Nâng Cao

### Chọn Tool Động

```python
from langchain.agents import create_agent
from langchain.agents.middleware import before_model

# Định nghĩa các bộ tools cho các ngữ cảnh khác nhau
admin_tools = [delete_user, modify_permissions, view_logs]
user_tools = [view_profile, update_settings]
guest_tools = [view_public_info]

@before_model
def select_tools_by_role(state, runtime):
    """Chọn tools dựa trên vai trò của người dùng."""
    user_role = state.get("user_role", "guest")

    tool_map = {
        "admin": admin_tools,
        "user": user_tools,
        "guest": guest_tools
    }

    available_tools = tool_map.get(user_role, guest_tools)
    return {"tools": available_tools}

agent = create_agent(
    model="gpt-4o",
    tools=admin_tools + user_tools + guest_tools,  # Tất cả tools
    middleware=[select_tools_by_role]
)

# Gọi với ngữ cảnh vai trò
result = agent.invoke({
    "messages": [{"role": "user", "content": "Delete user john@example.com"}],
    "user_role": "admin"  # Tools được lọc dựa trên điều này
})
```

### Phân Nhánh Cuộc Hội Thoại

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
agent = create_agent(
    model="gpt-4o",
    tools=tools,
    checkpointer=checkpointer
)

# Cuộc hội thoại chính
main_config = {"configurable": {"thread_id": "main"}}
agent.invoke({"messages": [{"role": "user", "content": "Let's plan a trip to Japan"}]}, main_config)
agent.invoke({"messages": [{"role": "user", "content": "I want to visit Tokyo and Kyoto"}]}, main_config)

# Nhánh: Khám phá tùy chọn tiết kiệm
budget_config = {"configurable": {"thread_id": "main-budget"}}
# Sao chép trạng thái từ main
state = checkpointer.get(main_config["configurable"])
checkpointer.put(budget_config["configurable"], state)

agent.invoke({"messages": [{"role": "user", "content": "What's the cheapest option?"}]}, budget_config)

# Nhánh: Khám phá tùy chọn sang trọng
luxury_config = {"configurable": {"thread_id": "main-luxury"}}
checkpointer.put(luxury_config["configurable"], state)

agent.invoke({"messages": [{"role": "user", "content": "What's the most luxurious option?"}]}, luxury_config)

# Cuộc hội thoại chính tiếp tục không bị ảnh hưởng
agent.invoke({"messages": [{"role": "user", "content": "How long should I stay?"}]}, main_config)
```

### Agent với Custom State

```python
from langchain.agents import create_agent
from typing import TypedDict, List, Optional

class CustomState(TypedDict):
    messages: List[dict]
    user_id: str
    session_start: str
    interaction_count: int
    preferences: Optional[dict]

from langchain.agents.middleware import before_model

@before_model
def track_interactions(state: CustomState, runtime):
    """Theo dõi số lần tương tác."""
    return {"interaction_count": state.get("interaction_count", 0) + 1}

@before_model
def inject_preferences(state: CustomState, runtime):
    """Thêm preferences của người dùng vào ngữ cảnh."""
    prefs = state.get("preferences", {})
    if prefs:
        pref_str = ", ".join([f"{k}: {v}" for k, v in prefs.items()])
        system_msg = {"role": "system", "content": f"User preferences: {pref_str}"}
        return {"messages": [system_msg, *state["messages"]]}
    return None

agent = create_agent(
    model="gpt-4o",
    tools=tools,
    state_schema=CustomState,
    middleware=[track_interactions, inject_preferences]
)

# Gọi với custom state
result = agent.invoke({
    "messages": [{"role": "user", "content": "Help me book a restaurant"}],
    "user_id": "user-123",
    "session_start": "2024-01-15T10:00:00",
    "interaction_count": 0,
    "preferences": {"cuisine": "Japanese", "budget": "moderate"}
})
```

### Middleware Giới Hạn Tốc Độ

```python
from langchain.agents.middleware import before_model
from datetime import datetime, timedelta
from collections import defaultdict

# Bộ giới hạn tốc độ đơn giản trong bộ nhớ
rate_limits = defaultdict(list)

@before_model
def rate_limit(state, runtime, max_requests=10, window_seconds=60):
    """Giới hạn tốc độ yêu cầu theo người dùng."""
    user_id = state.get("user_id", "anonymous")
    now = datetime.now()
    window_start = now - timedelta(seconds=window_seconds)

    # Dọn dẹp các yêu cầu cũ
    rate_limits[user_id] = [
        ts for ts in rate_limits[user_id]
        if ts > window_start
    ]

    if len(rate_limits[user_id]) >= max_requests:
        return {
            "messages": [
                *state["messages"],
                {
                    "role": "assistant",
                    "content": f"Rate limit exceeded. Please wait {window_seconds} seconds."
                }
            ]
        }

    rate_limits[user_id].append(now)
    return None
```

---

## Tham Khảo Nhanh

### Cài Đặt

```bash
# Package cốt lõi
pip install langchain

# Với các nhà cung cấp cụ thể
pip install "langchain[anthropic]"
pip install "langchain[openai]"
pip install "langchain[google-genai]"

# Cho các tính năng LangGraph
pip install langgraph

# Cho production persistence
pip install langgraph-checkpoint-postgres
```

### Biến Môi Trường

```bash
# OpenAI
export OPENAI_API_KEY="your-key"

# Anthropic
export ANTHROPIC_API_KEY="your-key"

# Google
export GOOGLE_API_KEY="your-key"

# LangSmith (tùy chọn, cho tracing)
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="your-langsmith-key"
```

### Các Import Phổ Biến

```python
# Cốt lõi
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model

# Middleware
from langchain.agents.middleware import before_model, after_model, wrap_tool_call

# Structured Output
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy

# Memory
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# Messages
from langchain.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
```

---

## Bước Tiếp Theo

- [Kiến trúc](./architecture.md) - Đi sâu vào kiến trúc agent
- [Thành phần](./components.md) - Tài liệu thành phần chi tiết
- [Mẫu](./patterns.md) - Các mẫu kiến trúc phổ biến
