# Các Mẫu Tích Hợp Công Cụ

Tài liệu này đề cập đến các mẫu tích hợp công cụ với AI agent, bao gồm gọi hàm, schema công cụ, chọn động và xử lý lỗi.

## 1. Các Mẫu Gọi Hàm

### Tổng Quan

Gọi hàm cho phép LLM gọi các hàm/công cụ bên ngoài theo cách có cấu trúc. Mô hình xuất dữ liệu có cấu trúc (tên hàm + tham số) có thể được phân tích và thực thi.

### Định Nghĩa Công Cụ Cơ Bản

#### Sử Dụng Decorators (LangChain)

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    """Schema đầu vào cho công cụ tìm kiếm."""
    query: str = Field(description="Truy vấn tìm kiếm")
    max_results: int = Field(default=5, ge=1, le=20, description="Số kết quả tối đa trả về")

@tool("web_search", args_schema=SearchInput)
def web_search(query: str, max_results: int = 5) -> str:
    """Tìm kiếm thông tin trên web.

    Args:
        query: Truy vấn tìm kiếm
        max_results: Số kết quả tối đa

    Returns:
        Kết quả tìm kiếm dạng chuỗi đã định dạng
    """
    # Triển khai
    results = perform_search(query, max_results)
    return format_results(results)
```

#### Sử Dụng Lớp BaseTool (CrewAI)

```python
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
from typing import Type, List, Any
import os

class WeatherToolInput(BaseModel):
    """Schema đầu vào cho công cụ thời tiết."""
    city: str = Field(..., description="Tên thành phố")
    units: str = Field(default="celsius", description="Đơn vị nhiệt độ")

class WeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Lấy thời tiết hiện tại cho một thành phố"
    args_schema: Type[BaseModel] = WeatherToolInput

    # Biến môi trường cần thiết
    env_vars: List[EnvVar] = [
        EnvVar(name="WEATHER_API_KEY", description="Khóa API", required=True)
    ]

    # Phụ thuộc package
    package_dependencies: List[str] = ["requests"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "WEATHER_API_KEY" not in os.environ:
            raise ValueError("Yêu cầu WEATHER_API_KEY")

    def _run(self, city: str, units: str = "celsius") -> str:
        """Thực thi đồng bộ."""
        # Triển khai
        return f"Thời tiết tại {city}: 22 độ {units}"

    async def _arun(self, city: str, units: str = "celsius") -> str:
        """Thực thi async."""
        return self._run(city, units)
```

#### Schema Công Cụ Thô

```python
# Định nghĩa schema công cụ thủ công
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Lấy thời tiết hiện tại cho một vị trí",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Tên thành phố, ví dụ: 'Hà Nội'"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Đơn vị nhiệt độ"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Gắn công cụ vào LLM
llm_with_tools = llm.bind(tools=tools)
response = llm_with_tools.invoke(messages)
```

---

## 2. Các Mẫu Schema Công Cụ

### Schema Pydantic với Xác Thực

```python
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class CreateTaskInput(BaseModel):
    """Schema cho tạo tác vụ với xác thực."""

    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Tiêu đề tác vụ"
    )

    description: Optional[str] = Field(
        None,
        max_length=1000,
        description="Mô tả chi tiết tác vụ"
    )

    priority: Priority = Field(
        default=Priority.MEDIUM,
        description="Mức độ ưu tiên tác vụ"
    )

    due_date: Optional[str] = Field(
        None,
        description="Ngày hết hạn định dạng YYYY-MM-DD"
    )

    tags: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Thẻ tác vụ"
    )

    @field_validator("due_date")
    @classmethod
    def validate_date_format(cls, v):
        if v is None:
            return v
        from datetime import datetime
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Ngày phải có định dạng YYYY-MM-DD")
        return v

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v):
        return [tag.lower().strip() for tag in v]
```

### Schema Lồng Nhau

```python
from pydantic import BaseModel, Field
from typing import List

class Address(BaseModel):
    street: str = Field(..., description="Địa chỉ đường")
    city: str = Field(..., description="Tên thành phố")
    country: str = Field(..., description="Mã quốc gia")
    postal_code: str = Field(..., description="Mã bưu chính")

class ContactInfo(BaseModel):
    email: str = Field(..., description="Địa chỉ email")
    phone: Optional[str] = Field(None, description="Số điện thoại")

class CreateCustomerInput(BaseModel):
    """Tạo bản ghi khách hàng mới."""
    name: str = Field(..., description="Họ tên khách hàng")
    contact: ContactInfo = Field(..., description="Thông tin liên hệ")
    addresses: List[Address] = Field(
        default_factory=list,
        description="Địa chỉ khách hàng"
    )

@tool("create_customer", args_schema=CreateCustomerInput)
def create_customer(name: str, contact: ContactInfo, addresses: List[Address]) -> str:
    """Tạo khách hàng mới với thông tin liên hệ và địa chỉ."""
    # Triển khai
    return f"Đã tạo khách hàng: {name}"
```

---

## 3. Chọn Công Cụ Động

### Tổng Quan

Đối với agent có nhiều công cụ, chọn động lọc các công cụ liên quan khi chạy để giảm kích thước prompt và cải thiện độ chính xác.

### Chọn Dựa Trên Decorator

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

def get_relevant_tools(state: dict, all_tools: list) -> list:
    """Chọn công cụ liên quan dựa trên trạng thái hiện tại."""
    # Chọn dựa trên danh mục
    category = state.get("category", "general")
    category_tools = {
        "finance": ["calculator", "stock_lookup", "currency_convert"],
        "research": ["web_search", "wiki_lookup", "arxiv_search"],
        "communication": ["send_email", "send_slack", "create_ticket"]
    }
    tool_names = category_tools.get(category, [])

    # Lọc công cụ
    return [t for t in all_tools if t.name in tool_names]

@wrap_model_call
def select_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Middleware để chọn công cụ động."""
    relevant_tools = get_relevant_tools(request.state, request.tools)
    return handler(request.override(tools=relevant_tools))

agent = create_agent(
    model="gpt-4o",
    tools=all_tools,  # Tất cả công cụ đăng ký
    middleware=[select_tools]
)
```

### Chọn Công Cụ Ngữ Nghĩa

```python
from langchain_openai import OpenAIEmbeddings
import numpy as np

class SemanticToolSelector:
    def __init__(self, tools: List):
        self.tools = tools
        self.embeddings = OpenAIEmbeddings()

        # Tính toán trước embedding công cụ từ mô tả
        self.tool_embeddings = {}
        for tool in tools:
            desc = f"{tool.name}: {tool.description}"
            self.tool_embeddings[tool.name] = self.embeddings.embed_query(desc)

    def select_tools(self, query: str, k: int = 5) -> List:
        """Chọn k công cụ liên quan nhất cho query."""
        query_embedding = self.embeddings.embed_query(query)

        # Tính độ tương đồng
        scores = []
        for tool in self.tools:
            tool_emb = self.tool_embeddings[tool.name]
            similarity = np.dot(query_embedding, tool_emb)
            scores.append((tool, similarity))

        # Sắp xếp theo độ tương đồng và trả về top k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [tool for tool, _ in scores[:k]]

# Sử dụng
selector = SemanticToolSelector(all_tools)
relevant_tools = selector.select_tools("Tôi cần tìm kiếm thông tin trên web")
agent = create_agent(model="gpt-4o", tools=relevant_tools)
```

### Tổ Chức Công Cụ Phân Cấp

```python
class ToolCategory:
    """Tổ chức công cụ thành các danh mục để chọn phân cấp."""

    def __init__(self, name: str, description: str, tools: List):
        self.name = name
        self.description = description
        self.tools = tools

# Định nghĩa danh mục
categories = [
    ToolCategory(
        name="search",
        description="Công cụ để tìm kiếm và truy xuất thông tin",
        tools=[web_search, wiki_search, arxiv_search]
    ),
    ToolCategory(
        name="calculation",
        description="Công cụ cho các phép toán",
        tools=[calculator, unit_converter, statistics]
    ),
    ToolCategory(
        name="communication",
        description="Công cụ để gửi tin nhắn và thông báo",
        tools=[email, slack, sms]
    )
]

# Chọn hai bước: đầu tiên danh mục, sau đó công cụ
@tool("select_category")
def select_category(task_description: str) -> str:
    """Chọn danh mục công cụ phù hợp nhất cho tác vụ."""
    # LLM chọn danh mục
    return llm.invoke(f"Danh mục nào cho: {task_description}")

def get_tools_for_category(category_name: str) -> List:
    """Lấy công cụ cho danh mục cụ thể."""
    for cat in categories:
        if cat.name == category_name:
            return cat.tools
    return []
```

---

## 4. Chuỗi Công Cụ

### Tổng Quan

Chuỗi công cụ kết nối nhiều cuộc gọi công cụ trong đó đầu ra của một công cụ trở thành đầu vào của công cụ khác.

### Chuỗi Tuần Tự

```python
from langgraph.graph import StateGraph, END

class ChainState(TypedDict):
    query: str
    search_results: str
    analysis: str
    summary: str

def search_step(state: ChainState):
    """Bước 1: Tìm kiếm thông tin."""
    results = web_search.invoke({"query": state["query"]})
    return {"search_results": results}

def analyze_step(state: ChainState):
    """Bước 2: Phân tích kết quả tìm kiếm."""
    analysis = analyzer.invoke({"data": state["search_results"]})
    return {"analysis": analysis}

def summarize_step(state: ChainState):
    """Bước 3: Tóm tắt phân tích."""
    summary = summarizer.invoke({"content": state["analysis"]})
    return {"summary": summary}

workflow = StateGraph(ChainState)
workflow.add_node("search", search_step)
workflow.add_node("analyze", analyze_step)
workflow.add_node("summarize", summarize_step)

workflow.add_edge("search", "analyze")
workflow.add_edge("analyze", "summarize")
workflow.add_edge("summarize", END)

workflow.set_entry_point("search")
chain = workflow.compile()
```

### Chuỗi Có Điều Kiện

```python
def route_after_search(state: ChainState) -> str:
    """Quyết định bước tiếp theo dựa trên kết quả tìm kiếm."""
    if not state["search_results"]:
        return "retry_search"
    if needs_deep_analysis(state["search_results"]):
        return "deep_analyze"
    return "quick_summarize"

workflow = StateGraph(ChainState)
workflow.add_node("search", search_step)
workflow.add_node("retry_search", retry_search_step)
workflow.add_node("deep_analyze", deep_analyze_step)
workflow.add_node("quick_summarize", quick_summarize_step)

workflow.add_conditional_edges("search", route_after_search, {
    "retry_search": "retry_search",
    "deep_analyze": "deep_analyze",
    "quick_summarize": "quick_summarize"
})
```

### Thực Thi Công Cụ Song Song

```python
import asyncio
from typing import List

async def parallel_tool_execution(tools: List, inputs: List[dict]) -> List:
    """Thực thi nhiều công cụ song song."""
    async def execute_tool(tool, input_data):
        return await tool.ainvoke(input_data)

    tasks = [execute_tool(t, i) for t, i in zip(tools, inputs)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Xử lý các ngoại lệ
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "tool": tools[i].name,
                "error": str(result)
            })
        else:
            processed_results.append({
                "tool": tools[i].name,
                "result": result
            })

    return processed_results

# Sử dụng
results = await parallel_tool_execution(
    [search_tool, wiki_tool, news_tool],
    [{"query": q} for q in ["AI agents", "LLM agents", "autonomous AI"]]
)
```

---

## 5. Xử Lý Lỗi và Phục Hồi

### Mẫu Thử Lại

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

class ToolWithRetry(BaseTool):
    name = "api_tool"
    description = "Công cụ với logic thử lại tích hợp"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutError, httpx.HTTPStatusError))
    )
    def _run(self, input: str) -> str:
        """Thực thi với tự động thử lại khi thất bại."""
        response = httpx.get(f"https://api.example.com/{input}", timeout=10)
        response.raise_for_status()
        return response.json()
```

### CrewAI Guardrails với Thử Lại

```python
from crewai import Task, TaskOutput
from typing import Tuple, Any
import json

def validate_json_output(result: TaskOutput) -> Tuple[bool, Any]:
    """Xác thực đầu ra là JSON hợp lệ."""
    try:
        data = json.loads(result.raw)
        # Xác thực bổ sung
        if "required_field" not in data:
            return (False, "Thiếu required_field trong phản hồi")
        return (True, data)
    except json.JSONDecodeError as e:
        return (False, f"JSON không hợp lệ: {e}")

task = Task(
    description="Tạo báo cáo JSON",
    expected_output="JSON hợp lệ với required_field",
    agent=analyst,
    guardrail=validate_json_output,
    guardrail_max_retries=3  # Agent thử lại tối đa 3 lần
)
```

### Chiến Lược Dự Phòng

```python
class ToolWithFallback:
    """Công cụ có dự phòng thay thế khi thất bại."""

    def __init__(self, primary_tool, fallback_tool):
        self.primary = primary_tool
        self.fallback = fallback_tool

    async def invoke(self, input: dict) -> str:
        try:
            return await self.primary.ainvoke(input)
        except Exception as e:
            print(f"Công cụ chính thất bại: {e}, thử dự phòng")
            try:
                return await self.fallback.ainvoke(input)
            except Exception as e2:
                return f"Cả hai công cụ đều thất bại. Chính: {e}, Dự phòng: {e2}"

# Sử dụng
search_with_fallback = ToolWithFallback(
    primary_tool=google_search,
    fallback_tool=bing_search
)
```

### Leo Thang Đến Con Người

```python
from langgraph.graph import StateGraph, END

class EscalationState(TypedDict):
    input: str
    tool_result: str
    error: Optional[str]
    requires_human: bool
    human_response: Optional[str]

def execute_tool(state: EscalationState):
    """Thực thi công cụ với xử lý lỗi."""
    try:
        result = tool.invoke(state["input"])
        return {"tool_result": result, "error": None, "requires_human": False}
    except Exception as e:
        # Xác định nếu cần leo thang đến con người
        if is_critical_error(e):
            return {
                "error": str(e),
                "requires_human": True,
                "tool_result": None
            }
        return {"error": str(e), "requires_human": False, "tool_result": None}

def human_escalation(state: EscalationState):
    """Yêu cầu can thiệp của con người."""
    # Trong production, điều này sẽ kích hoạt thông báo/workflow
    print(f"Cần can thiệp của con người cho: {state['error']}")
    # Chặn cho đến khi con người phản hồi (hoặc timeout)
    human_response = wait_for_human_response(state)
    return {"human_response": human_response}

def route_after_tool(state: EscalationState) -> str:
    if state["requires_human"]:
        return "human"
    if state["error"]:
        return "retry"
    return "complete"

workflow = StateGraph(EscalationState)
workflow.add_node("execute", execute_tool)
workflow.add_node("human", human_escalation)
workflow.add_node("retry", retry_step)
workflow.add_node("complete", complete_step)

workflow.add_conditional_edges("execute", route_after_tool)
workflow.add_edge("human", "complete")
workflow.add_edge("retry", "execute")
workflow.add_edge("complete", END)
```

### Suy Giảm Uyển Chuyển

```python
class GracefulTool(BaseTool):
    """Công cụ suy giảm uyển chuyển khi thất bại một phần."""

    name = "resilient_search"
    description = "Tìm kiếm trên nhiều nguồn với suy giảm uyển chuyển"

    def _run(self, query: str) -> str:
        sources = [
            ("google", self._search_google),
            ("bing", self._search_bing),
            ("duckduckgo", self._search_ddg)
        ]

        results = []
        errors = []

        for source_name, search_fn in sources:
            try:
                result = search_fn(query)
                results.append(f"[{source_name}] {result}")
            except Exception as e:
                errors.append(f"{source_name}: {e}")

        # Trả về những gì có được
        if results:
            output = "Kết quả tìm kiếm:\n" + "\n".join(results)
            if errors:
                output += f"\n\nLưu ý: Một số nguồn thất bại: {errors}"
            return output
        else:
            return f"Tất cả nguồn tìm kiếm đều thất bại: {errors}"
```

---

## 6. Thực Tiễn Tốt Nhất Công Cụ

### 1. Thiết Kế Công Cụ

```python
# Tốt: Công cụ rõ ràng, tập trung
@tool("get_user_email")
def get_user_email(user_id: str) -> str:
    """Lấy địa chỉ email cho người dùng cụ thể.

    Args:
        user_id: Định danh duy nhất của người dùng

    Returns:
        Địa chỉ email của người dùng hoặc thông báo lỗi
    """
    pass

# Xấu: Công cụ quá rộng
@tool("manage_user")
def manage_user(action: str, user_id: str, data: dict) -> str:
    """Quản lý người dùng - tạo, cập nhật, xóa hoặc lấy."""
    pass  # Quá nhiều trách nhiệm
```

### 2. Mô Tả Rõ Ràng

```python
# Mô tả tốt
@tool
def calculate_compound_interest(
    principal: float,
    rate: float,
    time: int,
    compounds_per_year: int = 12
) -> float:
    """Tính lãi kép trên khoản đầu tư.

    Sử dụng công cụ này khi bạn cần tính khoản đầu tư sẽ tăng
    bao nhiêu theo thời gian với lãi kép.

    Args:
        principal: Số tiền đầu tư ban đầu
        rate: Lãi suất hàng năm dạng thập phân (ví dụ: 0.05 cho 5%)
        time: Số năm
        compounds_per_year: Tần suất tính lãi (mặc định: 12 cho hàng tháng)

    Returns:
        Số tiền cuối cùng sau lãi kép

    Ví dụ:
        calculate_compound_interest(1000, 0.05, 10, 12)
        Trả về: 1647.01 (cho $1000 với 5% trong 10 năm, tính lãi hàng tháng)
    """
    pass
```

### 3. Xác Thực Đầu Vào

```python
from pydantic import BaseModel, Field, field_validator

class TransferFundsInput(BaseModel):
    from_account: str = Field(..., description="ID tài khoản nguồn")
    to_account: str = Field(..., description="ID tài khoản đích")
    amount: float = Field(..., gt=0, description="Số tiền chuyển")
    currency: str = Field(default="USD", description="Mã tiền tệ")

    @field_validator("from_account", "to_account")
    @classmethod
    def validate_account_format(cls, v):
        if not v.startswith("ACC-"):
            raise ValueError("ID tài khoản phải bắt đầu bằng 'ACC-'")
        return v

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, v):
        valid_currencies = ["USD", "EUR", "GBP", "JPY"]
        if v not in valid_currencies:
            raise ValueError(f"Tiền tệ phải là một trong {valid_currencies}")
        return v
```

### 4. Thông Báo Lỗi

```python
@tool
def fetch_stock_price(symbol: str) -> str:
    """Lấy giá cổ phiếu hiện tại cho mã ticker."""
    try:
        price = stock_api.get_price(symbol)
        return f"Giá hiện tại của {symbol}: ${price:.2f}"
    except SymbolNotFoundError:
        return f"Lỗi: Mã cổ phiếu '{symbol}' không tìm thấy. Vui lòng xác minh mã ticker."
    except APIRateLimitError:
        return "Lỗi: Vượt quá giới hạn rate. Vui lòng thử lại sau vài giây."
    except Exception as e:
        return f"Lỗi lấy giá cổ phiếu: {str(e)}. Vui lòng thử lại."
```

### 5. Tính Idempotent

```python
import hashlib
from functools import lru_cache

class IdempotentTool(BaseTool):
    """Công cụ đảm bảo các hoạt động idempotent."""

    def __init__(self):
        super().__init__()
        self._executed_operations = {}

    def _generate_operation_key(self, **kwargs) -> str:
        """Tạo khóa duy nhất cho hoạt động."""
        content = json.dumps(kwargs, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def _run(self, **kwargs) -> str:
        op_key = self._generate_operation_key(**kwargs)

        # Kiểm tra nếu đã thực thi
        if op_key in self._executed_operations:
            return f"Hoạt động đã được thực thi. Kết quả: {self._executed_operations[op_key]}"

        # Thực thi và lưu
        result = self._execute(**kwargs)
        self._executed_operations[op_key] = result
        return result
```

---

## So Sánh Framework

| Tính Năng | LangChain | CrewAI | AutoGen |
|-----------|-----------|--------|---------|
| Định nghĩa công cụ | @tool decorator | Lớp BaseTool | Function def |
| Schema | Pydantic | Pydantic | Dict/Pydantic |
| Hỗ trợ Async | Tích hợp | Tích hợp | Tích hợp |
| Chọn động | Middleware | Tùy chỉnh | Tùy chỉnh |
| Xử lý lỗi | Try/catch | Guardrails | Message handling |
| Chuỗi công cụ | LangGraph | Task context | Conversation |
