# Các Mẫu Thiết Kế Agent

Tài liệu này đề cập đến các mẫu suy luận và hành động cốt lõi được sử dụng bởi các AI agent riêng lẻ trên các framework chính.

## 1. ReAct (Suy Luận + Hành Động)

### Tổng Quan

Mẫu ReAct là mẫu agent nền tảng xen kẽ giữa các bước suy luận và hành động. Agent:
1. **Suy luận** về trạng thái hiện tại và thông tin/hành động cần thiết
2. **Hành động** bằng cách gọi công cụ hoặc tạo đầu ra
3. **Quan sát** kết quả
4. Lặp lại cho đến khi hoàn thành tác vụ

### Cách Hoạt Động

```
Truy vấn người dùng -> Suy luận -> Hành động (Gọi công cụ) -> Quan sát -> Suy luận -> Hành động (Câu trả lời cuối)
```

Agent duy trì một bảng ghi chép các suy nghĩ và quan sát trung gian, sử dụng chúng để đưa ra quyết định có căn cứ về các bước tiếp theo.

### Ví Dụ Triển Khai

#### LangChain Zero-Shot ReAct Agent

```python
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

agent = initialize_agent(
    tools=toolkit.get_tools(),
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    return_intermediate_steps=True
)
```

#### LangGraph ReAct Agent

```python
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Tìm kiếm thông tin."""
    return f"Kết quả cho {query}"

class AgentState(TypedDict):
    input: str
    agent_outcome: Optional[AgentOutcome]
    intermediate_steps: List[Tuple[AgentAction, str]]

def should_continue(state: AgentState) -> str:
    if isinstance(state["agent_outcome"], AgentFinish):
        return "end"
    return "continue"

workflow = StateGraph(AgentState)
workflow.add_node("agent", run_agent)
workflow.add_conditional_edges("agent", should_continue, {
    "end": END,
    "continue": "tools"
})
workflow.add_node("tools", execute_tools)
workflow.add_edge("tools", "agent")
```

### Thực Tiễn Tốt Nhất

- **Mô tả công cụ rõ ràng**: Agent dựa vào mô tả công cụ để quyết định sử dụng công cụ nào
- **Chế độ Verbose**: Bật ghi nhật ký verbose trong quá trình phát triển để hiểu suy luận
- **Giới hạn vòng lặp**: Đặt max_iterations để ngăn vòng lặp vô hạn
- **Temperature 0**: Sử dụng temperature thấp để suy luận nhất quán

---

## 2. Plan-and-Execute (Lập Kế Hoạch và Thực Thi)

### Tổng Quan

Mẫu Plan-and-Execute tách biệt lập kế hoạch khỏi thực thi:
1. **Giai đoạn Lập Kế Hoạch**: Tạo kế hoạch cấp cao với các bước
2. **Giai đoạn Thực Thi**: Thực thi từng bước, có thể lập kế hoạch lại nếu cần

Điều này phù hợp hơn cho các tác vụ phức tạp, nhiều bước mà việc lập kế hoạch trước cải thiện kết quả.

### Cách Hoạt Động

```
Truy vấn người dùng -> Agent Lập Kế Hoạch -> [Bước 1, Bước 2, Bước 3, ...]
                    -> Agent Thực Thi -> Thực thi Bước 1 -> Kết quả 1
                    -> Agent Thực Thi -> Thực thi Bước 2 -> Kết quả 2
                    -> ... -> Câu Trả Lời Cuối
```

### Ví Dụ Triển Khai

```python
from langgraph.graph import StateGraph, END

class PlanExecuteState(TypedDict):
    input: str
    plan: List[str]
    current_step: int
    results: List[str]
    final_answer: str

def planner(state: PlanExecuteState):
    """Tạo kế hoạch cho tác vụ."""
    plan = llm.invoke(f"Tạo kế hoạch từng bước cho: {state['input']}")
    return {"plan": parse_plan(plan), "current_step": 0}

def executor(state: PlanExecuteState):
    """Thực thi bước hiện tại."""
    step = state["plan"][state["current_step"]]
    result = execute_step(step)
    return {
        "results": state["results"] + [result],
        "current_step": state["current_step"] + 1
    }

def should_continue(state: PlanExecuteState):
    if state["current_step"] >= len(state["plan"]):
        return "synthesize"
    return "executor"

workflow = StateGraph(PlanExecuteState)
workflow.add_node("planner", planner)
workflow.add_node("executor", executor)
workflow.add_node("synthesize", synthesize_results)
workflow.add_edge("planner", "executor")
workflow.add_conditional_edges("executor", should_continue)
workflow.add_edge("synthesize", END)
```

### Khi Nào Sử Dụng

- Các tác vụ phức tạp với nhiều bước độc lập
- Các tác vụ yêu cầu phân bổ tài nguyên trước khi thực thi
- Khi phân tách tác vụ cải thiện độ chính xác
- Các tác vụ chạy dài được hưởng lợi từ checkpointing

---

## 3. Mẫu Phản Chiếu (Reflection)

### Tổng Quan

Các mẫu phản chiếu cho phép agent đánh giá và cải thiện đầu ra của chính mình:
1. Tạo phản hồi ban đầu
2. Phê bình phản hồi
3. Sửa đổi dựa trên phê bình
4. Lặp lại cho đến khi đạt yêu cầu

### Triển Khai Tự Phản Chiếu

```python
from langgraph.graph import StateGraph, END

class ReflectionState(TypedDict):
    input: str
    draft: str
    critique: str
    revision_count: int
    final_output: str

def generate_draft(state: ReflectionState):
    """Tạo phản hồi ban đầu."""
    draft = llm.invoke(f"Trả lời câu hỏi này: {state['input']}")
    return {"draft": draft, "revision_count": 0}

def critique(state: ReflectionState):
    """Phê bình bản nháp hiện tại."""
    critique = llm.invoke(f"""
    Phê bình phản hồi này về độ chính xác, đầy đủ và rõ ràng:

    Câu hỏi: {state['input']}
    Phản hồi: {state['draft']}

    Đưa ra đề xuất cụ thể để cải thiện.
    """)
    return {"critique": critique}

def revise(state: ReflectionState):
    """Sửa đổi dựa trên phê bình."""
    revised = llm.invoke(f"""
    Cải thiện phản hồi này dựa trên phê bình:

    Bản gốc: {state['draft']}
    Phê bình: {state['critique']}

    Đưa ra phản hồi được cải thiện.
    """)
    return {"draft": revised, "revision_count": state["revision_count"] + 1}

def should_continue(state: ReflectionState):
    if state["revision_count"] >= 3:
        return "finalize"
    if "không cần cải thiện" in state["critique"].lower():
        return "finalize"
    return "revise"

workflow = StateGraph(ReflectionState)
workflow.add_node("generate", generate_draft)
workflow.add_node("critique", critique)
workflow.add_node("revise", revise)
workflow.add_node("finalize", lambda s: {"final_output": s["draft"]})

workflow.add_edge("generate", "critique")
workflow.add_conditional_edges("critique", should_continue)
workflow.add_edge("revise", "critique")
workflow.add_edge("finalize", END)
```

### Các Loại Phản Chiếu

| Loại | Mô Tả | Trường Hợp Sử Dụng |
|------|-------|-------------------|
| Tự Phê Bình | Agent đánh giá đầu ra của chính mình | Cải thiện chất lượng |
| Đánh Giá Ngang Hàng | Agent khác đánh giá đầu ra | Hệ thống đa agent |
| Phản Chiếu Thực Thi | Phản chiếu về kết quả hành động | Học từ sai lầm |
| Siêu Nhận Thức | Phản chiếu về quy trình suy luận | Các tác vụ suy luận phức tạp |

---

## 4. Mẫu Tự Phê Bình

### Tổng Quan

Tự phê bình là một dạng phản chiếu chuyên biệt trong đó agent xác thực rõ ràng đầu ra của mình theo các tiêu chí.

### Triển Khai với Guardrails

```python
from crewai import Task, TaskOutput
import json

def validate_json_output(result: TaskOutput) -> Tuple[bool, Any]:
    """Xác thực và phân tích đầu ra JSON."""
    try:
        data = json.loads(result)
        # Logic xác thực bổ sung
        if "required_field" not in data:
            return (False, "Thiếu required_field")
        return (True, data)
    except json.JSONDecodeError as e:
        return (False, f"JSON không hợp lệ: {e}")

task = Task(
    description="Tạo báo cáo JSON",
    expected_output="Đối tượng JSON hợp lệ với required_field",
    agent=analyst,
    guardrail=validate_json_output,
    guardrail_max_retries=3  # Thử lại tối đa 3 lần
)
```

### Các Chiều Phê Bình

Các chiều phổ biến để tự phê bình:
- **Độ chính xác**: Thông tin có đúng không?
- **Đầy đủ**: Tất cả các khía cạnh có được đề cập không?
- **Liên quan**: Nó có trả lời câu hỏi không?
- **Rõ ràng**: Nó có được tổ chức tốt và dễ hiểu không?
- **An toàn**: Nó có tránh nội dung có hại không?

---

## 5. Mẫu Sử Dụng Công Cụ

### Tổng Quan

Sử dụng công cụ là nền tảng cho khả năng của agent. Agent sử dụng công cụ để:
- Truy cập thông tin bên ngoài (tìm kiếm, cơ sở dữ liệu)
- Thực hiện hành động (gửi email, tạo tệp)
- Thực thi mã
- Tương tác với API

### Định Nghĩa Công Cụ Cơ Bản

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="Truy vấn tìm kiếm")
    max_results: int = Field(default=5, description="Số kết quả tối đa")

@tool("web_search", args_schema=SearchInput)
def web_search(query: str, max_results: int = 5) -> str:
    """Tìm kiếm thông tin trên web."""
    # Triển khai
    return f"Kết quả tìm kiếm cho: {query}"
```

### Chiến Lược Chọn Công Cụ

| Chiến Lược | Mô Tả | Khi Nào Sử Dụng |
|-----------|-------|-----------------|
| Zero-Shot | Chọn dựa trên mô tả | Bộ công cụ đơn giản |
| Few-Shot | Bao gồm ví dụ | Công cụ phức tạp |
| Động | Lọc công cụ khi chạy | Bộ công cụ lớn |
| Phân cấp | Tổ chức công cụ theo danh mục | Nhiều công cụ liên quan |

---

## 6. Vòng Lặp Agentic

### Tổng Quan

Vòng lặp agentic là các mẫu lặp lại trong đó agent liên tục thực hiện hành động cho đến khi đạt được mục tiêu.

### Triển Khai

```python
from langgraph.graph import StateGraph, END

class LoopState(TypedDict):
    input: str
    iterations: int
    max_iterations: int
    output: str
    is_complete: bool

def agent_step(state: LoopState):
    """Thực thi một bước agent."""
    result = agent.invoke(state["input"])
    is_complete = check_completion(result)
    return {
        "output": result,
        "iterations": state["iterations"] + 1,
        "is_complete": is_complete
    }

def should_continue(state: LoopState):
    if state["is_complete"]:
        return END
    if state["iterations"] >= state["max_iterations"]:
        return END
    return "agent"

workflow = StateGraph(LoopState)
workflow.add_node("agent", agent_step)
workflow.add_conditional_edges("agent", should_continue)
workflow.set_entry_point("agent")
```

### Thực Tiễn Tốt Nhất Kiểm Soát Vòng Lặp

1. **Đặt Số Vòng Lặp Tối Đa**: Ngăn vòng lặp vô hạn
2. **Điều Kiện Thoát Rõ Ràng**: Định nghĩa khi nào tác vụ hoàn thành
3. **Theo Dõi Tiến Trình**: Ghi nhật ký tiến trình vòng lặp
4. **Xử Lý Timeout**: Đặt giới hạn thời gian tổng thể
5. **Checkpoint Trạng Thái**: Lưu trạng thái để khôi phục

---

## So Sánh Mẫu

| Mẫu | Suy Luận | Lập Kế Hoạch | Lặp Lại | Phù Hợp Cho |
|-----|----------|--------------|---------|-------------|
| ReAct | Theo bước | Không | Vòng lặp công cụ | Tác vụ chung |
| Plan-Execute | Tối thiểu | Trước | Các bước kế hoạch | Tác vụ phức tạp |
| Reflection | Phê bình | Không | Cải thiện | Chất lượng quan trọng |
| Self-Critique | Xác thực | Không | Thử lại | Đầu ra có cấu trúc |
| Agentic Loop | Liên tục | Không | Hướng mục tiêu | Tác vụ lặp |

## Hỗ Trợ Framework

| Mẫu | LangChain | LangGraph | CrewAI | AutoGen |
|-----|-----------|-----------|--------|---------|
| ReAct | Tích hợp | Tích hợp | Qua process | Qua agents |
| Plan-Execute | Template | Tùy chỉnh | Hierarchical | Tùy chỉnh |
| Reflection | Tùy chỉnh | Tích hợp | Qua tasks | Tùy chỉnh |
| Self-Critique | Tùy chỉnh | Tích hợp | Guardrails | Tùy chỉnh |
| Agentic Loop | Hạn chế | Tích hợp | Sequential | Tích hợp |
