# Các Mẫu Agent LangChain

Tài liệu này trình bày các mẫu kiến trúc phổ biến được sử dụng trong phát triển agent LangChain, bao gồm hệ thống single vs. multi-agent, chuỗi tuần tự, mẫu router và luồng công việc human-in-the-loop.

---

## Mục Lục

1. [Mẫu Single Agent](#mau-single-agent)
2. [Mẫu Multi-Agent](#mau-multi-agent)
3. [Chuỗi Tuần Tự](#chuoi-tuan-tu)
4. [Mẫu Router](#mau-router)
5. [Mẫu Plan-and-Execute](#mau-plan-and-execute)
6. [Mẫu Human-in-the-Loop](#mau-human-in-the-loop)
7. [Mẫu Workflow với LangGraph](#mau-workflow-voi-langgraph)

---

## Mẫu Single Agent

Mẫu đơn giản nhất trong đó một agent xử lý tất cả các nhiệm vụ.

### Single Agent Cơ bản

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

agent = create_agent(
    model="gpt-4o",
    tools=[search, calculator],
    system_prompt="""You are a helpful assistant that can search the web
    and perform calculations. Use the appropriate tool for each task."""
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is the population of Tokyo times 2?"}]
})
```

### Single Agent với Bộ Nhớ

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

agent = create_agent(
    model="gpt-4o",
    tools=[search, calculator],
    system_prompt="You are a helpful assistant.",
    checkpointer=checkpointer
)

# Cuộc hội thoại được duy trì qua các lần gọi
config = {"configurable": {"thread_id": "user-123"}}

agent.invoke({"messages": [{"role": "user", "content": "My name is Alice"}]}, config)
agent.invoke({"messages": [{"role": "user", "content": "What's my name?"}]}, config)
# Agent nhớ: "Your name is Alice"
```

### Khi Nào Sử Dụng Single Agent

| Trường hợp Sử dụng | Khuyến nghị |
|----------|----------------|
| Nhiệm vụ đơn giản | Lý tưởng |
| < 5 tools | Lý tưởng |
| Nhiệm vụ đồng nhất | Lý tưởng |
| Đa miền phức tạp | Xem xét multi-agent |
| Nhiều tools chuyên biệt | Xem xét multi-agent |

---

## Mẫu Multi-Agent

Hệ thống multi-agent phối hợp các agents chuyên biệt để xử lý các nhiệm vụ phức tạp.

### Mẫu 1: Supervisor Pattern

Một supervisor trung tâm phân công nhiệm vụ cho các sub-agents chuyên biệt.

```
                    +------------------+
                    |   Supervisor     |
                    |     Agent        |
                    +------------------+
                           |
           +---------------+---------------+
           |               |               |
           v               v               v
    +------------+  +------------+  +------------+
    |  Calendar  |  |   Email    |  |  Research  |
    |   Agent    |  |   Agent    |  |   Agent    |
    +------------+  +------------+  +------------+
```

```python
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

model = init_chat_model("claude-haiku-4-5-20251001")

# Bước 1: Định nghĩa các tools cấp thấp
@tool
def create_calendar_event(title: str, start_time: str, end_time: str) -> str:
    """Create a calendar event."""
    return f"Event created: {title} from {start_time} to {end_time}"

@tool
def send_email(to: list[str], subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {', '.join(to)} - Subject: {subject}"

# Bước 2: Tạo các sub-agents chuyên biệt
calendar_agent = create_agent(
    model,
    tools=[create_calendar_event],
    system_prompt="""You are a calendar scheduling assistant.
    Parse natural language requests into proper datetime formats.
    Always confirm what was scheduled."""
)

email_agent = create_agent(
    model,
    tools=[send_email],
    system_prompt="""You are an email assistant.
    Compose professional emails and confirm what was sent."""
)

# Bước 3: Bọc sub-agents thành tools cho supervisor
@tool
def schedule_event(request: str) -> str:
    """Schedule calendar events using natural language."""
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text

@tool
def manage_email(request: str) -> str:
    """Send emails using natural language."""
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text

# Bước 4: Tạo supervisor agent
supervisor = create_agent(
    model,
    tools=[schedule_event, manage_email],
    system_prompt="""You are a personal assistant.
    You can schedule events and send emails.
    Break down complex requests into appropriate tool calls."""
)

# Thực thi yêu cầu nhiều bước
result = supervisor.invoke({
    "messages": [{
        "role": "user",
        "content": "Schedule a meeting with design team Tuesday 2pm and email them a reminder"
    }]
})
```

### Mẫu 2: Parallel Agents

Nhiều agents làm việc độc lập trên các khía cạnh khác nhau của một nhiệm vụ.

```python
import asyncio
from langchain.agents import create_agent

# Tạo các agents chuyên biệt
research_agent = create_agent(
    model="gpt-4o",
    tools=[web_search],
    system_prompt="Research and summarize topics."
)

analysis_agent = create_agent(
    model="gpt-4o",
    tools=[data_analyzer],
    system_prompt="Analyze data and provide insights."
)

async def parallel_analysis(topic: str):
    """Chạy nhiều agents song song."""
    research_task = asyncio.create_task(
        research_agent.ainvoke({
            "messages": [{"role": "user", "content": f"Research: {topic}"}]
        })
    )
    analysis_task = asyncio.create_task(
        analysis_agent.ainvoke({
            "messages": [{"role": "user", "content": f"Analyze trends in: {topic}"}]
        })
    )

    research_result, analysis_result = await asyncio.gather(
        research_task, analysis_task
    )

    return {
        "research": research_result["messages"][-1].content,
        "analysis": analysis_result["messages"][-1].content
    }
```

### Mẫu 3: Pipeline Agents

Các agents xử lý dữ liệu tuần tự, mỗi agent xây dựng trên đầu ra trước đó.

```python
# Agent 1: Nghiên cứu
research_agent = create_agent(
    model="gpt-4o",
    tools=[web_search],
    system_prompt="Research topics and gather information."
)

# Agent 2: Tóm tắt
summary_agent = create_agent(
    model="gpt-4o",
    tools=[],
    system_prompt="Summarize research into key points."
)

# Agent 3: Viết
writer_agent = create_agent(
    model="gpt-4o",
    tools=[],
    system_prompt="Write polished content from summaries."
)

def pipeline(topic: str):
    # Bước 1: Nghiên cứu
    research = research_agent.invoke({
        "messages": [{"role": "user", "content": f"Research: {topic}"}]
    })

    # Bước 2: Tóm tắt
    summary = summary_agent.invoke({
        "messages": [{
            "role": "user",
            "content": f"Summarize this research: {research['messages'][-1].content}"
        }]
    })

    # Bước 3: Viết
    final = writer_agent.invoke({
        "messages": [{
            "role": "user",
            "content": f"Write an article from: {summary['messages'][-1].content}"
        }]
    })

    return final["messages"][-1].content
```

---

## Chuỗi Tuần Tự

Liên kết nhiều bước xử lý trong đó đầu ra chảy vào đầu vào.

### Chuỗi Tuần Tự Cơ bản

```python
from langchain_classic.chains import SequentialChain
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# Chain 1: Tạo dàn ý
outline_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Create an outline for an article about: {topic}"
)
outline_chain = LLMChain(llm=llm, prompt=outline_prompt, output_key="outline")

# Chain 2: Viết từ dàn ý
write_prompt = PromptTemplate(
    input_variables=["outline"],
    template="Write an article based on this outline:\n{outline}"
)
write_chain = LLMChain(llm=llm, prompt=write_prompt, output_key="article")

# Kết hợp các chains
sequential_chain = SequentialChain(
    chains=[outline_chain, write_chain],
    input_variables=["topic"],
    output_variables=["outline", "article"]
)

result = sequential_chain.invoke({"topic": "AI Agents"})
print(result["article"])
```

### Mẫu Tuần Tự LangGraph

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    topic: str
    outline: str
    article: str

def generate_outline(state: State):
    result = llm.invoke(f"Create outline for: {state['topic']}")
    return {"outline": result.content}

def write_article(state: State):
    result = llm.invoke(f"Write article from: {state['outline']}")
    return {"article": result.content}

# Xây dựng graph
builder = StateGraph(State)
builder.add_node("outline", generate_outline)
builder.add_node("write", write_article)
builder.add_edge(START, "outline")
builder.add_edge("outline", "write")
builder.add_edge("write", END)

chain = builder.compile()
result = chain.invoke({"topic": "AI Agents"})
```

---

## Mẫu Router

Định tuyến các yêu cầu đến các handlers khác nhau dựa trên nội dung hoặc ngữ cảnh.

### Router Dựa trên LLM

```python
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from typing import Literal

# Schema cho quyết định định tuyến
class Route(BaseModel):
    destination: Literal["story", "joke", "poem"]

# Router LLM với structured output
router = llm.with_structured_output(Route)

class State(TypedDict):
    input: str
    decision: str
    output: str

def route_input(state: State):
    """Xác định handler nào để sử dụng."""
    decision = router.invoke([
        {"role": "system", "content": "Route to story, joke, or poem based on request."},
        {"role": "user", "content": state["input"]}
    ])
    return {"decision": decision.destination}

def write_story(state: State):
    result = llm.invoke(f"Write a story about: {state['input']}")
    return {"output": result.content}

def write_joke(state: State):
    result = llm.invoke(f"Write a joke about: {state['input']}")
    return {"output": result.content}

def write_poem(state: State):
    result = llm.invoke(f"Write a poem about: {state['input']}")
    return {"output": result.content}

def route_decision(state: State):
    """Trả về node để truy cập dựa trên quyết định."""
    return f"write_{state['decision']}"

# Xây dựng router graph
builder = StateGraph(State)
builder.add_node("router", route_input)
builder.add_node("write_story", write_story)
builder.add_node("write_joke", write_joke)
builder.add_node("write_poem", write_poem)

builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    route_decision,
    {
        "write_story": "write_story",
        "write_joke": "write_joke",
        "write_poem": "write_poem"
    }
)
builder.add_edge("write_story", END)
builder.add_edge("write_joke", END)
builder.add_edge("write_poem", END)

router_graph = builder.compile()
```

### Router Knowledge Base

Định tuyến câu hỏi đến các nguồn kiến thức phù hợp:

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def search_api_docs(query: str) -> str:
    """Search API documentation."""
    # Tìm kiếm API docs vector store
    return "API documentation results..."

@tool
def search_tutorials(query: str) -> str:
    """Search tutorial content."""
    return "Tutorial results..."

@tool
def search_faq(query: str) -> str:
    """Search frequently asked questions."""
    return "FAQ results..."

# Router agent chọn nguồn phù hợp
router_agent = create_agent(
    model="gpt-4o",
    tools=[search_api_docs, search_tutorials, search_faq],
    system_prompt="""You are a documentation assistant.
    Route questions to the most appropriate knowledge source:
    - API docs: for technical API questions
    - Tutorials: for how-to guides and examples
    - FAQ: for common questions and troubleshooting"""
)
```

---

## Mẫu Plan-and-Execute

Chia các nhiệm vụ phức tạp thành các nhiệm vụ con, sau đó thực thi từng nhiệm vụ.

### LangGraph Plan-Execute

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from pydantic import BaseModel

class Step(BaseModel):
    description: str
    completed: bool = False

class PlanExecuteState(TypedDict):
    objective: str
    plan: List[Step]
    current_step: int
    results: List[str]
    final_answer: str

def create_plan(state: PlanExecuteState):
    """Tạo kế hoạch thực thi."""
    planner = llm.with_structured_output(List[Step])
    plan = planner.invoke([
        {"role": "system", "content": "Break this objective into 3-5 steps."},
        {"role": "user", "content": state["objective"]}
    ])
    return {"plan": plan, "current_step": 0, "results": []}

def execute_step(state: PlanExecuteState):
    """Thực thi bước hiện tại."""
    step = state["plan"][state["current_step"]]
    context = "\n".join(state["results"])

    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": f"Context: {context}\n\nExecute: {step.description}"
        }]
    })

    new_results = state["results"] + [result["messages"][-1].content]
    return {
        "results": new_results,
        "current_step": state["current_step"] + 1
    }

def synthesize(state: PlanExecuteState):
    """Tổng hợp câu trả lời cuối cùng từ các kết quả."""
    result = llm.invoke([
        {"role": "system", "content": "Synthesize these results into a final answer."},
        {"role": "user", "content": "\n\n".join(state["results"])}
    ])
    return {"final_answer": result.content}

def should_continue(state: PlanExecuteState):
    """Kiểm tra xem còn bước nào không."""
    if state["current_step"] >= len(state["plan"]):
        return "synthesize"
    return "execute"

# Xây dựng plan-execute graph
builder = StateGraph(PlanExecuteState)
builder.add_node("plan", create_plan)
builder.add_node("execute", execute_step)
builder.add_node("synthesize", synthesize)

builder.add_edge(START, "plan")
builder.add_edge("plan", "execute")
builder.add_conditional_edges(
    "execute",
    should_continue,
    {"execute": "execute", "synthesize": "synthesize"}
)
builder.add_edge("synthesize", END)

plan_execute = builder.compile()
```

---

## Mẫu Human-in-the-Loop

Tạm dừng thực thi để con người xem xét, phê duyệt hoặc nhập liệu.

### Luồng Phê Duyệt Cơ bản

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal, Optional, TypedDict

class ApprovalState(TypedDict):
    action_details: str
    status: Optional[Literal["pending", "approved", "rejected"]]

def approval_node(state: ApprovalState) -> Command[Literal["proceed", "cancel"]]:
    """Tạm dừng để phê duyệt của con người."""
    decision = interrupt({
        "question": "Approve this action?",
        "details": state["action_details"],
    })

    # Định tuyến dựa trên quyết định của con người
    return Command(goto="proceed" if decision else "cancel")

def proceed_node(state: ApprovalState):
    return {"status": "approved"}

def cancel_node(state: ApprovalState):
    return {"status": "rejected"}

# Xây dựng approval graph
builder = StateGraph(ApprovalState)
builder.add_node("approval", approval_node)
builder.add_node("proceed", proceed_node)
builder.add_node("cancel", cancel_node)
builder.add_edge(START, "approval")
builder.add_edge("proceed", END)
builder.add_edge("cancel", END)

checkpointer = MemorySaver()
approval_graph = builder.compile(checkpointer=checkpointer)

# Thực thi - tạm dừng tại phê duyệt
config = {"configurable": {"thread_id": "approval-123"}}
initial = approval_graph.invoke(
    {"action_details": "Transfer $500", "status": "pending"},
    config=config
)
print(initial["__interrupt__"])  # Hiển thị yêu cầu phê duyệt

# Tiếp tục với quyết định
resumed = approval_graph.invoke(Command(resume=True), config=config)
print(resumed["status"])  # "approved"
```

### Luồng Xem Xét và Chỉnh Sửa

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

class ReviewState(TypedDict):
    generated_text: str

def review_node(state: ReviewState):
    """Tạm dừng để con người xem xét và chỉnh sửa."""
    updated = interrupt({
        "instruction": "Review and edit this content",
        "content": state["generated_text"]
    })
    return {"generated_text": updated}

builder = StateGraph(ReviewState)
builder.add_node("review", review_node)
builder.add_edge(START, "review")
builder.add_edge("review", END)

checkpointer = MemorySaver()
review_graph = builder.compile(checkpointer=checkpointer)

# Bắt đầu với bản nháp
config = {"configurable": {"thread_id": "review-42"}}
initial = review_graph.invoke({"generated_text": "Initial draft"}, config)
# Hiển thị interrupt với nội dung để xem xét

# Tiếp tục với văn bản đã chỉnh sửa
final = review_graph.invoke(
    Command(resume="Improved draft after review"),
    config
)
print(final["generated_text"])  # "Improved draft after review"
```

### Agent với Phê Duyệt Trước Khi Hành Động

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

agent = create_agent(
    model="gpt-4o",
    tools=[dangerous_action, safe_action],
    checkpointer=checkpointer,
    interrupt_before=["dangerous_action"]  # Tạm dừng trước tool này
)

config = {"configurable": {"thread_id": "agent-123"}}

# Agent tạm dừng trước khi gọi dangerous_action
result = agent.invoke({
    "messages": [{"role": "user", "content": "Delete all files"}]
}, config)

# Xem xét hành động đang chờ
print(result["__interrupt__"])  # Hiển thị lời gọi tool để xem xét

# Phê duyệt và tiếp tục
result = agent.invoke(Command(resume=True), config)
```

---

## Mẫu Workflow với LangGraph

### State Graph với Retry

```python
from langgraph.types import RetryPolicy
from langgraph.graph import StateGraph

workflow = StateGraph(State)

# Thêm node với retry policy
workflow.add_node(
    "unreliable_api",
    call_api,
    retry_policy=RetryPolicy(max_attempts=3)
)
```

### Mẫu Khôi Phục Lỗi

```python
from langgraph.types import Command

def execute_with_recovery(state: State) -> Command[Literal["agent", "recover"]]:
    try:
        result = run_tool(state['tool_call'])
        return Command(update={"result": result}, goto="agent")
    except ToolError as e:
        # Cho agent thấy lỗi và điều chỉnh
        return Command(
            update={"result": f"Error: {str(e)}"},
            goto="agent"
        )
```

### Mẫu Thực Thi Song Song

```python
from langgraph.graph import StateGraph
import asyncio

class ParallelState(TypedDict):
    query: str
    results: dict

async def parallel_search(state: ParallelState):
    """Chạy nhiều tìm kiếm song song."""
    async def search_source(name, query):
        # Mô phỏng async search
        return f"{name} results for: {query}"

    results = await asyncio.gather(
        search_source("web", state["query"]),
        search_source("docs", state["query"]),
        search_source("code", state["query"])
    )

    return {"results": dict(zip(["web", "docs", "code"], results))}
```

---

## Hướng Dẫn Chọn Mẫu

| Mẫu | Trường hợp Sử dụng | Độ phức tạp | Khả năng Mở rộng |
|---------|----------|------------|-------------|
| Single Agent | Nhiệm vụ đơn giản, ít tools | Thấp | Hạn chế |
| Supervisor | Nhiệm vụ đa miền | Trung bình | Tốt |
| Parallel | Nhiệm vụ con độc lập | Trung bình | Xuất sắc |
| Pipeline | Xử lý tuần tự | Thấp | Tốt |
| Router | Phân phối dựa trên nội dung | Thấp | Xuất sắc |
| Plan-Execute | Nhiều bước phức tạp | Cao | Tốt |
| Human-in-Loop | Quyết định có rủi ro cao | Trung bình | Hạn chế |

---

## Bước Tiếp Theo

- [Ví dụ Code](./examples.md) - Các ví dụ triển khai hoàn chỉnh
- [Kiến trúc](./architecture.md) - Chi tiết kiến trúc cốt lõi
- [Thành phần](./components.md) - Tham khảo thành phần
