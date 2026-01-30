# Các Mẫu Thiết Kế Đa Agent

Tài liệu này đề cập đến các mẫu phối hợp cho các hệ thống có nhiều AI agent làm việc cùng nhau.

## 1. Mẫu Phân Cấp (Supervisor/Worker)

### Tổng Quan

Trong các mẫu phân cấp, một agent supervisor điều phối và giao nhiệm vụ cho các agent worker. Supervisor:
- Nhận tác vụ ban đầu
- Chia nhỏ thành các tác vụ con
- Giao cho các worker chuyên biệt
- Tổng hợp kết quả

### Kiến Trúc

```
                    ┌──────────────┐
                    │  Supervisor  │
                    │    Agent     │
                    └──────┬───────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Worker 1 │    │ Worker 2 │    │ Worker 3 │
    │(Nghiên cứu)│   │(Viết lách)│   │(Phân tích)│
    └──────────┘    └──────────┘    └──────────┘
```

### Triển Khai LangGraph

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph_supervisor import create_supervisor

# Định nghĩa các agent worker
research_agent = create_react_agent(
    model=model,
    tools=[search_tool, wiki_tool],
    name="researcher"
)

math_agent = create_react_agent(
    model=model,
    tools=[calculator_tool],
    name="mathematician"
)

# Tạo workflow supervisor
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt="Bạn là supervisor nhóm quản lý chuyên gia nghiên cứu và chuyên gia toán."
)

# Thêm bộ nhớ để lưu trữ
checkpointer = InMemorySaver()
store = InMemoryStore()

app = workflow.compile(
    checkpointer=checkpointer,
    store=store
)
```

### Triển Khai CrewAI

```python
from crewai import Agent, Crew, Task, Process

# Agent quản lý
manager = Agent(
    role="Quản Lý Dự Án",
    goal="Điều phối nỗ lực nhóm và đảm bảo thành công dự án",
    backstory="Quản lý dự án có kinh nghiệm giỏi phân công công việc",
    allow_delegation=True,
    verbose=True
)

# Các agent chuyên gia
researcher = Agent(
    role="Nhà Nghiên Cứu",
    goal="Cung cấp nghiên cứu và phân tích chính xác",
    backstory="Nhà nghiên cứu chuyên gia với kỹ năng phân tích sâu",
    allow_delegation=False
)

writer = Agent(
    role="Người Viết",
    goal="Tạo nội dung hấp dẫn",
    backstory="Người viết có kỹ năng tạo nội dung thu hút",
    allow_delegation=False
)

# Tạo crew phân cấp
crew = Crew(
    agents=[manager, researcher, writer],
    tasks=[project_task],
    process=Process.hierarchical,
    manager_llm="gpt-4o",
    verbose=True
)
```

### Phân Cấp Nhiều Cấp

```python
from langgraph_supervisor import create_supervisor

# Tạo supervisor cấp nhóm
research_team = create_supervisor(
    [research_agent, math_agent],
    model=model,
    supervisor_name="research_supervisor"
).compile(name="research_team")

writing_team = create_supervisor(
    [writing_agent, publishing_agent],
    model=model,
    supervisor_name="writing_supervisor"
).compile(name="writing_team")

# Tạo supervisor cấp cao nhất
top_level_supervisor = create_supervisor(
    [research_team, writing_team],
    model=model,
    supervisor_name="top_level_supervisor"
).compile(name="top_level_supervisor")
```

### Thực Tiễn Tốt Nhất

1. **Định Nghĩa Vai Trò Rõ Ràng**: Mỗi worker nên có vai trò cụ thể, được định nghĩa rõ
2. **Phạm Vi Giới Hạn**: Worker nên tập trung vào chuyên môn của họ
3. **Phân Công Rõ Ràng**: Supervisor nên nêu rõ worker nào được sử dụng
4. **Tổng Hợp Kết Quả**: Supervisor nên tổng hợp đầu ra của worker

---

## 2. Mẫu Cộng Tác (Ngang Hàng)

### Tổng Quan

Các mẫu cộng tác cho phép các agent làm việc cùng nhau như các đối tác ngang hàng, chia sẻ thông tin và xây dựng dựa trên công việc của nhau mà không có người điều phối trung tâm.

### Kiến Trúc Swarm

```
    ┌──────────┐     ┌──────────┐
    │  Agent A │◄───►│  Agent B │
    └────┬─────┘     └────┬─────┘
         │                │
         │    ┌──────────┐│
         └───►│  Agent C │◄┘
              └──────────┘
```

### Triển Khai LangGraph Swarm

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph_swarm import create_swarm

# Tạo agent với khả năng chuyển giao
alice = create_react_agent(
    model=model,
    tools=[search_tool, create_handoff_tool(bob)],
    name="Alice"
)

bob = create_react_agent(
    model=model,
    tools=[analysis_tool, create_handoff_tool(alice)],
    name="Bob"
)

# Tạo swarm
workflow = create_swarm(
    [alice, bob],
    default_active_agent="Alice"
)

# Biên dịch với bộ nhớ
checkpointer = InMemorySaver()
store = InMemoryStore()

app = workflow.compile(
    checkpointer=checkpointer,
    store=store
)
```

### Mẫu Hội Thoại AutoGen

```python
from autogen_core import (
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler
)

@dataclass
class Message:
    content: str
    sender: str

class CollaborativeAgent(RoutedAgent):
    def __init__(self, model_client, neighbors):
        super().__init__("Agent cộng tác")
        self._model_client = model_client
        self._neighbors = neighbors
        self._history = []

    @message_handler
    async def handle_message(self, message: Message, ctx):
        # Xử lý tin nhắn
        self._history.append(message)

        # Tạo phản hồi
        response = await self._model_client.create(
            self._system_messages + self._history
        )

        # Chia sẻ với hàng xóm nếu cần
        for neighbor in self._neighbors:
            await self.send_message(
                Message(content=response, sender=self.id),
                neighbor
            )

        return response
```

### Các Mẫu Cộng Tác

| Mẫu | Mô Tả | Trường Hợp Sử Dụng |
|-----|-------|-------------------|
| Round-Robin | Agent lần lượt thay phiên | Tinh chỉnh tuần tự |
| Broadcast | Tất cả agent nhận tất cả tin nhắn | Xây dựng đồng thuận |
| Handoff | Chuyển giao quyền kiểm soát rõ ràng | Xử lý chuyên biệt |
| Debate | Agent tranh luận các quan điểm | Đưa ra quyết định |

---

## 3. Mẫu Cạnh Tranh

### Tổng Quan

Các mẫu cạnh tranh có nhiều agent làm việc trên cùng một vấn đề, với đầu ra của họ được so sánh hoặc tổng hợp.

### Tranh Luận Đa Agent

```python
from autogen_core import RoutedAgent, default_subscription

@dataclass
class SolverRequest:
    content: str
    question: str

@dataclass
class IntermediateSolverResponse:
    content: str
    question: str
    answer: str
    round: int

@default_subscription
class DebateSolver(RoutedAgent):
    def __init__(self, model_client, num_neighbors, max_rounds):
        super().__init__("Bộ giải tranh luận")
        self._model_client = model_client
        self._num_neighbors = num_neighbors
        self._max_rounds = max_rounds
        self._history = []
        self._buffer = {}
        self._round = 0

    @message_handler
    async def handle_request(self, message: SolverRequest, ctx):
        # Tạo giải pháp
        self._history.append(UserMessage(content=message.content))
        response = await self._model_client.create(
            self._system_messages + self._history
        )

        # Trích xuất câu trả lời và chia sẻ với hàng xóm
        answer = extract_answer(response)
        self._round += 1

        if self._round < self._max_rounds:
            # Chia sẻ phản hồi trung gian
            await self.publish_message(
                IntermediateSolverResponse(
                    content=response,
                    question=message.question,
                    answer=answer,
                    round=self._round
                )
            )
        else:
            # Xuất bản câu trả lời cuối cùng
            await self.publish_message(FinalResponse(answer=answer))

    @message_handler
    async def handle_neighbor_response(self, message: IntermediateSolverResponse, ctx):
        # Tổng hợp giải pháp hàng xóm
        self._buffer.setdefault(message.round, []).append(message)

        if len(self._buffer[message.round]) == self._num_neighbors:
            # Chuẩn bị prompt tinh chỉnh với giải pháp hàng xóm
            prompt = "Xem xét các giải pháp này từ các agent khác:\n"
            for resp in self._buffer[message.round]:
                prompt += f"Giải pháp: {resp.content}\n"
            prompt += "Cung cấp câu trả lời đã tinh chỉnh của bạn."

            await self.send_message(
                SolverRequest(content=prompt, question=message.question),
                self.id
            )
```

### Bỏ Phiếu Ensemble

```python
class EnsembleCoordinator:
    def __init__(self, agents: List[Agent]):
        self.agents = agents

    async def solve(self, question: str) -> str:
        # Lấy câu trả lời từ tất cả agent
        answers = []
        for agent in self.agents:
            answer = await agent.invoke(question)
            answers.append(answer)

        # Tổng hợp bằng bỏ phiếu
        return self.majority_vote(answers)

    def majority_vote(self, answers: List[str]) -> str:
        # Đếm số lần xuất hiện và trả về phổ biến nhất
        from collections import Counter
        counts = Counter(answers)
        return counts.most_common(1)[0][0]
```

---

## 4. Pipeline Tuần Tự

### Tổng Quan

Pipeline tuần tự chuyển công việc qua một chuỗi agent, mỗi agent thực hiện một biến đổi hoặc phân tích cụ thể.

### Kiến Trúc

```
Đầu vào -> Agent 1 -> Agent 2 -> Agent 3 -> Đầu ra
           (Trích xuất) (Phân tích) (Tóm tắt)
```

### Quy Trình Tuần Tự CrewAI

```python
from crewai import Agent, Crew, Task, Process

# Định nghĩa các agent pipeline
extractor = Agent(
    role="Trích Xuất Dữ Liệu",
    goal="Trích xuất thông tin liên quan từ dữ liệu thô",
    backstory="Chuyên gia trích xuất dữ liệu"
)

analyst = Agent(
    role="Phân Tích Dữ Liệu",
    goal="Phân tích dữ liệu đã trích xuất để tìm insight",
    backstory="Chuyên gia phân tích dữ liệu"
)

reporter = Agent(
    role="Người Viết Báo Cáo",
    goal="Tạo báo cáo rõ ràng, có thể hành động",
    backstory="Người viết kỹ thuật có chuyên môn về dữ liệu"
)

# Định nghĩa các tác vụ tuần tự
extraction_task = Task(
    description="Trích xuất các chỉ số chính từ dữ liệu",
    agent=extractor,
    expected_output="Dữ liệu có cấu trúc với các chỉ số chính"
)

analysis_task = Task(
    description="Phân tích chỉ số và xác định xu hướng",
    agent=analyst,
    expected_output="Phân tích với các xu hướng đã xác định",
    context=[extraction_task]  # Phụ thuộc vào trích xuất
)

report_task = Task(
    description="Tạo báo cáo tóm tắt điều hành",
    agent=reporter,
    expected_output="Tóm tắt điều hành với khuyến nghị",
    context=[analysis_task]  # Phụ thuộc vào phân tích
)

# Tạo crew tuần tự
crew = Crew(
    agents=[extractor, analyst, reporter],
    tasks=[extraction_task, analysis_task, report_task],
    process=Process.sequential
)
```

### Pipeline LangGraph

```python
from langgraph.graph import StateGraph, END

class PipelineState(TypedDict):
    input: str
    extracted_data: str
    analysis: str
    report: str

def extract(state: PipelineState):
    result = extractor_agent.invoke(state["input"])
    return {"extracted_data": result}

def analyze(state: PipelineState):
    result = analyst_agent.invoke(state["extracted_data"])
    return {"analysis": result}

def report(state: PipelineState):
    result = reporter_agent.invoke(state["analysis"])
    return {"report": result}

workflow = StateGraph(PipelineState)
workflow.add_node("extract", extract)
workflow.add_node("analyze", analyze)
workflow.add_node("report", report)

workflow.add_edge("extract", "analyze")
workflow.add_edge("analyze", "report")
workflow.add_edge("report", END)

workflow.set_entry_point("extract")
app = workflow.compile()
```

---

## 5. Mẫu Router/Dispatcher

### Tổng Quan

Mẫu Router sử dụng bộ phân phối trung tâm để định tuyến yêu cầu đến agent chuyên biệt phù hợp dựa trên loại yêu cầu.

### Kiến Trúc

```
                    ┌──────────────┐
                    │    Router    │
                    │  (Phân loại) │
                    └──────┬───────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │   Tech   │    │   Sales  │    │ Support  │
    │  Agent   │    │  Agent   │    │  Agent   │
    └──────────┘    └──────────┘    └──────────┘
```

### Triển Khai

```python
from langgraph.graph import StateGraph, END

class RouterState(TypedDict):
    input: str
    category: str
    response: str

def router(state: RouterState):
    """Phân loại yêu cầu và xác định định tuyến."""
    classification = llm.invoke(f"""
    Phân loại yêu cầu này thành một trong: tech, sales, support

    Yêu cầu: {state['input']}

    Chỉ trả về tên danh mục.
    """)
    return {"category": classification.strip().lower()}

def route_to_agent(state: RouterState) -> str:
    """Định tuyến đến agent phù hợp dựa trên danh mục."""
    category = state["category"]
    if category == "tech":
        return "tech_agent"
    elif category == "sales":
        return "sales_agent"
    else:
        return "support_agent"

def tech_agent(state: RouterState):
    response = tech_specialist.invoke(state["input"])
    return {"response": response}

def sales_agent(state: RouterState):
    response = sales_specialist.invoke(state["input"])
    return {"response": response}

def support_agent(state: RouterState):
    response = support_specialist.invoke(state["input"])
    return {"response": response}

workflow = StateGraph(RouterState)
workflow.add_node("router", router)
workflow.add_node("tech_agent", tech_agent)
workflow.add_node("sales_agent", sales_agent)
workflow.add_node("support_agent", support_agent)

workflow.add_conditional_edges("router", route_to_agent)
workflow.add_edge("tech_agent", END)
workflow.add_edge("sales_agent", END)
workflow.add_edge("support_agent", END)

workflow.set_entry_point("router")
app = workflow.compile()
```

### Định Tuyến Ngữ Nghĩa

```python
from langchain_core.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

class SemanticRouter:
    def __init__(self, routes: Dict[str, Agent]):
        self.routes = routes
        self.embeddings = OpenAIEmbeddings()
        self.route_embeddings = {}

        # Tính toán trước embedding cho các route
        for name, agent in routes.items():
            description = f"{agent.role}: {agent.goal}"
            self.route_embeddings[name] = self.embeddings.embed_query(description)

    def route(self, query: str) -> str:
        query_embedding = self.embeddings.embed_query(query)

        best_route = None
        best_score = -1

        for name, route_embedding in self.route_embeddings.items():
            score = cosine_similarity([query_embedding], [route_embedding])[0][0]
            if score > best_score:
                best_score = score
                best_route = name

        return best_route
```

---

## So Sánh Mẫu

| Mẫu | Phối Hợp | Khả Năng Mở Rộng | Độ Phức Tạp | Trường Hợp Sử Dụng |
|-----|----------|------------------|-------------|-------------------|
| Phân cấp | Tập trung | Trung bình | Trung bình | Nhóm có cấu trúc |
| Cộng tác | Phân tán | Cao | Cao | Tác vụ sáng tạo |
| Cạnh tranh | Song song | Cao | Thấp | Yêu cầu độ chính xác |
| Tuần tự | Tuyến tính | Thấp | Thấp | Xử lý pipeline |
| Router | Tập trung | Cao | Trung bình | Định tuyến yêu cầu |

## So Sánh Framework

| Mẫu | LangGraph | CrewAI | AutoGen |
|-----|-----------|--------|---------|
| Phân cấp | Thư viện Supervisor | Process.hierarchical | Tùy chỉnh |
| Cộng tác | Thư viện Swarm | allow_delegation | Conversation |
| Cạnh tranh | Tùy chỉnh | Tùy chỉnh | Tranh luận đa agent |
| Tuần tự | Graph edges | Process.sequential | Tùy chỉnh |
| Router | Conditional edges | Tùy chỉnh | Topic routing |

## Thực Tiễn Tốt Nhất

### 1. Thiết Kế Agent
- Cho mỗi agent một vai trò rõ ràng, tập trung
- Giới hạn số công cụ mỗi agent
- Sử dụng tên và backstory mô tả

### 2. Giao Tiếp
- Định nghĩa giao thức tin nhắn rõ ràng
- Giới hạn kích thước tin nhắn để giảm ngữ cảnh
- Sử dụng định dạng có cấu trúc (JSON) để trao đổi dữ liệu

### 3. Phối Hợp
- Đặt điều kiện kết thúc rõ ràng
- Triển khai timeout cho tác vụ chạy dài
- Thêm handler dự phòng cho lỗi agent

### 4. Khả Năng Mở Rộng
- Sử dụng mẫu async để thực thi song song
- Triển khai pool agent cho các tình huống tải cao
- Xem xét hàng đợi tin nhắn cho các hệ thống lớn
