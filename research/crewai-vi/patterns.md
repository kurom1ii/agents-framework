# Patterns và Best Practices CrewAI

Tài liệu này bao gồm các pattern cộng tác, pattern thiết kế và best practices để xây dựng hệ thống CrewAI hiệu quả.

## Mục lục

1. [Patterns cộng tác Agent](#patterns-cộng-tác-agent)
2. [Patterns luồng Task](#patterns-luồng-task)
3. [Patterns tổ chức Crew](#patterns-tổ-chức-crew)
4. [Patterns Flow](#patterns-flow)
5. [Best Practices thiết kế](#best-practices-thiết-kế)
6. [Tối ưu hóa hiệu suất](#tối-ưu-hóa-hiệu-suất)
7. [Xử lý lỗi](#xử-lý-lỗi)
8. [Chiến lược Testing](#chiến-lược-testing)

---

## Patterns cộng tác Agent

### 1. Pattern ủy thác (Delegation)

Cho phép agent ủy thác các task con cho agent khác.

```python
from crewai import Agent

# Lead agent với delegation được bật
project_lead = Agent(
    role="Project Lead",
    goal="Coordinate project deliverables",
    backstory="Senior manager with 10 years experience",
    allow_delegation=True,  # Có thể ủy thác cho đồng đội
    verbose=True
)

# Các agent chuyên gia (không delegation)
researcher = Agent(
    role="Research Specialist",
    goal="Gather accurate data",
    backstory="Expert in data collection",
    allow_delegation=False  # Tập trung vào task của mình
)

writer = Agent(
    role="Content Writer",
    goal="Create compelling content",
    backstory="Award-winning writer",
    allow_delegation=False
)
```

**Khi `allow_delegation=True`**, agent có quyền truy cập:
- **Tool Delegate**: `Delegate work to coworker(task, context, coworker)`
- **Tool Ask**: `Ask question to coworker(question, context, coworker)`

### 2. Pipeline Nghiên cứu-Viết-Chỉnh sửa

Pattern tạo nội dung kinh điển với các chuyển giao rõ ràng.

```python
from crewai import Agent, Task, Crew, Process

# Agents
researcher = Agent(
    role="Senior Researcher",
    goal="Conduct thorough research on topics",
    backstory="Expert at finding and synthesizing information"
)

writer = Agent(
    role="Content Writer",
    goal="Transform research into engaging content",
    backstory="Skilled at making complex topics accessible"
)

editor = Agent(
    role="Senior Editor",
    goal="Polish content to publication quality",
    backstory="Meticulous editor with high standards"
)

# Tasks với luồng context
research_task = Task(
    description="Research {topic} comprehensively",
    expected_output="Detailed research notes with sources",
    agent=researcher
)

writing_task = Task(
    description="Write an article based on research findings",
    expected_output="Draft article in markdown format",
    agent=writer,
    context=[research_task]  # Nhận output từ research
)

editing_task = Task(
    description="Edit and polish the article",
    expected_output="Publication-ready article",
    agent=editor,
    context=[writing_task]  # Nhận bản thảo
)

crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential
)
```

### 3. Task đơn cộng tác

Nhiều agent đóng góp vào một task phức tạp đơn lẻ.

```python
lead_agent = Agent(
    role="Lead Analyst",
    goal="Produce comprehensive market analysis",
    backstory="Senior analyst coordinating team efforts",
    allow_delegation=True
)

market_expert = Agent(
    role="Market Expert",
    goal="Provide market-specific insights",
    backstory="Specialist in market dynamics",
    allow_delegation=False
)

data_analyst = Agent(
    role="Data Analyst",
    goal="Analyze quantitative data",
    backstory="Expert in statistical analysis",
    allow_delegation=False
)

# Task phức tạp đơn lẻ
analysis_task = Task(
    description="""
    Create a comprehensive market analysis report including:
    - Market size and growth trends
    - Competitive landscape
    - Statistical projections
    Delegate specific analyses to team members as needed.
    """,
    expected_output="Complete market analysis report",
    agent=lead_agent  # Lead điều phối, ủy thác cho những người khác
)
```

### 4. Quản lý phân cấp (Hierarchical)

Manager điều phối các chuyên gia cho công việc phức tạp.

```python
from crewai import Crew, Process

# Các agent chuyên gia
specialists = [
    Agent(role="Frontend Developer", goal="Build UI", backstory="..."),
    Agent(role="Backend Developer", goal="Build API", backstory="..."),
    Agent(role="QA Engineer", goal="Test quality", backstory="...")
]

# Manager phân công dựa trên chuyên môn
crew = Crew(
    agents=specialists,
    tasks=[feature_task, testing_task],
    process=Process.hierarchical,
    manager_llm="openai/gpt-4o"
)
```

---

## Patterns luồng Task

### 1. Dependencies tuần tự

Mỗi task xây dựng trên output của task trước.

```python
task1 = Task(
    description="Gather requirements",
    expected_output="Requirements document",
    agent=analyst
)

task2 = Task(
    description="Design solution based on requirements",
    expected_output="Design document",
    agent=architect,
    context=[task1]  # Phụ thuộc vào task1
)

task3 = Task(
    description="Implement the designed solution",
    expected_output="Working code",
    agent=developer,
    context=[task2]  # Phụ thuộc vào task2
)
```

### 2. Thực thi song song

Các task độc lập chạy đồng thời.

```python
# Các task này có thể chạy song song
research_web = Task(
    description="Research web sources",
    expected_output="Web research notes",
    agent=web_researcher,
    async_execution=True
)

research_papers = Task(
    description="Research academic papers",
    expected_output="Academic research notes",
    agent=academic_researcher,
    async_execution=True
)

# Task này chờ cả hai task song song
synthesis = Task(
    description="Synthesize all research findings",
    expected_output="Comprehensive research summary",
    agent=synthesizer,
    context=[research_web, research_papers]  # Chờ cả hai
)
```

### 3. Luồng Task có điều kiện

Sử dụng guardrails để điều khiển luồng.

```python
def quality_check(output):
    """Guardrail xác thực chất lượng output"""
    if "error" in output.raw.lower():
        return (False, "Output contains errors, please fix")
    if len(output.raw) < 500:
        return (False, "Output too short, please elaborate")
    return (True, output)

task = Task(
    description="Generate detailed analysis",
    expected_output="Comprehensive analysis",
    agent=analyst,
    guardrails=[quality_check],
    guardrail_max_retries=3
)
```

### 4. Human-in-the-Loop

Yêu cầu phê duyệt của con người cho các task quan trọng.

```python
approval_task = Task(
    description="Generate contract terms",
    expected_output="Legal contract draft",
    agent=legal_agent,
    human_input=True  # Yêu cầu đánh giá của con người
)
```

---

## Patterns tổ chức Crew

### 1. Crew đơn mục đích

Crew tập trung cho các domain cụ thể.

```python
# Crew nghiên cứu
research_crew = Crew(
    agents=[researcher, fact_checker],
    tasks=[research_task, verification_task],
    process=Process.sequential
)

# Crew nội dung
content_crew = Crew(
    agents=[writer, editor],
    tasks=[writing_task, editing_task],
    process=Process.sequential
)

# Sử dụng trong flow để điều phối
class ContentPipeline(Flow):
    @start()
    def research_phase(self):
        return research_crew.kickoff(inputs={...})

    @listen(research_phase)
    def content_phase(self, research):
        return content_crew.kickoff(inputs={"research": research.raw})
```

### 2. Crew có bật Memory

Crew học và ghi nhớ.

```python
learning_crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    memory=True,  # Bật tất cả loại memory
    verbose=True
)

# Nhiều lần chạy sẽ xây dựng memory
result1 = learning_crew.kickoff(inputs={"topic": "AI"})
result2 = learning_crew.kickoff(inputs={"topic": "ML"})  # Được lợi từ context trước
```

### 3. Crew tăng cường Knowledge

Crew với knowledge chuyên ngành.

```python
from crewai.knowledge.source import PDFKnowledgeSource

product_docs = PDFKnowledgeSource(file_path="product_manual.pdf")

support_crew = Crew(
    agents=[support_agent, escalation_agent],
    tasks=[triage_task, resolution_task],
    knowledge_sources=[product_docs]  # Knowledge chia sẻ
)
```

---

## Patterns Flow

### 1. Flow tuyến tính

Thực thi tuần tự đơn giản.

```python
from crewai.flow.flow import Flow, start, listen

class LinearFlow(Flow):
    @start()
    def step_one(self):
        return "Step 1 complete"

    @listen(step_one)
    def step_two(self, result):
        return f"Step 2 received: {result}"

    @listen(step_two)
    def step_three(self, result):
        return f"Final: {result}"
```

### 2. Flow phân nhánh

Đường thực thi có điều kiện.

```python
from crewai.flow.flow import Flow, start, listen, router

class BranchingFlow(Flow):
    @start()
    def classify_input(self):
        # Phân tích input
        return {"type": "urgent", "score": 0.9}

    @router(classify_input)
    def route_by_type(self, classification):
        if classification["type"] == "urgent":
            return "urgent_path"
        else:
            return "normal_path"

    @listen("urgent_path")
    def handle_urgent(self):
        return "Urgent handling complete"

    @listen("normal_path")
    def handle_normal(self):
        return "Normal handling complete"
```

### 3. Flow Fan-Out/Fan-In

Xử lý song song với tổng hợp.

```python
from crewai.flow.flow import Flow, start, listen, and_

class ParallelFlow(Flow):
    @start()
    def distribute(self):
        return "Work to distribute"

    @listen(distribute)
    def process_a(self, work):
        return f"A processed: {work}"

    @listen(distribute)
    def process_b(self, work):
        return f"B processed: {work}"

    @listen(distribute)
    def process_c(self, work):
        return f"C processed: {work}"

    @listen(and_(process_a, process_b, process_c))
    def aggregate(self, results):
        return f"Aggregated: {results}"
```

### 4. Vòng lặp tự đánh giá

Pattern cải tiến lặp đi lặp lại.

```python
class SelfEvalFlow(Flow):
    def __init__(self):
        super().__init__()
        self.max_iterations = 3
        self.iteration = 0

    @start()
    def generate_content(self):
        self.iteration += 1
        return "Generated content..."

    @router(generate_content)
    def evaluate(self, content):
        quality_score = self._assess_quality(content)
        if quality_score > 0.8 or self.iteration >= self.max_iterations:
            return "accept"
        return "revise"

    @listen("revise")
    def improve_content(self):
        # Kích hoạt lại generation
        return self.generate_content()

    @listen("accept")
    def finalize(self, content):
        return f"Final: {content}"

    def _assess_quality(self, content):
        # Triển khai đánh giá chất lượng
        return 0.7
```

### 5. Flow phản hồi con người

Đánh giá con người tương tác.

```python
from crewai.flow.flow import Flow, start, listen, human_feedback

class ReviewFlow(Flow):
    @start()
    def generate_draft(self):
        return "Draft document..."

    @human_feedback(
        outcomes=["approve", "revise", "reject"],
        prompt="Review the draft and decide:"
    )
    @listen(generate_draft)
    def review_step(self, draft):
        return draft

    @listen("approve")
    def publish(self, result):
        return "Published!"

    @listen("revise")
    def revise_draft(self, result):
        return "Revised draft..."

    @listen("reject")
    def archive(self, result):
        return "Archived"
```

---

## Best Practices thiết kế

### Thiết kế Agent

1. **Định nghĩa Role rõ ràng**
   ```python
   # Tốt: Role cụ thể
   role="Senior Python Backend Developer"

   # Không tốt: Role mơ hồ
   role="Developer"
   ```

2. **Goal có thể hành động**
   ```python
   # Tốt: Goal đo lường được
   goal="Reduce API response time to under 200ms"

   # Không tốt: Goal mơ hồ
   goal="Make things faster"
   ```

3. **Backstory phong phú**
   ```python
   backstory="""You are a 10-year veteran at a Fortune 500 tech company.
   You've led teams building high-scale distributed systems.
   You prioritize clean code, thorough testing, and documentation."""
   ```

4. **Chọn Tool phù hợp**
   ```python
   # Chỉ cấp tools mà agent cần
   researcher = Agent(
       role="Web Researcher",
       tools=[SerperDevTool(), WebsiteSearchTool()],  # Tools liên quan
       # Không thêm tools thực thi code cho researcher
   )
   ```

### Thiết kế Task

1. **Mô tả rõ ràng**
   ```python
   description="""
   Analyze the provided customer feedback data and:
   1. Identify top 5 recurring themes
   2. Calculate sentiment scores per theme
   3. Provide actionable recommendations

   Data source: {feedback_file}
   Analysis period: {date_range}
   """
   ```

2. **Output mong đợi cụ thể**
   ```python
   expected_output="""A structured report containing:
   - Executive summary (2-3 sentences)
   - Theme analysis table with sentiment scores
   - Prioritized recommendations (at least 3)
   - Supporting data visualizations (if applicable)
   """
   ```

3. **Sử dụng Structured Outputs**
   ```python
   from pydantic import BaseModel

   class AnalysisReport(BaseModel):
       summary: str
       themes: list[dict]
       recommendations: list[str]
       confidence_score: float

   task = Task(
       description="...",
       expected_output="...",
       output_pydantic=AnalysisReport
   )
   ```

### Thiết kế Crew

1. **Agent bổ sung cho nhau**
   - Mỗi agent nên có vai trò riêng biệt
   - Tránh trùng lặp trách nhiệm
   - Xem xét các điểm chuyển giao

2. **Loại Process phù hợp**
   ```python
   # Sequential: Pipeline rõ ràng, luồng dự đoán được
   Process.sequential

   # Hierarchical: Task phức tạp, cần điều phối
   Process.hierarchical
   ```

3. **Bật Memory để học**
   ```python
   crew = Crew(
       ...,
       memory=True,  # Agent học từ tương tác
       verbose=True   # Cho debugging
   )
   ```

---

## Tối ưu hóa hiệu suất

### 1. Giới hạn tốc độ

```python
# Giới hạn tốc độ cấp crew
crew = Crew(
    agents=[...],
    tasks=[...],
    max_rpm=60  # Tối đa request mỗi phút
)

# Giới hạn tốc độ cấp agent
agent = Agent(
    role="...",
    max_rpm=10  # Giới hạn của agent này
)
```

### 2. Caching

```python
# Bật tool caching (mặc định là True)
agent = Agent(
    role="...",
    cache=True
)

# Hàm cache tùy chỉnh
def should_cache(args, result):
    return len(result) > 100  # Chỉ cache kết quả đáng kể

tool.cache_function = should_cache
```

### 3. Quản lý Token

```python
agent = Agent(
    role="...",
    respect_context_window=True,  # Quản lý context tự động
    max_iter=15  # Giới hạn iterations
)
```

### 4. Thực thi Async

```python
import asyncio

async def run_crews():
    results = await asyncio.gather(
        crew1.akickoff(inputs={...}),
        crew2.akickoff(inputs={...}),
        crew3.akickoff(inputs={...})
    )
    return results
```

### 5. Tối ưu chi phí

```python
# Sử dụng model rẻ hơn cho task đơn giản
simple_agent = Agent(
    role="Data Formatter",
    llm="openai/gpt-3.5-turbo"  # Model rẻ hơn
)

# Sử dụng model mạnh hơn cho reasoning phức tạp
complex_agent = Agent(
    role="Strategic Planner",
    llm="openai/gpt-4o"  # Model có khả năng hơn
)

# Sử dụng function_calling_llm cho tool calls
agent = Agent(
    role="...",
    llm="openai/gpt-4o",
    function_calling_llm="openai/gpt-3.5-turbo"  # Rẻ hơn cho tool calls
)
```

---

## Xử lý lỗi

### 1. Retry cấp Agent

```python
agent = Agent(
    role="...",
    max_retry_limit=3,  # Thử lại khi lỗi
    max_execution_time=300  # Timeout 5 phút
)
```

### 2. Xác thực Guardrail

```python
def validate_output(output):
    try:
        data = json.loads(output.raw)
        if "error" in data:
            return (False, "Output contains error, retry")
        return (True, output)
    except json.JSONDecodeError:
        return (False, "Invalid JSON, please format correctly")

task = Task(
    description="...",
    guardrails=[validate_output],
    guardrail_max_retries=3
)
```

### 3. Callbacks để giám sát

```python
def step_monitor(step_output):
    print(f"Step completed: {step_output}")
    if "error" in str(step_output).lower():
        # Log hoặc cảnh báo
        logging.warning(f"Potential error: {step_output}")

def task_monitor(task_output):
    print(f"Task completed: {task_output.description}")
    # Gửi đến hệ thống giám sát

crew = Crew(
    agents=[...],
    tasks=[...],
    step_callback=step_monitor,
    task_callback=task_monitor
)
```

### 4. Xử lý lỗi Flow

```python
class RobustFlow(Flow):
    @start()
    def risky_operation(self):
        try:
            # Code rủi ro
            return result
        except Exception as e:
            self.state.error = str(e)
            return None

    @router(risky_operation)
    def check_result(self, result):
        if result is None:
            return "error_path"
        return "success_path"

    @listen("error_path")
    def handle_error(self):
        return f"Error handled: {self.state.error}"
```

---

## Chiến lược Testing

### 1. Unit Testing Agent

```python
import pytest
from unittest.mock import patch

def test_agent_creation():
    agent = Agent(
        role="Test Agent",
        goal="Test goals",
        backstory="Test backstory"
    )
    assert agent.role == "Test Agent"

@patch('crewai.Agent.execute_task')
def test_agent_execution(mock_execute):
    mock_execute.return_value = "Mocked result"
    # Test hành vi agent
```

### 2. Testing Task

```python
def test_task_output():
    task = Task(
        description="Test task",
        expected_output="Expected result",
        agent=mock_agent
    )
    # Xác minh cấu hình task
    assert task.description == "Test task"
```

### 3. Integration Testing Crew

```python
def test_crew_execution():
    crew = Crew(
        agents=[agent1, agent2],
        tasks=[task1, task2],
        verbose=False
    )

    result = crew.kickoff(inputs={"topic": "test"})

    assert result is not None
    assert len(result.tasks_output) == 2
```

### 4. Mocking LLM Calls

```python
from unittest.mock import MagicMock

def test_with_mocked_llm():
    mock_llm = MagicMock()
    mock_llm.call.return_value = "Mocked response"

    agent = Agent(
        role="Test",
        goal="Test",
        backstory="Test",
        llm=mock_llm
    )
    # Test với LLM được mock
```

---

## Các lỗi thường gặp cần tránh

1. **Vai trò Agent trùng lặp**
   - Mỗi agent nên có trách nhiệm riêng biệt
   - Tránh nhầm lẫn về ai làm gì

2. **Mô tả Task mơ hồ**
   - Cụ thể về những gì bạn muốn
   - Bao gồm kỳ vọng về định dạng

3. **Bỏ qua Context Windows**
   - Bật `respect_context_window=True`
   - Sử dụng RAG cho tài liệu lớn

4. **Không có giới hạn tốc độ**
   - Đặt `max_rpm` để tránh bị throttle API
   - Xem xét chi phí với số lần iteration cao

5. **Thiếu xử lý lỗi**
   - Sử dụng guardrails để validation
   - Đặt timeout phù hợp
   - Triển khai callbacks để giám sát

6. **Phân cấp quá phức tạp**
   - Bắt đầu với sequential process
   - Chỉ sử dụng hierarchical khi cần

7. **Không test từng bước**
   - Test agent riêng lẻ trước
   - Sau đó test kết hợp task
   - Cuối cùng test workflow crew đầy đủ
