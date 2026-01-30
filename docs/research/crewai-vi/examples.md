# Ví dụ Code CrewAI

Tài liệu này cung cấp các ví dụ code toàn diện cho các trường hợp sử dụng CrewAI khác nhau.

## Mục lục

1. [Thiết lập Crew cơ bản](#thiết-lập-crew-cơ-bản)
2. [Cấu hình dựa trên YAML](#cấu-hình-dựa-trên-yaml)
3. [Workflow đa Agent](#workflow-đa-agent)
4. [Tools tùy chỉnh](#tools-tùy-chỉnh)
5. [Ví dụ Flows](#ví-dụ-flows)
6. [Memory và Knowledge](#memory-và-knowledge)
7. [Patterns Production](#patterns-production)
8. [Ví dụ thực tế](#ví-dụ-thực-tế)

---

## Thiết lập Crew cơ bản

### Ví dụ tối thiểu

```python
from crewai import Agent, Task, Crew, Process

# Định nghĩa agents
researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI",
    backstory="""You are an expert at finding and analyzing information.
    You work at a leading tech think tank.""",
    verbose=True
)

writer = Agent(
    role="Tech Content Writer",
    goal="Create engaging content about AI discoveries",
    backstory="""You specialize in making complex topics accessible.
    You're known for your clear, engaging writing style.""",
    verbose=True
)

# Định nghĩa tasks
research_task = Task(
    description="Research the latest developments in {topic}",
    expected_output="A detailed report with key findings and sources",
    agent=researcher
)

writing_task = Task(
    description="Write an article based on the research findings",
    expected_output="A well-structured, engaging article in markdown format",
    agent=writer,
    context=[research_task]  # Sử dụng output của research làm context
)

# Tạo và chạy crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=True
)

# Thực thi
result = crew.kickoff(inputs={"topic": "AI Agents"})
print(result.raw)
```

### Với Structured Output

```python
from crewai import Agent, Task, Crew
from pydantic import BaseModel
from typing import List

class ResearchReport(BaseModel):
    title: str
    summary: str
    key_findings: List[str]
    sources: List[str]
    recommendations: List[str]

researcher = Agent(
    role="Research Analyst",
    goal="Produce comprehensive research reports",
    backstory="Expert researcher with analytical skills"
)

research_task = Task(
    description="Research {topic} and provide structured findings",
    expected_output="A structured research report",
    agent=researcher,
    output_pydantic=ResearchReport
)

crew = Crew(
    agents=[researcher],
    tasks=[research_task]
)

result = crew.kickoff(inputs={"topic": "quantum computing"})

# Truy cập structured output
report = result.pydantic
print(f"Title: {report.title}")
print(f"Findings: {report.key_findings}")
```

---

## Cấu hình dựa trên YAML

### Cấu trúc dự án

```
my_project/
├── src/my_project/
│   ├── crew.py
│   ├── main.py
│   ├── config/
│   │   ├── agents.yaml
│   │   └── tasks.yaml
│   └── tools/
│       └── custom_tool.py
```

### agents.yaml

```yaml
# config/agents.yaml
researcher:
  role: >
    {topic} Senior Data Researcher
  goal: >
    Uncover cutting-edge developments in {topic}
  backstory: >
    You're a seasoned researcher with a knack for uncovering the latest
    developments in {topic}. Known for your ability to find the most relevant
    information and present it in a clear and concise manner.

reporting_analyst:
  role: >
    {topic} Reporting Analyst
  goal: >
    Create detailed reports based on {topic} data analysis and research findings
  backstory: >
    You're a meticulous analyst with a keen eye for detail. You're known for
    your ability to turn complex data into clear and concise reports, making
    it easy for others to understand and act on the information you provide.
```

### tasks.yaml

```yaml
# config/tasks.yaml
research_task:
  description: >
    Conduct a thorough research about {topic}
    Make sure you find any interesting and relevant information given
    the current year is 2025.
  expected_output: >
    A list with 10 bullet points of the most relevant information about {topic}
  agent: researcher

reporting_task:
  description: >
    Review the context you got and expand each topic into a full section for a report.
    Make sure the report is detailed and contains any and all relevant information.
  expected_output: >
    A fully fledged report with the main topics, each with a full section of information.
    Formatted as markdown.
  agent: reporting_analyst
  output_file: report.md
```

### crew.py

```python
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from typing import List

@CrewBase
class LatestAiDevelopmentCrew():
    """LatestAiDevelopment crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            tools=[SerperDevTool()]
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Tạo LatestAiDevelopment crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
```

### main.py

```python
#!/usr/bin/env python
from my_project.crew import LatestAiDevelopmentCrew

def run():
    """Chạy crew."""
    inputs = {
        'topic': 'AI Agents'
    }
    LatestAiDevelopmentCrew().crew().kickoff(inputs=inputs)

if __name__ == "__main__":
    run()
```

---

## Workflow đa Agent

### Crew phân tích cổ phiếu

Dựa trên repository ví dụ chính thức của CrewAI.

```python
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import WebsiteSearchTool, ScrapeWebsiteTool

@CrewBase
class StockAnalysisCrew:
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def financial_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['financial_analyst'],
            verbose=True,
            tools=[
                ScrapeWebsiteTool(),
                WebsiteSearchTool(),
            ]
        )

    @agent
    def research_analyst_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['research_analyst'],
            verbose=True,
            tools=[
                ScrapeWebsiteTool(),
            ]
        )

    @agent
    def investment_advisor_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['investment_advisor'],
            verbose=True,
            tools=[
                ScrapeWebsiteTool(),
                WebsiteSearchTool(),
            ]
        )

    @task
    def financial_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['financial_analysis'],
            agent=self.financial_agent(),
        )

    @task
    def research(self) -> Task:
        return Task(
            config=self.tasks_config['research'],
            agent=self.research_analyst_agent(),
        )

    @task
    def recommend(self) -> Task:
        return Task(
            config=self.tasks_config['recommend'],
            agent=self.investment_advisor_agent(),
        )

    @crew
    def crew(self) -> Crew:
        """Tạo Stock Analysis crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
```

### Team phân cấp

```python
from crewai import Agent, Crew, Process, Task

# Các agent chuyên gia
frontend_dev = Agent(
    role="Senior Frontend Developer",
    goal="Build beautiful, responsive user interfaces",
    backstory="Expert in React, TypeScript, and modern CSS"
)

backend_dev = Agent(
    role="Senior Backend Developer",
    goal="Build robust, scalable APIs and services",
    backstory="Expert in Python, databases, and system design"
)

qa_engineer = Agent(
    role="QA Engineer",
    goal="Ensure software quality through comprehensive testing",
    backstory="Expert in test automation and quality assurance"
)

# Tasks
feature_task = Task(
    description="""
    Build a user authentication feature including:
    - Login/logout functionality
    - Password reset
    - Session management
    """,
    expected_output="Complete implementation with tests"
)

# Crew phân cấp với manager
crew = Crew(
    agents=[frontend_dev, backend_dev, qa_engineer],
    tasks=[feature_task],
    process=Process.hierarchical,
    manager_llm="openai/gpt-4o",
    verbose=True
)

result = crew.kickoff()
```

---

## Tools tùy chỉnh

### Phương pháp 1: Kế thừa BaseTool

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import requests

class WeatherInput(BaseModel):
    city: str = Field(..., description="City name to get weather for")
    units: str = Field(default="metric", description="Temperature units (metric/imperial)")

class WeatherTool(BaseTool):
    name: str = "Weather Lookup"
    description: str = "Get current weather for a city"
    args_schema: Type[BaseModel] = WeatherInput

    def _run(self, city: str, units: str = "metric") -> str:
        api_key = os.getenv("OPENWEATHER_API_KEY")
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "units": units, "appid": api_key}

        response = requests.get(url, params=params)
        data = response.json()

        if response.status_code == 200:
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"]
            return f"Weather in {city}: {temp}°, {desc}"
        return f"Error: Could not get weather for {city}"

# Sử dụng
weather_tool = WeatherTool()
agent = Agent(
    role="Weather Reporter",
    goal="Provide accurate weather information",
    backstory="Expert meteorologist",
    tools=[weather_tool]
)
```

### Phương pháp 2: Decorator @tool

```python
from crewai.tools import tool
import requests

@tool("Stock Price Lookup")
def get_stock_price(symbol: str) -> str:
    """Get the current stock price for a given symbol.

    Args:
        symbol: Stock ticker symbol (e.g., AAPL, GOOGL)
    """
    # Triển khai
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": api_key
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "Global Quote" in data:
        price = data["Global Quote"]["05. price"]
        return f"{symbol}: ${price}"
    return f"Could not find price for {symbol}"

# Sử dụng
agent = Agent(
    role="Financial Analyst",
    goal="Analyze stock performance",
    backstory="Expert in financial markets",
    tools=[get_stock_price]
)
```

### Tool Async

```python
from crewai.tools import tool
import aiohttp

@tool("Async Data Fetcher")
async def fetch_multiple_urls(urls: list[str]) -> str:
    """Fetch data from multiple URLs concurrently.

    Args:
        urls: List of URLs to fetch
    """
    async with aiohttp.ClientSession() as session:
        results = []
        for url in urls:
            async with session.get(url) as response:
                text = await response.text()
                results.append(f"{url}: {len(text)} chars")
        return "\n".join(results)
```

---

## Ví dụ Flows

### Flow cơ bản

```python
from crewai.flow.flow import Flow, start, listen
from pydantic import BaseModel

class ContentState(BaseModel):
    topic: str = ""
    research: str = ""
    draft: str = ""
    final: str = ""

class ContentFlow(Flow[ContentState]):

    @start()
    def initialize(self):
        self.state.topic = "AI in Healthcare"
        return self.state.topic

    @listen(initialize)
    def research_topic(self, topic):
        # Có thể khởi chạy research crew ở đây
        self.state.research = f"Research findings on {topic}..."
        return self.state.research

    @listen(research_topic)
    def write_draft(self, research):
        self.state.draft = f"Draft article based on: {research}"
        return self.state.draft

    @listen(write_draft)
    def finalize(self, draft):
        self.state.final = f"Final version: {draft}"
        return self.state.final

# Chạy
flow = ContentFlow()
result = flow.kickoff()
print(result)
print(f"Final state: {flow.state}")
```

### Flow vòng lặp tự đánh giá

Dựa trên ví dụ chính thức của CrewAI.

```python
from typing import Optional
from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel
from crewai import Agent, Crew, Task

class EvalState(BaseModel):
    content: str = ""
    feedback: Optional[str] = None
    valid: bool = False
    retry_count: int = 0

class SelfEvalFlow(Flow[EvalState]):

    @start("retry")
    def generate_content(self):
        print("Generating content...")

        writer = Agent(
            role="Content Writer",
            goal="Write engaging content",
            backstory="Expert writer"
        )

        task = Task(
            description=f"Write a short post about AI. Previous feedback: {self.state.feedback or 'None'}",
            expected_output="A concise, engaging post",
            agent=writer
        )

        crew = Crew(agents=[writer], tasks=[task])
        result = crew.kickoff()

        self.state.content = result.raw
        return self.state.content

    @router(generate_content)
    def evaluate_content(self):
        if self.state.retry_count > 3:
            return "max_retry_exceeded"

        # Đánh giá chất lượng nội dung
        evaluator = Agent(
            role="Content Evaluator",
            goal="Evaluate content quality",
            backstory="Expert editor"
        )

        eval_task = Task(
            description=f"Evaluate this content: {self.state.content}. Is it good enough?",
            expected_output="JSON with 'valid': true/false and 'feedback': string",
            agent=evaluator,
            output_json={"valid": bool, "feedback": str}
        )

        crew = Crew(agents=[evaluator], tasks=[eval_task])
        result = crew.kickoff()

        self.state.valid = result.json_dict.get("valid", False)
        self.state.feedback = result.json_dict.get("feedback", "")
        self.state.retry_count += 1

        if self.state.valid:
            return "complete"
        return "retry"

    @listen("complete")
    def save_result(self):
        print("Content is valid!")
        print(f"Final content: {self.state.content}")
        with open("output.txt", "w") as f:
            f.write(self.state.content)

    @listen("max_retry_exceeded")
    def handle_max_retries(self):
        print("Max retry count exceeded")
        print(f"Last content: {self.state.content}")
        print(f"Last feedback: {self.state.feedback}")

# Chạy
flow = SelfEvalFlow()
flow.kickoff()
```

### Flow với nhiều Crews

```python
from crewai.flow.flow import Flow, start, listen, router, or_
from crewai import Agent, Crew, Task
from pydantic import BaseModel

class PipelineState(BaseModel):
    raw_data: str = ""
    analysis: str = ""
    report: str = ""
    route: str = ""

class DataPipelineFlow(Flow[PipelineState]):

    @start()
    def ingest_data(self):
        # Crew nhập dữ liệu
        ingester = Agent(
            role="Data Ingester",
            goal="Collect and validate data",
            backstory="Expert in data pipelines"
        )

        task = Task(
            description="Collect sample data for analysis",
            expected_output="Raw data in JSON format",
            agent=ingester
        )

        crew = Crew(agents=[ingester], tasks=[task])
        result = crew.kickoff()
        self.state.raw_data = result.raw
        return self.state.raw_data

    @listen(ingest_data)
    def analyze_data(self, data):
        # Crew phân tích
        analyst = Agent(
            role="Data Analyst",
            goal="Analyze data patterns",
            backstory="Expert statistician"
        )

        task = Task(
            description=f"Analyze this data: {data}",
            expected_output="Analysis summary with key insights",
            agent=analyst
        )

        crew = Crew(agents=[analyst], tasks=[task])
        result = crew.kickoff()
        self.state.analysis = result.raw
        return self.state.analysis

    @router(analyze_data)
    def determine_report_type(self, analysis):
        if "critical" in analysis.lower():
            return "urgent_report"
        return "standard_report"

    @listen("urgent_report")
    def generate_urgent_report(self):
        self.state.report = f"URGENT REPORT: {self.state.analysis}"
        return self.state.report

    @listen("standard_report")
    def generate_standard_report(self):
        self.state.report = f"Standard Report: {self.state.analysis}"
        return self.state.report

    @listen(or_("urgent_report", "standard_report"))
    def distribute_report(self, report):
        print(f"Distributing: {report[:100]}...")
        return "Report distributed"

# Chạy
flow = DataPipelineFlow()
result = flow.kickoff()
```

---

## Memory và Knowledge

### Crew có bật Memory

```python
from crewai import Agent, Crew, Task

# Tạo agents
support_agent = Agent(
    role="Customer Support",
    goal="Help customers with their questions",
    backstory="Experienced support specialist",
    memory=True  # Memory cấp agent
)

# Tạo crew với memory
crew = Crew(
    agents=[support_agent],
    tasks=[...],
    memory=True,  # Bật memory ngắn hạn, dài hạn, entity
    verbose=True
)

# Tương tác đầu tiên
result1 = crew.kickoff(inputs={"query": "What are your hours?"})

# Tương tác thứ hai - sẽ nhớ context
result2 = crew.kickoff(inputs={"query": "And on weekends?"})
```

### Nguồn Knowledge

```python
from crewai import Agent, Crew, Task
from crewai.knowledge.source import (
    StringKnowledgeSource,
    PDFKnowledgeSource,
    TextFileKnowledgeSource
)

# Tạo nguồn knowledge
company_info = StringKnowledgeSource(
    content="""
    Company Name: TechCorp
    Founded: 2020
    Products: AI Assistant, Data Analyzer, AutoML Platform
    Support Hours: 9 AM - 6 PM EST, Monday to Friday
    """
)

product_docs = PDFKnowledgeSource(file_path="./docs/product_manual.pdf")
faq = TextFileKnowledgeSource(file_path="./docs/faq.txt")

# Agent với knowledge
support_agent = Agent(
    role="Customer Support Specialist",
    goal="Help customers with product questions",
    backstory="Expert on all company products",
    knowledge_sources=[company_info, product_docs, faq]
)

# Hoặc knowledge cấp crew (chia sẻ bởi tất cả agents)
crew = Crew(
    agents=[support_agent],
    tasks=[support_task],
    knowledge_sources=[company_info]  # Tất cả agents có thể truy cập
)
```

### Cấu hình Embedder tùy chỉnh

```python
from crewai import Crew

# Sử dụng Ollama cho embeddings local
crew = Crew(
    agents=[...],
    tasks=[...],
    memory=True,
    embedder={
        "provider": "ollama",
        "config": {
            "model": "mxbai-embed-large",
            "url": "http://localhost:11434/api/embeddings"
        }
    }
)

# Sử dụng OpenAI embeddings (mặc định)
crew = Crew(
    agents=[...],
    tasks=[...],
    memory=True,
    embedder={
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small"
        }
    }
)
```

---

## Patterns Production

### Streaming Output

```python
from crewai import Crew

crew = Crew(
    agents=[...],
    tasks=[...],
    stream=True  # Bật streaming
)

# Lấy streaming iterator
streaming = crew.kickoff(inputs={"topic": "AI"})

# Stream output real-time
for chunk in streaming:
    print(chunk.content, end="", flush=True)

# Lấy kết quả cuối cùng
final_result = streaming.result
```

### Thực thi Async

```python
import asyncio
from crewai import Crew

async def run_crews():
    crew1 = Crew(agents=[...], tasks=[...])
    crew2 = Crew(agents=[...], tasks=[...])
    crew3 = Crew(agents=[...], tasks=[...])

    # Chạy crews song song
    results = await asyncio.gather(
        crew1.akickoff(inputs={"topic": "AI"}),
        crew2.akickoff(inputs={"topic": "ML"}),
        crew3.akickoff(inputs={"topic": "Data Science"})
    )

    return results

# Chạy
results = asyncio.run(run_crews())
```

### Giới hạn tốc độ và xử lý lỗi

```python
from crewai import Agent, Crew, Task

# Agent có giới hạn tốc độ
agent = Agent(
    role="API Consumer",
    goal="Fetch data from APIs",
    backstory="Expert at API integrations",
    max_rpm=30,  # 30 requests mỗi phút
    max_retry_limit=3,  # Thử lại khi lỗi
    max_execution_time=300  # Timeout 5 phút
)

# Guardrails cho validation
def validate_json(output):
    import json
    try:
        json.loads(output.raw)
        return (True, output)
    except:
        return (False, "Output must be valid JSON")

task = Task(
    description="Fetch and process API data",
    expected_output="JSON response",
    agent=agent,
    guardrails=[validate_json],
    guardrail_max_retries=3
)

# Crew với callbacks
def on_task_complete(output):
    print(f"Task completed: {output.description}")

crew = Crew(
    agents=[agent],
    tasks=[task],
    max_rpm=60,  # Giới hạn tốc độ cấp crew
    task_callback=on_task_complete,
    output_log_file="execution.json"
)
```

### LLM khác nhau cho mục đích khác nhau

```python
from crewai import Agent, LLM

# Model mạnh cho reasoning phức tạp
strategic_llm = LLM(
    model="openai/gpt-4o",
    temperature=0.7,
    max_tokens=4000
)

# Model rẻ hơn cho task đơn giản
simple_llm = LLM(
    model="openai/gpt-3.5-turbo",
    temperature=0.5,
    max_tokens=2000
)

# Strategic planner sử dụng model mạnh
planner = Agent(
    role="Strategic Planner",
    goal="Develop comprehensive strategies",
    backstory="Senior strategist",
    llm=strategic_llm
)

# Data formatter sử dụng model rẻ hơn
formatter = Agent(
    role="Data Formatter",
    goal="Format data consistently",
    backstory="Data processing expert",
    llm=simple_llm,
    function_calling_llm=simple_llm  # Cũng dùng model rẻ cho tools
)
```

---

## Ví dụ thực tế

### Lên kế hoạch du lịch

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

# Agents
city_researcher = Agent(
    role="City Research Expert",
    goal="Research and compile information about cities",
    backstory="Expert in travel research",
    tools=[SerperDevTool(), ScrapeWebsiteTool()]
)

local_expert = Agent(
    role="Local Expert",
    goal="Provide local insights and recommendations",
    backstory="Local guide with deep knowledge"
)

trip_planner = Agent(
    role="Trip Planner",
    goal="Create comprehensive travel itineraries",
    backstory="Experienced travel planner"
)

# Tasks
city_research = Task(
    description="""Research {destination} for a trip:
    - Best time to visit
    - Major attractions
    - Local customs
    - Weather expectations
    """,
    expected_output="Comprehensive city guide",
    agent=city_researcher
)

local_insights = Task(
    description="Provide local recommendations for {destination}",
    expected_output="Local tips and hidden gems",
    agent=local_expert,
    context=[city_research]
)

create_itinerary = Task(
    description="""Create a {days}-day itinerary for {destination}
    considering the research and local insights""",
    expected_output="Day-by-day travel itinerary",
    agent=trip_planner,
    context=[city_research, local_insights],
    output_file="itinerary.md"
)

# Crew
trip_crew = Crew(
    agents=[city_researcher, local_expert, trip_planner],
    tasks=[city_research, local_insights, create_itinerary],
    process=Process.sequential,
    memory=True,
    verbose=True
)

# Chạy
result = trip_crew.kickoff(inputs={
    "destination": "Tokyo, Japan",
    "days": 7
})
```

### Pipeline Content Marketing

```python
from crewai.flow.flow import Flow, start, listen, router
from crewai import Agent, Crew, Task
from pydantic import BaseModel
from typing import List

class ContentState(BaseModel):
    topic: str = ""
    keywords: List[str] = []
    outline: str = ""
    draft: str = ""
    final: str = ""
    social_posts: List[str] = []

class ContentMarketingFlow(Flow[ContentState]):

    @start()
    def keyword_research(self):
        researcher = Agent(
            role="SEO Specialist",
            goal="Identify high-value keywords",
            backstory="Expert in SEO and content strategy"
        )

        task = Task(
            description=f"Research keywords for: {self.state.topic}",
            expected_output="List of 10 target keywords with search volume",
            agent=researcher
        )

        crew = Crew(agents=[researcher], tasks=[task])
        result = crew.kickoff()
        self.state.keywords = result.raw.split("\n")[:10]
        return self.state.keywords

    @listen(keyword_research)
    def create_outline(self, keywords):
        planner = Agent(
            role="Content Strategist",
            goal="Create engaging content outlines",
            backstory="Expert content planner"
        )

        task = Task(
            description=f"Create an outline for {self.state.topic} targeting: {keywords}",
            expected_output="Detailed article outline with sections",
            agent=planner
        )

        crew = Crew(agents=[planner], tasks=[task])
        result = crew.kickoff()
        self.state.outline = result.raw
        return self.state.outline

    @listen(create_outline)
    def write_draft(self, outline):
        writer = Agent(
            role="Content Writer",
            goal="Write engaging, SEO-optimized content",
            backstory="Professional content writer"
        )

        task = Task(
            description=f"Write an article based on this outline: {outline}",
            expected_output="Complete article in markdown",
            agent=writer
        )

        crew = Crew(agents=[writer], tasks=[task])
        result = crew.kickoff()
        self.state.draft = result.raw
        return self.state.draft

    @listen(write_draft)
    def edit_content(self, draft):
        editor = Agent(
            role="Senior Editor",
            goal="Polish content to publication quality",
            backstory="Experienced editor"
        )

        task = Task(
            description=f"Edit and improve this article: {draft}",
            expected_output="Publication-ready article",
            agent=editor
        )

        crew = Crew(agents=[editor], tasks=[task])
        result = crew.kickoff()
        self.state.final = result.raw
        return self.state.final

    @listen(edit_content)
    def create_social_posts(self, final):
        social_manager = Agent(
            role="Social Media Manager",
            goal="Create engaging social media content",
            backstory="Expert in social media marketing"
        )

        task = Task(
            description=f"Create 5 social media posts to promote: {final[:500]}",
            expected_output="5 social media posts for different platforms",
            agent=social_manager
        )

        crew = Crew(agents=[social_manager], tasks=[task])
        result = crew.kickoff()
        self.state.social_posts = result.raw.split("\n\n")
        return self.state.social_posts

# Chạy
flow = ContentMarketingFlow()
flow.state.topic = "The Future of AI in Healthcare"
result = flow.kickoff()

print("Final Article:", flow.state.final[:500])
print("Social Posts:", flow.state.social_posts)
```

---

## Tham chiếu lệnh CLI

```bash
# Tạo dự án crew mới
crewai create crew my_project

# Tạo dự án flow mới
crewai create flow my_flow

# Chạy crew
crewai run

# Cài đặt dependencies
crewai install

# Xem output của task
crewai log-tasks-outputs

# Replay từ task cụ thể
crewai replay -t task_id

# Reset memories
crewai reset-memories --short    # Memory ngắn hạn
crewai reset-memories --long     # Memory dài hạn
crewai reset-memories --entity   # Entity memory
crewai reset-memories --knowledge  # Knowledge

# Cập nhật CrewAI
crewai update
```

---

## Tài nguyên

- [Tài liệu chính thức](https://docs.crewai.com)
- [GitHub Repository](https://github.com/crewAIInc/crewAI)
- [Examples Repository](https://github.com/crewAIInc/crewAI-examples)
- [Diễn đàn cộng đồng](https://community.crewai.com)
