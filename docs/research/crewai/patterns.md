# CrewAI Patterns and Best Practices

This document covers collaboration patterns, design patterns, and best practices for building effective CrewAI systems.

## Table of Contents

1. [Agent Collaboration Patterns](#agent-collaboration-patterns)
2. [Task Flow Patterns](#task-flow-patterns)
3. [Crew Organization Patterns](#crew-organization-patterns)
4. [Flow Patterns](#flow-patterns)
5. [Design Best Practices](#design-best-practices)
6. [Performance Optimization](#performance-optimization)
7. [Error Handling](#error-handling)
8. [Testing Strategies](#testing-strategies)

---

## Agent Collaboration Patterns

### 1. Delegation Pattern

Enable agents to delegate subtasks to other agents.

```python
from crewai import Agent

# Lead agent with delegation enabled
project_lead = Agent(
    role="Project Lead",
    goal="Coordinate project deliverables",
    backstory="Senior manager with 10 years experience",
    allow_delegation=True,  # Can delegate to teammates
    verbose=True
)

# Specialist agents (no delegation)
researcher = Agent(
    role="Research Specialist",
    goal="Gather accurate data",
    backstory="Expert in data collection",
    allow_delegation=False  # Focused on own tasks
)

writer = Agent(
    role="Content Writer",
    goal="Create compelling content",
    backstory="Award-winning writer",
    allow_delegation=False
)
```

**When `allow_delegation=True`**, agents gain access to:
- **Delegate tool**: `Delegate work to coworker(task, context, coworker)`
- **Ask tool**: `Ask question to coworker(question, context, coworker)`

### 2. Research-Write-Edit Pipeline

Classic content creation pattern with clear handoffs.

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

# Tasks with context flow
research_task = Task(
    description="Research {topic} comprehensively",
    expected_output="Detailed research notes with sources",
    agent=researcher
)

writing_task = Task(
    description="Write an article based on research findings",
    expected_output="Draft article in markdown format",
    agent=writer,
    context=[research_task]  # Receives research output
)

editing_task = Task(
    description="Edit and polish the article",
    expected_output="Publication-ready article",
    agent=editor,
    context=[writing_task]  # Receives draft
)

crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential
)
```

### 3. Collaborative Single Task

Multiple agents contribute to a single complex task.

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

# Single complex task
analysis_task = Task(
    description="""
    Create a comprehensive market analysis report including:
    - Market size and growth trends
    - Competitive landscape
    - Statistical projections
    Delegate specific analyses to team members as needed.
    """,
    expected_output="Complete market analysis report",
    agent=lead_agent  # Lead coordinates, delegates to others
)
```

### 4. Hierarchical Management

Manager coordinates specialists for complex work.

```python
from crewai import Crew, Process

# Specialist agents
specialists = [
    Agent(role="Frontend Developer", goal="Build UI", backstory="..."),
    Agent(role="Backend Developer", goal="Build API", backstory="..."),
    Agent(role="QA Engineer", goal="Test quality", backstory="...")
]

# Manager delegates based on expertise
crew = Crew(
    agents=specialists,
    tasks=[feature_task, testing_task],
    process=Process.hierarchical,
    manager_llm="openai/gpt-4o"
)
```

---

## Task Flow Patterns

### 1. Sequential Dependencies

Each task builds on previous outputs.

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
    context=[task1]  # Depends on task1
)

task3 = Task(
    description="Implement the designed solution",
    expected_output="Working code",
    agent=developer,
    context=[task2]  # Depends on task2
)
```

### 2. Parallel Execution

Independent tasks run concurrently.

```python
# These can run in parallel
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

# This waits for both parallel tasks
synthesis = Task(
    description="Synthesize all research findings",
    expected_output="Comprehensive research summary",
    agent=synthesizer,
    context=[research_web, research_papers]  # Waits for both
)
```

### 3. Conditional Task Flow

Using guardrails to control flow.

```python
def quality_check(output):
    """Guardrail that validates output quality"""
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

Require human approval for critical tasks.

```python
approval_task = Task(
    description="Generate contract terms",
    expected_output="Legal contract draft",
    agent=legal_agent,
    human_input=True  # Requires human review
)
```

---

## Crew Organization Patterns

### 1. Single-Purpose Crews

Focused crews for specific domains.

```python
# Research crew
research_crew = Crew(
    agents=[researcher, fact_checker],
    tasks=[research_task, verification_task],
    process=Process.sequential
)

# Content crew
content_crew = Crew(
    agents=[writer, editor],
    tasks=[writing_task, editing_task],
    process=Process.sequential
)

# Use in a flow to orchestrate
class ContentPipeline(Flow):
    @start()
    def research_phase(self):
        return research_crew.kickoff(inputs={...})

    @listen(research_phase)
    def content_phase(self, research):
        return content_crew.kickoff(inputs={"research": research.raw})
```

### 2. Memory-Enabled Crews

Crews that learn and remember.

```python
learning_crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    memory=True,  # Enable all memory types
    verbose=True
)

# Multiple runs will build up memory
result1 = learning_crew.kickoff(inputs={"topic": "AI"})
result2 = learning_crew.kickoff(inputs={"topic": "ML"})  # Benefits from prior context
```

### 3. Knowledge-Enhanced Crews

Crews with domain-specific knowledge.

```python
from crewai.knowledge.source import PDFKnowledgeSource

product_docs = PDFKnowledgeSource(file_path="product_manual.pdf")

support_crew = Crew(
    agents=[support_agent, escalation_agent],
    tasks=[triage_task, resolution_task],
    knowledge_sources=[product_docs]  # Shared knowledge
)
```

---

## Flow Patterns

### 1. Linear Flow

Simple sequential execution.

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

### 2. Branching Flow

Conditional execution paths.

```python
from crewai.flow.flow import Flow, start, listen, router

class BranchingFlow(Flow):
    @start()
    def classify_input(self):
        # Analyze input
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

### 3. Fan-Out/Fan-In Flow

Parallel processing with aggregation.

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

### 4. Self-Evaluation Loop

Iterative improvement pattern.

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
        # Re-trigger generation
        return self.generate_content()

    @listen("accept")
    def finalize(self, content):
        return f"Final: {content}"

    def _assess_quality(self, content):
        # Implement quality assessment
        return 0.7
```

### 5. Human Feedback Flow

Interactive human review.

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

## Design Best Practices

### Agent Design

1. **Clear Role Definition**
   ```python
   # Good: Specific role
   role="Senior Python Backend Developer"

   # Bad: Vague role
   role="Developer"
   ```

2. **Actionable Goals**
   ```python
   # Good: Measurable goal
   goal="Reduce API response time to under 200ms"

   # Bad: Vague goal
   goal="Make things faster"
   ```

3. **Rich Backstory**
   ```python
   backstory="""You are a 10-year veteran at a Fortune 500 tech company.
   You've led teams building high-scale distributed systems.
   You prioritize clean code, thorough testing, and documentation."""
   ```

4. **Appropriate Tool Selection**
   ```python
   # Only give tools the agent needs
   researcher = Agent(
       role="Web Researcher",
       tools=[SerperDevTool(), WebsiteSearchTool()],  # Relevant tools
       # Don't add code execution tools to a researcher
   )
   ```

### Task Design

1. **Clear Descriptions**
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

2. **Specific Expected Outputs**
   ```python
   expected_output="""A structured report containing:
   - Executive summary (2-3 sentences)
   - Theme analysis table with sentiment scores
   - Prioritized recommendations (at least 3)
   - Supporting data visualizations (if applicable)
   """
   ```

3. **Use Structured Outputs**
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

### Crew Design

1. **Complementary Agents**
   - Each agent should have a distinct role
   - Avoid overlapping responsibilities
   - Consider the handoff points

2. **Right Process Type**
   ```python
   # Sequential: Clear pipeline, predictable flow
   Process.sequential

   # Hierarchical: Complex tasks, need coordination
   Process.hierarchical
   ```

3. **Enable Memory for Learning**
   ```python
   crew = Crew(
       ...,
       memory=True,  # Agents learn from interactions
       verbose=True   # For debugging
   )
   ```

---

## Performance Optimization

### 1. Rate Limiting

```python
# Crew-level rate limiting
crew = Crew(
    agents=[...],
    tasks=[...],
    max_rpm=60  # Max requests per minute
)

# Agent-level rate limiting
agent = Agent(
    role="...",
    max_rpm=10  # This agent's limit
)
```

### 2. Caching

```python
# Enable tool caching (default is True)
agent = Agent(
    role="...",
    cache=True
)

# Custom cache function
def should_cache(args, result):
    return len(result) > 100  # Only cache substantial results

tool.cache_function = should_cache
```

### 3. Token Management

```python
agent = Agent(
    role="...",
    respect_context_window=True,  # Auto-manage context
    max_iter=15  # Limit iterations
)
```

### 4. Async Execution

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

### 5. Cost Optimization

```python
# Use cheaper models for simple tasks
simple_agent = Agent(
    role="Data Formatter",
    llm="openai/gpt-3.5-turbo"  # Cheaper model
)

# Use powerful models for complex reasoning
complex_agent = Agent(
    role="Strategic Planner",
    llm="openai/gpt-4o"  # More capable model
)

# Use function_calling_llm for tool calls
agent = Agent(
    role="...",
    llm="openai/gpt-4o",
    function_calling_llm="openai/gpt-3.5-turbo"  # Cheaper for tool calls
)
```

---

## Error Handling

### 1. Agent-Level Retry

```python
agent = Agent(
    role="...",
    max_retry_limit=3,  # Retry on errors
    max_execution_time=300  # 5 minute timeout
)
```

### 2. Guardrail Validation

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

### 3. Callbacks for Monitoring

```python
def step_monitor(step_output):
    print(f"Step completed: {step_output}")
    if "error" in str(step_output).lower():
        # Log or alert
        logging.warning(f"Potential error: {step_output}")

def task_monitor(task_output):
    print(f"Task completed: {task_output.description}")
    # Send to monitoring system

crew = Crew(
    agents=[...],
    tasks=[...],
    step_callback=step_monitor,
    task_callback=task_monitor
)
```

### 4. Flow Error Handling

```python
class RobustFlow(Flow):
    @start()
    def risky_operation(self):
        try:
            # Risky code
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

## Testing Strategies

### 1. Unit Testing Agents

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
    # Test agent behavior
```

### 2. Testing Tasks

```python
def test_task_output():
    task = Task(
        description="Test task",
        expected_output="Expected result",
        agent=mock_agent
    )
    # Verify task configuration
    assert task.description == "Test task"
```

### 3. Integration Testing Crews

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
    # Test with mocked LLM
```

---

## Common Pitfalls to Avoid

1. **Overlapping Agent Roles**
   - Each agent should have distinct responsibilities
   - Avoid confusion about who does what

2. **Vague Task Descriptions**
   - Be specific about what you want
   - Include format expectations

3. **Ignoring Context Windows**
   - Enable `respect_context_window=True`
   - Use RAG for large documents

4. **No Rate Limiting**
   - Set `max_rpm` to avoid API throttling
   - Consider costs with high iteration counts

5. **Missing Error Handling**
   - Use guardrails for validation
   - Set appropriate timeouts
   - Implement callbacks for monitoring

6. **Overcomplicating Hierarchies**
   - Start with sequential process
   - Only use hierarchical when needed

7. **Not Testing Incrementally**
   - Test agents individually first
   - Then test task combinations
   - Finally test full crew workflows
