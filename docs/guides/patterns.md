# Multi-Agent Patterns Guide

Hướng dẫn sử dụng các patterns cho multi-agent systems.

## Tổng Quan

Agents Framework hỗ trợ 4 patterns chính:

1. **Hierarchical** - Supervisor điều phối workers
2. **Sequential** - Pipeline tuần tự
3. **Swarm** - Handoff giữa peers
4. **Router** - Định tuyến động

## 1. Hierarchical Pattern

Phù hợp khi có task phức tạp cần chia nhỏ.

### Cấu Trúc

```
        ┌────────────┐
        │ Supervisor │
        └─────┬──────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌───────┐ ┌───────┐ ┌───────┐
│Worker1│ │Worker2│ │Worker3│
└───────┘ └───────┘ └───────┘
```

### Implementation

```python
from agents_framework.teams.patterns.hierarchical import HierarchicalPattern
from agents_framework.teams.team import Team

# Tạo agents
supervisor = create_supervisor_agent()
researcher = create_researcher_agent()
writer = create_writer_agent()
reviewer = create_reviewer_agent()

# Setup pattern
pattern = HierarchicalPattern(
    supervisor="supervisor",
    workers=["researcher", "writer", "reviewer"],
    delegation_strategy="parallel",  # hoặc "sequential"
)

# Tạo team
team = Team(
    name="content-team",
    agents={
        "supervisor": supervisor,
        "researcher": researcher,
        "writer": writer,
        "reviewer": reviewer,
    },
    pattern=pattern,
)

# Run
result = await team.run("Viết bài blog về AI trong y tế")
```

### Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `parallel` | Workers chạy đồng thời | Tasks độc lập |
| `sequential` | Workers chạy tuần tự | Tasks phụ thuộc |
| `dynamic` | Supervisor quyết định | Tasks linh hoạt |

### Best Practices

- Supervisor chỉ coordinate, không làm task
- Workers nên specialized
- Aggregation logic trong supervisor

## 2. Sequential Pattern

Phù hợp cho pipelines xử lý data.

### Cấu Trúc

```
┌────────┐   ┌───────────┐   ┌──────────┐   ┌──────────┐
│ Input  │ → │ Analyzer  │ → │Processor │ → │Formatter │ → Output
└────────┘   └───────────┘   └──────────┘   └──────────┘
```

### Implementation

```python
from agents_framework.teams.patterns.sequential import SequentialPattern

pattern = SequentialPattern(
    stages=["analyzer", "processor", "formatter"],
    pass_output=True,  # Output stage trước → Input stage sau
)

team = Team(
    name="data-pipeline",
    agents={
        "analyzer": analyzer_agent,
        "processor": processor_agent,
        "formatter": formatter_agent,
    },
    pattern=pattern,
)

result = await team.run("Analyze and format this dataset")
```

### Options

```python
pattern = SequentialPattern(
    stages=["a", "b", "c"],
    pass_output=True,           # Chuyển output qua stages
    stop_on_failure=True,       # Dừng nếu stage fail
    stage_timeout=30.0,         # Timeout mỗi stage
)
```

### Use Cases

- ETL pipelines
- Document processing
- Multi-step analysis

## 3. Swarm Pattern

Phù hợp cho customer support, routing động.

### Cấu Trúc

```
┌─────────┐     ┌─────────┐
│ Triage  │ ←─→ │  Tech   │
└────┬────┘     └────┬────┘
     │               │
     └───────┬───────┘
             │
     ┌───────▼───────┐
     │    Billing    │
     └───────────────┘
```

### Implementation

```python
from agents_framework.teams.patterns.swarm import SwarmPattern, HandoffResult

# Agent có thể handoff
class TriageAgent:
    async def run(self, input: str) -> Union[str, HandoffResult]:
        if "technical" in input.lower():
            return HandoffResult(
                target_agent="tech_support",
                context={"issue_type": "technical"},
                message=input,
            )
        elif "billing" in input.lower():
            return HandoffResult(
                target_agent="billing",
                context={"issue_type": "billing"},
                message=input,
            )
        return "How can I help you?"

# Setup swarm
pattern = SwarmPattern(
    agents={
        "triage": triage_agent,
        "tech_support": tech_agent,
        "billing": billing_agent,
    },
    entry_point="triage",
    max_handoffs=5,  # Giới hạn số lần handoff
)

team = Team(
    name="support-swarm",
    agents=pattern.agents,
    pattern=pattern,
)

result = await team.run("Tôi cần hỗ trợ kỹ thuật về sản phẩm")
```

### Handoff Context

```python
HandoffResult(
    target_agent="tech_support",
    context={
        "issue_type": "technical",
        "priority": "high",
        "customer_tier": "premium",
    },
    message="Original user message",
    carry_history=True,  # Mang theo conversation history
)
```

### Best Practices

- Entry point agent làm triage
- Giới hạn max_handoffs để tránh loops
- Mỗi agent specialized cho domain

## 4. Router Pattern

Phù hợp khi cần route dựa trên content.

### Implementation

```python
from agents_framework.teams.patterns.router import RouterPattern

pattern = RouterPattern(
    router_agent="router",
    target_agents=["sales", "support", "technical"],
    routing_strategy="llm",  # hoặc "rule"
)

# Rule-based routing
pattern = RouterPattern(
    router_agent="router",
    target_agents=["sales", "support", "technical"],
    routing_strategy="rule",
    rules={
        r"buy|price|cost": "sales",
        r"help|issue|problem": "support",
        r"code|api|integration": "technical",
    },
)
```

### LLM-based Routing

```python
class RouterAgent:
    ROUTING_PROMPT = """
    Analyze the user request and determine the best agent:
    - sales: Purchase, pricing, quotes
    - support: General help, issues
    - technical: Code, API, integration

    Request: {input}
    Best agent:
    """

    async def route(self, input: str) -> str:
        response = await self.llm.generate([
            Message(role=MessageRole.USER, content=self.ROUTING_PROMPT.format(input=input))
        ])
        return response.content.strip().lower()
```

## Combining Patterns

Có thể combine patterns cho use cases phức tạp.

### Hierarchical + Sequential

```python
# Supervisor sử dụng sequential pipeline
class SupervisorAgent:
    def __init__(self):
        self.pipeline = SequentialPattern(
            stages=["research", "write", "review"],
        )

    async def run(self, task: str) -> str:
        # Delegate theo pipeline
        result = await self.pipeline.execute(task)
        return self.aggregate(result)
```

### Router + Swarm

```python
# Router chọn swarm phù hợp
pattern = RouterPattern(
    router_agent="router",
    target_agents=["support-swarm", "sales-swarm"],
)

support_swarm = SwarmPattern(
    agents={"triage": t, "tech": tech, "billing": b},
    entry_point="triage",
)
```

## Performance Tips

1. **Parallel when possible**: Dùng parallel delegation cho tasks độc lập
2. **Limit handoffs**: Set max_handoffs để tránh infinite loops
3. **Timeout per stage**: Set stage_timeout trong sequential
4. **Shared memory**: Dùng shared_memory cho team để giảm duplication
5. **Early exit**: Implement early termination khi đủ kết quả
