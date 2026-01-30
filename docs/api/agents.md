# Agents API Reference

API reference cho Agent system.

## BaseAgent

Agent cơ bản với LLM và Tools integration.

### Class Definition

```python
from agents_framework.agents.base import BaseAgent

class BaseAgent:
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        tools: Optional[ToolRegistry] = None,
        memory: Optional[MemoryManager] = None,
        system_prompt: Optional[str] = None,
        name: str = "agent",
    ):
        ...
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `llm_provider` | `BaseLLMProvider` | LLM provider instance |
| `tools` | `ToolRegistry` | Optional tool registry |
| `memory` | `MemoryManager` | Optional memory manager |
| `system_prompt` | `str` | System prompt for agent |
| `name` | `str` | Agent identifier |

### Methods

#### run(input: str) -> str

Chạy agent với input và trả về response.

```python
agent = BaseAgent(provider, tools=registry)
result = await agent.run("Tính 5 + 3")
```

#### add_tool(tool: BaseTool)

Thêm tool vào agent.

```python
agent.add_tool(calculator_tool)
```

## SupervisorAgent

Agent supervisor điều phối workers.

### Class Definition

```python
from agents_framework.agents.supervisor import SupervisorAgent

class SupervisorAgent(BaseAgent):
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        workers: List[str],
        strategy: str = "parallel",
        **kwargs,
    ):
        ...
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `llm_provider` | `BaseLLMProvider` | LLM provider |
| `workers` | `List[str]` | List of worker agent IDs |
| `strategy` | `str` | "parallel" hoặc "sequential" |

### Methods

#### delegate(task: str, worker_id: str) -> str

Giao task cho worker cụ thể.

```python
result = await supervisor.delegate("Research AI", "researcher")
```

#### aggregate(results: List[str]) -> str

Tổng hợp kết quả từ workers.

```python
final = await supervisor.aggregate([result1, result2])
```

## RouterAgent

Agent định tuyến requests.

### Class Definition

```python
from agents_framework.agents.router import RouterAgent

class RouterAgent(BaseAgent):
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        routes: Dict[str, str],
        **kwargs,
    ):
        ...
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `llm_provider` | `BaseLLMProvider` | LLM provider |
| `routes` | `Dict[str, str]` | Mapping pattern -> agent_id |

### Methods

#### route(input: str) -> str

Xác định agent phù hợp cho input.

```python
target_agent = await router.route("Tôi cần hỗ trợ kỹ thuật")
# Returns: "tech_support"
```

## AgentSpawner

Tạo agents động.

### Class Definition

```python
from agents_framework.agents.spawner import AgentSpawner

class AgentSpawner:
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        templates: Dict[str, AgentTemplate] = None,
    ):
        ...
```

### Methods

#### spawn(template_name: str, **config) -> BaseAgent

Tạo agent mới từ template.

```python
spawner = AgentSpawner(provider, templates={"worker": worker_template})
new_agent = spawner.spawn("worker", name="worker-1")
```

#### cleanup(agent_id: str)

Dọn dẹp agent đã hoàn thành.

```python
spawner.cleanup("worker-1")
```

## Example: Complete Agent Setup

```python
import asyncio
from agents_framework.llm.base import LLMConfig
from agents_framework.llm.providers.openai import OpenAIProvider
from agents_framework.agents.base import BaseAgent
from agents_framework.tools.base import tool
from agents_framework.tools.registry import ToolRegistry
from agents_framework.memory.short_term import SessionMemory
from agents_framework.memory.manager import MemoryManager

# Define tools
@tool(name="search", description="Search the web")
def search(query: str) -> str:
    return f"Found: {query} information"

@tool(name="calculate", description="Calculate expression")
def calculate(expr: str) -> str:
    return str(eval(expr))

async def main():
    # Setup
    config = LLMConfig(model="gpt-4", api_key="...")
    provider = OpenAIProvider(config)

    registry = ToolRegistry()
    registry.register(search)
    registry.register(calculate)

    memory = MemoryManager(
        short_term=SessionMemory(max_tokens=4000)
    )

    # Create agent
    agent = BaseAgent(
        llm_provider=provider,
        tools=registry,
        memory=memory,
        system_prompt="Bạn là trợ lý thông minh.",
        name="assistant",
    )

    # Run
    result = await agent.run("Tính 100 * 5 và tìm kiếm về Python")
    print(result)

asyncio.run(main())
```
