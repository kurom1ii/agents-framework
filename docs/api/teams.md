# Teams API Reference

API reference cho Multi-Agent Teams.

## Team

Lớp chính cho team orchestration.

### Class Definition

```python
from agents_framework.teams.team import Team

class Team:
    def __init__(
        self,
        name: str,
        agents: Dict[str, BaseAgent],
        pattern: TeamPattern,
        shared_memory: Optional[MemoryManager] = None,
    ):
        ...
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Team identifier |
| `agents` | `Dict[str, BaseAgent]` | Mapping agent_id -> agent |
| `pattern` | `TeamPattern` | Execution pattern |
| `shared_memory` | `MemoryManager` | Shared memory cho team |

### Methods

#### run(task: str) -> TeamResult

Chạy team với task.

```python
team = Team(
    name="research-team",
    agents={"researcher": r, "writer": w},
    pattern=HierarchicalPattern(supervisor="researcher"),
)
result = await team.run("Viết báo cáo về AI")
```

## AgentRegistry

Quản lý đăng ký agents.

### Class Definition

```python
from agents_framework.teams.registry import AgentRegistry

class AgentRegistry:
    def __init__(self):
        ...
```

### Methods

#### register(agent, agent_id: str, role: str = None)

Đăng ký agent vào registry.

```python
registry = AgentRegistry()
registry.register(my_agent, agent_id="agent-1", role="researcher")
```

#### get(agent_id: str) -> BaseAgent

Lấy agent theo ID.

```python
agent = registry.get("agent-1")
```

#### get_by_role(role: str) -> List[BaseAgent]

Lấy tất cả agents theo role.

```python
researchers = registry.get_by_role("researcher")
```

#### unregister(agent_id: str)

Xóa agent khỏi registry.

```python
registry.unregister("agent-1")
```

## MessageRouter

Route messages giữa agents.

### Class Definition

```python
from agents_framework.teams.router import MessageRouter, AgentMessage

class MessageRouter:
    def __init__(self):
        ...
```

### Methods

#### register_agent(agent_id: str, handler: Callable)

Đăng ký handler cho agent.

```python
router = MessageRouter()

async def handle_message(msg: AgentMessage):
    print(f"Received: {msg.content}")

router.register_agent("agent-1", handle_message)
```

#### route(message: AgentMessage)

Route message đến agent.

```python
msg = AgentMessage(
    sender_id="supervisor",
    receiver_id="agent-1",
    content="Do this task",
)
await router.route(msg)
```

#### broadcast(message: AgentMessage, targets: List[str])

Broadcast message đến nhiều agents.

```python
await router.broadcast(msg, targets=["agent-1", "agent-2", "agent-3"])
```

## Team Patterns

### HierarchicalPattern

Supervisor điều phối workers.

```python
from agents_framework.teams.patterns.hierarchical import HierarchicalPattern

pattern = HierarchicalPattern(
    supervisor="manager",
    workers=["worker-1", "worker-2"],
    delegation_strategy="parallel",
)
```

### SequentialPattern

Agents xử lý tuần tự.

```python
from agents_framework.teams.patterns.sequential import SequentialPattern

pattern = SequentialPattern(
    stages=["analyzer", "processor", "formatter"],
    pass_output=True,
)
```

### SwarmPattern

Handoff giữa các agents.

```python
from agents_framework.teams.patterns.swarm import SwarmPattern

pattern = SwarmPattern(
    agents={"triage": t, "tech": tech, "billing": b},
    entry_point="triage",
)
```

### RouterPattern

Route đến agent phù hợp.

```python
from agents_framework.teams.patterns.router import RouterPattern

pattern = RouterPattern(
    router_agent="router",
    target_agents=["sales", "support", "tech"],
)
```

## AgentMessage

Message giữa agents.

### Class Definition

```python
from agents_framework.teams.router import AgentMessage

@dataclass
class AgentMessage:
    sender_id: str
    receiver_id: str
    content: str
    message_type: str = "request"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `sender_id` | `str` | ID của agent gửi |
| `receiver_id` | `str` | ID của agent nhận |
| `content` | `str` | Nội dung message |
| `message_type` | `str` | "request", "response", "notification" |
| `metadata` | `Dict` | Metadata bổ sung |
| `timestamp` | `datetime` | Thời gian gửi |

## Example: Research Team

```python
import asyncio
from agents_framework.teams.team import Team
from agents_framework.teams.registry import AgentRegistry
from agents_framework.teams.patterns.hierarchical import HierarchicalPattern

async def main():
    # Create agents
    researcher = create_researcher_agent()
    writer = create_writer_agent()
    editor = create_editor_agent()

    # Register
    registry = AgentRegistry()
    registry.register(researcher, "researcher", role="researcher")
    registry.register(writer, "writer", role="writer")
    registry.register(editor, "editor", role="editor")

    # Create team
    pattern = HierarchicalPattern(
        supervisor="editor",
        workers=["researcher", "writer"],
    )

    team = Team(
        name="content-team",
        agents={
            "researcher": researcher,
            "writer": writer,
            "editor": editor,
        },
        pattern=pattern,
    )

    # Run
    result = await team.run("Viết bài về Machine Learning")
    print(result.final_output)

asyncio.run(main())
```
