# Các mẫu AutoGen

## Các mẫu hội thoại

### Mẫu trò chuyện hai tác tử

Mẫu đơn giản nhất cho tương tác tác tử.

```python
from autogen.agentchat import AssistantAgent, UserProxyAgent

# Tạo các tác tử
assistant = AssistantAgent(
    name="assistant",
    system_message="Bạn là một trợ lý AI hữu ích.",
    llm_config=llm_config,
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding"},
)

# Khởi tạo hội thoại
chat_result = user_proxy.initiate_chat(
    assistant,
    message="Viết một script Python để in 'Hello, world!'",
)
```

### Mẫu phản ánh

Một tác tử tạo, một tác tử khác phê bình.

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination

primary_agent = AssistantAgent(
    "primary",
    model_client=model_client,
    system_message="Bạn là một trợ lý AI hữu ích.",
)

critic_agent = AssistantAgent(
    "critic",
    model_client=model_client,
    system_message="Cung cấp phản hồi xây dựng. Phản hồi với 'APPROVE' khi hài lòng.",
)

team = RoundRobinGroupChat(
    [primary_agent, critic_agent],
    termination_condition=TextMentionTermination("APPROVE"),
)

result = await team.run(task="Viết một bài thơ về trí tuệ nhân tạo")
```

### Mẫu sử dụng công cụ

Tác tử sử dụng công cụ để hoàn thành tác vụ.

```python
from autogen_agentchat.agents import AssistantAgent

def get_weather(city: str) -> str:
    """Lấy thời tiết cho một thành phố."""
    return f"Thời tiết ở {city} là 72 độ và nắng."

def get_stock_price(symbol: str) -> str:
    """Lấy giá cổ phiếu cho một mã."""
    return f"Giá cổ phiếu của {symbol} là $150.00"

assistant = AssistantAgent(
    "assistant",
    model_client=model_client,
    tools=[get_weather, get_stock_price],
    system_message="Bạn là một trợ lý hữu ích có quyền truy cập công cụ thời tiết và cổ phiếu.",
)

result = await assistant.run(task="Thời tiết ở Tokyo thế nào? Và giá MSFT là bao nhiêu?")
```

### Mẫu tạo và thực thi mã

```python
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

assistant = AssistantAgent(
    name="assistant",
    system_message="Bạn là một trợ lý hữu ích. Viết tất cả mã bằng Python. Trả lời 'TERMINATE' khi hoàn thành.",
    model_client=model_client,
)

code_executor = CodeExecutorAgent(
    name="code_executor",
    code_executor=LocalCommandLineCodeExecutor(work_dir="coding"),
)

termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(10)
team = RoundRobinGroupChat([assistant, code_executor], termination_condition=termination)

await Console(team.run_stream(task="Viết và chạy một script tính số Fibonacci"))
```

## Các mẫu con người trong vòng lặp

### Mẫu phê duyệt

Người dùng phải phê duyệt trước khi tiếp tục.

```python
from autogen_agentchat.agents import UserProxyAgent

user_proxy = UserProxyAgent(
    "UserProxyAgent",
    description="Một đại diện cho người dùng để phê duyệt hoặc từ chối tác vụ."
)

def selector_func_with_user_proxy(messages):
    if messages[-1].source != planning_agent.name and messages[-1].source != user_proxy.name:
        return planning_agent.name

    if messages[-1].source == planning_agent.name:
        if messages[-2].source == user_proxy.name and "APPROVE" in messages[-1].content.upper():
            return None  # Tiếp tục đến tác tử tiếp theo
        return user_proxy.name  # Lấy phê duyệt người dùng

    if messages[-1].source == user_proxy.name:
        if "APPROVE" not in messages[-1].content.upper():
            return planning_agent.name  # Sửa đổi kế hoạch

    return None

team = SelectorGroupChat(
    [planning_agent, worker_agent, user_proxy],
    model_client=model_client,
    selector_func=selector_func_with_user_proxy,
)
```

### Phê duyệt thực thi mã

```python
from autogen_agentchat.agents import ApprovalRequest, ApprovalResponse

def approval_func(request: ApprovalRequest) -> ApprovalResponse:
    """Yêu cầu xác nhận người dùng trước khi thực thi mã."""
    print(f"Mã cần thực thi:\n{request.code}")
    user_input = input("Phê duyệt? (y/n): ").strip().lower()

    if user_input == 'y':
        return ApprovalResponse(approved=True, reason="Người dùng đã phê duyệt")
    else:
        return ApprovalResponse(approved=False, reason="Người dùng từ chối")

from autogen_ext.teams.magentic_one import MagenticOne

m1 = MagenticOne(client=client, approval_func=approval_func)
result = await Console(m1.run_stream(task="Viết một script Python"))
```

### Chế độ đầu vào con người (v0.2)

```python
# ALWAYS: Luôn yêu cầu đầu vào con người
user_proxy = UserProxyAgent(
    "user",
    human_input_mode="ALWAYS",
)

# TERMINATE: Yêu cầu đầu vào tại điều kiện kết thúc
user_proxy = UserProxyAgent(
    "user",
    human_input_mode="TERMINATE",
    is_termination_msg=lambda x: x.get("content", "").endswith("DONE"),
)

# NEVER: Hoàn toàn tự chủ
user_proxy = UserProxyAgent(
    "user",
    human_input_mode="NEVER",
)
```

## Các mẫu trò chuyện nhóm

### Trò chuyện nhóm Round Robin

Các tác tử nói theo vòng xoay cố định.

```python
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination

agent1 = AssistantAgent("Assistant1", model_client=model_client)
agent2 = AssistantAgent("Assistant2", model_client=model_client)
agent3 = AssistantAgent("Assistant3", model_client=model_client)

termination = MaxMessageTermination(10)
team = RoundRobinGroupChat([agent1, agent2, agent3], termination_condition=termination)

result = await team.run(task="Thảo luận các phương pháp giải quyết biến đổi khí hậu")
```

### Trò chuyện nhóm bộ chọn

Lựa chọn người nói động dựa trên LLM.

```python
from autogen_agentchat.teams import SelectorGroupChat

planner = AssistantAgent(
    "planner",
    model_client=model_client,
    system_message="Bạn là một tác tử lập kế hoạch. Chia nhỏ tác vụ thành các bước.",
    description="Lập kế hoạch và điều phối công việc.",
)

coder = AssistantAgent(
    "coder",
    model_client=model_client,
    system_message="Bạn viết mã Python.",
    description="Viết mã để triển khai giải pháp.",
)

reviewer = AssistantAgent(
    "reviewer",
    model_client=model_client,
    system_message="Bạn đánh giá mã để tìm lỗi và cải tiến.",
    description="Đánh giá và cải thiện chất lượng mã.",
)

selector_prompt = """Chọn một tác tử để thực hiện tác vụ tiếp theo.
{roles}

Hội thoại hiện tại:
{history}

Chọn từ {participants}. Chỉ chọn một tác tử.
Đảm bảo người lập kế hoạch gán tác vụ trước khi các tác tử khác bắt đầu.
"""

team = SelectorGroupChat(
    participants=[planner, coder, reviewer],
    model_client=model_client,
    selector_prompt=selector_prompt,
    termination_condition=TextMentionTermination("DONE"),
)
```

### Hàm bộ chọn tùy chỉnh

```python
def custom_selector(messages):
    """Logic tùy chỉnh cho việc chọn người nói."""
    last_speaker = messages[-1].source if messages else None

    if last_speaker == "planner":
        return "coder"
    elif last_speaker == "coder":
        return "reviewer"
    elif last_speaker == "reviewer":
        return "planner"
    else:
        return "planner"

team = SelectorGroupChat(
    participants=[planner, coder, reviewer],
    model_client=model_client,
    selector_func=custom_selector,
)
```

### Mẫu Swarm

Các tác tử chuyển giao dựa trên khả năng của họ.

```python
from autogen_agentchat.teams import Swarm

alice = AssistantAgent(
    "Alice",
    model_client=model_client,
    handoffs=["Bob", "Charlie"],
    system_message="Bạn là Alice. Chuyển giao cho Bob để viết mã, Charlie để đánh giá.",
)

bob = AssistantAgent(
    "Bob",
    model_client=model_client,
    handoffs=["Alice", "Charlie"],
    system_message="Bạn là Bob người viết mã. Chuyển giao lại cho Alice khi hoàn thành.",
)

charlie = AssistantAgent(
    "Charlie",
    model_client=model_client,
    handoffs=["Alice", "Bob"],
    system_message="Bạn là Charlie người đánh giá.",
)

team = Swarm([alice, bob, charlie], termination_condition=termination)
```

## Các mẫu hội thoại lồng nhau

### Mẫu nhóm lồng nhau

Một nhóm chứa các nhóm bên trong.

```python
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.base import Response

class NestedTeamAgent(BaseChatAgent):
    """Một tác tử chạy một nhóm bên trong."""

    def __init__(self, name: str, inner_team: RoundRobinGroupChat):
        super().__init__(name, description="Chạy một nhóm bên trong")
        self._inner_team = inner_team

    async def on_messages(self, messages, cancellation_token):
        # Chạy nhóm bên trong với các tin nhắn
        result = await self._inner_team.run(
            task=messages,
            cancellation_token=cancellation_token
        )
        # Trả về tin nhắn cuối cùng từ nhóm bên trong
        return Response(
            chat_message=result.messages[-1],
            inner_messages=result.messages[:-1]
        )

    async def on_reset(self, cancellation_token):
        await self._inner_team.reset()

    @property
    def produced_message_types(self):
        return (TextMessage,)

# Tạo nhóm bên trong
inner_agent1 = AssistantAgent("inner1", model_client=model_client)
inner_agent2 = AssistantAgent("inner2", model_client=model_client)
inner_team = RoundRobinGroupChat([inner_agent1, inner_agent2], max_turns=3)

# Tạo tác tử lồng nhau
nested_agent = NestedTeamAgent("nested", inner_team)

# Sử dụng trong nhóm bên ngoài
outer_team = RoundRobinGroupChat([nested_agent, outer_agent], termination_condition=termination)
```

### Tác tử Society of Mind

Cho các kịch bản lồng nhau phức tạp.

```python
from autogen_agentchat.agents import SocietyOfMindAgent

inner_team = RoundRobinGroupChat([agent1, agent2], max_turns=5)

society_agent = SocietyOfMindAgent(
    name="society",
    inner_team=inner_team,
    description="Một nhóm các chuyên gia làm việc cùng nhau",
)
```

## Mẫu tác tử như công cụ

Bọc các tác tử như công cụ có thể gọi cho các tác tử khác.

```python
from autogen_ext.agents import AgentTool

math_expert = AssistantAgent(
    "math_expert",
    model_client=model_client,
    system_message="Bạn là chuyên gia toán học. Giải các bài toán.",
)

chemistry_expert = AssistantAgent(
    "chemistry_expert",
    model_client=model_client,
    system_message="Bạn là chuyên gia hóa học.",
)

# Bọc như công cụ
math_tool = AgentTool(math_expert, return_value_as_last_message=True)
chemistry_tool = AgentTool(chemistry_expert, return_value_as_last_message=True)

# Trợ lý chung với công cụ chuyên gia
assistant = AssistantAgent(
    "assistant",
    model_client=model_client,
    tools=[math_tool, chemistry_tool],
    system_message="Sử dụng các chuyên gia cho câu hỏi chuyên môn.",
)
```

## Các mẫu kết thúc

### Kết thúc từ khóa

```python
termination = TextMentionTermination("TERMINATE")
```

### Giới hạn tin nhắn

```python
termination = MaxMessageTermination(10)
```

### Điều kiện kết hợp

```python
# Một trong hai điều kiện kích hoạt kết thúc
termination = TextMentionTermination("DONE") | MaxMessageTermination(20)

# Cả hai điều kiện đều cần thiết
termination = SomeCondition() & AnotherCondition()
```

### Kết thúc tùy chỉnh

```python
is_termination_msg = lambda x: x.get("content", "").rstrip().endswith("TERMINATE")

agent = AssistantAgent(
    "assistant",
    is_termination_msg=is_termination_msg,
    ...
)
```

## Mẫu lưu trữ trạng thái

```python
# Lưu trạng thái nhóm
state = await team.save_state()

# Lưu trữ trạng thái (ví dụ: vào file)
import json
with open("team_state.json", "w") as f:
    json.dump(state, f)

# Sau đó: Tải và tiếp tục
with open("team_state.json", "r") as f:
    state = json.load(f)

await team.load_state(state)
result = await team.run()  # Tiếp tục từ trạng thái đã lưu
```

## Mẫu Streaming

```python
from autogen_agentchat.ui import Console

# Stream ra console
await Console(team.run_stream(task="Tác vụ của bạn"))

# Xử lý streaming tùy chỉnh
async for message in team.run_stream(task="Tác vụ của bạn"):
    if hasattr(message, 'content'):
        print(f"[{message.source}]: {message.content}")
```

## Mẫu hủy bỏ

```python
from autogen_core import CancellationToken
import asyncio

cancellation_token = CancellationToken()

# Bắt đầu tác vụ
task = asyncio.create_task(
    team.run(task="Tác vụ chạy lâu", cancellation_token=cancellation_token)
)

# Hủy sau thời gian chờ
await asyncio.sleep(30)
cancellation_token.cancel()
```
