# Các thành phần AutoGen

## Các loại tác tử cốt lõi

### AssistantAgent

Tác tử chính được hỗ trợ bởi LLM cho các tác vụ mục đích chung.

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-4o")

assistant = AssistantAgent(
    name="assistant",
    model_client=model_client,
    system_message="Bạn là một trợ lý AI hữu ích.",
    tools=[my_tool],  # Tùy chọn: các công cụ đã đăng ký
    description="Một trợ lý hữu ích có thể trả lời câu hỏi.",
)
```

**Tham số chính:**
- `name`: Định danh duy nhất cho tác tử
- `model_client`: LLM client để sử dụng
- `system_message`: Hướng dẫn xác định hành vi tác tử
- `tools`: Danh sách các công cụ/hàm có thể gọi
- `description`: Được sử dụng bởi bộ chọn để hiểu khả năng tác tử
- `max_consecutive_auto_reply`: Giới hạn phản hồi tự động
- `reflect_on_tool_use`: Có suy ngẫm về kết quả công cụ hay không

**Khả năng:**
- Tạo ngôn ngữ tự nhiên
- Gọi công cụ/hàm
- Hỗ trợ đầu vào đa phương thức (với các mô hình có khả năng nhìn)
- Phản hồi streaming

### UserProxyAgent

Đại diện cho người dùng trong hội thoại, thu thập đầu vào hoặc mô phỏng hành vi người dùng.

```python
from autogen_agentchat.agents import UserProxyAgent

# v0.4 - Đơn giản hóa
user_proxy = UserProxyAgent("user_proxy")

# v0.2 - Cấu hình đầy đủ
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",  # ALWAYS, TERMINATE, hoặc NEVER
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
    system_message="Một quản trị viên con người.",
)
```

**Chế độ đầu vào con người (v0.2):**
- `ALWAYS`: Luôn yêu cầu đầu vào con người
- `TERMINATE`: Chỉ yêu cầu tại điều kiện kết thúc
- `NEVER`: Không bao giờ yêu cầu đầu vào con người (chế độ tự chủ)

### CodeExecutorAgent

Tác tử chuyên biệt để thực thi các khối mã.

```python
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

code_executor = CodeExecutorAgent(
    name="code_executor",
    code_executor=LocalCommandLineCodeExecutor(work_dir="coding"),
)
```

### ConversableAgent (v0.2)

Lớp cơ sở cho tất cả các tác tử có thể hội thoại trong v0.2.

```python
from autogen.agentchat import ConversableAgent

agent = ConversableAgent(
    name="conversable_agent",
    system_message="Bạn là một trợ lý hữu ích.",
    llm_config=llm_config,
    code_execution_config={"work_dir": "coding"},
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
)

# Đăng ký hàm phản hồi tùy chỉnh
def reply_func(recipient, messages, sender, config):
    # Logic tùy chỉnh
    return True, "Phản hồi tùy chỉnh"

agent.register_reply([ConversableAgent], reply_func, position=0)
```

## Các thành phần nhóm

### GroupChat (v0.2)

Cho phép nhiều tác tử tham gia vào một hội thoại chung.

```python
import autogen

groupchat = autogen.GroupChat(
    agents=[agent1, agent2, agent3],
    messages=[],
    max_round=12,
    speaker_selection_method="auto",  # hoặc "round_robin", "random", "manual"
)
```

### GroupChatManager (v0.2)

Điều phối luồng tin nhắn trong GroupChat.

```python
manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
)

# Bắt đầu hội thoại
user_proxy.initiate_chat(
    manager,
    message="Hãy thảo luận về kế hoạch dự án.",
)
```

### RoundRobinGroupChat (v0.4)

Các tác tử thay phiên nhau theo thứ tự cố định.

```python
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination

team = RoundRobinGroupChat(
    [primary_agent, critic_agent],
    termination_condition=TextMentionTermination("APPROVE"),
)

result = await team.run(task="Viết một bài thơ về AI")
```

### SelectorGroupChat (v0.4)

Sử dụng LLM để chọn động người nói tiếp theo.

```python
from autogen_agentchat.teams import SelectorGroupChat

selector_prompt = """Chọn một tác tử để thực hiện tác vụ.
{roles}
Ngữ cảnh hội thoại hiện tại:
{history}
Chọn từ {participants}. Chỉ chọn một tác tử.
"""

team = SelectorGroupChat(
    participants=[planner, coder, reviewer],
    model_client=model_client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=False,
    max_selector_attempts=3,
)
```

**Tham số chính:**
- `participants`: Danh sách các tác tử
- `model_client`: LLM cho việc chọn người nói
- `selector_prompt`: Mẫu prompt tùy chỉnh với `{roles}`, `{participants}`, `{history}`
- `allow_repeated_speaker`: Liệu cùng một tác tử có thể nói liên tiếp
- `selector_func`: Hàm lựa chọn tùy chỉnh tùy chọn

### Swarm (v0.4)

Các tác tử chuyển giao tác vụ cho nhau sử dụng lựa chọn dựa trên công cụ.

```python
from autogen_agentchat.teams import Swarm

agent1 = AssistantAgent(
    "Alice",
    model_client=model_client,
    handoffs=["Bob"],  # Có thể chuyển giao cho Bob
    system_message="Bạn là Alice, một chuyên gia về X.",
)

agent2 = AssistantAgent(
    "Bob",
    model_client=model_client,
    system_message="Bạn là Bob, một chuyên gia về Y.",
)

team = Swarm([agent1, agent2], termination_condition=termination)
```

## Bộ thực thi mã

### LocalCommandLineCodeExecutor

Thực thi mã trực tiếp trên máy chủ.

```python
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from pathlib import Path

work_dir = Path("coding")
work_dir.mkdir(exist_ok=True)

executor = LocalCommandLineCodeExecutor(work_dir=work_dir)

result = await executor.execute_code_blocks(
    code_blocks=[
        CodeBlock(language="python", code="print('Hello, World!')"),
    ],
    cancellation_token=CancellationToken(),
)
```

**Cảnh báo:** Truy cập trực tiếp hệ thống - sử dụng cẩn thận.

### DockerCommandLineCodeExecutor

Thực thi mã trong các container Docker cách ly.

```python
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

async with DockerCommandLineCodeExecutor(work_dir=work_dir) as executor:
    result = await executor.execute_code_blocks(
        code_blocks=[
            CodeBlock(language="python", code="print('Hello, World!')"),
        ],
        cancellation_token=CancellationToken(),
    )
```

**Lợi ích:**
- Cách ly khỏi hệ thống máy chủ
- Hình ảnh Docker có thể tùy chỉnh
- Môi trường thực thi an toàn hơn

## Model Clients

### OpenAIChatCompletionClient

```python
from autogen_ext.models.openai import OpenAIChatCompletionClient

client = OpenAIChatCompletionClient(
    model="gpt-4o",
    # api_key="sk-...",  # Tùy chọn nếu OPENAI_API_KEY đã được đặt
)
```

### AzureOpenAIChatCompletionClient

```python
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

client = AzureOpenAIChatCompletionClient(
    model="gpt-4o",
    azure_endpoint="https://your-endpoint.openai.azure.com/",
    api_version="2024-02-01",
)
```

### Client tùy chỉnh/Tương thích OpenAI

```python
client = OpenAIChatCompletionClient(
    model="your-model",
    base_url="https://your-compatible-api.com/v1",
    model_info={"vision": False, "json_output": True},
)
```

## Điều kiện kết thúc

```python
from autogen_agentchat.conditions import (
    MaxMessageTermination,
    TextMentionTermination,
    ExternalTermination,
)

# Dừng sau 10 tin nhắn
max_msg = MaxMessageTermination(10)

# Dừng khi "TERMINATE" xuất hiện
text_term = TextMentionTermination("TERMINATE")

# Kết hợp với HOẶC
combined = max_msg | text_term

# Kết hợp với VÀ
both_required = max_msg & text_term
```

## Công cụ và gọi hàm

### Định nghĩa công cụ

```python
def get_weather(city: str) -> str:
    """Lấy thời tiết cho một thành phố."""
    return f"Thời tiết ở {city} là 72 độ và nắng."

# Gắn vào tác tử
assistant = AssistantAgent(
    "assistant",
    model_client=model_client,
    tools=[get_weather],
)
```

### PythonCodeExecutionTool

```python
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

code_executor = DockerCommandLineCodeExecutor()
await code_executor.start()

coding_tool = PythonCodeExecutionTool(code_executor)
result = await coding_tool.run_json({"code": code}, cancellation_token)
```

### Đăng ký hàm (v0.2)

```python
from autogen.agentchat import register_function

register_function(
    get_weather,
    caller=tool_caller,
    executor=tool_executor,
)
```

## Bộ nhớ và RAG

### Trừu tượng bộ nhớ (v0.4)

```python
from autogen_agentchat.memory import ChromaDBVectorMemory

memory = ChromaDBVectorMemory(
    collection_name="agent_memory",
    persist_directory="./chroma_db",
)

# Thêm vào tác tử
assistant = AssistantAgent(
    "assistant",
    model_client=model_client,
    memory=memory,
)
```

## Bộ nhớ đệm

### ChatCompletionCache (v0.4)

```python
from autogen_ext.cache import DiskCacheStore, ChatCompletionCache

cache_store = DiskCacheStore(cache_dir="./cache")
cached_client = ChatCompletionCache(
    model_client=model_client,
    cache_store=cache_store,
)
```

## Các thành phần UI

### Đầu ra Console

```python
from autogen_agentchat.ui import Console

# Stream đầu ra ra console
await Console(team.run_stream(task="Tác vụ của bạn"))
```

## Tích hợp MCP

### McpWorkbench

```python
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

# Kết nối đến máy chủ MCP (ví dụ: Playwright)
workbench = McpWorkbench(
    StdioServerParams(
        command="npx",
        args=["@playwright/mcp"],
    )
)

assistant = AssistantAgent(
    "web_assistant",
    model_client=model_client,
    tools=workbench.tools,
)
```
