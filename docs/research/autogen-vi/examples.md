# Ví dụ mã AutoGen

## Cài đặt

```bash
# Gói cốt lõi
pip install -U "autogen-agentchat" "autogen-ext[openai]"

# Với thực thi mã Docker
pip install -U "autogen-ext[docker]"

# AutoGen Studio (Giao diện đồ họa)
pip install -U "autogenstudio"
autogenstudio ui --port 8080 --appdir ./my-app
```

## Thiết lập hội thoại cơ bản

### Hello World (v0.4)

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    # Tạo model client
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    # Tạo trợ lý
    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="Bạn là một trợ lý AI hữu ích.",
    )

    # Chạy một tác vụ đơn giản
    result = await assistant.run(task="Nói 'Hello World!'")
    print(result.messages[-1].content)

    # Dọn dẹp
    await model_client.close()

asyncio.run(main())
```

### Hội thoại hai tác tử (v0.2)

```python
import autogen

# Cấu hình
config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
    filter_dict={"model": ["gpt-4", "gpt-4o", "gpt-3.5-turbo"]},
)

llm_config = {"config_list": config_list, "cache_seed": 42}

# Tạo Trợ lý
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="Bạn là một trợ lý AI hữu ích.",
)

# Tạo User Proxy
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
    system_message="Một quản trị viên con người.",
)

# Bắt đầu hội thoại
chat_result = user_proxy.initiate_chat(
    assistant,
    message="Viết một hàm Python để tính giai thừa.",
    clear_history=True,
    silent=False,
)

# Truy cập kết quả
print(chat_result.summary)
print(chat_result.chat_history)
print(chat_result.cost)
```

### Hai tác tử với thực thi mã (v0.4)

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o", seed=42, temperature=0)

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

    stream = team.run_stream(task="Viết một script Python để in 'Hello, world!'")
    await Console(stream)

    await model_client.close()

asyncio.run(main())
```

## Các mẫu trò chuyện nhóm

### Trò chuyện nhóm Round Robin

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    # Tạo tác tử chính
    primary_agent = AssistantAgent(
        "primary",
        model_client=model_client,
        system_message="Bạn là một trợ lý AI hữu ích.",
    )

    # Tạo tác tử phê bình
    critic_agent = AssistantAgent(
        "critic",
        model_client=model_client,
        system_message="Cung cấp phản hồi xây dựng. Phản hồi với 'APPROVE' khi hài lòng.",
    )

    # Tạo nhóm với điều kiện kết thúc
    text_termination = TextMentionTermination("APPROVE")
    team = RoundRobinGroupChat(
        [primary_agent, critic_agent],
        termination_condition=text_termination,
    )

    # Chạy và hiển thị
    await Console(team.run_stream(task="Viết một bài haiku về lập trình"))

asyncio.run(main())
```

### Trò chuyện nhóm bộ chọn

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    # Tác tử lập kế hoạch
    planner = AssistantAgent(
        "planner",
        model_client=model_client,
        system_message="""Bạn là một tác tử lập kế hoạch.
        Chia nhỏ các tác vụ phức tạp thành các bước.
        Gán mỗi bước cho tác tử phù hợp.
        Nói 'DONE' khi tất cả các bước hoàn thành.""",
        description="Lập kế hoạch và điều phối công việc giữa các tác tử.",
    )

    # Tác tử viết mã
    coder = AssistantAgent(
        "coder",
        model_client=model_client,
        system_message="""Bạn là một tác tử viết mã.
        Viết mã Python sạch, có tài liệu.
        Làm theo hướng dẫn của người lập kế hoạch.""",
        description="Viết mã Python để triển khai giải pháp.",
    )

    # Tác tử đánh giá
    reviewer = AssistantAgent(
        "reviewer",
        model_client=model_client,
        system_message="""Bạn là một tác tử đánh giá mã.
        Đánh giá mã để tìm lỗi, phong cách và cải tiến.
        Cung cấp phản hồi cụ thể.""",
        description="Đánh giá mã và cung cấp phản hồi.",
    )

    # Prompt bộ chọn tùy chỉnh
    selector_prompt = """Chọn một tác tử để thực hiện tác vụ tiếp theo.

{roles}

Hội thoại hiện tại:
{history}

Đọc hội thoại trên. Chọn tác tử tiếp theo từ {participants}.
Đảm bảo người lập kế hoạch gán tác vụ trước khi các tác tử khác bắt đầu làm việc.
Chỉ chọn một tác tử.
"""

    team = SelectorGroupChat(
        participants=[planner, coder, reviewer],
        model_client=model_client,
        selector_prompt=selector_prompt,
        termination_condition=TextMentionTermination("DONE"),
    )

    await Console(team.run_stream(
        task="Tạo một hàm Python sắp xếp danh sách bằng quicksort"
    ))

asyncio.run(main())
```

### Mẫu Swarm

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import Swarm
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    # Tác tử phân loại - điểm tiếp xúc đầu tiên
    triage = AssistantAgent(
        "Triage",
        model_client=model_client,
        handoffs=["TechSupport", "Billing", "General"],
        system_message="""Bạn là một tác tử phân loại.
        Xác định loại trợ giúp người dùng cần.
        Chuyển giao cho TechSupport cho các vấn đề kỹ thuật.
        Chuyển giao cho Billing cho các vấn đề thanh toán.
        Chuyển giao cho General cho các câu hỏi khác.""",
    )

    tech_support = AssistantAgent(
        "TechSupport",
        model_client=model_client,
        handoffs=["Triage"],
        system_message="""Bạn là một tác tử hỗ trợ kỹ thuật.
        Giúp người dùng với các vấn đề kỹ thuật.
        Chuyển giao lại cho Triage nếu đây không phải vấn đề kỹ thuật.""",
    )

    billing = AssistantAgent(
        "Billing",
        model_client=model_client,
        handoffs=["Triage"],
        system_message="""Bạn là một tác tử thanh toán.
        Giúp người dùng với các vấn đề thanh toán và tài khoản.
        Chuyển giao lại cho Triage nếu đây không phải vấn đề thanh toán.""",
    )

    general = AssistantAgent(
        "General",
        model_client=model_client,
        handoffs=["Triage"],
        system_message="""Bạn là một tác tử hỗ trợ chung.
        Giúp với các thắc mắc chung.""",
    )

    termination = MaxMessageTermination(10)
    team = Swarm(
        [triage, tech_support, billing, general],
        termination_condition=termination,
    )

    stream = team.run_stream(task="Tôi không thể đăng nhập vào tài khoản và tôi cũng có câu hỏi về thanh toán")
    async for message in stream:
        print(f"[{message.source}]: {message.content}" if hasattr(message, 'content') else message)

asyncio.run(main())
```

## Tác tử thực thi mã

### Bộ thực thi mã Docker

```python
import asyncio
from pathlib import Path
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

async def main():
    work_dir = Path("coding")
    work_dir.mkdir(exist_ok=True)

    async with DockerCommandLineCodeExecutor(work_dir=work_dir) as executor:
        # Thực thi mã Python
        result = await executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(
                    language="python",
                    code="""
import numpy as np
print(f"Phiên bản Numpy: {np.__version__}")
arr = np.array([1, 2, 3, 4, 5])
print(f"Mảng: {arr}")
print(f"Tổng: {arr.sum()}")
"""
                ),
            ],
            cancellation_token=CancellationToken(),
        )
        print(result)

asyncio.run(main())
```

### Bộ thực thi mã cục bộ

```python
import asyncio
from pathlib import Path
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

async def main():
    work_dir = Path("coding")
    work_dir.mkdir(exist_ok=True)

    executor = LocalCommandLineCodeExecutor(work_dir=work_dir)

    result = await executor.execute_code_blocks(
        code_blocks=[
            CodeBlock(language="python", code="print('Xin chào từ bộ thực thi cục bộ!')"),
        ],
        cancellation_token=CancellationToken(),
    )
    print(result)

asyncio.run(main())
```

### Quy trình làm việc thực thi mã đầy đủ

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    # Trợ lý viết mã
    assistant = AssistantAgent(
        name="coder",
        model_client=model_client,
        system_message="""Bạn là một lập trình viên Python chuyên nghiệp.
        Viết mã sạch, hiệu quả với xử lý lỗi đúng cách.
        Bao gồm chú thích giải thích mã của bạn.
        Trả lời 'TERMINATE' khi tác vụ hoàn thành.""",
    )

    # Bộ thực thi mã với cách ly Docker
    async with DockerCommandLineCodeExecutor(work_dir="coding") as docker_executor:
        executor = CodeExecutorAgent(
            name="executor",
            code_executor=docker_executor,
        )

        termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(10)
        team = RoundRobinGroupChat(
            [assistant, executor],
            termination_condition=termination,
        )

        await Console(team.run_stream(
            task="Viết một script Python tải xuống trang web và đếm tần suất từ"
        ))

asyncio.run(main())
```

## Ví dụ sử dụng công cụ

### Đăng ký công cụ cơ bản

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Định nghĩa công cụ
def get_weather(city: str) -> str:
    """Lấy thời tiết hiện tại cho một thành phố.

    Args:
        city: Tên thành phố

    Returns:
        Mô tả thời tiết
    """
    # Dữ liệu thời tiết mô phỏng
    weather_data = {
        "tokyo": "72F và nắng",
        "london": "55F và mưa",
        "new york": "65F và nhiều mây",
    }
    return weather_data.get(city.lower(), f"Dữ liệu thời tiết không có sẵn cho {city}")

def calculate(expression: str) -> str:
    """Tính toán một biểu thức toán học.

    Args:
        expression: Biểu thức toán cần đánh giá

    Returns:
        Kết quả tính toán
    """
    try:
        result = eval(expression)
        return f"Kết quả: {result}"
    except Exception as e:
        return f"Lỗi: {e}"

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=[get_weather, calculate],
        system_message="Bạn là một trợ lý hữu ích với công cụ thời tiết và máy tính.",
    )

    result = await assistant.run(
        task="Thời tiết ở Tokyo thế nào? Và 15 * 24 + 38 bằng bao nhiêu?"
    )

    for message in result.messages:
        print(f"[{message.source}]: {message.content}")

asyncio.run(main())
```

### Đăng ký công cụ (v0.2)

```python
from autogen.agentchat import AssistantAgent, UserProxyAgent, register_function

llm_config = {
    "config_list": [{"model": "gpt-4o", "api_key": "sk-xxx"}],
    "seed": 42,
    "temperature": 0,
}

tool_caller = AssistantAgent(
    name="tool_caller",
    system_message="Bạn là một trợ lý hữu ích. Sử dụng công cụ khi cần.",
    llm_config=llm_config,
    max_consecutive_auto_reply=1,
)

tool_executor = UserProxyAgent(
    name="tool_executor",
    human_input_mode="NEVER",
    code_execution_config=False,
    llm_config=False,
)

def get_weather(city: str) -> str:
    """Lấy thời tiết cho một thành phố."""
    return f"Thời tiết ở {city} là 72 độ và nắng."

# Đăng ký hàm với cả hai tác tử
register_function(get_weather, caller=tool_caller, executor=tool_executor)

# Vòng lặp tương tác
while True:
    user_input = input("User: ")
    if user_input == "exit":
        break

    chat_result = tool_executor.initiate_chat(
        tool_caller,
        message=user_input,
        summary_method="reflection_with_llm",
    )
    print("Assistant:", chat_result.summary)
```

## Ví dụ con người trong vòng lặp

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    planner = AssistantAgent(
        "planner",
        model_client=model_client,
        system_message="Bạn là người lập kế hoạch. Tạo các kế hoạch từng bước.",
        description="Tạo các kế hoạch cho tác vụ.",
    )

    worker = AssistantAgent(
        "worker",
        model_client=model_client,
        system_message="Bạn thực hiện các tác vụ theo kế hoạch.",
        description="Thực hiện các tác vụ đã lên kế hoạch.",
    )

    user_proxy = UserProxyAgent(
        "user",
        description="Người dùng con người để phê duyệt.",
    )

    def selector_with_approval(messages):
        """Bộ chọn tùy chỉnh yêu cầu phê duyệt người dùng sau khi lập kế hoạch."""
        if not messages:
            return "planner"

        last_msg = messages[-1]

        # Sau khi người lập kế hoạch nói, lấy phê duyệt người dùng
        if last_msg.source == "planner":
            return "user"

        # Nếu người dùng phê duyệt, tiếp tục đến worker
        if last_msg.source == "user":
            if "approve" in last_msg.content.lower():
                return "worker"
            else:
                return "planner"  # Sửa đổi kế hoạch

        # Sau worker, kiểm tra với người lập kế hoạch
        if last_msg.source == "worker":
            return "planner"

        return None

    team = SelectorGroupChat(
        participants=[planner, worker, user_proxy],
        model_client=model_client,
        selector_func=selector_with_approval,
        termination_condition=MaxMessageTermination(10),
    )

    await Console(team.run_stream(task="Lập kế hoạch và thực hiện quy trình phân tích dữ liệu"))

asyncio.run(main())
```

## Ví dụ duyệt web MCP

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

async def main():
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        model_client_stream=True,
    )

    # Kết nối đến máy chủ Playwright MCP để duyệt web
    # Chạy trước: npx @playwright/mcp
    workbench = McpWorkbench(
        StdioServerParams(
            command="npx",
            args=["@playwright/mcp"],
        )
    )

    assistant = AssistantAgent(
        name="web_assistant",
        model_client=model_client,
        tools=workbench.tools,
        system_message="Bạn là một trợ lý duyệt web. Sử dụng các công cụ để điều hướng và tương tác với các trang web.",
    )

    await Console(assistant.run_stream(
        task="Đi đến news.ycombinator.com và lấy 3 tiêu đề bài viết hàng đầu"
    ))

asyncio.run(main())
```

## Ví dụ lưu trữ trạng thái

```python
import asyncio
import json
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    agent1 = AssistantAgent("agent1", model_client=model_client)
    agent2 = AssistantAgent("agent2", model_client=model_client)

    team = RoundRobinGroupChat(
        [agent1, agent2],
        termination_condition=MaxMessageTermination(5),
    )

    # Chạy phần đầu tiên
    result = await team.run(task="Đếm từ 1 đến 10, mỗi lần một số")
    print("Lần chạy đầu tiên hoàn thành")

    # Lưu trạng thái
    state = await team.save_state()
    with open("team_state.json", "w") as f:
        json.dump(state, f)

    # ... sau đó, trong phiên mới ...

    # Tải trạng thái và tiếp tục
    with open("team_state.json", "r") as f:
        state = json.load(f)

    await team.load_state(state)

    # Tiếp tục từ nơi đã dừng
    result = await team.run()  # Không cần tác vụ, tiếp tục trước đó
    print("Lần chạy tiếp tục hoàn thành")

asyncio.run(main())
```

## Ví dụ đa mô hình

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    # Các mô hình khác nhau cho các tác tử khác nhau
    gpt4_client = OpenAIChatCompletionClient(model="gpt-4o")
    gpt35_client = OpenAIChatCompletionClient(model="gpt-3.5-turbo")

    # GPT-4 cho suy luận phức tạp
    reasoner = AssistantAgent(
        "reasoner",
        model_client=gpt4_client,
        system_message="Bạn là chuyên gia về suy luận và phân tích phức tạp.",
    )

    # GPT-3.5 cho các tác vụ đơn giản hơn
    helper = AssistantAgent(
        "helper",
        model_client=gpt35_client,
        system_message="Bạn giúp với các tác vụ đơn giản và định dạng.",
    )

    team = RoundRobinGroupChat(
        [reasoner, helper],
        termination_condition=MaxMessageTermination(6),
    )

    result = await team.run(task="Giải thích máy tính lượng tử một cách đơn giản")

    # Dọn dẹp
    await gpt4_client.close()
    await gpt35_client.close()

asyncio.run(main())
```
