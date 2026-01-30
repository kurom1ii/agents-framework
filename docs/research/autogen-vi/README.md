# Nghiên cứu Framework AutoGen

> **AutoGen** là một framework được phát triển bởi Microsoft để tạo các ứng dụng AI đa tác tử có khả năng hoạt động tự chủ hoặc làm việc cùng con người.

**Kho mã nguồn:** [github.com/microsoft/autogen](https://github.com/microsoft/autogen)

**Tài liệu:** [microsoft.github.io/autogen](https://microsoft.github.io/autogen/)

**Giấy phép:** MIT (Mã nguồn), CC-BY-4.0 (Tài liệu)

**Yêu cầu:** Python 3.10 trở lên

---

## Mục lục

1. [Tổng quan](#tổng-quan)
2. [Kiến trúc cốt lõi](#kiến-trúc-cốt-lõi)
3. [Các thành phần chính](#các-thành-phần-chính)
4. [Tính năng độc đáo](#tính-năng-độc-đáo)
5. [Điểm mạnh và điểm yếu](#điểm-mạnh-và-điểm-yếu)
6. [Trường hợp sử dụng](#trường-hợp-sử-dụng)
7. [Bắt đầu nhanh](#bắt-đầu-nhanh)
8. [Tài liệu liên quan](#tài-liệu-liên-quan)

---

## Tổng quan

AutoGen là framework mã nguồn mở của Microsoft để xây dựng các ứng dụng AI đa tác tử. Nó cho phép các nhà phát triển tạo ra các hệ thống nơi nhiều tác tử AI cộng tác, giao tiếp và làm việc cùng nhau (hoặc với con người) để hoàn thành các tác vụ phức tạp.

### Đặc điểm chính

- **Điều phối đa tác tử**: Các tác tử có thể gọi các tác tử khác như công cụ
- **Con người trong vòng lặp**: Hỗ trợ hàng đầu cho giám sát và can thiệp của con người
- **Thực thi mã**: Thực thi an toàn mã được tạo qua Docker hoặc môi trường cục bộ
- **Các mẫu linh hoạt**: Hỗ trợ nhiều mẫu hội thoại khác nhau (round-robin, dựa trên bộ chọn, swarm)
- **Tùy chọn không cần mã**: AutoGen Studio cung cấp giao diện đồ họa để xây dựng quy trình làm việc

### Thông tin phiên bản

AutoGen có hai phiên bản chính với sự khác biệt đáng kể về kiến trúc:

| Phiên bản | Kiến trúc | Trạng thái |
|-----------|-----------|------------|
| v0.2.x | Đồng bộ, dựa trên `llm_config` | Ổn định, cũ |
| v0.4+ | Bất đồng bộ, hướng sự kiện, dựa trên `model_client` | Hiện tại, khuyến nghị |

---

## Kiến trúc cốt lõi

AutoGen tuân theo kiến trúc phân lớp:

```
+------------------------------------------+
|              AutoGen Studio              |
|      (Giao diện đồ họa không cần mã)     |
+------------------------------------------+
|              AgentChat API               |
|      (Cấp cao, các mẫu có định kiến)     |
+------------------------------------------+
|              Core API                    |
|      (Framework actor hướng sự kiện)     |
+------------------------------------------+
|              Extensions                  |
|   (LLM clients, bộ thực thi mã, công cụ) |
+------------------------------------------+
```

### Tác tử có thể hội thoại

Trừu tượng cơ bản trong AutoGen là **Tác tử có thể hội thoại** - các thực thể có thể:
- Gửi và nhận tin nhắn một cách tự chủ
- Duy trì trạng thái hội thoại
- Thực thi các hành động và sử dụng công cụ
- Giao tiếp với bất kỳ tác tử nào khác

### Các mẫu đa tác tử

| Mẫu | Mô tả | Trường hợp sử dụng |
|-----|-------|-------------------|
| Hai tác tử | Trao đổi qua lại đơn giản | Tạo mã, Hỏi đáp |
| Round Robin | Thứ tự lượt cố định | Tranh luận, chu kỳ đánh giá |
| Selector | LLM chọn người nói tiếp theo | Quy trình làm việc phức tạp |
| Swarm | Chuyển giao dựa trên công cụ | Dịch vụ khách hàng |
| Lồng nhau | Nhóm trong nhóm | Tác vụ phân cấp |

Xem [architecture.md](./architecture.md) để biết tài liệu kiến trúc chi tiết.

---

## Các thành phần chính

### Các loại tác tử

| Thành phần | Mô tả | Phiên bản |
|------------|-------|-----------|
| `AssistantAgent` | Tác tử mục đích chung được hỗ trợ bởi LLM | v0.2, v0.4 |
| `UserProxyAgent` | Đại diện người dùng / xử lý đầu vào người dùng | v0.2, v0.4 |
| `CodeExecutorAgent` | Thực thi các khối mã | v0.4 |
| `ConversableAgent` | Lớp cơ sở cho các tác tử tùy chỉnh | v0.2 |
| `BaseChatAgent` | Lớp cơ sở cho các tác tử tùy chỉnh | v0.4 |

### Các loại nhóm (v0.4)

| Thành phần | Mô tả |
|------------|-------|
| `RoundRobinGroupChat` | Các tác tử nói theo thứ tự cố định |
| `SelectorGroupChat` | LLM chọn người nói tiếp theo |
| `Swarm` | Các tác tử chuyển giao qua công cụ |

### Điều phối (v0.2)

| Thành phần | Mô tả |
|------------|-------|
| `GroupChat` | Chứa hội thoại đa tác tử |
| `GroupChatManager` | Điều phối luồng tin nhắn |

### Bộ thực thi mã

| Bộ thực thi | Mô tả | Độ an toàn |
|-------------|-------|------------|
| `DockerCommandLineCodeExecutor` | Thực thi trong container Docker | Cao |
| `LocalCommandLineCodeExecutor` | Thực thi trực tiếp trên máy chủ | Thấp |

Xem [components.md](./components.md) để biết tài liệu thành phần chi tiết.

---

## Tính năng độc đáo

### 1. Các mẫu con người trong vòng lặp

AutoGen ưu tiên giám sát của con người với nhiều mẫu can thiệp:

```python
# Phê duyệt của người dùng trước khi tiếp tục
user_proxy = UserProxyAgent("user", description="Con người để phê duyệt")

def selector_with_approval(messages):
    if messages[-1].source == "planner":
        return "user"  # Lấy phê duyệt của con người
    if "approve" in messages[-1].content.lower():
        return "worker"  # Tiếp tục công việc
    return "planner"  # Sửa đổi kế hoạch
```

### 2. Khả năng thực thi mã

Thực thi mã an toàn với cách ly Docker:

```python
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

async with DockerCommandLineCodeExecutor(work_dir="coding") as executor:
    result = await executor.execute_code_blocks(
        code_blocks=[CodeBlock(language="python", code="print('An toàn!')")],
        cancellation_token=CancellationToken(),
    )
```

### 3. Hội thoại lồng nhau

Các nhóm có thể chứa các nhóm bên trong cho quy trình làm việc phân cấp:

```python
class NestedTeamAgent(BaseChatAgent):
    def __init__(self, name, inner_team):
        self._inner_team = inner_team

    async def on_messages(self, messages, cancellation_token):
        result = await self._inner_team.run(task=messages)
        return Response(chat_message=result.messages[-1])
```

### 4. Tùy chỉnh tác tử

Tác tử tùy chỉnh với toàn quyền kiểm soát hành vi:

```python
class CustomAgent(BaseChatAgent):
    async def on_messages(self, messages, cancellation_token):
        # Logic tùy chỉnh
        return Response(chat_message=TextMessage(content="Phản hồi tùy chỉnh"))

    async def on_reset(self, cancellation_token):
        # Đặt lại trạng thái
        pass
```

### 5. Gọi công cụ/hàm

Các tác tử có thể sử dụng công cụ bên ngoài:

```python
def get_weather(city: str) -> str:
    """Lấy thời tiết cho một thành phố."""
    return f"72F và nắng ở {city}"

assistant = AssistantAgent(
    "assistant",
    model_client=model_client,
    tools=[get_weather],
)
```

### 6. Tích hợp MCP

Kết nối với các máy chủ Model Context Protocol:

```python
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

workbench = McpWorkbench(
    StdioServerParams(command="npx", args=["@playwright/mcp"])
)

assistant = AssistantAgent("web_agent", tools=workbench.tools)
```

Xem [patterns.md](./patterns.md) để biết tài liệu mẫu chi tiết.

---

## Điểm mạnh và điểm yếu

### Điểm mạnh

| Điểm mạnh | Mô tả |
|-----------|-------|
| **Hỗ trợ từ Microsoft** | Hỗ trợ doanh nghiệp mạnh mẽ và phát triển tích cực |
| **Kiến trúc linh hoạt** | Hỗ trợ nhiều mẫu đa tác tử đa dạng |
| **Con người trong vòng lặp** | Hỗ trợ hàng đầu cho giám sát của con người |
| **Thực thi mã** | Môi trường thực thi an toàn dựa trên Docker |
| **Tài liệu phong phú** | Tài liệu, ví dụ và hướng dẫn mở rộng |
| **AutoGen Studio** | Giao diện đồ họa không cần mã để tạo mẫu nhanh |
| **Không phụ thuộc mô hình** | Hoạt động với OpenAI, Azure và các API tương thích |
| **Lưu trữ trạng thái** | Lưu và tiếp tục trạng thái tác tử/nhóm |
| **Hỗ trợ streaming** | Đầu ra thời gian thực cho các tác vụ chạy lâu |
| **Cộng đồng tích cực** | Hệ sinh thái và cộng đồng đang phát triển |

### Điểm yếu

| Điểm yếu | Mô tả |
|----------|-------|
| **Thay đổi đột phá** | Di chuyển từ v0.2 sang v0.4 đòi hỏi tái cấu trúc đáng kể |
| **Độ phức tạp** | Nhiều mẫu có thể khiến người mới bối rối |
| **Đường cong học tập** | Hiểu khi nào sử dụng mẫu nào |
| **Ưu tiên bất đồng bộ (v0.4)** | Yêu cầu kiến thức async/await |
| **Nhầm lẫn phiên bản** | Hai phiên bản chính với các API khác nhau |
| **Phụ thuộc Docker** | Khuyến nghị cho thực thi mã an toàn nhưng thêm độ phức tạp |
| **Chi phí token** | Hội thoại đa tác tử tiêu thụ nhiều token hơn |
| **Gỡ lỗi** | Tương tác đa tác tử có thể khó gỡ lỗi |

### So sánh với các framework khác

| Khía cạnh | AutoGen | LangChain | CrewAI |
|-----------|---------|-----------|--------|
| Trọng tâm | Hội thoại đa tác tử | Chuỗi và tác tử | Đội ngũ dựa trên vai trò |
| Kiến trúc | Actor hướng sự kiện | Chuỗi tuần tự | Dựa trên tác vụ |
| Con người trong vòng lặp | Xuất sắc | Tốt | Tốt |
| Thực thi mã | Tích hợp sẵn (Docker) | Qua công cụ | Hạn chế |
| Đường cong học tập | Trung bình-Cao | Trung bình | Thấp |
| Tính linh hoạt | Rất cao | Cao | Trung bình |
| Tài liệu | Xuất sắc | Xuất sắc | Tốt |

---

## Trường hợp sử dụng

### Phù hợp cho

1. **Quy trình làm việc đa tác tử phức tạp**
   - Giải quyết vấn đề cộng tác
   - Điều phối hệ thống chuyên gia
   - Tác vụ nghiên cứu và phân tích

2. **Tạo và thực thi mã**
   - Trợ lý viết mã tự động
   - Đường ống phân tích dữ liệu
   - Tạo và kiểm tra script

3. **Hệ thống AI được giám sát bởi con người**
   - Quy trình đánh giá nội dung
   - Hệ thống phê duyệt quyết định
   - Hoàn thành tác vụ có hướng dẫn

4. **Tranh luận/Thảo luận cộng tác**
   - Phân tích đa góc nhìn
   - Phê bình và hoàn thiện
   - Hệ thống động não

5. **Dịch vụ khách hàng**
   - Định tuyến đa phòng ban (Swarm)
   - Xử lý leo thang
   - Chuyển giao tác tử chuyên biệt

### Ít phù hợp cho

1. **Tác vụ đơn tác tử đơn giản** - Chi phí có thể không hợp lý
2. **Thời gian thực độ trễ thấp** - Đa tác tử thêm độ trễ
3. **Môi trường tài nguyên hạn chế** - Yêu cầu Docker
4. **Quy trình làm việc xác định** - Lựa chọn dựa trên LLM thêm tính biến đổi

---

## Bắt đầu nhanh

### Cài đặt

```bash
# Gói cốt lõi
pip install -U "autogen-agentchat" "autogen-ext[openai]"

# Với hỗ trợ Docker
pip install -U "autogen-ext[docker]"

# AutoGen Studio (Giao diện đồ họa)
pip install -U "autogenstudio"
```

### Hello World (v0.4)

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="Bạn là một trợ lý hữu ích.",
    )

    result = await assistant.run(task="Nói 'Hello World!'")
    print(result.messages[-1].content)

    await model_client.close()

asyncio.run(main())
```

### Trò chuyện hai tác tử (v0.2)

```python
import autogen

config_list = [{"model": "gpt-4o", "api_key": "YOUR_KEY"}]
llm_config = {"config_list": config_list}

assistant = autogen.AssistantAgent("assistant", llm_config=llm_config)
user_proxy = autogen.UserProxyAgent("user", human_input_mode="NEVER")

user_proxy.initiate_chat(assistant, message="Kể cho tôi một câu chuyện cười")
```

### Trò chuyện nhóm (v0.4)

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    client = OpenAIChatCompletionClient(model="gpt-4o")

    writer = AssistantAgent("writer", model_client=client, system_message="Viết nội dung.")
    critic = AssistantAgent("critic", model_client=client, system_message="Phê bình. Nói APPROVE khi hoàn thành.")

    team = RoundRobinGroupChat([writer, critic], termination_condition=TextMentionTermination("APPROVE"))

    await Console(team.run_stream(task="Viết một bài haiku về AI"))

asyncio.run(main())
```

Xem [examples.md](./examples.md) để biết thêm ví dụ toàn diện.

---

## Tài liệu liên quan

- [Chi tiết kiến trúc](./architecture.md) - Kiến trúc cốt lõi và các mẫu thiết kế
- [Tham khảo thành phần](./components.md) - Tất cả các loại tác tử và nhóm
- [Các mẫu hội thoại](./patterns.md) - Các mẫu và quy trình làm việc đa tác tử
- [Ví dụ mã](./examples.md) - Các mẫu mã hoạt động

---

## Tài nguyên bên ngoài

- [Tài liệu chính thức](https://microsoft.github.io/autogen/)
- [Kho mã GitHub](https://github.com/microsoft/autogen)
- [AutoGen Studio](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/index.html)
- [Hướng dẫn di chuyển (v0.2 sang v0.4)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/migration-guide.html)
- [Gói PyPI](https://pypi.org/project/autogen-agentchat/)

---

*Cập nhật lần cuối: Tháng 1 năm 2025*
