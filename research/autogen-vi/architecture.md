# Kiến trúc AutoGen

## Tổng quan

AutoGen là một framework được phát triển bởi Microsoft để tạo các ứng dụng AI đa tác tử có khả năng hoạt động tự chủ hoặc làm việc cùng con người. Framework này đã phát triển đáng kể, với phiên bản 0.4+ đại diện cho việc viết lại từ đầu với kiến trúc bất đồng bộ, hướng sự kiện.

## Kiến trúc phân lớp

AutoGen tuân theo thiết kế kiến trúc phân lớp:

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

### Lớp Core

Core API cung cấp:
- **Truyền tin nhắn**: Giao tiếp bất đồng bộ giữa các tác tử
- **Tác tử hướng sự kiện**: Các tác tử phản hồi các sự kiện và tin nhắn
- **Runtime cục bộ/phân tán**: Hỗ trợ cả thực thi cục bộ và phân tán
- **Mô hình dựa trên actor**: Các tác tử như các actor độc lập với trạng thái riêng

### Lớp AgentChat

AgentChat là API cấp cao được khuyến nghị cho hầu hết người dùng:
- Mặc định trực quan để tạo mẫu nhanh
- Các loại tác tử được xây dựng sẵn (AssistantAgent, UserProxyAgent, CodeExecutorAgent)
- Các mẫu nhóm (RoundRobinGroupChat, SelectorGroupChat, Swarm)
- Hỗ trợ tích hợp cho các quy trình làm việc đa tác tử phổ biến

### Lớp Extensions

Các thành phần có thể cắm cho:
- LLM clients (OpenAI, Azure OpenAI, v.v.)
- Thực thi mã (Docker, Cục bộ)
- Công cụ và gọi hàm
- Tích hợp MCP (Model Context Protocol)

## Khái niệm tác tử có thể hội thoại

Trừu tượng cốt lõi trong AutoGen là **Tác tử có thể hội thoại** - một thực thể có thể gửi và nhận tin nhắn, tham gia hội thoại và thực thi các hành động.

### Đặc điểm chính

1. **Giao tiếp tự chủ**: Các tác tử có thể độc lập gửi và nhận tin nhắn
2. **Có trạng thái**: Mỗi tác tử duy trì lịch sử hội thoại và trạng thái riêng
3. **Hành vi có thể cấu hình**: Các tác tử có thể được tùy chỉnh với thông điệp hệ thống, công cụ và logic phản hồi
4. **Có thể tương tác**: Bất kỳ tác tử nào cũng có thể giao tiếp với bất kỳ tác tử nào khác

### Luồng giao tiếp tác tử

```
┌─────────────────┐         ┌─────────────────┐
│    Tác tử A     │         │    Tác tử B     │
│                 │ tin nhắn│                 │
│  ┌───────────┐  │────────>│  ┌───────────┐  │
│  │ on_message│  │         │  │ on_message│  │
│  └───────────┘  │<────────│  └───────────┘  │
│                 │ phản hồi│                 │
└─────────────────┘         └─────────────────┘
```

## Các mẫu hội thoại đa tác tử

### Hội thoại hai tác tử

Mẫu đơn giản nhất liên quan đến hai tác tử giao tiếp:

```python
# Tác tử A gửi tin nhắn đến Tác tử B
# Tác tử B xử lý và phản hồi
# Giao tiếp tiếp tục cho đến điều kiện kết thúc
```

### Trò chuyện nhóm (Round Robin)

Các tác tử thay phiên nhau theo thứ tự được xác định trước:

```
Tác tử 1 -> Tác tử 2 -> Tác tử 3 -> Tác tử 1 -> ...
```

### Trò chuyện nhóm bộ chọn

Bộ chọn dựa trên LLM chọn động người nói tiếp theo:

```
                 ┌──────────────┐
                 │   Bộ chọn    │
                 │    (LLM)     │
                 └──────┬───────┘
                        │ chọn
        ┌───────────────┼───────────────┐
        v               v               v
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │ Tác tử 1│    │ Tác tử 2│    │ Tác tử 3│
   └─────────┘    └─────────┘    └─────────┘
```

### Mẫu Swarm

Các tác tử chuyển giao tác vụ cho nhau dựa trên lựa chọn công cụ:

```python
agent1 = AssistantAgent(
    "Alice",
    handoffs=["Bob"],  # Có thể chuyển giao cho Bob
    system_message="Bạn là Alice..."
)
```

### Hội thoại lồng nhau

Các nhóm hoặc tác tử có thể chứa các nhóm bên trong, cho phép quy trình làm việc phân cấp:

```
┌───────────────────────────────────────────┐
│              Nhóm bên ngoài               │
│  ┌─────────────────────────────────────┐  │
│  │           Nhóm bên trong            │  │
│  │   Tác tử A <-> Tác tử B <-> Tác tử C│  │
│  └─────────────────────────────────────┘  │
│                    ^                       │
│                    │                       │
│         Tác tử D (Điều phối viên)         │
└───────────────────────────────────────────┘
```

## Điều phối tác tử

### Điều kiện kết thúc

AutoGen cung cấp các điều kiện kết thúc linh hoạt:

```python
from autogen_agentchat.conditions import (
    MaxMessageTermination,    # Dừng sau N tin nhắn
    TextMentionTermination,   # Dừng khi văn bản cụ thể xuất hiện
    ExternalTermination,      # Kích hoạt bên ngoài
)

# Kết hợp điều kiện với | (HOẶC) hoặc & (VÀ)
termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(10)
```

### Xử lý tin nhắn

Các tác tử xử lý tin nhắn thông qua phương thức `on_messages`:

```python
class CustomAgent(BaseChatAgent):
    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken
    ) -> Response:
        # Xử lý tin nhắn
        # Trả về phản hồi
        return Response(chat_message=TextMessage(...))
```

### Quản lý trạng thái

v0.4 giới thiệu lưu trữ trạng thái đúng cách:

```python
# Lưu trạng thái tác tử/nhóm
state = await team.save_state()

# Tiếp tục từ trạng thái đã lưu
await team.load_state(state)
```

## Các mô hình Runtime

### Thực thi đồng bộ

Thực thi chặn truyền thống cho các trường hợp sử dụng đơn giản.

### Thực thi bất đồng bộ

Cách tiếp cận được khuyến nghị trong v0.4:

```python
async def main():
    result = await team.run(task="Tác vụ của bạn ở đây")
```

### Thực thi streaming

Cho đầu ra thời gian thực và kết quả trung gian:

```python
async for message in team.run_stream(task="Tác vụ của bạn"):
    print(message)
```

## So sánh phiên bản (v0.2 vs v0.4)

| Khía cạnh | v0.2 | v0.4 |
|-----------|------|------|
| Kiến trúc | Đồng bộ | Bất đồng bộ, hướng sự kiện |
| Cấu hình mô hình | Dict `llm_config` | `model_client` tường minh |
| Tạo tác tử | `ConversableAgent` + `register_reply` | Lớp `BaseChatAgent` tùy chỉnh |
| Trò chuyện nhóm | `GroupChat` + `GroupChatManager` | `RoundRobinGroupChat`, `SelectorGroupChat` |
| Bộ nhớ đệm | Mặc định bật (`cache_seed`) | Tùy chọn qua `ChatCompletionCache` |
| Trạng thái | Xuất lịch sử thủ công | `save_state`/`load_state` |
| Streaming | Hạn chế | Hỗ trợ đầy đủ qua `run_stream` |

## Nguyên tắc thiết kế

1. **Mô đun hóa**: Các tác tử là mô đun và có thể có các mức độ truy cập thông tin khác nhau
2. **Sự tham gia của con người**: Ưu tiên con người trong vòng lặp để giám sát
3. **An toàn**: Khuyến nghị container Docker cho thực thi mã
4. **Linh hoạt**: Hỗ trợ các topology tác tử đặc thù ứng dụng
5. **Khả năng quan sát**: Ghi log và streaming tích hợp để gỡ lỗi
