# OpenAI Swarm / Agents SDK

## Tổng Quan

**OpenAI Swarm** là một framework thử nghiệm, mang tính giáo dục để khám phá các pattern điều phối multi-agent nhẹ. Nó đã được **thay thế bởi OpenAI Agents SDK**, phiên bản production-ready của Swarm và được đội ngũ OpenAI tích cực bảo trì.

- **Repository (Swarm Legacy)**: https://github.com/openai/swarm
- **Repository (Agents SDK)**: https://github.com/openai/openai-agents-python
- **Stars**: ~21,000 (Swarm)
- **Ngôn ngữ**: Python
- **License**: MIT

## Triết Lý Cốt Lõi

Swarm (và phiên bản kế nhiệm) tập trung vào việc làm cho **phối hợp** và **thực thi** agent trở nên nhẹ, có khả năng kiểm soát cao và dễ kiểm thử. Framework sử dụng hai trừu tượng nguyên thủy:

1. **Agents**: LLM được cấu hình với instructions, tools, guardrails và handoffs
2. **Handoffs**: Tool call đặc biệt để chuyển quyền điều khiển giữa các agent

Các nguyên thủy này đủ mạnh để thể hiện các động lực phong phú giữa tools và mạng lưới agents, cho phép developer xây dựng các giải pháp thực tế, có khả năng mở rộng trong khi tránh được đường cong học tập dốc.

## Các Khái Niệm Chính

### Agents

Một Agent đóng gói:
- **Instructions**: System prompt (có thể là chuỗi tĩnh hoặc hàm động)
- **Functions/Tools**: Các hàm Python mà agent có thể gọi
- **Model**: LLM sử dụng (mặc định: gpt-4o)
- **Tool Choice**: Cách agent chọn tools

```python
from agents import Agent, Runner

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant",
    tools=[get_weather],
)

result = Runner.run_sync(agent, "What's the weather?")
```

### Handoffs

Handoffs là cơ chế đặc biệt để chuyển quyền điều khiển giữa các agent. Khi một hàm trả về đối tượng Agent, việc thực thi sẽ chuyển sang agent đó.

```python
spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language.",
    handoffs=[spanish_agent, english_agent],
)
```

### Vòng Lặp Agent

Khi `Runner.run()` được gọi, một vòng lặp thực thi chạy cho đến khi có output cuối cùng:

1. Gọi LLM với model, settings và lịch sử tin nhắn
2. LLM trả về phản hồi (có thể bao gồm tool calls)
3. Nếu phản hồi có output cuối cùng, trả về và kết thúc vòng lặp
4. Nếu phản hồi có handoff, chuyển sang agent mới và quay lại bước 1
5. Xử lý tool calls và thêm tin nhắn phản hồi tool, sau đó quay lại bước 1

### Context Variables

Context variables cho phép truyền trạng thái qua các agent và function call:

```python
def instructions(context_variables):
    user_name = context_variables["user_name"]
    return f"Help the user, {user_name}, do whatever they want."

agent = Agent(instructions=instructions)
response = client.run(
    agent=agent,
    messages=[{"role": "user", "content": "Hi!"}],
    context_variables={"user_name": "John"}
)
```

### Functions/Tools

Tools là các hàm Python mà agent có thể gọi. SDK tự động chuyển đổi hàm thành JSON schema:

```python
from agents import function_tool

@function_tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"The weather in {city} is sunny."

agent = Agent(
    name="Weather Agent",
    tools=[get_weather],
)
```

## Tính Năng OpenAI Agents SDK (Phiên Bản Production)

### Các Khái Niệm Cốt Lõi

1. **Agents**: LLM được cấu hình với instructions, tools, guardrails và handoffs
2. **Handoffs**: Tool call đặc biệt để chuyển quyền điều khiển giữa các agent
3. **Guardrails**: Kiểm tra an toàn có thể cấu hình cho validation input và output
4. **Sessions**: Quản lý lịch sử hội thoại tự động qua các lần chạy agent
5. **Tracing**: Theo dõi tích hợp các lần chạy agent để debug và tối ưu hóa

### Sessions (Quản Lý Memory)

```python
from agents import Agent, Runner, SQLiteSession

agent = Agent(name="Assistant", instructions="Reply concisely.")
session = SQLiteSession("conversation_123")

# Lượt đầu tiên
result = await Runner.run(agent, "What city is the Golden Gate Bridge in?", session=session)
print(result.final_output)  # "San Francisco"

# Lượt thứ hai - agent nhớ ngữ cảnh
result = await Runner.run(agent, "What state is it in?", session=session)
print(result.final_output)  # "California"
```

### Tracing

SDK tự động trace các lần chạy agent với hỗ trợ các đích đến bên ngoài:
- Logfire
- AgentOps
- Braintrust
- Scorecard
- Keywords AI

### Không Phụ Thuộc Provider

SDK hỗ trợ:
- OpenAI Responses và Chat Completions APIs
- 100+ LLM khác thông qua tích hợp LiteLLM

## Kiến Trúc

### Swarm Core (Legacy)

```
swarm/
  __init__.py      # Export: Swarm, Agent
  core.py          # Class Swarm chính với run() và handle_tool_calls()
  types.py         # Các type Agent, Result, Response
  util.py          # function_to_json, tiện ích debug
  repl/            # REPL tương tác để testing
```

### Agents SDK (Hiện Tại)

```
agents/
  agent.py         # Định nghĩa Agent
  runner.py        # Runner để thực thi agents
  tools/           # Định nghĩa Tool và decorator function_tool
  memory/          # Triển khai Session (SQLite, Redis)
  tracing/         # Hạ tầng tracing
  extensions/      # Hỗ trợ voice, Redis sessions, v.v.
```

## Ví Dụ Trường Hợp Sử Dụng

1. **Triage Agent**: Định tuyến yêu cầu đến các agent chuyên biệt dựa trên nội dung
2. **Bot Dịch Vụ Khách Hàng**: Hệ thống multi-agent để xử lý các loại yêu cầu khác nhau
3. **Trợ Lý Mua Sắm Cá Nhân**: Agent với tools tìm kiếm sản phẩm và quản lý đơn hàng
4. **Bot Hỗ Trợ**: Agent giao diện người dùng + agent trung tâm trợ giúp với tools
5. **Dịch Vụ Khách Hàng Hàng Không**: Định tuyến multi-agent phức tạp

## Điểm Khác Biệt Chính

| Tính năng | Mô tả |
|-----------|-------|
| **Nhẹ** | Trừu tượng tối thiểu, chạy trên Chat Completions API |
| **Stateless** | Không lưu trạng thái giữa các lần gọi (như Chat Completions) |
| **Pattern Handoff** | Hỗ trợ native chuyển agent-to-agent |
| **Function-First** | Gọi trực tiếp hàm Python như tools |
| **Giáo dục** | Được thiết kế để dạy các pattern multi-agent |
| **Không Phụ Thuộc Provider** | Hỗ trợ 100+ LLM (Agents SDK) |
| **Tracing Tích Hợp** | Theo dõi và debug tự động |
| **Quản Lý Session** | Hỗ trợ session SQLite và Redis |

## Trường Hợp Sử Dụng Tốt Nhất

- **Định tuyến multi-agent**: Khi bạn cần các agent chuyên biệt cho các task khác nhau
- **Dịch vụ khách hàng**: Phân loại và handoff giữa các phòng ban
- **Workflow hội thoại**: Luồng hội thoại tuyến tính hoặc phân nhánh
- **Học pattern multi-agent**: Framework giáo dục để hiểu các khái niệm
- **Prototype nhẹ**: Proof-of-concept multi-agent nhanh
- **Ứng dụng production**: Với Agents SDK cho triển khai doanh nghiệp

## Hạn Chế

- **Tập trung vào OpenAI** (Swarm gốc): Chủ yếu thiết kế cho model OpenAI
- **Không có planning tích hợp**: Dựa vào LLM để quyết định handoffs
- **Memory tối thiểu**: Context variables chỉ cho trạng thái đơn giản (Swarm gốc)
- **Không tích hợp vector store**: Không có khả năng RAG tích hợp

## Cài Đặt

### Swarm (Legacy/Giáo dục)
```bash
pip install git+https://github.com/openai/swarm.git
```

### OpenAI Agents SDK (Production)
```bash
pip install openai-agents

# Với hỗ trợ voice
pip install 'openai-agents[voice]'

# Với hỗ trợ Redis session
pip install 'openai-agents[redis]'
```

## Tài Liệu Tham Khảo

- [OpenAI Swarm GitHub](https://github.com/openai/swarm)
- [OpenAI Agents SDK GitHub](https://github.com/openai/openai-agents-python)
- [Tài liệu Agents SDK](https://openai.github.io/openai-agents-python/)
- [Agents SDK JS/TS](https://github.com/openai/openai-agents-js)
