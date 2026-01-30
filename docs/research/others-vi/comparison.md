# So Sánh Các Framework AI Agent

## Tổng Quan

Tài liệu này cung cấp so sánh toàn diện về các framework AI agent chính, nêu bật các tính năng chính, kiến trúc và trường hợp sử dụng tốt nhất của chúng.

## Bảng Tham Khảo Nhanh

| Tính năng | OpenAI Swarm/Agents SDK | Semantic Kernel | LlamaIndex | Haystack |
|---------|------------------------|-----------------|------------|----------|
| **Trọng tâm chính** | Điều phối multi-agent | SDK AI doanh nghiệp | Data agents & RAG | Ứng dụng LLM dựa trên pipeline |
| **Nhà phát triển** | OpenAI | Microsoft | LlamaIndex Inc. | deepset |
| **Stars (GitHub)** | ~21,000 | ~27,000 | ~47,000 | ~24,000 |
| **Ngôn ngữ** | Python (JS/TS cho SDK) | C#, Python, Java | Python, TypeScript | Python |
| **License** | MIT | MIT | MIT | Apache 2.0 |
| **Sẵn sàng Production** | Có (Agents SDK) | Có | Có | Có |

## So Sánh Chi Tiết

### Triết Lý Kiến Trúc

| Framework | Cách tiếp cận | Trừu tượng chính |
|-----------|--------------|------------------|
| **OpenAI Swarm/Agents SDK** | Tối thiểu, stateless | Agents + Handoffs |
| **Semantic Kernel** | DI container doanh nghiệp | Kernel + Plugins |
| **LlamaIndex** | Tập trung dữ liệu | Query Engines + Tools |
| **Haystack** | Kết hợp pipeline | Components + Pipelines |

### Khả Năng Agent

| Khả năng | OpenAI Agents | Semantic Kernel | LlamaIndex | Haystack |
|----------|---------------|-----------------|------------|----------|
| Agent đơn | Có | Có | Có | Có |
| Multi-Agent | Có (handoffs) | Có (group chat) | Có (workflow) | Có (pipelines) |
| Gọi Tool | Có | Có (plugins) | Có | Có |
| Handoffs | Native | Qua điều phối | Qua workflow | Qua routing |
| Memory | Sessions (SQL/Redis) | Vector stores | ChatMemoryBuffer | Ngắn hạn/dài hạn |
| Streaming | Có | Có | Có | Có |
| Guardrails | Tích hợp | Qua middleware | Tùy chỉnh | Tùy chỉnh |

### Planning & Lập Luận

| Tính năng | OpenAI Agents | Semantic Kernel | LlamaIndex | Haystack |
|-----------|---------------|-----------------|------------|----------|
| Phương pháp Planning | Vòng lặp LLM-driven | Function calling | Agent loop | Định tuyến pipeline |
| Hỗ trợ ReAct | Không | Không | Có (ReActAgent) | Qua agents |
| Custom Planners | Qua tools | Deprecated (dùng FC) | Custom agents | Kết hợp pipeline |
| Theo dõi Step | Tracing | ChatHistory | Agent events | Pipeline tracing |

### Tích Hợp Dữ Liệu & RAG

| Tính năng | OpenAI Agents | Semantic Kernel | LlamaIndex | Haystack |
|-----------|---------------|-----------------|------------|----------|
| Hỗ trợ Vector Store | Không (bên ngoài) | Có (nhiều) | Có (nhiều) | Có (nhiều) |
| Tải Document | Không | Hạn chế | Mở rộng | Mở rộng |
| Query Engines | Không | Không | Tính năng cốt lõi | Qua retrievers |
| RAG Pipelines | Không | Qua plugins | Tính năng cốt lõi | Tính năng cốt lõi |
| Knowledge Graphs | Không | Không | Có | Hạn chế |

### Tính Năng Doanh Nghiệp

| Tính năng | OpenAI Agents | Semantic Kernel | LlamaIndex | Haystack |
|-----------|---------------|-----------------|------------|----------|
| Đa ngôn ngữ | Python, JS/TS | C#, Python, Java | Python, TS | Python |
| Tích hợp Cloud | OpenAI API | Azure-first | Cloud-agnostic | Cloud-agnostic |
| Observability | Tracing tích hợp | Events/middleware | LlamaTrace | Telemetry |
| Bảo mật | Guardrails | Kiểm soát doanh nghiệp | Tùy chỉnh | Doanh nghiệp |
| Hỗ trợ DI | Không | Native | Hạn chế | Hạn chế |

### Hỗ Trợ Model

| Provider | OpenAI Agents | Semantic Kernel | LlamaIndex | Haystack |
|----------|---------------|-----------------|------------|----------|
| OpenAI | Native | Có | Có | Có |
| Azure OpenAI | Có | Native | Có | Có |
| Anthropic | Qua LiteLLM | Có | Có | Có |
| Google | Qua LiteLLM | Có | Có | Có |
| HuggingFace | Qua LiteLLM | Có | Có | Có |
| Local (Ollama) | Qua LiteLLM | Có | Có | Có |
| 100+ LLMs | Có (LiteLLM) | Nhiều | Nhiều | Nhiều |

## Khuyến Nghị Trường Hợp Sử Dụng

### Khi Nào Sử Dụng OpenAI Swarm/Agents SDK

**Tốt nhất cho:**
- Định tuyến và handoffs multi-agent
- Ứng dụng dịch vụ khách hàng
- Workflows hội thoại
- Prototyping nhanh hệ thống multi-agent
- Đội ngũ đã sử dụng OpenAI

**Ví dụ trường hợp sử dụng:**
- Phân loại hỗ trợ khách hàng
- Định tuyến yêu cầu đa phòng ban
- Trợ lý tương tác với agents chuyên biệt
- Dự án giáo dục học pattern multi-agent

### Khi Nào Sử Dụng Semantic Kernel

**Tốt nhất cho:**
- Ứng dụng doanh nghiệp
- Dự án trong hệ sinh thái Microsoft/Azure
- Ứng dụng .NET
- Tự động hóa quy trình kinh doanh phức tạp
- Dự án cần hỗ trợ đa ngôn ngữ

**Ví dụ trường hợp sử dụng:**
- Chatbot doanh nghiệp với tích hợp Azure
- Tự động hóa quy trình kinh doanh
- Trợ lý kiến thức nội bộ
- Phát triển ứng dụng đa ngôn ngữ

### Khi Nào Sử Dụng LlamaIndex

**Tốt nhất cho:**
- Ứng dụng agent xử lý dữ liệu nặng
- Hệ thống RAG với nhu cầu truy xuất phức tạp
- Tổng hợp dữ liệu đa nguồn
- Hệ thống hỏi đáp tài liệu
- Công cụ nghiên cứu và phân tích

**Ví dụ trường hợp sử dụng:**
- Tìm kiếm và hỏi đáp tài liệu
- Trợ lý nghiên cứu
- Agents truy vấn SQL
- Chatbots knowledge base
- Hệ thống truy xuất đa index

### Khi Nào Sử Dụng Haystack

**Tốt nhất cho:**
- Hệ thống RAG production
- Pipelines nhiều bước phức tạp
- Ứng dụng tìm kiếm lai
- Workflows xử lý tùy chỉnh
- Triển khai doanh nghiệp

**Ví dụ trường hợp sử dụng:**
- Tìm kiếm tài liệu doanh nghiệp
- Hệ thống trả lời câu hỏi
- Pipelines trích xuất nội dung
- Ứng dụng tìm kiếm semantic
- Kết hợp Agent + RAG

## Phân Tích Tính Năng Chi Tiết

### Điều Phối Multi-Agent

| Framework | Pattern | Triển khai |
|-----------|---------|------------|
| **OpenAI Agents** | Handoffs | Function trả về đối tượng Agent, quyền điều khiển chuyển |
| **Semantic Kernel** | Group Chat | Nhiều agents hợp tác trong ngữ cảnh chung |
| **LlamaIndex** | AgentWorkflow | Định nghĩa handoff rõ ràng trong workflow |
| **Haystack** | Định tuyến Pipeline | ConditionalRouter điều hướng đến agent components |

### Hệ Thống Tool/Plugin

**OpenAI Agents SDK:**
```python
@function_tool
def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny"
```

**Semantic Kernel:**
```python
@kernel_function(description="Get weather")
def get_weather(self, city: str) -> str:
    return f"Weather in {city}: Sunny"
```

**LlamaIndex:**
```python
tool = FunctionTool.from_defaults(fn=get_weather)
```

**Haystack:**
```python
@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny"
```

### Quản Lý Memory/Session

| Framework | Ngắn hạn | Dài hạn | Lưu trữ |
|-----------|----------|---------|---------|
| OpenAI Agents | Session class | Session stores | SQLite, Redis |
| Semantic Kernel | ChatHistory | Vector stores | Nhiều backend |
| LlamaIndex | ChatMemoryBuffer | Storage context | Nhiều backend |
| Haystack | Conversation | Memory components | Custom stores |

## Cân Nhắc Hiệu Năng

### Chi Phí & Độ Phức Tạp

| Framework | Độ phức tạp Setup | Chi phí Runtime | Đường cong Học |
|-----------|------------------|-----------------|----------------|
| OpenAI Agents | Thấp | Thấp | Thấp |
| Semantic Kernel | Trung bình-Cao | Trung bình | Trung bình-Cao |
| LlamaIndex | Trung bình | Trung bình-Cao (indexing) | Trung bình |
| Haystack | Trung bình | Trung bình | Trung bình |

### Patterns Mở Rộng

| Framework | Mở rộng Ngang | Thiết kế Stateless | Agents Phân tán |
|-----------|---------------|-------------------|-----------------|
| OpenAI Agents | Dễ (stateless) | Native | Qua state bên ngoài |
| Semantic Kernel | Enterprise patterns | Qua DI | Qua services |
| LlamaIndex | Qua external stores | Có thể cấu hình | Hạn chế |
| Haystack | Qua Hayhooks | Dựa trên pipeline | Qua components |

## Hệ Sinh Thái Tích Hợp

### Số Lượng Tích Hợp

| Danh mục | OpenAI Agents | Semantic Kernel | LlamaIndex | Haystack |
|----------|---------------|-----------------|------------|----------|
| LLMs | 100+ (LiteLLM) | ~20 | ~50 | ~30 |
| Vector Stores | Bên ngoài | ~10 | ~30 | ~15 |
| Data Loaders | Không | Hạn chế | 300+ | ~50 |
| Tools/APIs | Tùy chỉnh | Plugins | LlamaHub | Tích hợp |

## Ma Trận Tổng Hợp

### Điểm Mạnh

| Framework | Điểm mạnh chính |
|-----------|----------------|
| **OpenAI Agents** | Đơn giản, nhẹ, handoffs native, tracing tích hợp |
| **Semantic Kernel** | Sẵn sàng doanh nghiệp, đa ngôn ngữ, tích hợp Azure, plugins |
| **LlamaIndex** | Tốt nhất cho RAG, data connectors mở rộng, query engines như tools |
| **Haystack** | Linh hoạt pipeline, sẵn sàng production, luồng dữ liệu rõ ràng |

### Hạn Chế

| Framework | Hạn chế chính |
|-----------|--------------|
| **OpenAI Agents** | Không có RAG tích hợp, memory hạn chế, tập trung OpenAI |
| **Semantic Kernel** | Setup phức tạp, tập trung Microsoft, đường cong học tập |
| **LlamaIndex** | Tập trung dữ liệu, phức tạp cho trường hợp đơn giản, nặng cho không-RAG |
| **Haystack** | Dài dòng, tính năng agent mới hơn, đường cong học pipeline |

## Hướng Dẫn Quyết Định

```
Cần handoffs multi-agent với setup tối thiểu?
  --> OpenAI Agents SDK

Xây dựng ứng dụng doanh nghiệp trong .NET/Azure?
  --> Semantic Kernel

Cần agents trên dữ liệu/tài liệu phức tạp?
  --> LlamaIndex

Muốn pipelines rõ ràng, có thể kết hợp?
  --> Haystack

Cần kết hợp RAG + agents?
  --> LlamaIndex hoặc Haystack

Bắt đầu mới, muốn đơn giản?
  --> OpenAI Agents SDK

Cần hỗ trợ đa ngôn ngữ (C#, Python, Java)?
  --> Semantic Kernel
```

## Tài Liệu Tham Khảo

- [OpenAI Swarm](https://github.com/openai/swarm)
- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
- [Microsoft Semantic Kernel](https://github.com/microsoft/semantic-kernel)
- [LlamaIndex](https://github.com/run-llama/llama_index)
- [Haystack](https://github.com/deepset-ai/haystack)
