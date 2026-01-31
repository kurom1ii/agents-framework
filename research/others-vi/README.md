# Nghiên Cứu Các Framework AI Agent

Thư mục này chứa nghiên cứu toàn diện về các framework AI agent đáng chú ý. Nghiên cứu bao gồm kiến trúc, tính năng chính, trường hợp sử dụng và so sánh để hỗ trợ các quyết định thiết kế.

## Nội Dung

| File | Mô Tả |
|------|-------|
| [swarm.md](./swarm.md) | OpenAI Swarm & Agents SDK - Điều phối multi-agent nhẹ |
| [semantic-kernel.md](./semantic-kernel.md) | Microsoft Semantic Kernel - SDK AI cho doanh nghiệp |
| [llamaindex.md](./llamaindex.md) | LlamaIndex - Data agent và framework RAG |
| [haystack.md](./haystack.md) | Haystack - Ứng dụng LLM dựa trên pipeline |
| [comparison.md](./comparison.md) | So sánh chi tiết tất cả các framework |

## Tổng Quan Framework

### OpenAI Swarm / Agents SDK
- **Trọng tâm**: Điều phối multi-agent với handoff
- **Pattern chính**: Agents + Handoffs
- **Phù hợp nhất cho**: Dịch vụ khách hàng, định tuyến, workflow hội thoại
- **Repository**: https://github.com/openai/openai-agents-python

### Microsoft Semantic Kernel
- **Trọng tâm**: Phát triển ứng dụng AI cho doanh nghiệp
- **Pattern chính**: Kernel + Plugins (DI container)
- **Phù hợp nhất cho**: Ứng dụng doanh nghiệp, tích hợp Azure, dự án .NET
- **Repository**: https://github.com/microsoft/semantic-kernel

### LlamaIndex
- **Trọng tâm**: Data agent và ứng dụng RAG
- **Pattern chính**: Query Engine làm Agent Tool
- **Phù hợp nhất cho**: Hỏi đáp tài liệu, trợ lý nghiên cứu, agent xử lý dữ liệu
- **Repository**: https://github.com/run-llama/llama_index

### Haystack
- **Trọng tâm**: Pipeline LLM sẵn sàng cho production
- **Pattern chính**: Pipeline có thể kết hợp
- **Phù hợp nhất cho**: RAG doanh nghiệp, workflow phức tạp, tìm kiếm lai
- **Repository**: https://github.com/deepset-ai/haystack

## So Sánh Nhanh

| Tính năng | Swarm/Agents | Semantic Kernel | LlamaIndex | Haystack |
|-----------|--------------|-----------------|------------|----------|
| Stars | ~21K | ~27K | ~47K | ~24K |
| Ngôn ngữ | Python, JS/TS | C#, Python, Java | Python, TS | Python |
| Multi-Agent | Handoffs | Group Chat | Workflows | Pipelines |
| RAG tích hợp | Không | Qua plugin | Native | Native |
| Doanh nghiệp | Đang phát triển | Mạnh | Trung bình | Mạnh |

## Điểm Chính

### Các Design Pattern Quan Sát Được

1. **Agent Loop Pattern**: Tất cả framework đều triển khai vòng lặp tương tự:
   - Nhận input -> LLM quyết định hành động -> Thực thi tool -> Trả về cho LLM

2. **Tool Abstraction**: Mọi framework đều wrap function thành tool:
   - Dựa trên decorator (`@function_tool`, `@kernel_function`, `@tool`)
   - Dựa trên class (`FunctionTool`, `Tool`, `ComponentTool`)

3. **Chiến lược Memory**:
   - Dựa trên session (Agents SDK)
   - Hỗ trợ vector store (Semantic Kernel, LlamaIndex, Haystack)
   - Buffer lịch sử chat (tất cả)

4. **Các Pattern Multi-Agent**:
   - Trả về handoff (Swarm)
   - Điều phối/Group chat (Semantic Kernel)
   - Định nghĩa workflow (LlamaIndex)
   - Định tuyến pipeline (Haystack)

### Xu Hướng Mới Nổi

1. **Function Calling như Planning**: Function calling native của LLM thay thế planner dựa trên prompt
2. **Tích hợp MCP**: Hỗ trợ Model Context Protocol đang phát triển (Semantic Kernel)
3. **Không phụ thuộc Provider**: Tất cả framework đang hướng tới hỗ trợ đa provider
4. **Observability tích hợp**: Tracing và monitoring đang trở thành tiêu chuẩn
5. **Sessions/Memory**: Quản lý trạng thái tốt hơn qua các lượt hội thoại

## Chọn Framework

```
Định tuyến multi-agent đơn giản?      --> OpenAI Agents SDK
Ứng dụng doanh nghiệp/.NET?           --> Semantic Kernel
Agent dữ liệu/tài liệu phức tạp?      --> LlamaIndex
Pipeline RAG production?               --> Haystack
Học pattern agent?                     --> OpenAI Swarm (giáo dục)
```

## Phương Pháp Nghiên Cứu

Nghiên cứu này được biên soạn bằng cách:
1. Phân tích repository GitHub chính thức và README
2. Xem xét tài liệu chính thức
3. Kiểm tra cấu trúc mã nguồn và pattern
4. So sánh bộ tính năng và khả năng
5. Xác định trường hợp sử dụng từ ví dụ thực tế

## Cập Nhật Lần Cuối

Tháng 1, 2026
