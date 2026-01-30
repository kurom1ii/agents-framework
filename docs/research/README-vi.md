# Nghiên Cứu Các Framework Agent

Nghiên cứu toàn diện về các AI Agent Framework phổ biến để thiết kế agents framework riêng.

**Ngày tạo**: 2026-01-31

## Tổng quan nhanh

| Framework | Loại | Ngôn ngữ | Đặc điểm chính |
|-----------|------|----------|----------------|
| [LangChain](./langchain/) | Đa năng | Python | LangGraph, hơn 100 nhà cung cấp model, ReAct agents |
| [CrewAI](./crewai/) | Đa tác nhân | Python | Agent dựa trên vai trò, Flows, hơn 100 công cụ |
| [AutoGen](./autogen/) | Hội thoại | Python | Trò chuyện nhóm, Thực thi mã, Con người trong vòng lặp |
| [OpenClaw](./openclaw/) | Trợ lý cá nhân | TypeScript | Đa kênh, Kiến trúc Gateway, Ưu tiên cục bộ |
| [OpenAI Swarm/Agents SDK](./others/swarm.md) | Đa tác nhân | Python | Chuyển giao, Nhẹ, Quy trình |
| [Semantic Kernel](./others/semantic-kernel.md) | Doanh nghiệp | C#/Python | Kiến trúc plugin, Tích hợp Azure |
| [LlamaIndex](./others/llamaindex.md) | Agent dữ liệu | Python | Công cụ truy vấn, Tập trung vào RAG |
| [Haystack](./others/haystack.md) | Dựa trên pipeline | Python | Pipeline module hóa, Sẵn sàng cho production |

## Cấu trúc thư mục

```
docs/research/
├── README.md                 # File này - Tổng hợp toàn bộ
├── langchain/                # LangChain Framework
│   ├── README.md             # Tổng quan
│   ├── architecture.md       # Kiến trúc
│   ├── components.md         # Các thành phần
│   ├── patterns.md           # Mẫu thiết kế
│   └── examples.md           # Ví dụ mã nguồn
├── crewai/                   # CrewAI Framework
│   ├── README.md
│   ├── architecture.md
│   ├── components.md
│   ├── patterns.md
│   └── examples.md
├── autogen/                  # Microsoft AutoGen
│   ├── README.md
│   ├── architecture.md
│   ├── components.md
│   ├── patterns.md
│   └── examples.md
├── openclaw/                 # OpenClaw Trợ Lý Cá Nhân
│   ├── README.md
│   ├── architecture.md
│   ├── components.md
│   └── examples.md
├── others/                   # Các framework khác
│   ├── README.md
│   ├── swarm.md              # OpenAI Swarm/Agents SDK
│   ├── semantic-kernel.md    # Microsoft Semantic Kernel
│   ├── llamaindex.md         # LlamaIndex Agents
│   ├── haystack.md           # Haystack Agents
│   └── comparison.md         # So sánh tổng hợp
└── patterns/                 # Mẫu thiết kế chung
    ├── README.md
    ├── agent-patterns.md     # ReAct, Plan-Execute, Phản ánh
    ├── multi-agent-patterns.md # Giám sát, Swarm, Tuần tự
    ├── memory-patterns.md    # Ngắn hạn, Dài hạn, Vector
    ├── tool-patterns.md      # Gọi hàm, Schema
    └── observability.md      # Theo dõi, Ghi log, Số liệu
```

## So sánh Framework

### Theo Trường hợp sử dụng

| Trường hợp sử dụng | Framework được khuyên dùng |
|----------|----------------------------|
| Ứng dụng LLM nhanh | LangChain |
| Cộng tác đa tác nhân | CrewAI, AutoGen |
| Trợ lý AI cá nhân | OpenClaw |
| Doanh nghiệp/.NET | Semantic Kernel |
| Agent dữ liệu/RAG | LlamaIndex |
| Pipeline production | Haystack |
| Đa tác nhân nhẹ | OpenAI Agents SDK |

### Theo Mẫu kiến trúc

| Mẫu | Framework |
|---------|-----------|
| **Dựa trên đồ thị** | LangChain/LangGraph |
| **Dựa trên vai trò** | CrewAI |
| **Dựa trên hội thoại** | AutoGen |
| **Gateway/Hub** | OpenClaw |
| **Dựa trên chuyển giao** | OpenAI Swarm |
| **Dựa trên plugin** | Semantic Kernel |
| **Dựa trên pipeline** | Haystack |

### Theo Tính năng

| Tính năng | LangChain | CrewAI | AutoGen | OpenClaw |
|-----------|:---------:|:------:|:-------:|:--------:|
| Đa tác nhân | ✅ | ✅ | ✅ | ✅ |
| Hệ thống bộ nhớ | ✅ | ✅ | ✅ | ✅ |
| Tích hợp công cụ | ✅ | ✅ | ✅ | ✅ |
| Con người trong vòng lặp | ✅ | ✅ | ✅ | ✅ |
| Truyền phát | ✅ | ✅ | ✅ | ✅ |
| Lưu trữ | ✅ | ✅ | ✅ | ✅ |
| Đa kênh | ❌ | ❌ | ❌ | ✅ |
| Quan sát tích hợp | ✅ (LangSmith) | ✅ | ❌ | ✅ |
| Cấu hình YAML | ❌ | ✅ | ❌ | ✅ |
| Hỗ trợ giọng nói | ❌ | ❌ | ❌ | ✅ |

## Những hiểu biết chính cho Thiết kế Agent Framework

### 1. Vòng lặp Agent cốt lõi
Tất cả framework đều triển khai mẫu tương tự:
```
Đầu vào → LLM → Quyết định (công cụ/phản hồi) → Thực thi → Lặp lại
```

### 2. Trừu tượng hóa Agent
Các thành phần cốt lõi của một Agent:
- **Danh tính**: Vai trò, mục tiêu, hướng dẫn
- **Khả năng**: Công cụ, hàm
- **Bộ nhớ**: Ngữ cảnh, lịch sử
- **LLM**: Backend model

### 3. Các mẫu Đa tác nhân
- **Giám sát**: Điều phối trung tâm ủy quyền cho các worker
- **Swarm/Chuyển giao**: Các agent chuyển giao cho nhau
- **Pipeline tuần tự**: Chuỗi các agent
- **Trò chuyện nhóm**: Các agent thảo luận cùng nhau

### 4. Kiến trúc Bộ nhớ
- **Ngắn hạn**: Bộ đệm hội thoại (cửa sổ giới hạn)
- **Dài hạn**: Kho vector (tìm kiếm ngữ nghĩa)
- **Theo tập**: Lưu trữ sự kiện/trải nghiệm
- **Thực thể**: Kiến thức có cấu trúc

### 5. Tích hợp Công cụ
- Dựa trên decorator: hàm `@tool`
- Dựa trên lớp: kế thừa `BaseTool`
- Dựa trên schema: định nghĩa JSON/Pydantic

### 6. Quản lý Trạng thái
- **Điểm kiểm tra**: Lưu/khôi phục trạng thái agent
- **Phiên**: Ngữ cảnh theo cuộc hội thoại
- **Trạng thái chia sẻ**: Giao tiếp giữa các agent

## Khuyến nghị cho Framework Tùy chỉnh

### Các thành phần MVP tối thiểu
1. **Lớp Agent**: Vai trò, hướng dẫn, công cụ, LLM
2. **Registry công cụ**: Registry hàm với xác thực schema
3. **Vòng lặp Agent**: Vòng lặp suy luận kiểu ReAct
4. **Bộ nhớ**: Bộ đệm hội thoại + kho vector tùy chọn
5. **Bộ thực thi**: Sandbox thực thi công cụ

### Tính năng nâng cao (Giai đoạn 2)
1. **Điều phối đa tác nhân**: Mẫu giám sát hoặc chuyển giao
2. **Lưu trữ**: Điểm kiểm tra/khôi phục trạng thái
3. **Truyền phát**: Truyền phát phản hồi thời gian thực
4. **Quan sát**: Theo dõi, ghi log
5. **Con người trong vòng lặp**: Luồng phê duyệt

### Khuyến nghị Kiến trúc
1. **Phân tách mối quan tâm**: Định nghĩa agent vs thực thi
2. **LLM có thể thay thế**: Giao diện LLM trừu tượng
3. **Ưu tiên bất đồng bộ**: Hỗ trợ các hoạt động đồng thời
4. **An toàn kiểu**: Sử dụng Pydantic/TypeBox cho schema
5. **Khả năng mở rộng**: Hệ thống plugin/mở rộng

## Tài liệu tham khảo

### Tài liệu chính thức
- LangChain: https://python.langchain.com/docs/
- CrewAI: https://docs.crewai.com/
- AutoGen: https://microsoft.github.io/autogen/
- OpenClaw: https://docs.openclaw.ai/
- OpenAI Agents SDK: https://openai.github.io/openai-agents-python/
- Semantic Kernel: https://learn.microsoft.com/semantic-kernel/
- LlamaIndex: https://docs.llamaindex.ai/
- Haystack: https://docs.haystack.deepset.ai/

### Kho lưu trữ GitHub
- https://github.com/langchain-ai/langchain
- https://github.com/crewAIInc/crewAI
- https://github.com/microsoft/autogen
- https://github.com/openclaw/openclaw
- https://github.com/openai/openai-agents-python
- https://github.com/microsoft/semantic-kernel
- https://github.com/run-llama/llama_index
- https://github.com/deepset-ai/haystack

---

*Nghiên cứu này được tạo tự động bởi Claude Code với mục đích thiết kế custom agents framework.*
