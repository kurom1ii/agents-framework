# Kiến trúc CrewAI

## Tổng quan

CrewAI triển khai kiến trúc đa agent tập trung vào khái niệm "crew" - các nhóm cộng tác của AI agent làm việc cùng nhau để hoàn thành các tác vụ phức tạp. Kiến trúc nhấn mạnh thiết kế dựa trên vai trò, thực thi hướng mục tiêu và điều phối linh hoạt.

## Các thành phần kiến trúc cốt lõi

### 1. Lớp Agent

Agent là các khối xây dựng cơ bản của CrewAI. Mỗi agent là một thực thể AI tự chủ với:

```
+------------------------------------------+
|                 AGENT                     |
+------------------------------------------+
| Identity (Danh tính)                      |
|   - role: str (chức năng/chuyên môn)      |
|   - goal: str (mục tiêu)                  |
|   - backstory: str (bối cảnh/tính cách)   |
+------------------------------------------+
| Capabilities (Khả năng)                   |
|   - tools: List[BaseTool]                 |
|   - knowledge_sources: List[...]          |
|   - llm: LLM                              |
+------------------------------------------+
| Behavior (Hành vi)                        |
|   - allow_delegation: bool                |
|   - allow_code_execution: bool            |
|   - memory: bool                          |
|   - verbose: bool                         |
+------------------------------------------+
| Constraints (Ràng buộc)                   |
|   - max_iter: int                         |
|   - max_rpm: int                          |
|   - max_execution_time: int               |
+------------------------------------------+
```

### 2. Lớp Task

Task đại diện cho các đơn vị công việc cần được agent hoàn thành:

```
+------------------------------------------+
|                  TASK                     |
+------------------------------------------+
| Definition (Định nghĩa)                   |
|   - description: str                      |
|   - expected_output: str                  |
|   - agent: Agent (tùy chọn)               |
+------------------------------------------+
| I/O Configuration (Cấu hình I/O)          |
|   - context: List[Task]                   |
|   - output_file: str                      |
|   - output_json: Type[BaseModel]          |
|   - output_pydantic: Type[BaseModel]      |
+------------------------------------------+
| Execution Control (Điều khiển thực thi)   |
|   - async_execution: bool                 |
|   - human_input: bool                     |
|   - guardrails: List[Callable]            |
|   - callback: Callable                    |
+------------------------------------------+
```

### 3. Lớp Crew

Crew điều phối sự cộng tác giữa các agent:

```
+------------------------------------------+
|                  CREW                     |
+------------------------------------------+
| Composition (Thành phần)                  |
|   - agents: List[Agent]                   |
|   - tasks: List[Task]                     |
+------------------------------------------+
| Orchestration (Điều phối)                 |
|   - process: Process (seq/hierarchical)   |
|   - manager_llm: LLM (cho hierarchical)   |
|   - manager_agent: Agent (tùy chọn)       |
+------------------------------------------+
| Shared Resources (Tài nguyên chia sẻ)     |
|   - memory: bool                          |
|   - cache: bool                           |
|   - knowledge_sources: List[...]          |
|   - embedder: Dict                        |
+------------------------------------------+
| Callbacks & Logging                       |
|   - step_callback: Callable               |
|   - task_callback: Callable               |
|   - output_log_file: str                  |
+------------------------------------------+
```

### 4. Lớp Flow

Flow cung cấp điều phối cấp cao hơn với các pattern hướng sự kiện:

```
+------------------------------------------+
|                  FLOW                     |
+------------------------------------------+
| Control Primitives (Các primitive điều khiển)
|   - @start(): Điểm vào                   |
|   - @listen(): Xử lý sự kiện             |
|   - @router(): Định tuyến có điều kiện   |
|   - or_(): Trigger any-of                |
|   - and_(): Trigger all-of               |
+------------------------------------------+
| State Management (Quản lý trạng thái)     |
|   - Unstructured: trạng thái kiểu dict   |
|   - Structured: Pydantic BaseModel        |
|   - ID duy nhất tự động tạo              |
+------------------------------------------+
| Persistence (Lưu trữ)                     |
|   - @persist: Lưu trữ trạng thái         |
|   - SQLiteFlowPersistence (mặc định)     |
|   - Triển khai FlowPersistence tùy chỉnh |
+------------------------------------------+
| Human-in-Loop (Con người tham gia)        |
|   - @human_feedback: Cổng đánh giá       |
|   - Kết quả định tuyến qua LLM           |
|   - Truy cập last_human_feedback         |
|   - Theo dõi human_feedback_history      |
+------------------------------------------+
| Execution (Thực thi)                      |
|   - kickoff(): Thực thi đồng bộ          |
|   - kickoff_async(): Thực thi bất đồng bộ|
|   - plot(): Trực quan hóa Flow           |
|   - stream: Hỗ trợ streaming output      |
+------------------------------------------+
```

#### Chi tiết Decorator của Flow

| Decorator | Mô tả | Cách sử dụng |
|-----------|-------|--------------|
| `@start()` | Điểm vào, có thể được gated bởi label hoặc điều kiện | `@start()` hoặc `@start("label")` |
| `@listen(method)` | Kích hoạt khi method hoàn thành | `@listen(method_name)` |
| `@router(method)` | Trả về label cho định tuyến có điều kiện | Trả về string label |
| `@persist` | Bật lưu trữ trạng thái | Cấp class hoặc method |
| `@human_feedback` | Thu thập đánh giá/quyết định của con người | Với danh sách outcomes |

#### Toán tử điều kiện

| Toán tử | Mô tả | Ví dụ |
|---------|-------|-------|
| `or_(a, b)` | Kích hoạt khi BẤT KỲ method nào phát ra | `@listen(or_(method1, method2))` |
| `and_(a, b)` | Kích hoạt khi TẤT CẢ method phát ra | `@listen(and_(method1, method2))` |

## Sơ đồ kiến trúc hệ thống

```
                    +------------------+
                    |   User/Client    |
                    +--------+---------+
                             |
                             v
+-----------------------------------------------------------+
|                       LỚP FLOW                             |
|  +-------+   +-------+   +--------+   +----------------+  |
|  |@start |-->|@listen|-->|@router |-->|@human_feedback |  |
|  +-------+   +-------+   +--------+   +----------------+  |
|                    |                                       |
|              Quản lý trạng thái                            |
|              (Structured/Unstructured)                     |
+-----------------------------------------------------------+
                             |
                             v
+-----------------------------------------------------------+
|                       LỚP CREW                             |
|  +--------------------------------------------------+     |
|  |                  CREW                             |     |
|  |  Process: Sequential | Hierarchical               |     |
|  |  Manager: LLM hoặc Agent (chỉ hierarchical)       |     |
|  +--------------------------------------------------+     |
|         |                                                  |
|         v                                                  |
|  +------+-------+-------+-------+                         |
|  |Task 1|Task 2 |Task 3 |Task N |  (Thực thi theo thứ tự) |
|  +------+-------+-------+-------+                         |
+-----------------------------------------------------------+
                             |
                             v
+-----------------------------------------------------------+
|                      LỚP AGENT                             |
|  +------------+  +------------+  +------------+           |
|  |  Agent 1   |  |  Agent 2   |  |  Agent N   |           |
|  |  (Role)    |  |  (Role)    |  |  (Role)    |           |
|  +------------+  +------------+  +------------+           |
|        |               |               |                   |
|        v               v               v                   |
|  +---------+     +---------+     +---------+              |
|  | Tools   |     | Tools   |     | Tools   |              |
|  +---------+     +---------+     +---------+              |
+-----------------------------------------------------------+
                             |
                             v
+-----------------------------------------------------------+
|                    LỚP NỀN TẢNG                            |
|  +----------+  +----------+  +-----------+  +----------+  |
|  |   LLM    |  |  Memory  |  | Knowledge |  | Embedder |  |
|  | Provider |  |  System  |  |  Sources  |  |          |  |
|  +----------+  +----------+  +-----------+  +----------+  |
+-----------------------------------------------------------+
```

## Các loại Process

### Sequential Process

Các task thực thi theo thứ tự được định nghĩa. Đầu ra của mỗi task có thể được truyền làm context cho các task tiếp theo.

```
Task 1 --> Task 2 --> Task 3 --> Task N --> Đầu ra cuối
   |          ^
   +----------+ (Đầu ra chảy như context)
```

**Đặc điểm**:
- Thứ tự thực thi xác định
- Đầu ra của một task trở thành context cho task tiếp theo
- Workflow đơn giản và dự đoán được
- Tốt nhất cho workflow kiểu pipeline

### Hierarchical Process

Một manager agent điều phối việc phân công task và xác nhận hoàn thành công việc.

```
                 +------------------+
                 |  Manager Agent   |
                 | (Lập kế hoạch/   |
                 |   Điều phối)     |
                 +--------+---------+
                          |
        +-----------------+-----------------+
        |                 |                 |
        v                 v                 v
  +-----------+     +-----------+     +-----------+
  |  Agent 1  |     |  Agent 2  |     |  Agent N  |
  | (Delegate)|     | (Delegate)|     | (Delegate)|
  +-----------+     +-----------+     +-----------+
```

**Đặc điểm**:
- Manager lập kế hoạch, phân công và xác nhận
- Phân công task động dựa trên khả năng của agent
- Linh hoạt hơn nhưng yêu cầu manager LLM
- Tốt nhất cho workflow phức tạp, thích ứng

## Kiến trúc Memory

CrewAI triển khai hệ thống memory đa tầng:

```
+-----------------------------------------------------------+
|                    HỆ THỐNG MEMORY                         |
+-----------------------------------------------------------+
|                                                            |
|  +------------------+     +------------------+             |
|  | Memory Ngắn hạn  |     | Memory Dài hạn   |             |
|  | (Phiên hiện tại) |     | (Xuyên phiên)    |             |
|  | ChromaDB + RAG   |     | SQLite3          |             |
|  +------------------+     +------------------+             |
|                                                            |
|  +------------------+     +------------------+             |
|  |  Entity Memory   |     | External Memory  |             |
|  | (Người, địa điểm)|     | (Mem0, tùy chỉnh)|             |
|  | ChromaDB + RAG   |     |                  |             |
|  +------------------+     +------------------+             |
|                                                            |
|  +--------------------------------------------------+     |
|  |              Contextual Memory                    |     |
|  | (Kết hợp tất cả loại memory cho context liền mạch)|     |
|  +--------------------------------------------------+     |
+-----------------------------------------------------------+
```

### Các loại Memory

| Loại | Lưu trữ | Mục đích |
|------|---------|----------|
| Ngắn hạn | ChromaDB | Tương tác gần đây trong thực thi hiện tại |
| Dài hạn | SQLite3 | Insights có giá trị xuyên phiên |
| Entity | ChromaDB | Thông tin về thực thể (người, địa điểm, khái niệm) |
| External | Có thể cắm | Tích hợp với nhà cung cấp memory bên ngoài |
| Contextual | Kết hợp | Context thống nhất từ tất cả loại memory |

## Kiến trúc Knowledge

Các nguồn knowledge cung cấp cho agent thông tin chuyên ngành:

```
+-----------------------------------------------------------+
|                  HỆ THỐNG KNOWLEDGE                        |
+-----------------------------------------------------------+
|                                                            |
|  Nguồn tích hợp:                                           |
|  +--------+ +--------+ +--------+ +--------+ +--------+   |
|  |  Text  | |  PDF   | |  CSV   | | Excel  | |  JSON  |   |
|  +--------+ +--------+ +--------+ +--------+ +--------+   |
|                                                            |
|  Nguồn Web:                                                |
|  +--------------------------------------------------+     |
|  |            CrewDoclingSource (URLs)               |     |
|  +--------------------------------------------------+     |
|                                                            |
|  Nguồn tùy chỉnh:                                          |
|  +--------------------------------------------------+     |
|  |       BaseKnowledgeSource (subclass)              |     |
|  +--------------------------------------------------+     |
|                                                            |
|  Lưu trữ:                                                  |
|  +--------------------------------------------------+     |
|  |  ChromaDB (mặc định) | Qdrant (tùy chọn)          |     |
|  +--------------------------------------------------+     |
|                                                            |
+-----------------------------------------------------------+
```

## Kiến trúc tích hợp Tool

```
+-----------------------------------------------------------+
|                    HỆ THỐNG TOOL                           |
+-----------------------------------------------------------+
|                                                            |
|  Tools tích hợp:                                           |
|  +----------+ +----------+ +----------+ +----------+      |
|  |  Search  | |   RAG    | |  Scrape  | |  Code    |      |
|  | (Serper) | | (Vector) | |  (Web)   | | (Interp) |      |
|  +----------+ +----------+ +----------+ +----------+      |
|                                                            |
|  Tools tùy chỉnh:                                          |
|  +--------------------------------------------------+     |
|  |  BaseTool Subclass  |  @tool Decorator            |     |
|  +--------------------------------------------------+     |
|                                                            |
|  Tính năng:                                                |
|  - Caching (cache_function)                               |
|  - Xử lý lỗi                                              |
|  - Hỗ trợ Async                                           |
|  - Pydantic input schemas                                  |
|                                                            |
+-----------------------------------------------------------+
```

## Tích hợp LLM

CrewAI không phụ thuộc LLM và hỗ trợ nhiều nhà cung cấp:

| Nhà cung cấp | Model Prefix | Tính năng chính |
|--------------|--------------|-----------------|
| OpenAI | `openai/` | Responses API, streaming |
| Anthropic | `anthropic/` | Extended thinking, Claude models |
| Google | `gemini/` | Multimodal, Vertex AI |
| Azure | `azure/` | Enterprise, deployments |
| AWS Bedrock | `bedrock/` | Converse API, guardrails |
| Ollama | `ollama/` | Local models |
| Groq | `groq/` | Fast inference |
| Và nhiều hơn... | | |

## Hệ thống Event

CrewAI phát ra các event để giám sát và observability:

```
+-----------------------------------------------------------+
|                    HỆ THỐNG EVENT                          |
+-----------------------------------------------------------+
|                                                            |
|  Memory Events:                                            |
|  - MemoryQueryCompletedEvent                              |
|  - MemorySaveCompletedEvent                               |
|  - MemorySaveFailedEvent                                  |
|                                                            |
|  Knowledge Events:                                         |
|  - KnowledgeRetrievalStartedEvent                         |
|  - KnowledgeRetrievalCompletedEvent                       |
|  - KnowledgeQueryFailedEvent                              |
|                                                            |
|  LLM Events:                                               |
|  - LLMStreamChunkEvent (streaming)                        |
|                                                            |
|  Đăng ký qua BaseEventListener subclass                    |
+-----------------------------------------------------------+
```

## Luồng thực thi

### Thực thi Crew tiêu chuẩn

```
1. Crew.kickoff(inputs)
       |
2. Khởi tạo agents với config
       |
3. Với mỗi task (dựa trên process):
       |
   3a. Giao cho agent (sequential) hoặc delegate (hierarchical)
       |
   3b. Agent thực thi với tools/knowledge/memory
       |
   3c. Áp dụng guardrails (validate/transform)
       |
   3d. Thực thi callbacks
       |
   3e. Truyền output làm context cho task tiếp theo
       |
4. Trả về CrewOutput (raw, pydantic, json_dict)
```

### Thực thi Flow

```
1. Flow.kickoff()
       |
2. Thực thi các method @start (điểm vào)
       |
3. Với mỗi output của method:
       |
   3a. Cập nhật state
       |
   3b. Kích hoạt các method @listen khớp với output
       |
   3c. Thực thi @router cho phân nhánh có điều kiện
       |
   3d. Xử lý @human_feedback nếu có
       |
4. Lưu trữ state nếu @persist được bật
       |
5. Trả về output của method cuối cùng
```

## Kiến trúc triển khai

### Phát triển Local

```
Máy của Developer
    |
    +-- Dự án CrewAI
    |       |
    |       +-- crew.py (agents, tasks)
    |       +-- main.py (entry)
    |       +-- config/ (YAML)
    |       +-- tools/ (tùy chỉnh)
    |
    +-- .env (API keys)
    |
    +-- Local LLM (Ollama) hoặc Remote API
```

### Production (Enterprise)

```
+-----------------------------------------------------------+
|                 CrewAI Enterprise                          |
+-----------------------------------------------------------+
|                                                            |
|  +------------------+    +------------------+              |
|  | Quản lý môi trường|    |    Giám sát      |              |
|  | (Redeploy an toàn)|    | (Xem live run)   |              |
|  +------------------+    +------------------+              |
|                                                            |
|  +------------------+    +------------------+              |
|  |    Triggers      |    |   Quản lý Team   |              |
|  | (Gmail, Slack,   |    |   (RBAC)         |              |
|  |  Salesforce...)  |    |                  |              |
|  +------------------+    +------------------+              |
|                                                            |
|  +--------------------------------------------------+     |
|  |           Crew Control Plane                      |     |
|  | (Tracing, quản lý tập trung, bảo mật, analytics)  |     |
|  +--------------------------------------------------+     |
|                                                            |
+-----------------------------------------------------------+
```

## Nguyên tắc thiết kế

1. **Thiết kế dựa trên vai trò**: Agent có role, goal và backstory rõ ràng
2. **Khả năng kết hợp**: Agent, task và tool là modular và tái sử dụng được
3. **Linh hoạt**: Hỗ trợ nhiều loại process và pattern thực thi
4. **Observability**: Logging, event và giám sát tích hợp
5. **Sẵn sàng Production**: Memory, persistence, guardrails và tính năng enterprise
6. **Không phụ thuộc LLM**: Hoạt động với mọi nhà cung cấp LLM lớn
7. **Trải nghiệm Developer**: API đơn giản, công cụ CLI, cấu hình YAML
