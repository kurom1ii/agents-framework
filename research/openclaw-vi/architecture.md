# Kiến Trúc OpenClaw

Tài liệu này cung cấp phân tích chi tiết về kiến trúc của OpenClaw.

## Kiến Trúc Gateway

Gateway là thành phần trung tâm của OpenClaw - một server WebSocket đóng vai trò là control plane thống nhất cho tất cả các hoạt động.

### Nguyên Tắc Thiết Kế Gateway

1. **Điểm Kiểm Soát Duy Nhất**: Một Gateway trên mỗi host quản lý tất cả các bề mặt nhắn tin
2. **Ưu Tiên WebSocket**: Tất cả giao tiếp sử dụng WebSocket với payload JSON
3. **Hướng Sự Kiện**: Server đẩy sự kiện để cập nhật thời gian thực
4. **Giao Thức Có Kiểu**: Schema TypeBox cho giao tiếp an toàn kiểu

### Các Thành Phần Gateway

```
                    +----------------------------------+
                    |           Gateway Server         |
                    +----------------------------------+
                    |                                  |
    +---------------+---------------+------------------+
    |               |               |                  |
    v               v               v                  v
+-------+     +---------+     +----------+      +----------+
|Channel|     | Session |     |  Agent   |      |   Tool   |
|Manager|     | Manager |     | Runtime  |      | Registry |
+-------+     +---------+     +----------+      +----------+
    |               |               |                  |
    v               v               v                  v
+--------+    +---------+     +---------+       +---------+
|WhatsApp|    |sessions.|     |p-mono   |       |exec,    |
|Telegram|    |json     |     |runtime  |       |browser, |
|Slack...|    |JSONL    |     |         |       |canvas...|
+--------+    +---------+     +---------+       +---------+
```

### Giao Thức Truyền Tải

Tất cả giao tiếp theo mẫu request/response với các sự kiện đẩy từ server:

```typescript
// Request (Yêu cầu)
{
  type: "req",
  id: string,        // ID yêu cầu duy nhất
  method: string,    // Tên phương thức
  params: object     // Tham số phương thức
}

// Response (Phản hồi)
{
  type: "res",
  id: string,        // Khớp với ID yêu cầu
  ok: boolean,       // Cờ thành công
  payload?: object,  // Dữ liệu thành công
  error?: object     // Chi tiết lỗi
}

// Event (Sự kiện - đẩy từ server)
{
  type: "event",
  event: string,     // Tên sự kiện
  payload: object,   // Dữ liệu sự kiện
  seq?: number,      // Số thứ tự
  stateVersion?: number
}
```

### Vòng Đời Kết Nối

```
Client                           Gateway
  |                                 |
  |---- req:connect --------------->|
  |<------ res:hello-ok ------------|  (bao gồm snapshot)
  |                                 |
  |<------ event:presence ----------|
  |<------ event:tick --------------|
  |                                 |
  |------- req:agent -------------->|
  |<------ res:agent (accepted) ----|
  |<------ event:agent -------------|  (streaming)
  |<------ res:agent (final) -------|
  |                                 |
```

### Mô Hình Xác Thực

OpenClaw hỗ trợ nhiều chế độ xác thực:

1. **Xác thực Token**: Token gateway trong `connect.params.auth.token`
2. **Ghép Nối Thiết Bị**: Danh tính thiết bị với mã ghép nối
3. **Danh Tính Tailscale**: Tự động tin tưởng dựa trên header Tailscale
4. **Xác Thực Mật Khẩu**: Mật khẩu chia sẻ cho phơi bày công khai

```json5
{
  gateway: {
    auth: {
      mode: "password",          // hoặc "token"
      password: "bí_mật",
      allowTailscale: true,      // Tin tưởng danh tính Tailscale
      allowLocal: true           // Tự động phê duyệt loopback
    }
  }
}
```

## Kiến Trúc Agent Runtime

### Tích Hợp p-mono

OpenClaw sử dụng phiên bản sửa đổi của agent runtime p-mono:

```
+------------------------+
|    Agent Runtime       |
+------------------------+
|  - Model Provider      |  (Anthropic, OpenAI, Google, v.v.)
|  - Tool Registry       |  (Tích hợp sẵn + Skills)
|  - Context Manager     |  (Lắp ráp system prompt)
|  - Stream Handler      |  (Streaming công cụ + khối)
+------------------------+
```

### Luồng Thực Thi Công Cụ

```
Tin nhắn Người dùng
     |
     v
+--------------------+
| Message Processor  |
+--------------------+
     |
     v
+--------------------+
| Agent Loop         |<----+
+--------------------+     |
     |                     |
     v                     |
+--------------------+     |
| Tool Invocation    |-----+
+--------------------+
     |
     v
+--------------------+
| Response Assembly  |
+--------------------+
     |
     v
Giao hàng qua Channel
```

### Lắp Ráp Ngữ Cảnh

System prompt được lắp ráp từ nhiều nguồn:

```
+----------------------------+
|     System Prompt          |
+----------------------------+
| 1. Base prompt (templates) |
| 2. AGENTS.md (workspace)   |
| 3. SOUL.md (persona)       |
| 4. TOOLS.md (hướng dẫn)    |
| 5. IDENTITY.md (tên)       |
| 6. USER.md (hồ sơ người dùng) |
| 7. Skills (file SKILL.md)  |
| 8. Runtime context         |
+----------------------------+
```

## Kiến Trúc Session

### Giải Quyết Session Key

```
Tin nhắn Trực tiếp (DM)
  |
  +-- dmScope: "main"
  |     -> agent:<agentId>:<mainKey>
  |
  +-- dmScope: "per-peer"
  |     -> agent:<agentId>:dm:<peerId>
  |
  +-- dmScope: "per-channel-peer"
  |     -> agent:<agentId>:<channel>:dm:<peerId>
  |
  +-- dmScope: "per-account-channel-peer"
        -> agent:<agentId>:<channel>:<accountId>:dm:<peerId>

Chat Nhóm
  -> agent:<agentId>:<channel>:group:<groupId>

Thread/Topic
  -> agent:<agentId>:<channel>:group:<groupId>:topic:<threadId>
```

### Vòng Đời Session

```
+----------------+     +-----------------+     +----------------+
|   Tin nhắn Mới | --> | Session Lookup  | --> | Tồn tại?       |
+----------------+     +-----------------+     +----------------+
                                                    |
                       +----------------+      +----+----+
                       | Tạo Session    | <--- |  Không  |
                       +----------------+      +---------+
                                |
                       +--------v--------+     +---------+
                       |  Kiểm tra Hết hạn| <---|   Có    |
                       +-----------------+     +---------+
                                |
                 +--------------+---------------+
                 |                              |
            +----v----+                    +----v----+
            | Hết hạn |                    | Hợp lệ  |
            +---------+                    +---------+
                 |                              |
            +----v--------+                +----v----+
            | Reset       |                | Tái sử  |
            | (ID mới)    |                | dụng    |
            +-------------+                +---------+
```

### Lưu Trữ Session

```
~/.openclaw/agents/<agentId>/sessions/
├── sessions.json          # Kho session (key -> metadata)
├── <sessionId>.jsonl      # Bản ghi (định dạng JSONL)
└── <sessionId>-topic-<threadId>.jsonl  # Bản ghi thread
```

## Kiến Trúc Channel

### Giao Diện Channel

Tất cả các channel triển khai một giao diện chung:

```typescript
interface Channel {
  // Vòng đời
  start(): Promise<void>;
  stop(): Promise<void>;

  // Trạng thái
  isConnected(): boolean;
  getStatus(): ChannelStatus;

  // Nhắn tin
  send(message: OutboundMessage): Promise<void>;

  // Sự kiện
  onMessage(handler: MessageHandler): void;
  onTyping(handler: TypingHandler): void;
}
```

### Định Tuyến Channel

```
Tin nhắn Đến
      |
      v
+------------------+
| Channel Adapter  |  (WhatsApp, Telegram, v.v.)
+------------------+
      |
      v
+------------------+
| Quy tắc Định tuyến |
+------------------+
      |
      +-- Kiểm tra danh sách allowFrom
      +-- Kiểm tra danh sách cho phép nhóm
      +-- Áp dụng chính sách DM (pairing/open)
      +-- Giải quyết ánh xạ agent
      |
      v
+------------------+
| Session Resolver |
+------------------+
      |
      v
+------------------+
| Agent Runtime    |
+------------------+
```

### Hỗ Trợ Đa Tài Khoản

```json5
{
  agents: {
    agents: [
      { id: "personal", model: "anthropic/claude-opus-4-5" },
      { id: "work", model: "openai/gpt-5.2" }
    ],
    routing: {
      whatsapp: {
        "+1234567890": { agent: "personal" },
        "+0987654321": { agent: "work" }
      }
    }
  }
}
```

## Kiến Trúc Extension

### Cấu Trúc Plugin

```
extensions/<tên-channel>/
├── package.json         # Dependencies
├── src/
│   ├── index.ts        # Điểm vào chính
│   ├── channel.ts      # Triển khai channel
│   └── ...
└── README.md
```

### Plugin SDK

Extensions sử dụng plugin SDK:

```typescript
import { definePlugin, ChannelPlugin } from 'openclaw/plugin-sdk';

export default definePlugin({
  name: 'my-channel',
  version: '1.0.0',

  channels: [{
    id: 'my-channel',
    name: 'My Channel',

    async start(config) {
      // Khởi tạo channel
    },

    async stop() {
      // Dọn dẹp
    },

    async send(message) {
      // Gửi tin nhắn đi
    }
  }]
});
```

## Kiến Trúc Bộ Nhớ

### Các Lớp Bộ Nhớ

```
+------------------------+
|    Memory System       |
+------------------------+
        |
        +-- Nhật ký Hàng ngày (memory/YYYY-MM-DD.md)
        |     - Chỉ thêm
        |     - Đọc hôm nay + hôm qua
        |
        +-- Dài hạn (MEMORY.md)
        |     - Sự kiện được chọn lọc
        |     - Chỉ main session
        |
        +-- Vector Index (SQLite)
              - Tìm kiếm ngữ nghĩa
              - Hybrid BM25 + vector
```

### Pipeline Tìm Kiếm Bộ Nhớ

```
Truy vấn
  |
  v
+-------------------+
| Query Embedding   |  (OpenAI / Gemini / Local)
+-------------------+
  |
  v
+-------------------+
| Vector Search     |  (sqlite-vec)
+-------------------+
  |
  v
+-------------------+
| BM25 Search       |  (FTS5)
+-------------------+
  |
  v
+-------------------+
| Score Fusion      |  (pha trộn có trọng số)
+-------------------+
  |
  v
Kết quả (snippets + metadata)
```

## Kiến Trúc Node

Nodes là các endpoint thiết bị cung cấp khả năng cục bộ:

### Đăng Ký Node

```
Node (macOS/iOS/Android)
        |
        |---- Kết nối WebSocket (role: "node")
        v
+------------------+
| Gateway          |
+------------------+
        |
        v
+------------------+
| Node Registry    |
+------------------+
        |
        +-- node.list (khám phá nodes)
        +-- node.describe (lấy khả năng)
        +-- node.invoke (thực thi lệnh)
```

### Khả Năng Node

```json5
{
  role: "node",
  caps: ["canvas", "camera", "screen", "location", "notifications"],
  commands: {
    "canvas.push": { /* schema */ },
    "camera.snap": { /* schema */ },
    "screen.record": { /* schema */ },
    "location.get": { /* schema */ },
    "system.notify": { /* schema */ }
  },
  permissions: {
    "screen": "granted",
    "camera": "denied",
    "location": "undetermined"
  }
}
```

## Kiến Trúc Bảo Mật

### Các Lớp Bảo Mật

```
+-----------------------+
|   Chính sách DM       |  (pairing/open)
+-----------------------+
          |
          v
+-----------------------+
|   Danh sách Cho phép  |  (theo channel)
+-----------------------+
          |
          v
+-----------------------+
|   Tool Sandboxing     |  (theo session)
+-----------------------+
          |
          v
+-----------------------+
|   Phê duyệt Exec      |  (theo lệnh)
+-----------------------+
```

### Các Chế Độ Sandbox

```json5
{
  agents: {
    defaults: {
      sandbox: {
        mode: "non-main",        // Sandbox các session không phải main
        workspaceAccess: "rw",   // ro, rw, none
        allowlist: [
          "bash", "process", "read", "write", "edit"
        ],
        denylist: [
          "browser", "canvas", "nodes", "cron"
        ]
      }
    }
  }
}
```

## Kiến Trúc Triển Khai

### Triển Khai Cục Bộ

```
+-------------------+
|   Máy của Bạn     |
+-------------------+
|                   |
|  +-------------+  |
|  |   Gateway   |  |
|  | :18789 (WS) |  |
|  +-------------+  |
|        |          |
|  +-----+-----+    |
|  |           |    |
|  v           v    |
| WhatsApp  Telegram|
|  (web)    (API)   |
+-------------------+
```

### Triển Khai Từ Xa

```
+------------------+      +-------------------+
|   Mac của Bạn    |      |   Linux VPS       |
+------------------+      +-------------------+
|                  |      |                   |
|  macOS App  <----+------+----> Gateway      |
|  (SSH tunnel)    |      |      :18789       |
|                  |      |                   |
+------------------+      |   +----+----+     |
                          |   |         |     |
                          |   v         v     |
                          | WhatsApp Telegram |
                          +-------------------+
```

### Tích Hợp Tailscale

```json5
{
  gateway: {
    tailscale: {
      mode: "serve",     // hoặc "funnel"
      resetOnExit: true
    },
    bind: "loopback"     // Bắt buộc với Tailscale
  }
}
```

## Cân Nhắc Hiệu Suất

### Điểm Nghẽn

1. **Độ Trễ LLM**: Điểm nghẽn chính là thời gian phản hồi API model
2. **Kết Nối Channel**: Mỗi channel duy trì kết nối persistent
3. **Đánh Chỉ Mục Bộ Nhớ**: File bộ nhớ lớn có thể làm chậm đánh chỉ mục vector

### Tối Ưu Hóa

1. **Block Streaming**: Gửi phản hồi khi hoàn thành
2. **Session Pruning**: Cắt kết quả công cụ cũ để giảm ngữ cảnh
3. **Embedding Cache**: Tránh re-embedding nội dung không thay đổi
4. **Batch Embeddings**: Sử dụng OpenAI Batch API cho các công việc đánh chỉ mục lớn

## Cân Nhắc Khả Năng Mở Rộng

OpenClaw được thiết kế cho các use case **trợ lý cá nhân đơn người dùng**:

- **Mở Rộng Ngang**: Không được thiết kế cho điều này (một Gateway trên mỗi host)
- **Mở Rộng Dọc**: Xử lý nhiều channel trên một Gateway duy nhất
- **Multi-Agent**: Nhiều danh tính agent trên một Gateway (không phải multi-tenant)

Đối với các kịch bản đa người dùng, chạy các instance OpenClaw riêng biệt cho mỗi người dùng.

## Định Tuyến Agent (Agent Routing)

Định tuyến agent trong OpenClaw cho phép điều hướng tin nhắn đến các agent khác nhau dựa trên nhiều tiêu chí. Đây là tính năng quan trọng cho việc triển khai nhiều agent với các vai trò khác nhau.

### Kiến Trúc Định Tuyến

```
Tin nhắn Đến
      |
      v
+------------------+
| Channel Adapter  |
+------------------+
      |
      v
+------------------+
| Routing Engine   |  <-- Quyết định agent nào xử lý
+------------------+
      |
      +-- Kiểm tra routing theo channel
      +-- Kiểm tra routing theo người gửi
      +-- Kiểm tra routing theo nhóm
      +-- Fallback đến agent mặc định
      |
      v
+------------------+
| Agent Selector   |
+------------------+
      |
      +-- agent: "personal"
      +-- agent: "work"
      +-- agent: "assistant"
      |
      v
+------------------+
| Target Agent     |
+------------------+
```

### Các Chiến Lược Định Tuyến

#### 1. Định Tuyến Theo Channel và Người Gửi

```json5
{
  agents: {
    agents: [
      { id: "personal", model: "anthropic/claude-opus-4-5" },
      { id: "work", model: "openai/gpt-5.2" },
      { id: "family", model: "anthropic/claude-sonnet-4" }
    ],
    routing: {
      // Định tuyến theo WhatsApp
      whatsapp: {
        "+1234567890": { agent: "personal" },  // Số cá nhân
        "+0987654321": { agent: "work" },      // Số công việc
        "*": { agent: "family" }               // Mặc định cho WhatsApp
      },
      // Định tuyến theo Telegram
      telegram: {
        "boss_username": { agent: "work" },
        "friend_username": { agent: "personal" }
      },
      // Định tuyến theo Discord
      discord: {
        guilds: {
          "work-guild-id": { agent: "work" },
          "gaming-guild-id": { agent: "personal" }
        }
      }
    }
  }
}
```

#### 2. Định Tuyến Theo Nhóm

```json5
{
  agents: {
    routing: {
      telegram: {
        groups: {
          "-100123456789": { agent: "work" },     // Nhóm công việc
          "-100987654321": { agent: "personal" }, // Nhóm bạn bè
          "*": {
            agent: "assistant",
            requireMention: true  // Chỉ phản hồi khi được @mention
          }
        }
      }
    }
  }
}
```

#### 3. Định Tuyến Động Dựa Trên Nội Dung

OpenClaw hỗ trợ routing hooks để định tuyến động:

```typescript
// hooks/route-by-content.js
export async function routeMessage(context) {
  const { message, channel, sender } = context;

  // Phân tích nội dung tin nhắn
  if (message.text.includes('#work') || message.text.includes('#công-việc')) {
    return { agent: 'work' };
  }

  if (message.text.includes('#personal') || message.text.includes('#cá-nhân')) {
    return { agent: 'personal' };
  }

  // Định tuyến dựa trên thời gian
  const hour = new Date().getHours();
  if (hour >= 9 && hour <= 17) {
    return { agent: 'work' };  // Giờ làm việc
  }

  return { agent: 'personal' };  // Ngoài giờ làm việc
}
```

### Cấu Hình Routing Trong Config

```json5
{
  agents: {
    defaults: {
      workspace: "~/.openclaw/workspace",
      model: "anthropic/claude-opus-4-5"
    },
    agents: [
      {
        id: "main",
        model: "anthropic/claude-opus-4-5",
        workspace: "~/.openclaw/workspace/main",
        // Cấu hình riêng cho agent này
        thinkingLevel: "high",
        sandbox: { mode: "none" }
      },
      {
        id: "helper",
        model: "anthropic/claude-sonnet-4",
        workspace: "~/.openclaw/workspace/helper",
        thinkingLevel: "medium",
        // Agent helper có quyền hạn chế
        sandbox: {
          mode: "always",
          allowlist: ["read", "write", "bash"],
          denylist: ["browser", "canvas"]
        }
      }
    ],
    // Quy tắc routing
    routing: {
      default: { agent: "main" },
      whatsapp: {
        "+1234567890": { agent: "main" },
        "*": { agent: "helper" }
      }
    }
  }
}
```

## Giao Tiếp Agent-to-Agent (A2A)

OpenClaw cung cấp hệ thống giao tiếp Agent-to-Agent (A2A) mạnh mẽ, cho phép các agent tương tác và phối hợp với nhau.

### Kiến Trúc A2A

```
+------------------+       +------------------+
|    Agent A       |       |    Agent B       |
|   (Personal)     |       |    (Work)        |
+------------------+       +------------------+
        |                          |
        |   sessions_send()        |
        +------------------------->|
        |                          |
        |   sessions_history()     |
        |<-------------------------|
        |                          |
        +-------- Gateway ---------+
                    |
           +--------+--------+
           |                 |
    sessions_list()    sessions_spawn()
```

### Các Công Cụ A2A Tích Hợp Sẵn

#### 1. `sessions_list` - Khám Phá Sessions

Cho phép agent khám phá các session đang hoạt động:

```typescript
// Agent sử dụng công cụ sessions_list
const activeSessions = await sessions_list({
  filter: "active",      // active, all, idle
  limit: 10,
  includeTranscript: false
});

// Kết quả
{
  sessions: [
    {
      sessionKey: "agent:work:main",
      agentId: "work",
      lastActivity: "2026-01-31T02:45:00Z",
      contextTokens: 4500,
      displayName: "Work Assistant"
    },
    {
      sessionKey: "agent:personal:dm:+1234567890",
      agentId: "personal",
      lastActivity: "2026-01-31T02:30:00Z",
      contextTokens: 2100,
      displayName: "Personal Chat with Alice"
    }
  ]
}
```

#### 2. `sessions_send` - Gửi Tin Nhắn Đến Session Khác

Cho phép một agent gửi tin nhắn đến session/agent khác:

```typescript
// Agent Personal gửi tin nhắn đến Agent Work
await sessions_send({
  sessionKey: "agent:work:main",
  message: "Bạn có thể kiểm tra lịch họp ngày mai không?",
  priority: "normal",  // low, normal, high
  waitForResponse: true,
  timeout: 30000  // 30 giây
});

// Kết quả
{
  delivered: true,
  response: "Ngày mai bạn có 2 cuộc họp: 9:00 AM Standup, 2:00 PM Sprint Review",
  responseTime: 2500  // ms
}
```

#### 3. `sessions_history` - Truy Cập Lịch Sử Session

Cho phép agent đọc lịch sử của session khác (với quyền):

```typescript
// Đọc lịch sử của session work
const history = await sessions_history({
  sessionKey: "agent:work:main",
  limit: 20,
  since: "2026-01-30T00:00:00Z",
  format: "summary"  // full, summary, messages_only
});

// Kết quả
{
  messages: [
    {
      role: "user",
      content: "Tạo báo cáo tuần",
      timestamp: "2026-01-30T14:00:00Z"
    },
    {
      role: "assistant",
      content: "Đã tạo báo cáo...",
      timestamp: "2026-01-30T14:02:00Z",
      toolsUsed: ["read", "write"]
    }
  ],
  summary: "Session tập trung vào báo cáo và lập kế hoạch"
}
```

#### 4. `sessions_spawn` - Tạo Sub-Agent

Cho phép agent tạo sub-agent cho tác vụ cụ thể:

```typescript
// Tạo sub-agent cho tác vụ nghiên cứu
const subAgent = await sessions_spawn({
  parentSession: "agent:main:current",
  agentConfig: {
    id: "researcher",
    model: "anthropic/claude-sonnet-4",
    purpose: "Nghiên cứu tài liệu về AI agents",
    tools: ["web_search", "web_fetch", "read", "write"],
    maxTurns: 10,
    timeout: 300000  // 5 phút
  },
  task: "Tìm hiểu về các framework AI agent phổ biến và tóm tắt ưu nhược điểm",
  isolated: true,  // Session riêng biệt
  reportBack: true  // Báo cáo kết quả về parent
});

// Kết quả khi sub-agent hoàn thành
{
  sessionId: "agent:researcher:task-001",
  status: "completed",
  result: "Đã tìm thấy 5 framework chính...",
  tokensUsed: 15000,
  duration: 45000  // ms
}
```

### Mẫu Giao Tiếp A2A

#### Mẫu 1: Ủy Quyền Tác Vụ (Task Delegation)

```
+------------------+
|   Main Agent     |
+------------------+
        |
        | 1. Nhận yêu cầu phức tạp
        v
+------------------+
| Phân tích yêu cầu|
+------------------+
        |
        | 2. Quyết định ủy quyền
        v
+-------+--------+-------+
|       |        |       |
v       v        v       v
Research  Code   Data   Review
Agent    Agent  Agent   Agent
        |
        | 3. Thu thập kết quả
        v
+------------------+
| Tổng hợp & Phản hồi|
+------------------+
```

**Ví dụ code:**

```typescript
// Main agent ủy quyền tác vụ
async function handleComplexRequest(request) {
  // Phân tích yêu cầu
  const tasks = analyzeRequest(request);

  // Spawn các sub-agents song song
  const results = await Promise.all([
    sessions_spawn({
      agentConfig: { id: "researcher", purpose: "Nghiên cứu" },
      task: tasks.research
    }),
    sessions_spawn({
      agentConfig: { id: "coder", purpose: "Viết code" },
      task: tasks.coding
    }),
    sessions_spawn({
      agentConfig: { id: "reviewer", purpose: "Review" },
      task: tasks.review
    })
  ]);

  // Tổng hợp kết quả
  return synthesizeResults(results);
}
```

#### Mẫu 2: Chuỗi Xử Lý (Pipeline)

```
Request --> Agent A --> Agent B --> Agent C --> Response
            (Phân tích) (Xử lý)    (Format)
```

**Ví dụ code:**

```typescript
// Pipeline A2A
async function processPipeline(input) {
  // Bước 1: Agent phân tích
  const analysis = await sessions_send({
    sessionKey: "agent:analyzer:main",
    message: `Phân tích: ${input}`,
    waitForResponse: true
  });

  // Bước 2: Agent xử lý
  const processed = await sessions_send({
    sessionKey: "agent:processor:main",
    message: `Xử lý dựa trên: ${analysis.response}`,
    waitForResponse: true
  });

  // Bước 3: Agent format
  const formatted = await sessions_send({
    sessionKey: "agent:formatter:main",
    message: `Format kết quả: ${processed.response}`,
    waitForResponse: true
  });

  return formatted.response;
}
```

#### Mẫu 3: Hội Thoại Đa Agent (Multi-Agent Conversation)

```
+---------+     +---------+     +---------+
| Agent A |<--->| Gateway |<--->| Agent B |
+---------+     +---------+     +---------+
                    ^
                    |
                +---------+
                | Agent C |
                +---------+
```

**Ví dụ code:**

```typescript
// Hội thoại đa agent
async function multiAgentDiscussion(topic) {
  const agents = ["expert-a", "expert-b", "moderator"];
  let conversation = [];

  // Bắt đầu thảo luận
  let currentMessage = `Thảo luận về: ${topic}`;

  for (let round = 0; round < 3; round++) {
    for (const agentId of agents) {
      const response = await sessions_send({
        sessionKey: `agent:${agentId}:main`,
        message: currentMessage,
        context: conversation,  // Chia sẻ ngữ cảnh
        waitForResponse: true
      });

      conversation.push({
        agent: agentId,
        message: response.response,
        round: round
      });

      currentMessage = response.response;
    }
  }

  // Moderator tổng kết
  return await sessions_send({
    sessionKey: "agent:moderator:main",
    message: "Tổng kết cuộc thảo luận",
    context: conversation,
    waitForResponse: true
  });
}
```

### Bảo Mật A2A

#### Quyền Truy Cập Session

```json5
{
  agents: {
    agents: [
      {
        id: "main",
        a2a: {
          // Ai có thể gửi tin nhắn đến agent này
          allowIncoming: ["helper", "researcher"],
          // Agent này có thể gửi đến ai
          allowOutgoing: ["*"],  // Tất cả
          // Quyền đọc lịch sử
          historyAccess: {
            "helper": "summary",     // Chỉ tóm tắt
            "researcher": "full",    // Toàn bộ
            "*": "none"              // Mặc định: không
          }
        }
      },
      {
        id: "helper",
        a2a: {
          allowIncoming: ["main"],
          allowOutgoing: ["main"],
          historyAccess: { "*": "none" }
        }
      }
    ]
  }
}
```

#### Sandbox Cho Sub-Agents

```json5
{
  agents: {
    spawnDefaults: {
      // Cấu hình mặc định cho sub-agents
      sandbox: {
        mode: "always",
        workspaceAccess: "ro",  // Chỉ đọc
        allowlist: ["read", "web_search"],
        denylist: ["bash", "write", "canvas", "sessions_spawn"],
        maxTokens: 50000,
        timeout: 300000  // 5 phút
      },
      // Giới hạn spawn chain
      maxSpawnDepth: 2,  // Sub-agent không thể spawn sub-sub-agent
      maxConcurrentSpawns: 3
    }
  }
}
```

### Ứng Dụng Thực Tế A2A

#### 1. Trợ Lý Đa Chuyên Môn

```json5
{
  agents: {
    agents: [
      { id: "coordinator", model: "claude-opus-4-5", purpose: "Điều phối" },
      { id: "coder", model: "claude-sonnet-4", purpose: "Lập trình" },
      { id: "writer", model: "claude-sonnet-4", purpose: "Viết văn bản" },
      { id: "researcher", model: "claude-haiku", purpose: "Nghiên cứu nhanh" }
    ],
    routing: {
      default: { agent: "coordinator" }  // Mọi yêu cầu đến coordinator trước
    }
  }
}
```

#### 2. Hệ Thống Review Code

```
Developer Request
       |
       v
+------------------+
| Coordinator      |
+------------------+
       |
       +---> Code Analyzer (phân tích cú pháp)
       +---> Security Checker (kiểm tra bảo mật)
       +---> Performance Reviewer (đánh giá hiệu suất)
       +---> Style Checker (kiểm tra style)
       |
       v
+------------------+
| Tổng hợp Review  |
+------------------+
```

#### 3. Nghiên Cứu Tự Động

```typescript
// Hệ thống nghiên cứu với nhiều agents
const researchSystem = {
  searcher: { purpose: "Tìm kiếm thông tin", tools: ["web_search"] },
  reader: { purpose: "Đọc và trích xuất", tools: ["web_fetch", "read"] },
  summarizer: { purpose: "Tóm tắt", tools: ["write"] },
  critic: { purpose: "Đánh giá độ tin cậy", tools: ["web_search"] }
};

async function conductResearch(topic) {
  // Bước 1: Tìm kiếm
  const sources = await sessions_spawn({
    agentConfig: researchSystem.searcher,
    task: `Tìm 10 nguồn uy tín về: ${topic}`
  });

  // Bước 2: Đọc song song
  const readings = await Promise.all(
    sources.result.map(source =>
      sessions_spawn({
        agentConfig: researchSystem.reader,
        task: `Đọc và trích xuất thông tin từ: ${source}`
      })
    )
  );

  // Bước 3: Tóm tắt
  const summary = await sessions_spawn({
    agentConfig: researchSystem.summarizer,
    task: `Tóm tắt các thông tin: ${JSON.stringify(readings)}`
  });

  // Bước 4: Đánh giá
  const evaluation = await sessions_spawn({
    agentConfig: researchSystem.critic,
    task: `Đánh giá độ tin cậy của: ${summary.result}`
  });

  return { summary: summary.result, evaluation: evaluation.result };
}
```

### So Sánh A2A Với Các Framework Khác

| Tính năng | OpenClaw A2A | LangChain | AutoGen | CrewAI |
|-----------|--------------|-----------|---------|--------|
| Session Discovery | ✅ Có | ❌ Không | ⚠️ Hạn chế | ❌ Không |
| Cross-Session Messaging | ✅ Có | ❌ Không | ⚠️ Trong group | ⚠️ Trong crew |
| History Access | ✅ Có | ❌ Không | ⚠️ Hạn chế | ❌ Không |
| Dynamic Spawning | ✅ Có | ⚠️ Manual | ✅ Có | ⚠️ Predefined |
| Security Controls | ✅ Chi tiết | ❌ Không | ⚠️ Cơ bản | ⚠️ Cơ bản |
| Persistent Sessions | ✅ Có | ❌ Không | ❌ Không | ❌ Không |

### Điểm Chính Cho Agents Framework

Dựa trên nghiên cứu OpenClaw, các tính năng A2A nên triển khai trong agents_framework:

1. **Session Registry**: Đăng ký và khám phá các agent sessions
2. **Inter-Agent Messaging**: Giao tiếp trực tiếp giữa các agents
3. **Shared Context**: Chia sẻ ngữ cảnh có kiểm soát
4. **Task Delegation**: Ủy quyền tác vụ với spawn sub-agents
5. **Pipeline Support**: Hỗ trợ chuỗi xử lý agent
6. **Security Controls**: Kiểm soát quyền truy cập chi tiết
