# Các Thành Phần OpenClaw

Tài liệu này cung cấp tài liệu chi tiết về các thành phần chính của OpenClaw.

## Gateway Server

Gateway là server WebSocket trung tâm đóng vai trò là control plane cho tất cả các hoạt động OpenClaw.

### Các File Chính

```
src/gateway/
├── server.impl.ts          # Triển khai server chính
├── server.ts               # Exports server
├── client.ts               # Xử lý kết nối client
├── auth.ts                 # Logic xác thực
├── session-utils.ts        # Tiện ích session
├── server-channels.ts      # Quản lý channel
├── server-chat.ts          # Xử lý chat
├── server-methods.ts       # Registry phương thức RPC
├── protocol/               # Định nghĩa giao thức
└── server-methods/         # Triển khai phương thức riêng lẻ
```

### Các Phương Thức Gateway

| Phương thức | Mô tả |
|--------|-------------|
| `connect` | Bắt tay ban đầu với xác thực |
| `health` | Kiểm tra sức khỏe hệ thống |
| `status` | Trạng thái gateway |
| `agent` | Bắt đầu chạy agent |
| `send` | Gửi tin nhắn đến channel |
| `sessions.list` | Liệt kê các session đang hoạt động |
| `sessions.patch` | Cập nhật cài đặt session |
| `sessions.reset` | Reset một session |
| `config.get` | Lấy cấu hình |
| `config.patch` | Cập nhật cấu hình |
| `cron.list` | Liệt kê các công việc cron |
| `cron.trigger` | Kích hoạt thủ công công việc cron |

### Các Sự Kiện

| Sự kiện | Mô tả |
|-------|-------------|
| `agent` | Hoạt động agent (thinking, sử dụng công cụ, phản hồi) |
| `presence` | Thay đổi presence hệ thống |
| `tick` | Heartbeat định kỳ |
| `health` | Cập nhật trạng thái sức khỏe |
| `chat` | Sự kiện tin nhắn chat |
| `shutdown` | Gateway đang tắt |

## Agent Runtime

Agent runtime xử lý các tương tác LLM và thực thi công cụ.

### Các File Chính

```
src/agents/
├── agent-paths.ts          # Tiện ích đường dẫn workspace
├── agent-scope.ts          # Phạm vi agent
├── agent-config.ts         # Cấu hình agent
├── tools/                  # Định nghĩa công cụ
│   ├── bash.ts
│   ├── browser.ts
│   ├── canvas.ts
│   ├── read.ts
│   ├── write.ts
│   ├── edit.ts
│   └── ...
└── skills/                 # Loader skill
```

### Các Công Cụ Tích Hợp Sẵn

#### Công Cụ Hệ Thống File

| Công cụ | Mô tả |
|------|-------------|
| `read` | Đọc nội dung file |
| `write` | Ghi file ra đĩa |
| `edit` | Chỉnh sửa file với patches |
| `glob` | Tìm file theo pattern |
| `grep` | Tìm kiếm nội dung file |

#### Công Cụ Thực Thi

| Công cụ | Mô tả |
|------|-------------|
| `bash` | Thực thi lệnh shell |
| `process` | Quản lý tiến trình |
| `apply_patch` | Áp dụng patches unified diff |

#### Công Cụ Trình Duyệt

| Công cụ | Mô tả |
|------|-------------|
| `browser_navigate` | Điều hướng đến URL |
| `browser_action` | Thực hiện hành động trình duyệt |
| `browser_snapshot` | Chụp nhanh trang |
| `browser_screenshot` | Chụp ảnh màn hình |
| `browser_upload` | Tải lên file |

#### Công Cụ Canvas

| Công cụ | Mô tả |
|------|-------------|
| `canvas_push` | Đẩy nội dung HTML |
| `canvas_reset` | Reset canvas |
| `canvas_eval` | Đánh giá JS trong canvas |
| `canvas_snapshot` | Chụp nhanh trạng thái canvas |

#### Công Cụ Giao Tiếp

| Công cụ | Mô tả |
|------|-------------|
| `send_message` | Gửi qua channel nhắn tin |
| `sessions_list` | Liệt kê các session đang hoạt động |
| `sessions_send` | Gửi đến session khác |
| `sessions_history` | Lấy lịch sử session |
| `sessions_spawn` | Spawn sub-agent |

#### Công Cụ Tiện Ích

| Công cụ | Mô tả |
|------|-------------|
| `think` | Chế độ thinking mở rộng |
| `memory_search` | Tìm kiếm file bộ nhớ |
| `memory_get` | Đọc file bộ nhớ |
| `web_search` | Tìm kiếm web |
| `web_fetch` | Lấy nội dung web |

### Cấu Hình Công Cụ

```json5
{
  tools: {
    exec: {
      enabled: true,
      allowlist: ["bash", "process"],
      denylist: [],
      applyPatch: true,
      timeout: 120000
    },
    browser: {
      enabled: true,
      chromePath: "/đường/dẫn/đến/chrome",
      headless: true
    },
    canvas: {
      enabled: true,
      port: 18793
    }
  }
}
```

## Quản Lý Session

Sessions theo dõi trạng thái hội thoại qua các tin nhắn.

### Các File Chính

```
src/sessions/
├── level-overrides.ts      # Ghi đè cấp độ thinking
├── model-overrides.ts      # Ghi đè lựa chọn model
├── send-policy.ts          # Chính sách giao hàng tin nhắn
├── session-key-utils.ts    # Tiện ích session key
├── session-label.ts        # Gán nhãn session
└── transcript-events.ts    # Xử lý sự kiện bản ghi

src/gateway/
├── session-utils.ts        # Tiện ích session cốt lõi
├── session-utils.fs.ts     # Hoạt động file session
├── sessions-patch.ts       # Patching session
└── sessions-resolve.ts     # Giải quyết session
```

### Cấu Trúc Dữ Liệu Session

```typescript
interface Session {
  sessionId: string;           // ID duy nhất
  sessionKey: string;          // Khóa tra cứu
  agentId: string;             // Agent liên kết
  updatedAt: string;           // Timestamp cập nhật cuối
  inputTokens: number;         // Sử dụng token
  outputTokens: number;
  totalTokens: number;
  contextTokens: number;       // Kích thước ngữ cảnh hiện tại
  thinkingLevel?: string;      // Ghi đè
  verboseLevel?: boolean;      // Ghi đè
  model?: string;              // Ghi đè
  sendPolicy?: string;         // Chính sách giao hàng
  groupActivation?: string;    // Chế độ kích hoạt nhóm
  displayName?: string;        // Nhãn UI
  channel?: string;            // Channel nguồn
  origin?: SessionOrigin;      // Metadata nguồn
}
```

### Chính Sách Reset Session

```json5
{
  session: {
    reset: {
      mode: "daily",           // daily, idle, hoặc combined
      atHour: 4,               // Giờ reset (giờ địa phương)
      idleMinutes: 120         // Timeout không hoạt động
    },
    resetByType: {
      dm: { mode: "idle", idleMinutes: 240 },
      group: { mode: "daily", atHour: 4 },
      thread: { mode: "idle", idleMinutes: 60 }
    },
    resetByChannel: {
      discord: { mode: "idle", idleMinutes: 10080 }
    },
    resetTriggers: ["/new", "/reset"]
  }
}
```

## Hệ Thống Channel

Channels là các adapter nền tảng nhắn tin.

### Các Channel Cốt Lõi

| Channel | Thư viện | Vị trí |
|---------|---------|----------|
| WhatsApp | Baileys | `extensions/whatsapp/` |
| Telegram | grammY | `extensions/telegram/` |
| Slack | Bolt | `extensions/slack/` |
| Discord | discord.js | `extensions/discord/` |
| Signal | signal-cli | `extensions/signal/` |
| iMessage | imsg | `extensions/imessage/` |

### Các Channel Mở Rộng

| Channel | Mô tả |
|---------|-------------|
| Microsoft Teams | Tích hợp Bot Framework |
| Google Chat | Chat API |
| Matrix | Giao thức Matrix |
| BlueBubbles | iMessage thay thế |
| Zalo | Nhắn tin Việt Nam |
| Mattermost | Chat tự lưu trữ |
| Line | Nhắn tin LINE |
| Nostr | Mạng xã hội phi tập trung |
| Twitch | Chat stream |

### Cấu Hình Channel

```json5
{
  channels: {
    whatsapp: {
      enabled: true,
      allowFrom: ["+1234567890"],
      groups: {
        "*": { requireMention: true }
      }
    },
    telegram: {
      botToken: "123456:ABCDEF",
      allowFrom: ["username"],
      groups: {
        "*": { requireMention: true }
      }
    },
    discord: {
      token: "discord-bot-token",
      dm: {
        policy: "pairing",
        allowFrom: []
      },
      guilds: {
        "guild-id": {
          channels: ["channel-id"]
        }
      }
    },
    slack: {
      botToken: "xoxb-...",
      appToken: "xapp-...",
      dm: {
        policy: "pairing"
      }
    }
  }
}
```

## Hệ Thống Bộ Nhớ

Hệ thống bộ nhớ cung cấp lưu trữ kiến thức bền vững.

### Các File Chính

```
src/memory/
├── memory-index.ts         # Đánh chỉ mục bộ nhớ
├── memory-search.ts        # Triển khai tìm kiếm
├── embeddings/             # Nhà cung cấp embedding
│   ├── openai.ts
│   ├── gemini.ts
│   └── local.ts
└── store/                  # Backend lưu trữ
    └── sqlite.ts

extensions/memory-core/     # Plugin bộ nhớ cốt lõi
extensions/memory-lancedb/  # Plugin LanceDB
```

### Cấu Hình Bộ Nhớ

```json5
{
  agents: {
    defaults: {
      memorySearch: {
        enabled: true,
        provider: "openai",           // openai, gemini, local
        model: "text-embedding-3-small",
        fallback: "openai",

        // Đường dẫn bổ sung để đánh chỉ mục
        extraPaths: ["../team-docs"],

        // Vector store
        store: {
          path: "~/.openclaw/memory/{agentId}.sqlite",
          vector: {
            enabled: true             // Sử dụng sqlite-vec
          }
        },

        // Tìm kiếm hybrid
        query: {
          hybrid: {
            enabled: true,
            vectorWeight: 0.7,
            textWeight: 0.3
          }
        },

        // Cài đặt cache
        cache: {
          enabled: true,
          maxEntries: 50000
        },

        // Thử nghiệm: bộ nhớ session
        experimental: {
          sessionMemory: false
        }
      }
    }
  }
}
```

### Các Công Cụ Bộ Nhớ

```typescript
// memory_search - Tìm kiếm ngữ nghĩa qua các file bộ nhớ
{
  query: string;           // Truy vấn tìm kiếm
  maxResults?: number;     // Kết quả tối đa (mặc định: 5)
}

// Phản hồi
{
  results: Array<{
    text: string;          // Snippet (tối đa 700 ký tự)
    path: string;          // Đường dẫn file
    lineStart: number;     // Dòng bắt đầu
    lineEnd: number;       // Dòng kết thúc
    score: number;         // Điểm liên quan
    provider: string;      // Nhà cung cấp embedding
    model: string;         // Model embedding
    fallback: boolean;     // Đã sử dụng fallback?
  }>;
}

// memory_get - Đọc nội dung file bộ nhớ
{
  path: string;            // Đường dẫn tương đối workspace
  startLine?: number;      // Dòng bắt đầu
  numLines?: number;       // Số dòng
}
```

## Hệ Thống Skills

Skills là các mẫu prompt và quy trình làm việc có thể tái sử dụng.

### Các File Chính

```
skills/                     # Skills đi kèm
├── sag/                    # Tìm kiếm và trả lời
│   └── SKILL.md
├── git/                    # Hoạt động Git
│   └── SKILL.md
├── imsg/                   # iMessage
│   └── SKILL.md
└── ...

src/agents/skills/          # Loader skill
```

### Vị Trí Skill

1. **Bundled**: Đi kèm với OpenClaw (`skills/`)
2. **Managed**: Người dùng cài đặt (`~/.openclaw/skills/`)
3. **Workspace**: Cụ thể theo dự án (`<workspace>/skills/`)

### Cấu Trúc Skill

```
skills/<tên-skill>/
├── SKILL.md            # Định nghĩa skill chính
├── README.md           # Tài liệu tùy chọn
└── assets/             # Assets tùy chọn
```

### Định Dạng SKILL.md

```markdown
---
name: my-skill
description: Làm điều gì đó hữu ích
requireEnv:
  - MY_API_KEY
requireConfig:
  - mySkill.enabled
---

# My Skill

Hướng dẫn cho agent về cách sử dụng skill này...

## Cách Sử Dụng

Khi người dùng hỏi về X, làm Y...
```

### Cấu Hình Skill

```json5
{
  skills: {
    // Bật/tắt skills cụ thể
    enabled: ["sag", "git", "imsg"],
    disabled: [],

    // Cấu hình cụ thể theo skill
    sag: {
      searchProvider: "brave"
    }
  }
}
```

## Hệ Thống Cấu Hình

OpenClaw sử dụng hệ thống cấu hình phân lớp.

### Các File Chính

```
src/config/
├── config.ts               # Tải cấu hình
├── config-schema.ts        # Định nghĩa schema
├── config-defaults.ts      # Giá trị mặc định
└── config-validation.ts    # Xác thực
```

### Vị Trí Cấu Hình

1. **File**: `~/.openclaw/openclaw.json` (hoặc JSON5)
2. **Environment**: Biến `OPENCLAW_*`
3. **CLI**: Tham số dòng lệnh
4. **Runtime**: Patches API Gateway

### Cấu Trúc Cấu Hình

```json5
{
  // Cài đặt agent
  agent: {
    model: "anthropic/claude-opus-4-5",
    skipBootstrap: false
  },

  // Thiết lập multi-agent
  agents: {
    defaults: {
      workspace: "~/.openclaw/workspace",
      model: "anthropic/claude-opus-4-5",
      thinkingLevel: "medium",
      sandbox: { mode: "non-main" }
    },
    agents: [
      { id: "personal", model: "anthropic/claude-opus-4-5" },
      { id: "work", model: "openai/gpt-5.2" }
    ],
    routing: { /* ánh xạ channel -> agent */ }
  },

  // Cài đặt gateway
  gateway: {
    port: 18789,
    bind: "loopback",
    auth: { mode: "password", password: "bí_mật" }
  },

  // Cài đặt session
  session: {
    dmScope: "main",
    reset: { mode: "daily", atHour: 4 }
  },

  // Cài đặt channel
  channels: {
    whatsapp: { /* ... */ },
    telegram: { /* ... */ },
    discord: { /* ... */ }
  },

  // Cài đặt công cụ
  tools: {
    exec: { enabled: true },
    browser: { enabled: true },
    canvas: { enabled: true }
  },

  // Cài đặt nhà cung cấp model
  models: {
    providers: {
      anthropic: { apiKey: "sk-..." },
      openai: { apiKey: "sk-..." }
    }
  },

  // Cài đặt plugin
  plugins: {
    slots: {
      memory: "memory-core"
    }
  }
}
```

## Hệ Thống CLI

CLI cung cấp truy cập dòng lệnh đến chức năng OpenClaw.

### Các File Chính

```
src/cli/
├── index.ts                # Điểm vào CLI
├── commands/               # Triển khai lệnh
├── progress.ts             # Chỉ báo tiến trình
└── prompts.ts              # Prompt tương tác

src/commands/
├── gateway.ts              # Lệnh gateway
├── agent.ts                # Lệnh agent
├── send.ts                 # Lệnh gửi
├── channels.ts             # Lệnh channel
├── sessions.ts             # Lệnh session
├── config.ts               # Lệnh config
├── doctor.ts               # Lệnh chẩn đoán
└── ...
```

### Các Lệnh Chính

```bash
# Quản lý gateway
openclaw gateway run          # Khởi động gateway
openclaw gateway stop         # Dừng gateway
openclaw gateway status       # Kiểm tra trạng thái

# Tương tác agent
openclaw agent --message "Xin chào"
openclaw agent --thinking high

# Nhắn tin
openclaw message send --to +1234567890 --message "Xin chào"

# Quản lý channel
openclaw channels status      # Trạng thái channel
openclaw channels login       # Thiết lập xác thực channel

# Quản lý session
openclaw sessions list        # Liệt kê sessions
openclaw sessions reset <key> # Reset session

# Cấu hình
openclaw config get           # Hiển thị config
openclaw config set <path> <value>

# Chẩn đoán
openclaw doctor               # Chạy chẩn đoán
openclaw status               # Trạng thái hệ thống

# Thiết lập
openclaw onboard              # Wizard onboarding
openclaw setup                # Khởi tạo workspace
```

## Điều Khiển Trình Duyệt

OpenClaw bao gồm khả năng tự động hóa trình duyệt.

### Các File Chính

```
src/browser/
├── browser.ts              # Controller trình duyệt chính
├── browser-pool.ts         # Pool instance trình duyệt
├── page-utils.ts           # Tiện ích trang
├── snapshot.ts             # Chụp nhanh trang
└── actions.ts              # Hành động trình duyệt
```

### Cấu Hình Trình Duyệt

```json5
{
  browser: {
    enabled: true,
    chromePath: "/đường/dẫn/đến/chrome",  // Tự động phát hiện nếu không đặt
    headless: false,
    profiles: {
      default: {
        dataDir: "~/.openclaw/browser/default"
      }
    },
    timeout: 30000,
    viewport: {
      width: 1280,
      height: 720
    }
  }
}
```

## Hệ Thống Canvas

Canvas cung cấp không gian làm việc trực quan điều khiển bởi agent.

### Các File Chính

```
src/canvas-host/
├── canvas-server.ts        # HTTP server canvas
├── a2ui/                   # Bundle A2UI
└── templates/              # Mẫu HTML
```

### Cấu Hình Canvas

```json5
{
  tools: {
    canvas: {
      enabled: true,
      port: 18793,
      host: "127.0.0.1"
    }
  }
}
```

### A2UI (Agent-to-UI)

A2UI là giao thức để agents tạo UI tương tác:

```typescript
// Agent đẩy nội dung
canvas_push({
  html: "<div>Xin chào Thế giới</div>",
  css: "div { color: blue; }",
  js: "console.log('đã tải')"
});

// Agent đánh giá JS
canvas_eval({
  expression: "document.body.innerHTML"
});

// Agent chụp nhanh
canvas_snapshot({
  format: "png",
  fullPage: true
});
```

## Hệ Thống Node

Nodes là các endpoint thiết bị cho khả năng cục bộ.

### Các File Chính

```
src/node-host/
├── node-server.ts          # Server node
├── node-registry.ts        # Đăng ký node
└── node-commands.ts        # Xử lý lệnh

// Ứng dụng di động
apps/ios/                   # Ứng dụng iOS
apps/android/               # Ứng dụng Android
apps/macos/                 # Ứng dụng macOS
```

### Các Lệnh Node

| Lệnh | Nền tảng | Mô tả |
|---------|----------|-------------|
| `canvas.push` | Tất cả | Đẩy HTML đến canvas |
| `canvas.reset` | Tất cả | Reset canvas |
| `camera.snap` | iOS/Android/macOS | Chụp ảnh |
| `camera.clip` | iOS/Android/macOS | Quay video |
| `screen.record` | iOS/Android/macOS | Quay màn hình |
| `location.get` | iOS/Android | Lấy vị trí |
| `system.notify` | macOS | Gửi thông báo |
| `system.run` | macOS | Chạy lệnh shell |

### Cấu Hình Node

```json5
{
  nodes: {
    autoApprove: false,      // Tự động phê duyệt nodes cục bộ
    policy: {
      camera: "prompt",      // always, prompt, deny
      location: "prompt",
      screen: "prompt"
    }
  }
}
```

## Hệ Thống Cron

Công việc cron cho phép chạy agent theo lịch.

### Các File Chính

```
src/cron/
├── cron-manager.ts         # Quản lý công việc cron
├── cron-parser.ts          # Phân tích lịch
└── cron-runner.ts          # Thực thi công việc
```

### Cấu Hình Cron

```json5
{
  cron: {
    jobs: [
      {
        id: "morning-briefing",
        schedule: "0 7 * * *",        // 7 AM hàng ngày
        message: "Cho tôi bản tóm tắt buổi sáng",
        agent: "personal",
        isolated: true,               // Session mới mỗi lần chạy
        channel: "telegram",          // Channel giao hàng
        target: "123456789"           // Mục tiêu giao hàng
      },
      {
        id: "weekly-report",
        schedule: "0 18 * * 5",       // Thứ 6 6 PM
        message: "Tạo tóm tắt tuần",
        thinking: "high"
      }
    ]
  }
}
```

## Hệ Thống Hooks

Hooks cho phép xử lý tùy chỉnh tại các điểm khác nhau.

### Các File Chính

```
src/hooks/
├── hooks.ts                # Thực thi hook
├── hook-types.ts           # Định nghĩa kiểu hook
└── hook-registry.ts        # Đăng ký hook
```

### Các Hooks Có Sẵn

| Hook | Điểm Kích hoạt |
|------|---------------|
| `inbound.before` | Trước khi xử lý tin nhắn đến |
| `inbound.after` | Sau khi xử lý tin nhắn đến |
| `outbound.before` | Trước khi gửi phản hồi |
| `outbound.after` | Sau khi gửi phản hồi |
| `agent.before` | Trước khi chạy agent |
| `agent.after` | Sau khi chạy agent |
| `tool.before` | Trước khi thực thi công cụ |
| `tool.after` | Sau khi thực thi công cụ |

### Cấu Hình Hook

```json5
{
  hooks: {
    "inbound.before": [
      {
        type: "script",
        path: "~/.openclaw/hooks/log-inbound.js"
      }
    ],
    "outbound.before": [
      {
        type: "webhook",
        url: "https://example.com/hook",
        method: "POST"
      }
    ]
  }
}
```
