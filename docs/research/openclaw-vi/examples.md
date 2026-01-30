# Ví Dụ Code OpenClaw

Tài liệu này cung cấp các ví dụ code thực tế để làm việc với OpenClaw.

## Cài Đặt

### Cài Đặt npm/pnpm

```bash
# Cài đặt toàn cục với npm
npm install -g openclaw@latest

# Hoặc với pnpm
pnpm add -g openclaw@latest

# Xác minh cài đặt
openclaw --version
```

### Cài Đặt Từ Source

```bash
# Clone repository
git clone https://github.com/openclaw/openclaw.git
cd openclaw

# Cài đặt dependencies
pnpm install

# Build UI
pnpm ui:build

# Build TypeScript
pnpm build

# Chạy onboarding
pnpm openclaw onboard --install-daemon

# Chế độ phát triển (auto-reload)
pnpm gateway:watch
```

### Cài Đặt Docker

```bash
# Sử dụng Docker Compose
git clone https://github.com/openclaw/openclaw.git
cd openclaw

# Chạy script thiết lập
./docker-setup.sh

# Khởi động với Docker Compose
docker-compose up -d
```

## Bắt Đầu Nhanh

### 1. Onboarding

```bash
# Chạy wizard onboarding (khuyến nghị)
openclaw onboard --install-daemon

# Wizard sẽ:
# - Tạo ~/.openclaw/openclaw.json
# - Khởi tạo các file workspace
# - Thiết lập xác thực
# - Cấu hình channels
# - Cài đặt gateway daemon
```

### 2. Thiết Lập Thủ Công

```bash
# Khởi tạo workspace
openclaw setup

# Cấu hình nhà cung cấp model
openclaw config set agent.model "anthropic/claude-opus-4-5"

# Thêm API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Khởi động gateway
openclaw gateway --port 18789 --verbose
```

### 3. Tương Tác Đầu Tiên

```bash
# Gửi tin nhắn đến agent
openclaw agent --message "Xin chào, bạn có thể làm gì?"

# Với chế độ thinking mở rộng
openclaw agent --message "Giải thích về máy tính lượng tử" --thinking high
```

## Ví Dụ Cấu Hình

### Cấu Hình Tối Thiểu

```json5
// ~/.openclaw/openclaw.json
{
  agent: {
    model: "anthropic/claude-opus-4-5"
  }
}
```

### Cấu Hình WhatsApp

```json5
{
  agent: {
    model: "anthropic/claude-opus-4-5"
  },
  channels: {
    whatsapp: {
      enabled: true,
      // Chỉ cho phép tin nhắn từ các số này
      allowFrom: [
        "+1234567890",
        "+0987654321"
      ],
      // Cài đặt nhóm
      groups: {
        "*": {
          requireMention: true,  // Phải @mention trong nhóm
          allowFrom: []          // Danh sách cho phép nhóm bổ sung
        }
      }
    }
  }
}
```

### Cấu Hình Telegram

```json5
{
  agent: {
    model: "anthropic/claude-opus-4-5"
  },
  channels: {
    telegram: {
      botToken: "123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
      allowFrom: ["username1", "username2"],
      groups: {
        "*": {
          requireMention: true
        }
      }
    }
  }
}
```

### Cấu Hình Discord

```json5
{
  agent: {
    model: "anthropic/claude-opus-4-5"
  },
  channels: {
    discord: {
      token: "your-discord-bot-token",
      dm: {
        policy: "pairing",      // Yêu cầu ghép nối cho DMs
        allowFrom: []           // ID người dùng được phép
      },
      guilds: {
        "guild-id-123": {
          channels: ["channel-id-456"],
          requireMention: true
        }
      }
    }
  }
}
```

### Cấu Hình Đa Kênh

```json5
{
  agent: {
    model: "anthropic/claude-opus-4-5"
  },
  channels: {
    whatsapp: {
      enabled: true,
      allowFrom: ["+1234567890"]
    },
    telegram: {
      botToken: "your-bot-token",
      allowFrom: ["yourusername"]
    },
    discord: {
      token: "your-discord-token",
      dm: { policy: "pairing" }
    },
    slack: {
      botToken: "xoxb-...",
      appToken: "xapp-...",
      dm: { policy: "pairing" }
    }
  }
}
```

### Cấu Hình Đa Agent

```json5
{
  agents: {
    defaults: {
      workspace: "~/.openclaw/workspace",
      model: "anthropic/claude-opus-4-5",
      thinkingLevel: "medium"
    },
    agents: [
      {
        id: "personal",
        model: "anthropic/claude-opus-4-5",
        workspace: "~/.openclaw/workspace/personal"
      },
      {
        id: "work",
        model: "openai/gpt-5.2",
        workspace: "~/.openclaw/workspace/work"
      }
    ],
    routing: {
      whatsapp: {
        "+1234567890": { agent: "personal" },
        "+0987654321": { agent: "work" }
      },
      telegram: {
        "personal_bot": { agent: "personal" },
        "work_bot": { agent: "work" }
      }
    }
  }
}
```

### Cấu Hình Session

```json5
{
  session: {
    // Phạm vi session DM
    dmScope: "main",  // hoặc "per-peer", "per-channel-peer"

    // Chính sách reset session
    reset: {
      mode: "daily",
      atHour: 4,          // 4 AM giờ địa phương
      idleMinutes: 120    // Cũng reset sau 2 giờ không hoạt động
    },

    // Ghi đè theo loại
    resetByType: {
      dm: { mode: "idle", idleMinutes: 240 },
      group: { mode: "daily", atHour: 4 },
      thread: { mode: "idle", idleMinutes: 60 }
    },

    // Liên kết danh tính qua các channel
    identityLinks: {
      alice: [
        "telegram:123456789",
        "discord:987654321012345678",
        "whatsapp:+1234567890"
      ]
    }
  }
}
```

### Cấu Hình Bảo Mật

```json5
{
  gateway: {
    port: 18789,
    bind: "loopback",           // Chỉ kết nối cục bộ
    auth: {
      mode: "password",
      password: "mật-khẩu-bảo-mật-của-bạn",
      allowLocal: true,         // Tự động phê duyệt loopback
      allowTailscale: true      // Tin tưởng danh tính Tailscale
    }
  },
  agents: {
    defaults: {
      sandbox: {
        mode: "non-main",       // Sandbox các session không phải main
        workspaceAccess: "ro",  // Chỉ đọc workspace trong sandbox
        allowlist: [
          "bash", "read", "write", "edit", "glob", "grep"
        ],
        denylist: [
          "browser", "canvas", "nodes", "cron"
        ]
      }
    }
  }
}
```

### Cấu Hình Bộ Nhớ

```json5
{
  agents: {
    defaults: {
      memorySearch: {
        enabled: true,
        provider: "openai",
        model: "text-embedding-3-small",

        // Đường dẫn bổ sung để đánh chỉ mục
        extraPaths: [
          "../team-docs",
          "/đường/dẫn/đến/ghi-chú"
        ],

        // Cài đặt tìm kiếm hybrid
        query: {
          hybrid: {
            enabled: true,
            vectorWeight: 0.7,
            textWeight: 0.3
          }
        },

        // Caching
        cache: {
          enabled: true,
          maxEntries: 50000
        }
      },

      // Flush bộ nhớ trước khi nén
      compaction: {
        reserveTokensFloor: 20000,
        memoryFlush: {
          enabled: true,
          softThresholdTokens: 4000
        }
      }
    }
  }
}
```

### Cấu Hình Công Việc Cron

```json5
{
  cron: {
    jobs: [
      {
        id: "morning-briefing",
        schedule: "0 7 * * *",          // 7 AM hàng ngày
        message: "Cho tôi bản tóm tắt buổi sáng",
        agent: "personal",
        isolated: true,                 // Session mới
        channel: "telegram",
        target: "123456789"
      },
      {
        id: "weekly-backup",
        schedule: "0 2 * * 0",          // Chủ nhật 2 AM
        message: "Chạy quy trình backup hàng tuần",
        thinking: "high"
      },
      {
        id: "news-digest",
        schedule: "0 18 * * 1-5",       // Ngày thường 6 PM
        message: "Tóm tắt tin tức công nghệ hôm nay",
        channel: "discord",
        target: "channel-id"
      }
    ]
  }
}
```

### Cấu Hình Trình Duyệt

```json5
{
  browser: {
    enabled: true,
    headless: false,           // Hiển thị cửa sổ trình duyệt
    chromePath: null,          // Tự động phát hiện
    timeout: 30000,
    profiles: {
      default: {
        dataDir: "~/.openclaw/browser/default"
      },
      work: {
        dataDir: "~/.openclaw/browser/work"
      }
    }
  }
}
```

## Ví Dụ Sử Dụng CLI

### Quản Lý Gateway

```bash
# Khởi động gateway ở foreground
openclaw gateway run --port 18789 --verbose

# Khởi động như background daemon
openclaw gateway run --daemon

# Dừng gateway
openclaw gateway stop

# Kiểm tra trạng thái gateway
openclaw gateway status

# Khởi động lại gateway
openclaw gateway restart
```

### Tương Tác Agent

```bash
# Tin nhắn đơn giản
openclaw agent --message "Thời tiết hôm nay thế nào?"

# Với chế độ thinking
openclaw agent --message "Phân tích code này" --thinking high

# Chỉ định model
openclaw agent --message "Xin chào" --model "openai/gpt-5.2"

# Từ file
openclaw agent --file ./prompt.txt

# Chế độ tương tác
openclaw agent --interactive
```

### Gửi Tin Nhắn

```bash
# Gửi đến số WhatsApp
openclaw message send \
  --channel whatsapp \
  --to "+1234567890" \
  --message "Xin chào từ OpenClaw"

# Gửi đến Telegram
openclaw message send \
  --channel telegram \
  --to "username" \
  --message "Xin chào!"

# Gửi với tệp đính kèm
openclaw message send \
  --channel whatsapp \
  --to "+1234567890" \
  --message "Xem cái này" \
  --file ./image.png
```

### Quản Lý Channel

```bash
# Kiểm tra trạng thái tất cả channel
openclaw channels status

# Trạng thái chi tiết với probes
openclaw channels status --deep

# Đăng nhập WhatsApp (mã QR)
openclaw channels login whatsapp

# Đăng xuất khỏi channel
openclaw channels logout telegram
```

### Quản Lý Session

```bash
# Liệt kê tất cả sessions
openclaw sessions list

# Liệt kê sessions hoạt động (60 phút cuối)
openclaw sessions list --active 60

# Đầu ra JSON
openclaw sessions list --json

# Reset session cụ thể
openclaw sessions reset "agent:default:main"

# Xóa tất cả sessions
openclaw sessions clear --confirm
```

### Quản Lý Cấu Hình

```bash
# Hiển thị config hiện tại
openclaw config get

# Lấy giá trị cụ thể
openclaw config get agent.model

# Đặt giá trị
openclaw config set agent.model "openai/gpt-5.2"

# Đặt cấu hình channel
openclaw config set channels.telegram.botToken "123:ABC"

# Xóa giá trị
openclaw config delete channels.discord
```

### Chẩn Đoán

```bash
# Chạy kiểm tra chẩn đoán
openclaw doctor

# Chẩn đoán chi tiết
openclaw doctor --verbose

# Kiểm tra lĩnh vực cụ thể
openclaw doctor --check channels
openclaw doctor --check auth
openclaw doctor --check workspace
```

## Ví Dụ Workspace

### AGENTS.md (Hướng Dẫn)

```markdown
# Hướng Dẫn Vận Hành

## Danh Tính
Bạn là Molty, một trợ lý AI hữu ích.

## Quy Tắc Cốt Lõi
1. Ngắn gọn và trực tiếp
2. Hỏi câu hỏi làm rõ khi cần
3. Viết code khi nó giúp giải thích
4. Nhớ thông tin quan trọng

## Bộ Nhớ
- Viết ghi chú hàng ngày vào memory/YYYY-MM-DD.md
- Cập nhật MEMORY.md cho các sự kiện dài hạn
- Đọc các file bộ nhớ khi ngữ cảnh có thể giúp ích

## Công Cụ
- Sử dụng bash cho các hoạt động hệ thống
- Sử dụng browser để nghiên cứu web
- Sử dụng canvas cho đầu ra trực quan
```

### SOUL.md (Tính Cách)

```markdown
# Tâm Hồn

## Giọng Nói
- Thân thiện nhưng chuyên nghiệp
- Rõ ràng và ngắn gọn
- Thỉnh thoảng sử dụng câu đùa về tôm hùm

## Ranh Giới
- Không bao giờ chia sẻ API keys hoặc mật khẩu
- Không thực thi lệnh phá hoại mà không xác nhận
- Tôn trọng quyền riêng tư người dùng

## Sở Thích
- Ưu tiên ví dụ code hơn giải thích dài dòng
- Sử dụng định dạng markdown
- Chia các tác vụ phức tạp thành các bước
```

### TOOLS.md (Hướng Dẫn Công Cụ)

```markdown
# Ghi Chú Công Cụ

## bash
- Luôn sử dụng đường dẫn tuyệt đối
- Xác nhận các hoạt động phá hoại
- Sử dụng timeout cho các lệnh chạy lâu

## browser
- Ưu tiên headless cho các tác vụ nền
- Chụp ảnh màn hình để xác minh
- Đóng tabs khi xong

## memory_search
- Sử dụng để nhớ lại các cuộc hội thoại trước
- Kiểm tra bộ nhớ trước khi hỏi các câu hỏi lặp lại
```

### Ví Dụ Skill (SKILL.md)

```markdown
---
name: git-helper
description: Trợ lý hoạt động Git
requireEnv: []
---

# Git Helper

Khi người dùng hỏi về các hoạt động git:

1. Kiểm tra branch hiện tại: `git branch --show-current`
2. Kiểm tra trạng thái: `git status --short`
3. Đối với commits, sử dụng định dạng conventional commit

## Các Hoạt Động Thường Dùng

- **Branch mới**: `git checkout -b feature/<tên>`
- **Commit**: `git commit -m "type: mô tả"`
- **Push**: `git push -u origin <branch>`
```

## Ví Dụ Gateway API

### Kết Nối WebSocket (TypeScript)

```typescript
import WebSocket from 'ws';

const ws = new WebSocket('ws://127.0.0.1:18789');

// Handler kết nối
ws.on('open', () => {
  // Gửi yêu cầu kết nối
  ws.send(JSON.stringify({
    type: 'req',
    id: 'connect-1',
    method: 'connect',
    params: {
      auth: { token: 'your-gateway-token' },
      clientId: 'my-client',
      deviceId: 'device-123'
    }
  }));
});

// Handler tin nhắn
ws.on('message', (data) => {
  const msg = JSON.parse(data.toString());

  if (msg.type === 'res' && msg.id === 'connect-1') {
    console.log('Đã kết nối:', msg.payload);
  }

  if (msg.type === 'event') {
    console.log('Sự kiện:', msg.event, msg.payload);
  }
});
```

### Gửi Yêu Cầu Agent

```typescript
// Sau khi kết nối thành công...

function sendAgentRequest(message: string) {
  const requestId = `agent-${Date.now()}`;

  ws.send(JSON.stringify({
    type: 'req',
    id: requestId,
    method: 'agent',
    params: {
      message,
      sessionKey: 'agent:default:main',
      options: {
        thinkingLevel: 'medium',
        stream: true
      }
    }
  }));

  return requestId;
}

// Xử lý sự kiện agent
ws.on('message', (data) => {
  const msg = JSON.parse(data.toString());

  if (msg.type === 'event' && msg.event === 'agent') {
    const { event, payload } = msg.payload;

    switch (event) {
      case 'thinking':
        console.log('Đang suy nghĩ:', payload.content);
        break;
      case 'tool_start':
        console.log('Công cụ:', payload.name);
        break;
      case 'text_delta':
        process.stdout.write(payload.delta);
        break;
      case 'done':
        console.log('\nHoàn thành');
        break;
    }
  }
});
```

### Gửi Tin Nhắn Qua API

```typescript
function sendMessage(channel: string, to: string, message: string) {
  const requestId = `send-${Date.now()}`;

  ws.send(JSON.stringify({
    type: 'req',
    id: requestId,
    method: 'send',
    params: {
      channel,
      to,
      message,
      idempotencyKey: requestId
    }
  }));

  return requestId;
}

// Sử dụng
sendMessage('telegram', 'username', 'Xin chào từ API!');
```

### Liệt Kê Sessions

```typescript
function listSessions() {
  const requestId = `sessions-${Date.now()}`;

  ws.send(JSON.stringify({
    type: 'req',
    id: requestId,
    method: 'sessions.list',
    params: {}
  }));
}

// Xử lý phản hồi
ws.on('message', (data) => {
  const msg = JSON.parse(data.toString());

  if (msg.type === 'res' && msg.ok) {
    console.log('Sessions:', msg.payload.sessions);
  }
});
```

## Phát Triển Extension

### Tạo Extension Channel

```typescript
// extensions/my-channel/src/index.ts
import { definePlugin } from 'openclaw/plugin-sdk';

export default definePlugin({
  name: 'my-channel',
  version: '1.0.0',

  channels: [{
    id: 'my-channel',
    name: 'My Custom Channel',

    configSchema: {
      type: 'object',
      properties: {
        apiKey: { type: 'string' },
        webhookUrl: { type: 'string' }
      },
      required: ['apiKey']
    },

    async start(config, context) {
      // Khởi tạo channel của bạn
      console.log('Đang khởi động my-channel với config:', config);

      // Thiết lập handler tin nhắn
      context.onInbound((message) => {
        // Chuyển tiếp đến OpenClaw
        context.deliver({
          from: message.sender,
          text: message.content,
          channel: 'my-channel'
        });
      });
    },

    async stop() {
      // Dọn dẹp
      console.log('Đang dừng my-channel');
    },

    async send(message) {
      // Gửi tin nhắn đi
      console.log('Đang gửi:', message);

      // Gọi API của bạn
      await fetch('https://api.example.com/send', {
        method: 'POST',
        body: JSON.stringify({
          to: message.to,
          text: message.text
        })
      });
    },

    isConnected() {
      return true;
    },

    getStatus() {
      return {
        connected: true,
        lastActivity: new Date().toISOString()
      };
    }
  }]
});
```

### Cấu Hình Package

```json
{
  "name": "openclaw-my-channel",
  "version": "1.0.0",
  "main": "dist/index.js",
  "dependencies": {
    "openclaw": "*"
  },
  "peerDependencies": {
    "openclaw": ">=2024.1.0"
  }
}
```

## Testing

### Chạy Tests

```bash
# Chạy tất cả tests
pnpm test

# Chạy với coverage
pnpm test:coverage

# Chạy file test cụ thể
pnpm test src/gateway/session-utils.test.ts

# Chạy e2e tests
pnpm test:e2e

# Chạy live tests (yêu cầu API keys)
CLAWDBOT_LIVE_TEST=1 pnpm test:live
```

### Viết Tests

```typescript
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { createTestGateway } from '../test-helpers/gateway';

describe('Quản Lý Session', () => {
  let gateway: TestGateway;

  beforeAll(async () => {
    gateway = await createTestGateway();
  });

  afterAll(async () => {
    await gateway.close();
  });

  it('nên tạo session mới', async () => {
    const result = await gateway.call('sessions.list', {});
    expect(result.sessions).toBeInstanceOf(Array);
  });

  it('nên reset session', async () => {
    const result = await gateway.call('sessions.reset', {
      sessionKey: 'agent:default:main'
    });
    expect(result.ok).toBe(true);
  });
});
```

## Ví Dụ Docker

### docker-compose.yml

```yaml
version: '3.8'

services:
  openclaw:
    build: .
    ports:
      - "18789:18789"
      - "18793:18793"
    volumes:
      - ./config:/root/.openclaw
      - ./workspace:/root/.openclaw/workspace
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENCLAW_GATEWAY_TOKEN=${GATEWAY_TOKEN}
    restart: unless-stopped
```

### Dockerfile

```dockerfile
FROM node:22-slim

WORKDIR /app

# Cài đặt dependencies
COPY package.json pnpm-lock.yaml ./
RUN npm install -g pnpm && pnpm install --frozen-lockfile

# Copy source
COPY . .

# Build
RUN pnpm build

# Expose ports
EXPOSE 18789 18793

# Chạy gateway
CMD ["node", "dist/entry.js", "gateway", "run", "--bind", "0.0.0.0"]
```
