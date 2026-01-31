# OpenClaw Code Examples

This document provides practical code examples for working with OpenClaw.

## Installation

### npm/pnpm Installation

```bash
# Install globally with npm
npm install -g openclaw@latest

# Or with pnpm
pnpm add -g openclaw@latest

# Verify installation
openclaw --version
```

### From Source

```bash
# Clone the repository
git clone https://github.com/openclaw/openclaw.git
cd openclaw

# Install dependencies
pnpm install

# Build UI
pnpm ui:build

# Build TypeScript
pnpm build

# Run onboarding
pnpm openclaw onboard --install-daemon

# Development mode (auto-reload)
pnpm gateway:watch
```

### Docker Installation

```bash
# Using Docker Compose
git clone https://github.com/openclaw/openclaw.git
cd openclaw

# Run setup script
./docker-setup.sh

# Start with Docker Compose
docker-compose up -d
```

## Quick Start

### 1. Onboarding

```bash
# Run the onboarding wizard (recommended)
openclaw onboard --install-daemon

# The wizard will:
# - Create ~/.openclaw/openclaw.json
# - Initialize workspace files
# - Set up authentication
# - Configure channels
# - Install gateway daemon
```

### 2. Manual Setup

```bash
# Initialize workspace
openclaw setup

# Configure model provider
openclaw config set agent.model "anthropic/claude-opus-4-5"

# Add API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Start gateway
openclaw gateway --port 18789 --verbose
```

### 3. First Interaction

```bash
# Send a message to the agent
openclaw agent --message "Hello, what can you do?"

# With extended thinking
openclaw agent --message "Explain quantum computing" --thinking high
```

## Configuration Examples

### Minimal Configuration

```json5
// ~/.openclaw/openclaw.json
{
  agent: {
    model: "anthropic/claude-opus-4-5"
  }
}
```

### WhatsApp Configuration

```json5
{
  agent: {
    model: "anthropic/claude-opus-4-5"
  },
  channels: {
    whatsapp: {
      enabled: true,
      // Only allow messages from these numbers
      allowFrom: [
        "+1234567890",
        "+0987654321"
      ],
      // Group settings
      groups: {
        "*": {
          requireMention: true,  // Must @mention in groups
          allowFrom: []          // Additional group allowlist
        }
      }
    }
  }
}
```

### Telegram Configuration

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

### Discord Configuration

```json5
{
  agent: {
    model: "anthropic/claude-opus-4-5"
  },
  channels: {
    discord: {
      token: "your-discord-bot-token",
      dm: {
        policy: "pairing",      // Require pairing for DMs
        allowFrom: []           // Allowlisted user IDs
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

### Multi-Channel Configuration

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

### Multi-Agent Configuration

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

### Session Configuration

```json5
{
  session: {
    // DM session scope
    dmScope: "main",  // or "per-peer", "per-channel-peer"

    // Session reset policy
    reset: {
      mode: "daily",
      atHour: 4,          // 4 AM local time
      idleMinutes: 120    // Also reset after 2 hours idle
    },

    // Per-type overrides
    resetByType: {
      dm: { mode: "idle", idleMinutes: 240 },
      group: { mode: "daily", atHour: 4 },
      thread: { mode: "idle", idleMinutes: 60 }
    },

    // Link identities across channels
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

### Security Configuration

```json5
{
  gateway: {
    port: 18789,
    bind: "loopback",           // Only local connections
    auth: {
      mode: "password",
      password: "your-secure-password",
      allowLocal: true,         // Auto-approve loopback
      allowTailscale: true      // Trust Tailscale identity
    }
  },
  agents: {
    defaults: {
      sandbox: {
        mode: "non-main",       // Sandbox non-main sessions
        workspaceAccess: "ro",  // Read-only workspace in sandbox
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

### Memory Configuration

```json5
{
  agents: {
    defaults: {
      memorySearch: {
        enabled: true,
        provider: "openai",
        model: "text-embedding-3-small",

        // Additional paths to index
        extraPaths: [
          "../team-docs",
          "/path/to/notes"
        ],

        // Hybrid search settings
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

      // Pre-compaction memory flush
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

### Cron Jobs Configuration

```json5
{
  cron: {
    jobs: [
      {
        id: "morning-briefing",
        schedule: "0 7 * * *",          // 7 AM daily
        message: "Give me my morning briefing",
        agent: "personal",
        isolated: true,                 // Fresh session
        channel: "telegram",
        target: "123456789"
      },
      {
        id: "weekly-backup",
        schedule: "0 2 * * 0",          // Sunday 2 AM
        message: "Run weekly backup routine",
        thinking: "high"
      },
      {
        id: "news-digest",
        schedule: "0 18 * * 1-5",       // Weekdays 6 PM
        message: "Summarize today's tech news",
        channel: "discord",
        target: "channel-id"
      }
    ]
  }
}
```

### Browser Configuration

```json5
{
  browser: {
    enabled: true,
    headless: false,           // Show browser window
    chromePath: null,          // Auto-detect
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

## CLI Usage Examples

### Gateway Management

```bash
# Start gateway in foreground
openclaw gateway run --port 18789 --verbose

# Start as background daemon
openclaw gateway run --daemon

# Stop gateway
openclaw gateway stop

# Check gateway status
openclaw gateway status

# Restart gateway
openclaw gateway restart
```

### Agent Interaction

```bash
# Simple message
openclaw agent --message "What's the weather like?"

# With thinking mode
openclaw agent --message "Analyze this code" --thinking high

# Specify model
openclaw agent --message "Hello" --model "openai/gpt-5.2"

# From file
openclaw agent --file ./prompt.txt

# Interactive mode
openclaw agent --interactive
```

### Message Sending

```bash
# Send to WhatsApp number
openclaw message send \
  --channel whatsapp \
  --to "+1234567890" \
  --message "Hello from OpenClaw"

# Send to Telegram
openclaw message send \
  --channel telegram \
  --to "username" \
  --message "Hello!"

# Send with attachment
openclaw message send \
  --channel whatsapp \
  --to "+1234567890" \
  --message "Check this out" \
  --file ./image.png
```

### Channel Management

```bash
# Check all channel status
openclaw channels status

# Detailed status with probes
openclaw channels status --deep

# Login to WhatsApp (QR code)
openclaw channels login whatsapp

# Logout from channel
openclaw channels logout telegram
```

### Session Management

```bash
# List all sessions
openclaw sessions list

# List active sessions (last 60 minutes)
openclaw sessions list --active 60

# JSON output
openclaw sessions list --json

# Reset specific session
openclaw sessions reset "agent:default:main"

# Clear all sessions
openclaw sessions clear --confirm
```

### Configuration Management

```bash
# Show current config
openclaw config get

# Get specific value
openclaw config get agent.model

# Set value
openclaw config set agent.model "openai/gpt-5.2"

# Set channel config
openclaw config set channels.telegram.botToken "123:ABC"

# Delete value
openclaw config delete channels.discord
```

### Diagnostics

```bash
# Run diagnostic checks
openclaw doctor

# Verbose diagnostics
openclaw doctor --verbose

# Check specific area
openclaw doctor --check channels
openclaw doctor --check auth
openclaw doctor --check workspace
```

## Workspace Examples

### AGENTS.md (Instructions)

```markdown
# Operating Instructions

## Identity
You are Molty, a helpful AI assistant.

## Core Rules
1. Be concise and direct
2. Ask clarifying questions when needed
3. Write code when it helps explain
4. Remember important information

## Memory
- Write daily notes to memory/YYYY-MM-DD.md
- Update MEMORY.md for long-term facts
- Read memory files when context might help

## Tools
- Use bash for system operations
- Use browser for web research
- Use canvas for visual output
```

### SOUL.md (Persona)

```markdown
# Soul

## Voice
- Friendly but professional
- Clear and concise
- Occasionally uses lobster puns

## Boundaries
- Never share API keys or passwords
- Don't execute destructive commands without confirmation
- Respect user privacy

## Preferences
- Prefer code examples over lengthy explanations
- Use markdown formatting
- Break complex tasks into steps
```

### TOOLS.md (Tool Guidance)

```markdown
# Tool Notes

## bash
- Always use absolute paths
- Confirm destructive operations
- Use timeout for long-running commands

## browser
- Prefer headless for background tasks
- Take screenshots for verification
- Close tabs when done

## memory_search
- Use for recalling past conversations
- Check memory before asking repeated questions
```

### Skill Example (SKILL.md)

```markdown
---
name: git-helper
description: Git operations helper
requireEnv: []
---

# Git Helper

When the user asks about git operations:

1. Check current branch: `git branch --show-current`
2. Check status: `git status --short`
3. For commits, use conventional commit format

## Common Operations

- **New branch**: `git checkout -b feature/<name>`
- **Commit**: `git commit -m "type: description"`
- **Push**: `git push -u origin <branch>`
```

## Gateway API Examples

### WebSocket Connection (TypeScript)

```typescript
import WebSocket from 'ws';

const ws = new WebSocket('ws://127.0.0.1:18789');

// Connection handler
ws.on('open', () => {
  // Send connect request
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

// Message handler
ws.on('message', (data) => {
  const msg = JSON.parse(data.toString());

  if (msg.type === 'res' && msg.id === 'connect-1') {
    console.log('Connected:', msg.payload);
  }

  if (msg.type === 'event') {
    console.log('Event:', msg.event, msg.payload);
  }
});
```

### Sending Agent Request

```typescript
// After successful connect...

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

// Handle agent events
ws.on('message', (data) => {
  const msg = JSON.parse(data.toString());

  if (msg.type === 'event' && msg.event === 'agent') {
    const { event, payload } = msg.payload;

    switch (event) {
      case 'thinking':
        console.log('Thinking:', payload.content);
        break;
      case 'tool_start':
        console.log('Tool:', payload.name);
        break;
      case 'text_delta':
        process.stdout.write(payload.delta);
        break;
      case 'done':
        console.log('\nComplete');
        break;
    }
  }
});
```

### Sending Messages via API

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

// Usage
sendMessage('telegram', 'username', 'Hello from API!');
```

### Listing Sessions

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

// Handle response
ws.on('message', (data) => {
  const msg = JSON.parse(data.toString());

  if (msg.type === 'res' && msg.ok) {
    console.log('Sessions:', msg.payload.sessions);
  }
});
```

## Extension Development

### Creating a Channel Extension

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
      // Initialize your channel
      console.log('Starting my-channel with config:', config);

      // Set up message handler
      context.onInbound((message) => {
        // Forward to OpenClaw
        context.deliver({
          from: message.sender,
          text: message.content,
          channel: 'my-channel'
        });
      });
    },

    async stop() {
      // Cleanup
      console.log('Stopping my-channel');
    },

    async send(message) {
      // Send outbound message
      console.log('Sending:', message);

      // Call your API
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

### Package Configuration

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

### Running Tests

```bash
# Run all tests
pnpm test

# Run with coverage
pnpm test:coverage

# Run specific test file
pnpm test src/gateway/session-utils.test.ts

# Run e2e tests
pnpm test:e2e

# Run live tests (requires API keys)
CLAWDBOT_LIVE_TEST=1 pnpm test:live
```

### Writing Tests

```typescript
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { createTestGateway } from '../test-helpers/gateway';

describe('Session Management', () => {
  let gateway: TestGateway;

  beforeAll(async () => {
    gateway = await createTestGateway();
  });

  afterAll(async () => {
    await gateway.close();
  });

  it('should create new session', async () => {
    const result = await gateway.call('sessions.list', {});
    expect(result.sessions).toBeInstanceOf(Array);
  });

  it('should reset session', async () => {
    const result = await gateway.call('sessions.reset', {
      sessionKey: 'agent:default:main'
    });
    expect(result.ok).toBe(true);
  });
});
```

## Docker Examples

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

# Install dependencies
COPY package.json pnpm-lock.yaml ./
RUN npm install -g pnpm && pnpm install --frozen-lockfile

# Copy source
COPY . .

# Build
RUN pnpm build

# Expose ports
EXPOSE 18789 18793

# Run gateway
CMD ["node", "dist/entry.js", "gateway", "run", "--bind", "0.0.0.0"]
```
