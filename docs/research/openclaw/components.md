# OpenClaw Components

This document provides detailed documentation of OpenClaw's key components.

## Gateway Server

The Gateway is the central WebSocket server that acts as the control plane for all OpenClaw operations.

### Key Files

```
src/gateway/
├── server.impl.ts          # Main server implementation
├── server.ts               # Server exports
├── client.ts               # Client connection handling
├── auth.ts                 # Authentication logic
├── session-utils.ts        # Session utilities
├── server-channels.ts      # Channel management
├── server-chat.ts          # Chat handling
├── server-methods.ts       # RPC method registry
├── protocol/               # Protocol definitions
└── server-methods/         # Individual method implementations
```

### Gateway Methods

| Method | Description |
|--------|-------------|
| `connect` | Initial handshake with auth |
| `health` | System health check |
| `status` | Gateway status |
| `agent` | Start agent run |
| `send` | Send message to channel |
| `sessions.list` | List active sessions |
| `sessions.patch` | Update session settings |
| `sessions.reset` | Reset a session |
| `config.get` | Get configuration |
| `config.patch` | Update configuration |
| `cron.list` | List cron jobs |
| `cron.trigger` | Manually trigger cron job |

### Events

| Event | Description |
|-------|-------------|
| `agent` | Agent activity (thinking, tool use, response) |
| `presence` | System presence changes |
| `tick` | Periodic heartbeat |
| `health` | Health status updates |
| `chat` | Chat message events |
| `shutdown` | Gateway shutting down |

## Agent Runtime

The agent runtime handles LLM interactions and tool execution.

### Key Files

```
src/agents/
├── agent-paths.ts          # Workspace path utilities
├── agent-scope.ts          # Agent scoping
├── agent-config.ts         # Agent configuration
├── tools/                  # Tool definitions
│   ├── bash.ts
│   ├── browser.ts
│   ├── canvas.ts
│   ├── read.ts
│   ├── write.ts
│   ├── edit.ts
│   └── ...
└── skills/                 # Skill loader
```

### Built-in Tools

#### File System Tools

| Tool | Description |
|------|-------------|
| `read` | Read file contents |
| `write` | Write file to disk |
| `edit` | Edit file with patches |
| `glob` | Find files by pattern |
| `grep` | Search file contents |

#### Execution Tools

| Tool | Description |
|------|-------------|
| `bash` | Execute shell commands |
| `process` | Process management |
| `apply_patch` | Apply unified diff patches |

#### Browser Tools

| Tool | Description |
|------|-------------|
| `browser_navigate` | Navigate to URL |
| `browser_action` | Perform browser action |
| `browser_snapshot` | Take page snapshot |
| `browser_screenshot` | Take screenshot |
| `browser_upload` | Upload file |

#### Canvas Tools

| Tool | Description |
|------|-------------|
| `canvas_push` | Push HTML content |
| `canvas_reset` | Reset canvas |
| `canvas_eval` | Evaluate JS in canvas |
| `canvas_snapshot` | Snapshot canvas state |

#### Communication Tools

| Tool | Description |
|------|-------------|
| `send_message` | Send via messaging channel |
| `sessions_list` | List active sessions |
| `sessions_send` | Send to another session |
| `sessions_history` | Get session history |
| `sessions_spawn` | Spawn sub-agent |

#### Utility Tools

| Tool | Description |
|------|-------------|
| `think` | Extended thinking mode |
| `memory_search` | Search memory files |
| `memory_get` | Read memory file |
| `web_search` | Web search |
| `web_fetch` | Fetch web content |

### Tool Configuration

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
      chromePath: "/path/to/chrome",
      headless: true
    },
    canvas: {
      enabled: true,
      port: 18793
    }
  }
}
```

## Session Management

Sessions track conversation state across messages.

### Key Files

```
src/sessions/
├── level-overrides.ts      # Thinking level overrides
├── model-overrides.ts      # Model selection overrides
├── send-policy.ts          # Message delivery policy
├── session-key-utils.ts    # Session key utilities
├── session-label.ts        # Session labeling
└── transcript-events.ts    # Transcript event handling

src/gateway/
├── session-utils.ts        # Core session utilities
├── session-utils.fs.ts     # Session file operations
├── sessions-patch.ts       # Session patching
└── sessions-resolve.ts     # Session resolution
```

### Session Data Structure

```typescript
interface Session {
  sessionId: string;           // Unique ID
  sessionKey: string;          // Lookup key
  agentId: string;             // Associated agent
  updatedAt: string;           // Last update timestamp
  inputTokens: number;         // Token usage
  outputTokens: number;
  totalTokens: number;
  contextTokens: number;       // Current context size
  thinkingLevel?: string;      // Override
  verboseLevel?: boolean;      // Override
  model?: string;              // Override
  sendPolicy?: string;         // Delivery policy
  groupActivation?: string;    // Group trigger mode
  displayName?: string;        // UI label
  channel?: string;            // Source channel
  origin?: SessionOrigin;      // Source metadata
}
```

### Session Reset Policies

```json5
{
  session: {
    reset: {
      mode: "daily",           // daily, idle, or combined
      atHour: 4,               // Reset hour (local time)
      idleMinutes: 120         // Idle timeout
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

## Channel System

Channels are messaging platform adapters.

### Core Channels

| Channel | Library | Location |
|---------|---------|----------|
| WhatsApp | Baileys | `extensions/whatsapp/` |
| Telegram | grammY | `extensions/telegram/` |
| Slack | Bolt | `extensions/slack/` |
| Discord | discord.js | `extensions/discord/` |
| Signal | signal-cli | `extensions/signal/` |
| iMessage | imsg | `extensions/imessage/` |

### Extension Channels

| Channel | Description |
|---------|-------------|
| Microsoft Teams | Bot Framework integration |
| Google Chat | Chat API |
| Matrix | Matrix protocol |
| BlueBubbles | Alternative iMessage |
| Zalo | Vietnamese messaging |
| Mattermost | Self-hosted chat |
| Line | LINE messaging |
| Nostr | Decentralized social |
| Twitch | Stream chat |

### Channel Configuration

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

## Memory System

The memory system provides persistent knowledge storage.

### Key Files

```
src/memory/
├── memory-index.ts         # Memory indexing
├── memory-search.ts        # Search implementation
├── embeddings/             # Embedding providers
│   ├── openai.ts
│   ├── gemini.ts
│   └── local.ts
└── store/                  # Storage backends
    └── sqlite.ts

extensions/memory-core/     # Core memory plugin
extensions/memory-lancedb/  # LanceDB plugin
```

### Memory Configuration

```json5
{
  agents: {
    defaults: {
      memorySearch: {
        enabled: true,
        provider: "openai",           // openai, gemini, local
        model: "text-embedding-3-small",
        fallback: "openai",

        // Additional paths to index
        extraPaths: ["../team-docs"],

        // Vector store
        store: {
          path: "~/.openclaw/memory/{agentId}.sqlite",
          vector: {
            enabled: true             // Use sqlite-vec
          }
        },

        // Hybrid search
        query: {
          hybrid: {
            enabled: true,
            vectorWeight: 0.7,
            textWeight: 0.3
          }
        },

        // Cache settings
        cache: {
          enabled: true,
          maxEntries: 50000
        },

        // Experimental: session memory
        experimental: {
          sessionMemory: false
        }
      }
    }
  }
}
```

### Memory Tools

```typescript
// memory_search - Semantic search over memory files
{
  query: string;           // Search query
  maxResults?: number;     // Max results (default: 5)
}

// Response
{
  results: Array<{
    text: string;          // Snippet (700 chars max)
    path: string;          // File path
    lineStart: number;     // Starting line
    lineEnd: number;       // Ending line
    score: number;         // Relevance score
    provider: string;      // Embedding provider
    model: string;         // Embedding model
    fallback: boolean;     // Used fallback?
  }>;
}

// memory_get - Read memory file content
{
  path: string;            // Workspace-relative path
  startLine?: number;      // Start line
  numLines?: number;       // Number of lines
}
```

## Skills System

Skills are reusable prompt templates and workflows.

### Key Files

```
skills/                     # Bundled skills
├── sag/                    # Search and answer
│   └── SKILL.md
├── git/                    # Git operations
│   └── SKILL.md
├── imsg/                   # iMessage
│   └── SKILL.md
└── ...

src/agents/skills/          # Skill loader
```

### Skill Locations

1. **Bundled**: Shipped with OpenClaw (`skills/`)
2. **Managed**: User-installed (`~/.openclaw/skills/`)
3. **Workspace**: Project-specific (`<workspace>/skills/`)

### Skill Structure

```
skills/<skill-name>/
├── SKILL.md            # Main skill definition
├── README.md           # Optional documentation
└── assets/             # Optional assets
```

### SKILL.md Format

```markdown
---
name: my-skill
description: Does something useful
requireEnv:
  - MY_API_KEY
requireConfig:
  - mySkill.enabled
---

# My Skill

Instructions for the agent on how to use this skill...

## Usage

When the user asks about X, do Y...
```

### Skill Configuration

```json5
{
  skills: {
    // Enable/disable specific skills
    enabled: ["sag", "git", "imsg"],
    disabled: [],

    // Skill-specific config
    sag: {
      searchProvider: "brave"
    }
  }
}
```

## Configuration System

OpenClaw uses a layered configuration system.

### Key Files

```
src/config/
├── config.ts               # Configuration loading
├── config-schema.ts        # Schema definitions
├── config-defaults.ts      # Default values
└── config-validation.ts    # Validation
```

### Configuration Locations

1. **File**: `~/.openclaw/openclaw.json` (or JSON5)
2. **Environment**: `OPENCLAW_*` variables
3. **CLI**: Command-line arguments
4. **Runtime**: Gateway API patches

### Configuration Structure

```json5
{
  // Agent settings
  agent: {
    model: "anthropic/claude-opus-4-5",
    skipBootstrap: false
  },

  // Multi-agent setup
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
    routing: { /* channel -> agent mapping */ }
  },

  // Gateway settings
  gateway: {
    port: 18789,
    bind: "loopback",
    auth: { mode: "password", password: "secret" }
  },

  // Session settings
  session: {
    dmScope: "main",
    reset: { mode: "daily", atHour: 4 }
  },

  // Channel settings
  channels: {
    whatsapp: { /* ... */ },
    telegram: { /* ... */ },
    discord: { /* ... */ }
  },

  // Tool settings
  tools: {
    exec: { enabled: true },
    browser: { enabled: true },
    canvas: { enabled: true }
  },

  // Model provider settings
  models: {
    providers: {
      anthropic: { apiKey: "sk-..." },
      openai: { apiKey: "sk-..." }
    }
  },

  // Plugin settings
  plugins: {
    slots: {
      memory: "memory-core"
    }
  }
}
```

## CLI System

The CLI provides command-line access to OpenClaw functionality.

### Key Files

```
src/cli/
├── index.ts                # CLI entry
├── commands/               # Command implementations
├── progress.ts             # Progress indicators
└── prompts.ts              # Interactive prompts

src/commands/
├── gateway.ts              # Gateway commands
├── agent.ts                # Agent commands
├── send.ts                 # Send commands
├── channels.ts             # Channel commands
├── sessions.ts             # Session commands
├── config.ts               # Config commands
├── doctor.ts               # Diagnostic commands
└── ...
```

### Key Commands

```bash
# Gateway management
openclaw gateway run          # Start gateway
openclaw gateway stop         # Stop gateway
openclaw gateway status       # Check status

# Agent interaction
openclaw agent --message "Hello"
openclaw agent --thinking high

# Messaging
openclaw message send --to +1234567890 --message "Hello"

# Channel management
openclaw channels status      # Channel status
openclaw channels login       # Setup channel auth

# Session management
openclaw sessions list        # List sessions
openclaw sessions reset <key> # Reset session

# Configuration
openclaw config get           # Show config
openclaw config set <path> <value>

# Diagnostics
openclaw doctor               # Run diagnostics
openclaw status               # System status

# Setup
openclaw onboard              # Onboarding wizard
openclaw setup                # Initialize workspace
```

## Browser Control

OpenClaw includes browser automation capabilities.

### Key Files

```
src/browser/
├── browser.ts              # Main browser controller
├── browser-pool.ts         # Browser instance pool
├── page-utils.ts           # Page utilities
├── snapshot.ts             # Page snapshotting
└── actions.ts              # Browser actions
```

### Browser Configuration

```json5
{
  browser: {
    enabled: true,
    chromePath: "/path/to/chrome",  // Auto-detect if not set
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

## Canvas System

The Canvas provides an agent-driven visual workspace.

### Key Files

```
src/canvas-host/
├── canvas-server.ts        # Canvas HTTP server
├── a2ui/                   # A2UI bundle
└── templates/              # HTML templates
```

### Canvas Configuration

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

A2UI is a protocol for agents to generate interactive UIs:

```typescript
// Agent pushes content
canvas_push({
  html: "<div>Hello World</div>",
  css: "div { color: blue; }",
  js: "console.log('loaded')"
});

// Agent evaluates JS
canvas_eval({
  expression: "document.body.innerHTML"
});

// Agent takes snapshot
canvas_snapshot({
  format: "png",
  fullPage: true
});
```

## Node System

Nodes are device endpoints for local capabilities.

### Key Files

```
src/node-host/
├── node-server.ts          # Node server
├── node-registry.ts        # Node registration
└── node-commands.ts        # Command handling

// Mobile apps
apps/ios/                   # iOS app
apps/android/               # Android app
apps/macos/                 # macOS app
```

### Node Commands

| Command | Platform | Description |
|---------|----------|-------------|
| `canvas.push` | All | Push HTML to canvas |
| `canvas.reset` | All | Reset canvas |
| `camera.snap` | iOS/Android/macOS | Take photo |
| `camera.clip` | iOS/Android/macOS | Record video |
| `screen.record` | iOS/Android/macOS | Screen recording |
| `location.get` | iOS/Android | Get location |
| `system.notify` | macOS | Send notification |
| `system.run` | macOS | Run shell command |

### Node Configuration

```json5
{
  nodes: {
    autoApprove: false,      // Auto-approve local nodes
    policy: {
      camera: "prompt",      // always, prompt, deny
      location: "prompt",
      screen: "prompt"
    }
  }
}
```

## Cron System

Cron jobs enable scheduled agent runs.

### Key Files

```
src/cron/
├── cron-manager.ts         # Cron job management
├── cron-parser.ts          # Schedule parsing
└── cron-runner.ts          # Job execution
```

### Cron Configuration

```json5
{
  cron: {
    jobs: [
      {
        id: "morning-briefing",
        schedule: "0 7 * * *",        // 7 AM daily
        message: "Give me my morning briefing",
        agent: "personal",
        isolated: true,               // Fresh session each run
        channel: "telegram",          // Delivery channel
        target: "123456789"           // Delivery target
      },
      {
        id: "weekly-report",
        schedule: "0 18 * * 5",       // Friday 6 PM
        message: "Generate weekly summary",
        thinking: "high"
      }
    ]
  }
}
```

## Hooks System

Hooks allow custom processing at various points.

### Key Files

```
src/hooks/
├── hooks.ts                # Hook execution
├── hook-types.ts           # Hook type definitions
└── hook-registry.ts        # Hook registration
```

### Available Hooks

| Hook | Trigger Point |
|------|---------------|
| `inbound.before` | Before processing inbound message |
| `inbound.after` | After processing inbound message |
| `outbound.before` | Before sending response |
| `outbound.after` | After sending response |
| `agent.before` | Before agent run |
| `agent.after` | After agent run |
| `tool.before` | Before tool execution |
| `tool.after` | After tool execution |

### Hook Configuration

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
