# OpenClaw Framework Research

**Repository**: https://github.com/openclaw/openclaw
**License**: MIT
**Last Updated**: 2026-01-31

## Overview

OpenClaw is a **personal AI assistant framework** designed to run locally on your own devices. Unlike cloud-based AI assistants, OpenClaw is a self-hosted solution that connects to messaging platforms you already use, providing a unified interface for AI-powered interactions.

The name is inspired by "Molty," a space lobster AI assistant mascot.

## Executive Summary

### What OpenClaw Is

OpenClaw is a **Gateway-centric personal AI assistant** that:

- Runs a single unified control plane (Gateway) on your local machine
- Connects to 12+ messaging channels (WhatsApp, Telegram, Slack, Discord, Signal, iMessage, Microsoft Teams, Google Chat, Matrix, WebChat, and more)
- Supports multiple LLM providers (Anthropic Claude, OpenAI GPT, Google Gemini, and others)
- Provides voice wake and talk mode capabilities via companion apps (macOS, iOS, Android)
- Features a Canvas system for visual agent-driven workspaces
- Uses WebSocket-based communication for all client interactions

### Key Design Philosophy

1. **Local-First**: The Gateway runs on your machine, not in the cloud
2. **Single Control Plane**: One Gateway manages all channels, sessions, and tools
3. **Multi-Channel Inbox**: Chat with your AI through any messaging platform
4. **Session Continuity**: Maintains context across conversations and platforms
5. **Extensible Architecture**: Plugin system for adding new channels and features

## Core Architecture

See [architecture.md](./architecture.md) for detailed architecture documentation.

### High-Level Architecture Diagram

```
WhatsApp / Telegram / Slack / Discord / Signal / iMessage / Teams / Google Chat / Matrix / WebChat
                              |
                              v
              +-------------------------------+
              |           Gateway             |
              |      (Control Plane)          |
              |   ws://127.0.0.1:18789        |
              +---------------+---------------+
                              |
          +-------------------+-------------------+
          |           |           |           |
          v           v           v           v
     Pi Agent     CLI Tools    WebChat     macOS/iOS/Android
     Runtime      (openclaw)    UI         Companion Apps
```

### Core Components

| Component | Purpose |
|-----------|---------|
| **Gateway** | Central WebSocket server managing all connections, sessions, and routing |
| **Agent Runtime** | Embedded p-mono-based LLM agent with tool execution |
| **Sessions** | Per-conversation state management with persistence |
| **Channels** | Messaging platform connectors (WhatsApp, Telegram, etc.) |
| **Tools** | Built-in and custom capabilities (bash, browser, canvas, etc.) |
| **Skills** | Reusable prompt templates and workflows |
| **Nodes** | Device-specific endpoints (camera, location, notifications) |

## Key Components

See [components.md](./components.md) for detailed component documentation.

### Gateway Server

The heart of OpenClaw is the Gateway server - a WebSocket-based control plane that:
- Manages all channel connections
- Routes messages to appropriate sessions
- Handles authentication and authorization
- Coordinates tool execution
- Manages session lifecycle

### Agent Runtime

Based on **p-mono**, the agent runtime provides:
- Tool streaming and block streaming
- Thinking mode support (multiple levels)
- Model failover and rotation
- Session context management
- Sub-agent coordination

### Session Model

OpenClaw's session model is sophisticated:
- **Main Session**: Primary 1:1 conversation with the assistant
- **Group Sessions**: Isolated per-group/channel contexts
- **Cron Sessions**: Scheduled task contexts
- **Webhook Sessions**: Event-triggered contexts

### Channel System

Extensive multi-channel support:
- **Core**: WhatsApp (Baileys), Telegram (grammY), Slack (Bolt), Discord (discord.js)
- **Extensions**: Signal, iMessage, Microsoft Teams, Google Chat, Matrix, Zalo, BlueBubbles

## Unique Features

### 1. Multi-Channel Personal Assistant

Unlike other frameworks focused on single-use cases, OpenClaw is designed as a **personal assistant that follows you** across all your messaging platforms.

### 2. Gateway-Based Architecture

The central Gateway pattern provides:
- Single source of truth for all state
- Unified protocol for all clients
- Consistent security model
- Hot-reload configuration support

### 3. Session Continuity

Smart session management that:
- Maintains context across devices and channels
- Supports per-peer, per-channel-peer, and main session modes
- Allows identity linking across platforms
- Implements intelligent session reset policies

### 4. Memory System

Plain Markdown-based memory with:
- Daily logs (`memory/YYYY-MM-DD.md`)
- Long-term memory (`MEMORY.md`)
- Vector search via embeddings (local or remote)
- Automatic pre-compaction memory flush

### 5. Agent-to-Agent Communication

Built-in tools for multi-agent coordination:
- `sessions_list` - Discover active sessions
- `sessions_send` - Message other sessions
- `sessions_history` - Access session transcripts

### 6. Canvas & A2UI

Visual agent-driven workspace:
- Agent can push/reset HTML content
- Supports evaluation and snapshots
- Available on macOS, iOS, and Android

### 7. Voice Wake & Talk Mode

Always-on speech capabilities:
- Wake word detection
- Continuous conversation
- ElevenLabs TTS integration

## Strengths and Weaknesses

### Strengths

1. **Comprehensive Channel Support**: No other framework supports as many messaging platforms out-of-the-box
2. **Local-First Design**: Privacy-focused with all data staying on your devices
3. **Production Ready**: Extensive testing, Docker support, and operational tooling
4. **Strong Security Model**: DM pairing, allowlists, sandboxing options
5. **Active Development**: Regular releases, active Discord community
6. **Well-Documented**: Comprehensive docs at docs.openclaw.ai
7. **Model Agnostic**: Supports multiple LLM providers with failover

### Weaknesses

1. **Complexity**: High barrier to entry due to extensive configuration options
2. **Single-User Focus**: Designed as a personal assistant, not multi-tenant
3. **Node.js Dependency**: Requires Node 22+ runtime
4. **Learning Curve**: Many concepts to understand (Gateway, sessions, channels, skills)
5. **Resource Requirements**: Running multiple channels simultaneously can be resource-intensive

### Ideal Use Cases

- Personal AI assistant across all messaging platforms
- Home automation integration
- Development workflow automation
- Cross-platform notification and messaging hub
- Voice-enabled AI assistant on macOS/iOS/Android

### Less Suitable For

- Multi-tenant SaaS applications
- Simple single-purpose chatbots
- Serverless deployments
- Applications requiring minimal setup

## Code Examples

See [examples.md](./examples.md) for detailed code examples.

### Quick Start

```bash
# Install globally
npm install -g openclaw@latest

# Run onboarding wizard
openclaw onboard --install-daemon

# Start the gateway
openclaw gateway --port 18789 --verbose

# Send a message
openclaw message send --to +1234567890 --message "Hello from OpenClaw"

# Talk to the assistant
openclaw agent --message "What can you do?" --thinking high
```

### Minimal Configuration

```json5
// ~/.openclaw/openclaw.json
{
  agent: {
    model: "anthropic/claude-opus-4-5"
  },
  channels: {
    whatsapp: {
      allowFrom: ["+1234567890"]
    }
  }
}
```

## Technology Stack

| Layer | Technology |
|-------|------------|
| Language | TypeScript (ESM) |
| Runtime | Node.js 22+ (Bun supported for dev) |
| Build | pnpm, tsc |
| Testing | Vitest |
| Linting | Oxlint, Oxfmt |
| Packaging | npm, Docker |
| Protocol | WebSocket + JSON |
| Schemas | TypeBox |
| Database | SQLite (sessions, memory) |

## Project Structure

```
openclaw/
├── src/                    # Core source code
│   ├── gateway/            # WebSocket server
│   ├── agents/             # Agent runtime
│   ├── sessions/           # Session management
│   ├── channels/           # Channel abstractions
│   ├── cli/                # CLI wiring
│   ├── commands/           # CLI commands
│   ├── browser/            # Browser automation
│   ├── canvas-host/        # Canvas/A2UI hosting
│   ├── memory/             # Memory system
│   └── ...
├── extensions/             # Channel plugins
│   ├── whatsapp/
│   ├── telegram/
│   ├── slack/
│   ├── discord/
│   └── ...
├── docs/                   # Documentation
├── apps/                   # Companion apps
│   ├── macos/
│   ├── ios/
│   └── android/
├── skills/                 # Bundled skills
├── ui/                     # Web UI
└── packages/               # Shared packages
```

## Comparison with Other Frameworks

| Feature | OpenClaw | LangChain | AutoGen | CrewAI |
|---------|----------|-----------|---------|--------|
| Primary Focus | Personal Assistant | LLM Apps | Multi-Agent | Multi-Agent |
| Channel Support | 12+ | None | None | None |
| Local-First | Yes | No | No | No |
| Voice Support | Yes | No | No | No |
| Session Management | Advanced | Basic | Basic | Basic |
| Memory System | Built-in Vector | Plugin | Plugin | Plugin |
| Mobile Apps | Yes | No | No | No |

## Key Takeaways

1. **Gateway Pattern**: Central WebSocket server as the control plane is a powerful architectural choice for multi-channel applications

2. **Session Model**: The sophisticated session system (main, per-peer, per-channel-peer) with identity linking is worth studying

3. **Memory Design**: Plain Markdown files with vector search overlay is a pragmatic, human-readable approach

4. **Extension System**: Plugin architecture via `extensions/` demonstrates clean separation of channel implementations

5. **Multi-Agent Tools**: Built-in session tools for agent-to-agent communication show how to coordinate multiple AI agents

## Further Reading

- [Official Documentation](https://docs.openclaw.ai)
- [GitHub Repository](https://github.com/openclaw/openclaw)
- [DeepWiki Analysis](https://deepwiki.com/openclaw/openclaw)
- [Discord Community](https://discord.gg/clawd)
- [ClawdHub (Skills Registry)](https://clawdhub.com)
