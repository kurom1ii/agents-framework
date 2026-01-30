# OpenClaw Architecture

This document provides a detailed analysis of OpenClaw's architecture.

## Gateway Architecture

The Gateway is the central component of OpenClaw - a WebSocket server that acts as the unified control plane for all operations.

### Gateway Design Principles

1. **Single Point of Control**: One Gateway per host manages all messaging surfaces
2. **WebSocket-First**: All communication uses WebSocket with JSON payloads
3. **Event-Driven**: Server pushes events for real-time updates
4. **Typed Protocol**: TypeBox schemas for type-safe communication

### Gateway Components

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

### Wire Protocol

All communication follows a request/response pattern with server-push events:

```typescript
// Request
{
  type: "req",
  id: string,        // Unique request ID
  method: string,    // Method name
  params: object     // Method parameters
}

// Response
{
  type: "res",
  id: string,        // Matches request ID
  ok: boolean,       // Success flag
  payload?: object,  // Success data
  error?: object     // Error details
}

// Event (server push)
{
  type: "event",
  event: string,     // Event name
  payload: object,   // Event data
  seq?: number,      // Sequence number
  stateVersion?: number
}
```

### Connection Lifecycle

```
Client                           Gateway
  |                                 |
  |---- req:connect --------------->|
  |<------ res:hello-ok ------------|  (includes snapshot)
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

### Authentication Model

OpenClaw supports multiple authentication modes:

1. **Token Auth**: Gateway token in `connect.params.auth.token`
2. **Device Pairing**: Device identity with pairing codes
3. **Tailscale Identity**: Auto-trust based on Tailscale headers
4. **Password Auth**: Shared password for public exposure

```json5
{
  gateway: {
    auth: {
      mode: "password",          // or "token"
      password: "secret",
      allowTailscale: true,      // Trust Tailscale identity
      allowLocal: true           // Auto-approve loopback
    }
  }
}
```

## Agent Runtime Architecture

### p-mono Integration

OpenClaw uses a modified version of the p-mono agent runtime:

```
+------------------------+
|    Agent Runtime       |
+------------------------+
|  - Model Provider      |  (Anthropic, OpenAI, Google, etc.)
|  - Tool Registry       |  (Built-in + Skills)
|  - Context Manager     |  (System prompt assembly)
|  - Stream Handler      |  (Tool + block streaming)
+------------------------+
```

### Tool Execution Flow

```
User Message
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
Channel Delivery
```

### Context Assembly

The system prompt is assembled from multiple sources:

```
+----------------------------+
|     System Prompt          |
+----------------------------+
| 1. Base prompt (templates) |
| 2. AGENTS.md (workspace)   |
| 3. SOUL.md (persona)       |
| 4. TOOLS.md (guidance)     |
| 5. IDENTITY.md (name)      |
| 6. USER.md (user profile)  |
| 7. Skills (SKILL.md files) |
| 8. Runtime context         |
+----------------------------+
```

## Session Architecture

### Session Key Resolution

```
Direct Message (DM)
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

Group Chat
  -> agent:<agentId>:<channel>:group:<groupId>

Thread/Topic
  -> agent:<agentId>:<channel>:group:<groupId>:topic:<threadId>
```

### Session Lifecycle

```
+----------------+     +-----------------+     +----------------+
|   New Message  | --> | Session Lookup  | --> | Exists?        |
+----------------+     +-----------------+     +----------------+
                                                    |
                       +----------------+      +----+----+
                       | Create Session | <--- |   No    |
                       +----------------+      +---------+
                                |
                       +--------v--------+     +---------+
                       |  Check Expiry   | <---|   Yes   |
                       +-----------------+     +---------+
                                |
                 +--------------+---------------+
                 |                              |
            +----v----+                    +----v----+
            | Expired |                    |  Valid  |
            +---------+                    +---------+
                 |                              |
            +----v--------+                +----v----+
            | Reset       |                | Reuse   |
            | (new ID)    |                | Session |
            +-------------+                +---------+
```

### Session Storage

```
~/.openclaw/agents/<agentId>/sessions/
├── sessions.json          # Session store (key -> metadata)
├── <sessionId>.jsonl      # Transcript (JSONL format)
└── <sessionId>-topic-<threadId>.jsonl  # Thread transcripts
```

## Channel Architecture

### Channel Interface

All channels implement a common interface:

```typescript
interface Channel {
  // Lifecycle
  start(): Promise<void>;
  stop(): Promise<void>;

  // Status
  isConnected(): boolean;
  getStatus(): ChannelStatus;

  // Messaging
  send(message: OutboundMessage): Promise<void>;

  // Events
  onMessage(handler: MessageHandler): void;
  onTyping(handler: TypingHandler): void;
}
```

### Channel Routing

```
Inbound Message
      |
      v
+------------------+
| Channel Adapter  |  (WhatsApp, Telegram, etc.)
+------------------+
      |
      v
+------------------+
| Routing Rules    |
+------------------+
      |
      +-- Check allowFrom list
      +-- Check group allowlist
      +-- Apply DM policy (pairing/open)
      +-- Resolve agent mapping
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

### Multi-Account Support

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

## Extension Architecture

### Plugin Structure

```
extensions/<channel-name>/
├── package.json         # Dependencies
├── src/
│   ├── index.ts        # Main entry point
│   ├── channel.ts      # Channel implementation
│   └── ...
└── README.md
```

### Plugin SDK

Extensions use the plugin SDK:

```typescript
import { definePlugin, ChannelPlugin } from 'openclaw/plugin-sdk';

export default definePlugin({
  name: 'my-channel',
  version: '1.0.0',

  channels: [{
    id: 'my-channel',
    name: 'My Channel',

    async start(config) {
      // Initialize channel
    },

    async stop() {
      // Cleanup
    },

    async send(message) {
      // Send outbound message
    }
  }]
});
```

## Memory Architecture

### Memory Layers

```
+------------------------+
|    Memory System       |
+------------------------+
        |
        +-- Daily Logs (memory/YYYY-MM-DD.md)
        |     - Append-only
        |     - Read today + yesterday
        |
        +-- Long-term (MEMORY.md)
        |     - Curated facts
        |     - Main session only
        |
        +-- Vector Index (SQLite)
              - Semantic search
              - Hybrid BM25 + vector
```

### Memory Search Pipeline

```
Query
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
| Score Fusion      |  (weighted blend)
+-------------------+
  |
  v
Results (snippets + metadata)
```

## Node Architecture

Nodes are device endpoints that provide local capabilities:

### Node Registration

```
Node (macOS/iOS/Android)
        |
        |---- WebSocket connect (role: "node")
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
        +-- node.list (discover nodes)
        +-- node.describe (get capabilities)
        +-- node.invoke (execute command)
```

### Node Capabilities

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

## Security Architecture

### Security Layers

```
+-----------------------+
|   DM Policy           |  (pairing/open)
+-----------------------+
          |
          v
+-----------------------+
|   Allowlists          |  (per-channel)
+-----------------------+
          |
          v
+-----------------------+
|   Tool Sandboxing     |  (per-session)
+-----------------------+
          |
          v
+-----------------------+
|   Exec Approvals      |  (per-command)
+-----------------------+
```

### Sandbox Modes

```json5
{
  agents: {
    defaults: {
      sandbox: {
        mode: "non-main",        // Sandbox non-main sessions
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

## Deployment Architecture

### Local Deployment

```
+-------------------+
|   Your Machine    |
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

### Remote Deployment

```
+------------------+      +-------------------+
|   Your Mac       |      |   Linux VPS       |
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

### Tailscale Integration

```json5
{
  gateway: {
    tailscale: {
      mode: "serve",     // or "funnel"
      resetOnExit: true
    },
    bind: "loopback"     // Required with Tailscale
  }
}
```

## Performance Considerations

### Bottlenecks

1. **LLM Latency**: Primary bottleneck is model API response time
2. **Channel Connections**: Each channel maintains persistent connections
3. **Memory Indexing**: Large memory files can slow vector indexing

### Optimizations

1. **Block Streaming**: Send responses as they complete
2. **Session Pruning**: Trim old tool results to reduce context
3. **Embedding Cache**: Avoid re-embedding unchanged content
4. **Batch Embeddings**: Use OpenAI Batch API for large indexing jobs

## Scalability Considerations

OpenClaw is designed for **single-user personal assistant** use cases:

- **Horizontal Scaling**: Not designed for it (single Gateway per host)
- **Vertical Scaling**: Handles multiple channels on a single Gateway
- **Multi-Agent**: Multiple agent identities on one Gateway (not multi-tenant)

For multi-user scenarios, run separate OpenClaw instances per user.
