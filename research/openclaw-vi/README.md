# Nghiên Cứu Framework OpenClaw

**Repository**: https://github.com/openclaw/openclaw
**Giấy phép**: MIT
**Cập nhật lần cuối**: 2026-01-31

## Tổng Quan

OpenClaw là một **framework trợ lý AI cá nhân** được thiết kế để chạy cục bộ trên thiết bị của bạn. Không giống như các trợ lý AI dựa trên đám mây, OpenClaw là giải pháp tự lưu trữ kết nối với các nền tảng nhắn tin mà bạn đã sử dụng, cung cấp giao diện thống nhất cho các tương tác được hỗ trợ bởi AI.

Tên gọi được lấy cảm hứng từ "Molty," một linh vật trợ lý AI tôm hùm không gian.

## Tóm Tắt Điều Hành

### OpenClaw Là Gì

OpenClaw là một **trợ lý AI cá nhân tập trung vào Gateway** với các đặc điểm:

- Chạy một control plane thống nhất duy nhất (Gateway) trên máy cục bộ của bạn
- Kết nối với hơn 12 kênh nhắn tin (WhatsApp, Telegram, Slack, Discord, Signal, iMessage, Microsoft Teams, Google Chat, Matrix, WebChat, và nhiều hơn nữa)
- Hỗ trợ nhiều nhà cung cấp LLM (Anthropic Claude, OpenAI GPT, Google Gemini, và các nhà cung cấp khác)
- Cung cấp khả năng đánh thức bằng giọng nói và chế độ nói chuyện thông qua các ứng dụng đồng hành (macOS, iOS, Android)
- Tính năng hệ thống Canvas cho không gian làm việc trực quan điều khiển bởi agent
- Sử dụng giao tiếp dựa trên WebSocket cho tất cả các tương tác client

### Triết Lý Thiết Kế Chính

1. **Ưu tiên Cục bộ**: Gateway chạy trên máy của bạn, không phải trên đám mây
2. **Control Plane Duy nhất**: Một Gateway quản lý tất cả các kênh, phiên và công cụ
3. **Hộp thư Đa kênh**: Trò chuyện với AI của bạn thông qua bất kỳ nền tảng nhắn tin nào
4. **Tính liên tục của Phiên**: Duy trì ngữ cảnh qua các cuộc hội thoại và nền tảng
5. **Kiến trúc Mở rộng**: Hệ thống plugin để thêm các kênh và tính năng mới

## Kiến Trúc Cốt Lõi

Xem [architecture.md](./architecture.md) để có tài liệu kiến trúc chi tiết.

### Sơ Đồ Kiến Trúc Cấp Cao

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
     Runtime      (openclaw)    UI         Ứng dụng Đồng hành
```

### Các Thành Phần Cốt Lõi

| Thành phần | Mục đích |
|-----------|---------|
| **Gateway** | Server WebSocket trung tâm quản lý tất cả kết nối, phiên và định tuyến |
| **Agent Runtime** | Agent LLM dựa trên p-mono nhúng với thực thi công cụ |
| **Sessions** | Quản lý trạng thái theo từng cuộc hội thoại với tính bền vững |
| **Channels** | Các connector nền tảng nhắn tin (WhatsApp, Telegram, v.v.) |
| **Tools** | Khả năng tích hợp sẵn và tùy chỉnh (bash, browser, canvas, v.v.) |
| **Skills** | Các mẫu prompt và quy trình làm việc có thể tái sử dụng |
| **Nodes** | Các endpoint cụ thể theo thiết bị (camera, vị trí, thông báo) |

## Các Thành Phần Chính

Xem [components.md](./components.md) để có tài liệu thành phần chi tiết.

### Gateway Server

Trái tim của OpenClaw là Gateway server - một control plane dựa trên WebSocket:
- Quản lý tất cả kết nối kênh
- Định tuyến tin nhắn đến các phiên thích hợp
- Xử lý xác thực và ủy quyền
- Điều phối thực thi công cụ
- Quản lý vòng đời phiên

### Agent Runtime

Dựa trên **p-mono**, agent runtime cung cấp:
- Streaming công cụ và streaming khối
- Hỗ trợ chế độ thinking (nhiều cấp độ)
- Failover và xoay vòng model
- Quản lý ngữ cảnh phiên
- Điều phối sub-agent

### Mô Hình Session

Mô hình session của OpenClaw rất tinh vi:
- **Main Session**: Cuộc hội thoại 1:1 chính với trợ lý
- **Group Sessions**: Ngữ cảnh cô lập theo từng nhóm/kênh
- **Cron Sessions**: Ngữ cảnh tác vụ theo lịch
- **Webhook Sessions**: Ngữ cảnh kích hoạt theo sự kiện

### Hệ Thống Channel

Hỗ trợ đa kênh mở rộng:
- **Core**: WhatsApp (Baileys), Telegram (grammY), Slack (Bolt), Discord (discord.js)
- **Extensions**: Signal, iMessage, Microsoft Teams, Google Chat, Matrix, Zalo, BlueBubbles

## Tính Năng Độc Đáo

### 1. Trợ Lý Cá Nhân Đa Kênh

Không giống như các framework khác tập trung vào các use case đơn lẻ, OpenClaw được thiết kế như một **trợ lý cá nhân theo bạn** qua tất cả các nền tảng nhắn tin.

### 2. Kiến Trúc Dựa Trên Gateway

Mẫu Gateway trung tâm cung cấp:
- Nguồn sự thật duy nhất cho tất cả trạng thái
- Giao thức thống nhất cho tất cả client
- Mô hình bảo mật nhất quán
- Hỗ trợ cấu hình hot-reload

### 3. Tính Liên Tục Của Phiên

Quản lý phiên thông minh:
- Duy trì ngữ cảnh qua các thiết bị và kênh
- Hỗ trợ các chế độ per-peer, per-channel-peer, và main session
- Cho phép liên kết danh tính qua các nền tảng
- Triển khai các chính sách reset phiên thông minh

### 4. Hệ Thống Bộ Nhớ

Bộ nhớ dựa trên Markdown thuần túy với:
- Nhật ký hàng ngày (`memory/YYYY-MM-DD.md`)
- Bộ nhớ dài hạn (`MEMORY.md`)
- Tìm kiếm vector qua embeddings (cục bộ hoặc từ xa)
- Tự động flush bộ nhớ trước khi nén

### 5. Giao Tiếp Agent-to-Agent

Công cụ tích hợp sẵn cho điều phối đa agent:
- `sessions_list` - Khám phá các phiên đang hoạt động
- `sessions_send` - Gửi tin nhắn đến các phiên khác
- `sessions_history` - Truy cập bản ghi phiên

### 6. Canvas & A2UI

Không gian làm việc trực quan điều khiển bởi agent:
- Agent có thể push/reset nội dung HTML
- Hỗ trợ đánh giá và chụp nhanh
- Có sẵn trên macOS, iOS, và Android

### 7. Chế Độ Đánh Thức & Nói Chuyện Bằng Giọng Nói

Khả năng giọng nói luôn bật:
- Phát hiện từ đánh thức
- Hội thoại liên tục
- Tích hợp ElevenLabs TTS

## Điểm Mạnh và Điểm Yếu

### Điểm Mạnh

1. **Hỗ Trợ Kênh Toàn Diện**: Không có framework nào khác hỗ trợ nhiều nền tảng nhắn tin sẵn có như vậy
2. **Thiết Kế Ưu Tiên Cục Bộ**: Tập trung vào quyền riêng tư với tất cả dữ liệu ở lại trên thiết bị của bạn
3. **Sẵn Sàng Sản Xuất**: Kiểm thử mở rộng, hỗ trợ Docker, và công cụ vận hành
4. **Mô Hình Bảo Mật Mạnh**: Ghép nối DM, danh sách cho phép, tùy chọn sandboxing
5. **Phát Triển Tích Cực**: Phát hành thường xuyên, cộng đồng Discord hoạt động
6. **Tài Liệu Đầy Đủ**: Tài liệu toàn diện tại docs.openclaw.ai
7. **Bất Khả Tri Model**: Hỗ trợ nhiều nhà cung cấp LLM với failover

### Điểm Yếu

1. **Phức Tạp**: Rào cản gia nhập cao do các tùy chọn cấu hình mở rộng
2. **Tập Trung Người Dùng Đơn**: Được thiết kế như trợ lý cá nhân, không phải multi-tenant
3. **Phụ Thuộc Node.js**: Yêu cầu runtime Node 22+
4. **Đường Cong Học Tập**: Nhiều khái niệm cần hiểu (Gateway, sessions, channels, skills)
5. **Yêu Cầu Tài Nguyên**: Chạy nhiều kênh đồng thời có thể tốn tài nguyên

### Trường Hợp Sử Dụng Lý Tưởng

- Trợ lý AI cá nhân qua tất cả các nền tảng nhắn tin
- Tích hợp tự động hóa nhà thông minh
- Tự động hóa quy trình làm việc phát triển
- Hub thông báo và nhắn tin đa nền tảng
- Trợ lý AI hỗ trợ giọng nói trên macOS/iOS/Android

### Ít Phù Hợp Cho

- Ứng dụng SaaS multi-tenant
- Chatbot đơn mục đích đơn giản
- Triển khai serverless
- Ứng dụng yêu cầu thiết lập tối thiểu

## Ví Dụ Code

Xem [examples.md](./examples.md) để có ví dụ code chi tiết.

### Bắt Đầu Nhanh

```bash
# Cài đặt toàn cục
npm install -g openclaw@latest

# Chạy wizard onboarding
openclaw onboard --install-daemon

# Khởi động gateway
openclaw gateway --port 18789 --verbose

# Gửi tin nhắn
openclaw message send --to +1234567890 --message "Xin chào từ OpenClaw"

# Nói chuyện với trợ lý
openclaw agent --message "Bạn có thể làm gì?" --thinking high
```

### Cấu Hình Tối Thiểu

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

## Ngăn Xếp Công Nghệ

| Lớp | Công nghệ |
|-------|------------|
| Ngôn ngữ | TypeScript (ESM) |
| Runtime | Node.js 22+ (Bun hỗ trợ cho dev) |
| Build | pnpm, tsc |
| Testing | Vitest |
| Linting | Oxlint, Oxfmt |
| Đóng gói | npm, Docker |
| Giao thức | WebSocket + JSON |
| Schemas | TypeBox |
| Cơ sở dữ liệu | SQLite (sessions, memory) |

## Cấu Trúc Dự Án

```
openclaw/
├── src/                    # Code nguồn cốt lõi
│   ├── gateway/            # WebSocket server
│   ├── agents/             # Agent runtime
│   ├── sessions/           # Quản lý phiên
│   ├── channels/           # Trừu tượng hóa kênh
│   ├── cli/                # Đấu nối CLI
│   ├── commands/           # Các lệnh CLI
│   ├── browser/            # Tự động hóa trình duyệt
│   ├── canvas-host/        # Hosting Canvas/A2UI
│   ├── memory/             # Hệ thống bộ nhớ
│   └── ...
├── extensions/             # Plugin kênh
│   ├── whatsapp/
│   ├── telegram/
│   ├── slack/
│   ├── discord/
│   └── ...
├── docs/                   # Tài liệu
├── apps/                   # Ứng dụng đồng hành
│   ├── macos/
│   ├── ios/
│   └── android/
├── skills/                 # Skills đi kèm
├── ui/                     # Web UI
└── packages/               # Packages chia sẻ
```

## So Sánh Với Các Framework Khác

| Tính năng | OpenClaw | LangChain | AutoGen | CrewAI |
|---------|----------|-----------|---------|--------|
| Trọng tâm Chính | Trợ lý Cá nhân | Ứng dụng LLM | Đa Agent | Đa Agent |
| Hỗ trợ Kênh | 12+ | Không | Không | Không |
| Ưu tiên Cục bộ | Có | Không | Không | Không |
| Hỗ trợ Giọng nói | Có | Không | Không | Không |
| Quản lý Phiên | Nâng cao | Cơ bản | Cơ bản | Cơ bản |
| Hệ thống Bộ nhớ | Vector Tích hợp | Plugin | Plugin | Plugin |
| Ứng dụng Di động | Có | Không | Không | Không |

## Điểm Chính Rút Ra

1. **Mẫu Gateway**: Server WebSocket trung tâm làm control plane là lựa chọn kiến trúc mạnh mẽ cho các ứng dụng đa kênh

2. **Mô Hình Session**: Hệ thống session tinh vi (main, per-peer, per-channel-peer) với liên kết danh tính đáng nghiên cứu

3. **Thiết Kế Bộ Nhớ**: File Markdown thuần túy với lớp phủ tìm kiếm vector là cách tiếp cận thực dụng, dễ đọc cho con người

4. **Hệ Thống Extension**: Kiến trúc plugin qua `extensions/` thể hiện sự tách biệt rõ ràng các triển khai kênh

5. **Công Cụ Đa Agent**: Công cụ session tích hợp cho giao tiếp agent-to-agent cho thấy cách điều phối nhiều AI agent

## Đọc Thêm

- [Tài liệu Chính thức](https://docs.openclaw.ai)
- [GitHub Repository](https://github.com/openclaw/openclaw)
- [Phân tích DeepWiki](https://deepwiki.com/openclaw/openclaw)
- [Cộng đồng Discord](https://discord.gg/clawd)
- [ClawdHub (Registry Skills)](https://clawdhub.com)
