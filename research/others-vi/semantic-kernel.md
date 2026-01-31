# Microsoft Semantic Kernel

## Tổng Quan

**Semantic Kernel** là SDK sẵn sàng cho doanh nghiệp của Microsoft để xây dựng các AI agent thông minh và hệ thống multi-agent. Đây là framework không phụ thuộc model, trao quyền cho developer xây dựng, điều phối và triển khai các AI agent với độ tin cậy và tính linh hoạt cấp doanh nghiệp.

- **Repository**: https://github.com/microsoft/semantic-kernel
- **Stars**: ~27,000
- **Ngôn ngữ**: C#, Python, Java
- **License**: MIT
- **Tài liệu**: https://learn.microsoft.com/en-us/semantic-kernel/

## Triết Lý Cốt Lõi

Semantic Kernel được thiết kế như một "Dependency Injection container cho AI" - **Kernel** là thành phần trung tâm quản lý tất cả services và plugins cần thiết để chạy ứng dụng AI. Nó tuân theo các pattern doanh nghiệp quen thuộc với developer .NET trong khi cung cấp trải nghiệm nhất quán trên Python và Java.

## Các Khái Niệm Chính

### Kernel

Kernel là container kiểu DI trung tâm chứa tất cả services và plugins:

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

kernel = Kernel()
kernel.add_service(AzureChatCompletion(model_id, endpoint, api_key))
kernel.add_plugin(TimePlugin(), plugin_name="TimePlugin")
```

Khi gọi một prompt, kernel:
1. Chọn AI service phù hợp
2. Xây dựng prompt từ template
3. Gửi prompt đến AI
4. Nhận và phân tích phản hồi
5. Trả về phản hồi LLM

### Services

Services bao gồm AI services (chat completion, embeddings) và các runtime services khác (logging, HTTP clients). Chúng tuân theo pattern Service Provider từ .NET.

```python
# Python
kernel.add_service(AzureChatCompletion(model_id, endpoint, api_key))

// C#
builder.AddAzureOpenAIChatCompletion(modelId, endpoint, apiKey);
```

### Plugins

Plugins là các thành phần có thể hành động mở rộng khả năng của model - lấy dữ liệu từ database, gọi API bên ngoài, hoặc chạy native code.

```python
from typing import Annotated
from semantic_kernel.functions import kernel_function

class MenuPlugin:
    @kernel_function(description="Provides a list of specials from the menu.")
    def get_specials(self) -> Annotated[str, "Returns the specials from the menu."]:
        return """
        Special Soup: Clam Chowder
        Special Salad: Cobb Salad
        """

    @kernel_function(description="Provides the price of the requested menu item.")
    def get_item_price(
        self, menu_item: Annotated[str, "The name of the menu item."]
    ) -> Annotated[str, "Returns the price of the menu item."]:
        return "$9.99"
```

### Agents

Semantic Kernel cung cấp framework agent toàn diện:

```python
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

agent = ChatCompletionAgent(
    service=AzureChatCompletion(),
    name="SK-Assistant",
    instructions="You are a helpful assistant.",
    plugins=[MenuPlugin()],
)

response = await agent.get_response(messages="What is the price of the soup?")
```

### Hệ Thống Multi-Agent

Semantic Kernel hỗ trợ điều phối nhiều agent chuyên biệt:

```python
billing_agent = ChatCompletionAgent(
    service=AzureChatCompletion(),
    name="BillingAgent",
    instructions="You handle billing issues."
)

refund_agent = ChatCompletionAgent(
    service=AzureChatCompletion(),
    name="RefundAgent",
    instructions="Assist users with refund inquiries.",
)

triage_agent = ChatCompletionAgent(
    service=OpenAIChatCompletion(),
    name="TriageAgent",
    instructions="Evaluate user requests and forward to BillingAgent or RefundAgent.",
    plugins=[billing_agent, refund_agent],
)
```

## Planning (Function Calling)

Semantic Kernel sử dụng khả năng function-calling native của LLM làm cơ chế planning chính. Các planner Stepwise và Handlebars cũ đã bị deprecated.

### Cách Function Calling Hoạt Động

1. Nhận lịch sử chat + JSON schema cho các function có sẵn (plugins)
2. Model quyết định trả về text hoặc gọi function với tham số
3. Nếu function được gọi, hệ thống thực thi nó
4. Kết quả function được trả về cho model
5. Model kiểm tra kết quả và có thể gọi thêm function hoặc trả lời
6. Lặp lại cho đến khi hoàn thành

```python
from semantic_kernel.connectors.ai import FunctionChoiceBehavior

execution_settings = AzureChatPromptExecutionSettings()
execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

result = await chat_completion.get_chat_message_content(
    chat_history=history,
    settings=execution_settings,
    kernel=kernel
)
```

### Gọi Function Song Song

Models có thể gọi functions tuần tự hoặc song song (OpenAI models 1106+), hữu ích cho việc thực thi nhanh hơn các task phức tạp.

## Tích Hợp Memory

Semantic Kernel hỗ trợ tích hợp với nhiều vector database cho memory:

- Azure AI Search
- Elasticsearch
- Chroma
- Pinecone
- Qdrant
- Weaviate

Memory store cho phép:
- Tìm kiếm semantic trên tài liệu
- Lịch sử hội thoại
- Truy xuất kiến thức cho RAG

## Tính Năng Chính

### Tính Linh Hoạt Model
- OpenAI và Azure OpenAI
- Hugging Face
- NVIDIA NIM
- Ollama (local)
- LMStudio (local)
- ONNX runtime

### Hệ Sinh Thái Plugin
- Native code functions
- Prompt templates
- OpenAPI specs
- Hỗ trợ Model Context Protocol (MCP)

### Hỗ Trợ Multimodal
- Xử lý text
- Input hình ảnh
- Xử lý audio

### Process Framework
Model các quy trình kinh doanh phức tạp với cách tiếp cận workflow có cấu trúc.

### Sẵn Sàng Cho Doanh Nghiệp
- Observability tích hợp
- Tính năng bảo mật
- API ổn định
- Middleware/event hooks cho logging, status và responsible AI

## Kiến Trúc

```
semantic-kernel/
├── python/
│   └── semantic_kernel/
│       ├── agents/              # Framework agent
│       │   ├── autogen/         # Tích hợp AutoGen
│       │   ├── azure_ai/        # Azure AI agents
│       │   ├── bedrock/         # AWS Bedrock agents
│       │   ├── chat_completion/ # Chat completion agents
│       │   ├── copilot_studio/  # Tích hợp Copilot Studio
│       │   ├── group_chat/      # Điều phối group chat
│       │   ├── open_ai/         # OpenAI agents
│       │   ├── orchestration/   # Các pattern điều phối
│       │   └── strategies/      # Chiến lược agent
│       ├── connectors/          # Connector AI service
│       ├── functions/           # Kernel functions
│       ├── memory/              # Memory/vector stores
│       └── processes/           # Process framework
├── dotnet/                      # Triển khai .NET
└── java/                        # Triển khai Java
```

## Các Loại Agent

| Loại Agent | Mô tả |
|-----------|-------|
| ChatCompletionAgent | Agent cơ bản sử dụng chat completion |
| AzureAIAgent | Agent sử dụng Azure AI services |
| OpenAIAssistantAgent | Agent sử dụng OpenAI Assistants API |
| BedrockAgent | Agent sử dụng AWS Bedrock |
| CopilotStudioAgent | Tích hợp với Microsoft Copilot Studio |

## Tích Hợp MCP Server

Semantic Kernel có thể expose kernel functions như MCP servers:

```python
from semantic_kernel.functions import kernel_function

@kernel_function()
def echo_function(message: str, extra: str = "") -> str:
    return f"Function echo: {message} {extra}"

kernel.add_function("echo", echo_function, "echo_function")
server = kernel.as_mcp_server(server_name="sk")
```

## Trường Hợp Sử Dụng Tốt Nhất

- **Ứng dụng AI doanh nghiệp**: Được xây dựng cho production với các pattern doanh nghiệp
- **Hệ sinh thái Microsoft**: Tích hợp Azure liền mạch
- **Ứng dụng .NET**: Hỗ trợ C# hàng đầu
- **Workflow phức tạp**: Process framework cho logic nghiệp vụ
- **Nhu cầu đa provider**: Thiết kế không phụ thuộc model
- **Ngành được quản lý**: Tính năng bảo mật và tuân thủ
- **Ứng dụng RAG**: Tích hợp vector store

## Hạn Chế

- **Đường cong học tập**: Các pattern doanh nghiệp có thể phức tạp cho use case đơn giản
- **Tập trung Microsoft**: Trải nghiệm tốt nhất trong stack Azure/Microsoft
- **Nặng**: Thiết lập nhiều hơn các giải pháp nhẹ
- **Tài liệu**: Phát triển nhanh, tài liệu có thể chậm trễ

## Cài Đặt

### Python
```bash
pip install semantic-kernel
```

### .NET
```bash
dotnet add package Microsoft.SemanticKernel
dotnet add package Microsoft.SemanticKernel.Agents.Core
```

### Java
Xem [semantic-kernel-java build](https://github.com/microsoft/semantic-kernel-java/blob/main/BUILD.md)

## So Sánh Với Các Framework Khác

| Tính năng | Semantic Kernel | LangChain | LlamaIndex |
|-----------|----------------|-----------|------------|
| Trọng tâm chính | Điều phối AI doanh nghiệp | Framework ứng dụng LLM | Data agent cho RAG |
| Ngôn ngữ | C#, Python, Java | Python, JavaScript | Python, TypeScript |
| Memory | Tích hợp vector store | Vector stores + loại memory | Storage context tích hợp |
| Planning | Function calling | Nhiều planner | Agent workflows |
| Doanh nghiệp | Mạnh | Trung bình | Trung bình |

## Tài Liệu Tham Khảo

- [Semantic Kernel GitHub](https://github.com/microsoft/semantic-kernel)
- [Tài liệu](https://learn.microsoft.com/en-us/semantic-kernel/)
- [Hướng dẫn Bắt đầu Nhanh](https://learn.microsoft.com/en-us/semantic-kernel/get-started/quick-start-guide)
- [Xây dựng Agents](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/)
- [Cộng đồng Discord](https://aka.ms/SKDiscord)
- [Cookbook](https://github.com/microsoft/SemanticKernelCookBook)
