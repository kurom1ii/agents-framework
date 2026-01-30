# Microsoft Semantic Kernel

## Overview

**Semantic Kernel** is Microsoft's enterprise-ready SDK for building intelligent AI agents and multi-agent systems. It is a model-agnostic framework that empowers developers to build, orchestrate, and deploy AI agents with enterprise-grade reliability and flexibility.

- **Repository**: https://github.com/microsoft/semantic-kernel
- **Stars**: ~27,000
- **Languages**: C#, Python, Java
- **License**: MIT
- **Documentation**: https://learn.microsoft.com/en-us/semantic-kernel/

## Core Philosophy

Semantic Kernel is designed as a "Dependency Injection container for AI" - the **Kernel** is the central component that manages all services and plugins necessary to run AI applications. It follows enterprise patterns familiar to .NET developers while providing a consistent experience across Python and Java.

## Key Concepts

### The Kernel

The Kernel is the central DI-style container that holds all services and plugins:

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

kernel = Kernel()
kernel.add_service(AzureChatCompletion(model_id, endpoint, api_key))
kernel.add_plugin(TimePlugin(), plugin_name="TimePlugin")
```

When invoking a prompt, the kernel:
1. Selects the appropriate AI service
2. Builds the prompt from a template
3. Sends the prompt to the AI
4. Receives and parses the response
5. Returns the LLM response

### Services

Services include AI services (chat completion, embeddings) and other runtime services (logging, HTTP clients). They follow the Service Provider pattern from .NET.

```python
# Python
kernel.add_service(AzureChatCompletion(model_id, endpoint, api_key))

// C#
builder.AddAzureOpenAIChatCompletion(modelId, endpoint, apiKey);
```

### Plugins

Plugins are actionable components that extend the model's capabilities - fetching data from databases, calling external APIs, or running native code.

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

Semantic Kernel provides a comprehensive agent framework:

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

### Multi-Agent Systems

Semantic Kernel supports orchestrating multiple specialized agents:

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

Semantic Kernel uses the LLM's native function-calling capability as its primary planning mechanism. The older Stepwise and Handlebars planners have been deprecated.

### How Function Calling Works

1. Receive chat history + JSON schemas for available functions (plugins)
2. Model decides to return text or call a function with parameters
3. If function is called, the system invokes it
4. Function result is returned to the model
5. Model inspects results and may call more functions or reply
6. Repeat until completion

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

### Parallel Function Calling

Models can call functions sequentially or in parallel (OpenAI models 1106+), useful for faster execution of complex tasks.

## Memory Integration

Semantic Kernel supports integration with various vector databases for memory:

- Azure AI Search
- Elasticsearch
- Chroma
- Pinecone
- Qdrant
- Weaviate

Memory stores enable:
- Semantic search over documents
- Conversation history
- Knowledge retrieval for RAG

## Key Features

### Model Flexibility
- OpenAI and Azure OpenAI
- Hugging Face
- NVIDIA NIM
- Ollama (local)
- LMStudio (local)
- ONNX runtime

### Plugin Ecosystem
- Native code functions
- Prompt templates
- OpenAPI specs
- Model Context Protocol (MCP) support

### Multimodal Support
- Text processing
- Vision inputs
- Audio processing

### Process Framework
Model complex business processes with structured workflow approach.

### Enterprise Ready
- Built-in observability
- Security features
- Stable APIs
- Middleware/events hooks for logging, status, and responsible AI

## Architecture

```
semantic-kernel/
├── python/
│   └── semantic_kernel/
│       ├── agents/              # Agent framework
│       │   ├── autogen/         # AutoGen integration
│       │   ├── azure_ai/        # Azure AI agents
│       │   ├── bedrock/         # AWS Bedrock agents
│       │   ├── chat_completion/ # Chat completion agents
│       │   ├── copilot_studio/  # Copilot Studio integration
│       │   ├── group_chat/      # Group chat orchestration
│       │   ├── open_ai/         # OpenAI agents
│       │   ├── orchestration/   # Orchestration patterns
│       │   └── strategies/      # Agent strategies
│       ├── connectors/          # AI service connectors
│       ├── functions/           # Kernel functions
│       ├── memory/              # Memory/vector stores
│       └── processes/           # Process framework
├── dotnet/                      # .NET implementation
└── java/                        # Java implementation
```

## Agent Types

| Agent Type | Description |
|-----------|-------------|
| ChatCompletionAgent | Basic agent using chat completion |
| AzureAIAgent | Agent using Azure AI services |
| OpenAIAssistantAgent | Agent using OpenAI Assistants API |
| BedrockAgent | Agent using AWS Bedrock |
| CopilotStudioAgent | Integration with Microsoft Copilot Studio |

## MCP Server Integration

Semantic Kernel can expose kernel functions as MCP servers:

```python
from semantic_kernel.functions import kernel_function

@kernel_function()
def echo_function(message: str, extra: str = "") -> str:
    return f"Function echo: {message} {extra}"

kernel.add_function("echo", echo_function, "echo_function")
server = kernel.as_mcp_server(server_name="sk")
```

## Best Use Cases

- **Enterprise AI applications**: Built for production with enterprise patterns
- **Microsoft ecosystem**: Seamless Azure integration
- **.NET applications**: First-class C# support
- **Complex workflows**: Process framework for business logic
- **Multi-provider needs**: Model-agnostic design
- **Regulated industries**: Security and compliance features
- **RAG applications**: Vector store integrations

## Limitations

- **Learning curve**: Enterprise patterns can be complex for simple use cases
- **Microsoft-centric**: Best experience in Azure/Microsoft stack
- **Heavyweight**: More setup than lightweight alternatives
- **Documentation**: Rapidly evolving, documentation may lag

## Installation

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
See [semantic-kernel-java build](https://github.com/microsoft/semantic-kernel-java/blob/main/BUILD.md)

## Comparison with Other Frameworks

| Feature | Semantic Kernel | LangChain | LlamaIndex |
|---------|----------------|-----------|------------|
| Primary Focus | Enterprise AI orchestration | LLM application framework | Data agents for RAG |
| Languages | C#, Python, Java | Python, JavaScript | Python, TypeScript |
| Memory | Vector store integrations | Vector stores + memory types | Built-in storage context |
| Planning | Function calling | Various planners | Agent workflows |
| Enterprise | Strong | Moderate | Moderate |

## References

- [Semantic Kernel GitHub](https://github.com/microsoft/semantic-kernel)
- [Documentation](https://learn.microsoft.com/en-us/semantic-kernel/)
- [Getting Started Guide](https://learn.microsoft.com/en-us/semantic-kernel/get-started/quick-start-guide)
- [Building Agents](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/)
- [Discord Community](https://aka.ms/SKDiscord)
- [Cookbook](https://github.com/microsoft/SemanticKernelCookBook)
