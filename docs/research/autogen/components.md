# AutoGen Components

## Core Agent Types

### AssistantAgent

The primary LLM-powered agent for general-purpose tasks.

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-4o")

assistant = AssistantAgent(
    name="assistant",
    model_client=model_client,
    system_message="You are a helpful AI assistant.",
    tools=[my_tool],  # Optional: registered tools
    description="A helpful assistant that can answer questions.",
)
```

**Key Parameters:**
- `name`: Unique identifier for the agent
- `model_client`: The LLM client to use
- `system_message`: Instructions that define agent behavior
- `tools`: List of callable tools/functions
- `description`: Used by selectors to understand agent capabilities
- `max_consecutive_auto_reply`: Limit on automatic responses
- `reflect_on_tool_use`: Whether to reflect on tool results

**Capabilities:**
- Natural language generation
- Tool/function calling
- Multi-modal input support (with vision-capable models)
- Streaming responses

### UserProxyAgent

Represents a human user in the conversation, collecting input or simulating user behavior.

```python
from autogen_agentchat.agents import UserProxyAgent

# v0.4 - Simplified
user_proxy = UserProxyAgent("user_proxy")

# v0.2 - Full configuration
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",  # ALWAYS, TERMINATE, or NEVER
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
    system_message="A human admin.",
)
```

**Human Input Modes (v0.2):**
- `ALWAYS`: Always ask for human input
- `TERMINATE`: Ask only at termination conditions
- `NEVER`: Never ask for human input (autonomous mode)

### CodeExecutorAgent

Specialized agent for executing code blocks.

```python
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

code_executor = CodeExecutorAgent(
    name="code_executor",
    code_executor=LocalCommandLineCodeExecutor(work_dir="coding"),
)
```

### ConversableAgent (v0.2)

The base class for all conversable agents in v0.2.

```python
from autogen.agentchat import ConversableAgent

agent = ConversableAgent(
    name="conversable_agent",
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
    code_execution_config={"work_dir": "coding"},
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
)

# Register custom reply function
def reply_func(recipient, messages, sender, config):
    # Custom logic
    return True, "Custom reply"

agent.register_reply([ConversableAgent], reply_func, position=0)
```

## Team Components

### GroupChat (v0.2)

Enables multiple agents to participate in a shared conversation.

```python
import autogen

groupchat = autogen.GroupChat(
    agents=[agent1, agent2, agent3],
    messages=[],
    max_round=12,
    speaker_selection_method="auto",  # or "round_robin", "random", "manual"
)
```

### GroupChatManager (v0.2)

Orchestrates the flow of messages in a GroupChat.

```python
manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
)

# Start the conversation
user_proxy.initiate_chat(
    manager,
    message="Let's discuss the project plan.",
)
```

### RoundRobinGroupChat (v0.4)

Agents take turns in a fixed order.

```python
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination

team = RoundRobinGroupChat(
    [primary_agent, critic_agent],
    termination_condition=TextMentionTermination("APPROVE"),
)

result = await team.run(task="Write a poem about AI")
```

### SelectorGroupChat (v0.4)

Uses an LLM to dynamically select the next speaker.

```python
from autogen_agentchat.teams import SelectorGroupChat

selector_prompt = """Select an agent to perform task.
{roles}
Current conversation context:
{history}
Select from {participants}. Only select one agent.
"""

team = SelectorGroupChat(
    participants=[planner, coder, reviewer],
    model_client=model_client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=False,
    max_selector_attempts=3,
)
```

**Key Parameters:**
- `participants`: List of agents
- `model_client`: LLM for speaker selection
- `selector_prompt`: Custom prompt template with `{roles}`, `{participants}`, `{history}`
- `allow_repeated_speaker`: Whether the same agent can speak consecutively
- `selector_func`: Optional custom selection function

### Swarm (v0.4)

Agents hand off tasks to each other using tool-based selection.

```python
from autogen_agentchat.teams import Swarm

agent1 = AssistantAgent(
    "Alice",
    model_client=model_client,
    handoffs=["Bob"],  # Can hand off to Bob
    system_message="You are Alice, an expert in X.",
)

agent2 = AssistantAgent(
    "Bob",
    model_client=model_client,
    system_message="You are Bob, an expert in Y.",
)

team = Swarm([agent1, agent2], termination_condition=termination)
```

## Code Executors

### LocalCommandLineCodeExecutor

Executes code directly on the host machine.

```python
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from pathlib import Path

work_dir = Path("coding")
work_dir.mkdir(exist_ok=True)

executor = LocalCommandLineCodeExecutor(work_dir=work_dir)

result = await executor.execute_code_blocks(
    code_blocks=[
        CodeBlock(language="python", code="print('Hello, World!')"),
    ],
    cancellation_token=CancellationToken(),
)
```

**Warning:** Direct system access - use with caution.

### DockerCommandLineCodeExecutor

Executes code in isolated Docker containers.

```python
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

async with DockerCommandLineCodeExecutor(work_dir=work_dir) as executor:
    result = await executor.execute_code_blocks(
        code_blocks=[
            CodeBlock(language="python", code="print('Hello, World!')"),
        ],
        cancellation_token=CancellationToken(),
    )
```

**Benefits:**
- Isolation from host system
- Customizable Docker images
- Safer execution environment

## Model Clients

### OpenAIChatCompletionClient

```python
from autogen_ext.models.openai import OpenAIChatCompletionClient

client = OpenAIChatCompletionClient(
    model="gpt-4o",
    # api_key="sk-...",  # Optional if OPENAI_API_KEY is set
)
```

### AzureOpenAIChatCompletionClient

```python
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

client = AzureOpenAIChatCompletionClient(
    model="gpt-4o",
    azure_endpoint="https://your-endpoint.openai.azure.com/",
    api_version="2024-02-01",
)
```

### Custom/OpenAI-Compatible Clients

```python
client = OpenAIChatCompletionClient(
    model="your-model",
    base_url="https://your-compatible-api.com/v1",
    model_info={"vision": False, "json_output": True},
)
```

## Termination Conditions

```python
from autogen_agentchat.conditions import (
    MaxMessageTermination,
    TextMentionTermination,
    ExternalTermination,
)

# Stop after 10 messages
max_msg = MaxMessageTermination(10)

# Stop when "TERMINATE" appears
text_term = TextMentionTermination("TERMINATE")

# Combine with OR
combined = max_msg | text_term

# Combine with AND
both_required = max_msg & text_term
```

## Tools and Function Calling

### Defining Tools

```python
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"The weather in {city} is 72 degrees and sunny."

# Attach to agent
assistant = AssistantAgent(
    "assistant",
    model_client=model_client,
    tools=[get_weather],
)
```

### PythonCodeExecutionTool

```python
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

code_executor = DockerCommandLineCodeExecutor()
await code_executor.start()

coding_tool = PythonCodeExecutionTool(code_executor)
result = await coding_tool.run_json({"code": code}, cancellation_token)
```

### Registering Functions (v0.2)

```python
from autogen.agentchat import register_function

register_function(
    get_weather,
    caller=tool_caller,
    executor=tool_executor,
)
```

## Memory and RAG

### Memory Abstraction (v0.4)

```python
from autogen_agentchat.memory import ChromaDBVectorMemory

memory = ChromaDBVectorMemory(
    collection_name="agent_memory",
    persist_directory="./chroma_db",
)

# Add to agent
assistant = AssistantAgent(
    "assistant",
    model_client=model_client,
    memory=memory,
)
```

## Caching

### ChatCompletionCache (v0.4)

```python
from autogen_ext.cache import DiskCacheStore, ChatCompletionCache

cache_store = DiskCacheStore(cache_dir="./cache")
cached_client = ChatCompletionCache(
    model_client=model_client,
    cache_store=cache_store,
)
```

## UI Components

### Console Output

```python
from autogen_agentchat.ui import Console

# Stream output to console
await Console(team.run_stream(task="Your task"))
```

## MCP Integration

### McpWorkbench

```python
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

# Connect to MCP server (e.g., Playwright)
workbench = McpWorkbench(
    StdioServerParams(
        command="npx",
        args=["@playwright/mcp"],
    )
)

assistant = AssistantAgent(
    "web_assistant",
    model_client=model_client,
    tools=workbench.tools,
)
```
