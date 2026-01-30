# AutoGen Code Examples

## Installation

```bash
# Core packages
pip install -U "autogen-agentchat" "autogen-ext[openai]"

# With Docker code execution
pip install -U "autogen-ext[docker]"

# AutoGen Studio (GUI)
pip install -U "autogenstudio"
autogenstudio ui --port 8080 --appdir ./my-app
```

## Basic Conversation Setup

### Hello World (v0.4)

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    # Create model client
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    # Create assistant
    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="You are a helpful AI assistant.",
    )

    # Run a simple task
    result = await assistant.run(task="Say 'Hello World!'")
    print(result.messages[-1].content)

    # Clean up
    await model_client.close()

asyncio.run(main())
```

### Two-Agent Conversation (v0.2)

```python
import autogen

# Configuration
config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
    filter_dict={"model": ["gpt-4", "gpt-4o", "gpt-3.5-turbo"]},
)

llm_config = {"config_list": config_list, "cache_seed": 42}

# Create Assistant
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="You are a helpful AI assistant.",
)

# Create User Proxy
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
    system_message="A human admin.",
)

# Start conversation
chat_result = user_proxy.initiate_chat(
    assistant,
    message="Write a Python function to calculate factorial.",
    clear_history=True,
    silent=False,
)

# Access results
print(chat_result.summary)
print(chat_result.chat_history)
print(chat_result.cost)
```

### Two-Agent with Code Execution (v0.4)

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o", seed=42, temperature=0)

    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant. Write all code in Python. Reply 'TERMINATE' when done.",
        model_client=model_client,
    )

    code_executor = CodeExecutorAgent(
        name="code_executor",
        code_executor=LocalCommandLineCodeExecutor(work_dir="coding"),
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(10)
    team = RoundRobinGroupChat([assistant, code_executor], termination_condition=termination)

    stream = team.run_stream(task="Write a Python script to print 'Hello, world!'")
    await Console(stream)

    await model_client.close()

asyncio.run(main())
```

## Group Chat Patterns

### Round Robin Group Chat

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    # Create primary agent
    primary_agent = AssistantAgent(
        "primary",
        model_client=model_client,
        system_message="You are a helpful AI assistant.",
    )

    # Create critic agent
    critic_agent = AssistantAgent(
        "critic",
        model_client=model_client,
        system_message="Provide constructive feedback. Respond with 'APPROVE' when satisfied.",
    )

    # Create team with termination condition
    text_termination = TextMentionTermination("APPROVE")
    team = RoundRobinGroupChat(
        [primary_agent, critic_agent],
        termination_condition=text_termination,
    )

    # Run and display
    await Console(team.run_stream(task="Write a haiku about programming"))

asyncio.run(main())
```

### Selector Group Chat

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    # Planning agent
    planner = AssistantAgent(
        "planner",
        model_client=model_client,
        system_message="""You are a planning agent.
        Break down complex tasks into steps.
        Assign each step to the appropriate agent.
        Say 'DONE' when all steps are complete.""",
        description="Plans and coordinates work between agents.",
    )

    # Coding agent
    coder = AssistantAgent(
        "coder",
        model_client=model_client,
        system_message="""You are a coding agent.
        Write clean, documented Python code.
        Follow the planner's instructions.""",
        description="Writes Python code to implement solutions.",
    )

    # Reviewer agent
    reviewer = AssistantAgent(
        "reviewer",
        model_client=model_client,
        system_message="""You are a code review agent.
        Review code for bugs, style, and improvements.
        Provide specific feedback.""",
        description="Reviews code and provides feedback.",
    )

    # Custom selector prompt
    selector_prompt = """Select an agent to perform the next task.

{roles}

Current conversation:
{history}

Read the above conversation. Select the next agent from {participants}.
Make sure the planner assigns tasks before other agents begin working.
Only select one agent.
"""

    team = SelectorGroupChat(
        participants=[planner, coder, reviewer],
        model_client=model_client,
        selector_prompt=selector_prompt,
        termination_condition=TextMentionTermination("DONE"),
    )

    await Console(team.run_stream(
        task="Create a Python function that sorts a list using quicksort"
    ))

asyncio.run(main())
```

### Swarm Pattern

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import Swarm
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    # Triage agent - first point of contact
    triage = AssistantAgent(
        "Triage",
        model_client=model_client,
        handoffs=["TechSupport", "Billing", "General"],
        system_message="""You are a triage agent.
        Determine what kind of help the user needs.
        Hand off to TechSupport for technical issues.
        Hand off to Billing for payment issues.
        Hand off to General for other questions.""",
    )

    tech_support = AssistantAgent(
        "TechSupport",
        model_client=model_client,
        handoffs=["Triage"],
        system_message="""You are a technical support agent.
        Help users with technical issues.
        Hand back to Triage if this isn't a technical issue.""",
    )

    billing = AssistantAgent(
        "Billing",
        model_client=model_client,
        handoffs=["Triage"],
        system_message="""You are a billing agent.
        Help users with payment and account issues.
        Hand back to Triage if this isn't a billing issue.""",
    )

    general = AssistantAgent(
        "General",
        model_client=model_client,
        handoffs=["Triage"],
        system_message="""You are a general support agent.
        Help with general inquiries.""",
    )

    termination = MaxMessageTermination(10)
    team = Swarm(
        [triage, tech_support, billing, general],
        termination_condition=termination,
    )

    stream = team.run_stream(task="I can't log into my account and I also have a billing question")
    async for message in stream:
        print(f"[{message.source}]: {message.content}" if hasattr(message, 'content') else message)

asyncio.run(main())
```

## Code Execution Agents

### Docker Code Executor

```python
import asyncio
from pathlib import Path
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

async def main():
    work_dir = Path("coding")
    work_dir.mkdir(exist_ok=True)

    async with DockerCommandLineCodeExecutor(work_dir=work_dir) as executor:
        # Execute Python code
        result = await executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(
                    language="python",
                    code="""
import numpy as np
print(f"Numpy version: {np.__version__}")
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Sum: {arr.sum()}")
"""
                ),
            ],
            cancellation_token=CancellationToken(),
        )
        print(result)

asyncio.run(main())
```

### Local Code Executor

```python
import asyncio
from pathlib import Path
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

async def main():
    work_dir = Path("coding")
    work_dir.mkdir(exist_ok=True)

    executor = LocalCommandLineCodeExecutor(work_dir=work_dir)

    result = await executor.execute_code_blocks(
        code_blocks=[
            CodeBlock(language="python", code="print('Hello from local executor!')"),
        ],
        cancellation_token=CancellationToken(),
    )
    print(result)

asyncio.run(main())
```

### Full Code Execution Workflow

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    # Code writing assistant
    assistant = AssistantAgent(
        name="coder",
        model_client=model_client,
        system_message="""You are an expert Python programmer.
        Write clean, efficient code with proper error handling.
        Include comments explaining your code.
        Reply 'TERMINATE' when the task is complete.""",
    )

    # Code executor with Docker isolation
    async with DockerCommandLineCodeExecutor(work_dir="coding") as docker_executor:
        executor = CodeExecutorAgent(
            name="executor",
            code_executor=docker_executor,
        )

        termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(10)
        team = RoundRobinGroupChat(
            [assistant, executor],
            termination_condition=termination,
        )

        await Console(team.run_stream(
            task="Write a Python script that downloads a webpage and counts word frequency"
        ))

asyncio.run(main())
```

## Tool Use Examples

### Basic Tool Registration

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Define tools
def get_weather(city: str) -> str:
    """Get current weather for a city.

    Args:
        city: Name of the city

    Returns:
        Weather description
    """
    # Simulated weather data
    weather_data = {
        "tokyo": "72F and sunny",
        "london": "55F and rainy",
        "new york": "65F and cloudy",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")

def calculate(expression: str) -> str:
    """Calculate a mathematical expression.

    Args:
        expression: Math expression to evaluate

    Returns:
        Calculation result
    """
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=[get_weather, calculate],
        system_message="You are a helpful assistant with weather and calculator tools.",
    )

    result = await assistant.run(
        task="What's the weather in Tokyo? Also, what's 15 * 24 + 38?"
    )

    for message in result.messages:
        print(f"[{message.source}]: {message.content}")

asyncio.run(main())
```

### Tool Registration (v0.2)

```python
from autogen.agentchat import AssistantAgent, UserProxyAgent, register_function

llm_config = {
    "config_list": [{"model": "gpt-4o", "api_key": "sk-xxx"}],
    "seed": 42,
    "temperature": 0,
}

tool_caller = AssistantAgent(
    name="tool_caller",
    system_message="You are a helpful assistant. Use tools when needed.",
    llm_config=llm_config,
    max_consecutive_auto_reply=1,
)

tool_executor = UserProxyAgent(
    name="tool_executor",
    human_input_mode="NEVER",
    code_execution_config=False,
    llm_config=False,
)

def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is 72 degrees and sunny."

# Register the function with both agents
register_function(get_weather, caller=tool_caller, executor=tool_executor)

# Interactive loop
while True:
    user_input = input("User: ")
    if user_input == "exit":
        break

    chat_result = tool_executor.initiate_chat(
        tool_caller,
        message=user_input,
        summary_method="reflection_with_llm",
    )
    print("Assistant:", chat_result.summary)
```

## Human-in-the-Loop Example

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    planner = AssistantAgent(
        "planner",
        model_client=model_client,
        system_message="You are a planner. Create step-by-step plans.",
        description="Creates plans for tasks.",
    )

    worker = AssistantAgent(
        "worker",
        model_client=model_client,
        system_message="You execute tasks according to the plan.",
        description="Executes planned tasks.",
    )

    user_proxy = UserProxyAgent(
        "user",
        description="Human user for approval.",
    )

    def selector_with_approval(messages):
        """Custom selector that requires user approval after planning."""
        if not messages:
            return "planner"

        last_msg = messages[-1]

        # After planner speaks, get user approval
        if last_msg.source == "planner":
            return "user"

        # If user approves, continue to worker
        if last_msg.source == "user":
            if "approve" in last_msg.content.lower():
                return "worker"
            else:
                return "planner"  # Revise plan

        # After worker, check with planner
        if last_msg.source == "worker":
            return "planner"

        return None

    team = SelectorGroupChat(
        participants=[planner, worker, user_proxy],
        model_client=model_client,
        selector_func=selector_with_approval,
        termination_condition=MaxMessageTermination(10),
    )

    await Console(team.run_stream(task="Plan and execute a data analysis workflow"))

asyncio.run(main())
```

## MCP Web Browsing Example

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

async def main():
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        model_client_stream=True,
    )

    # Connect to Playwright MCP server for web browsing
    # First run: npx @playwright/mcp
    workbench = McpWorkbench(
        StdioServerParams(
            command="npx",
            args=["@playwright/mcp"],
        )
    )

    assistant = AssistantAgent(
        name="web_assistant",
        model_client=model_client,
        tools=workbench.tools,
        system_message="You are a web browsing assistant. Use the tools to navigate and interact with web pages.",
    )

    await Console(assistant.run_stream(
        task="Go to news.ycombinator.com and get the top 3 article titles"
    ))

asyncio.run(main())
```

## State Persistence Example

```python
import asyncio
import json
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    agent1 = AssistantAgent("agent1", model_client=model_client)
    agent2 = AssistantAgent("agent2", model_client=model_client)

    team = RoundRobinGroupChat(
        [agent1, agent2],
        termination_condition=MaxMessageTermination(5),
    )

    # Run first part
    result = await team.run(task="Count from 1 to 10, one at a time")
    print("First run complete")

    # Save state
    state = await team.save_state()
    with open("team_state.json", "w") as f:
        json.dump(state, f)

    # ... later, in a new session ...

    # Load state and continue
    with open("team_state.json", "r") as f:
        state = json.load(f)

    await team.load_state(state)

    # Continue from where we left off
    result = await team.run()  # No task needed, continues previous
    print("Continued run complete")

asyncio.run(main())
```

## Multi-Model Example

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    # Different models for different agents
    gpt4_client = OpenAIChatCompletionClient(model="gpt-4o")
    gpt35_client = OpenAIChatCompletionClient(model="gpt-3.5-turbo")

    # GPT-4 for complex reasoning
    reasoner = AssistantAgent(
        "reasoner",
        model_client=gpt4_client,
        system_message="You are an expert at complex reasoning and analysis.",
    )

    # GPT-3.5 for simpler tasks
    helper = AssistantAgent(
        "helper",
        model_client=gpt35_client,
        system_message="You help with simple tasks and formatting.",
    )

    team = RoundRobinGroupChat(
        [reasoner, helper],
        termination_condition=MaxMessageTermination(6),
    )

    result = await team.run(task="Explain quantum computing simply")

    # Clean up
    await gpt4_client.close()
    await gpt35_client.close()

asyncio.run(main())
```
