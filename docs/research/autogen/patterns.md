# AutoGen Patterns

## Conversation Patterns

### Two-Agent Chat Pattern

The simplest pattern for agent interaction.

```python
from autogen.agentchat import AssistantAgent, UserProxyAgent

# Create agents
assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful AI assistant.",
    llm_config=llm_config,
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding"},
)

# Initiate conversation
chat_result = user_proxy.initiate_chat(
    assistant,
    message="Write a Python script to print 'Hello, world!'",
)
```

### Reflection Pattern

One agent creates, another critiques.

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination

primary_agent = AssistantAgent(
    "primary",
    model_client=model_client,
    system_message="You are a helpful AI assistant.",
)

critic_agent = AssistantAgent(
    "critic",
    model_client=model_client,
    system_message="Provide constructive feedback. Respond with 'APPROVE' when satisfied.",
)

team = RoundRobinGroupChat(
    [primary_agent, critic_agent],
    termination_condition=TextMentionTermination("APPROVE"),
)

result = await team.run(task="Write a poem about artificial intelligence")
```

### Tool Use Pattern

Agent uses tools to accomplish tasks.

```python
from autogen_agentchat.agents import AssistantAgent

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"The weather in {city} is 72 degrees and sunny."

def get_stock_price(symbol: str) -> str:
    """Get stock price for a symbol."""
    return f"The stock price of {symbol} is $150.00"

assistant = AssistantAgent(
    "assistant",
    model_client=model_client,
    tools=[get_weather, get_stock_price],
    system_message="You are a helpful assistant with access to weather and stock tools.",
)

result = await assistant.run(task="What's the weather in Tokyo and the price of MSFT?")
```

### Code Generation and Execution Pattern

```python
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

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

await Console(team.run_stream(task="Write and run a script that calculates fibonacci numbers"))
```

## Human-in-the-Loop Patterns

### Approval Pattern

User must approve before proceeding.

```python
from autogen_agentchat.agents import UserProxyAgent

user_proxy = UserProxyAgent(
    "UserProxyAgent",
    description="A proxy for the user to approve or disapprove tasks."
)

def selector_func_with_user_proxy(messages):
    if messages[-1].source != planning_agent.name and messages[-1].source != user_proxy.name:
        return planning_agent.name

    if messages[-1].source == planning_agent.name:
        if messages[-2].source == user_proxy.name and "APPROVE" in messages[-1].content.upper():
            return None  # Proceed to next agent
        return user_proxy.name  # Get user approval

    if messages[-1].source == user_proxy.name:
        if "APPROVE" not in messages[-1].content.upper():
            return planning_agent.name  # Revise plan

    return None

team = SelectorGroupChat(
    [planning_agent, worker_agent, user_proxy],
    model_client=model_client,
    selector_func=selector_func_with_user_proxy,
)
```

### Code Execution Approval

```python
from autogen_agentchat.agents import ApprovalRequest, ApprovalResponse

def approval_func(request: ApprovalRequest) -> ApprovalResponse:
    """Request user confirmation before code execution."""
    print(f"Code to execute:\n{request.code}")
    user_input = input("Approve? (y/n): ").strip().lower()

    if user_input == 'y':
        return ApprovalResponse(approved=True, reason="User approved")
    else:
        return ApprovalResponse(approved=False, reason="User denied")

from autogen_ext.teams.magentic_one import MagenticOne

m1 = MagenticOne(client=client, approval_func=approval_func)
result = await Console(m1.run_stream(task="Write a Python script"))
```

### Human Input Modes (v0.2)

```python
# ALWAYS: Always request human input
user_proxy = UserProxyAgent(
    "user",
    human_input_mode="ALWAYS",
)

# TERMINATE: Request input at termination conditions
user_proxy = UserProxyAgent(
    "user",
    human_input_mode="TERMINATE",
    is_termination_msg=lambda x: x.get("content", "").endswith("DONE"),
)

# NEVER: Fully autonomous
user_proxy = UserProxyAgent(
    "user",
    human_input_mode="NEVER",
)
```

## Group Chat Patterns

### Round Robin Group Chat

Agents speak in fixed rotation.

```python
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination

agent1 = AssistantAgent("Assistant1", model_client=model_client)
agent2 = AssistantAgent("Assistant2", model_client=model_client)
agent3 = AssistantAgent("Assistant3", model_client=model_client)

termination = MaxMessageTermination(10)
team = RoundRobinGroupChat([agent1, agent2, agent3], termination_condition=termination)

result = await team.run(task="Discuss approaches to solving climate change")
```

### Selector Group Chat

LLM-based dynamic speaker selection.

```python
from autogen_agentchat.teams import SelectorGroupChat

planner = AssistantAgent(
    "planner",
    model_client=model_client,
    system_message="You are a planning agent. Break down tasks into steps.",
    description="Plans and coordinates work.",
)

coder = AssistantAgent(
    "coder",
    model_client=model_client,
    system_message="You write Python code.",
    description="Writes code to implement solutions.",
)

reviewer = AssistantAgent(
    "reviewer",
    model_client=model_client,
    system_message="You review code for bugs and improvements.",
    description="Reviews and improves code quality.",
)

selector_prompt = """Select an agent to perform the next task.
{roles}

Current conversation:
{history}

Select from {participants}. Only select one agent.
Make sure the planner assigns tasks before other agents begin.
"""

team = SelectorGroupChat(
    participants=[planner, coder, reviewer],
    model_client=model_client,
    selector_prompt=selector_prompt,
    termination_condition=TextMentionTermination("DONE"),
)
```

### Custom Selector Function

```python
def custom_selector(messages):
    """Custom logic for speaker selection."""
    last_speaker = messages[-1].source if messages else None

    if last_speaker == "planner":
        return "coder"
    elif last_speaker == "coder":
        return "reviewer"
    elif last_speaker == "reviewer":
        return "planner"
    else:
        return "planner"

team = SelectorGroupChat(
    participants=[planner, coder, reviewer],
    model_client=model_client,
    selector_func=custom_selector,
)
```

### Swarm Pattern

Agents hand off based on their capabilities.

```python
from autogen_agentchat.teams import Swarm

alice = AssistantAgent(
    "Alice",
    model_client=model_client,
    handoffs=["Bob", "Charlie"],
    system_message="You are Alice. Hand off to Bob for coding, Charlie for review.",
)

bob = AssistantAgent(
    "Bob",
    model_client=model_client,
    handoffs=["Alice", "Charlie"],
    system_message="You are Bob the coder. Hand back to Alice when done.",
)

charlie = AssistantAgent(
    "Charlie",
    model_client=model_client,
    handoffs=["Alice", "Bob"],
    system_message="You are Charlie the reviewer.",
)

team = Swarm([alice, bob, charlie], termination_condition=termination)
```

## Nested Conversation Patterns

### Nested Team Pattern

A team that contains inner teams.

```python
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.base import Response

class NestedTeamAgent(BaseChatAgent):
    """An agent that runs an inner team."""

    def __init__(self, name: str, inner_team: RoundRobinGroupChat):
        super().__init__(name, description="Runs an inner team")
        self._inner_team = inner_team

    async def on_messages(self, messages, cancellation_token):
        # Run inner team with the messages
        result = await self._inner_team.run(
            task=messages,
            cancellation_token=cancellation_token
        )
        # Return the last message from inner team
        return Response(
            chat_message=result.messages[-1],
            inner_messages=result.messages[:-1]
        )

    async def on_reset(self, cancellation_token):
        await self._inner_team.reset()

    @property
    def produced_message_types(self):
        return (TextMessage,)

# Create inner team
inner_agent1 = AssistantAgent("inner1", model_client=model_client)
inner_agent2 = AssistantAgent("inner2", model_client=model_client)
inner_team = RoundRobinGroupChat([inner_agent1, inner_agent2], max_turns=3)

# Create nested agent
nested_agent = NestedTeamAgent("nested", inner_team)

# Use in outer team
outer_team = RoundRobinGroupChat([nested_agent, outer_agent], termination_condition=termination)
```

### Society of Mind Agent

For complex nested scenarios.

```python
from autogen_agentchat.agents import SocietyOfMindAgent

inner_team = RoundRobinGroupChat([agent1, agent2], max_turns=5)

society_agent = SocietyOfMindAgent(
    name="society",
    inner_team=inner_team,
    description="A team of experts working together",
)
```

## Agent as Tool Pattern

Wrap agents as callable tools for other agents.

```python
from autogen_ext.agents import AgentTool

math_expert = AssistantAgent(
    "math_expert",
    model_client=model_client,
    system_message="You are a math expert. Solve math problems.",
)

chemistry_expert = AssistantAgent(
    "chemistry_expert",
    model_client=model_client,
    system_message="You are a chemistry expert.",
)

# Wrap as tools
math_tool = AgentTool(math_expert, return_value_as_last_message=True)
chemistry_tool = AgentTool(chemistry_expert, return_value_as_last_message=True)

# General assistant with expert tools
assistant = AssistantAgent(
    "assistant",
    model_client=model_client,
    tools=[math_tool, chemistry_tool],
    system_message="Use experts for specialized questions.",
)
```

## Termination Patterns

### Keyword Termination

```python
termination = TextMentionTermination("TERMINATE")
```

### Message Limit

```python
termination = MaxMessageTermination(10)
```

### Combined Conditions

```python
# Either condition triggers termination
termination = TextMentionTermination("DONE") | MaxMessageTermination(20)

# Both conditions required
termination = SomeCondition() & AnotherCondition()
```

### Custom Termination

```python
is_termination_msg = lambda x: x.get("content", "").rstrip().endswith("TERMINATE")

agent = AssistantAgent(
    "assistant",
    is_termination_msg=is_termination_msg,
    ...
)
```

## State Persistence Pattern

```python
# Save team state
state = await team.save_state()

# Store state (e.g., to file)
import json
with open("team_state.json", "w") as f:
    json.dump(state, f)

# Later: Load and resume
with open("team_state.json", "r") as f:
    state = json.load(f)

await team.load_state(state)
result = await team.run()  # Continues from saved state
```

## Streaming Pattern

```python
from autogen_agentchat.ui import Console

# Stream to console
await Console(team.run_stream(task="Your task"))

# Custom streaming handler
async for message in team.run_stream(task="Your task"):
    if hasattr(message, 'content'):
        print(f"[{message.source}]: {message.content}")
```

## Cancellation Pattern

```python
from autogen_core import CancellationToken
import asyncio

cancellation_token = CancellationToken()

# Start task
task = asyncio.create_task(
    team.run(task="Long running task", cancellation_token=cancellation_token)
)

# Cancel after timeout
await asyncio.sleep(30)
cancellation_token.cancel()
```
