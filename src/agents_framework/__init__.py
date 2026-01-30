"""Agents Framework - A modular AI agents framework.

This framework provides building blocks for creating AI agents with:
- LLM provider abstraction (OpenAI, Anthropic, Ollama)
- Tool system with decorators and registries
- Memory system with short-term and long-term storage
- Context management with token counting and compaction
- Agent base classes and patterns
- Execution engine with ReAct pattern

Example:
    from agents_framework.llm import LLMConfig, OpenAIProvider
    from agents_framework.tools import tool, ToolRegistry
    from agents_framework.agents import BaseAgent, AgentRole, WorkerAgent
    from agents_framework.context import ContextManager, ContextBudget
    from agents_framework.execution import AgentLoop, AgentRunner

    # Create an LLM provider
    provider = OpenAIProvider(LLMConfig(model="gpt-4o"))

    # Define tools
    @tool(description="Search the web")
    async def search(query: str) -> str:
        return f"Results for: {query}"

    # Create context manager
    budget = ContextBudget(total_tokens=16000)
    context = ContextManager(budget=budget, model="gpt-4o")

    # Create an agent
    role = AgentRole(name="assistant", description="Helpful assistant")
    # agent = MyAgent(role=role, llm=provider)

    # Use the execution engine
    loop = AgentLoop(llm=provider, tool_registry=registry)
    result = await loop.run("What is the weather?")
"""

__version__ = "0.1.0"

# Re-export main components for convenience
from .llm import (
    LLMConfig,
    LLMProvider,
    LLMResponse,
    Message,
    MessageRole,
)
from .tools import (
    BaseTool,
    ToolRegistry,
    tool,
)
from .agents import (
    AgentConfig,
    AgentRole,
    AgentStatus,
    BaseAgent,
    Task,
    TaskResult,
    WorkerAgent,
    WorkerConfig,
)
from .memory import (
    MemoryItem,
    MemoryQuery,
    MemoryStore,
)
from .context import (
    ContextBudget,
    ContextManager,
    CompactionStrategy,
    ConversationSummarizer,
)
from .execution import (
    AgentLoop,
    AgentRunner,
    LoopConfig,
    HookRegistry,
    Hook,
    HookType,
)
from .routing import (
    RoutingEngine,
    RoutingRequest,
    RoutingResult,
    RoutingRule,
    RoutingConfig,
    StaticRouter,
    PatternRouter,
)

__all__ = [
    "__version__",
    # LLM
    "LLMConfig",
    "LLMProvider",
    "LLMResponse",
    "Message",
    "MessageRole",
    # Tools
    "BaseTool",
    "ToolRegistry",
    "tool",
    # Agents
    "AgentConfig",
    "AgentRole",
    "AgentStatus",
    "BaseAgent",
    "Task",
    "TaskResult",
    "WorkerAgent",
    "WorkerConfig",
    # Memory
    "MemoryItem",
    "MemoryQuery",
    "MemoryStore",
    # Context
    "ContextBudget",
    "ContextManager",
    "CompactionStrategy",
    "ConversationSummarizer",
    # Execution
    "AgentLoop",
    "AgentRunner",
    "LoopConfig",
    "HookRegistry",
    "Hook",
    "HookType",
    # Routing
    "RoutingEngine",
    "RoutingRequest",
    "RoutingResult",
    "RoutingRule",
    "RoutingConfig",
    "StaticRouter",
    "PatternRouter",
]
