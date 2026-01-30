"""Agents Framework - A modular AI agents framework.

This framework provides building blocks for creating AI agents with:
- LLM provider abstraction (OpenAI, Anthropic, Ollama)
- Tool system with decorators and registries
- Memory system with short-term and long-term storage
- Agent base classes and patterns

Example:
    from agents_framework.llm import LLMConfig, OpenAIProvider
    from agents_framework.tools import tool, ToolRegistry
    from agents_framework.agents import BaseAgent, AgentRole

    # Create an LLM provider
    provider = OpenAIProvider(LLMConfig(model="gpt-4o"))

    # Define tools
    @tool(description="Search the web")
    async def search(query: str) -> str:
        return f"Results for: {query}"

    # Create an agent
    role = AgentRole(name="assistant", description="Helpful assistant")
    # agent = MyAgent(role=role, llm=provider)
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
)
from .memory import (
    MemoryItem,
    MemoryQuery,
    MemoryStore,
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
    # Memory
    "MemoryItem",
    "MemoryQuery",
    "MemoryStore",
]
