"""Local fixtures for skills tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import pytest

from agents_framework.llm.base import (
    BaseLLMProvider,
    LLMConfig,
    LLMResponse,
    Message,
    ToolDefinition,
)
from agents_framework.skills.base import (
    BaseSkill,
    FunctionSkill,
    SkillCategory,
    SkillContext,
    SkillMetadata,
    SkillResult,
    skill,
)
from agents_framework.skills.registry import SkillRegistry


# ============================================================================
# Mock LLM Provider (copied from main conftest for independence)
# ============================================================================


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, config: LLMConfig, responses: Optional[List[LLMResponse]] = None):
        super().__init__(config)
        self.responses = responses or []
        self.call_count = 0
        self.last_messages: Optional[List[Message]] = None
        self.last_tools: Optional[List[ToolDefinition]] = None

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a mock response."""
        self.last_messages = messages
        self.last_tools = tools
        self.call_count += 1

        if self.responses:
            return self.responses[min(self.call_count - 1, len(self.responses) - 1)]

        return LLMResponse(
            content="Mock response",
            model=self.config.model,
            finish_reason="stop",
        )

    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream mock response chunks."""
        self.last_messages = messages
        self.last_tools = tools
        self.call_count += 1

        async def _stream() -> AsyncIterator[str]:
            chunks = ["Hello", " ", "World", "!"]
            for chunk in chunks:
                yield chunk

        async for chunk in _stream():
            yield chunk

    def supports_tools(self) -> bool:
        """Return True for mock provider."""
        return True


# ============================================================================
# Mock Memory Store (copied from main conftest for independence)
# ============================================================================


class MockMemoryStore:
    """Mock memory store for testing."""

    def __init__(self):
        from agents_framework.memory.base import MemoryItem

        self.items: Dict[str, MemoryItem] = {}

    async def store(self, item: Any) -> str:
        """Store a memory item."""
        self.items[item.id] = item
        return item.id

    async def retrieve(self, query: Any) -> List[Any]:
        """Retrieve memory items."""
        results = list(self.items.values())
        if hasattr(query, "namespace") and query.namespace:
            results = [r for r in results if r.namespace == query.namespace]
        return results[: query.limit] if hasattr(query, "limit") else results

    async def delete(self, item_id: str) -> bool:
        """Delete a memory item."""
        if item_id in self.items:
            del self.items[item_id]
            return True
        return False

    async def clear(self, namespace: Optional[str] = None) -> None:
        """Clear memory items."""
        if namespace:
            self.items = {k: v for k, v in self.items.items() if v.namespace != namespace}
        else:
            self.items.clear()

    async def get(self, item_id: str) -> Optional[Any]:
        """Get a memory item by ID."""
        return self.items.get(item_id)

    async def count(self, namespace: Optional[str] = None) -> int:
        """Count memory items."""
        if namespace:
            return len([v for v in self.items.values() if v.namespace == namespace])
        return len(self.items)


# ============================================================================
# Mock Classes for Skills Testing
# ============================================================================


class MockToolRegistry:
    """Mock tool registry for testing skills that search tools."""

    def __init__(self, tools: Optional[List[BaseTool]] = None):
        self._tools = tools or []

    def list_tools(self) -> List[BaseTool]:
        """List all registered tools."""
        return self._tools

    def list_names(self) -> List[str]:
        """List all tool names."""
        return [t.name for t in self._tools]


class SimpleMockTool:
    """Simple mock tool for testing (not inheriting from BaseTool)."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    async def execute(self, **kwargs: Any) -> Any:
        return "mock result"


# ============================================================================
# Skill Fixtures
# ============================================================================


@pytest.fixture
def skill_registry() -> SkillRegistry:
    """Create a fresh skill registry."""
    return SkillRegistry()


@pytest.fixture
def skill_context() -> SkillContext:
    """Create a basic skill context."""
    return SkillContext(
        agent_id="test-agent",
        task_id="test-task",
        session_id="test-session",
        config={"test_key": "test_value"},
    )


@pytest.fixture
def skill_context_with_llm(sample_llm_config: LLMConfig) -> SkillContext:
    """Create a skill context with a mock LLM provider."""
    mock_llm = MockLLMProvider(
        sample_llm_config,
        responses=[
            LLMResponse(
                content="This is a test summary.",
                model="test-model",
                finish_reason="stop",
            )
        ],
    )
    return SkillContext(
        agent_id="test-agent",
        task_id="test-task",
        session_id="test-session",
        llm=mock_llm,
    )


@pytest.fixture
def skill_context_with_memory() -> SkillContext:
    """Create a skill context with a mock memory store."""
    mock_memory = MockMemoryStore()
    return SkillContext(
        agent_id="test-agent",
        task_id="test-task",
        session_id="test-session",
        memory=mock_memory,
    )


@pytest.fixture
def skill_context_with_tools() -> SkillContext:
    """Create a skill context with a mock tool registry."""
    tools = [
        SimpleMockTool("search_web", "Search the web for information"),
        SimpleMockTool("calculate", "Perform mathematical calculations"),
        SimpleMockTool("read_file", "Read contents of a file"),
    ]
    mock_tools = MockToolRegistry(tools)
    return SkillContext(
        agent_id="test-agent",
        task_id="test-task",
        session_id="test-session",
        tools=mock_tools,
    )


@pytest.fixture
def skill_context_full(sample_llm_config: LLMConfig) -> SkillContext:
    """Create a skill context with all dependencies."""
    mock_llm = MockLLMProvider(
        sample_llm_config,
        responses=[
            LLMResponse(
                content="This is a test response.",
                model="test-model",
                finish_reason="stop",
            )
        ],
    )
    mock_memory = MockMemoryStore()
    tools = [
        SimpleMockTool("search_web", "Search the web for information"),
        SimpleMockTool("calculate", "Perform mathematical calculations"),
    ]
    mock_tools = MockToolRegistry(tools)

    return SkillContext(
        agent_id="test-agent",
        task_id="test-task",
        session_id="test-session",
        llm=mock_llm,
        memory=mock_memory,
        tools=mock_tools,
        config={"test_key": "test_value"},
    )


@pytest.fixture
def sample_skill_metadata() -> SkillMetadata:
    """Create sample skill metadata."""
    return SkillMetadata(
        name="test_skill",
        description="A test skill for testing",
        version="1.0.0",
        author="Test Author",
        category=SkillCategory.GENERAL,
        tags=["test", "sample"],
        requires_llm=False,
        timeout=30.0,
    )


# ============================================================================
# Sample Skill Classes for Testing
# ============================================================================


class SimpleTestSkill(BaseSkill):
    """A simple test skill that returns a greeting."""

    def __init__(self):
        super().__init__(
            metadata=SkillMetadata(
                name="simple_test",
                description="A simple test skill",
                category=SkillCategory.GENERAL,
                tags=["test"],
            )
        )

    async def execute(
        self,
        context: SkillContext,
        greeting_name: str = "World",
    ) -> str:
        return f"Hello, {greeting_name}!"


class LLMRequiredSkill(BaseSkill):
    """A skill that requires an LLM provider."""

    def __init__(self):
        super().__init__(
            metadata=SkillMetadata(
                name="llm_required",
                description="A skill that requires LLM",
                category=SkillCategory.REASONING,
                requires_llm=True,
            )
        )

    async def execute(
        self,
        context: SkillContext,
        prompt: str,
    ) -> str:
        from agents_framework.llm import Message, MessageRole

        messages = [Message(role=MessageRole.USER, content=prompt)]
        response = await context.llm.generate(messages)
        return response.content


class SlowSkill(BaseSkill):
    """A skill that takes a long time to execute."""

    def __init__(self, delay: float = 0.1):
        super().__init__(
            metadata=SkillMetadata(
                name="slow_skill",
                description="A slow skill for timeout testing",
                timeout=0.05,  # Very short timeout
            )
        )
        self.delay = delay

    async def execute(
        self,
        context: SkillContext,
    ) -> str:
        import asyncio

        await asyncio.sleep(self.delay)
        return "Done!"


class FailingSkill(BaseSkill):
    """A skill that always raises an error."""

    def __init__(self, error_message: str = "Skill failed"):
        super().__init__(
            metadata=SkillMetadata(
                name="failing_skill",
                description="A skill that always fails",
            )
        )
        self.error_message = error_message

    async def execute(
        self,
        context: SkillContext,
    ) -> str:
        raise RuntimeError(self.error_message)


@pytest.fixture
def simple_test_skill() -> SimpleTestSkill:
    """Create a simple test skill."""
    return SimpleTestSkill()


@pytest.fixture
def llm_required_skill() -> LLMRequiredSkill:
    """Create an LLM-required skill."""
    return LLMRequiredSkill()


@pytest.fixture
def slow_skill() -> SlowSkill:
    """Create a slow skill for timeout testing."""
    return SlowSkill()


@pytest.fixture
def failing_skill() -> FailingSkill:
    """Create a failing skill."""
    return FailingSkill()


# ============================================================================
# LLM Config Fixtures (needed for skill_context_with_llm)
# ============================================================================


@pytest.fixture
def sample_llm_config() -> LLMConfig:
    """Create sample LLM configuration."""
    return LLMConfig(
        model="test-model",
        api_key="test-api-key",
        temperature=0.7,
        max_tokens=1000,
        timeout=30.0,
    )
