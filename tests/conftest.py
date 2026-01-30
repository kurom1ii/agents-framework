"""Common test fixtures and configuration for agents_framework tests."""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

import pytest

from agents_framework.llm.base import (
    BaseLLMProvider,
    LLMConfig,
    LLMResponse,
    Message,
    MessageRole,
    RetryConfig,
    ToolCall,
    ToolDefinition,
)
from agents_framework.memory.base import (
    MemoryConfig,
    MemoryItem,
    MemoryQuery,
    MemoryType,
)
from agents_framework.tools.base import BaseTool


# ============================================================================
# LLM Fixtures
# ============================================================================


@pytest.fixture
def sample_messages() -> List[Message]:
    """Create sample messages for testing."""
    return [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="Hello, how are you?"),
        Message(role=MessageRole.ASSISTANT, content="I'm doing well, thank you!"),
    ]


@pytest.fixture
def sample_tool_definitions() -> List[ToolDefinition]:
    """Create sample tool definitions for testing."""
    return [
        ToolDefinition(
            name="search",
            description="Search the web for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        ),
        ToolDefinition(
            name="calculate",
            description="Perform mathematical calculations",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"},
                },
                "required": ["expression"],
            },
        ),
    ]


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


@pytest.fixture
def sample_retry_config() -> RetryConfig:
    """Create sample retry configuration."""
    return RetryConfig(
        max_retries=3,
        base_delay=0.1,
        max_delay=1.0,
        exponential_base=2.0,
        jitter=False,
    )


@pytest.fixture
def sample_llm_response() -> LLMResponse:
    """Create sample LLM response."""
    return LLMResponse(
        content="This is a test response.",
        model="test-model",
        finish_reason="stop",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


@pytest.fixture
def sample_tool_call() -> ToolCall:
    """Create sample tool call."""
    return ToolCall(
        id="call_123",
        name="search",
        arguments={"query": "test query"},
    )


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


@pytest.fixture
def mock_llm_provider(sample_llm_config: LLMConfig) -> MockLLMProvider:
    """Create mock LLM provider."""
    return MockLLMProvider(sample_llm_config)


# ============================================================================
# Memory Fixtures
# ============================================================================


@pytest.fixture
def sample_memory_item() -> MemoryItem:
    """Create sample memory item."""
    return MemoryItem(
        id="test-item-1",
        content="This is a test memory item.",
        metadata={"source": "test", "importance": "high"},
        memory_type=MemoryType.SHORT_TERM,
        namespace="test-namespace",
    )


@pytest.fixture
def sample_memory_items() -> List[MemoryItem]:
    """Create multiple sample memory items."""
    return [
        MemoryItem(
            id=f"test-item-{i}",
            content=f"Memory item content {i}",
            metadata={"index": i},
            memory_type=MemoryType.SHORT_TERM,
            namespace="test-namespace",
        )
        for i in range(5)
    ]


@pytest.fixture
def sample_memory_query() -> MemoryQuery:
    """Create sample memory query."""
    return MemoryQuery(
        query_text="test query",
        namespace="test-namespace",
        limit=10,
    )


@pytest.fixture
def sample_memory_config() -> MemoryConfig:
    """Create sample memory configuration."""
    return MemoryConfig(
        namespace="test",
        max_items=100,
        default_ttl=3600,
        enable_embeddings=False,
    )


class MockMemoryStore:
    """Mock memory store for testing."""

    def __init__(self):
        self.items: Dict[str, MemoryItem] = {}

    async def store(self, item: MemoryItem) -> str:
        """Store a memory item."""
        self.items[item.id] = item
        return item.id

    async def retrieve(self, query: MemoryQuery) -> List[MemoryItem]:
        """Retrieve memory items."""
        results = list(self.items.values())
        if query.namespace:
            results = [r for r in results if r.namespace == query.namespace]
        return results[: query.limit]

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

    async def get(self, item_id: str) -> Optional[MemoryItem]:
        """Get a memory item by ID."""
        return self.items.get(item_id)

    async def count(self, namespace: Optional[str] = None) -> int:
        """Count memory items."""
        if namespace:
            return len([v for v in self.items.values() if v.namespace == namespace])
        return len(self.items)


@pytest.fixture
def mock_memory_store() -> MockMemoryStore:
    """Create mock memory store."""
    return MockMemoryStore()


# ============================================================================
# Tool Fixtures
# ============================================================================


class MockTool(BaseTool):
    """Mock tool for testing."""

    name = "mock_tool"
    description = "A mock tool for testing"

    def __init__(self, return_value: Any = "success"):
        super().__init__()
        self.return_value = return_value
        self.call_count = 0
        self.last_kwargs: Optional[Dict[str, Any]] = None

    async def execute(self, **kwargs: Any) -> Any:
        """Execute the mock tool."""
        self.call_count += 1
        self.last_kwargs = kwargs
        return self.return_value


class FailingTool(BaseTool):
    """Tool that always fails for testing error handling."""

    name = "failing_tool"
    description = "A tool that always fails"

    def __init__(self, error_message: str = "Tool execution failed"):
        super().__init__()
        self.error_message = error_message

    async def execute(self, **kwargs: Any) -> Any:
        """Raise an exception."""
        raise RuntimeError(self.error_message)


@pytest.fixture
def mock_tool() -> MockTool:
    """Create mock tool."""
    return MockTool()


@pytest.fixture
def failing_tool() -> FailingTool:
    """Create failing tool."""
    return FailingTool()


# ============================================================================
# Common Test Data
# ============================================================================


@pytest.fixture
def sample_json_schema() -> Dict[str, Any]:
    """Create sample JSON schema for testing."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The name"},
            "age": {"type": "integer", "minimum": 0},
            "email": {"type": "string", "format": "email"},
        },
        "required": ["name", "age"],
    }


@pytest.fixture
def sample_valid_data() -> Dict[str, Any]:
    """Create sample data that matches the schema."""
    return {
        "name": "Test User",
        "age": 25,
        "email": "test@example.com",
    }


@pytest.fixture
def sample_invalid_data() -> Dict[str, Any]:
    """Create sample data that doesn't match the schema."""
    return {
        "name": 123,  # Should be string
        "age": -5,  # Should be >= 0
    }
