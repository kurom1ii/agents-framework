"""Fixtures for integration tests.

Provides mock MCP servers, test containers, and integration-specific fixtures.
"""

import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
# Mocks available if needed

from agents_framework.llm.base import LLMConfig
from tests.conftest import MockLLMProvider


# ============================================================================
# Mock MCP Server
# ============================================================================

@dataclass
class MockMCPTool:
    """Mock MCP tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any]


class MockMCPClient:
    """Mock MCP client for testing without real server."""

    def __init__(self, tools: Optional[List[MockMCPTool]] = None):
        self.tools = tools or []
        self.connected = False
        self.tool_call_history: List[Dict[str, Any]] = []
        self._tool_handlers: Dict[str, Any] = {}

    async def connect(self):
        """Simulate connection to MCP server."""
        self.connected = True

    async def disconnect(self):
        """Simulate disconnection."""
        self.connected = False

    def list_tools(self) -> List[MockMCPTool]:
        """Return registered tools."""
        return self.tools

    def register_tool_handler(self, name: str, handler):
        """Register a handler for a tool."""
        self._tool_handlers[name] = handler

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool and track the call."""
        if not self.connected:
            raise RuntimeError("Not connected to MCP server")

        self.tool_call_history.append({"name": name, "arguments": arguments})

        if name in self._tool_handlers:
            handler = self._tool_handlers[name]
            if callable(handler):
                return handler(**arguments)
            return handler

        # Default mock responses
        mock_responses = {
            "read_file": f"Content of file: {arguments.get('path', 'unknown')}",
            "write_file": "File written successfully",
            "search": f"Search results for: {arguments.get('query', '')}",
            "execute": f"Executed: {arguments.get('command', '')}",
        }
        return mock_responses.get(name, f"Mock response for {name}")


@pytest.fixture
def mock_mcp_client():
    """Create a mock MCP client with common tools."""
    tools = [
        MockMCPTool(
            name="read_file",
            description="Read contents of a file",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
        MockMCPTool(
            name="write_file",
            description="Write contents to a file",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        ),
        MockMCPTool(
            name="search",
            description="Search for content",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
    ]
    return MockMCPClient(tools)


# ============================================================================
# LLM Fixtures
# ============================================================================

@pytest.fixture
def integration_llm_config() -> LLMConfig:
    """LLM config for integration tests."""
    return LLMConfig(
        model="test-model",
        api_key="test-key",
        temperature=0.0,  # Deterministic for tests
    )


@pytest.fixture
def mock_llm(integration_llm_config) -> MockLLMProvider:
    """Mock LLM provider for integration tests."""
    return MockLLMProvider(integration_llm_config)


# ============================================================================
# Memory Fixtures
# ============================================================================

@pytest.fixture
def temp_memory_dir(tmp_path):
    """Temporary directory for memory persistence tests."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    return memory_dir


@pytest.fixture
def temp_session_dir(tmp_path):
    """Temporary directory for session tests."""
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    return session_dir


# ============================================================================
# Team Fixtures
# ============================================================================

class MockAgent:
    """Simple mock agent for team tests."""

    def __init__(self, agent_id: str, role: str = "worker"):
        self.agent_id = agent_id
        self.role = role
        self.received_messages: List[Any] = []
        self.responses: List[str] = []

    async def process(self, message: Any) -> str:
        """Process a message."""
        self.received_messages.append(message)
        if self.responses:
            return self.responses.pop(0)
        return f"Response from {self.agent_id}"


@pytest.fixture
def mock_agents():
    """Create a set of mock agents for team tests."""
    return {
        "supervisor": MockAgent("supervisor", role="supervisor"),
        "researcher": MockAgent("researcher", role="researcher"),
        "writer": MockAgent("writer", role="writer"),
        "analyst": MockAgent("analyst", role="analyst"),
    }
