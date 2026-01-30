"""Fixtures for end-to-end tests.

Provides realistic test scenarios and fixtures for complete workflow testing.
"""

import pytest
from typing import Dict, List
from dataclasses import dataclass, field

from agents_framework.llm.base import Message, MessageRole
from agents_framework.tools.base import tool
from agents_framework.tools.registry import ToolRegistry
from tests.conftest import MockLLMProvider


# ============================================================================
# Realistic Tool Fixtures
# ============================================================================

@pytest.fixture
def research_tools():
    """Tools for research scenarios."""
    registry = ToolRegistry()

    @tool(name="web_search", description="Search the web for information")
    def web_search(query: str) -> str:
        return f"Search results for '{query}': Found 10 relevant articles about the topic."

    @tool(name="read_article", description="Read an article from URL")
    def read_article(url: str) -> str:
        return f"Article content from {url}: This is a detailed article about the requested topic..."

    @tool(name="summarize", description="Summarize text content")
    def summarize(text: str) -> str:
        return f"Summary: {text[:100]}..." if len(text) > 100 else f"Summary: {text}"

    @tool(name="save_note", description="Save a note to memory")
    def save_note(title: str, content: str) -> str:
        return f"Note '{title}' saved successfully."

    registry.register(web_search)
    registry.register(read_article)
    registry.register(summarize)
    registry.register(save_note)

    return registry


@pytest.fixture
def coding_tools():
    """Tools for coding scenarios."""
    registry = ToolRegistry()

    @tool(name="read_file", description="Read a file")
    def read_file(path: str) -> str:
        return f"def hello():\n    print('Hello from {path}')"

    @tool(name="write_file", description="Write to a file")
    def write_file(path: str, content: str) -> str:
        return f"Written {len(content)} bytes to {path}"

    @tool(name="run_tests", description="Run unit tests")
    def run_tests(test_path: str) -> str:
        return "All 5 tests passed!"

    @tool(name="lint_code", description="Lint code for errors")
    def lint_code(path: str) -> str:
        return "No linting errors found."

    registry.register(read_file)
    registry.register(write_file)
    registry.register(run_tests)
    registry.register(lint_code)

    return registry


# ============================================================================
# Scenario Builders
# ============================================================================

@dataclass
class E2EScenario:
    """Represents an end-to-end test scenario."""
    name: str
    description: str
    initial_message: str
    expected_tool_sequence: List[str]
    expected_final_keywords: List[str]
    tool_responses: Dict[str, str] = field(default_factory=dict)


@pytest.fixture
def research_scenario():
    """Research task scenario."""
    return E2EScenario(
        name="research_task",
        description="Agent researches a topic and produces a summary",
        initial_message="Research the benefits of Python for data science",
        expected_tool_sequence=["web_search", "summarize"],
        expected_final_keywords=["Python", "data science"],
        tool_responses={
            "web_search": "Python is excellent for data science due to libraries like pandas, numpy, and scikit-learn.",
            "summarize": "Summary: Python is ideal for data science with rich libraries.",
        },
    )


@pytest.fixture
def coding_scenario():
    """Coding task scenario."""
    return E2EScenario(
        name="coding_task",
        description="Agent reviews and tests code",
        initial_message="Review and test the file src/main.py",
        expected_tool_sequence=["read_file", "lint_code", "run_tests"],
        expected_final_keywords=["tests", "passed"],
        tool_responses={
            "read_file": "def main():\n    print('Hello World')",
            "lint_code": "No errors found",
            "run_tests": "All tests passed",
        },
    )


# ============================================================================
# Mock Agent Workflow
# ============================================================================

class MockWorkflowAgent:
    """Mock agent that executes a predefined workflow."""

    def __init__(
        self,
        llm_provider: MockLLMProvider,
        tool_registry: ToolRegistry,
    ):
        self.llm = llm_provider
        self.tools = tool_registry
        self.conversation: List[Message] = []
        self.tool_calls_made: List[str] = []

    async def run(self, user_message: str) -> str:
        """Run the agent workflow."""
        self.conversation.append(
            Message(role=MessageRole.USER, content=user_message)
        )

        max_iterations = 10
        for _ in range(max_iterations):
            response = await self.llm.generate(
                self.conversation,
                tools=self.tools.to_definitions(),
            )

            if response.has_tool_calls:
                for tc in response.tool_calls:
                    self.tool_calls_made.append(tc.name)
                    tool_obj = self.tools.get(tc.name)
                    if tool_obj:
                        result = await tool_obj.run(**tc.arguments)
                        self.conversation.append(
                            Message(
                                role=MessageRole.TOOL,
                                content=str(result.output),
                                tool_call_id=tc.id,
                            )
                        )
            else:
                return response.content

        return "Max iterations reached"


@pytest.fixture
def mock_workflow_agent(mock_llm_provider, research_tools):
    """Create a mock workflow agent."""
    return MockWorkflowAgent(mock_llm_provider, research_tools)


# ============================================================================
# Memory Fixtures for E2E
# ============================================================================

@pytest.fixture
def persistent_memory_dir(tmp_path):
    """Directory for persistent memory in E2E tests."""
    memory_dir = tmp_path / "e2e_memory"
    memory_dir.mkdir()
    return memory_dir


@pytest.fixture
def multi_session_dir(tmp_path):
    """Directory for multi-session E2E tests."""
    session_dir = tmp_path / "e2e_sessions"
    session_dir.mkdir()
    return session_dir
