"""Integration tests - Test component interactions.

Kiểm tra các components hoạt động cùng nhau:
- Agent + Tools
- Agent + Memory
- Team + Agents
"""

import pytest
from agents_framework.llm.base import LLMConfig, Message, MessageRole, ToolCall
from agents_framework.tools.base import BaseTool, tool
from agents_framework.tools.registry import ToolRegistry
from agents_framework.tools.executor import ToolExecutor
from agents_framework.memory.base import MemoryItem, MemoryType
from agents_framework.memory.short_term import SessionMemory
from tests.conftest import MockLLMProvider


# ============================================================================
# Integration: Agent + Tools
# ============================================================================

class TestAgentToolsIntegration:
    """Test Agent và Tools hoạt động cùng nhau."""

    @pytest.mark.asyncio
    async def test_full_tool_execution_flow(self):
        """Test flow hoàn chỉnh: LLM request → Tool execute → Response."""
        # Setup
        @tool(name="add", description="Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        registry = ToolRegistry()
        registry.register(add)
        executor = ToolExecutor(registry)

        config = LLMConfig(model="test", api_key="key")
        tool_calls = [[ToolCall(id="1", name="add", arguments={"a": 5, "b": 3})]]
        provider = MockLLMProvider(
            config,
            responses=["", "Kết quả là 8"],
            tool_calls=tool_calls,
        )

        # Step 1: User request
        messages = [Message(role=MessageRole.USER, content="Tính 5 + 3")]

        # Step 2: LLM returns tool call
        response = await provider.generate(messages, tools=registry.to_definitions())
        assert response.has_tool_calls
        assert response.tool_calls[0].name == "add"

        # Step 3: Execute tool
        result = await executor.execute("add", a=5, b=3)
        assert result.success
        assert result.output == 8

        # Step 4: Get final response
        messages.append(Message(role=MessageRole.TOOL, content="8", tool_call_id="1"))
        final = await provider.generate(messages)
        assert "8" in final.content

    @pytest.mark.asyncio
    async def test_multiple_tools_in_registry(self):
        """Registry quản lý nhiều tools cùng lúc."""
        @tool(name="multiply", description="Multiply")
        def multiply(a: int, b: int) -> int:
            return a * b

        @tool(name="divide", description="Divide")
        def divide(a: int, b: int) -> float:
            return a / b

        registry = ToolRegistry()
        registry.register(multiply)
        registry.register(divide)

        executor = ToolExecutor(registry)

        # Execute both
        r1 = await executor.execute("multiply", a=6, b=7)
        r2 = await executor.execute("divide", a=20, b=4)

        assert r1.output == 42
        assert r2.output == 5.0


# ============================================================================
# Integration: Agent + Memory
# ============================================================================

class TestAgentMemoryIntegration:
    """Test Agent và Memory hoạt động cùng nhau."""

    @pytest.mark.asyncio
    async def test_conversation_stored_in_memory(self):
        """Conversation được lưu vào memory."""
        memory = SessionMemory(max_tokens=1000)

        # Add conversation messages
        memory.add_message("user", "Xin chào!")
        memory.add_message("assistant", "Chào bạn, tôi có thể giúp gì?")
        memory.add_message("user", "Hôm nay thời tiết thế nào?")

        # Check memory
        messages = memory.get_messages()
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert "Xin chào" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_memory_with_context_retrieval(self):
        """Memory có thể retrieve context cho agent."""
        memory = SessionMemory(max_tokens=500)

        # Store some context
        memory.add_message("user", "Tên tôi là Minh")
        memory.add_message("assistant", "Xin chào Minh!")
        memory.add_message("user", "Tôi thích Python")

        # Get context for new request
        context = memory.get_context_string()
        assert "Minh" in context
        assert "Python" in context


# ============================================================================
# Integration: Team + Agents
# ============================================================================

class TestTeamAgentsIntegration:
    """Test Team orchestration với multiple agents."""

    @pytest.mark.asyncio
    async def test_message_routing_between_agents(self):
        """Messages được route đúng giữa các agents."""
        from agents_framework.teams.router import MessageRouter, AgentMessage

        router = MessageRouter()
        received = {"agent1": [], "agent2": []}

        async def handler1(msg):
            received["agent1"].append(msg)

        async def handler2(msg):
            received["agent2"].append(msg)

        router.register_agent("agent1", handler1)
        router.register_agent("agent2", handler2)

        # Send to agent1
        msg1 = AgentMessage(sender_id="supervisor", receiver_id="agent1", content="Task 1")
        await router.route(msg1)

        # Send to agent2
        msg2 = AgentMessage(sender_id="supervisor", receiver_id="agent2", content="Task 2")
        await router.route(msg2)

        assert len(received["agent1"]) == 1
        assert len(received["agent2"]) == 1
        assert received["agent1"][0].content == "Task 1"

    @pytest.mark.asyncio
    async def test_agent_registry_with_roles(self):
        """Agent registry quản lý agents theo role."""
        from agents_framework.teams.registry import AgentRegistry

        registry = AgentRegistry()

        class FakeAgent:
            def __init__(self, name):
                self.name = name

        # Register với roles
        registry.register(FakeAgent("r1"), agent_id="r1", role="researcher")
        registry.register(FakeAgent("r2"), agent_id="r2", role="researcher")
        registry.register(FakeAgent("w1"), agent_id="w1", role="writer")

        # Find by role
        researchers = registry.get_by_role("researcher")
        writers = registry.get_by_role("writer")

        assert len(researchers) == 2
        assert len(writers) == 1


# ============================================================================
# Integration: MCP Client
# ============================================================================

class TestMCPIntegration:
    """Test MCP client integration with mock server."""

    @pytest.mark.asyncio
    async def test_mcp_client_tool_discovery(self, mock_mcp_client):
        """MCP client discovers tools from server."""
        await mock_mcp_client.connect()

        tools = mock_mcp_client.list_tools()
        assert len(tools) == 3
        tool_names = [t.name for t in tools]
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "search" in tool_names

        await mock_mcp_client.disconnect()

    @pytest.mark.asyncio
    async def test_mcp_client_tool_execution(self, mock_mcp_client):
        """MCP client executes tools correctly."""
        await mock_mcp_client.connect()

        result = await mock_mcp_client.call_tool("read_file", {"path": "/tmp/test.txt"})
        assert "Content of file" in result
        assert "/tmp/test.txt" in result

        # Verify call history
        assert len(mock_mcp_client.tool_call_history) == 1
        assert mock_mcp_client.tool_call_history[0]["name"] == "read_file"

        await mock_mcp_client.disconnect()

    @pytest.mark.asyncio
    async def test_mcp_client_with_custom_handler(self, mock_mcp_client):
        """MCP client with custom tool handler."""
        await mock_mcp_client.connect()

        # Register custom handler
        mock_mcp_client.register_tool_handler(
            "search",
            lambda query: f"Found 5 results for: {query}"
        )

        result = await mock_mcp_client.call_tool("search", {"query": "Python async"})
        assert "Found 5 results" in result
        assert "Python async" in result

        await mock_mcp_client.disconnect()

    @pytest.mark.asyncio
    async def test_mcp_client_not_connected_error(self, mock_mcp_client):
        """MCP client raises error when not connected."""
        with pytest.raises(RuntimeError, match="Not connected"):
            await mock_mcp_client.call_tool("read_file", {"path": "/tmp/test.txt"})


# ============================================================================
# Integration: Sessions
# ============================================================================

class TestSessionIntegration:
    """Test session management integration."""

    @pytest.mark.asyncio
    async def test_session_store_persistence(self, temp_session_dir):
        """Session store persists sessions to disk."""
        from agents_framework.sessions import FileSessionStore, Session

        store = FileSessionStore(base_path=temp_session_dir)

        # Create and save session
        session = Session(
            session_id="test-123",
            session_key="agent:test:main",
            agent_id="test-agent",
        )
        await store.save(session)

        # Load session
        loaded = await store.load("agent:test:main")
        assert loaded is not None
        assert loaded.session_id == "test-123"
        assert loaded.agent_id == "test-agent"

    @pytest.mark.asyncio
    async def test_session_manager_lifecycle(self, temp_session_dir):
        """Session manager handles session lifecycle."""
        from agents_framework.sessions import (
            SessionManager,
            SessionConfig,
            FileSessionStore,
        )

        store = FileSessionStore(base_path=temp_session_dir)
        config = SessionConfig(agent_id="lifecycle-test")
        manager = SessionManager(store=store, config=config)

        # Get or create session
        session = await manager.get_or_create("user:123")
        assert session is not None
        assert session.agent_id == "lifecycle-test"

        # Session should be retrievable
        same_session = await manager.get_or_create("user:123")
        assert same_session.session_id == session.session_id

    @pytest.mark.asyncio
    async def test_session_transcript_storage(self, temp_session_dir):
        """Transcript storage works with sessions."""
        from agents_framework.sessions import (
            FileTranscriptStore,
            TranscriptEntry,
        )

        store = FileTranscriptStore(base_path=temp_session_dir)
        session_key = "agent:test:transcript"

        # Add entries
        entry1 = TranscriptEntry(role="user", content="Hello")
        entry2 = TranscriptEntry(role="assistant", content="Hi there!")

        await store.append(session_key, entry1)
        await store.append(session_key, entry2)

        # Load entries
        entries = await store.load(session_key)
        assert len(entries) == 2
        assert entries[0].role == "user"
        assert entries[1].role == "assistant"

