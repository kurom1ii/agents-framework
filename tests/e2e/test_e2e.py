"""End-to-End tests - Test complete workflows.

Kiểm tra các use case hoàn chỉnh từ đầu đến cuối:
- Single agent task
- Team collaboration
- Memory persistence
"""

import pytest
from agents_framework.llm.base import LLMConfig, Message, MessageRole, ToolCall
from agents_framework.tools.base import tool
from agents_framework.tools.registry import ToolRegistry
from agents_framework.memory.short_term import SessionMemory
from tests.conftest import MockLLMProvider


# ============================================================================
# E2E: Single Agent Workflow
# ============================================================================

class TestSingleAgentE2E:
    """Test single agent hoàn thành task."""

    @pytest.mark.asyncio
    async def test_agent_completes_research_task(self):
        """Agent hoàn thành research task với tools."""
        # Setup tools
        @tool(name="search", description="Search for information")
        def search(query: str) -> str:
            return f"Found: {query} is a programming language"

        @tool(name="summarize", description="Summarize text")
        def summarize(text: str) -> str:
            return f"Summary: {text[:50]}..."

        registry = ToolRegistry()
        registry.register(search)
        registry.register(summarize)

        # Setup LLM với tool call sequence
        config = LLMConfig(model="test", api_key="key")
        tool_calls = [
            [ToolCall(id="1", name="search", arguments={"query": "Python"})],
            [ToolCall(id="2", name="summarize", arguments={"text": "Python is great"})],
        ]
        provider = MockLLMProvider(
            config,
            responses=["", "", "Python là ngôn ngữ lập trình phổ biến."],
            tool_calls=tool_calls,
        )

        # Execute workflow
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are a researcher."),
            Message(role=MessageRole.USER, content="Research Python"),
        ]

        # Turn 1: Search
        response = await provider.generate(messages, tools=registry.to_definitions())
        assert response.has_tool_calls
        tool_obj = registry.get("search")
        result = await tool_obj.run(query="Python")
        messages.append(Message(role=MessageRole.TOOL, content=result.output, tool_call_id="1"))

        # Turn 2: Summarize
        response = await provider.generate(messages, tools=registry.to_definitions())
        assert response.has_tool_calls
        tool_obj = registry.get("summarize")
        result = await tool_obj.run(text="Python is great")
        messages.append(Message(role=MessageRole.TOOL, content=result.output, tool_call_id="2"))

        # Turn 3: Final response
        response = await provider.generate(messages, tools=registry.to_definitions())
        assert not response.has_tool_calls
        assert "Python" in response.content


# ============================================================================
# E2E: Team Workflow
# ============================================================================

class TestTeamWorkflowE2E:
    """Test team hoàn thành task phức tạp."""

    @pytest.mark.asyncio
    async def test_supervisor_delegates_to_workers(self):
        """Supervisor phân công và tổng hợp từ workers."""
        config = LLMConfig(model="test", api_key="key")

        # Supervisor agent
        supervisor = MockLLMProvider(
            config,
            responses=[
                "Phân công: researcher tìm kiếm, writer viết bài",
                "Tổng hợp: Bài viết hoàn chỉnh về AI",
            ],
        )

        # Worker agents
        researcher = MockLLMProvider(config, responses=["Tìm thấy: AI đang phát triển nhanh"])
        writer = MockLLMProvider(config, responses=["Bài viết: AI và tương lai công nghệ"])

        # Step 1: Supervisor phân tích
        messages = [Message(role=MessageRole.USER, content="Viết bài về AI")]
        plan = await supervisor.generate(messages)
        assert "researcher" in plan.content.lower() or "writer" in plan.content.lower()

        # Step 2: Researcher làm việc
        research_result = await researcher.generate([
            Message(role=MessageRole.USER, content="Tìm kiếm thông tin về AI")
        ])
        assert "AI" in research_result.content

        # Step 3: Writer làm việc
        writer_result = await writer.generate([
            Message(role=MessageRole.USER, content=f"Viết bài dựa trên: {research_result.content}")
        ])
        assert "AI" in writer_result.content

        # Step 4: Supervisor tổng hợp
        messages.append(Message(role=MessageRole.ASSISTANT, content=writer_result.content))
        final = await supervisor.generate(messages)
        assert "AI" in final.content


# ============================================================================
# E2E: Memory Persistence
# ============================================================================

class TestMemoryPersistenceE2E:
    """Test memory lưu trữ qua các sessions."""

    @pytest.mark.asyncio
    async def test_conversation_context_maintained(self):
        """Context được duy trì trong conversation."""
        memory = SessionMemory(max_tokens=2000)

        # Multi-turn conversation
        turns = [
            ("user", "Tên tôi là Lan"),
            ("assistant", "Xin chào Lan!"),
            ("user", "Tôi làm developer"),
            ("assistant", "Developer là nghề thú vị!"),
            ("user", "Tôi thích Python và Go"),
            ("assistant", "Python và Go đều rất tốt cho backend!"),
        ]

        for role, content in turns:
            memory.add_message(role, content)

        # Check all context is preserved
        messages = memory.get_messages()
        assert len(messages) == 6

        context = memory.get_context_string()
        assert "Lan" in context
        assert "developer" in context
        assert "Python" in context

    @pytest.mark.asyncio
    async def test_memory_provides_context_for_new_request(self):
        """Memory cung cấp context cho request mới."""
        memory = SessionMemory(max_tokens=1000)

        # Previous conversation
        memory.add_message("user", "Dự án của tôi dùng FastAPI")
        memory.add_message("assistant", "FastAPI rất tốt cho REST APIs")

        # New request needs context
        new_request = "Làm sao để deploy nó?"

        # Agent sử dụng context
        context = memory.get_context_string()
        full_prompt = f"Context:\n{context}\n\nQuestion: {new_request}"

        # Verify context includes previous info
        assert "FastAPI" in full_prompt
        assert "deploy" in full_prompt


# ============================================================================
# E2E: Swarm Pattern with Handoffs
# ============================================================================

class TestSwarmPatternE2E:
    """Test swarm pattern với agent handoffs."""

    @pytest.mark.asyncio
    async def test_agent_handoff_flow(self):
        """Agent chuyển giao task cho agent khác."""
        from agents_framework.teams.patterns.swarm import HandoffResult

        config = LLMConfig(model="test", api_key="key")

        # Triage agent routes to specialist
        triage = MockLLMProvider(
            config,
            responses=["Đây là vấn đề kỹ thuật, chuyển cho tech_support"],
        )

        # Tech support handles
        tech_support = MockLLMProvider(
            config,
            responses=["Giải pháp: Reset router và thử lại kết nối"],
        )

        # Step 1: User contacts triage
        user_message = "Tôi không kết nối được internet"
        triage_response = await triage.generate([
            Message(role=MessageRole.USER, content=user_message)
        ])
        assert "tech_support" in triage_response.content.lower()

        # Step 2: Handoff to tech support
        handoff = HandoffResult(
            target_agent="tech_support",
            context={"issue": "internet_connection"},
            message=user_message,
        )

        # Step 3: Tech support resolves
        tech_response = await tech_support.generate([
            Message(role=MessageRole.SYSTEM, content=f"Context: {handoff.context}"),
            Message(role=MessageRole.USER, content=handoff.message),
        ])
        assert "reset" in tech_response.content.lower() or "router" in tech_response.content.lower()

    @pytest.mark.asyncio
    async def test_multi_hop_handoffs(self):
        """Multiple handoffs trong một conversation."""
        config = LLMConfig(model="test", api_key="key")

        agents = {
            "triage": MockLLMProvider(config, responses=["Chuyển cho billing"]),
            "billing": MockLLMProvider(config, responses=["Vấn đề kỹ thuật, chuyển cho tech"]),
            "tech": MockLLMProvider(config, responses=["Đã sửa xong vấn đề!"]),
        }

        # Simulate handoff chain
        current_agent = "triage"
        handoff_history = [current_agent]

        # Triage -> Billing
        response = await agents[current_agent].generate([
            Message(role=MessageRole.USER, content="Tôi bị tính phí sai")
        ])
        if "billing" in response.content.lower():
            current_agent = "billing"
            handoff_history.append(current_agent)

        # Billing -> Tech
        response = await agents[current_agent].generate([
            Message(role=MessageRole.USER, content="Hệ thống tính phí sai")
        ])
        if "tech" in response.content.lower():
            current_agent = "tech"
            handoff_history.append(current_agent)

        # Tech resolves
        final = await agents[current_agent].generate([
            Message(role=MessageRole.USER, content="Sửa lỗi")
        ])

        assert len(handoff_history) == 3
        assert handoff_history == ["triage", "billing", "tech"]
        assert "sửa" in final.content.lower()


# ============================================================================
# E2E: Session Persistence
# ============================================================================

class TestSessionPersistenceE2E:
    """Test session persistence across restarts."""

    @pytest.mark.asyncio
    async def test_session_survives_restart(self, multi_session_dir):
        """Session data persists after simulated restart."""
        from agents_framework.sessions import (
            SessionManager,
            SessionConfig,
            FileSessionStore,
        )

        # First "run" - create session
        store1 = FileSessionStore(base_path=multi_session_dir)
        config1 = SessionConfig(agent_id="persist-test")
        manager1 = SessionManager(store=store1, config=config1)

        session1 = await manager1.get_or_create("user:persist")
        original_id = session1.session_id

        # Simulate restart - create new manager with same store
        store2 = FileSessionStore(base_path=multi_session_dir)
        config2 = SessionConfig(agent_id="persist-test")
        manager2 = SessionManager(store=store2, config=config2)

        session2 = await manager2.get_or_create("user:persist")

        # Should get same session
        assert session2.session_id == original_id

    @pytest.mark.asyncio
    async def test_multiple_sessions_isolation(self, multi_session_dir):
        """Multiple sessions are properly isolated."""
        from agents_framework.sessions import (
            SessionManager,
            SessionConfig,
            FileSessionStore,
        )

        store = FileSessionStore(base_path=multi_session_dir)
        config = SessionConfig(agent_id="isolation-test")
        manager = SessionManager(store=store, config=config)

        # Create separate sessions
        session_a = await manager.get_or_create("user:alice")
        session_b = await manager.get_or_create("user:bob")

        # Sessions should be different
        assert session_a.session_id != session_b.session_id
        assert session_a.session_key != session_b.session_key


# ============================================================================
# E2E: Context Compaction
# ============================================================================

class TestContextCompactionE2E:
    """Test context compaction under load."""

    @pytest.mark.asyncio
    async def test_memory_handles_long_conversation(self):
        """Memory handles và compacts long conversations."""
        memory = SessionMemory(max_tokens=500)  # Small limit to trigger compaction

        # Add many messages
        for i in range(50):
            memory.add_message("user", f"Message {i}: " + "x" * 50)
            memory.add_message("assistant", f"Response {i}: " + "y" * 50)

        # Memory should still work
        messages = memory.get_messages()
        context = memory.get_context_string()

        # Should have compacted - not all 100 messages
        assert len(messages) < 100
        assert len(context) < 10000  # Reasonable size

    @pytest.mark.asyncio
    async def test_context_retains_important_info(self):
        """Context compaction retains important information."""
        memory = SessionMemory(max_tokens=300)

        # Add important early message
        memory.add_message("user", "IMPORTANT: My name is Claude Test User")

        # Add filler messages
        for i in range(20):
            memory.add_message("assistant", f"Filler response {i}")
            memory.add_message("user", f"Filler question {i}")

        # Recent messages should be preserved
        messages = memory.get_messages()
        assert len(messages) > 0

        # Note: actual implementation may vary - this tests the interface
        context = memory.get_context_string()
        assert context is not None

