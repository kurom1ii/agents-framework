"""Tests for built-in skills (summarize, search, plan).

Tests cover:
- SummarizeSkill
- SearchSkill
- PlanSkill
- Plan and PlanStep data structures
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mark all async tests in this module to use pytest-asyncio
pytestmark = pytest.mark.asyncio

from agents_framework.llm.base import LLMConfig, LLMResponse
from agents_framework.skills.base import SkillCategory, SkillContext

from .conftest import MockLLMProvider


# ============================================================================
# Summarize Skill Tests
# ============================================================================


class TestSummarizeSkill:
    """Tests for SummarizeSkill."""

    @pytest.fixture
    def summarize_skill(self):
        """Create a SummarizeSkill instance."""
        from agents_framework.skills.builtin.summarize import SummarizeSkill

        return SummarizeSkill()

    def test_summarize_skill_metadata(self, summarize_skill):
        """Test summarize skill metadata."""
        assert summarize_skill.name == "summarize"
        assert "summarize" in summarize_skill.description.lower()
        assert summarize_skill.metadata.category == SkillCategory.TEXT
        assert summarize_skill.metadata.requires_llm is True
        assert "summarization" in summarize_skill.metadata.tags

    async def test_summarize_skill_requires_llm(
        self, summarize_skill, skill_context: SkillContext
    ):
        """Test that summarize skill fails without LLM."""
        result = await summarize_skill.run(skill_context, text="Some text to summarize")

        assert result.success is False
        assert "LLM" in result.error or "requires" in result.error

    async def test_summarize_skill_with_llm(
        self, summarize_skill, sample_llm_config: LLMConfig
    ):
        """Test summarize skill with LLM provider."""
        mock_llm = MockLLMProvider(
            sample_llm_config,
            responses=[
                LLMResponse(
                    content="This is a concise summary.",
                    model="test-model",
                    finish_reason="stop",
                )
            ],
        )
        context = SkillContext(llm=mock_llm)

        result = await summarize_skill.run(
            context,
            text="This is a long text that needs to be summarized. It contains many details and should be condensed.",
        )

        assert result.success is True
        assert result.output == "This is a concise summary."
        assert mock_llm.call_count == 1

    async def test_summarize_skill_with_options(
        self, summarize_skill, sample_llm_config: LLMConfig
    ):
        """Test summarize skill with various options."""
        mock_llm = MockLLMProvider(
            sample_llm_config,
            responses=[
                LLMResponse(
                    content="- Point 1\n- Point 2\n- Point 3",
                    model="test-model",
                    finish_reason="stop",
                )
            ],
        )
        context = SkillContext(llm=mock_llm)

        result = await summarize_skill.run(
            context,
            text="Long text here",
            max_length=50,
            style="bullet_points",
            focus="key findings",
            language="English",
        )

        assert result.success is True
        # Verify the prompt was built with the options
        last_messages = mock_llm.last_messages
        user_message = last_messages[-1].content
        assert "50" in user_message  # max_length
        assert "bullet" in user_message.lower()  # style
        assert "key findings" in user_message  # focus
        assert "English" in user_message  # language

    def test_summarize_skill_build_prompt(self, summarize_skill):
        """Test prompt building with different styles."""
        # Brief style
        prompt = summarize_skill._build_prompt(
            text="Test text",
            max_length=100,
            style="brief",
            focus=None,
            language=None,
        )
        assert "brief" in prompt.lower() or "concise" in prompt.lower()

        # Detailed style
        prompt = summarize_skill._build_prompt(
            text="Test text",
            max_length=100,
            style="detailed",
            focus=None,
            language=None,
        )
        assert "detailed" in prompt.lower()

        # Bullet points style
        prompt = summarize_skill._build_prompt(
            text="Test text",
            max_length=100,
            style="bullet_points",
            focus=None,
            language=None,
        )
        assert "bullet" in prompt.lower()

    async def test_summarize_multiple(
        self, summarize_skill, sample_llm_config: LLMConfig
    ):
        """Test summarizing multiple texts."""
        mock_llm = MockLLMProvider(
            sample_llm_config,
            responses=[
                LLMResponse(content=f"Summary {i}", model="test", finish_reason="stop")
                for i in range(3)
            ],
        )
        context = SkillContext(llm=mock_llm)

        texts = ["Text 1", "Text 2", "Text 3"]
        summaries = await summarize_skill.summarize_multiple(context, texts)

        assert len(summaries) == 3
        assert mock_llm.call_count == 3


# ============================================================================
# Search Skill Tests
# ============================================================================


class TestSearchSkill:
    """Tests for SearchSkill."""

    @pytest.fixture
    def search_skill(self):
        """Create a SearchSkill instance."""
        from agents_framework.skills.builtin.search import SearchSkill

        return SearchSkill()

    def test_search_skill_metadata(self, search_skill):
        """Test search skill metadata."""
        assert search_skill.name == "search"
        assert "search" in search_skill.description.lower()
        assert search_skill.metadata.category == SkillCategory.SEARCH
        assert search_skill.metadata.requires_llm is False
        assert "search" in search_skill.metadata.tags

    async def test_search_skill_no_sources(self, search_skill, skill_context: SkillContext):
        """Test search skill with no sources available."""
        result = await search_skill.run(skill_context, query="test query")

        assert result.success is True
        assert result.output == []  # Empty list when no sources

    async def test_search_skill_with_tools(
        self, search_skill, skill_context_with_tools: SkillContext
    ):
        """Test search skill searching through tools."""
        result = await search_skill.run(
            skill_context_with_tools,
            query="search",
            source="tools",
        )

        assert result.success is True
        results = result.output
        assert len(results) > 0
        # Should find the search_web tool
        assert any("search" in r["content"].lower() for r in results)

    async def test_search_skill_with_tools_partial_match(
        self, search_skill, skill_context_with_tools: SkillContext
    ):
        """Test search skill with partial query matches."""
        result = await search_skill.run(
            skill_context_with_tools,
            query="calculate",
            source="tools",
            max_results=5,
        )

        assert result.success is True
        results = result.output
        assert len(results) > 0
        assert any("calculate" in r["content"].lower() for r in results)

    async def test_search_skill_max_results(
        self, search_skill, skill_context_with_tools: SkillContext
    ):
        """Test search skill respects max_results."""
        result = await search_skill.run(
            skill_context_with_tools,
            query="",  # Empty query to match nothing specific
            source="tools",
            max_results=1,
        )

        assert result.success is True
        assert len(result.output) <= 1

    async def test_search_skill_min_score_filter(
        self, search_skill, skill_context_with_tools: SkillContext
    ):
        """Test search skill filters by minimum score."""
        result = await search_skill.run(
            skill_context_with_tools,
            query="xyz_nonexistent",  # Query that won't match well
            source="tools",
            min_score=0.95,  # High threshold
        )

        assert result.success is True
        # High min_score should filter out low-scoring results
        for r in result.output:
            assert r["score"] >= 0.95

    async def test_search_skill_all_sources(
        self, search_skill, skill_context_with_tools: SkillContext
    ):
        """Test search skill searching all sources."""
        result = await search_skill.run(
            skill_context_with_tools,
            query="search",
            source="all",
        )

        assert result.success is True

    async def test_search_with_llm_synthesis(self, search_skill, sample_llm_config: LLMConfig):
        """Test search_with_llm method."""
        mock_llm = MockLLMProvider(
            sample_llm_config,
            responses=[
                LLMResponse(
                    content="Based on the search results, here is the answer.",
                    model="test-model",
                    finish_reason="stop",
                )
            ],
        )
        context = SkillContext(llm=mock_llm)

        search_results = [
            {"content": "Result 1", "source": "memory", "score": 0.9, "metadata": {}},
            {"content": "Result 2", "source": "tools", "score": 0.8, "metadata": {}},
        ]

        answer = await search_skill.search_with_llm(
            context,
            query="What is the answer?",
            search_results=search_results,
        )

        assert "Based on the search results" in answer
        assert mock_llm.call_count == 1

    async def test_search_with_llm_no_provider(self, search_skill, skill_context: SkillContext):
        """Test search_with_llm fails without LLM."""
        with pytest.raises(ValueError, match="LLM provider"):
            await search_skill.search_with_llm(
                skill_context,
                query="test",
                search_results=[],
            )

    def test_search_source_enum(self):
        """Test SearchSource enum values."""
        from agents_framework.skills.builtin.search import SearchSource

        assert SearchSource.MEMORY == "memory"
        assert SearchSource.TOOLS == "tools"
        assert SearchSource.WEB == "web"
        assert SearchSource.ALL == "all"

    def test_search_result_dataclass(self):
        """Test SearchResult dataclass."""
        from agents_framework.skills.builtin.search import SearchResult

        result = SearchResult(
            content="Test content",
            source="memory",
            score=0.85,
            metadata={"key": "value"},
        )

        assert result.content == "Test content"
        assert result.source == "memory"
        assert result.score == 0.85
        assert result.metadata == {"key": "value"}


# ============================================================================
# Plan Skill Tests
# ============================================================================


class TestPlanSkill:
    """Tests for PlanSkill."""

    @pytest.fixture
    def plan_skill(self):
        """Create a PlanSkill instance."""
        from agents_framework.skills.builtin.plan import PlanSkill

        return PlanSkill()

    def test_plan_skill_metadata(self, plan_skill):
        """Test plan skill metadata."""
        assert plan_skill.name == "plan"
        assert "plan" in plan_skill.description.lower()
        assert plan_skill.metadata.category == SkillCategory.PLANNING
        assert plan_skill.metadata.requires_llm is True
        assert "planning" in plan_skill.metadata.tags

    async def test_plan_skill_requires_llm(self, plan_skill, skill_context: SkillContext):
        """Test that plan skill fails without LLM."""
        result = await plan_skill.run(skill_context, goal="Create a project plan")

        assert result.success is False
        assert "LLM" in result.error or "requires" in result.error

    async def test_plan_skill_creates_plan(self, plan_skill, sample_llm_config: LLMConfig):
        """Test plan skill creates a plan."""
        mock_llm = MockLLMProvider(
            sample_llm_config,
            responses=[
                LLMResponse(
                    content="""1. Research the topic (action: search)
2. Gather requirements (action: ask_user) - after step 1
3. Create outline (action: write) - after step 2
4. Review and finalize""",
                    model="test-model",
                    finish_reason="stop",
                )
            ],
        )
        context = SkillContext(llm=mock_llm)

        result = await plan_skill.run(context, goal="Write a blog post")

        assert result.success is True
        plan = result.output
        assert "goal" in plan
        assert plan["goal"] == "Write a blog post"
        assert "steps" in plan
        assert len(plan["steps"]) >= 1

    async def test_plan_skill_with_max_steps(self, plan_skill, sample_llm_config: LLMConfig):
        """Test plan skill respects max_steps in prompt."""
        mock_llm = MockLLMProvider(
            sample_llm_config,
            responses=[
                LLMResponse(
                    content="1. Step one\n2. Step two\n3. Step three",
                    model="test-model",
                    finish_reason="stop",
                )
            ],
        )
        context = SkillContext(llm=mock_llm)

        await plan_skill.run(context, goal="Simple goal", max_steps=3)

        # Verify max_steps was included in prompt
        user_message = mock_llm.last_messages[-1].content
        assert "3" in user_message

    async def test_plan_skill_with_available_tools(
        self, plan_skill, sample_llm_config: LLMConfig
    ):
        """Test plan skill uses available tools."""
        mock_llm = MockLLMProvider(
            sample_llm_config,
            responses=[
                LLMResponse(
                    content="1. Search for info (action: search)\n2. Calculate result (action: calculate)",
                    model="test-model",
                    finish_reason="stop",
                )
            ],
        )
        context = SkillContext(llm=mock_llm)

        await plan_skill.run(
            context,
            goal="Research and calculate",
            available_tools=["search", "calculate"],
        )

        user_message = mock_llm.last_messages[-1].content
        assert "search" in user_message
        assert "calculate" in user_message

    async def test_plan_skill_include_actions(
        self, plan_skill, sample_llm_config: LLMConfig
    ):
        """Test plan skill action inclusion option."""
        mock_llm = MockLLMProvider(
            sample_llm_config,
            responses=[
                LLMResponse(
                    content="1. Do something",
                    model="test-model",
                    finish_reason="stop",
                )
            ],
        )
        context = SkillContext(llm=mock_llm)

        await plan_skill.run(context, goal="Goal", include_actions=False)

        # When include_actions is False and no tools, should not ask for tool suggestions
        assert mock_llm.call_count == 1

    def test_plan_skill_parse_step_line(self, plan_skill):
        """Test parsing individual step lines."""
        # Standard numbered step
        step = plan_skill._parse_step_line("1. Do something important", 1)
        assert step is not None
        assert step.description == "Do something important"
        assert step.id == "step_1"

        # Step with action
        step = plan_skill._parse_step_line("2. Search for info (action: search)", 2)
        assert step is not None
        assert step.action == "search"

        # Step with dependencies
        step = plan_skill._parse_step_line("3. Analyze results - after step 2", 3)
        assert step is not None
        assert "step_2" in step.dependencies

        # Empty line
        step = plan_skill._parse_step_line("", 1)
        assert step is None

    def test_plan_skill_parse_plan_response(self, plan_skill):
        """Test parsing complete plan response."""
        response = """1. Research the topic (action: search)
2. Analyze findings - after step 1
3. Write summary (action: write) - after step 2
4. Review and finalize"""

        plan = plan_skill._parse_plan_response("Test goal", response)

        assert plan.goal == "Test goal"
        assert len(plan.steps) == 4
        assert plan.steps[0].action == "search"
        assert "step_1" in plan.steps[1].dependencies

    async def test_plan_skill_refine_plan(self, plan_skill, sample_llm_config: LLMConfig):
        """Test refining an existing plan."""
        mock_llm = MockLLMProvider(
            sample_llm_config,
            responses=[
                LLMResponse(
                    content="1. New refined step one\n2. New refined step two",
                    model="test-model",
                    finish_reason="stop",
                )
            ],
        )
        context = SkillContext(llm=mock_llm)

        original_plan = {
            "goal": "Original goal",
            "steps": [
                {"id": "step_1", "description": "Step one", "action": None}
            ],
            "status": "pending",
        }

        refined = await plan_skill.refine_plan(
            context,
            plan=original_plan,
            feedback="Add more detail",
        )

        assert refined["goal"] == "Original goal"
        assert len(refined["steps"]) >= 1

    async def test_plan_skill_refine_plan_requires_llm(
        self, plan_skill, skill_context: SkillContext
    ):
        """Test that refine_plan fails without LLM."""
        with pytest.raises(ValueError, match="LLM provider"):
            await plan_skill.refine_plan(
                skill_context,
                plan={"goal": "test", "steps": []},
                feedback="Add more steps",
            )


# ============================================================================
# Plan Data Structures Tests
# ============================================================================


class TestPlanDataStructures:
    """Tests for Plan and PlanStep data structures."""

    def test_plan_status_enum(self):
        """Test PlanStatus enum values."""
        from agents_framework.skills.builtin.plan import PlanStatus

        assert PlanStatus.PENDING == "pending"
        assert PlanStatus.IN_PROGRESS == "in_progress"
        assert PlanStatus.COMPLETED == "completed"
        assert PlanStatus.FAILED == "failed"
        assert PlanStatus.SKIPPED == "skipped"

    def test_plan_step_creation(self):
        """Test PlanStep dataclass creation."""
        from agents_framework.skills.builtin.plan import PlanStatus, PlanStep

        step = PlanStep(
            id="step_1",
            description="Do something",
            action="search",
            dependencies=["step_0"],
        )

        assert step.id == "step_1"
        assert step.description == "Do something"
        assert step.action == "search"
        assert step.dependencies == ["step_0"]
        assert step.status == PlanStatus.PENDING
        assert step.result is None

    def test_plan_step_to_dict(self):
        """Test PlanStep to_dict method."""
        from agents_framework.skills.builtin.plan import PlanStep

        step = PlanStep(
            id="step_1",
            description="Test step",
            action="test_action",
        )

        step_dict = step.to_dict()

        assert step_dict["id"] == "step_1"
        assert step_dict["description"] == "Test step"
        assert step_dict["action"] == "test_action"
        assert step_dict["status"] == "pending"

    def test_plan_creation(self):
        """Test Plan dataclass creation."""
        from agents_framework.skills.builtin.plan import Plan, PlanStatus, PlanStep

        steps = [
            PlanStep(id="step_1", description="First step"),
            PlanStep(id="step_2", description="Second step", dependencies=["step_1"]),
        ]

        plan = Plan(goal="Achieve something", steps=steps)

        assert plan.goal == "Achieve something"
        assert len(plan.steps) == 2
        assert plan.status == PlanStatus.PENDING

    def test_plan_to_dict(self):
        """Test Plan to_dict method."""
        from agents_framework.skills.builtin.plan import Plan, PlanStep

        plan = Plan(
            goal="Test goal",
            steps=[PlanStep(id="step_1", description="Step 1")],
            metadata={"key": "value"},
        )

        plan_dict = plan.to_dict()

        assert plan_dict["goal"] == "Test goal"
        assert len(plan_dict["steps"]) == 1
        assert plan_dict["status"] == "pending"
        assert plan_dict["metadata"] == {"key": "value"}

    def test_plan_get_next_steps_no_dependencies(self):
        """Test getting next steps when no dependencies."""
        from agents_framework.skills.builtin.plan import Plan, PlanStep

        plan = Plan(
            goal="Test",
            steps=[
                PlanStep(id="step_1", description="Step 1"),
                PlanStep(id="step_2", description="Step 2"),
            ],
        )

        next_steps = plan.get_next_steps()

        # Both steps should be ready since no dependencies
        assert len(next_steps) == 2

    def test_plan_get_next_steps_with_dependencies(self):
        """Test getting next steps respects dependencies."""
        from agents_framework.skills.builtin.plan import Plan, PlanStatus, PlanStep

        plan = Plan(
            goal="Test",
            steps=[
                PlanStep(id="step_1", description="Step 1"),
                PlanStep(id="step_2", description="Step 2", dependencies=["step_1"]),
                PlanStep(id="step_3", description="Step 3", dependencies=["step_2"]),
            ],
        )

        # Initially only step_1 should be ready
        next_steps = plan.get_next_steps()
        assert len(next_steps) == 1
        assert next_steps[0].id == "step_1"

        # Complete step_1
        plan.mark_step_complete("step_1")

        # Now step_2 should be ready
        next_steps = plan.get_next_steps()
        assert len(next_steps) == 1
        assert next_steps[0].id == "step_2"

    def test_plan_is_complete(self):
        """Test checking if plan is complete."""
        from agents_framework.skills.builtin.plan import Plan, PlanStatus, PlanStep

        plan = Plan(
            goal="Test",
            steps=[
                PlanStep(id="step_1", description="Step 1"),
                PlanStep(id="step_2", description="Step 2"),
            ],
        )

        assert plan.is_complete() is False

        plan.mark_step_complete("step_1")
        assert plan.is_complete() is False

        plan.mark_step_complete("step_2")
        assert plan.is_complete() is True

    def test_plan_is_complete_with_skipped(self):
        """Test is_complete considers skipped steps as complete."""
        from agents_framework.skills.builtin.plan import Plan, PlanStatus, PlanStep

        plan = Plan(
            goal="Test",
            steps=[
                PlanStep(id="step_1", description="Step 1", status=PlanStatus.COMPLETED),
                PlanStep(id="step_2", description="Step 2", status=PlanStatus.SKIPPED),
            ],
        )

        assert plan.is_complete() is True

    def test_plan_mark_step_complete(self):
        """Test marking a step complete."""
        from agents_framework.skills.builtin.plan import Plan, PlanStatus, PlanStep

        plan = Plan(
            goal="Test",
            steps=[PlanStep(id="step_1", description="Step 1")],
        )

        plan.mark_step_complete("step_1", result="Success!")

        assert plan.steps[0].status == PlanStatus.COMPLETED
        assert plan.steps[0].result == "Success!"
        assert plan.status == PlanStatus.COMPLETED

    def test_plan_mark_step_failed(self):
        """Test marking a step as failed."""
        from agents_framework.skills.builtin.plan import Plan, PlanStatus, PlanStep

        plan = Plan(
            goal="Test",
            steps=[PlanStep(id="step_1", description="Step 1")],
        )

        plan.mark_step_failed("step_1", error="Something went wrong")

        assert plan.steps[0].status == PlanStatus.FAILED
        assert plan.steps[0].result == {"error": "Something went wrong"}
        assert plan.status == PlanStatus.FAILED


# ============================================================================
# Integration Tests
# ============================================================================


class TestBuiltinSkillsIntegration:
    """Integration tests for built-in skills."""

    async def test_search_then_summarize(self, sample_llm_config: LLMConfig):
        """Test using search results with summarize."""
        from agents_framework.skills.builtin.search import SearchSkill
        from agents_framework.skills.builtin.summarize import SummarizeSkill

        mock_llm = MockLLMProvider(
            sample_llm_config,
            responses=[
                LLMResponse(
                    content="Summary of the search results.",
                    model="test-model",
                    finish_reason="stop",
                )
            ],
        )

        # Create context with mock tools for search
        from .conftest import MockToolRegistry, SimpleMockTool

        tools = [SimpleMockTool("web_search", "Search the web")]
        context = SkillContext(
            llm=mock_llm,
            tools=MockToolRegistry(tools),
        )

        # Search for something
        search_skill = SearchSkill()
        search_result = await search_skill.run(
            context,
            query="search",
            source="tools",
        )

        assert search_result.success is True

        # Summarize the results
        summarize_skill = SummarizeSkill()
        results_text = "\n".join(
            r["content"] for r in search_result.output
        )

        if results_text:
            summary_result = await summarize_skill.run(
                context,
                text=results_text,
            )
            assert summary_result.success is True

    async def test_plan_with_tool_context(self, sample_llm_config: LLMConfig):
        """Test planning skill uses tools from context."""
        from agents_framework.skills.builtin.plan import PlanSkill

        from .conftest import MockToolRegistry, SimpleMockTool

        mock_llm = MockLLMProvider(
            sample_llm_config,
            responses=[
                LLMResponse(
                    content="1. Search (action: search)\n2. Analyze (action: analyze)",
                    model="test-model",
                    finish_reason="stop",
                )
            ],
        )

        tools = [
            SimpleMockTool("search", "Search for information"),
            SimpleMockTool("analyze", "Analyze data"),
        ]

        context = SkillContext(
            llm=mock_llm,
            tools=MockToolRegistry(tools),
        )

        plan_skill = PlanSkill()
        result = await plan_skill.run(context, goal="Research and analyze")

        assert result.success is True

        # Verify tools were included in prompt
        user_message = mock_llm.last_messages[-1].content
        assert "search" in user_message.lower()
        assert "analyze" in user_message.lower()


# ============================================================================
# Fixture for LLM config needed by tests
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
