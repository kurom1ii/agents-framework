"""Tests for the skills base module.

Tests cover:
- BaseSkill class
- FunctionSkill class
- @skill decorator
- SkillContext
- SkillResult
- SkillMetadata
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import pytest

# Mark all async tests in this module to use pytest-asyncio
pytestmark = pytest.mark.asyncio

from agents_framework.skills.base import (
    BaseSkill,
    FunctionSkill,
    SkillCategory,
    SkillContext,
    SkillMetadata,
    SkillResult,
    skill,
    skill_method,
)


# ============================================================================
# SkillCategory Tests
# ============================================================================


class TestSkillCategory:
    """Tests for SkillCategory enum."""

    def test_category_values(self):
        """Test that all expected categories exist."""
        assert SkillCategory.GENERAL == "general"
        assert SkillCategory.TEXT == "text"
        assert SkillCategory.DATA == "data"
        assert SkillCategory.SEARCH == "search"
        assert SkillCategory.PLANNING == "planning"
        assert SkillCategory.CODE == "code"
        assert SkillCategory.COMMUNICATION == "communication"
        assert SkillCategory.REASONING == "reasoning"

    def test_category_is_string_enum(self):
        """Test that categories are string-based."""
        assert isinstance(SkillCategory.GENERAL, str)
        assert SkillCategory.TEXT.value == "text"


# ============================================================================
# SkillMetadata Tests
# ============================================================================


class TestSkillMetadata:
    """Tests for SkillMetadata dataclass."""

    def test_metadata_creation_minimal(self):
        """Test creating metadata with minimal required fields."""
        metadata = SkillMetadata(
            name="test_skill",
            description="A test skill",
        )

        assert metadata.name == "test_skill"
        assert metadata.description == "A test skill"
        assert metadata.version == "1.0.0"
        assert metadata.author == ""
        assert metadata.category == SkillCategory.GENERAL
        assert metadata.tags == []
        assert metadata.requires_llm is False
        assert metadata.timeout == 60.0

    def test_metadata_creation_full(self):
        """Test creating metadata with all fields."""
        metadata = SkillMetadata(
            name="full_skill",
            description="A fully configured skill",
            version="2.0.0",
            author="Test Author",
            category=SkillCategory.TEXT,
            tags=["text", "processing"],
            requires_llm=True,
            timeout=120.0,
        )

        assert metadata.name == "full_skill"
        assert metadata.description == "A fully configured skill"
        assert metadata.version == "2.0.0"
        assert metadata.author == "Test Author"
        assert metadata.category == SkillCategory.TEXT
        assert metadata.tags == ["text", "processing"]
        assert metadata.requires_llm is True
        assert metadata.timeout == 120.0


# ============================================================================
# SkillContext Tests
# ============================================================================


class TestSkillContext:
    """Tests for SkillContext dataclass."""

    def test_context_creation_empty(self):
        """Test creating an empty context."""
        context = SkillContext()

        assert context.agent_id is None
        assert context.task_id is None
        assert context.session_id is None
        assert context.llm is None
        assert context.memory is None
        assert context.tools is None
        assert context.config == {}

    def test_context_creation_with_values(self):
        """Test creating context with values."""
        context = SkillContext(
            agent_id="agent-123",
            task_id="task-456",
            session_id="session-789",
            config={"key1": "value1", "key2": 42},
        )

        assert context.agent_id == "agent-123"
        assert context.task_id == "task-456"
        assert context.session_id == "session-789"
        assert context.config == {"key1": "value1", "key2": 42}

    def test_context_get_config(self):
        """Test getting configuration values."""
        context = SkillContext(
            config={"existing_key": "existing_value"},
        )

        assert context.get_config("existing_key") == "existing_value"
        assert context.get_config("missing_key") is None
        assert context.get_config("missing_key", "default") == "default"


# ============================================================================
# SkillResult Tests
# ============================================================================


class TestSkillResult:
    """Tests for SkillResult dataclass."""

    def test_result_success(self):
        """Test creating a success result."""
        result = SkillResult(
            success=True,
            output="Test output",
            execution_time=0.5,
        )

        assert result.success is True
        assert result.output == "Test output"
        assert result.error is None
        assert result.metadata == {}
        assert result.execution_time == 0.5

    def test_result_failure(self):
        """Test creating a failure result."""
        result = SkillResult(
            success=False,
            error="Something went wrong",
            execution_time=0.1,
        )

        assert result.success is False
        assert result.output is None
        assert result.error == "Something went wrong"

    def test_result_with_metadata(self):
        """Test result with metadata."""
        result = SkillResult(
            success=True,
            output="data",
            metadata={"source": "test", "count": 5},
        )

        assert result.metadata == {"source": "test", "count": 5}


# ============================================================================
# BaseSkill Tests
# ============================================================================


class TestBaseSkill:
    """Tests for BaseSkill abstract class."""

    def test_skill_init_with_metadata(self, sample_skill_metadata: SkillMetadata):
        """Test initializing skill with metadata."""
        from .conftest import SimpleTestSkill

        skill = SimpleTestSkill()

        assert skill.name == "simple_test"
        assert skill.description == "A simple test skill"
        assert skill.metadata.category == SkillCategory.GENERAL

    def test_skill_init_with_name_override(self):
        """Test skill with name/description override."""

        class OverrideSkill(BaseSkill):
            """Original description."""

            def __init__(self):
                super().__init__(name="overridden_name", description="Overridden description")

            async def execute(self, context: SkillContext, **kwargs: Any) -> str:
                return "result"

        skill = OverrideSkill()

        assert skill.name == "overridden_name"
        assert skill.description == "Overridden description"

    def test_skill_name_property(self, simple_test_skill):
        """Test skill name property."""
        assert simple_test_skill.name == "simple_test"

    def test_skill_description_property(self, simple_test_skill):
        """Test skill description property."""
        assert simple_test_skill.description == "A simple test skill"

    async def test_skill_execute(self, simple_test_skill, skill_context):
        """Test direct skill execution."""
        result = await simple_test_skill.execute(skill_context, greeting_name="Alice")
        assert result == "Hello, Alice!"

    async def test_skill_execute_default_args(self, simple_test_skill, skill_context):
        """Test skill execution with default arguments."""
        result = await simple_test_skill.execute(skill_context)
        assert result == "Hello, World!"

    async def test_skill_run_success(self, simple_test_skill, skill_context):
        """Test skill run method returns SkillResult on success."""
        result = await simple_test_skill.run(skill_context, name="Bob")

        assert isinstance(result, SkillResult)
        assert result.success is True
        assert result.output == "Hello, Bob!"
        assert result.error is None
        assert result.execution_time > 0

    async def test_skill_run_without_context(self, simple_test_skill):
        """Test skill run with no context provided."""
        result = await simple_test_skill.run(name="Charlie")

        assert result.success is True
        assert result.output == "Hello, Charlie!"

    async def test_skill_run_with_error(self, failing_skill, skill_context):
        """Test skill run method catches errors."""
        result = await failing_skill.run(skill_context)

        assert result.success is False
        assert "Skill failed" in result.error
        assert result.execution_time > 0

    async def test_skill_run_llm_required_without_llm(self, llm_required_skill, skill_context):
        """Test LLM-required skill fails without LLM provider."""
        result = await llm_required_skill.run(skill_context, prompt="test")

        assert result.success is False
        assert "requires an LLM provider" in result.error

    async def test_skill_run_llm_required_with_llm(self, llm_required_skill, skill_context_with_llm):
        """Test LLM-required skill succeeds with LLM provider."""
        result = await llm_required_skill.run(skill_context_with_llm, prompt="test prompt")

        assert result.success is True
        assert result.output is not None

    async def test_skill_run_timeout(self, slow_skill, skill_context):
        """Test skill run times out correctly."""
        result = await slow_skill.run(skill_context)

        assert result.success is False
        assert "timed out" in result.error

    def test_skill_get_parameters_schema(self, simple_test_skill):
        """Test getting skill parameters schema."""
        schema = simple_test_skill.get_parameters_schema()

        assert isinstance(schema, dict)
        assert "type" in schema or "properties" in schema

    def test_skill_repr(self, simple_test_skill):
        """Test skill string representation."""
        repr_str = repr(simple_test_skill)

        assert "SimpleTestSkill" in repr_str
        assert "simple_test" in repr_str


# ============================================================================
# FunctionSkill Tests
# ============================================================================


class TestFunctionSkill:
    """Tests for FunctionSkill class."""

    def test_function_skill_from_sync_function(self):
        """Test creating FunctionSkill from sync function."""

        def greet(name: str) -> str:
            """Greet a person by name."""
            return f"Hello, {name}!"

        skill = FunctionSkill(greet)

        assert skill.name == "greet"
        assert "Greet a person" in skill.description
        assert skill._is_async is False

    def test_function_skill_from_async_function(self):
        """Test creating FunctionSkill from async function."""

        async def greet_async(name: str) -> str:
            """Greet a person asynchronously."""
            return f"Hello, {name}!"

        skill = FunctionSkill(greet_async)

        assert skill.name == "greet_async"
        assert skill._is_async is True

    def test_function_skill_with_name_override(self):
        """Test FunctionSkill with name override."""

        def my_func(x: int) -> int:
            """Original description."""
            return x * 2

        skill = FunctionSkill(my_func, name="custom_name", description="Custom description")

        assert skill.name == "custom_name"
        assert skill.description == "Custom description"

    def test_function_skill_with_category(self):
        """Test FunctionSkill with category."""

        def text_func(text: str) -> str:
            """Process text."""
            return text.upper()

        skill = FunctionSkill(text_func, category=SkillCategory.TEXT)

        assert skill.metadata.category == SkillCategory.TEXT

    def test_function_skill_with_tags(self):
        """Test FunctionSkill with tags."""

        def tagged_func() -> str:
            """Tagged function."""
            return "result"

        skill = FunctionSkill(tagged_func, tags=["tag1", "tag2"])

        assert skill.metadata.tags == ["tag1", "tag2"]

    def test_function_skill_requires_llm(self):
        """Test FunctionSkill with LLM requirement."""

        def llm_func() -> str:
            """Function requiring LLM."""
            return "result"

        skill = FunctionSkill(llm_func, requires_llm=True)

        assert skill.metadata.requires_llm is True

    async def test_function_skill_execute_sync(self, skill_context):
        """Test executing sync function skill."""

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        skill = FunctionSkill(add)
        result = await skill.execute(skill_context, a=3, b=4)

        assert result == 7

    async def test_function_skill_execute_async(self, skill_context):
        """Test executing async function skill."""

        async def async_add(a: int, b: int) -> int:
            """Add two numbers asynchronously."""
            return a + b

        skill = FunctionSkill(async_add)
        result = await skill.execute(skill_context, a=5, b=6)

        assert result == 11

    def test_function_skill_with_context_param(self):
        """Test FunctionSkill detects context parameter."""

        def func_with_context(context: SkillContext, value: int) -> int:
            """Function that uses context."""
            return value * 2

        skill = FunctionSkill(func_with_context)

        assert skill._has_context is True

    async def test_function_skill_passes_context(self, skill_context):
        """Test that context is passed to function when expected."""

        def func_with_context(context: SkillContext, value: int) -> str:
            """Function that uses context."""
            return f"Agent: {context.agent_id}, Value: {value}"

        skill = FunctionSkill(func_with_context)
        result = await skill.execute(skill_context, value=42)

        assert "Agent: test-agent" in result
        assert "Value: 42" in result

    def test_function_skill_get_parameters_schema(self):
        """Test getting parameters schema from function skill."""

        def typed_func(name: str, age: int, active: bool = True) -> str:
            """A typed function."""
            return f"{name}, {age}, {active}"

        skill = FunctionSkill(typed_func)
        schema = skill.get_parameters_schema()

        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]


# ============================================================================
# @skill Decorator Tests
# ============================================================================


class TestSkillDecorator:
    """Tests for @skill decorator."""

    def test_skill_decorator_basic(self):
        """Test basic skill decorator usage."""

        @skill()
        def simple_skill(text: str) -> str:
            """A simple skill function."""
            return text.upper()

        assert isinstance(simple_skill, FunctionSkill)
        assert simple_skill.name == "simple_skill"

    def test_skill_decorator_with_name(self):
        """Test skill decorator with custom name."""

        @skill(name="custom_skill")
        def my_function(x: int) -> int:
            """My function."""
            return x * 2

        assert my_function.name == "custom_skill"

    def test_skill_decorator_with_description(self):
        """Test skill decorator with custom description."""

        @skill(description="Custom description here")
        def another_func() -> str:
            """Original docstring."""
            return "result"

        assert another_func.description == "Custom description here"

    def test_skill_decorator_with_category(self):
        """Test skill decorator with category."""

        @skill(category=SkillCategory.DATA)
        def data_skill(data: str) -> str:
            """Process data."""
            return data

        assert data_skill.metadata.category == SkillCategory.DATA

    def test_skill_decorator_with_tags(self):
        """Test skill decorator with tags."""

        @skill(tags=["ml", "prediction"])
        def ml_skill(features: list) -> float:
            """ML prediction skill."""
            return 0.95

        assert "ml" in ml_skill.metadata.tags
        assert "prediction" in ml_skill.metadata.tags

    def test_skill_decorator_requires_llm(self):
        """Test skill decorator with LLM requirement."""

        @skill(requires_llm=True)
        def llm_skill(prompt: str) -> str:
            """LLM-based skill."""
            return "response"

        assert llm_skill.metadata.requires_llm is True

    async def test_skill_decorator_async_function(self, skill_context):
        """Test skill decorator on async function."""

        @skill(name="async_greet")
        async def greet(name: str) -> str:
            """Greet someone asynchronously."""
            await asyncio.sleep(0.001)
            return f"Hello, {name}!"

        result = await greet.execute(skill_context, name="World")
        assert result == "Hello, World!"

    async def test_skill_decorator_run_method(self, skill_context):
        """Test that decorated skill has run method."""

        @skill()
        def echo(message: str) -> str:
            """Echo a message."""
            return message

        result = await echo.run(skill_context, message="test")

        assert result.success is True
        assert result.output == "test"


# ============================================================================
# @skill_method Decorator Tests
# ============================================================================


class TestSkillMethodDecorator:
    """Tests for @skill_method decorator."""

    def test_skill_method_marks_method(self):
        """Test that skill_method decorator marks methods correctly."""

        class TestClass:
            @skill_method(name="my_method", description="My method description")
            def do_something(self) -> str:
                """Original docstring."""
                return "done"

        assert hasattr(TestClass.do_something, "_is_skill_method")
        assert TestClass.do_something._is_skill_method is True
        assert TestClass.do_something._skill_name == "my_method"
        assert TestClass.do_something._skill_description == "My method description"

    def test_skill_method_defaults_to_function_name(self):
        """Test skill_method uses function name by default."""

        class TestClass:
            @skill_method()
            def another_method(self) -> str:
                """Another method docstring."""
                return "result"

        assert TestClass.another_method._skill_name == "another_method"
        assert TestClass.another_method._skill_description == "Another method docstring."

    def test_skill_method_callable(self):
        """Test that skill_method decorated methods are still callable."""

        class TestClass:
            @skill_method()
            def callable_method(self, x: int) -> int:
                """Multiply by 2."""
                return x * 2

        obj = TestClass()
        result = obj.callable_method(5)
        assert result == 10


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestSkillEdgeCases:
    """Tests for edge cases and error handling."""

    def test_skill_with_empty_docstring(self):
        """Test skill creation with no docstring."""

        class NoDocSkill(BaseSkill):
            def __init__(self):
                super().__init__(name="no_doc")

            async def execute(self, context: SkillContext, **kwargs: Any) -> str:
                return "result"

        skill = NoDocSkill()
        assert skill.name == "no_doc"
        assert skill.description == ""  # Empty string when no docstring

    def test_skill_metadata_modification(self, simple_test_skill):
        """Test that metadata can be modified after creation."""
        simple_test_skill.metadata.tags.append("new_tag")
        assert "new_tag" in simple_test_skill.metadata.tags

    async def test_skill_with_complex_return_type(self, skill_context):
        """Test skill returning complex data types."""

        @skill()
        def complex_skill() -> dict:
            """Return complex data."""
            return {"nested": {"data": [1, 2, 3]}, "flag": True}

        result = await complex_skill.run(skill_context)

        assert result.success is True
        assert result.output == {"nested": {"data": [1, 2, 3]}, "flag": True}

    async def test_skill_with_optional_params(self, skill_context):
        """Test skill with optional parameters."""

        @skill()
        def optional_skill(
            required: str,
            optional: Optional[str] = None,
            default: str = "default_value",
        ) -> str:
            """Skill with optional parameters."""
            return f"{required}-{optional}-{default}"

        result = await optional_skill.execute(skill_context, required="test")
        assert result == "test-None-default_value"

    async def test_skill_execution_time_tracking(self, skill_context):
        """Test that execution time is tracked."""

        @skill()
        async def timed_skill() -> str:
            """A skill that takes some time."""
            await asyncio.sleep(0.01)
            return "done"

        result = await timed_skill.run(skill_context)

        assert result.success is True
        assert result.execution_time >= 0.01

    def test_skill_category_assignment(self):
        """Test various category assignments."""

        @skill(category=SkillCategory.CODE)
        def code_skill() -> str:
            """Code skill."""
            return ""

        @skill(category=SkillCategory.REASONING)
        def reasoning_skill() -> str:
            """Reasoning skill."""
            return ""

        assert code_skill.metadata.category == SkillCategory.CODE
        assert reasoning_skill.metadata.category == SkillCategory.REASONING
