"""Base skill classes and decorators for the agents framework.

This module provides the skill system foundation:
- BaseSkill abstract class
- @skill decorator for creating skills from functions
- Skill context for execution
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

from agents_framework.tools.schema import generate_function_schema


class SkillCategory(str, Enum):
    """Categories for organizing skills."""

    GENERAL = "general"
    TEXT = "text"
    DATA = "data"
    SEARCH = "search"
    PLANNING = "planning"
    CODE = "code"
    COMMUNICATION = "communication"
    REASONING = "reasoning"


@dataclass
class SkillMetadata:
    """Metadata for a skill.

    Attributes:
        name: Skill name.
        description: Skill description.
        version: Skill version.
        author: Skill author.
        category: Skill category.
        tags: List of tags for discovery.
        requires_llm: Whether the skill requires an LLM.
        timeout: Default execution timeout in seconds.
    """

    name: str
    description: str
    version: str = "1.0.0"
    author: str = ""
    category: SkillCategory = SkillCategory.GENERAL
    tags: List[str] = field(default_factory=list)
    requires_llm: bool = False
    timeout: float = 60.0


@dataclass
class SkillContext:
    """Context passed to skill execution.

    Contains information about the execution environment
    and access to shared resources.

    Attributes:
        agent_id: ID of the executing agent.
        task_id: ID of the current task.
        session_id: ID of the current session.
        llm: Optional LLM provider for skills that need it.
        memory: Optional memory store.
        tools: Optional tool registry.
        config: Additional configuration.
    """

    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    llm: Optional[Any] = None  # LLMProvider
    memory: Optional[Any] = None  # MemoryStore
    tools: Optional[Any] = None  # ToolRegistry
    config: Dict[str, Any] = field(default_factory=dict)

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Configuration key.
            default: Default value if key not found.

        Returns:
            Configuration value.
        """
        return self.config.get(key, default)


@dataclass
class SkillResult:
    """Result of a skill execution.

    Attributes:
        success: Whether execution succeeded.
        output: The skill output.
        error: Error message if failed.
        metadata: Additional metadata about the execution.
        execution_time: Time taken to execute in seconds.
    """

    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0


class BaseSkill(ABC):
    """Abstract base class for skills.

    Skills are higher-level abstractions than tools, representing
    reusable capabilities that may combine multiple operations,
    use LLMs, or implement complex logic.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[SkillMetadata] = None,
    ):
        """Initialize the skill.

        Args:
            name: Optional name override.
            description: Optional description override.
            metadata: Optional metadata for the skill.
        """
        self.metadata = metadata or SkillMetadata(
            name=name or self.__class__.__name__,
            description=description or self.__doc__ or "",
        )

        if name:
            self.metadata.name = name
        if description:
            self.metadata.description = description

    @property
    def name(self) -> str:
        """Get skill name."""
        return self.metadata.name

    @property
    def description(self) -> str:
        """Get skill description."""
        return self.metadata.description

    @abstractmethod
    async def execute(
        self,
        context: SkillContext,
        **kwargs: Any,
    ) -> Any:
        """Execute the skill.

        Args:
            context: Execution context.
            **kwargs: Skill-specific arguments.

        Returns:
            The skill result.
        """
        pass

    async def run(
        self,
        context: Optional[SkillContext] = None,
        **kwargs: Any,
    ) -> SkillResult:
        """Run the skill with error handling and timing.

        Args:
            context: Optional execution context.
            **kwargs: Skill-specific arguments.

        Returns:
            SkillResult with output or error.
        """
        import time

        ctx = context or SkillContext()
        start_time = time.perf_counter()

        try:
            # Check LLM requirement
            if self.metadata.requires_llm and ctx.llm is None:
                return SkillResult(
                    success=False,
                    error="This skill requires an LLM provider",
                )

            result = await asyncio.wait_for(
                self.execute(ctx, **kwargs),
                timeout=self.metadata.timeout,
            )

            execution_time = time.perf_counter() - start_time
            return SkillResult(
                success=True,
                output=result,
                execution_time=execution_time,
            )

        except asyncio.TimeoutError:
            execution_time = time.perf_counter() - start_time
            return SkillResult(
                success=False,
                error=f"Skill timed out after {self.metadata.timeout}s",
                execution_time=execution_time,
            )
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return SkillResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
            )

    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get JSON Schema for skill parameters.

        Returns:
            JSON Schema dictionary.
        """
        return generate_function_schema(self.execute).get(
            "parameters",
            {"type": "object", "properties": {}},
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# Type variable for function skill
F = TypeVar("F", bound=Callable[..., Any])


class FunctionSkill(BaseSkill):
    """Skill wrapper for regular functions."""

    def __init__(
        self,
        func: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: SkillCategory = SkillCategory.GENERAL,
        requires_llm: bool = False,
        tags: Optional[List[str]] = None,
    ):
        """Initialize the function skill.

        Args:
            func: The function to wrap.
            name: Optional name override.
            description: Optional description override.
            category: Skill category.
            requires_llm: Whether skill requires LLM.
            tags: Optional tags for discovery.
        """
        self._func = func
        self._is_async = asyncio.iscoroutinefunction(func)
        self._has_context = self._check_context_param()

        # Generate schema from function
        schema = generate_function_schema(
            func,
            description=description,
            name=name,
        )

        metadata = SkillMetadata(
            name=schema["name"],
            description=schema["description"],
            category=category,
            requires_llm=requires_llm,
            tags=tags or [],
        )

        super().__init__(metadata=metadata)
        self._parameters = schema["parameters"]

    def _check_context_param(self) -> bool:
        """Check if function accepts context parameter."""
        sig = inspect.signature(self._func)
        for param in sig.parameters.values():
            if param.name == "context":
                return True
            if param.annotation is SkillContext:
                return True
        return False

    async def execute(
        self,
        context: SkillContext,
        **kwargs: Any,
    ) -> Any:
        """Execute the wrapped function."""
        # Add context if function expects it
        if self._has_context:
            kwargs["context"] = context

        if self._is_async:
            return await self._func(**kwargs)
        else:
            return self._func(**kwargs)

    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get JSON Schema for skill parameters."""
        return self._parameters


def skill(
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: SkillCategory = SkillCategory.GENERAL,
    requires_llm: bool = False,
    tags: Optional[List[str]] = None,
) -> Callable[[F], FunctionSkill]:
    """Decorator to convert a function into a skill.

    Args:
        name: Optional skill name. Defaults to function name.
        description: Optional description. Defaults to function docstring.
        category: Skill category.
        requires_llm: Whether skill requires an LLM.
        tags: Optional tags for discovery.

    Returns:
        A FunctionSkill instance wrapping the function.

    Example:
        @skill(name="summarize", category=SkillCategory.TEXT)
        async def summarize_text(text: str, max_length: int = 100) -> str:
            '''Summarize the given text.

            Args:
                text: Text to summarize.
                max_length: Maximum length of summary.

            Returns:
                Summarized text.
            '''
            # Implementation
            return summary
    """
    def decorator(func: F) -> FunctionSkill:
        return FunctionSkill(
            func,
            name=name,
            description=description,
            category=category,
            requires_llm=requires_llm,
            tags=tags,
        )

    return decorator


def skill_method(
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[[F], F]:
    """Decorator to mark a method as a skill entry point.

    Use this decorator on methods of a class that inherits from BaseSkill
    to provide additional metadata or to create multiple skills from
    a single class.

    Args:
        name: Optional name override.
        description: Optional description override.

    Returns:
        The decorated method with skill metadata.
    """
    def decorator(func: F) -> F:
        func._skill_name = name or func.__name__  # type: ignore
        func._skill_description = description or func.__doc__  # type: ignore
        func._is_skill_method = True  # type: ignore
        return func

    return decorator
