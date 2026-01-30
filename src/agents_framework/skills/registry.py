"""Skill registry for managing and discovering skills.

This module provides the skill registry and loader:
- SkillRegistry for managing skills
- Skill loader for loading from files/modules
- Skill executor with context management
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
    Union,
)

from .base import (
    BaseSkill,
    FunctionSkill,
    SkillCategory,
    SkillContext,
    SkillMetadata,
    SkillResult,
)


class SkillRegistry:
    """Registry for managing skills.

    Provides skill registration, lookup, and execution capabilities.
    Supports both class-based skills (BaseSkill) and function-based skills.
    """

    def __init__(self):
        """Initialize an empty skill registry."""
        self._skills: Dict[str, BaseSkill] = {}
        self._categories: Dict[SkillCategory, List[str]] = {
            cat: [] for cat in SkillCategory
        }
        self._tags: Dict[str, List[str]] = {}

    def register(
        self,
        skill: Union[BaseSkill, Callable[..., Any]],
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: SkillCategory = SkillCategory.GENERAL,
        tags: Optional[List[str]] = None,
    ) -> BaseSkill:
        """Register a skill in the registry.

        Args:
            skill: A BaseSkill instance or a callable to wrap.
            name: Optional name override.
            description: Optional description override.
            category: Skill category.
            tags: Optional tags for discovery.

        Returns:
            The registered BaseSkill instance.

        Raises:
            ValueError: If a skill with the same name already exists.
        """
        if isinstance(skill, BaseSkill):
            registered_skill = skill
            if name:
                registered_skill.metadata.name = name
            if description:
                registered_skill.metadata.description = description
        else:
            # Wrap callable as FunctionSkill
            registered_skill = FunctionSkill(
                skill,
                name=name,
                description=description,
                category=category,
                tags=tags,
            )

        skill_name = registered_skill.name

        if skill_name in self._skills:
            raise ValueError(
                f"Skill '{skill_name}' is already registered. "
                "Use a different name or unregister the existing skill first."
            )

        self._skills[skill_name] = registered_skill

        # Update category index
        cat = registered_skill.metadata.category
        if skill_name not in self._categories[cat]:
            self._categories[cat].append(skill_name)

        # Update tag index
        for tag in registered_skill.metadata.tags:
            if tag not in self._tags:
                self._tags[tag] = []
            if skill_name not in self._tags[tag]:
                self._tags[tag].append(skill_name)

        return registered_skill

    def register_all(
        self,
        skills: Iterable[Union[BaseSkill, Callable[..., Any]]],
    ) -> List[BaseSkill]:
        """Register multiple skills at once.

        Args:
            skills: An iterable of skills to register.

        Returns:
            List of registered BaseSkill instances.
        """
        return [self.register(skill) for skill in skills]

    def unregister(self, name: str) -> Optional[BaseSkill]:
        """Unregister a skill by name.

        Args:
            name: The name of the skill to unregister.

        Returns:
            The unregistered skill if found, None otherwise.
        """
        skill = self._skills.pop(name, None)
        if skill:
            # Remove from category index
            cat = skill.metadata.category
            if name in self._categories[cat]:
                self._categories[cat].remove(name)

            # Remove from tag index
            for tag in skill.metadata.tags:
                if tag in self._tags and name in self._tags[tag]:
                    self._tags[tag].remove(name)

        return skill

    def get(self, name: str) -> Optional[BaseSkill]:
        """Get a skill by name.

        Args:
            name: The name of the skill to retrieve.

        Returns:
            The skill if found, None otherwise.
        """
        return self._skills.get(name)

    def get_or_raise(self, name: str) -> BaseSkill:
        """Get a skill by name, raising an error if not found.

        Args:
            name: The name of the skill to retrieve.

        Returns:
            The skill.

        Raises:
            KeyError: If the skill is not found.
        """
        skill = self._skills.get(name)
        if skill is None:
            available = ", ".join(self._skills.keys()) or "none"
            raise KeyError(
                f"Skill '{name}' not found. Available skills: {available}"
            )
        return skill

    def has(self, name: str) -> bool:
        """Check if a skill is registered.

        Args:
            name: The name of the skill to check.

        Returns:
            True if the skill is registered, False otherwise.
        """
        return name in self._skills

    def list_skills(self) -> List[BaseSkill]:
        """List all registered skills.

        Returns:
            List of all registered skills.
        """
        return list(self._skills.values())

    def list_names(self) -> List[str]:
        """List all registered skill names.

        Returns:
            List of skill names.
        """
        return list(self._skills.keys())

    def list_by_category(self, category: SkillCategory) -> List[BaseSkill]:
        """List skills in a category.

        Args:
            category: The category to filter by.

        Returns:
            List of skills in the category.
        """
        names = self._categories.get(category, [])
        return [self._skills[name] for name in names if name in self._skills]

    def list_by_tag(self, tag: str) -> List[BaseSkill]:
        """List skills with a specific tag.

        Args:
            tag: The tag to filter by.

        Returns:
            List of skills with the tag.
        """
        names = self._tags.get(tag, [])
        return [self._skills[name] for name in names if name in self._skills]

    def search(
        self,
        query: str,
        category: Optional[SkillCategory] = None,
        tags: Optional[List[str]] = None,
    ) -> List[BaseSkill]:
        """Search for skills matching criteria.

        Args:
            query: Text to search in name and description.
            category: Optional category filter.
            tags: Optional tags filter (any match).

        Returns:
            List of matching skills.
        """
        results = []
        query_lower = query.lower()

        for skill in self._skills.values():
            # Check category
            if category and skill.metadata.category != category:
                continue

            # Check tags
            if tags and not any(t in skill.metadata.tags for t in tags):
                continue

            # Check query in name or description
            if query_lower in skill.name.lower() or \
               query_lower in skill.description.lower():
                results.append(skill)

        return results

    async def execute(
        self,
        name: str,
        context: Optional[SkillContext] = None,
        **kwargs: Any,
    ) -> SkillResult:
        """Execute a skill by name.

        Args:
            name: The name of the skill to execute.
            context: Optional execution context.
            **kwargs: Arguments to pass to the skill.

        Returns:
            SkillResult with success status and output or error.
        """
        skill = self.get(name)
        if skill is None:
            return SkillResult(
                success=False,
                error=f"Skill '{name}' not found",
            )
        return await skill.run(context, **kwargs)

    def clear(self) -> None:
        """Clear all registered skills."""
        self._skills.clear()
        self._categories = {cat: [] for cat in SkillCategory}
        self._tags.clear()

    def __len__(self) -> int:
        """Return the number of registered skills."""
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        """Check if a skill is registered."""
        return name in self._skills

    def __iter__(self):
        """Iterate over registered skills."""
        return iter(self._skills.values())

    def __repr__(self) -> str:
        return f"SkillRegistry(skills={list(self._skills.keys())})"


# Global default registry
_default_registry = SkillRegistry()


def get_default_registry() -> SkillRegistry:
    """Get the default global skill registry."""
    return _default_registry


def register_skill(
    skill: Union[BaseSkill, Callable[..., Any]],
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: SkillCategory = SkillCategory.GENERAL,
    tags: Optional[List[str]] = None,
) -> BaseSkill:
    """Register a skill in the default registry.

    Args:
        skill: A BaseSkill instance or a callable to wrap.
        name: Optional name override.
        description: Optional description override.
        category: Skill category.
        tags: Optional tags.

    Returns:
        The registered BaseSkill instance.
    """
    return _default_registry.register(
        skill,
        name=name,
        description=description,
        category=category,
        tags=tags,
    )


class SkillLoader:
    """Loader for discovering and loading skills from files and modules."""

    def __init__(self, registry: Optional[SkillRegistry] = None):
        """Initialize the skill loader.

        Args:
            registry: Optional registry to load skills into.
        """
        self.registry = registry or get_default_registry()

    def load_from_module(self, module_name: str) -> List[BaseSkill]:
        """Load skills from a Python module.

        Args:
            module_name: Fully qualified module name.

        Returns:
            List of loaded skills.
        """
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ImportError(f"Could not import module '{module_name}': {e}")

        return self._extract_skills_from_module(module)

    def load_from_file(self, file_path: Union[str, Path]) -> List[BaseSkill]:
        """Load skills from a Python file.

        Args:
            file_path: Path to the Python file.

        Returns:
            List of loaded skills.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.suffix == ".py":
            raise ValueError(f"Expected .py file, got: {path.suffix}")

        # Create a unique module name
        module_name = f"_skills_{path.stem}_{id(path)}"

        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for: {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
            return self._extract_skills_from_module(module)
        finally:
            # Clean up the module
            sys.modules.pop(module_name, None)

    def load_from_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
    ) -> List[BaseSkill]:
        """Load skills from all Python files in a directory.

        Args:
            directory: Path to the directory.
            recursive: Whether to search recursively.

        Returns:
            List of loaded skills.
        """
        path = Path(directory)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        if not path.is_dir():
            raise ValueError(f"Expected directory, got file: {path}")

        skills = []
        pattern = "**/*.py" if recursive else "*.py"

        for file_path in path.glob(pattern):
            if file_path.name.startswith("_"):
                continue
            try:
                loaded = self.load_from_file(file_path)
                skills.extend(loaded)
            except Exception as e:
                # Log but continue loading other files
                print(f"Error loading skills from {file_path}: {e}")

        return skills

    def _extract_skills_from_module(self, module: Any) -> List[BaseSkill]:
        """Extract skill instances from a module.

        Args:
            module: The Python module.

        Returns:
            List of skill instances.
        """
        skills = []

        for name in dir(module):
            if name.startswith("_"):
                continue

            obj = getattr(module, name)

            # Check if it's already a skill instance
            if isinstance(obj, BaseSkill):
                if not self.registry.has(obj.name):
                    self.registry.register(obj)
                    skills.append(obj)
                continue

            # Check if it's a skill class
            if isinstance(obj, type) and issubclass(obj, BaseSkill):
                if obj is BaseSkill or obj is FunctionSkill:
                    continue
                instance = obj()
                if not self.registry.has(instance.name):
                    self.registry.register(instance)
                    skills.append(instance)

        return skills


@dataclass
class SkillExecutorConfig:
    """Configuration for skill executor.

    Attributes:
        timeout: Default execution timeout in seconds.
        max_concurrent: Maximum concurrent skill executions.
        retry_on_error: Whether to retry failed executions.
        max_retries: Maximum number of retries.
    """

    timeout: float = 60.0
    max_concurrent: int = 5
    retry_on_error: bool = False
    max_retries: int = 3


class SkillExecutor:
    """Executor for running skills with context management.

    Provides a managed way to execute skills with:
    - Timeout handling
    - Concurrency limits
    - Context propagation
    - Retry logic
    """

    def __init__(
        self,
        registry: Optional[SkillRegistry] = None,
        config: Optional[SkillExecutorConfig] = None,
    ):
        """Initialize the executor.

        Args:
            registry: Skill registry to use.
            config: Execution configuration.
        """
        self.registry = registry or get_default_registry()
        self.config = config or SkillExecutorConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._default_context: Optional[SkillContext] = None

    def set_default_context(self, context: SkillContext) -> None:
        """Set the default context for all skill executions.

        Args:
            context: The default context to use.
        """
        self._default_context = context

    async def execute(
        self,
        skill_name: str,
        context: Optional[SkillContext] = None,
        **kwargs: Any,
    ) -> SkillResult:
        """Execute a skill by name.

        Args:
            skill_name: Name of the skill to execute.
            context: Optional execution context.
            **kwargs: Arguments to pass to the skill.

        Returns:
            SkillResult with the result and metadata.
        """
        import time

        skill = self.registry.get(skill_name)
        if skill is None:
            return SkillResult(
                success=False,
                error=f"Skill '{skill_name}' not found",
            )

        ctx = context or self._default_context or SkillContext()
        retries = 0
        start_time = time.perf_counter()

        async with self._semaphore:
            while True:
                try:
                    result = await skill.run(ctx, **kwargs)

                    if result.success or not self.config.retry_on_error:
                        return result

                    # Retry on error if configured
                    if retries < self.config.max_retries:
                        retries += 1
                        await asyncio.sleep(0.5 * retries)
                        continue

                    return result

                except Exception as e:
                    if self.config.retry_on_error and retries < self.config.max_retries:
                        retries += 1
                        await asyncio.sleep(0.5 * retries)
                        continue

                    execution_time = time.perf_counter() - start_time
                    return SkillResult(
                        success=False,
                        error=str(e),
                        execution_time=execution_time,
                    )

    async def execute_many(
        self,
        skill_calls: List[Dict[str, Any]],
        context: Optional[SkillContext] = None,
    ) -> List[SkillResult]:
        """Execute multiple skills concurrently.

        Args:
            skill_calls: List of skill call dicts with keys:
                - name: Skill name
                - kwargs: Dict of arguments
            context: Optional shared context.

        Returns:
            List of SkillResults in the same order as inputs.
        """
        tasks = [
            self.execute(
                skill_name=call["name"],
                context=context,
                **call.get("kwargs", {}),
            )
            for call in skill_calls
        ]

        return await asyncio.gather(*tasks)

    def register(self, skill: BaseSkill) -> BaseSkill:
        """Register a skill with the executor's registry.

        Args:
            skill: The skill to register.

        Returns:
            The registered skill.
        """
        return self.registry.register(skill)

    def get_skill(self, name: str) -> Optional[BaseSkill]:
        """Get a skill from the registry.

        Args:
            name: Skill name.

        Returns:
            The skill if found, None otherwise.
        """
        return self.registry.get(name)
