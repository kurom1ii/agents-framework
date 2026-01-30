"""Skills package for the agents framework.

This package provides the skill system for defining, registering, and
executing higher-level capabilities that agents can use.

Skills differ from tools in that they:
- May combine multiple operations
- Can use LLMs for complex reasoning
- Support execution context with memory and tools
- Have richer metadata and categorization

Example:
    # Using the @skill decorator
    from agents_framework.skills import skill, SkillCategory

    @skill(name="analyze", category=SkillCategory.REASONING)
    async def analyze_data(data: str, context: SkillContext) -> str:
        '''Analyze the given data.

        Args:
            data: Data to analyze.
            context: Execution context.

        Returns:
            Analysis results.
        '''
        # Use LLM from context if available
        if context.llm:
            # ... LLM-based analysis
            pass
        return results

    # Using class-based skills
    from agents_framework.skills import BaseSkill, SkillContext

    class MySkill(BaseSkill):
        name = "my_skill"
        description = "Does something useful"

        async def execute(self, context: SkillContext, **kwargs) -> Any:
            return "result"

    # Using the registry
    from agents_framework.skills import SkillRegistry

    registry = SkillRegistry()
    registry.register(analyze_data)
    registry.register(MySkill())

    # Execute skills
    result = await registry.execute("analyze", context=ctx, data="...")
"""

from .base import (
    BaseSkill,
    FunctionSkill,
    SkillCategory,
    SkillContext,
    SkillMetadata,
    SkillResult,
    skill,
    skill_method,
)
from .registry import (
    SkillExecutor,
    SkillExecutorConfig,
    SkillLoader,
    SkillRegistry,
    get_default_registry,
    register_skill,
)

__all__ = [
    # Base classes
    "BaseSkill",
    "FunctionSkill",
    "SkillCategory",
    "SkillContext",
    "SkillMetadata",
    "SkillResult",
    # Decorators
    "skill",
    "skill_method",
    # Registry
    "SkillExecutor",
    "SkillExecutorConfig",
    "SkillLoader",
    "SkillRegistry",
    "get_default_registry",
    "register_skill",
]
