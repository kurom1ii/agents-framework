"""Built-in skills for the agents framework.

This module provides a set of common skills that are useful
for many agent workflows:

- SummarizeSkill: Summarize text content
- SearchSkill: Search for information
- PlanSkill: Create execution plans
"""

from .summarize import SummarizeSkill
from .search import SearchSkill
from .plan import PlanSkill

__all__ = [
    "SummarizeSkill",
    "SearchSkill",
    "PlanSkill",
]


def register_builtin_skills() -> None:
    """Register all built-in skills with the default registry."""
    from agents_framework.skills import get_default_registry

    registry = get_default_registry()

    # Register built-in skills
    registry.register(SummarizeSkill())
    registry.register(SearchSkill())
    registry.register(PlanSkill())
