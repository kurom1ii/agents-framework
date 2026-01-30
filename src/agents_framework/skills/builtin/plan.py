"""Plan skill for creating execution plans.

This skill provides planning capabilities for breaking down
complex tasks into actionable steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from agents_framework.skills.base import (
    BaseSkill,
    SkillCategory,
    SkillContext,
    SkillMetadata,
)


class PlanStatus(str, Enum):
    """Status of a plan or step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """A single step in a plan.

    Attributes:
        id: Unique step identifier.
        description: Description of what to do.
        action: Optional action/tool to use.
        dependencies: IDs of steps that must complete first.
        status: Current status of the step.
        result: Result after execution.
    """

    id: str
    description: str
    action: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    status: PlanStatus = PlanStatus.PENDING
    result: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "action": self.action,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "result": self.result,
        }


@dataclass
class Plan:
    """A complete execution plan.

    Attributes:
        goal: The overall goal of the plan.
        steps: List of steps to achieve the goal.
        status: Overall plan status.
        metadata: Additional metadata.
    """

    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    status: PlanStatus = PlanStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "goal": self.goal,
            "steps": [step.to_dict() for step in self.steps],
            "status": self.status.value,
            "metadata": self.metadata,
        }

    def get_next_steps(self) -> List[PlanStep]:
        """Get steps that are ready to execute.

        Returns:
            List of steps whose dependencies are complete.
        """
        completed_ids = {
            step.id for step in self.steps
            if step.status == PlanStatus.COMPLETED
        }

        ready = []
        for step in self.steps:
            if step.status != PlanStatus.PENDING:
                continue
            if all(dep in completed_ids for dep in step.dependencies):
                ready.append(step)

        return ready

    def is_complete(self) -> bool:
        """Check if all steps are complete."""
        return all(
            step.status in (PlanStatus.COMPLETED, PlanStatus.SKIPPED)
            for step in self.steps
        )

    def mark_step_complete(
        self,
        step_id: str,
        result: Optional[Any] = None,
    ) -> None:
        """Mark a step as complete.

        Args:
            step_id: ID of the step to mark.
            result: Optional result of the step.
        """
        for step in self.steps:
            if step.id == step_id:
                step.status = PlanStatus.COMPLETED
                step.result = result
                break

        # Update overall status
        if self.is_complete():
            self.status = PlanStatus.COMPLETED

    def mark_step_failed(
        self,
        step_id: str,
        error: Optional[str] = None,
    ) -> None:
        """Mark a step as failed.

        Args:
            step_id: ID of the step to mark.
            error: Optional error message.
        """
        for step in self.steps:
            if step.id == step_id:
                step.status = PlanStatus.FAILED
                step.result = {"error": error}
                break

        self.status = PlanStatus.FAILED


class PlanSkill(BaseSkill):
    """Skill for creating and managing execution plans.

    Uses an LLM to break down complex tasks into actionable steps
    and optionally identifies tools/actions for each step.
    """

    def __init__(self):
        """Initialize the plan skill."""
        super().__init__(
            metadata=SkillMetadata(
                name="plan",
                description="Create an execution plan for achieving a goal",
                category=SkillCategory.PLANNING,
                tags=["planning", "task-decomposition", "workflow"],
                requires_llm=True,
            )
        )

    async def execute(
        self,
        context: SkillContext,
        goal: str,
        max_steps: int = 10,
        include_actions: bool = True,
        available_tools: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create an execution plan for the given goal.

        Args:
            context: Execution context with LLM provider.
            goal: The goal to plan for.
            max_steps: Maximum number of steps in the plan.
            include_actions: Whether to include tool/action suggestions.
            available_tools: Optional list of available tool names.

        Returns:
            Plan as a dictionary.

        Raises:
            ValueError: If LLM is not available.
        """
        if context.llm is None:
            raise ValueError("LLM provider is required for planning")

        # Get available tools if not provided
        if available_tools is None and context.tools is not None:
            available_tools = context.tools.list_names()

        # Build the planning prompt
        prompt = self._build_prompt(
            goal=goal,
            max_steps=max_steps,
            include_actions=include_actions,
            available_tools=available_tools or [],
        )

        from agents_framework.llm import Message, MessageRole

        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=self._get_system_prompt(),
            ),
            Message(
                role=MessageRole.USER,
                content=prompt,
            ),
        ]

        response = await context.llm.generate(messages)

        # Parse the response into a plan
        plan = self._parse_plan_response(goal, response.content)

        return plan.to_dict()

    def _get_system_prompt(self) -> str:
        """Get the system prompt for planning."""
        return """You are an expert task planner. Your role is to break down complex goals into clear, actionable steps.

When creating a plan:
1. Break the goal into logical, sequential steps
2. Identify dependencies between steps
3. Keep steps clear and specific
4. If tools are available, suggest which tool to use for each step

Format your response as a numbered list of steps. For each step, include:
- A clear description of what to do
- [Optional] The tool/action to use (in parentheses)
- [Optional] Dependencies on other steps (e.g., "after step 1")

Example:
1. Gather requirements from the user (action: ask_user)
2. Research available options (action: search) - after step 1
3. Analyze and compare options (action: analyze) - after step 2
4. Present recommendations to user (action: respond) - after step 3
"""

    def _build_prompt(
        self,
        goal: str,
        max_steps: int,
        include_actions: bool,
        available_tools: List[str],
    ) -> str:
        """Build the planning prompt."""
        parts = [f"Create a plan to achieve this goal: {goal}"]

        parts.append(f"\nProvide up to {max_steps} steps.")

        if include_actions and available_tools:
            tools_str = ", ".join(available_tools)
            parts.append(f"\nAvailable tools/actions: {tools_str}")
            parts.append("Suggest which tool to use for each step where applicable.")

        return "\n".join(parts)

    def _parse_plan_response(self, goal: str, response: str) -> Plan:
        """Parse LLM response into a Plan object.

        Args:
            goal: The original goal.
            response: LLM response text.

        Returns:
            Parsed Plan object.
        """
        steps = []
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try to parse numbered step
            step_data = self._parse_step_line(line, len(steps) + 1)
            if step_data:
                steps.append(step_data)

        return Plan(goal=goal, steps=steps)

    def _parse_step_line(
        self,
        line: str,
        step_num: int,
    ) -> Optional[PlanStep]:
        """Parse a single step line.

        Args:
            line: The line to parse.
            step_num: Default step number.

        Returns:
            PlanStep if parsing succeeds, None otherwise.
        """
        import re

        # Remove common prefixes like "1.", "1)", "Step 1:", etc.
        patterns = [
            r"^\d+[\.\)]\s*",
            r"^Step\s+\d+[:\s]*",
            r"^-\s*",
            r"^\*\s*",
        ]

        description = line
        for pattern in patterns:
            description = re.sub(pattern, "", description, flags=re.IGNORECASE)

        if not description.strip():
            return None

        # Extract action if present (in parentheses)
        action = None
        action_match = re.search(r"\((?:action:|tool:)?\s*(\w+)\)", description)
        if action_match:
            action = action_match.group(1)
            description = re.sub(r"\s*\([^)]+\)\s*", " ", description)

        # Extract dependencies
        dependencies = []
        dep_match = re.search(
            r"(?:after|depends on|requires)\s+(?:step\s+)?(\d+(?:\s*,\s*\d+)*)",
            description,
            re.IGNORECASE,
        )
        if dep_match:
            dep_nums = re.findall(r"\d+", dep_match.group(1))
            dependencies = [f"step_{num}" for num in dep_nums]
            description = re.sub(
                r"\s*-?\s*(?:after|depends on|requires)\s+[^.]*",
                "",
                description,
                flags=re.IGNORECASE,
            )

        return PlanStep(
            id=f"step_{step_num}",
            description=description.strip(),
            action=action,
            dependencies=dependencies,
        )

    async def refine_plan(
        self,
        context: SkillContext,
        plan: Dict[str, Any],
        feedback: str,
    ) -> Dict[str, Any]:
        """Refine a plan based on feedback.

        Args:
            context: Execution context with LLM.
            plan: Current plan dictionary.
            feedback: Feedback for refinement.

        Returns:
            Refined plan dictionary.
        """
        if context.llm is None:
            raise ValueError("LLM provider is required for planning")

        from agents_framework.llm import Message, MessageRole

        # Format current plan
        plan_text = self._format_plan_for_prompt(plan)

        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=self._get_system_prompt(),
            ),
            Message(
                role=MessageRole.USER,
                content=f"Current plan:\n{plan_text}\n\nFeedback: {feedback}\n\nPlease provide a refined plan.",
            ),
        ]

        response = await context.llm.generate(messages)

        refined = self._parse_plan_response(plan["goal"], response.content)
        return refined.to_dict()

    def _format_plan_for_prompt(self, plan: Dict[str, Any]) -> str:
        """Format a plan dictionary for inclusion in a prompt."""
        lines = [f"Goal: {plan.get('goal', '')}"]
        for i, step in enumerate(plan.get("steps", []), 1):
            action_str = f" (action: {step['action']})" if step.get("action") else ""
            lines.append(f"{i}. {step['description']}{action_str}")
        return "\n".join(lines)
