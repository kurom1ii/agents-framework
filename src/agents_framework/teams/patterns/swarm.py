"""Swarm team pattern implementation.

This module implements a swarm pattern where agents can hand off
conversation control to other agents dynamically. This creates
a fluid, multi-agent conversation with seamless transitions.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

from .base import (
    BasePattern,
    PatternContext,
    PatternResult,
    PatternStatus,
    StepResult,
)

if TYPE_CHECKING:
    from agents_framework.agents.base import BaseAgent, Task, TaskResult


class HandoffReason(str, Enum):
    """Reason for handing off to another agent."""

    CAPABILITY_MATCH = "capability_match"
    EXPLICIT_REQUEST = "explicit_request"
    TASK_COMPLETE = "task_complete"
    UNABLE_TO_HELP = "unable_to_help"
    ESCALATION = "escalation"


@dataclass
class HandoffRequest:
    """Request to hand off conversation to another agent.

    Attributes:
        target_agent_id: ID or name of the target agent.
        reason: Why the handoff is occurring.
        message: Message to pass to the target agent.
        context_updates: Variables to update in context.
        preserve_history: Whether to include conversation history.
    """

    target_agent_id: str
    reason: HandoffReason = HandoffReason.EXPLICIT_REQUEST
    message: str = ""
    context_updates: Dict[str, Any] = field(default_factory=dict)
    preserve_history: bool = True


@dataclass
class SwarmMessage:
    """A message in the swarm conversation.

    Attributes:
        sender_id: ID of the agent who sent the message.
        sender_name: Name of the sender agent.
        content: Message content.
        role: Role (user, assistant, system).
        timestamp: When the message was sent.
        metadata: Additional message metadata.
    """

    sender_id: str
    sender_name: str
    content: str
    role: str = "assistant"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SwarmState:
    """State of the swarm execution.

    Attributes:
        active_agent_id: Currently active agent ID.
        conversation_history: Full conversation history.
        handoff_history: History of agent handoffs.
        agent_call_counts: Number of times each agent was invoked.
        context_variables: Shared context variables.
    """

    active_agent_id: Optional[str] = None
    conversation_history: List[SwarmMessage] = field(default_factory=list)
    handoff_history: List[Dict[str, str]] = field(default_factory=list)
    agent_call_counts: Dict[str, int] = field(default_factory=dict)
    context_variables: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SwarmConfig:
    """Configuration for the swarm pattern.

    Attributes:
        max_handoffs: Maximum number of handoffs before termination.
        max_turns: Maximum conversation turns.
        allow_self_handoff: Whether an agent can hand off to itself.
        default_agent_id: Agent to use when no specific target is found.
        handoff_timeout: Timeout for handoff operations.
        require_explicit_handoff: Whether agents must explicitly request handoffs.
    """

    max_handoffs: int = 10
    max_turns: int = 50
    allow_self_handoff: bool = False
    default_agent_id: Optional[str] = None
    handoff_timeout: float = 30.0
    require_explicit_handoff: bool = False


class SwarmPattern(BasePattern):
    """Swarm pattern for dynamic agent handoffs.

    This pattern allows agents to dynamically hand off conversation
    control to other agents based on:
    - Capability requirements
    - Explicit handoff requests
    - Task completion signals
    - Escalation needs

    The pattern maintains conversation continuity and shared context
    across all participating agents.

    Example usage:
        swarm = SwarmPattern()
        swarm.register_agent(agent1, capabilities=["coding", "debugging"])
        swarm.register_agent(agent2, capabilities=["research", "analysis"])

        result = await swarm.execute(task, [agent1, agent2])
    """

    def __init__(
        self,
        config: Optional[SwarmConfig] = None,
        handoff_fn: Optional[Callable] = None,
    ):
        """Initialize the swarm pattern.

        Args:
            config: Configuration for the swarm.
            handoff_fn: Custom function to determine handoffs.
        """
        super().__init__(name="swarm")
        self.config = config or SwarmConfig()
        self._handoff_fn = handoff_fn
        self._state = SwarmState()
        self._agent_capabilities: Dict[str, List[str]] = {}
        self._agent_registry: Dict[str, BaseAgent] = {}

    @property
    def state(self) -> SwarmState:
        """Get the current swarm state."""
        return self._state

    def register_agent(
        self,
        agent: BaseAgent,
        capabilities: Optional[List[str]] = None,
    ) -> None:
        """Register an agent with the swarm.

        Args:
            agent: The agent to register.
            capabilities: Optional list of capabilities for this agent.
        """
        self._agent_registry[agent.id] = agent
        self._agent_capabilities[agent.id] = (
            capabilities or agent.role.capabilities
        )

    async def execute(
        self,
        task: Task,
        agents: List[BaseAgent],
        context: Optional[PatternContext] = None,
    ) -> PatternResult:
        """Execute the swarm pattern.

        The first agent starts the conversation. Agents can hand off
        to each other based on capabilities and explicit requests.

        Args:
            task: The initial task/message.
            agents: List of participating agents.
            context: Optional initial context.

        Returns:
            PatternResult with conversation output.
        """
        if not await self.validate_agents(agents):
            return PatternResult(
                pattern_name=self.name,
                status=PatternStatus.FAILED,
                error="No agents provided for the swarm",
            )

        ctx = self._create_context(context)
        self._status = PatternStatus.RUNNING
        self._state = SwarmState()
        start_time = time.time()
        steps: List[StepResult] = []

        # Register all agents
        for agent in agents:
            self.register_agent(agent)

        # Set initial active agent
        initial_agent = agents[0]
        self._state.active_agent_id = initial_agent.id
        self._state.context_variables = ctx.variables.copy()

        # Set default agent if not specified
        if not self.config.default_agent_id:
            self.config.default_agent_id = initial_agent.id

        try:
            # Add initial user message
            self._state.conversation_history.append(
                SwarmMessage(
                    sender_id="user",
                    sender_name="user",
                    content=task.description,
                    role="user",
                )
            )

            handoff_count = 0
            turn_count = 0

            while (
                handoff_count < self.config.max_handoffs
                and turn_count < self.config.max_turns
            ):
                current_agent = self._agent_registry.get(
                    self._state.active_agent_id or ""
                )

                if not current_agent:
                    break

                # Execute the current agent's turn
                step_start = time.time()
                result, handoff = await self._execute_turn(
                    current_agent, task, ctx
                )
                step_duration = (time.time() - step_start) * 1000

                # Track agent calls
                self._state.agent_call_counts[current_agent.id] = (
                    self._state.agent_call_counts.get(current_agent.id, 0) + 1
                )

                steps.append(
                    self._create_step_result(
                        agent=current_agent,
                        output=result.output if result.success else None,
                        success=result.success,
                        error=result.error,
                        duration_ms=step_duration,
                    )
                )

                if result.success:
                    # Add agent response to conversation
                    response_content = (
                        result.output if isinstance(result.output, str)
                        else str(result.output) if result.output
                        else ""
                    )
                    self._state.conversation_history.append(
                        SwarmMessage(
                            sender_id=current_agent.id,
                            sender_name=current_agent.role.name,
                            content=response_content,
                            role="assistant",
                        )
                    )

                if handoff:
                    # Process handoff request
                    target_agent = await self._process_handoff(
                        handoff, current_agent, ctx
                    )

                    if target_agent:
                        self._state.handoff_history.append({
                            "from": current_agent.id,
                            "to": target_agent.id,
                            "reason": handoff.reason.value,
                        })
                        self._state.active_agent_id = target_agent.id
                        handoff_count += 1

                        # Update context with handoff updates
                        if handoff.context_updates:
                            ctx.update(handoff.context_updates)
                            self._state.context_variables.update(
                                handoff.context_updates
                            )
                    else:
                        # No valid handoff target, continue with current
                        turn_count += 1
                else:
                    # No handoff requested, check if we should continue
                    if self._is_conversation_complete(result, ctx):
                        break
                    turn_count += 1

                # Update task for next turn (if needed)
                ctx.history.append(steps[-1])

            self._status = PatternStatus.COMPLETED

            # Build final output from conversation
            final_output = self._build_final_output()

            return PatternResult(
                pattern_name=self.name,
                status=PatternStatus.COMPLETED,
                final_output=final_output,
                steps=steps,
                context=ctx,
                total_duration_ms=(time.time() - start_time) * 1000,
                metadata={
                    "handoff_count": handoff_count,
                    "turn_count": turn_count,
                    "agent_calls": self._state.agent_call_counts,
                    "handoff_history": self._state.handoff_history,
                },
            )

        except Exception as e:
            self._status = PatternStatus.FAILED
            return PatternResult(
                pattern_name=self.name,
                status=PatternStatus.FAILED,
                error=str(e),
                steps=steps,
                context=ctx,
                total_duration_ms=(time.time() - start_time) * 1000,
            )

    async def _execute_turn(
        self,
        agent: BaseAgent,
        task: Task,
        context: PatternContext,
    ) -> tuple[TaskResult, Optional[HandoffRequest]]:
        """Execute a single conversation turn with an agent.

        Args:
            agent: The agent to execute.
            task: The current task.
            context: Execution context.

        Returns:
            Tuple of (TaskResult, optional HandoffRequest).
        """
        from agents_framework.agents.base import Task as AgentTask, TaskResult

        # Build context with conversation history
        conversation = [
            {"role": msg.role, "content": msg.content, "sender": msg.sender_name}
            for msg in self._state.conversation_history
        ]

        agent_task = AgentTask(
            description=task.description,
            context={
                "conversation_history": conversation,
                "context_variables": self._state.context_variables,
                "available_agents": list(self._agent_registry.keys()),
                **context.variables,
            },
        )

        try:
            result = await asyncio.wait_for(
                agent.run(agent_task),
                timeout=self.config.handoff_timeout,
            )

            # Check for handoff in result
            handoff = self._extract_handoff(result, agent)

            return result, handoff

        except asyncio.TimeoutError:
            return TaskResult(
                task_id=agent_task.id,
                success=False,
                error="Agent turn timed out",
            ), None

    def _extract_handoff(
        self,
        result: TaskResult,
        current_agent: BaseAgent,
    ) -> Optional[HandoffRequest]:
        """Extract handoff request from agent result.

        Checks if the result contains a handoff signal, either as
        a structured HandoffRequest or as a special output format.
        """
        if not result.success:
            return None

        output = result.output

        # Check if output is a HandoffRequest
        if isinstance(output, HandoffRequest):
            return output

        # Check if output is a dict with handoff info
        if isinstance(output, dict):
            if "handoff_to" in output:
                target = output["handoff_to"]
                # Validate target
                if not self.config.allow_self_handoff and target == current_agent.id:
                    return None
                return HandoffRequest(
                    target_agent_id=target,
                    reason=HandoffReason(output.get("reason", "explicit_request")),
                    message=output.get("message", ""),
                    context_updates=output.get("context_updates", {}),
                )

            # Check for capability-based handoff
            if "required_capability" in output:
                target = self._find_agent_by_capability(
                    output["required_capability"],
                    exclude_id=current_agent.id if not self.config.allow_self_handoff else None,
                )
                if target:
                    return HandoffRequest(
                        target_agent_id=target,
                        reason=HandoffReason.CAPABILITY_MATCH,
                        context_updates=output.get("context_updates", {}),
                    )

        # Use custom handoff function if provided
        if self._handoff_fn:
            return self._handoff_fn(result, current_agent, self._state)

        return None

    async def _process_handoff(
        self,
        handoff: HandoffRequest,
        from_agent: BaseAgent,
        context: PatternContext,
    ) -> Optional[BaseAgent]:
        """Process a handoff request and find the target agent.

        Args:
            handoff: The handoff request.
            from_agent: The agent initiating the handoff.
            context: Execution context.

        Returns:
            The target agent if found, None otherwise.
        """
        target_id = handoff.target_agent_id

        # Direct lookup by ID
        if target_id in self._agent_registry:
            return self._agent_registry[target_id]

        # Lookup by name
        for agent_id, agent in self._agent_registry.items():
            if agent.role.name == target_id:
                return agent

        # Fallback to default agent
        if self.config.default_agent_id:
            return self._agent_registry.get(self.config.default_agent_id)

        return None

    def _find_agent_by_capability(
        self,
        capability: str,
        exclude_id: Optional[str] = None,
    ) -> Optional[str]:
        """Find an agent that has a specific capability.

        Args:
            capability: The required capability.
            exclude_id: Agent ID to exclude from search.

        Returns:
            Agent ID if found, None otherwise.
        """
        for agent_id, capabilities in self._agent_capabilities.items():
            if agent_id == exclude_id:
                continue
            if capability in capabilities:
                return agent_id
        return None

    def _is_conversation_complete(
        self,
        result: TaskResult,
        context: PatternContext,
    ) -> bool:
        """Check if the conversation should end.

        Args:
            result: Latest result from an agent.
            context: Execution context.

        Returns:
            True if conversation should end.
        """
        if not result.success:
            return True

        output = result.output

        # Check for explicit completion signal
        if isinstance(output, dict):
            if output.get("complete", False):
                return True
            if output.get("action") == "end_conversation":
                return True

        # Check context for completion flag
        if context.get("conversation_complete", False):
            return True

        return False

    def _build_final_output(self) -> Dict[str, Any]:
        """Build the final output from the conversation."""
        return {
            "messages": [
                {
                    "sender": msg.sender_name,
                    "content": msg.content,
                    "role": msg.role,
                }
                for msg in self._state.conversation_history
            ],
            "final_response": (
                self._state.conversation_history[-1].content
                if self._state.conversation_history
                else None
            ),
            "context_variables": self._state.context_variables,
        }

    def handoff(
        self,
        target_agent_id: str,
        reason: HandoffReason = HandoffReason.EXPLICIT_REQUEST,
        message: str = "",
        context_updates: Optional[Dict[str, Any]] = None,
    ) -> HandoffRequest:
        """Create a handoff request.

        This is a convenience method for agents to create handoff requests.

        Args:
            target_agent_id: ID or name of the target agent.
            reason: Reason for the handoff.
            message: Message to pass to the target.
            context_updates: Variables to update in context.

        Returns:
            HandoffRequest object.
        """
        return HandoffRequest(
            target_agent_id=target_agent_id,
            reason=reason,
            message=message,
            context_updates=context_updates or {},
        )

    def get_agent_capabilities(self, agent_id: str) -> List[str]:
        """Get capabilities of a registered agent."""
        return self._agent_capabilities.get(agent_id, [])

    def get_conversation_history(self) -> List[SwarmMessage]:
        """Get the full conversation history."""
        return self._state.conversation_history.copy()

    def reset_state(self) -> None:
        """Reset the swarm state for re-execution."""
        self._state = SwarmState()
        self._status = PatternStatus.PENDING
