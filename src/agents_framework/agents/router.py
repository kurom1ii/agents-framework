"""Router agent implementation.

This module provides a RouterAgent that routes incoming tasks to
appropriate agents based on LLM decisions, rules, or capabilities.
"""

from __future__ import annotations

import asyncio
import re
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
    AgentConfig,
    AgentRole,
    AgentStatus,
    BaseAgent,
    Task,
    TaskResult,
)

if TYPE_CHECKING:
    from agents_framework.llm import LLMProvider


class RoutingStrategy(str, Enum):
    """Strategy for routing decisions."""

    LLM = "llm"  # Use LLM to decide routing
    RULE_BASED = "rule_based"  # Use predefined rules
    CAPABILITY = "capability"  # Match based on capabilities
    ROUND_ROBIN = "round_robin"  # Distribute evenly
    FALLBACK = "fallback"  # Use in order until one succeeds


@dataclass
class RoutingRule:
    """A rule for routing tasks.

    Attributes:
        name: Name of the rule.
        pattern: Regex pattern or keyword to match.
        target_agent_id: ID of the target agent.
        priority: Priority (higher = checked first).
        condition: Optional function for custom matching.
        metadata: Additional rule metadata.
    """

    name: str
    pattern: Optional[str] = None
    target_agent_id: Optional[str] = None
    target_capability: Optional[str] = None
    priority: int = 0
    condition: Optional[Callable[[Task], bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Result of a routing decision.

    Attributes:
        target_agent_id: ID of the selected agent.
        strategy_used: Which strategy made the decision.
        confidence: Confidence score (0-1) for LLM decisions.
        reason: Explanation for the routing decision.
        fallback_agents: Ordered list of fallback agent IDs.
    """

    target_agent_id: str
    strategy_used: RoutingStrategy
    confidence: float = 1.0
    reason: str = ""
    fallback_agents: List[str] = field(default_factory=list)


@dataclass
class RouterConfig:
    """Configuration for the router agent.

    Attributes:
        default_strategy: Default routing strategy.
        fallback_enabled: Whether to try fallback agents on failure.
        max_fallback_attempts: Maximum number of fallback attempts.
        llm_routing_prompt: Custom prompt for LLM-based routing.
        confidence_threshold: Minimum confidence for LLM routing.
        timeout_per_agent: Timeout for each routed agent call.
    """

    default_strategy: RoutingStrategy = RoutingStrategy.LLM
    fallback_enabled: bool = True
    max_fallback_attempts: int = 3
    llm_routing_prompt: Optional[str] = None
    confidence_threshold: float = 0.7
    timeout_per_agent: float = 60.0


class RouterAgent(BaseAgent):
    """Agent that routes tasks to other agents.

    The RouterAgent acts as a dispatcher, analyzing incoming tasks
    and routing them to the most appropriate agent based on:
    - LLM-based semantic understanding
    - Rule-based pattern matching
    - Capability matching
    - Round-robin distribution
    - Fallback chains

    Example:
        router = RouterAgent(
            role=AgentRole(name="router", description="Routes tasks"),
            llm=llm_provider,
        )

        # Register agents to route to
        router.register_agent(coding_agent, capabilities=["code", "debug"])
        router.register_agent(research_agent, capabilities=["search", "analyze"])

        # Add routing rules
        router.add_rule(RoutingRule(
            name="code_request",
            pattern=r"(write|create|code|implement)",
            target_capability="code",
        ))

        # Route a task
        result = await router.run(task)
    """

    def __init__(
        self,
        role: Optional[AgentRole] = None,
        llm: Optional[LLMProvider] = None,
        config: Optional[AgentConfig] = None,
        router_config: Optional[RouterConfig] = None,
    ):
        """Initialize the router agent.

        Args:
            role: Role definition for the router.
            llm: LLM provider for semantic routing.
            config: Agent configuration.
            router_config: Router-specific configuration.
        """
        if role is None:
            role = AgentRole(
                name="router",
                description="Routes tasks to appropriate agents",
                capabilities=["routing", "delegation"],
            )
        super().__init__(role=role, llm=llm, config=config)

        self.router_config = router_config or RouterConfig()
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_capabilities: Dict[str, List[str]] = {}
        self._rules: List[RoutingRule] = []
        self._round_robin_index = 0

    def register_agent(
        self,
        agent: BaseAgent,
        capabilities: Optional[List[str]] = None,
    ) -> None:
        """Register an agent that can receive routed tasks.

        Args:
            agent: The agent to register.
            capabilities: Optional capabilities for routing decisions.
        """
        self._agents[agent.id] = agent
        self._agent_capabilities[agent.id] = (
            capabilities or agent.role.capabilities
        )

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the router.

        Args:
            agent_id: ID of the agent to unregister.

        Returns:
            True if the agent was unregistered.
        """
        if agent_id in self._agents:
            del self._agents[agent_id]
            del self._agent_capabilities[agent_id]
            return True
        return False

    def add_rule(self, rule: RoutingRule) -> None:
        """Add a routing rule.

        Args:
            rule: The routing rule to add.
        """
        self._rules.append(rule)
        # Sort rules by priority (highest first)
        self._rules.sort(key=lambda r: -r.priority)

    def remove_rule(self, name: str) -> bool:
        """Remove a routing rule by name.

        Args:
            name: Name of the rule to remove.

        Returns:
            True if the rule was removed.
        """
        original_len = len(self._rules)
        self._rules = [r for r in self._rules if r.name != name]
        return len(self._rules) < original_len

    async def run(self, task: Union[str, Task]) -> TaskResult:
        """Route and execute a task.

        Args:
            task: The task to route and execute.

        Returns:
            TaskResult from the routed agent.
        """
        # Convert string to Task if needed
        if isinstance(task, str):
            task = Task(description=task)

        self._status = AgentStatus.BUSY

        try:
            # Make routing decision
            decision = await self._make_routing_decision(task)

            if not decision:
                self._status = AgentStatus.ERROR
                return TaskResult(
                    task_id=task.id,
                    success=False,
                    error="Unable to determine routing for task",
                )

            # Execute with the selected agent
            result = await self._execute_with_fallback(task, decision)

            self._status = AgentStatus.IDLE
            return result

        except Exception as e:
            self._status = AgentStatus.ERROR
            return TaskResult(
                task_id=task.id,
                success=False,
                error=f"Router error: {str(e)}",
            )

    async def _make_routing_decision(self, task: Task) -> Optional[RoutingDecision]:
        """Make a routing decision based on the configured strategy.

        Args:
            task: The task to route.

        Returns:
            RoutingDecision if a target is found, None otherwise.
        """
        strategy = self.router_config.default_strategy

        if strategy == RoutingStrategy.LLM:
            decision = await self._llm_routing(task)
            if decision:
                return decision
            # Fallback to rule-based if LLM fails
            strategy = RoutingStrategy.RULE_BASED

        if strategy == RoutingStrategy.RULE_BASED:
            decision = self._rule_based_routing(task)
            if decision:
                return decision
            # Fallback to capability matching
            strategy = RoutingStrategy.CAPABILITY

        if strategy == RoutingStrategy.CAPABILITY:
            decision = self._capability_routing(task)
            if decision:
                return decision

        if strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_routing()

        # Final fallback: use first available agent
        if self._agents:
            first_agent_id = list(self._agents.keys())[0]
            return RoutingDecision(
                target_agent_id=first_agent_id,
                strategy_used=RoutingStrategy.FALLBACK,
                reason="Fallback to first available agent",
                fallback_agents=list(self._agents.keys())[1:],
            )

        return None

    async def _llm_routing(self, task: Task) -> Optional[RoutingDecision]:
        """Use LLM to make routing decision.

        Args:
            task: The task to route.

        Returns:
            RoutingDecision based on LLM analysis.
        """
        if not self.llm:
            return None

        # Build agent descriptions for the prompt
        agent_descriptions = []
        for agent_id, agent in self._agents.items():
            capabilities = self._agent_capabilities.get(agent_id, [])
            agent_descriptions.append(
                f"- {agent.role.name} (ID: {agent_id}): "
                f"{agent.role.description}. Capabilities: {', '.join(capabilities)}"
            )

        if not agent_descriptions:
            return None

        # Build routing prompt
        prompt = self.router_config.llm_routing_prompt or self._default_routing_prompt()

        from agents_framework.llm import Message, MessageRole

        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=prompt,
            ),
            Message(
                role=MessageRole.USER,
                content=f"""Available agents:
{chr(10).join(agent_descriptions)}

Task to route:
{task.description}

Task context: {task.context}

Respond with JSON: {{"agent_id": "...", "confidence": 0.0-1.0, "reason": "..."}}""",
            ),
        ]

        try:
            response = await self.llm.generate(messages)

            # Parse LLM response
            decision = self._parse_llm_response(response.content)

            if decision and decision.confidence >= self.router_config.confidence_threshold:
                return decision

        except Exception:
            pass

        return None

    def _parse_llm_response(self, content: str) -> Optional[RoutingDecision]:
        """Parse LLM response into a RoutingDecision.

        Args:
            content: Raw LLM response.

        Returns:
            RoutingDecision if successfully parsed.
        """
        import json

        try:
            # Try to extract JSON from response
            json_match = re.search(r"\{[^}]+\}", content)
            if json_match:
                data = json.loads(json_match.group())
                agent_id = data.get("agent_id")

                if agent_id and agent_id in self._agents:
                    return RoutingDecision(
                        target_agent_id=agent_id,
                        strategy_used=RoutingStrategy.LLM,
                        confidence=float(data.get("confidence", 0.8)),
                        reason=data.get("reason", "LLM routing decision"),
                    )
        except (json.JSONDecodeError, ValueError):
            pass

        return None

    def _rule_based_routing(self, task: Task) -> Optional[RoutingDecision]:
        """Apply routing rules to find target agent.

        Args:
            task: The task to route.

        Returns:
            RoutingDecision if a rule matches.
        """
        for rule in self._rules:
            if self._rule_matches(rule, task):
                # Find target agent
                target_id = rule.target_agent_id

                if not target_id and rule.target_capability:
                    target_id = self._find_agent_by_capability(
                        rule.target_capability
                    )

                if target_id and target_id in self._agents:
                    return RoutingDecision(
                        target_agent_id=target_id,
                        strategy_used=RoutingStrategy.RULE_BASED,
                        reason=f"Matched rule: {rule.name}",
                    )

        return None

    def _rule_matches(self, rule: RoutingRule, task: Task) -> bool:
        """Check if a rule matches the task.

        Args:
            rule: The routing rule.
            task: The task to check.

        Returns:
            True if the rule matches.
        """
        # Custom condition takes priority
        if rule.condition:
            return rule.condition(task)

        # Pattern matching
        if rule.pattern:
            pattern = re.compile(rule.pattern, re.IGNORECASE)
            if pattern.search(task.description):
                return True

            # Also check context values
            for value in task.context.values():
                if isinstance(value, str) and pattern.search(value):
                    return True

        return False

    def _capability_routing(self, task: Task) -> Optional[RoutingDecision]:
        """Route based on required capabilities in task.

        Args:
            task: The task to route.

        Returns:
            RoutingDecision if capability match found.
        """
        # Check for required capabilities in task
        required = task.required_capabilities

        if not required:
            # Try to extract from task description
            required = self._extract_capabilities(task.description)

        for capability in required:
            target_id = self._find_agent_by_capability(capability)
            if target_id:
                return RoutingDecision(
                    target_agent_id=target_id,
                    strategy_used=RoutingStrategy.CAPABILITY,
                    reason=f"Matched capability: {capability}",
                )

        return None

    def _round_robin_routing(self) -> Optional[RoutingDecision]:
        """Distribute tasks evenly across agents.

        Returns:
            RoutingDecision with next agent in rotation.
        """
        if not self._agents:
            return None

        agent_ids = list(self._agents.keys())
        target_id = agent_ids[self._round_robin_index % len(agent_ids)]
        self._round_robin_index += 1

        return RoutingDecision(
            target_agent_id=target_id,
            strategy_used=RoutingStrategy.ROUND_ROBIN,
            reason=f"Round-robin selection (index: {self._round_robin_index - 1})",
        )

    def _find_agent_by_capability(self, capability: str) -> Optional[str]:
        """Find an agent with a specific capability.

        Args:
            capability: The required capability.

        Returns:
            Agent ID if found.
        """
        for agent_id, capabilities in self._agent_capabilities.items():
            if capability.lower() in [c.lower() for c in capabilities]:
                return agent_id
        return None

    def _extract_capabilities(self, text: str) -> List[str]:
        """Extract potential capability requirements from text.

        Args:
            text: Text to analyze.

        Returns:
            List of potential capability keywords.
        """
        # Common capability keywords
        capability_keywords = {
            "code": ["code", "program", "develop", "implement", "write code"],
            "search": ["search", "find", "look up", "research"],
            "analyze": ["analyze", "examine", "review", "evaluate"],
            "summarize": ["summarize", "condense", "tldr", "brief"],
            "translate": ["translate", "convert language"],
            "debug": ["debug", "fix", "troubleshoot", "error"],
            "design": ["design", "architect", "plan", "structure"],
        }

        found = []
        text_lower = text.lower()

        for capability, keywords in capability_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found.append(capability)
                    break

        return found

    async def _execute_with_fallback(
        self,
        task: Task,
        decision: RoutingDecision,
    ) -> TaskResult:
        """Execute task with fallback on failure.

        Args:
            task: The task to execute.
            decision: Routing decision with target and fallbacks.

        Returns:
            TaskResult from successful execution.
        """
        # Try primary target
        target_agent = self._agents.get(decision.target_agent_id)

        if target_agent:
            try:
                result = await asyncio.wait_for(
                    target_agent.run(task),
                    timeout=self.router_config.timeout_per_agent,
                )
                if result.success:
                    result.metadata["routed_by"] = self.id
                    result.metadata["routing_decision"] = {
                        "target": decision.target_agent_id,
                        "strategy": decision.strategy_used.value,
                        "reason": decision.reason,
                    }
                    return result
            except asyncio.TimeoutError:
                pass  # Try fallback

        # Try fallback agents if enabled
        if not self.router_config.fallback_enabled:
            return TaskResult(
                task_id=task.id,
                success=False,
                error=f"Primary agent {decision.target_agent_id} failed",
            )

        fallbacks = decision.fallback_agents or [
            aid for aid in self._agents.keys()
            if aid != decision.target_agent_id
        ]

        for i, fallback_id in enumerate(fallbacks):
            if i >= self.router_config.max_fallback_attempts:
                break

            fallback_agent = self._agents.get(fallback_id)
            if not fallback_agent:
                continue

            try:
                result = await asyncio.wait_for(
                    fallback_agent.run(task),
                    timeout=self.router_config.timeout_per_agent,
                )
                if result.success:
                    result.metadata["routed_by"] = self.id
                    result.metadata["routing_decision"] = {
                        "target": fallback_id,
                        "strategy": RoutingStrategy.FALLBACK.value,
                        "reason": f"Fallback attempt {i + 1}",
                        "original_target": decision.target_agent_id,
                    }
                    return result
            except asyncio.TimeoutError:
                continue

        return TaskResult(
            task_id=task.id,
            success=False,
            error="All agents failed or timed out",
        )

    def _default_routing_prompt(self) -> str:
        """Get the default routing prompt for LLM-based routing."""
        return """You are a task router. Your job is to analyze incoming tasks and
determine which agent is best suited to handle them.

Consider:
1. The task's requirements and complexity
2. Each agent's capabilities and expertise
3. The context and any specific requirements

Respond with a JSON object containing:
- agent_id: The ID of the best-suited agent
- confidence: Your confidence in this choice (0.0 to 1.0)
- reason: A brief explanation of why this agent was chosen

Be precise and base your decision on the agents' stated capabilities."""

    def get_registered_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered agents.

        Returns:
            Dictionary with agent info.
        """
        return {
            agent_id: {
                "name": agent.role.name,
                "description": agent.role.description,
                "capabilities": self._agent_capabilities.get(agent_id, []),
                "status": agent.status.value,
            }
            for agent_id, agent in self._agents.items()
        }

    def get_routing_rules(self) -> List[Dict[str, Any]]:
        """Get all routing rules.

        Returns:
            List of rule information.
        """
        return [
            {
                "name": rule.name,
                "pattern": rule.pattern,
                "target_agent_id": rule.target_agent_id,
                "target_capability": rule.target_capability,
                "priority": rule.priority,
            }
            for rule in self._rules
        ]
