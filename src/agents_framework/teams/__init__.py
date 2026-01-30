"""Teams package for multi-agent coordination.

This package provides infrastructure for organizing agents into teams
and coordinating their activities using various patterns.

Subpackages:
    - patterns: Team patterns for agent orchestration
        - HierarchicalPattern: Supervisor-worker pattern
        - SequentialPattern: Pipeline pattern
        - SwarmPattern: Dynamic handoff pattern

Modules:
    - registry: Agent registry for managing agents
    - router: Message router for inter-agent communication
    - team: Team orchestration for multi-agent collaboration

Example:
    from agents_framework.teams import (
        AgentRegistry,
        MessageRouter,
        Team,
        TeamConfig,
        TeamExecutionStrategy,
    )

    # Create a team
    team = Team(TeamConfig(
        name="research_team",
        strategy=TeamExecutionStrategy.COLLABORATIVE,
    ))

    # Add agents to the team
    team.add_member(researcher)
    team.add_member(analyst)

    # Execute tasks collaboratively
    await team.start()
    result = await team.run("Research and analyze market trends")
    await team.stop()
"""

from .registry import (
    AgentInfo,
    AgentLookupError,
    AgentRegistry,
    get_default_registry,
    register_agent,
)
from .router import (
    AgentMessage,
    MessageAcknowledgment,
    MessageHandler,
    MessagePriority,
    MessageQueue,
    MessageRouter,
    MessageStatus,
    RoutingStrategy,
)
from .team import (
    ResultMerger,
    SharedContext,
    TaskDivider,
    Team,
    TeamConfig,
    TeamExecutionStrategy,
    TeamState,
    TeamTaskResult,
)

__all__ = [
    # Registry
    "AgentInfo",
    "AgentLookupError",
    "AgentRegistry",
    "get_default_registry",
    "register_agent",
    # Router
    "AgentMessage",
    "MessageAcknowledgment",
    "MessageHandler",
    "MessagePriority",
    "MessageQueue",
    "MessageRouter",
    "MessageStatus",
    "RoutingStrategy",
    # Team
    "ResultMerger",
    "SharedContext",
    "TaskDivider",
    "Team",
    "TeamConfig",
    "TeamExecutionStrategy",
    "TeamState",
    "TeamTaskResult",
]
