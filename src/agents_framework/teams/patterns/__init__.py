"""Team patterns package.

This package provides various patterns for orchestrating teams of agents:
- HierarchicalPattern: Supervisor-worker pattern with task decomposition
- SequentialPattern: Pipeline pattern with stage-to-stage data passing
- SwarmPattern: Dynamic handoff pattern with conversation continuity

Example:
    from agents_framework.teams.patterns import (
        HierarchicalPattern,
        SequentialPattern,
        SwarmPattern,
        PatternContext,
        PatternResult,
    )

    # Create a hierarchical team
    pattern = HierarchicalPattern()
    result = await pattern.execute(task, [supervisor, worker1, worker2])

    # Create a sequential pipeline
    pipeline = SequentialPattern()
    result = await pipeline.execute(task, [stage1_agent, stage2_agent])

    # Create a swarm with dynamic handoffs
    swarm = SwarmPattern()
    result = await swarm.execute(task, agents)
"""

from .base import (
    BasePattern,
    PatternContext,
    PatternResult,
    PatternStatus,
    StepResult,
    TeamPattern,
)
from .hierarchical import (
    EscalationLevel,
    EscalationRequest,
    HierarchicalConfig,
    HierarchicalPattern,
    SubTask,
)
from .sequential import (
    BranchConfig,
    PipelineStage,
    PipelineState,
    SequentialConfig,
    SequentialPattern,
    StageStatus,
)
from .swarm import (
    HandoffReason,
    HandoffRequest,
    SwarmConfig,
    SwarmMessage,
    SwarmPattern,
    SwarmState,
)

__all__ = [
    # Base
    "BasePattern",
    "PatternContext",
    "PatternResult",
    "PatternStatus",
    "StepResult",
    "TeamPattern",
    # Hierarchical
    "EscalationLevel",
    "EscalationRequest",
    "HierarchicalConfig",
    "HierarchicalPattern",
    "SubTask",
    # Sequential
    "BranchConfig",
    "PipelineStage",
    "PipelineState",
    "SequentialConfig",
    "SequentialPattern",
    "StageStatus",
    # Swarm
    "HandoffReason",
    "HandoffRequest",
    "SwarmConfig",
    "SwarmMessage",
    "SwarmPattern",
    "SwarmState",
]
