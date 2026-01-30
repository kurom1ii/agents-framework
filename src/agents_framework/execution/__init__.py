"""Agent execution module for the agents framework.

This package provides the execution engine for running agents, including:
- AgentLoop implementing the ReAct (Thought -> Action -> Observation) pattern
- AgentRunner for execution management
- Hook system for lifecycle events
- Streaming support

Example:
    from agents_framework.execution import AgentLoop, AgentRunner, HookRegistry

    # Create an agent loop
    loop = AgentLoop(agent=my_agent, max_iterations=10)
    result = await loop.run("What is the weather?")

    # With hooks
    hooks = HookRegistry()
    hooks.register("pre_execute", my_logging_hook)
    runner = AgentRunner(hooks=hooks)
    result = await runner.run(agent, task)
"""

from .loop import (
    AgentLoop,
    AgentRunner,
    LoopConfig,
    LoopState,
    LoopStep,
    StepType,
    TerminationReason,
)
from .hooks import (
    Hook,
    HookRegistry,
    HookType,
    HookContext,
    LoggingHook,
    ValidationHook,
)

__all__ = [
    # Loop
    "AgentLoop",
    "AgentRunner",
    "LoopConfig",
    "LoopState",
    "LoopStep",
    "StepType",
    "TerminationReason",
    # Hooks
    "Hook",
    "HookRegistry",
    "HookType",
    "HookContext",
    "LoggingHook",
    "ValidationHook",
]
