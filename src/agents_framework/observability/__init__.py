"""Observability module for tracing, logging, and metrics.

This module provides comprehensive observability capabilities for
the agents framework, including:

- Structured logging with context propagation
- Distributed tracing with spans and traces
- Metrics collection for monitoring and alerting

Example:
    # Logging with context
    from agents_framework.observability import get_logger, LogContext, set_context

    logger = get_logger("my_agent")
    set_context(LogContext(agent_id="agent-1", correlation_id="req-123"))
    logger.info("Processing task", task_id="task-456")

    # Tracing execution
    from agents_framework.observability import trace, SpanKind

    @trace(name="process_task", kind=SpanKind.AGENT)
    async def process_task(task: Task) -> TaskResult:
        return await agent.run(task)

    # Metrics collection
    from agents_framework.observability import get_metrics_collector

    metrics = get_metrics_collector()
    metrics.inc_counter("tool_calls_total", labels={"tool": "search"})
    metrics.observe_histogram("tool_execution_duration_seconds", 0.5)
"""

from .logging import (
    AgentLogger,
    ConsoleExporter,
    FileExporter,
    LogConfig,
    LogContext,
    LogExporter,
    LogLevel,
    clear_context,
    clear_global_context,
    configure_logging,
    get_context,
    get_global_context,
    get_logger,
    set_context,
    set_global_context,
)
from .tracing import (
    ConsoleSpanExporter,
    InMemorySpanExporter,
    Span,
    SpanContext,
    SpanEvent,
    SpanExporter,
    SpanKind,
    SpanStatus,
    Tracer,
    TracerConfig,
    get_current_span,
    get_tracer,
    set_current_span,
    trace,
)
from .metrics import (
    ConsoleMetricsExporter,
    Counter,
    FileMetricsExporter,
    Gauge,
    Histogram,
    MetricData,
    MetricPoint,
    MetricsCollector,
    MetricsConfig,
    MetricsExporter,
    MetricType,
    get_metrics_collector,
    timed,
)

__all__ = [
    # Logging
    "AgentLogger",
    "ConsoleExporter",
    "FileExporter",
    "LogConfig",
    "LogContext",
    "LogExporter",
    "LogLevel",
    "clear_context",
    "clear_global_context",
    "configure_logging",
    "get_context",
    "get_global_context",
    "get_logger",
    "set_context",
    "set_global_context",
    # Tracing
    "ConsoleSpanExporter",
    "InMemorySpanExporter",
    "Span",
    "SpanContext",
    "SpanEvent",
    "SpanExporter",
    "SpanKind",
    "SpanStatus",
    "Tracer",
    "TracerConfig",
    "get_current_span",
    "get_tracer",
    "set_current_span",
    "trace",
    # Metrics
    "ConsoleMetricsExporter",
    "Counter",
    "FileMetricsExporter",
    "Gauge",
    "Histogram",
    "MetricData",
    "MetricPoint",
    "MetricsCollector",
    "MetricsConfig",
    "MetricsExporter",
    "MetricType",
    "get_metrics_collector",
    "timed",
]
