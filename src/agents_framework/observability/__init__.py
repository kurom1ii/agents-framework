"""Observability module for tracing, logging, and metrics."""

from .logger import (
    AgentLogger,
    LogContext,
    get_logger,
    set_global_context,
    clear_global_context,
)
from .tracer import (
    Span,
    SpanKind,
    Tracer,
    get_tracer,
    trace,
)
from .metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsCollector,
    get_metrics_collector,
)

__all__ = [
    # Logger
    "AgentLogger",
    "LogContext",
    "get_logger",
    "set_global_context",
    "clear_global_context",
    # Tracer
    "Span",
    "SpanKind",
    "Tracer",
    "get_tracer",
    "trace",
    # Metrics
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsCollector",
    "get_metrics_collector",
]
