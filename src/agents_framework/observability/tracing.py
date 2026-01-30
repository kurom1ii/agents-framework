"""Tracing for agent execution with spans and traces.

This module provides distributed tracing capabilities for:
- Tracking agent execution flow
- Performance monitoring
- Request correlation
- Debugging multi-agent interactions
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
)


class SpanKind(str, Enum):
    """Kind of span indicating the role in the trace."""

    INTERNAL = "internal"  # Default internal operation
    CLIENT = "client"  # Outgoing request (e.g., LLM call)
    SERVER = "server"  # Incoming request handler
    PRODUCER = "producer"  # Message producer
    CONSUMER = "consumer"  # Message consumer
    AGENT = "agent"  # Agent execution
    TOOL = "tool"  # Tool execution
    LLM = "llm"  # LLM provider call


class SpanStatus(str, Enum):
    """Status of a span."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanContext:
    """Context for span propagation.

    Attributes:
        trace_id: Unique identifier for the entire trace.
        span_id: Unique identifier for this span.
        parent_span_id: ID of the parent span (if any).
        baggage: Key-value pairs for cross-cutting concerns.
    """

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)

    def child_context(self) -> "SpanContext":
        """Create a child span context."""
        return SpanContext(
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=self.span_id,
            baggage=self.baggage.copy(),
        )


@dataclass
class SpanEvent:
    """An event within a span.

    Attributes:
        name: Event name.
        timestamp: When the event occurred.
        attributes: Event attributes.
    """

    name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """Represents a unit of work in a trace.

    A span tracks the execution of a single operation within a trace,
    recording timing, status, and contextual information.

    Attributes:
        name: Name of the operation being traced.
        context: Span context with trace/span IDs.
        kind: Kind of span (internal, client, etc.).
        status: Current status of the span.
        start_time: When the span started.
        end_time: When the span ended (None if still active).
        attributes: Key-value pairs with span metadata.
        events: List of events that occurred during the span.
    """

    name: str
    context: SpanContext
    kind: SpanKind = SpanKind.INTERNAL
    status: SpanStatus = SpanStatus.UNSET
    status_message: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)

    # Timing in nanoseconds for precise measurement
    _start_ns: int = field(default_factory=time.perf_counter_ns)
    _end_ns: Optional[int] = None

    @property
    def trace_id(self) -> str:
        """Get the trace ID."""
        return self.context.trace_id

    @property
    def span_id(self) -> str:
        """Get the span ID."""
        return self.context.span_id

    @property
    def parent_span_id(self) -> Optional[str]:
        """Get the parent span ID."""
        return self.context.parent_span_id

    @property
    def duration_ns(self) -> Optional[int]:
        """Get span duration in nanoseconds."""
        if self._end_ns is None:
            return None
        return self._end_ns - self._start_ns

    @property
    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        duration_ns = self.duration_ns
        if duration_ns is None:
            return None
        return duration_ns / 1_000_000

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        """Set multiple span attributes."""
        self.attributes.update(attributes)

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> SpanEvent:
        """Add an event to the span."""
        event = SpanEvent(
            name=name,
            timestamp=datetime.utcnow(),
            attributes=attributes or {},
        )
        self.events.append(event)
        return event

    def set_status(
        self,
        status: SpanStatus,
        message: Optional[str] = None,
    ) -> None:
        """Set the span status."""
        self.status = status
        self.status_message = message

    def set_ok(self) -> None:
        """Set span status to OK."""
        self.set_status(SpanStatus.OK)

    def set_error(self, message: Optional[str] = None) -> None:
        """Set span status to ERROR."""
        self.set_status(SpanStatus.ERROR, message)

    def record_exception(
        self,
        exception: BaseException,
        escaped: bool = False,
    ) -> None:
        """Record an exception as a span event."""
        self.add_event(
            name="exception",
            attributes={
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
                "exception.escaped": escaped,
            },
        )
        self.set_error(str(exception))

    def end(self) -> None:
        """End the span, recording the end time."""
        if self.end_time is None:
            self.end_time = datetime.utcnow()
            self._end_ns = time.perf_counter_ns()
            if self.status == SpanStatus.UNSET:
                self.set_ok()

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for export."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "kind": self.kind.value,
            "status": self.status.value,
            "status_message": self.status_message,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": [
                {
                    "name": e.name,
                    "timestamp": e.timestamp.isoformat(),
                    "attributes": e.attributes,
                }
                for e in self.events
            ],
        }

    def __enter__(self) -> "Span":
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        """Exit context manager."""
        if exc_val is not None:
            self.record_exception(exc_val, escaped=True)
        self.end()

    async def __aenter__(self) -> "Span":
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        """Exit async context manager."""
        if exc_val is not None:
            self.record_exception(exc_val, escaped=True)
        self.end()


# Context variable for current span
_current_span: ContextVar[Optional[Span]] = ContextVar(
    "current_span", default=None
)


class SpanExporter:
    """Base class for span exporters."""

    async def export(self, spans: List[Span]) -> None:
        """Export spans to the backend.

        Args:
            spans: List of spans to export.
        """
        raise NotImplementedError

    async def shutdown(self) -> None:
        """Shut down the exporter."""
        pass


class ConsoleSpanExporter(SpanExporter):
    """Exports spans to the console for debugging."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    async def export(self, spans: List[Span]) -> None:
        """Print spans to console."""
        for span in spans:
            duration = span.duration_ms or 0
            status = span.status.value

            if self.verbose:
                print(
                    f"[TRACE] {span.trace_id[:8]}... "
                    f"| {span.name} [{span.kind.value}] "
                    f"| {duration:.2f}ms | {status}"
                )
                if span.attributes:
                    for key, value in span.attributes.items():
                        print(f"        {key}: {value}")
            else:
                print(
                    f"[TRACE] {span.name} | {duration:.2f}ms | {status}"
                )


class InMemorySpanExporter(SpanExporter):
    """Stores spans in memory for testing and debugging."""

    def __init__(self, max_spans: int = 1000):
        self.spans: List[Span] = []
        self.max_spans = max_spans

    async def export(self, spans: List[Span]) -> None:
        """Store spans in memory."""
        self.spans.extend(spans)
        # Trim if needed
        if len(self.spans) > self.max_spans:
            self.spans = self.spans[-self.max_spans:]

    def clear(self) -> None:
        """Clear stored spans."""
        self.spans.clear()

    def get_spans_by_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace ID."""
        return [s for s in self.spans if s.trace_id == trace_id]

    def get_spans_by_name(self, name: str) -> List[Span]:
        """Get all spans with a given name."""
        return [s for s in self.spans if s.name == name]


@dataclass
class TracerConfig:
    """Configuration for the tracer.

    Attributes:
        service_name: Name of the service being traced.
        exporters: List of span exporters.
        sample_rate: Rate at which to sample traces (0.0 to 1.0).
        max_attributes: Maximum number of attributes per span.
        max_events: Maximum number of events per span.
    """

    service_name: str = "agents_framework"
    exporters: List[SpanExporter] = field(default_factory=list)
    sample_rate: float = 1.0
    max_attributes: int = 128
    max_events: int = 128


class Tracer:
    """Tracer for creating and managing spans.

    Provides an interface for creating distributed traces across
    agent execution, tool calls, and LLM interactions.
    """

    def __init__(self, config: Optional[TracerConfig] = None):
        """Initialize the tracer.

        Args:
            config: Optional tracer configuration.
        """
        self.config = config or TracerConfig()
        self._pending_spans: List[Span] = []

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: Optional[Union[Span, SpanContext]] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """Start a new span.

        Args:
            name: Name of the span.
            kind: Kind of span.
            parent: Optional parent span or context.
            attributes: Optional initial attributes.

        Returns:
            A new Span instance.
        """
        # Determine parent context
        if parent is None:
            current = get_current_span()
            if current is not None:
                parent_context = current.context.child_context()
            else:
                parent_context = SpanContext(
                    trace_id=str(uuid.uuid4()),
                    span_id=str(uuid.uuid4()),
                )
        elif isinstance(parent, Span):
            parent_context = parent.context.child_context()
        else:
            parent_context = parent.child_context()

        span = Span(
            name=name,
            context=parent_context,
            kind=kind,
            attributes=attributes or {},
        )

        # Set as current span
        _current_span.set(span)

        return span

    def start_as_current_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """Start a new span and set it as the current span.

        This is a convenience method that creates a span and
        automatically manages the current span context.

        Args:
            name: Name of the span.
            kind: Kind of span.
            attributes: Optional initial attributes.

        Returns:
            A new Span instance set as current.
        """
        return self.start_span(name, kind=kind, attributes=attributes)

    async def end_span(self, span: Span) -> None:
        """End a span and export it.

        Args:
            span: The span to end and export.
        """
        span.end()
        self._pending_spans.append(span)

        # Export if we have enough spans
        if len(self._pending_spans) >= 10:
            await self.flush()

    async def flush(self) -> None:
        """Flush pending spans to exporters."""
        if not self._pending_spans:
            return

        spans_to_export = self._pending_spans.copy()
        self._pending_spans.clear()

        for exporter in self.config.exporters:
            try:
                await exporter.export(spans_to_export)
            except Exception as e:
                # Log but don't raise - tracing should not break the app
                print(f"Error exporting spans: {e}")

    async def shutdown(self) -> None:
        """Shut down the tracer, flushing remaining spans."""
        await self.flush()
        for exporter in self.config.exporters:
            await exporter.shutdown()


# Global tracer instance
_tracer: Optional[Tracer] = None


def get_tracer(config: Optional[TracerConfig] = None) -> Tracer:
    """Get the global tracer instance.

    Args:
        config: Optional configuration for the tracer.

    Returns:
        The global Tracer instance.
    """
    global _tracer
    if _tracer is None:
        _tracer = Tracer(config)
    return _tracer


def get_current_span() -> Optional[Span]:
    """Get the current active span.

    Returns:
        The current Span if any, None otherwise.
    """
    return _current_span.get()


def set_current_span(span: Optional[Span]) -> None:
    """Set the current active span.

    Args:
        span: The span to set as current.
    """
    _current_span.set(span)


# Type variable for generic function decoration
F = TypeVar("F", bound=Callable[..., Any])


def trace(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """Decorator to trace a function.

    Args:
        name: Optional span name. Defaults to function name.
        kind: Kind of span.
        attributes: Optional attributes to add to the span.

    Returns:
        Decorated function with tracing.

    Example:
        @trace(name="process_data", kind=SpanKind.INTERNAL)
        async def process_data(data: str) -> str:
            return data.upper()
    """
    def decorator(func: F) -> F:
        span_name = name or func.__name__
        is_async = asyncio.iscoroutinefunction(func)

        if is_async:
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = get_tracer()
                span = tracer.start_span(
                    name=span_name,
                    kind=kind,
                    attributes=attributes,
                )
                try:
                    result = await func(*args, **kwargs)
                    span.set_ok()
                    return result
                except Exception as e:
                    span.record_exception(e, escaped=True)
                    raise
                finally:
                    await tracer.end_span(span)

            return async_wrapper  # type: ignore
        else:
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = get_tracer()
                span = tracer.start_span(
                    name=span_name,
                    kind=kind,
                    attributes=attributes,
                )
                try:
                    result = func(*args, **kwargs)
                    span.set_ok()
                    return result
                except Exception as e:
                    span.record_exception(e, escaped=True)
                    raise
                finally:
                    span.end()

            return sync_wrapper  # type: ignore

    return decorator
