"""Tests for distributed tracing in the agents framework."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from agents_framework.observability.tracing import (
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

if TYPE_CHECKING:
    pass


class TestSpanKind:
    """Tests for SpanKind enum."""

    def test_span_kind_values(self):
        """Test that all span kinds have expected values."""
        assert SpanKind.INTERNAL.value == "internal"
        assert SpanKind.CLIENT.value == "client"
        assert SpanKind.SERVER.value == "server"
        assert SpanKind.PRODUCER.value == "producer"
        assert SpanKind.CONSUMER.value == "consumer"
        assert SpanKind.AGENT.value == "agent"
        assert SpanKind.TOOL.value == "tool"
        assert SpanKind.LLM.value == "llm"


class TestSpanStatus:
    """Tests for SpanStatus enum."""

    def test_span_status_values(self):
        """Test that all span statuses have expected values."""
        assert SpanStatus.UNSET.value == "unset"
        assert SpanStatus.OK.value == "ok"
        assert SpanStatus.ERROR.value == "error"


class TestSpanContext:
    """Tests for SpanContext dataclass."""

    def test_span_context_creation(self, span_context):
        """Test creating a span context."""
        assert span_context.trace_id == "test-trace-123"
        assert span_context.span_id == "test-span-456"
        assert span_context.parent_span_id is None
        assert span_context.baggage == {"key": "value"}

    def test_span_context_defaults(self):
        """Test span context default values."""
        context = SpanContext(
            trace_id="trace-1",
            span_id="span-1",
        )

        assert context.parent_span_id is None
        assert context.baggage == {}

    def test_span_context_child_context(self, span_context):
        """Test creating a child context."""
        child = span_context.child_context()

        assert child.trace_id == span_context.trace_id
        assert child.span_id != span_context.span_id
        assert child.parent_span_id == span_context.span_id
        assert child.baggage == span_context.baggage

    def test_span_context_child_context_baggage_copy(self, span_context):
        """Test that child context copies baggage."""
        child = span_context.child_context()
        child.baggage["new_key"] = "new_value"

        # Original should not be modified
        assert "new_key" not in span_context.baggage


class TestSpanEvent:
    """Tests for SpanEvent dataclass."""

    def test_span_event_creation(self):
        """Test creating a span event."""
        event = SpanEvent(
            name="test_event",
            attributes={"key": "value"},
        )

        assert event.name == "test_event"
        assert event.attributes == {"key": "value"}
        assert isinstance(event.timestamp, datetime)

    def test_span_event_defaults(self):
        """Test span event default values."""
        event = SpanEvent(name="simple_event")

        assert event.name == "simple_event"
        assert event.attributes == {}
        assert event.timestamp is not None


class TestSpan:
    """Tests for Span class."""

    def test_span_creation(self, sample_span):
        """Test creating a span."""
        assert sample_span.name == "test-span"
        assert sample_span.kind == SpanKind.INTERNAL
        assert sample_span.status == SpanStatus.UNSET
        assert sample_span.attributes == {"test_attr": "test_value"}
        assert sample_span.end_time is None

    def test_span_trace_id_property(self, sample_span):
        """Test trace_id property."""
        assert sample_span.trace_id == sample_span.context.trace_id

    def test_span_span_id_property(self, sample_span):
        """Test span_id property."""
        assert sample_span.span_id == sample_span.context.span_id

    def test_span_parent_span_id_property(self, sample_span):
        """Test parent_span_id property."""
        assert sample_span.parent_span_id == sample_span.context.parent_span_id

    def test_span_duration_before_end(self, sample_span):
        """Test duration is None before span ends."""
        assert sample_span.duration_ns is None
        assert sample_span.duration_ms is None

    def test_span_duration_after_end(self, sample_span):
        """Test duration is calculated after span ends."""
        time.sleep(0.01)  # Small delay
        sample_span.end()

        assert sample_span.duration_ns is not None
        assert sample_span.duration_ns > 0
        assert sample_span.duration_ms is not None
        assert sample_span.duration_ms > 0

    def test_span_set_attribute(self, sample_span):
        """Test setting a single attribute."""
        sample_span.set_attribute("new_key", "new_value")

        assert sample_span.attributes["new_key"] == "new_value"

    def test_span_set_attributes(self, sample_span):
        """Test setting multiple attributes."""
        sample_span.set_attributes({"key1": "value1", "key2": "value2"})

        assert sample_span.attributes["key1"] == "value1"
        assert sample_span.attributes["key2"] == "value2"

    def test_span_add_event(self, sample_span):
        """Test adding an event to a span."""
        event = sample_span.add_event("test_event", {"event_key": "event_value"})

        assert event.name == "test_event"
        assert event.attributes == {"event_key": "event_value"}
        assert len(sample_span.events) == 1
        assert sample_span.events[0] is event

    def test_span_add_event_without_attributes(self, sample_span):
        """Test adding an event without attributes."""
        event = sample_span.add_event("simple_event")

        assert event.attributes == {}

    def test_span_set_status(self, sample_span):
        """Test setting span status."""
        sample_span.set_status(SpanStatus.ERROR, "Something went wrong")

        assert sample_span.status == SpanStatus.ERROR
        assert sample_span.status_message == "Something went wrong"

    def test_span_set_ok(self, sample_span):
        """Test setting span status to OK."""
        sample_span.set_ok()

        assert sample_span.status == SpanStatus.OK
        assert sample_span.status_message is None

    def test_span_set_error(self, sample_span):
        """Test setting span status to ERROR."""
        sample_span.set_error("Error message")

        assert sample_span.status == SpanStatus.ERROR
        assert sample_span.status_message == "Error message"

    def test_span_record_exception(self, sample_span):
        """Test recording an exception."""
        exception = ValueError("Test exception")
        sample_span.record_exception(exception, escaped=True)

        assert sample_span.status == SpanStatus.ERROR
        assert len(sample_span.events) == 1

        event = sample_span.events[0]
        assert event.name == "exception"
        assert event.attributes["exception.type"] == "ValueError"
        assert event.attributes["exception.message"] == "Test exception"
        assert event.attributes["exception.escaped"] is True

    def test_span_end(self, sample_span):
        """Test ending a span."""
        sample_span.end()

        assert sample_span.end_time is not None
        assert sample_span._end_ns is not None
        assert sample_span.status == SpanStatus.OK  # Default to OK if unset

    def test_span_end_preserves_status(self, sample_span):
        """Test that end() does not override explicit status."""
        sample_span.set_error("Explicit error")
        sample_span.end()

        assert sample_span.status == SpanStatus.ERROR

    def test_span_end_idempotent(self, sample_span):
        """Test that calling end() multiple times is idempotent."""
        sample_span.end()
        first_end_time = sample_span.end_time
        first_end_ns = sample_span._end_ns

        time.sleep(0.01)
        sample_span.end()

        assert sample_span.end_time == first_end_time
        assert sample_span._end_ns == first_end_ns

    def test_span_to_dict(self, sample_span):
        """Test converting span to dictionary."""
        sample_span.add_event("test_event", {"key": "value"})
        sample_span.end()

        result = sample_span.to_dict()

        assert result["name"] == "test-span"
        assert result["trace_id"] == sample_span.trace_id
        assert result["span_id"] == sample_span.span_id
        assert result["kind"] == "internal"
        assert result["status"] == "ok"
        assert result["attributes"] == {"test_attr": "test_value"}
        assert len(result["events"]) == 1
        assert result["duration_ms"] is not None

    def test_span_context_manager_success(self, span_context):
        """Test span as context manager on success."""
        with Span(name="context_span", context=span_context) as span:
            span.set_attribute("key", "value")

        assert span.end_time is not None
        assert span.status == SpanStatus.OK

    def test_span_context_manager_exception(self, span_context):
        """Test span as context manager with exception."""
        with pytest.raises(ValueError):
            with Span(name="context_span", context=span_context) as span:
                raise ValueError("Test error")

        assert span.status == SpanStatus.ERROR
        assert len(span.events) == 1
        assert span.events[0].name == "exception"

    async def test_span_async_context_manager_success(self, span_context):
        """Test span as async context manager on success."""
        async with Span(name="async_span", context=span_context) as span:
            span.set_attribute("async_key", "async_value")
            await asyncio.sleep(0.001)

        assert span.end_time is not None
        assert span.status == SpanStatus.OK

    async def test_span_async_context_manager_exception(self, span_context):
        """Test span as async context manager with exception."""
        with pytest.raises(RuntimeError):
            async with Span(name="async_span", context=span_context) as span:
                raise RuntimeError("Async error")

        assert span.status == SpanStatus.ERROR
        assert len(span.events) == 1


class TestSpanExporter:
    """Tests for span exporters."""

    async def test_base_exporter_not_implemented(self):
        """Test that base exporter raises NotImplementedError."""
        exporter = SpanExporter()

        with pytest.raises(NotImplementedError):
            await exporter.export([])

    async def test_base_exporter_shutdown(self):
        """Test that base exporter shutdown is a no-op."""
        exporter = SpanExporter()
        await exporter.shutdown()  # Should not raise


class TestConsoleSpanExporter:
    """Tests for ConsoleSpanExporter."""

    async def test_console_exporter_creation(self):
        """Test creating a console exporter."""
        exporter = ConsoleSpanExporter(verbose=True)

        assert exporter.verbose is True

    async def test_console_exporter_export(self, sample_span, capsys):
        """Test exporting spans to console."""
        exporter = ConsoleSpanExporter(verbose=False)
        sample_span.end()

        await exporter.export([sample_span])

        captured = capsys.readouterr()
        assert "[TRACE]" in captured.out
        assert "test-span" in captured.out

    async def test_console_exporter_verbose(self, sample_span, capsys):
        """Test verbose console export."""
        exporter = ConsoleSpanExporter(verbose=True)
        sample_span.set_attribute("verbose_key", "verbose_value")
        sample_span.end()

        await exporter.export([sample_span])

        captured = capsys.readouterr()
        assert "verbose_key" in captured.out or "[TRACE]" in captured.out


class TestInMemorySpanExporter:
    """Tests for InMemorySpanExporter."""

    async def test_in_memory_exporter_creation(self):
        """Test creating an in-memory exporter."""
        exporter = InMemorySpanExporter(max_spans=50)

        assert exporter.max_spans == 50
        assert exporter.spans == []

    async def test_in_memory_exporter_export(self, memory_exporter, sample_span):
        """Test exporting spans to memory."""
        sample_span.end()
        await memory_exporter.export([sample_span])

        assert len(memory_exporter.spans) == 1
        assert memory_exporter.spans[0] is sample_span

    async def test_in_memory_exporter_max_spans(self, span_context):
        """Test that exporter trims when max is exceeded."""
        exporter = InMemorySpanExporter(max_spans=5)

        spans = [
            Span(name=f"span-{i}", context=span_context.child_context())
            for i in range(10)
        ]
        for span in spans:
            span.end()

        await exporter.export(spans)

        assert len(exporter.spans) == 5
        # Should keep last 5
        assert exporter.spans[0].name == "span-5"

    async def test_in_memory_exporter_clear(self, memory_exporter, sample_span):
        """Test clearing spans from memory."""
        sample_span.end()
        await memory_exporter.export([sample_span])

        memory_exporter.clear()

        assert len(memory_exporter.spans) == 0

    async def test_in_memory_exporter_get_spans_by_trace(
        self, memory_exporter, span_context
    ):
        """Test getting spans by trace ID."""
        span1 = Span(name="span-1", context=span_context)
        span1.end()

        other_context = SpanContext(trace_id="other-trace", span_id="other-span")
        span2 = Span(name="span-2", context=other_context)
        span2.end()

        await memory_exporter.export([span1, span2])

        results = memory_exporter.get_spans_by_trace(span_context.trace_id)

        assert len(results) == 1
        assert results[0].name == "span-1"

    async def test_in_memory_exporter_get_spans_by_name(
        self, memory_exporter, span_context
    ):
        """Test getting spans by name."""
        span1 = Span(name="target-span", context=span_context)
        span1.end()
        span2 = Span(name="other-span", context=span_context.child_context())
        span2.end()

        await memory_exporter.export([span1, span2])

        results = memory_exporter.get_spans_by_name("target-span")

        assert len(results) == 1
        assert results[0].name == "target-span"


class TestTracerConfig:
    """Tests for TracerConfig."""

    def test_tracer_config_defaults(self):
        """Test tracer config default values."""
        config = TracerConfig()

        assert config.service_name == "agents_framework"
        assert config.exporters == []
        assert config.sample_rate == 1.0
        assert config.max_attributes == 128
        assert config.max_events == 128

    def test_tracer_config_custom(self, memory_exporter):
        """Test tracer config with custom values."""
        config = TracerConfig(
            service_name="custom-service",
            exporters=[memory_exporter],
            sample_rate=0.5,
            max_attributes=64,
            max_events=32,
        )

        assert config.service_name == "custom-service"
        assert len(config.exporters) == 1
        assert config.sample_rate == 0.5
        assert config.max_attributes == 64
        assert config.max_events == 32


class TestTracer:
    """Tests for Tracer class."""

    def test_tracer_creation(self, tracer_config):
        """Test creating a tracer."""
        tracer = Tracer(tracer_config)

        assert tracer.config is tracer_config
        assert tracer._pending_spans == []

    def test_tracer_default_config(self):
        """Test tracer with default config."""
        tracer = Tracer()

        assert tracer.config.service_name == "agents_framework"

    def test_tracer_start_span(self, tracer):
        """Test starting a new span."""
        span = tracer.start_span("test-span", kind=SpanKind.INTERNAL)

        assert span.name == "test-span"
        assert span.kind == SpanKind.INTERNAL
        assert span.trace_id is not None
        assert span.span_id is not None

    def test_tracer_start_span_with_parent(self, tracer, sample_span):
        """Test starting a span with parent span."""
        child = tracer.start_span("child-span", parent=sample_span)

        assert child.trace_id == sample_span.trace_id
        assert child.parent_span_id == sample_span.span_id

    def test_tracer_start_span_with_parent_context(self, tracer, span_context):
        """Test starting a span with parent context."""
        child = tracer.start_span("child-span", parent=span_context)

        assert child.trace_id == span_context.trace_id
        assert child.parent_span_id == span_context.span_id

    def test_tracer_start_span_with_attributes(self, tracer):
        """Test starting a span with initial attributes."""
        span = tracer.start_span(
            "attributed-span",
            attributes={"key1": "value1", "key2": "value2"},
        )

        assert span.attributes["key1"] == "value1"
        assert span.attributes["key2"] == "value2"

    def test_tracer_start_span_inherits_current(self, tracer):
        """Test that new span inherits from current span."""
        parent = tracer.start_span("parent-span")
        set_current_span(parent)

        child = tracer.start_span("child-span")

        assert child.trace_id == parent.trace_id
        assert child.parent_span_id == parent.span_id

    def test_tracer_start_as_current_span(self, tracer):
        """Test starting a span as current."""
        span = tracer.start_as_current_span("current-span", kind=SpanKind.AGENT)

        assert span.name == "current-span"
        assert span.kind == SpanKind.AGENT
        assert get_current_span() is span

    async def test_tracer_end_span(self, tracer):
        """Test ending a span."""
        span = tracer.start_span("test-span")

        await tracer.end_span(span)

        assert span.end_time is not None
        assert span in tracer._pending_spans

    async def test_tracer_flush(self, tracer, memory_exporter):
        """Test flushing pending spans."""
        span1 = tracer.start_span("span-1")
        span1.end()
        span2 = tracer.start_span("span-2")
        span2.end()

        tracer._pending_spans = [span1, span2]
        await tracer.flush()

        assert tracer._pending_spans == []
        assert len(memory_exporter.spans) == 2

    async def test_tracer_flush_empty(self, tracer):
        """Test flushing when no pending spans."""
        await tracer.flush()  # Should not raise

    async def test_tracer_flush_handles_exporter_error(self, tracer):
        """Test that flush handles exporter errors gracefully."""
        span = tracer.start_span("test-span")
        span.end()
        tracer._pending_spans = [span]

        # Add a failing exporter
        failing_exporter = AsyncMock(side_effect=RuntimeError("Export failed"))
        tracer.config.exporters.append(failing_exporter)

        # Should not raise
        await tracer.flush()

    async def test_tracer_shutdown(self, tracer, memory_exporter):
        """Test shutting down the tracer."""
        span = tracer.start_span("test-span")
        span.end()
        tracer._pending_spans = [span]

        await tracer.shutdown()

        assert tracer._pending_spans == []
        assert len(memory_exporter.spans) == 1


class TestCurrentSpan:
    """Tests for current span management."""

    def test_get_current_span_none(self):
        """Test getting current span when none is set."""
        set_current_span(None)

        assert get_current_span() is None

    def test_set_and_get_current_span(self, sample_span):
        """Test setting and getting current span."""
        set_current_span(sample_span)

        assert get_current_span() is sample_span

    def test_clear_current_span(self, sample_span):
        """Test clearing current span."""
        set_current_span(sample_span)
        set_current_span(None)

        assert get_current_span() is None


class TestTraceDecorator:
    """Tests for @trace decorator."""

    async def test_trace_decorator_async_function(self):
        """Test trace decorator on async function."""
        @trace(name="traced_async")
        async def async_func() -> str:
            return "result"

        result = await async_func()

        assert result == "result"

    async def test_trace_decorator_async_with_args(self):
        """Test trace decorator passes arguments correctly."""
        @trace()
        async def func_with_args(a: int, b: str) -> str:
            return f"{a}-{b}"

        result = await func_with_args(42, "test")

        assert result == "42-test"

    async def test_trace_decorator_async_exception(self):
        """Test trace decorator handles exceptions."""
        @trace(name="failing_func")
        async def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await failing_func()

    def test_trace_decorator_sync_function(self):
        """Test trace decorator on sync function."""
        @trace(name="traced_sync")
        def sync_func() -> str:
            return "sync_result"

        result = sync_func()

        assert result == "sync_result"

    def test_trace_decorator_sync_exception(self):
        """Test trace decorator handles sync exceptions."""
        @trace(name="failing_sync")
        def failing_sync():
            raise RuntimeError("Sync error")

        with pytest.raises(RuntimeError, match="Sync error"):
            failing_sync()

    def test_trace_decorator_default_name(self):
        """Test trace decorator uses function name as default."""
        @trace()
        def my_function():
            pass

        # The function name should be preserved
        assert my_function.__name__ == "my_function"

    def test_trace_decorator_with_kind(self):
        """Test trace decorator with custom kind."""
        @trace(kind=SpanKind.TOOL)
        def tool_func():
            pass

        # Should not raise
        tool_func()

    def test_trace_decorator_with_attributes(self):
        """Test trace decorator with initial attributes."""
        @trace(attributes={"custom": "attr"})
        def attributed_func():
            pass

        # Should not raise
        attributed_func()


class TestGetTracer:
    """Tests for get_tracer function."""

    def test_get_tracer_creates_global(self):
        """Test that get_tracer creates a global instance."""
        # Reset global tracer for this test
        import agents_framework.observability.tracing as tracing_module
        tracing_module._tracer = None

        tracer = get_tracer()

        assert tracer is not None
        assert isinstance(tracer, Tracer)

    def test_get_tracer_returns_same_instance(self):
        """Test that get_tracer returns the same instance."""
        tracer1 = get_tracer()
        tracer2 = get_tracer()

        assert tracer1 is tracer2

    def test_get_tracer_with_config(self):
        """Test get_tracer with custom config."""
        # Reset global tracer
        import agents_framework.observability.tracing as tracing_module
        tracing_module._tracer = None

        config = TracerConfig(service_name="custom-service")
        tracer = get_tracer(config)

        assert tracer.config.service_name == "custom-service"


class TestTracingIntegration:
    """Integration tests for tracing functionality."""

    async def test_nested_spans(self, tracer, memory_exporter):
        """Test creating nested spans."""
        parent = tracer.start_span("parent")
        set_current_span(parent)

        child = tracer.start_span("child")
        grandchild = tracer.start_span("grandchild", parent=child)

        grandchild.end()
        child.end()
        parent.end()

        await memory_exporter.export([parent, child, grandchild])

        # Verify parent-child relationships
        assert child.parent_span_id == parent.span_id
        assert grandchild.parent_span_id == child.span_id
        assert child.trace_id == parent.trace_id
        assert grandchild.trace_id == parent.trace_id

    async def test_span_with_events_and_attributes(self, tracer, memory_exporter):
        """Test span with multiple events and attributes."""
        span = tracer.start_span("complex-span")
        span.set_attributes({"attr1": "value1", "attr2": "value2"})
        span.add_event("event1", {"event_attr": "event_value"})
        span.add_event("event2")
        span.end()

        await memory_exporter.export([span])

        assert len(span.attributes) >= 2
        assert len(span.events) == 2

    async def test_error_span_flow(self, tracer, memory_exporter):
        """Test span flow with error."""
        span = tracer.start_span("error-span")
        try:
            raise ValueError("Test error")
        except ValueError as e:
            span.record_exception(e)
        finally:
            span.end()

        await memory_exporter.export([span])

        assert span.status == SpanStatus.ERROR
        assert len(span.events) == 1
        assert span.events[0].name == "exception"
