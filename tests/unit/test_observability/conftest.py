"""Local fixtures for observability tests."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from agents_framework.observability.logging import (
    AgentLogger,
    LogConfig,
    LogContext,
    LogLevel,
    clear_context,
    clear_global_context,
)
from agents_framework.observability.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsCollector,
    MetricsConfig,
    MetricsExporter,
)
from agents_framework.observability.tracing import (
    InMemorySpanExporter,
    Span,
    SpanContext,
    SpanExporter,
    SpanKind,
    Tracer,
    TracerConfig,
    set_current_span,
)


# ============================================================================
# Logging Fixtures
# ============================================================================


@pytest.fixture
def log_config() -> LogConfig:
    """Create a basic log configuration for testing."""
    return LogConfig(
        level=LogLevel.DEBUG,
        include_timestamp=True,
        include_caller=False,
        json_format=False,
    )


@pytest.fixture
def agent_logger(log_config: LogConfig) -> AgentLogger:
    """Create an agent logger for testing."""
    return AgentLogger("test_logger", log_config)


@pytest.fixture
def log_context() -> LogContext:
    """Create a sample log context."""
    return LogContext(
        correlation_id="test-correlation-123",
        agent_id="test-agent-1",
        task_id="test-task-456",
        session_id="test-session-789",
        extra={"environment": "test"},
    )


@pytest.fixture(autouse=True)
def cleanup_logging_context():
    """Clean up logging context after each test."""
    yield
    clear_context()
    clear_global_context()


# ============================================================================
# Tracing Fixtures
# ============================================================================


@pytest.fixture
def span_context() -> SpanContext:
    """Create a sample span context."""
    return SpanContext(
        trace_id="test-trace-123",
        span_id="test-span-456",
        parent_span_id=None,
        baggage={"key": "value"},
    )


@pytest.fixture
def sample_span(span_context: SpanContext) -> Span:
    """Create a sample span."""
    return Span(
        name="test-span",
        context=span_context,
        kind=SpanKind.INTERNAL,
        attributes={"test_attr": "test_value"},
    )


@pytest.fixture
def memory_exporter() -> InMemorySpanExporter:
    """Create an in-memory span exporter for testing."""
    return InMemorySpanExporter(max_spans=100)


@pytest.fixture
def tracer_config(memory_exporter: InMemorySpanExporter) -> TracerConfig:
    """Create a tracer configuration with in-memory exporter."""
    return TracerConfig(
        service_name="test-service",
        exporters=[memory_exporter],
        sample_rate=1.0,
    )


@pytest.fixture
def tracer(tracer_config: TracerConfig) -> Tracer:
    """Create a tracer for testing."""
    return Tracer(tracer_config)


@pytest.fixture(autouse=True)
def cleanup_current_span():
    """Clean up current span after each test."""
    yield
    set_current_span(None)


# ============================================================================
# Metrics Fixtures
# ============================================================================


class MockMetricsExporter(MetricsExporter):
    """Mock metrics exporter for testing."""

    def __init__(self):
        self.exported_metrics: List[Dict[str, Any]] = []
        self.export_count = 0
        self.shutdown_called = False

    async def export(self, metrics: Dict[str, Any]) -> None:
        """Store exported metrics."""
        self.exported_metrics.append(metrics.copy())
        self.export_count += 1

    async def shutdown(self) -> None:
        """Mark shutdown as called."""
        self.shutdown_called = True


@pytest.fixture
def mock_exporter() -> MockMetricsExporter:
    """Create a mock metrics exporter."""
    return MockMetricsExporter()


@pytest.fixture
def metrics_config(mock_exporter: MockMetricsExporter) -> MetricsConfig:
    """Create a metrics configuration for testing."""
    return MetricsConfig(
        service_name="test-service",
        exporters=[mock_exporter],
        export_interval=1.0,
        enable_runtime_metrics=False,
    )


@pytest.fixture
def metrics_collector(metrics_config: MetricsConfig) -> MetricsCollector:
    """Create a metrics collector for testing."""
    return MetricsCollector(metrics_config)


@pytest.fixture
def counter() -> Counter:
    """Create a test counter."""
    return Counter(
        name="test_counter",
        description="Test counter for unit tests",
        unit="count",
    )


@pytest.fixture
def gauge() -> Gauge:
    """Create a test gauge."""
    return Gauge(
        name="test_gauge",
        description="Test gauge for unit tests",
        unit="items",
    )


@pytest.fixture
def histogram() -> Histogram:
    """Create a test histogram."""
    return Histogram(
        name="test_histogram",
        description="Test histogram for unit tests",
        unit="seconds",
        buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    )
