"""Metrics collection for agent execution.

This module provides metrics collection capabilities for:
- Execution time tracking
- Token usage monitoring
- Counter, gauge, and histogram metrics
- Multiple exporters (console, file)
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union


class MetricType(str, Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class MetricPoint:
    """A single metric data point.

    Attributes:
        value: The metric value.
        timestamp: When the metric was recorded.
        labels: Labels/tags for the metric.
    """

    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricData:
    """Container for metric data and metadata.

    Attributes:
        name: Metric name.
        type: Type of metric (counter, gauge, histogram).
        description: Human-readable description.
        unit: Unit of measurement.
        points: List of recorded data points.
    """

    name: str
    type: MetricType
    description: str = ""
    unit: str = ""
    points: List[MetricPoint] = field(default_factory=list)

    def add_point(
        self,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Add a data point."""
        self.points.append(MetricPoint(value=value, labels=labels or {}))

    def get_latest(self) -> Optional[float]:
        """Get the latest recorded value."""
        if not self.points:
            return None
        return self.points[-1].value

    def get_sum(self) -> float:
        """Get the sum of all values."""
        return sum(p.value for p in self.points)

    def get_average(self) -> Optional[float]:
        """Get the average value."""
        if not self.points:
            return None
        return mean(p.value for p in self.points)

    def clear(self) -> None:
        """Clear all data points."""
        self.points.clear()


class Counter:
    """A monotonically increasing counter metric.

    Use counters for counting events like requests, errors, etc.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        labels: Optional[Dict[str, str]] = None,
    ):
        """Initialize the counter.

        Args:
            name: Metric name.
            description: Human-readable description.
            unit: Unit of measurement.
            labels: Default labels for this counter.
        """
        self.name = name
        self.description = description
        self.unit = unit
        self.default_labels = labels or {}
        self._value: float = 0.0
        self._label_values: Dict[str, float] = defaultdict(float)

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the counter.

        Args:
            value: Amount to increment by (must be positive).
            labels: Optional labels for this increment.
        """
        if value < 0:
            raise ValueError("Counter can only be incremented with positive values")

        self._value += value

        if labels:
            label_key = self._labels_to_key(labels)
            self._label_values[label_key] += value

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get the current counter value.

        Args:
            labels: Optional labels to filter by.

        Returns:
            Current counter value.
        """
        if labels:
            label_key = self._labels_to_key(labels)
            return self._label_values.get(label_key, 0.0)
        return self._value

    def reset(self) -> None:
        """Reset the counter to zero."""
        self._value = 0.0
        self._label_values.clear()

    def _labels_to_key(self, labels: Dict[str, str]) -> str:
        """Convert labels dict to a hashable key."""
        merged = {**self.default_labels, **labels}
        return json.dumps(merged, sort_keys=True)


class Gauge:
    """A metric that can go up and down.

    Use gauges for values that can increase or decrease,
    like memory usage, active connections, queue size, etc.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        labels: Optional[Dict[str, str]] = None,
    ):
        """Initialize the gauge.

        Args:
            name: Metric name.
            description: Human-readable description.
            unit: Unit of measurement.
            labels: Default labels for this gauge.
        """
        self.name = name
        self.description = description
        self.unit = unit
        self.default_labels = labels or {}
        self._value: float = 0.0
        self._label_values: Dict[str, float] = {}

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set the gauge value.

        Args:
            value: Value to set.
            labels: Optional labels for this value.
        """
        self._value = value
        if labels:
            label_key = self._labels_to_key(labels)
            self._label_values[label_key] = value

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the gauge value.

        Args:
            value: Amount to increment by.
            labels: Optional labels.
        """
        self._value += value
        if labels:
            label_key = self._labels_to_key(labels)
            self._label_values[label_key] = (
                self._label_values.get(label_key, 0.0) + value
            )

    def dec(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Decrement the gauge value.

        Args:
            value: Amount to decrement by.
            labels: Optional labels.
        """
        self._value -= value
        if labels:
            label_key = self._labels_to_key(labels)
            self._label_values[label_key] = (
                self._label_values.get(label_key, 0.0) - value
            )

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get the current gauge value.

        Args:
            labels: Optional labels to filter by.

        Returns:
            Current gauge value.
        """
        if labels:
            label_key = self._labels_to_key(labels)
            return self._label_values.get(label_key, 0.0)
        return self._value

    def _labels_to_key(self, labels: Dict[str, str]) -> str:
        """Convert labels dict to a hashable key."""
        merged = {**self.default_labels, **labels}
        return json.dumps(merged, sort_keys=True)


class Histogram:
    """A histogram for tracking value distributions.

    Use histograms for measuring things like request latency,
    response sizes, etc.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        buckets: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Initialize the histogram.

        Args:
            name: Metric name.
            description: Human-readable description.
            unit: Unit of measurement.
            buckets: Bucket boundaries for histogram.
            labels: Default labels for this histogram.
        """
        self.name = name
        self.description = description
        self.unit = unit
        self.default_labels = labels or {}

        # Default buckets for latency in seconds
        self.buckets = buckets or [
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
        ]

        self._values: List[float] = []
        self._bucket_counts: Dict[float, int] = {b: 0 for b in self.buckets}
        self._bucket_counts[float("inf")] = 0
        self._sum: float = 0.0
        self._count: int = 0

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a value.

        Args:
            value: Value to record.
            labels: Optional labels for this observation.
        """
        self._values.append(value)
        self._sum += value
        self._count += 1

        # Update bucket counts
        for bucket in sorted(self._bucket_counts.keys()):
            if value <= bucket:
                self._bucket_counts[bucket] += 1

    def get_count(self) -> int:
        """Get the total number of observations."""
        return self._count

    def get_sum(self) -> float:
        """Get the sum of all observations."""
        return self._sum

    def get_average(self) -> Optional[float]:
        """Get the average value."""
        if self._count == 0:
            return None
        return self._sum / self._count

    def get_median(self) -> Optional[float]:
        """Get the median value."""
        if not self._values:
            return None
        return median(self._values)

    def get_stddev(self) -> Optional[float]:
        """Get the standard deviation."""
        if len(self._values) < 2:
            return None
        return stdev(self._values)

    def get_percentile(self, percentile: float) -> Optional[float]:
        """Get a percentile value (0-100).

        Args:
            percentile: Percentile to calculate (0-100).

        Returns:
            The percentile value.
        """
        if not self._values:
            return None

        sorted_values = sorted(self._values)
        idx = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(idx, len(sorted_values) - 1)]

    def get_bucket_counts(self) -> Dict[float, int]:
        """Get bucket counts."""
        return self._bucket_counts.copy()

    def reset(self) -> None:
        """Reset the histogram."""
        self._values.clear()
        self._bucket_counts = {b: 0 for b in self.buckets}
        self._bucket_counts[float("inf")] = 0
        self._sum = 0.0
        self._count = 0


class MetricsExporter:
    """Base class for metrics exporters."""

    async def export(self, metrics: Dict[str, Any]) -> None:
        """Export metrics to the backend.

        Args:
            metrics: Dictionary of metric data to export.
        """
        raise NotImplementedError

    async def shutdown(self) -> None:
        """Shut down the exporter."""
        pass


class ConsoleMetricsExporter(MetricsExporter):
    """Exports metrics to the console for debugging."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    async def export(self, metrics: Dict[str, Any]) -> None:
        """Print metrics to console."""
        print("\n=== Metrics ===")
        for name, data in metrics.items():
            print(f"  {name}: {data}")
        print("===============\n")


class FileMetricsExporter(MetricsExporter):
    """Exports metrics to a file."""

    def __init__(
        self,
        path: Union[str, Path] = "metrics.jsonl",
        append: bool = True,
    ):
        self.path = Path(path)
        self.append = append

    async def export(self, metrics: Dict[str, Any]) -> None:
        """Write metrics to file as JSON lines."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self.append else "w"

        with open(self.path, mode) as f:
            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": metrics,
            }
            f.write(json.dumps(record) + "\n")


@dataclass
class MetricsConfig:
    """Configuration for the metrics collector.

    Attributes:
        service_name: Name of the service.
        exporters: List of metric exporters.
        export_interval: How often to export metrics (seconds).
        enable_runtime_metrics: Whether to collect Python runtime metrics.
    """

    service_name: str = "agents_framework"
    exporters: List[MetricsExporter] = field(default_factory=list)
    export_interval: float = 60.0
    enable_runtime_metrics: bool = True


class MetricsCollector:
    """Central metrics collector and registry.

    Manages all metrics for the agents framework, including
    counters, gauges, and histograms.
    """

    def __init__(self, config: Optional[MetricsConfig] = None):
        """Initialize the metrics collector.

        Args:
            config: Optional configuration.
        """
        self.config = config or MetricsConfig()
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._export_task: Optional[asyncio.Task[None]] = None

        # Pre-defined metrics for agents framework
        self._setup_default_metrics()

    def _setup_default_metrics(self) -> None:
        """Set up default metrics for the framework."""
        # Execution metrics
        self.create_counter(
            "agent_executions_total",
            "Total number of agent executions",
        )
        self.create_counter(
            "tool_calls_total",
            "Total number of tool calls",
        )
        self.create_counter(
            "llm_requests_total",
            "Total number of LLM API requests",
        )
        self.create_counter(
            "llm_errors_total",
            "Total number of LLM API errors",
        )

        # Token metrics
        self.create_counter(
            "tokens_input_total",
            "Total input tokens used",
            unit="tokens",
        )
        self.create_counter(
            "tokens_output_total",
            "Total output tokens used",
            unit="tokens",
        )

        # Latency metrics
        self.create_histogram(
            "agent_execution_duration_seconds",
            "Agent execution duration",
            unit="seconds",
        )
        self.create_histogram(
            "tool_execution_duration_seconds",
            "Tool execution duration",
            unit="seconds",
        )
        self.create_histogram(
            "llm_request_duration_seconds",
            "LLM API request duration",
            unit="seconds",
        )

        # Active counts
        self.create_gauge(
            "active_agents",
            "Number of currently active agents",
        )
        self.create_gauge(
            "active_tasks",
            "Number of currently active tasks",
        )

    def create_counter(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        labels: Optional[Dict[str, str]] = None,
    ) -> Counter:
        """Create and register a counter.

        Args:
            name: Counter name.
            description: Human-readable description.
            unit: Unit of measurement.
            labels: Default labels.

        Returns:
            The created Counter.
        """
        counter = Counter(name, description, unit, labels)
        self._counters[name] = counter
        return counter

    def create_gauge(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        labels: Optional[Dict[str, str]] = None,
    ) -> Gauge:
        """Create and register a gauge.

        Args:
            name: Gauge name.
            description: Human-readable description.
            unit: Unit of measurement.
            labels: Default labels.

        Returns:
            The created Gauge.
        """
        gauge = Gauge(name, description, unit, labels)
        self._gauges[name] = gauge
        return gauge

    def create_histogram(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        buckets: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> Histogram:
        """Create and register a histogram.

        Args:
            name: Histogram name.
            description: Human-readable description.
            unit: Unit of measurement.
            buckets: Bucket boundaries.
            labels: Default labels.

        Returns:
            The created Histogram.
        """
        histogram = Histogram(name, description, unit, buckets, labels)
        self._histograms[name] = histogram
        return histogram

    def get_counter(self, name: str) -> Optional[Counter]:
        """Get a counter by name."""
        return self._counters.get(name)

    def get_gauge(self, name: str) -> Optional[Gauge]:
        """Get a gauge by name."""
        return self._gauges.get(name)

    def get_histogram(self, name: str) -> Optional[Histogram]:
        """Get a histogram by name."""
        return self._histograms.get(name)

    def inc_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter by name.

        Args:
            name: Counter name.
            value: Amount to increment.
            labels: Optional labels.
        """
        counter = self._counters.get(name)
        if counter:
            counter.inc(value, labels)

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge value by name.

        Args:
            name: Gauge name.
            value: Value to set.
            labels: Optional labels.
        """
        gauge = self._gauges.get(name)
        if gauge:
            gauge.set(value, labels)

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a histogram observation by name.

        Args:
            name: Histogram name.
            value: Value to observe.
            labels: Optional labels.
        """
        histogram = self._histograms.get(name)
        if histogram:
            histogram.observe(value, labels)

    def record_tokens(
        self,
        input_tokens: int,
        output_tokens: int,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record token usage.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            labels: Optional labels.
        """
        self.inc_counter("tokens_input_total", float(input_tokens), labels)
        self.inc_counter("tokens_output_total", float(output_tokens), labels)

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary.

        Returns:
            Dictionary with all metric values.
        """
        metrics: Dict[str, Any] = {}

        # Counters
        for name, counter in self._counters.items():
            metrics[name] = counter.get()

        # Gauges
        for name, gauge in self._gauges.items():
            metrics[name] = gauge.get()

        # Histograms
        for name, histogram in self._histograms.items():
            metrics[name] = {
                "count": histogram.get_count(),
                "sum": histogram.get_sum(),
                "average": histogram.get_average(),
                "p50": histogram.get_percentile(50),
                "p95": histogram.get_percentile(95),
                "p99": histogram.get_percentile(99),
            }

        return metrics

    async def export(self) -> None:
        """Export all metrics to configured exporters."""
        metrics = self.get_all_metrics()

        for exporter in self.config.exporters:
            try:
                await exporter.export(metrics)
            except Exception as e:
                print(f"Error exporting metrics: {e}")

    async def start_periodic_export(self) -> None:
        """Start periodic metric export task."""
        if self._export_task is not None:
            return

        async def export_loop() -> None:
            while True:
                await asyncio.sleep(self.config.export_interval)
                await self.export()

        self._export_task = asyncio.create_task(export_loop())

    async def stop_periodic_export(self) -> None:
        """Stop periodic metric export task."""
        if self._export_task is not None:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass
            self._export_task = None

    async def shutdown(self) -> None:
        """Shut down the metrics collector."""
        await self.stop_periodic_export()
        await self.export()  # Final export
        for exporter in self.config.exporters:
            await exporter.shutdown()


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector(
    config: Optional[MetricsConfig] = None,
) -> MetricsCollector:
    """Get the global metrics collector instance.

    Args:
        config: Optional configuration for the collector.

    Returns:
        The global MetricsCollector instance.
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(config)
    return _metrics_collector


# Type variable for generic function decoration
F = TypeVar("F", bound=Callable[..., Any])


def timed(
    histogram_name: str,
    labels: Optional[Dict[str, str]] = None,
) -> Callable[[F], F]:
    """Decorator to time function execution.

    Args:
        histogram_name: Name of histogram to record to.
        labels: Optional labels for the observation.

    Returns:
        Decorated function with timing.

    Example:
        @timed("agent_execution_duration_seconds")
        async def process_task(task: Task) -> TaskResult:
            # ... processing
            return result
    """
    from functools import wraps

    def decorator(func: F) -> F:
        is_async = asyncio.iscoroutinefunction(func)

        if is_async:
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    duration = time.perf_counter() - start
                    collector = get_metrics_collector()
                    collector.observe_histogram(histogram_name, duration, labels)

            return async_wrapper  # type: ignore
        else:
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    duration = time.perf_counter() - start
                    collector = get_metrics_collector()
                    collector.observe_histogram(histogram_name, duration, labels)

            return sync_wrapper  # type: ignore

    return decorator
