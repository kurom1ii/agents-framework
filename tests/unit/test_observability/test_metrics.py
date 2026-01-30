"""Tests for metrics collection in the agents framework."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from agents_framework.observability.metrics import (
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

if TYPE_CHECKING:
    from .conftest import MockMetricsExporter


class TestMetricType:
    """Tests for MetricType enum."""

    def test_metric_type_values(self):
        """Test that all metric types have expected values."""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"


class TestMetricPoint:
    """Tests for MetricPoint dataclass."""

    def test_metric_point_creation(self):
        """Test creating a metric point."""
        point = MetricPoint(
            value=42.5,
            labels={"env": "test"},
        )

        assert point.value == 42.5
        assert point.labels == {"env": "test"}
        assert point.timestamp is not None

    def test_metric_point_defaults(self):
        """Test metric point default values."""
        point = MetricPoint(value=10.0)

        assert point.value == 10.0
        assert point.labels == {}
        assert point.timestamp is not None


class TestMetricData:
    """Tests for MetricData class."""

    def test_metric_data_creation(self):
        """Test creating metric data."""
        data = MetricData(
            name="test_metric",
            type=MetricType.COUNTER,
            description="A test metric",
            unit="count",
        )

        assert data.name == "test_metric"
        assert data.type == MetricType.COUNTER
        assert data.description == "A test metric"
        assert data.unit == "count"
        assert data.points == []

    def test_metric_data_add_point(self):
        """Test adding a data point."""
        data = MetricData(name="test", type=MetricType.GAUGE)
        data.add_point(42.0, {"label": "value"})

        assert len(data.points) == 1
        assert data.points[0].value == 42.0
        assert data.points[0].labels == {"label": "value"}

    def test_metric_data_add_point_no_labels(self):
        """Test adding a point without labels."""
        data = MetricData(name="test", type=MetricType.GAUGE)
        data.add_point(10.0)

        assert data.points[0].labels == {}

    def test_metric_data_get_latest_empty(self):
        """Test get_latest with no points."""
        data = MetricData(name="test", type=MetricType.GAUGE)

        assert data.get_latest() is None

    def test_metric_data_get_latest(self):
        """Test get_latest returns last value."""
        data = MetricData(name="test", type=MetricType.GAUGE)
        data.add_point(1.0)
        data.add_point(2.0)
        data.add_point(3.0)

        assert data.get_latest() == 3.0

    def test_metric_data_get_sum(self):
        """Test get_sum calculates correctly."""
        data = MetricData(name="test", type=MetricType.COUNTER)
        data.add_point(10.0)
        data.add_point(20.0)
        data.add_point(30.0)

        assert data.get_sum() == 60.0

    def test_metric_data_get_sum_empty(self):
        """Test get_sum with no points."""
        data = MetricData(name="test", type=MetricType.COUNTER)

        assert data.get_sum() == 0.0

    def test_metric_data_get_average(self):
        """Test get_average calculates correctly."""
        data = MetricData(name="test", type=MetricType.GAUGE)
        data.add_point(10.0)
        data.add_point(20.0)
        data.add_point(30.0)

        assert data.get_average() == 20.0

    def test_metric_data_get_average_empty(self):
        """Test get_average with no points."""
        data = MetricData(name="test", type=MetricType.GAUGE)

        assert data.get_average() is None

    def test_metric_data_clear(self):
        """Test clearing data points."""
        data = MetricData(name="test", type=MetricType.GAUGE)
        data.add_point(1.0)
        data.add_point(2.0)

        data.clear()

        assert data.points == []


class TestCounter:
    """Tests for Counter metric."""

    def test_counter_creation(self, counter):
        """Test creating a counter."""
        assert counter.name == "test_counter"
        assert counter.description == "Test counter for unit tests"
        assert counter.unit == "count"
        assert counter.get() == 0.0

    def test_counter_inc_default(self, counter):
        """Test incrementing by default (1)."""
        counter.inc()

        assert counter.get() == 1.0

    def test_counter_inc_custom_value(self, counter):
        """Test incrementing by custom value."""
        counter.inc(5.0)
        counter.inc(3.0)

        assert counter.get() == 8.0

    def test_counter_inc_negative_raises(self, counter):
        """Test that incrementing by negative value raises."""
        with pytest.raises(ValueError, match="positive"):
            counter.inc(-1.0)

    def test_counter_with_labels(self):
        """Test counter with labels."""
        counter = Counter(name="labeled_counter", labels={"env": "test"})

        counter.inc(1.0, {"method": "GET"})
        counter.inc(2.0, {"method": "POST"})
        counter.inc(3.0, {"method": "GET"})

        assert counter.get({"method": "GET"}) == 4.0
        assert counter.get({"method": "POST"}) == 2.0
        assert counter.get() == 6.0  # Total

    def test_counter_labels_merge_with_defaults(self):
        """Test that labels merge with default labels."""
        counter = Counter(name="test", labels={"env": "test"})

        counter.inc(1.0, {"method": "GET"})

        # Label key should include default labels
        label_key = counter._labels_to_key({"method": "GET"})
        expected = json.dumps({"env": "test", "method": "GET"}, sort_keys=True)
        assert label_key == expected

    def test_counter_reset(self, counter):
        """Test resetting a counter."""
        counter.inc(10.0)
        counter.inc(5.0, {"label": "value"})

        counter.reset()

        assert counter.get() == 0.0
        assert counter.get({"label": "value"}) == 0.0


class TestGauge:
    """Tests for Gauge metric."""

    def test_gauge_creation(self, gauge):
        """Test creating a gauge."""
        assert gauge.name == "test_gauge"
        assert gauge.description == "Test gauge for unit tests"
        assert gauge.unit == "items"
        assert gauge.get() == 0.0

    def test_gauge_set(self, gauge):
        """Test setting gauge value."""
        gauge.set(42.0)

        assert gauge.get() == 42.0

    def test_gauge_set_with_labels(self, gauge):
        """Test setting gauge with labels."""
        gauge.set(10.0, {"region": "us-east"})
        gauge.set(20.0, {"region": "us-west"})

        assert gauge.get({"region": "us-east"}) == 10.0
        assert gauge.get({"region": "us-west"}) == 20.0

    def test_gauge_inc(self, gauge):
        """Test incrementing gauge."""
        gauge.set(10.0)
        gauge.inc(5.0)

        assert gauge.get() == 15.0

    def test_gauge_inc_with_labels(self, gauge):
        """Test incrementing gauge with labels."""
        gauge.set(10.0, {"type": "a"})
        gauge.inc(5.0, {"type": "a"})

        assert gauge.get({"type": "a"}) == 15.0

    def test_gauge_dec(self, gauge):
        """Test decrementing gauge."""
        gauge.set(10.0)
        gauge.dec(3.0)

        assert gauge.get() == 7.0

    def test_gauge_dec_with_labels(self, gauge):
        """Test decrementing gauge with labels."""
        gauge.set(10.0, {"type": "a"})
        gauge.dec(3.0, {"type": "a"})

        assert gauge.get({"type": "a"}) == 7.0

    def test_gauge_can_go_negative(self, gauge):
        """Test that gauge can go negative."""
        gauge.set(5.0)
        gauge.dec(10.0)

        assert gauge.get() == -5.0


class TestHistogram:
    """Tests for Histogram metric."""

    def test_histogram_creation(self, histogram):
        """Test creating a histogram."""
        assert histogram.name == "test_histogram"
        assert histogram.description == "Test histogram for unit tests"
        assert histogram.unit == "seconds"
        assert histogram.buckets == [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

    def test_histogram_default_buckets(self):
        """Test histogram has default buckets."""
        h = Histogram(name="default_buckets")

        expected = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        assert h.buckets == expected

    def test_histogram_observe(self, histogram):
        """Test observing values."""
        histogram.observe(0.1)
        histogram.observe(0.2)
        histogram.observe(0.3)

        assert histogram.get_count() == 3
        assert histogram.get_sum() == pytest.approx(0.6)

    def test_histogram_get_average(self, histogram):
        """Test getting average."""
        histogram.observe(10.0)
        histogram.observe(20.0)
        histogram.observe(30.0)

        assert histogram.get_average() == 20.0

    def test_histogram_get_average_empty(self, histogram):
        """Test getting average when empty."""
        assert histogram.get_average() is None

    def test_histogram_get_median(self, histogram):
        """Test getting median."""
        for val in [1, 2, 3, 4, 5]:
            histogram.observe(float(val))

        assert histogram.get_median() == 3.0

    def test_histogram_get_median_empty(self, histogram):
        """Test getting median when empty."""
        assert histogram.get_median() is None

    def test_histogram_get_stddev(self, histogram):
        """Test getting standard deviation."""
        for val in [2, 4, 4, 4, 5, 5, 7, 9]:
            histogram.observe(float(val))

        stddev = histogram.get_stddev()
        assert stddev is not None
        # Python's statistics.stdev uses sample std dev (n-1 denominator)
        assert stddev == pytest.approx(2.14, abs=0.1)

    def test_histogram_get_stddev_empty(self, histogram):
        """Test getting stddev when empty."""
        assert histogram.get_stddev() is None

    def test_histogram_get_stddev_single_value(self, histogram):
        """Test getting stddev with single value."""
        histogram.observe(5.0)

        assert histogram.get_stddev() is None

    def test_histogram_get_percentile(self, histogram):
        """Test getting percentiles."""
        for i in range(1, 101):
            histogram.observe(float(i))

        p50 = histogram.get_percentile(50)
        p95 = histogram.get_percentile(95)
        p99 = histogram.get_percentile(99)

        assert p50 is not None
        assert p50 == pytest.approx(50, abs=2)
        assert p95 is not None
        assert p95 == pytest.approx(95, abs=2)
        assert p99 is not None
        assert p99 == pytest.approx(99, abs=2)

    def test_histogram_get_percentile_empty(self, histogram):
        """Test getting percentile when empty."""
        assert histogram.get_percentile(50) is None

    def test_histogram_bucket_counts(self, histogram):
        """Test bucket counts are updated."""
        # Buckets: [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, inf]
        histogram.observe(0.005)  # <= 0.01
        histogram.observe(0.02)   # <= 0.05
        histogram.observe(0.08)   # <= 0.1
        histogram.observe(0.3)    # <= 0.5
        histogram.observe(10.0)   # <= inf

        buckets = histogram.get_bucket_counts()

        assert buckets[0.01] >= 1
        assert buckets[0.05] >= 2
        assert buckets[0.1] >= 3
        assert buckets[0.5] >= 4
        assert buckets[float("inf")] == 5

    def test_histogram_reset(self, histogram):
        """Test resetting histogram."""
        histogram.observe(1.0)
        histogram.observe(2.0)

        histogram.reset()

        assert histogram.get_count() == 0
        assert histogram.get_sum() == 0.0
        assert histogram.get_average() is None


class TestMetricsExporter:
    """Tests for metrics exporters."""

    async def test_base_exporter_not_implemented(self):
        """Test that base exporter raises NotImplementedError."""
        exporter = MetricsExporter()

        with pytest.raises(NotImplementedError):
            await exporter.export({})

    async def test_base_exporter_shutdown(self):
        """Test that base exporter shutdown is a no-op."""
        exporter = MetricsExporter()
        await exporter.shutdown()  # Should not raise


class TestConsoleMetricsExporter:
    """Tests for ConsoleMetricsExporter."""

    async def test_console_exporter_creation(self):
        """Test creating a console exporter."""
        exporter = ConsoleMetricsExporter(verbose=True)

        assert exporter.verbose is True

    async def test_console_exporter_export(self, capsys):
        """Test exporting metrics to console."""
        exporter = ConsoleMetricsExporter()

        await exporter.export({"test_metric": 42})

        captured = capsys.readouterr()
        assert "Metrics" in captured.out
        assert "test_metric" in captured.out


class TestFileMetricsExporter:
    """Tests for FileMetricsExporter."""

    async def test_file_exporter_creation(self):
        """Test creating a file exporter."""
        exporter = FileMetricsExporter(path="/tmp/test.jsonl", append=False)

        assert exporter.path == Path("/tmp/test.jsonl")
        assert exporter.append is False

    async def test_file_exporter_export(self):
        """Test exporting metrics to file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            exporter = FileMetricsExporter(path=path, append=True)

            await exporter.export({"metric1": 10, "metric2": 20})
            await exporter.export({"metric1": 15})

            with open(path) as f:
                lines = f.readlines()

            assert len(lines) == 2

            record1 = json.loads(lines[0])
            assert "timestamp" in record1
            assert record1["metrics"]["metric1"] == 10

            record2 = json.loads(lines[1])
            assert record2["metrics"]["metric1"] == 15
        finally:
            Path(path).unlink(missing_ok=True)

    async def test_file_exporter_creates_parent_dirs(self):
        """Test that file exporter creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "metrics.jsonl"
            exporter = FileMetricsExporter(path=path)

            await exporter.export({"test": 1})

            assert path.exists()


class TestMetricsConfig:
    """Tests for MetricsConfig."""

    def test_metrics_config_defaults(self):
        """Test metrics config default values."""
        config = MetricsConfig()

        assert config.service_name == "agents_framework"
        assert config.exporters == []
        assert config.export_interval == 60.0
        assert config.enable_runtime_metrics is True

    def test_metrics_config_custom(self, mock_exporter):
        """Test metrics config with custom values."""
        config = MetricsConfig(
            service_name="custom-service",
            exporters=[mock_exporter],
            export_interval=30.0,
            enable_runtime_metrics=False,
        )

        assert config.service_name == "custom-service"
        assert len(config.exporters) == 1
        assert config.export_interval == 30.0
        assert config.enable_runtime_metrics is False


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_collector_creation(self, metrics_config):
        """Test creating a metrics collector."""
        collector = MetricsCollector(metrics_config)

        assert collector.config is metrics_config

    def test_collector_default_config(self):
        """Test collector with default config."""
        collector = MetricsCollector()

        assert collector.config.service_name == "agents_framework"

    def test_collector_has_default_metrics(self, metrics_collector):
        """Test that collector sets up default metrics."""
        # Check counters
        assert metrics_collector.get_counter("agent_executions_total") is not None
        assert metrics_collector.get_counter("tool_calls_total") is not None
        assert metrics_collector.get_counter("llm_requests_total") is not None
        assert metrics_collector.get_counter("llm_errors_total") is not None
        assert metrics_collector.get_counter("tokens_input_total") is not None
        assert metrics_collector.get_counter("tokens_output_total") is not None

        # Check gauges
        assert metrics_collector.get_gauge("active_agents") is not None
        assert metrics_collector.get_gauge("active_tasks") is not None

        # Check histograms
        assert metrics_collector.get_histogram("agent_execution_duration_seconds") is not None
        assert metrics_collector.get_histogram("tool_execution_duration_seconds") is not None
        assert metrics_collector.get_histogram("llm_request_duration_seconds") is not None

    def test_collector_create_counter(self, metrics_collector):
        """Test creating a counter."""
        counter = metrics_collector.create_counter(
            "custom_counter",
            description="Custom counter",
            unit="requests",
        )

        assert counter.name == "custom_counter"
        assert metrics_collector.get_counter("custom_counter") is counter

    def test_collector_create_gauge(self, metrics_collector):
        """Test creating a gauge."""
        gauge = metrics_collector.create_gauge(
            "custom_gauge",
            description="Custom gauge",
            unit="connections",
        )

        assert gauge.name == "custom_gauge"
        assert metrics_collector.get_gauge("custom_gauge") is gauge

    def test_collector_create_histogram(self, metrics_collector):
        """Test creating a histogram."""
        histogram = metrics_collector.create_histogram(
            "custom_histogram",
            description="Custom histogram",
            unit="ms",
            buckets=[10, 50, 100, 500],
        )

        assert histogram.name == "custom_histogram"
        assert histogram.buckets == [10, 50, 100, 500]
        assert metrics_collector.get_histogram("custom_histogram") is histogram

    def test_collector_inc_counter(self, metrics_collector):
        """Test incrementing counter by name."""
        metrics_collector.inc_counter("agent_executions_total", 5.0)

        counter = metrics_collector.get_counter("agent_executions_total")
        assert counter is not None
        assert counter.get() == 5.0

    def test_collector_inc_counter_nonexistent(self, metrics_collector):
        """Test incrementing nonexistent counter does nothing."""
        metrics_collector.inc_counter("nonexistent_counter", 5.0)
        # Should not raise

    def test_collector_set_gauge(self, metrics_collector):
        """Test setting gauge by name."""
        metrics_collector.set_gauge("active_agents", 10.0)

        gauge = metrics_collector.get_gauge("active_agents")
        assert gauge is not None
        assert gauge.get() == 10.0

    def test_collector_set_gauge_nonexistent(self, metrics_collector):
        """Test setting nonexistent gauge does nothing."""
        metrics_collector.set_gauge("nonexistent_gauge", 10.0)
        # Should not raise

    def test_collector_observe_histogram(self, metrics_collector):
        """Test observing histogram by name."""
        metrics_collector.observe_histogram("agent_execution_duration_seconds", 0.5)

        histogram = metrics_collector.get_histogram("agent_execution_duration_seconds")
        assert histogram is not None
        assert histogram.get_count() == 1
        assert histogram.get_sum() == 0.5

    def test_collector_observe_histogram_nonexistent(self, metrics_collector):
        """Test observing nonexistent histogram does nothing."""
        metrics_collector.observe_histogram("nonexistent_histogram", 0.5)
        # Should not raise

    def test_collector_record_tokens(self, metrics_collector):
        """Test recording token usage."""
        metrics_collector.record_tokens(100, 50)

        input_counter = metrics_collector.get_counter("tokens_input_total")
        output_counter = metrics_collector.get_counter("tokens_output_total")

        assert input_counter is not None
        assert input_counter.get() == 100.0
        assert output_counter is not None
        assert output_counter.get() == 50.0

    def test_collector_record_tokens_with_labels(self, metrics_collector):
        """Test recording token usage with labels."""
        metrics_collector.record_tokens(100, 50, {"model": "gpt-4"})
        metrics_collector.record_tokens(200, 100, {"model": "gpt-3.5"})

        input_counter = metrics_collector.get_counter("tokens_input_total")
        assert input_counter is not None
        assert input_counter.get() == 300.0

    def test_collector_get_all_metrics(self, metrics_collector):
        """Test getting all metrics as dictionary."""
        metrics_collector.inc_counter("agent_executions_total", 5.0)
        metrics_collector.set_gauge("active_agents", 3.0)
        metrics_collector.observe_histogram("agent_execution_duration_seconds", 0.5)

        all_metrics = metrics_collector.get_all_metrics()

        assert "agent_executions_total" in all_metrics
        assert all_metrics["agent_executions_total"] == 5.0

        assert "active_agents" in all_metrics
        assert all_metrics["active_agents"] == 3.0

        assert "agent_execution_duration_seconds" in all_metrics
        assert all_metrics["agent_execution_duration_seconds"]["count"] == 1

    async def test_collector_export(self, metrics_collector, mock_exporter):
        """Test exporting all metrics."""
        metrics_collector.inc_counter("agent_executions_total", 3.0)

        await metrics_collector.export()

        assert mock_exporter.export_count == 1
        assert "agent_executions_total" in mock_exporter.exported_metrics[0]

    async def test_collector_export_handles_error(self, metrics_collector):
        """Test that export handles exporter errors gracefully."""
        failing_exporter = AsyncMock(side_effect=RuntimeError("Export failed"))
        metrics_collector.config.exporters.append(failing_exporter)

        # Should not raise
        await metrics_collector.export()

    async def test_collector_periodic_export(self, metrics_collector, mock_exporter):
        """Test starting and stopping periodic export."""
        metrics_collector.config.export_interval = 0.1

        await metrics_collector.start_periodic_export()

        # Wait for at least one export
        await asyncio.sleep(0.15)

        await metrics_collector.stop_periodic_export()

        assert mock_exporter.export_count >= 1

    async def test_collector_start_periodic_idempotent(self, metrics_collector):
        """Test that starting periodic export is idempotent."""
        await metrics_collector.start_periodic_export()
        task1 = metrics_collector._export_task

        await metrics_collector.start_periodic_export()
        task2 = metrics_collector._export_task

        assert task1 is task2

        await metrics_collector.stop_periodic_export()

    async def test_collector_stop_periodic_when_not_running(self, metrics_collector):
        """Test stopping periodic export when not running."""
        await metrics_collector.stop_periodic_export()
        # Should not raise

    async def test_collector_shutdown(self, metrics_collector, mock_exporter):
        """Test shutting down the collector."""
        metrics_collector.config.export_interval = 0.1
        await metrics_collector.start_periodic_export()

        await metrics_collector.shutdown()

        assert metrics_collector._export_task is None
        assert mock_exporter.export_count >= 1
        assert mock_exporter.shutdown_called is True


class TestGetMetricsCollector:
    """Tests for get_metrics_collector function."""

    def test_get_metrics_collector_creates_global(self):
        """Test that get_metrics_collector creates a global instance."""
        # Reset global collector for this test
        import agents_framework.observability.metrics as metrics_module
        metrics_module._metrics_collector = None

        collector = get_metrics_collector()

        assert collector is not None
        assert isinstance(collector, MetricsCollector)

    def test_get_metrics_collector_returns_same_instance(self):
        """Test that get_metrics_collector returns the same instance."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()

        assert collector1 is collector2

    def test_get_metrics_collector_with_config(self):
        """Test get_metrics_collector with custom config."""
        # Reset global collector
        import agents_framework.observability.metrics as metrics_module
        metrics_module._metrics_collector = None

        config = MetricsConfig(service_name="custom-service")
        collector = get_metrics_collector(config)

        assert collector.config.service_name == "custom-service"


class TestTimedDecorator:
    """Tests for @timed decorator."""

    async def test_timed_decorator_async_function(self):
        """Test timed decorator on async function."""
        # Reset collector
        import agents_framework.observability.metrics as metrics_module
        metrics_module._metrics_collector = None

        collector = get_metrics_collector()
        collector.create_histogram("test_async_duration")

        @timed("test_async_duration")
        async def async_func() -> str:
            await asyncio.sleep(0.01)
            return "result"

        result = await async_func()

        assert result == "result"
        histogram = collector.get_histogram("test_async_duration")
        assert histogram is not None
        assert histogram.get_count() == 1
        assert histogram.get_sum() > 0

    async def test_timed_decorator_async_exception(self):
        """Test timed decorator records time even on exception."""
        import agents_framework.observability.metrics as metrics_module
        metrics_module._metrics_collector = None

        collector = get_metrics_collector()
        collector.create_histogram("test_failing_duration")

        @timed("test_failing_duration")
        async def failing_func():
            await asyncio.sleep(0.01)
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await failing_func()

        histogram = collector.get_histogram("test_failing_duration")
        assert histogram is not None
        assert histogram.get_count() == 1

    def test_timed_decorator_sync_function(self):
        """Test timed decorator on sync function."""
        import agents_framework.observability.metrics as metrics_module
        metrics_module._metrics_collector = None

        collector = get_metrics_collector()
        collector.create_histogram("test_sync_duration")

        @timed("test_sync_duration")
        def sync_func() -> str:
            time.sleep(0.01)
            return "sync_result"

        result = sync_func()

        assert result == "sync_result"
        histogram = collector.get_histogram("test_sync_duration")
        assert histogram is not None
        assert histogram.get_count() == 1

    def test_timed_decorator_sync_exception(self):
        """Test timed decorator records time on sync exception."""
        import agents_framework.observability.metrics as metrics_module
        metrics_module._metrics_collector = None

        collector = get_metrics_collector()
        collector.create_histogram("test_sync_fail_duration")

        @timed("test_sync_fail_duration")
        def sync_failing():
            time.sleep(0.01)
            raise RuntimeError("Sync error")

        with pytest.raises(RuntimeError):
            sync_failing()

        histogram = collector.get_histogram("test_sync_fail_duration")
        assert histogram is not None
        assert histogram.get_count() == 1

    def test_timed_decorator_with_labels(self):
        """Test timed decorator with labels."""
        import agents_framework.observability.metrics as metrics_module
        metrics_module._metrics_collector = None

        collector = get_metrics_collector()
        collector.create_histogram("labeled_duration")

        @timed("labeled_duration", labels={"method": "test"})
        def labeled_func():
            pass

        labeled_func()

        histogram = collector.get_histogram("labeled_duration")
        assert histogram is not None
        assert histogram.get_count() == 1

    def test_timed_decorator_preserves_function_name(self):
        """Test that timed decorator preserves function name."""
        @timed("some_histogram")
        def my_named_function():
            pass

        assert my_named_function.__name__ == "my_named_function"

    async def test_timed_decorator_async_preserves_function_name(self):
        """Test that timed decorator preserves async function name."""
        @timed("some_histogram")
        async def my_async_function():
            pass

        assert my_async_function.__name__ == "my_async_function"


class TestMetricsIntegration:
    """Integration tests for metrics functionality."""

    async def test_full_metrics_workflow(self, mock_exporter):
        """Test a complete metrics workflow."""
        config = MetricsConfig(
            service_name="integration-test",
            exporters=[mock_exporter],
            export_interval=60.0,
        )
        collector = MetricsCollector(config)

        # Record various metrics
        collector.inc_counter("agent_executions_total", 1.0)
        collector.inc_counter("tool_calls_total", 3.0)
        collector.record_tokens(500, 200)
        collector.set_gauge("active_agents", 2.0)
        collector.observe_histogram("agent_execution_duration_seconds", 0.5)
        collector.observe_histogram("agent_execution_duration_seconds", 1.2)

        # Export
        await collector.export()

        # Verify
        assert mock_exporter.export_count == 1
        metrics = mock_exporter.exported_metrics[0]

        assert metrics["agent_executions_total"] == 1.0
        assert metrics["tool_calls_total"] == 3.0
        assert metrics["tokens_input_total"] == 500.0
        assert metrics["tokens_output_total"] == 200.0
        assert metrics["active_agents"] == 2.0
        assert metrics["agent_execution_duration_seconds"]["count"] == 2

    def test_metrics_with_multiple_label_dimensions(self):
        """Test metrics with multiple label dimensions."""
        counter = Counter(name="requests", labels={"service": "api"})

        counter.inc(1.0, {"method": "GET", "path": "/users"})
        counter.inc(2.0, {"method": "POST", "path": "/users"})
        counter.inc(1.0, {"method": "GET", "path": "/items"})

        assert counter.get({"method": "GET", "path": "/users"}) == 1.0
        assert counter.get({"method": "POST", "path": "/users"}) == 2.0
        assert counter.get({"method": "GET", "path": "/items"}) == 1.0
        assert counter.get() == 4.0  # Total
