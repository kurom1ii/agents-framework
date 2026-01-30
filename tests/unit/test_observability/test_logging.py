"""Tests for structured logging in the agents framework."""

from __future__ import annotations

import logging
from io import StringIO
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from agents_framework.observability.logging import (
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

if TYPE_CHECKING:
    pass


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_log_level_values(self):
        """Test that all log levels have expected values."""
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.WARNING.value == "warning"
        assert LogLevel.ERROR.value == "error"
        assert LogLevel.CRITICAL.value == "critical"

    def test_log_level_to_int_debug(self):
        """Test DEBUG level converts to correct integer."""
        assert LogLevel.DEBUG.to_int() == logging.DEBUG

    def test_log_level_to_int_info(self):
        """Test INFO level converts to correct integer."""
        assert LogLevel.INFO.to_int() == logging.INFO

    def test_log_level_to_int_warning(self):
        """Test WARNING level converts to correct integer."""
        assert LogLevel.WARNING.to_int() == logging.WARNING

    def test_log_level_to_int_error(self):
        """Test ERROR level converts to correct integer."""
        assert LogLevel.ERROR.to_int() == logging.ERROR

    def test_log_level_to_int_critical(self):
        """Test CRITICAL level converts to correct integer."""
        assert LogLevel.CRITICAL.to_int() == logging.CRITICAL


class TestLogContext:
    """Tests for LogContext dataclass."""

    def test_log_context_creation(self):
        """Test creating a log context with all fields."""
        context = LogContext(
            correlation_id="corr-123",
            agent_id="agent-1",
            task_id="task-456",
            session_id="session-789",
            extra={"custom_key": "custom_value"},
        )

        assert context.correlation_id == "corr-123"
        assert context.agent_id == "agent-1"
        assert context.task_id == "task-456"
        assert context.session_id == "session-789"
        assert context.extra == {"custom_key": "custom_value"}

    def test_log_context_defaults(self):
        """Test log context has correct defaults."""
        context = LogContext()

        assert context.correlation_id is None
        assert context.agent_id is None
        assert context.task_id is None
        assert context.session_id is None
        assert context.extra == {}

    def test_log_context_to_dict_full(self):
        """Test converting full context to dictionary."""
        context = LogContext(
            correlation_id="corr-123",
            agent_id="agent-1",
            task_id="task-456",
            session_id="session-789",
            extra={"environment": "test"},
        )

        result = context.to_dict()

        assert result["correlation_id"] == "corr-123"
        assert result["agent_id"] == "agent-1"
        assert result["task_id"] == "task-456"
        assert result["session_id"] == "session-789"
        assert result["environment"] == "test"

    def test_log_context_to_dict_partial(self):
        """Test converting partial context to dictionary."""
        context = LogContext(
            correlation_id="corr-123",
            agent_id=None,
            task_id="task-456",
        )

        result = context.to_dict()

        assert "correlation_id" in result
        assert "agent_id" not in result
        assert "task_id" in result
        assert "session_id" not in result

    def test_log_context_to_dict_empty(self):
        """Test converting empty context to dictionary."""
        context = LogContext()

        result = context.to_dict()

        assert result == {}

    def test_log_context_with_extra(self):
        """Test creating new context with extra data."""
        original = LogContext(
            correlation_id="corr-123",
            agent_id="agent-1",
            extra={"key1": "value1"},
        )

        new_context = original.with_extra(key2="value2", key3="value3")

        # Original should be unchanged
        assert original.extra == {"key1": "value1"}

        # New context should have merged extra
        assert new_context.correlation_id == "corr-123"
        assert new_context.agent_id == "agent-1"
        assert new_context.extra == {"key1": "value1", "key2": "value2", "key3": "value3"}

    def test_log_context_with_extra_override(self):
        """Test that with_extra can override existing extra keys."""
        original = LogContext(extra={"key1": "original"})

        new_context = original.with_extra(key1="overridden")

        assert original.extra == {"key1": "original"}
        assert new_context.extra == {"key1": "overridden"}


class TestContextManagement:
    """Tests for context variable management functions."""

    def test_set_and_get_context(self):
        """Test setting and getting log context."""
        context = LogContext(correlation_id="test-123")

        set_context(context)
        retrieved = get_context()

        assert retrieved is context
        assert retrieved.correlation_id == "test-123"

    def test_get_context_when_none(self):
        """Test getting context when none is set."""
        clear_context()

        result = get_context()

        assert result is None

    def test_clear_context(self):
        """Test clearing the log context."""
        set_context(LogContext(correlation_id="test-123"))

        clear_context()

        assert get_context() is None

    def test_set_global_context(self):
        """Test setting global context."""
        set_global_context(service="test-service", version="1.0.0")

        result = get_global_context()

        assert result["service"] == "test-service"
        assert result["version"] == "1.0.0"

    def test_set_global_context_merge(self):
        """Test that global context merges values."""
        set_global_context(key1="value1")
        set_global_context(key2="value2")

        result = get_global_context()

        assert result["key1"] == "value1"
        assert result["key2"] == "value2"

    def test_clear_global_context(self):
        """Test clearing global context."""
        set_global_context(key="value")

        clear_global_context()

        assert get_global_context() == {}

    def test_get_global_context_returns_copy(self):
        """Test that get_global_context returns a copy."""
        set_global_context(key="value")

        result = get_global_context()
        result["modified"] = "yes"

        # Original should not be modified
        assert "modified" not in get_global_context()


class TestLogExporters:
    """Tests for log exporter configurations."""

    def test_log_exporter_base(self):
        """Test base log exporter configuration."""
        exporter = LogExporter(name="test", min_level=LogLevel.INFO)

        assert exporter.name == "test"
        assert exporter.min_level == LogLevel.INFO

    def test_console_exporter_defaults(self):
        """Test console exporter default configuration."""
        exporter = ConsoleExporter()

        assert exporter.name == "console"
        assert exporter.min_level == LogLevel.DEBUG
        assert exporter.colors is True
        assert exporter.format_style == "console"

    def test_console_exporter_custom(self):
        """Test console exporter with custom configuration."""
        stream = StringIO()
        exporter = ConsoleExporter(
            stream=stream,
            colors=False,
            format_style="json",
        )

        assert exporter.stream is stream
        assert exporter.colors is False
        assert exporter.format_style == "json"

    def test_file_exporter_defaults(self):
        """Test file exporter default configuration."""
        exporter = FileExporter()

        assert exporter.name == "file"
        assert exporter.path == "agents.log"
        assert exporter.max_bytes == 10 * 1024 * 1024
        assert exporter.backup_count == 5
        assert exporter.format_style == "json"

    def test_file_exporter_custom(self):
        """Test file exporter with custom configuration."""
        exporter = FileExporter(
            path="/var/log/custom.log",
            max_bytes=5 * 1024 * 1024,
            backup_count=10,
            format_style="text",
        )

        assert exporter.path == "/var/log/custom.log"
        assert exporter.max_bytes == 5 * 1024 * 1024
        assert exporter.backup_count == 10
        assert exporter.format_style == "text"


class TestLogConfig:
    """Tests for LogConfig dataclass."""

    def test_log_config_defaults(self):
        """Test log config default values."""
        config = LogConfig()

        assert config.level == LogLevel.INFO
        assert config.include_timestamp is True
        assert config.include_caller is True
        assert config.json_format is False
        assert len(config.exporters) == 1
        assert isinstance(config.exporters[0], ConsoleExporter)

    def test_log_config_custom(self):
        """Test log config with custom values."""
        exporters = [ConsoleExporter(), FileExporter()]
        config = LogConfig(
            level=LogLevel.DEBUG,
            exporters=exporters,
            include_timestamp=False,
            include_caller=False,
            json_format=True,
        )

        assert config.level == LogLevel.DEBUG
        assert len(config.exporters) == 2
        assert config.include_timestamp is False
        assert config.include_caller is False
        assert config.json_format is True

    def test_log_config_adds_default_exporter_when_empty(self):
        """Test that post_init adds console exporter if exporters is empty."""
        config = LogConfig(exporters=[])

        assert len(config.exporters) == 1
        assert isinstance(config.exporters[0], ConsoleExporter)


class TestAgentLogger:
    """Tests for AgentLogger class."""

    def test_logger_creation(self, log_config):
        """Test creating an agent logger."""
        logger = AgentLogger("test.module", log_config)

        assert logger.name == "test.module"
        assert logger.config is log_config

    def test_logger_default_config(self):
        """Test logger uses default config when none provided."""
        logger = AgentLogger("test.module")

        assert logger.config is not None
        assert logger.config.level == LogLevel.INFO

    def test_logger_bind_creates_new_instance(self, agent_logger):
        """Test that bind creates a new logger instance."""
        bound = agent_logger.bind(custom_field="value")

        assert bound is not agent_logger
        assert bound.name == agent_logger.name

    def test_logger_get_merged_context_empty(self, agent_logger):
        """Test getting merged context when nothing is set."""
        result = agent_logger._get_merged_context()

        assert result == {}

    def test_logger_get_merged_context_with_global(self, agent_logger):
        """Test getting merged context with global context."""
        set_global_context(service="test")

        result = agent_logger._get_merged_context()

        assert result["service"] == "test"

    def test_logger_get_merged_context_with_log_context(self, agent_logger, log_context):
        """Test getting merged context with log context."""
        set_context(log_context)

        result = agent_logger._get_merged_context()

        assert result["correlation_id"] == log_context.correlation_id
        assert result["agent_id"] == log_context.agent_id

    def test_logger_get_merged_context_with_extra(self, agent_logger):
        """Test getting merged context with extra kwargs."""
        result = agent_logger._get_merged_context(custom_key="custom_value")

        assert result["custom_key"] == "custom_value"

    def test_logger_get_merged_context_priority(self, agent_logger, log_context):
        """Test that extra kwargs take priority over global and log context."""
        set_global_context(key="global")
        log_context.extra = {"key": "context"}
        set_context(log_context)

        result = agent_logger._get_merged_context(key="extra")

        assert result["key"] == "extra"

    def test_logger_log_method(self, agent_logger):
        """Test the generic log method."""
        with patch.object(agent_logger, "info") as mock_info:
            agent_logger.log(LogLevel.INFO, "test message", key="value")

            mock_info.assert_called_once_with("test message", key="value")

    def test_logger_debug_method(self, agent_logger):
        """Test debug logging."""
        # This just verifies the method can be called without error
        agent_logger.debug("Debug message", extra_key="extra_value")

    def test_logger_info_method(self, agent_logger):
        """Test info logging."""
        agent_logger.info("Info message", extra_key="extra_value")

    def test_logger_warning_method(self, agent_logger):
        """Test warning logging."""
        agent_logger.warning("Warning message", extra_key="extra_value")

    def test_logger_error_method(self, agent_logger):
        """Test error logging."""
        agent_logger.error("Error message", extra_key="extra_value")

    def test_logger_critical_method(self, agent_logger):
        """Test critical logging."""
        agent_logger.critical("Critical message", extra_key="extra_value")

    def test_logger_exception_method(self, agent_logger):
        """Test exception logging."""
        try:
            raise ValueError("Test exception")
        except ValueError:
            agent_logger.exception("Exception occurred", extra_key="extra_value")


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_creates_new(self):
        """Test that get_logger creates a new logger."""
        # Use unique name to avoid conflicts with other tests
        logger = get_logger("unique_test_logger_1")

        assert isinstance(logger, AgentLogger)
        assert logger.name == "unique_test_logger_1"

    def test_get_logger_returns_existing(self):
        """Test that get_logger returns existing logger for same name."""
        logger1 = get_logger("shared_logger_test")
        logger2 = get_logger("shared_logger_test")

        assert logger1 is logger2

    def test_get_logger_with_config(self):
        """Test get_logger with custom configuration."""
        config = LogConfig(level=LogLevel.DEBUG)
        logger = get_logger("configured_logger_test", config)

        assert logger.config.level == LogLevel.DEBUG

    def test_get_logger_config_ignored_for_existing(self):
        """Test that config is ignored if logger already exists."""
        # Create initial logger
        _ = get_logger("existing_config_test", LogConfig(level=LogLevel.INFO))

        # Try to get with different config
        logger2 = get_logger("existing_config_test", LogConfig(level=LogLevel.DEBUG))

        # Should still have original config
        assert logger2.config.level == LogLevel.INFO


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_logging_clears_loggers(self):
        """Test that configure_logging clears existing loggers."""
        # Create a logger first
        _ = get_logger("to_be_cleared_test")

        # Configure with new settings
        config = LogConfig(level=LogLevel.WARNING)
        configure_logging(config)

        # Getting the same logger should now create a new one
        new_logger = get_logger("to_be_cleared_test")
        # New logger should be freshly created
        assert isinstance(new_logger, AgentLogger)


class TestLoggingIntegration:
    """Integration tests for logging functionality."""

    def test_logging_with_full_context(self, agent_logger):
        """Test logging with all context sources."""
        # Set up global context
        set_global_context(service="test-service", environment="testing")

        # Set up log context
        log_ctx = LogContext(
            correlation_id="corr-123",
            agent_id="agent-1",
        )
        set_context(log_ctx)

        # Get merged context
        result = agent_logger._get_merged_context(request_id="req-456")

        # Verify all sources are merged
        assert result["service"] == "test-service"
        assert result["environment"] == "testing"
        assert result["correlation_id"] == "corr-123"
        assert result["agent_id"] == "agent-1"
        assert result["request_id"] == "req-456"

    def test_logging_context_isolation(self):
        """Test that logging contexts are properly isolated."""
        context1 = LogContext(correlation_id="context-1")
        context2 = LogContext(correlation_id="context-2")

        set_context(context1)
        assert get_context().correlation_id == "context-1"

        set_context(context2)
        assert get_context().correlation_id == "context-2"
