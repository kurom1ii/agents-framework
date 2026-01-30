"""Structured logging with context for the agents framework.

This module provides a logging system with:
- Structured logging using structlog
- Context propagation
- Correlation IDs for request tracking
- Multiple exporters (console, file)
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO, Union

try:
    import structlog
    from structlog.types import Processor
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False
    structlog = None  # type: ignore


class LogLevel(str, Enum):
    """Log levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    def to_int(self) -> int:
        """Convert to logging module integer level."""
        return {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }[self]


@dataclass
class LogContext:
    """Context for structured logging.

    Holds contextual data that should be included in all log entries
    within the current execution scope.

    Attributes:
        correlation_id: ID for tracking related requests/operations.
        agent_id: ID of the current agent.
        task_id: ID of the current task.
        session_id: ID of the current session.
        extra: Additional context data.
    """

    correlation_id: Optional[str] = None
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging."""
        result: Dict[str, Any] = {}
        if self.correlation_id:
            result["correlation_id"] = self.correlation_id
        if self.agent_id:
            result["agent_id"] = self.agent_id
        if self.task_id:
            result["task_id"] = self.task_id
        if self.session_id:
            result["session_id"] = self.session_id
        result.update(self.extra)
        return result

    def with_extra(self, **kwargs: Any) -> "LogContext":
        """Create a new context with additional data."""
        new_extra = {**self.extra, **kwargs}
        return LogContext(
            correlation_id=self.correlation_id,
            agent_id=self.agent_id,
            task_id=self.task_id,
            session_id=self.session_id,
            extra=new_extra,
        )


# Context variable for storing current log context
_log_context: ContextVar[Optional[LogContext]] = ContextVar(
    "log_context", default=None
)

# Global context that applies to all log entries
_global_context: Dict[str, Any] = {}


def set_context(context: LogContext) -> None:
    """Set the current log context."""
    _log_context.set(context)


def get_context() -> Optional[LogContext]:
    """Get the current log context."""
    return _log_context.get()


def clear_context() -> None:
    """Clear the current log context."""
    _log_context.set(None)


def set_global_context(**kwargs: Any) -> None:
    """Set global context that applies to all log entries."""
    _global_context.update(kwargs)


def clear_global_context() -> None:
    """Clear all global context."""
    _global_context.clear()


def get_global_context() -> Dict[str, Any]:
    """Get the global context."""
    return _global_context.copy()


@dataclass
class LogExporter:
    """Base configuration for log exporters."""

    name: str
    min_level: LogLevel = LogLevel.DEBUG


@dataclass
class ConsoleExporter(LogExporter):
    """Console log exporter configuration."""

    name: str = "console"
    stream: TextIO = field(default_factory=lambda: sys.stdout)
    colors: bool = True
    format_style: str = "console"  # "console" or "json"


@dataclass
class FileExporter(LogExporter):
    """File log exporter configuration."""

    name: str = "file"
    path: Union[str, Path] = "agents.log"
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    format_style: str = "json"  # "json" or "text"


@dataclass
class LogConfig:
    """Configuration for the logging system.

    Attributes:
        level: Minimum log level.
        exporters: List of log exporters.
        include_timestamp: Whether to include timestamps.
        include_caller: Whether to include caller info.
        json_format: Whether to use JSON format by default.
    """

    level: LogLevel = LogLevel.INFO
    exporters: List[LogExporter] = field(default_factory=list)
    include_timestamp: bool = True
    include_caller: bool = True
    json_format: bool = False

    def __post_init__(self) -> None:
        if not self.exporters:
            self.exporters = [ConsoleExporter()]


class AgentLogger:
    """Structured logger for agents with context support.

    Provides structured logging with automatic context inclusion,
    correlation ID tracking, and support for multiple exporters.
    """

    def __init__(
        self,
        name: str,
        config: Optional[LogConfig] = None,
    ):
        """Initialize the logger.

        Args:
            name: Logger name (usually module name).
            config: Optional logging configuration.
        """
        self.name = name
        self.config = config or LogConfig()
        self._stdlib_logger = logging.getLogger(name)
        self._stdlib_logger.setLevel(self.config.level.to_int())

        # Set up structlog if available
        if HAS_STRUCTLOG:
            self._setup_structlog()
            self._logger = structlog.get_logger(name)
        else:
            self._logger = None
            self._setup_stdlib_logging()

    def _setup_structlog(self) -> None:
        """Configure structlog with processors."""
        if not HAS_STRUCTLOG:
            return

        processors: List[Processor] = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
        ]

        if self.config.include_caller:
            processors.append(structlog.processors.CallsiteParameterAdder())

        # Add final processor based on format
        if self.config.json_format:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(
                structlog.dev.ConsoleRenderer(colors=True)
            )

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(
                self.config.level.to_int()
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=False,
        )

    def _setup_stdlib_logging(self) -> None:
        """Configure standard library logging as fallback."""
        if not self._stdlib_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self._stdlib_logger.addHandler(handler)

    def _get_merged_context(self, **extra: Any) -> Dict[str, Any]:
        """Merge all context sources into a single dict."""
        context: Dict[str, Any] = {}

        # Add global context
        context.update(_global_context)

        # Add current log context
        log_context = get_context()
        if log_context:
            context.update(log_context.to_dict())

        # Add extra kwargs
        context.update(extra)

        return context

    def bind(self, **kwargs: Any) -> "AgentLogger":
        """Create a new logger with bound context.

        Args:
            **kwargs: Context to bind to the logger.

        Returns:
            New AgentLogger with bound context.
        """
        new_logger = AgentLogger(self.name, self.config)
        if HAS_STRUCTLOG and self._logger:
            new_logger._logger = self._logger.bind(**kwargs)
        return new_logger

    def debug(self, msg: str, **kwargs: Any) -> None:
        """Log a debug message."""
        context = self._get_merged_context(**kwargs)
        if HAS_STRUCTLOG and self._logger:
            self._logger.debug(msg, **context)
        else:
            self._stdlib_logger.debug(msg, extra=context)

    def info(self, msg: str, **kwargs: Any) -> None:
        """Log an info message."""
        context = self._get_merged_context(**kwargs)
        if HAS_STRUCTLOG and self._logger:
            self._logger.info(msg, **context)
        else:
            self._stdlib_logger.info(msg, extra=context)

    def warning(self, msg: str, **kwargs: Any) -> None:
        """Log a warning message."""
        context = self._get_merged_context(**kwargs)
        if HAS_STRUCTLOG and self._logger:
            self._logger.warning(msg, **context)
        else:
            self._stdlib_logger.warning(msg, extra=context)

    def error(self, msg: str, **kwargs: Any) -> None:
        """Log an error message."""
        context = self._get_merged_context(**kwargs)
        if HAS_STRUCTLOG and self._logger:
            self._logger.error(msg, **context)
        else:
            self._stdlib_logger.error(msg, extra=context)

    def critical(self, msg: str, **kwargs: Any) -> None:
        """Log a critical message."""
        context = self._get_merged_context(**kwargs)
        if HAS_STRUCTLOG and self._logger:
            self._logger.critical(msg, **context)
        else:
            self._stdlib_logger.critical(msg, extra=context)

    def exception(self, msg: str, **kwargs: Any) -> None:
        """Log an exception with traceback."""
        context = self._get_merged_context(**kwargs)
        if HAS_STRUCTLOG and self._logger:
            self._logger.exception(msg, **context)
        else:
            self._stdlib_logger.exception(msg, extra=context)

    def log(self, level: LogLevel, msg: str, **kwargs: Any) -> None:
        """Log a message at the specified level."""
        method = getattr(self, level.value)
        method(msg, **kwargs)


# Default logger instance
_loggers: Dict[str, AgentLogger] = {}


def get_logger(
    name: str = "agents_framework",
    config: Optional[LogConfig] = None,
) -> AgentLogger:
    """Get or create a logger instance.

    Args:
        name: Logger name.
        config: Optional configuration. Only used when creating new logger.

    Returns:
        AgentLogger instance.
    """
    if name not in _loggers:
        _loggers[name] = AgentLogger(name, config)
    return _loggers[name]


def configure_logging(config: LogConfig) -> None:
    """Configure the global logging system.

    Args:
        config: Logging configuration to apply.
    """
    # Clear existing loggers
    _loggers.clear()

    # Set up file handlers if needed
    for exporter in config.exporters:
        if isinstance(exporter, FileExporter):
            _setup_file_handler(exporter)


def _setup_file_handler(exporter: FileExporter) -> None:
    """Set up a file handler for logging."""
    from logging.handlers import RotatingFileHandler

    path = Path(exporter.path) if isinstance(exporter.path, str) else exporter.path
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        path,
        maxBytes=exporter.max_bytes,
        backupCount=exporter.backup_count,
    )
    handler.setLevel(exporter.min_level.to_int())

    if exporter.format_style == "json":
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"name": "%(name)s", "message": "%(message)s"}'
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    handler.setFormatter(formatter)

    # Add to root logger
    logging.getLogger().addHandler(handler)
