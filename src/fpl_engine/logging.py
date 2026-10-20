"""Structured logging configuration using structlog."""

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor

from .config import get_settings


def _configure_structlog() -> None:
    """Configure structlog with appropriate processors."""
    settings = get_settings()

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.log_format == "json":
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        shared_processors.append(
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
        )

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def configure_logging() -> None:
    """Configure the root logger and structlog."""
    settings = get_settings()

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=log_level,
        force=True,
    )

    _configure_structlog()


def get_logger(name: str | None = None, **context: Any) -> structlog.stdlib.BoundLogger:
    """Get a structured logger with optional context.

    Args:
        name: Logger name (defaults to caller's module).
        **context: Additional context to bind to the logger.

    Returns:
        A bound logger instance.
    """
    logger = structlog.get_logger(name)
    if context:
        logger = logger.bind(**context)
    return logger  # type: ignore[no-any-return]


# Convenience function for direct stdlib logger access
def get_stdlib_logger(name: str | None = None) -> logging.Logger:
    """Get a standard library logger."""
    return logging.getLogger(name)
