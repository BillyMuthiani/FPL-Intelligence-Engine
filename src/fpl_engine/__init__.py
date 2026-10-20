"""FPL Intelligence Engine - Core package."""

from .config import Settings, get_settings
from .logging import configure_logging, get_logger

__version__ = "0.1.0"

__all__ = [
    "Settings",
    "configure_logging",
    "get_logger",
    "get_settings",
]
