"""Data ingestion package.

This package contains pure normalization helpers that map raw FPL API
documents into the canonical models defined in :mod:`fpl_engine.models`. No
HTTP requests or filesystem access happen here -- callers supply an already
fetched payload (e.g. the dict returned by
:meth:`fpl_engine.providers.FPLClient.get_bootstrap_data`).

Keeping the raw FPL schema confined to this module preserves the separation
between extraction and normalization required by the architecture and keeps the
canonical models source-agnostic.
"""
from __future__ import annotations

from .bootstrap import (
    Conflict,
    Rejection,
    ValidationReport,
    normalize_gameweeks,
    normalize_players,
    normalize_teams,
)
from .coordinator import IngestionCoordinator
from .exceptions import IngestionCoordinatorError, IngestionError
from .models import IngestionResult
from .processed import ProcessedWriter

__all__ = [
    "Conflict",
    "IngestionCoordinator",
    "IngestionCoordinatorError",
    "IngestionError",
    "IngestionResult",
    "ProcessedWriter",
    "Rejection",
    "ValidationReport",
    "normalize_gameweeks",
    "normalize_players",
    "normalize_teams",
]
