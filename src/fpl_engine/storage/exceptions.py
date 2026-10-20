"""Exception types for the raw-snapshot storage layer."""

from __future__ import annotations


class CacheError(Exception):
    """Raised on filesystem write failures or unrecoverable cache errors."""
