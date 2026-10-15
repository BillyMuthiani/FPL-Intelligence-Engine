"""Filesystem snapshot persistence for raw API responses.

This package provides :class:`~fpl_engine.storage.cache.SnapshotCache`,
a standalone persistence layer that writes raw JSON payloads to disk
with sidecar metadata, verifies SHA-256 integrity on every read, and
supports TTL-based freshness checks with stale-fallback retrieval.
"""

from .cache import SnapshotCache
from .exceptions import CacheError
from .models import Snapshot, SnapshotMetadata

__all__ = [
    "CacheError",
    "Snapshot",
    "SnapshotCache",
    "SnapshotMetadata",
]
