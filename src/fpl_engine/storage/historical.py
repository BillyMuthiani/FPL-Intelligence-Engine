"""Historical snapshot storage for reproducible backtesting.

This module provides :class:`HistoricalSnapshotStore`, a companion to
:class:`~fpl_engine.storage.cache.SnapshotCache` that persists raw-API
JSON responses tagged with gameweek and deadline metadata.  Snapshots are
immutable: once written, a snapshot is never silently replaced.  Integrity is
verified via SHA-256 on every retrieval.

Snapshots are stored under ``raw_data_dir / "historical" /`` to keep them
separate from the live (TTL-based) cache managed by ``SnapshotCache``.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError, field_validator

from ..config import Settings, get_settings
from ..logging import get_logger
from .cache import (
    _format_timestamp,
    _meta_path_to_data_path,
    _safe_unlink,
    _sanitize_endpoint,
    _serialize_metadata,
    _serialize_payload,
    _write_atomic,
)
from .exceptions import CacheError
from .models import SnapshotMetadata

log = get_logger(__name__)

__all__ = [
    "HistoricalSnapshot",
    "HistoricalSnapshotMetadata",
    "HistoricalSnapshotStore",
]


class HistoricalSnapshotMetadata(SnapshotMetadata):
    """Provenance metadata for a single historical raw-API snapshot.

    Extends :class:`~fpl_engine.storage.models.SnapshotMetadata` with optional
    gameweek and deadline fields so that snapshots can be correlated to
    specific FPL gameweeks.

    ``retrieved_at`` records when the snapshot was actually collected from the
    API.  It is **never** assumed to equal the official gameweek deadline unless
    the caller explicitly supplies ``deadline_time``.
    """

    gameweek: int | None = None
    deadline_time: datetime | None = None

    @field_validator("gameweek")
    @classmethod
    def _validate_gameweek(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError(f"gameweek must be >= 1, got {v}")  # noqa: TRY003
        return v

    @field_validator("deadline_time")
    @classmethod
    def _validate_deadline_time(cls, v: datetime | None) -> datetime | None:
        if v is not None:
            if v.tzinfo is None:
                raise ValueError("deadline_time must be timezone-aware")  # noqa: TRY003
            if v.utcoffset() != UTC.utcoffset(None):
                raise ValueError("deadline_time must be UTC")  # noqa: TRY003
        return v


@dataclass(frozen=True)
class HistoricalSnapshot:
    """A validated historical snapshot: the parsed payload plus its metadata."""

    payload: Any
    metadata: HistoricalSnapshotMetadata


class HistoricalSnapshotStore:
    """Immutable filesystem store for historical raw-API snapshots.

    Snapshots are persisted as ``{timestamp}_{hash8}.json`` +
    ``{timestamp}_{hash8}.meta.json`` pairs beneath
    ``raw_data_dir / "historical" / {sanitized-endpoint} /``, using the same
    deterministic serialization and SHA-256 integrity verification as
    :class:`~fpl_engine.storage.cache.SnapshotCache`.

    Design guarantees:

    * **Immutability:** storing a payload at the same ``retrieved_at`` produces
      the same filename (content-hash suffix), so re-storing identical data is
      a no-op.  Different content at the same timestamp produces a different
      filename -- an earlier snapshot is never silently replaced.
    * **Integrity:** every read verifies SHA-256 and ``content_length``;
      corrupt or mismatched pairs are silently skipped (same convention as
      ``SnapshotCache``).
    * **Offline retrieval:** once stored, snapshots can be retrieved without
      any network access.
    * **Atomicity:** writes use a temp-file + ``Path.replace`` pattern.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings: Settings = settings if settings is not None else get_settings()

    @property
    def base_dir(self) -> Path:
        """Root directory under which all historical snapshots live."""
        return Path(self._settings.raw_data_dir) / "historical"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def store(
        self,
        endpoint: str,
        payload: Any,
        *,
        source: str,
        url: str,
        retrieved_at: datetime,
        gameweek: int | None = None,
        deadline_time: datetime | None = None,
    ) -> HistoricalSnapshot:
        """Serialize, hash, and atomically persist a raw payload snapshot.

        Parameters
        ----------
        endpoint
            Raw endpoint path, e.g. ``"/bootstrap-static/"``.
        The fetched value to persist (typically a dict from
        ``/bootstrap-static/`` or ``/element-summary/{id}/``, or a list
        from ``/fixtures/``).
        source
            Human-readable source identifier, e.g. ``"fpl-api"``.
        url
            Fully-qualified URL from which the data was retrieved.
        retrieved_at
            UTC timestamp recording **when** the snapshot was collected.
            Must be timezone-aware and in UTC.
        gameweek
            Optional FPL gameweek tag for correlation.  Not inferred from
            ``retrieved_at``.
        deadline_time
            Optional official gameweek deadline.  Callers must not rely on
            ``retrieved_at`` equalling the deadline.

        Returns
        -------
        HistoricalSnapshot
            The stored snapshot (payload + metadata) so the caller can use the
            result immediately without re-reading from disk.

        Raises
        ------
        ValueError
            If *retrieved_at* is not UTC-aware, or if gameweek/deadline_time
            fail validation.
        CacheError
            If the payload cannot be serialized or the atomic file write fails.
        """
        if retrieved_at.tzinfo is None:
            raise ValueError("retrieved_at must be timezone-aware")  # noqa: TRY003
        if retrieved_at.utcoffset() != UTC.utcoffset(None):
            raise ValueError("retrieved_at must be UTC")  # noqa: TRY003

        json_bytes = _serialize_payload(payload)
        content_sha256 = hashlib.sha256(json_bytes).hexdigest()

        metadata = HistoricalSnapshotMetadata(
            schema_version="1",
            source=source,
            endpoint=endpoint,
            url=url,
            retrieved_at=retrieved_at,
            content_sha256=content_sha256,
            content_length=len(json_bytes),
            content_encoding="utf-8",
            json_serialization="compact-sorted-keys",
            gameweek=gameweek,
            deadline_time=deadline_time,
        )

        self._atomic_write_pair(
            endpoint, retrieved_at, content_sha256, json_bytes, metadata
        )

        log.info(
            "Stored historical snapshot",
            endpoint=endpoint,
            gameweek=gameweek,
            retrieved_at=retrieved_at.isoformat(),
        )

        return HistoricalSnapshot(payload=payload, metadata=metadata)

    def get(
        self,
        endpoint: str,
        *,
        gameweek: int | None = None,
        before: datetime | None = None,
    ) -> HistoricalSnapshot | None:
        """Retrieve the newest valid snapshot matching the given criteria.

        Parameters
        ----------
        endpoint
            Raw endpoint path to filter on.
        gameweek
            If provided, only snapshots tagged with this gameweek are
            considered.
        before
            If provided, only snapshots with ``retrieved_at <= before`` are
            considered.  This enforces a prediction-time **cutoff**: only
            snapshots that were actually collected at or before *before*
            are eligible.  A snapshot merely *tagged* with a gameweek does
            not by itself prove it was collected before that gameweek's
            deadline -- callers must supply the deadline or cutoff here.

        Returns
        -------
        HistoricalSnapshot | None
            The newest matching snapshot, or ``None`` if none exists.
            Corrupt, unsupported, or incomplete pairs are silently skipped.
        """
        snapshots = self._scan_valid_snapshots(endpoint)
        if gameweek is not None:
            snapshots = [s for s in snapshots if s.metadata.gameweek == gameweek]
        if before is not None:
            snapshots = [s for s in snapshots if s.metadata.retrieved_at <= before]
        if not snapshots:
            return None
        return snapshots[0]  # newest-first

    def get_all(self, endpoint: str) -> list[HistoricalSnapshot]:
        """Return all valid snapshots for *endpoint*, newest first."""
        return self._scan_valid_snapshots(endpoint)

    def has_snapshot(
        self,
        endpoint: str,
        *,
        gameweek: int | None = None,
        before: datetime | None = None,
    ) -> bool:
        """Return True if at least one valid snapshot exists for *endpoint*.

        When *gameweek* is provided, the check is restricted to snapshots
        tagged with that gameweek.

        When *before* is provided, only snapshots with ``retrieved_at <= before``
        are considered.  This lets callers ask "do I already have a snapshot
        for this gameweek that was collected **before** this cutoff?" -- a
        question distinct from "does any snapshot for this gameweek exist,
        regardless of when it was collected."
        """
        return self.get(endpoint, gameweek=gameweek, before=before) is not None

    def clear(self) -> int:
        """Delete all historical snapshots.

        Returns the number of top-level directories removed.  Intended for
        test cleanup; not typically used in production.

        Raises
        ------
        CacheError
            If the base directory cannot be removed.
        """
        if not self.base_dir.is_dir():
            return 0
        removed = 0
        for child in sorted(self.base_dir.iterdir()):
            if child.is_dir():
                with contextlib.suppress(OSError):
                    child.rmdir()
                    removed += 1
        try:
            self.base_dir.rmdir()
        except OSError as exc:
            raise CacheError(  # noqa: TRY003
                f"Failed to remove historical base dir {self.base_dir}: {exc}"
            ) from exc
        return removed

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _snapshot_dir(self, endpoint: str) -> Path:
        """Return the directory holding snapshots for *endpoint*."""
        return self.base_dir / _sanitize_endpoint(endpoint)

    def _scan_valid_snapshots(self, endpoint: str) -> list[HistoricalSnapshot]:
        """Enumerate all valid snapshots for *endpoint*, newest first.

        Snapshots are ordered by ``metadata.retrieved_at`` (read from the
        metadata sidecar), **never** by filesystem modification time.
        Invalid, corrupt, unsupported, or incomplete pairs are skipped.
        """
        dir_path = self._snapshot_dir(endpoint)
        if not dir_path.is_dir():
            return []

        valid: list[HistoricalSnapshot] = []
        for meta_path in sorted(dir_path.glob("*.meta.json")):
            try:
                snap = self._load_candidate(meta_path, endpoint=endpoint)
            except OSError:
                continue
            if snap is not None:
                valid.append(snap)

        valid.sort(key=lambda s: s.metadata.retrieved_at, reverse=True)
        return valid

    def _load_candidate(
        self, meta_path: Path, *, endpoint: str
    ) -> HistoricalSnapshot | None:
        """Validate a single snapshot candidate.

        Returns ``None`` if the metadata or data file is corrupt,
        unsupported, incomplete, or if the metadata ``endpoint`` does not
        match the requested endpoint (defending against path-sanitization
        collisions).  SHA-256 and content-length mismatches are also
        silently skipped.
        """
        json_path = _meta_path_to_data_path(meta_path)

        if not json_path.is_file():
            return None

        try:
            meta_text = meta_path.read_text(encoding="utf-8")
        except OSError:
            return None

        try:
            metadata = HistoricalSnapshotMetadata.model_validate_json(meta_text)
        except (ValidationError, ValueError):
            return None

        if metadata.endpoint != endpoint:
            return None

        try:
            raw_bytes = json_path.read_bytes()
        except OSError:
            return None

        if len(raw_bytes) != metadata.content_length:
            return None

        if hashlib.sha256(raw_bytes).hexdigest() != metadata.content_sha256:
            return None

        try:
            payload = json.loads(raw_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

        if not isinstance(payload, (dict, list)):
            return None

        return HistoricalSnapshot(payload=payload, metadata=metadata)

    def _atomic_write_pair(
        self,
        endpoint: str,
        retrieved_at: datetime,
        content_sha256: str,
        json_bytes: bytes,
        metadata: HistoricalSnapshotMetadata,
    ) -> None:
        """Atomically write the data file and metadata sidecar.

        Writing the data file first, then metadata, means a crash leaves
        either a complete pair or a data-only file without metadata -- both
        are safe because only ``*.meta.json`` files are scanned.
        """
        dir_path = self._snapshot_dir(endpoint)

        filename_base = f"{_format_timestamp(retrieved_at)}_{content_sha256[:8]}"
        data_path = dir_path / f"{filename_base}.json"
        meta_path = dir_path / f"{filename_base}.meta.json"

        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CacheError(  # noqa: TRY003
                f"Cannot create snapshot directory {dir_path}: {exc}"
            ) from exc

        try:
            _write_atomic(data_path, json_bytes)
            meta_bytes = _serialize_metadata(metadata)
            _write_atomic(meta_path, meta_bytes)
        except OSError as exc:
            for p in (
                data_path.parent / (data_path.name + ".tmp"),
                meta_path.parent / (meta_path.name + ".tmp"),
            ):
                _safe_unlink(p)
            raise CacheError(  # noqa: TRY003
                f"Failed to write historical snapshot for {endpoint}: {exc}"
            ) from exc
