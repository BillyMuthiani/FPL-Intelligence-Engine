"""Filesystem persistence layer for raw API response snapshots.

SnapshotCache handles:

* Deterministic serialization of raw payloads (compact UTF-8 JSON, sorted keys).
* SHA-256 integrity verification on every read (never returns corrupt data).
* Timestamped snapshot naming with deterministic directory layout.
* TTL-based freshness checks and stale-but-valid fallback retrieval.
* Atomic writes via temp files + ``os.replace``.

The cache has **no knowledge of HTTP**. It receives an already-fetched
payload (a ``dict``) and a provenance block, and persists them. An
external orchestrator is responsible for deciding when to fetch fresh
data from the network and when to fall back to a stale snapshot.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..config import Settings, get_settings
from .exceptions import CacheError
from .models import Snapshot, SnapshotMetadata


def _sanitize_endpoint(endpoint: str) -> str:
    """Convert a raw endpoint path into a filesystem-safe slug."""
    stripped = endpoint.strip("/")
    return stripped.replace("/", "_")


def _format_timestamp(dt: datetime) -> str:
    """Format a UTC datetime as a filesystem-safe sortable string.

    Colons are replaced with hyphens for Windows compatibility
    (``:`` is illegal in Windows filenames).
    """
    return dt.strftime("%Y-%m-%dT%H-%M-%SZ")


def _serialize_payload(payload: dict[str, Any]) -> bytes:
    """Serialize a payload dict to deterministic compact UTF-8 JSON bytes.

    Settings: ``ensure_ascii=False``, ``sort_keys=True``,
    ``separators=(",", ":")`` -- guarantees byte-identical output for
    the same content regardless of input key ordering.
    """
    json_str = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return json_str.encode("utf-8")


def _serialize_metadata(metadata: SnapshotMetadata) -> bytes:
    """Serialize snapshot metadata to deterministic compact UTF-8 JSON."""
    meta_dict = metadata.model_dump(mode="json")
    meta_str = json.dumps(
        meta_dict,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return meta_str.encode("utf-8")


def _safe_unlink(path: Path) -> None:
    """Delete *path* if it exists, ignoring errors."""
    with contextlib.suppress(OSError):
        path.unlink()




def _write_atomic(path: Path, data: bytes) -> None:
    """Write *data* to *path* atomically via a temp file + ``Path.replace``.

    The temp file is created in the same directory (same filesystem) to
    guarantee ``Path.replace`` is atomic on both POSIX and Windows/NTFS.
    On failure, the temp file is removed.
    """
    tmp_path = path.parent / (path.name + ".tmp")
    try:
        with tmp_path.open("wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(path)
    except BaseException:
        _safe_unlink(tmp_path)
        raise


def _meta_path_to_data_path(meta_path: Path) -> Path:
    """Derive the data-file path from a metadata sidecar path.

    ``foo.meta.json`` -> ``foo.json``
    """
    base = meta_path.name.removesuffix(".meta.json")
    return meta_path.parent / f"{base}.json"


class SnapshotCache:
    """Pure filesystem persistence for raw-API JSON snapshots.

    Parameters
    ----------
    settings
        Project :class:`~fpl_engine.config.Settings`; only
        ``settings.raw_data_dir`` is consulted. Defaults to the
        global settings singleton.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings: Settings = settings if settings is not None else get_settings()


    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def store_raw(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        source: str,
        url: str,
        retrieved_at: datetime | None = None,
    ) -> Snapshot:
        """Serialize, hash, and atomically persist a raw payload snapshot.

        Parameters
        ----------
        endpoint
            Raw endpoint path, e.g. ``"/bootstrap-static/"``.
        payload
            The fetched dict to persist.
        source
            Human-readable source identifier, e.g. ``"fpl-api"``.
        url
            Fully-qualified URL from which the data was retrieved.
        retrieved_at
            UTC timestamp for provenance. If ``None`` (default),
            ``datetime.now(UTC)`` is used. Tests may inject
            an explicit value for deterministic behavior.

        Returns
        -------
        Snapshot
            The stored snapshot (payload + metadata) so the caller can
            use the result immediately without re-reading from disk.

        Raises
        ------
        CacheError
            If the payload cannot be serialized or the file write fails.
        ValidationError
            If the provenance inputs fail metadata validation.
        """
        if retrieved_at is None:
            retrieved_at = datetime.now(UTC)

        json_bytes = _serialize_payload(payload)
        content_sha256 = hashlib.sha256(json_bytes).hexdigest()

        metadata = SnapshotMetadata(
            schema_version="1",
            source=source,
            endpoint=endpoint,
            url=url,
            retrieved_at=retrieved_at,
            content_sha256=content_sha256,
            content_length=len(json_bytes),
            content_encoding="utf-8",
            json_serialization="compact-sorted-keys",
        )

        self._atomic_write_pair(
            endpoint, retrieved_at, content_sha256, json_bytes, metadata
        )

        return Snapshot(payload=payload, metadata=metadata)

    def get_cached(
        self,
        endpoint: str,
        *,
        ttl_seconds: float | None = None,
        now: datetime | None = None,
    ) -> Snapshot | None:
        """Return the newest valid snapshot for *endpoint* if it is fresh.

        Parameters
        ----------
        endpoint
            Raw endpoint path, e.g. ``"/bootstrap-static/"``.
        ttl_seconds
            Maximum age in seconds for the snapshot to be considered
            fresh. If ``None``, no TTL check is applied (identical to
            :meth:`get_cached_any`).

        Returns
        -------
        Snapshot | None
            The newest valid snapshot within the TTL, or ``None`` if no
            fresh valid snapshot exists. Corrupt, unsupported, or
            incomplete snapshots are skipped silently.
        """
        if now is None:
            now = datetime.now(UTC)
        for snap in self._scan_valid_snapshots(endpoint):
            if ttl_seconds is not None:
                age = (now - snap.metadata.retrieved_at).total_seconds()
                if age > ttl_seconds:
                    continue
            return snap
        return None

    def get_cached_any(self, endpoint: str) -> Snapshot | None:
        """Return the newest **valid** snapshot, regardless of staleness.

        Unlike :meth:`get_cached`, this method performs **no TTL check**.
        It is intended for orchestrator-level stale-fallback: when the API
        is unavailable, the caller may use this method to retrieve a
        valid-but-stale snapshot without ever accepting corrupted data.

        Returns ``None`` if no valid snapshot exists for *endpoint*.
        """
        return self.get_cached(endpoint, ttl_seconds=None)


    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _snapshot_dir(self, endpoint: str) -> Path | None:
        """Return the directory holding snapshots for *endpoint*, or None."""
        dir_path = Path(self._settings.raw_data_dir) / _sanitize_endpoint(endpoint)
        if not dir_path.is_dir():
            return None
        return dir_path

    def _scan_valid_snapshots(self, endpoint: str) -> list[Snapshot]:
        """Enumerate all valid snapshots for *endpoint*, newest first.

        Snapshots are ordered by ``metadata.retrieved_at`` (read from the
        metadata sidecar), **never** by filesystem modification time.
        Invalid, corrupt, unsupported, or incomplete pairs are skipped.
        """
        dir_path = self._snapshot_dir(endpoint)
        if dir_path is None:
            return []

        valid: list[Snapshot] = []
        for meta_path in sorted(dir_path.glob("*.meta.json")):
            try:
                snap = self._load_candidate(meta_path)
            except OSError:
                continue
            if snap is not None:
                valid.append(snap)

        valid.sort(key=lambda s: s.metadata.retrieved_at, reverse=True)
        return valid

    def _load_candidate(self, meta_path: Path) -> Snapshot | None:
        """Validate a single snapshot candidate.

        Returns ``None`` if the metadata or data file is corrupt,
        unsupported, or incomplete.
        """
        json_path = _meta_path_to_data_path(meta_path)

        if not json_path.is_file():
            return None

        try:
            meta_text = meta_path.read_text(encoding="utf-8")
        except OSError:
            return None

        try:
            metadata = SnapshotMetadata.model_validate_json(meta_text)
        except (ValidationError, ValueError):
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

        return Snapshot(payload=payload, metadata=metadata)

    def _atomic_write_pair(
        self,
        endpoint: str,
        retrieved_at: datetime,
        content_sha256: str,
        json_bytes: bytes,
        metadata: SnapshotMetadata,
    ) -> None:
        """Atomically write the data file and metadata sidecar."""
        dir_path = Path(self._settings.raw_data_dir) / _sanitize_endpoint(endpoint)

        filename_base = f"{_format_timestamp(retrieved_at)}_{content_sha256[:8]}"
        data_path = dir_path / f"{filename_base}.json"
        meta_path = dir_path / f"{filename_base}.meta.json"

        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CacheError(  # noqa: TRY003
                f"Cannot create snapshot directory {dir_path}: {exc}"
            ) from exc

        # Write data file first, then metadata (metadata written last so
        # a crash leaves either a complete pair or a data-only file
        # without metadata -- both are safe because only .meta.json
        # files are scanned for candidates).
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
                f"Failed to write snapshot for {endpoint}: {exc}"
            ) from exc
