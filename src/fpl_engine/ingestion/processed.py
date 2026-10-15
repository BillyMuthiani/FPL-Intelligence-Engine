"""Processed-data writer for canonical ingestion output.

The :class:`ProcessedWriter` serializes normalized entities (players, teams,
gameweeks) together with their raw-snapshot provenance into a single,
deterministic, versioned JSON bundle and writes it atomically to
``processed_data_dir``.

Design notes
------------
* Uses only the standard library ``json`` module and Pydantic's
  ``model_dump(mode="json")`` - no additional dependencies.
* The output file is written via a temporary file followed by
  ``Path.replace``, guaranteeing that the destination file is never
  partially written.
* Temporary files are cleaned up on any failure.
* Serialization is deterministic: ``sort_keys=True`` and compact
  separators ensure byte-identical output for identical input.
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from ..models import Gameweek, Player, Team
from ..storage.exceptions import CacheError
from ..storage.models import SnapshotMetadata

__all__ = ["PROCESSED_FILENAME", "PROCESSED_SCHEMA_VERSION", "ProcessedWriter"]

#: Version of the processed-data bundle schema.  Bump when the top-level
#: structure changes in a backwards-incompatible way.
PROCESSED_SCHEMA_VERSION = "1"

#: Filename for the single processed-data bundle.
PROCESSED_FILENAME = "bootstrap_processed.json"


def _safe_unlink(path: Path) -> None:
    """Delete *path* if it exists, ignoring errors."""
    with contextlib.suppress(OSError):
        if path.exists():
            path.unlink()


class ProcessedWriter:
    """Atomically persist a versioned processed-data JSON bundle.

    Parameters
    ----------
    settings
        Project :class:`~fpl_engine.config.Settings`; only
        ``settings.processed_data_dir`` is consulted.  Defaults to the
        global settings singleton.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings: Settings = settings if settings is not None else get_settings()

    @property
    def output_path(self) -> Path:
        """Resolved path to the processed-data bundle."""
        return Path(self._settings.processed_data_dir) / PROCESSED_FILENAME

    def write(
        self,
        players: list[Player],
        teams: list[Team],
        gameweeks: list[Gameweek],
        snapshot_metadata: SnapshotMetadata,
    ) -> None:
        """Serialize and atomically write the processed-data bundle.

        Parameters
        ----------
        players
            Normalized canonical player entities.
        teams
            Normalized canonical team entities.
        gameweeks
            Normalized canonical gameweek entities.
        snapshot_metadata
            Provenance metadata from the raw snapshot that was selected
            for this run.  Embedded into the bundle so each processed
            file can be traced back to its source.

        Raises
        ------
        CacheError
            If the bundle cannot be serialized or the atomic file write
            fails (e.g. disk full, permission denied).
        """
        bundle = _build_bundle(players, teams, gameweeks, snapshot_metadata)
        raw = json.dumps(
            bundle,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        _atomic_write(self.output_path, raw)


def _build_bundle(
    players: list[Player],
    teams: list[Team],
    gameweeks: list[Gameweek],
    snapshot_metadata: SnapshotMetadata,
) -> dict[str, Any]:
    """Construct the bundle dict.

    The dict itself is not order-guaranteed, but the caller serializes with
    ``sort_keys=True``, so the final bytes are deterministic.
    """
    return {
        "schema_version": PROCESSED_SCHEMA_VERSION,
        "source_snapshot": snapshot_metadata.model_dump(mode="json"),
        "players": [p.model_dump(mode="json") for p in players],
        "teams": [t.model_dump(mode="json") for t in teams],
        "gameweeks": [g.model_dump(mode="json") for g in gameweeks],
    }


def _atomic_write(target: Path, data: bytes) -> None:
    """Write *data* to *target* atomically via a temp file + ``Path.replace``.

    The temp file lives in the same directory (same filesystem) to
    guarantee ``Path.replace`` is atomic on both POSIX and Windows/NTFS.
    On any failure the temp file is removed so the final output never
    contains a partially written document.

    ``OSError`` is wrapped in :class:`~fpl_engine.storage.exceptions.CacheError`
    for a consistent error type across the persistence layer.  Non-OS
    exceptions propagate as-is so they are never silenced.
    """
    tmp_path = target.parent / (target.name + ".tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(target)
    except OSError as exc:
        _safe_unlink(tmp_path)
        raise CacheError(  # noqa: TRY003
            f"Failed to write processed data to {target}: {exc}"
        ) from exc
    except BaseException:
        _safe_unlink(tmp_path)
        raise
