"""Result type for the ingestion coordinator.

This module defines :class:`IngestionResult`, an immutable dataclass that
bundles together the normalized canonical entities, their per-section
:class:`~fpl_engine.ingestion.bootstrap.ValidationReport` objects, and
provenance / freshness metadata so that callers can trace exactly where the
data came from and whether it is still within its freshness window.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..models import Gameweek, Player, Team
from ..storage.models import SnapshotMetadata
from .bootstrap import ValidationReport

__all__ = ["IngestionResult"]


@dataclass(frozen=True)
class IngestionResult:
    """Structured outcome of a complete ingestion run.

    Attributes
    ----------
    players
        Normalized canonical player entities.
    teams
        Normalized canonical team entities.
    gameweeks
        Normalized canonical gameweek entities.
    player_report
        Validation report produced by :func:`normalize_players`.
    team_report
        Validation report produced by :func:`normalize_teams`.
    gameweek_report
        Validation report produced by :func:`normalize_gameweeks`.
    snapshot_metadata
        Provenance metadata of the raw snapshot that was selected for this
        run (the original ``retrieved_at``, SHA-256, etc.).
    is_fresh
        ``True`` when the data came from the live API or a cache snapshot
        within the configured TTL; ``False`` for stale-fallback data.
    source
        Human-readable label identifying where the selected data originated:

        * ``"live"`` - freshly fetched from the provider via the API.
        * ``"fresh_cache"`` - used from the raw snapshot cache without an
          API request (within TTL).
        * ``"stale_cache"`` - used from a stale cache snapshot as a
          fallback after an API failure.
    """

    players: list[Player]
    teams: list[Team]
    gameweeks: list[Gameweek]
    player_report: ValidationReport
    team_report: ValidationReport
    gameweek_report: ValidationReport
    snapshot_metadata: SnapshotMetadata
    is_fresh: bool
    source: str
