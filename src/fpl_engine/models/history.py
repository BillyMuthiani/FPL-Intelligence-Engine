"""Per-gameweek match history observation model.

Provides a typed, validated view of the per-gameweek entries available from
the FPL ``/api/element-summary/{player_id}/`` ``history`` array.  Only
defensive-relevant fields are modelled; the full history payload contains
many additional statistics that are out of scope for this module.

Fields are populated from the per-gameweek ``history`` array entries, which
contain observed (not expected) values.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from .fixture import Fixture


class PlayerMatchHistory(BaseModel):
    """A single player's observed statistics for one gameweek match.

    All fields are observed (actual) values from the FPL element-summary
    ``history`` array entry for the corresponding fixture.

    Attributes:
        player_id: FPL player identifier (element id).
        fixture_id: FPL fixture identifier.
        gameweek: FPL gameweek number (``round`` in API data).
        minutes: Minutes played in the match.  0 if the player did not
            appear.
        goals_conceded: Goals conceded while the player was on the pitch
            (0/1 values from the API; 0 for players who weren't goalkeeper
            or defender).
        clean_sheet: Whether the player's team kept a clean sheet.
        saves: Goals saved by the goalkeeper.
        goals_scored: Goals scored by the player.
        assists: Assists by the player.
        tackles: Tackles won.
        clearances_blocks_interceptions: Combined CBI count from the API.
        recoveries: Ball recoveries.
        yellow_cards: Yellow cards received.
        red_cards: Red cards received (including second yellows).
        own_goals: Own goals scored.
        penalties_saved: Penalties saved.
        penalties_missed: Penalties missed.
        bonus: Bonus points awarded (0-3).
        bps: Bonus points system raw score.
        total_points: Total FPL points earned in the match.
        outcome_time: UTC datetime when the match outcome became observable.
               None when not derivable from the element-summary endpoint alone.
        kickoff_time: UTC datetime when the fixture kicked off (match start).
               Available from the FPL ``/api/fixtures/`` endpoint (returned as
               an ISO string) and joined via ``fixture_id``.  ``None`` when
               fixture metadata has not been joined.
    """

    player_id: int = Field(description="FPL player ID", ge=0)
    fixture_id: int = Field(description="FPL fixture ID", ge=0)
    gameweek: int = Field(description="Gameweek number (1-38)", ge=1, le=38)

    minutes: int = Field(default=0, ge=0, le=120, description="Minutes played")

    goals_conceded: int = Field(default=0, ge=0, description="Goals conceded (on-pitch)")
    clean_sheet: bool = Field(default=False, description="Clean sheet kept")
    saves: int = Field(default=0, ge=0, description="Saves (GK)")
    goals_scored: int = Field(default=0, ge=0, description="Goals scored")
    assists: int = Field(default=0, ge=0, description="Assists")

    tackles: int = Field(default=0, ge=0, description="Tackles won")
    clearances_blocks_interceptions: int = Field(
        default=0, ge=0, description="Combined CBI from FPL API"
    )
    recoveries: int = Field(default=0, ge=0, description="Ball recoveries")

    yellow_cards: int = Field(default=0, ge=0, description="Yellow cards")
    red_cards: int = Field(default=0, ge=0, description="Red cards")
    own_goals: int = Field(default=0, ge=0, description="Own goals")
    penalties_saved: int = Field(default=0, ge=0, description="Penalties saved")
    penalties_missed: int = Field(default=0, ge=0, description="Penalties missed")

    bonus: int = Field(default=0, ge=0, le=3, description="Bonus points (0-3)")
    bps: int = Field(default=0, ge=0, description="Bonus points system score")
    total_points: int = Field(default=0, ge=0, description="Total FPL points")

    outcome_time: datetime | None = Field(
        default=None,
        description=(
            "UTC datetime when the match outcome became observable (i.e. the "
            "fixture kickoff time + 90 minutes, or when FPL marked the result "
            "as final).  Used for leakage-safe temporal filtering in feature "
            "generation.  ``None`` when it cannot be established from the "
            "element-summary endpoint alone — callers should join fixture "
            "metadata from the ``/api/fixtures/`` endpoint."
        ),
    )

    kickoff_time: datetime | None = Field(
        default=None,
        description=(
            "UTC datetime when the fixture kicked off (match start).  "
            "Available from the FPL ``/api/fixtures/`` endpoint as an ISO "
            "string, joined via ``fixture_id``.  Used as the pre-match "
            "prediction cutoff in leakage-safe evaluation.  ``None`` when "
            "fixture metadata has not been joined."
        ),
    )

    @field_validator("outcome_time", "kickoff_time")
    @classmethod
    def _validate_timestamp(cls, v: datetime | None) -> datetime | None:
        """Ensure non-None timestamps are UTC-aware and in UTC."""
        if v is not None:
            if v.tzinfo is None:
                raise ValueError(  # noqa: TRY003
                    "must be timezone-aware (UTC)"
                )
            if v.utcoffset() != UTC.utcoffset(None):
                raise ValueError(  # noqa: TRY003
                    "must be in UTC"
                )
        return v

    @field_validator("player_id", "fixture_id", mode="before")
    @classmethod
    def _validate_id(cls, v: object) -> int:
        if isinstance(v, bool) or not isinstance(v, int):
            exc = TypeError("player_id and fixture_id must be integers")
            raise exc
        return v

    @field_validator("gameweek", mode="before")
    @classmethod
    def _validate_gameweek(cls, v: object) -> int:
        if isinstance(v, bool) or not isinstance(v, int):
            exc = TypeError("gameweek must be an integer")
            raise exc
        return v


__all__ = [
    "PlayerMatchHistory",
    "defensive_contribution_from_history",
    "normalize_match_history",
    "normalize_match_history_with_fixtures",
]


def normalize_match_history(
    player_id: int,
    raw: dict[str, Any],
    outcome_time: datetime | None = None,
    kickoff_time: datetime | None = None,
) -> PlayerMatchHistory:
    """Convert a raw FPL element-summary history entry to a ``PlayerMatchHistory``.

    The FPL ``/api/element-summary/{player_id}/`` endpoint returns a
    ``history`` array whose entries use ``round`` for the gameweek number
    and ``fixture`` for the fixture ID.  This function maps those raw
    keys to the canonical model fields and preserves missing values
    as model defaults (see module docstring for missing-vs-zero semantics).

    Parameters
    ----------
    player_id
        The FPL player (element) ID that these statistics belong to.
    raw
        A single history-dict entry from the API response.
    outcome_time
        Optional UTC datetime when the match outcome became observable.
        Required for leakage-safe temporal filtering.  When ``None``,
        callers cannot verify temporal safety for this observation.
    kickoff_time
        Optional UTC datetime when the fixture kicked off (match start).
        Available from the FPL ``/api/fixtures/`` endpoint.  Used as the
        pre-match prediction cutoff in leakage-safe evaluation.  When
        ``None``, callers cannot verify that training observations were
        pre-match.

    Returns
    -------
    PlayerMatchHistory
        A validated model instance.

    Raises
    ------
    pydantic.ValidationError
        If required fields are missing or values violate constraints.
    """
    return PlayerMatchHistory(
        player_id=player_id,
        fixture_id=raw.get("fixture", 0),
        gameweek=raw.get("round", 0),
        minutes=raw.get("minutes", 0),
        goals_conceded=raw.get("goals_conceded", 0),
        clean_sheet=bool(raw.get("clean_sheet", False)),
        saves=raw.get("saves", 0),
        goals_scored=raw.get("goals_scored", 0),
        assists=raw.get("assists", 0),
        tackles=raw.get("tackles", 0),
        clearances_blocks_interceptions=raw.get(
            "clearances_blocks_interceptions", 0
        ),
        recoveries=raw.get("recoveries", 0),
        yellow_cards=raw.get("yellow_cards", 0),
        red_cards=raw.get("red_cards", 0),
        own_goals=raw.get("own_goals", 0),
        penalties_saved=raw.get("penalties_saved", 0),
        penalties_missed=raw.get("penalties_missed", 0),
        bonus=raw.get("bonus", 0),
        bps=raw.get("bps", 0),
        total_points=raw.get("total_points", 0),
        outcome_time=outcome_time,
        kickoff_time=kickoff_time,
    )


def normalize_match_history_with_fixtures(
    player_id: int,
    raw: dict[str, Any],
    fixtures: Mapping[int, Fixture],
    outcome_time: datetime | None = None,
) -> PlayerMatchHistory:
    """Normalize a history entry, joining fixture metadata for kickoff time.

    The FPL ``/api/element-summary/{player_id}/`` ``history`` array does not
    include ``kickoff_time``.  Callers that have already fetched fixture
    metadata (from ``/api/fixtures/``) can pass a mapping of
    ``fixture_id -> Fixture`` to populate the ``kickoff_time`` field, which
    serves as the pre-match prediction cutoff for leakage-safe evaluation.

    Parameters
    ----------
    player_id
        The FPL player (element) ID.
    raw
        A single history-dict entry from the API response.
    fixtures
        Mapping from fixture ID to ``Fixture`` objects (e.g. as returned by
        ``FPLClient.get_fixtures``).
    outcome_time
        Optional UTC datetime when the match outcome became observable.
        When ``None`` (the common case — the API does not provide an
        outcome-availability timestamp), callers should rely on
        ``kickoff_time`` (joined from ``fixtures``) for temporal filtering.

    Returns
    -------
    PlayerMatchHistory
        A validated model instance with ``kickoff_time`` populated from
        fixture metadata when available.

    Raises
    ------
    pydantic.ValidationError
        If required fields are missing or values violate constraints.
    """
    fixture_id = raw.get("fixture", 0)
    kickoff_time: datetime | None = None
    if fixture_id in fixtures:
        fixture = fixtures[fixture_id]
        if fixture.kickoff_time is not None:
            kickoff_time = datetime.fromisoformat(fixture.kickoff_time)
    return normalize_match_history(
        player_id=player_id,
        raw=raw,
        outcome_time=outcome_time,
        kickoff_time=kickoff_time,
    )


def defensive_contribution_from_history(
    history: PlayerMatchHistory,
    threshold: int,
) -> bool:
    """Determine whether a historical match met the FPL DC threshold.

    Uses the same CBIT/CBIRT logic as ``scoring.calculate_fpl_points``:

    - DEF: 10+ CBIT (tackles + CBI) -> True
    - MID/FWD: 12+ CBIRT (tackles + CBI + recoveries) -> True

    Note:
        The caller must pass the position-appropriate threshold (``10`` for
        defenders, ``12`` for midfielders/forwards).  Goalkeepers are not
        eligible for DC points and this function should not be called for them.

    Parameters
    ----------
    history
        A single match observation.  The caller is responsible for ensuring
        the player was on the pitch (minutes > 0) for this observation.
    threshold
        The DC threshold: 10 for defenders, 12 for midfielders/forwards.

    Returns
    -------
    bool
        True if the DC threshold was met for this match.

    Raises
    ------
    ValueError
        If threshold is not 10 or 12.

    Limitations
    -----------
    This function cannot distinguish "missing" from "zero" for defensive
    statistics on historical observations because the FPL element-summary API
    always includes these fields as integers (0 when the player did not record
    the stat).  If a player did not play (minutes == 0), the caller should
    skip evaluation entirely.
    """
    if threshold not in (10, 12):
        exc = ValueError(
            f"threshold must be 10 (DEF) or 12 (MID/FWD), got {threshold}"
        )
        raise exc

    cbit = history.tackles + history.clearances_blocks_interceptions

    if threshold == 10:
        # DEF: CBIT only (tackles + clearances + blocks + interceptions)
        return cbit >= threshold
    else:
        # MID/FWD: CBIRT = CBIT + ball recoveries
        cbirt = cbit + history.recoveries
        return cbirt >= threshold


