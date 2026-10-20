"""Player gameweek record types and descriptive metrics calculator.

Public API:
    - :class:`FixtureGameweekRecord` — per-fixture performance snapshot
    - :class:`GameweekStats` — aggregated gameweek-level statistics
    - :class:`PlayerMetrics` — comprehensive descriptive metrics for a player
    - :func:`calculate_player_metrics` — pure function producing PlayerMetrics
    - :func:`points_per_90` — generic per-90 rate helper
    - :func:`points_per_million` — points per £1m value helper
    - :func:`clean_sheet_rate` — clean sheet rate helper
    - :func:`points_std_dev` — population standard deviation
    - :func:`points_last_n_average` — rolling average form helper
    - :class:`MetricsError` — raised on invalid input

This module is independent of the Streamlit UI and the ingestion layer.
All functions are pure and deterministic.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import pstdev
from typing import Any

from fpl_engine.analytics.scoring import (
    DEFAULT_SCORING_RULES,
    MatchStats,
    ScoringDecomposition,
)
from fpl_engine.models import Player, PlayerPosition

__all__ = [
    "FixtureGameweekRecord",
    "GameweekStats",
    "MetricsError",
    "PlayerMetrics",
    "calculate_player_metrics",
    "clean_sheet_rate",
    "points_last_n_average",
    "points_per_90",
    "points_per_million",
    "points_std_dev",
]


class MetricsError(ValueError):
    """Raised when player metrics calculation encounters invalid input."""


def _is_real_int(value: Any) -> bool:
    """Return True only if value is an int and NOT a bool."""
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class FixtureGameweekRecord:
    """One player's performance in a single fixture.

    For double gameweeks, a player will have multiple records with
    the same ``gameweek`` number but different ``fixture_id`` values.

    Attributes:
        gameweek: FPL gameweek number. Must be a real integer in [1, 38].
        fixture_id: FPL fixture ID. Must be a real integer and unique
            within this player's record set. Used for deterministic ordering
            — fixture identity is never inferred from statistics.
        match_stats: Per-fixture statistics (validated 0-120 minutes).
            Scoring is computed per-fixture before aggregation.
        decomposition: Reconstructed FPL points breakdown for this fixture,
            computed via ``calculate_fpl_points(position, match_stats)``.
        observed_points: Official FPL points awarded for this fixture,
            if available. ``None`` if unobserved or future. Never silently
            substituted with ``decomposition.total``.
    """

    gameweek: int
    fixture_id: int
    match_stats: MatchStats
    decomposition: ScoringDecomposition
    observed_points: int | None = None

    def __post_init__(self) -> None:
        if not _is_real_int(self.gameweek):
            raise MetricsError(  # noqa: TRY003
                f"gameweek must be an int, got {type(self.gameweek).__name__} "
                f"({self.gameweek!r})"
            )
        if not _is_real_int(self.fixture_id):
            raise MetricsError(  # noqa: TRY003
                f"fixture_id must be an int, got {type(self.fixture_id).__name__} "
                f"({self.fixture_id!r})"
            )
        if not 1 <= self.gameweek <= 38:
            raise MetricsError(f"gameweek must be in [1, 38], got {self.gameweek}")  # noqa: TRY003
        if not isinstance(self.match_stats, MatchStats):
            raise MetricsError(  # noqa: TRY003
                f"match_stats must be MatchStats, got {type(self.match_stats).__name__}"
            )
        if not isinstance(self.decomposition, ScoringDecomposition):
            raise MetricsError(  # noqa: TRY003
                f"decomposition must be ScoringDecomposition, got "
                f"{type(self.decomposition).__name__}"
            )
        if self.observed_points is not None:
            if not _is_real_int(self.observed_points):
                raise MetricsError(  # noqa: TRY003
                    f"observed_points must be int or None, got "
                    f"{type(self.observed_points).__name__} ({self.observed_points!r})"
                )
            if self.observed_points < 0:
                raise MetricsError(  # noqa: TRY003
                    f"observed_points must be non-negative, got {self.observed_points}"
                )


@dataclass(frozen=True)
class GameweekStats:
    """Aggregated gameweek-level statistics.

    Unlike :class:`MatchStats`, ``minutes`` is allowed to exceed 120
    for double gameweeks. This type exists to avoid modifying Stage 1
    validation in :mod:`fpl_engine.analytics.scoring`.

    Attributes:
        minutes: Total minutes played in the GW (can exceed 120 for DGW).
        goals, assists, etc.: Sums across all fixtures in this GW.
        clean_sheet: True only if ALL fixtures have ``clean_sheet=True``
            AND ALL fixtures have ``minutes > 0``. A zero-minute fixture
            in a GW disqualifies the entire GW's clean sheet.
        total_points_reconstructed: Sum of ``decomposition.total`` across
            all fixtures in this GW.
        total_points_observed: Sum of ``observed_points`` across all fixtures
            if every fixture in this GW has ``observed_points is not None``;
            otherwise ``None``. Missing observed points make the entire GW's
            observed total unavailable — never substituted with reconstructed.
        fixture_count: Number of fixtures aggregated into this GW.
    """

    minutes: int
    goals: int
    assists: int
    goals_conceded: int
    saves: int
    penalties_saved: int
    penalties_missed: int
    yellow_cards: int
    red_cards: int
    own_goals: int
    clean_sheet: bool
    bonus: int
    total_points_reconstructed: int
    total_points_observed: int | None
    fixture_count: int

    def __post_init__(self) -> None:
        if self.minutes < 0:
            raise MetricsError(f"minutes must be non-negative, got {self.minutes}")  # noqa: TRY003
        for name in (
            "goals", "assists", "goals_conceded", "saves",
            "penalties_saved", "penalties_missed", "yellow_cards",
            "red_cards", "own_goals",
        ):
            if getattr(self, name) < 0:
                raise MetricsError(  # noqa: TRY003
                    f"{name} must be non-negative, got {getattr(self, name)}"
                )
        if self.bonus < 0:
            raise MetricsError(f"bonus must be non-negative, got {self.bonus}")  # noqa: TRY003
        if self.fixture_count < 1:
            raise MetricsError(  # noqa: TRY003
                f"fixture_count must be >= 1, got {self.fixture_count}"
            )


@dataclass(frozen=True)
class PlayerMetrics:
    """Descriptive player metrics through a gameweek cutoff.

    All count-based and rate-based metrics are gameweek-level:
    a double gameweek counts as one in denominators.

    Observed and reconstructed metrics are kept strictly separate.
    See module docstring and :func:`calculate_player_metrics` for details.
    """

    # --- Player identity ---
    player_id: int
    position: PlayerPosition

    # --- Gameweek-level counts (DGW counts once) ---
    matches_played: int            # unique GWs with minutes > 0
    sample_size: int                # total unique GWs after cutoff
    total_fixtures: int            # total FixtureGameweekRecords processed
    total_minutes: int

    # --- Raw event aggregates ---
    goals: int
    assists: int
    goals_conceded: int
    clean_sheets: int
    saves: int
    penalties_saved: int
    penalties_missed: int
    yellow_cards: int
    red_cards: int
    own_goals: int
    bonus: int

    # --- Observed points (per-GW) ---
    observed_gw_count: int           # GWs where total_points_observed is not None
    total_points_observed: int
    points_per_90_observed: float | None
    points_season_avg_observed: float | None
    points_last_1_observed: int | None
    points_last_3_avg_observed: float | None
    points_last_5_avg_observed: float | None
    points_std_dev_observed: float | None

    # --- Reconstructed points (per-fixture scoring → GW aggregate) ---
    total_points_reconstructed: int
    points_per_90_reconstructed: float | None
    points_season_avg_reconstructed: float | None
    points_std_dev_reconstructed: float | None

    # --- Defensive metrics (GW-level) ---
    clean_sheet_rate: float | None
    clean_sheet_eligible: bool

    # --- Cutoff metadata ---
    cutoff_gameweek: int | None


def points_per_90(total_points: int, total_minutes: int) -> float | None:
    """Return points per 90 minutes.

    Args:
        total_points: Accumulated FPL points.
        total_minutes: Accumulated minutes played.

    Returns:
        ``total_points / total_minutes * 90`` as a float, or ``None``
        when ``total_minutes == 0``.
    """
    if total_minutes == 0:
        return None
    return (total_points / total_minutes) * 90.0


def points_per_million(total_points: int, price_tenths: int) -> float | None:
    """Return points per million pounds.

    Args:
        total_points: Accumulated FPL points.
        price_tenths: Player price in tenths of millions (£13.0m = 130).

    Returns:
        ``total_points / (price_tenths / 10.0)``, or ``None`` when
        ``price_tenths == 0``.

    Price timing is the caller's responsibility; this function does not
    infer or look up prices.
    """
    if price_tenths == 0:
        return None
    price_millions = price_tenths / 10.0
    return total_points / price_millions


def clean_sheet_rate(clean_sheets: int, matches_played: int) -> float | None:
    """Return clean sheets per game-week played.

    A double gameweek counts once in the denominator (``matches_played``
    is already gameweek-level).

    Args:
        clean_sheets: Number of gameweeks with a clean sheet.
        matches_played: Number of gameweeks with minutes > 0.

    Returns:
        ``clean_sheets / matches_played``, or ``None`` when
        ``matches_played == 0``. Position-agnostic.
    """
    if matches_played == 0:
        return None
    return clean_sheets / matches_played


def points_std_dev(points: Sequence[int]) -> float | None:
    """Return population standard deviation of point values.

    Args:
        points: Sequence of integer point totals (typically one per GW).

    Returns:
        Population standard deviation, or ``None`` if fewer than 2 values
        are provided.
    """
    if len(points) < 2:
        return None
    return float(pstdev(points))


def points_last_n_average(points: Sequence[int], n: int) -> float | None:
    """Return the average of the last ``n`` point values.

    Assumes ``points`` is sorted chronologically ascending (oldest first).

    Args:
        points: Chronological sequence of integer point totals.
        n: Window size.

    Returns:
        Mean of the last ``n`` values, or ``None`` if fewer than ``n``
        values are available or ``n <= 0``.
    """
    if n <= 0:
        return None
    if len(points) < n:
        return None
    return float(sum(points[-n:]) / n)


def _validate_player(player: Any) -> None:
    if not isinstance(player, Player):
        raise MetricsError(  # noqa: TRY003
            f"player must be a Player instance, got {type(player).__name__}"
        )


def _validate_records(
    fixture_records: Sequence[FixtureGameweekRecord],
) -> None:
    fixture_ids: set[int] = set()
    for i, record in enumerate(fixture_records):
        if not isinstance(record, FixtureGameweekRecord):
            raise MetricsError(  # noqa: TRY003
                f"fixture_records[{i}] must be FixtureGameweekRecord, got "
                f"{type(record).__name__}"
            )
        if record.fixture_id in fixture_ids:
            raise MetricsError(  # noqa: TRY003
                f"duplicate fixture_id {record.fixture_id} found in records"
            )
        fixture_ids.add(record.fixture_id)


def _aggregate_fixture(
    gw: int, fixtures: list[FixtureGameweekRecord]
) -> GameweekStats:
    """Aggregate fixture records for a single gameweek.

    Applies the conservative DGW clean-sheet rule and partial-observed rule.
    """
    if not fixtures:
        raise MetricsError(f"no fixtures for gameweek {gw}")  # noqa: TRY003

    minutes = sum(f.match_stats.minutes for f in fixtures)
    goals = sum(f.match_stats.goals for f in fixtures)
    assists = sum(f.match_stats.assists for f in fixtures)
    goals_conceded = sum(f.match_stats.goals_conceded for f in fixtures)
    saves = sum(f.match_stats.saves for f in fixtures)
    penalties_saved = sum(f.match_stats.penalties_saved for f in fixtures)
    penalties_missed = sum(f.match_stats.penalties_missed for f in fixtures)
    yellow_cards = sum(f.match_stats.yellow_cards for f in fixtures)
    red_cards = sum(f.match_stats.red_cards for f in fixtures)
    own_goals = sum(f.match_stats.own_goals for f in fixtures)
    bonus = sum(f.match_stats.bonus for f in fixtures)

    all_clean_sheets = all(f.match_stats.clean_sheet for f in fixtures)
    all_minutes_positive = all(f.match_stats.minutes > 0 for f in fixtures)
    clean_sheet = all_clean_sheets and all_minutes_positive

    total_reconstructed = sum(f.decomposition.total for f in fixtures)

    observed_values = [f.observed_points for f in fixtures]
    if all(v is not None for v in observed_values):
        total_observed: int | None = sum(v for v in observed_values if v is not None)
    else:
        total_observed = None

    return GameweekStats(
        minutes=minutes,
        goals=goals,
        assists=assists,
        goals_conceded=goals_conceded,
        saves=saves,
        penalties_saved=penalties_saved,
        penalties_missed=penalties_missed,
        yellow_cards=yellow_cards,
        red_cards=red_cards,
        own_goals=own_goals,
        clean_sheet=clean_sheet,
        bonus=bonus,
        total_points_reconstructed=total_reconstructed,
        total_points_observed=total_observed,
        fixture_count=len(fixtures),
    )


def calculate_player_metrics(
    player: Player,
    fixture_records: Sequence[FixtureGameweekRecord],
    cutoff_gameweek: int | None = None,
) -> PlayerMetrics:
    """Compute descriptive metrics from fixture-level gameweek records.

    Processing steps:
        1. Validate ``player`` is a :class:`Player` instance.
        2. Validate ``fixture_records`` types and ``fixture_id`` uniqueness.
        3. Validate ``cutoff_gameweek`` (if not None, must be a real int >= 0).
        4. Filter out records where ``gameweek > cutoff_gameweek``.
        5. Sort remaining records by ``(gameweek, fixture_id)`` ascending.
        6. Aggregate fixtures into :class:`GameweekStats` per unique ``gameweek``.
        7. Compute all metrics from the aggregated :class:`GameweekStats` list.

    Scoring separation:
        - ``calculate_fpl_points`` is applied per-fixture (Stage 1) before
          aggregation — defensive contributions, clean sheets, goals conceded,
          and appearance points are computed on individual ``MatchStats``.
        - Per-GW ``decomposition.total`` sums are computed from per-fixture
          decomps — never by re-scoring aggregated ``MatchStats``.

    Args:
        player: Player entity (provides position and id).
        fixture_records: Per-fixture records (may include multiple per GW
            for double gameweeks). Need not be pre-sorted.
        cutoff_gameweek: Inclusive upper GW bound. ``None`` = all records.
            ``0`` = valid, produces empty metrics. Negative = raises.

    Returns:
        :class:`PlayerMetrics` with zero counts and ``None`` rates for empty
        input.

    Raises:
        MetricsError: If player type is wrong, fixture_id is duplicated,
            cutoff is negative or non-int, or any record fails validation.
    """
    _validate_player(player)
    _validate_records(fixture_records)

    if cutoff_gameweek is not None:
        if not _is_real_int(cutoff_gameweek):
            raise MetricsError(  # noqa: TRY003
                f"cutoff_gameweek must be int or None, got "
                f"{type(cutoff_gameweek).__name__} ({cutoff_gameweek!r})"
            )
        if cutoff_gameweek < 0:
            raise MetricsError(  # noqa: TRY003
                f"cutoff_gameweek must be >= 0, got {cutoff_gameweek}"
            )

    if not fixture_records:
        gw_stats: list[GameweekStats] = []
    else:
        # Sort for deterministic processing: (gameweek, fixture_id)
        sorted_records = sorted(
            fixture_records, key=lambda r: (r.gameweek, r.fixture_id)
        )

        # Apply cutoff filter BEFORE aggregation
        if cutoff_gameweek is not None:
            sorted_records = [
                r for r in sorted_records if r.gameweek <= cutoff_gameweek
            ]

        # Group by gameweek
        gw_groups: dict[int, list[FixtureGameweekRecord]] = defaultdict(list)
        for record in sorted_records:
            gw_groups[record.gameweek].append(record)

        # Aggregate each GW
        gw_stats = [
            _aggregate_fixture(gw, fixtures)
            for gw, fixtures in sorted(gw_groups.items())
        ]

    return _compute_metrics(
        player=player,
        gw_stats=gw_stats,
        cutoff_gameweek=cutoff_gameweek,
        total_fixtures=len(fixture_records),
    )


def _compute_metrics(
    player: Player,
    gw_stats: list[GameweekStats],
    cutoff_gameweek: int | None,
    total_fixtures: int,
) -> PlayerMetrics:
    """Compute PlayerMetrics from aggregated GameweekStats.

    Assumes ``gw_stats`` is already sorted by gameweek ascending.
    """
    sample_size = len(gw_stats)
    matches_played = sum(1 for gs in gw_stats if gs.minutes > 0)
    total_minutes = sum(gs.minutes for gs in gw_stats)

    goals = sum(gs.goals for gs in gw_stats)
    assists = sum(gs.assists for gs in gw_stats)
    goals_conceded = sum(gs.goals_conceded for gs in gw_stats)
    clean_sheets = sum(
        1 for gs in gw_stats if gs.clean_sheet and gs.minutes > 0
    )
    saves = sum(gs.saves for gs in gw_stats)
    penalties_saved = sum(gs.penalties_saved for gs in gw_stats)
    penalties_missed = sum(gs.penalties_missed for gs in gw_stats)
    yellow_cards = sum(gs.yellow_cards for gs in gw_stats)
    red_cards = sum(gs.red_cards for gs in gw_stats)
    own_goals = sum(gs.own_goals for gs in gw_stats)
    bonus = sum(gs.bonus for gs in gw_stats)

    # Observed points — only from GWs where all fixtures have observed_points
    observed_series = [
        gs.total_points_observed for gs in gw_stats
        if gs.total_points_observed is not None
    ]
    # Minutes only for GWs with observed data
    observed_minutes = sum(
        gs.minutes for gs in gw_stats
        if gs.total_points_observed is not None
    )
    total_points_observed = sum(observed_series)
    observed_gw_count = len(observed_series)

    # Reconstructed points — from ALL GWs
    reconstructed_series = [gs.total_points_reconstructed for gs in gw_stats]
    total_points_reconstructed = sum(reconstructed_series)

    # Observed rates and stats
    points_per_90_observed = points_per_90(total_points_observed, observed_minutes)
    points_season_avg_observed = (
        total_points_observed / observed_gw_count if observed_gw_count > 0 else None
    )

    if observed_series:
        points_last_1_observed = observed_series[-1]
        points_last_3_avg_observed = points_last_n_average(observed_series, 3)
        points_last_5_avg_observed = points_last_n_average(observed_series, 5)
        points_std_dev_observed = points_std_dev(observed_series)
    else:
        points_last_1_observed = None
        points_last_3_avg_observed = None
        points_last_5_avg_observed = None
        points_std_dev_observed = None

    # Reconstructed rates and stats
    points_per_90_reconstructed = points_per_90(
        total_points_reconstructed, total_minutes
    )
    points_season_avg_reconstructed = (
        total_points_reconstructed / sample_size if sample_size > 0 else None
    )
    points_std_dev_reconstructed = points_std_dev(reconstructed_series)

    # Clean sheet metrics (GW-level)
    cs_rate = clean_sheet_rate(clean_sheets, matches_played)
    clean_sheet_eligible = (
        player.position in DEFAULT_SCORING_RULES.clean_sheet_points
    )

    return PlayerMetrics(
        player_id=player.id,
        position=player.position,
        matches_played=matches_played,
        sample_size=sample_size,
        total_fixtures=total_fixtures,
        total_minutes=total_minutes,
        goals=goals,
        assists=assists,
        goals_conceded=goals_conceded,
        clean_sheets=clean_sheets,
        saves=saves,
        penalties_saved=penalties_saved,
        penalties_missed=penalties_missed,
        yellow_cards=yellow_cards,
        red_cards=red_cards,
        own_goals=own_goals,
        bonus=bonus,
        observed_gw_count=observed_gw_count,
        total_points_observed=total_points_observed,
        points_per_90_observed=points_per_90_observed,
        points_season_avg_observed=points_season_avg_observed,
        points_last_1_observed=points_last_1_observed,
        points_last_3_avg_observed=points_last_3_avg_observed,
        points_last_5_avg_observed=points_last_5_avg_observed,
        points_std_dev_observed=points_std_dev_observed,
        total_points_reconstructed=total_points_reconstructed,
        points_per_90_reconstructed=points_per_90_reconstructed,
        points_season_avg_reconstructed=points_season_avg_reconstructed,
        points_std_dev_reconstructed=points_std_dev_reconstructed,
        clean_sheet_rate=cs_rate,
        clean_sheet_eligible=clean_sheet_eligible,
        cutoff_gameweek=cutoff_gameweek,
    )
