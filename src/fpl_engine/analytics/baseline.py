"""Baseline analytics for transparent player comparison and ranking.

Stage 3 of the FPL Intelligence Engine analytics stack.

This module provides deterministic, explainable ranking functions that
operate on the outputs of Stage 1 (scoring) and Stage 2 (player metrics).
No machine learning, no simulation, no future-data inference.

Public API:
    - :class:`PlayerBaseline` — descriptive profile combining metrics, price, and fixture context
    - :func:`rank_players_by_value` — rank by points-per-million (historical observed points vs current price)
    - :func:`rank_players_by_consistency` — rank by stability of observed points
    - :func:`rank_players_by_form` — rank by recent gameweek form
    - :func:`compare_players` — produce side-by-side PlayerBaseline objects for a subset
    - :func:`basic_expected_points` — season-aggregate xG/xA proxy for expected FPL points

Temporal safety:
    All ranking functions accept a ``cutoff_gameweek`` parameter forwarded to
    :func:`calculate_player_metrics`, which filters records where
    ``gameweek > cutoff_gameweek`` before aggregation.  This ensures rankings
    computed at historical points in time never incorporate future data.

    The cutoff is also applied to fixture-difficulty calculations so that only
    fixtures played up to and including ``cutoff_gameweek`` contribute to
    ``avg_fixture_difficulty``.  The ``finished`` flag on each fixture provides
    an additional guard against unplayed fixtures.

Price semantics:
    ``points_per_million`` uses ``Player.now_cost`` (the *current* price in tenths
    of millions).  Historical value analysis with per-gameweek price snapshots is
    not yet supported — see ``docs/PROJECT_SPEC.md`` for details.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from fpl_engine.analytics.metrics import (
    FixtureGameweekRecord,
    PlayerMetrics,
    calculate_player_metrics,
    points_per_million,
)
from fpl_engine.analytics.scoring import DEFAULT_SCORING_RULES
from fpl_engine.models import Fixture, Player, PlayerPosition

__all__ = [
    "PlayerBaseline",
    "basic_expected_points",
    "compare_players",
    "rank_players_by_consistency",
    "rank_players_by_form",
    "rank_players_by_value",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayerBaseline:
    """Deterministic descriptive profile for a single player.

    Wraps :class:`~fpl_engine.analytics.metrics.PlayerMetrics` with price
    information and optional fixture-difficulty context to support
    transparent ranking and comparison.

    Price note:
        ``price_tenths`` reflects the **current** price (``Player.now_cost``).
        Using current price against historical points is a deliberate,
        documented trade-off — true per-gameweek value analysis requires
        historical price snapshots, which are not yet available.

    Attributes:
        player_id: FPL player ID.
        name: Player web name (e.g., ``"Salah"``).
        full_name: Player's full name.
        position: :class:`PlayerPosition`.
        team_id: FPL team ID.
        price_tenths: Current price in tenths of millions (£13.0m = 130).
        metrics: Stage 2 :class:`PlayerMetrics` for this player.
        points_per_million_value: ``total_points_observed`` divided by price
            in millions, or ``None`` if price is zero.
        avg_fixture_difficulty: Mean fixture difficulty rating (1-5) across
            finished fixtures for the player's team, or ``None`` if unavailable.
        form_3gw: Average observed points over the last 3 gameweeks, or ``None``.
        form_5gw: Average observed points over the last 5 gameweeks, or ``None``.
        consistency_std: Population standard deviation of per-GW observed points,
            or ``None`` if fewer than 2 observed gameweeks.
    """

    player_id: int
    name: str
    full_name: str
    position: PlayerPosition
    team_id: int
    price_tenths: int
    metrics: PlayerMetrics
    points_per_million_value: float | None
    avg_fixture_difficulty: float | None
    form_3gw: float | None
    form_5gw: float | None
    consistency_std: float | None
    cutoff_gameweek: int | None

    @property
    def total_points_observed(self) -> int:
        """Total observed FPL points (delegates to ``metrics``)."""
        return self.metrics.total_points_observed


# ---------------------------------------------------------------------------
# Fixture-difficulty helper
# ---------------------------------------------------------------------------


def _calculate_avg_fixture_difficulty(
    team_id: int,
    fixtures: Sequence[Fixture],
    cutoff_gameweek: int | None = None,
) -> float | None:
    """Mean FDR (1-5) across finished fixtures involving ``team_id``.

    Only fixtures where ``finished == True`` are included — future or
    unplayed fixtures are excluded to prevent future-information leakage.

    When ``cutoff_gameweek`` is provided, fixtures whose ``event`` exceeds
    the cutoff are also excluded, preventing temporal data leakage in
    backtesting scenarios.  Fixtures with a missing ``event`` are skipped
    for safety.

    Args:
        team_id: The FPL team ID to look up.
        fixtures: Sequence of :class:`Fixture` objects to search.
        cutoff_gameweek: Optional upper bound on fixture gameweek.  Only
            fixtures with ``event <= cutoff_gameweek`` are considered.
            When ``None`` (default), no cutoff filter is applied.

    Returns:
        Average numeric difficulty (1.0 to 5.0), or ``None`` if no finished
        fixtures involving the team remain after filtering.
    """
    difficulties: list[int] = []
    for fx in fixtures:
        if not fx.finished:
            continue
        if cutoff_gameweek is not None and (
            fx.event is None or fx.event > cutoff_gameweek
        ):
            continue
        if team_id == fx.team_h:
            difficulties.append(fx.team_h_difficulty)
        elif team_id == fx.team_a:
            difficulties.append(fx.team_a_difficulty)
    if not difficulties:
        return None
    return sum(difficulties) / len(difficulties)


# ---------------------------------------------------------------------------
# Baseline expected points proxy
# ---------------------------------------------------------------------------


def basic_expected_points(
    player: Player,
    expected_goals: float | None = None,
    expected_assists: float | None = None,
    bonus_share: float | None = None,
) -> float | None:
    """
Season-aggregate xG/xA points-proxy estimator.

    Computes a rough expected-points estimate using position-specific goal
    values from :data:`DEFAULT_SCORING_RULES` plus assist points:

    .. code-block::

        expected_points = xG * goal_points[position] + xA * assist_points
                        + (bonus_share * 2.0 if bonus_share is not None else 0)

    .. warning::

        This is a **season-aggregate descriptive proxy**, **not** a
        forward-looking per-gameweek projection.  It converts season-total xG
        and xA into an FPL-points-equivalent figure.  It does **not**
        account for minutes played, fixture difficulty, opponent strength,
        player availability, or upcoming fixtures.  Use it for coarse
        season-to-date value assessment only.

    Args:
        player: Player entity (provides position for goal point lookup).
        expected_goals: Expected goals for the player. If ``None``, falls back
            to ``player.expected_goals``.
        expected_assists: Expected assists. If ``None``, falls back to
            ``player.expected_assists``.
        bonus_share: Optional fraction (0.0-1.0) of maximum bonus points
            (2 points per game).  If ``None``, no bonus component.

    Returns:
        Expected points proxy as a float, or ``None`` if both xG and xA
        are unavailable or negative.

    Notes:
        - Goal points are position-specific: GKP=10, DEF=6, MID=5, FWD=4 (2026/27).
        - Assist points are 3 for all positions.
        - Maximum bonus per game is 3, but the proxy uses 2 as a conservative
          cap (average bonus for goal-scorers).
        - When only one of xG/xA is supplied, the other is treated as 0.0.
    """
    xg = expected_goals if expected_goals is not None else player.expected_goals
    xa = expected_assists if expected_assists is not None else player.expected_assists

    if xg is None and xa is None:
        return None
    if (xg is not None and xg < 0) or (xa is not None and xa < 0):
        return None

    xg_val = xg if xg is not None else 0.0
    xa_val = xa if xa is not None else 0.0

    rules = DEFAULT_SCORING_RULES
    goal_pts = rules.goal_points[player.position]
    assist_pts = rules.assist_points

    total = xg_val * goal_pts + xa_val * assist_pts

    if bonus_share is not None:
        total += min(bonus_share, 1.0) * 2.0

    return round(total, 4)


# ---------------------------------------------------------------------------
# PlayerBaseline construction helper
# ---------------------------------------------------------------------------


def _build_baseline(
    player: Player,
    fixture_records: Sequence[FixtureGameweekRecord],
    fixtures: Sequence[Fixture] | None,
    cutoff_gameweek: int | None,
    metrics: PlayerMetrics | None = None,
) -> PlayerBaseline:
    """Build a :class:`PlayerBaseline` from a player, their fixture records,
    and optional team fixtures for difficulty context.
    """
    if metrics is None:
        metrics = calculate_player_metrics(
            player, fixture_records, cutoff_gameweek=cutoff_gameweek
        )

    ppm = (
        points_per_million(
            metrics.total_points_observed,
            player.now_cost,
        )
        if player.now_cost != 0
        else None
    )

    avg_diff = _calculate_avg_fixture_difficulty(
        player.team_id, fixtures or [], cutoff_gameweek
    )

    return PlayerBaseline(
        player_id=player.id,
        name=player.web_name,
        full_name=player.full_name,
        position=player.position,
        team_id=player.team_id,
        price_tenths=player.now_cost,
        metrics=metrics,
        points_per_million_value=ppm,
        avg_fixture_difficulty=avg_diff,
        form_3gw=metrics.points_last_3_avg_observed,
        form_5gw=metrics.points_last_5_avg_observed,
        consistency_std=metrics.points_std_dev_observed,
        cutoff_gameweek=cutoff_gameweek,
    )


# ---------------------------------------------------------------------------
# Ranking functions
# ---------------------------------------------------------------------------


def rank_players_by_value(
    players: Sequence[Player],
    fixtures_by_player: Mapping[int, Sequence[FixtureGameweekRecord]],
    fixtures_by_team: Mapping[int, Sequence[Fixture]] | None = None,
    cutoff_gameweek: int | None = None,
    min_gameweeks: int = 4,
) -> list[PlayerBaseline]:
    """Rank players by historical points-per-million value.

    Computes :func:`calculate_player_metrics` for each player using their
    fixture records, then ranks by ``points_per_million_value`` descending.

    Players are excluded if:
        - Their ``observed_gw_count`` is below ``min_gameweeks``.
        - Their price (``now_cost``) is zero.
        - Their ``points_per_million_value`` is ``None``.

    Tie-breaking (deterministic):
        1. Higher ``points_per_million_value``
        2. Higher ``total_points_observed``
        3. Higher ``total_minutes``
        4. Lower ``player_id``

    Args:
        players: Sequence of :class:`Player` entities to rank.
        fixtures_by_player: Mapping of ``player_id`` -> sequence of
            :class:`FixtureGameweekRecord` for that player.
        fixtures_by_team: Optional mapping of ``team_id`` -> :class:`Fixture`
            objects for fixture-difficulty context.
        cutoff_gameweek: Inclusive upper GW bound forwarded to
            :func:`calculate_player_metrics`.
        min_gameweeks: Minimum observed GW count for inclusion.

    Returns:
        List of :class:`PlayerBaseline` sorted by value descending.
    """
    baselines: list[PlayerBaseline] = []

    for player in players:
        records = list(fixtures_by_player.get(player.id, []))
        if not records:
            continue

        metrics = calculate_player_metrics(
            player, records, cutoff_gameweek=cutoff_gameweek
        )

        if metrics.observed_gw_count < min_gameweeks:
            continue
        if player.now_cost == 0:
            continue

        team_fixtures = (
            fixtures_by_team.get(player.team_id, []) if fixtures_by_team else None
        )
        baseline = _build_baseline(
            player, records, team_fixtures, cutoff_gameweek, metrics=metrics
        )

        if baseline.points_per_million_value is None:
            continue

        baselines.append(baseline)

    return sorted(
        baselines,
        key=lambda b: (
            -(b.points_per_million_value or 0.0),
            -b.total_points_observed,
            -b.metrics.total_minutes,
            b.player_id,
        ),
    )


def rank_players_by_consistency(
    players: Sequence[Player],
    fixtures_by_player: Mapping[int, Sequence[FixtureGameweekRecord]],
    cutoff_gameweek: int | None = None,
    min_gameweeks: int = 4,
) -> list[PlayerBaseline]:
    """Rank players by consistency (lower standard deviation first).

    Uses ``PlayerMetrics.points_std_dev_observed`` as the consistency measure.
    Lower standard deviation means more predictable (consistent) performance.

    Players are excluded if:
        - Their ``observed_gw_count`` is below ``min_gameweeks``.
        - Their ``consistency_std`` is ``None`` (fewer than 2 observed GWs).

    Tie-breaking (deterministic):
        1. Lower ``consistency_std`` (more consistent first)
        2. Higher ``total_points_observed``
        3. Higher ``total_minutes``
        4. Lower ``player_id``

    Args:
        players: Sequence of :class:`Player` entities to rank.
        fixtures_by_player: Mapping of ``player_id`` ->
            :class:`FixtureGameweekRecord` sequences.
        cutoff_gameweek: Inclusive upper GW bound forwarded to
            :func:`calculate_player_metrics`.
        min_gameweeks: Minimum observed GW count for inclusion.

    Returns:
        List of :class:`PlayerBaseline` sorted by std dev ascending, then
        by total points descending.
    """
    baselines: list[PlayerBaseline] = []

    for player in players:
        records = list(fixtures_by_player.get(player.id, []))
        if not records:
            continue

        metrics = calculate_player_metrics(
            player, records, cutoff_gameweek=cutoff_gameweek
        )

        if metrics.observed_gw_count < min_gameweeks:
            continue

        std = metrics.points_std_dev_observed
        if std is None:
            continue

        baseline = _build_baseline(
            player, records, None, cutoff_gameweek, metrics=metrics
        )
        baselines.append(baseline)

    return sorted(
        baselines,
        key=lambda b: (
            b.consistency_std if b.consistency_std is not None else float("inf"),
            -b.total_points_observed,
            -b.metrics.total_minutes,
            b.player_id,
        ),
    )


def rank_players_by_form(
    players: Sequence[Player],
    fixtures_by_player: Mapping[int, Sequence[FixtureGameweekRecord]],
    cutoff_gameweek: int | None = None,
    n_gameweeks: int = 5,
    min_gameweeks: int = 4,
) -> list[PlayerBaseline]:
    """Rank players by recent form (last-N gameweek average points).

    Uses ``PlayerMetrics`` form averages for the requested ``n_gameweeks``
    window.  Higher average = better form.

    Players are excluded if:
        - Their ``observed_gw_count`` is below ``min_gameweeks``.
        - Their form score is ``None`` (insufficient observations for window).

    Tie-breaking (deterministic):
        1. Higher form score
        2. Higher ``total_points_observed``
        3. Higher ``total_minutes``
        4. Lower ``player_id``

    Args:
        players: Sequence of :class:`Player` entities to rank.
        fixtures_by_player: Mapping of ``player_id`` ->
            :class:`FixtureGameweekRecord` sequences.
        cutoff_gameweek: Inclusive upper GW bound forwarded to
            :func:`calculate_player_metrics`.
        n_gameweeks: Form window size (3 or 5 recommended).
        min_gameweeks: Minimum observed GW count for inclusion.

    Returns:
        List of :class:`PlayerBaseline` sorted by form descending.
    """
    baselines: list[PlayerBaseline] = []

    for player in players:
        records = list(fixtures_by_player.get(player.id, []))
        if not records:
            continue

        metrics = calculate_player_metrics(
            player, records, cutoff_gameweek=cutoff_gameweek
        )

        if metrics.observed_gw_count < min_gameweeks:
            continue

        # Determine which form metric to use once, then reuse it for both
        # eligibility and sorting so they can never disagree.
        # n_gameweeks == 3 uses 3-gw form; every other value (including
        # custom windows) falls back to 5-gw form, matching the
        # documented behaviour.
        form_score = (
            metrics.points_last_3_avg_observed
            if n_gameweeks == 3
            else metrics.points_last_5_avg_observed
        )
        if form_score is None:
            continue

        baseline = _build_baseline(
            player, records, None, cutoff_gameweek, metrics=metrics
        )
        baselines.append(baseline)

    return sorted(
        baselines,
        key=lambda b: (
            -(b.form_5gw or 0.0)
            if n_gameweeks != 3
            else -(b.form_3gw or 0.0),
            -b.total_points_observed,
            -b.metrics.total_minutes,
            b.player_id,
        ),
    )


def compare_players(
    player_ids: Sequence[int],
    players: Sequence[Player],
    fixtures_by_player: Mapping[int, Sequence[FixtureGameweekRecord]],
    fixtures_by_team: Mapping[int, Sequence[Fixture]] | None = None,
    cutoff_gameweek: int | None = None,
) -> list[PlayerBaseline]:
    """Produce side-by-side :class:`PlayerBaseline` profiles for a subset.

    Unlike the ranking functions, this does **not** filter by minimum
    gameweeks or apply eligibility thresholds — it returns all requested
    players so the caller can inspect their full profiles, including
    ``None`` values where data is insufficient.

    The output is sorted by ``player_id`` for deterministic ordering.

    Args:
        player_ids: IDs of players to compare.
        players: Pool of :class:`Player` entities to select from.
        fixtures_by_player: Mapping of ``player_id`` ->
            :class:`FixtureGameweekRecord` sequences.
        fixtures_by_team: Optional mapping for fixture-difficulty context.
        cutoff_gameweek: Inclusive upper GW bound forwarded to
            :func:`calculate_player_metrics`.

    Returns:
        List of :class:`PlayerBaseline` for found players, sorted by
        ``player_id``.
    """
    player_map = {p.id: p for p in players}
    result: list[PlayerBaseline] = []
    seen: set[int] = set()

    for pid in player_ids:
        if pid in seen or pid not in player_map:
            continue
        seen.add(pid)

        player = player_map[pid]
        records = list(fixtures_by_player.get(pid, []))
        team_fixtures = (
            fixtures_by_team.get(player.team_id, []) if fixtures_by_team else None
        )

        baseline = _build_baseline(
            player, records, team_fixtures, cutoff_gameweek
        )
        result.append(baseline)

    return sorted(result, key=lambda b: b.player_id)
