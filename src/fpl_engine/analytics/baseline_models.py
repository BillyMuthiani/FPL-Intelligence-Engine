"""Baseline models for expected-points prediction.

Stage 4B of the FPL Intelligence Engine analytics stack.

This module provides transparent, deterministic baseline models that convert
raw FPL data (player statistics, fixture information, team records) into
:class:`~fpl_engine.analytics.prediction.MatchPrediction` objects.

The baseline models make **no machine-learning claims**.  They are intentionally
simple, explainable functions that serve as:

- A reproducible starting point for expected-points estimation.
- A performance baseline against which more complex models are compared.
- A fall-back when advanced features (e.g. pre-trained xG models) are unavailable.

Design principles:
    - **Transparency**: Every assumption is documented and traceable.
    - **Honest uncertainty**: Unavailable data is preserved as ``None``, not
      silently replaced with misleading defaults.
    - **No data leakage**: The module performs no I/O.  Temporal filtering
      is delegated to :func:`_validate_cutoff` and forward-passed to the
      defensive-contribution model, which applies leakage-safe look-back
      windows when a ``prediction_cutoff`` is supplied.  Callers remain
      responsible for ensuring only information available at prediction
      time is supplied for non-DC features.
    - **Contract completeness**: :func:`predict_match_baseline` returns a valid
      :class:`~fpl_engine.analytics.prediction.MatchPrediction` with all
      ``None`` values resolved to 0.0 at the boundary.

Approximations and their limitations:
    - **Goals/assists/saves/bonus/own goals**: Season per-90 rates are
      projected to fixture-level expected counts by multiplying by
      ``exp_minutes / 90.0``.  ``exp_minutes`` is the unconditional
      expected minutes (FPL's ``expected_minutes`` when available,
      otherwise ``90.0 * start_prob``).  Bayesian shrinkage toward a
      position benchmark is available via :func:`_shrink_rate` but is
      not yet wired into the production pipeline.
    - **Goals conceded**: Projected using a Poisson team model on defensive
      and offensive strength. The linear ``floor(gc/2)`` deduction is applied
      inside :func:`~fpl_engine.analytics.prediction.predict_expected_points`.
    - **Saves**: Projected from per-90 historical rate (goalkeeper-only).
    - **Clean sheet**: Poisson probability ``P(0) = exp(-lambda)`` where
      ``lambda`` is the expected goals conceded rate for the team vs.
      opponent.  Treated as P(CS | 60+ min) by the scoring engine under
      an independence assumption.
    - **Playing time**: Derived from FPL's ``chance_of_playing_next_round``
      (with status fallback) and ``expected_minutes``.  The P(60+) heuristic
      is **UNCALIBRATED** and should be replaced by a data-driven model.
    - **Yellow/red cards**: Not exposed by the FPL player API, so always 0.0.
    - **Penalties**: Not modeled (insufficient data for reliable rate estimation).
    - **Defensive contribution**: Estimated from per-gameweek match history
      via the Stage 4H :mod:`~fpl_engine.analytics.defensive_probability`
      module.  The historical-frequency baseline (default) computes the
      shrunken proportion of prior appearances reaching the DC action
      threshold.  Falls back to a position-specific prior when no history
      is available.  GKP is ineligible (always 0.0).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fpl_engine.analytics.defensive_probability import (
    DC_DEFAULT_MINUTES,
    predict_defensive_contribution_probability,
)
from fpl_engine.analytics.prediction import MatchPrediction
from fpl_engine.analytics.scoring import FPLScoringRules
from fpl_engine.models import (
    Fixture,
    Player,
    PlayerMatchHistory,
    PlayerPosition,
    PlayerStatus,
    Team,
)

__all__ = [
    "BaselineFeatures",
    "BaselineModelError",
    "PositionBenchmark",
    "compute_position_benchmarks",
    "extract_baseline_features",
    "predict_match_baseline",
]

# Historical EPL average goals per team per match (2014-2024).
# Used as the denominator in the Poisson rate adjustment.
LEAGUE_AVG_GOALS_PER_TEAM: float = 2.72 / 2.0  # ~ 1.36

# Bayesian shrinkage strength (kappa).  Higher values apply more regularization
# toward the prior rate.  A value of 3.0 corresponds to ~3 matches of
# pseudo-observation weight, suitable for early-season stabilization.
SHRINKAGE_STRENGTH: float = 3.0

# Minimum minutes required for a player to contribute to position benchmarks.
# Filters out bench players whose per-90 rates are not representative.
MIN_BENCHMARK_MINUTES: int = 270  # ~ 3 full matches

# FPL awards a maximum of 3 bonus points per match.
MAX_BONUS_PER_MATCH: float = 3.0


def _is_finite_number(value: Any) -> bool:
    """Return True if *value* is a finite int or float, but **not** bool.

    Rejects ``NaN`` and ``±inf``, which are technically floats but would
    break arithmetic and validation semantics downstream.  ``bool`` is
    rejected because ``True``/``False`` are instances of ``int`` in Python.
    """
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_valid_probability_value(value: Any) -> bool:
    """Return True if *value* is a real number strictly in [0, 1]."""
    return _is_finite_number(value) and 0.0 <= float(value) <= 1.0


class BaselineModelError(ValueError):
    """Raised when baseline model construction encounters invalid inputs."""


def _validate_cutoff(dt: datetime | None, label: str) -> None:
    """Raise :class:`BaselineModelError` if *dt* is present but not UTC-aware.

    Parameters
    ----------
    dt
        The datetime (or ``None``) to validate.
    label
        Human-readable name used in the error message.
    """
    if dt is None:
        return
    if dt.tzinfo is None:
        raise BaselineModelError(  # noqa: TRY003
            f"{label} must be timezone-aware, got naive datetime"
        )
    if dt.utcoffset() != UTC.utcoffset(None):
        raise BaselineModelError(  # noqa: TRY003
            f"{label} must be in UTC, got {dt.tzinfo!r}"
        )




@dataclass(frozen=True)
class PositionBenchmark:
    """Position-average per-90 rates computed from a pool of players.

    These rates serve as Bayesian priors for shrinkage.  Players with fewer
    than MIN_BENCHMARK_MINUTES minutes played are excluded from the
    averages.

    Attributes:
        position: The player position this benchmark represents.
        xg_per_90: Average expected goals per 90 minutes for the position.
        xa_per_90: Average expected assists per 90 minutes.
        saves_per_90: Average saves per 90 minutes (meaningful for GK only).
        bonus_per_90: Average bonus points per 90 minutes.
        matches_played: Average matches played (minutes / 90) for eligible players.
    """

    position: PlayerPosition
    xg_per_90: float = 0.0
    xa_per_90: float = 0.0
    saves_per_90: float = 0.0
    bonus_per_90: float = 0.0
    matches_played: float = 0.0


@dataclass(frozen=True)
class BaselineFeatures:
    """All baseline features for a single player-fixture pair.

    This dataclass preserves None for unavailable expectations to
    distinguish "data unavailable" from "genuine zero".
    Use predict_match_baseline to resolve None to 0.0 when
    constructing a MatchPrediction.

    Attributes:
        player_id: FPL player identifier.
        fixture_id: FPL fixture identifier.
        gameweek: FPL gameweek number.
        position: Player position.
        is_home: Whether the player"s team is playing at home.

        start_probability: P(starts the match), in [0, 1].
        probability_appears: P(plays at least 1 minute), in [0, 1].
            Equals start_probability for the baseline (no sub model).
        probability_60_plus: P(plays 60+ minutes), in [0, 1].
        clean_sheet_probability: Poisson P(team keeps CS), or None
            if team data is unavailable.  The scoring engine treats this
            as an approximation of P(CS | player plays 60+ min) under the
            independence assumption (documented in prediction module).

        expected_goals: Fixture-level expected goals (per-90 rate
            projected to expected fixture minutes).  Always non-None;
            0.0 when the player lacks sufficient data.
        expected_assists: Fixture-level expected assists (per-90 rate
            projected to expected fixture minutes).  Always non-None;
            0.0 when the player lacks sufficient data.
        expected_goals_conceded: Fixture-level expected goals conceded
            (per-90 rate projected to expected fixture minutes).  None
            for non-GK/DEF positions or when data is unavailable.
        expected_saves: Fixture-level expected saves (per-90 rate
            projected to expected fixture minutes).  None for non-GK
            or when saves data is unavailable.
        expected_bonus: Projected bonus points in FPL-point units,
            capped at [0, 3] (per-90 rate projected to expected fixture
            minutes).  None if bonus or minutes data is unavailable.
        expected_own_goals: Fixture-level expected own goals (per-90
            rate projected to expected fixture minutes).  None when the
            player has fewer than MIN_BENCHMARK_MINUTES minutes.

        expected_yellow_cards: Always 0.0 (FPL API does not expose YC totals).
        expected_red_cards: Always 0.0 (FPL API does not expose RC totals).
        expected_penalties_saved: Always 0.0 (insufficient data for rate).
        expected_penalties_missed: Always 0.0 (insufficient data for rate).
        defensive_contribution_probability: P(DC threshold met | player
            appears), estimated from per-gameweek history when available
            (Stage 4H).  Falls back to a position-specific prior when
            no history is supplied.  GKP always 0.0 (ineligible).

        cutoff_gameweek: Metadata only. Does NOT filter player aggregates.
        expected_minutes: Resolved expected minutes for the fixture
            (unconditional — incorporates appearance probability when
            FPL's expected_minutes is not supplied).
        matches_played: Player's career minutes / 90.
        defensive_contribution_probability_from_history: Whether the
            DC probability was derived from the player's personal history
            (True) or falls back to the population prior (False).  False
            when no prior appearances were available, and for GKP
            (ineligible).  Informational only — does not affect scoring.
    """

    # Identifiers
    player_id: int
    fixture_id: int
    gameweek: int
    position: PlayerPosition
    is_home: bool

    # Probabilities
    start_probability: float
    probability_appears: float
    probability_60_plus: float
    clean_sheet_probability: float | None

    # Count/rate expectations (None = unavailable)
    expected_goals: float | None
    expected_assists: float | None
    expected_goals_conceded: float | None
    expected_saves: float | None
    expected_bonus: float | None
    expected_own_goals: float | None

    # Unavailable / always zero (documented for completeness)
    expected_yellow_cards: float
    expected_red_cards: float
    expected_penalties_saved: float
    expected_penalties_missed: float
    defensive_contribution_probability: float

    # Metadata
    cutoff_gameweek: int | None
    expected_minutes: float
    matches_played: float
    defensive_contribution_probability_from_history: bool = True

    def __post_init__(self) -> None:
        """Validate field contracts at construction time.

        Raises :class:`BaselineModelError` for invalid values.  ``None``
        is preserved for optional fields — this validation only rejects
        malformed non-``None`` values (NaN, inf, bool, out-of-range).
        """
        # --- Probability fields (non-optional, [0, 1]) ---
        for name in (
            "start_probability",
            "probability_appears",
            "probability_60_plus",
            "defensive_contribution_probability",
        ):
            value = getattr(self, name)
            if not _is_valid_probability_value(value):
                raise BaselineModelError(  # noqa: TRY003
                    f"{name} must be a finite probability in [0, 1], got {value!r}"
                )

        # --- clean_sheet_probability (optional, None or [0, 1]) ---
        cs = self.clean_sheet_probability
        if cs is not None and not _is_valid_probability_value(cs):
            raise BaselineModelError(  # noqa: TRY003
                f"clean_sheet_probability must be a probability in [0, 1] "
                f"or None, got {cs!r}"
            )

        # --- Count/rate fields (optional, None or non-negative finite) ---
        for name in (
            "expected_goals",
            "expected_assists",
            "expected_goals_conceded",
            "expected_saves",
            "expected_own_goals",
        ):
            value = getattr(self, name)
            if value is not None and (not _is_finite_number(value) or float(value) < 0):
                raise BaselineModelError(  # noqa: TRY003
                    f"{name} must be a non-negative finite number or None, "
                    f"got {value!r}"
                )

        # --- expected_bonus (optional, None or [0, 3]) ---
        bonus = self.expected_bonus
        if bonus is not None and (
            not _is_finite_number(bonus)
            or not 0.0 <= float(bonus) <= MAX_BONUS_PER_MATCH
        ):
                raise BaselineModelError(  # noqa: TRY003
                    f"expected_bonus must be in [0, {MAX_BONUS_PER_MATCH}] "
                    f"or None, got {bonus!r}"
                )

        # --- Always-zero fields (non-optional, non-negative finite) ---
        for name in (
            "expected_yellow_cards",
            "expected_red_cards",
            "expected_penalties_saved",
            "expected_penalties_missed",
        ):
            value = getattr(self, name)
            if not _is_finite_number(value) or float(value) < 0:
                raise BaselineModelError(  # noqa: TRY003
                    f"{name} must be a non-negative finite number, got {value!r}"
                )



# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_per_90(total: float | None, minutes: int) -> float | None:
    """Convert a season total to a per-90 rate.

    Returns None if total is None or minutes is zero,
    since dividing by zero is undefined.
    """
    if total is None or minutes <= 0:
        return None
    return total / minutes * 90.0


def _shrink_rate(observed_rate: float, prior_rate: float, matches: float) -> float:
    """Bayesian shrinkage of observed rate toward a prior.

    Blend: shrunk = (observed * n + prior * kappa) / (n + kappa)
    where n is matches played and kappa is SHRINKAGE_STRENGTH.
    """
    if matches <= 0:
        return prior_rate
    return (
        observed_rate * matches + prior_rate * SHRINKAGE_STRENGTH
    ) / (matches + SHRINKAGE_STRENGTH)


def _compute_start_probability(player: Player) -> float:
    """Derive P(starts the match) from FPL data.

    Priority: chance_of_playing_next_round, then status fallback.
    """
    cop = player.chance_of_playing_next_round
    if cop is not None:
        return max(0.0, min(cop / 100.0, 1.0))
    if player.status == PlayerStatus.DOUBTFUL:
        return 0.5
    if player.status == PlayerStatus.AVAILABLE:
        return 1.0
    return 0.0


def _resolve_expected_minutes(player: Player, start_prob: float) -> float:
    """Resolve expected minutes for the fixture.

    Uses FPL ``expected_minutes`` when available (the player's
    unconditional expected minutes for the gameweek).  Otherwise
    infers ``90.0 * start_prob``: a proportional estimate reflecting
    both minutes-if-playing and probability-of-appearing.

    The returned value is the **unconditional** expected minutes
    (i.e. it already incorporates appearance probability).  Per-90
    rates in :func:`extract_baseline_features` are multiplied by
    ``exp_minutes / 90.0`` to obtain fixture-level expected counts —
    no additional appearance-probability multiplier is needed.
    """
    if player.expected_minutes is not None:
        return float(player.expected_minutes)
    return 90.0 * start_prob


def _heuristic_60plus_factor(expected_minutes: float) -> float:
    """UNCALIBRATED heuristic for P(60+ minutes | player appears).

    Maps expected minutes to a probability. Not calibrated to real
    bench/rotation data. Should be replaced by a data-driven model.

    - >= 90 min -> 0.95 (likely starter, slight rotation risk)
    - >= 60 min -> 0.75 (rotation risk)
    - >= 30 min -> 0.20 (bench player, unlikely to reach 60)
    - < 30 min -> 0.0
    """
    if expected_minutes >= 90:
        return 0.95
    if expected_minutes >= 60:
        return 0.75
    if expected_minutes >= 30:
        return 0.20
    return 0.0


def _safe_mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of values, or 0.0 if empty."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _project_goals_conceded_rate(team: Team, opponent: Team, is_home: bool) -> float:
    """Project expected goals conceded for team vs opponent.

    Uses team defensive and opponent offensive strength normalized
    by league average and venue (home/away).

    Strengths are on a 1-5 scale where 3.0 = league average.
    """
    if is_home:
        team_def = team.strength_defence_home
        opp_atk = opponent.strength_attack_away
    else:
        team_def = team.strength_defence_away
        opp_atk = opponent.strength_attack_home
    if team_def <= 0 or opp_atk <= 0:
        return LEAGUE_AVG_GOALS_PER_TEAM
    norm = 3.0
    # Stronger own defence (team_def) should *reduce* expected goals conceded,
    # while stronger opponent attack (opp_atk) should *increase* it.
    # FPL strength values are on a 1-5 scale where 3.0 = league average.
    return LEAGUE_AVG_GOALS_PER_TEAM * (opp_atk / norm) / (team_def / norm)



def compute_position_benchmarks(players: Sequence[Player]) -> dict[PlayerPosition, PositionBenchmark]:
    """Compute per-position benchmarks from a population of players.

    Aggregates season-long statistics to per-90 rates and computes
    the arithmetic mean of each rate within each position bucket.
    These benchmarks serve as Bayesian priors for the baseline models.

    Only players with at least MIN_BENCHMARK_MINUTES minutes
    are included in the benchmark calculation, so that bench players
    do not distort the per-90 rates.

    Args:
        players: Sequence of historical Player objects with season totals.
    Returns:
        Dict mapping PlayerPosition to PositionBenchmark.
    """
    buckets: dict[PlayerPosition, list[Player]] = {
        pos: [] for pos in PlayerPosition
    }
    for p in players:
        buckets[p.position].append(p)

    result: dict[PlayerPosition, PositionBenchmark] = {}
    for pos, bucket_players in buckets.items():
        eligible = [
            p for p in bucket_players if p.minutes >= MIN_BENCHMARK_MINUTES
        ]
        if not eligible:
            result[pos] = PositionBenchmark(
                position=pos,
                xg_per_90=0.0, xa_per_90=0.0, saves_per_90=0.0, bonus_per_90=0.0,
                matches_played=0.0,
            )
            continue
        n_matches = _safe_mean([p.minutes / 90.0 for p in eligible])
        xg_rates = [
            r for r in (_to_per_90(p.expected_goals, p.minutes) for p in eligible)
            if r is not None
        ]
        xa_rates = [
            r for r in (_to_per_90(p.expected_assists, p.minutes) for p in eligible)
            if r is not None
        ]
        saves_rates = [
            r for r in (_to_per_90(p.saves, p.minutes) for p in eligible)
            if r is not None
        ]
        bonus_rates = [
            r for r in (_to_per_90(p.bonus, p.minutes) for p in eligible)
            if r is not None
        ]
        result[pos] = PositionBenchmark(
            position=pos,
            xg_per_90=_safe_mean(xg_rates),
            xa_per_90=_safe_mean(xa_rates),
            saves_per_90=_safe_mean(saves_rates),
            bonus_per_90=_safe_mean(bonus_rates),
            matches_played=n_matches,
        )
    return result



def extract_baseline_features(
    player: Player,
    fixture: Fixture,
    team: Team | None,
    opponent: Team | None,
    benchmarks: dict[PlayerPosition, PositionBenchmark],
    gameweek: int,
    cutoff_gameweek: int | None = None,
    history: Sequence[PlayerMatchHistory] | None = None,
    prediction_cutoff: datetime | None = None,
) -> BaselineFeatures:
    """Extract baseline features for a single player-fixture pair.

    Combines player season statistics, fixture context, team strength,
    position benchmarks, and FPL availability data into a
    BaselineFeatures object.

    Per-90 season rates (xG, xA, xGC, saves, bonus, own goals) are
    projected to fixture-level expected counts by multiplying by
    ``exp_minutes / 90.0``.  ``exp_minutes`` is the unconditional
    expected minutes — FPL's ``expected_minutes`` when available,
    otherwise ``90.0 * start_prob``.  This means appearance probability
    is incorporated exactly once via the projection factor.

    Args:
        player: Player with season statistics.
        fixture: Fixture providing team IDs and venue.
        team: Player's team, or None if unavailable.
        opponent: Opponent team, or None if unavailable.
        benchmarks: Position benchmarks from compute_position_benchmarks.
        gameweek: FPL gameweek number.
        cutoff_gameweek: Optional metadata for temporal filtering.
        history: Per-gameweek match history for the player (from the
            element-summary endpoint).  Only entries with ``gameweek <
            gameweek`` and ``minutes > 0`` are used to estimate the
            defensive-contribution probability.  When None or empty,
            a position-specific prior is used as a conservative fallback.
        prediction_cutoff: Optional UTC datetime representing the latest
            point at which information was available for the prediction.
            When supplied, only history records with ``outcome_time`` at
            or before the cutoff are used, preventing leakage from
            postponed or rescheduled fixtures whose gameweek label is
            earlier but whose outcome was observed later.

    Returns:
        BaselineFeatures with all baseline expectations for the fixture.
    """
    _validate_cutoff(prediction_cutoff, "prediction_cutoff")

    is_home = fixture.is_team_home(player.team_id)
    start_prob = _compute_start_probability(player)
    exp_minutes = _resolve_expected_minutes(player, start_prob)
    # _heuristic_60plus_factor estimates P(60+ | appears) and expects
    # conditional minutes (E[minutes | appears]).  _resolve_expected_minutes
    # always returns unconditional expected minutes (either the FPL-supplied
    # expected_minutes or the 90 * start_prob fallback), so de-dilute by
    # start_prob to recover conditional minutes before applying the heuristic.
    # When start_prob is 0, the player cannot play, so P(60+) is 0 regardless
    # of expected_minutes (avoiding division-by-zero).
    if start_prob > 0.0:
        prob_60_plus = _heuristic_60plus_factor(exp_minutes / start_prob) * start_prob
    else:
        prob_60_plus = 0.0
    bm = benchmarks.get(player.position)
    if bm is None:
        bm = PositionBenchmark(
            position=player.position,
            xg_per_90=0.0, xa_per_90=0.0, saves_per_90=0.0, bonus_per_90=0.0,
            matches_played=0.0,
        )

    matches_played = player.minutes / 90.0 if player.minutes > 0 else 0.0

    # Projection factor: converts per-90 rates to fixture-level expected counts.
    # exp_minutes is unconditional (already incorporates appearance probability),
    # so this is applied exactly once — no additional start_prob multiplier.
    # When start_prob is 0, the player cannot play; zero the projection factor
    # to ensure all playing-dependent expectations are zero, preserving
    # E[events] = P(appears) * E[events | appears] = 0.  The exp_minutes
    # metadata retains the resolved value for reference.
    projection_factor = exp_minutes / 90.0 if start_prob > 0.0 else 0.0

    # -- Expected goals --
    xg = player.expected_goals_per_90
    if xg is None and player.expected_goals is not None and player.minutes > 0:
        xg = _to_per_90(player.expected_goals, player.minutes)
    if xg is None and player.minutes >= MIN_BENCHMARK_MINUTES:
        xg = bm.xg_per_90
    if xg is None:
        xg = 0.0
    xg = xg * projection_factor

    # -- Expected assists --
    xa = player.expected_assists_per_90
    if xa is None and player.expected_assists is not None and player.minutes > 0:
        xa = _to_per_90(player.expected_assists, player.minutes)
    if xa is None and player.minutes >= MIN_BENCHMARK_MINUTES:
        xa = bm.xa_per_90
    if xa is None:
        xa = 0.0
    xa = xa * projection_factor

    # -- Expected goals conceded (GK/DEF only) --
    xgc = player.expected_goals_conceded_per_90
    if xgc is None and player.expected_goals_conceded is not None and player.minutes > 0:
        xgc = _to_per_90(player.expected_goals_conceded, player.minutes)
    if player.position not in (PlayerPosition.GOALKEEPER, PlayerPosition.DEFENDER):
        xgc = None
    if xgc is not None:
        xgc = xgc * projection_factor

    # -- Expected saves (GK only) --
    saves_rate = _to_per_90(player.saves, player.minutes) if player.position == PlayerPosition.GOALKEEPER else None
    if saves_rate is not None:
        saves_rate = saves_rate * projection_factor

    # -- Expected bonus --
    bonus_rate = _to_per_90(player.bonus, player.minutes)
    if bonus_rate is not None:
        # Project the season per-90 rate to expected fixture minutes,
        # then cap at the FPL maximum of MAX_BONUS_PER_MATCH.
        bonus_rate = min(bonus_rate * projection_factor, MAX_BONUS_PER_MATCH)

    # -- Clean sheet probability via Poisson model --
    cs_prob = None
    if team is not None and opponent is not None:
        lam = _project_goals_conceded_rate(team, opponent, is_home)
        cs_prob = math.exp(-lam)

    # -- Expected own goals --
    og = None
    if player.minutes >= MIN_BENCHMARK_MINUTES and player.minutes > 0:
        og = player.own_goals / player.minutes * 90.0
    if og is not None:
        og = og * projection_factor

    # -- Defensive contribution probability --
    # P(DC threshold met | player appears), estimated from per-gameweek
    # match history.  Uses only observations strictly before the target
    # fixture (leakage guard).  GKP is ineligible (returns 0.0).
    # When no history is supplied, falls back to a position-specific prior.
    #
    # This probability is *conditional on appearance*.  The scoring
    # engine applies P(appear) exactly once:
    #   E[DC points] = P(appear) * P(DC | appear) * DC_points
    # so we do NOT multiply by start_prob here.
    conditional_minutes = (
        exp_minutes / start_prob if start_prob > 0.0 else DC_DEFAULT_MINUTES
    )
    dc_result = predict_defensive_contribution_probability(
        history=history,
        position=player.position,
        target_gameweek=gameweek,
        expected_minutes=conditional_minutes,
        prediction_cutoff=prediction_cutoff,
    )
    dc_prob = dc_result.probability
    dc_from_history = dc_result.n_observations > 0

    return BaselineFeatures(
        player_id=player.id,
        fixture_id=fixture.id,
        gameweek=gameweek,
        position=player.position,
        is_home=is_home,

        start_probability=start_prob,
        probability_appears=start_prob,
        probability_60_plus=prob_60_plus,
        clean_sheet_probability=cs_prob,

        expected_goals=xg,
        expected_assists=xa,
        expected_goals_conceded=xgc,
        expected_saves=saves_rate,
        expected_bonus=bonus_rate,
        expected_own_goals=og,

        expected_yellow_cards=0.0,
        expected_red_cards=0.0,
        expected_penalties_saved=0.0,
        expected_penalties_missed=0.0,
        defensive_contribution_probability=dc_prob,
        defensive_contribution_probability_from_history=dc_from_history,

        cutoff_gameweek=cutoff_gameweek,
        expected_minutes=exp_minutes,
        matches_played=matches_played,
    )



def predict_match_baseline(
    features: BaselineFeatures,
    rules: FPLScoringRules | None = None,  # noqa: ARG001
) -> MatchPrediction:
    """Convert baseline features into a MatchPrediction.

    Resolves all None values to 0.0 at the boundary, ensuring
    the returned MatchPrediction is fully specified and can be
    passed directly to predict_expected_points for scoring.

    Args:
        features: BaselineFeatures extracted via extract_baseline_features.
        rules: Scoring rules (accepted for API symmetry; not used
             in the baseline model but available for future hooks).
    Returns:
        A MatchPrediction with None values resolved to 0.0.
    """

    def _resolve(value: float | None) -> float:
        return value if value is not None else 0.0

    return MatchPrediction(
        fixture_id=features.fixture_id,
        gameweek=features.gameweek,
        start_probability=features.start_probability,
        probability_appears=features.probability_appears,
        probability_60_plus=features.probability_60_plus,
        clean_sheet_probability=_resolve(features.clean_sheet_probability),
        defensive_contribution_probability=features.defensive_contribution_probability,
        defensive_contribution_probability_from_history=features.defensive_contribution_probability_from_history,
        expected_goals=_resolve(features.expected_goals),
        expected_assists=_resolve(features.expected_assists),
        expected_goals_conceded=_resolve(features.expected_goals_conceded),
        expected_saves=_resolve(features.expected_saves),
        expected_yellow_cards=features.expected_yellow_cards,
        expected_red_cards=features.expected_red_cards,
        expected_own_goals=_resolve(features.expected_own_goals),
        expected_penalties_saved=features.expected_penalties_saved,
        expected_penalties_missed=features.expected_penalties_missed,
        expected_bonus=_resolve(features.expected_bonus),
    )
