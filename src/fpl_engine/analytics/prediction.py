"""Prediction contracts and expected-points calculator for FPL.

Stage 4 - Prediction Engine:
    - MatchPrediction - probabilistic prediction for a single fixture
    - PlayerGameweekPrediction - aggregated fixtures prediction for a GW
    - ExpectedPointsResult - expected FPL points with component breakdown
    - predict_expected_points - pure function converting predictions to
      expected points
    - PredictionError - raised on invalid prediction inputs

This module provides the bridge between probabilistic match predictions
(produced by a future modelling layer) and FPL expected points. It is a
pure, deterministic transformation - it performs no I/O and makes no
claims about temporal safety. Preventing data leakage (e.g., excluding
fixtures beyond a cutoff gameweek) is the responsibility of the caller
that constructs PlayerGameweekPrediction objects.

Scoring formulas
----------------
Exact in expectation (linear in the count/rate):
    Goals, assists, yellow cards, red cards, own goals, penalty misses.
    Each is multiplied by a fixed per-unit point value from
    FPLScoringRules.

Exact in expectation (probability-based):
    Appearance points (three mutually exclusive outcomes), penalty saves
    and misses (0/1 indicators - expected value equals probability).

Approximation (nonlinear / threshold-based):
    Goals conceded: FPL deducts points via floor(gc / 2), a nonlinear
    function. The linear approximation E[gc] / 2 * penalty ignores
    the floor quantisation and is exact only when E[gc] is an even
    integer. The exact value requires the full goals-conceded
    distribution.

    Saves (GK only): FPL awards points via
    floor(saves / saves_per_point), also nonlinear. The linear
    approximation E[saves] / saves_per_point * pts is exact only at
    multiples of saves_per_point. Floor division is intentionally
    not applied to the expected value.

Probability-based (explicit conditional probability supplied by caller):
    Clean sheet: P(60+ min) * P(CS) * CS_points
    where P(CS) is the team's clean-sheet probability.  The scoring
    engine treats P(CS) as an approximation of P(CS | player plays 60+
    min) under the independence assumption: the player's 60+ minute
    appearance and the team keeping a clean sheet are assumed
    independent.  The P(60+) factor is applied exactly once.

    Defensive contribution: P(appear) * P(threshold met | appear) *
    DC_points. The probability of meeting the CBIT/CBIRT threshold is
    supplied directly by the caller, not derived from expected defensive
    stats.

Input categories
----------------
Fixture-level expected counts (can exceed 1):
    expected_goals, expected_assists, expected_goals_conceded,
    expected_saves, expected_yellow_cards, expected_red_cards,
    expected_own_goals.

Probabilities (in [0, 1]):
    start_probability (informational only), probability_appears,
    probability_60_plus, clean_sheet_probability,
    defensive_contribution_probability, expected_penalties_saved,
    expected_penalties_missed.

Already in FPL-point units:
    expected_bonus (validated to [0, 3]).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from fpl_engine.analytics.scoring import (
    DEFAULT_SCORING_RULES,
    FPLScoringRules,
)
from fpl_engine.models.player import PlayerPosition

__all__ = [
    "ExpectedPointsResult",
    "MatchPrediction",
    "PlayerGameweekPrediction",
    "PredictionError",
    "predict_expected_points",
]


class PredictionError(ValueError):
    """Raised when prediction contract validation fails."""


def _is_real_int(value: Any) -> bool:
    """Return True only if *value* is an int and NOT a bool."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real_float(value: Any) -> bool:
    """Return True only if *value* is a finite float or int, but NOT a bool.

    Rejects ``NaN`` and ``±inf``, which are technically floats but would
    break arithmetic and validation semantics downstream.
    """
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_valid_probability(value: Any) -> bool:
    """Return True if *value* is a real number in [0, 1]."""
    return _is_real_float(value) and 0.0 <= value <= 1.0


@dataclass(frozen=True)
class MatchPrediction:
    """Probabilistic prediction for a single fixture.

    The ``start_probability`` field is informational only - it is
    validated but not used in point calculations. This distinguishes
    "starts" from "appears": a player may start but be substituted
    before playing any minutes, or come on as a substitute without
    starting. Bench probability is therefore NOT defined as
    ``1 - start_probability`` - ``probability_appears`` already captures
    the probability of getting on the pitch at all.

    Attributes:
        fixture_id: FPL fixture identifier.
        gameweek: FPL gameweek number, must be in [1, 38].
        start_probability: P(starts the match) in [0, 1]. Informational only.
        probability_appears: P(player plays at least 1 minute) in [0, 1].
        probability_60_plus: P(player plays 60+ minutes) in [0, 1].
            Must be <= probability_appears.
        clean_sheet_probability: P(team keeps CS).  The scoring engine
            treats this as an approximation of P(CS | player plays 60+ min)
            under the independence assumption (see module docstring).
        defensive_contribution_probability: P(threshold met | player plays).
        defensive_contribution_probability_from_history: Whether the DC
            probability was derived from the player's personal history (True)
            or falls back to the population prior (False).  False for GKP
            (ineligible) and when no prior appearances were available.
            Informational only — does not affect scoring.
        expected_goals: Fixture-level expected goals (expected count).
        expected_assists: Fixture-level expected assists (expected count).
        expected_goals_conceded: Fixture-level expected goals conceded
            (expected count).  See module docstring for approximation.
        expected_saves: Fixture-level expected saves (GK only).  See
            module docstring for approximation.
        expected_penalties_saved: P(penalty saved) in [0, 1].
            One-event-per-fixture assumption.
        expected_penalties_missed: P(penalty missed) in [0, 1].
            One-event-per-fixture assumption.
        expected_yellow_cards: Expected yellow cards (expected count).
        expected_red_cards: Expected red cards (expected count).
        expected_own_goals: Expected own goals (expected count).
        expected_bonus: Expected bonus, FPL-point units, in [0, 3].
    """

    fixture_id: int
    gameweek: int
    start_probability: float = 0.0
    probability_appears: float = 0.0
    probability_60_plus: float = 0.0
    clean_sheet_probability: float = 0.0
    defensive_contribution_probability: float = 0.0
    defensive_contribution_probability_from_history: bool = True
    expected_goals: float = 0.0
    expected_assists: float = 0.0
    expected_goals_conceded: float = 0.0
    expected_saves: float = 0.0
    expected_yellow_cards: float = 0.0
    expected_red_cards: float = 0.0
    expected_own_goals: float = 0.0
    expected_penalties_saved: float = 0.0
    expected_penalties_missed: float = 0.0
    expected_bonus: float = 0.0

    def __post_init__(self) -> None:
        # Identifier validation
        if not _is_real_int(self.fixture_id):
            raise PredictionError(  # noqa: TRY003
                f"fixture_id must be an int, got {type(self.fixture_id).__name__} "
                f"({self.fixture_id!r})"
            )
        if not _is_real_int(self.gameweek):
            raise PredictionError(  # noqa: TRY003
                f"gameweek must be an int, got {type(self.gameweek).__name__} ({self.gameweek!r})"
            )
        if not 1 <= self.gameweek <= 38:
            raise PredictionError(  # noqa: TRY003
                f"gameweek must be in [1, 38], got {self.gameweek}"
            )

        # Probability fields in [0, 1]
        for name in (
            "start_probability",
            "probability_appears",
            "probability_60_plus",
            "clean_sheet_probability",
            "defensive_contribution_probability",
        ):
            value = getattr(self, name)
            if not _is_valid_probability(value):
                raise PredictionError(  # noqa: TRY003
                    f"{name} must be a probability in [0, 1], got {value!r}"
                )

        # Penalty fields are probabilities (one event per fixture)
        for name in ("expected_penalties_saved", "expected_penalties_missed"):
            value = getattr(self, name)
            if not _is_valid_probability(value):
                raise PredictionError(  # noqa: TRY003
                    f"{name} must be a probability in [0, 1] "
                    f"(at most one event per fixture), got {value!r}"
                )

        # Count/rate expectation fields >= 0
        for name in (
            "expected_goals",
            "expected_assists",
            "expected_goals_conceded",
            "expected_saves",
            "expected_yellow_cards",
            "expected_red_cards",
            "expected_own_goals",
        ):
            value = getattr(self, name)
            if not _is_real_float(value) or value < 0:
                raise PredictionError(  # noqa: TRY003
                    f"{name} must be a non-negative number, got {value!r}"
                )

        # Bonus is already in FPL-point units in [0, 3]
        if not _is_real_float(self.expected_bonus) or not 0.0 <= self.expected_bonus <= 3.0:
            raise PredictionError(  # noqa: TRY003
                f"expected_bonus must be in [0, 3] (FPL-point units), got {self.expected_bonus!r}"
            )

        # Probability ordering: 60+ cannot exceed appears
        if self.probability_60_plus > self.probability_appears:
            raise PredictionError(  # noqa: TRY003
                f"probability_60_plus ({self.probability_60_plus}) must be <= "
                f"probability_appears ({self.probability_appears})"
            )


@dataclass(frozen=True)
class PlayerGameweekPrediction:
    """Prediction for a player across a single gameweek (supports DGW).

    All MatchPrediction entries must share the same gameweek and have
    unique fixture_id values.

    Attributes:
        player_id: FPL player identifier.
        position: Player position (used for scoring-rule lookups).
        gameweek: FPL gameweek number, must be in [1, 38].
        fixtures: Tuple of MatchPrediction, one per fixture. Non-empty;
            all fixture_id values must be unique and all gameweek values
            must match self.gameweek.
    """

    player_id: int
    position: PlayerPosition
    gameweek: int
    fixtures: tuple[MatchPrediction, ...]

    def __post_init__(self) -> None:
        if not _is_real_int(self.player_id):
            raise PredictionError(  # noqa: TRY003
                f"player_id must be an int, got {type(self.player_id).__name__} "
                f"({self.player_id!r})"
            )
        if not isinstance(self.position, PlayerPosition):
            raise PredictionError(  # noqa: TRY003
                f"position must be a PlayerPosition enum, got {type(self.position).__name__}"
            )
        if not _is_real_int(self.gameweek):
            raise PredictionError(  # noqa: TRY003
                f"gameweek must be an int, got {type(self.gameweek).__name__}"
            )
        if not 1 <= self.gameweek <= 38:
            raise PredictionError(  # noqa: TRY003
                f"gameweek must be in [1, 38], got {self.gameweek}"
            )
        if not isinstance(self.fixtures, tuple):
            raise PredictionError(  # noqa: TRY003
                f"fixtures must be a tuple, got {type(self.fixtures).__name__}"
            )
        if len(self.fixtures) == 0:
            raise PredictionError("fixtures must not be empty")  # noqa: TRY003

        seen_ids: set[int] = set()
        for mp in self.fixtures:
            if not isinstance(mp, MatchPrediction):
                raise PredictionError(  # noqa: TRY003
                    f"each fixture must be a MatchPrediction, got {type(mp).__name__}"
                )
            if mp.gameweek != self.gameweek:
                raise PredictionError(  # noqa: TRY003
                    f"fixture gameweek ({mp.gameweek}) must match "
                    f"prediction gameweek ({self.gameweek})"
                )
            if mp.fixture_id in seen_ids:
                raise PredictionError(  # noqa: TRY003
                    f"duplicate fixture_id {mp.fixture_id} in prediction"
                )
            seen_ids.add(mp.fixture_id)


@dataclass(frozen=True)
class ExpectedPointsResult:
    """Expected FPL points for a player in a gameweek.

    All expected-point fields are floats maintained at full precision.
    Rounding for display is the caller's responsibility. The
    ``total_expected_points`` field equals the sum of all component fields.

    Attributes:
        total_expected_points: Sum of all component expected points.
        expected_appearance_points: Appearance (0/1/2) points in expectation.
        expected_goal_points: Goal points in expectation.
        expected_assist_points: Assist points in expectation.
        expected_clean_sheet_points: Clean sheet points in expectation.
            Approximation - assumes independence between playing 60+
            minutes and the team keeping a clean sheet.
        expected_goals_conceded_points: Goals-conceded deduction in
            expectation. Approximation - linearisation of a floor-based
            rule.
        expected_save_points: Save points (GK only) in expectation.
            Approximation - linearisation of a floor-based rule.
        expected_penalty_save_points: Penalty-save points (GK only).
        expected_penalty_miss_points: Penalty-miss deductions.
        expected_yellow_card_points: Yellow-card deductions in expectation.
        expected_red_card_points: Red-card deductions in expectation.
        expected_own_goal_points: Own-goal deductions in expectation.
        expected_defensive_contribution_points: Defensive contribution
            points in expectation.
        expected_bonus_points: Bonus points (already in FPL-point units).
        fixture_count: Number of fixtures aggregated.
        rules: Scoring rules used (for transparency).
    """

    total_expected_points: float
    expected_appearance_points: float
    expected_goal_points: float
    expected_assist_points: float
    expected_clean_sheet_points: float
    expected_goals_conceded_points: float
    expected_save_points: float
    expected_penalty_save_points: float
    expected_penalty_miss_points: float
    expected_yellow_card_points: float
    expected_red_card_points: float
    expected_own_goal_points: float
    expected_defensive_contribution_points: float
    expected_bonus_points: float
    fixture_count: int
    rules: FPLScoringRules


def predict_expected_points(
    prediction: PlayerGameweekPrediction,
    rules: FPLScoringRules | None = None,
) -> ExpectedPointsResult:
    """Convert a probabilistic prediction into expected FPL points.

    This is a pure, deterministic function. It makes no I/O calls,
    performs no network access, and does not modify any external state.
    It does not enforce temporal safety (e.g., cutoff gameweeks) -
    callers must ensure that ``prediction`` contains only fixtures for
    which predictions are available at the desired prediction date.

    Scoring formulas (see module docstring for full details):

    - Linear (exact): goals, assists, cards, own goals, penalty misses.
    - Probability-based (exact): appearance points, penalty saves/misses.
    - Approximate: clean sheet (independence assumption), goals conceded
      (floor quantisation), saves (floor quantisation), defensive
      contribution (threshold probability supplied by caller).

    Args:
        prediction: A validated PlayerGameweekPrediction.
        rules: Scoring rules to use. Defaults to DEFAULT_SCORING_RULES.

    Returns:
        An ExpectedPointsResult with a full component breakdown.

    Raises:
        PredictionError: If prediction is not a PlayerGameweekPrediction.
    """
    if not isinstance(prediction, PlayerGameweekPrediction):
        raise PredictionError(  # noqa: TRY003
            f"prediction must be a PlayerGameweekPrediction, got {type(prediction).__name__}"
        )

    r = rules if rules is not None else DEFAULT_SCORING_RULES
    position = prediction.position

    # Accumulators - full precision floats
    appearance_pts = 0.0
    goal_pts = 0.0
    assist_pts = 0.0
    clean_sheet_pts = 0.0
    goals_conceded_pts = 0.0
    save_pts = 0.0
    penalty_save_pts = 0.0
    penalty_miss_pts = 0.0
    yellow_card_pts = 0.0
    red_card_pts = 0.0
    own_goal_pts = 0.0
    defensive_contribution_pts = 0.0
    bonus_pts = 0.0

    # Position eligibility lookups (computed once, outside the fixture loop)
    cs_eligible = position in r.clean_sheet_points
    gc_eligible = position in r.goals_conceded_eligible
    save_eligible = position in r.save_eligible
    dc_eligible = position in r.defensive_contribution_eligible

    for mp in prediction.fixtures:
        # --- Appearance points ---
        # Three mutually exclusive outcomes:
        #   P(0 min)    = 1 - probability_appears
        #   P(1-59 min) = probability_appears - probability_60_plus
        #   P(60+ min)  = probability_60_plus
        appearance_pts += (
            mp.probability_appears - mp.probability_60_plus
        ) * r.appearance_points_below + mp.probability_60_plus * r.appearance_points_above_or_equal

        # --- Goal points (linear, exact) ---
        goal_pts += mp.expected_goals * r.goal_points[position]

        # --- Assist points (linear, exact) ---
        assist_pts += mp.expected_assists * r.assist_points

        # --- Clean sheet points (probability-based) ---
        # E[CS] = P(60+ min) * P(CS | 60+ min) * CS_points
        if cs_eligible:
            clean_sheet_pts += (
                mp.probability_60_plus * mp.clean_sheet_probability * r.clean_sheet_points[position]
            )

        # --- Goals conceded (GK/DEF only, approximation) ---
        # FPL: floor(gc / 2) * penalty. Linearised; exact only when
        # E[gc] is an even integer. Floor division NOT applied to
        # the expected value.
        if gc_eligible:
            goals_conceded_pts += (mp.expected_goals_conceded / 2.0) * r.goals_conceded_penalty

        # --- Save points (GK only, approximation) ---
        # FPL: floor(saves / saves_per_point) * save_points.
        # Linearised; exact only at multiples of saves_per_point.
        # Floor division NOT applied to the expected value.
        if save_eligible:
            save_pts += (mp.expected_saves / r.saves_per_point) * r.save_points

        # --- Penalty save points (GK only, exact) ---
        # At most one penalty per fixture, so expected value = probability.
        if position == PlayerPosition.GOALKEEPER:
            penalty_save_pts += mp.expected_penalties_saved * r.penalty_save_points

        # --- Penalty miss points (all positions, exact) ---
        penalty_miss_pts += mp.expected_penalties_missed * r.penalty_miss_points

        # --- Card deductions (linear, exact) ---
        yellow_card_pts += mp.expected_yellow_cards * r.yellow_card_points
        red_card_pts += mp.expected_red_cards * r.red_card_points

        # --- Own goal deduction (all positions, linear, exact) ---
        own_goal_pts += mp.expected_own_goals * r.own_goal_points

        # --- Defensive contribution (probability-based) ---
        # E[DC] = P(appear) * P(threshold met | appear) * DC_points
        if dc_eligible:
            defensive_contribution_pts += (
                mp.probability_appears
                * mp.defensive_contribution_probability
                * r.defensive_contribution_points
            )

        # --- Bonus points (already in FPL-point units, exact) ---
        bonus_pts += mp.expected_bonus

    # Total - sum of all components, maintained at full precision
    total = (
        appearance_pts
        + goal_pts
        + assist_pts
        + clean_sheet_pts
        + goals_conceded_pts
        + save_pts
        + penalty_save_pts
        + penalty_miss_pts
        + yellow_card_pts
        + red_card_pts
        + own_goal_pts
        + defensive_contribution_pts
        + bonus_pts
    )

    return ExpectedPointsResult(
        total_expected_points=total,
        expected_appearance_points=appearance_pts,
        expected_goal_points=goal_pts,
        expected_assist_points=assist_pts,
        expected_clean_sheet_points=clean_sheet_pts,
        expected_goals_conceded_points=goals_conceded_pts,
        expected_save_points=save_pts,
        expected_penalty_save_points=penalty_save_pts,
        expected_penalty_miss_points=penalty_miss_pts,
        expected_yellow_card_points=yellow_card_pts,
        expected_red_card_points=red_card_pts,
        expected_own_goal_points=own_goal_pts,
        expected_defensive_contribution_points=defensive_contribution_pts,
        expected_bonus_points=bonus_pts,
        fixture_count=len(prediction.fixtures),
        rules=r,
    )
