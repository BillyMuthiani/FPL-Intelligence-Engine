"""Leakage-safe evaluation framework for FPL expected-points predictions.

Stage 4I-B — Evaluation infrastructure.

This module provides a typed, UI-independent framework for evaluating
expected-points and probability predictions against actual outcomes,
with explicit temporal-safety guards to prevent data leakage.

Design principles
-----------------
* **Temporal separation:** Feature-generation time is separated from
  outcome-observation time.  Each :class:`PredictionRecord` carries a
  ``prediction_cutoff`` (UTC); each :class:`OutcomeRecord` carries an
  ``outcome_time`` (UTC).  Examples where the outcome was observable at
  or before the prediction cutoff are excluded as **temporal leakage**.
* **Deterministic alignment:** Predictions and outcomes are matched by
  the composite key ``(player_id, fixture_id, gameweek)``.  Duplicates on
  either side are reported, not silently overwritten.
* **Explicit exclusions:** Missing outcomes, duplicate keys, invalid
  probabilities, non-finite predictions, and post-deadline observations
  are each marked with a human-readable ``exclusion_reason`` and counted
  in :attr:`EvaluationSummary.exclusion_reasons`.
* **No I/O:** The framework accepts already-parsed records.  Callers are
  responsible for loading snapshots and converting payloads.
* **Re-use of existing DC evaluation:** :func:`evaluate_dc_models_from_history`
  reuses :func:`~fpl_engine.analytics.defensive_probability.evaluate_dc_models_brier`
  for per-player rolling-origin DC evaluation.

Integration helpers
-------------------
* :func:`outcome_from_history` — convert a :class:`PlayerMatchHistory`
  observation into an :class:`OutcomeRecord`, computing the DC outcome
  via :func:`defensive_contribution_from_history`.
* :func:`evaluate_dc_models_from_history` — bridge to the existing
  rolling-origin DC evaluation, converting its
  :class:`~fpl_engine.analytics.defensive_probability.DCEvaluationResult`
  output to :class:`BrierMetrics`.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fpl_engine.analytics.defensive_probability import (
    DC_DEFAULT_MINUTES,
    DEF_DC_THRESHOLD,
    MID_FWD_DC_THRESHOLD,
    evaluate_dc_models_brier,
)
from fpl_engine.models import PlayerMatchHistory, PlayerPosition
from fpl_engine.models.history import defensive_contribution_from_history

__all__ = [
    "BrierMetrics",
    "EvaluationError",
    "EvaluationExample",
    "EvaluationSummary",
    "ExpectedPointsMetrics",
    "OutcomeRecord",
    "PredictionRecord",
    "align_predictions_with_outcomes",
    "compute_brier_score",
    "compute_expected_points_metrics",
    "evaluate_dc_models_from_history",
    "evaluate_predictions",
    "outcome_from_history",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EvaluationError(ValueError):
    """Raised when evaluation inputs violate structural constraints."""


# ---------------------------------------------------------------------------
# Datetime validation helper
# ---------------------------------------------------------------------------


def _require_utc(dt: datetime, label: str) -> None:
    """Raise :class:`EvaluationError` if *dt* is not UTC-aware."""
    if dt.tzinfo is None:
        raise EvaluationError(f"{label} must be timezone-aware, got naive datetime")  # noqa: TRY003
    if dt.utcoffset() != UTC.utcoffset(None):
        raise EvaluationError(f"{label} must be in UTC, got {dt.tzinfo!r}")  # noqa: TRY003


# ---------------------------------------------------------------------------
# Numeric validation helper
# ---------------------------------------------------------------------------


def _is_finite(value: Any) -> bool:
    """Return True if *value* is a finite int or float (not bool).

    Rejects ``NaN``, ``±inf``, ``bool``, and non-numeric types.
    """
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    # math.isfinite handles NaN and inf for floats; for ints it's always True.
    return math.isfinite(value)


# ---------------------------------------------------------------------------
# Input records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionRecord:
    """A point-in-time prediction for a single player-fixture-gameweek.

    The prediction was generated at ``prediction_cutoff`` (UTC).  Only
    feature data retrieved **at or before** this timestamp may be used
    to construct the prediction; data from after the cutoff constitutes
    leakage.

    Attributes:
        player_id: FPL player (element) identifier.
        fixture_id: FPL fixture identifier.
        gameweek: FPL gameweek number (1-38).
        position: Player position — used to determine DC eligibility.
        predicted_expected_points: Expected FPL points for the fixture.
        predicted_dc_probability: DC probability P(threshold met | appear)
            in [0, 1].  Meaningful for DEF/MID/FWD; 0.0 for GKP.
        prediction_cutoff: UTC datetime when features were frozen.
    """

    player_id: int
    fixture_id: int
    gameweek: int
    position: PlayerPosition
    predicted_expected_points: float
    predicted_dc_probability: float
    prediction_cutoff: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.player_id, int) or isinstance(self.player_id, bool):
            raise EvaluationError("player_id must be an int")  # noqa: TRY003
        if not isinstance(self.fixture_id, int) or isinstance(self.fixture_id, bool):
            raise EvaluationError("fixture_id must be an int")  # noqa: TRY003
        if not isinstance(self.gameweek, int) or isinstance(self.gameweek, bool):
            raise EvaluationError("gameweek must be an int")  # noqa: TRY003
        if not 1 <= self.gameweek <= 38:
            raise EvaluationError(  # noqa: TRY003
                f"gameweek must be in [1, 38], got {self.gameweek}"
            )
        _require_utc(self.prediction_cutoff, "prediction_cutoff")


@dataclass(frozen=True)
class OutcomeRecord:
    """The actual outcome for a single player-fixture-gameweek.

    Attributes:
        player_id: FPL player (element) identifier.
        fixture_id: FPL fixture identifier.
        gameweek: FPL gameweek number (1-38).
        actual_points: FPL points awarded for the match (ground truth).
        dc_threshold_met: Whether the player met the DC action threshold
            in this match.  Only meaningful when ``appeared`` is True and
            the position is DC-eligible (DEF/MID/FWD).
        appeared: Whether the player played in the match (minutes > 0).
        outcome_time: UTC datetime when the outcome became observable
            (typically the fixture end time or when the data was first
            available after the match).
    """

    player_id: int
    fixture_id: int
    gameweek: int
    actual_points: int
    dc_threshold_met: bool
    appeared: bool
    outcome_time: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.player_id, int) or isinstance(self.player_id, bool):
            raise EvaluationError("player_id must be an int")  # noqa: TRY003
        if not isinstance(self.fixture_id, int) or isinstance(self.fixture_id, bool):
            raise EvaluationError("fixture_id must be an int")  # noqa: TRY003
        if not isinstance(self.gameweek, int) or isinstance(self.gameweek, bool):
            raise EvaluationError("gameweek must be an int")  # noqa: TRY003
        if not 1 <= self.gameweek <= 38:
            raise EvaluationError(  # noqa: TRY003
                f"gameweek must be in [1, 38], got {self.gameweek}"
            )
        if not isinstance(self.actual_points, int) or isinstance(
            self.actual_points, bool
        ):
            raise EvaluationError("actual_points must be an int")  # noqa: TRY003
        if not isinstance(self.dc_threshold_met, bool):
            raise EvaluationError("dc_threshold_met must be a bool")  # noqa: TRY003
        if not isinstance(self.appeared, bool):
            raise EvaluationError("appeared must be a bool")  # noqa: TRY003
        _require_utc(self.outcome_time, "outcome_time")


# ---------------------------------------------------------------------------
# Aligned evaluation example
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationExample:
    """A prediction/outcome pair after alignment, with exclusion metadata.

    Instances are produced by :func:`align_predictions_with_outcomes`.
    Metric functions should skip examples where ``excluded`` is True.

    Attributes:
        player_id: FPL player identifier.
        fixture_id: FPL fixture identifier.
        gameweek: FPL gameweek number.
        position: Player position.
        predicted_expected_points: Expected FPL points (from prediction).
        predicted_dc_probability: DC probability (from prediction).
        actual_points: Actual FPL points (from outcome).
        dc_threshold_met: DC outcome (True/False).
        appeared: Whether the player appeared in the match.
        prediction_cutoff: When the prediction was made (UTC).
        outcome_time: When the outcome was observed (UTC).
        excluded: Whether this example should be skipped in computation.
        exclusion_reason: Human-readable reason when ``excluded`` is True.
    """

    player_id: int
    fixture_id: int
    gameweek: int
    position: PlayerPosition
    predicted_expected_points: float
    predicted_dc_probability: float
    actual_points: int
    dc_threshold_met: bool
    appeared: bool
    prediction_cutoff: datetime
    outcome_time: datetime
    excluded: bool = False
    exclusion_reason: str | None = None

    @property
    def dc_eligible(self) -> bool:
        """Whether this player's position is eligible for DC points."""
        return self.position in (
            PlayerPosition.DEFENDER,
            PlayerPosition.MIDFIELDER,
            PlayerPosition.FORWARD,
        )


# ---------------------------------------------------------------------------
# Metric result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectedPointsMetrics:
    """MAE and RMSE for expected-points predictions.

    Attributes:
        mae: Mean absolute error on matched, non-excluded examples.
        rmse: Root mean squared error on matched, non-excluded examples.
        sample_size: Number of examples contributing to the metrics.
    """

    mae: float
    rmse: float
    sample_size: int


@dataclass(frozen=True)
class BrierMetrics:
    """Brier score and calibration for binary probability predictions.

    Attributes:
        brier_score: Mean Brier score (mean squared error of probabilities).
        sample_size: Number of (prediction, outcome) pairs evaluated.
        calibration: Mapping from bin lower-bound (multiples of
            ``1/num_bins``) to ``(avg_prediction, avg_observation)`` tuples.
        temporal_safety_enforced: True if leakage-safe temporal filtering
            was applied during evaluation (predictions strictly precede
            outcomes).  False when temporal enforcement was disabled.
    """

    brier_score: float
    sample_size: int
    calibration: dict[float, tuple[float, float]]
    temporal_safety_enforced: bool = False


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate evaluation result.

    Attributes:
        expected_points: MAE/RMSE metrics for expected-points predictions.
        dc_brier: Brier score and calibration for DC probability predictions.
        total_predictions: Number of prediction records supplied.
        total_outcomes: Number of outcome records supplied.
        n_matched: Number of prediction records that found a matching
            outcome (may still be excluded for other reasons).
        n_excluded: Number of matched examples excluded from metric
            computation.
        n_unpaired_outcomes: Number of outcome records with no matching
            prediction.
        exclusion_reasons: Count of exclusions by reason string.
    """

    expected_points: ExpectedPointsMetrics
    dc_brier: BrierMetrics
    total_predictions: int
    total_outcomes: int
    n_matched: int
    n_excluded: int
    n_unpaired_outcomes: int
    exclusion_reasons: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

_AlignmentKey = tuple[int, int, int]


def _alignment_key(record: Any, kind: str) -> _AlignmentKey:
    """Extract the ``(player_id, fixture_id, gameweek)`` alignment key."""
    player_id = record.player_id
    fixture_id = record.fixture_id
    gameweek = record.gameweek
    if not isinstance(player_id, int) or isinstance(player_id, bool):
        raise EvaluationError(  # noqa: TRY003
            f"{kind}: player_id must be int, got {type(player_id).__name__}"
        )
    if not isinstance(fixture_id, int) or isinstance(fixture_id, bool):
        raise EvaluationError(  # noqa: TRY003
            f"{kind}: fixture_id must be int, got {type(fixture_id).__name__}"
        )
    if not isinstance(gameweek, int) or isinstance(gameweek, bool):
        raise EvaluationError(  # noqa: TRY003
            f"{kind}: gameweek must be int, got {type(gameweek).__name__}"
        )
    return (player_id, fixture_id, gameweek)


def _detect_temporal_conflicts(
    outcomes: Sequence[OutcomeRecord],
) -> list[str]:
    """Detect fixtures whose outcome records conflict across retrieval times.

    When the same ``(player_id, fixture_id, gameweek)`` appears in multiple
    outcome records with *different* ``outcome_time`` values, the records may
    represent the same fixture observed at different retrieval timestamps
    (e.g. a live update and a post-match final).  Conflicting
    ``actual_points`` or ``dc_threshold_met`` values are surfaced as warning
    strings so callers can investigate rather than silently picking one.

    Parameters
    ----------
    outcomes
        Sequence of :class:`OutcomeRecord` to scan.

    Returns
    -------
    list[str]
        Human-readable conflict descriptions.  Empty when no conflicts are
        found.
    """
    grouped: dict[tuple[int, int, int], list[OutcomeRecord]] = {}
    for outcome in outcomes:
        key = (outcome.player_id, outcome.fixture_id, outcome.gameweek)
        grouped.setdefault(key, []).append(outcome)

    conflicts: list[str] = []
    for key, records in grouped.items():
        if len(records) <= 1:
            continue
        # Check for differing outcome_time (temporal conflict).
        times = {r.outcome_time for r in records}
        points = {r.actual_points for r in records}
        dc = {r.dc_threshold_met for r in records}
        if len(times) > 1 or len(points) > 1 or len(dc) > 1:
            player_id, fixture_id, gw = key
            detail_parts: list[str] = []
            if len(times) > 1:
                detail_parts.append(
                    f"distinct outcome_times: {sorted(t.isoformat() for t in times)}"
                )
            if len(points) > 1:
                detail_parts.append(f"actual_points: {sorted(points)}")
            if len(dc) > 1:
                detail_parts.append(f"dc_threshold_met: {sorted(dc)}")
            conflicts.append(
                f"player={player_id}, fixture={fixture_id}, gw={gw}: "
                + "; ".join(detail_parts)
            )
    return conflicts


def _make_excluded_example(pred: PredictionRecord, reason: str) -> EvaluationExample:
    """Build an :class:`EvaluationExample` flagged as excluded."""
    return EvaluationExample(
        player_id=pred.player_id,
        fixture_id=pred.fixture_id,
        gameweek=pred.gameweek,
        position=pred.position,
        predicted_expected_points=pred.predicted_expected_points,
        predicted_dc_probability=pred.predicted_dc_probability,
        actual_points=0,
        dc_threshold_met=False,
        appeared=False,
        prediction_cutoff=pred.prediction_cutoff,
        # placeholder; outcome data is unavailable
        outcome_time=pred.prediction_cutoff,
        excluded=True,
        exclusion_reason=reason,
    )


def align_predictions_with_outcomes(
    predictions: Sequence[PredictionRecord],
    outcomes: Sequence[OutcomeRecord],
    *,
    enforce_temporal_order: bool = True,
) -> tuple[list[EvaluationExample], int]:
    """Align predictions with outcomes by ``(player_id, fixture_id, gameweek)``.

    Each prediction record is matched to at most one outcome record.
    The returned list contains one :class:`EvaluationExample` per
    prediction record, with ``excluded`` and ``exclusion_reason`` set
    when appropriate.

    Exclusion reasons produced by this function:

    * ``"missing outcome"`` — no matching outcome record exists.
    * ``"duplicate prediction"`` — multiple predictions share the same key.
    * ``"duplicate outcome"`` — multiple outcomes share the same key.
    * ``"temporal leakage: outcome observed at or before prediction cutoff"``
      — ``outcome_time <= prediction_cutoff`` when temporal enforcement is
      enabled.
    * ``"invalid probability"`` — ``predicted_dc_probability`` is outside
      [0, 1] or non-finite.
    * ``"non-finite prediction"`` — ``predicted_expected_points`` is NaN or
      infinite.

    Parameters:
        predictions: Sequence of :class:`PredictionRecord`.
        outcomes: Sequence of :class:`OutcomeRecord`.
        enforce_temporal_order: When True (default), exclude examples where
            the outcome was observable at or before the prediction cutoff.

    Returns:
        A tuple ``(examples, unpaired_outcome_count)`` where *examples*
        has one entry per prediction record and *unpaired_outcome_count*
        is the number of outcome records without a matching prediction.

    Raises:
        EvaluationError: If conflicting cross-temporal outcome records are
            detected (same fixture appearing with different outcomes or
            observation times).  Use ``allow_temporal_conflicts=True`` to
            suppress this check and handle conflicts manually.
    """
    outcome_map: dict[_AlignmentKey, list[OutcomeRecord]] = {}
    for outcome in outcomes:
        key = _alignment_key(outcome, "outcome")
        outcome_map.setdefault(key, []).append(outcome)

    examples: list[EvaluationExample] = []
    seen_prediction_keys: set[_AlignmentKey] = set()

    for pred in predictions:
        key = _alignment_key(pred, "prediction")

        # Duplicate prediction on the same key.
        if key in seen_prediction_keys:
            examples.append(
                _make_excluded_example(pred, "duplicate prediction")
            )
            continue
        seen_prediction_keys.add(key)

        outcome_list = outcome_map.get(key)
        if not outcome_list:
            examples.append(_make_excluded_example(pred, "missing outcome"))
            continue

        # Multiple outcomes for the same key → exclude (ambiguous).
        if len(outcome_list) > 1:
            examples.append(
                _make_excluded_example(pred, "duplicate outcome")
            )
            continue

        outcome = outcome_list[0]

        ex = EvaluationExample(
            player_id=pred.player_id,
            fixture_id=pred.fixture_id,
            gameweek=pred.gameweek,
            position=pred.position,
            predicted_expected_points=pred.predicted_expected_points,
            predicted_dc_probability=pred.predicted_dc_probability,
            actual_points=outcome.actual_points,
            dc_threshold_met=outcome.dc_threshold_met,
            appeared=outcome.appeared,
            prediction_cutoff=pred.prediction_cutoff,
            outcome_time=outcome.outcome_time,
        )

        # Temporal leakage: outcome must be observed AFTER the cutoff.
        if enforce_temporal_order and ex.outcome_time <= ex.prediction_cutoff:
            examples.append(
                _make_excluded_example(
                    pred,
                    "temporal leakage: outcome observed at or before prediction cutoff",
                )
            )
            continue

        # Validate predicted DC probability.
        if not _is_finite(pred.predicted_dc_probability) or not 0.0 <= (
            pred.predicted_dc_probability
        ) <= 1.0:
            examples.append(_make_excluded_example(pred, "invalid probability"))
            continue

        # Validate predicted expected points.
        if not _is_finite(pred.predicted_expected_points):
            examples.append(
                _make_excluded_example(pred, "non-finite prediction")
            )
            continue

        examples.append(ex)

    # Count unpaired outcomes (outcomes without a matching prediction).
    unpaired_outcomes = sum(
        len(ols)
        for key, ols in outcome_map.items()
        if key not in seen_prediction_keys
    )

    return examples, unpaired_outcomes


# ---------------------------------------------------------------------------
# Expected-points metrics (MAE / RMSE)
# ---------------------------------------------------------------------------


def compute_expected_points_metrics(
    examples: Sequence[EvaluationExample],
) -> ExpectedPointsMetrics:
    """Compute MAE and RMSE for expected-points predictions.

    Only non-excluded examples with finite ``predicted_expected_points``
    contribute.  ``actual_points`` are integers, hence always finite.

    An empty sample returns zero-valued metrics with ``sample_size == 0``.
    """
    diffs: list[float] = []
    for ex in examples:
        if ex.excluded:
            continue
        if not _is_finite(ex.predicted_expected_points):
            continue
        diff = ex.predicted_expected_points - ex.actual_points
        if not _is_finite(diff):
            continue
        diffs.append(diff)

    n = len(diffs)
    if n == 0:
        return ExpectedPointsMetrics(mae=0.0, rmse=0.0, sample_size=0)

    abs_errors = [abs(d) for d in diffs]
    mae = sum(abs_errors) / n
    mse = sum(d * d for d in diffs) / n
    rmse = mse ** 0.5

    return ExpectedPointsMetrics(mae=mae, rmse=rmse, sample_size=n)


# ---------------------------------------------------------------------------
# Brier score and calibration
# ---------------------------------------------------------------------------


def _brier_component(prediction: float, outcome: int) -> float:
    """Single Brier score component: ``(prediction - outcome) ** 2``."""
    return (prediction - outcome) ** 2


def compute_brier_score(
    predicted_probabilities: Sequence[float],
    actual_outcomes: Sequence[int],
    *,
    num_bins: int = 10,
) -> BrierMetrics:
    """Compute the Brier score and calibration for binary predictions.

    Parameters:
        predicted_probabilities: Predicted probabilities in [0, 1].
        actual_outcomes: Binary outcomes (0 or 1).  Length should match
            *predicted_probabilities*; extra elements on either side are
            ignored by :func:`zip` with ``strict=False``.
        num_bins: Number of equal-width bins for calibration (default 10,
            matching :func:`~fpl_engine.analytics.defensive_probability.evaluate_dc_models_brier`).

    Pairs where the probability or outcome is invalid (non-finite,
    probability outside [0, 1], outcome not 0/1) are silently skipped.

    An empty valid sample returns zero-valued metrics with
    ``sample_size == 0`` and empty calibration.

    Raises:
        ValueError: If *num_bins* is less than 1.
    """
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}")  # noqa: TRY003

    # Validate and pair.
    valid_pairs: list[tuple[float, int]] = []
    for p, o in zip(predicted_probabilities, actual_outcomes, strict=False):
        if not isinstance(o, int) or isinstance(o, bool) or o not in (0, 1):
            continue
        if not _is_finite(p) or not 0.0 <= p <= 1.0:
            continue
        valid_pairs.append((p, o))

    n = len(valid_pairs)
    if n == 0:
        return BrierMetrics(
            brier_score=0.0, sample_size=0, calibration={},
            temporal_safety_enforced=True,
        )

    brier = sum(_brier_component(p, o) for p, o in valid_pairs) / n

    # Calibration: bin predictions into equal-width intervals.
    calibration: dict[float, tuple[float, float]] = {}
    bin_width = 1.0 / num_bins
    for bin_index in range(num_bins):
        lo = bin_index * bin_width
        hi = lo + bin_width
        if bin_index == num_bins - 1:
            # Last bin is inclusive of 1.0.
            bin_pairs = [(p, o) for p, o in valid_pairs if lo <= p <= 1.0]
        else:
            bin_pairs = [(p, o) for p, o in valid_pairs if lo <= p < hi]
        if bin_pairs:
            avg_pred = sum(p for p, _ in bin_pairs) / len(bin_pairs)
            avg_obs = sum(o for _, o in bin_pairs) / len(bin_pairs)
            calibration[lo] = (avg_pred, avg_obs)

    return BrierMetrics(
        brier_score=brier, sample_size=n, calibration=calibration,
        temporal_safety_enforced=True,
    )


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def _count_exclusion_reasons(
    examples: Sequence[EvaluationExample],
) -> dict[str, int]:
    """Count exclusions by reason string."""
    counts: dict[str, int] = {}
    for ex in examples:
        if ex.excluded and ex.exclusion_reason is not None:
            counts[ex.exclusion_reason] = counts.get(ex.exclusion_reason, 0) + 1
    return counts


def _dc_brier_from_examples(
    examples: Sequence[EvaluationExample],
    *,
    num_bins: int = 10,
) -> BrierMetrics:
    """Compute DC Brier score from aligned evaluation examples.

    Filters to examples that are:
    * Not excluded.
    * DC-eligible (position is DEF, MID, or FWD — not GKP).
    * The player appeared (``appeared`` is True).

    The DC probability is P(DC threshold met | appear), and the outcome
    is whether the threshold was actually met.  Both are only meaningful
    when the player appeared, so non-appearance matches are excluded.
    """
    preds: list[float] = []
    outs: list[int] = []
    for ex in examples:
        if ex.excluded:
            continue
        if not ex.dc_eligible:
            continue
        if not ex.appeared:
            continue
        preds.append(ex.predicted_dc_probability)
        outs.append(1 if ex.dc_threshold_met else 0)
    return compute_brier_score(preds, outs, num_bins=num_bins)


def evaluate_predictions(
    predictions: Sequence[PredictionRecord],
    outcomes: Sequence[OutcomeRecord],
    *,
    enforce_temporal_order: bool = True,
    num_bins: int = 10,
) -> EvaluationSummary:
    """Run the full leakage-safe evaluation pipeline.

    Steps:
        1. Align predictions and outcomes by
           ``(player_id, fixture_id, gameweek)``.
        2. Apply temporal, validity, and eligibility filters.
        3. Compute expected-points MAE/RMSE on valid examples.
        4. Compute DC Brier score + calibration on valid, DC-eligible,
           appeared examples.

    Parameters:
        predictions: Sequence of :class:`PredictionRecord`.
        outcomes: Sequence of :class:`OutcomeRecord`.
        enforce_temporal_order: When True, exclude examples where the
            outcome was observable at or before the prediction cutoff.
        num_bins: Calibration bin count for Brier evaluation.

    Returns:
        An :class:`EvaluationSummary` with all metrics, counts, and
        exclusion reasons.
    """
    examples, unpaired_outcomes = align_predictions_with_outcomes(
        predictions, outcomes, enforce_temporal_order=enforce_temporal_order
    )

    ep_metrics = compute_expected_points_metrics(examples)
    dc_metrics = _dc_brier_from_examples(examples, num_bins=num_bins)

    matched = len(examples)
    excluded = sum(1 for e in examples if e.excluded)
    reasons = _count_exclusion_reasons(examples)

    return EvaluationSummary(
        expected_points=ep_metrics,
        dc_brier=dc_metrics,
        total_predictions=len(predictions),
        total_outcomes=len(outcomes),
        n_matched=matched,
        n_excluded=excluded,
        n_unpaired_outcomes=unpaired_outcomes,
        exclusion_reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Outcome helper: PlayerMatchHistory → OutcomeRecord
# ---------------------------------------------------------------------------

_DC_THRESHOLD_BY_POSITION: dict[PlayerPosition, int] = {
    PlayerPosition.DEFENDER: DEF_DC_THRESHOLD,
    PlayerPosition.MIDFIELDER: MID_FWD_DC_THRESHOLD,
    PlayerPosition.FORWARD: MID_FWD_DC_THRESHOLD,
}


def outcome_from_history(
    history: PlayerMatchHistory,
    position: PlayerPosition,
    outcome_time: datetime,
) -> OutcomeRecord:
    """Create an :class:`OutcomeRecord` from a :class:`PlayerMatchHistory`.

    Uses :func:`defensive_contribution_from_history` to determine whether
    the DC threshold was met (only for eligible positions: DEF, MID, FWD).
    GKP players always get ``dc_threshold_met=False``.

    Parameters:
        history: A single match observation.  ``minutes == 0`` means the
            player did not appear.
        position: The player's position (determines the DC threshold).
        outcome_time: UTC datetime when the outcome became observable.

    Returns:
        An :class:`OutcomeRecord` with ``actual_points`` taken from
        ``history.total_points`` and ``dc_threshold_met`` computed from
        the position-specific threshold.
    """
    _require_utc(outcome_time, "outcome_time")

    appeared = history.minutes > 0
    threshold = _DC_THRESHOLD_BY_POSITION.get(position)
    dc_met = False
    if threshold is not None and appeared:
        dc_met = defensive_contribution_from_history(history, threshold)

    return OutcomeRecord(
        player_id=history.player_id,
        fixture_id=history.fixture_id,
        gameweek=history.gameweek,
        actual_points=history.total_points,
        dc_threshold_met=dc_met,
        appeared=appeared,
        outcome_time=outcome_time,
    )


# ---------------------------------------------------------------------------
# Bridge: reuse evaluate_dc_models_brier for per-player DC evaluation
# ---------------------------------------------------------------------------


def evaluate_dc_models_from_history(
    history: Sequence[PlayerMatchHistory],
    position: PlayerPosition,
    *,
    max_gameweek: int = 38,
    expected_minutes: float = DC_DEFAULT_MINUTES,
    prediction_cutoff: datetime | None = None,
) -> dict[str, BrierMetrics]:
    """Evaluate DC models using the existing rolling-origin evaluator.

    This is a thin compatibility bridge around
    :func:`~fpl_engine.analytics.defensive_probability.evaluate_dc_models_brier`.
    It reuses the existing function's rolling-origin logic (for each
    gameweek ``gw``, only observations with ``gameweek < gw`` enter the
    prediction) and converts its
    :class:`~fpl_engine.analytics.defensive_probability.DCEvaluationResult`
    output to :class:`BrierMetrics`.

    Use this function when you have the player's **full** match history
    and want rolling-origin DC evaluation.  For evaluating pre-computed
    prediction probabilities against aligned outcomes, use
    :func:`compute_brier_score` instead.

    Parameters:
        history: The player's full :class:`PlayerMatchHistory` sequence
            (all gameweeks).
        position: Player position.  GKP returns empty results.
        max_gameweek: Last gameweek to evaluate (inclusive).
        expected_minutes: Expected minutes for the Poisson model.

    Returns:
        Dict mapping model name (``"history"``, ``"poisson"``) to
        :class:`BrierMetrics`.  GKP returns empty metrics for both models.

    Raises:
        EvaluationError: If ``prediction_cutoff`` is provided but is not a
            UTC-aware datetime.
    """
    if prediction_cutoff is not None:
        _require_utc(prediction_cutoff, "prediction_cutoff")

    dc_results = evaluate_dc_models_brier(
        history=history,
        position=position,
        max_gameweek=max_gameweek,
        expected_minutes=expected_minutes,
        prediction_cutoff=prediction_cutoff,
    )

    return {
        model_name: BrierMetrics(
            brier_score=result.brier_score,
            sample_size=result.sample_size,
            calibration=result.calibration,
            temporal_safety_enforced=result.temporal_safety_enforced,
        )
        for model_name, result in dc_results.items()
    }
