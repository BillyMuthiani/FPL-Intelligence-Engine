"""Tests for the leakage-safe evaluation framework (Stage 4I-B).

Covers:
* ``PredictionRecord`` / ``OutcomeRecord`` dataclass validation.
* ``EvaluationExample`` and ``dc_eligible`` property.
* ``align_predictions_with_outcomes`` — temporal, duplicate, and validity guards.
* ``compute_expected_points_metrics`` — MAE/RMSE with exclusion handling.
* ``compute_brier_score`` — Brier score, calibration bins, invalid-skip semantics.
* ``evaluate_predictions`` — end-to-end orchestration.
* ``outcome_from_history`` — conversion from ``PlayerMatchHistory`` to ``OutcomeRecord``.
* ``evaluate_dc_models_from_history`` — bridge to ``evaluate_dc_models_brier``.
* ``prediction_cutoff`` parameter and ``temporal_safety_enforced`` flag.
* Cross-temporal conflict detection via ``_detect_temporal_conflicts``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from fpl_engine.analytics.evaluation import (
    BrierMetrics,
    EvaluationError,
    EvaluationExample,
    EvaluationSummary,
    OutcomeRecord,
    PredictionRecord,
    _detect_temporal_conflicts,
    align_predictions_with_outcomes,
    compute_brier_score,
    compute_expected_points_metrics,
    evaluate_dc_models_from_history,
    evaluate_predictions,
    outcome_from_history,
)
from fpl_engine.models import (
    PlayerMatchHistory,
    PlayerPosition,
)

# ---------------------------------------------------------------------------
# PredictionRecord validation
# ---------------------------------------------------------------------------


class TestPredictionRecord:
    """Unit tests for ``PredictionRecord`` dataclass."""

    def test_valid_construction(self) -> None:
        pr = _pred()
        assert pr.player_id == 1
        assert pr.fixture_id == 10
        assert pr.gameweek == 1
        assert pr.position == PlayerPosition.MIDFIELDER
        assert pr.predicted_expected_points == 5.0
        assert pr.predicted_dc_probability == 0.3
        assert pr.prediction_cutoff == _utc(2024, 8, 10, 12)

    def test_frozen(self) -> None:
        pr = _pred()
        with pytest.raises(AttributeError):
            pr.player_id = 99  # type: ignore[misc]

    def test_naive_cutoff_raises(self) -> None:
        with pytest.raises(EvaluationError, match="timezone-aware"):
            PredictionRecord(
                player_id=1,
                fixture_id=1,
                gameweek=1,
                position=PlayerPosition.MIDFIELDER,
                predicted_expected_points=1.0,
                predicted_dc_probability=0.5,
                prediction_cutoff=_naive_now(),
            )

    def test_non_utc_timezone_raises(self) -> None:
        tz = timezone(timedelta(hours=5))
        with pytest.raises(EvaluationError, match="must be in UTC"):
            PredictionRecord(
                player_id=1,
                fixture_id=1,
                gameweek=1,
                position=PlayerPosition.MIDFIELDER,
                predicted_expected_points=1.0,
                predicted_dc_probability=0.5,
                prediction_cutoff=datetime(2024, 8, 10, 12, tzinfo=tz),
            )

    def test_player_id_must_be_int(self) -> None:
        with pytest.raises(EvaluationError, match="player_id must be an int"):
            PredictionRecord(
                player_id="1",  # type: ignore[arg-type]
                fixture_id=1,
                gameweek=1,
                position=PlayerPosition.MIDFIELDER,
                predicted_expected_points=1.0,
                predicted_dc_probability=0.5,
                prediction_cutoff=_utc(),
            )

    def test_player_id_bool_rejected(self) -> None:
        with pytest.raises(EvaluationError, match="player_id must be an int"):
            PredictionRecord(
                player_id=True,  # type: ignore[arg-type]
                fixture_id=1,
                gameweek=1,
                position=PlayerPosition.MIDFIELDER,
                predicted_expected_points=1.0,
                predicted_dc_probability=0.5,
                prediction_cutoff=_utc(),
            )

    def test_fixture_id_must_be_int(self) -> None:
        with pytest.raises(EvaluationError, match="fixture_id must be an int"):
            PredictionRecord(
                player_id=1,
                fixture_id=1.0,  # type: ignore[arg-type]
                gameweek=1,
                position=PlayerPosition.MIDFIELDER,
                predicted_expected_points=1.0,
                predicted_dc_probability=0.5,
                prediction_cutoff=_utc(),
            )

    def test_gameweek_out_of_range_low(self) -> None:
        with pytest.raises(EvaluationError, match="gameweek must be in"):
            PredictionRecord(
                player_id=1,
                fixture_id=1,
                gameweek=0,
                position=PlayerPosition.MIDFIELDER,
                predicted_expected_points=1.0,
                predicted_dc_probability=0.5,
                prediction_cutoff=_utc(),
            )

    def test_gameweek_out_of_range_high(self) -> None:
        with pytest.raises(EvaluationError, match="gameweek must be in"):
            PredictionRecord(
                player_id=1,
                fixture_id=1,
                gameweek=39,
                position=PlayerPosition.MIDFIELDER,
                predicted_expected_points=1.0,
                predicted_dc_probability=0.5,
                prediction_cutoff=_utc(),
            )


# ---------------------------------------------------------------------------
# OutcomeRecord validation
# ---------------------------------------------------------------------------


class TestOutcomeRecord:
    """Unit tests for ``OutcomeRecord`` dataclass."""

    def test_valid_construction(self) -> None:
        oc = _outcome()
        assert oc.player_id == 1
        assert oc.actual_points == 3
        assert oc.dc_threshold_met is True
        assert oc.appeared is True
        assert oc.outcome_time == _utc(2024, 8, 17, 14)

    def test_frozen(self) -> None:
        oc = _outcome()
        with pytest.raises(AttributeError):
            oc.actual_points = 0  # type: ignore[misc]

    def test_naive_outcome_time_raises(self) -> None:
        with pytest.raises(EvaluationError, match="timezone-aware"):
            OutcomeRecord(
                player_id=1,
                fixture_id=1,
                gameweek=1,
                actual_points=0,
                dc_threshold_met=False,
                appeared=False,
                outcome_time=_naive_now(),
            )

    def test_non_utc_outcome_time_raises(self) -> None:
        tz = timezone(timedelta(hours=-3))
        with pytest.raises(EvaluationError, match="must be in UTC"):
            OutcomeRecord(
                player_id=1,
                fixture_id=1,
                gameweek=1,
                actual_points=0,
                dc_threshold_met=False,
                appeared=False,
                outcome_time=datetime(2024, 8, 17, 14, tzinfo=tz),
            )

    def test_actual_points_must_be_int(self) -> None:
        with pytest.raises(EvaluationError, match="actual_points must be an int"):
            OutcomeRecord(
                player_id=1,
                fixture_id=1,
                gameweek=1,
                actual_points=3.5,  # type: ignore[arg-type]
                dc_threshold_met=False,
                appeared=False,
                outcome_time=_utc(),
            )

    def test_actual_points_bool_rejected(self) -> None:
        with pytest.raises(EvaluationError, match="actual_points must be an int"):
            OutcomeRecord(
                player_id=1,
                fixture_id=1,
                gameweek=1,
                actual_points=True,  # type: ignore[arg-type]
                dc_threshold_met=False,
                appeared=False,
                outcome_time=_utc(),
            )

    def test_dc_threshold_met_must_be_bool(self) -> None:
        with pytest.raises(EvaluationError, match="dc_threshold_met must be a bool"):
            OutcomeRecord(
                player_id=1,
                fixture_id=1,
                gameweek=1,
                actual_points=0,
                dc_threshold_met=1,  # type: ignore[arg-type]
                appeared=True,
                outcome_time=_utc(),
            )

    def test_appeared_must_be_bool(self) -> None:
        with pytest.raises(EvaluationError, match="appeared must be a bool"):
            OutcomeRecord(
                player_id=1,
                fixture_id=1,
                gameweek=1,
                actual_points=0,
                dc_threshold_met=False,
                appeared="yes",  # type: ignore[arg-type]
                outcome_time=_utc(),
            )

    def test_gameweek_too_high(self) -> None:
        with pytest.raises(EvaluationError, match="gameweek must be in"):
            OutcomeRecord(
                player_id=1,
                fixture_id=1,
                gameweek=99,
                actual_points=0,
                dc_threshold_met=False,
                appeared=False,
                outcome_time=_utc(),
            )



def _utc(year: int = 2024, month: int = 8, day: int = 17, hour: int = 14,
    minute: int = 0, second: int = 0) -> datetime:
    """Return a timezone-aware UTC datetime."""
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


def _naive_now() -> datetime:
    """Return a naive (tz-naive) datetime — should always raise."""
    return datetime(2024, 8, 17, 14, 0, 0)  # noqa: DTZ001


def _pred(
    player_id: int = 1,
    fixture_id: int = 10,
    gameweek: int = 1,
    position: PlayerPosition = PlayerPosition.MIDFIELDER,
    expected_points: float = 5.0,
    dc_probability: float = 0.3,
    cutoff: datetime | None = None,
) -> PredictionRecord:
    """Create a valid ``PredictionRecord`` with sensible defaults."""
    if cutoff is None:
        cutoff = _utc(2024, 8, 10, 12)
    return PredictionRecord(
        player_id=player_id,
        fixture_id=fixture_id,
        gameweek=gameweek,
        position=position,
        predicted_expected_points=expected_points,
        predicted_dc_probability=dc_probability,
        prediction_cutoff=cutoff,
    )


def _outcome(
    player_id: int = 1,
    fixture_id: int = 10,
    gameweek: int = 1,
    actual_points: int = 3,
    dc_met: bool = True,
    appeared: bool = True,
    outcome_time: datetime | None = None,
) -> OutcomeRecord:
    """Create a valid ``OutcomeRecord`` with sensible defaults."""
    if outcome_time is None:
        outcome_time = _utc(2024, 8, 17, 14)
    return OutcomeRecord(
        player_id=player_id,
        fixture_id=fixture_id,
        gameweek=gameweek,
        actual_points=actual_points,
        dc_threshold_met=dc_met,
        appeared=appeared,
        outcome_time=outcome_time,
    )


def _match_history(
    player_id: int = 1,
    fixture_id: int = 10,
    gameweek: int = 1,
    minutes: int = 90,
    total_points: int = 3,
    tackles: int = 0,
    cbi: int = 0,
    recoveries: int = 0,
    clean_sheet: bool = True,
) -> PlayerMatchHistory:
    """Create a ``PlayerMatchHistory`` with controllable defensive stats."""
    return PlayerMatchHistory(
        player_id=player_id,
        fixture_id=fixture_id,
        gameweek=gameweek,
        minutes=minutes,
        goals_conceded=0,
        clean_sheet=clean_sheet,
        saves=0,
        goals_scored=0,
        assists=0,
        tackles=tackles,
        clearances_blocks_interceptions=cbi,
        recoveries=recoveries,
        yellow_cards=0,
        red_cards=0,
        own_goals=0,
        penalties_saved=0,
        penalties_missed=0,
        bonus=0,
        bps=0,
        total_points=total_points,
    )

# ---------------------------------------------------------------------------
# EvaluationExample
# ---------------------------------------------------------------------------


class TestEvaluationExample:
    """Unit tests for ``EvaluationExample`` and ``dc_eligible`` property."""

    def test_dc_eligible_midfielder(self) -> None:
        ex = EvaluationExample(
            player_id=1, fixture_id=1, gameweek=1,
            position=PlayerPosition.MIDFIELDER,
            predicted_expected_points=5.0,
            predicted_dc_probability=0.3,
            actual_points=3, dc_threshold_met=True, appeared=True,
            prediction_cutoff=_utc(), outcome_time=_utc(),
        )
        assert ex.dc_eligible is True

    def test_dc_eligible_defender(self) -> None:
        ex = EvaluationExample(
            player_id=1, fixture_id=1, gameweek=1,
            position=PlayerPosition.DEFENDER,
            predicted_expected_points=2.0,
            predicted_dc_probability=0.4,
            actual_points=1, dc_threshold_met=True, appeared=True,
            prediction_cutoff=_utc(), outcome_time=_utc(),
        )
        assert ex.dc_eligible is True

    def test_dc_eligible_forward(self) -> None:
        ex = EvaluationExample(
            player_id=1, fixture_id=1, gameweek=1,
            position=PlayerPosition.FORWARD,
            predicted_expected_points=4.0,
            predicted_dc_probability=0.2,
            actual_points=0, dc_threshold_met=False, appeared=True,
            prediction_cutoff=_utc(), outcome_time=_utc(),
        )
        assert ex.dc_eligible is True

    def test_dc_eligible_goalkeeper(self) -> None:
        ex = EvaluationExample(
            player_id=1, fixture_id=1, gameweek=1,
            position=PlayerPosition.GOALKEEPER,
            predicted_expected_points=2.0,
            predicted_dc_probability=0.0,
            actual_points=0, dc_threshold_met=False, appeared=True,
            prediction_cutoff=_utc(), outcome_time=_utc(),
        )
        assert ex.dc_eligible is False

    def test_default_excluded_is_false(self) -> None:
        ex = EvaluationExample(
            player_id=1, fixture_id=1, gameweek=1,
            position=PlayerPosition.MIDFIELDER,
            predicted_expected_points=5.0,
            predicted_dc_probability=0.3,
            actual_points=3, dc_threshold_met=True, appeared=True,
            prediction_cutoff=_utc(), outcome_time=_utc(),
        )
        assert ex.excluded is False
        assert ex.exclusion_reason is None

# ---------------------------------------------------------------------------
# align_predictions_with_outcomes
# ---------------------------------------------------------------------------


class TestAlignPredictionsWithOutcomes:
    """Tests for the alignment function and its exclusion logic."""

    def test_successful_alignment(self) -> None:
        pred = _pred(cutoff=_utc(2024, 8, 10, 12))
        outcome = _outcome(outcome_time=_utc(2024, 8, 17, 14))
        examples, unpaired = align_predictions_with_outcomes([pred], [outcome])
        assert len(examples) == 1
        assert examples[0].excluded is False
        assert unpaired == 0
        assert examples[0].actual_points == 3
        assert examples[0].dc_threshold_met is True

    def test_missing_outcome_excluded(self) -> None:
        pred = _pred()
        examples, _unpaired = align_predictions_with_outcomes([pred], [])
        assert len(examples) == 1
        assert examples[0].excluded is True
        assert examples[0].exclusion_reason == "missing outcome"

    def test_duplicate_prediction_excluded(self) -> None:
        pred1 = _pred()
        pred2 = _pred()  # same key
        outcome = _outcome()
        examples, _unpaired = align_predictions_with_outcomes([pred1, pred2], [outcome])
        assert len(examples) == 2
        assert examples[0].excluded is False
        assert examples[1].excluded is True
        assert examples[1].exclusion_reason == "duplicate prediction"

    def test_duplicate_outcome_excluded(self) -> None:
        pred = _pred()
        oc1 = _outcome()
        oc2 = _outcome()  # same key as oc1
        examples, _unpaired = align_predictions_with_outcomes([pred], [oc1, oc2])
        assert len(examples) == 1
        assert examples[0].excluded is True
        assert examples[0].exclusion_reason == "duplicate outcome"

    def test_temporal_leakage_excluded_when_enforced(self) -> None:
        pred = _pred(cutoff=_utc(2024, 8, 20, 12))
        outcome = _outcome(outcome_time=_utc(2024, 8, 17, 14))
        examples, _unpaired = align_predictions_with_outcomes(
            [pred], [outcome], enforce_temporal_order=True,
        )
        assert examples[0].excluded is True
        assert "temporal leakage" in (examples[0].exclusion_reason or "")

    def test_temporal_leakage_allowed_when_not_enforced(self) -> None:
        pred = _pred(cutoff=_utc(2024, 8, 20, 12))
        outcome = _outcome(outcome_time=_utc(2024, 8, 17, 14))
        examples, _unpaired = align_predictions_with_outcomes(
            [pred], [outcome], enforce_temporal_order=False,
        )
        assert examples[0].excluded is False

    def test_equal_times_excluded(self) -> None:
        t = _utc(2024, 8, 17, 14)
        pred = _pred(cutoff=t)
        outcome = _outcome(outcome_time=t)
        examples, _ = align_predictions_with_outcomes([pred], [outcome])
        assert examples[0].excluded is True
        assert "temporal leakage" in (examples[0].exclusion_reason or "")

    def test_invalid_probability_excluded(self) -> None:
        pred = _pred(dc_probability=1.5)
        outcome = _outcome()
        examples, _ = align_predictions_with_outcomes([pred], [outcome])
        assert examples[0].excluded is True
        assert examples[0].exclusion_reason == "invalid probability"

    def test_nan_probability_excluded(self) -> None:
        pred = _pred(dc_probability=float("nan"))
        outcome = _outcome()
        examples, _ = align_predictions_with_outcomes([pred], [outcome])
        assert examples[0].excluded is True
        assert examples[0].exclusion_reason == "invalid probability"

    def test_negative_probability_excluded(self) -> None:
        pred = _pred(dc_probability=-0.1)
        outcome = _outcome()
        examples, _ = align_predictions_with_outcomes([pred], [outcome])
        assert examples[0].excluded is True
    def test_non_finite_expected_points_excluded(self) -> None:
        pred = _pred(expected_points=float("inf"))
        outcome = _outcome()
        examples, _ = align_predictions_with_outcomes([pred], [outcome])
        assert examples[0].excluded is True
        assert examples[0].exclusion_reason == "non-finite prediction"

    def test_nan_expected_points_excluded(self) -> None:
        pred = _pred(expected_points=float("nan"))
        outcome = _outcome()
        examples, _ = align_predictions_with_outcomes([pred], [outcome])
        assert examples[0].excluded is True

    def test_unpaired_outcomes_counted(self) -> None:
        pred = _pred()
        oc1 = _outcome()
        oc2 = _outcome(player_id=99)
        oc3 = _outcome(fixture_id=99)
        _, unpaired = align_predictions_with_outcomes([pred], [oc1, oc2, oc3])
        assert unpaired == 2

    def test_empty_inputs(self) -> None:
        examples, unpaired = align_predictions_with_outcomes([], [])
        assert examples == []
        assert unpaired == 0

    def test_multiple_distinct_keys_all_matched(self) -> None:
        preds = [
            _pred(player_id=1, fixture_id=10, gameweek=1),
            _pred(player_id=2, fixture_id=11, gameweek=2),
        ]
        outcomes = [
            _outcome(player_id=1, fixture_id=10, gameweek=1),
            _outcome(player_id=2, fixture_id=11, gameweek=2),
        ]
        examples, unpaired = align_predictions_with_outcomes(preds, outcomes)
        assert len(examples) == 2
        assert all(not ex.excluded for ex in examples)
        assert unpaired == 0

    def test_strictly_after_boundary(self) -> None:
        cutoff = _utc(2024, 8, 17, 14, 0, 0)
        outcome_time = cutoff + timedelta(seconds=1)
        pred = _pred(cutoff=cutoff)
        outcome = _outcome(outcome_time=outcome_time)
        examples, _ = align_predictions_with_outcomes([pred], [outcome])
        assert examples[0].excluded is False

# ---------------------------------------------------------------------------
# compute_expected_points_metrics
# ---------------------------------------------------------------------------


class TestComputeExpectedPointsMetrics:
    """Tests for MAE/RMSE computation."""

    def test_perfect_predictions(self) -> None:
        ex = EvaluationExample(
            player_id=1, fixture_id=1, gameweek=1,
            position=PlayerPosition.MIDFIELDER,
            predicted_expected_points=3.0,
            predicted_dc_probability=0.3,
            actual_points=3, dc_threshold_met=True, appeared=True,
            prediction_cutoff=_utc(), outcome_time=_utc(),
        )
        metrics = compute_expected_points_metrics([ex])
        assert metrics.mae == 0.0
        assert metrics.rmse == 0.0
        assert metrics.sample_size == 1

    def test_single_error(self) -> None:
        ex = EvaluationExample(
            player_id=1, fixture_id=1, gameweek=1,
            position=PlayerPosition.MIDFIELDER,
            predicted_expected_points=5.0,
            predicted_dc_probability=0.3,
            actual_points=3, dc_threshold_met=True, appeared=True,
            prediction_cutoff=_utc(), outcome_time=_utc(),
        )
        metrics = compute_expected_points_metrics([ex])
        assert metrics.mae == pytest.approx(2.0)
        assert metrics.rmse == pytest.approx(2.0)
        assert metrics.sample_size == 1

    def test_multiple_errors(self) -> None:
        examples = []
        examples.append(EvaluationExample(
            player_id=1, fixture_id=1, gameweek=1,
            position=PlayerPosition.MIDFIELDER,
            predicted_expected_points=5.0,
            predicted_dc_probability=0.3,
            actual_points=3, dc_threshold_met=True, appeared=True,
            prediction_cutoff=_utc(), outcome_time=_utc(),
        ))
        examples.append(EvaluationExample(
            player_id=2, fixture_id=2, gameweek=1,
            position=PlayerPosition.MIDFIELDER,
            predicted_expected_points=2.0,
            predicted_dc_probability=0.3,
            actual_points=4, dc_threshold_met=False, appeared=True,
            prediction_cutoff=_utc(), outcome_time=_utc(),
        ))
        metrics = compute_expected_points_metrics(examples)
        assert metrics.mae == pytest.approx(2.0)
        assert metrics.rmse == pytest.approx(2.0)
        assert metrics.sample_size == 2

    def test_excluded_examples_skipped(self) -> None:
        ex = EvaluationExample(
            player_id=1, fixture_id=1, gameweek=1,
            position=PlayerPosition.MIDFIELDER,
            predicted_expected_points=5.0,
            predicted_dc_probability=0.3,
            actual_points=3, dc_threshold_met=True, appeared=True,
            prediction_cutoff=_utc(), outcome_time=_utc(),
            excluded=True, exclusion_reason="test",
        )
        metrics = compute_expected_points_metrics([ex])
        assert metrics.mae == 0.0
        assert metrics.rmse == 0.0
        assert metrics.sample_size == 0

    def test_non_finite_predictions_skipped(self) -> None:
        ex = EvaluationExample(
            player_id=1, fixture_id=1, gameweek=1,
            position=PlayerPosition.MIDFIELDER,
            predicted_expected_points=float("nan"),
            predicted_dc_probability=0.3,
            actual_points=3, dc_threshold_met=True, appeared=True,
            prediction_cutoff=_utc(), outcome_time=_utc(),
        )
        metrics = compute_expected_points_metrics([ex])
        assert metrics.sample_size == 0

    def test_empty_examples(self) -> None:
        metrics = compute_expected_points_metrics([])
        assert metrics.mae == 0.0
        assert metrics.rmse == 0.0
        assert metrics.sample_size == 0

# ---------------------------------------------------------------------------
# compute_brier_score
# ---------------------------------------------------------------------------


class TestComputeBrierScore:
    """Tests for Brier score computation and calibration."""

    def test_perfect_predictions(self) -> None:
        metrics = compute_brier_score([1.0, 0.0], [1, 0])
        assert metrics.brier_score == 0.0
        assert metrics.sample_size == 2

    def test_worst_predictions(self) -> None:
        metrics = compute_brier_score([1.0, 0.0], [0, 1])
        assert metrics.brier_score == 1.0
        assert metrics.sample_size == 2

    def test_single_prediction(self) -> None:
        metrics = compute_brier_score([0.7], [1])
        assert metrics.brier_score == pytest.approx((0.3) ** 2)
        assert metrics.sample_size == 1

    def test_multiple_predictions(self) -> None:
        preds = [0.9, 0.8, 0.3, 0.1]
        outcomes = [1, 1, 0, 0]
        metrics = compute_brier_score(preds, outcomes)
        expected = sum((p - o) ** 2 for p, o in zip(preds, outcomes, strict=False)) / len(preds)
        assert metrics.brier_score == pytest.approx(expected)
        assert metrics.sample_size == 4

    def test_empty_input(self) -> None:
        metrics = compute_brier_score([], [])
        assert metrics.brier_score == 0.0
        assert metrics.sample_size == 0
        assert metrics.calibration == {}

    def test_invalid_probabilities_skipped(self) -> None:
        metrics = compute_brier_score([0.5, 1.5, -0.1, 0.3], [1, 0, 0, 1])
        assert metrics.sample_size == 2
        expected = ((0.5 - 1) ** 2 + (0.3 - 1) ** 2) / 2
        assert metrics.brier_score == pytest.approx(expected)

    def test_nan_probabilities_skipped(self) -> None:
        metrics = compute_brier_score([float("nan"), 0.5, float("inf")], [1, 1, 0])
        assert metrics.sample_size == 1
        assert metrics.brier_score == pytest.approx((0.5 - 1) ** 2)

    def test_invalid_outcomes_skipped(self) -> None:
        metrics = compute_brier_score([0.5, 0.5, 0.5], [1, 2, 0])
        assert metrics.sample_size == 2
        expected = ((0.5 - 1) ** 2 + (0.5 - 0) ** 2) / 2
        assert metrics.brier_score == pytest.approx(expected)

    def test_non_int_outcomes_skipped(self) -> None:
        metrics = compute_brier_score([0.5, 0.5], [1.0, 0.0])
        assert metrics.sample_size == 0
        assert metrics.brier_score == 0.0

    def test_bool_outcomes_skipped(self) -> None:
        metrics = compute_brier_score([0.5, 0.5], [True, False])
        assert metrics.sample_size == 0

    def test_calibration_bins_populated(self) -> None:
        preds = [0.05, 0.15, 0.25, 0.95]
        outcomes = [1, 0, 0, 1]
        metrics = compute_brier_score(preds, outcomes, num_bins=10)
        assert len(metrics.calibration) > 0
        assert 0.0 in metrics.calibration
        avg_pred, avg_obs = metrics.calibration[0.0]
        assert avg_pred == pytest.approx(0.05)
        assert avg_obs == 1.0

    def test_calibration_last_bin_inclusive(self) -> None:
        metrics = compute_brier_score([1.0], [1], num_bins=10)
        assert 0.9 in metrics.calibration
        avg_pred, _avg_obs = metrics.calibration[0.9]
        assert avg_pred == pytest.approx(1.0)

    def test_custom_num_bins(self) -> None:
        metrics = compute_brier_score([0.1, 0.5], [1, 0], num_bins=5)
        assert len(metrics.calibration) == 2

    def test_invalid_num_bins_raises(self) -> None:
        with pytest.raises(ValueError, match="num_bins must be >= 1"):
            compute_brier_score([0.5], [1], num_bins=0)

    def test_mismatched_lengths_truncates(self) -> None:
        metrics = compute_brier_score([0.5, 0.3], [1])
        assert metrics.sample_size == 1

    def test_extra_outcomes_ignored(self) -> None:
        metrics = compute_brier_score([0.5], [1, 0])
        assert metrics.sample_size == 1

# ---------------------------------------------------------------------------
# evaluate_predictions (top-level orchestrator)
# ---------------------------------------------------------------------------


class TestEvaluatePredictions:
    """End-to-end tests for the full evaluation pipeline."""

    def test_perfect_predictions_summary(self) -> None:
        pred = _pred(cutoff=_utc(2024, 8, 10, 12), expected_points=3.0)
        outcome = _outcome(outcome_time=_utc(2024, 8, 17, 14))
        summary = evaluate_predictions([pred], [outcome])
        assert isinstance(summary, EvaluationSummary)
        assert summary.total_predictions == 1
        assert summary.total_outcomes == 1
        assert summary.n_matched == 1
        assert summary.n_excluded == 0
        assert summary.n_unpaired_outcomes == 0
        assert summary.expected_points.mae == 0.0
        assert summary.expected_points.sample_size == 1
        # DC: P=0.3, outcome=1 (DC met), appeared=True, eligible
        # brier = (0.3 - 1)^2 = 0.49
        assert summary.dc_brier.brier_score == pytest.approx(0.49)
        assert summary.dc_brier.sample_size == 1

    def test_excluded_example_appears_in_summary(self) -> None:
        pred = _pred()
        summary = evaluate_predictions([pred], [])
        assert summary.n_matched == 1
        assert summary.n_excluded == 1
        assert summary.exclusion_reasons.get("missing outcome") == 1
        assert summary.expected_points.sample_size == 0

    def test_temporal_leakage_summary(self) -> None:
        pred = _pred(cutoff=_utc(2024, 8, 20, 12))
        outcome = _outcome(outcome_time=_utc(2024, 8, 17, 14))
        summary = evaluate_predictions([pred], [outcome])
        assert summary.n_excluded == 1
        assert summary.exclusion_reasons.get(
            "temporal leakage: outcome observed at or before prediction cutoff"
        ) == 1

    def test_empty_inputs(self) -> None:
        summary = evaluate_predictions([], [])
        assert summary.total_predictions == 0
        assert summary.total_outcomes == 0
        assert summary.n_matched == 0
        assert summary.n_excluded == 0
        assert summary.expected_points.sample_size == 0
        assert summary.dc_brier.sample_size == 0

    def test_mixed_examples(self) -> None:
        """Three predictions: one valid, one missing outcome, one temporal leak."""
        preds = [
            _pred(player_id=1, cutoff=_utc(2024, 8, 10, 12)),
            _pred(player_id=2, cutoff=_utc(2024, 8, 10, 12)),
            _pred(player_id=3, cutoff=_utc(2024, 8, 20, 12)),
        ]
        outcomes = [
            _outcome(player_id=1, outcome_time=_utc(2024, 8, 17, 14)),
            # player 2 has no outcome -> missing
            _outcome(player_id=3, outcome_time=_utc(2024, 8, 17, 14)),
        ]
        summary = evaluate_predictions(preds, outcomes)
        assert summary.total_predictions == 3
        assert summary.total_outcomes == 2
        assert summary.n_matched == 3
        assert summary.n_excluded == 2
        assert summary.exclusion_reasons.get("missing outcome") == 1
        assert summary.exclusion_reasons.get(
            "temporal leakage: outcome observed at or before prediction cutoff"
        ) == 1
        assert summary.expected_points.sample_size == 1

    def test_dc_brier_excludes_non_appeared(self) -> None:
        pred = _pred(cutoff=_utc(2024, 8, 10, 12), dc_probability=0.5)
        outcome = _outcome(
            outcome_time=_utc(2024, 8, 17, 14),
            appeared=False, dc_met=False,
        )
        summary = evaluate_predictions([pred], [outcome])
        assert summary.dc_brier.sample_size == 0

    def test_dc_brier_excludes_gkp(self) -> None:
        pred = _pred(
            cutoff=_utc(2024, 8, 10, 12),
            position=PlayerPosition.GOALKEEPER,
            dc_probability=0.0,
        )
        outcome = _outcome(
            outcome_time=_utc(2024, 8, 17, 14),
            appeared=True, dc_met=False,
        )
        summary = evaluate_predictions([pred], [outcome])
        assert summary.dc_brier.sample_size == 0

    def test_no_temporal_enforcement(self) -> None:
        pred = _pred(cutoff=_utc(2024, 8, 20, 12))
        outcome = _outcome(outcome_time=_utc(2024, 8, 17, 14))
        summary = evaluate_predictions(
            [pred], [outcome], enforce_temporal_order=False
        )
        assert summary.n_excluded == 0
        assert summary.expected_points.sample_size == 1

# ---------------------------------------------------------------------------
# outcome_from_history
# ---------------------------------------------------------------------------


class TestOutcomeFromHistory:
    """Tests for ``outcome_from_history`` conversion helper."""

    def _base_time(self) -> datetime:
        return _utc(2024, 8, 17, 14)

    def test_defender_dc_met(self) -> None:
        # DEF: needs 10+ CBIT (tackles + CBI)
        history = _match_history(
            minutes=90, tackles=6, cbi=5, recoveries=0,
            total_points=1,
        )
        outcome = outcome_from_history(history, PlayerPosition.DEFENDER, self._base_time())
        assert outcome.actual_points == 1
        assert outcome.appeared is True
        assert outcome.dc_threshold_met is True  # 6+5=11 >= 10

    def test_defender_dc_not_met(self) -> None:
        history = _match_history(
            minutes=90, tackles=3, cbi=4,  # 7 < 10
            total_points=0,
        )
        outcome = outcome_from_history(history, PlayerPosition.DEFENDER, self._base_time())
        assert outcome.dc_threshold_met is False
        assert outcome.appeared is True

    def test_midfielder_dc_met(self) -> None:
        # MID: needs 12+ CBIRT (tackles + CBI + recoveries)
        history = _match_history(
            minutes=90, tackles=5, cbi=4, recoveries=4,  # 13 >= 12
            total_points=1,
        )
        outcome = outcome_from_history(history, PlayerPosition.MIDFIELDER, self._base_time())
        assert outcome.dc_threshold_met is True

    def test_midfielder_dc_not_met(self) -> None:
        history = _match_history(
            minutes=90, tackles=3, cbi=3, recoveries=3,  # 9 < 12
            total_points=0,
        )
        outcome = outcome_from_history(history, PlayerPosition.MIDFIELDER, self._base_time())
        assert outcome.dc_threshold_met is False

    def test_forward_dc_met(self) -> None:
        history = _match_history(
            minutes=90, tackles=4, cbi=4, recoveries=5,  # 13 >= 12
            total_points=1,
        )
        outcome = outcome_from_history(history, PlayerPosition.FORWARD, self._base_time())
        assert outcome.dc_threshold_met is True

    def test_forward_dc_not_met(self) -> None:
        history = _match_history(
            minutes=90, tackles=3, cbi=3, recoveries=3,  # 9 < 12
            total_points=0,
        )
        outcome = outcome_from_history(history, PlayerPosition.FORWARD, self._base_time())
        assert outcome.dc_threshold_met is False

    def test_goalkeeper_dc_always_false(self) -> None:
        history = _match_history(minutes=90, tackles=10, cbi=10, total_points=0)
        outcome = outcome_from_history(history, PlayerPosition.GOALKEEPER, self._base_time())
        assert outcome.dc_threshold_met is False
        assert outcome.appeared is True

    def test_player_did_not_appear(self) -> None:
        history = _match_history(minutes=0, total_points=0)
        outcome = outcome_from_history(history, PlayerPosition.MIDFIELDER, self._base_time())
        assert outcome.appeared is False
        assert outcome.dc_threshold_met is False

    def test_naive_outcome_time_raises(self) -> None:
        history = _match_history(minutes=90)
        with pytest.raises(EvaluationError, match="timezone-aware"):
            outcome_from_history(history, PlayerPosition.MIDFIELDER, _naive_now())

    def test_non_utc_outcome_time_raises(self) -> None:
        tz = timezone(timedelta(hours=2))
        history = _match_history(minutes=90)
        with pytest.raises(EvaluationError, match="must be in UTC"):
            outcome_from_history(
                history, PlayerPosition.MIDFIELDER,
                datetime(2024, 8, 17, 14, tzinfo=tz),
            )

    def test_fields_copied_from_history(self) -> None:
        history = _match_history(
            player_id=42, fixture_id=7, gameweek=5, total_points=6, minutes=90,
        )
        outcome = outcome_from_history(history, PlayerPosition.FORWARD, self._base_time())
        assert outcome.player_id == 42
        assert outcome.fixture_id == 7
        assert outcome.gameweek == 5
        assert outcome.actual_points == 6

    def test_dc_threshold_exactly_boundary(self) -> None:
        """DEF: exactly 10 CBIT -> should meet threshold."""
        history = _match_history(minutes=90, tackles=5, cbi=5, total_points=0)
        outcome = outcome_from_history(history, PlayerPosition.DEFENDER, self._base_time())
        assert outcome.dc_threshold_met is True

    def test_dc_threshold_just_below_boundary(self) -> None:
        """DEF: 9 CBIT -> should not meet threshold."""
        history = _match_history(minutes=90, tackles=4, cbi=5, total_points=0)
        outcome = outcome_from_history(history, PlayerPosition.DEFENDER, self._base_time())
        assert outcome.dc_threshold_met is False

# ---------------------------------------------------------------------------
# evaluate_dc_models_from_history (bridge to evaluate_dc_models_brier)
# ---------------------------------------------------------------------------


class TestEvaluateDcModelsFromHistory:
    """Tests for the bridge function to ``evaluate_dc_models_brier``."""

    def _build_history(self, position: PlayerPosition) -> list[PlayerMatchHistory]:
        """Build a synthetic multi-gameweek history for testing.

        Creates a player who appears in every gameweek with enough
        defensive actions to meet the DC threshold.
        """
        histories = []
        threshold_map = {
            PlayerPosition.DEFENDER: 10,
            PlayerPosition.MIDFIELDER: 12,
            PlayerPosition.FORWARD: 12,
            PlayerPosition.GOALKEEPER: 10,
        }
        threshold = threshold_map[position]
        tackles = threshold
        cbi = 0
        recoveries = 0

        for gw in range(1, 10):
            histories.append(
                PlayerMatchHistory(
                    player_id=1,
                    fixture_id=gw * 10,
                    gameweek=gw,
                    minutes=90,
                    total_points=1,
                    tackles=tackles,
                    clearances_blocks_interceptions=cbi,
                    recoveries=recoveries,
                )
            )
        return histories

    def test_defender_history_returns_both_models(self) -> None:
        history = self._build_history(PlayerPosition.DEFENDER)
        results = evaluate_dc_models_from_history(history, PlayerPosition.DEFENDER)
        assert "history" in results
        assert "poisson" in results

    def test_midfielder_history_returns_both_models(self) -> None:
        history = self._build_history(PlayerPosition.MIDFIELDER)
        results = evaluate_dc_models_from_history(history, PlayerPosition.MIDFIELDER)
        assert "history" in results
        assert "poisson" in results

    def test_forward_history_returns_both_models(self) -> None:
        history = self._build_history(PlayerPosition.FORWARD)
        results = evaluate_dc_models_from_history(history, PlayerPosition.FORWARD)
        assert "history" in results
        assert "poisson" in results

    def test_goalkeeper_returns_empty(self) -> None:
        history = self._build_history(PlayerPosition.GOALKEEPER)
        results = evaluate_dc_models_from_history(history, PlayerPosition.GOALKEEPER)
        for _model_name, metrics in results.items():
            assert metrics.sample_size == 0

    def test_results_are_brier_metrics(self) -> None:
        history = self._build_history(PlayerPosition.DEFENDER)
        results = evaluate_dc_models_from_history(history, PlayerPosition.DEFENDER)
        for _model_name, metrics in results.items():
            assert isinstance(metrics, BrierMetrics)
            assert isinstance(metrics.brier_score, float)
            assert isinstance(metrics.sample_size, int)
            assert isinstance(metrics.calibration, dict)

    def test_sample_size_grows_with_more_history(self) -> None:
        short = self._build_history(PlayerPosition.DEFENDER)[:5]
        long = self._build_history(PlayerPosition.DEFENDER)
        short_results = evaluate_dc_models_from_history(short, PlayerPosition.DEFENDER)
        long_results = evaluate_dc_models_from_history(long, PlayerPosition.DEFENDER)
        assert long_results["history"].sample_size > short_results["history"].sample_size

    def test_brier_score_in_valid_range(self) -> None:
        history = self._build_history(PlayerPosition.MIDFIELDER)
        results = evaluate_dc_models_from_history(history, PlayerPosition.MIDFIELDER)
        for _model_name, metrics in results.items():
            assert 0.0 <= metrics.brier_score <= 1.0


# ---------------------------------------------------------------------------
# Integration: outcome_from_history -> evaluate_predictions
# ---------------------------------------------------------------------------


class TestIntegrationOutcomeHistory:
    """Integration tests using ``outcome_from_history`` with the evaluation pipeline."""

    def test_full_pipeline_with_history_outcomes(self) -> None:
        """End-to-end: create predictions, derive outcomes from history, evaluate."""
        cutoff = _utc(2024, 8, 10, 12)
        outcome_time = _utc(2024, 8, 17, 14)

        pred = _pred(
            player_id=1, fixture_id=10, gameweek=1,
            position=PlayerPosition.MIDFIELDER,
            expected_points=2.5,
            dc_probability=0.4,
            cutoff=cutoff,
        )

        history = _match_history(
            player_id=1, fixture_id=10, gameweek=1,
            minutes=90, tackles=5, cbi=4, recoveries=4,  # CBIRT=13 >= 12 -> DC met
            total_points=1,
        )
        outcome = outcome_from_history(history, PlayerPosition.MIDFIELDER, outcome_time)

        summary = evaluate_predictions([pred], [outcome])
        assert summary.n_matched == 1
        assert summary.n_excluded == 0
        assert summary.expected_points.sample_size == 1
        # MAE = |2.5 - 1| = 1.5
        assert summary.expected_points.mae == pytest.approx(1.5)
        # DC Brier: P=0.4, outcome=1 (DC met), appeared=True, eligible
        assert summary.dc_brier.brier_score == pytest.approx((0.4 - 1) ** 2)
        assert summary.dc_brier.sample_size == 1


# ---------------------------------------------------------------------------
# BrierMetrics temporal_safety_enforced field
# ---------------------------------------------------------------------------


class TestBrierMetricsTemporalSafety:
    """Tests for the temporal_safety_enforced field on BrierMetrics."""

    def test_compute_brier_score_has_temporal_safety(self) -> None:
        """compute_brier_score sets temporal_safety_enforced=True (caller-managed)."""
        metrics = compute_brier_score([0.5, 0.6], [1, 0])
        assert metrics.temporal_safety_enforced is True

    def test_empty_brier_score_has_temporal_safety(self) -> None:
        """Empty compute_brier_score still reports temporal_safety_enforced=True."""
        metrics = compute_brier_score([], [])
        assert metrics.temporal_safety_enforced is True

    def test_brier_metrics_default_temporal_safety_is_false(self) -> None:
        """BrierMetrics without explicit temporal_safety_enforced defaults to False."""
        metrics = BrierMetrics(brier_score=0.1, sample_size=10, calibration={})
        assert metrics.temporal_safety_enforced is False


# ---------------------------------------------------------------------------
# Temporal conflict detection
# ---------------------------------------------------------------------------


class TestTemporalConflictDetection:
    """Tests for cross-temporal outcome conflict detection."""

    def test_no_conflicts_with_single_outcome(self) -> None:
        """Single outcome per key produces no conflicts."""
        outcomes = [_outcome(player_id=1, fixture_id=10, gameweek=1)]
        conflicts = _detect_temporal_conflicts(outcomes)
        assert conflicts == []

    def test_conflict_same_fixture_different_outcome_time(self) -> None:
        """Same (player, fixture, gw) with different outcome_times triggers conflict."""
        outcomes = [
            _outcome(
                player_id=1, fixture_id=10, gameweek=1,
                outcome_time=_utc(2024, 8, 17, 14),
                actual_points=3,
            ),
            _outcome(
                player_id=1, fixture_id=10, gameweek=1,
                outcome_time=_utc(2024, 8, 18, 14),
                actual_points=1,
            ),
        ]
        conflicts = _detect_temporal_conflicts(outcomes)
        assert len(conflicts) == 1
        assert "player=1, fixture=10, gw=1" in conflicts[0]

    def test_no_conflict_different_gameweeks(self) -> None:
        """Same fixture but different gameweeks is not a conflict."""
        outcomes = [
            _outcome(player_id=1, fixture_id=10, gameweek=1,
                     outcome_time=_utc(2024, 8, 17)),
            _outcome(player_id=1, fixture_id=10, gameweek=2,
                     outcome_time=_utc(2024, 8, 24)),
        ]
        conflicts = _detect_temporal_conflicts(outcomes)
        assert conflicts == []

    def test_conflict_same_time_different_points(self) -> None:
        """Same outcome_time but different actual_points triggers conflict."""
        t = _utc(2024, 8, 17, 14)
        outcomes = [
            _outcome(player_id=1, fixture_id=10, gameweek=1,
                     outcome_time=t, actual_points=3),
            _outcome(player_id=1, fixture_id=10, gameweek=1,
                     outcome_time=t, actual_points=1),
        ]
        conflicts = _detect_temporal_conflicts(outcomes)
        assert len(conflicts) == 1
        assert "actual_points" in conflicts[0]


# ---------------------------------------------------------------------------
# prediction_cutoff / temporal_safety tests for evaluate_dc_models_from_history
# ---------------------------------------------------------------------------


class TestEvaluateDcModelsFromHistoryCutoff:
    """Tests for prediction_cutoff and temporal_safety_enforced in the bridge."""

    def test_prediction_cutoff_forwarded_to_brier(self) -> None:
        """prediction_cutoff is validated and forwarded to evaluate_dc_models_brier."""
        history = [
            _make_history_bridge(gameweek=i, tackles=5, cbi=8)
            for i in range(1, 6)
        ]
        cutoff = _utc(2024, 8, 17)
        results = evaluate_dc_models_from_history(
            history=history,
            position=PlayerPosition.DEFENDER,
            prediction_cutoff=cutoff,
        )
        for metrics in results.values():
            assert metrics.temporal_safety_enforced is True

    def test_no_cutoff_backward_compatible(self) -> None:
        """Without prediction_cutoff, temporal_safety_enforced is False (no times)."""
        history = [
            _make_history_bridge(gameweek=i, tackles=5, cbi=8)
            for i in range(1, 6)
        ]
        results = evaluate_dc_models_from_history(
            history=history,
            position=PlayerPosition.DEFENDER,
        )
        # No outcome_time or kickoff_time -> gameweek-only filtering
        for metrics in results.values():
            if metrics.sample_size > 0:
                assert metrics.temporal_safety_enforced is False

    def test_naive_prediction_cutoff_raises(self) -> None:
        """A naive prediction_cutoff raises EvaluationError."""
        history = [_make_history_bridge(gameweek=1, tackles=5, cbi=8)]
        with pytest.raises(EvaluationError, match="timezone-aware"):
            evaluate_dc_models_from_history(
                history=history,
                position=PlayerPosition.DEFENDER,
                prediction_cutoff=datetime(2024, 8, 17),  # noqa: DTZ001
            )

    def test_non_utc_prediction_cutoff_raises(self) -> None:
        """A non-UTC prediction_cutoff raises EvaluationError."""
        tz_plus5 = timezone(timedelta(hours=5))
        history = [_make_history_bridge(gameweek=1, tackles=5, cbi=8)]
        with pytest.raises(EvaluationError, match="must be in UTC"):
            evaluate_dc_models_from_history(
                history=history,
                position=PlayerPosition.DEFENDER,
                prediction_cutoff=datetime(2024, 8, 17, tzinfo=tz_plus5),
            )

    def test_temporal_safety_true_with_outcome_time_no_cutoff(self) -> None:
        """When outcome_time is present (no cutoff), temporal safety is True."""
        history = [
            _make_history_bridge(
                gameweek=i, tackles=5, cbi=8,
                outcome_time=_utc(2024, 8, i, 14),
            )
            for i in range(1, 6)
        ]
        results = evaluate_dc_models_from_history(
            history=history,
            position=PlayerPosition.DEFENDER,
        )
        for metrics in results.values():
            assert metrics.temporal_safety_enforced is True


def _make_history_bridge(
    *,
    player_id: int = 1,
    fixture_id: int = 100,
    gameweek: int = 1,
    minutes: int = 90,
    tackles: int = 0,
    cbi: int = 0,
    recoveries: int = 0,
    outcome_time: datetime | None = None,
    kickoff_time: datetime | None = None,
    total_points: int = 3,
) -> PlayerMatchHistory:
    """Build a PlayerMatchHistory for bridge tests."""
    return PlayerMatchHistory(
        player_id=player_id,
        fixture_id=fixture_id,
        gameweek=gameweek,
        minutes=minutes,
        tackles=tackles,
        clearances_blocks_interceptions=cbi,
        recoveries=recoveries,
        outcome_time=outcome_time,
        kickoff_time=kickoff_time,
        total_points=total_points,
    )
