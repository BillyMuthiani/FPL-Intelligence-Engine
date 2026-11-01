"""Tests for the defensive probability model — Stage 4H.

Covers:
    - Private helpers: _dc_threshold, _dc_action_count, _is_eligible
    - Public API: predict_defensive_contribution_probability,
      dc_action_count, evaluate_dc_models_brier
    - Dataclasses: DCPredictionResult, DCEvaluationResult, DCModelError
    - Threshold and eligibility semantics
    - Prior fallback (no-history) behaviour
    - Brier score and calibration logic for evaluate_dc_models_brier
    - Temporal-safety: prediction_cutoff forwarding in _appearance_history,
      historical_dc_probability, poisson_dc_probability, and
      predict_defensive_contribution_probability
    - Temporal-safety: outcome_time-based filtering in evaluate_dc_models_brier
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta, timezone

import pytest

from fpl_engine.analytics.defensive_probability import (
    DC_DEFAULT_MINUTES,
    DC_MODEL_HISTORY,
    DC_MODEL_POISSON,
    DC_POSITION_PRIOR,
    DC_PRIOR_PSEUDO_COUNT,
    DEF_DC_THRESHOLD,
    MID_FWD_DC_THRESHOLD,
    DCEvaluationResult,
    DCModelError,
    DCPredictionResult,
    _appearance_history,
    _poisson_geq,
    _poisson_sf,
    dc_action_count,
    evaluate_dc_models_brier,
    historical_dc_probability,
    poisson_dc_probability,
    predict_defensive_contribution_probability,
)
from fpl_engine.models import PlayerMatchHistory, PlayerPosition

# ---------------------------------------------------------------------------
# Test data factories
# ---------------------------------------------------------------------------


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    """Build a UTC-aware datetime for test data."""
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def _make_history(
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
) -> PlayerMatchHistory:
    """Build a PlayerMatchHistory with specified defensive stats."""
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
    )
# ---------------------------------------------------------------------------
# DCPredictionResult
# ---------------------------------------------------------------------------


class TestDCPredictionResult:

    def test_fields(self) -> None:
        """DCPredictionResult stores all five fields correctly."""
        result = DCPredictionResult(
            probability=0.42,
            model=DC_MODEL_POISSON,
            n_observations=15,
            shrunk=True,
            prior_used=0.25,
        )
        assert result.probability == 0.42
        assert result.model == DC_MODEL_POISSON
        assert result.n_observations == 15
        assert result.shrunk is True
        assert result.prior_used == 0.25

    def test_history_model(self) -> None:
        result = DCPredictionResult(
            probability=0.35,
            model=DC_MODEL_HISTORY,
            n_observations=20,
            shrunk=False,
            prior_used=0.35,
        )
        assert result.model == DC_MODEL_HISTORY
        assert result.n_observations == 20


# ---------------------------------------------------------------------------
# DCEvaluationResult
# ---------------------------------------------------------------------------


class TestDCEvaluationResult:

    def test_defaults(self) -> None:
        result = DCEvaluationResult(
            model=DC_MODEL_HISTORY,
            brier_score=0.0,
            sample_size=0,
            calibration={},
            position=PlayerPosition.MIDFIELDER,
        )
        assert result.model == DC_MODEL_HISTORY
        assert result.brier_score == 0.0
        assert result.sample_size == 0
        assert result.calibration == {}
        assert result.position == PlayerPosition.MIDFIELDER

    def test_with_data(self) -> None:
        cal: dict[float, tuple[float, float]] = {0.0: (0.05, 0.1)}
        result = DCEvaluationResult(
            model=DC_MODEL_POISSON,
            brier_score=0.15,
            sample_size=100,
            calibration=cal,
            position=PlayerPosition.DEFENDER,
        )
        assert result.model == DC_MODEL_POISSON
        assert result.brier_score == 0.15
        assert result.sample_size == 100
        assert result.calibration == cal
        assert result.position == PlayerPosition.DEFENDER


# ---------------------------------------------------------------------------
# Eligibility and threshold helpers
# ---------------------------------------------------------------------------


class TestEligibilityAndThreshold:

    def test_gkp_ineligible(self) -> None:
        from fpl_engine.analytics.defensive_probability import _is_eligible

        assert not _is_eligible(PlayerPosition.GOALKEEPER)

    @pytest.mark.parametrize(
        "position",
        [PlayerPosition.DEFENDER, PlayerPosition.MIDFIELDER, PlayerPosition.FORWARD],
    )
    def test_eligible_positions(self, position: PlayerPosition) -> None:
        from fpl_engine.analytics.defensive_probability import _is_eligible

        assert _is_eligible(position)

    def test_def_threshold_is_10(self) -> None:
        from fpl_engine.analytics.defensive_probability import _dc_threshold

        assert _dc_threshold(PlayerPosition.DEFENDER) == 10

    def test_mid_fwd_threshold_is_12(self) -> None:
        from fpl_engine.analytics.defensive_probability import _dc_threshold

        assert _dc_threshold(PlayerPosition.MIDFIELDER) == 12
        assert _dc_threshold(PlayerPosition.FORWARD) == 12

    def test_threshold_gkp_raises(self) -> None:
        from fpl_engine.analytics.defensive_probability import _dc_threshold

        with pytest.raises(ValueError, match="not eligible"):
            _dc_threshold(PlayerPosition.GOALKEEPER)


# ---------------------------------------------------------------------------
# _dc_action_count helper
# ---------------------------------------------------------------------------


class TestDCActionCount:

    def test_def_cbit(self) -> None:
        from fpl_engine.analytics.defensive_probability import _dc_action_count

        h = _make_history(tackles=5, cbi=8, recoveries=3)
        assert _dc_action_count(h, PlayerPosition.DEFENDER) == 13

    def test_mid_cbirt(self) -> None:
        from fpl_engine.analytics.defensive_probability import _dc_action_count

        h = _make_history(tackles=3, cbi=5, recoveries=7)
        assert _dc_action_count(h, PlayerPosition.MIDFIELDER) == 15

    def test_fwd_cbirt(self) -> None:
        from fpl_engine.analytics.defensive_probability import _dc_action_count

        h = _make_history(tackles=2, cbi=4, recoveries=6)
        assert _dc_action_count(h, PlayerPosition.FORWARD) == 12


# ---------------------------------------------------------------------------
# dc_action_count public wrapper
# ---------------------------------------------------------------------------


class TestDcActionCountPublic:

    def test_returns_correct_count(self) -> None:
        h = _make_history(tackles=5, cbi=8, recoveries=3)
        assert dc_action_count(h, PlayerPosition.DEFENDER) == 13

    def test_mid_includes_recoveries(self) -> None:
        h = _make_history(tackles=4, cbi=3, recoveries=5)
        assert dc_action_count(h, PlayerPosition.MIDFIELDER) == 12

    def test_gkp_not_raised(self) -> None:
        """dc_action_count does not enforce eligibility; it just computes."""
        result = dc_action_count(
            _make_history(tackles=3, cbi=2, recoveries=1),
            PlayerPosition.GOALKEEPER,
        )
        # GKP falls through to the else branch (CBIT + recoveries = 6)
        assert result == 6
# ---------------------------------------------------------------------------
# Historical DC probability
# ---------------------------------------------------------------------------


class TestHistoricalDcProbability:

    def test_no_history_returns_prior(self) -> None:
        result = historical_dc_probability(
            history=[], position=PlayerPosition.MIDFIELDER, target_gameweek=5
        )
        assert result.probability == pytest.approx(
            DC_POSITION_PRIOR[PlayerPosition.MIDFIELDER]
        )
        assert result.model == DC_MODEL_HISTORY

    def test_all_successes_shrinks_toward_one(self) -> None:
        history = [
            _make_history(gameweek=i, tackles=5, cbi=8)
            for i in range(1, 11)
        ]
        result = historical_dc_probability(
            history, PlayerPosition.MIDFIELDER, target_gameweek=11
        )
        assert result.probability > 0.7

    def test_all_failures_shrinks_toward_zero(self) -> None:
        history = [
            _make_history(gameweek=i, tackles=0, cbi=0)
            for i in range(1, 11)
        ]
        result = historical_dc_probability(
            history, PlayerPosition.MIDFIELDER, target_gameweek=11
        )
        assert result.probability < 0.2

    def test_leakage_guard_excludes_target_gw(self) -> None:
        prior = _make_history(gameweek=5, tackles=5, cbi=8)
        prior_old = _make_history(gameweek=4, tackles=5, cbi=8)
        result = historical_dc_probability(
            [prior, prior_old],
            position=PlayerPosition.MIDFIELDER,
            target_gameweek=5,
        )
        pseudo = DC_PRIOR_PSEUDO_COUNT
        prior_val = DC_POSITION_PRIOR[PlayerPosition.MIDFIELDER]
        expected = (pseudo * prior_val + 1) / (pseudo + 1)
        assert result.probability == pytest.approx(expected)

    def test_uses_custom_prior_and_pseudo_count(self) -> None:
        history = [_make_history(gameweek=1, tackles=0, cbi=0)]
        result = historical_dc_probability(
            history,
            position=PlayerPosition.DEFENDER,
            target_gameweek=5,
            prior=0.3,
            pseudo_count=3.0,
        )
        expected = (3.0 * 0.3 + 0) / (3.0 + 1)
        assert result.probability == pytest.approx(expected)

    def test_gkp_returns_zero(self) -> None:
        result = historical_dc_probability(
            history=[], position=PlayerPosition.GOALKEEPER, target_gameweek=5
        )
        assert result.probability == 0.0
        assert result.model == DC_MODEL_HISTORY
        assert result.n_observations == 0
# ---------------------------------------------------------------------------
# Poisson DC probability
# ---------------------------------------------------------------------------


class TestPoissonDcProbability:

    def test_no_history_returns_prior(self) -> None:
        result = poisson_dc_probability(
            history=[], position=PlayerPosition.MIDFIELDER, target_gameweek=5
        )
        assert result.probability == pytest.approx(
            DC_POSITION_PRIOR[PlayerPosition.MIDFIELDER]
        )
        assert result.model == DC_MODEL_POISSON

    def test_high_action_rate_high_probability(self) -> None:
        """Very high action rate → probability well above prior."""
        history = [
            _make_history(gameweek=i, tackles=20, cbi=30, recoveries=20)
            for i in range(1, 11)
        ]
        result = poisson_dc_probability(
            history, PlayerPosition.MIDFIELDER, target_gameweek=11
        )
        assert result.probability > DC_POSITION_PRIOR[PlayerPosition.MIDFIELDER]

    def test_low_action_rate_low_probability(self) -> None:
        history = [
            _make_history(gameweek=i, tackles=0, cbi=0, recoveries=0)
            for i in range(1, 11)
        ]
        result = poisson_dc_probability(
            history, PlayerPosition.MIDFIELDER, target_gameweek=11
        )
        assert result.probability < 0.2

    def test_expected_minutes_zero(self) -> None:
        """When expected_minutes=0, Poisson rate is 0 → falls back to prior."""
        history = [
            _make_history(gameweek=i, tackles=5, cbi=8, recoveries=7)
            for i in range(1, 11)
        ]
        result = poisson_dc_probability(
            history, position=PlayerPosition.MIDFIELDER, target_gameweek=11,
            expected_minutes=0,
        )
        # Falls back to prior probability (shrunk=True)
        assert result.probability == pytest.approx(
            DC_POSITION_PRIOR[PlayerPosition.MIDFIELDER]
        )
        assert result.shrunk is True
        assert result.n_observations == 0

    def test_uses_custom_expected_minutes(self) -> None:
        """Lower expected_minutes than full 90 reduces probability."""
        history = [
            _make_history(gameweek=i, tackles=20, cbi=30, recoveries=20)
            for i in range(1, 11)
        ]
        result_full = poisson_dc_probability(
            history, position=PlayerPosition.MIDFIELDER, target_gameweek=11,
            expected_minutes=90,
        )
        result_partial = poisson_dc_probability(
            history, position=PlayerPosition.MIDFIELDER, target_gameweek=11,
            expected_minutes=60,
        )
        # Fewer expected minutes → lower probability
        assert result_partial.probability < result_full.probability
        assert 0.0 < result_partial.probability < 1.0

    def test_gkp_returns_zero(self) -> None:
        result = poisson_dc_probability(
            history=[], position=PlayerPosition.GOALKEEPER, target_gameweek=5
        )
        assert result.probability == 0.0
        assert result.model == DC_MODEL_POISSON


# ---------------------------------------------------------------------------
# predict_defensive_contribution_probability dispatch
# ---------------------------------------------------------------------------


class TestPredictDefensiveContributionProbability:

    def test_gkp_returns_zero(self) -> None:
        result = predict_defensive_contribution_probability(
            history=[],
            position=PlayerPosition.GOALKEEPER,
            target_gameweek=5,
        )
        assert result.probability == 0.0

    def test_default_model_is_history(self) -> None:
        result = predict_defensive_contribution_probability(
            history=[],
            position=PlayerPosition.DEFENDER,
            target_gameweek=5,
        )
        assert result.model == DC_MODEL_HISTORY

    def test_model_selection_history(self) -> None:
        result = predict_defensive_contribution_probability(
            history=[],
            position=PlayerPosition.DEFENDER,
            target_gameweek=5,
            model=DC_MODEL_HISTORY,
        )
        assert result.model == DC_MODEL_HISTORY

    def test_model_selection_poisson(self) -> None:
        result = predict_defensive_contribution_probability(
            history=[],
            position=PlayerPosition.MIDFIELDER,
            target_gameweek=5,
            model=DC_MODEL_POISSON,
        )
        assert result.model == DC_MODEL_POISSON

    def test_invalid_model_raises(self) -> None:
        with pytest.raises(DCModelError, match="Unknown model"):
            predict_defensive_contribution_probability(
                history=[],
                position=PlayerPosition.MIDFIELDER,
                target_gameweek=5,
                model="invalid",
            )

    def test_history_with_data(self) -> None:
        history = [
            _make_history(gameweek=1, tackles=5, cbi=8),
            _make_history(gameweek=2, tackles=3, cbi=2),
        ]
        result = predict_defensive_contribution_probability(
            history=history,
            position=PlayerPosition.DEFENDER,
            target_gameweek=3,
        )
        assert result.model == DC_MODEL_HISTORY
        assert 0.0 < result.probability < 1.0

    def test_none_history_treated_as_empty(self) -> None:
        result = predict_defensive_contribution_probability(
            history=None,
            position=PlayerPosition.MIDFIELDER,
            target_gameweek=5,
        )
        assert result.probability == pytest.approx(
            DC_POSITION_PRIOR[PlayerPosition.MIDFIELDER]
        )


# ---------------------------------------------------------------------------
# Brier score helper
# ---------------------------------------------------------------------------


class TestBrierScore:

    def test_perfect_prediction(self) -> None:
        from fpl_engine.analytics.defensive_probability import (
            _brier_score_component,
        )

        assert _brier_score_component(1.0, 1) == 0.0
        assert _brier_score_component(0.0, 0) == 0.0

    def test_worst_prediction(self) -> None:
        from fpl_engine.analytics.defensive_probability import (
            _brier_score_component,
        )

        assert _brier_score_component(1.0, 0) == 1.0
        assert _brier_score_component(0.0, 1) == 1.0

    def test_intermediate(self) -> None:
        from fpl_engine.analytics.defensive_probability import (
            _brier_score_component,
        )

        assert _brier_score_component(0.5, 1) == 0.25
        assert _brier_score_component(0.5, 0) == 0.25


# ---------------------------------------------------------------------------
# evaluate_dc_models_brier
# ---------------------------------------------------------------------------


class TestEvaluateDcModelsBrier:

    def test_gkp_returns_empty_results(self) -> None:
        results = evaluate_dc_models_brier(
            history=[],
            position=PlayerPosition.GOALKEEPER,
            max_gameweek=5,
        )
        assert "history" in results
        assert "poisson" in results
        assert results["history"].sample_size == 0
        assert results["poisson"].sample_size == 0
        assert results["history"].brier_score == 0.0
        assert results["poisson"].brier_score == 0.0

    def test_no_history_returns_empty(self) -> None:
        results = evaluate_dc_models_brier(
            history=[],
            position=PlayerPosition.MIDFIELDER,
            max_gameweek=5,
        )
        assert results["history"].sample_size == 0
        assert results["poisson"].sample_size == 0

    def test_with_synthetic_history(self) -> None:
        history = [
            _make_history(gameweek=i, tackles=5, cbi=8)
            for i in range(1, 11)
        ]
        results = evaluate_dc_models_brier(
            history=history,
            position=PlayerPosition.DEFENDER,
            max_gameweek=10,
        )
        assert results["history"].sample_size == 9
        assert results["poisson"].sample_size == 9

    def test_brier_score_in_valid_range(self) -> None:
        history = [
            _make_history(gameweek=i, tackles=5, cbi=8)
            for i in range(1, 11)
        ]
        results = evaluate_dc_models_brier(
            history=history,
            position=PlayerPosition.DEFENDER,
            max_gameweek=10,
        )
        for model_result in results.values():
            assert 0.0 <= model_result.brier_score <= 1.0

    def test_calibration_structure(self) -> None:
        history = [
            _make_history(gameweek=i, tackles=5, cbi=8)
            for i in range(1, 30)
        ]
        results = evaluate_dc_models_brier(
            history=history,
            position=PlayerPosition.DEFENDER,
            max_gameweek=29,
        )
        for model_result in results.values():
            for bin_start, (avg_pred, avg_obs) in model_result.calibration.items():
                assert isinstance(bin_start, float)
                assert 0.0 <= avg_pred <= 1.0
                assert 0.0 <= avg_obs <= 1.0

    def test_skips_zero_minute_matches(self) -> None:
        history = [
            _make_history(gameweek=1, tackles=5, cbi=8, minutes=90),
            _make_history(gameweek=2, tackles=0, cbi=0, minutes=0),
            _make_history(gameweek=3, tackles=5, cbi=8, minutes=90),
        ]
        results = evaluate_dc_models_brier(
            history=history,
            position=PlayerPosition.DEFENDER,
            max_gameweek=3,
        )
        assert results["history"].sample_size == 1

    def test_eval_gw_end_capped_at_max_gameweek(self) -> None:
        history = [
            _make_history(gameweek=i, tackles=5, cbi=8)
            for i in range(1, 6)
        ]
        results = evaluate_dc_models_brier(
            history=history,
            position=PlayerPosition.DEFENDER,
            max_gameweek=20,
        )
        assert results["history"].sample_size == 4


# ---------------------------------------------------------------------------
# Integration: full pipeline from history to probability
# ---------------------------------------------------------------------------


class TestIntegration:

    def test_defender_with_strong_history(self) -> None:
        history = [
            _make_history(gameweek=i, tackles=5, cbi=8)
            for i in range(1, 21)
        ]
        result = predict_defensive_contribution_probability(
            history=history,
            position=PlayerPosition.DEFENDER,
            target_gameweek=21,
        )
        assert result.probability > 0.8

    def test_forward_with_poor_history(self) -> None:
        history = [
            _make_history(gameweek=i, tackles=1, cbi=1, recoveries=1)
            for i in range(1, 21)
        ]
        result = predict_defensive_contribution_probability(
            history=history,
            position=PlayerPosition.FORWARD,
            target_gameweek=21,
        )
        assert result.probability < 0.1

    def test_probability_in_unit_interval(self) -> None:
        history = [
            _make_history(gameweek=i, tackles=3, cbi=3, recoveries=3)
            for i in range(1, 11)
        ]
        for pos in [
            PlayerPosition.DEFENDER,
            PlayerPosition.MIDFIELDER,
            PlayerPosition.FORWARD,
        ]:
            result = predict_defensive_contribution_probability(
                history=history,
                position=pos,
                target_gameweek=11,
            )
            assert 0.0 <= result.probability <= 1.0

    def test_dc_default_minutes_constant(self) -> None:
        assert DC_DEFAULT_MINUTES == 90.0

    def test_dc_position_priors_coverage(self) -> None:
        for pos in [
            PlayerPosition.DEFENDER,
            PlayerPosition.MIDFIELDER,
            PlayerPosition.FORWARD,
        ]:
            assert pos in DC_POSITION_PRIOR
            assert DC_POSITION_PRIOR[pos] > 0.0
        assert DC_POSITION_PRIOR[PlayerPosition.GOALKEEPER] == 0.0

    def test_dc_thresholds_constants(self) -> None:
        assert DEF_DC_THRESHOLD == 10
        assert MID_FWD_DC_THRESHOLD == 12


# ---------------------------------------------------------------------------
# Poisson numerical stability and correctness
# ---------------------------------------------------------------------------


def _direct_poisson_geq(k: int, lam: float, max_terms: int = 5000) -> float:
    """Compute P(X >= k) for Poisson(lam) via direct PMF summation.

    Independent reference implementation for validating the gamma-based
    implementation in the module under test.
    """
    if lam <= 0.0:
        return 1.0 if k <= 0 else 0.0
    total = 0.0
    for j in range(k, k + max_terms):
        log_pmf = -lam + j * math.log(lam) - math.lgamma(j + 1)
        pmf = math.exp(log_pmf)
        total += pmf
        if pmf < 1e-20:
            break
    return min(1.0, max(0.0, total))


class TestPoissonNumerical:
    """Numerical stability and correctness of the Poisson implementation."""

    @pytest.mark.parametrize(
        "k, lam",
        [
            (10, 100),
            (10, 1_000),
            (10, 10_000),
            (10, 100_000),
            (1, 1_000_000),
            (5, 1e6),
        ],
    )
    def test_large_lambda_returns_one(self, k: int, lam: float) -> None:
        """When λ >> k, P(X >= k) should be exactly 1.0 (no overflow)."""
        result = _poisson_geq(k, lam)
        assert result == 1.0
        assert math.isfinite(result)

    @pytest.mark.parametrize("k", [10, 12, 20])
    @pytest.mark.parametrize("lam", [0.001, 0.01, 0.1])
    def test_very_small_lambda_near_zero(self, k: int, lam: float) -> None:
        """When λ << k, P(X >= k) is extremely small (effectively zero)."""
        result = _poisson_geq(k, lam)
        assert result < 1e-10
        assert math.isfinite(result)
        assert 0.0 <= result <= 1.0

    def test_agreement_with_direct_pmf_summation(self) -> None:
        """Compare _poisson_geq against independent direct PMF summation."""
        test_cases = [
            (10, 2.5),
            (10, 10.0),
            (10, 12.0),
            (12, 12.0),
            (12, 5.0),
            (5, 3.0),
            (3, 8.0),
        ]
        for k, lam in test_cases:
            direct = _direct_poisson_geq(k, lam)
            gamma_impl = _poisson_geq(k, lam)
            assert gamma_impl == pytest.approx(direct, abs=1e-10), (
                f"Mismatch at k={k}, lam={lam}: "
                f"gamma={gamma_impl}, direct={direct}"
            )

    def test_zero_lambda_edge_case(self) -> None:
        """λ=0 means no events can ever occur."""
        assert _poisson_geq(5, 0.0) == 0.0
        assert _poisson_geq(0, 0.0) == 1.0

    def test_negative_lambda_raises(self) -> None:
        with pytest.raises(ValueError, match="lam must be non-negative"):
            _poisson_geq(10, -1.0)
        with pytest.raises(ValueError, match="lam must be non-negative"):
            _poisson_sf(10, -0.1)

    def test_survival_function_boundary(self) -> None:
        """P(X > 0) = 1 - P(X = 0) = 1 - exp(-λ)."""
        for lam in [0.5, 1.0, 5.0, 10.0]:
            expected = 1.0 - math.exp(-lam)
            assert _poisson_sf(0, lam) == pytest.approx(expected, abs=1e-10)

    @pytest.mark.parametrize(
        "k, lam",
        [
            (1, 0.5),
            (5, 2.5),
            (10, 5.0),
            (12, 10.0),
            (15, 20.0),
            (1, 1.0),
            (3, 0.1),
            (20, 100.0),
            (1, 1000.0),
        ],
    )
    def test_all_probabilities_in_unit_interval(self, k: int, lam: float) -> None:
        """All Poisson outputs must be finite and in [0, 1]."""
        result = _poisson_geq(k, lam)
        assert math.isfinite(result)
        assert 0.0 <= result <= 1.0

    def test_monotonic_decrease_in_k(self) -> None:
        """P(X >= k) must decrease as k increases (for fixed λ)."""
        lam = 5.0
        prev = _poisson_geq(1, lam)
        for k in range(2, 30):
            curr = _poisson_geq(k, lam)
            assert curr <= prev + 1e-15
            prev = curr


class TestShrunkBoundary:
    """Exact boundary behavior of the shrunk flag in DCPredictionResult."""

    def test_shrunk_true_when_total_below_pseudo_count(self) -> None:
        """shrunk=True when n_observations < pseudo_count."""
        history = [_make_history(gameweek=1, tackles=5, cbi=8)]
        result = historical_dc_probability(
            history, PlayerPosition.MIDFIELDER, target_gameweek=5,
            pseudo_count=10.0,
        )
        assert result.shrunk is True
        assert result.n_observations == 1

    def test_shrunk_false_when_total_at_pseudo_count(self) -> None:
        """shrunk=False when n_observations >= pseudo_count (exact boundary)."""
        history = [
            _make_history(gameweek=i, tackles=5, cbi=8) for i in range(1, 6)
        ]
        result = historical_dc_probability(
            history, PlayerPosition.MIDFIELDER, target_gameweek=6,
            pseudo_count=5.0,
        )
        assert result.shrunk is False
        assert result.n_observations == 5

    def test_shrunk_false_when_total_exceeds_pseudo_count(self) -> None:
        history = [
            _make_history(gameweek=i, tackles=5, cbi=8) for i in range(1, 11)
        ]
        result = historical_dc_probability(
            history, PlayerPosition.MIDFIELDER, target_gameweek=11,
            pseudo_count=5.0,
        )
        assert result.shrunk is False
        assert result.n_observations == 10

    def test_shrunk_false_for_gkp_historical(self) -> None:
        result = historical_dc_probability(
            [], PlayerPosition.GOALKEEPER, target_gameweek=5
        )
        assert result.shrunk is False
        assert result.n_observations == 0
        assert result.probability == 0.0

    def test_shrunk_false_for_gkp_poisson(self) -> None:
        result = poisson_dc_probability(
            [], PlayerPosition.GOALKEEPER, target_gameweek=5
        )
        assert result.shrunk is False
        assert result.n_observations == 0
        assert result.probability == 0.0

    def test_shrunk_true_for_prior_fallback_no_history(self) -> None:
        """When no history is supplied, the prior is returned with shrunk=True."""
        result = historical_dc_probability(
            [], PlayerPosition.DEFENDER, target_gameweek=5
        )
        assert result.shrunk is True
        assert result.n_observations == 0
        assert result.probability == pytest.approx(
            DC_POSITION_PRIOR[PlayerPosition.DEFENDER]
        )


# ---------------------------------------------------------------------------
# Poisson prior-rate calculation
# ---------------------------------------------------------------------------


class TestPoissonPriorRate:
    """Verify the Poisson prior-rate formula: lambda_prior = prior * threshold."""

    def test_prior_rate_defender(self) -> None:
        """lambda_prior for DEF = prior * threshold = 0.35 * 10 = 3.5."""
        prior = DC_POSITION_PRIOR[PlayerPosition.DEFENDER]
        threshold = DEF_DC_THRESHOLD
        expected_lambda_prior = prior * threshold
        assert expected_lambda_prior == pytest.approx(3.5)

    def test_prior_rate_midfielder(self) -> None:
        """lambda_prior for MID = prior * threshold = 0.25 * 12 = 3.0."""
        prior = DC_POSITION_PRIOR[PlayerPosition.MIDFIELDER]
        expected_lambda_prior = prior * MID_FWD_DC_THRESHOLD
        assert expected_lambda_prior == pytest.approx(3.0)

    def test_prior_rate_forward(self) -> None:
        """lambda_prior for FWD = prior * threshold = 0.15 * 12 = 1.8."""
        prior = DC_POSITION_PRIOR[PlayerPosition.FORWARD]
        expected_lambda_prior = prior * MID_FWD_DC_THRESHOLD
        assert expected_lambda_prior == pytest.approx(1.8)

    def test_poisson_matches_prior_rate_when_obs_matches(self) -> None:
        """When observed rate equals prior rate, lambda_shrunk == prior * threshold.

        Construct history so that lambda_obs = prior * threshold, then
        verify the resulting probability equals _poisson_geq(threshold,
        prior * threshold), confirming the prior-rate formula is applied.
        """
        prior = DC_POSITION_PRIOR[PlayerPosition.DEFENDER]
        threshold = DEF_DC_THRESHOLD
        lambda_prior = prior * threshold  # 3.5

        # Need lambda_obs = (total_actions / total_minutes) * expected_minutes = 3.5
        # Using 10 matches of 90 min each (total_minutes=900) and expected_minutes=90:
        #   total_actions = 3.5 * 900 / 90 = 35
        # 5 matches with 4 tackles + 5 with 3 tackles = 35.
        history: list[PlayerMatchHistory] = []
        for i in range(1, 11):
            tackles = 4 if i <= 5 else 3
            history.append(
                _make_history(gameweek=i, tackles=tackles, cbi=0, minutes=90)
            )

        result = poisson_dc_probability(
            history, PlayerPosition.DEFENDER, target_gameweek=11,
            expected_minutes=DC_DEFAULT_MINUTES,
        )

        # lambda_obs = (35 / 900) * 90 = 3.5 = lambda_prior
        # lambda_shrunk = (3.5 + 3.5 * 5.0) / (1 + 5.0) = 3.5 * (1 + 5) / (1 + 5) = 3.5
        expected_prob = _poisson_geq(threshold, lambda_prior)
        assert result.probability == pytest.approx(expected_prob, abs=1e-10)
        assert result.n_observations == 10


# ---------------------------------------------------------------------------
# Probability bounds and finite outputs across valid input ranges
# ---------------------------------------------------------------------------


class TestProbabilityBounds:
    """Verify probability bounds and finite outputs across valid input ranges."""

    @pytest.mark.parametrize("minutes", [0, 1, 45, 60, 90, 120])
    def test_historical_probabilities_in_range(self, minutes: int) -> None:
        history = [
            _make_history(
                gameweek=i, tackles=3, cbi=5, recoveries=2, minutes=minutes
            )
            for i in range(1, 11)
        ]
        for pos in [
            PlayerPosition.DEFENDER,
            PlayerPosition.MIDFIELDER,
            PlayerPosition.FORWARD,
        ]:
            result = historical_dc_probability(
                history, pos, target_gameweek=11
            )
            assert math.isfinite(result.probability)
            assert 0.0 <= result.probability <= 1.0

    @pytest.mark.parametrize("minutes", [0, 1, 45, 60, 90, 120])
    def test_poisson_probabilities_in_range(self, minutes: int) -> None:
        history = [
            _make_history(
                gameweek=i, tackles=3, cbi=5, recoveries=2, minutes=minutes
            )
            for i in range(1, 11)
        ]
        for pos in [
            PlayerPosition.DEFENDER,
            PlayerPosition.MIDFIELDER,
            PlayerPosition.FORWARD,
        ]:
            result = poisson_dc_probability(
                history, pos, target_gameweek=11
            )
            assert math.isfinite(result.probability)
            assert 0.0 <= result.probability <= 1.0

    @pytest.mark.parametrize("exp_minutes", [0, 1, 30, 45, 60, 90, 120])
    def test_poisson_expected_minutes_variations(self, exp_minutes: int) -> None:
        history = [
            _make_history(gameweek=i, tackles=3, cbi=5, recoveries=2)
            for i in range(1, 11)
        ]
        result = poisson_dc_probability(
            history, PlayerPosition.DEFENDER, target_gameweek=11,
            expected_minutes=exp_minutes,
        )
        assert math.isfinite(result.probability)
        assert 0.0 <= result.probability <= 1.0
# ---------------------------------------------------------------------------
# Duplicate observation detection
# ---------------------------------------------------------------------------


class TestDuplicateDetection:
    """Verify that duplicate player-match records are detected and rejected."""

    def test_duplicate_fixture_gameweek_raises(self) -> None:
        """Same (fixture_id, gameweek) appears twice -> DCModelError."""
        dup = _make_history(gameweek=3, fixture_id=100, tackles=5, cbi=8)
        history = [dup, dup]
        with pytest.raises(DCModelError, match="Duplicate player-match"):
            historical_dc_probability(
                history, PlayerPosition.DEFENDER, target_gameweek=4
            )

    def test_duplicate_poisson_model_also_raises(self) -> None:
        """The dedup check covers the Poisson model path too."""
        dup = _make_history(gameweek=3, fixture_id=100, tackles=5, cbi=8)
        history = [dup, dup]
        with pytest.raises(DCModelError, match="Duplicate player-match"):
            poisson_dc_probability(
                history, PlayerPosition.DEFENDER, target_gameweek=4
            )

    def test_duplicate_across_gameweeks_ok(self) -> None:
        """Same fixture_id in different gameweeks is not a duplicate."""
        history = [
            _make_history(gameweek=1, fixture_id=100, tackles=5, cbi=8),
            _make_history(gameweek=2, fixture_id=100, tackles=5, cbi=8),
        ]
        result = historical_dc_probability(
            history, PlayerPosition.DEFENDER, target_gameweek=3
        )
        assert result.n_observations == 2

    def test_distinct_fixtures_same_gw_ok(self) -> None:
        """Different fixture_ids in the same gameweek is not a duplicate."""
        history = [
            _make_history(gameweek=1, fixture_id=100, tackles=5, cbi=8),
            _make_history(gameweek=1, fixture_id=200, tackles=5, cbi=8),
        ]
        result = historical_dc_probability(
            history, PlayerPosition.DEFENDER, target_gameweek=2
        )
        assert result.n_observations == 2

    def test_zero_minute_duplicates_not_flagged(self) -> None:
        """Zero-minute entries are filtered before dedup check."""
        bench1 = _make_history(gameweek=1, fixture_id=100, minutes=0)
        bench2 = _make_history(gameweek=2, fixture_id=200, minutes=0)
        starter = _make_history(gameweek=3, fixture_id=100, tackles=5, cbi=8)
        result = historical_dc_probability(
            [bench1, bench2, starter],
            PlayerPosition.DEFENDER,
            target_gameweek=4,
        )
        assert result.n_observations == 1

    def test_multiple_duplicates_reported(self) -> None:
        """Multiple unique duplicate pairs produce a clear error message."""
        h1 = _make_history(gameweek=1, fixture_id=100, tackles=3, cbi=8)
        h2 = _make_history(gameweek=2, fixture_id=200, tackles=3, cbi=8)
        history = [h1, h2, h1, h2]
        with pytest.raises(DCModelError, match="fixture=100, gw=1"):
            historical_dc_probability(
                history, PlayerPosition.DEFENDER, target_gameweek=3
            )

    def test_no_error_for_empty_history(self) -> None:
        """Empty history doesn't trigger dedup."""
        historical_dc_probability(
            [], PlayerPosition.DEFENDER, target_gameweek=5
        )
# ---------------------------------------------------------------------------
# Provenance: n_observations signal for from_history flag
# ---------------------------------------------------------------------------


class TestProvenanceBoundary:
    """Verify the n_observations signal distinguishes prior fallback from data.

    The ``defensive_contribution_probability_from_history`` field on
    BaselineFeatures and MatchPrediction is derived from
    ``DCPredictionResult.n_observations > 0``.  These tests verify that
    signal is correct for all model/position branches.
    """

    def test_no_history_zero_observations(self) -> None:
        """No history -> n_observations=0 (prior-only fallback)."""
        result = predict_defensive_contribution_probability(
            history=None,
            position=PlayerPosition.DEFENDER,
            target_gameweek=5,
        )
        assert result.n_observations == 0

    def test_history_present_positive_observations(self) -> None:
        """History present -> n_observations > 0 (data-backed)."""
        history = [
            _make_history(gameweek=i, tackles=5, cbi=8) for i in range(1, 6)
        ]
        result = predict_defensive_contribution_probability(
            history=history,
            position=PlayerPosition.DEFENDER,
            target_gameweek=6,
        )
        assert result.n_observations == 5

    def test_gkp_zero_observations(self) -> None:
        """GKP is ineligible -> n_observations=0, probability=0.0."""
        result = predict_defensive_contribution_probability(
            history=None,
            position=PlayerPosition.GOALKEEPER,
            target_gameweek=5,
        )
        assert result.n_observations == 0
        assert result.probability == 0.0

    def test_empty_history_zero_observations(self) -> None:
        """Empty list -> n_observations=0."""
        result = predict_defensive_contribution_probability(
            history=[],
            position=PlayerPosition.DEFENDER,
            target_gameweek=5,
        )
        assert result.n_observations == 0

    def test_poisson_no_history_zero_observations(self) -> None:
        """Poisson model, no history -> n_observations=0."""
        result = poisson_dc_probability(
            [], PlayerPosition.MIDFIELDER, target_gameweek=5
        )
        assert result.n_observations == 0
        assert result.shrunk is True
        assert result.n_observations == 0


# ---------------------------------------------------------------------------
# Temporal-safety: _appearance_history with prediction_cutoff
# ---------------------------------------------------------------------------


class TestAppearanceHistoryTemporalFiltering:
    """Tests for _appearance_history with the prediction_cutoff parameter."""

    def test_cutoff_excludes_future_outcome_time(self) -> None:
        """Records with outcome_time after cutoff are excluded."""
        cutoff = _utc(2024, 9, 15, 12)
        history = [
            _make_history(
                gameweek=1, outcome_time=_utc(2024, 8, 17), minutes=90,
                tackles=5, cbi=8,
            ),
            # Postponed: gameweek 2 but played on Sep 20 (after cutoff)
            _make_history(
                gameweek=2, outcome_time=_utc(2024, 9, 20), minutes=90,
                tackles=5, cbi=8,
            ),
        ]
        result = _appearance_history(
            history, target_gameweek=3, prediction_cutoff=cutoff,
        )
        assert len(result) == 1
        assert result[0].gameweek == 1

    def test_cutoff_exact_boundary_included(self) -> None:
        """Records with outcome_time == cutoff are included (<=)."""
        cutoff = _utc(2024, 9, 15, 12)
        history = [
            _make_history(
                gameweek=1, outcome_time=cutoff, minutes=90, tackles=5, cbi=8,
            ),
        ]
        result = _appearance_history(
            history, target_gameweek=2, prediction_cutoff=cutoff,
        )
        assert len(result) == 1

    def test_cutoff_excludes_no_outcome_time(self) -> None:
        """Records without outcome_time are excluded when cutoff is provided."""
        cutoff = _utc(2024, 9, 15)
        history = [
            _make_history(gameweek=1, minutes=90, tackles=5, cbi=8),
        ]
        result = _appearance_history(
            history, target_gameweek=2, prediction_cutoff=cutoff,
        )
        assert len(result) == 0

    def test_no_cutoff_keeps_no_outcome_time(self) -> None:
        """Without cutoff, records without outcome_time are included (backward compat)."""
        history = [
            _make_history(gameweek=1, minutes=90, tackles=5, cbi=8),
        ]
        result = _appearance_history(
            history, target_gameweek=2, prediction_cutoff=None,
        )
        assert len(result) == 1

    def test_cutoff_excludes_higher_gameweek_even_with_early_outcome(self) -> None:
        """Gameweek-level filter still applies alongside cutoff filter."""
        cutoff = _utc(2024, 12, 31)
        history = [
            _make_history(
                gameweek=1, outcome_time=_utc(2024, 8, 17), minutes=90,
                tackles=5, cbi=8,
            ),
            _make_history(
                gameweek=3, outcome_time=_utc(2024, 9, 10), minutes=90,
                tackles=5, cbi=8,
            ),
        ]
        # GW3 is excluded by gameweek filter (gw < 3); only GW1 survives
        result = _appearance_history(
            history, target_gameweek=3, prediction_cutoff=cutoff,
        )
        assert len(result) == 1
        assert result[0].gameweek == 1

    def test_cutoff_excludes_target_gw_fixture(self) -> None:
        """The target gameweek's own fixture is excluded by gameweek filter."""
        cutoff = _utc(2024, 12, 31)
        history = [
            _make_history(
                gameweek=3, outcome_time=_utc(2024, 9, 10), minutes=90,
                tackles=5, cbi=8,
            ),
        ]
        result = _appearance_history(
            history, target_gameweek=3, prediction_cutoff=cutoff,
        )
        assert len(result) == 0

    def test_cutoff_excludes_zero_minute_appearances(self) -> None:
        """Zero-minute records are excluded by the appearance filter."""
        cutoff = _utc(2024, 12, 31)
        history = [
            _make_history(
                gameweek=1, outcome_time=_utc(2024, 8, 17), minutes=0,
                tackles=5, cbi=8,
            ),
        ]
        result = _appearance_history(
            history, target_gameweek=2, prediction_cutoff=cutoff,
        )
        assert len(result) == 0

    def test_naive_cutoff_raises(self) -> None:
        """A tz-naive prediction_cutoff raises DCModelError."""
        naive = datetime(2024, 9, 15, 12)  # noqa: DTZ001
        with pytest.raises(DCModelError, match="timezone-aware"):
            _appearance_history([], target_gameweek=2, prediction_cutoff=naive)

    def test_non_utc_cutoff_raises(self) -> None:
        """A non-UTC timezone on prediction_cutoff raises DCModelError."""
        tz_plus5 = timezone(timedelta(hours=5))
        with pytest.raises(DCModelError, match="UTC"):
            _appearance_history(
                [], target_gameweek=2,
                prediction_cutoff=datetime(2024, 9, 15, 12, tzinfo=tz_plus5),
            )

    def test_postponed_fixture_excluded_when_after_cutoff(self) -> None:
        """A fixture whose gameweek label is early but outcome_time is late
        is correctly excluded by the temporal filter."""
        cutoff = _utc(2024, 9, 15, 12)
        # GW1 fixture scheduled for Aug 17 but postponed to Oct 10
        history = [
            _make_history(
                gameweek=1, outcome_time=_utc(2024, 10, 10), minutes=90,
                tackles=5, cbi=8,
            ),
        ]
        result = _appearance_history(
            history, target_gameweek=2, prediction_cutoff=cutoff,
        )
        assert len(result) == 0

    def test_postponed_fixture_included_when_before_cutoff(self) -> None:
        """A fixture whose outcome_time falls before the cutoff is included
        even if its gameweek label seems 'late'."""
        cutoff = _utc(2024, 10, 15, 12)
        # GW1 fixture postponed to Oct 10 — still before our Oct 15 cutoff
        history = [
            _make_history(
                gameweek=1, outcome_time=_utc(2024, 10, 10), minutes=90,
                tackles=5, cbi=8,
            ),
        ]
        result = _appearance_history(
            history, target_gameweek=2, prediction_cutoff=cutoff,
        )
        assert len(result) == 1

    def test_mixed_records_filters_correctly(self) -> None:
        """A mix of eligible and ineligible records yields only the eligible ones."""
        cutoff = _utc(2024, 9, 20, 0)
        history = [
            # Before cutoff, has outcome_time -> included
            _make_history(
                gameweek=1, outcome_time=_utc(2024, 8, 17), minutes=90,
                tackles=5, cbi=8,
            ),
            # After cutoff -> excluded
            _make_history(
                gameweek=2, outcome_time=_utc(2024, 9, 25), minutes=90,
                tackles=5, cbi=8,
            ),
            # Before cutoff but no outcome_time -> excluded
            _make_history(
                gameweek=3, outcome_time=None, minutes=90,
                tackles=5, cbi=8,
            ),
            # After cutoff -> excluded
            _make_history(
                gameweek=4, outcome_time=_utc(2024, 10, 5), minutes=90,
                tackles=5, cbi=8,
            ),
        ]
        result = _appearance_history(
            history, target_gameweek=5, prediction_cutoff=cutoff,
        )
        assert len(result) == 1
        assert result[0].gameweek == 1


# ---------------------------------------------------------------------------
# Temporal-safety: public model functions with prediction_cutoff
# ---------------------------------------------------------------------------


class TestDcModelTemporalSafety:
    """Tests for prediction_cutoff threading through public DC model functions."""

    def test_historical_dc_probability_threads_cutoff(self) -> None:
        """predict_cutoff is forwarded to _appearance_history in the
        historical-frequency model."""
        cutoff = _utc(2024, 9, 15)
        history = [
            _make_history(
                gameweek=1, outcome_time=_utc(2024, 8, 17), minutes=90,
                tackles=5, cbi=8,
            ),
            # After cutoff -> excluded
            _make_history(
                gameweek=2, outcome_time=_utc(2024, 9, 20), minutes=90,
                tackles=5, cbi=8,
            ),
        ]
        result_with_cutoff = historical_dc_probability(
            history, PlayerPosition.DEFENDER, target_gameweek=3,
            prediction_cutoff=cutoff,
        )
        result_without_cutoff = historical_dc_probability(
            history, PlayerPosition.DEFENDER, target_gameweek=3,
        )
        # With cutoff: only GW1 -> 1 observation
        assert result_with_cutoff.n_observations == 1
        # Without cutoff: both GWs -> 2 observations
        assert result_without_cutoff.n_observations == 2

    def test_poisson_dc_probability_threads_cutoff(self) -> None:
        """predict_cutoff is forwarded to _appearance_history in the Poisson
        model."""
        cutoff = _utc(2024, 9, 15)
        history = [
            _make_history(
                gameweek=1, outcome_time=_utc(2024, 8, 17), minutes=90,
                tackles=5, cbi=8, recoveries=7,
            ),
            _make_history(
                gameweek=2, outcome_time=_utc(2024, 9, 20), minutes=90,
                tackles=5, cbi=8, recoveries=7,
            ),
        ]
        result = poisson_dc_probability(
            history, PlayerPosition.MIDFIELDER, target_gameweek=3,
            prediction_cutoff=cutoff,
        )
        assert result.n_observations == 1

    def test_predict_dispatch_threads_cutoff(self) -> None:
        """predict_defensive_contribution_probability forwards cutoff to both
        model back-ends."""
        cutoff = _utc(2024, 9, 15)
        history = [
            _make_history(
                gameweek=1, outcome_time=_utc(2024, 8, 17), minutes=90,
                tackles=5, cbi=8, recoveries=7,
            ),
            _make_history(
                gameweek=2, outcome_time=_utc(2024, 9, 20), minutes=90,
                tackles=5, cbi=8, recoveries=7,
            ),
        ]
        # History model
        hist_result = predict_defensive_contribution_probability(
            history=history, position=PlayerPosition.MIDFIELDER,
            target_gameweek=3, prediction_cutoff=cutoff,
        )
        assert hist_result.n_observations == 1
        # Poisson model
        pois_result = predict_defensive_contribution_probability(
            history=history, position=PlayerPosition.MIDFIELDER,
            target_gameweek=3, model="poisson", prediction_cutoff=cutoff,
        )
        assert pois_result.n_observations == 1

    def test_naive_cutoff_raises_through_predict(self) -> None:
        """A naive cutoff raises when dispatched through the public predict API."""
        naive = datetime(2024, 9, 15, 12)  # noqa: DTZ001
        with pytest.raises(DCModelError, match="timezone-aware"):
            predict_defensive_contribution_probability(
                history=[], position=PlayerPosition.DEFENDER,
                target_gameweek=2, prediction_cutoff=naive,
            )


# ---------------------------------------------------------------------------
# Temporal-safety: evaluate_dc_models_brier with outcome_time
# ---------------------------------------------------------------------------


class TestEvaluateBrierTemporalSafety:
    """Tests for outcome_time-based filtering inside evaluate_dc_models_brier."""

    def test_postponed_fixture_excluded_from_lookback(self) -> None:
        """When the target fixture's outcome_time is known, prior fixtures
        whose outcome_time is later are excluded from the look-back.

        This test verifies that the brier_score differs between the
        temporally-filtered case (with outcome_time) and the unfiltered
        backward-compat case (without outcome_time).  The *sample_size*
        (number of evaluated gameweeks) is the same in both — the filtering
        changes which observations feed each prediction, not how many
        gameweeks are evaluated.

        Scenario: GW3 fixture was postponed to Oct 10, while GW4 was
        played Sep 10.  When evaluating GW4, GW3 (gw 3 < 4) would normally
        enter the look-back, but its outcome_time (Oct 10) exceeds GW4's
        outcome_time (Sep 10), so it is excluded.
        """
        # Build history with outcome_time (temporally filtered)
        history_filtered = [
            _make_history(
                gameweek=1, outcome_time=_utc(2024, 8, 17), minutes=90,
                tackles=5, cbi=8,
            ),
            _make_history(
                gameweek=2, outcome_time=_utc(2024, 8, 24), minutes=90,
                tackles=5, cbi=8,
            ),
            # GW3 — postponed to Oct 10
            _make_history(
                gameweek=3, outcome_time=_utc(2024, 10, 10), minutes=90,
                tackles=5, cbi=8,
            ),
            # GW4 — Sep 10 (before GW3's postponed date)
            _make_history(
                gameweek=4, outcome_time=_utc(2024, 9, 10), minutes=90,
                tackles=5, cbi=8,
            ),
        ]

        # Same history without outcome_time (backward-compat, no filtering)
        history_unfiltered = [
            _make_history(
                gameweek=h.gameweek, minutes=h.minutes,
                tackles=h.tackles, cbi=h.clearances_blocks_interceptions,
            )
            for h in history_filtered
        ]

        results_filtered = evaluate_dc_models_brier(
            history=history_filtered, position=PlayerPosition.DEFENDER,
            max_gameweek=10,
        )
        results_unfiltered = evaluate_dc_models_brier(
            history=history_unfiltered, position=PlayerPosition.DEFENDER,
            max_gameweek=10,
        )

        # Same number of evaluated gameweeks (GW2-GW4 = 3 predictions)
        assert results_filtered["history"].sample_size == 3
        assert results_unfiltered["history"].sample_size == 3

        # But the brier_score must differ because GW3 is excluded from
        # GW4's look-back only in the filtered case.
        assert results_filtered["history"].brier_score != pytest.approx(
            results_unfiltered["history"].brier_score, abs=1e-6
        )

    def test_no_outcome_time_backward_compat(self) -> None:
        """When no outcome_time is set, behavior matches pre-change filtering."""
        history = [
            _make_history(gameweek=i, tackles=5, cbi=8)
            for i in range(1, 11)
        ]
        results = evaluate_dc_models_brier(
            history=history, position=PlayerPosition.DEFENDER, max_gameweek=10,
        )
        # 9 predictions (GW2 through GW10)
        assert results["history"].sample_size == 9

    def test_all_fixtures_chronological(self) -> None:
        """With outcome_time on all records in chronological order,
        temporal filtering does not exclude any fixtures."""
        history = [
            _make_history(
                gameweek=i,
                outcome_time=_utc(2024, 8, 17) + timedelta(days=i),
                minutes=90, tackles=5, cbi=8,
            )
            for i in range(1, 11)
        ]
        results = evaluate_dc_models_brier(
            history=history, position=PlayerPosition.DEFENDER, max_gameweek=10,
        )
        # All matches are in chronological order by outcome_time
        # so no fixtures are excluded → 9 obs
        assert results["history"].sample_size == 9

    def test_partial_outcome_time_uses_what_is_available(self) -> None:
        """When some records have outcome_time and others don't,
        filtering applies only when the target has outcome_time.

        If the target fixture (gw_match) has outcome_time=None, no
        temporal filtering is applied at that step — backward-compatible
        behavior.
        """
        history = [
            # GW1 — has outcome_time
            _make_history(
                gameweek=1, outcome_time=_utc(2024, 8, 17), minutes=90,
                tackles=5, cbi=8,
            ),
            # GW2 — no outcome_time (target fixture)
            _make_history(
                gameweek=2, outcome_time=None, minutes=90,
                tackles=5, cbi=8,
            ),
            # GW3 — has outcome_time (target fixture)
            _make_history(
                gameweek=3, outcome_time=_utc(2024, 9, 10), minutes=90,
                tackles=5, cbi=8,
            ),
        ]
        results = evaluate_dc_models_brier(
            history=history, position=PlayerPosition.DEFENDER, max_gameweek=10,
        )
        # GW2: target has no outcome_time → no temporal filter → prior=[GW1] (1 obs)
        # GW3: target has outcome_time Sep10 → prior=[GW1] (Aug17 < Sep10, 1 obs)
        # Total = 1 + 1 = 2
        assert results["history"].sample_size == 2


# ---------------------------------------------------------------------------
# prediction_cutoff / temporal_safety_enforced tests
# ---------------------------------------------------------------------------


class TestPredictionCutoff:
    """Tests for the ``prediction_cutoff`` parameter and ``temporal_safety_enforced`` flag."""

    def test_temporal_safety_enforced_true_with_explicit_cutoff(self) -> None:
        """When prediction_cutoff is provided, temporal_safety_enforced is True."""
        history = [
            _make_history(gameweek=i, outcome_time=_utc(2024, 8, i, 14), tackles=5, cbi=8)
            for i in range(1, 6)
        ]
        cutoff = _utc(2024, 8, 17, 14)
        results = evaluate_dc_models_brier(
            history=history,
            position=PlayerPosition.DEFENDER,
            max_gameweek=5,
            prediction_cutoff=cutoff,
        )
        for model_result in results.values():
            assert model_result.temporal_safety_enforced is True

    def test_temporal_safety_enforced_false_without_cutoff_or_times(self) -> None:
        """Without prediction_cutoff or outcome_times, temporal_safety_enforced is False."""
        history = [
            _make_history(gameweek=i, outcome_time=None, tackles=5, cbi=8)
            for i in range(1, 6)
        ]
        results = evaluate_dc_models_brier(
            history=history,
            position=PlayerPosition.DEFENDER,
            max_gameweek=5,
        )
        for model_result in results.values():
            assert model_result.temporal_safety_enforced is False

    def test_temporal_safety_enforced_true_with_kickoff_time(self) -> None:
        """When outcome_time is None but kickoff_time is present, temporal safety applies."""
        history = [
            _make_history(
                gameweek=i,
                outcome_time=None,
                kickoff_time=_utc(2024, 8, i, 14),
                tackles=5, cbi=8,
            )
            for i in range(1, 6)
        ]
        results = evaluate_dc_models_brier(
            history=history,
            position=PlayerPosition.DEFENDER,
            max_gameweek=5,
        )
        for model_result in results.values():
            assert model_result.temporal_safety_enforced is True

    def test_naive_prediction_cutoff_raises(self) -> None:
        """A naive (tz-unaware) prediction_cutoff raises DCModelError."""
        history = [
            _make_history(gameweek=1, outcome_time=None, tackles=5, cbi=8),
        ]
        with pytest.raises(DCModelError, match="timezone-aware"):
            evaluate_dc_models_brier(
                history=history,
                position=PlayerPosition.DEFENDER,
                prediction_cutoff=datetime(2024, 8, 17),  # noqa: DTZ001
            )

    def test_non_utc_prediction_cutoff_raises(self) -> None:
        """A non-UTC timezone prediction_cutoff raises DCModelError."""
        tz_plus5 = timezone(timedelta(hours=5))
        history = [
            _make_history(gameweek=1, outcome_time=None, tackles=5, cbi=8),
        ]
        with pytest.raises(DCModelError, match="must be in UTC"):
            evaluate_dc_models_brier(
                history=history,
                position=PlayerPosition.DEFENDER,
                prediction_cutoff=datetime(2024, 8, 17, tzinfo=tz_plus5),
            )

    def test_explicit_cutoff_filters_prior_observations(self) -> None:
        """Observations with outcome_time before cutoff are included; at-or-after excluded."""
        history = [
            _make_history(
                gameweek=1,
                outcome_time=_utc(2024, 8, 10, 14),
                kickoff_time=_utc(2024, 8, 10, 12),
                tackles=5, cbi=8,
            ),
            _make_history(
                gameweek=2,
                outcome_time=_utc(2024, 8, 20, 14),
                kickoff_time=_utc(2024, 8, 20, 12),
                tackles=5, cbi=8,
            ),
            _make_history(
                gameweek=3,
                outcome_time=_utc(2024, 8, 25, 14),  # after cutoff
                kickoff_time=_utc(2024, 8, 25, 12),
                tackles=5, cbi=8,
            ),
            _make_history(
                gameweek=4,
                outcome_time=_utc(2024, 9, 1, 14),
                kickoff_time=_utc(2024, 9, 1, 12),
                tackles=5, cbi=8,
            ),
        ]
        # Cutoff between GW2 and GW3 outcomes
        cutoff = _utc(2024, 8, 22, 0)

        results = evaluate_dc_models_brier(
            history=history,
            position=PlayerPosition.DEFENDER,
            max_gameweek=4,
            prediction_cutoff=cutoff,
        )
        # GW2: target, prior=[GW1] (Aug10 < Aug22 cutoff) -> 1 training obs
        # GW3: target, prior=[GW1, GW2] (both < Aug22 cutoff) -> 2 training obs
        # GW4: target, prior=[GW1, GW2] (GW3 Aug25 >= Aug22, excluded) -> 2 training obs
        # sample_size counts evaluated fixtures (GW2, GW3, GW4) = 3
        assert results["history"].sample_size == 3

    def test_explicit_cutoff_excludes_at_boundary(self) -> None:
        """Observation exactly at the cutoff is excluded from training (strict <)."""
        boundary = _utc(2024, 8, 10, 14)
        history = [
            _make_history(
                gameweek=1,
                outcome_time=boundary,  # exactly at cutoff -> excluded from training
                tackles=5, cbi=8,
            ),
            _make_history(
                gameweek=2,  # target
                outcome_time=_utc(2024, 8, 20, 14),
                tackles=5, cbi=8,
            ),
        ]
        results = evaluate_dc_models_brier(
            history=history,
            position=PlayerPosition.DEFENDER,
            max_gameweek=2,
            prediction_cutoff=boundary,
        )
        # GW2 is the target: prior=[GW1] but GW1's outcome_time == cutoff -> excluded
        # Model falls back to prior (no training data), still makes a prediction.
        # sample_size counts evaluated fixtures, not training observations.
        # So sample_size == 1, but the prediction uses only the prior.
        assert results["history"].sample_size == 1
        # With 0 training observations, the probability should equal the prior
        # (no contribution from historical data).
        prior = DC_POSITION_PRIOR[PlayerPosition.DEFENDER]
        assert results["history"].brier_score == pytest.approx((prior - 1) ** 2)

    def test_gkp_temporal_safety_with_cutoff(self) -> None:
        """GKP results still report temporal_safety_enforced when cutoff is given."""
        results = evaluate_dc_models_brier(
            history=[],
            position=PlayerPosition.GOALKEEPER,
            prediction_cutoff=_utc(2024, 8, 17),
        )
        for model_result in results.values():
            assert model_result.temporal_safety_enforced is True
