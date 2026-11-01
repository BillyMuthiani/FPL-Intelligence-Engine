"""Analytics package — FPL scoring and point calculation.

Public API:
    Stage 1 — Scoring:
        - :class:`FPLScoringRules` — immutable configuration of 2026/27 scoring rules
        - :data:`DEFAULT_SCORING_RULES` — module-level singleton of the 2026/27 rules
        - :class:`MatchStats` — immutable input statistics for a single match
        - :class:`ScoringDecomposition` — immutable breakdown of points by category
        - :func:`calculate_fpl_points` — pure function computing points for a single match
        - :class:`ScoringError` — raised on invalid or inconsistent inputs

    Stage 2 — Player metrics:
        - :class:`FixtureGameweekRecord` — per-fixture performance snapshot
        - :class:`GameweekStats` — aggregated gameweek-level statistics
        - :class:`PlayerMetrics` — comprehensive descriptive metrics for a player
        - :func:`calculate_player_metrics` — pure function producing PlayerMetrics
        - :func:`points_per_90` — generic per-90 rate helper
        - :func:`points_per_million` — points per million £ helper
        - :func:`clean_sheet_rate` — clean sheet rate helper
        - :func:`points_std_dev` — population standard deviation helper
        - :func:`points_last_n_average` — rolling average form helper
        - :class:`MetricsError` — raised on invalid metrics input

    Stage 4 — Prediction:
        - :class:`MatchPrediction` — probabilistic prediction for a single fixture
        - :class:`PlayerGameweekPrediction` — aggregated fixtures prediction for a GW
        - :class:`ExpectedPointsResult` — expected FPL points with component breakdown
        - :func:`predict_expected_points` — pure function converting predictions to expected points
        - :class:`PredictionError` — raised on invalid prediction inputs

    Stage 4B — Baseline models:
        - :class:`PositionBenchmark` — per-position per-90 statistical benchmarks
        - :class:`BaselineFeatures` — extracted features for a player-fixture pair
        - :func:`compute_position_benchmarks` — aggregate population statistics by position
        - :func:`extract_baseline_features` — derive baseline features from raw FPL data
        - :func:`predict_match_baseline` — convert features into a MatchPrediction
        - :class:`BaselineModelError` — raised on invalid baseline model inputs

    The naming convention for the default singleton is ``DEFAULT_SCORING_RULES``
    (uppercase ``CONSTANT_NAME``), following PEP 8 for module-level constants
    and matching the existing pattern in :mod:`fpl_engine.storage.models`
    (``SUPPORTED_SCHEMA_VERSIONS``).
"""

from __future__ import annotations

from .baseline import (
    PlayerBaseline,
    basic_expected_points,
    compare_players,
    rank_players_by_consistency,
    rank_players_by_form,
    rank_players_by_value,
)
from .baseline_models import (
    BaselineFeatures,
    BaselineModelError,
    PositionBenchmark,
    compute_position_benchmarks,
    extract_baseline_features,
    predict_match_baseline,
)
from .defensive_probability import (
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
    dc_action_count,
    evaluate_dc_models_brier,
    historical_dc_probability,
    poisson_dc_probability,
    predict_defensive_contribution_probability,
)
from .evaluation import (
    BrierMetrics,
    EvaluationError,
    EvaluationExample,
    EvaluationSummary,
    ExpectedPointsMetrics,
    OutcomeRecord,
    PredictionRecord,
    align_predictions_with_outcomes,
    compute_brier_score,
    compute_expected_points_metrics,
    evaluate_dc_models_from_history,
    evaluate_predictions,
    outcome_from_history,
)
from .metrics import (
    FixtureGameweekRecord,
    GameweekStats,
    MetricsError,
    PlayerMetrics,
    calculate_player_metrics,
    clean_sheet_rate,
    points_last_n_average,
    points_per_90,
    points_per_million,
    points_std_dev,
)
from .prediction import (
    ExpectedPointsResult,
    MatchPrediction,
    PlayerGameweekPrediction,
    PredictionError,
    predict_expected_points,
)
from .scoring import (
    DEFAULT_SCORING_RULES,
    FPLScoringRules,
    MatchStats,
    ScoringDecomposition,
    ScoringError,
    calculate_fpl_points,
)

__all__ = [
    "DC_DEFAULT_MINUTES",
    "DC_MODEL_HISTORY",
    "DC_MODEL_POISSON",
    "DC_POSITION_PRIOR",
    "DC_PRIOR_PSEUDO_COUNT",
    "DEFAULT_SCORING_RULES",
    "DEF_DC_THRESHOLD",
    "MID_FWD_DC_THRESHOLD",
    "BaselineFeatures",
    "BaselineModelError",
    "BrierMetrics",
    "DCEvaluationResult",
    "DCModelError",
    "DCPredictionResult",
    "EvaluationError",
    "EvaluationExample",
    "EvaluationSummary",
    "ExpectedPointsMetrics",
    "ExpectedPointsResult",
    "FPLScoringRules",
    "FixtureGameweekRecord",
    "GameweekStats",
    "MatchPrediction",
    "MatchStats",
    "MetricsError",
    "OutcomeRecord",
    "PlayerBaseline",
    "PlayerGameweekPrediction",
    "PlayerMetrics",
    "PositionBenchmark",
    "PredictionError",
    "PredictionRecord",
    "ScoringDecomposition",
    "ScoringError",
    "align_predictions_with_outcomes",
    "basic_expected_points",
    "calculate_fpl_points",
    "calculate_player_metrics",
    "clean_sheet_rate",
    "compare_players",
    "compute_brier_score",
    "compute_expected_points_metrics",
    "compute_position_benchmarks",
    "dc_action_count",
    "defensive_contribution_from_history",
    "evaluate_dc_models_brier",
    "evaluate_dc_models_from_history",
    "evaluate_predictions",
    "extract_baseline_features",
    "historical_dc_probability",
    "outcome_from_history",
    "points_last_n_average",
    "points_per_90",
    "points_per_million",
    "points_std_dev",
    "poisson_dc_probability",
    "predict_defensive_contribution_probability",
    "predict_expected_points",
    "predict_match_baseline",
    "rank_players_by_consistency",
    "rank_players_by_form",
    "rank_players_by_value",
]
