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
from .scoring import (
    DEFAULT_SCORING_RULES,
    FPLScoringRules,
    MatchStats,
    ScoringDecomposition,
    ScoringError,
    calculate_fpl_points,
)

__all__ = [
    "DEFAULT_SCORING_RULES",
    # Stage 1: Scoring
    "FPLScoringRules",
    # Stage 2: Player metrics
    "FixtureGameweekRecord",
    "GameweekStats",
    "MatchStats",
    "MetricsError",
    # Stage 3: Baseline analytics
    "PlayerBaseline",
    "PlayerMetrics",
    "ScoringDecomposition",
    "ScoringError",
    "basic_expected_points",
    "calculate_fpl_points",
    "calculate_player_metrics",
    "clean_sheet_rate",
    "compare_players",
    "points_last_n_average",
    "points_per_90",
    "points_per_million",
    "points_std_dev",
    "rank_players_by_consistency",
    "rank_players_by_form",
    "rank_players_by_value",
]
