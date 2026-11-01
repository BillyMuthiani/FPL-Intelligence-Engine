"""Tests for baseline_models module — Stage 4B baseline prediction features.

Covers:
    - Private helpers: _to_per_90, _shrink_rate, _compute_start_probability,
      _resolve_expected_minutes, _heuristic_60plus_factor, _safe_mean,
      _project_goals_conceded_rate
    - Public API: compute_position_benchmarks, extract_baseline_features,
      predict_match_baseline
    - Dataclasses: PositionBenchmark, BaselineFeatures
    - Error handling: BaselineModelError
    - Temporal safety: prediction_cutoff filtering in extract_baseline_features
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timezone
from typing import Any

import pytest

from fpl_engine.analytics.baseline_models import (
    LEAGUE_AVG_GOALS_PER_TEAM,
    MAX_BONUS_PER_MATCH,
    MIN_BENCHMARK_MINUTES,
    SHRINKAGE_STRENGTH,
    BaselineFeatures,
    BaselineModelError,
    PositionBenchmark,
    _compute_start_probability,
    _heuristic_60plus_factor,
    _project_goals_conceded_rate,
    _resolve_expected_minutes,
    _safe_mean,
    _shrink_rate,
    _to_per_90,
    compute_position_benchmarks,
    extract_baseline_features,
    predict_match_baseline,
)
from fpl_engine.analytics.prediction import MatchPrediction
from fpl_engine.models import (
    Fixture,
    Player,
    PlayerMatchHistory,
    PlayerPosition,
    PlayerStatus,
    Team,
)

# ---------------------------------------------------------------------------
# Test data factories
# ---------------------------------------------------------------------------


def make_team(
    team_id: int = 14,
    short_name: str = "LIV",
    name: str = "Liverpool",
    strength: int = 5,
    strength_overall_home: int = 5,
    strength_overall_away: int = 4,
    strength_attack_home: int = 5,
    strength_attack_away: int = 4,
    strength_defence_home: int = 4,
    strength_defence_away: int = 4,
    goals_for: int = 70,
    goals_against: int = 30,
    played: int = 30,
) -> Team:
    """Create a Team with realistic FPL strength values (1-5 scale, 3.0 = avg)."""
    return Team(
        id=team_id,
        code=3,
        name=name,
        short_name=short_name,
        strength=strength,
        strength_overall_home=strength_overall_home,
        strength_overall_away=strength_overall_away,
        strength_attack_home=strength_attack_home,
        strength_attack_away=strength_attack_away,
        strength_defence_home=strength_defence_home,
        strength_defence_away=strength_defence_away,
        played=played,
        win=20,
        loss=5,
        draw=5,
        goals_for=goals_for,
        goals_against=goals_against,
    )


def make_fixture(
    fixture_id: int = 1,
    team_h: int = 14,
    team_a: int = 1,
    event: int = 1,
) -> Fixture:
    return Fixture(
        id=fixture_id,
        code=fixture_id * 100,
        event=event,
        team_h=team_h,
        team_a=team_a,
        team_h_difficulty=2,
        team_a_difficulty=4,
        kickoff_time="2024-08-17T14:00:00Z",
    )


def make_player(
    player_id: int = 1,
    team_id: int = 14,
    position: PlayerPosition = PlayerPosition.MIDFIELDER,
    status: PlayerStatus = PlayerStatus.AVAILABLE,
    minutes: int = 2700,
    expected_goals: float | None = 10.0,
    expected_assists: float | None = 8.0,
    expected_goals_conceded: float | None = 30.0,
    expected_goals_per_90: float | None = None,
    expected_assists_per_90: float | None = None,
    expected_goals_conceded_per_90: float | None = None,
    expected_minutes: int | None = 90,
    chance_of_playing_next_round: int | None = 100,
    chance_of_playing_this_round: int | None = None,
    saves: int = 0,
    bonus: int = 50,
    own_goals: int = 0,
    goals_scored: int = 10,
    assists: int = 5,
    yellow_cards: int = 2,
    red_cards: int = 0,
    tackles: int = 30,
    interceptions: int = 20,
    clearances: int = 10,
    blocks: int = 5,
) -> Player:
    """Create a Player with configurable stats for testing."""
    return Player(
        id=player_id,
        web_name="TestPlayer",
        first_name="Test",
        second_name="Player",
        team_id=team_id,
        position=position,
        status=status,
        now_cost=80,
        code=player_id * 1000,
        photo="test.jpg",
        element_type=position.position_id,
        team_code=team_id,
        minutes=minutes,
        expected_goals=expected_goals,
        expected_assists=expected_assists,
        expected_goals_conceded=expected_goals_conceded,
        expected_goals_per_90=expected_goals_per_90,
        expected_assists_per_90=expected_assists_per_90,
        expected_goals_conceded_per_90=expected_goals_conceded_per_90,
        expected_minutes=expected_minutes,
        chance_of_playing_next_round=chance_of_playing_next_round,
        chance_of_playing_this_round=chance_of_playing_this_round,
        saves=saves,
        bonus=bonus,
        own_goals=own_goals,
        goals_scored=goals_scored,
        assists=assists,
        goals_conceded=10,
        clean_sheets=15,
        yellow_cards=yellow_cards,
        red_cards=red_cards,
        tackles=tackles,
        interceptions=interceptions,
        clearances=clearances,
        blocks=blocks,
    )


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constants."""

    def test_league_avg_goals(self) -> None:
        assert pytest.approx(1.36) == LEAGUE_AVG_GOALS_PER_TEAM

    def test_shrinkage_strength(self) -> None:
        assert SHRINKAGE_STRENGTH == 3.0

    def test_min_benchmark_minutes(self) -> None:
        assert MIN_BENCHMARK_MINUTES == 270

    def test_max_bonus(self) -> None:
        assert MAX_BONUS_PER_MATCH == 3.0


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestPositionBenchmark:
    """Tests for PositionBenchmark dataclass."""

    def test_immutable(self) -> None:
        bm = PositionBenchmark(position=PlayerPosition.MIDFIELDER)
        with pytest.raises(AttributeError):
            bm.xg_per_90 = 0.5  # type: ignore[misc]

    def test_defaults(self) -> None:
        bm = PositionBenchmark(position=PlayerPosition.FORWARD)
        assert bm.position == PlayerPosition.FORWARD
        assert bm.xg_per_90 == 0.0
        assert bm.xa_per_90 == 0.0
        assert bm.saves_per_90 == 0.0
        assert bm.bonus_per_90 == 0.0
        assert bm.matches_played == 0.0

    def test_custom_values(self) -> None:
        bm = PositionBenchmark(
            position=PlayerPosition.GOALKEEPER,
            xg_per_90=0.1,
            xa_per_90=0.05,
            saves_per_90=3.0,
            bonus_per_90=0.05,
            matches_played=30.0,
        )
        assert bm.position == PlayerPosition.GOALKEEPER
        assert bm.xg_per_90 == pytest.approx(0.1)
        assert bm.saves_per_90 == pytest.approx(3.0)
        assert bm.matches_played == 30.0


class TestBaselineFeatures:
    """Tests for BaselineFeatures dataclass."""

    def test_immutable(self) -> None:
        features = BaselineFeatures(
            player_id=1, fixture_id=1, gameweek=1,
            position=PlayerPosition.MIDFIELDER, is_home=True,
            start_probability=1.0, probability_appears=1.0,
            probability_60_plus=0.95, clean_sheet_probability=None,
            expected_goals=0.5, expected_assists=0.3,
            expected_goals_conceded=None, expected_saves=None,
            expected_bonus=None, expected_own_goals=None,
            expected_yellow_cards=0.0, expected_red_cards=0.0,
            expected_penalties_saved=0.0, expected_penalties_missed=0.0,
            defensive_contribution_probability=0.0,
            defensive_contribution_probability_from_history=True,
            cutoff_gameweek=None, expected_minutes=90.0, matches_played=30.0,
        )
        with pytest.raises(AttributeError):
            features.expected_goals = 0.6  # type: ignore[misc]

    def test_none_preserved(self) -> None:
        """None values should be preserved to distinguish 'unavailable' from 'zero'."""
        features = BaselineFeatures(
            player_id=1, fixture_id=1, gameweek=1,
            position=PlayerPosition.FORWARD, is_home=False,
            start_probability=0.8, probability_appears=0.8,
            probability_60_plus=0.6, clean_sheet_probability=None,
            expected_goals=None, expected_assists=None,
            expected_goals_conceded=None, expected_saves=None,
            expected_bonus=None, expected_own_goals=None,
            expected_yellow_cards=0.0, expected_red_cards=0.0,
            expected_penalties_saved=0.0, expected_penalties_missed=0.0,
            defensive_contribution_probability=0.0,
            defensive_contribution_probability_from_history=False,
            cutoff_gameweek=None, expected_minutes=75.0, matches_played=10.0,
        )
        assert features.clean_sheet_probability is None
        assert features.expected_goals is None
        assert features.expected_assists is None
        assert features.expected_own_goals is None
        assert features.expected_yellow_cards == 0.0

    def test_from_history_defaults_to_true(self) -> None:
        """The provenance flag defaults to True (data-backed assumption)."""
        features = BaselineFeatures(
            player_id=1, fixture_id=1, gameweek=1,
            position=PlayerPosition.MIDFIELDER, is_home=True,
            start_probability=1.0, probability_appears=1.0,
            probability_60_plus=0.95, clean_sheet_probability=None,
            expected_goals=0.5, expected_assists=0.3,
            expected_goals_conceded=None, expected_saves=None,
            expected_bonus=None, expected_own_goals=None,
            expected_yellow_cards=0.0, expected_red_cards=0.0,
            expected_penalties_saved=0.0, expected_penalties_missed=0.0,
            defensive_contribution_probability=0.0,
            cutoff_gameweek=None, expected_minutes=90.0, matches_played=30.0,
        )
        assert (
            features.defensive_contribution_probability_from_history is True
        )

    def test_from_history_can_be_false(self) -> None:
        """The provenance flag can be explicitly set to False (prior fallback)."""
        features = BaselineFeatures(
            player_id=1, fixture_id=1, gameweek=1,
            position=PlayerPosition.MIDFIELDER, is_home=True,
            start_probability=1.0, probability_appears=1.0,
            probability_60_plus=0.95, clean_sheet_probability=None,
            expected_goals=0.5, expected_assists=0.3,
            expected_goals_conceded=None, expected_saves=None,
            expected_bonus=None, expected_own_goals=None,
            expected_yellow_cards=0.0, expected_red_cards=0.0,
            expected_penalties_saved=0.0, expected_penalties_missed=0.0,
            defensive_contribution_probability=0.25,
            defensive_contribution_probability_from_history=False,
            cutoff_gameweek=None, expected_minutes=90.0, matches_played=30.0,
        )
        assert (
            features.defensive_contribution_probability_from_history is False
        )
    """E5 tests for BaselineFeatures.__post_init__ input validation."""

    def _valid_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Return a dict of valid BaselineFeatures kwargs."""
        kwargs: dict[str, Any] = {
            "player_id": 1,
            "fixture_id": 1,
            "gameweek": 1,
            "position": PlayerPosition.MIDFIELDER,
            "is_home": True,
            "start_probability": 1.0,
            "probability_appears": 1.0,
            "probability_60_plus": 0.95,
            "clean_sheet_probability": None,
            "expected_goals": 0.5,
            "expected_assists": 0.3,
            "expected_goals_conceded": None,
            "expected_saves": None,
            "expected_bonus": None,
            "expected_own_goals": None,
            "expected_yellow_cards": 0.0,
            "expected_red_cards": 0.0,
            "expected_penalties_saved": 0.0,
            "expected_penalties_missed": 0.0,
            "defensive_contribution_probability": 0.0,
            "defensive_contribution_probability_from_history": True,
            "cutoff_gameweek": None,
            "expected_minutes": 90.0,
            "matches_played": 30.0,
        }
        kwargs.update(overrides)
        return kwargs

    def test_valid_construction_succeeds(self) -> None:
        features = BaselineFeatures(**self._valid_kwargs())
        assert features.expected_goals == pytest.approx(0.5)

    def test_none_values_accepted(self) -> None:
        features = BaselineFeatures(
            **self._valid_kwargs(
                clean_sheet_probability=None,
                expected_goals=None,
                expected_assists=None,
                expected_goals_conceded=None,
                expected_saves=None,
                expected_bonus=None,
                expected_own_goals=None,
            )
        )
        assert features.expected_goals is None

    def test_nan_start_probability_rejected(self) -> None:
        with pytest.raises(BaselineModelError, match="start_probability"):
            BaselineFeatures(**self._valid_kwargs(start_probability=float("nan")))

    def test_inf_start_probability_rejected(self) -> None:
        with pytest.raises(BaselineModelError, match="start_probability"):
            BaselineFeatures(**self._valid_kwargs(start_probability=float("inf")))

    def test_probability_above_one_rejected(self) -> None:
        with pytest.raises(BaselineModelError, match="probability_60_plus"):
            BaselineFeatures(**self._valid_kwargs(probability_60_plus=1.5))

    def test_negative_probability_rejected(self) -> None:
        with pytest.raises(BaselineModelError, match="probability_appears"):
            BaselineFeatures(**self._valid_kwargs(probability_appears=-0.1))

    def test_bool_probability_rejected(self) -> None:
        with pytest.raises(BaselineModelError, match="start_probability"):
            BaselineFeatures(**self._valid_kwargs(start_probability=True))  # type: ignore[arg-type]

    def test_nan_count_rejected(self) -> None:
        with pytest.raises(BaselineModelError, match="expected_goals"):
            BaselineFeatures(**self._valid_kwargs(expected_goals=float("nan")))

    def test_negative_count_rejected(self) -> None:
        with pytest.raises(BaselineModelError, match="expected_assists"):
            BaselineFeatures(**self._valid_kwargs(expected_assists=-0.5))

    def test_bonus_above_cap_rejected(self) -> None:
        with pytest.raises(BaselineModelError, match="expected_bonus"):
            BaselineFeatures(**self._valid_kwargs(expected_bonus=3.5))

    def test_bonus_negative_rejected(self) -> None:
        with pytest.raises(BaselineModelError, match="expected_bonus"):
            BaselineFeatures(**self._valid_kwargs(expected_bonus=-0.1))

    def test_negative_yellow_cards_rejected(self) -> None:
        with pytest.raises(BaselineModelError, match="expected_yellow_cards"):
            BaselineFeatures(**self._valid_kwargs(expected_yellow_cards=-1.0))

    def test_bool_count_rejected(self) -> None:
        with pytest.raises(BaselineModelError, match="expected_goals"):
            BaselineFeatures(**self._valid_kwargs(expected_goals=True))  # type: ignore[arg-type]

    def test_boundary_zero_probability_accepted(self) -> None:
        features = BaselineFeatures(
            **self._valid_kwargs(
                start_probability=0.0,
                probability_appears=0.0,
                probability_60_plus=0.0,
            )
        )
        assert features.start_probability == 0.0

    def test_boundary_one_probability_accepted(self) -> None:
        features = BaselineFeatures(
            **self._valid_kwargs(
                start_probability=1.0,
                probability_appears=1.0,
                probability_60_plus=1.0,
            )
        )
        assert features.probability_60_plus == 1.0

    def test_boundary_zero_count_accepted(self) -> None:
        features = BaselineFeatures(**self._valid_kwargs(expected_goals=0.0))
        assert features.expected_goals == 0.0

    def test_boundary_bonus_zero_accepted(self) -> None:
        features = BaselineFeatures(**self._valid_kwargs(expected_bonus=0.0))
        assert features.expected_bonus == 0.0

    def test_boundary_bonus_three_accepted(self) -> None:
        features = BaselineFeatures(**self._valid_kwargs(expected_bonus=3.0))
        assert features.expected_bonus == 3.0


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestToPer90:
    """Tests for _to_per_90 helper."""

    def test_normal_conversion(self) -> None:
        assert _to_per_90(90.0, 2700) == pytest.approx(3.0)

    def test_zero_minutes_returns_none(self) -> None:
        assert _to_per_90(10.0, 0) is None

    def test_negative_minutes_returns_none(self) -> None:
        assert _to_per_90(10.0, -100) is None

    def test_none_total_returns_none(self) -> None:
        assert _to_per_90(None, 2700) is None

    def test_zero_total_returns_zero(self) -> None:
        assert _to_per_90(0.0, 90) == pytest.approx(0.0)

    def test_integer_inputs(self) -> None:
        assert _to_per_90(180, 2700) == pytest.approx(6.0)


class TestShrinkRate:
    """Tests for _shrink_rate helper."""

    def test_zero_matches_returns_prior(self) -> None:
        result = _shrink_rate(observed_rate=10.0, prior_rate=2.0, matches=0.0)
        assert result == pytest.approx(2.0)

    def test_normal_shrinkage(self) -> None:
        # shrunk = (10*5 + 2*3) / (5+3) = (50+6)/8 = 7.0
        result = _shrink_rate(observed_rate=10.0, prior_rate=2.0, matches=5.0)
        assert result == pytest.approx(7.0)

    def test_high_matches_converges_to_observed(self) -> None:
        result = _shrink_rate(
            observed_rate=10.0, prior_rate=2.0, matches=10000.0
        )
        assert result == pytest.approx(10.0, abs=0.01)

    def test_zero_matches_and_zero_prior(self) -> None:
        result = _shrink_rate(observed_rate=5.0, prior_rate=0.0, matches=0.0)
        assert result == pytest.approx(0.0)


class TestComputeStartProbability:
    """Tests for _compute_start_probability helper."""

    def test_chance_of_playing_next_round_takes_priority(self) -> None:
        player = make_player(chance_of_playing_next_round=80)
        assert _compute_start_probability(player) == pytest.approx(0.8)

    def test_doubtful_status(self) -> None:
        player = make_player(
            chance_of_playing_next_round=None,
            status=PlayerStatus.DOUBTFUL,
        )
        assert _compute_start_probability(player) == pytest.approx(0.5)

    def test_available_status(self) -> None:
        player = make_player(
            chance_of_playing_next_round=None,
            status=PlayerStatus.AVAILABLE,
        )
        assert _compute_start_probability(player) == pytest.approx(1.0)

    def test_injured_status(self) -> None:
        player = make_player(
            chance_of_playing_next_round=None,
            status=PlayerStatus.INJURED,
        )
        assert _compute_start_probability(player) == pytest.approx(0.0)

    def test_suspended_status(self) -> None:
        player = make_player(
            chance_of_playing_next_round=None,
            status=PlayerStatus.SUSPENDED,
        )
        assert _compute_start_probability(player) == pytest.approx(0.0)

    def test_zero_chance(self) -> None:
        player = make_player(chance_of_playing_next_round=0)
        assert _compute_start_probability(player) == pytest.approx(0.0)

    def test_chance_over_100_clamped(self) -> None:
        player = make_player(chance_of_playing_next_round=150)
        assert _compute_start_probability(player) == pytest.approx(1.0)

    def test_negative_chance_clamped(self) -> None:
        player = make_player(chance_of_playing_next_round=-20)
        assert _compute_start_probability(player) == pytest.approx(0.0)


class TestResolveExpectedMinutes:
    """Tests for _resolve_expected_minutes helper."""

    def test_uses_expected_minutes_when_available(self) -> None:
        player = make_player(expected_minutes=60)
        assert _resolve_expected_minutes(player, 1.0) == pytest.approx(60.0)

    def test_infers_90_times_start_prob_when_expected_minutes_none(self) -> None:
        player = make_player(expected_minutes=None)
        assert _resolve_expected_minutes(player, 0.5) == pytest.approx(45.0)

    def test_returns_zero_when_start_prob_zero(self) -> None:
        player = make_player(expected_minutes=None)
        assert _resolve_expected_minutes(player, 0.0) == pytest.approx(0.0)

    def test_zero_start_prob_with_expected_minutes(self) -> None:
        player = make_player(expected_minutes=30)
        assert _resolve_expected_minutes(player, 0.0) == pytest.approx(30.0)


class TestHeuristic60Plus:
    """Tests for _heuristic_60plus_factor helper."""

    def test_90_minutes(self) -> None:
        assert _heuristic_60plus_factor(90.0) == pytest.approx(0.95)

    def test_89_minutes(self) -> None:
        assert _heuristic_60plus_factor(89.0) == pytest.approx(0.75)

    def test_60_minutes(self) -> None:
        assert _heuristic_60plus_factor(60.0) == pytest.approx(0.75)

    def test_59_minutes(self) -> None:
        assert _heuristic_60plus_factor(59.0) == pytest.approx(0.20)

    def test_30_minutes(self) -> None:
        assert _heuristic_60plus_factor(30.0) == pytest.approx(0.20)

    def test_29_minutes(self) -> None:
        assert _heuristic_60plus_factor(29.0) == pytest.approx(0.0)

    def test_zero_minutes(self) -> None:
        assert _heuristic_60plus_factor(0.0) == pytest.approx(0.0)

    def test_negative_minutes(self) -> None:
        assert _heuristic_60plus_factor(-5.0) == pytest.approx(0.0)


class TestSafeMean:
    """Tests for _safe_mean helper."""

    def test_normal_mean(self) -> None:
        assert _safe_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_empty_list_returns_zero(self) -> None:
        assert _safe_mean([]) == pytest.approx(0.0)

    def test_single_value(self) -> None:
        assert _safe_mean([5.0]) == pytest.approx(5.0)

    def test_negative_values(self) -> None:
        assert _safe_mean([-3.0, 3.0]) == pytest.approx(0.0)


class TestProjectGoalsConcededRate:
    """Tests for _project_goals_conceded_rate helper."""

    def test_home_team_strong_defense(self) -> None:
        team = make_team(strength_defence_home=5, strength_attack_away=5)
        opponent = make_team(
            team_id=1, short_name="ARS", name="Arsenal",
            strength_attack_away=3, strength_defence_home=3,
        )
        rate = _project_goals_conceded_rate(team, opponent, is_home=True)
        # Strong defence (5) reduces goals conceded; average attack (3)
        assert rate == pytest.approx(1.36 * (3.0 / 3.0) / (5.0 / 3.0))

    def test_away_team(self) -> None:
        team = make_team(strength_defence_away=4, strength_attack_home=5)
        opponent = make_team(
            team_id=1, short_name="ARS", name="Arsenal",
            strength_attack_away=4, strength_attack_home=3,
        )
        rate = _project_goals_conceded_rate(team, opponent, is_home=False)
        # Defence 4 reduces vs league avg; attack 3 is league avg
        assert rate == pytest.approx(1.36 * (3.0 / 3.0) / (4.0 / 3.0))

    def test_zero_strength_returns_league_avg(self) -> None:
        team = make_team(strength_defence_home=0, strength_defence_away=0)
        opponent = make_team(
            team_id=1, short_name="ARS", name="Arsenal",
            strength_attack_away=0, strength_attack_home=0,
        )
        rate = _project_goals_conceded_rate(team, opponent, is_home=True)
        assert rate == pytest.approx(LEAGUE_AVG_GOALS_PER_TEAM)

    def test_league_avg_strength(self) -> None:
        """Team and opponent both at league-average (3.0) -> league avg."""
        team = make_team(
            strength_defence_home=3, strength_defence_away=3,
            strength_attack_home=3, strength_attack_away=3,
        )
        opponent = make_team(
            team_id=1, short_name="ARS", name="Arsenal",
            strength_defence_home=3, strength_defence_away=3,
            strength_attack_home=3, strength_attack_away=3,
        )
        rate = _project_goals_conceded_rate(team, opponent, is_home=True)
        assert rate == pytest.approx(LEAGUE_AVG_GOALS_PER_TEAM)

    def test_stronger_defence_reduces_expected_goals(self) -> None:
        """Holding opponent attack constant, stronger defence -> lower lambda."""
        base_team = make_team(
            strength_defence_home=3, strength_defence_away=3,
            strength_attack_home=3, strength_attack_away=3,
        )
        strong_team = make_team(
            strength_defence_home=5, strength_defence_away=5,
            strength_attack_home=3, strength_attack_away=3,
        )
        opponent = make_team(
            team_id=1, short_name="ARS", name="Arsenal",
            strength_attack_away=3, strength_attack_home=3,
            strength_defence_home=3, strength_defence_away=3,
        )
        base_rate = _project_goals_conceded_rate(base_team, opponent, is_home=True)
        strong_rate = _project_goals_conceded_rate(strong_team, opponent, is_home=True)
        assert strong_rate < base_rate

    def test_stronger_attack_increases_expected_goals(self) -> None:
        """Holding team defence constant, stronger opponent attack -> higher lambda."""
        team = make_team(
            strength_defence_home=3, strength_defence_away=3,
            strength_attack_home=3, strength_attack_away=3,
        )
        weak_opponent = make_team(
            team_id=1, short_name="ARS", name="Arsenal",
            strength_attack_away=1, strength_attack_home=1,
            strength_defence_home=3, strength_defence_away=3,
        )
        strong_opponent = make_team(
            team_id=1, short_name="ARS", name="Arsenal",
            strength_attack_away=5, strength_attack_home=5,
            strength_defence_home=3, strength_defence_away=3,
        )
        weak_rate = _project_goals_conceded_rate(team, weak_opponent, is_home=True)
        strong_rate = _project_goals_conceded_rate(team, strong_opponent, is_home=True)
        assert strong_rate > weak_rate


# ---------------------------------------------------------------------------
# compute_position_benchmarks tests
# ---------------------------------------------------------------------------
    """Tests for compute_position_benchmarks function."""

    def test_empty_players(self) -> None:
        """Empty player list returns benchmarks for all positions with zero rates."""
        result = compute_position_benchmarks([])
        for pos in PlayerPosition:
            assert pos in result
            assert result[pos].position == pos
            assert result[pos].xg_per_90 == 0.0
            assert result[pos].matches_played == 0.0

    def test_filters_low_minutes(self) -> None:
        """Players below MIN_BENCHMARK_MINUTES should be excluded."""
        player_good = make_player(
            player_id=1,
            position=PlayerPosition.MIDFIELDER,
            minutes=300,
            expected_goals=3.0,
        )
        player_bad = make_player(
            player_id=2,
            position=PlayerPosition.MIDFIELDER,
            minutes=200,
            expected_goals=3.0,
        )
        result = compute_position_benchmarks([player_good, player_bad])
        mid_bm = result[PlayerPosition.MIDFIELDER]
        assert mid_bm.xg_per_90 == pytest.approx(0.9)

    def test_xg_per_90_computed_from_total(self) -> None:
        """When expected_goals_per_90 is None, compute from total/minutes."""
        player = make_player(
            player_id=1,
            position=PlayerPosition.FORWARD,
            minutes=500,
            expected_goals=10.0,
            expected_goals_per_90=None,
        )
        result = compute_position_benchmarks([player])
        fwd_bm = result[PlayerPosition.FORWARD]
        # xg_per_90 = 10.0 / 500 * 90 = 1.8
        assert fwd_bm.xg_per_90 == pytest.approx(1.8)

    def test_falls_back_to_total_divided_by_minutes(self) -> None:
        """When per_90 is None but total exists, compute from total/minutes."""
        player = make_player(
            player_id=1,
            position=PlayerPosition.FORWARD,
            minutes=900,
            expected_goals=18.0,
            expected_goals_per_90=None,
        )
        result = compute_position_benchmarks([player])
        fwd_bm = result[PlayerPosition.FORWARD]
        assert fwd_bm.xg_per_90 == pytest.approx(1.8)

    def test_aggregates_multiple_players(self) -> None:
        """Benchmarks should be the average across eligible players."""
        p1 = make_player(
            player_id=1, position=PlayerPosition.MIDFIELDER,
            minutes=900, expected_goals=18.0, expected_assists=9.0,
            bonus=20,
        )
        p2 = make_player(
            player_id=2, position=PlayerPosition.MIDFIELDER,
            minutes=900, expected_goals=9.0, expected_assists=18.0,
            bonus=10,
        )
        result = compute_position_benchmarks([p1, p2])
        mid_bm = result[PlayerPosition.MIDFIELDER]
        assert mid_bm.xg_per_90 == pytest.approx(1.35)
        assert mid_bm.xa_per_90 == pytest.approx(1.35)
        assert mid_bm.bonus_per_90 == pytest.approx(1.5)

    def test_saves_per_90_only_for_goalkeeper(self) -> None:
        """Saves benchmark should be computed for goalkeepers."""
        gk = make_player(
            player_id=1, position=PlayerPosition.GOALKEEPER,
            minutes=900, saves=100,
        )
        result = compute_position_benchmarks([gk])
        gk_bm = result[PlayerPosition.GOALKEEPER]
        assert gk_bm.saves_per_90 == pytest.approx(10.0)
        for pos in [PlayerPosition.DEFENDER, PlayerPosition.MIDFIELDER, PlayerPosition.FORWARD]:
            assert result[pos].saves_per_90 == 0.0

    def test_none_expected_goals_excluded_from_xg_rate(self) -> None:
        """Players with None expected_goals should not contribute to xg_per_90."""
        p1 = make_player(
            player_id=1, position=PlayerPosition.MIDFIELDER,
            minutes=900, expected_goals=None, expected_goals_per_90=1.0,
        )
        p2 = make_player(
            player_id=2, position=PlayerPosition.MIDFIELDER,
            minutes=900, expected_goals=9.0, expected_goals_per_90=None,
        )
        result = compute_position_benchmarks([p1, p2])
        mid_bm = result[PlayerPosition.MIDFIELDER]
        # p1: expected_goals=None -> excluded; p2: 9.0/900*90=0.9 -> mean=0.9
        assert mid_bm.xg_per_90 == pytest.approx(0.9)

    def test_matches_played(self) -> None:
        """matches_played should be average minutes/90 across eligible players."""
        p1 = make_player(player_id=1, position=PlayerPosition.MIDFIELDER, minutes=900)
        p2 = make_player(player_id=2, position=PlayerPosition.MIDFIELDER, minutes=2700)
        result = compute_position_benchmarks([p1, p2])
        mid_bm = result[PlayerPosition.MIDFIELDER]
        assert mid_bm.matches_played == pytest.approx(20.0)

    def test_all_positions_present(self) -> None:
        """All four positions should always be in the result."""
        result = compute_position_benchmarks([])
        assert set(result.keys()) == set(PlayerPosition)


# ---------------------------------------------------------------------------
# extract_baseline_features tests
# ---------------------------------------------------------------------------


class TestExtractBaselineFeatures:
    """Tests for extract_baseline_features function."""

    @pytest.fixture
    def benchmarks(self) -> dict:
        """Position benchmarks with known values."""
        return compute_position_benchmarks([
            make_player(
                player_id=1, position=PlayerPosition.MIDFIELDER,
                minutes=2700, expected_goals=20.0, expected_assists=15.0,
                bonus=40,
            ),
        ])

    def test_basic_extraction(self, benchmarks: dict) -> None:
        player = make_player(
            position=PlayerPosition.MIDFIELDER,
            minutes=2700,
            expected_goals_per_90=0.7,
            expected_assists_per_90=0.5,
            bonus=40,
        )
        fixture = make_fixture()
        team = make_team(team_id=14, short_name="LIV")
        opponent = make_team(team_id=1, short_name="ARS", name="Arsenal")
        features = extract_baseline_features(
            player, fixture, team, opponent, benchmarks, gameweek=1
        )
        assert features.player_id == player.id
        assert features.fixture_id == fixture.id
        assert features.gameweek == 1
        assert features.position == PlayerPosition.MIDFIELDER
        assert features.is_home is True

    def test_clean_sheet_probability_with_teams(self, benchmarks: dict) -> None:
        player = make_player(position=PlayerPosition.DEFENDER, minutes=2700)
        fixture = make_fixture(team_h=14, team_a=1)
        team = make_team(team_id=14, short_name="LIV")
        opponent = make_team(team_id=1, short_name="ARS", name="Arsenal")
        features = extract_baseline_features(
            player, fixture, team, opponent, benchmarks, gameweek=1
        )
        assert features.clean_sheet_probability is not None
        assert 0.0 <= features.clean_sheet_probability <= 1.0

    def test_clean_sheet_none_without_teams(self, benchmarks: dict) -> None:
        player = make_player(position=PlayerPosition.DEFENDER, minutes=2700)
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, team=None, opponent=None,
            benchmarks=benchmarks, gameweek=1,
        )
        assert features.clean_sheet_probability is None

    def test_expected_goals_from_per_90(self, benchmarks: dict) -> None:
        player = make_player(
            position=PlayerPosition.MIDFIELDER,
            minutes=2700,
            expected_goals_per_90=0.5,
            expected_goals=15.0,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.expected_goals == pytest.approx(0.5)

    def test_expected_goals_fallback_from_total(self, benchmarks: dict) -> None:
        player = make_player(
            position=PlayerPosition.MIDFIELDER,
            minutes=900,
            expected_goals_per_90=None,
            expected_goals=9.0,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.expected_goals == pytest.approx(0.9)

    def test_expected_goals_fallback_to_benchmark(self, benchmarks: dict) -> None:
        player = make_player(
            position=PlayerPosition.MIDFIELDER,
            minutes=2700,
            expected_goals_per_90=None,
            expected_goals=None,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.expected_goals is not None
        assert features.expected_goals > 0.0

    def test_expected_goals_fallback_to_zero_short_minutes(self, benchmarks: dict) -> None:
        player = make_player(
            position=PlayerPosition.MIDFIELDER,
            minutes=100,
            expected_goals_per_90=None,
            expected_goals=None,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.expected_goals == 0.0

    def test_expected_goals_conceded_none_for_forward(self, benchmarks: dict) -> None:
        player = make_player(
            position=PlayerPosition.FORWARD,
            minutes=2700,
            expected_goals_conceded_per_90=None,
            expected_goals_conceded=20.0,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.expected_goals_conceded is None

    def test_expected_goals_conceded_for_defender(self, benchmarks: dict) -> None:
        player = make_player(
            position=PlayerPosition.DEFENDER,
            minutes=2700,
            expected_goals_conceded_per_90=None,
            expected_goals_conceded=20.0,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.expected_goals_conceded is not None
        assert features.expected_goals_conceded > 0.0

    def test_expected_saves_only_for_goalkeeper(self, benchmarks: dict) -> None:
        gk = make_player(
            position=PlayerPosition.GOALKEEPER,
            minutes=2700, saves=100,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            gk, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.expected_saves is not None
        assert features.expected_saves == pytest.approx(100 / 2700 * 90.0)

    def test_expected_saves_none_for_non_gk(self, benchmarks: dict) -> None:
        player = make_player(position=PlayerPosition.MIDFIELDER, minutes=2700)
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.expected_saves is None

    def test_expected_bonus_from_per_90(self, benchmarks: dict) -> None:
        """With full 90 expected minutes, projection factor is 1.0."""
        player = make_player(
            position=PlayerPosition.MIDFIELDER,
            minutes=2700, bonus=40,
            expected_goals_per_90=0.7,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.expected_bonus is not None
        # Per-90 rate * (90/90) projection, no cap needed
        assert features.expected_bonus == pytest.approx(40 / 2700 * 90.0)

    def test_expected_bonus_short_minutes_projection(self, benchmarks: dict) -> None:
        """Bonus projected to fewer expected minutes scales proportionally."""
        player = make_player(
            position=PlayerPosition.MIDFIELDER,
            minutes=2700, bonus=40,
            expected_minutes=45,
            chance_of_playing_next_round=None,
            status=PlayerStatus.AVAILABLE,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.expected_bonus is not None
        # 40/2700*90 = 1.333 per-90; projected to 45 min: 1.333 * 45/90 = 0.667
        assert features.expected_bonus == pytest.approx(40 / 2700 * 90.0 * 45.0 / 90.0)

    def test_expected_bonus_capped_at_max(self, benchmarks: dict) -> None:
        """Projected bonus above MAX_BONUS_PER_MATCH is capped at 3.0."""
        player = make_player(
            position=PlayerPosition.MIDFIELDER,
            minutes=900, bonus=300,
            expected_goals_per_90=0.7,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.expected_bonus is not None
        # Per-90 = 300/900*90 = 30.0, projected to 90 min = 30.0, capped at 3.0
        assert features.expected_bonus == pytest.approx(MAX_BONUS_PER_MATCH)

    def test_expected_bonus_zero_when_not_playing(self, benchmarks: dict) -> None:
        """When chance_of_playing is 0 and no expected_minutes, bonus is 0."""
        player = make_player(
            position=PlayerPosition.MIDFIELDER,
            minutes=2700, bonus=40,
            chance_of_playing_next_round=0,
            expected_minutes=None,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.expected_bonus == 0.0

    # ------------------------------------------------------------------
    # D2 regression: bonus projection correctness
    # ------------------------------------------------------------------

    def test_bonus_full_minutes_uses_factor_one(self, benchmarks: dict) -> None:
        """Bonus with exp_minutes=90 (start_prob=1): factor = 1.0."""
        player = make_player(
            position=PlayerPosition.MIDFIELDER,
            minutes=2700, bonus=60,
        )
        features = extract_baseline_features(
            player, make_fixture(), None, None, benchmarks, gameweek=1
        )
        assert features.expected_bonus == pytest.approx(60 / 2700 * 90.0)

    def test_bonus_partial_availability_half(self, benchmarks: dict) -> None:
        """Bonus scaled by start_prob when expected_minutes is None."""
        player = make_player(
            position=PlayerPosition.MIDFIELDER,
            minutes=2700, bonus=60,
            expected_minutes=None,
            chance_of_playing_next_round=50,
        )
        features = extract_baseline_features(
            player, make_fixture(), None, None, benchmarks, gameweek=1
        )
        # start_prob=0.5 → exp_minutes=45 → projection_factor=0.5
        assert features.expected_bonus == pytest.approx(
            60 / 2700 * 90.0 * 45.0 / 90.0
        )

    def test_bonus_zero_when_not_playing(self, benchmarks: dict) -> None:
        """Bonus is 0.0 when start_prob=0 and no expected_minutes."""
        player = make_player(
            position=PlayerPosition.MIDFIELDER,
            minutes=2700, bonus=60,
            expected_minutes=None,
            chance_of_playing_next_round=0,
        )
        features = extract_baseline_features(
            player, make_fixture(), None, None, benchmarks, gameweek=1
        )
        assert features.expected_bonus == 0.0

    def test_bonus_capped_at_max(self, benchmarks: dict) -> None:
        """Bonus capped at MAX_BONUS_PER_MATCH after projection."""
        player = make_player(
            position=PlayerPosition.MIDFIELDER,
            minutes=600, bonus=300,
        )
        features = extract_baseline_features(
            player, make_fixture(), None, None, benchmarks, gameweek=1
        )
        # bonus_per_90 = 300/600*90 = 45.0; projected = 45.0; capped at 3.0
        assert features.expected_bonus == pytest.approx(MAX_BONUS_PER_MATCH)

    def test_expected_own_goals_none_short_minutes(self, benchmarks: dict) -> None:
        player = make_player(
            position=PlayerPosition.MIDFIELDER, minutes=100, own_goals=0,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.expected_own_goals is None

    def test_expected_own_goals_computed_long_minutes(self, benchmarks: dict) -> None:
        player = make_player(
            position=PlayerPosition.MIDFIELDER, minutes=2700, own_goals=2,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.expected_own_goals == pytest.approx(2 / 2700 * 90.0)

    # ------------------------------------------------------------------
    # D1 regression: fixture-level projection of per-90 rates
    # ------------------------------------------------------------------

    def test_xg_projected_full_90_minutes(self, benchmarks: dict) -> None:
        """Per-90 xG with exp_minutes=90: projection factor = 1.0."""
        player = make_player(
            position=PlayerPosition.MIDFIELDER,
            minutes=2700,
            expected_goals_per_90=0.5,
        )
        features = extract_baseline_features(
            player, make_fixture(), None, None, benchmarks, gameweek=1
        )
        assert features.expected_goals == pytest.approx(0.5)

    def test_xg_projected_half_minutes(self, benchmarks: dict) -> None:
        """Per-90 xG with expected_minutes=45: projection factor = 0.5."""
        player = make_player(
            position=PlayerPosition.MIDFIELDER,
            minutes=2700,
            expected_goals_per_90=0.5,
            expected_minutes=45,
            chance_of_playing_next_round=100,
        )
        features = extract_baseline_features(
            player, make_fixture(), None, None, benchmarks, gameweek=1
        )
        assert features.expected_goals == pytest.approx(0.25)

    def test_xg_projected_partial_availability(self, benchmarks: dict) -> None:
        """Without expected_minutes, exp_minutes = 90 * start_prob."""
        player = make_player(
            position=PlayerPosition.MIDFIELDER,
            minutes=2700,
            expected_goals_per_90=0.5,
            expected_minutes=None,
            chance_of_playing_next_round=50,
        )
        features = extract_baseline_features(
            player, make_fixture(), None, None, benchmarks, gameweek=1
        )
        # start_prob=0.5 → exp_minutes=45 → projection_factor=0.5
        assert features.expected_goals == pytest.approx(0.25)

    def test_xg_zero_when_not_playing(self, benchmarks: dict) -> None:
        """Zero expected minutes → zero fixture-level expectation."""
        player = make_player(
            position=PlayerPosition.MIDFIELDER,
            minutes=2700,
            expected_goals_per_90=0.5,
            expected_minutes=None,
            chance_of_playing_next_round=0,
        )
        features = extract_baseline_features(
            player, make_fixture(), None, None, benchmarks, gameweek=1
        )
        assert features.expected_goals == pytest.approx(0.0)

    def test_zero_availability_suppresses_event_counts(self, benchmarks: dict) -> None:
        """When expected_minutes > 0 but chance_of_playing is 0, all
        playing-dependent expected counts must be zero while the
        expected_minutes metadata is preserved."""
        player = make_player(
            position=PlayerPosition.MIDFIELDER,
            minutes=2700,
            expected_goals_per_90=0.7,
            expected_assists_per_90=0.5,
            bonus=40,
            expected_minutes=30,
            chance_of_playing_next_round=0,
        )
        features = extract_baseline_features(
            player, make_fixture(), None, None, benchmarks, gameweek=1
        )
        # Metadata preserved
        assert features.expected_minutes == pytest.approx(30.0)
        # Probabilities zero
        assert features.start_probability == pytest.approx(0.0)
        assert features.probability_appears == pytest.approx(0.0)
        assert features.probability_60_plus == pytest.approx(0.0)
        # Playing-dependent counts suppressed to zero
        assert features.expected_goals == pytest.approx(0.0)
        assert features.expected_assists == pytest.approx(0.0)
        assert features.expected_bonus == pytest.approx(0.0)

    def test_xa_projected_proportional(self, benchmarks: dict) -> None:
        """Expected assists projected the same way as expected goals."""
        player = make_player(
            position=PlayerPosition.MIDFIELDER,
            minutes=2700,
            expected_assists_per_90=0.3,
            expected_minutes=45,
            chance_of_playing_next_round=100,
        )
        features = extract_baseline_features(
            player, make_fixture(), None, None, benchmarks, gameweek=1
        )
        assert features.expected_assists == pytest.approx(0.15)

    def test_saves_projected_gk(self, benchmarks: dict) -> None:
        """GK saves projected to fixture-level count."""
        gk = make_player(
            position=PlayerPosition.GOALKEEPER,
            minutes=2700, saves=100,
            expected_minutes=45,
            chance_of_playing_next_round=100,
        )
        features = extract_baseline_features(
            gk, make_fixture(), None, None, benchmarks, gameweek=1
        )
        assert features.expected_saves == pytest.approx(
            100 / 2700 * 90.0 * 45.0 / 90.0
        )

    def test_xgc_projected_defender(self, benchmarks: dict) -> None:
        """Defender xGC projected to fixture-level count."""
        player = make_player(
            position=PlayerPosition.DEFENDER,
            minutes=2700,
            expected_goals_conceded_per_90=0.8,
            expected_minutes=45,
            chance_of_playing_next_round=100,
        )
        features = extract_baseline_features(
            player, make_fixture(), None, None, benchmarks, gameweek=1
        )
        assert features.expected_goals_conceded == pytest.approx(0.4)

    def test_own_goals_projected_to_fixture(self, benchmarks: dict) -> None:
        """Own-goals per-90 rate projected to fixture-level count."""
        player = make_player(
            position=PlayerPosition.MIDFIELDER, minutes=2700, own_goals=2,
            expected_minutes=45,
            chance_of_playing_next_round=100,
        )
        features = extract_baseline_features(
            player, make_fixture(), None, None, benchmarks, gameweek=1
        )
        assert features.expected_own_goals == pytest.approx(
            2 / 2700 * 90.0 * 45.0 / 90.0
        )

    def test_projection_no_double_application(self, benchmarks: dict) -> None:
        """Projection factor applied exactly once."""
        player = make_player(
            position=PlayerPosition.MIDFIELDER,
            minutes=2700,
            expected_goals_per_90=0.4,
            expected_minutes=None,
            chance_of_playing_next_round=50,
        )
        features = extract_baseline_features(
            player, make_fixture(), None, None, benchmarks, gameweek=1
        )
        # exp_minutes = 45, projection_factor = 0.5
        # xg = 0.4 * 0.5 = 0.2 (NOT 0.4 * 0.5 * 0.5)
        assert features.expected_goals == pytest.approx(0.2)

    def test_consistency_with_expected_points(self, benchmarks: dict) -> None:
        """Projected fixture-level counts feed correctly into scoring."""
        from fpl_engine.analytics.prediction import (
            PlayerGameweekPrediction,
            predict_expected_points,
        )
        player = make_player(
            position=PlayerPosition.MIDFIELDER,
            minutes=2700,
            expected_goals_per_90=0.5,
            expected_assists_per_90=0.3,
            bonus=20,
            expected_minutes=45,
            chance_of_playing_next_round=100,
        )
        fixture = make_fixture(team_h=14, team_a=1)
        team = make_team(team_id=14, short_name="LIV")
        opponent = make_team(team_id=1, short_name="ARS", name="Arsenal")
        features = extract_baseline_features(
            player, fixture, team, opponent, benchmarks, gameweek=1
        )
        mp = predict_match_baseline(features)
        pgw = PlayerGameweekPrediction(
            player_id=player.id,
            position=player.position,
            gameweek=1,
            fixtures=(mp,),
        )
        result = predict_expected_points(pgw)
        # exp_minutes=45 → projection_factor=0.5
        # prob_60_plus = heuristic(45) * 1.0 = 0.20
        assert result.expected_appearance_points == pytest.approx(
            (1.0 - 0.20) * 1 + 0.20 * 2
        )
        assert result.expected_goal_points == pytest.approx(0.5 * 0.5 * 5)
        assert result.expected_assist_points == pytest.approx(0.3 * 0.5 * 3)
        assert result.expected_bonus_points == pytest.approx(
            20 / 2700 * 90.0 * 45.0 / 90.0
        )

    def test_start_probability_from_chance_of_playing(self, benchmarks: dict) -> None:
        player = make_player(
            position=PlayerPosition.MIDFIELDER, minutes=2700,
            chance_of_playing_next_round=75,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.start_probability == pytest.approx(0.75)
        assert features.probability_appears == pytest.approx(0.75)
        # probability_60_plus = heuristic * start_prob, so must not exceed it
        assert features.probability_60_plus <= features.probability_appears

    def test_prob_60_plus_invariant_not_exceeds_appears(self, benchmarks: dict) -> None:
        """Regression: chance_of_playing < 100 must not let P(60+) exceed P(appears)."""
        player = make_player(
            position=PlayerPosition.MIDFIELDER, minutes=2700,
            chance_of_playing_next_round=75,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert 0.0 <= features.probability_60_plus <= features.probability_appears <= 1.0
        # With expected_minutes=90 default: heuristic=0.95, start_prob=0.75
        assert features.probability_60_plus == pytest.approx(0.95 * 0.75)

    def test_prob_60_plus_invariant_full_appearance(self, benchmarks: dict) -> None:
        """With chance_of_playing=100 and expected_minutes=90, P(60+) = heuristic."""
        player = make_player(
            position=PlayerPosition.MIDFIELDER, minutes=2700,
            chance_of_playing_next_round=100,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.probability_60_plus == pytest.approx(0.95)
        assert features.probability_60_plus <= features.probability_appears

    def test_prob_60_plus_invariant_zero_appearance(self, benchmarks: dict) -> None:
        """With chance_of_playing=0, both P(appears) and P(60+) are 0."""
        player = make_player(
            position=PlayerPosition.MIDFIELDER, minutes=2700,
            chance_of_playing_next_round=0,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.probability_appears == pytest.approx(0.0)
        assert features.probability_60_plus == pytest.approx(0.0)

    def test_prob_60plus_fallback_no_double_dilution(self, benchmarks: dict) -> None:
        """When expected_minutes is None, the P(60+) heuristic must receive
        conditional minutes (90.0), not the diluted unconditional fallback
        (90 * start_prob).  At start_prob=0.75 the fallback exp_minutes=67.5
        would fall in the '>= 60' tier (0.75), but conditional minutes=90.0
        correctly yield the '>= 90' tier (0.95).
        """
        player = make_player(
            position=PlayerPosition.MIDFIELDER, minutes=2700,
            chance_of_playing_next_round=75,
            expected_minutes=None,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        # exp_minutes = 67.5; conditional = 67.5/0.75 = 90.0; heuristic(90)=0.95
        assert features.probability_60_plus == pytest.approx(0.95 * 0.75)

    def test_prob_60plus_zero_appearance_without_expected_minutes(
        self, benchmarks: dict
    ) -> None:
        """With expected_minutes=None and start_prob=0, P(60+)=0 and
        conditional minutes cannot be computed (0/0), so the code falls
        back to heuristic(0)=0."""
        player = make_player(
            position=PlayerPosition.MIDFIELDER, minutes=2700,
            chance_of_playing_next_round=0,
            expected_minutes=None,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.probability_appears == pytest.approx(0.0)
        assert features.probability_60_plus == pytest.approx(0.0)

    def test_prob_60plus_dedilution_with_expected_minutes(self, benchmarks: dict) -> None:
        """When expected_minutes is provided and start_prob < 1, the heuristic
        must receive conditional minutes (expected_minutes / start_prob),
        not the raw unconditional expected_minutes value.

        Scenario: expected_minutes=45 (unconditional, like FPL Analyst's
        xMins), chance_of_playing_next_round=75 (start_prob=0.75).

        Conditional minutes = 45 / 0.75 = 60.0
        heuristic(60.0) = 0.75  (>= 60 min tier)
        P(60+) = 0.75 * 0.75 = 0.5625

        Without de-dilution: heuristic(45) = 0.20  (>= 30 min tier)
        P(60+) = 0.20 * 0.75 = 0.15

        The fix ensures P(60+) = 0.5625, not 0.15.
        """
        player = make_player(
            position=PlayerPosition.MIDFIELDER, minutes=2700,
            expected_minutes=45,
            chance_of_playing_next_round=75,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.start_probability == pytest.approx(0.75)
        assert features.probability_appears == pytest.approx(0.75)
        # Conditional minutes = 45/0.75 = 60.0; heuristic(60) = 0.75
        assert features.probability_60_plus == pytest.approx(0.75 * 0.75)

    def test_prob_60plus_dedilution_crosses_90min_tier(self, benchmarks: dict) -> None:
        """When de-dilution pushes conditional minutes into the >=90 tier,
        the heuristic must use 0.95, not the lower >=60 tier value of 0.75.

        Scenario: expected_minutes=80 (unconditional, like FPL Analyst's xMins),
        chance_of_playing_next_round=80 (start_prob=0.8).

        Conditional minutes = 80 / 0.8 = 100.0
        heuristic(100.0) = 0.95  (>= 90 min tier)
        P(60+) = 0.95 * 0.8 = 0.76

        Without de-dilution: heuristic(80) = 0.75  (>= 60, < 90 min tier)
        P(60+) = 0.75 * 0.8 = 0.60

        The fix ensures P(60+) = 0.76, not 0.60.
        """
        player = make_player(
            position=PlayerPosition.MIDFIELDER, minutes=2700,
            expected_minutes=80,
            chance_of_playing_next_round=80,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.start_probability == pytest.approx(0.8)
        assert features.probability_appears == pytest.approx(0.8)
        # Conditional minutes = 80/0.8 = 100.0; heuristic(100) = 0.95
        assert features.probability_60_plus == pytest.approx(0.95 * 0.8)

    def test_prob_60plus_expected_minutes_with_full_availability(self, benchmarks: dict) -> None:
        """With start_prob=1.0, de-dilution is a no-op: conditional = unconditional.
        """
        player = make_player(
            position=PlayerPosition.MIDFIELDER, minutes=2700,
            expected_minutes=45,
            chance_of_playing_next_round=100,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        # start_prob=1.0: conditional = 45/1.0 = 45; heuristic(45) = 0.20
        assert features.probability_60_plus == pytest.approx(0.20 * 1.0)

    def test_predict_match_baseline_with_partial_appearance(self, benchmarks: dict) -> None:
        """End-to-end: predict_match_baseline must not crash when P(appears) < 1."""
        player = make_player(
            position=PlayerPosition.MIDFIELDER, minutes=2700,
            expected_goals_per_90=0.5, expected_assists_per_90=0.3,
            bonus=20, chance_of_playing_next_round=75,
        )
        fixture = make_fixture(team_h=14, team_a=1)
        team = make_team(team_id=14)
        opponent = make_team(team_id=1, short_name="ARS", name="Arsenal")
        features = extract_baseline_features(
            player, fixture, team, opponent, benchmarks, gameweek=1
        )
        prediction = predict_match_baseline(features)
        assert isinstance(prediction, MatchPrediction)
        # The invariant must hold in the MatchPrediction as well
        assert prediction.probability_60_plus <= prediction.probability_appears

    def test_zero_availability_yields_zero_expected_points(self, benchmarks: dict) -> None:
        """End-to-end: contradictory inputs (expected_minutes=30, start_prob=0)
        must produce zero expected points at the scoring level."""
        from fpl_engine.analytics.prediction import (
            PlayerGameweekPrediction,
            predict_expected_points,
        )
        player = make_player(
            position=PlayerPosition.MIDFIELDER, minutes=2700,
            expected_goals_per_90=0.7, expected_assists_per_90=0.5,
            bonus=40, expected_minutes=30,
            chance_of_playing_next_round=0,
        )
        fixture = make_fixture(team_h=14, team_a=1)
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        mp = predict_match_baseline(features)
        pgw = PlayerGameweekPrediction(
            player_id=player.id, position=player.position,
            gameweek=1, fixtures=(mp,),
        )
        result = predict_expected_points(pgw)
        assert result.total_expected_points == pytest.approx(0.0)
        assert result.expected_goal_points == pytest.approx(0.0)
        assert result.expected_assist_points == pytest.approx(0.0)
        assert result.expected_bonus_points == pytest.approx(0.0)
        assert result.expected_appearance_points == pytest.approx(0.0)

    def test_is_home_false(self, benchmarks: dict) -> None:
        player = make_player(
            player_id=10, team_id=1,
            position=PlayerPosition.MIDFIELDER, minutes=2700,
        )
        fixture = make_fixture(team_h=14, team_a=1)
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.is_home is False

    def test_cutoff_gameweek_passed_through(self, benchmarks: dict) -> None:
        player = make_player(position=PlayerPosition.MIDFIELDER, minutes=2700)
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=5,
            cutoff_gameweek=3,
        )
        assert features.cutoff_gameweek == 3

    def test_cutoff_gameweek_defaults_none(self, benchmarks: dict) -> None:
        player = make_player(position=PlayerPosition.MIDFIELDER, minutes=2700)
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=5,
        )
        assert features.cutoff_gameweek is None

    def test_expected_minutes_passed_through(self, benchmarks: dict) -> None:
        player = make_player(
            position=PlayerPosition.MIDFIELDER, minutes=2700,
            expected_minutes=60,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.expected_minutes == pytest.approx(60.0)

    def test_matches_played_computed(self, benchmarks: dict) -> None:
        player = make_player(
            position=PlayerPosition.MIDFIELDER, minutes=2700,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.matches_played == pytest.approx(30.0)

    def test_always_zero_fields(self, benchmarks: dict) -> None:
        player = make_player(position=PlayerPosition.MIDFIELDER, minutes=2700)
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.expected_yellow_cards == 0.0
        assert features.expected_red_cards == 0.0
        assert features.expected_penalties_saved == 0.0
        assert features.expected_penalties_missed == 0.0

    def test_defensive_contribution_probability_gkp_zero(self, benchmarks: dict) -> None:
        """Goalkeeper is ineligible for DC points (always 0.0)."""
        player = make_player(position=PlayerPosition.GOALKEEPER, minutes=2700)
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.defensive_contribution_probability == 0.0

    @pytest.mark.parametrize(
        ("position", "expected_prior"),
        [
            (PlayerPosition.DEFENDER, 0.35),
            (PlayerPosition.MIDFIELDER, 0.25),
            (PlayerPosition.FORWARD, 0.15),
        ],
    )
    def test_dc_probability_prior_fallback(
        self, benchmarks: dict, position: PlayerPosition, expected_prior: float
    ) -> None:
        """Without history, DC probability falls back to position prior."""
        player = make_player(position=position, minutes=2700)
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.defensive_contribution_probability == pytest.approx(expected_prior)

    def test_benchmark_fallback_when_benchmarks_empty(self) -> None:
        player = make_player(
            position=PlayerPosition.MIDFIELDER, minutes=2700,
            expected_goals_per_90=0.5, expected_assists_per_90=0.3,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, {}, gameweek=1
        )
        assert features.expected_goals == pytest.approx(0.5)
        assert features.expected_assists == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Production-path tests: expected_minutes=None fallback
#
# These tests exercise the production fallback path in
# _resolve_expected_minutes, which fires when the FPL API does not supply
# expected_minutes (it never does — confirmed via live bootstrap-static
# fetch: absent from element_stats, element keys, and game_config).
# The function therefore always falls back to 90.0 * start_probability.
#
# All expected values are computed by hand using documented FPL constants
# and independently verified arithmetic.  No implementation formula is
# duplicated.
# ---------------------------------------------------------------------------


class TestProductionFallback:
    """Tests for the expected_minutes=None production fallback path.

    The FPL API does not expose an ``expected_minutes`` field (confirmed
    via live bootstrap-static fetch — absent from ``element_stats``,
    ``element_types``, and ``game_config``).  The
    :func:`_resolve_expected_minutes` function therefore always falls
    back to ``90.0 * start_prob`` in production.

    These tests independently verify that fallback path produces correct
    values, computed from first principles using documented FPL constants
    (not by replaying the implementation code).
    """

    @pytest.fixture
    def benchmarks(self) -> dict:
        """Position benchmarks with known values (same as TestExtractBaselineFeatures)."""
        return compute_position_benchmarks([
            make_player(
                player_id=1, position=PlayerPosition.MIDFIELDER,
                minutes=2700, expected_goals=20.0, expected_assists=15.0,
                bonus=40,
            ),
        ])

    # D1: Full availability via chance_of_playing with expected_minutes=None
    def test_full_availability_expected_minutes_none(self, benchmarks: dict) -> None:
        """Full availability (chance_of_playing=100) with expected_minutes=None.

        Production fallback: exp_minutes = 90 * 1.0 = 90.0.  Projection
        factor = 90.0 / 90.0 = 1.0.  De-diluted conditional minutes =
        90.0 / 1.0 = 90.0.  heuristic(90) = 0.95.

        Expected event counts = per-90 rates * projection_factor (1.0),
        i.e. the raw per-90 values with no scaling applied.
        """
        player = make_player(
            position=PlayerPosition.MIDFIELDER,
            minutes=2700,
            expected_goals_per_90=0.7,
            expected_assists_per_90=0.5,
            bonus=40,
            expected_minutes=None,
            chance_of_playing_next_round=100,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        # start_probability from chance_of_playing=100
        assert features.start_probability == pytest.approx(1.0)
        # Fallback: 90 * 1.0 = 90.0
        assert features.expected_minutes == pytest.approx(90.0)
        # probability_appears = start_probability = 1.0
        assert features.probability_appears == pytest.approx(1.0)
        # De-dilution: conditional = 90/1.0 = 90; heuristic(90) = 0.95; *1.0
        assert features.probability_60_plus == pytest.approx(0.95)
        # Event counts: per-90 * projection_factor (1.0)
        assert features.expected_goals == pytest.approx(0.7)
        assert features.expected_assists == pytest.approx(0.5)
        # bonus_per_90 = 40/2700*90 = 1.3333; * projection_factor (1.0); capped at 3.0
        assert features.expected_bonus == pytest.approx(40.0 / 2700.0 * 90.0)
        # own_goals=0 -> 0.0
        assert features.expected_own_goals == pytest.approx(0.0)
        # matches_played = 2700/90 = 30.0
        assert features.matches_played == pytest.approx(30.0)

    # D2: Status-based fallback (AVAILABLE) with expected_minutes=None
    def test_status_fallback_available_expected_minutes_none(self, benchmarks: dict) -> None:
        """Status-based fallback: chance_of_playing=None, status=AVAILABLE.

        This is the most common production state: the FPL API returns
        ``chance_of_playing_next_round=None`` and ``expected_minutes=None``
        for fit players.  ``status='a'`` (AVAILABLE) yields start_prob=1.0,
        so the result is identical to full availability.
        """
        player = make_player(
            position=PlayerPosition.MIDFIELDER,
            minutes=2700,
            expected_goals_per_90=0.7,
            expected_assists_per_90=0.5,
            bonus=40,
            expected_minutes=None,
            chance_of_playing_next_round=None,
            status=PlayerStatus.AVAILABLE,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        # status=AVAILABLE -> start_prob=1.0
        assert features.start_probability == pytest.approx(1.0)
        # Fallback: 90 * 1.0 = 90.0
        assert features.expected_minutes == pytest.approx(90.0)
        assert features.probability_appears == pytest.approx(1.0)
        # heuristic(90) = 0.95; * 1.0
        assert features.probability_60_plus == pytest.approx(0.95)
        # Event counts: per-90 * projection_factor (1.0)
        assert features.expected_goals == pytest.approx(0.7)
        assert features.expected_assists == pytest.approx(0.5)

    # D3: Status-based fallback (DOUBTFUL) with expected_minutes=None
    def test_status_fallback_doubtful_expected_minutes_none(self, benchmarks: dict) -> None:
        """Status-based fallback: chance_of_playing=None, status=DOUBTFUL.

        ``status='d'`` (DOUBTFUL) yields start_prob=0.5.  Fallback:
        exp_minutes = 90 * 0.5 = 45.0.  Projection factor = 0.5.
        De-diluted conditional minutes = 45.0 / 0.5 = 90.0; heuristic(90) = 0.95.
        P(60+) = 0.95 * 0.5 = 0.475.

        Expected event counts = per-90 rates * projection_factor (0.5).
        """
        player = make_player(
            position=PlayerPosition.MIDFIELDER,
            minutes=2700,
            expected_goals_per_90=0.7,
            expected_assists_per_90=0.5,
            bonus=40,
            expected_minutes=None,
            chance_of_playing_next_round=None,
            status=PlayerStatus.DOUBTFUL,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        # status=DOUBTFUL -> start_prob=0.5
        assert features.start_probability == pytest.approx(0.5)
        # Fallback: 90 * 0.5 = 45.0
        assert features.expected_minutes == pytest.approx(45.0)
        assert features.probability_appears == pytest.approx(0.5)
        # De-dilution: conditional = 45/0.5 = 90; heuristic(90) = 0.95; * 0.5
        assert features.probability_60_plus == pytest.approx(0.95 * 0.5)
        # Event counts: per-90 * projection_factor (0.5)
        assert features.expected_goals == pytest.approx(0.7 * 0.5)
        assert features.expected_assists == pytest.approx(0.5 * 0.5)
        # bonus_per_90 = 40/2700*90 = 1.3333; * 0.5 = 0.6667; capped at 3.0
        assert features.expected_bonus == pytest.approx(40.0 / 2700.0 * 90.0 * 0.5)

    # D4: Zero availability with expected_minutes=None
    def test_zero_availability_expected_minutes_none(self, benchmarks: dict) -> None:
        """Zero availability (chance_of_playing=0) with expected_minutes=None.

        start_prob=0 -> exp_minutes=0, projection_factor=0, P(60+)=0.
        All playing-dependent event counts must be zero.
        """
        player = make_player(
            position=PlayerPosition.MIDFIELDER,
            minutes=2700,
            expected_goals_per_90=0.7,
            expected_assists_per_90=0.5,
            bonus=40,
            expected_minutes=None,
            chance_of_playing_next_round=0,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        # start_prob via chance_of_playing=0
        assert features.start_probability == pytest.approx(0.0)
        # Fallback: 90 * 0.0 = 0.0
        assert features.expected_minutes == pytest.approx(0.0)
        assert features.probability_appears == pytest.approx(0.0)
        # start_prob=0 -> P(60+) = 0 (short-circuit, no heuristic call)
        assert features.probability_60_plus == pytest.approx(0.0)
        # All playing-dependent counts suppressed to zero
        assert features.expected_goals == pytest.approx(0.0)
        assert features.expected_assists == pytest.approx(0.0)
        assert features.expected_bonus == pytest.approx(0.0)
        assert features.expected_own_goals == pytest.approx(0.0)

    # D5: Event-count assertions — no double dilution through full pipeline
    def test_event_counts_not_double_diluted(self, benchmarks: dict) -> None:
        """Event counts must NOT be multiplied by probability_60_plus.

        This is the "no double dilution" invariant: ``expected_goals``,
        ``expected_assists``, and ``expected_bonus`` are projected using
        ``projection_factor`` (= exp_minutes / 90, which already
        incorporates start_prob), NOT additionally multiplied by
        ``probability_60_plus``.

        Verified end-to-end through ``predict_expected_points``: if double
        dilution occurred, ``goal_pts`` would be ``0.7 * 0.95 * 5`` instead
        of ``0.7 * 5``.
        """
        from fpl_engine.analytics.prediction import (
            PlayerGameweekPrediction,
            predict_expected_points,
        )

        player = make_player(
            position=PlayerPosition.MIDFIELDER,
            minutes=2700,
            expected_goals_per_90=0.7,
            expected_assists_per_90=0.5,
            bonus=40,
            expected_minutes=None,
            chance_of_playing_next_round=100,
        )
        fixture = make_fixture(team_h=14, team_a=1)
        team = make_team(team_id=14, short_name="LIV")
        opponent = make_team(team_id=1, short_name="ARS", name="Arsenal")
        features = extract_baseline_features(
            player, fixture, team, opponent, benchmarks, gameweek=1
        )

        # --- Feature-level assertions (no double dilution) ---
        # projection_factor = 1.0 (start_prob=1.0); NOT multiplied by P(60+)=0.95
        assert features.expected_goals == pytest.approx(0.7)
        assert features.expected_assists == pytest.approx(0.5)
        assert features.expected_bonus == pytest.approx(40.0 / 2700.0 * 90.0)

        # --- End-to-end through scoring engine ---
        mp = predict_match_baseline(features)
        pgw = PlayerGameweekPrediction(
            player_id=player.id,
            position=player.position,
            gameweek=1,
            fixtures=(mp,),
        )
        result = predict_expected_points(pgw)

        # Independently calculated expected points (MID, full availability):
        # start_prob=1.0, exp_minutes=90, projection_factor=1.0
        # P(60+) = 0.95 * 1.0 = 0.95
        #
        # appearance_pts: P(<60 min) * 1 + P(>=60 min) * 2
        #   = (1.0 - 0.95) * 1 + 0.95 * 2 = 0.05 + 1.90 = 1.95
        appearance_pts = (1.0 - 0.95) * 1 + 0.95 * 2
        # goal_pts: 0.7 * 5 = 3.5  (NOT 0.7 * 0.95 * 5 = 3.325)
        goal_pts = 0.7 * 5
        # assist_pts: 0.5 * 3 = 1.5  (NOT 0.5 * 0.95 * 3 = 1.425)
        assist_pts = 0.5 * 3

        # Clean sheet probability via Poisson model (independently computed)
        lam = _project_goals_conceded_rate(team, opponent, is_home=True)
        cs_prob = math.exp(-lam)
        # cs_pts = P(60+) * P(CS) * CS_points(MID=1)
        cs_pts = 0.95 * cs_prob * 1

        # bonus_pts: bonus_per_90 * projection_factor(1.0), capped at 3.0
        bonus_pts = min(40.0 / 2700.0 * 90.0 * 1.0, MAX_BONUS_PER_MATCH)

        # dc_pts = P(appear) * P(DC | appear) * DC_points
        # No history -> DC probability falls back to MID prior (0.25).
        dc_prob = 0.25
        dc_pts = 1.0 * dc_prob * 2.0

        # Verify no double dilution in scoring output
        assert result.expected_appearance_points == pytest.approx(appearance_pts)
        assert result.expected_goal_points == pytest.approx(goal_pts)
        assert result.expected_assist_points == pytest.approx(assist_pts)
        assert result.expected_bonus_points == pytest.approx(bonus_pts)

        # gc_pts=0 (MID not eligible), save_pts=0 (not GK), dc_pts computed above
        expected_total = (
            appearance_pts + goal_pts + assist_pts + cs_pts + bonus_pts + dc_pts
        )
        assert result.total_expected_points == pytest.approx(expected_total)


# ---------------------------------------------------------------------------
# predict_match_baseline tests
# ---------------------------------------------------------------------------


class TestPredictMatchBaseline:
    """Tests for predict_match_baseline function."""

    def _make_features(self, **kwargs) -> BaselineFeatures:
        """Create BaselineFeatures with sensible defaults."""
        defaults = {
            "player_id": 1, "fixture_id": 1, "gameweek": 1,
            "position": PlayerPosition.MIDFIELDER, "is_home": True,
            "start_probability": 1.0, "probability_appears": 1.0,
            "probability_60_plus": 0.95, "clean_sheet_probability": None,
            "expected_goals": 0.5, "expected_assists": 0.3,
            "expected_goals_conceded": None, "expected_saves": None,
            "expected_bonus": None, "expected_own_goals": None,
            "expected_yellow_cards": 0.0, "expected_red_cards": 0.0,
            "expected_penalties_saved": 0.0, "expected_penalties_missed": 0.0,
            "defensive_contribution_probability": 0.0,
            "defensive_contribution_probability_from_history": True,
            "cutoff_gameweek": None, "expected_minutes": 90.0,
            "matches_played": 30.0,
        }
        defaults.update(kwargs)
        return BaselineFeatures(**defaults)

    def test_returns_match_prediction(self) -> None:
        features = self._make_features()
        result = predict_match_baseline(features)
        assert isinstance(result, MatchPrediction)
        assert result.fixture_id == 1
        assert result.gameweek == 1

    def test_none_resolved_to_zero(self) -> None:
        features = self._make_features(
            clean_sheet_probability=None,
            expected_goals=None,
            expected_assists=None,
            expected_goals_conceded=None,
            expected_saves=None,
            expected_bonus=None,
            expected_own_goals=None,
        )
        result = predict_match_baseline(features)
        assert result.clean_sheet_probability == 0.0
        assert result.expected_goals == 0.0
        assert result.expected_assists == 0.0
        assert result.expected_goals_conceded == 0.0
        assert result.expected_saves == 0.0
        assert result.expected_bonus == 0.0
        assert result.expected_own_goals == 0.0

    def test_non_none_values_preserved(self) -> None:
        features = self._make_features(
            clean_sheet_probability=0.5,
            expected_goals=1.2,
            expected_assists=0.4,
            expected_bonus=1.5,
        )
        result = predict_match_baseline(features)
        assert result.clean_sheet_probability == pytest.approx(0.5)
        assert result.expected_goals == pytest.approx(1.2)
        assert result.expected_assists == pytest.approx(0.4)
        assert result.expected_bonus == pytest.approx(1.5)

    def test_always_zero_fields_preserved(self) -> None:
        features = self._make_features()
        result = predict_match_baseline(features)
        assert result.expected_yellow_cards == 0.0
        assert result.expected_red_cards == 0.0
        assert result.expected_penalties_saved == 0.0
        assert result.expected_penalties_missed == 0.0

    def test_probabilities_preserved(self) -> None:
        features = self._make_features(
            start_probability=0.8,
            probability_appears=0.8,
            probability_60_plus=0.6,
        )
        result = predict_match_baseline(features)
        assert result.start_probability == pytest.approx(0.8)
        assert result.probability_appears == pytest.approx(0.8)
        assert result.probability_60_plus == pytest.approx(0.6)

    def test_rules_parameter_accepted(self) -> None:
        """The rules parameter should be accepted (even if unused)."""
        from fpl_engine.analytics.scoring import FPLScoringRules
        features = self._make_features()
        result = predict_match_baseline(features, rules=FPLScoringRules())
        assert isinstance(result, MatchPrediction)

    def test_frozen_features_not_modified(self) -> None:
        """Features should remain unchanged after prediction."""
        features = self._make_features(
            clean_sheet_probability=None,
            expected_goals=1.0,
        )
        predict_match_baseline(features)
        # frozen dataclass - verify values untouched
        assert features.expected_goals == 1.0
        assert features.clean_sheet_probability is None


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end integration tests for the full pipeline."""

    def test_benchmarks_to_features_to_prediction(self) -> None:
        """Full pipeline: players -> benchmarks -> features -> prediction."""
        players = [
            make_player(
                player_id=1, position=PlayerPosition.MIDFIELDER,
                minutes=2700, expected_goals=20.0, expected_assists=15.0,
                bonus=40, expected_goals_per_90=0.7, expected_assists_per_90=0.5,
            ),
        ]
        benchmarks = compute_position_benchmarks(players)

        player = make_player(
            player_id=10, position=PlayerPosition.MIDFIELDER,
            minutes=2700,
            expected_goals_per_90=0.5,
            expected_assists_per_90=0.3,
            bonus=20,
            chance_of_playing_next_round=100,
        )
        fixture = make_fixture(team_h=14, team_a=1)
        team = make_team(team_id=14, short_name="LIV")
        opponent = make_team(
            team_id=1, short_name="ARS", name="Arsenal",
        )

        features = extract_baseline_features(
            player, fixture, team, opponent, benchmarks, gameweek=1
        )
        prediction = predict_match_baseline(features)

        assert isinstance(prediction, MatchPrediction)
        assert prediction.gameweek == 1
        assert prediction.fixture_id == fixture.id
        assert prediction.clean_sheet_probability is not None
        assert 0.0 < prediction.clean_sheet_probability < 1.0
        assert prediction.expected_goals == pytest.approx(0.5)
        # DC provenance: MID with no history → prior fallback (from_history=False)
        assert (
            prediction.defensive_contribution_probability_from_history
            is False
        )
        """Pipeline feeds into predict_expected_points for scoring."""
        from fpl_engine.analytics.prediction import (
            PlayerGameweekPrediction,
            predict_expected_points,
        )
        players = [
            make_player(
                player_id=1, position=PlayerPosition.MIDFIELDER,
                minutes=2700, expected_goals=20.0, expected_goals_per_90=0.7,
            ),
        ]
        benchmarks = compute_position_benchmarks(players)

        player = make_player(
            player_id=10, position=PlayerPosition.MIDFIELDER,
            minutes=2700,
            expected_goals_per_90=0.5,
        )
        fixture = make_fixture(team_h=14, team_a=1)
        team = make_team(team_id=14, short_name="LIV")
        opponent = make_team(team_id=1, short_name="ARS", name="Arsenal")

        features = extract_baseline_features(
            player, fixture, team, opponent, benchmarks, gameweek=1
        )
        mp = predict_match_baseline(features)

        pgw = PlayerGameweekPrediction(
            player_id=player.id,
            position=player.position,
            gameweek=1,
            fixtures=(mp,),
        )
        result = predict_expected_points(pgw)
        assert result.total_expected_points > 0.0
        assert result.fixture_count == 1

    def test_pipeline_without_teams(self) -> None:
        """Pipeline works without team data (CS prob resolved to 0)."""
        players = [
            make_player(
                player_id=1, position=PlayerPosition.MIDFIELDER,
                minutes=2700, expected_goals=20.0, expected_goals_per_90=0.7,
            ),
        ]
        benchmarks = compute_position_benchmarks(players)

        player = make_player(
            player_id=10, position=PlayerPosition.MIDFIELDER,
            minutes=2700, expected_goals_per_90=0.5,
        )
        fixture = make_fixture()

        features = extract_baseline_features(
            player, fixture, team=None, opponent=None,
            benchmarks=benchmarks, gameweek=2,
        )
        assert features.clean_sheet_probability is None

        mp = predict_match_baseline(features)
        assert mp.clean_sheet_probability == 0.0
        assert mp.expected_goals == pytest.approx(0.5)

    def test_defender_xgc_computed(self) -> None:
        """Defenders should have expected_goals_conceded populated."""
        players = [
            make_player(
                player_id=1, position=PlayerPosition.DEFENDER,
                minutes=2700, expected_goals_conceded=30.0,
                expected_goals_conceded_per_90=1.0,
            ),
        ]
        benchmarks = compute_position_benchmarks(players)

        player = make_player(
            player_id=10, position=PlayerPosition.DEFENDER,
            minutes=2700, expected_goals_conceded_per_90=0.8,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.expected_goals_conceded is not None
        assert features.expected_goals_conceded == pytest.approx(0.8)

    def test_forward_xgc_none(self) -> None:
        """Forwards should never have expected_goals_conceded."""
        players = [
            make_player(
                player_id=1, position=PlayerPosition.FORWARD,
                minutes=2700, expected_goals=15.0, expected_goals_per_90=0.5,
            ),
        ]
        benchmarks = compute_position_benchmarks(players)

        player = make_player(
            player_id=10, position=PlayerPosition.FORWARD,
            minutes=2700, expected_goals_per_90=0.5,
        )
        fixture = make_fixture()
        features = extract_baseline_features(
            player, fixture, None, None, benchmarks, gameweek=1
        )
        assert features.expected_goals_conceded is None

    def test_dgw_end_to_end_expected_points(self) -> None:
        """D5: double-gameweek end-to-end with manual verification.

        Two fixtures for the same player are built through the full
        baseline pipeline (benchmarks → features → MatchPrediction →
        PlayerGameweekPrediction → predict_expected_points).  The
        result is verified against a hand-computed sum of fixture-level
        contributions, confirming that fixture-specific projections are
        not combined incorrectly and probabilities are not multiplied
        twice.
        """
        from fpl_engine.analytics.prediction import (
            PlayerGameweekPrediction,
            predict_expected_points,
        )
        from fpl_engine.analytics.scoring import DEFAULT_SCORING_RULES

        players = [
            make_player(
                player_id=1, position=PlayerPosition.MIDFIELDER,
                minutes=2700, expected_goals=20.0, expected_goals_per_90=0.7,
                expected_assists=15.0, expected_assists_per_90=0.5,
                bonus=40,
            ),
        ]
        benchmarks = compute_position_benchmarks(players)

        player = make_player(
            player_id=10, position=PlayerPosition.MIDFIELDER,
            minutes=2700,
            expected_goals_per_90=0.7,
            expected_assists_per_90=0.5,
            bonus=40,
            chance_of_playing_next_round=100,
            expected_minutes=90,
        )
        team = make_team(team_id=14, short_name="LIV")
        opponent = make_team(team_id=1, short_name="ARS", name="Arsenal")

        # Fixture 1 (home for LIV)
        fixture1 = make_fixture(fixture_id=1, team_h=14, team_a=1, event=5)
        features1 = extract_baseline_features(
            player, fixture1, team, opponent, benchmarks, gameweek=5
        )
        mp1 = predict_match_baseline(features1)

        # Fixture 2 (also home for LIV — same teams, different fixture_id)
        fixture2 = make_fixture(fixture_id=2, team_h=14, team_a=1, event=5)
        features2 = extract_baseline_features(
            player, fixture2, team, opponent, benchmarks, gameweek=5
        )
        mp2 = predict_match_baseline(features2)

        # Both fixtures share the same player/team → identical projections
        assert mp1.fixture_id == 1
        assert mp2.fixture_id == 2
        assert mp1.expected_goals == pytest.approx(mp2.expected_goals)
        assert mp1.expected_bonus == pytest.approx(mp2.expected_bonus)

        pgw = PlayerGameweekPrediction(
            player_id=player.id,
            position=player.position,
            gameweek=5,
            fixtures=(mp1, mp2),
        )
        result = predict_expected_points(pgw)

        # --- Manual calculation for ONE fixture ---
        r = DEFAULT_SCORING_RULES
        pos = PlayerPosition.MIDFIELDER
        proj_factor = 90.0 / 90.0  # exp_minutes=90, start_prob=1.0

        xg = 0.7 * proj_factor
        xa = 0.5 * proj_factor
        bonus = min(40 / 2700 * 90.0 * proj_factor, MAX_BONUS_PER_MATCH)
        prob_appears = 1.0
        prob_60_plus = 0.95 * 1.0  # heuristic(90) = 0.95

        # CS probability (Poisson model on team strengths)
        lam = _project_goals_conceded_rate(team, opponent, is_home=True)
        cs_prob = math.exp(-lam)

        appearance_pts = (prob_appears - prob_60_plus) * 1 + prob_60_plus * 2
        goal_pts = xg * r.goal_points[pos]
        assist_pts = xa * r.assist_points
        cs_pts = prob_60_plus * cs_prob * r.clean_sheet_points[pos]
        gc_pts = 0.0  # midfielder not eligible
        save_pts = 0.0  # not GK
        bonus_pts = bonus
        # DC: P(appear) * P(DC | appear) * DC_points; no history -> MID prior 0.25
        dc_pts = prob_appears * 0.25 * r.defensive_contribution_points

        fixture_total = (
            appearance_pts + goal_pts + assist_pts + cs_pts + gc_pts
            + save_pts + bonus_pts + dc_pts
        )

        # DGW = 2 * fixture_total (both fixtures identical)
        assert result.fixture_count == 2
        assert result.expected_appearance_points == pytest.approx(2 * appearance_pts)
        assert result.expected_goal_points == pytest.approx(2 * goal_pts)
        assert result.expected_assist_points == pytest.approx(2 * assist_pts)
        assert result.expected_clean_sheet_points == pytest.approx(2 * cs_pts)
        assert result.expected_goals_conceded_points == pytest.approx(2 * gc_pts)
        assert result.expected_save_points == pytest.approx(2 * save_pts)
        assert result.expected_bonus_points == pytest.approx(2 * bonus_pts)
        assert result.total_expected_points == pytest.approx(2 * fixture_total)

        # Probabilities must NOT be multiplied twice:
        # prob_60_plus should appear only in appearance and CS, not in goals.
        assert mp1.probability_60_plus == pytest.approx(0.95)
        assert mp1.expected_goals == pytest.approx(0.7)  # not 0.7 * 0.95


# ---------------------------------------------------------------------------
# Module exports tests
# ---------------------------------------------------------------------------


class TestBaselineModelError:
    """Tests for the BaselineModelError exception."""

    def test_is_value_error(self) -> None:
        from fpl_engine.analytics.baseline_models import BaselineModelError
        assert issubclass(BaselineModelError, ValueError)

    def test_can_raise_and_catch(self) -> None:
        from fpl_engine.analytics.baseline_models import BaselineModelError
        with pytest.raises(BaselineModelError):
            raise BaselineModelError("test")


class TestModuleExports:
    """Tests for module __all__ and public exports."""

    def test_all_contains_position_benchmark(self) -> None:
        import fpl_engine.analytics.baseline_models as bm
        assert "PositionBenchmark" in bm.__all__

    def test_all_contains_baseline_features(self) -> None:
        import fpl_engine.analytics.baseline_models as bm
        assert "BaselineFeatures" in bm.__all__

    def test_all_contains_compute_position_benchmarks(self) -> None:
        import fpl_engine.analytics.baseline_models as bm
        assert "compute_position_benchmarks" in bm.__all__

    def test_all_contains_extract_baseline_features(self) -> None:
        import fpl_engine.analytics.baseline_models as bm
        assert "extract_baseline_features" in bm.__all__

    def test_all_contains_predict_match_baseline(self) -> None:
        import fpl_engine.analytics.baseline_models as bm
        assert "predict_match_baseline" in bm.__all__

    def test_all_contains_baseline_model_error(self) -> None:
        import fpl_engine.analytics.baseline_models as bm
        assert "BaselineModelError" in bm.__all__

    def test_importable_from_analytics_package(self) -> None:
        from fpl_engine.analytics import (
            BaselineFeatures,
            BaselineModelError,
            PositionBenchmark,
            compute_position_benchmarks,
            extract_baseline_features,
            predict_match_baseline,
        )
        assert callable(compute_position_benchmarks)
        assert callable(extract_baseline_features)
        assert callable(predict_match_baseline)
        assert isinstance(BaselineModelError, type)
        assert isinstance(PositionBenchmark, type)
        assert isinstance(BaselineFeatures, type)


# ---------------------------------------------------------------------------
# Temporal-safety: extract_baseline_features with prediction_cutoff
# ---------------------------------------------------------------------------


class TestExtractBaselineFeaturesTemporalSafety:
    """Tests for prediction_cutoff enforcement in extract_baseline_features.

    The DC probability is the only component that consumes per-gameweek
    history, so the cutoff filtering is most impactful there.  These tests
    verify that the cutoff is validated, forwarded to the DC model, and
    correctly filters history records.
    """

    @pytest.fixture
    def benchmarks(self) -> dict:
        """Position benchmarks with known values."""
        return compute_position_benchmarks([
            make_player(
                player_id=1, position=PlayerPosition.MIDFIELDER,
                minutes=2700, expected_goals=20.0, expected_assists=15.0,
                bonus=40,
            ),
        ])

    @staticmethod
    def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
        return datetime(year, month, day, hour, tzinfo=UTC)

    def test_naive_cutoff_raises(self, benchmarks: dict) -> None:
        """A tz-naive prediction_cutoff raises BaselineModelError."""
        from fpl_engine.analytics.baseline_models import BaselineModelError

        player = make_player(position=PlayerPosition.MIDFIELDER, minutes=2700)
        with pytest.raises(BaselineModelError, match="timezone-aware"):
            extract_baseline_features(
                player, make_fixture(), None, None, benchmarks,
                gameweek=1, history=[],
                prediction_cutoff=datetime(2024, 9, 15, 12),  # noqa: DTZ001
            )

    def test_non_utc_cutoff_raises(self, benchmarks: dict) -> None:
        """A non-UTC timezone on prediction_cutoff raises BaselineModelError."""
        from fpl_engine.analytics.baseline_models import BaselineModelError

        cutoff = datetime(
            2024, 9, 15, 12, tzinfo=timezone(__import__("datetime").timedelta(hours=5)),
        )
        player = make_player(position=PlayerPosition.MIDFIELDER, minutes=2700)
        with pytest.raises(BaselineModelError, match="UTC"):
            extract_baseline_features(
                player, make_fixture(), None, None, benchmarks,
                gameweek=1, history=[], prediction_cutoff=cutoff,
            )

    def test_no_cutoff_backward_compat(self, benchmarks: dict) -> None:
        """Without prediction_cutoff, all prior-gameweek history is used."""
        from fpl_engine.models import PlayerMatchHistory

        player = make_player(
            position=PlayerPosition.MIDFIELDER, minutes=2700,
            expected_goals_per_90=0.5,
        )
        history = [
            PlayerMatchHistory(
                player_id=1, fixture_id=i, gameweek=i, minutes=90,
                tackles=5, clearances_blocks_interceptions=5,
                recoveries=5,
            )
            for i in range(1, 5)
        ]
        features = extract_baseline_features(
            player, make_fixture(), None, None, benchmarks,
            gameweek=5, history=history,
        )
        # MID with 4 prior appearances meeting DC threshold (CBIRT=15 >= 12)
        # -> from_history = True (shrunk prior)
        assert features.defensive_contribution_probability_from_history is True

    def test_cutoff_filters_history_excludes_future(self, benchmarks: dict) -> None:
        """History records with outcome_time after cutoff are excluded,
        yielding prior fallback when all records are post-cutoff."""
        from fpl_engine.analytics.defensive_probability import (
            DC_POSITION_PRIOR,
        )
        from fpl_engine.models import PlayerMatchHistory

        player = make_player(
            position=PlayerPosition.MIDFIELDER, minutes=2700,
            expected_goals_per_90=0.5,
        )
        future = self._utc(2024, 10, 1)
        history = [
            PlayerMatchHistory(
                player_id=1, fixture_id=i, gameweek=i, minutes=90,
                tackles=5, clearances_blocks_interceptions=5,
                recoveries=5, outcome_time=future,
            )
            for i in range(1, 5)
        ]
        cutoff = self._utc(2024, 9, 15)
        features = extract_baseline_features(
            player, make_fixture(), None, None, benchmarks,
            gameweek=5, history=history, prediction_cutoff=cutoff,
        )
        # All history excluded -> DC falls back to prior (0.25 for MID)
        expected_prior = DC_POSITION_PRIOR[PlayerPosition.MIDFIELDER]
        assert features.defensive_contribution_probability == pytest.approx(
            expected_prior
        )
        assert features.defensive_contribution_probability_from_history is False

    def test_cutoff_boundary_includes_observation(self, benchmarks: dict) -> None:
        """History records with outcome_time == cutoff are included."""
        from fpl_engine.models import PlayerMatchHistory

        player = make_player(
            position=PlayerPosition.MIDFIELDER, minutes=2700,
            expected_goals_per_90=0.5,
        )
        cutoff = self._utc(2024, 9, 15)
        history = [
            PlayerMatchHistory(
                player_id=1, fixture_id=1, gameweek=1, minutes=90,
                tackles=5, clearances_blocks_interceptions=0,
                recoveries=0, outcome_time=cutoff,
            ),
        ]
        features = extract_baseline_features(
            player, make_fixture(), None, None, benchmarks,
            gameweek=2, history=history, prediction_cutoff=cutoff,
        )
        assert features.defensive_contribution_probability_from_history is True

    def test_cutoff_mixed_history_filters_correctly(self, benchmarks: dict) -> None:
        """Only records with outcome_time at or before cutoff are included."""
        player = make_player(
            position=PlayerPosition.DEFENDER, minutes=2700,
            expected_goals_per_90=0.3,
        )
        cutoff = self._utc(2024, 9, 20)
        history = [
            # Included: before cutoff
            PlayerMatchHistory(
                player_id=1, fixture_id=1, gameweek=1, minutes=90,
                tackles=5, clearances_blocks_interceptions=5,
                recoveries=0, outcome_time=self._utc(2024, 8, 17),
            ),
            # Excluded: after cutoff
            PlayerMatchHistory(
                player_id=1, fixture_id=2, gameweek=2, minutes=90,
                tackles=5, clearances_blocks_interceptions=5,
                recoveries=0, outcome_time=self._utc(2024, 10, 1),
            ),
            # Excluded: no outcome_time
            PlayerMatchHistory(
                player_id=1, fixture_id=3, gameweek=3, minutes=90,
                tackles=5, clearances_blocks_interceptions=5,
                recoveries=0, outcome_time=None,
            ),
        ]
        features = extract_baseline_features(
            player, make_fixture(), None, None, benchmarks,
            gameweek=4, history=history, prediction_cutoff=cutoff,
        )
        # Only GW1 included -> from_history=True
        assert features.defensive_contribution_probability_from_history is True
