"""Tests for prediction contracts and expected-points calculator."""

from __future__ import annotations

import pytest

from fpl_engine.analytics import (
    DEFAULT_SCORING_RULES,
    MatchStats,
    calculate_fpl_points,
    predict_expected_points,
)
from fpl_engine.analytics.prediction import (
    MatchPrediction,
    PlayerGameweekPrediction,
    PredictionError,
)
from fpl_engine.models import PlayerPosition


def _make_match_prediction(
    fixture_id: int = 1,
    gameweek: int = 1,
    *,
    start_probability: float = 1.0,
    probability_appears: float = 1.0,
    probability_60_plus: float = 1.0,
    clean_sheet_probability: float = 0.0,
    defensive_contribution_probability: float = 0.0,
    defensive_contribution_probability_from_history: bool = True,
    expected_goals: float = 0.0,
    expected_assists: float = 0.0,
    expected_goals_conceded: float = 0.0,
    expected_saves: float = 0.0,
    expected_penalties_saved: float = 0.0,
    expected_penalties_missed: float = 0.0,
    expected_yellow_cards: float = 0.0,
    expected_red_cards: float = 0.0,
    expected_own_goals: float = 0.0,
    expected_bonus: float = 0.0,
) -> MatchPrediction:
    """Create a MatchPrediction with deterministic full-90-min defaults."""
    return MatchPrediction(
        fixture_id=fixture_id,
        gameweek=gameweek,
        start_probability=start_probability,
        probability_appears=probability_appears,
        probability_60_plus=probability_60_plus,
        clean_sheet_probability=clean_sheet_probability,
        defensive_contribution_probability=defensive_contribution_probability,
        defensive_contribution_probability_from_history=defensive_contribution_probability_from_history,
        expected_goals=expected_goals,
        expected_assists=expected_assists,
        expected_goals_conceded=expected_goals_conceded,
        expected_saves=expected_saves,
        expected_penalties_saved=expected_penalties_saved,
        expected_penalties_missed=expected_penalties_missed,
        expected_yellow_cards=expected_yellow_cards,
        expected_red_cards=expected_red_cards,
        expected_own_goals=expected_own_goals,
        expected_bonus=expected_bonus,
    )


def _make_gw_prediction(
    player_id: int = 1,
    position: PlayerPosition = PlayerPosition.MIDFIELDER,
    gameweek: int = 1,
    fixtures: tuple[MatchPrediction, ...] | None = None,
) -> PlayerGameweekPrediction:
    """Create a PlayerGameweekPrediction with a single default fixture."""
    if fixtures is None:
        fixtures = (_make_match_prediction(fixture_id=1, gameweek=gameweek),)
    return PlayerGameweekPrediction(
        player_id=player_id,
        position=position,
        gameweek=gameweek,
        fixtures=fixtures,
    )


# ---------------------------------------------------------------------------
# MatchPrediction validation
# ---------------------------------------------------------------------------


class TestMatchPredictionValidation:
    def test_valid_defaults(self) -> None:
        mp = MatchPrediction(fixture_id=1, gameweek=1)
        assert mp.fixture_id == 1
        assert mp.gameweek == 1
        assert mp.start_probability == 0.0
        assert mp.probability_appears == 0.0

    def test_fixture_id_must_be_int(self) -> None:
        with pytest.raises(PredictionError, match="fixture_id"):
            MatchPrediction(fixture_id=1.5, gameweek=1)  # type: ignore[arg-type]

    def test_fixture_id_bool_rejected(self) -> None:
        with pytest.raises(PredictionError, match="fixture_id"):
            MatchPrediction(fixture_id=True, gameweek=1)

    def test_gameweek_must_be_int(self) -> None:
        with pytest.raises(PredictionError, match="gameweek"):
            MatchPrediction(fixture_id=1, gameweek=1.0)  # type: ignore[arg-type]

    def test_gameweek_below_range(self) -> None:
        with pytest.raises(PredictionError, match="gameweek"):
            MatchPrediction(fixture_id=1, gameweek=0)

    def test_gameweek_above_range(self) -> None:
        with pytest.raises(PredictionError, match="gameweek"):
            MatchPrediction(fixture_id=1, gameweek=39)

    def test_gameweek_bool_rejected(self) -> None:
        with pytest.raises(PredictionError, match="gameweek"):
            MatchPrediction(fixture_id=1, gameweek=True)

    @pytest.mark.parametrize(
        "field",
        [
            "start_probability",
            "probability_appears",
            "probability_60_plus",
            "clean_sheet_probability",
            "defensive_contribution_probability",
        ],
    )
    def test_probability_out_of_range(self, field: str) -> None:
        with pytest.raises(PredictionError, match=field):
            MatchPrediction(
                fixture_id=1,
                gameweek=1,
                **{field: 1.5},
            )

    def test_probability_negative(self) -> None:
        with pytest.raises(PredictionError, match="probability_appears"):
            MatchPrediction(
                fixture_id=1,
                gameweek=1,
                probability_appears=-0.1,
            )

    def test_probability_bool_rejected(self) -> None:
        with pytest.raises(PredictionError, match="probability_appears"):
            MatchPrediction(
                fixture_id=1,
                gameweek=1,
                probability_appears=True,
            )

    def test_penalty_prob_out_of_range(self) -> None:
        with pytest.raises(PredictionError, match="expected_penalties_saved"):
            MatchPrediction(
                fixture_id=1,
                gameweek=1,
                expected_penalties_saved=1.5,
            )

    def test_penalty_miss_prob_out_of_range(self) -> None:
        with pytest.raises(PredictionError, match="expected_penalties_missed"):
            MatchPrediction(
                fixture_id=1,
                gameweek=1,
                expected_penalties_missed=-0.1,
            )

    @pytest.mark.parametrize(
        "field",
        [
            "expected_goals",
            "expected_assists",
            "expected_goals_conceded",
            "expected_saves",
            "expected_yellow_cards",
            "expected_red_cards",
            "expected_own_goals",
        ],
    )
    def test_count_field_negative(self, field: str) -> None:
        with pytest.raises(PredictionError, match=field):
            MatchPrediction(
                fixture_id=1,
                gameweek=1,
                **{field: -1.0},
            )

    def test_expected_bonus_above_3(self) -> None:
        with pytest.raises(PredictionError, match="expected_bonus"):
            MatchPrediction(fixture_id=1, gameweek=1, expected_bonus=3.5)

    def test_expected_bonus_negative(self) -> None:
        with pytest.raises(PredictionError, match="expected_bonus"):
            MatchPrediction(fixture_id=1, gameweek=1, expected_bonus=-0.1)

    def test_60_plus_exceeds_appears(self) -> None:
        with pytest.raises(PredictionError, match="probability_60_plus"):
            MatchPrediction(
                fixture_id=1,
                gameweek=1,
                probability_appears=0.3,
                probability_60_plus=0.5,
            )

    def test_probabilities_boundary_values(self) -> None:
        """0.0 and 1.0 are valid probability boundaries."""
        mp = MatchPrediction(
            fixture_id=1,
            gameweek=1,
            start_probability=0.0,
            probability_appears=1.0,
            probability_60_plus=1.0,
            clean_sheet_probability=0.0,
            defensive_contribution_probability=1.0,
        )
        assert mp.probability_60_plus == 1.0


# ---------------------------------------------------------------------------
# PlayerGameweekPrediction validation
# ---------------------------------------------------------------------------


class TestPlayerGameweekPredictionValidation:
    def test_valid_defaults(self) -> None:
        gw = _make_gw_prediction()
        assert gw.player_id == 1
        assert gw.position == PlayerPosition.MIDFIELDER
        assert gw.gameweek == 1
        assert len(gw.fixtures) == 1

    def test_player_id_must_be_int(self) -> None:
        with pytest.raises(PredictionError, match="player_id"):
            _make_gw_prediction(player_id=1.5)  # type: ignore[arg-type]

    def test_player_id_bool_rejected(self) -> None:
        with pytest.raises(PredictionError, match="player_id"):
            _make_gw_prediction(player_id=True)

    def test_position_must_be_enum(self) -> None:
        with pytest.raises(PredictionError, match="position"):
            PlayerGameweekPrediction(
                player_id=1,
                position="MID",  # type: ignore[arg-type]
                gameweek=1,
                fixtures=(_make_match_prediction(),),
            )

    def test_gameweek_below_range(self) -> None:
        with pytest.raises(PredictionError, match="gameweek"):
            _make_gw_prediction(gameweek=0)

    def test_gameweek_above_range(self) -> None:
        with pytest.raises(PredictionError, match="gameweek"):
            _make_gw_prediction(gameweek=39)

    def test_fixtures_must_be_tuple(self) -> None:
        with pytest.raises(PredictionError, match="tuple"):
            PlayerGameweekPrediction(
                player_id=1,
                position=PlayerPosition.MIDFIELDER,
                gameweek=1,
                fixtures=[_make_match_prediction()],  # type: ignore[arg-type]
            )

    def test_fixtures_empty(self) -> None:
        with pytest.raises(PredictionError, match="empty"):
            PlayerGameweekPrediction(
                player_id=1,
                position=PlayerPosition.MIDFIELDER,
                gameweek=1,
                fixtures=(),
            )

    def test_fixture_not_match_prediction(self) -> None:
        with pytest.raises(PredictionError, match="MatchPrediction"):
            PlayerGameweekPrediction(
                player_id=1,
                position=PlayerPosition.MIDFIELDER,
                gameweek=1,
                fixtures=(MatchStats(),),  # type: ignore[arg-type]
            )

    def test_fixture_gameweek_mismatch(self) -> None:
        with pytest.raises(PredictionError, match="gameweek"):
            _make_gw_prediction(
                gameweek=1,
                fixtures=(_make_match_prediction(gameweek=2),),
            )

    def test_duplicate_fixture_id(self) -> None:
        with pytest.raises(PredictionError, match="duplicate"):
            _make_gw_prediction(
                fixtures=(
                    _make_match_prediction(fixture_id=1, gameweek=1),
                    _make_match_prediction(fixture_id=1, gameweek=1),
                ),
            )


# ---------------------------------------------------------------------------
# predict_expected_points input validation
# ---------------------------------------------------------------------------


class TestPredictInputValidation:
    def test_non_prediction_raises(self) -> None:
        with pytest.raises(PredictionError, match="PlayerGameweekPrediction"):
            predict_expected_points("not a prediction")  # type: ignore[arg-type]

    def test_custom_rules_used(self) -> None:
        """Should accept a custom FPLScoringRules instance."""
        from fpl_engine.analytics.scoring import FPLScoringRules

        custom = FPLScoringRules(assist_points=99)
        gw = _make_gw_prediction(
            fixtures=(_make_match_prediction(expected_assists=1.0),),
        )
        result = predict_expected_points(gw, rules=custom)
        assert result.expected_assist_points == 99.0
        assert result.rules is custom


# ---------------------------------------------------------------------------
# Appearance probabilities
# ---------------------------------------------------------------------------


class TestAppearanceProbabilities:
    def test_full_90_minutes(self) -> None:
        """Full 90 min: P(appears)=1, P(60+)=1 -> 2 points."""
        gw = _make_gw_prediction(
            fixtures=(
                _make_match_prediction(
                    probability_appears=1.0,
                    probability_60_plus=1.0,
                ),
            ),
        )
        result = predict_expected_points(gw)
        assert result.expected_appearance_points == 2.0

    def test_sub_below_60(self) -> None:
        """Sub <60 min: P(appears)=1, P(60+)=0 -> 1 point."""
        gw = _make_gw_prediction(
            fixtures=(
                _make_match_prediction(
                    probability_appears=1.0,
                    probability_60_plus=0.0,
                ),
            ),
        )
        result = predict_expected_points(gw)
        assert result.expected_appearance_points == 1.0

    def test_bench(self) -> None:
        """Bench: P(appears)=0, P(60+)=0 -> 0 points."""
        gw = _make_gw_prediction(
            fixtures=(
                _make_match_prediction(
                    probability_appears=0.0,
                    probability_60_plus=0.0,
                ),
            ),
        )
        result = predict_expected_points(gw)
        assert result.expected_appearance_points == 0.0

    def test_partial_probabilities(self) -> None:
        """P(appears)=0.5, P(60+)=0.3 -> 0.2*1 + 0.3*2 = 0.8."""
        gw = _make_gw_prediction(
            fixtures=(
                _make_match_prediction(
                    probability_appears=0.5,
                    probability_60_plus=0.3,
                ),
            ),
        )
        result = predict_expected_points(gw)
        assert result.expected_appearance_points == 0.8

    def test_start_probability_not_used(self) -> None:
        """start_probability is informational only; should not affect points."""
        mp1 = _make_match_prediction(
            start_probability=1.0,
            probability_appears=1.0,
            probability_60_plus=1.0,
        )
        mp2 = _make_match_prediction(
            start_probability=0.0,
            probability_appears=1.0,
            probability_60_plus=1.0,
        )
        r1 = predict_expected_points(_make_gw_prediction(fixtures=(mp1,)))
        r2 = predict_expected_points(_make_gw_prediction(fixtures=(mp2,)))
        assert r1.expected_appearance_points == r2.expected_appearance_points
        assert r1.total_expected_points == r2.total_expected_points


# ---------------------------------------------------------------------------
# Scoring parity with calculate_fpl_points (exact, linear components)
# ---------------------------------------------------------------------------


def _parity_case(
    position: PlayerPosition,
    minutes: int = 90,
    goals: int = 0,
    assists: int = 0,
    goals_conceded: int = 0,
    saves: int = 0,
    clean_sheet: bool = False,
    bonus: int = 0,
    penalties_saved: int = 0,
    penalties_missed: int = 0,
    yellow_cards: int = 0,
    red_cards: int = 0,
    own_goals: int = 0,
) -> tuple[PlayerGameweekPrediction, int]:
    """Build a deterministic prediction and return (prediction, expected_total).

    goals_conceded must be even and saves must be a multiple of 3 for
    parity (due to floor-division approximations in FPL scoring).
    """
    if minutes == 0:
        prob_appears = 0.0
        prob_60_plus = 0.0
    elif minutes < 60:
        prob_appears = 1.0
        prob_60_plus = 0.0
    else:
        prob_appears = 1.0
        prob_60_plus = 1.0

    mp = MatchPrediction(
        fixture_id=1,
        gameweek=1,
        probability_appears=prob_appears,
        probability_60_plus=prob_60_plus,
        clean_sheet_probability=1.0 if clean_sheet else 0.0,
        expected_goals=float(goals),
        expected_assists=float(assists),
        expected_goals_conceded=float(goals_conceded),
        expected_saves=float(saves),
        expected_penalties_saved=float(penalties_saved),
        expected_penalties_missed=float(penalties_missed),
        expected_yellow_cards=float(yellow_cards),
        expected_red_cards=float(red_cards),
        expected_own_goals=float(own_goals),
        expected_bonus=float(bonus),
    )
    ms = MatchStats(
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
        bonus=bonus,
        clean_sheet=clean_sheet,
    )
    gw = PlayerGameweekPrediction(
        player_id=1,
        position=position,
        gameweek=1,
        fixtures=(mp,),
    )
    decomp = calculate_fpl_points(position, ms)
    return gw, decomp.total


class TestScoringParity:
    """Verify predict_expected_points == calculate_fpl_points for exact
    (linear) components with deterministic inputs."""

    def test_mid_full90_no_stats(self) -> None:
        gw, expected = _parity_case(PlayerPosition.MIDFIELDER, minutes=90)
        result = predict_expected_points(gw)
        assert result.total_expected_points == expected
        assert result.total_expected_points == 2.0

    def test_mid_full90_one_goal(self) -> None:
        gw, expected = _parity_case(PlayerPosition.MIDFIELDER, minutes=90, goals=1)
        result = predict_expected_points(gw)
        assert result.total_expected_points == expected
        assert result.total_expected_points == 7.0  # 2 + 5

    def test_mid_full90_two_goals(self) -> None:
        gw, expected = _parity_case(PlayerPosition.MIDFIELDER, minutes=90, goals=2)
        result = predict_expected_points(gw)
        assert result.total_expected_points == expected
        assert result.total_expected_points == 12.0  # 2 + 10

    def test_mid_full90_assist(self) -> None:
        gw, expected = _parity_case(PlayerPosition.MIDFIELDER, minutes=90, assists=1)
        result = predict_expected_points(gw)
        assert result.total_expected_points == expected
        assert result.total_expected_points == 5.0  # 2 + 3

    def test_mid_full90_clean_sheet(self) -> None:
        gw, expected = _parity_case(PlayerPosition.MIDFIELDER, minutes=90, clean_sheet=True)
        result = predict_expected_points(gw)
        assert result.total_expected_points == expected
        assert result.total_expected_points == 3.0  # 2 + 1

    def test_def_full90_clean_sheet_no_gc(self) -> None:
        gw, expected = _parity_case(
            PlayerPosition.DEFENDER,
            minutes=90,
            clean_sheet=True,
            goals_conceded=0,
        )
        result = predict_expected_points(gw)
        assert result.total_expected_points == expected
        assert result.total_expected_points == 6.0  # 2 + 4

    def test_def_full90_two_gc(self) -> None:
        gw, expected = _parity_case(
            PlayerPosition.DEFENDER,
            minutes=90,
            clean_sheet=False,
            goals_conceded=2,
        )
        result = predict_expected_points(gw)
        assert result.total_expected_points == expected
        assert result.total_expected_points == 1.0  # 2 + (-1)

    def test_fwd_full90_two_goals(self) -> None:
        gw, expected = _parity_case(
            PlayerPosition.FORWARD,
            minutes=90,
            goals=2,
        )
        result = predict_expected_points(gw)
        assert result.total_expected_points == expected
        assert result.total_expected_points == 10.0  # 2 + 8

    def test_sub_30_minutes(self) -> None:
        gw, expected = _parity_case(PlayerPosition.MIDFIELDER, minutes=30)
        result = predict_expected_points(gw)
        assert result.total_expected_points == expected
        assert result.total_expected_points == 1.0

    def test_bench_zero_minutes(self) -> None:
        gw, expected = _parity_case(PlayerPosition.MIDFIELDER, minutes=0)
        result = predict_expected_points(gw)
        assert result.total_expected_points == expected
        assert result.total_expected_points == 0.0

    def test_penalty_miss(self) -> None:
        gw, expected = _parity_case(
            PlayerPosition.FORWARD,
            minutes=90,
            penalties_missed=1,
        )
        result = predict_expected_points(gw)
        assert result.total_expected_points == expected
        assert result.total_expected_points == 0.0  # 2 + (-2)

    def test_yellow_card(self) -> None:
        gw, expected = _parity_case(
            PlayerPosition.MIDFIELDER,
            minutes=90,
            yellow_cards=1,
        )
        result = predict_expected_points(gw)
        assert result.total_expected_points == expected
        assert result.total_expected_points == 1.0  # 2 + (-1)

    def test_red_card(self) -> None:
        gw, expected = _parity_case(
            PlayerPosition.FORWARD,
            minutes=90,
            red_cards=1,
        )
        result = predict_expected_points(gw)
        assert result.total_expected_points == expected
        assert result.total_expected_points == -1.0  # 2 + (-3)

    def test_own_goal(self) -> None:
        gw, expected = _parity_case(
            PlayerPosition.MIDFIELDER,
            minutes=90,
            own_goals=1,
        )
        result = predict_expected_points(gw)
        assert result.total_expected_points == expected
        assert result.total_expected_points == 0.0  # 2 + (-2)

    def test_gk_full90_three_saves(self) -> None:
        gw, expected = _parity_case(
            PlayerPosition.GOALKEEPER,
            minutes=90,
            saves=3,
        )
        result = predict_expected_points(gw)
        assert result.total_expected_points == expected
        assert result.total_expected_points == 3.0  # 2 + 1

    def test_gk_full90_penalty_saved(self) -> None:
        gw, expected = _parity_case(
            PlayerPosition.GOALKEEPER,
            minutes=90,
            clean_sheet=True,
            goals_conceded=0,
            penalties_saved=1,
        )
        result = predict_expected_points(gw)
        assert result.total_expected_points == expected
        assert result.total_expected_points == 11.0  # 2 + 4 + 5

    def test_gk_full90_six_saves_clean_sheet(self) -> None:
        gw, expected = _parity_case(
            PlayerPosition.GOALKEEPER,
            minutes=90,
            saves=6,
            clean_sheet=True,
            goals_conceded=0,
        )
        result = predict_expected_points(gw)
        assert result.total_expected_points == expected
        assert result.total_expected_points == 8.0  # 2 + 4 + 2

    def test_gk_full90_bonus_3(self) -> None:
        gw, expected = _parity_case(
            PlayerPosition.GOALKEEPER,
            minutes=90,
            bonus=3,
        )
        result = predict_expected_points(gw)
        assert result.total_expected_points == expected
        assert result.total_expected_points == 5.0  # 2 + 3

    def test_fwd_no_clean_sheet_points(self) -> None:
        """FORWARD clean sheet yields 0 points (not eligible)."""
        gw = _make_gw_prediction(
            position=PlayerPosition.FORWARD,
            fixtures=(
                _make_match_prediction(
                    probability_appears=1.0,
                    probability_60_plus=1.0,
                    clean_sheet_probability=1.0,
                    expected_goals_conceded=0.0,
                ),
            ),
        )
        result = predict_expected_points(gw)
        assert result.expected_clean_sheet_points == 0.0
        assert result.expected_goals_conceded_points == 0.0
        assert result.total_expected_points == 2.0


# ---------------------------------------------------------------------------
# Position-specific scoring
# ---------------------------------------------------------------------------


class TestPositionSpecificScoring:
    @pytest.mark.parametrize(
        "position,goal_pts",
        [
            (PlayerPosition.GOALKEEPER, 10),
            (PlayerPosition.DEFENDER, 6),
            (PlayerPosition.MIDFIELDER, 5),
            (PlayerPosition.FORWARD, 4),
        ],
    )
    def test_goal_value_by_position(
        self,
        position: PlayerPosition,
        goal_pts: int,
    ) -> None:
        gw = _make_gw_prediction(
            position=position,
            fixtures=(_make_match_prediction(expected_goals=1.0),),
        )
        result = predict_expected_points(gw)
        # 2 appearance points + goal_pts per goal
        assert result.expected_goal_points == float(goal_pts)
        assert result.total_expected_points == 2.0 + goal_pts

    @pytest.mark.parametrize(
        "position",
        [
            PlayerPosition.GOALKEEPER,
            PlayerPosition.DEFENDER,
            PlayerPosition.MIDFIELDER,
        ],
    )
    def test_clean_sheet_by_position(self, position: PlayerPosition) -> None:
        cs_points = DEFAULT_SCORING_RULES.clean_sheet_points[position]
        gw = _make_gw_prediction(
            position=position,
            fixtures=(
                _make_match_prediction(
                    probability_appears=1.0,
                    probability_60_plus=1.0,
                    clean_sheet_probability=1.0,
                ),
            ),
        )
        result = predict_expected_points(gw)
        assert result.expected_clean_sheet_points == float(cs_points)

    def test_fwd_clean_sheet_zero(self) -> None:
        gw = _make_gw_prediction(
            position=PlayerPosition.FORWARD,
            fixtures=(
                _make_match_prediction(
                    probability_appears=1.0,
                    probability_60_plus=1.0,
                    clean_sheet_probability=1.0,
                ),
            ),
        )
        result = predict_expected_points(gw)
        assert result.expected_clean_sheet_points == 0.0

    def test_fwd_goals_conceded_zero(self) -> None:
        """FORWARD not eligible for goals-conceded deductions."""
        gw = _make_gw_prediction(
            position=PlayerPosition.FORWARD,
            fixtures=(_make_match_prediction(expected_goals_conceded=5.0),),
        )
        result = predict_expected_points(gw)
        assert result.expected_goals_conceded_points == 0.0

    def test_fwd_saves_zero(self) -> None:
        gw = _make_gw_prediction(
            position=PlayerPosition.FORWARD,
            fixtures=(_make_match_prediction(expected_saves=9.0),),
        )
        result = predict_expected_points(gw)
        assert result.expected_save_points == 0.0

    def test_non_gk_penalty_save_zero(self) -> None:
        """Only GKs get penalty-save points."""
        for pos in (
            PlayerPosition.DEFENDER,
            PlayerPosition.MIDFIELDER,
            PlayerPosition.FORWARD,
        ):
            gw = _make_gw_prediction(
                position=pos,
                fixtures=(
                    _make_match_prediction(
                        expected_penalties_saved=1.0,
                    ),
                ),
            )
            result = predict_expected_points(gw)
            assert result.expected_penalty_save_points == 0.0


# ---------------------------------------------------------------------------
# Double gameweek aggregation
# ---------------------------------------------------------------------------


class TestDoubleGameweek:
    def test_dgw_two_goals_sum(self) -> None:
        """Two fixtures each with 1 goal: total goal pts = 2 * 5."""
        gw = _make_gw_prediction(
            fixtures=(
                _make_match_prediction(
                    fixture_id=1,
                    gameweek=1,
                    expected_goals=1.0,
                ),
                _make_match_prediction(
                    fixture_id=2,
                    gameweek=1,
                    expected_goals=1.0,
                ),
            ),
        )
        result = predict_expected_points(gw)
        assert result.fixture_count == 2
        assert result.expected_appearance_points == 4.0  # 2 + 2
        assert result.expected_goal_points == 10.0  # 5 + 5
        assert result.total_expected_points == 14.0

    def test_dgw_one_starts_one_bench(self) -> None:
        """One fixture starts, one bench - only first contributes."""
        gw = _make_gw_prediction(
            fixtures=(
                _make_match_prediction(
                    fixture_id=1,
                    gameweek=1,
                    probability_appears=1.0,
                    probability_60_plus=1.0,
                ),
                _make_match_prediction(
                    fixture_id=2,
                    gameweek=1,
                    probability_appears=0.0,
                    probability_60_plus=0.0,
                ),
            ),
        )
        result = predict_expected_points(gw)
        assert result.fixture_count == 2
        assert result.expected_appearance_points == 2.0  # 2 + 0
        assert result.total_expected_points == 2.0

    def test_dgw_fractional_probabilities(self) -> None:
        """Fractional appearance probabilities across two fixtures."""
        gw = _make_gw_prediction(
            fixtures=(
                _make_match_prediction(
                    fixture_id=1,
                    gameweek=1,
                    probability_appears=0.8,
                    probability_60_plus=0.5,
                ),
                _make_match_prediction(
                    fixture_id=2,
                    gameweek=1,
                    probability_appears=0.6,
                    probability_60_plus=0.4,
                ),
            ),
        )
        result = predict_expected_points(gw)
        # Fixture 1: 0.3 * 1 + 0.5 * 2 = 0.3 + 1.0 = 1.3
        # Fixture 2: 0.2 * 1 + 0.4 * 2 = 0.2 + 0.8 = 1.0
        assert result.expected_appearance_points == 2.3


# ---------------------------------------------------------------------------
# Approximation boundary tests (nonlinear components)
# ---------------------------------------------------------------------------


class TestApproximationBoundaries:
    """Goals conceded and saves use floor-division scoring in FPL.
    The linear approximation diverges from the exact floor value at
    non-multiple inputs. These tests verify the documented boundaries."""

    def test_goals_conceded_single_exact_zero(self) -> None:
        """1 goal: exact = floor(1/2)*(-1) = 0, approx = 0.5*(-1) = -0.5.
        They differ - the approximation is a linearisation."""
        gw = _make_gw_prediction(
            position=PlayerPosition.DEFENDER,
            fixtures=(
                _make_match_prediction(
                    probability_appears=1.0,
                    probability_60_plus=1.0,
                    expected_goals_conceded=1.0,
                ),
            ),
        )
        result = predict_expected_points(gw)
        # Approx: (1.0 / 2.0) * (-1) = -0.5
        assert result.expected_goals_conceded_points == -0.5
        # Exact would be 0 (floor(1 // 2) = 0)
        assert result.expected_goals_conceded_points != 0.0

    def test_goals_conceded_even_parity(self) -> None:
        """2 goals: exact = floor(2/2)*(-1) = -1, approx = -1.0. Parity."""
        gw, expected = _parity_case(
            PlayerPosition.DEFENDER,
            minutes=90,
            goals_conceded=2,
        )
        result = predict_expected_points(gw)
        assert result.total_expected_points == expected
        assert result.expected_goals_conceded_points == -1.0

    def test_goals_conceded_odd_differs(self) -> None:
        """3 goals: exact = floor(3/2)*(-1) = -1, approx = -1.5. Differ."""
        gw = _make_gw_prediction(
            position=PlayerPosition.DEFENDER,
            fixtures=(
                _make_match_prediction(
                    probability_appears=1.0,
                    probability_60_plus=1.0,
                    expected_goals_conceded=3.0,
                ),
            ),
        )
        result = predict_expected_points(gw)
        # Approx: (3.0 / 2.0) * (-1) = -1.5
        assert result.expected_goals_conceded_points == -1.5
        # Exact would be -1 (floor(3 // 2) = 1, penalty -1)
        assert result.expected_goals_conceded_points != -1.0

    def test_saves_below_threshold(self) -> None:
        """1 save: exact = floor(1/3)*1 = 0, approx = 1/3. Differ."""
        gw = _make_gw_prediction(
            position=PlayerPosition.GOALKEEPER,
            fixtures=(
                _make_match_prediction(
                    probability_appears=1.0,
                    probability_60_plus=1.0,
                    expected_saves=1.0,
                ),
            ),
        )
        result = predict_expected_points(gw)
        # Approx: (1.0 / 3) * 1 = 0.333...
        assert result.expected_save_points == pytest.approx(1.0 / 3.0)
        # Exact would be 0
        assert result.expected_save_points != 0.0

    def test_saves_at_threshold_parity(self) -> None:
        """3 saves: exact = floor(3/3)*1 = 1, approx = 1.0. Parity."""
        gw, expected = _parity_case(
            PlayerPosition.GOALKEEPER,
            minutes=90,
            saves=3,
        )
        result = predict_expected_points(gw)
        assert result.total_expected_points == expected
        assert result.expected_save_points == 1.0

    def test_saves_above_threshold_differs(self) -> None:
        """4 saves: exact = floor(4/3)*1 = 1, approx = 4/3. Differ."""
        gw = _make_gw_prediction(
            position=PlayerPosition.GOALKEEPER,
            fixtures=(
                _make_match_prediction(
                    probability_appears=1.0,
                    probability_60_plus=1.0,
                    expected_saves=4.0,
                ),
            ),
        )
        result = predict_expected_points(gw)
        # Approx: (4.0 / 3) * 1 = 1.333...
        assert result.expected_save_points == pytest.approx(4.0 / 3.0)
        # Exact would be 1
        assert result.expected_save_points != 1.0


# ---------------------------------------------------------------------------
# Component sum consistency
# ---------------------------------------------------------------------------


class TestComponentConsistency:
    def test_total_equals_sum_of_components(self) -> None:
        """total_expected_points must equal the sum of all component fields."""
        mp = _make_match_prediction(
            expected_goals=1.5,
            expected_assists=0.7,
            clean_sheet_probability=0.6,
            probability_appears=0.9,
            probability_60_plus=0.7,
            expected_goals_conceded=1.3,
            expected_saves=4.2,
            expected_bonus=1.5,
            expected_yellow_cards=0.3,
            expected_red_cards=0.05,
            expected_own_goals=0.1,
            expected_penalties_saved=0.2,
            expected_penalties_missed=0.1,
            defensive_contribution_probability=0.5,
        )
        gw = _make_gw_prediction(
            position=PlayerPosition.DEFENDER,
            fixtures=(mp,),
        )
        result = predict_expected_points(gw)
        component_sum = (
            result.expected_appearance_points
            + result.expected_goal_points
            + result.expected_assist_points
            + result.expected_clean_sheet_points
            + result.expected_goals_conceded_points
            + result.expected_save_points
            + result.expected_penalty_save_points
            + result.expected_penalty_miss_points
            + result.expected_yellow_card_points
            + result.expected_red_card_points
            + result.expected_own_goal_points
            + result.expected_defensive_contribution_points
            + result.expected_bonus_points
        )
        assert result.total_expected_points == pytest.approx(component_sum)

    def test_total_equals_sum_of_components_dgw(self) -> None:
        """Same check across a DGW with two fixtures."""
        fixtures = (
            _make_match_prediction(
                fixture_id=1,
                gameweek=1,
                expected_goals=0.5,
                expected_assists=0.3,
                expected_yellow_cards=0.8,
                expected_bonus=0.5,
            ),
            _make_match_prediction(
                fixture_id=2,
                gameweek=1,
                expected_goals=0.2,
            ),
        )
        gw = _make_gw_prediction(
            position=PlayerPosition.MIDFIELDER,
            fixtures=fixtures,
        )
        result = predict_expected_points(gw)
        component_sum = (
            result.expected_appearance_points
            + result.expected_goal_points
            + result.expected_assist_points
            + result.expected_clean_sheet_points
            + result.expected_goals_conceded_points
            + result.expected_save_points
            + result.expected_penalty_save_points
            + result.expected_penalty_miss_points
            + result.expected_yellow_card_points
            + result.expected_red_card_points
            + result.expected_own_goal_points
            + result.expected_defensive_contribution_points
            + result.expected_bonus_points
        )
        assert result.total_expected_points == pytest.approx(component_sum)
        assert result.fixture_count == 2


# ---------------------------------------------------------------------------
# Defensive contribution
# ---------------------------------------------------------------------------


class TestDefensiveContribution:
    def test_dc_midfielder_with_probability(self) -> None:
        """MID: P(appear)=1, P(DC)=0.5 -> 1.0 * 0.5 * 2 = 1.0."""
        gw = _make_gw_prediction(
            position=PlayerPosition.MIDFIELDER,
            fixtures=(
                _make_match_prediction(
                    probability_appears=1.0,
                    defensive_contribution_probability=0.5,
                ),
            ),
        )
        result = predict_expected_points(gw)
        assert result.expected_defensive_contribution_points == 1.0

    def test_dc_defender_with_probability(self) -> None:
        """DEF: same formula applies."""
        gw = _make_gw_prediction(
            position=PlayerPosition.DEFENDER,
            fixtures=(
                _make_match_prediction(
                    probability_appears=0.8,
                    probability_60_plus=0.8,
                    defensive_contribution_probability=0.75,
                ),
            ),
        )
        result = predict_expected_points(gw)
        # 0.8 * 0.75 * 2 = 1.2
        assert result.expected_defensive_contribution_points == pytest.approx(1.2)

    def test_dc_forward_with_probability(self) -> None:
        """FWD: eligible for DC."""
        gw = _make_gw_prediction(
            position=PlayerPosition.FORWARD,
            fixtures=(
                _make_match_prediction(
                    probability_appears=1.0,
                    defensive_contribution_probability=0.3,
                ),
            ),
        )
        result = predict_expected_points(gw)
        assert result.expected_defensive_contribution_points == pytest.approx(0.6)

    def test_dc_zero_probability(self) -> None:
        """P(DC)=0 -> no DC points."""
        gw = _make_gw_prediction(
            fixtures=(
                _make_match_prediction(
                    probability_appears=1.0,
                    defensive_contribution_probability=0.0,
                ),
            ),
        )
        result = predict_expected_points(gw)
        assert result.expected_defensive_contribution_points == 0.0

    def test_dc_bench_zero(self) -> None:
        """P(appear)=0 -> no DC points even if P(DC)=1."""
        gw = _make_gw_prediction(
            fixtures=(
                _make_match_prediction(
                    probability_appears=0.0,
                    probability_60_plus=0.0,
                    defensive_contribution_probability=1.0,
                ),
            ),
        )
        result = predict_expected_points(gw)
        assert result.expected_defensive_contribution_points == 0.0


# ---------------------------------------------------------------------------
# Defensive contribution provenance
# ---------------------------------------------------------------------------


class TestDfContributionProvenance:
    """Verify the from_history flag distinguishes prior fallback from data."""

    def test_default_from_history_is_true(self) -> None:
        """MatchPrediction defaults the flag to True (data-backed assumption)."""
        mp = MatchPrediction(fixture_id=1, gameweek=1)
        assert mp.defensive_contribution_probability_from_history is True

    def test_from_history_true_for_data_backed(self) -> None:
        mp = MatchPrediction(
            fixture_id=1,
            gameweek=1,
            defensive_contribution_probability=0.5,
            defensive_contribution_probability_from_history=True,
        )
        assert mp.defensive_contribution_probability_from_history is True

    def test_from_history_false_for_prior_fallback(self) -> None:
        mp = MatchPrediction(
            fixture_id=1,
            gameweek=1,
            defensive_contribution_probability=0.25,
            defensive_contribution_probability_from_history=False,
        )
        assert mp.defensive_contribution_probability_from_history is False


# ---------------------------------------------------------------------------
# Full precision verification
# ---------------------------------------------------------------------------


class TestPrecision:
    def test_no_rounding_appearance(self) -> None:
        """Fractional probabilities produce exact float results."""
        gw = _make_gw_prediction(
            fixtures=(
                _make_match_prediction(
                    probability_appears=0.333,
                    probability_60_plus=0.111,
                ),
            ),
        )
        result = predict_expected_points(gw)
        # (0.333 - 0.111) * 1 + 0.111 * 2 = 0.222 + 0.222 = 0.444
        assert result.expected_appearance_points == pytest.approx(0.444)

    def test_no_rounding_goals(self) -> None:
        """Fractional expected goals produce exact float results."""
        gw = _make_gw_prediction(
            fixtures=(
                _make_match_prediction(
                    expected_goals=0.333333333,
                ),
            ),
        )
        result = predict_expected_points(gw)
        assert result.expected_goal_points == 0.333333333 * 5

    def test_no_rounding_goals_conceded(self) -> None:
        """Fractional goals conceded produce exact float results."""
        gw = _make_gw_prediction(
            position=PlayerPosition.DEFENDER,
            fixtures=(
                _make_match_prediction(
                    probability_appears=1.0,
                    probability_60_plus=1.0,
                    expected_goals_conceded=1.7,
                ),
            ),
        )
        result = predict_expected_points(gw)
        # (1.7 / 2.0) * (-1) = -0.85
        assert result.expected_goals_conceded_points == -0.85

    def test_total_matches_manual_calculation(self) -> None:
        """Verify total against a hand-computed value with many decimals."""
        mp = _make_match_prediction(
            probability_appears=0.85,
            probability_60_plus=0.65,
            clean_sheet_probability=0.4,
            expected_goals=1.7,
            expected_assists=0.4,
            expected_goals_conceded=1.2,
            expected_bonus=1.5,
            expected_yellow_cards=0.3,
            defensive_contribution_probability=0.6,
        )
        gw = _make_gw_prediction(
            position=PlayerPosition.DEFENDER,
            fixtures=(mp,),
        )
        result = predict_expected_points(gw)
        # Manual calculation:
        # appearance: (0.85 - 0.65) * 1 + 0.65 * 2 = 0.20 + 1.30 = 1.50
        # goals: 1.7 * 6 = 10.2
        # assists: 0.4 * 3 = 1.2
        # clean sheet: 0.65 * 0.4 * 4 = 1.04
        # goals conceded: (1.2 / 2.0) * (-1) = -0.60
        # bonus: 1.5
        # DC: 0.85 * 0.6 * 2 = 1.02
        # yellow: 0.3 * (-1) = -0.3
        expected_total = 1.50 + 10.2 + 1.2 + 1.04 - 0.60 + 1.5 + 1.02 - 0.3
        assert result.total_expected_points == pytest.approx(expected_total)
        assert result.fixture_count == 1
        assert result.rules is DEFAULT_SCORING_RULES


# ---------------------------------------------------------------------------
# Non-finite value rejection (regression)
# ---------------------------------------------------------------------------


class TestNonFiniteRejection:
    """Guard against NaN and ±inf slipping through ``_is_real_float``.

    Previously ``_is_real_float`` accepted any float or int (excluding bool)
    without checking finiteness, so ``NaN`` and ``±inf`` could bypass the
    type check.  For count/rate fields whose only range guard is ``value < 0``,
    this meant ``NaN`` and ``+inf`` were silently accepted, producing garbage
    expected-points results downstream.
    """

    @pytest.mark.parametrize(
        "nonfinite",
        [float("nan"), float("inf"), float("-inf")],
        ids=["nan", "pos_inf", "neg_inf"],
    )
    @pytest.mark.parametrize(
        "field",
        [
            "expected_goals",
            "expected_assists",
            "expected_goals_conceded",
            "expected_saves",
            "expected_yellow_cards",
            "expected_red_cards",
            "expected_own_goals",
        ],
    )
    def test_count_field_rejects_nonfinite(self, field: str, nonfinite: float) -> None:
        """Every count/rate field must reject NaN, +inf, and -inf."""
        with pytest.raises(PredictionError, match=field):
            MatchPrediction(fixture_id=1, gameweek=1, **{field: nonfinite})

    @pytest.mark.parametrize(
        "nonfinite",
        [float("nan"), float("inf"), float("-inf")],
        ids=["nan", "pos_inf", "neg_inf"],
    )
    @pytest.mark.parametrize(
        "field",
        [
            "start_probability",
            "probability_appears",
            "probability_60_plus",
            "clean_sheet_probability",
            "defensive_contribution_probability",
        ],
    )
    def test_probability_field_rejects_nonfinite(self, field: str, nonfinite: float) -> None:
        """Probability fields must reject NaN, +inf, and -inf."""
        with pytest.raises(PredictionError, match=field):
            MatchPrediction(fixture_id=1, gameweek=1, **{field: nonfinite})

    @pytest.mark.parametrize(
        "nonfinite",
        [float("nan"), float("inf"), float("-inf")],
        ids=["nan", "pos_inf", "neg_inf"],
    )
    @pytest.mark.parametrize("field", ["expected_penalties_saved", "expected_penalties_missed"])
    def test_penalty_field_rejects_nonfinite(self, field: str, nonfinite: float) -> None:
        """Penalty fields must reject NaN, +inf, and -inf."""
        with pytest.raises(PredictionError, match=field):
            MatchPrediction(fixture_id=1, gameweek=1, **{field: nonfinite})

    @pytest.mark.parametrize(
        "nonfinite",
        [float("nan"), float("inf"), float("-inf")],
        ids=["nan", "pos_inf", "neg_inf"],
    )
    def test_expected_bonus_rejects_nonfinite(self, nonfinite: float) -> None:
        """expected_bonus must reject NaN, +inf, and -inf."""
        with pytest.raises(PredictionError, match="expected_bonus"):
            MatchPrediction(fixture_id=1, gameweek=1, expected_bonus=nonfinite)

    def test_valid_finite_floats_accepted(self) -> None:
        """Sanity check: ordinary finite floats are still accepted."""
        mp = MatchPrediction(
            fixture_id=1,
            gameweek=1,
            expected_goals=1.5,
            expected_assists=0.3,
            expected_goals_conceded=2.0,
            expected_saves=3.7,
            expected_yellow_cards=0.1,
            expected_red_cards=0.0,
            expected_own_goals=0.0,
            start_probability=0.5,
            probability_appears=1.0,
            probability_60_plus=0.8,
            clean_sheet_probability=0.3,
            defensive_contribution_probability=0.2,
            expected_penalties_saved=1.0,
            expected_penalties_missed=0.0,
            expected_bonus=2.0,
        )
        assert mp.expected_goals == 1.5
        assert mp.expected_assists == 0.3
        assert mp.expected_bonus == 2.0

    def test_integer_count_fields_accepted(self) -> None:
        """Integer values for count/rate fields are accepted (real floats)."""
        mp = MatchPrediction(
            fixture_id=1,
            gameweek=1,
            expected_goals=2,  # type: ignore[arg-type]
            expected_assists=1,  # type: ignore[arg-type]
        )
        assert mp.expected_goals == 2
        assert mp.expected_assists == 1
