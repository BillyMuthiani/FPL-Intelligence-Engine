"""Tests for FPL scoring rules and single-match point calculation.

All point values verified against the 2026/27 FPL rules:
https://fantasy.premierleague.com/en/help/rules
"""

from __future__ import annotations

import pytest

from fpl_engine.analytics import (
    DEFAULT_SCORING_RULES,
    FPLScoringRules,
    MatchStats,
    ScoringDecomposition,
    ScoringError,
    calculate_fpl_points,
)
from fpl_engine.models.player import PlayerPosition


class TestFPLScoringRules:
    """Tests for the FPLScoringRules configuration object."""

    def test_default_rules_exist(self) -> None:
        assert isinstance(DEFAULT_SCORING_RULES, FPLScoringRules)

    def test_default_rules_goalkeeper_goals_10(self) -> None:
        """2026/27 change: GK goals = 10 points (not 6)."""
        assert DEFAULT_SCORING_RULES.goal_points[PlayerPosition.GOALKEEPER] == 10

    def test_default_rules_defender_goals_6(self) -> None:
        assert DEFAULT_SCORING_RULES.goal_points[PlayerPosition.DEFENDER] == 6

    def test_default_rules_midfielder_goals_5(self) -> None:
        assert DEFAULT_SCORING_RULES.goal_points[PlayerPosition.MIDFIELDER] == 5

    def test_default_rules_forward_goals_4(self) -> None:
        assert DEFAULT_SCORING_RULES.goal_points[PlayerPosition.FORWARD] == 4

    def test_default_rules_penalty_miss_minus_2(self) -> None:
        """2026/27 change: penalty miss = -2 (was -5)."""
        assert DEFAULT_SCORING_RULES.penalty_miss_points == -2

    def test_default_rules_assist_points_3(self) -> None:
        assert DEFAULT_SCORING_RULES.assist_points == 3

    def test_default_rules_appearance_points(self) -> None:
        assert DEFAULT_SCORING_RULES.appearance_points_below == 1
        assert DEFAULT_SCORING_RULES.appearance_points_above_or_equal == 2

    def test_default_rules_clean_sheet_points(self) -> None:
        assert DEFAULT_SCORING_RULES.clean_sheet_points[PlayerPosition.GOALKEEPER] == 4
        assert DEFAULT_SCORING_RULES.clean_sheet_points[PlayerPosition.DEFENDER] == 4
        assert DEFAULT_SCORING_RULES.clean_sheet_points[PlayerPosition.MIDFIELDER] == 1
        assert PlayerPosition.FORWARD not in DEFAULT_SCORING_RULES.clean_sheet_points

    def test_default_rules_defensive_contribution(self) -> None:
        """Defensive contribution defaults match 2026/27 rules."""
        assert DEFAULT_SCORING_RULES.defensive_contribution_def_threshold == 10
        assert DEFAULT_SCORING_RULES.defensive_contribution_mid_fwd_threshold == 12
        assert DEFAULT_SCORING_RULES.defensive_contribution_points == 2
        assert DEFAULT_SCORING_RULES.defensive_contribution_max == 2

    def test_immutability_prevents_attribute_reassignment(self) -> None:
        """Frozen dataclass prevents reassignment of top-level fields."""
        rules = FPLScoringRules()
        with pytest.raises((AttributeError, Exception)):
            rules.assist_points = 5  # type: ignore[misc]

    def test_immutability_of_dict_fields(self) -> None:
        """Dict fields are wrapped in MappingProxyType — truly immutable."""
        rules = FPLScoringRules()
        with pytest.raises(TypeError):
            rules.goal_points[PlayerPosition.FORWARD] = 100  # type: ignore[index]

    def test_custom_rules_override(self) -> None:
        """Custom rules with different values produce different results."""
        custom = FPLScoringRules(
            goal_points={
                PlayerPosition.GOALKEEPER: 5,
                PlayerPosition.DEFENDER: 4,
                PlayerPosition.MIDFIELDER: 3,
                PlayerPosition.FORWARD: 2,
            },
            assist_points=2,
        )
        stats = MatchStats(minutes=90, goals=1)
        result = calculate_fpl_points(PlayerPosition.FORWARD, stats, rules=custom)
        assert result.goal_points == 2
        assert result.total == 2 + 2  # goal + appearance

    def test_validation_rejects_missing_position_in_goal_points(self) -> None:
        """Custom rules must define goal points for all 4 positions."""
        with pytest.raises(ScoringError, match="goal_points missing entry"):
            FPLScoringRules(
                goal_points={
                    PlayerPosition.GOALKEEPER: 10,
                    PlayerPosition.DEFENDER: 6,
                    PlayerPosition.MIDFIELDER: 5,
                    # FORWARD missing
                }
            )

    def test_validation_rejects_negative_threshold(self) -> None:
        """Configuration validation rejects non-positive thresholds."""
        with pytest.raises(ScoringError, match="must be positive"):
            FPLScoringRules(defensive_contribution_def_threshold=0)

    def test_validation_rejects_positive_penalty(self) -> None:
        """Configuration validation rejects positive penalty values."""
        with pytest.raises(ScoringError, match="must be non-positive"):
            FPLScoringRules(goals_conceded_penalty=1)

    def test_validation_rejects_max_below_points(self) -> None:
        """defensive_contribution_max must be >= defensive_contribution_points."""
        with pytest.raises(ScoringError, match="defensive_contribution_max"):
            FPLScoringRules(
                defensive_contribution_points=3,
                defensive_contribution_max=2,
            )

    def test_default_scoring_rules_exported_from_package(self) -> None:
        """The public API exposes the default rules singleton."""
        import fpl_engine.analytics

        assert fpl_engine.analytics.DEFAULT_SCORING_RULES is DEFAULT_SCORING_RULES

    def test_default_rules_goals_conceded_eligible_gk_def_only(self) -> None:
        """Goals conceded deduction only applies to GK and DEF."""
        assert PlayerPosition.GOALKEEPER in DEFAULT_SCORING_RULES.goals_conceded_eligible
        assert PlayerPosition.DEFENDER in DEFAULT_SCORING_RULES.goals_conceded_eligible
        assert PlayerPosition.MIDFIELDER not in DEFAULT_SCORING_RULES.goals_conceded_eligible
        assert PlayerPosition.FORWARD not in DEFAULT_SCORING_RULES.goals_conceded_eligible


class TestMatchStats:
    """Tests for MatchStats input validation."""

    def test_valid_match_stats(self) -> None:
        """Valid stats construct without error."""
        stats = MatchStats(
            minutes=90,
            goals=2,
            assists=1,
            clean_sheet=True,
        )
        assert stats.minutes == 90
        assert stats.goals == 2

    def test_all_defaults_zero(self) -> None:
        """MatchStats with no arguments has all-zero stats, no clean sheet."""
        stats = MatchStats()
        assert stats.minutes == 0
        assert stats.goals == 0
        assert stats.bonus == 0
        assert stats.clean_sheet is False

    def test_negative_minutes_raises(self) -> None:
        with pytest.raises(ScoringError, match="minutes"):
            MatchStats(minutes=-1)

    def test_minutes_exceeding_120_raises(self) -> None:
        with pytest.raises(ScoringError, match="120"):
            MatchStats(minutes=121)

    def test_minutes_at_120_acceptable(self) -> None:
        """120 minutes is the maximum valid value (two 60-minute halves in extra time)."""
        stats = MatchStats(minutes=120)
        assert stats.minutes == 120

    def test_negative_goals_raises(self) -> None:
        with pytest.raises(ScoringError, match="goals must be non-negative"):
            MatchStats(goals=-1)

    def test_negative_assists_raises(self) -> None:
        with pytest.raises(ScoringError, match="assists must be non-negative"):
            MatchStats(assists=-1)

    def test_negative_saves_raises(self) -> None:
        with pytest.raises(ScoringError, match="saves must be non-negative"):
            MatchStats(saves=-1)

    def test_negative_tackles_raises(self) -> None:
        with pytest.raises(ScoringError, match="tackles must be non-negative"):
            MatchStats(tackles=-1)

    def test_bonus_out_of_range_raises(self) -> None:
        with pytest.raises(ScoringError, match="bonus must be"):
            MatchStats(bonus=4)

    def test_bonus_negative_raises(self) -> None:
        with pytest.raises(ScoringError, match="bonus must be"):
            MatchStats(bonus=-1)

    def test_bonus_valid_values(self) -> None:
        """Bonus can be 0, 1, 2, or 3."""
        for val in (0, 1, 2, 3):
            stats = MatchStats(bonus=val)
            assert stats.bonus == val

    def test_frozen_dataclass(self) -> None:
        """MatchStats is frozen — attributes cannot be reassigned."""
        stats = MatchStats(minutes=90)
        with pytest.raises((AttributeError, Exception)):
            stats.minutes = 60  # type: ignore[misc]

    def test_goals_conceded_documented_as_on_pitch(self) -> None:
        """goals_conceded field accepts values independently of position."""
        # A forward's team conceding while they're on the pitch — stats accept it
        stats = MatchStats(minutes=90, goals_conceded=2)
        assert stats.goals_conceded == 2


class TestCalculateFplPoints:
    """Tests for the calculate_fpl_points function."""

    # --- Appearance points ---

    def test_appearance_points_zero_minutes(self) -> None:
        stats = MatchStats(minutes=0)
        result = calculate_fpl_points(PlayerPosition.MIDFIELDER, stats)
        assert result.appearance_points == 0

    def test_appearance_points_below_60(self) -> None:
        stats = MatchStats(minutes=30)
        result = calculate_fpl_points(PlayerPosition.MIDFIELDER, stats)
        assert result.appearance_points == 1

    def test_appearance_points_at_60(self) -> None:
        stats = MatchStats(minutes=60)
        result = calculate_fpl_points(PlayerPosition.MIDFIELDER, stats)
        assert result.appearance_points == 2

    def test_appearance_points_above_60(self) -> None:
        stats = MatchStats(minutes=90)
        result = calculate_fpl_points(PlayerPosition.MIDFIELDER, stats)
        assert result.appearance_points == 2

    # --- Goal points by position ---

    def test_goalkeeper_goals_10_points(self) -> None:
        """2026/27 change: GK goals = 10 points (not 6)."""
        stats = MatchStats(minutes=90, goals=2)
        result = calculate_fpl_points(PlayerPosition.GOALKEEPER, stats)
        assert result.goal_points == 20
        assert result.total == 20 + 2  # goals + appearance

    def test_defender_goals_6_points(self) -> None:
        stats = MatchStats(minutes=90, goals=1)
        result = calculate_fpl_points(PlayerPosition.DEFENDER, stats)
        assert result.goal_points == 6

    def test_midfielder_goals_5_points(self) -> None:
        stats = MatchStats(minutes=90, goals=3)
        result = calculate_fpl_points(PlayerPosition.MIDFIELDER, stats)
        assert result.goal_points == 15

    def test_forward_goals_4_points(self) -> None:
        stats = MatchStats(minutes=90, goals=2)
        result = calculate_fpl_points(PlayerPosition.FORWARD, stats)
        assert result.goal_points == 8

    # --- Assist points ---

    def test_assist_points_all_positions(self) -> None:
        """Assists are 3 points for every position."""
        for pos in PlayerPosition:
            stats = MatchStats(minutes=90, assists=1)
            result = calculate_fpl_points(pos, stats)
            assert result.assist_points == 3

    # --- Clean sheet ---

    def test_clean_sheet_gk_4_points(self) -> None:
        stats = MatchStats(minutes=90, clean_sheet=True)
        result = calculate_fpl_points(PlayerPosition.GOALKEEPER, stats)
        assert result.clean_sheet_points == 4

    def test_clean_sheet_def_4_points(self) -> None:
        stats = MatchStats(minutes=90, clean_sheet=True)
        result = calculate_fpl_points(PlayerPosition.DEFENDER, stats)
        assert result.clean_sheet_points == 4

    def test_clean_sheet_mid_1_point(self) -> None:
        stats = MatchStats(minutes=90, clean_sheet=True)
        result = calculate_fpl_points(PlayerPosition.MIDFIELDER, stats)
        assert result.clean_sheet_points == 1

    def test_no_clean_sheet_for_forward(self) -> None:
        stats = MatchStats(minutes=90, clean_sheet=True)
        result = calculate_fpl_points(PlayerPosition.FORWARD, stats)
        assert result.clean_sheet_points == 0

    def test_clean_sheet_ineligible_due_to_minutes(self) -> None:
        """Clean sheet not awarded if player played fewer than 60 minutes."""
        stats = MatchStats(minutes=30, clean_sheet=True)
        result = calculate_fpl_points(PlayerPosition.DEFENDER, stats)
        assert result.clean_sheet_points == 0

    def test_clean_sheet_at_60_minute_boundary(self) -> None:
        """Exactly 60 minutes is eligible for clean sheet."""
        stats = MatchStats(minutes=60, clean_sheet=True)
        result = calculate_fpl_points(PlayerPosition.DEFENDER, stats)
        assert result.clean_sheet_points == 4

    def test_clean_sheet_not_awarded_if_false(self) -> None:
        """clean_sheet=True must be set for points; default is False."""
        stats = MatchStats(minutes=90)
        result = calculate_fpl_points(PlayerPosition.GOALKEEPER, stats)
        assert result.clean_sheet_points == 0

    # --- Goals conceded ---

    def test_goals_conceded_gk_def_only(self) -> None:
        """Goals conceded deduction only applies to GK and DEF."""
        # GK: 4 goals conceded -> 4//2 = 2 -> 2 * -1 = -2
        stats_gk = MatchStats(minutes=90, goals_conceded=4)
        result_gk = calculate_fpl_points(PlayerPosition.GOALKEEPER, stats_gk)
        assert result_gk.goals_conceded_points == -2

        # DEF: 2 goals conceded -> 2//2 = 1 -> 1 * -1 = -1
        stats_def = MatchStats(minutes=90, goals_conceded=2)
        result_def = calculate_fpl_points(PlayerPosition.DEFENDER, stats_def)
        assert result_def.goals_conceded_points == -1

        # MID: 4 goals conceded -> 0 (not eligible)
        stats_mid = MatchStats(minutes=90, goals_conceded=4)
        result_mid = calculate_fpl_points(PlayerPosition.MIDFIELDER, stats_mid)
        assert result_mid.goals_conceded_points == 0

        # FWD: 4 goals conceded -> 0 (not eligible)
        stats_fwd = MatchStats(minutes=90, goals_conceded=4)
        result_fwd = calculate_fpl_points(PlayerPosition.FORWARD, stats_fwd)
        assert result_fwd.goals_conceded_points == 0

    def test_goals_conceded_rounding(self) -> None:
        """Floor division: 0-1 -> 0, 2-3 -> -1, 4-5 -> -2."""
        for goals, expected in [(0, 0), (1, 0), (2, -1), (3, -1), (4, -2), (5, -2)]:
            stats = MatchStats(minutes=90, goals_conceded=goals)
            result = calculate_fpl_points(PlayerPosition.DEFENDER, stats)
            assert result.goals_conceded_points == expected, (
                f"goals_conceded={goals} -> expected {expected}, "
                f"got {result.goals_conceded_points}"
            )

    # --- Save points ---

    def test_save_points_gk(self) -> None:
        """GK earns 1 point per 3 saves."""
        stats = MatchStats(minutes=90, saves=6)
        result = calculate_fpl_points(PlayerPosition.GOALKEEPER, stats)
        assert result.save_points == 2

    def test_save_points_below_threshold(self) -> None:
        """2 saves is below the 3-save threshold -> 0 points."""
        stats = MatchStats(minutes=90, saves=2)
        result = calculate_fpl_points(PlayerPosition.GOALKEEPER, stats)
        assert result.save_points == 0

    def test_save_points_exact_threshold(self) -> None:
        """Exactly 3 saves -> 1 point."""
        stats = MatchStats(minutes=90, saves=3)
        result = calculate_fpl_points(PlayerPosition.GOALKEEPER, stats)
        assert result.save_points == 1

    def test_no_save_points_for_non_gk(self) -> None:
        """Only goalkeepers earn save points."""
        stats = MatchStats(minutes=90, saves=6)
        result = calculate_fpl_points(PlayerPosition.DEFENDER, stats)
        assert result.save_points == 0

    # --- Penalty saves (GK only) ---

    def test_penalty_save_gk_only(self) -> None:
        """Penalty save points awarded ONLY to goalkeepers (requirement #2)."""
        stats = MatchStats(minutes=90, penalties_saved=2)

        result_gk = calculate_fpl_points(PlayerPosition.GOALKEEPER, stats)
        assert result_gk.penalty_save_points == 10

        result_def = calculate_fpl_points(PlayerPosition.DEFENDER, stats)
        assert result_def.penalty_save_points == 0

        result_mid = calculate_fpl_points(PlayerPosition.MIDFIELDER, stats)
        assert result_mid.penalty_save_points == 0

        result_fwd = calculate_fpl_points(PlayerPosition.FORWARD, stats)
        assert result_fwd.penalty_save_points == 0

    def test_penalty_save_single(self) -> None:
        """One penalty saved by GK = +5 points."""
        stats = MatchStats(minutes=90, penalties_saved=1)
        result = calculate_fpl_points(PlayerPosition.GOALKEEPER, stats)
        assert result.penalty_save_points == 5

    # --- Penalty misses ---

    def test_penalty_miss_minus_2(self) -> None:
        """2026/27 change: penalty miss = -2 (was -5)."""
        stats = MatchStats(minutes=90, penalties_missed=1)
        result = calculate_fpl_points(PlayerPosition.MIDFIELDER, stats)
        assert result.penalty_miss_points == -2

    def test_penalty_miss_multiple(self) -> None:
        stats = MatchStats(minutes=90, penalties_missed=2)
        result = calculate_fpl_points(PlayerPosition.FORWARD, stats)
        assert result.penalty_miss_points == -4

    def test_penalty_miss_all_positions(self) -> None:
        """Penalty miss deduction applies to all positions."""
        stats = MatchStats(minutes=90, penalties_missed=1)
        for pos in PlayerPosition:
            result = calculate_fpl_points(pos, stats)
            assert result.penalty_miss_points == -2

    # --- Cards ---

    def test_yellow_card_minus_1(self) -> None:
        stats = MatchStats(minutes=90, yellow_cards=1)
        result = calculate_fpl_points(PlayerPosition.MIDFIELDER, stats)
        assert result.yellow_card_points == -1

    def test_yellow_card_multiple(self) -> None:
        stats = MatchStats(minutes=90, yellow_cards=2)
        result = calculate_fpl_points(PlayerPosition.MIDFIELDER, stats)
        assert result.yellow_card_points == -2

    def test_red_card_minus_3(self) -> None:
        stats = MatchStats(minutes=90, red_cards=1)
        result = calculate_fpl_points(PlayerPosition.DEFENDER, stats)
        assert result.red_card_points == -3

    # --- Own goals ---

    def test_own_goal_minus_2(self) -> None:
        stats = MatchStats(minutes=90, own_goals=1)
        result = calculate_fpl_points(PlayerPosition.MIDFIELDER, stats)
        assert result.own_goal_points == -2

    def test_own_goal_applies_to_all_positions(self) -> None:
        """Own goals deduct points for every position."""
        stats = MatchStats(minutes=90, own_goals=1)
        for pos in PlayerPosition:
            result = calculate_fpl_points(pos, stats)
            assert result.own_goal_points == -2

    # --- Defensive contribution ---

    def test_defensive_contribution_def_threshold(self) -> None:
        """DEF needs 10+ CBIT (clearances+blocks+interceptions+tackles) for +2."""
        # 4+3+2+1 = 10 CBIT
        stats = MatchStats(
            minutes=90, tackles=4, interceptions=3, clearances=2, blocks=1
        )
        result = calculate_fpl_points(PlayerPosition.DEFENDER, stats)
        assert result.defensive_contribution_points == 2

    def test_defensive_contribution_def_below_threshold(self) -> None:
        """DEF with 9 CBIT gets 0 defensive contribution points."""
        stats = MatchStats(
            minutes=90, tackles=3, interceptions=2, clearances=2, blocks=2
        )
        result = calculate_fpl_points(PlayerPosition.DEFENDER, stats)
        assert result.defensive_contribution_points == 0

    def test_defensive_contribution_mid_includes_recoveries(self) -> None:
        """MID needs 12+ CBIRT (CBIT + recoveries) for +2 points."""
        # 3+2+2+2 = 9 CBIT + 3 recoveries = 12 CBIRT
        stats = MatchStats(
            minutes=90,
            tackles=3,
            interceptions=2,
            clearances=2,
            blocks=2,
            recoveries=3,
        )
        result = calculate_fpl_points(PlayerPosition.MIDFIELDER, stats)
        assert result.defensive_contribution_points == 2

    def test_defensive_contribution_mid_below_threshold(self) -> None:
        """MID needs 12 CBIRT; 11 gets 0."""
        stats = MatchStats(
            minutes=90,
            tackles=3,
            interceptions=2,
            clearances=2,
            blocks=1,
            recoveries=3,
        )
        result = calculate_fpl_points(PlayerPosition.MIDFIELDER, stats)
        assert result.defensive_contribution_points == 0

    def test_defensive_contribution_fwd_uses_recoveries(self) -> None:
        """FWD needs 12+ CBIRT including recoveries."""
        stats = MatchStats(
            minutes=90,
            tackles=3,
            interceptions=2,
            clearances=2,
            blocks=2,
            recoveries=3,
        )
        result = calculate_fpl_points(PlayerPosition.FORWARD, stats)
        assert result.defensive_contribution_points == 2

    def test_defensive_contribution_recoveries_not_counted_for_def(self) -> None:
        """DEF uses only CBIT (no recoveries)."""
        # 9 CBIT + 3 recoveries: effective for DEF is 9
        stats = MatchStats(
            minutes=90,
            tackles=3,
            interceptions=2,
            clearances=2,
            blocks=2,
            recoveries=3,
        )
        result = calculate_fpl_points(PlayerPosition.DEFENDER, stats)
        assert result.defensive_contribution_points == 0

    def test_defensive_contribution_capped_at_2(self) -> None:
        """Even with 30 CBIT, DEF only gets 2 defensive contribution points."""
        stats = MatchStats(
            minutes=90,
            tackles=8,
            interceptions=7,
            clearances=6,
            blocks=9,
        )
        result = calculate_fpl_points(PlayerPosition.DEFENDER, stats)
        assert result.defensive_contribution_points == 2

    def test_no_defensive_contribution_for_gk(self) -> None:
        """Goalkeepers are not eligible for defensive contribution."""
        stats = MatchStats(
            minutes=90, tackles=5, interceptions=5, clearances=5, blocks=5
        )
        result = calculate_fpl_points(PlayerPosition.GOALKEEPER, stats)
        assert result.defensive_contribution_points == 0

    # --- Bonus ---

    def test_bonus_passed_through(self) -> None:
        """Bonus points are taken directly from input, not recomputed."""
        stats = MatchStats(minutes=90, bonus=3)
        result = calculate_fpl_points(PlayerPosition.MIDFIELDER, stats)
        assert result.bonus_points == 3

    def test_bonus_default_zero(self) -> None:
        stats = MatchStats(minutes=90)
        result = calculate_fpl_points(PlayerPosition.MIDFIELDER, stats)
        assert result.bonus_points == 0

    # --- Invalid input handling ---

    def test_invalid_position_type_raises(self) -> None:
        stats = MatchStats(minutes=90)
        with pytest.raises(ScoringError, match="position"):
            calculate_fpl_points("MID", stats)  # type: ignore[arg-type]

    def test_invalid_match_stats_type_raises(self) -> None:
        with pytest.raises(ScoringError, match="match_stats"):
            calculate_fpl_points(
                PlayerPosition.MIDFIELDER, {"minutes": 90}  # type: ignore[arg-type]
            )

    def test_default_rules_used_when_none_passed(self) -> None:
        """When rules=None, DEFAULT_SCORING_RULES (2026/27) is used."""
        stats = MatchStats(minutes=90, goals=1)
        result = calculate_fpl_points(PlayerPosition.FORWARD, stats)
        assert result.goal_points == 4  # default rule: FWD goal = 4

    # --- Combined match events ---

    def test_combined_match_events_gk(self) -> None:
        """GK: 90 min (+2), 1 goal (+10), 5 saves (+1), 1 pen save (+5),
        clean sheet (+4), 2 goals conceded (-1), 1 yellow (-1), bonus 1.
        Total = 2 + 10 + 1 + 5 + 4 - 1 - 1 + 1 = 21.
        """
        stats = MatchStats(
            minutes=90,
            goals=1,
            goals_conceded=2,
            saves=5,
            penalties_saved=1,
            clean_sheet=True,
            yellow_cards=1,
            bonus=1,
        )
        result = calculate_fpl_points(PlayerPosition.GOALKEEPER, stats)
        assert result.appearance_points == 2
        assert result.goal_points == 10
        assert result.save_points == 1
        assert result.penalty_save_points == 5
        assert result.clean_sheet_points == 4
        assert result.goals_conceded_points == -1
        assert result.yellow_card_points == -1
        assert result.bonus_points == 1
        assert result.total == 21

    def test_combined_match_events_def(self) -> None:
        """DEF: 90 min (+2), 1 goal (+6), 1 assist (+3), clean sheet (+4),
        2 goals conceded (-1), 10 CBIT (+2), 1 yellow (-1), bonus 3.
        Total = 2 + 6 + 3 + 4 - 1 + 2 - 1 + 3 = 18.
        """
        stats = MatchStats(
            minutes=90,
            goals=1,
            assists=1,
            goals_conceded=2,
            clean_sheet=True,
            tackles=4,
            interceptions=3,
            clearances=2,
            blocks=1,
            yellow_cards=1,
            bonus=3,
        )
        result = calculate_fpl_points(PlayerPosition.DEFENDER, stats)
        assert result.appearance_points == 2
        assert result.goal_points == 6
        assert result.assist_points == 3
        assert result.clean_sheet_points == 4
        assert result.goals_conceded_points == -1
        assert result.defensive_contribution_points == 2
        assert result.yellow_card_points == -1
        assert result.bonus_points == 3
        assert result.total == 18

    def test_combined_match_events_fwd(self) -> None:
        """FWD: 90 min (+2), 2 goals (+8), 1 assist (+3), 1 pen miss (-2), bonus 2.
        Total = 2 + 8 + 3 - 2 + 2 = 13.
        """
        stats = MatchStats(
            minutes=90,
            goals=2,
            assists=1,
            penalties_missed=1,
            bonus=2,
        )
        result = calculate_fpl_points(PlayerPosition.FORWARD, stats)
        assert result.appearance_points == 2
        assert result.goal_points == 8
        assert result.assist_points == 3
        assert result.penalty_miss_points == -2
        assert result.bonus_points == 2
        assert result.total == 13

    def test_combined_match_events_mid(self) -> None:
        """MID: 90 min (+2), 1 goal (+5), 2 assists (+6), clean sheet (+1),
        12 CBIRT (+2), 1 own goal (-2), bonus 1.
        Total = 2 + 5 + 6 + 1 + 2 - 2 + 1 = 15.
        """
        stats = MatchStats(
            minutes=90,
            goals=1,
            assists=2,
            clean_sheet=True,
            tackles=3,
            interceptions=2,
            clearances=2,
            blocks=2,
            recoveries=3,
            own_goals=1,
            bonus=1,
        )
        result = calculate_fpl_points(PlayerPosition.MIDFIELDER, stats)
        assert result.appearance_points == 2
        assert result.goal_points == 5
        assert result.assist_points == 6
        assert result.clean_sheet_points == 1
        assert result.defensive_contribution_points == 2
        assert result.own_goal_points == -2
        assert result.bonus_points == 1
        assert result.total == 15

    # --- Edge cases ---

    def test_zero_event_match(self) -> None:
        """Player with all stats zero gets total 0."""
        stats = MatchStats()
        result = calculate_fpl_points(PlayerPosition.MIDFIELDER, stats)
        assert result.total == 0
        assert result.appearance_points == 0

    def test_zero_minutes_ignores_clean_sheet(self) -> None:
        """0 minutes: even if clean_sheet=True, no clean sheet points."""
        stats = MatchStats(minutes=0, clean_sheet=True)
        result = calculate_fpl_points(PlayerPosition.DEFENDER, stats)
        assert result.appearance_points == 0
        assert result.clean_sheet_points == 0
        assert result.total == 0

    # --- Consistency invariant ---

    def test_total_equals_sum_of_components(self) -> None:
        """The total field must equal the sum of all component fields."""
        stats = MatchStats(
            minutes=90,
            goals=2,
            assists=1,
            goals_conceded=3,
            saves=7,
            penalties_saved=1,
            penalties_missed=1,
            yellow_cards=1,
            red_cards=0,
            own_goals=0,
            tackles=5,
            interceptions=3,
            clearances=4,
            blocks=2,
            recoveries=4,
            bonus=3,
            clean_sheet=True,
        )
        for pos in PlayerPosition:
            result = calculate_fpl_points(pos, stats)
            component_sum = (
                result.appearance_points
                + result.goal_points
                + result.assist_points
                + result.clean_sheet_points
                + result.goals_conceded_points
                + result.save_points
                + result.penalty_save_points
                + result.penalty_miss_points
                + result.yellow_card_points
                + result.red_card_points
                + result.own_goal_points
                + result.defensive_contribution_points
                + result.bonus_points
            )
            assert result.total == component_sum, f"Total mismatch for {pos}"

    def test_scoring_decomposition_is_frozen(self) -> None:
        """ScoringDecomposition is frozen — cannot be mutated."""
        stats = MatchStats(minutes=90)
        result = calculate_fpl_points(PlayerPosition.MIDFIELDER, stats)
        with pytest.raises((AttributeError, Exception)):
            result.total = 100  # type: ignore[misc]

    def test_scoring_decomposition_type(self) -> None:
        stats = MatchStats(minutes=90, goals=1)
        result = calculate_fpl_points(PlayerPosition.MIDFIELDER, stats)
        assert isinstance(result, ScoringDecomposition)
