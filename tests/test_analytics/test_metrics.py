"""Tests for player metrics calculation."""

from __future__ import annotations

import pytest

from fpl_engine.analytics import (
    DEFAULT_SCORING_RULES,
    FixtureGameweekRecord,
    GameweekStats,
    MatchStats,
    MetricsError,
    calculate_fpl_points,
    calculate_player_metrics,
    clean_sheet_rate,
    points_last_n_average,
    points_per_90,
    points_per_million,
    points_std_dev,
)
from fpl_engine.models.player import Player, PlayerPosition


def _make_player(pos: PlayerPosition, pid: int = 1, cost: int = 100) -> Player:
    """Create a minimal valid Player for testing."""
    return Player(
        id=pid, web_name=f"P{pid}", first_name="Test", second_name="Player",
        team_id=1, position=pos, now_cost=cost, code=1,
        element_type=pos.position_id, team_code=1,
    )


def _make_match_stats(**overrides) -> MatchStats:
    defaults = {
        "minutes": 90, "goals": 0, "assists": 0, "goals_conceded": 0,
        "saves": 0, "penalties_saved": 0, "penalties_missed": 0,
        "yellow_cards": 0, "red_cards": 0, "own_goals": 0,
        "tackles": 0, "interceptions": 0, "clearances": 0, "blocks": 0,
        "recoveries": 0, "bonus": 0, "clean_sheet": False,
    }
    defaults.update(overrides)
    return MatchStats(**defaults)


def _make_record(
    gw: int, fixture_id: int,
    pos: PlayerPosition = PlayerPosition.MIDFIELDER,
    observed_points: int | None = None,
    **stats_overrides,
) -> FixtureGameweekRecord:
    ms = _make_match_stats(**stats_overrides)
    decomp = calculate_fpl_points(pos, ms)
    return FixtureGameweekRecord(
        gameweek=gw, fixture_id=fixture_id, match_stats=ms,
        decomposition=decomp, observed_points=observed_points,
    )


class TestHelperFunctions:
    def test_points_per_90_normal(self) -> None:
        assert points_per_90(42, 900) == 4.2

    def test_points_per_90_zero_minutes(self) -> None:
        assert points_per_90(42, 0) is None

    def test_points_per_90_zero_points(self) -> None:
        assert points_per_90(0, 90) == 0.0

    def test_points_per_90_zero_zero(self) -> None:
        assert points_per_90(0, 0) is None

    def test_points_per_million_normal(self) -> None:
        assert points_per_million(42, 85) == pytest.approx(4.941176)

    def test_points_per_million_zero_price(self) -> None:
        assert points_per_million(42, 0) is None

    def test_points_per_million_zero_points(self) -> None:
        assert points_per_million(0, 100) == 0.0

    def test_clean_sheet_rate_normal(self) -> None:
        assert clean_sheet_rate(3, 10) == 0.3

    def test_clean_sheet_rate_zero_matches(self) -> None:
        assert clean_sheet_rate(0, 0) is None

    def test_clean_sheet_rate_zero_sheets(self) -> None:
        assert clean_sheet_rate(0, 10) == 0.0

    def test_points_std_dev_normal(self) -> None:
        assert points_std_dev([4, 6]) == 1.0

    def test_points_std_dev_single(self) -> None:
        assert points_std_dev([4]) is None

    def test_points_std_dev_empty(self) -> None:
        assert points_std_dev([]) is None

    def test_points_std_dev_identical(self) -> None:
        assert points_std_dev([5, 5, 5]) == 0.0

    def test_points_last_n_avg_normal(self) -> None:
        assert points_last_n_average([1, 2, 3], 3) == 2.0

    def test_points_last_n_avg_insufficient(self) -> None:
        assert points_last_n_average([1, 2], 3) is None

    def test_points_last_n_avg_zero_n(self) -> None:
        assert points_last_n_average([1, 2, 3], 0) is None

    def test_points_last_n_avg_negative_n(self) -> None:
        assert points_last_n_average([1, 2, 3], -1) is None


class TestFixtureGameweekRecordValidation:
    def test_valid_record(self) -> None:
        ms = _make_match_stats(minutes=90, goals=1)
        decomp = calculate_fpl_points(PlayerPosition.MIDFIELDER, ms)
        record = FixtureGameweekRecord(
            gameweek=1, fixture_id=1, match_stats=ms,
            decomposition=decomp, observed_points=6,
        )
        assert record.gameweek == 1
        assert record.fixture_id == 1
        assert record.observed_points == 6

    def test_gameweek_zero_raises(self) -> None:
        ms = _make_match_stats()
        decomp = calculate_fpl_points(PlayerPosition.MIDFIELDER, ms)
        with pytest.raises(MetricsError, match="gameweek must be in"):
            FixtureGameweekRecord(
                gameweek=0, fixture_id=1, match_stats=ms,
                decomposition=decomp, observed_points=None,
            )

    def test_gameweek_39_raises(self) -> None:
        ms = _make_match_stats()
        decomp = calculate_fpl_points(PlayerPosition.MIDFIELDER, ms)
        with pytest.raises(MetricsError, match="gameweek must be in"):
            FixtureGameweekRecord(
                gameweek=39, fixture_id=1, match_stats=ms,
                decomposition=decomp, observed_points=None,
            )

    def test_boolean_gameweek_raises(self) -> None:
        ms = _make_match_stats()
        decomp = calculate_fpl_points(PlayerPosition.MIDFIELDER, ms)
        with pytest.raises(MetricsError, match="must be an int"):
            FixtureGameweekRecord(
                gameweek=True,  # type: ignore[arg-type]
                fixture_id=1, match_stats=ms,
                decomposition=decomp, observed_points=None,
            )

    def test_boolean_fixture_id_raises(self) -> None:
        ms = _make_match_stats()
        decomp = calculate_fpl_points(PlayerPosition.MIDFIELDER, ms)
        with pytest.raises(MetricsError, match="must be an int"):
            FixtureGameweekRecord(
                gameweek=1, fixture_id=True,  # type: ignore[arg-type]
                match_stats=ms, decomposition=decomp, observed_points=None,
            )

    def test_negative_observed_points_raises(self) -> None:
        ms = _make_match_stats()
        decomp = calculate_fpl_points(PlayerPosition.MIDFIELDER, ms)
        with pytest.raises(MetricsError, match="must be non-negative"):
            FixtureGameweekRecord(
                gameweek=1, fixture_id=1, match_stats=ms,
                decomposition=decomp, observed_points=-1,
            )

    def test_wrong_match_stats_type_raises(self) -> None:
        decomp = calculate_fpl_points(PlayerPosition.MIDFIELDER, _make_match_stats())
        with pytest.raises(MetricsError, match="match_stats must be MatchStats"):
            FixtureGameweekRecord(
                gameweek=1, fixture_id=1,
                match_stats=object(),  # type: ignore[arg-type]
                decomposition=decomp, observed_points=None,
            )

    def test_wrong_decomposition_type_raises(self) -> None:
        ms = _make_match_stats()
        with pytest.raises(MetricsError, match="decomposition must be ScoringDecomposition"):
            FixtureGameweekRecord(
                gameweek=1, fixture_id=1, match_stats=ms,
                decomposition=object(),  # type: ignore[arg-type]
                observed_points=None,
            )


class TestCalculatePlayerMetricsBasic:
    def test_empty_records(self) -> None:
        player = _make_player(PlayerPosition.MIDFIELDER)
        result = calculate_player_metrics(player, [])
        assert result.sample_size == 0
        assert result.matches_played == 0
        assert result.total_minutes == 0
        assert result.total_points_observed == 0
        assert result.total_points_reconstructed == 0
        assert result.points_per_90_observed is None
        assert result.points_per_90_reconstructed is None
        assert result.clean_sheet_rate is None
        assert result.cutoff_gameweek is None

    def test_single_gw_calculations(self) -> None:
        player = _make_player(PlayerPosition.MIDFIELDER)
        record = _make_record(1, 1, PlayerPosition.MIDFIELDER,
                              observed_points=10, goals=1, assists=1)
        result = calculate_player_metrics(player, [record])
        assert result.sample_size == 1
        assert result.matches_played == 1
        assert result.total_minutes == 90
        assert result.goals == 1
        assert result.assists == 1
        assert result.total_points_observed == 10
        assert result.total_points_reconstructed == record.decomposition.total
        assert result.points_per_90_observed == pytest.approx(10.0)
        assert result.points_season_avg_observed == 10.0
        assert result.points_last_1_observed == 10
        assert result.points_last_3_avg_observed is None
        assert result.points_last_5_avg_observed is None
        assert result.points_std_dev_observed is None

    def test_unsorted_input_sorted_deterministically(self) -> None:
        player = _make_player(PlayerPosition.MIDFIELDER)
        r1 = _make_record(1, 1, PlayerPosition.MIDFIELDER, observed_points=3)
        r2 = _make_record(2, 2, PlayerPosition.MIDFIELDER, observed_points=5)
        r3 = _make_record(3, 3, PlayerPosition.MIDFIELDER, observed_points=7)
        result = calculate_player_metrics(player, [r3, r1, r2])
        assert result.points_last_1_observed == 7
        assert result.total_points_observed == 15

    def test_shuffled_input_same_output(self) -> None:
        player = _make_player(PlayerPosition.MIDFIELDER)
        records = [
            _make_record(1, 1, PlayerPosition.MIDFIELDER, observed_points=3, goals=1),
            _make_record(2, 2, PlayerPosition.MIDFIELDER, observed_points=5, goals=2),
            _make_record(3, 3, PlayerPosition.MIDFIELDER, observed_points=7, goals=1),
        ]
        result1 = calculate_player_metrics(player, records)
        result2 = calculate_player_metrics(player, list(reversed(records)))
        assert result1 == result2


class TestDuplicateFixtureId:
    def test_duplicate_fixture_id_raises(self) -> None:
        player = _make_player(PlayerPosition.MIDFIELDER)
        r1 = _make_record(1, 1, PlayerPosition.MIDFIELDER, observed_points=2)
        r2 = _make_record(2, 1, PlayerPosition.MIDFIELDER, observed_points=3)
        with pytest.raises(MetricsError, match="duplicate fixture_id"):
            calculate_player_metrics(player, [r1, r2])


class TestCutoffBehavior:
    def test_cutoff_zero_returns_empty(self) -> None:
        player = _make_player(PlayerPosition.MIDFIELDER)
        records = [_make_record(1, 1, PlayerPosition.MIDFIELDER, observed_points=2)]
        result = calculate_player_metrics(player, records, cutoff_gameweek=0)
        assert result.sample_size == 0
        assert result.matches_played == 0
        assert result.total_minutes == 0

    def test_negative_cutoff_raises(self) -> None:
        player = _make_player(PlayerPosition.MIDFIELDER)
        records = [_make_record(1, 1, PlayerPosition.MIDFIELDER, observed_points=2)]
        with pytest.raises(MetricsError, match="must be >= 0"):
            calculate_player_metrics(player, records, cutoff_gameweek=-1)

    def test_boolean_cutoff_raises(self) -> None:
        player = _make_player(PlayerPosition.MIDFIELDER)
        records = [_make_record(1, 1, PlayerPosition.MIDFIELDER, observed_points=2)]
        with pytest.raises(MetricsError, match="must be int or None"):
            calculate_player_metrics(player, records, cutoff_gameweek=True)  # type: ignore[arg-type]

    def test_records_after_cutoff_excluded(self) -> None:
        player = _make_player(PlayerPosition.MIDFIELDER)
        records = [
            _make_record(1, 1, PlayerPosition.MIDFIELDER, observed_points=2),
            _make_record(2, 2, PlayerPosition.MIDFIELDER, observed_points=4),
            _make_record(3, 3, PlayerPosition.MIDFIELDER, observed_points=6),
        ]
        result = calculate_player_metrics(player, records, cutoff_gameweek=2)
        assert result.sample_size == 2
        assert result.total_points_observed == 6
        assert result.total_fixtures == 3

    def test_cutoff_above_all_gameweeks(self) -> None:
        player = _make_player(PlayerPosition.MIDFIELDER)
        records = [
            _make_record(i, i, PlayerPosition.MIDFIELDER, observed_points=i)
            for i in range(1, 11)
        ]
        result = calculate_player_metrics(player, records, cutoff_gameweek=100)
        assert result.sample_size == 10

    def test_invalid_player_type_raises(self) -> None:
        records = [_make_record(1, 1)]
        with pytest.raises(MetricsError, match="must be a Player instance"):
            calculate_player_metrics("not_a_player", records)  # type: ignore[arg-type]


class TestDoubleGameweek:
    def test_dgw_minutes_exceed_120(self) -> None:
        player = _make_player(PlayerPosition.DEFENDER)
        r1 = _make_record(1, 100, PlayerPosition.DEFENDER,
                          observed_points=6, minutes=90, clean_sheet=True)
        r2 = _make_record(1, 101, PlayerPosition.DEFENDER,
                          observed_points=6, minutes=90, clean_sheet=True)
        result = calculate_player_metrics(player, [r1, r2])
        assert result.sample_size == 1
        assert result.matches_played == 1
        assert result.total_minutes == 180
        assert result.total_fixtures == 2
        assert result.total_points_observed == 12

    def test_dgw_clean_sheet_requires_all_fixtures(self) -> None:
        player = _make_player(PlayerPosition.DEFENDER)
        r1 = _make_record(1, 100, PlayerPosition.DEFENDER, observed_points=4,
                          minutes=90, clean_sheet=True)
        r2 = _make_record(1, 101, PlayerPosition.DEFENDER, observed_points=2,
                          minutes=90, clean_sheet=False)
        result = calculate_player_metrics(player, [r1, r2])
        assert result.clean_sheets == 0
        assert result.matches_played == 1

    def test_dgw_zero_minute_fixture_disallows_clean_sheet(self) -> None:
        player = _make_player(PlayerPosition.DEFENDER)
        r1 = _make_record(1, 100, PlayerPosition.DEFENDER, observed_points=1,
                          minutes=0, clean_sheet=True)
        r2 = _make_record(1, 101, PlayerPosition.DEFENDER, observed_points=6,
                          minutes=90, clean_sheet=True)
        result = calculate_player_metrics(player, [r1, r2])
        assert result.clean_sheets == 0

    def test_dgw_partial_observation(self) -> None:
        player = _make_player(PlayerPosition.DEFENDER)
        r1 = _make_record(1, 100, PlayerPosition.DEFENDER, observed_points=4,
                          minutes=90, clean_sheet=True)
        r2 = _make_record(1, 101, PlayerPosition.DEFENDER, observed_points=None,
                          minutes=90, clean_sheet=True)
        result = calculate_player_metrics(player, [r1, r2])
        assert result.observed_gw_count == 0
        assert result.total_points_observed == 0
        assert result.points_last_1_observed is None

    def test_dgw_reconstructed_always_computed(self) -> None:
        player = _make_player(PlayerPosition.DEFENDER)
        r1 = _make_record(1, 100, PlayerPosition.DEFENDER, observed_points=None,
                          minutes=90, clean_sheet=True)
        r2 = _make_record(1, 101, PlayerPosition.DEFENDER, observed_points=None,
                          minutes=90, clean_sheet=True)
        result = calculate_player_metrics(player, [r1, r2])
        expected_rec = r1.decomposition.total + r2.decomposition.total
        assert result.total_points_reconstructed == expected_rec


class TestObservedVsReconstructed:
    def test_divergence_reported_independently(self) -> None:
        player = _make_player(PlayerPosition.MIDFIELDER)
        ms = _make_match_stats(minutes=90, goals=1)
        decomp = calculate_fpl_points(PlayerPosition.MIDFIELDER, ms)
        record = FixtureGameweekRecord(
            gameweek=1, fixture_id=1, match_stats=ms,
            decomposition=decomp, observed_points=99,
        )
        result = calculate_player_metrics(player, [record])
        assert result.total_points_observed == 99
        assert result.total_points_reconstructed == decomp.total
        assert result.total_points_observed != result.total_points_reconstructed


class TestZeroDenominators:
    def test_all_zero_minutes(self) -> None:
        player = _make_player(PlayerPosition.MIDFIELDER)
        records = [
            _make_record(1, 1, PlayerPosition.MIDFIELDER, observed_points=0, minutes=0),
            _make_record(2, 2, PlayerPosition.MIDFIELDER, observed_points=0, minutes=0),
        ]
        result = calculate_player_metrics(player, records)
        assert result.matches_played == 0
        assert result.points_per_90_observed is None
        assert result.points_per_90_reconstructed is None
        assert result.clean_sheet_rate is None

    def test_no_observed_points_all_none(self) -> None:
        player = _make_player(PlayerPosition.MIDFIELDER)
        records = [
            _make_record(1, 1, PlayerPosition.MIDFIELDER, observed_points=None, goals=1),
            _make_record(2, 2, PlayerPosition.MIDFIELDER, observed_points=None, goals=1),
        ]
        result = calculate_player_metrics(player, records)
        assert result.observed_gw_count == 0
        assert result.total_points_observed == 0
        assert result.points_per_90_observed is None
        assert result.points_season_avg_observed is None
        assert result.points_std_dev_observed is None
        assert result.total_points_reconstructed > 0


class TestFormWindows:
    def test_points_last_3_sufficient(self) -> None:
        player = _make_player(PlayerPosition.MIDFIELDER)
        records = [
            _make_record(i, i, PlayerPosition.MIDFIELDER, observed_points=i)
            for i in range(1, 6)
        ]
        result = calculate_player_metrics(player, records)
        assert result.points_last_3_avg_observed == pytest.approx(4.0)
        assert result.points_last_5_avg_observed == pytest.approx(3.0)

    def test_points_last_3_insufficient(self) -> None:
        player = _make_player(PlayerPosition.MIDFIELDER)
        records = [
            _make_record(1, 1, PlayerPosition.MIDFIELDER, observed_points=3),
            _make_record(2, 2, PlayerPosition.MIDFIELDER, observed_points=5),
        ]
        result = calculate_player_metrics(player, records)
        assert result.points_last_3_avg_observed is None
        assert result.points_last_5_avg_observed is None

    def test_form_uses_gw_not_fixture(self) -> None:
        """Form windows operate on GW-level observed points."""
        player = _make_player(PlayerPosition.MIDFIELDER)
        records = [
            _make_record(1, 100, PlayerPosition.MIDFIELDER, observed_points=2),
            _make_record(1, 101, PlayerPosition.MIDFIELDER, observed_points=2),
            _make_record(2, 200, PlayerPosition.MIDFIELDER, observed_points=4),
            _make_record(3, 300, PlayerPosition.MIDFIELDER, observed_points=6),
        ]
        result = calculate_player_metrics(player, records)
        assert result.observed_gw_count == 3
        assert result.points_last_1_observed == 6
        assert result.points_last_3_avg_observed == pytest.approx(14 / 3)


class TestCleanSheetSemantics:
    def test_fwd_not_clean_sheet_eligible(self) -> None:
        player = _make_player(PlayerPosition.FORWARD)
        records = [
            _make_record(1, 1, PlayerPosition.FORWARD, observed_points=5,
                         minutes=90, goals=1),
        ]
        result = calculate_player_metrics(player, records)
        assert result.clean_sheet_eligible is False

    def test_gk_clean_sheet_eligible(self) -> None:
        player = _make_player(PlayerPosition.GOALKEEPER)
        records = [
            _make_record(1, 1, PlayerPosition.GOALKEEPER, observed_points=6,
                         minutes=90, clean_sheet=True, saves=5),
        ]
        result = calculate_player_metrics(player, records)
        assert result.clean_sheet_eligible is True
        assert result.clean_sheet_rate == 1.0

    def test_mid_clean_sheet_eligible(self) -> None:
        player = _make_player(PlayerPosition.MIDFIELDER)
        result = calculate_player_metrics(player, [])
        assert result.clean_sheet_eligible is True

    def test_zero_minute_gw_no_clean_sheet(self) -> None:
        player = _make_player(PlayerPosition.DEFENDER)
        ms = _make_match_stats(minutes=0, clean_sheet=True)
        decomp = calculate_fpl_points(PlayerPosition.DEFENDER, ms)
        record = FixtureGameweekRecord(
            gameweek=1, fixture_id=1, match_stats=ms,
            decomposition=decomp, observed_points=None,
        )
        result = calculate_player_metrics(player, [record])
        assert result.matches_played == 0
        assert result.clean_sheets == 0
        assert result.clean_sheet_rate is None
        assert result.matches_played == 0
        assert result.clean_sheets == 0
        assert result.clean_sheet_rate is None


class TestStdDevReconstructed:
    def test_std_dev_two_samples(self) -> None:
        player = _make_player(PlayerPosition.MIDFIELDER)
        records = [
            _make_record(1, 1, PlayerPosition.MIDFIELDER, observed_points=2),
            _make_record(2, 2, PlayerPosition.MIDFIELDER, observed_points=4),
        ]
        result = calculate_player_metrics(player, records)
        assert result.points_std_dev_reconstructed is not None

    def test_std_dev_single_sample(self) -> None:
        player = _make_player(PlayerPosition.MIDFIELDER)
        records = [_make_record(1, 1, PlayerPosition.MIDFIELDER, observed_points=2)]
        result = calculate_player_metrics(player, records)
        assert result.points_std_dev_reconstructed is None


class TestGameweekStatsDirect:
    def test_minutes_can_exceed_120(self) -> None:
        stats = GameweekStats(
            minutes=180, goals=2, assists=0, goals_conceded=1,
            saves=0, penalties_saved=0, penalties_missed=0,
            yellow_cards=0, red_cards=0, own_goals=0,
            clean_sheet=False, bonus=0,
            total_points_reconstructed=12, total_points_observed=12,
            fixture_count=2,
        )
        assert stats.minutes == 180

    def test_negative_minutes_raises(self) -> None:
        with pytest.raises(MetricsError, match="minutes must be non-negative"):
            GameweekStats(
                minutes=-1, goals=0, assists=0, goals_conceded=0,
                saves=0, penalties_saved=0, penalties_missed=0,
                yellow_cards=0, red_cards=0, own_goals=0,
                clean_sheet=False, bonus=0,
                total_points_reconstructed=0, total_points_observed=0,
                fixture_count=1,
            )

    def test_zero_fixture_count_raises(self) -> None:
        with pytest.raises(MetricsError, match="fixture_count must be >= 1"):
            GameweekStats(
                minutes=0, goals=0, assists=0, goals_conceded=0,
                saves=0, penalties_saved=0, penalties_missed=0,
                yellow_cards=0, red_cards=0, own_goals=0,
                clean_sheet=False, bonus=0,
                total_points_reconstructed=0, total_points_observed=None,
                fixture_count=0,
            )


class TestFrozenDataclass:
    def test_player_metrics_is_frozen(self) -> None:
        player = _make_player(PlayerPosition.MIDFIELDER)
        result = calculate_player_metrics(player, [])
        with pytest.raises((AttributeError, Exception)):
            result.player_id = 999  # type: ignore[misc]

    def test_fixture_record_is_frozen(self) -> None:
        ms = _make_match_stats()
        decomp = calculate_fpl_points(PlayerPosition.MIDFIELDER, ms)
        record = FixtureGameweekRecord(
            gameweek=1, fixture_id=1, match_stats=ms,
            decomposition=decomp, observed_points=None,
        )
        with pytest.raises((AttributeError, Exception)):
            record.gameweek = 2  # type: ignore[misc]


class TestRegressionScoringUnchanged:
    def test_scoring_still_works(self) -> None:
        ms = MatchStats(minutes=90, goals=1, assists=1, clean_sheet=True)
        result = calculate_fpl_points(PlayerPosition.MIDFIELDER, ms)
        assert result.appearance_points == 2
        assert result.goal_points == 5
        assert result.assist_points == 3
        assert result.clean_sheet_points == 1
        assert result.total == 11

    def test_match_stats_validation_still_enforced(self) -> None:
        with pytest.raises(ValueError):
            MatchStats(minutes=200)

    def test_scoring_rules_immutable(self) -> None:
        rules = DEFAULT_SCORING_RULES
        assert rules.clean_sheet_points[PlayerPosition.GOALKEEPER] == 4
        assert rules.clean_sheet_points[PlayerPosition.MIDFIELDER] == 1
        assert PlayerPosition.FORWARD not in rules.clean_sheet_points


class TestIntegrationFullSeason:
    def test_38_gw_season(self) -> None:
        player = _make_player(PlayerPosition.MIDFIELDER)
        records: list[FixtureGameweekRecord] = []
        for gw in range(1, 39):
            rec = _make_record(
                gw, gw * 10, PlayerPosition.MIDFIELDER,
                observed_points=gw % 6, goals=1 if gw % 2 == 0 else 0,
            )
            records.append(rec)
        result = calculate_player_metrics(player, records)
        assert result.sample_size == 38
        assert result.matches_played == 38
        assert result.total_fixtures == 38
        assert result.total_points_observed == sum(gw % 6 for gw in range(1, 39))
        assert result.observed_gw_count == 38
        assert result.points_last_1_observed == 38 % 6
        assert result.points_std_dev_observed is not None
        assert result.clean_sheet_eligible is True
