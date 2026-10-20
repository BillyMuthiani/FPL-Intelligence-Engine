"""Tests for baseline analytics ranking functions."""

from __future__ import annotations

from unittest import mock

import pytest

from fpl_engine.analytics import (
    PlayerBaseline,
    basic_expected_points,
    calculate_fpl_points,
    compare_players,
    rank_players_by_consistency,
    rank_players_by_form,
    rank_players_by_value,
)
from fpl_engine.analytics.metrics import (
    FixtureGameweekRecord,
    calculate_player_metrics,
)
from fpl_engine.analytics.scoring import MatchStats
from fpl_engine.models import Fixture, Player, PlayerPosition


def _make_player(
    pos: PlayerPosition = PlayerPosition.MIDFIELDER,
    pid: int = 1,
    team_id: int = 1,
    cost: int = 100,
) -> Player:
    """Create a minimal valid Player for testing."""
    return Player(
        id=pid,
        web_name=f"P{pid}",
        first_name="Test",
        second_name=f"Player{pid}",
        team_id=team_id,
        position=pos,
        now_cost=cost,
        code=pid,
        element_type=pos.position_id,
        team_code=team_id,
    )


def _make_match_stats(**overrides) -> MatchStats:
    """Create a MatchStats with sensible defaults, applying overrides."""
    defaults: dict[str, object] = {
        "minutes": 90,
        "goals": 0,
        "assists": 0,
        "goals_conceded": 0,
        "saves": 0,
        "penalties_saved": 0,
        "penalties_missed": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "own_goals": 0,
        "tackles": 0,
        "interceptions": 0,
        "clearances": 0,
        "blocks": 0,
        "recoveries": 0,
        "bonus": 0,
        "clean_sheet": False,
    }
    defaults.update(overrides)
    return MatchStats(**defaults)  # type: ignore[arg-type]


def _make_record(
    gw: int,
    fixture_id: int,
    pos: PlayerPosition = PlayerPosition.MIDFIELDER,
    observed_points: int | None = None,
    **stats_overrides,
) -> FixtureGameweekRecord:
    """Create a FixtureGameweekRecord with scoring decomposition."""
    ms = _make_match_stats(**stats_overrides)
    decomp = calculate_fpl_points(pos, ms)
    return FixtureGameweekRecord(
        gameweek=gw,
        fixture_id=fixture_id,
        match_stats=ms,
        decomposition=decomp,
        observed_points=observed_points,
    )


# ---------------------------------------------------------------------------
# Basic expected points proxy
# ---------------------------------------------------------------------------


class TestBasicExpectedPoints:
    def test_midfielder_xg_xa(self) -> None:
        player = _make_player(PlayerPosition.MIDFIELDER)
        result = basic_expected_points(player, expected_goals=10.0, expected_assists=5.0)
        # MID goal = 5 pts, assist = 3 pts
        # 10*5 + 5*3 = 50 + 15 = 65
        assert result == 65.0

    def test_forward_goal_value(self) -> None:
        player = _make_player(PlayerPosition.FORWARD)
        result = basic_expected_points(player, expected_goals=5.0, expected_assists=2.0)
        # FWD goal = 4 pts, assist = 3 pts
        # 5*4 + 2*3 = 20 + 6 = 26
        assert result == 26.0

    def test_defender_goal_value(self) -> None:
        player = _make_player(PlayerPosition.DEFENDER)
        result = basic_expected_points(player, expected_goals=3.0, expected_assists=1.0)
        # DEF goal = 6 pts, assist = 3 pts
        # 3*6 + 1*3 = 18 + 3 = 21
        assert result == 21.0

    def test_goalkeeper_goal_value(self) -> None:
        player = _make_player(PlayerPosition.GOALKEEPER)
        result = basic_expected_points(player, expected_goals=1.0, expected_assists=0.0)
        # GKP goal = 10 pts
        assert result == 10.0

    def test_falls_back_to_player_xg(self) -> None:
        player = Player(
            id=1,
            web_name="P1",
            first_name="Test",
            second_name="Player1",
            team_id=1,
            position=PlayerPosition.MIDFIELDER,
            now_cost=100,
            code=1,
            element_type=3,
            team_code=1,
            expected_goals=8.0,
            expected_assists=4.0,
        )
        result = basic_expected_points(player)
        # 8*5 + 4*3 = 40 + 12 = 52
        assert result == 52.0

    def test_bonus_share(self) -> None:
        player = _make_player(PlayerPosition.MIDFIELDER)
        result = basic_expected_points(
            player, expected_goals=0.0, expected_assists=0.0, bonus_share=0.5
        )
        # Only bonus: 0.5 * 2.0 = 1.0
        assert result == 1.0

    def test_bonus_share_capped(self) -> None:
        player = _make_player(PlayerPosition.MIDFIELDER)
        result = basic_expected_points(
            player, expected_goals=0.0, expected_assists=0.0, bonus_share=2.0
        )
        # Capped at 1.0 * 2.0 = 2.0
        assert result == 2.0

    def test_no_xg_or_xa_returns_none(self) -> None:
        player = _make_player(PlayerPosition.MIDFIELDER)
        assert basic_expected_points(player) is None

    def test_only_xa_supplied(self) -> None:
        """When xG is missing, it is treated as 0 and only xA contributes."""
        player = _make_player(PlayerPosition.MIDFIELDER)
        result = basic_expected_points(player, expected_goals=None, expected_assists=5.0)
        # xG=0 (missing), xA=5 → 0*5 + 5*3 = 15
        assert result == 15.0

    def test_only_xg_supplied(self) -> None:
        """When xA is missing, it is treated as 0 and only xG contributes."""
        player = _make_player(PlayerPosition.MIDFIELDER)
        result = basic_expected_points(
            player, expected_goals=10.0, expected_assists=None
        )
        # xG=10, xA=0 (missing) -> 10*5 + 0*3 = 50
        assert result == 50.0

    def test_negative_xa_returns_none(self) -> None:
        """Negative xA is treated as invalid data → None."""
        player = _make_player(PlayerPosition.MIDFIELDER)
        assert basic_expected_points(
            player, expected_goals=5.0, expected_assists=-1.0
        ) is None

    def test_both_zero_returns_zero(self) -> None:
        """xG=0 and xA=0 is a genuine zero, not None."""
        player = _make_player(PlayerPosition.MIDFIELDER)
        result = basic_expected_points(
            player, expected_goals=0.0, expected_assists=0.0
        )
        assert result == 0.0

    def test_bonus_with_xg_and_xa(self) -> None:
        """bonus_share adds a capped component to xG + xA estimate."""
        player = _make_player(PlayerPosition.FORWARD)
        result = basic_expected_points(
            player, expected_goals=2.0, expected_assists=1.0, bonus_share=0.5
        )
        # FWD goal = 4, assist = 3
        # 2*4 + 1*3 + 0.5*2 = 8 + 3 + 1 = 12
        assert result == 12.0

    def test_negative_xg_returns_none(self) -> None:
        player = _make_player(PlayerPosition.MIDFIELDER)
        assert basic_expected_points(
            player, expected_goals=-1.0, expected_assists=2.0
        ) is None


# ---------------------------------------------------------------------------
# PlayerBaseline total_points_observed property
# ---------------------------------------------------------------------------


class TestPlayerBaseline:
    def test_total_points_observed_property(self) -> None:
        player = _make_player(pid=1, team_id=1)
        records = [
            _make_record(gw=1, fixture_id=1, observed_points=6, goals=1),
            _make_record(gw=2, fixture_id=2, observed_points=8, assists=1),
        ]
        metrics = calculate_player_metrics(player, records)
        assert metrics.total_points_observed == 14

    def test_total_points_observed_delegates_to_metrics(self) -> None:
        player = _make_player(pid=1, team_id=1)
        records = [
            _make_record(gw=g, fixture_id=g, observed_points=10)
            for g in range(1, 5)
        ]
        metrics = calculate_player_metrics(player, records)
        baseline = PlayerBaseline(
            player_id=player.id,
            name=player.web_name,
            full_name=player.full_name,
            position=player.position,
            team_id=player.team_id,
            price_tenths=player.now_cost,
            metrics=metrics,
            points_per_million_value=None,
            avg_fixture_difficulty=None,
            form_3gw=metrics.points_last_3_avg_observed,
            form_5gw=metrics.points_last_5_avg_observed,
            consistency_std=metrics.points_std_dev_observed,
            cutoff_gameweek=None,
        )
        assert baseline.total_points_observed == metrics.total_points_observed


# ---------------------------------------------------------------------------
# rank_players_by_value
# ---------------------------------------------------------------------------


class TestRankPlayersByValue:
    def test_sorts_by_ppm_descending(self) -> None:
        """Higher PPM should rank above lower PPM."""
        player_a = _make_player(pid=1, cost=100, team_id=1)  # 40 pts → 4.0 ppm
        player_b = _make_player(pid=2, cost=100, team_id=2)  # 80 pts → 8.0 ppm

        fixtures_by_player = {
            1: [_make_record(gw=g, fixture_id=g, observed_points=10) for g in range(1, 5)],
            2: [_make_record(gw=g, fixture_id=100 + g, observed_points=20) for g in range(1, 5)],
        }

        result = rank_players_by_value([player_a, player_b], fixtures_by_player)

        assert len(result) == 2
        assert result[0].player_id == 2
        assert result[1].player_id == 1
        assert result[0].points_per_million_value == pytest.approx(8.0)
        assert result[1].points_per_million_value == pytest.approx(4.0)

    def test_excludes_min_gameweeks_filter(self) -> None:
        """Players with fewer than min_gameweeks are excluded."""
        player = _make_player(pid=1, cost=100)
        fixtures_by_player = {
            1: [
                _make_record(gw=1, fixture_id=1, observed_points=10),
                _make_record(gw=2, fixture_id=2, observed_points=10),
            ],
        }
        result = rank_players_by_value([player], fixtures_by_player)
        assert len(result) == 0

    def test_excludes_zero_price(self) -> None:
        """Players with now_cost=0 are excluded."""
        player = _make_player(pid=1, cost=0)
        fixtures_by_player = {
            1: [_make_record(gw=g, fixture_id=g, observed_points=10) for g in range(1, 5)],
        }
        result = rank_players_by_value([player], fixtures_by_player)
        assert len(result) == 0

    def test_tie_break_total_points(self) -> None:
        """When PPM is equal, higher total points wins."""
        player_a = _make_player(pid=1, cost=200)  # 40 pts → 2.0 ppm
        player_b = _make_player(pid=2, cost=300)  # 60 pts → 2.0 ppm

        fixtures_by_player = {
            1: [_make_record(gw=g, fixture_id=g, observed_points=10) for g in range(1, 5)],
            2: [_make_record(gw=g, fixture_id=100 + g, observed_points=15) for g in range(1, 5)],
        }

        result = rank_players_by_value([player_a, player_b], fixtures_by_player)
        assert result[0].player_id == 2
        assert result[1].player_id == 1

    def test_tie_break_total_minutes(self) -> None:
        """When PPM and total points are equal, higher total minutes wins."""
        player_a = _make_player(pid=1, cost=100)
        player_b = _make_player(pid=2, cost=100)

        fixtures_by_player = {
            1: [_make_record(gw=g, fixture_id=g, observed_points=10, minutes=90) for g in range(1, 5)],
            2: [_make_record(gw=g, fixture_id=100 + g, observed_points=10, minutes=60) for g in range(1, 5)],
        }

        result = rank_players_by_value([player_a, player_b], fixtures_by_player)
        assert result[0].player_id == 1
        assert result[1].player_id == 2

    def test_tie_break_player_id(self) -> None:
        """When all else is equal, lower player_id wins."""
        player_a = _make_player(pid=1, cost=100)
        player_b = _make_player(pid=2, cost=100)

        fixtures_by_player = {
            1: [_make_record(gw=g, fixture_id=g, observed_points=10, minutes=90) for g in range(1, 5)],
            2: [_make_record(gw=g, fixture_id=100 + g, observed_points=10, minutes=90) for g in range(1, 5)],
        }

        result = rank_players_by_value([player_b, player_a], fixtures_by_player)
        assert result[0].player_id == 1
        assert result[1].player_id == 2

    def test_fdr_respects_cutoff(self) -> None:
        """Fix B: cutoff_gameweek must filter fixtures in FDR calculation."""
        player = _make_player(pid=1, cost=100, team_id=10)
        fixtures = [
            Fixture(
                id=i, code=i, event=gw, team_h=10, team_a=20,
                finished=True, team_h_difficulty=diff,
            )
            for i, gw, diff in [
                (1, 1, 2), (2, 2, 3), (3, 3, 4), (4, 4, 5),
                (5, 5, 1), (6, 6, 2), (7, 7, 3), (8, 8, 4),
            ]
        ]
        fixtures_by_team = {10: fixtures}
        fixtures_by_player = {
            1: [_make_record(gw=g, fixture_id=g, observed_points=10) for g in range(1, 5)],
        }

        result_no_cutoff = rank_players_by_value(
            [player], fixtures_by_player, fixtures_by_team=fixtures_by_team
        )
        result_cutoff = rank_players_by_value(
            [player], fixtures_by_player, fixtures_by_team=fixtures_by_team,
            cutoff_gameweek=4,
        )

        # No cutoff: (2+3+4+5+1+2+3+4)/8 = 3.0
        assert result_no_cutoff[0].avg_fixture_difficulty == pytest.approx(3.0)
        # Cutoff=4: (2+3+4+5)/4 = 3.5
        assert result_cutoff[0].avg_fixture_difficulty == pytest.approx(3.5)

    def test_zero_ppm_players_tie_break(self) -> None:
        """Players with 0 points and positive cost have PPM=0.0 (not None)
        and should be tie-broken by player_id."""
        player_a = _make_player(pid=1, cost=100)
        player_b = _make_player(pid=2, cost=100)

        fixtures_by_player = {
            1: [_make_record(gw=g, fixture_id=g, observed_points=0) for g in range(1, 6)],
            2: [_make_record(gw=g, fixture_id=100 + g, observed_points=0) for g in range(1, 6)],
        }

        result = rank_players_by_value([player_b, player_a], fixtures_by_player)

        assert len(result) == 2
        assert [b.player_id for b in result] == [1, 2]
        assert result[0].points_per_million_value == 0.0

    def test_cutoff_gameweek_filters_future(self) -> None:
        """cutoff_gameweek should exclude records beyond it."""
        player = _make_player(pid=1, cost=100)
        fixtures_by_player = {
            1: [_make_record(gw=g, fixture_id=g, observed_points=10) for g in range(1, 7)],
        }
        result = rank_players_by_value([player], fixtures_by_player, cutoff_gameweek=4)
        assert len(result) == 1
        assert result[0].metrics.observed_gw_count == 4
        assert result[0].total_points_observed == 40

    def test_cutoff_excludes_below_min(self) -> None:
        """cutoff that leaves fewer than min_gameweeks should exclude."""
        player = _make_player(pid=1, cost=100)
        fixtures_by_player = {
            1: [_make_record(gw=g, fixture_id=g, observed_points=10) for g in range(1, 7)],
        }
        result = rank_players_by_value(
            [player], fixtures_by_player, cutoff_gameweek=3, min_gameweeks=4
        )
        assert len(result) == 0

    def test_fixtures_by_team_context(self) -> None:
        """Passing fixtures_by_team should compute avg_fixture_difficulty."""
        player = _make_player(pid=1, cost=100, team_id=14)
        fixtures = [
            Fixture(
                id=1, code=1, event=1, team_h=14, team_a=1,
                team_h_difficulty=2, team_a_difficulty=4,
                kickoff_time="2024-08-17T14:00:00Z", finished=True,
            ),
            Fixture(
                id=2, code=2, event=2, team_h=1, team_a=14,
                team_h_difficulty=3, team_a_difficulty=3,
                kickoff_time="2024-08-24T14:00:00Z", finished=True,
            ),
        ]
        fixtures_by_team = {14: fixtures}
        fixtures_by_player = {
            1: [_make_record(gw=g, fixture_id=g, observed_points=10) for g in range(1, 5)],
        }

        result = rank_players_by_value(
            [player], fixtures_by_player, fixtures_by_team=fixtures_by_team
        )
        assert len(result) == 1
        # Home diff=2 + away diff=3 → avg=2.5
        assert result[0].avg_fixture_difficulty == pytest.approx(2.5)

    def test_unfinished_fixtures_excluded_from_fdr(self) -> None:
        """Unfinished fixtures should not contribute to avg_fixture_difficulty."""
        player = _make_player(pid=1, cost=100, team_id=14)
        fixtures = [
            Fixture(
                id=1, code=1, event=1, team_h=14, team_a=1,
                team_h_difficulty=2, team_a_difficulty=4,
                kickoff_time="2024-08-17T14:00:00Z", finished=True,
            ),
            Fixture(
                id=2, code=2, event=2, team_h=1, team_a=14,
                team_h_difficulty=3, team_a_difficulty=3,
                kickoff_time="2024-08-24T14:00:00Z", finished=False,
            ),
        ]
        fixtures_by_team = {14: fixtures}
        fixtures_by_player = {
            1: [_make_record(gw=g, fixture_id=g, observed_points=10) for g in range(1, 5)],
        }

        result = rank_players_by_value(
            [player], fixtures_by_player, fixtures_by_team=fixtures_by_team
        )
        # Only finished fixture: home diff=2
        assert result[0].avg_fixture_difficulty == pytest.approx(2.0)

    def test_fixture_difficulty_with_cutoff(self) -> None:
        """cutoff_gameweek should exclude finished fixtures beyond the cutoff.

        Two finished fixtures exist: one before the cutoff and one after.
        Only the one before the cutoff should contribute to avg difficulty.
        """
        player = _make_player(pid=1, cost=100, team_id=14)
        fixtures = [
            Fixture(
                id=1, code=1, event=1, team_h=14, team_a=1,
                team_h_difficulty=2, team_a_difficulty=4,
                kickoff_time="2024-08-17T14:00:00Z", finished=True,
            ),
            Fixture(
                id=2, code=2, event=5, team_h=1, team_a=14,
                team_h_difficulty=5, team_a_difficulty=5,
                kickoff_time="2024-09-14T14:00:00Z", finished=True,
            ),
        ]
        fixtures_by_team = {14: fixtures}
        fixtures_by_player = {
            1: [_make_record(gw=g, fixture_id=g, observed_points=10) for g in range(1, 9)],
        }

        # With cutoff=4: only event=1 fixture (difficulty 2) counts.
        # The fixture at event=5 is finished but beyond cutoff=4.
        result = rank_players_by_value(
            [player], fixtures_by_player, fixtures_by_team=fixtures_by_team,
            cutoff_gameweek=4,
        )
        assert len(result) == 1
        assert result[0].avg_fixture_difficulty == pytest.approx(2.0)

        # Without cutoff: both fixtures count (difficulties 2 + 5).
        result_no_cutoff = rank_players_by_value(
            [player], fixtures_by_player, fixtures_by_team=fixtures_by_team
        )
        assert result_no_cutoff[0].avg_fixture_difficulty == pytest.approx(3.5)

    def test_empty_players_returns_empty(self) -> None:
        assert rank_players_by_value([], {}) == []

    def test_player_without_records_skipped(self) -> None:
        player = _make_player(pid=1, cost=100)
        assert len(rank_players_by_value([player], {})) == 0


# ---------------------------------------------------------------------------
# rank_players_by_consistency
# ---------------------------------------------------------------------------


class TestRankPlayersByConsistency:
    def test_sorts_by_std_ascending(self) -> None:
        """Lower std dev = more consistent = ranks first."""
        player_a = _make_player(pid=1, cost=100)  # always 10 → std=0
        player_b = _make_player(pid=2, cost=100)  # alternating 2, 18 → std=8

        fixtures_by_player = {
            1: [_make_record(gw=g, fixture_id=g, observed_points=10) for g in range(1, 5)],
            2: [
                _make_record(gw=1, fixture_id=1, observed_points=2),
                _make_record(gw=2, fixture_id=2, observed_points=18),
                _make_record(gw=3, fixture_id=3, observed_points=2),
                _make_record(gw=4, fixture_id=4, observed_points=18),
            ],
        }

        result = rank_players_by_consistency([player_a, player_b], fixtures_by_player)
        assert len(result) == 2
        assert result[0].player_id == 1  # std=0.0
        assert result[1].player_id == 2  # std=8.0

    def test_tie_break_total_points(self) -> None:
        """When std is equal, higher total points wins."""
        player_a = _make_player(pid=1, cost=100)  # [3,5,3,5] → std=1.0, total=16
        player_b = _make_player(pid=2, cost=100)  # [4,6,4,6] → std=1.0, total=20

        fixtures_by_player = {
            1: [
                _make_record(gw=1, fixture_id=1, observed_points=3),
                _make_record(gw=2, fixture_id=2, observed_points=5),
                _make_record(gw=3, fixture_id=3, observed_points=3),
                _make_record(gw=4, fixture_id=4, observed_points=5),
            ],
            2: [
                _make_record(gw=1, fixture_id=5, observed_points=4),
                _make_record(gw=2, fixture_id=6, observed_points=6),
                _make_record(gw=3, fixture_id=7, observed_points=4),
                _make_record(gw=4, fixture_id=8, observed_points=6),
            ],
        }

        result = rank_players_by_consistency([player_a, player_b], fixtures_by_player)
        assert result[0].player_id == 2  # Same std, but B has more total points
        assert result[1].player_id == 1

    def test_excludes_min_gameweeks(self) -> None:
        player = _make_player(pid=1, cost=100)
        fixtures_by_player = {
            1: [
                _make_record(gw=1, fixture_id=1, observed_points=10),
                _make_record(gw=2, fixture_id=2, observed_points=10),
            ],
        }
        result = rank_players_by_consistency([player], fixtures_by_player)
        assert len(result) == 0

    def test_empty_returns_empty(self) -> None:
        assert rank_players_by_consistency([], {}) == []


# ---------------------------------------------------------------------------
# rank_players_by_form
# ---------------------------------------------------------------------------


class TestRankPlayersByForm:
    def test_sorts_by_form_5gw_descending(self) -> None:
        """Higher 5-GW form average should rank first."""
        player_a = _make_player(pid=1, cost=100)  # avg 2 pts/GW
        player_b = _make_player(pid=2, cost=100)  # avg 8 pts/GW

        fixtures_by_player = {
            1: [_make_record(gw=g, fixture_id=g, observed_points=2) for g in range(1, 6)],
            2: [_make_record(gw=g, fixture_id=100 + g, observed_points=8) for g in range(1, 6)],
        }

        result = rank_players_by_form([player_a, player_b], fixtures_by_player)
        assert result[0].player_id == 2
        assert result[1].player_id == 1

    def test_form_3gw(self) -> None:
        """Using n_gameweeks=3 should rank by 3-GW form."""
        player_a = _make_player(pid=1, cost=100)  # last 3: [0,10,0] → avg 3.33
        player_b = _make_player(pid=2, cost=100)  # last 3: [10,10,10] → avg 10

        fixtures_by_player = {
            1: [
                _make_record(gw=1, fixture_id=1, observed_points=0),
                _make_record(gw=2, fixture_id=2, observed_points=10),
                _make_record(gw=3, fixture_id=3, observed_points=0),
            ],
            2: [
                _make_record(gw=1, fixture_id=4, observed_points=10),
                _make_record(gw=2, fixture_id=5, observed_points=10),
                _make_record(gw=3, fixture_id=6, observed_points=10),
            ],
        }

        result = rank_players_by_form(
            [player_a, player_b], fixtures_by_player, n_gameweeks=3, min_gameweeks=3
        )
        assert result[0].player_id == 2
        assert result[1].player_id == 1

    def test_tie_break_total_points(self) -> None:
        """When form is equal, higher total points wins."""
        player_a = _make_player(pid=1, cost=100)  # 5 GWs x 5 pts = 25 total
        player_b = _make_player(pid=2, cost=100)  # 6 GWs x 5 pts = 30 total

        fixtures_by_player = {
            1: [_make_record(gw=g, fixture_id=g, observed_points=5) for g in range(1, 6)],
            2: [_make_record(gw=g, fixture_id=100 + g, observed_points=5) for g in range(1, 7)],
        }

        result = rank_players_by_form([player_a, player_b], fixtures_by_player)
        assert result[0].player_id == 2  # More total points
        assert result[1].player_id == 1

    def test_excludes_min_gameweeks(self) -> None:
        player = _make_player(pid=1, cost=100)
        fixtures_by_player = {
            1: [
                _make_record(gw=1, fixture_id=1, observed_points=10),
                _make_record(gw=2, fixture_id=2, observed_points=10),
            ],
        }
        result = rank_players_by_form([player], fixtures_by_player)
        assert len(result) == 0

    def test_form_none_excluded(self) -> None:
        """Player with insufficient data for form window is excluded."""
        player = _make_player(pid=1, cost=100)
        # Only 3 GWs, need 5 for form_5gw
        fixtures_by_player = {
            1: [_make_record(gw=g, fixture_id=g, observed_points=10) for g in range(1, 4)],
        }
        # min_gameweeks=3, but form_5gw needs 5 GWs → None → excluded
        result = rank_players_by_form([player], fixtures_by_player, min_gameweeks=3)
        assert len(result) == 0

    def test_custom_window_falls_back_to_5gw(self) -> None:
        """Custom n_gameweeks not 3 or 5 falls back to 5-gw form."""
        player = _make_player(pid=1, cost=100)
        fixtures_by_player = {
            1: [_make_record(gw=g, fixture_id=g, observed_points=10) for g in range(1, 6)],
        }
        result = rank_players_by_form([player], fixtures_by_player, n_gameweeks=4)
        assert len(result) == 1
        assert result[0].form_5gw == pytest.approx(10.0)

    def test_custom_window_sort_matches_eligibility(self) -> None:
        """Regression: custom n_gameweeks must use 5-gw form for both
        eligibility and sort so rankings never disagree.

        Players C and D are constructed so that their 3-gw and 5-gw form
        averages diverge — sorting by 3-gw would produce a different order
        than sorting by 5-gw.  With n_gameweeks=4 the 5-gw form must be
        used consistently for both inclusion and ordering.
        """
        player_c = _make_player(pid=3, cost=100)  # form_5gw=30, form_3gw=30
        player_d = _make_player(pid=4, cost=100)  # form_5gw=20, form_3gw=33.33

        fixtures_by_player = {
            3: [_make_record(gw=g, fixture_id=g, observed_points=30) for g in range(1, 6)],
            4: [
                _make_record(gw=1, fixture_id=1, observed_points=0),
                _make_record(gw=2, fixture_id=2, observed_points=0),
                _make_record(gw=3, fixture_id=3, observed_points=33),
                _make_record(gw=4, fixture_id=4, observed_points=33),
                _make_record(gw=5, fixture_id=5, observed_points=34),
            ],
        }

        result = rank_players_by_form(
            [player_c, player_d], fixtures_by_player, n_gameweeks=4
        )

        # Only form_5gw should drive both inclusion and ordering.
        assert [b.player_id for b in result] == [3, 4]
        assert result[0].form_5gw == pytest.approx(30.0)
        assert result[1].form_5gw == pytest.approx(20.0)
        # The buggy sort would have put D first (form_3gw=33.33 > C's 30.0).
        assert result[0].form_3gw == pytest.approx(30.0)

    def test_incomplete_dgw_excluded_from_form(self) -> None:
        """An incomplete DGW (one fixture unobserved) should be entirely
        excluded from observed form, not partially counted."""
        player = _make_player(pid=1, cost=100)
        # 6 GWs; GW 3 is a DGW with one finished fixture (100 pts) and
        # one unobserved fixture (observed_points=None).
        fixtures_by_player = {
            1: [
                _make_record(gw=1, fixture_id=1, observed_points=10),
                _make_record(gw=2, fixture_id=2, observed_points=10),
                _make_record(gw=3, fixture_id=30, observed_points=100),
                _make_record(gw=3, fixture_id=31, observed_points=None),
                _make_record(gw=4, fixture_id=4, observed_points=10),
                _make_record(gw=5, fixture_id=5, observed_points=10),
                _make_record(gw=6, fixture_id=6, observed_points=10),
            ],
        }

        result = rank_players_by_form([player], fixtures_by_player)

        assert len(result) == 1
        # GW 3 excluded from observed series → 5 observed GWs, 50 total pts
        assert result[0].metrics.observed_gw_count == 5
        assert result[0].total_points_observed == 50
        assert result[0].form_5gw == pytest.approx(10.0)

    def test_empty_returns_empty(self) -> None:
        assert rank_players_by_form([], {}) == []


# ---------------------------------------------------------------------------
# compare_players
# ---------------------------------------------------------------------------


class TestComparePlayers:
    def test_returns_requested_players(self) -> None:
        player_a = _make_player(pid=1, cost=100)
        player_b = _make_player(pid=2, cost=100)
        fixtures_by_player = {
            1: [_make_record(gw=g, fixture_id=g, observed_points=10) for g in range(1, 5)],
            2: [_make_record(gw=g, fixture_id=100 + g, observed_points=6) for g in range(1, 5)],
        }

        result = compare_players([1, 2], [player_a, player_b], fixtures_by_player)
        assert len(result) == 2
        assert result[0].player_id == 1
        assert result[1].player_id == 2

    def test_no_min_gameweeks_filter(self) -> None:
        """compare_players should not filter by min_gameweeks."""
        player = _make_player(pid=1, cost=100)
        fixtures_by_player = {
            1: [_make_record(gw=1, fixture_id=1, observed_points=10)],
        }

        result = compare_players([1], [player], fixtures_by_player)
        assert len(result) == 1
        assert result[0].metrics.observed_gw_count == 1
        assert result[0].total_points_observed == 10

    def test_missing_player_id_skipped(self) -> None:
        player = _make_player(pid=1, cost=100)
        fixtures_by_player = {
            1: [_make_record(gw=g, fixture_id=g, observed_points=10) for g in range(1, 5)],
        }

        result = compare_players([999, 1], [player], fixtures_by_player)
        assert len(result) == 1
        assert result[0].player_id == 1

    def test_sorted_by_player_id(self) -> None:
        player_a = _make_player(pid=5, cost=100)
        player_b = _make_player(pid=3, cost=100)
        player_c = _make_player(pid=1, cost=100)
        fixtures_by_player = {
            pid: [_make_record(gw=g, fixture_id=g, observed_points=10) for g in range(1, 5)]
            for pid in (1, 3, 5)
        }

        result = compare_players([5, 3, 1], [player_a, player_b, player_c], fixtures_by_player)
        assert len(result) == 3
        assert [b.player_id for b in result] == [1, 3, 5]

    def test_empty_player_ids_returns_empty(self) -> None:
        assert compare_players([], [], {}) == []

    def test_player_without_records(self) -> None:
        """Player with no fixture records still returns a baseline."""
        player = _make_player(pid=1, cost=100)
        result = compare_players([1], [player], {})
        assert len(result) == 1
        assert result[0].metrics.observed_gw_count == 0
        assert result[0].metrics.total_points_observed == 0

    def test_duplicate_ids_deduplicated(self) -> None:
        """Duplicate player IDs in the input should yield each player once."""
        player_a = _make_player(pid=1, cost=100)
        player_b = _make_player(pid=2, cost=100)
        player_c = _make_player(pid=3, cost=100)

        fixtures_by_player = {
            pid: [_make_record(gw=g, fixture_id=g, observed_points=10) for g in range(1, 5)]
            for pid in (1, 2, 3)
        }

        result = compare_players(
            [1, 1, 2, 3, 1, 2],
            [player_a, player_b, player_c],
            fixtures_by_player,
        )

        assert len(result) == 3
        assert [b.player_id for b in result] == [1, 2, 3]

    def test_cutoff_applied(self) -> None:
        """compare_players should respect cutoff_gameweek."""
        player = _make_player(pid=1, cost=100)
        records = [
            _make_record(gw=g, fixture_id=g, observed_points=10)
            for g in range(1, 11)
        ]
        fixtures_by_player = {1: records}

        result = compare_players([1], [player], fixtures_by_player, cutoff_gameweek=5)
        assert len(result) == 1
        assert result[0].metrics.observed_gw_count == 5
        assert result[0].total_points_observed == 50


# ---------------------------------------------------------------------------
# Temporal safety: cutoff filtering
# ---------------------------------------------------------------------------


class TestTemporalSafety:
    def test_cutoff_excludes_future_records(self) -> None:
        """cutoff_gameweek should exclude records where gw > cutoff."""
        player = _make_player(pid=1, cost=100)
        fixtures_by_player = {
            1: [_make_record(gw=g, fixture_id=g, observed_points=10) for g in range(1, 11)],
        }

        result_no_cutoff = rank_players_by_value([player], fixtures_by_player)
        result_cutoff_5 = rank_players_by_value(
            [player], fixtures_by_player, cutoff_gameweek=5
        )

        # Without cutoff: 10 GWs, 100 points; PPM = 100/10 = 10.0
        assert result_no_cutoff[0].metrics.observed_gw_count == 10
        assert result_no_cutoff[0].total_points_observed == 100

        # With cutoff=5: 5 GWs, 50 points; PPM = 50/10 = 5.0
        assert result_cutoff_5[0].metrics.observed_gw_count == 5
        assert result_cutoff_5[0].total_points_observed == 50


# ---------------------------------------------------------------------------
# Metrics computation reuse
# ---------------------------------------------------------------------------


class TestMetricsReuse:
    """Verify that ranking functions compute metrics once per player instead
    of recomputing inside ``_build_baseline`` (Fix C regression)."""

    def test_form_ranking_computes_metrics_once(self) -> None:
        player_a = _make_player(pid=1, cost=100)
        player_b = _make_player(pid=2, cost=100)
        fixtures_by_player = {
            1: [_make_record(gw=g, fixture_id=g, observed_points=10) for g in range(1, 6)],
            2: [_make_record(gw=g, fixture_id=100 + g, observed_points=20) for g in range(1, 6)],
        }

        with mock.patch(
            "fpl_engine.analytics.baseline.calculate_player_metrics",
            side_effect=calculate_player_metrics,
        ) as spy:
            rank_players_by_form([player_a, player_b], fixtures_by_player)

        # One call per player — _build_baseline must NOT call it again.
        assert spy.call_count == 2

    def test_value_ranking_computes_metrics_once(self) -> None:
        player_a = _make_player(pid=1, cost=100)
        player_b = _make_player(pid=2, cost=100)
        fixtures_by_player = {
            1: [_make_record(gw=g, fixture_id=g, observed_points=10) for g in range(1, 5)],
            2: [_make_record(gw=g, fixture_id=100 + g, observed_points=20) for g in range(1, 5)],
        }

        with mock.patch(
            "fpl_engine.analytics.baseline.calculate_player_metrics",
            side_effect=calculate_player_metrics,
        ) as spy:
            rank_players_by_value([player_a, player_b], fixtures_by_player)

        assert spy.call_count == 2
