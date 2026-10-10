"""Tests for the bootstrap normalization layer (offline, no live API)."""
from unittest.mock import patch

import pytest

from fpl_engine.ingestion import (
    IngestionError,
    normalize_gameweeks,
    normalize_players,
    normalize_teams,
)
from fpl_engine.models import PlayerPosition
from fpl_engine.providers import FPLClient


def _player(**over: object) -> dict[str, object]:
    """Minimal complete FPL ``elements`` entry."""
    base: dict[str, object] = {
        "id": 1, "web_name": "Salah", "first_name": "Mohamed",
        "second_name": "Salah", "team": 14, "element_type": 3, "status": "a",
        "now_cost": 130, "code": 123456, "photo": "salah.jpg", "bps": 500,
        "expected_goals": "15.5", "chance_of_playing_this_round": 100,
        "chance_of_playing_next_round": 100, "expected_minutes": 90,
        "penalties_order": 1, "influence": "120.5", "value_season": "1.5",
        "clean_sheets": 0, "goals_conceded": 0,
    }
    return {**base, **over}


def _team(**over: object) -> dict[str, object]:
    """Minimal complete FPL ``teams`` entry."""
    base: dict[str, object] = {
        "id": 14, "code": 3, "name": "Liverpool", "short_name": "LIV",
        "strength": 5, "strength_overall_home": 5, "strength_overall_away": 4,
        "strength_attack_home": 5, "strength_attack_away": 4,
        "strength_defence_home": 4, "strength_defence_away": 4,
        "played": 30, "win": 20, "loss": 5, "draw": 5, "points": 65,
        "position": 2, "goals_for": 70, "goals_against": 30,
        "form": "WWDWL", "pulse_id": 14,
    }
    return {**base, **over}


def _event(**over: object) -> dict[str, object]:
    """Minimal complete FPL ``events`` entry."""
    base: dict[str, object] = {
        "id": 1, "name": "Gameweek 1",
        "deadline_time": "2024-08-16T17:30:00Z",
        "deadline_time_epoch": 1723829400,
        "deadline_time_game_offset": 0, "finished": False, "data_checked": False,
        "is_previous": False, "is_current": True, "is_next": False,
        "cup_leagues_created": False, "h2h_leagues_created": False,
        "transfers_made": 0, "most_captained": 1, "most_vice_captained": 2,
        "average_entry_score": 0, "highest_score": 0, "highest_scoring_entry": None,
        "top_element_info": None, "chip_plays": [],
    }
    return {**base, **over}


class TestNormalizePlayers:
    def test_valid_player_parsed(self) -> None:
        players, report = normalize_players(
            {"elements": [_player()], "teams": [_team()]}
        )
        assert len(players) == 1
        p = players[0]
        assert p.id == 1
        assert p.web_name == "Salah"
        assert p.full_name == "Mohamed Salah"
        assert p.team_id == 14
        assert p.position == PlayerPosition.MIDFIELDER
        assert p.now_cost == 130
        assert p.team_code == 3  # derived from the teams map
        assert p.expected_goals == 15.5
        assert report.accepted == 1
        assert report.rejected == 0
        assert report.conflicts == 0
        assert report.duplicates_skipped == 0
        assert report.total == 1

    def test_missing_required_field_is_rejected(self) -> None:
        bad = _player()
        del bad["web_name"]
        players, report = normalize_players(
            {"elements": [_player(), bad], "teams": [_team()]}
        )
        assert len(players) == 1
        assert report.accepted == 1
        assert report.rejected == 1
        assert any("web_name" in r.reason for r in report.rejections)

    def test_all_records_invalid_raises_unusable(self) -> None:
        bad = _player()
        del bad["web_name"]
        with pytest.raises(IngestionError):
            normalize_players({"elements": [bad], "teams": [_team()]})

    def test_missing_optional_is_none_genuine_zero_kept(self) -> None:
        elem_missing = _player()
        del elem_missing["expected_goals"]
        elem_zero = _player(id=2, web_name="Zero")
        elem_zero["bps"] = 0
        players, _ = normalize_players(
            {"elements": [elem_missing, elem_zero], "teams": [_team()]}
        )
        assert players[0].expected_goals is None
        assert players[1].bps == 0

    def test_absent_nullable_optional_fields_are_none(self) -> None:
        elem = _player()
        for key in (
            "chance_of_playing_this_round", "chance_of_playing_next_round",
            "expected_minutes", "penalties_order", "influence", "value_season",
        ):
            del elem[key]
        players, report = normalize_players(
            {"elements": [elem], "teams": [_team()]}
        )
        assert report.accepted == 1
        p = players[0]
        assert p.chance_of_playing_this_round is None
        assert p.chance_of_playing_next_round is None
        assert p.expected_minutes is None
        assert p.penalties_order is None
        assert p.influence is None
        assert p.value_season is None

    def test_non_dict_element_rejected(self) -> None:
        players, report = normalize_players(
            {"elements": [_player(), "not-a-dict"], "teams": [_team()]}
        )
        assert len(players) == 1
        assert report.rejected == 1
        assert report.rejections[0].reason == "element is not a dict"

    def test_invalid_status_recorded_as_rejection(self) -> None:
            valid = _player()
            invalid = _player(id=2, web_name="Bad")
            invalid["status"] = "z"
            _players, report = normalize_players(
                {"elements": [valid, invalid], "teams": [_team()]}
            )
            assert report.accepted == 1
            assert report.rejected == 1
            assert report.rejections[0].source_id == "element:2"
            assert "PlayerStatus" in report.rejections[0].reason

    def test_duplicate_identifiers_deduped(self) -> None:
        elem = _player()
        players, report = normalize_players(
            {"elements": [elem, dict(elem)], "teams": [_team()]}
        )
        assert len(players) == 1
        assert report.accepted == 1
        assert report.duplicates_skipped == 1

    def test_conflicting_duplicates_excluded_from_output(self) -> None:
        """Conflicting duplicate IDs are excluded from output (neither wins)."""
        valid = _player(id=2)
        e1 = _player()
        e2 = _player()
        e2["web_name"] = "M Salah"
        players, report = normalize_players(
            {"elements": [valid, e1, e2], "teams": [_team()]}
        )
        assert len(players) == 1
        assert players[0].id == 2
        assert report.accepted == 1
        assert report.conflicts == 1
        assert report.conflicts_detail[0].identifier == "element:1"
        assert report.total == 3

    def test_conflicting_duplicates_order_independent(self) -> None:
        """Conflicting duplicates excluded regardless of which record comes first."""
        valid = _player(id=2)

        def make_conflict():
            a = _player()
            b = _player()
            b["web_name"] = "M Salah"
            return a, b

        a1, b1 = make_conflict()
        r1, p1 = normalize_players(
            {"elements": [valid, a1, b1], "teams": [_team()]}
        )
        a2, b2 = make_conflict()
        r2, p2 = normalize_players(
            {"elements": [valid, b2, a2], "teams": [_team()]}
        )

        assert len(r1) == 1 and len(r2) == 1
        assert r1[0].id == 2 and r2[0].id == 2
        assert p1.accepted == 1 and p2.accepted == 1
        assert p1.conflicts == 1 and p2.conflicts == 1
        assert p1.total == 3 and p2.total == 3

    def test_conflict_permanent_exclusion_aba(self) -> None:
        """Once an ID is conflicted, an exact-duplicate of the first
        record is still counted as a conflict -- the ID is never restored."""
        valid = _player(id=99)
        p1 = _player()
        p2 = _player()
        p2["web_name"] = "X"
        p3 = dict(p1)
        players, report = normalize_players(
            {"elements": [valid, p1, p2, p3], "teams": [_team()]}
        )
        assert len(players) == 1
        assert players[0].id == 99
        assert report.accepted == 1
        assert report.conflicts == 2
        assert report.duplicates_skipped == 0
        assert report.total == 4

    def test_three_way_conflict_abc(self) -> None:
        """All records sharing a conflicting id are counted as conflicts."""
        valid = _player(id=99)
        p1 = _player()
        p2 = _player()
        p2["web_name"] = "X"
        p3 = _player()
        p3["web_name"] = "Y"
        players, report = normalize_players(
            {"elements": [valid, p1, p2, p3], "teams": [_team()]}
        )
        assert len(players) == 1
        assert players[0].id == 99
        assert report.accepted == 1
        assert report.conflicts == 2
        assert report.duplicates_skipped == 0
        assert report.total == 4

    def test_conflict_order_independent_aba(self) -> None:
        """A/B/A sequence produces same counts regardless of record order."""
        def make():
            valid = _player(id=99)
            a = _player()
            b = _player()
            b["web_name"] = "X"
            c = dict(a)
            return valid, a, b, c

        v1, a1, b1, c1 = make()
        r1, p1 = normalize_players(
            {"elements": [v1, a1, b1, c1], "teams": [_team()]}
        )

        v2, a2, b2, c2 = make()
        r2, p2 = normalize_players(
            {"elements": [v2, b2, a2, c2], "teams": [_team()]}
        )

        assert len(r1) == 1 and len(r2) == 1
        assert r1[0].id == 99 and r2[0].id == 99
        assert p1.accepted == 1 and p2.accepted == 1
        assert p1.conflicts == 2 and p2.conflicts == 2
        assert p1.duplicates_skipped == 0 and p2.duplicates_skipped == 0
        assert p1.total == 4 and p2.total == 4

    def test_missing_teams_key_raises(self) -> None:
        """When the teams section is absent, the section is unusable."""
        with pytest.raises(IngestionError) as exc_info:
            normalize_players({"elements": [_player()]})
        assert "players" in str(exc_info.value.__notes__)

    def test_unknown_team_id_rejected(self) -> None:
        """A player referencing a team not in the teams list is rejected."""
        valid = _player(id=2)
        bad = _player(id=1, team=99)
        players, report = normalize_players(
            {"elements": [valid, bad], "teams": [_team()]})
        assert len(players) == 1
        assert players[0].id == 2
        assert report.accepted == 1
        assert report.rejected == 1
        assert "team 99" in report.rejections[0].reason

    def test_empty_teams_section_raises(self) -> None:
        """An empty teams list makes all players unresolvable -> IngestionError."""
        with pytest.raises(IngestionError) as exc_info:
            normalize_players({"elements": [_player()], "teams": []})
        assert "players" in str(exc_info.value.__notes__)

    def test_malformed_payload_raises(self) -> None:
        with pytest.raises(IngestionError):
            normalize_players("not a dict")  # type: ignore[arg-type]
        with pytest.raises(IngestionError):
            normalize_players({"teams": []})
        with pytest.raises(IngestionError):
            normalize_players({"elements": "nope"})

    def test_report_counts_consistent(self) -> None:
        good = _player()
        good_dup = dict(good)
        bad = _player(id=2)
        del bad["now_cost"]
        players, report = normalize_players(
            {"elements": [good, good_dup, bad], "teams": [_team()]}
        )
        assert len(players) == 1
        assert report.accepted == 1
        assert report.duplicates_skipped == 1
        assert report.rejected == 1
        assert len(report.rejections) == 1
        assert report.total == 3


class TestNormalizeTeams:
    def test_valid_team_parsed(self) -> None:
        teams, report = normalize_teams({"teams": [_team()]})
        assert len(teams) == 1
        assert teams[0].short_name == "LIV"
        assert teams[0].code == 3
        assert report.accepted == 1

    def test_missing_required_field_rejected(self) -> None:
        bad = _team()
        del bad["short_name"]
        teams, report = normalize_teams({"teams": [_team(), bad]})
        assert len(teams) == 1
        assert report.rejected == 1

    def test_all_invalid_raises(self) -> None:
        bad = _team()
        del bad["name"]
        with pytest.raises(IngestionError):
            normalize_teams({"teams": [bad]})

    def test_duplicate_teams_deduped(self) -> None:
        teams, report = normalize_teams({"teams": [_team(), _team()]})
        assert len(teams) == 1
        assert report.duplicates_skipped == 1

    def test_conflicting_duplicate_teams(self) -> None:
        t1 = _team()
        t2 = _team()
        t2["name"] = "Liverpool FC"
        valid = _team(id=15, code=4, name="Arsenal", short_name="ARS")
        teams, report = normalize_teams({"teams": [valid, t1, t2]})
        assert len(teams) == 1
        assert teams[0].id == 15
        assert report.accepted == 1
        assert report.conflicts == 1
        assert report.total == 3

    def test_conflict_permanent_exclusion_aba(self) -> None:
        """Once an ID is conflicted, an exact-duplicate of the first
        record is still counted as a conflict -- the ID is never restored."""
        valid = _team(id=15, code=4, name="Arsenal", short_name="ARS")
        t1 = _team()
        t2 = _team()
        t2["name"] = "Liverpool FC"
        t3 = dict(t1)
        teams, report = normalize_teams({"teams": [valid, t1, t2, t3]})
        assert len(teams) == 1
        assert teams[0].id == 15
        assert report.accepted == 1
        assert report.conflicts == 2
        assert report.duplicates_skipped == 0
        assert report.total == 4

    def test_three_way_conflict_abc(self) -> None:
        """All records sharing a conflicting id are counted as conflicts."""
        valid = _team(id=15, code=4, name="Arsenal", short_name="ARS")
        t1 = _team()
        t2 = _team()
        t2["name"] = "Liverpool FC"
        t3 = _team()
        t3["short_name"] = "LFC"
        teams, report = normalize_teams({"teams": [valid, t1, t2, t3]})
        assert len(teams) == 1
        assert teams[0].id == 15
        assert report.accepted == 1
        assert report.conflicts == 2
        assert report.duplicates_skipped == 0
        assert report.total == 4

    def test_conflict_order_independent_aba(self) -> None:
        """A/B/A sequence produces same counts regardless of record order."""
        def make():
            valid = _team(id=15, code=4, name="Arsenal", short_name="ARS")
            a = _team()
            b = _team()
            b["name"] = "Liverpool FC"
            c = dict(a)
            return valid, a, b, c

        v1, a1, b1, c1 = make()
        r1, p1 = normalize_teams({"teams": [v1, a1, b1, c1]})

        v2, a2, b2, c2 = make()
        r2, p2 = normalize_teams({"teams": [v2, b2, a2, c2]})

        assert len(r1) == 1 and len(r2) == 1
        assert r1[0].id == 15 and r2[0].id == 15
        assert p1.accepted == 1 and p2.accepted == 1
        assert p1.conflicts == 2 and p2.conflicts == 2
        assert p1.duplicates_skipped == 0 and p2.duplicates_skipped == 0
        assert p1.total == 4 and p2.total == 4

    def test_malformed_payload_raises(self) -> None:
        with pytest.raises(IngestionError):
            normalize_teams("x")  # type: ignore[arg-type]
        with pytest.raises(IngestionError):
            normalize_teams({"elements": []})


class TestNormalizeGameweeks:
    def test_valid_event_parsed(self) -> None:
        gws, report = normalize_gameweeks({"events": [_event()]})
        assert len(gws) == 1
        assert gws[0].id == 1
        assert gws[0].is_current is True
        assert gws[0].is_active is True
        assert report.accepted == 1

    def test_empty_section_returns_empty(self) -> None:
        gws, report = normalize_gameweeks({"events": []})
        assert gws == []
        assert report.accepted == 0
        assert report.total == 0

    def test_missing_required_field_rejected(self) -> None:
        bad = _event()
        del bad["name"]
        gws, report = normalize_gameweeks({"events": [_event(), bad]})
        assert len(gws) == 1
        assert report.rejected == 1

    def test_all_invalid_raises(self) -> None:
        bad = _event()
        del bad["deadline_time"]
        with pytest.raises(IngestionError):
            normalize_gameweeks({"events": [bad]})

    def test_conflicting_duplicate_events(self) -> None:
        e1 = _event()
        e2 = _event()
        e2["is_current"] = False
        e2["is_next"] = True
        valid = _event(id=2)
        gws, report = normalize_gameweeks({"events": [valid, e1, e2]})
        assert len(gws) == 1
        assert gws[0].id == 2
        assert report.accepted == 1
        assert report.conflicts == 1
        assert report.total == 3

    def test_conflicting_duplicate_events_order_independent(self) -> None:
        """Conflicting event duplicates excluded regardless of order."""
        valid = _event(id=2)

        def make_conflict():
            a = _event()
            b = _event()
            b["is_current"] = False
            b["is_next"] = True
            return a, b

        a1, b1 = make_conflict()
        r1, p1 = normalize_gameweeks({"events": [valid, a1, b1]})
        a2, b2 = make_conflict()
        r2, p2 = normalize_gameweeks({"events": [valid, b2, a2]})

        assert len(r1) == 1 and len(r2) == 1
        assert r1[0].id == 2 and r2[0].id == 2
        assert p1.accepted == 1 and p2.accepted == 1
        assert p1.conflicts == 1 and p2.conflicts == 1
        assert p1.total == 3 and p2.total == 3

    def test_conflict_permanent_exclusion_aba(self) -> None:
        """Once an ID is conflicted, an exact-duplicate of the first
        record is still counted as a conflict -- the ID is never restored."""
        valid = _event(id=2)
        e1 = _event()
        e2 = _event()
        e2["is_current"] = False
        e2["is_next"] = True
        e3 = dict(e1)
        gws, report = normalize_gameweeks({"events": [valid, e1, e2, e3]})
        assert len(gws) == 1
        assert gws[0].id == 2
        assert report.accepted == 1
        assert report.conflicts == 2
        assert report.duplicates_skipped == 0
        assert report.total == 4

    def test_three_way_conflict_abc(self) -> None:
        """All records sharing a conflicting id are counted as conflicts."""
        valid = _event(id=2)
        e1 = _event()
        e2 = _event()
        e2["is_current"] = False
        e2["is_next"] = True
        e3 = _event()
        e3["is_previous"] = True
        gws, report = normalize_gameweeks({"events": [valid, e1, e2, e3]})
        assert len(gws) == 1
        assert gws[0].id == 2
        assert report.accepted == 1
        assert report.conflicts == 2
        assert report.duplicates_skipped == 0
        assert report.total == 4

    def test_conflict_order_independent_aba(self) -> None:
        """A/B/A sequence produces same counts regardless of record order."""
        def make():
            valid = _event(id=2)
            a = _event()
            b = _event()
            b["is_current"] = False
            b["is_next"] = True
            c = dict(a)
            return valid, a, b, c

        v1, a1, b1, c1 = make()
        r1, p1 = normalize_gameweeks({"events": [v1, a1, b1, c1]})

        v2, a2, b2, c2 = make()
        r2, p2 = normalize_gameweeks({"events": [v2, b2, a2, c2]})

        assert len(r1) == 1 and len(r2) == 1
        assert r1[0].id == 2 and r2[0].id == 2
        assert p1.accepted == 1 and p2.accepted == 1
        assert p1.conflicts == 2 and p2.conflicts == 2
        assert p1.duplicates_skipped == 0 and p2.duplicates_skipped == 0
        assert p1.total == 4 and p2.total == 4

    def test_malformed_payload_raises(self) -> None:
        with pytest.raises(IngestionError):
            normalize_gameweeks("x")  # type: ignore[arg-type]
        with pytest.raises(IngestionError):
            normalize_gameweeks({"teams": []})


class TestCompatibility:
    def test_players_match_fpl_client(
        self, test_settings, mock_bootstrap_data
    ) -> None:
        client = FPLClient(test_settings)
        with patch.object(client, "_request", return_value=mock_bootstrap_data):
            fpl_players = client.get_players()
        norm_players, report = normalize_players(mock_bootstrap_data)
        assert report.accepted == len(fpl_players)
        assert [p.model_dump() for p in norm_players] == [
            p.model_dump() for p in fpl_players
        ]

    def test_teams_match_fpl_client(
        self, test_settings, mock_bootstrap_data
    ) -> None:
        client = FPLClient(test_settings)
        with patch.object(client, "_request", return_value=mock_bootstrap_data):
            fpl_teams = client.get_teams()
        norm_teams, report = normalize_teams(mock_bootstrap_data)
        assert report.accepted == len(fpl_teams)
        assert [t.model_dump() for t in norm_teams] == [
            t.model_dump() for t in fpl_teams
        ]



