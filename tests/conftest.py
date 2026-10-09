"""Pytest configuration and shared fixtures."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from fpl_engine.config import Settings
from fpl_engine.models import Fixture, Gameweek, Player, PlayerPosition, PlayerStatus, Team
from fpl_engine.providers import FPLClient


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Provide test settings."""
    return Settings(
        environment="test",
        debug=True,
        fpl_base_url="https://fantasy.premierleague.com/api",
        fpl_timeout_seconds=5.0,
        log_level="DEBUG",
        log_format="console",
    )


@pytest.fixture
def mock_bootstrap_data() -> dict[str, Any]:
    """Load mock bootstrap data from fixture file."""
    fixture_path = Path(__file__).parent / "fixtures" / "bootstrap_static.json"
    if fixture_path.exists():
        with open(fixture_path) as f:
            return json.load(f)
    return _create_minimal_bootstrap()


@pytest.fixture
def mock_fixtures_data() -> list[dict[str, Any]]:
    """Load mock fixtures data from fixture file."""
    fixture_path = Path(__file__).parent / "fixtures" / "fixtures.json"
    if fixture_path.exists():
        with open(fixture_path) as f:
            return json.load(f)
    return []


@pytest.fixture
def sample_player() -> Player:
    """Create a sample player for testing."""
    return Player(
        id=1,
        web_name="Salah",
        first_name="Mohamed",
        second_name="Salah",
        team_id=14,
        position=PlayerPosition.MIDFIELDER,
        status=PlayerStatus.AVAILABLE,
        now_cost=130,
        code=123456,
        photo="salah.jpg",
        element_type=3,
        team_code=14,
    )


@pytest.fixture
def sample_team() -> Team:
    """Create a sample team for testing."""
    return Team(
        id=14,
        code=3,
        name="Liverpool",
        short_name="LIV",
        strength=5,
        strength_overall_home=5,
        strength_overall_away=4,
        strength_attack_home=5,
        strength_attack_away=4,
        strength_defence_home=4,
        strength_defence_away=4,
    )


@pytest.fixture
def sample_fixture() -> Fixture:
    """Create a sample fixture for testing."""
    return Fixture(
        id=1,
        code=12345,
        event=1,
        team_h=14,
        team_a=1,
        team_h_difficulty=2,
        team_a_difficulty=4,
        kickoff_time="2024-08-17T14:00:00Z",
    )


@pytest.fixture
def sample_gameweek() -> Gameweek:
    """Create a sample gameweek for testing."""
    return Gameweek(
        id=1,
        name="Gameweek 1",
        deadline_time="2024-08-16T17:30:00Z",
        deadline_time_epoch=1723829400,
        deadline_time_game_offset=0,
        is_current=True,
    )


@pytest.fixture
def mock_fpl_client(mock_bootstrap_data: dict[str, Any], mock_fixtures_data: list[dict[str, Any]]) -> Mock:
    """Create a mocked FPL client."""
    client = Mock(spec=FPLClient)
    client.get_bootstrap_data.return_value = mock_bootstrap_data
    client.get_fixtures.return_value = []
    client.get_gameweeks.return_value = []
    client.get_teams.return_value = []
    client.get_players.return_value = []
    return client


def _create_minimal_bootstrap() -> dict[str, Any]:
    """Create minimal bootstrap data for testing."""
    return {
        "elements": [
            {
                "id": 1,
                "web_name": "Salah",
                "first_name": "Mohamed",
                "second_name": "Salah",
                "team": 14,
                "element_type": 3,
                "status": "a",
                "now_cost": 130,
                "cost_change_start": 0,
                "cost_change_event": 0,
                "cost_change_event_fall": 0,
                "cost_change_start_fall": 0,
                "selected_by_percent": "50.0",
                "transfers_in": 100000,
                "transfers_out": 50000,
                "transfers_in_event": 1000,
                "transfers_out_event": 500,
                "form": "8.5",
                "total_points": 200,
                "points_per_game": "7.2",
                "event_points": 8,
                "expected_goals": "15.5",
                "expected_assists": "8.2",
                "expected_goal_involvements": "23.7",
                "expected_goals_conceded": "0.0",
                "expected_goals_per_90": "0.65",
                "expected_assists_per_90": "0.35",
                "expected_goal_involvements_per_90": "1.0",
                "expected_goals_conceded_per_90": "0.0",
                "minutes": 2700,
                "starts": 30,
                "expected_minutes": 90,
                "goals_scored": 18,
                "assists": 10,
                "penalties_missed": 0,
                "penalties_saved": 0,
                "penalties_order": 1,
                "direct_freekicks_order": 2,
                "corners_order": 1,
                "clean_sheets": 0,
                "goals_conceded": 0,
                "own_goals": 0,
                "saves": 0,
                "bonus": 50,
                "bps": 500,
                "tackles": 20,
                "interceptions": 15,
                "clearances": 10,
                "blocks": 5,
                "influence": "120.5",
                "creativity": "95.2",
                "threat": "145.8",
                "ict_index": "361.5",
                "value_form": "1.2",
                "value_season": "1.5",
                "in_dreamteam": True,
                "dreamteam_count": 5,
                "code": 123456,
                "photo": "salah.jpg",
                "chance_of_playing_this_round": 100,
                "chance_of_playing_next_round": 100,
            }
        ],
        "teams": [
            {
                "id": 14,
                "code": 3,
                "name": "Liverpool",
                "short_name": "LIV",
                "strength": 5,
                "strength_overall_home": 5,
                "strength_overall_away": 4,
                "strength_attack_home": 5,
                "strength_attack_away": 4,
                "strength_defence_home": 4,
                "strength_defence_away": 4,
                "played": 30,
                "win": 20,
                "loss": 5,
                "draw": 5,
                "points": 65,
                "position": 2,
                "goals_for": 70,
                "goals_against": 30,
                "form": "WWDWL",
                "pulse_id": 14,
            }
        ],
        "element_types": [
            {"id": 1, "singular_name": "Goalkeeper", "plural_name": "Goalkeepers"},
            {"id": 2, "singular_name": "Defender", "plural_name": "Defenders"},
            {"id": 3, "singular_name": "Midfielder", "plural_name": "Midfielders"},
            {"id": 4, "singular_name": "Forward", "plural_name": "Forwards"},
        ],
        "events": [
            {
                "id": 1,
                "name": "Gameweek 1",
                "deadline_time": "2024-08-16T17:30:00Z",
                "deadline_time_epoch": 1723829400,
                "deadline_time_game_offset": 0,
                "finished": False,
                "data_checked": False,
                "is_previous": False,
                "is_current": True,
                "is_next": False,
                "cup_leagues_created": False,
                "h2h_leagues_created": False,
                "transfers_made": 0,
                "average_entry_score": 0,
                "highest_score": 0,
            }
        ],
    }
