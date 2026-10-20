"""Tests for providers module."""

from unittest.mock import Mock, patch

import httpx
import pytest

from fpl_engine.config import Settings
from fpl_engine.models import PlayerPosition, PlayerStatus
from fpl_engine.providers import FPLClient
from fpl_engine.providers.protocols import (
    BootstrapDataProvider,
    CompositeDataProvider,
    FixtureDataProvider,
    GameweekDataProvider,
    PlayerDataProvider,
    TeamDataProvider,
)


class TestProtocols:
    """Tests for provider protocols."""

    def test_protocol_inheritance(self) -> None:
        """Test that CompositeDataProvider inherits from all protocols."""
        assert issubclass(CompositeDataProvider, PlayerDataProvider)
        assert issubclass(CompositeDataProvider, TeamDataProvider)
        assert issubclass(CompositeDataProvider, FixtureDataProvider)
        assert issubclass(CompositeDataProvider, GameweekDataProvider)
        assert issubclass(CompositeDataProvider, BootstrapDataProvider)

    def test_fpl_client_implements_composite(self) -> None:
        """Test that FPLClient implements CompositeDataProvider."""
        assert issubclass(FPLClient, CompositeDataProvider)


class TestFPLClient:
    """Tests for FPLClient."""

    @pytest.fixture
    def settings(self) -> Settings:
        """Create test settings."""
        return Settings(
            environment="test",
            fpl_base_url="https://fantasy.premierleague.com/api",
            fpl_timeout_seconds=5.0,
            fpl_max_retries=0,  # No retries in tests
        )

    @pytest.fixture
    def client(self, settings: Settings) -> FPLClient:
        """Create FPLClient instance."""
        return FPLClient(settings)

    def test_client_initialization(self, client: FPLClient) -> None:
        """Test client initialization."""
        assert client._settings is not None
        assert client._client is None  # Lazy initialization

    def test_context_manager(self, client: FPLClient) -> None:
        """Test context manager protocol."""
        with client as c:
            assert c is client
            assert c._client is not None
        # Client should be closed after context
        assert client._client is None or client._client.is_closed

    def test_get_bootstrap_data_caches(self, client: FPLClient) -> None:
        """Test bootstrap data caching."""
        mock_data = {"elements": [], "teams": [], "element_types": [], "events": []}

        with patch.object(client, "_request", return_value=mock_data) as mock_request:
            result1 = client.get_bootstrap_data()
            result2 = client.get_bootstrap_data()

            assert result1 == mock_data
            assert result2 == mock_data
            # Should only call _request once due to caching
            mock_request.assert_called_once_with("/bootstrap-static/")

    def test_clear_cache(self, client: FPLClient) -> None:
        """Test cache clearing."""
        mock_data = {"elements": [], "teams": [], "element_types": [], "events": []}

        with patch.object(client, "_request", return_value=mock_data):
            client.get_bootstrap_data()
            assert client._bootstrap_cache is not None

            client.clear_cache()
            assert client._bootstrap_cache is None

    def test_get_players_parses_correctly(self, client: FPLClient) -> None:
        """Test player parsing from bootstrap data."""
        mock_bootstrap = {
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
            "events": [],
        }

        with patch.object(client, "_request", return_value=mock_bootstrap):
            players = client.get_players()

        assert len(players) == 1
        player = players[0]
        assert player.id == 1
        assert player.web_name == "Salah"
        assert player.first_name == "Mohamed"
        assert player.second_name == "Salah"
        assert player.team_id == 14
        assert player.position == PlayerPosition.MIDFIELDER
        assert player.status == PlayerStatus.AVAILABLE
        assert player.now_cost == 130
        assert player.price == 13.0
        assert player.full_name == "Mohamed Salah"

    def test_get_teams_parses_correctly(self, client: FPLClient) -> None:
        """Test team parsing from bootstrap data."""
        mock_bootstrap = {
            "elements": [],
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
            "element_types": [],
            "events": [],
        }

        with patch.object(client, "_request", return_value=mock_bootstrap):
            teams = client.get_teams()

        assert len(teams) == 1
        team = teams[0]
        assert team.id == 14
        assert team.name == "Liverpool"
        assert team.short_name == "LIV"
        assert team.strength == 5

    def test_get_fixtures_parses_correctly(self, client: FPLClient) -> None:
        """Test fixture parsing."""
        mock_fixtures = [
            {
                "id": 1,
                "code": 12345,
                "event": 1,
                "team_h": 14,
                "team_a": 1,
                "team_h_score": None,
                "team_a_score": None,
                "finished": False,
                "started": False,
                "minutes": 0,
                "provisional_start_time": "2024-08-17T14:00:00Z",
                "kickoff_time": "2024-08-17T14:00:00Z",
                "event_name": "Gameweek 1",
                "team_h_difficulty": 2,
                "team_a_difficulty": 4,
                "stats": [],
                "team_h_name": "LIV",
                "team_a_name": "ARS",
            }
        ]

        with patch.object(client, "_request", return_value=mock_fixtures):
            fixtures = client.get_fixtures()

        assert len(fixtures) == 1
        fixture = fixtures[0]
        assert fixture.id == 1
        assert fixture.event == 1
        assert fixture.team_h == 14
        assert fixture.team_a == 1
        assert fixture.finished is False

    def test_get_fixtures_filters_by_gameweek(self, client: FPLClient) -> None:
        """Test fixture filtering by gameweek."""
        mock_fixtures = [
            {"id": 1, "code": 1, "event": 1, "team_h": 1, "team_a": 2, "finished": False, "started": False, "minutes": 0, "team_h_difficulty": 3, "team_a_difficulty": 3, "stats": []},
            {"id": 2, "code": 2, "event": 2, "team_h": 3, "team_a": 4, "finished": False, "started": False, "minutes": 0, "team_h_difficulty": 3, "team_a_difficulty": 3, "stats": []},
        ]

        with patch.object(client, "_request", return_value=mock_fixtures):
            gw1_fixtures = client.get_fixtures(gameweek=1)
            gw2_fixtures = client.get_fixtures(gameweek=2)
            all_fixtures = client.get_fixtures()

        assert len(gw1_fixtures) == 1
        assert gw1_fixtures[0].event == 1
        assert len(gw2_fixtures) == 1
        assert gw2_fixtures[0].event == 2
        assert len(all_fixtures) == 2

    def test_get_gameweeks_parses_correctly(self, client: FPLClient) -> None:
        """Test gameweek parsing."""
        mock_bootstrap = {
            "elements": [],
            "teams": [],
            "element_types": [],
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

        with patch.object(client, "_request", return_value=mock_bootstrap):
            gameweeks = client.get_gameweeks()

        assert len(gameweeks) == 1
        gw = gameweeks[0]
        assert gw.id == 1
        assert gw.name == "Gameweek 1"
        assert gw.is_current is True
        assert gw.is_active is True

    def test_get_current_gameweek(self, client: FPLClient) -> None:
        """Test getting current gameweek."""
        mock_bootstrap = {
            "elements": [],
            "teams": [],
            "element_types": [],
            "events": [
                {"id": 1, "name": "Gameweek 1", "deadline_time": "2024-08-16T17:30:00Z", "deadline_time_epoch": 1, "deadline_time_game_offset": 0, "is_current": True, "is_next": False},
                {"id": 2, "name": "Gameweek 2", "deadline_time": "2024-08-23T17:30:00Z", "deadline_time_epoch": 2, "deadline_time_game_offset": 0, "is_current": False, "is_next": True},
            ],
        }

        with patch.object(client, "_request", return_value=mock_bootstrap):
            current = client.get_current_gameweek()
            next_gw = client.get_next_gameweek()

        assert current is not None
        assert current.id == 1
        assert next_gw is not None
        assert next_gw.id == 2

    def test_request_retry_on_5xx(self, settings: Settings) -> None:
        """Test retry logic on 5xx errors - should succeed after retries."""
        # Create client with retries
        settings.fpl_max_retries = 2
        settings.fpl_retry_backoff_factor = 0.01
        retry_client = FPLClient(settings)

        call_count = 0

        def mock_get(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                exc = httpx.HTTPStatusError("Server Error", request=Mock(), response=Mock(status_code=500))
                raise exc
            return Mock(json=lambda: {"elements": [], "teams": [], "element_types": [], "events": []}, status_code=200)

        with patch.object(retry_client._get_client(), "get", side_effect=mock_get):
            # Should succeed after 2 retries (3 total calls)
            result = retry_client.get_bootstrap_data()
            assert result == {"elements": [], "teams": [], "element_types": [], "events": []}

        assert call_count == 3  # Initial + 2 retries

    def test_request_raises_on_4xx(self, client: FPLClient) -> None:
        """Test that 4xx errors are raised immediately without retry."""
        def mock_request(*_args, **_kwargs):
            exc = httpx.HTTPStatusError("Not Found", request=Mock(), response=Mock(status_code=404))
            raise exc

        with patch.object(client, "_request", side_effect=mock_request), pytest.raises(httpx.HTTPStatusError):
            client.get_bootstrap_data()

    def test_safe_float(self) -> None:
        """Test _safe_float helper."""
        assert FPLClient._safe_float("1.5") == 1.5
        assert FPLClient._safe_float(1.5) == 1.5
        assert FPLClient._safe_float(1) == 1.0
        assert FPLClient._safe_float(None) is None
        assert FPLClient._safe_float("") is None
        assert FPLClient._safe_float("invalid") is None


class TestFPLClientErrorHandling:
    """Tests for error handling in FPLClient."""

    @pytest.fixture
    def settings(self) -> Settings:
        """Create test settings."""
        return Settings(
            environment="test",
            fpl_base_url="https://fantasy.premierleague.com/api",
            fpl_timeout_seconds=5.0,
            fpl_max_retries=0,
        )

    @pytest.fixture
    def client(self, settings: Settings) -> FPLClient:
        """Create FPLClient instance."""
        return FPLClient(settings)

    def test_network_error_raises(self, client: FPLClient) -> None:
        """Test that network errors are raised."""
        def mock_get(*_args, **_kwargs):
            exc = httpx.ConnectError("Connection failed")
            raise exc

        with patch.object(client._get_client(), "get", side_effect=mock_get), pytest.raises(httpx.ConnectError):
            client.get_bootstrap_data()

    def test_timeout_error_raises(self, client: FPLClient) -> None:
        """Test that timeout errors are raised."""
        def mock_get(*_args, **_kwargs):
            exc = httpx.TimeoutException("Request timeout")
            raise exc

        with patch.object(client._get_client(), "get", side_effect=mock_get), pytest.raises(httpx.TimeoutException):
            client.get_bootstrap_data()
