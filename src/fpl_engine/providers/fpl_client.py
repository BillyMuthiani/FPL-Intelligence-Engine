"""Official FPL API client implementation."""

import time
from typing import Any

import httpx

from ..config import Settings, get_settings
from ..logging import get_logger
from ..models import Fixture, Gameweek, Player, PlayerPosition, PlayerStatus, Team
from .protocols import (
    BootstrapDataProvider,
    FixtureDataProvider,
    GameweekDataProvider,
    PlayerDataProvider,
    TeamDataProvider,
)

log = get_logger(__name__)


class FPLClient(
    PlayerDataProvider,
    TeamDataProvider,
    FixtureDataProvider,
    GameweekDataProvider,
    BootstrapDataProvider,
):
    """Official FPL API client implementing all provider protocols."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._client: httpx.Client | None = None
        self._bootstrap_cache: dict[str, Any] | None = None

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self._settings.fpl_base_url,
                timeout=self._settings.fpl_timeout_seconds,
                headers={
                    "User-Agent": "FPL-Intelligence-Engine/0.1.0",
                    "Accept": "application/json",
                },
            )
        return self._client

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "FPLClient":
        self._get_client()  # Initialize client on enter
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        self.close()

    def _request(self, endpoint: str) -> Any:
        """Make a GET request with retry logic."""
        client = self._get_client()
        max_retries = self._settings.fpl_max_retries
        backoff = self._settings.fpl_retry_backoff_factor

        last_exception: BaseException | None = None
        for attempt in range(max_retries + 1):
            try:
                response = client.get(endpoint)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                last_exception = e
                if e.response.status_code >= 500 and attempt < max_retries:
                    time.sleep(backoff * (2**attempt))
                    continue
                log.error(
                    "FPL API request failed",
                    endpoint=endpoint,
                    status_code=e.response.status_code,
                    attempt=attempt + 1,
                )
                raise
            except httpx.RequestError as e:
                last_exception = e
                if attempt < max_retries:
                    time.sleep(backoff * (2**attempt))
                    continue
                log.error("FPL API request error", endpoint=endpoint, error=str(e), attempt=attempt + 1)
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Request failed without exception")

    def get_bootstrap_data(self) -> dict[str, Any]:
        """Get full bootstrap-static data."""
        if self._bootstrap_cache is None:
            self._bootstrap_cache = self._request("/bootstrap-static/")
        return self._bootstrap_cache

    def clear_cache(self) -> None:
        """Clear bootstrap cache."""
        self._bootstrap_cache = None

    # PlayerDataProvider implementation
    def get_players(self) -> list[Player]:
        """Get all current players."""
        data = self.get_bootstrap_data()
        elements = data.get("elements", [])
        teams = {t["id"]: t for t in data.get("teams", [])}
        element_types = {et["id"]: et for et in data.get("element_types", [])}

        players = []
        for elem in elements:
            team_data = teams.get(elem["team"], {})
            position_data = element_types.get(elem["element_type"], {})

            player = Player(
                id=elem["id"],
                web_name=elem["web_name"],
                first_name=elem["first_name"],
                second_name=elem["second_name"],
                team_id=elem["team"],
                position=PlayerPosition.from_id(elem["element_type"]),
                status=PlayerStatus(elem.get("status", "a")),
                now_cost=elem["now_cost"],
                cost_change_start=elem.get("cost_change_start", 0),
                cost_change_event=elem.get("cost_change_event", 0),
                cost_change_event_fall=elem.get("cost_change_event_fall", 0),
                cost_change_start_fall=elem.get("cost_change_start_fall", 0),
                selected_by_percent=float(elem.get("selected_by_percent", 0)),
                transfers_in=elem.get("transfers_in", 0),
                transfers_out=elem.get("transfers_out", 0),
                transfers_in_event=elem.get("transfers_in_event", 0),
                transfers_out_event=elem.get("transfers_out_event", 0),
                form=float(elem.get("form", 0)),
                total_points=elem.get("total_points", 0),
                points_per_game=float(elem.get("points_per_game", 0)),
                event_points=elem.get("event_points", 0),
                expected_goals=self._safe_float(elem.get("expected_goals")),
                expected_assists=self._safe_float(elem.get("expected_assists")),
                expected_goal_involvements=self._safe_float(elem.get("expected_goal_involvements")),
                expected_goals_conceded=self._safe_float(elem.get("expected_goals_conceded")),
                expected_goals_per_90=self._safe_float(elem.get("expected_goals_per_90")),
                expected_assists_per_90=self._safe_float(elem.get("expected_assists_per_90")),
                expected_goal_involvements_per_90=self._safe_float(elem.get("expected_goal_involvements_per_90")),
                expected_goals_conceded_per_90=self._safe_float(elem.get("expected_goals_conceded_per_90")),
                minutes=elem.get("minutes", 0),
                starts=elem.get("starts", 0),
                expected_minutes=elem.get("expected_minutes"),
                goals_scored=elem.get("goals_scored", 0),
                assists=elem.get("assists", 0),
                penalties_missed=elem.get("penalties_missed", 0),
                penalties_saved=elem.get("penalties_saved", 0),
                penalties_order=elem.get("penalties_order"),
                direct_freekicks_order=elem.get("direct_freekicks_order"),
                corners_order=elem.get("corners_order"),
                clean_sheets=elem.get("clean_sheets", 0),
                goals_conceded=elem.get("goals_conceded", 0),
                own_goals=elem.get("own_goals", 0),
                saves=elem.get("saves", 0),
                bonus=elem.get("bonus", 0),
                bps=elem.get("bps", 0),
                tackles=elem.get("tackles", 0),
                interceptions=elem.get("interceptions", 0),
                clearances=elem.get("clearances", 0),
                blocks=elem.get("blocks", 0),
                influence=self._safe_float(elem.get("influence")),
                creativity=self._safe_float(elem.get("creativity")),
                threat=self._safe_float(elem.get("threat")),
                ict_index=self._safe_float(elem.get("ict_index")),
                value_form=self._safe_float(elem.get("value_form")),
                value_season=self._safe_float(elem.get("value_season")),
                in_dreamteam=elem.get("in_dreamteam", False),
                dreamteam_count=elem.get("dreamteam_count", 0),
                code=elem["code"],
                photo=elem.get("photo", ""),
                element_type=elem["element_type"],
                team_code=team_data.get("code", 0),
                chance_of_playing_this_round=elem.get("chance_of_playing_this_round"),
                chance_of_playing_next_round=elem.get("chance_of_playing_next_round"),
            )
            players.append(player)

        return players

    def get_player(self, player_id: int) -> Player | None:
        """Get a single player by ID."""
        players = self.get_players()
        for player in players:
            if player.id == player_id:
                return player
        return None

    def get_player_history(self, player_id: int) -> list[dict[str, Any]]:
        """Get player's gameweek history."""
        data = self._request(f"/element-summary/{player_id}/")
        return data.get("history", [])  # type: ignore[no-any-return]

    def get_player_gameweek_stats(self, player_id: int, gameweek: int) -> dict[str, Any] | None:
        """Get player stats for a specific gameweek."""
        history = self.get_player_history(player_id)
        for entry in history:
            if entry.get("round") == gameweek:
                return entry
        return None

    # TeamDataProvider implementation
    def get_teams(self) -> list[Team]:
        """Get all teams."""
        data = self.get_bootstrap_data()
        teams_data = data.get("teams", [])

        teams = []
        for t in teams_data:
            team = Team(
                id=t["id"],
                code=t["code"],
                name=t["name"],
                short_name=t["short_name"],
                strength=t.get("strength", 0),
                strength_overall_home=t.get("strength_overall_home", 0),
                strength_overall_away=t.get("strength_overall_away", 0),
                strength_attack_home=t.get("strength_attack_home", 0),
                strength_attack_away=t.get("strength_attack_away", 0),
                strength_defence_home=t.get("strength_defence_home", 0),
                strength_defence_away=t.get("strength_defence_away", 0),
                played=t.get("played", 0),
                win=t.get("win", 0),
                loss=t.get("loss", 0),
                draw=t.get("draw", 0),
                points=t.get("points", 0),
                position=t.get("position", 0),
                goals_for=t.get("goals_for", 0),
                goals_against=t.get("goals_against", 0),
                form=t.get("form"),
                pulse_id=t.get("pulse_id"),
            )
            teams.append(team)

        return teams

    def get_team(self, team_id: int) -> Team | None:
        """Get a single team by ID."""
        teams = self.get_teams()
        for team in teams:
            if team.id == team_id:
                return team
        return None

    # FixtureDataProvider implementation
    def get_fixtures(self, gameweek: int | None = None) -> list[Fixture]:
        """Get fixtures, optionally filtered by gameweek."""
        data = self._request("/fixtures/")
        fixtures = []

        for f in data:
            if not isinstance(f, dict):
                continue
            if gameweek is not None and f.get("event") != gameweek:
                continue

            fixture = Fixture(
                id=f["id"],
                code=f["code"],
                event=f.get("event", 0),
                team_h=f["team_h"],
                team_a=f["team_a"],
                team_h_score=f.get("team_h_score"),
                team_a_score=f.get("team_a_score"),
                finished=f.get("finished", False),
                started=f.get("started", False),
                minutes=f.get("minutes", 0),
                provisional_start_time=f.get("provisional_start_time"),
                kickoff_time=f.get("kickoff_time"),
                event_name=f.get("event_name"),
                team_h_difficulty=f.get("team_h_difficulty", 3),
                team_a_difficulty=f.get("team_a_difficulty", 3),
                stats=f.get("stats", []),
                team_h_name=f.get("team_h_name"),
                team_a_name=f.get("team_a_name"),
            )
            fixtures.append(fixture)

        return fixtures

    def get_fixture(self, fixture_id: int) -> Fixture | None:
        """Get a single fixture by ID."""
        fixtures = self.get_fixtures()
        for fixture in fixtures:
            if fixture.id == fixture_id:
                return fixture
        return None

    def get_team_fixtures(
        self, team_id: int, gameweek_start: int | None = None, gameweek_end: int | None = None
    ) -> list[Fixture]:
        """Get fixtures for a specific team."""
        fixtures = self.get_fixtures()
        result = []

        for f in fixtures:
            if f.team_h != team_id and f.team_a != team_id:
                continue
            if gameweek_start is not None and f.event < gameweek_start:
                continue
            if gameweek_end is not None and f.event > gameweek_end:
                continue
            result.append(f)

        return result

    # GameweekDataProvider implementation
    def get_gameweeks(self) -> list[Gameweek]:
        """Get all gameweeks."""
        data = self.get_bootstrap_data()
        events = data.get("events", [])

        gameweeks = []
        for e in events:
            gw = Gameweek(
                id=e["id"],
                name=e["name"],
                deadline_time=e["deadline_time"],
                deadline_time_epoch=e["deadline_time_epoch"],
                deadline_time_game_offset=e["deadline_time_game_offset"],
                finished=e.get("finished", False),
                data_checked=e.get("data_checked", False),
                is_previous=e.get("is_previous", False),
                is_current=e.get("is_current", False),
                is_next=e.get("is_next", False),
                cup_leagues_created=e.get("cup_leagues_created", False),
                h2h_leagues_created=e.get("h2h_leagues_created", False),
                top_element_info=e.get("top_element_info"),
                chip_plays=e.get("chip_plays", []),
                transfers_made=e.get("transfers_made", 0),
                most_captained=e.get("most_captained"),
                most_vice_captained=e.get("most_vice_captained"),
                average_entry_score=e.get("average_entry_score", 0),
                highest_score=e.get("highest_score", 0),
                highest_scoring_entry=e.get("highest_scoring_entry"),
            )
            gameweeks.append(gw)

        return gameweeks

    def get_current_gameweek(self) -> Gameweek | None:
        """Get the current gameweek."""
        for gw in self.get_gameweeks():
            if gw.is_current:
                return gw
        return None

    def get_next_gameweek(self) -> Gameweek | None:
        """Get the next gameweek."""
        for gw in self.get_gameweeks():
            if gw.is_next:
                return gw
        return None

    def get_gameweek(self, gameweek_id: int) -> Gameweek | None:
        """Get a specific gameweek."""
        for gw in self.get_gameweeks():
            if gw.id == gameweek_id:
                return gw
        return None

    @staticmethod
    def _safe_float(value: str | float | int | None) -> float | None:
        """Safely convert value to float."""
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
