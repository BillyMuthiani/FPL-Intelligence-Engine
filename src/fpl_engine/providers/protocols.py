"""Provider protocols for data abstraction."""

from typing import Any, Protocol, runtime_checkable

from ..models import Fixture, Gameweek, Player, Team


@runtime_checkable
class PlayerDataProvider(Protocol):
    """Protocol for player data providers."""

    def get_players(self) -> list[Player]:
        """Get all current players."""
        ...

    def get_player(self, player_id: int) -> Player | None:
        """Get a single player by ID."""
        ...

    def get_player_history(self, player_id: int) -> list[dict[str, Any]]:
        """Get player's gameweek history."""
        ...

    def get_player_gameweek_stats(self, player_id: int, gameweek: int) -> dict[str, Any] | None:
        """Get player stats for a specific gameweek."""
        ...


@runtime_checkable
class TeamDataProvider(Protocol):
    """Protocol for team data providers."""

    def get_teams(self) -> list[Team]:
        """Get all teams."""
        ...

    def get_team(self, team_id: int) -> Team | None:
        """Get a single team by ID."""
        ...


@runtime_checkable
class FixtureDataProvider(Protocol):
    """Protocol for fixture data providers."""

    def get_fixtures(self, gameweek: int | None = None) -> list[Fixture]:
        """Get fixtures, optionally filtered by gameweek."""
        ...

    def get_fixture(self, fixture_id: int) -> Fixture | None:
        """Get a single fixture by ID."""
        ...

    def get_team_fixtures(self, team_id: int, gameweek_start: int | None = None, gameweek_end: int | None = None) -> list[Fixture]:
        """Get fixtures for a specific team."""
        ...


@runtime_checkable
class GameweekDataProvider(Protocol):
    """Protocol for gameweek data providers."""

    def get_gameweeks(self) -> list[Gameweek]:
        """Get all gameweeks."""
        ...

    def get_current_gameweek(self) -> Gameweek | None:
        """Get the current gameweek."""
        ...

    def get_next_gameweek(self) -> Gameweek | None:
        """Get the next gameweek."""
        ...

    def get_gameweek(self, gameweek_id: int) -> Gameweek | None:
        """Get a specific gameweek."""
        ...


@runtime_checkable
class BootstrapDataProvider(Protocol):
    """Protocol for bootstrap static data provider.

    Combines all core data in a single call for efficiency.
    """

    def get_bootstrap_data(self) -> dict[str, Any]:
        """Get full bootstrap-static data from FPL."""
        ...


@runtime_checkable
class CompositeDataProvider(
    PlayerDataProvider,
    TeamDataProvider,
    FixtureDataProvider,
    GameweekDataProvider,
    BootstrapDataProvider,
    Protocol,
):
    """Composite protocol for providers that implement all data access."""
    pass
