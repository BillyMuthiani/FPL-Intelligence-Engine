"""Squad model for FPL squad representation."""


from pydantic import BaseModel, Field, field_validator

from .player import Player, PlayerPosition


class SquadPlayer(BaseModel):
    """A player in a squad with position info."""

    player: Player = Field(description="Player entity")
    position: int = Field(description="Position in squad (1-15)")
    is_captain: bool = Field(default=False, description="Is captain")
    is_vice_captain: bool = Field(default=False, description="Is vice-captain")
    multiplier: int = Field(default=1, description="Chip multiplier (1, 2 for TC, 3 for BB bench)")

    @field_validator("position")
    @classmethod
    def _validate_position(cls, v: int) -> int:
        if not 1 <= v <= 15:
            exc = ValueError("Squad position must be between 1 and 15")
            raise exc
        return v

    @field_validator("multiplier")
    @classmethod
    def _validate_multiplier(cls, v: int) -> int:
        if v not in (1, 2, 3):
            exc = ValueError("Multiplier must be 1, 2, or 3")
            raise exc
        return v


class Squad(BaseModel):
    """FPL squad with 15 players."""

    players: list[SquadPlayer] = Field(default_factory=list, description="15 squad players")
    bank: float = Field(default=0.0, description="Remaining budget in millions")
    free_transfers: int = Field(default=1, description="Available free transfers")
    gameweek: int = Field(description="Current gameweek")
    total_value: float = Field(default=0.0, description="Total squad value in millions")

    # Chips
    wildcard_available: bool = Field(default=True, description="Wildcard available")
    free_hit_available: bool = Field(default=True, description="Free Hit available")
    bench_boost_available: bool = Field(default=True, description="Bench Boost available")
    triple_captain_available: bool = Field(default=True, description="Triple Captain available")

    # Used chips this season
    wildcard_used_gw: int | None = Field(default=None, description="GW wildcard used")
    free_hit_used_gw: int | None = Field(default=None, description="GW free hit used")
    bench_boost_used_gw: int | None = Field(default=None, description="GW bench boost used")
    triple_captain_used_gw: int | None = Field(default=None, description="GW triple captain used")

    @field_validator("players")
    @classmethod
    def _validate_squad_size(cls, v: list[SquadPlayer]) -> list[SquadPlayer]:
        if len(v) > 15:
            exc = ValueError("Squad cannot have more than 15 players")
            raise exc
        return v

    def get_starting_xi(self) -> list[SquadPlayer]:
        """Get starting XI based on position (1-11)."""
        return [p for p in self.players if 1 <= p.position <= 11]

    def get_bench(self) -> list[SquadPlayer]:
        """Get bench players (positions 12-15)."""
        return [p for p in self.players if 12 <= p.position <= 15]

    def get_captain(self) -> SquadPlayer | None:
        """Get captain."""
        for p in self.players:
            if p.is_captain:
                return p
        return None

    def get_vice_captain(self) -> SquadPlayer | None:
        """Get vice-captain."""
        for p in self.players:
            if p.is_vice_captain:
                return p
        return None

    def get_by_position(self, position: PlayerPosition) -> list[SquadPlayer]:
        """Get all squad players of a given position."""
        return [sp for sp in self.players if sp.player.position == position]

    def get_starting_by_position(self, position: PlayerPosition) -> list[SquadPlayer]:
        """Get starting XI players of a given position."""
        return [sp for sp in self.get_starting_xi() if sp.player.position == position]

    def count_position(self, position: PlayerPosition, starting_only: bool = False) -> int:
        """Count players of a given position."""
        players = self.get_starting_by_position(position) if starting_only else self.get_by_position(position)
        return len(players)

    def validate_formation(self) -> tuple[bool, list[str]]:
        """Validate squad formation constraints.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Count starting positions
        gk = self.count_position(PlayerPosition.GOALKEEPER, starting_only=True)
        def_ = self.count_position(PlayerPosition.DEFENDER, starting_only=True)
        mid = self.count_position(PlayerPosition.MIDFIELDER, starting_only=True)
        fwd = self.count_position(PlayerPosition.FORWARD, starting_only=True)

        if gk != 1:
            errors.append(f"Must have exactly 1 GK in starting XI, got {gk}")
        if def_ < 3 or def_ > 5:
            errors.append(f"Must have 3-5 DEF in starting XI, got {def_}")
        if mid < 3 or mid > 5:
            errors.append(f"Must have 3-5 MID in starting XI, got {mid}")
        if fwd < 1 or fwd > 3:
            errors.append(f"Must have 1-3 FWD in starting XI, got {fwd}")
        if gk + def_ + mid + fwd != 11:
            errors.append(f"Starting XI must have 11 players, got {gk + def_ + mid + fwd}")

        # Check squad totals
        total_gk = self.count_position(PlayerPosition.GOALKEEPER)
        total_def = self.count_position(PlayerPosition.DEFENDER)
        total_mid = self.count_position(PlayerPosition.MIDFIELDER)
        total_fwd = self.count_position(PlayerPosition.FORWARD)

        if total_gk != 2:
            errors.append(f"Squad must have 2 GK, got {total_gk}")
        if total_def != 5:
            errors.append(f"Squad must have 5 DEF, got {total_def}")
        if total_mid != 5:
            errors.append(f"Squad must have 5 MID, got {total_mid}")
        if total_fwd != 3:
            errors.append(f"Squad must have 3 FWD, got {total_fwd}")

        return len(errors) == 0, errors

    def validate_club_constraints(self) -> tuple[bool, list[str]]:
        """Validate 3-player per club limit.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        club_counts: dict[int, int] = {}

        for sp in self.players:
            club_counts[sp.player.team_id] = club_counts.get(sp.player.team_id, 0) + 1

        for club_id, count in club_counts.items():
            if count > 3:
                errors.append(f"Club {club_id} has {count} players (max 3)")

        return len(errors) == 0, errors

    def get_formation_string(self) -> str:
        """Get formation string (e.g., '3-4-3')."""
        def_ = self.count_position(PlayerPosition.DEFENDER, starting_only=True)
        mid = self.count_position(PlayerPosition.MIDFIELDER, starting_only=True)
        fwd = self.count_position(PlayerPosition.FORWARD, starting_only=True)
        return f"{def_}-{mid}-{fwd}"

    @property
    def starting_xi_count(self) -> int:
        return len(self.get_starting_xi())

    @property
    def bench_count(self) -> int:
        return len(self.get_bench())
