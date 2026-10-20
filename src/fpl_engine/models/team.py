"""Team model."""


from pydantic import BaseModel, Field


class Team(BaseModel):
    """Canonical team entity."""

    id: int = Field(description="Unique FPL team ID")
    code: int = Field(description="Team code (for fixtures)")
    name: str = Field(description="Full team name (e.g., 'Manchester City')")
    short_name: str = Field(description="Short name (e.g., 'MCI')")
    strength: int = Field(default=0, description="Overall team strength")
    strength_overall_home: int = Field(default=0, description="Overall home strength")
    strength_overall_away: int = Field(default=0, description="Overall away strength")
    strength_attack_home: int = Field(default=0, description="Attack home strength")
    strength_attack_away: int = Field(default=0, description="Attack away strength")
    strength_defence_home: int = Field(default=0, description="Defence home strength")
    strength_defence_away: int = Field(default=0, description="Defence away strength")

    # Season stats
    played: int = Field(default=0, description="Games played")
    win: int = Field(default=0, description="Wins")
    loss: int = Field(default=0, description="Losses")
    draw: int = Field(default=0, description="Draws")
    points: int = Field(default=0, description="League points")
    position: int = Field(default=0, description="League position")

    # Goals
    goals_for: int = Field(default=0, description="Goals scored")
    goals_against: int = Field(default=0, description="Goals conceded")
    form: str | None = Field(default=None, description="Recent form string (e.g., 'WDLWW')")

    # Metadata
    pulse_id: int | None = Field(default=None, description="External pulse ID")

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Team):
            return NotImplemented
        return self.id == other.id
