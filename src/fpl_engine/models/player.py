"""Player model and related enums."""

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class PlayerPosition(StrEnum):
    """FPL player positions."""

    GOALKEEPER = "GKP"
    DEFENDER = "DEF"
    MIDFIELDER = "MID"
    FORWARD = "FWD"

    @property
    def position_id(self) -> int:
        """Return the FPL position ID (1-4)."""
        mapping = {
            PlayerPosition.GOALKEEPER: 1,
            PlayerPosition.DEFENDER: 2,
            PlayerPosition.MIDFIELDER: 3,
            PlayerPosition.FORWARD: 4,
        }
        return mapping[self]

    @classmethod
    def from_id(cls, position_id: int) -> "PlayerPosition":
        """Create PlayerPosition from FPL position ID."""
        mapping = {
            1: cls.GOALKEEPER,
            2: cls.DEFENDER,
            3: cls.MIDFIELDER,
            4: cls.FORWARD,
        }
        if position_id not in mapping:
            exc = ValueError(f"Invalid position ID: {position_id}")
            raise exc
        return mapping[position_id]


class PlayerStatus(StrEnum):
    """Player availability status."""

    AVAILABLE = "a"
    INJURED = "i"
    SUSPENDED = "s"
    UNAVAILABLE = "u"
    DOUBTFUL = "d"
    NOT_IN_SQUAD = "n"


class Player(BaseModel):
    """Canonical player entity."""

    id: int = Field(description="Unique FPL player ID")
    web_name: str = Field(description="Player's web name (e.g., 'Salah')")
    first_name: str = Field(description="Player's first name")
    second_name: str = Field(description="Player's last name")
    team_id: int = Field(description="FPL team ID")
    position: PlayerPosition = Field(description="Player position")
    status: PlayerStatus = Field(default=PlayerStatus.AVAILABLE, description="Availability status")

    # Pricing
    now_cost: int = Field(description="Current price in tenths of millions (e.g., 130 = £13.0m)")
    cost_change_start: int = Field(default=0, description="Price change since season start")
    cost_change_event: int = Field(default=0, description="Price change since last GW")
    cost_change_event_fall: int = Field(default=0, description="Price fall since last GW")
    cost_change_start_fall: int = Field(default=0, description="Price fall since season start")

    # Ownership
    selected_by_percent: float = Field(default=0.0, description="Percentage of managers owning player")
    transfers_in: int = Field(default=0, description="Transfers in this GW")
    transfers_out: int = Field(default=0, description="Transfers out this GW")
    transfers_in_event: int = Field(default=0, description="Transfers in this event")
    transfers_out_event: int = Field(default=0, description="Transfers out this event")

    # Form and points
    form: float = Field(default=0.0, description="Form (last few GWs average points)")
    total_points: int = Field(default=0, description="Total season points")
    points_per_game: float = Field(default=0.0, description="Points per game average")
    event_points: int = Field(default=0, description="Points in current GW")

    # Expected stats (from FPL)
    expected_goals: float | None = Field(default=None, description="Season xG")
    expected_assists: float | None = Field(default=None, description="Season xA")
    expected_goal_involvements: float | None = Field(default=None, description="Season xGI")
    expected_goals_conceded: float | None = Field(default=None, description="Season xGC")
    expected_goals_per_90: float | None = Field(default=None, description="xG per 90")
    expected_assists_per_90: float | None = Field(default=None, description="xA per 90")
    expected_goal_involvements_per_90: float | None = Field(default=None, description="xGI per 90")
    expected_goals_conceded_per_90: float | None = Field(default=None, description="xGC per 90")

    # Minutes and appearances
    minutes: int = Field(default=0, description="Total minutes played")
    starts: int = Field(default=0, description="Games started")
    expected_minutes: int | None = Field(default=None, description="Expected minutes for next GW")

    # Attacking
    goals_scored: int = Field(default=0, description="Goals scored")
    assists: int = Field(default=0, description="Assists")
    penalties_missed: int = Field(default=0, description="Penalties missed")
    penalties_saved: int = Field(default=0, description="Penalties saved (GK)")
    penalties_order: int | None = Field(default=None, description="Penalty taker order")
    direct_freekicks_order: int | None = Field(default=None, description="Free-kick taker order")
    corners_order: int | None = Field(default=None, description="Corner taker order")

    # Defensive
    clean_sheets: int = Field(default=0, description="Clean sheets")
    goals_conceded: int = Field(default=0, description="Goals conceded (GK/DEF)")
    own_goals: int = Field(default=0, description="Own goals")
    saves: int = Field(default=0, description="Saves (GK)")
    bonus: int = Field(default=0, description="Bonus points")
    bps: int = Field(default=0, description="Bonus points system score")

    # Advanced defensive
    tackles: int = Field(default=0, description="Tackles")
    interceptions: int = Field(default=0, description="Interceptions")
    clearances: int = Field(default=0, description="Clearances")
    blocks: int = Field(default=0, description="Blocks")

    # Influence metrics
    influence: float | None = Field(default=None, description="Influence score")
    creativity: float | None = Field(default=None, description="Creativity score")
    threat: float | None = Field(default=None, description="Threat score")
    ict_index: float | None = Field(default=None, description="ICT index")

    # Value
    value_form: float | None = Field(default=None, description="Value form (points per million)")
    value_season: float | None = Field(default=None, description="Value season")

    # Chips
    in_dreamteam: bool = Field(default=False, description="In dreamteam this GW")
    dreamteam_count: int = Field(default=0, description="Times in dreamteam")

    # Metadata
    code: int = Field(description="Player code for photos")
    photo: str = Field(default="", description="Photo filename")
    element_type: int = Field(description="Position type (1-4)")
    team_code: int = Field(description="Team code")
    chance_of_playing_this_round: int | None = Field(default=None, description="Chance of playing %")
    chance_of_playing_next_round: int | None = Field(default=None, description="Chance of playing next GW %")

    @field_validator("position", mode="before")
    @classmethod
    def _validate_position(cls, v: int | str | PlayerPosition) -> PlayerPosition:
        if isinstance(v, PlayerPosition):
            return v
        if isinstance(v, int):
            return PlayerPosition.from_id(v)
        if isinstance(v, str):
            try:
                return PlayerPosition(v.upper())
            except ValueError:
                return PlayerPosition.from_id(int(v))
        exc = TypeError(f"Invalid position type: {type(v)}")
        raise exc

    @property
    def full_name(self) -> str:
        """Return player's full name."""
        return f"{self.first_name} {self.second_name}"

    @property
    def price(self) -> float:
        """Return price in millions."""
        return self.now_cost / 10.0

    @property
    def is_available(self) -> bool:
        """Check if player is available for selection."""
        return self.status == PlayerStatus.AVAILABLE

    @property
    def is_goalkeeper(self) -> bool:
        return self.position == PlayerPosition.GOALKEEPER

    @property
    def is_defender(self) -> bool:
        return self.position == PlayerPosition.DEFENDER

    @property
    def is_midfielder(self) -> bool:
        return self.position == PlayerPosition.MIDFIELDER

    @property
    def is_forward(self) -> bool:
        return self.position == PlayerPosition.FORWARD
