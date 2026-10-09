"""Gameweek model."""

from typing import Any

from pydantic import BaseModel, Field


class Gameweek(BaseModel):
    """Canonical gameweek entity."""

    id: int = Field(description="Gameweek number (1-38)")
    name: str = Field(description="Gameweek name (e.g., 'Gameweek 1')")
    deadline_time: str = Field(description="Deadline time (ISO format)")
    deadline_time_epoch: int = Field(description="Deadline as Unix timestamp")
    deadline_time_game_offset: float = Field(description="Hours offset from game start")
    finished: bool = Field(default=False, description="Whether GW is finished")
    data_checked: bool = Field(default=False, description="Whether data has been checked")
    is_previous: bool = Field(default=False, description="Is previous GW")
    is_current: bool = Field(default=False, description="Is current GW")
    is_next: bool = Field(default=False, description="Is next GW")
    cup_leagues_created: bool = Field(default=False, description="Cup leagues created")
    h2h_leagues_created: bool = Field(default=False, description="H2H leagues created")

    # Chip status
    top_element_info: dict[str, Any] | None = Field(default=None, description="Top element info")
    chip_plays: list[dict[str, Any]] = Field(default_factory=list, description="Chip plays this GW")

    # Transfers
    transfers_made: int = Field(default=0, description="Total transfers made this GW")
    most_captained: int | None = Field(default=None, description="Most captained player ID")
    most_vice_captained: int | None = Field(default=None, description="Most VC player ID")

    # Averages
    average_entry_score: int = Field(default=0, description="Average entry score")
    highest_score: int = Field(default=0, description="Highest score this GW")
    highest_scoring_entry: int | None = Field(default=None, description="Highest scoring entry ID")

    @property
    def is_active(self) -> bool:
        """Check if this is the active gameweek."""
        return self.is_current or self.is_next

    def __lt__(self, other: "Gameweek") -> bool:
        return self.id < other.id
