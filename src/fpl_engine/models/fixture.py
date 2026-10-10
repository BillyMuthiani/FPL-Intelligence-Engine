"""Fixture model and related enums."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class FixtureDifficulty(StrEnum):
    """Fixture difficulty rating."""

    VERY_EASY = "very_easy"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"

    @classmethod
    def from_fdr(cls, fdr: int) -> "FixtureDifficulty":
        """Create difficulty from FPL FDR (1-5)."""
        mapping = {
            1: cls.VERY_EASY,
            2: cls.EASY,
            3: cls.MEDIUM,
            4: cls.HARD,
            5: cls.VERY_HARD,
        }
        return mapping.get(fdr, cls.MEDIUM)

    @property
    def numeric(self) -> int:
        """Return numeric value 1-5."""
        mapping = {
            self.VERY_EASY: 1,
            self.EASY: 2,
            self.MEDIUM: 3,
            self.HARD: 4,
            self.VERY_HARD: 5,
        }
        return mapping[self]


class Fixture(BaseModel):
    """Canonical fixture entity."""

    id: int = Field(description="Unique fixture ID")
    code: int = Field(description="Fixture code")
    event: int = Field(description="Gameweek number")
    team_h: int = Field(description="Home team ID")
    team_a: int = Field(description="Away team ID")
    team_h_score: int | None = Field(default=None, description="Home team score")
    team_a_score: int | None = Field(default=None, description="Away team score")
    finished: bool = Field(default=False, description="Whether fixture is finished")
    started: bool = Field(default=False, description="Whether fixture has started")
    minutes: int = Field(default=0, description="Minutes played")
    provisional_start_time: str | None = Field(default=None, description="Provisional start time (ISO)")
    kickoff_time: str | None = Field(default=None, description="Confirmed kickoff time (ISO)")
    event_name: str | None = Field(default=None, description="Event name")

    # Difficulty ratings (1-5)
    team_h_difficulty: int = Field(default=3, description="Home team difficulty")
    team_a_difficulty: int = Field(default=3, description="Away team difficulty")

    # Stats (populated after fixture)
    stats: list[dict[str, Any]] = Field(default_factory=list, description="Fixture statistics")

    # Team info (denormalized for convenience)
    team_h_name: str | None = Field(default=None, description="Home team short name")
    team_a_name: str | None = Field(default=None, description="Away team short name")

    def get_difficulty(self, is_home: bool) -> FixtureDifficulty:
        """Get difficulty for a specific team."""
        if is_home:
            return FixtureDifficulty.from_fdr(self.team_h_difficulty)
        return FixtureDifficulty.from_fdr(self.team_a_difficulty)

    def get_opponent_id(self, team_id: int) -> int | None:
        """Get opponent team ID for a given team."""
        if team_id == self.team_h:
            return self.team_a
        if team_id == self.team_a:
            return self.team_h
        return None

    def is_team_home(self, team_id: int) -> bool:
        """Check if team is playing at home."""
        return team_id == self.team_h

    def is_team_away(self, team_id: int) -> bool:
        """Check if team is playing away."""
        return team_id == self.team_a

    @property
    def is_finished(self) -> bool:
        return self.finished

    @property
    def result(self) -> str | None:
        """Return result string if finished."""
        if not self.finished or self.team_h_score is None or self.team_a_score is None:
            return None
        return f"{self.team_h_score}-{self.team_a_score}"
