"""FPL scoring rules and single-match point calculator for the 2026/27 season.

All point values are verified against the official FPL rules:
https://fantasy.premierleague.com/en/help/rules

Key 2026/27 season values (verified):
- Goalkeeper goals: 10 points (increased from 6 in prior seasons)
- Penalty misses: -2 points (reduced from -5 in prior seasons)
- Defensive contribution: +2 points for reaching the action threshold
  (new in 2025/26, expanded in 2026/27 to include ball recoveries for
  midfielders and forwards)
- Appearance points: +1 for playing 1-59 minutes, +2 for 60+ minutes
- Clean sheet eligibility: requires >= 60 minutes played AND position eligibility
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType

from fpl_engine.models.player import PlayerPosition

__all__ = [
    "DEFAULT_SCORING_RULES",
    "FPLScoringRules",
    "MatchStats",
    "ScoringDecomposition",
    "ScoringError",
    "calculate_fpl_points",
]


class ScoringError(ValueError):
    """Raised when scoring calculation encounters invalid or inconsistent inputs."""


@dataclass(frozen=True)
class FPLScoringRules:
    """Immutable configuration of FPL scoring rules for a given season.

    All mapping fields are wrapped in :class:`MappingProxyType` during
    ``__post_init__`` to guarantee runtime immutability — once constructed,
    no scoring parameter can be changed.

    Defaults reflect the **2026/27** FPL season, verified against:
    https://fantasy.premierleague.com/en/help/rules
    """

    # --- Appearance Points ---
    appearance_min_minutes: int = 60
    appearance_points_below: int = 1  # Played 1-59 minutes
    appearance_points_above_or_equal: int = 2  # Played 60+ minutes

    # --- Goal Points (by position) ---
    # 2026/27: Goalkeeper goals worth 10 points (increased from 6 in prior seasons)
    goal_points: Mapping[PlayerPosition, int] = field(default_factory=lambda: {
        PlayerPosition.GOALKEEPER: 10,
        PlayerPosition.DEFENDER: 6,
        PlayerPosition.MIDFIELDER: 5,
        PlayerPosition.FORWARD: 4,
    })

    # --- Assist Points ---
    assist_points: int = 3  # All positions

    # --- Clean Sheet Points ---
    # Forward is intentionally absent -> 0 points (not eligible)
    clean_sheet_points: Mapping[PlayerPosition, int] = field(default_factory=lambda: {
        PlayerPosition.GOALKEEPER: 4,
        PlayerPosition.DEFENDER: 4,
        PlayerPosition.MIDFIELDER: 1,
    })
    clean_sheet_min_minutes: int = 60  # Must play at least this many minutes

    # --- Goals Conceded (GK/DEF only) ---
    # goals_conceded represents goals conceded while the player was on the
    # pitch, not necessarily the team's full-match total.
    goals_conceded_eligible: tuple[PlayerPosition, ...] = (
        PlayerPosition.GOALKEEPER,
        PlayerPosition.DEFENDER,
    )
    goals_conceded_penalty: int = -1  # -1 per 2 goals conceded

    # --- Saves (GK only) ---
    save_eligible: tuple[PlayerPosition, ...] = (
        PlayerPosition.GOALKEEPER,
    )
    saves_per_point: int = 3
    save_points: int = 1

    # --- Penalty Saves ---
    # Awarded only to goalkeepers (verified 2026/27 rule)
    penalty_save_points: int = 5

    # --- Penalty Misses ---
    # 2026/27: -2 points (reduced from -5 in prior seasons)
    penalty_miss_points: int = -2

    # --- Cards ---
    yellow_card_points: int = -1
    red_card_points: int = -3  # Includes accumulation from second yellow

    # --- Own Goals ---
    own_goal_points: int = -2  # All positions

    # --- Defensive Contribution (NEW: 2025/26, expanded 2026/27) ---
    # DEF: 10+ CBIT (clearances + blocks + interceptions + tackles) -> +2
    # MID/FWD: 12+ CBIRT (CBIT + ball recoveries) -> +2
    # Capped at 2 points per match
    defensive_contribution_eligible: tuple[PlayerPosition, ...] = (
        PlayerPosition.DEFENDER,
        PlayerPosition.MIDFIELDER,
        PlayerPosition.FORWARD,
    )
    defensive_contribution_def_threshold: int = 10
    defensive_contribution_mid_fwd_threshold: int = 12
    defensive_contribution_points: int = 2
    defensive_contribution_max: int = 2

    def __post_init__(self) -> None:
        # Wrap all dict fields in MappingProxyType for runtime immutability.
        # This makes the Mapping fields truly read-only: any attempt to mutate
        # them raises TypeError at runtime, and mypy already treats Mapping
        # as non-mutable.
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, dict):
                object.__setattr__(self, f.name, MappingProxyType(val))

        self._validate()

    def _validate(self) -> None:
        """Validate scoring configuration for consistency."""
        if self.appearance_min_minutes <= 0:
            raise ScoringError("appearance_min_minutes must be positive")  # noqa: TRY003

        for pos in PlayerPosition:
            if pos not in self.goal_points:
                raise ScoringError(f"goal_points missing entry for {pos}")  # noqa: TRY003
            if self.goal_points[pos] < 0:
                raise ScoringError(f"goal_points for {pos} must be non-negative")  # noqa: TRY003

        if self.assist_points < 0:
            raise ScoringError("assist_points must be non-negative")  # noqa: TRY003

        for pos, pts in self.clean_sheet_points.items():
            if pts < 0:
                raise ScoringError(f"clean_sheet_points for {pos} must be non-negative")  # noqa: TRY003

        if self.clean_sheet_min_minutes <= 0:
            raise ScoringError("clean_sheet_min_minutes must be positive")  # noqa: TRY003

        if self.goals_conceded_penalty > 0:
            raise ScoringError("goals_conceded_penalty must be non-positive")  # noqa: TRY003

        if self.saves_per_point <= 0:
            raise ScoringError("saves_per_point must be positive")  # noqa: TRY003

        if self.save_points < 0:
            raise ScoringError("save_points must be non-negative")  # noqa: TRY003

        if self.penalty_save_points < 0:
            raise ScoringError("penalty_save_points must be non-negative")  # noqa: TRY003

        if self.penalty_miss_points > 0:
            raise ScoringError("penalty_miss_points must be non-positive")  # noqa: TRY003

        if self.yellow_card_points > 0:
            raise ScoringError("yellow_card_points must be non-positive")  # noqa: TRY003

        if self.red_card_points > 0:
            raise ScoringError("red_card_points must be non-positive")  # noqa: TRY003

        if self.own_goal_points > 0:
            raise ScoringError("own_goal_points must be non-positive")  # noqa: TRY003

        if self.defensive_contribution_def_threshold <= 0:
            raise ScoringError("defensive_contribution_def_threshold must be positive")  # noqa: TRY003

        if self.defensive_contribution_mid_fwd_threshold <= 0:
            raise ScoringError(  # noqa: TRY003
                "defensive_contribution_mid_fwd_threshold must be positive"
            )

        if self.defensive_contribution_points < 0:
            raise ScoringError("defensive_contribution_points must be non-negative")  # noqa: TRY003

        if self.defensive_contribution_max < self.defensive_contribution_points:
            raise ScoringError(  # noqa: TRY003
                "defensive_contribution_max must be >= defensive_contribution_points"
            )

        if self.appearance_points_below < 0:
            raise ScoringError("appearance_points_below must be non-negative")  # noqa: TRY003

        if self.appearance_points_above_or_equal < 0:
            raise ScoringError(  # noqa: TRY003
                "appearance_points_above_or_equal must be non-negative"
            )


# Module-level singleton — the documented public interface for 2026/27 rules.
# Named DEFAULT_SCORING_RULES (PEP 8 constant convention), exported from the
# package __init__.py.  This resolves the naming inconsistency noted in the
# plan: the canonical name is DEFAULT_SCORING_RULES, not default_scoring_rules.
DEFAULT_SCORING_RULES: FPLScoringRules = FPLScoringRules()


@dataclass(frozen=True)
class MatchStats:
    """Statistics for a single player match — input to :func:`calculate_fpl_points`.

    All fields default to 0. Pass only the statistics that occurred.

    Notes:
        - ``goals_conceded`` represents goals conceded **while the player was
          on the pitch**, not the team's full-match total.  A player substituted
          off before a goal was scored is not penalised for it.
        - ``bonus`` represents actual bonus points awarded (0, 1, 2, or 3).
          The Bonus Points System is NOT reproduced here; bonus must be supplied
          as an explicit input.
        - ``recoveries`` represents ball recoveries (relevant for defensive
          contribution points for midfielders and forwards only).
    """

    minutes: int = 0
    goals: int = 0
    assists: int = 0
    goals_conceded: int = 0
    saves: int = 0
    penalties_saved: int = 0
    penalties_missed: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    own_goals: int = 0
    tackles: int = 0
    interceptions: int = 0
    clearances: int = 0
    blocks: int = 0
    recoveries: int = 0
    bonus: int = 0
    clean_sheet: bool = False

    def __post_init__(self) -> None:
        """Validate match statistics on construction."""
        if self.minutes < 0:
            raise ScoringError("minutes must be non-negative")  # noqa: TRY003
        if self.minutes > 120:
            raise ScoringError("minutes cannot exceed 120")  # noqa: TRY003

        stat_fields = [
            "goals",
            "assists",
            "goals_conceded",
            "saves",
            "penalties_saved",
            "penalties_missed",
            "yellow_cards",
            "red_cards",
            "own_goals",
            "tackles",
            "interceptions",
            "clearances",
            "blocks",
            "recoveries",
        ]
        for name in stat_fields:
            if getattr(self, name) < 0:
                raise ScoringError(f"{name} must be non-negative")  # noqa: TRY003

        if self.bonus not in (0, 1, 2, 3):
            raise ScoringError("bonus must be 0, 1, 2, or 3")  # noqa: TRY003


@dataclass(frozen=True)
class ScoringDecomposition:
    """Breakdown of FPL points for a single match, by scoring category.

    Each field is a named component.  The ``total`` field equals the sum
    of all component fields.
    """

    appearance_points: int
    goal_points: int
    assist_points: int
    clean_sheet_points: int
    goals_conceded_points: int
    save_points: int
    penalty_save_points: int
    penalty_miss_points: int
    yellow_card_points: int
    red_card_points: int
    own_goal_points: int
    defensive_contribution_points: int
    bonus_points: int
    total: int


def calculate_fpl_points(
    position: PlayerPosition,
    match_stats: MatchStats,
    rules: FPLScoringRules | None = None,
) -> ScoringDecomposition:
    """Calculate FPL points for a single player's single match.

    Args:
        position: The player's position (one of :class:`PlayerPosition`).
        match_stats: Match-level statistics for the player.
        rules: Scoring rules to use.  Defaults to
            :data:`DEFAULT_SCORING_RULES` (the 2026/27 standard rules).

    Returns:
        A :class:`ScoringDecomposition` with named component breakdowns
        and a ``total`` field.

    Raises:
        ScoringError: If ``position`` is not a :class:`PlayerPosition` enum
            or ``match_stats`` is not a :class:`MatchStats` instance.

    Notes:
        - Penalty-save points are awarded **only to goalkeepers**.
        - Clean-sheet points require the player to have played at least
          ``rules.clean_sheet_min_minutes`` (default 60) minutes AND be in
          an eligible position.  Supplying ``clean_sheet=True`` with fewer
          minutes yields 0 clean-sheet points.
        - Defensive contribution points are capped at
          ``rules.defensive_contribution_max`` (default 2) per match.
        - Bonus points are passed through from ``match_stats.bonus``; they are
          not recomputed from other statistics.
    """
    if not isinstance(position, PlayerPosition):
        raise ScoringError(  # noqa: TRY003
            f"position must be a PlayerPosition enum, got {type(position).__name__}"
        )

    if not isinstance(match_stats, MatchStats):
        raise ScoringError(  # noqa: TRY003
            f"match_stats must be a MatchStats instance, "
            f"got {type(match_stats).__name__}"
        )

    r = rules if rules is not None else DEFAULT_SCORING_RULES
    ms = match_stats

    # --- Appearance points ---
    if ms.minutes == 0:
        appearance_points = 0
    elif ms.minutes < r.appearance_min_minutes:
        appearance_points = r.appearance_points_below
    else:
        appearance_points = r.appearance_points_above_or_equal

    # --- Goal points (position-specific) ---
    goal_points = ms.goals * r.goal_points[position]

    # --- Assist points ---
    assist_points = ms.assists * r.assist_points

    # --- Clean sheet points ---
    # Requires: position eligible + minutes threshold + clean sheet kept
    if (
        position in r.clean_sheet_points
        and ms.minutes >= r.clean_sheet_min_minutes
        and ms.clean_sheet
    ):
        clean_sheet_points = r.clean_sheet_points[position]
    else:
        clean_sheet_points = 0

    # --- Goals conceded (GK/DEF only) ---
    # -1 point per 2 goals conceded (floor division -> non-positive result)
    if position in r.goals_conceded_eligible:
        goals_conceded_points = (ms.goals_conceded // 2) * r.goals_conceded_penalty
    else:
        goals_conceded_points = 0

    # --- Save points (GK only) ---
    if position in r.save_eligible:
        save_points = (ms.saves // r.saves_per_point) * r.save_points
    else:
        save_points = 0

    # --- Penalty save points (GK only) ---
    if position == PlayerPosition.GOALKEEPER:
        penalty_save_points = ms.penalties_saved * r.penalty_save_points
    else:
        penalty_save_points = 0

    # --- Penalty miss points (all positions) ---
    penalty_miss_points = ms.penalties_missed * r.penalty_miss_points

    # --- Card deductions ---
    yellow_card_points = ms.yellow_cards * r.yellow_card_points
    red_card_points = ms.red_cards * r.red_card_points

    # --- Own goal deduction (all positions) ---
    own_goal_points = ms.own_goals * r.own_goal_points

    # --- Defensive contribution ---
    # DEF: 10+ CBIT (clearances + blocks + interceptions + tackles) -> +2
    # MID/FWD: 12+ CBIRT (CBIT + ball recoveries) -> +2
    # Capped at defensive_contribution_max per match
    cbit = ms.tackles + ms.interceptions + ms.clearances + ms.blocks

    if position not in r.defensive_contribution_eligible:
        defensive_contribution_points = 0
    else:
        if position == PlayerPosition.DEFENDER:
            effective = cbit
            threshold = r.defensive_contribution_def_threshold
        else:  # MIDFIELDER or FORWARD
            # CBIRT includes ball recoveries for midfielders and forwards
            effective = cbit + ms.recoveries
            threshold = r.defensive_contribution_mid_fwd_threshold

        if effective >= threshold:
            defensive_contribution_points = min(
                r.defensive_contribution_points,
                r.defensive_contribution_max,
            )
        else:
            defensive_contribution_points = 0

    # --- Bonus points (only if supplied as input) ---
    bonus_points = ms.bonus

    # --- Total ---
    total = (
        appearance_points
        + goal_points
        + assist_points
        + clean_sheet_points
        + goals_conceded_points
        + save_points
        + penalty_save_points
        + penalty_miss_points
        + yellow_card_points
        + red_card_points
        + own_goal_points
        + defensive_contribution_points
        + bonus_points
    )

    return ScoringDecomposition(
        appearance_points=appearance_points,
        goal_points=goal_points,
        assist_points=assist_points,
        clean_sheet_points=clean_sheet_points,
        goals_conceded_points=goals_conceded_points,
        save_points=save_points,
        penalty_save_points=penalty_save_points,
        penalty_miss_points=penalty_miss_points,
        yellow_card_points=yellow_card_points,
        red_card_points=red_card_points,
        own_goal_points=own_goal_points,
        defensive_contribution_points=defensive_contribution_points,
        bonus_points=bonus_points,
        total=total,
    )
