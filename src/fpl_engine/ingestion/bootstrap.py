"""Bootstrap-static normalization.

Map raw ``bootstrap-static`` documents from the official FPL API into validated
canonical models. The field mapping intentionally mirrors
``FPLClient.get_players`` / ``get_teams`` / ``get_gameweeks`` so that normalized
output is byte-for-byte equivalent to what the existing provider yields on
well-formed input, while adding explicit validation, deduplication and a
structured :class:`ValidationReport`.

Missing vs. zero semantics:
  * Nullable model fields (``float | None`` / ``int | None``) carry ``None``
    when the source key is absent, distinguishing "unknown" from a genuine ``0``.
  * Non-nullable fields with a model default rely on that default when the key
    is absent -- the field's declared default is the single source of truth, so
    this module does not invent new imputations. Required fields, if absent,
    cause the record to be rejected rather than coerced.

Conflict and deduplication semantics:
  * Exact-duplicate records (same id, identical content) are counted as
    ``duplicates_skipped`` and produce a single output record.
  * Conflicting records (same id, different content) cause the identifier to
    be excluded from the normalized output entirely -- neither the first nor the
    second occurrence is returned. Each conflict is counted in
    ``conflicts`` with a matching ``Conflict`` detail entry. The identifier is
    added to a ``conflicted_ids`` set so that later records with the same id
    are also counted as conflicts and can never restore the identifier to the
    output. In other words, once conflicted, always conflicted.
  * ``conflicts`` counts conflicting **records**, not conflicted **identifiers**;
    one identifier may contribute multiple conflict records. The number of
    distinct conflicted identifiers can be recovered as
    ``len({c.identifier for c in conflicts_detail})``.
  * When all records for a section are rejected or conflicted (i.e.
    ``accepted == 0`` with a non-empty total), :func:`_finalize` raises
    :class:`IngestionError` because the section is structurally unusable.

Empty sections:
  * A section that is present but empty (``events: []``) yields an empty
    list and a report with ``total == 0``; no :class:`IngestionError` is raised.
  * Whether an empty section makes a *complete* bootstrap dataset unusable is
    a decision for a higher-level ingestion coordinator, not for these
    per-section normalizers.

Provenance:
  * No timestamps, URLs, hashes or other provenance values are fabricated.
    Each normalizer operates solely on the fields present in the input dict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ..models import Gameweek, Player, PlayerPosition, PlayerStatus, Team
from .exceptions import IngestionError

__all__ = [
    "Conflict",
    "Rejection",
    "ValidationReport",
    "normalize_gameweeks",
    "normalize_players",
    "normalize_teams",
]


# Required source keys: those indexed positionally by the existing provider.
_REQUIRED_PLAYER_FIELDS: tuple[str, ...] = (
    "id",
    "web_name",
    "first_name",
    "second_name",
    "team",
    "element_type",
    "now_cost",
    "code",
)
_REQUIRED_TEAM_FIELDS: tuple[str, ...] = ("id", "code", "name", "short_name")
_REQUIRED_EVENT_FIELDS: tuple[str, ...] = (
    "id",
    "name",
    "deadline_time",
    "deadline_time_epoch",
    "deadline_time_game_offset",
)


def _to_float(value: Any) -> float | None:
    """Mirror ``FPLClient._safe_float``: None/'' -> None, else ``float(value)``."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Rejection:
    """A record that could not be normalized."""

    source_id: str
    reason: str


@dataclass(frozen=True)
class Conflict:
    """A duplicate identifier whose content differed from the accepted record."""

    identifier: str
    reason: str


@dataclass
class ValidationReport:
    """Outcome summary of a normalization run.

    Counter semantics:
      * ``accepted`` — records that survived all checks and appear in output.
      * ``rejected`` — records discarded for missing required fields or invalid
        values.
      * ``duplicates_skipped`` — exact duplicates of an already-processed record
        (same id, identical content) that were silently dropped.
      * ``conflicts`` — records that could not be used because their identifier
        conflicted with a previously seen record *or* because the identifier
        has already been marked as conflicted (the record cannot restore it).
        This counter counts conflicting **records**, not conflicted
        **identifiers**; one identifier may contribute multiple conflict
        records.
      * ``seen`` — total records iterated (regardless of outcome).
      * ``total`` — alias for ``seen``; every record observed is counted.

    An identifier that is conflicted is permanently excluded from the normalized
    output: later records with the same id can never restore it.
    """

    accepted: int = 0
    rejected: int = 0
    conflicts: int = 0
    duplicates_skipped: int = 0
    seen: int = 0
    rejections: list[Rejection] = field(default_factory=list)
    conflicts_detail: list[Conflict] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Total number of records observed."""
        return self.seen


def _missing_keys(record: dict[str, Any], required: tuple[str, ...]) -> list[str]:
    """Return the subset of ``required`` keys absent from ``record``."""
    return [k for k in required if k not in record]


def _source_id(label: str, record_id: Any) -> str:
    """Stable identifier for a record used in report entries."""
    return f"{label}:{record_id}" if record_id is not None else f"{label}:unknown"


def _require_section(
    data: dict[str, Any], key: str, label: str
) -> list[dict[str, Any]]:
    """Return ``data[key]`` as a list or raise ``IngestionError`` if malformed."""
    section = data.get(key)
    if section is None:
        err = IngestionError("missing section")
        err.add_note(f"bootstrap payload missing '{label}' section")
        raise err
    if not isinstance(section, list):
        err = IngestionError("invalid section")
        err.add_note(f"bootstrap '{label}' must be a list")
        raise err
    return section


def _index_teams(data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Index the bootstrap ``teams`` array by team id (tolerant of bad entries)."""
    teams_raw = data.get("teams", [])
    if not isinstance(teams_raw, list):
        return {}
    indexed: dict[int, dict[str, Any]] = {}
    for team in teams_raw:
        if not isinstance(team, dict):
            continue
        team_id = team.get("id")
        if isinstance(team_id, bool) or not isinstance(team_id, (int, float)):
            continue
        indexed[int(team_id)] = team
    return indexed


def _finalize(report: ValidationReport, label: str) -> None:
    """Raise ``IngestionError`` if every record was rejected (unusable result)."""
    if report.accepted == 0 and report.total > 0:
        err = IngestionError("empty result")
        err.add_note(
            f"{label}: {report.rejected} rejected, "
            f"{report.conflicts} conflicts, {report.duplicates_skipped} duplicates"
        )
        raise err


def normalize_players(data: dict[str, Any]) -> tuple[list[Player], ValidationReport]:
    """Normalize the ``elements`` section of a bootstrap-static document.

    Raises:
        IngestionError: if ``data`` is not a dict, ``elements`` is absent/invalid,
            or no valid players remain after normalization.
    """
    if not isinstance(data, dict):
        err = IngestionError("payload not a dict")
        err.add_note("normalize_players expects a bootstrap-static document dict")
        raise err
    elements = _require_section(data, "elements", "elements")
    teams = _index_teams(data)

    report = ValidationReport()
    accepted: dict[int, Player] = {}
    seen_raw: dict[int, dict[str, Any]] = {}
    conflicted_ids: set[int] = set()

    for elem in elements:
        report.seen += 1
        if not isinstance(elem, dict):
            report.rejected += 1
            report.rejections.append(
                Rejection(source_id="element:unknown", reason="element is not a dict")
            )
            continue

        elem_id = elem.get("id")
        missing = _missing_keys(elem, _REQUIRED_PLAYER_FIELDS)
        if missing:
            report.rejected += 1
            report.rejections.append(
                Rejection(
                    source_id=_source_id("element", elem_id),
                    reason=f"missing required fields: {', '.join(missing)}",
                )
            )
            continue

        team_id = elem["team"]
        if team_id not in teams:
            report.rejected += 1
            report.rejections.append(
                Rejection(
                    source_id=_source_id("element", elem_id),
                    reason=f"player references team {team_id} not present in teams section",
                )
            )
            continue

        try:
            player = _build_player(elem, teams)
        except (ValidationError, ValueError) as exc:
            report.rejected += 1
            report.rejections.append(
                Rejection(source_id=_source_id("element", elem_id), reason=str(exc))
            )
            continue

        eid = int(elem["id"])
        if eid in conflicted_ids:
            report.conflicts += 1
            report.conflicts_detail.append(
                Conflict(
                    identifier=_source_id("element", eid),
                    reason="conflicting duplicate records",
                )
            )
            continue
        if eid in seen_raw:
            if elem == seen_raw[eid]:
                report.duplicates_skipped += 1
            else:
                if eid in accepted:
                    del accepted[eid]
                    report.accepted -= 1
                conflicted_ids.add(eid)
                report.conflicts += 1
                report.conflicts_detail.append(
                    Conflict(
                        identifier=_source_id("element", eid),
                        reason="conflicting duplicate records",
                    )
                )
            continue

        seen_raw[eid] = elem
        accepted[eid] = player
        report.accepted += 1

    _finalize(report, "players")
    return list(accepted.values()), report


def _build_player(
    elem: dict[str, Any], teams: dict[int, dict[str, Any]]
) -> Player:
    """Construct a :class:`Player` from a raw element dict.

    The team's ``code`` is resolved from the bootstrap ``teams`` array (matching
    the field the existing provider sets). Callers must verify that the team
    reference exists before calling; a missing team raises ``KeyError``.
    """
    team_data = teams[elem["team"]]
    return Player(
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
        expected_goals=_to_float(elem.get("expected_goals")),
        expected_assists=_to_float(elem.get("expected_assists")),
        expected_goal_involvements=_to_float(elem.get("expected_goal_involvements")),
        expected_goals_conceded=_to_float(elem.get("expected_goals_conceded")),
        expected_goals_per_90=_to_float(elem.get("expected_goals_per_90")),
        expected_assists_per_90=_to_float(elem.get("expected_assists_per_90")),
        expected_goal_involvements_per_90=_to_float(
            elem.get("expected_goal_involvements_per_90")
        ),
        expected_goals_conceded_per_90=_to_float(
            elem.get("expected_goals_conceded_per_90")
        ),
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
        influence=_to_float(elem.get("influence")),
        creativity=_to_float(elem.get("creativity")),
        threat=_to_float(elem.get("threat")),
        ict_index=_to_float(elem.get("ict_index")),
        value_form=_to_float(elem.get("value_form")),
        value_season=_to_float(elem.get("value_season")),
        in_dreamteam=elem.get("in_dreamteam", False),
        dreamteam_count=elem.get("dreamteam_count", 0),
        code=elem["code"],
        photo=elem.get("photo", ""),
        element_type=elem["element_type"],
        team_code=team_data["code"],
        chance_of_playing_this_round=elem.get("chance_of_playing_this_round"),
        chance_of_playing_next_round=elem.get("chance_of_playing_next_round"),
    )


def normalize_teams(data: dict[str, Any]) -> tuple[list[Team], ValidationReport]:
    """Normalize the ``teams`` section of a bootstrap-static document.

    Raises:
        IngestionError: if ``data`` is not a dict, ``teams`` is absent/invalid,
            or no valid teams remain after normalization.
    """
    if not isinstance(data, dict):
        err = IngestionError("payload not a dict")
        err.add_note("normalize_teams expects a bootstrap-static document dict")
        raise err
    teams_raw = _require_section(data, "teams", "teams")

    report = ValidationReport()
    accepted: dict[int, Team] = {}
    seen_raw: dict[int, dict[str, Any]] = {}
    conflicted_ids: set[int] = set()

    for team in teams_raw:
        report.seen += 1
        if not isinstance(team, dict):
            report.rejected += 1
            report.rejections.append(
                Rejection(source_id="team:unknown", reason="team is not a dict")
            )
            continue

        team_id = team.get("id")
        missing = _missing_keys(team, _REQUIRED_TEAM_FIELDS)
        if missing:
            report.rejected += 1
            report.rejections.append(
                Rejection(
                    source_id=_source_id("team", team_id),
                    reason=f"missing required fields: {', '.join(missing)}",
                )
            )
            continue

        try:
            team_obj = Team(
                id=team["id"],
                code=team["code"],
                name=team["name"],
                short_name=team["short_name"],
                strength=team.get("strength", 0),
                strength_overall_home=team.get("strength_overall_home", 0),
                strength_overall_away=team.get("strength_overall_away", 0),
                strength_attack_home=team.get("strength_attack_home", 0),
                strength_attack_away=team.get("strength_attack_away", 0),
                strength_defence_home=team.get("strength_defence_home", 0),
                strength_defence_away=team.get("strength_defence_away", 0),
                played=team.get("played", 0),
                win=team.get("win", 0),
                loss=team.get("loss", 0),
                draw=team.get("draw", 0),
                points=team.get("points", 0),
                position=team.get("position", 0),
                goals_for=team.get("goals_for", 0),
                goals_against=team.get("goals_against", 0),
                form=team.get("form"),
                pulse_id=team.get("pulse_id"),
            )
        except (ValidationError, ValueError) as exc:
            report.rejected += 1
            report.rejections.append(
                Rejection(source_id=_source_id("team", team_id), reason=str(exc))
            )
            continue

        tid = int(team["id"])
        if tid in conflicted_ids:
            report.conflicts += 1
            report.conflicts_detail.append(
                Conflict(
                    identifier=_source_id("team", tid),
                    reason="conflicting duplicate records",
                )
            )
            continue
        if tid in seen_raw:
            if team == seen_raw[tid]:
                report.duplicates_skipped += 1
            else:
                if tid in accepted:
                    del accepted[tid]
                    report.accepted -= 1
                conflicted_ids.add(tid)
                report.conflicts += 1
                report.conflicts_detail.append(
                    Conflict(
                        identifier=_source_id("team", tid),
                        reason="conflicting duplicate records",
                    )
                )
            continue

        seen_raw[tid] = team
        accepted[tid] = team_obj
        report.accepted += 1

    _finalize(report, "teams")
    return list(accepted.values()), report


def normalize_gameweeks(
    data: dict[str, Any],
) -> tuple[list[Gameweek], ValidationReport]:
    """Normalize the ``events`` section of a bootstrap-static document.

    Raises:
        IngestionError: if ``data`` is not a dict, ``events`` is absent/invalid,
            or no valid gameweeks remain after normalization.
    """
    if not isinstance(data, dict):
        err = IngestionError("payload not a dict")
        err.add_note("normalize_gameweeks expects a bootstrap-static document dict")
        raise err
    events = _require_section(data, "events", "events")

    report = ValidationReport()
    accepted: dict[int, Gameweek] = {}
    seen_raw: dict[int, dict[str, Any]] = {}
    conflicted_ids: set[int] = set()

    for event in events:
        report.seen += 1
        if not isinstance(event, dict):
            report.rejected += 1
            report.rejections.append(
                Rejection(source_id="event:unknown", reason="event is not a dict")
            )
            continue

        event_id = event.get("id")
        missing = _missing_keys(event, _REQUIRED_EVENT_FIELDS)
        if missing:
            report.rejected += 1
            report.rejections.append(
                Rejection(
                    source_id=_source_id("event", event_id),
                    reason=f"missing required fields: {', '.join(missing)}",
                )
            )
            continue

        try:
            gameweek = Gameweek(
                id=event["id"],
                name=event["name"],
                deadline_time=event["deadline_time"],
                deadline_time_epoch=event["deadline_time_epoch"],
                deadline_time_game_offset=event["deadline_time_game_offset"],
                finished=event.get("finished", False),
                data_checked=event.get("data_checked", False),
                is_previous=event.get("is_previous", False),
                is_current=event.get("is_current", False),
                is_next=event.get("is_next", False),
                cup_leagues_created=event.get("cup_leagues_created", False),
                h2h_leagues_created=event.get("h2h_leagues_created", False),
                top_element_info=event.get("top_element_info"),
                chip_plays=event.get("chip_plays", []),
                transfers_made=event.get("transfers_made", 0),
                most_captained=event.get("most_captained"),
                most_vice_captained=event.get("most_vice_captained"),
                average_entry_score=event.get("average_entry_score", 0),
                highest_score=event.get("highest_score", 0),
                highest_scoring_entry=event.get("highest_scoring_entry"),
            )
        except (ValidationError, ValueError) as exc:
            report.rejected += 1
            report.rejections.append(
                Rejection(source_id=_source_id("event", event_id), reason=str(exc))
            )
            continue

        eid = int(event["id"])
        if eid in conflicted_ids:
            report.conflicts += 1
            report.conflicts_detail.append(
                Conflict(
                    identifier=_source_id("event", eid),
                    reason="conflicting duplicate records",
                )
            )
            continue
        if eid in seen_raw:
            if event == seen_raw[eid]:
                report.duplicates_skipped += 1
            else:
                if eid in accepted:
                    del accepted[eid]
                    report.accepted -= 1
                conflicted_ids.add(eid)
                report.conflicts += 1
                report.conflicts_detail.append(
                    Conflict(
                        identifier=_source_id("event", eid),
                        reason="conflicting duplicate records",
                    )
                )
            continue

        seen_raw[eid] = event
        accepted[eid] = gameweek
        report.accepted += 1

    _finalize(report, "gameweeks")
    return list(accepted.values()), report




