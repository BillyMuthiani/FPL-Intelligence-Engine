"""Tests for the processed-data writer (offline, no live API)."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fpl_engine.config import Settings
from fpl_engine.ingestion.processed import (
    PROCESSED_FILENAME,
    PROCESSED_SCHEMA_VERSION,
    ProcessedWriter,
)
from fpl_engine.models import Gameweek, Player, PlayerPosition, Team
from fpl_engine.storage.exceptions import CacheError
from fpl_engine.storage.models import SnapshotMetadata

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def fixed_time() -> datetime:
    return datetime(2024, 1, 15, 12, 30, 0, tzinfo=UTC)


@pytest.fixture
def snapshot_metadata(fixed_time: datetime) -> SnapshotMetadata:
    return SnapshotMetadata(
        schema_version="1",
        source="fpl-api",
        endpoint="/bootstrap-static/",
        url="https://fantasy.premierleague.com/api/bootstrap-static/",
        retrieved_at=fixed_time,
        content_sha256="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        content_length=1234,
    )


@pytest.fixture
def sample_player() -> Player:
    return Player(
        id=1,
        web_name="Salah",
        first_name="Mohamed",
        second_name="Salah",
        team_id=14,
        position=PlayerPosition.MIDFIELDER,
        now_cost=130,
        code=123456,
        element_type=3,
        team_code=14,
    )


@pytest.fixture
def sample_team() -> Team:
    return Team(
        id=14,
        code=3,
        name="Liverpool",
        short_name="LIV",
    )


@pytest.fixture
def sample_gameweek() -> Gameweek:
    return Gameweek(
        id=1,
        name="Gameweek 1",
        deadline_time="2024-08-16T17:30:00Z",
        deadline_time_epoch=1723829400,
        deadline_time_game_offset=0,
        is_current=True,
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
    )


@pytest.fixture
def writer(settings: Settings) -> ProcessedWriter:
    return ProcessedWriter(settings)


# --------------------------------------------------------------------------- #
# Schema and serialization
# --------------------------------------------------------------------------- #


class TestWriteSchema:
    def test_writes_single_json_file(
        self,
        writer: ProcessedWriter,
        tmp_path: Path,
        sample_player: Player,
        sample_team: Team,
        sample_gameweek: Gameweek,
        snapshot_metadata: SnapshotMetadata,
    ) -> None:
        writer.write([sample_player], [sample_team], [sample_gameweek], snapshot_metadata)
        files = list((tmp_path / "processed").glob("*.json"))
        assert len(files) == 1
        assert files[0].name == PROCESSED_FILENAME

    def test_schema_version_present(
        self,
        writer: ProcessedWriter,
        sample_player: Player,
        sample_team: Team,
        sample_gameweek: Gameweek,
        snapshot_metadata: SnapshotMetadata,
    ) -> None:
        writer.write([sample_player], [sample_team], [sample_gameweek], snapshot_metadata)
        bundle = json.loads(writer.output_path.read_text())
        assert bundle["schema_version"] == PROCESSED_SCHEMA_VERSION

    def test_source_snapshot_provenance(
        self,
        writer: ProcessedWriter,
        sample_player: Player,
        sample_team: Team,
        sample_gameweek: Gameweek,
        snapshot_metadata: SnapshotMetadata,
    ) -> None:
        writer.write([sample_player], [sample_team], [sample_gameweek], snapshot_metadata)
        bundle = json.loads(writer.output_path.read_text())
        src = bundle["source_snapshot"]
        assert src["endpoint"] == "/bootstrap-static/"
        assert src["url"] == "https://fantasy.premierleague.com/api/bootstrap-static/"
        assert src["content_sha256"] == snapshot_metadata.content_sha256
        assert src["content_length"] == 1234
        assert src["source"] == "fpl-api"
        assert datetime.fromisoformat(src["retrieved_at"]) == datetime(
            2024, 1, 15, 12, 30, 0, tzinfo=UTC
        )

    def test_players_serialized_with_correct_fields(
        self,
        writer: ProcessedWriter,
        sample_player: Player,
        sample_team: Team,
        sample_gameweek: Gameweek,
        snapshot_metadata: SnapshotMetadata,
    ) -> None:
        writer.write([sample_player], [sample_team], [sample_gameweek], snapshot_metadata)
        bundle = json.loads(writer.output_path.read_text())
        assert len(bundle["players"]) == 1
        p = bundle["players"][0]
        assert p["id"] == 1
        assert p["web_name"] == "Salah"
        assert p["position"] == "MID"  # StrEnum serialized as string
        assert p["team_id"] == 14

    def test_nulls_preserved(
        self,
        writer: ProcessedWriter,
        sample_team: Team,
        sample_gameweek: Gameweek,
        snapshot_metadata: SnapshotMetadata,
    ) -> None:
        player = Player(
            id=2,
            web_name="Unknown",
            first_name="Unknown",
            second_name="Player",
            team_id=1,
            position=PlayerPosition.DEFENDER,
            now_cost=50,
            code=1,
            element_type=2,
            team_code=1,
            expected_goals=None,
            influence=None,
        )
        writer.write([player], [sample_team], [sample_gameweek], snapshot_metadata)
        bundle = json.loads(writer.output_path.read_text())
        assert bundle["players"][0]["expected_goals"] is None
        assert bundle["players"][0]["influence"] is None

    def test_deterministic_output(
        self,
        writer: ProcessedWriter,
        sample_player: Player,
        sample_team: Team,
        sample_gameweek: Gameweek,
        snapshot_metadata: SnapshotMetadata,
    ) -> None:
        """Two writes with identical input must produce identical bytes."""
        writer.write([sample_player], [sample_team], [sample_gameweek], snapshot_metadata)
        raw1 = writer.output_path.read_bytes()

        writer.write([sample_player], [sample_team], [sample_gameweek], snapshot_metadata)
        raw2 = writer.output_path.read_bytes()

        assert raw1 == raw2


# --------------------------------------------------------------------------- #
# Atomic replacement and cleanup
# --------------------------------------------------------------------------- #


class TestAtomicWrite:
    def test_no_temp_file_after_success(
        self,
        writer: ProcessedWriter,
        tmp_path: Path,
        sample_player: Player,
        sample_team: Team,
        sample_gameweek: Gameweek,
        snapshot_metadata: SnapshotMetadata,
    ) -> None:
        writer.write([sample_player], [sample_team], [sample_gameweek], snapshot_metadata)
        tmp_files = list((tmp_path / "processed").glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_temp_file_cleaned_up_on_failure(
        self,
        writer: ProcessedWriter,
        sample_player: Player,
        sample_team: Team,
        sample_gameweek: Gameweek,
        snapshot_metadata: SnapshotMetadata,
    ) -> None:
        """If the atomic write fails mid-way, the temp file must be removed."""
        target = writer.output_path
        target.parent.mkdir(parents=True, exist_ok=True)
        # Create a directory where the temp file should be — open("wb")
        # on it will raise IsADirectoryError (an OSError).
        target.parent.joinpath(target.name + ".tmp").mkdir()

        with pytest.raises((OSError, CacheError)):
            writer.write([sample_player], [sample_team], [sample_gameweek], snapshot_metadata)

        tmp_files = [
            f for f in target.parent.iterdir()
            if f.name.endswith(".tmp") and f.is_file()
        ]
        assert len(tmp_files) == 0

    def test_existing_file_replaced_atomically(
        self,
        writer: ProcessedWriter,
        sample_player: Player,
        sample_team: Team,
        sample_gameweek: Gameweek,
        snapshot_metadata: SnapshotMetadata,
    ) -> None:
        """A previous valid file must be fully replaced, not appended."""
        writer.write([sample_player], [sample_team], [sample_gameweek], snapshot_metadata)
        original_raw = writer.output_path.read_bytes()

        player2 = Player(
            id=99,
            web_name="Haaland",
            first_name="Erling",
            second_name="Haaland",
            team_id=1,
            position=PlayerPosition.FORWARD,
            now_cost=150,
            code=999999,
            element_type=4,
            team_code=1,
        )
        writer.write([player2], [sample_team], [sample_gameweek], snapshot_metadata)
        new_raw = writer.output_path.read_bytes()

        assert original_raw != new_raw
        bundle = json.loads(writer.output_path.read_text())
        assert bundle["players"][0]["web_name"] == "Haaland"


# --------------------------------------------------------------------------- #
# Persistence failure
# --------------------------------------------------------------------------- #


class TestPersistenceFailure:
    def test_cache_error_when_dir_uncreatable(
        self,
        tmp_path: Path,
        sample_player: Player,
        sample_team: Team,
        sample_gameweek: Gameweek,
        snapshot_metadata: SnapshotMetadata,
    ) -> None:
        """A regular file where the processed directory is expected causes CacheError."""
        blocker = tmp_path / "processed_file"
        blocker.touch()  # Create a file, not a directory

        settings = Settings(processed_data_dir=blocker)
        writer = ProcessedWriter(settings)

        with pytest.raises(CacheError):
            writer.write([sample_player], [sample_team], [sample_gameweek], snapshot_metadata)


