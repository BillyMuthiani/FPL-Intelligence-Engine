"""Tests for the ingestion coordinator (offline, no live API).

All tests use temporary directories, deterministic clocks, and mocked
providers.  No network access is required.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import httpx
import pytest

from fpl_engine.config import Settings
from fpl_engine.ingestion import IngestionCoordinator, IngestionCoordinatorError
from fpl_engine.ingestion.processed import ProcessedWriter
from fpl_engine.providers.protocols import BootstrapDataProvider
from fpl_engine.storage import SnapshotCache
from fpl_engine.storage.exceptions import CacheError

# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #

_FIXED_TIME = datetime(2024, 1, 15, 12, 30, 0, tzinfo=UTC)


def _valid_bootstrap() -> dict[str, Any]:
    """Minimal valid bootstrap-static payload (one player, team, event)."""
    return {
        "elements": [
            {
                "id": 1,
                "web_name": "Salah",
                "first_name": "Mohamed",
                "second_name": "Salah",
                "team": 14,
                "element_type": 3,
                "status": "a",
                "now_cost": 130,
                "code": 123456,
                "photo": "salah.jpg",
                "bps": 500,
                "expected_goals": "15.5",
                "chance_of_playing_this_round": 100,
                "chance_of_playing_next_round": 100,
                "expected_minutes": 90,
                "penalties_order": 1,
                "influence": "120.5",
                "value_season": "1.5",
                "clean_sheets": 0,
                "goals_conceded": 0,
            }
        ],
        "teams": [
            {
                "id": 14,
                "code": 3,
                "name": "Liverpool",
                "short_name": "LIV",
                "strength": 5,
                "strength_overall_home": 5,
                "strength_overall_away": 4,
                "strength_attack_home": 5,
                "strength_attack_away": 4,
                "strength_defence_home": 4,
                "strength_defence_away": 4,
                "played": 30,
                "win": 20,
                "loss": 5,
                "draw": 5,
                "points": 65,
                "position": 2,
                "goals_for": 70,
                "goals_against": 30,
                "form": "WWDWL",
                "pulse_id": 14,
            }
        ],
        "events": [
            {
                "id": 1,
                "name": "Gameweek 1",
                "deadline_time": "2024-08-16T17:30:00Z",
                "deadline_time_epoch": 1723829400,
                "deadline_time_game_offset": 0,
                "finished": False,
                "data_checked": False,
                "is_previous": False,
                "is_current": True,
                "is_next": False,
                "cup_leagues_created": False,
                "h2h_leagues_created": False,
                "transfers_made": 0,
                "most_captained": 1,
                "most_vice_captained": 2,
                "average_entry_score": 0,
                "highest_score": 0,
                "highest_scoring_entry": None,
                "top_element_info": None,
                "chip_plays": [],
            }
        ],
    }


def _empty_bootstrap() -> dict[str, Any]:
    """Bootstrap payload where all three required sections are present but empty."""
    return {"elements": [], "teams": [], "events": []}


def _make_provider(
    data: dict[str, Any] | None = None,
    side_effect: BaseException | None = None,
) -> Mock:
    """Create a mock :class:`BootstrapDataProvider`."""
    provider = Mock(spec=BootstrapDataProvider)
    if side_effect is not None:
        provider.get_bootstrap_data.side_effect = side_effect
    else:
        provider.get_bootstrap_data.return_value = data or _valid_bootstrap()
    return provider


def _make_request_error(message: str = "Connection refused") -> httpx.RequestError:
    """Create an ``httpx.RequestError`` without needing a full Request object."""
    return httpx.RequestError(message, request=None)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def fixed_time() -> datetime:
    return _FIXED_TIME


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
    )


@pytest.fixture
def cache(settings: Settings) -> SnapshotCache:
    return SnapshotCache(settings)


@pytest.fixture
def writer(settings: Settings) -> ProcessedWriter:
    return ProcessedWriter(settings)


@pytest.fixture
def endpoint() -> str:
    return "/bootstrap-static/"


# --------------------------------------------------------------------------- #
# API success / cache interaction
# --------------------------------------------------------------------------- #


class TestCoordinatorFlow:
    """Tests for the core run() decision tree."""

    def test_api_success_no_cache(
        self,
        settings: Settings,
        cache: SnapshotCache,
        writer: ProcessedWriter,
        endpoint: str,
        fixed_time: datetime,
    ) -> None:
        """API success with no cache — stores raw, normalizes, writes processed."""
        provider = _make_provider()
        coord = IngestionCoordinator(
            provider, cache, writer, ttl_seconds=3600.0, endpoint=endpoint
        )

        result = coord.run(now=fixed_time)

        assert result.source == "live"
        assert result.is_fresh is True
        assert result.snapshot_metadata.retrieved_at == fixed_time
        assert result.snapshot_metadata.source == "fpl-api"
        assert result.snapshot_metadata.endpoint == endpoint
        assert len(result.players) == 1
        assert len(result.teams) == 1
        assert len(result.gameweeks) == 1
        # Processed file must exist
        assert writer.output_path.exists()
        # API must have been called
        provider.get_bootstrap_data.assert_called_once()
        # Raw snapshot must have been stored (data + meta sidecar)
        raw_dir = settings.raw_data_dir / "bootstrap-static"
        raw_data_files = [
            f for f in raw_dir.glob("*.json")
            if not f.name.endswith(".meta.json")
        ]
        assert len(raw_data_files) == 1
        raw_meta_files = list(raw_dir.glob("*.meta.json"))
        assert len(raw_meta_files) == 1

    def test_fresh_cache_hit_no_api(
        self,
        cache: SnapshotCache,
        writer: ProcessedWriter,
        endpoint: str,
        fixed_time: datetime,
    ) -> None:
        """Fresh cached snapshot — coordinator skips the API entirely."""
        data = _valid_bootstrap()
        cache.store_raw(
            endpoint, data,
            source="fpl-api", url="https://fantasy.premierleague.com/api/bootstrap-static/",
            retrieved_at=fixed_time,
        )

        provider = _make_provider()
        coord = IngestionCoordinator(
            provider, cache, writer, ttl_seconds=3600.0, endpoint=endpoint
        )

        # now == fixed_time → snapshot is 0 seconds old → fresh
        result = coord.run(now=fixed_time)

        assert result.source == "fresh_cache"
        assert result.is_fresh is True
        assert result.snapshot_metadata.retrieved_at == fixed_time
        assert len(result.players) == 1
        provider.get_bootstrap_data.assert_not_called()

    def test_expired_cache_triggers_api(
        self,
        cache: SnapshotCache,
        writer: ProcessedWriter,
        endpoint: str,
        fixed_time: datetime,
    ) -> None:
        """Expired cache (beyond TTL) forces a fresh API call."""
        data = _valid_bootstrap()
        stale_time = fixed_time - timedelta(hours=24)
        cache.store_raw(
            endpoint, data,
            source="fpl-api", url="https://example.com/",
            retrieved_at=stale_time,
        )

        provider = _make_provider()
        coord = IngestionCoordinator(
            provider, cache, writer, ttl_seconds=3600.0, endpoint=endpoint
        )

        result = coord.run(now=fixed_time)

        assert result.source == "live"
        assert result.is_fresh is True
        assert result.snapshot_metadata.retrieved_at == fixed_time
        provider.get_bootstrap_data.assert_called_once()


# --------------------------------------------------------------------------- #
# Stale fallback scenarios
# --------------------------------------------------------------------------- #


class TestStaleFallback:
    def test_api_failure_with_stale_fallback_enabled(
        self,
        settings: Settings,
        cache: SnapshotCache,
        writer: ProcessedWriter,
        endpoint: str,
        fixed_time: datetime,
    ) -> None:
        """API fails, stale snapshot available → use stale, mark is_fresh=False."""
        data = _valid_bootstrap()
        stale_time = fixed_time - timedelta(hours=48)
        cache.store_raw(
            endpoint, data,
            source="fpl-api", url="https://example.com/",
            retrieved_at=stale_time,
        )

        provider = _make_provider(side_effect=_make_request_error())
        coord = IngestionCoordinator(
            provider, cache, writer, ttl_seconds=3600.0, endpoint=endpoint
        )

        result = coord.run(now=fixed_time, allow_stale_fallback=True)

        assert result.source == "stale_cache"
        assert result.is_fresh is False
        assert result.snapshot_metadata.retrieved_at == stale_time
        assert len(result.players) == 1
        # No new snapshot should be stored (API failed, used stale fallback)
        raw_dir = settings.raw_data_dir / "bootstrap-static"
        raw_data_files = [
            f for f in raw_dir.glob("*.json")
            if not f.name.endswith(".meta.json")
        ]
        assert len(raw_data_files) == 1

    def test_api_failure_stale_fallback_disabled(
        self,
        cache: SnapshotCache,
        writer: ProcessedWriter,
        endpoint: str,
        fixed_time: datetime,
    ) -> None:
        """API fails, stale fallback disabled → raise IngestionCoordinatorError."""
        data = _valid_bootstrap()
        stale_time = fixed_time - timedelta(hours=48)
        cache.store_raw(
            endpoint, data,
            source="fpl-api", url="https://example.com/",
            retrieved_at=stale_time,
        )

        provider = _make_provider(side_effect=_make_request_error())
        coord = IngestionCoordinator(
            provider, cache, writer, ttl_seconds=3600.0, endpoint=endpoint
        )

        with pytest.raises(IngestionCoordinatorError):
            coord.run(now=fixed_time, allow_stale_fallback=False)

    def test_api_failure_no_usable_snapshot(
        self,
        cache: SnapshotCache,
        writer: ProcessedWriter,
        endpoint: str,
        fixed_time: datetime,
    ) -> None:
        """API fails, no stale snapshot available → raise error."""
        provider = _make_provider(side_effect=_make_request_error())
        coord = IngestionCoordinator(
            provider, cache, writer, ttl_seconds=3600.0, endpoint=endpoint
        )

        with pytest.raises(IngestionCoordinatorError):
            coord.run(now=fixed_time, allow_stale_fallback=True)


# --------------------------------------------------------------------------- #
# Corrupt cache
# --------------------------------------------------------------------------- #


class TestCorruptCache:
    def test_corrupt_cache_falls_back_to_api(
        self,
        settings: Settings,
        cache: SnapshotCache,
        writer: ProcessedWriter,
        endpoint: str,
        fixed_time: datetime,
    ) -> None:
        """Corrupt snapshot is skipped by cache; coordinator uses API."""
        data = _valid_bootstrap()
        cache.store_raw(
            endpoint, data,
            source="fpl-api", url="https://example.com/",
            retrieved_at=fixed_time,
        )

        snap_dir = Path(settings.raw_data_dir) / "bootstrap-static"
        data_files = [
            f for f in snap_dir.glob("*.json")
            if not f.name.endswith(".meta.json")
        ]
        assert len(data_files) == 1
        data_files[0].write_text("CORRUPTED DATA", encoding="utf-8")

        provider = _make_provider()
        coord = IngestionCoordinator(
            provider, cache, writer, ttl_seconds=3600.0, endpoint=endpoint
        )

        result = coord.run(now=fixed_time)
        assert result.source == "live"
        assert result.is_fresh is True
        assert len(result.players) == 1
        provider.get_bootstrap_data.assert_called_once()


# --------------------------------------------------------------------------- #
# Invalid payload / normalization failure
# --------------------------------------------------------------------------- #


class TestInvalidPayload:
    def test_invalid_bootstrap_structure_raises(
        self,
        cache: SnapshotCache,
        writer: ProcessedWriter,
        endpoint: str,
        fixed_time: datetime,
    ) -> None:
        """Missing required top-level section → IngestionError → co-ordination error."""
        provider = _make_provider(data={"elements": [], "teams": []})  # no events
        coord = IngestionCoordinator(
            provider, cache, writer, ttl_seconds=3600.0, endpoint=endpoint
        )

        with pytest.raises(IngestionCoordinatorError):
            coord.run(now=fixed_time)

    def test_normalization_failure_prevents_processed_write(
        self,
        cache: SnapshotCache,
        writer: ProcessedWriter,
        endpoint: str,
        fixed_time: datetime,
    ) -> None:
        """All-rejected players → IngestionError → no processed file written."""
        bad_player = {
            "id": 1, "first_name": "X", "second_name": "Y",
            "team": 14, "element_type": 3, "now_cost": 100, "code": 1,
            # Missing required field "web_name"
        }
        bootstrap = _valid_bootstrap()
        provider = _make_provider(
            data={"elements": [bad_player], "teams": bootstrap["teams"]},
        )
        coord = IngestionCoordinator(
            provider, cache, writer, ttl_seconds=3600.0, endpoint=endpoint
        )

        with pytest.raises(IngestionCoordinatorError):
            coord.run(now=fixed_time)

        assert not writer.output_path.exists()

    def test_all_empty_sections_rejected(
        self,
        cache: SnapshotCache,
        writer: ProcessedWriter,
        endpoint: str,
        fixed_time: datetime,
    ) -> None:
        """All three sections present but empty → coordinator rejects."""
        provider = _make_provider(data=_empty_bootstrap())
        coord = IngestionCoordinator(
            provider, cache, writer, ttl_seconds=3600.0, endpoint=endpoint
        )

        with pytest.raises(IngestionCoordinatorError) as exc_info:
            coord.run(now=fixed_time)
        assert "all three required sections empty" in str(exc_info.value).lower()


# --------------------------------------------------------------------------- #
# Provenance / freshness flags
# --------------------------------------------------------------------------- #


class TestFreshnessFlags:
    def test_live_result_has_correct_provenance(
        self,
        cache: SnapshotCache,
        writer: ProcessedWriter,
        endpoint: str,
        fixed_time: datetime,
    ) -> None:
        """Live API result: source=live, is_fresh=True, retrieved_at=now."""
        provider = _make_provider()
        coord = IngestionCoordinator(
            provider, cache, writer, ttl_seconds=3600.0, endpoint=endpoint,
            url="https://fantasy.premierleague.com/api/bootstrap-static/",
        )

        result = coord.run(now=fixed_time)

        assert result.source == "live"
        assert result.is_fresh is True
        assert result.snapshot_metadata.retrieved_at == fixed_time

    def test_stale_result_preserves_original_timestamp(
        self,
        cache: SnapshotCache,
        writer: ProcessedWriter,
        endpoint: str,
        fixed_time: datetime,
    ) -> None:
        """Stale fallback: source=stale_cache, is_fresh=False, original timestamp."""
        data = _valid_bootstrap()
        stale_time = fixed_time - timedelta(hours=48)
        cache.store_raw(
            endpoint, data,
            source="fpl-api", url="https://example.com/",
            retrieved_at=stale_time,
        )

        provider = _make_provider(side_effect=_make_request_error())
        coord = IngestionCoordinator(
            provider, cache, writer, ttl_seconds=3600.0, endpoint=endpoint
        )

        result = coord.run(now=fixed_time, allow_stale_fallback=True)

        assert result.source == "stale_cache"
        assert result.is_fresh is False
        assert result.snapshot_metadata.retrieved_at == stale_time

    def test_no_raw_file_written_on_api_failure(
        self,
        settings: Settings,
        cache: SnapshotCache,
        writer: ProcessedWriter,
        endpoint: str,
        fixed_time: datetime,
    ) -> None:
        """On API failure with no snapshot, no raw file is written."""
        provider = _make_provider(side_effect=_make_request_error())
        coord = IngestionCoordinator(
            provider, cache, writer, ttl_seconds=3600.0, endpoint=endpoint
        )

        with pytest.raises(IngestionCoordinatorError):
            coord.run(now=fixed_time)

        raw_dir = Path(settings.raw_data_dir) / "bootstrap-static"
        raw_files = list(raw_dir.glob("*.json")) if raw_dir.exists() else []
        assert len(raw_files) == 0


# --------------------------------------------------------------------------- #
# Processed persistence failure propagation
# --------------------------------------------------------------------------- #


class TestProcessedPersistenceFailure:
    def test_writer_cache_error_propagates(
        self,
        cache: SnapshotCache,
        endpoint: str,
        fixed_time: datetime,
    ) -> None:
        """CacheError from ProcessedWriter.write → IngestionCoordinatorError."""
        mock_writer = Mock(spec=ProcessedWriter)
        mock_writer.write.side_effect = CacheError("disk full")

        provider = _make_provider()
        coord = IngestionCoordinator(
            provider, cache, mock_writer, ttl_seconds=3600.0, endpoint=endpoint
        )

        with pytest.raises(IngestionCoordinatorError) as exc_info:
            coord.run(now=fixed_time)
        assert "Failed to persist processed data" in str(exc_info.value)

    def test_raw_persistence_failure_propagates(
        self,
        writer: ProcessedWriter,
        endpoint: str,
        fixed_time: datetime,
    ) -> None:
        """CacheError from store_raw → IngestionCoordinatorError."""
        mock_cache = Mock(spec=SnapshotCache)
        mock_cache.get_cached.return_value = None
        mock_cache.get_cached_any.return_value = None
        mock_cache.store_raw.side_effect = CacheError("disk full")

        provider = _make_provider()
        coord = IngestionCoordinator(
            provider, mock_cache, writer, ttl_seconds=3600.0, endpoint=endpoint
        )

        with pytest.raises(IngestionCoordinatorError) as exc_info:
            coord.run(now=fixed_time)
        assert "Failed to persist raw bootstrap snapshot" in str(exc_info.value)

    def test_write_processed_false_skips_writer(
        self,
        cache: SnapshotCache,
        endpoint: str,
        fixed_time: datetime,
    ) -> None:
        """write_processed=False: writer is never called."""
        mock_writer = Mock(spec=ProcessedWriter)
        provider = _make_provider()
        coord = IngestionCoordinator(
            provider, cache, mock_writer, ttl_seconds=3600.0, endpoint=endpoint
        )

        result = coord.run(now=fixed_time, write_processed=False)
        mock_writer.write.assert_not_called()
        assert result.source == "live"

    def test_writer_none_skips_write(
        self,
        cache: SnapshotCache,
        endpoint: str,
        fixed_time: datetime,
    ) -> None:
        """Writer=None: coordinator works without writing processed data."""
        provider = _make_provider()
        coord = IngestionCoordinator(
            provider, cache, writer=None, ttl_seconds=3600.0, endpoint=endpoint
        )

        result = coord.run(now=fixed_time)
        assert result.source == "live"
        assert len(result.players) == 1

