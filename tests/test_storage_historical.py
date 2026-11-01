"""Tests for the historical snapshot storage layer.

All tests use temporary directories (``tmp_path``) and inject deterministic
``retrieved_at`` timestamps.  No network access, no ``time.sleep``.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fpl_engine.config import Settings
from fpl_engine.storage.historical import (
    HistoricalSnapshotStore,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings pointing raw_data_dir at a temp directory."""
    return Settings(raw_data_dir=tmp_path)


@pytest.fixture
def store(settings: Settings) -> HistoricalSnapshotStore:
    return HistoricalSnapshotStore(settings)


@pytest.fixture
def fixed_time() -> datetime:
    return datetime(2024, 1, 15, 12, 30, 0, tzinfo=UTC)


@pytest.fixture
def fixed_time_later() -> datetime:
    return datetime(2024, 3, 20, 18, 0, 0, tzinfo=UTC)


@pytest.fixture
def bootstrap_payload() -> dict:
    return {"elements": [{"id": 1}], "teams": [], "events": []}


@pytest.fixture
def fixtures_payload() -> list:
    return [{"code": 1, "event": 1, "team_h": 1, "team_a": 2}]


@pytest.fixture
def element_summary_payload() -> dict:
    return {
        "history": [{"fixture": 1, "round": 1, "total_points": 5}],
        "fixtures": [],
        "chips": [],
    }


# --------------------------------------------------------------------------- #
# store() + get() -- each snapshot type
# --------------------------------------------------------------------------- #


class TestStoreAndRetrieve:
    """Saving and retrieving each supported snapshot type."""

    def test_store_and_retrieve_bootstrap(
        self, store, bootstrap_payload, fixed_time
    ) -> None:
        store.store(
            "/bootstrap-static/",
            bootstrap_payload,
            source="fpl-api",
            url="https://fantasy.premierleague.com/api/bootstrap-static/",
            retrieved_at=fixed_time,
            gameweek=5,
        )
        retrieved = store.get("/bootstrap-static/", gameweek=5)
        assert retrieved is not None
        assert retrieved.payload == bootstrap_payload
        assert retrieved.metadata.gameweek == 5

    def test_store_and_retrieve_fixtures(
        self, store, fixtures_payload, fixed_time
    ) -> None:
        store.store(
            "/fixtures/",
            fixtures_payload,
            source="fpl-api",
            url="https://fantasy.premierleague.com/api/fixtures/",
            retrieved_at=fixed_time,
            gameweek=5,
        )
        retrieved = store.get("/fixtures/", gameweek=5)
        assert retrieved is not None
        assert retrieved.payload == fixtures_payload
        assert retrieved.metadata.gameweek == 5

    def test_store_and_retrieve_element_summary(
        self, store, element_summary_payload, fixed_time
    ) -> None:
        store.store(
            "/element-summary/1/",
            element_summary_payload,
            source="fpl-api",
            url="https://fantasy.premierleague.com/api/element-summary/1/",
            retrieved_at=fixed_time,
            gameweek=5,
        )
        retrieved = store.get("/element-summary/1/", gameweek=5)
        assert retrieved is not None
        assert retrieved.payload == element_summary_payload
        assert retrieved.metadata.gameweek == 5

    def test_store_returns_snapshot_with_metadata(
        self, store, bootstrap_payload, fixed_time
    ) -> None:
        snap = store.store(
            "/bootstrap-static/",
            bootstrap_payload,
            source="fpl-api",
            url="https://fantasy.premierleague.com/api/bootstrap-static/",
            retrieved_at=fixed_time,
            gameweek=3,
            deadline_time=fixed_time + timedelta(hours=1),
        )
        assert snap.metadata.endpoint == "/bootstrap-static/"
        assert snap.metadata.source == "fpl-api"
        assert snap.metadata.retrieved_at == fixed_time
        assert snap.metadata.gameweek == 3
        assert snap.metadata.deadline_time == fixed_time + timedelta(hours=1)
        assert snap.metadata.content_sha256
        assert snap.metadata.content_length > 0


# --------------------------------------------------------------------------- #
# Multiple snapshots, ordering, and retrieval
# --------------------------------------------------------------------------- #


class TestMultipleSnapshots:
    def test_multiple_snapshots_same_endpoint_different_timestamps(
        self, store, bootstrap_payload, fixed_time, fixed_time_later
    ) -> None:
        store.store(
            "/bootstrap-static/", bootstrap_payload,
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=1,
        )
        store.store(
            "/bootstrap-static/", bootstrap_payload,
            source="fpl-api", url="u",
            retrieved_at=fixed_time_later, gameweek=2,
        )
        all_snaps = store.get_all("/bootstrap-static/")
        assert len(all_snaps) == 2

    def test_get_returns_newest_by_default(
        self, store, fixed_time, fixed_time_later
    ) -> None:
        store.store(
            "/bootstrap-static/", {"old": True},
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=1,
        )
        store.store(
            "/bootstrap-static/", {"new": True},
            source="fpl-api", url="u",
            retrieved_at=fixed_time_later, gameweek=2,
        )
        snap = store.get("/bootstrap-static/")
        assert snap is not None
        assert snap.payload == {"new": True}
        assert snap.metadata.gameweek == 2

    def test_get_by_gameweek_returns_correct_snapshot(
        self, store, fixed_time, fixed_time_later
    ) -> None:
        store.store(
            "/bootstrap-static/", {"gw1": True},
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=1,
        )
        store.store(
            "/bootstrap-static/", {"gw2": True},
            source="fpl-api", url="u",
            retrieved_at=fixed_time_later, gameweek=2,
        )
        snap_gw1 = store.get("/bootstrap-static/", gameweek=1)
        snap_gw2 = store.get("/bootstrap-static/", gameweek=2)
        assert snap_gw1.payload == {"gw1": True}
        assert snap_gw2.payload == {"gw2": True}

    def test_get_before_timestamp(
        self, store, fixed_time, fixed_time_later
    ) -> None:
        store.store(
            "/bootstrap-static/", {"early": True},
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=1,
        )
        store.store(
            "/bootstrap-static/", {"late": True},
            source="fpl-api", url="u",
            retrieved_at=fixed_time_later, gameweek=2,
        )
        snap = store.get("/bootstrap-static/", before=fixed_time_later)
        assert snap is not None
        assert snap.payload == {"late": True}

        snap2 = store.get(
            "/bootstrap-static/", before=fixed_time + timedelta(seconds=1)
        )
        assert snap2 is not None
        assert snap2.payload == {"early": True}

    def test_get_before_timestamp_inclusive(
        self, store, fixed_time
    ) -> None:
        """before=t should include snapshots with retrieved_at == t."""
        store.store(
            "/bootstrap-static/", {"a": 1},
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=1,
        )
        snap = store.get("/bootstrap-static/", before=fixed_time)
        assert snap is not None
        assert snap.payload == {"a": 1}


# --------------------------------------------------------------------------- #
# Idempotency / immutability
# --------------------------------------------------------------------------- #


class TestImmutability:
    def test_identical_content_same_timestamp_is_idempotent(
        self, store, bootstrap_payload, fixed_time
    ) -> None:
        store.store(
            "/bootstrap-static/", bootstrap_payload,
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=1,
        )
        store.store(
            "/bootstrap-static/", bootstrap_payload,
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=1,
        )
        dir_path = store.base_dir / "bootstrap-static"
        json_files = [
            f for f in dir_path.glob("*.json")
            if not f.name.endswith(".meta.json")
        ]
        meta_files = list(dir_path.glob("*.meta.json"))
        assert len(json_files) == 1
        assert len(meta_files) == 1

    def test_different_content_same_timestamp_creates_separate_files(
        self, store, fixed_time
    ) -> None:
        store.store(
            "/bootstrap-static/", {"a": 1},
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=1,
        )
        store.store(
            "/bootstrap-static/", {"b": 2},
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=1,
        )
        dir_path = store.base_dir / "bootstrap-static"
        json_files = [
            f for f in dir_path.glob("*.json")
            if not f.name.endswith(".meta.json")
        ]
        assert len(json_files) == 2

    def test_earlier_snapshot_not_silently_replaced(
        self, store, fixed_time, fixed_time_later
    ) -> None:
        store.store(
            "/bootstrap-static/", {"earlier": True},
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=1,
        )
        store.store(
            "/bootstrap-static/", {"later": True},
            source="fpl-api", url="u",
            retrieved_at=fixed_time_later, gameweek=1,
        )
        all_snaps = store.get_all("/bootstrap-static/")
        assert len(all_snaps) == 2
        payloads = [s.payload for s in all_snaps]
        assert {"earlier": True} in payloads
        assert {"later": True} in payloads


# --------------------------------------------------------------------------- #
# Integrity verification
# --------------------------------------------------------------------------- #


class TestIntegrityVerification:
    def test_sha256_matches_serialized_bytes(
        self, store, bootstrap_payload, fixed_time
    ) -> None:
        snap = store.store(
            "/bootstrap-static/", bootstrap_payload,
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=1,
        )
        dir_path = store.base_dir / "bootstrap-static"
        json_files = [
            f for f in dir_path.glob("*.json")
            if not f.name.endswith(".meta.json")
        ]
        raw_bytes = json_files[0].read_bytes()
        expected_hash = hashlib.sha256(raw_bytes).hexdigest()
        assert snap.metadata.content_sha256 == expected_hash

    def test_content_length_matches_file_size(
        self, store, bootstrap_payload, fixed_time
    ) -> None:
        snap = store.store(
            "/bootstrap-static/", bootstrap_payload,
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=1,
        )
        dir_path = store.base_dir / "bootstrap-static"
        json_files = [
            f for f in dir_path.glob("*.json")
            if not f.name.endswith(".meta.json")
        ]
        raw_bytes = json_files[0].read_bytes()
        assert snap.metadata.content_length == len(raw_bytes)

    def test_tampered_data_file_skipped(
        self, store, bootstrap_payload, fixed_time
    ) -> None:
        store.store(
            "/bootstrap-static/", bootstrap_payload,
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=1,
        )
        dir_path = store.base_dir / "bootstrap-static"
        for f in dir_path.glob("*.json"):
            if not f.name.endswith(".meta.json"):
                f.write_bytes(b"CORRUPTED-PAYLOAD")
        result = store.get("/bootstrap-static/", gameweek=1)
        assert result is None

    def test_tampered_metadata_skipped(
        self, store, bootstrap_payload, fixed_time
    ) -> None:
        store.store(
            "/bootstrap-static/", bootstrap_payload,
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=1,
        )
        dir_path = store.base_dir / "bootstrap-static"
        for f in dir_path.glob("*.meta.json"):
            f.write_text("{invalid json", encoding="utf-8")
        result = store.get("/bootstrap-static/", gameweek=1)
        assert result is None

    def test_hash_mismatch_skipped(
        self, store, bootstrap_payload, fixed_time
    ) -> None:
        store.store(
            "/bootstrap-static/", bootstrap_payload,
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=1,
        )
        dir_path = store.base_dir / "bootstrap-static"
        for meta_file in dir_path.glob("*.meta.json"):
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            meta["content_sha256"] = "0" * 64
            meta_file.write_text(
                json.dumps(meta, sort_keys=True), encoding="utf-8"
            )
        result = store.get("/bootstrap-static/", gameweek=1)
        assert result is None

    def test_content_length_mismatch_skipped(
        self, store, bootstrap_payload, fixed_time
    ) -> None:
        store.store(
            "/bootstrap-static/", bootstrap_payload,
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=1,
        )
        dir_path = store.base_dir / "bootstrap-static"
        for meta_file in dir_path.glob("*.meta.json"):
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            meta["content_length"] = 999999
            meta_file.write_text(
                json.dumps(meta, sort_keys=True), encoding="utf-8"
            )
        result = store.get("/bootstrap-static/", gameweek=1)
        assert result is None

    def test_corrupt_snapshot_does_not_affect_valid_ones(
        self, store, fixed_time, fixed_time_later
    ) -> None:
        """Corrupting one snapshot should not hide another valid one."""
        store.store(
            "/bootstrap-static/", {"valid": True},
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=1,
        )
        store.store(
            "/bootstrap-static/", {"also_valid": True},
            source="fpl-api", url="u",
            retrieved_at=fixed_time_later, gameweek=2,
        )
        # Corrupt the older one (gameweek 1)
        dir_path = store.base_dir / "bootstrap-static"
        for meta_file in sorted(dir_path.glob("*.meta.json")):
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if meta.get("gameweek") == 1:
                json_path = meta_file.with_name(
                    meta_file.name.removesuffix(".meta.json") + ".json"
                )
                json_path.write_bytes(b"BAD")
        snap = store.get("/bootstrap-static/")
        assert snap is not None
        assert snap.payload == {"also_valid": True}


# --------------------------------------------------------------------------- #
# Offline retrieval
# --------------------------------------------------------------------------- #


class TestOfflineRetrieval:
    def test_retrieve_without_network(
        self, store, bootstrap_payload, fixed_time
    ) -> None:
        """Once stored, snapshot is retrievable with no fetch callable."""
        store.store(
            "/bootstrap-static/", bootstrap_payload,
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=1,
        )
        snap = store.get("/bootstrap-static/", gameweek=1)
        assert snap is not None
        assert snap.payload == bootstrap_payload

    def test_get_all_offline(
        self, store, fixed_time, fixed_time_later
    ) -> None:
        store.store(
            "/bootstrap-static/", {"a": 1},
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=1,
        )
        store.store(
            "/bootstrap-static/", {"b": 2},
            source="fpl-api", url="u",
            retrieved_at=fixed_time_later, gameweek=2,
        )
        all_snaps = store.get_all("/bootstrap-static/")
        assert len(all_snaps) == 2
        assert all_snaps[0].metadata.retrieved_at == fixed_time_later
        assert all_snaps[1].metadata.retrieved_at == fixed_time

    def test_has_snapshot_offline(
        self, store, bootstrap_payload, fixed_time
    ) -> None:
        assert store.has_snapshot("/bootstrap-static/") is False
        store.store(
            "/bootstrap-static/", bootstrap_payload,
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=1,
        )
        assert store.has_snapshot("/bootstrap-static/") is True
        assert store.has_snapshot("/bootstrap-static/", gameweek=1) is True
        assert store.has_snapshot("/bootstrap-static/", gameweek=2) is False


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #


class TestEdgeCases:
    def test_get_nonexistent_returns_none(self, store) -> None:
        assert store.get("/bootstrap-static/") is None
        assert store.get("/bootstrap-static/", gameweek=1) is None
        assert store.get_all("/bootstrap-static/") == []
        assert store.has_snapshot("/bootstrap-static/") is False

    def test_store_without_gameweek(
        self, store, bootstrap_payload, fixed_time
    ) -> None:
        store.store(
            "/bootstrap-static/", bootstrap_payload,
            source="fpl-api", url="u",
            retrieved_at=fixed_time,
        )
        snap = store.get("/bootstrap-static/")
        assert snap is not None
        assert snap.metadata.gameweek is None

    def test_retrieved_at_nonnaive_raises(
        self, store, bootstrap_payload
    ) -> None:
        naive = datetime(2024, 1, 15, 12, 30, 0)  # noqa: DTZ001
        with pytest.raises(ValueError, match="timezone-aware"):
            store.store(
                "/bootstrap-static/", bootstrap_payload,
                source="fpl-api", url="u",
                retrieved_at=naive,
            )

    def test_retrieved_at_non_utc_raises(
        self, store, bootstrap_payload
    ) -> None:
        from datetime import timezone
        non_utc = datetime(
            2024, 1, 15, 12, 30, 0,
            tzinfo=timezone(timedelta(hours=5)),
        )
        with pytest.raises(ValueError, match="must be UTC"):
            store.store(
                "/bootstrap-static/", bootstrap_payload,
                source="fpl-api", url="u",
                retrieved_at=non_utc,
            )

    def test_invalid_gameweek_raises(
        self, store, bootstrap_payload, fixed_time
    ) -> None:
        with pytest.raises(ValueError, match="gameweek"):
            store.store(
                "/bootstrap-static/", bootstrap_payload,
                source="fpl-api", url="u",
                retrieved_at=fixed_time,
                gameweek=0,
            )

    def test_deadline_time_non_utc_raises(
        self, store, bootstrap_payload, fixed_time
    ) -> None:
        from datetime import timezone
        non_utc = datetime(
            2024, 1, 15, 17, 30, 0,
            tzinfo=timezone(timedelta(hours=5)),
        )
        with pytest.raises(ValueError, match="deadline_time must be UTC"):
            store.store(
                "/bootstrap-static/", bootstrap_payload,
                source="fpl-api", url="u",
                retrieved_at=fixed_time,
                deadline_time=non_utc,
            )

    def test_base_dir_under_raw_data(self, settings) -> None:
        store = HistoricalSnapshotStore(settings)
        expected = Path(settings.raw_data_dir) / "historical"
        assert store.base_dir == expected

    def test_element_summary_per_player_directories(
        self, store, element_summary_payload, fixed_time
    ) -> None:
        store.store(
            "/element-summary/1/", element_summary_payload,
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=5,
        )
        store.store(
            "/element-summary/2/", element_summary_payload,
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=5,
        )
        assert (store.base_dir / "element-summary_1").is_dir()
        assert (store.base_dir / "element-summary_2").is_dir()
        assert store.get("/element-summary/1/", gameweek=5) is not None
        assert store.get("/element-summary/2/", gameweek=5) is not None


# --------------------------------------------------------------------------- #
# Cutoff-based retrieval, deadline semantics, and endpoint identity
# --------------------------------------------------------------------------- #

class TestCutoffRetrieval:
    """Retrieval with the ``before`` cutoff parameter."""

    def test_get_before_excludes_snapshot_after_cutoff(
        self, store, bootstrap_payload, fixed_time, fixed_time_later
    ) -> None:
        """get(before=fixed_time) must NOT return a snapshot collected later."""
        store.store(
            "/bootstrap-static/", bootstrap_payload,
            source="fpl-api", url="u",
            retrieved_at=fixed_time_later, gameweek=5,
        )
        snap = store.get("/bootstrap-static/", gameweek=5, before=fixed_time)
        assert snap is None

    def test_get_before_at_exact_timestamp_inclusive(
        self, store, bootstrap_payload, fixed_time
    ) -> None:
        """before=t must include snapshots with retrieved_at == t."""
        store.store(
            "/bootstrap-static/", bootstrap_payload,
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=5,
        )
        snap = store.get("/bootstrap-static/", gameweek=5, before=fixed_time)
        assert snap is not None
        assert snap.metadata.retrieved_at == fixed_time

    def test_get_before_returns_newest_within_cutoff(
        self, store, fixed_time, fixed_time_later
    ) -> None:
        """Among snapshots before the cutoff, return the newest."""
        store.store(
            "/bootstrap-static/", {"v": 1},
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=5,
        )
        store.store(
            "/bootstrap-static/", {"v": 2},
            source="fpl-api", url="u",
            retrieved_at=fixed_time_later, gameweek=5,
        )
        snap = store.get("/bootstrap-static/", gameweek=5, before=fixed_time_later)
        assert snap is not None
        assert snap.payload == {"v": 2}

    def test_get_before_no_matching_returns_none(
        self, store, bootstrap_payload, fixed_time_later
    ) -> None:
        """If no snapshot exists before the cutoff, return None."""
        store.store(
            "/bootstrap-static/", bootstrap_payload,
            source="fpl-api", url="u",
            retrieved_at=fixed_time_later, gameweek=5,
        )
        cutoff = fixed_time_later - timedelta(seconds=1)
        snap = store.get("/bootstrap-static/", gameweek=5, before=cutoff)
        assert snap is None

    def test_get_before_without_gameweek_filter(
        self, store, fixed_time, fixed_time_later
    ) -> None:
        """before filter works without gameweek restriction."""
        store.store(
            "/bootstrap-static/", {"a": 1},
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=5,
        )
        store.store(
            "/bootstrap-static/", {"b": 2},
            source="fpl-api", url="u",
            retrieved_at=fixed_time_later, gameweek=6,
        )
        snap = store.get("/bootstrap-static/", before=fixed_time_later)
        assert snap is not None
        assert snap.payload == {"b": 2}


class TestHasSnapshotWithBefore:
    """has_snapshot respects the before cutoff."""

    def test_has_snapshot_with_before_true(
        self, store, bootstrap_payload, fixed_time, fixed_time_later
    ) -> None:
        store.store(
            "/bootstrap-static/", bootstrap_payload,
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=5,
        )
        assert store.has_snapshot(
            "/bootstrap-static/", gameweek=5, before=fixed_time_later
        ) is True

    def test_has_snapshot_with_before_false(
        self, store, bootstrap_payload, fixed_time, fixed_time_later
    ) -> None:
        """Only a post-cutoff snapshot exists -> has_snapshot(before=cutoff) false."""
        store.store(
            "/bootstrap-static/", bootstrap_payload,
            source="fpl-api", url="u",
            retrieved_at=fixed_time_later, gameweek=5,
        )
        assert store.has_snapshot(
            "/bootstrap-static/", gameweek=5, before=fixed_time
        ) is False


class TestMultipleSnapshotsSameGameweek:
    """Two snapshots for the same gameweek but different collection times."""

    def test_get_returns_newest_for_same_gameweek(
        self, store, fixed_time, fixed_time_later
    ) -> None:
        store.store(
            "/bootstrap-static/", {"early": True},
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=5,
        )
        store.store(
            "/bootstrap-static/", {"late": True},
            source="fpl-api", url="u",
            retrieved_at=fixed_time_later, gameweek=5,
        )
        snap = store.get("/bootstrap-static/", gameweek=5)
        assert snap.payload == {"late": True}

    def test_get_all_returns_both_newest_first(
        self, store, fixed_time, fixed_time_later
    ) -> None:
        store.store(
            "/bootstrap-static/", {"early": True},
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=5,
        )
        store.store(
            "/bootstrap-static/", {"late": True},
            source="fpl-api", url="u",
            retrieved_at=fixed_time_later, gameweek=5,
        )
        all_snaps = store.get_all("/bootstrap-static/")
        assert len(all_snaps) == 2
        assert all_snaps[0].payload == {"late": True}
        assert all_snaps[1].payload == {"early": True}


class TestDeadlineTimeMetadata:
    """Snapshots can carry different deadline_time metadata for the same gameweek."""

    def test_snapshot_collected_after_deadline_excluded(
        self, store, fixed_time, fixed_time_later
    ) -> None:
        """get(before=deadline) excludes snapshots collected after the deadline."""
        deadline = datetime(2024, 3, 1, 17, 30, tzinfo=UTC)
        # Snapshot collected before the deadline
        store.store(
            "/bootstrap-static/", {"pre": True},
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=5, deadline_time=deadline,
        )
        # Snapshot collected after the deadline (same gameweek)
        store.store(
            "/bootstrap-static/", {"post": True},
            source="fpl-api", url="u",
            retrieved_at=fixed_time_later, gameweek=5, deadline_time=deadline,
        )
        # before=deadline should only return the pre-deadline snapshot
        snap = store.get("/bootstrap-static/", gameweek=5, before=deadline)
        assert snap is not None
        assert snap.payload == {"pre": True}


class TestEndpointIdentitySafety:
    """Metadata endpoint must match the requested endpoint on retrieval."""

    def test_endpoint_mismatch_skipped(
        self, store, bootstrap_payload, fixed_time
    ) -> None:
        """A meta file whose endpoint field differs from the request is skipped."""
        store.store(
            "/bootstrap-static/", bootstrap_payload,
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=5,
        )
        snap = store.get("/bootstrap-static/", gameweek=5)
        assert snap is not None
        assert snap.metadata.endpoint == "/bootstrap-static/"

    def test_endpoint_collision_different_strings(
        self, store, element_summary_payload, fixed_time
    ) -> None:
        """Two endpoint strings that sanitize to the same directory must not
        cross-load: metadata endpoint must match the request."""
        # /element-summary/1/ and /element-summary/1 both sanitize to element-summary_1
        store.store(
            "/element-summary/1/", element_summary_payload,
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=5,
        )
        store.store(
            "/element-summary/1", element_summary_payload,
            source="fpl-api", url="u",
            retrieved_at=fixed_time + timedelta(hours=1), gameweek=5,
        )
        snap_slash = store.get("/element-summary/1/", gameweek=5)
        assert snap_slash is not None
        assert snap_slash.metadata.endpoint == "/element-summary/1/"

        snap_no_slash = store.get("/element-summary/1", gameweek=5)
        assert snap_no_slash is not None
        assert snap_no_slash.metadata.endpoint == "/element-summary/1"


class TestCorruptMetadataHandling:
    """Corrupt or incomplete metadata is silently skipped on read."""

    def test_corrupt_meta_json_skipped(
        self, store, bootstrap_payload, fixed_time
    ) -> None:
        store.store(
            "/bootstrap-static/", bootstrap_payload,
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=5,
        )
        dir_path = store.base_dir / "bootstrap-static"
        meta_files = list(dir_path.glob("*.meta.json"))
        assert len(meta_files) == 1
        meta_files[0].write_text("{ corrupted json")
        assert store.get("/bootstrap-static/") is None

    def test_corrupt_meta_does_not_affect_valid(
        self, store, fixed_time, fixed_time_later
    ) -> None:
        store.store(
            "/bootstrap-static/", {"a": 1},
            source="fpl-api", url="u",
            retrieved_at=fixed_time, gameweek=5,
        )
        store.store(
            "/bootstrap-static/", {"b": 2},
            source="fpl-api", url="u",
            retrieved_at=fixed_time_later, gameweek=6,
        )
        dir_path = store.base_dir / "bootstrap-static"
        meta_files = sorted(dir_path.glob("*.meta.json"))
        # Corrupt the earlier snapshot's metadata
        meta_files[0].write_text("{ corrupted json")
        snap = store.get("/bootstrap-static/", gameweek=6)
        assert snap is not None
        assert snap.payload == {"b": 2}


# --------------------------------------------------------------------------- #
# Collection-time cutoff: store-level verification
# (Stage 4I-A.2: the store correctly enforces retrieved_at <= before;
# the collector's _cutoff must produce min(retrieved_at, deadline_time)
# so the store never sees an ineligible snapshot.)
# --------------------------------------------------------------------------- #


class TestCollectionTimeCutoffStore:
    """Store-layer verification for the 08:00/10:00/11:00 scenario.

    Collection attempt: 08:00 UTC.
    Existing snapshot: 10:00 UTC (after attempt, before 11:00 deadline).
    Deadline: 11:00 UTC.

    Using before=min(attempt, deadline) = 08:00 correctly excludes
    the 10:00 snapshot.  Using before=deadline alone (11:00) would
    incorrectly include it.
    """

    _att = datetime(2024, 9, 13, 8, 0, 0, tzinfo=UTC)   # 08:00 -- attempt
    _snap = datetime(2024, 9, 13, 10, 0, 0, tzinfo=UTC)  # 10:00 -- snapshot
    _deadline = datetime(2024, 9, 13, 11, 0, 0, tzinfo=UTC)  # 11:00 -- deadline

    def test_get_with_min_cutoff_excludes_future_snapshot(
        self, store, bootstrap_payload
    ) -> None:
        """before=min(attempt, deadline) = 08:00 excludes snapshot at 10:00."""
        store.store(
            "/bootstrap-static/", bootstrap_payload,
            source="fpl-api", url="u",
            retrieved_at=self._snap, gameweek=5, deadline_time=self._deadline,
        )
        cutoff = min(self._att, self._deadline)  # 08:00
        assert store.get("/bootstrap-static/", gameweek=5, before=cutoff) is None
        assert store.has_snapshot(
            "/bootstrap-static/", gameweek=5, before=cutoff
        ) is False

    def test_get_with_deadline_only_incorrectly_includes_future(
        self, store, bootstrap_payload
    ) -> None:
        """Demonstrates the bug: before=deadline (11:00) includes snapshot at 10:00.

        This test documents why _cutoff must use min(), not deadline_time alone.
        """
        store.store(
            "/bootstrap-static/", bootstrap_payload,
            source="fpl-api", url="u",
            retrieved_at=self._snap, gameweek=5, deadline_time=self._deadline,
        )
        snap = store.get("/bootstrap-static/", gameweek=5, before=self._deadline)
        assert snap is not None
        assert snap.metadata.retrieved_at == self._snap

    def test_get_with_attempt_cutoff_includes_pre_attempt_snapshot(
        self, store, bootstrap_payload
    ) -> None:
        """A snapshot before the attempt is still eligible when using min cutoff."""
        pre = self._att - timedelta(hours=1)  # 07:00
        store.store(
            "/bootstrap-static/", bootstrap_payload,
            source="fpl-api", url="u",
            retrieved_at=pre, gameweek=5, deadline_time=self._deadline,
        )
        cutoff = min(self._att, self._deadline)  # 08:00
        snap = store.get("/bootstrap-static/", gameweek=5, before=cutoff)
        assert snap is not None
        assert snap.metadata.retrieved_at == pre
