"""Tests for the raw-snapshot filesystem cache.

All tests use temporary directories (``tmp_path``) and inject deterministic
``retrieved_at`` timestamps. No network access, no ``time.sleep``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from fpl_engine.config import Settings
from fpl_engine.storage import SnapshotCache
from fpl_engine.storage.exceptions import CacheError
from fpl_engine.storage.models import SnapshotMetadata


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings pointing ``raw_data_dir`` at a temporary directory."""
    return Settings(raw_data_dir=tmp_path)


@pytest.fixture
def cache(settings: Settings) -> SnapshotCache:
    return SnapshotCache(settings)


@pytest.fixture
def fixed_time() -> datetime:
    """A deterministic UTC timestamp for test snapshots."""
    return datetime(2024, 1, 15, 12, 30, 0, tzinfo=UTC)


@pytest.fixture
def payload() -> dict:
    return {"elements": [{"id": 1, "name": "Player"}], "teams": []}



# --------------------------------------------------------------------------- #
# store_raw -- basic behavior
# --------------------------------------------------------------------------- #


class TestStoreRaw:
    def test_creates_json_and_meta_files(self, cache, tmp_path, payload, fixed_time) -> None:
        cache.store_raw(
            "/bootstrap-static/", payload,
            source="fpl-api", url="https://example.com/",
            retrieved_at=fixed_time,
        )
        files = sorted((tmp_path / "bootstrap-static").iterdir())
        assert len(files) == 2

    def test_returns_snapshot_with_correct_payload(self, cache, payload, fixed_time) -> None:
        snap = cache.store_raw(
            "/bootstrap-static/", payload,
            source="fpl-api", url="https://example.com/",
            retrieved_at=fixed_time,
        )
        assert snap.payload == payload

    def test_sha256_matches_serialized_bytes(self, cache, tmp_path, payload, fixed_time) -> None:
        snap = cache.store_raw(
            "/bootstrap-static/", payload,
            source="fpl-api", url="https://example.com/",
            retrieved_at=fixed_time,
        )
        json_files = [f for f in tmp_path.rglob("*.json") if not f.name.endswith(".meta.json")]
        raw_bytes = json_files[0].read_bytes()
        expected_hash = hashlib.sha256(raw_bytes).hexdigest()
        assert snap.metadata.content_sha256 == expected_hash

    def test_content_length_matches_file_size(self, cache, tmp_path, payload, fixed_time) -> None:
        snap = cache.store_raw(
            "/bootstrap-static/", payload,
            source="fpl-api", url="https://example.com/",
            retrieved_at=fixed_time,
        )
        json_files = [f for f in tmp_path.rglob("*.json") if not f.name.endswith(".meta.json")]
        raw_bytes = json_files[0].read_bytes()
        assert snap.metadata.content_length == len(raw_bytes)

    def test_uses_injected_timestamp(self, cache, payload, fixed_time) -> None:
        snap = cache.store_raw(
            "/bootstrap-static/", payload,
            source="fpl-api", url="https://example.com/",
            retrieved_at=fixed_time,
        )
        assert snap.metadata.retrieved_at == fixed_time

    def test_default_timestamp_is_utc_now(self, cache, payload) -> None:
        snap = cache.store_raw(
            "/bootstrap-static/", payload,
            source="fpl-api", url="https://example.com/",
        )
        assert snap.metadata.retrieved_at.tzinfo is not None
        assert snap.metadata.retrieved_at.utcoffset() == UTC.utcoffset(None)

    def test_creates_raw_data_directory(self, cache, payload, fixed_time) -> None:
        cache.store_raw(
            "/bootstrap-static/", payload,
            source="fpl-api", url="u",
            retrieved_at=fixed_time,
        )
        assert (Path(cache._settings.raw_data_dir) / "bootstrap-static").is_dir()

    def test_deterministic_serialization(self, cache, payload, fixed_time) -> None:
        cache.store_raw("/bootstrap-static/", payload, source="fpl-api", url="u", retrieved_at=fixed_time)
        reversed_payload = {k: payload[k] for k in reversed(payload)}
        cache.store_raw("/bootstrap-static/", reversed_payload, source="fpl-api", url="u", retrieved_at=fixed_time)
        json_files = [f for f in Path(cache._settings.raw_data_dir).rglob("*.json") if not f.name.endswith(".meta.json")]
        assert len(json_files) == 1

    def test_no_temp_files_after_store(self, cache, tmp_path, payload, fixed_time) -> None:
        cache.store_raw("/bootstrap-static/", payload, source="fpl-api", url="u", retrieved_at=fixed_time)
        assert len(list(tmp_path.rglob("*.tmp"))) == 0



# --------------------------------------------------------------------------- #
# get_cached -- basic retrieval
# --------------------------------------------------------------------------- #


class TestGetCached:
    def test_returns_none_when_no_cache(self, cache) -> None:
        assert cache.get_cached("/bootstrap-static/") is None

    def test_returns_snapshot_when_fresh(self, cache, payload, fixed_time) -> None:
        cache.store_raw("/bootstrap-static/", payload, source="fpl-api", url="u", retrieved_at=fixed_time)
        snap = cache.get_cached("/bootstrap-static/")
        assert snap is not None
        assert snap.payload == payload
        assert snap.metadata.retrieved_at == fixed_time

    def test_returns_none_for_unknown_endpoint(self, cache, payload, fixed_time) -> None:
        cache.store_raw("/bootstrap-static/", payload, source="fpl-api", url="u", retrieved_at=fixed_time)
        assert cache.get_cached("/players/") is None


# --------------------------------------------------------------------------- #
# TTL behavior
# --------------------------------------------------------------------------- #


class TestTTL:
    def test_fresh_within_ttl(self, cache, payload, fixed_time) -> None:
        cache.store_raw("/bootstrap-static/", payload, source="fpl-api", url="u", retrieved_at=fixed_time)
        snap = cache.get_cached("/bootstrap-static/", ttl_seconds=3600, now=fixed_time + timedelta(seconds=1800))
        assert snap is not None

    def test_stale_beyond_ttl_returns_none(self, cache, payload) -> None:
        old_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        cache.store_raw("/bootstrap-static/", payload, source="fpl-api", url="u", retrieved_at=old_time)
        snap = cache.get_cached("/bootstrap-static/", ttl_seconds=60)
        assert snap is None

    def test_ttl_none_returns_any_valid(self, cache, payload) -> None:
        old_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        cache.store_raw("/bootstrap-static/", payload, source="fpl-api", url="u", retrieved_at=old_time)
        snap = cache.get_cached("/bootstrap-static/", ttl_seconds=None)
        assert snap is not None
        assert snap.payload == payload


# --------------------------------------------------------------------------- #
# get_cached_any -- stale fallback
# --------------------------------------------------------------------------- #


class TestGetCachedAny:
    def test_returns_stale_valid_snapshot(self, cache, payload) -> None:
        old_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        cache.store_raw("/bootstrap-static/", payload, source="fpl-api", url="u", retrieved_at=old_time)
        snap = cache.get_cached_any("/bootstrap-static/")
        assert snap is not None
        assert snap.metadata.retrieved_at == old_time

    def test_returns_none_when_no_valid_snapshot(self, cache) -> None:
        assert cache.get_cached_any("/bootstrap-static/") is None



# --------------------------------------------------------------------------- #
# Metadata validation
# --------------------------------------------------------------------------- #


class TestMetadataValidation:
    def test_invalid_sha256_raises(self) -> None:
        with pytest.raises(ValidationError):
            SnapshotMetadata(
                schema_version="1", source="fpl-api", endpoint="/bootstrap-static/",
                url="u", retrieved_at=datetime(2024, 1, 1, tzinfo=UTC),
                content_sha256="not-a-valid-hash", content_length=100,
            )

    def test_negative_content_length_raises(self) -> None:
        with pytest.raises(ValidationError):
            SnapshotMetadata(
                schema_version="1", source="fpl-api", endpoint="/bootstrap-static/",
                url="u", retrieved_at=datetime(2024, 1, 1, tzinfo=UTC),
                content_sha256="a" * 64, content_length=-1,
            )

    def test_naive_datetime_raises(self) -> None:
        with pytest.raises(ValidationError):
            SnapshotMetadata(
                schema_version="1", source="fpl-api", endpoint="/bootstrap-static/",
                url="u", retrieved_at=datetime(2024, 1, 1),  # noqa: DTZ001
                content_sha256="a" * 64, content_length=0,
            )

    def test_non_utc_timezone_raises(self) -> None:
        with pytest.raises(ValidationError):
            SnapshotMetadata(
                schema_version="1", source="fpl-api", endpoint="/bootstrap-static/",
                url="u", retrieved_at=datetime(2024, 1, 1, tzinfo=timezone(timedelta(hours=5))),
                content_sha256="a" * 64, content_length=0,
            )

    def test_empty_source_raises(self) -> None:
        with pytest.raises(ValidationError):
            SnapshotMetadata(
                schema_version="1", source="", endpoint="/bootstrap-static/",
                url="u", retrieved_at=datetime(2024, 1, 1, tzinfo=UTC),
                content_sha256="a" * 64, content_length=0,
            )

    def test_endpoint_without_leading_slash_raises(self) -> None:
        with pytest.raises(ValidationError):
            SnapshotMetadata(
                schema_version="1", source="fpl-api", endpoint="bootstrap-static/",
                url="u", retrieved_at=datetime(2024, 1, 1, tzinfo=UTC),
                content_sha256="a" * 64, content_length=0,
            )

    def test_unsupported_schema_version_raises(self) -> None:
        with pytest.raises(ValidationError):
            SnapshotMetadata(
                schema_version="99", source="fpl-api", endpoint="/bootstrap-static/",
                url="u", retrieved_at=datetime(2024, 1, 1, tzinfo=UTC),
                content_sha256="a" * 64, content_length=0,
            )

    def test_supported_schema_version_accepted(self) -> None:
        meta = SnapshotMetadata(
            schema_version="1", source="fpl-api", endpoint="/bootstrap-static/",
            url="u", retrieved_at=datetime(2024, 1, 1, tzinfo=UTC),
            content_sha256="a" * 64, content_length=0,
        )
        assert meta.schema_version == "1"
        assert "1" in SnapshotMetadata.SUPPORTED_SCHEMA_VERSIONS



# --------------------------------------------------------------------------- #
# Corruption handling during load
# --------------------------------------------------------------------------- #


class TestCorruptionHandling:
    def test_corrupted_meta_json_returns_none(self, cache, payload, fixed_time) -> None:
        cache.store_raw("/bootstrap-static/", payload, source="fpl-api", url="u", retrieved_at=fixed_time)
        meta_files = list(Path(cache._settings.raw_data_dir).rglob("*.meta.json"))
        meta_files[0].write_text("{ INVALID JSON", encoding="utf-8")
        assert cache.get_cached("/bootstrap-static/") is None

    def test_sha_mismatch_returns_none(self, cache, payload, fixed_time) -> None:
        cache.store_raw("/bootstrap-static/", payload, source="fpl-api", url="u", retrieved_at=fixed_time)
        json_files = [f for f in Path(cache._settings.raw_data_dir).rglob("*.json") if not f.name.endswith(".meta.json")]
        json_files[0].write_bytes(b"TAMPERED")
        assert cache.get_cached("/bootstrap-static/") is None

    def test_truncated_json_returns_none(self, cache, payload, fixed_time) -> None:
        cache.store_raw("/bootstrap-static/", payload, source="fpl-api", url="u", retrieved_at=fixed_time)
        json_files = [f for f in Path(cache._settings.raw_data_dir).rglob("*.json") if not f.name.endswith(".meta.json")]
        raw = json_files[0].read_bytes()
        json_files[0].write_bytes(raw[: len(raw) // 2])
        assert cache.get_cached("/bootstrap-static/") is None

    def test_missing_companion_json_returns_none(self, cache, payload, fixed_time) -> None:
        cache.store_raw("/bootstrap-static/", payload, source="fpl-api", url="u", retrieved_at=fixed_time)
        json_files = [f for f in Path(cache._settings.raw_data_dir).rglob("*.json") if not f.name.endswith(".meta.json")]
        json_files[0].unlink()
        assert cache.get_cached("/bootstrap-static/") is None

    def test_missing_companion_meta_returns_none(self, cache, payload, fixed_time) -> None:
        cache.store_raw("/bootstrap-static/", payload, source="fpl-api", url="u", retrieved_at=fixed_time)
        meta_files = list(Path(cache._settings.raw_data_dir).rglob("*.meta.json"))
        meta_files[0].unlink()
        assert cache.get_cached("/bootstrap-static/") is None

    def test_unsupported_schema_version_skipped_on_load(self, cache, payload, fixed_time) -> None:
        cache.store_raw("/bootstrap-static/", payload, source="fpl-api", url="u", retrieved_at=fixed_time)
        meta_files = list(Path(cache._settings.raw_data_dir).rglob("*.meta.json"))
        meta_dict = json.loads(meta_files[0].read_text(encoding="utf-8"))
        meta_dict["schema_version"] = "99"
        meta_files[0].write_text(json.dumps(meta_dict), encoding="utf-8")
        assert cache.get_cached("/bootstrap-static/") is None

    def test_content_length_mismatch_returns_none(self, cache, payload, fixed_time) -> None:
        cache.store_raw("/bootstrap-static/", payload, source="fpl-api", url="u", retrieved_at=fixed_time)
        meta_files = list(Path(cache._settings.raw_data_dir).rglob("*.meta.json"))
        meta_dict = json.loads(meta_files[0].read_text(encoding="utf-8"))
        meta_dict["content_length"] = 99999
        meta_files[0].write_text(json.dumps(meta_dict), encoding="utf-8")
        assert cache.get_cached("/bootstrap-static/") is None

    def test_corrupt_files_do_not_crash(self, cache, payload, fixed_time) -> None:
        cache.store_raw("/bootstrap-static/", payload, source="fpl-api", url="u", retrieved_at=fixed_time)
        dir_path = Path(cache._settings.raw_data_dir) / "bootstrap-static"
        (dir_path / "garbage.meta.json").write_text("not json", encoding="utf-8")
        snap = cache.get_cached("/bootstrap-static/")
        assert snap is not None
        assert snap.payload == payload



# --------------------------------------------------------------------------- #
# Multi-snapshot selection
# --------------------------------------------------------------------------- #


class TestSnapshotSelection:
    def test_returns_newest_valid_snapshot(self, cache) -> None:
        t1 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        t2 = datetime(2024, 1, 2, 0, 0, 0, tzinfo=UTC)
        cache.store_raw("/bootstrap-static/", {"a": 1}, source="fpl-api", url="u", retrieved_at=t1)
        cache.store_raw("/bootstrap-static/", {"b": 2}, source="fpl-api", url="u", retrieved_at=t2)
        snap = cache.get_cached_any("/bootstrap-static/")
        assert snap is not None
        assert snap.payload == {"b": 2}
        assert snap.metadata.retrieved_at == t2

    def test_falls_back_to_older_when_newer_corrupt(self, cache) -> None:
        t1 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        t2 = datetime(2024, 1, 2, 0, 0, 0, tzinfo=UTC)
        cache.store_raw("/bootstrap-static/", {"older": True}, source="fpl-api", url="u", retrieved_at=t1)
        cache.store_raw("/bootstrap-static/", {"newer": True}, source="fpl-api", url="u", retrieved_at=t2)
        dir_path = Path(cache._settings.raw_data_dir) / "bootstrap-static"
        for mf in sorted(dir_path.glob("*.meta.json")):
            meta = json.loads(mf.read_text())
            if meta["retrieved_at"].startswith("2024-01-02"):
                data_file = mf.with_name(mf.name.removesuffix(".meta.json") + ".json")
                data_file.write_bytes(b"CORRUPT")
        snap = cache.get_cached_any("/bootstrap-static/")
        assert snap is not None
        assert snap.payload == {"older": True}

    def test_identical_payload_same_timestamp_overwrites(self, cache, payload, fixed_time) -> None:
        cache.store_raw("/bootstrap-static/", payload, source="fpl-api", url="u", retrieved_at=fixed_time)
        cache.store_raw("/bootstrap-static/", payload, source="fpl-api", url="u", retrieved_at=fixed_time)
        json_files = [f for f in Path(cache._settings.raw_data_dir).rglob("*.json") if not f.name.endswith(".meta.json")]
        meta_files = list(Path(cache._settings.raw_data_dir).rglob("*.meta.json"))
        assert len(json_files) == 1
        assert len(meta_files) == 1

    def test_different_payloads_same_timestamp_different_files(self, cache, fixed_time) -> None:
        cache.store_raw("/bootstrap-static/", {"a": 1}, source="fpl-api", url="u", retrieved_at=fixed_time)
        cache.store_raw("/bootstrap-static/", {"b": 2}, source="fpl-api", url="u", retrieved_at=fixed_time)
        json_files = [f for f in Path(cache._settings.raw_data_dir).rglob("*.json") if not f.name.endswith(".meta.json")]
        assert len(json_files) == 2


# --------------------------------------------------------------------------- #
# Endpoint sanitization
# --------------------------------------------------------------------------- #


class TestEndpointSanitization:
    def test_multiple_endpoints_separate_dirs(self, cache, payload, fixed_time) -> None:
        cache.store_raw("/bootstrap-static/", payload, source="fpl-api", url="u", retrieved_at=fixed_time)
        cache.store_raw("/players/", {"id": 1}, source="fpl-api", url="u2", retrieved_at=fixed_time)
        dirs = [d for d in Path(cache._settings.raw_data_dir).iterdir() if d.is_dir()]
        dir_names = [d.name for d in dirs]
        assert "bootstrap-static" in dir_names
        assert "players" in dir_names

    def test_nested_endpoint_sanitized(self, cache, payload, fixed_time) -> None:
        cache.store_raw("/players/1/", payload, source="fpl-api", url="u", retrieved_at=fixed_time)
        dirs = [d for d in Path(cache._settings.raw_data_dir).iterdir() if d.is_dir()]
        assert any(d.name == "players_1" for d in dirs)


# --------------------------------------------------------------------------- #
# Write error handling
# --------------------------------------------------------------------------- #


class TestWriteErrors:
    @pytest.mark.skipif(sys.platform == "win32", reason="chmod does not enforce directory permissions on Windows")
    def test_write_to_readonly_dir_raises_cache_error(self, payload, fixed_time) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            cache = SnapshotCache(Settings(raw_data_dir=tmp_dir))
            ep_dir = tmp_dir / "bootstrap-static"
            ep_dir.mkdir()
            ep_dir.chmod(0o555)
            try:
                with pytest.raises(CacheError):
                    cache.store_raw(
                        "/bootstrap-static/", payload,
                        source="fpl-api", url="u",
                        retrieved_at=fixed_time,
                    )
            finally:
                ep_dir.chmod(0o755)

