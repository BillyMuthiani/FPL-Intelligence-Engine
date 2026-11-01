"""Tests for the historical data collector (offline, no live API).

All tests use a ``Mock`` raw-fetcher and a real ``HistoricalSnapshotStore``
backed by a temporary directory.  No network access is required.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from fpl_engine.config import Settings
from fpl_engine.ingestion.historical import (
    HistoricalCollectionError,
    HistoricalDataCollector,
)
from fpl_engine.storage.historical import HistoricalSnapshotStore

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

_FIXED_TIME = datetime(2024, 9, 13, 11, 0, 0, tzinfo=UTC)
_GW5_DEADLINE = datetime(2024, 9, 13, 17, 30, 0, tzinfo=UTC)
_POST_DEADLINE_TIME = datetime(2024, 9, 13, 20, 0, 0, tzinfo=UTC)

# 08:00 / 10:00 / 11:00 scenario for collection-time safety audit:
_ATTEMPT_08 = datetime(2024, 9, 13, 8, 0, 0, tzinfo=UTC)   # 08:00 — current attempt
_SNAPSHOT_10 = datetime(2024, 9, 13, 10, 0, 0, tzinfo=UTC)  # 10:00 — existing snapshot (after attempt)
# _FIXED_TIME (11:00) serves as the deadline_time in these tests


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(raw_data_dir=tmp_path)


@pytest.fixture
def store(settings: Settings) -> HistoricalSnapshotStore:
    return HistoricalSnapshotStore(settings)


@pytest.fixture
def bootstrap_payload() -> dict:
    return {"elements": [{"id": 1}], "teams": [], "events": []}


@pytest.fixture
def fixtures_payload() -> list:
    return [{"code": 1, "event": 5, "team_h": 1, "team_a": 2}]


@pytest.fixture
def element_summary_payload() -> dict:
    return {"history": [{"fixture": 1, "round": 5, "total_points": 6}], "fixtures": []}


# --------------------------------------------------------------------------- #
# collect_bootstrap
# --------------------------------------------------------------------------- #


class TestCollectBootstrap:
    def test_collect_success(
        self, store, bootstrap_payload
    ) -> None:
        fetch = Mock(return_value=bootstrap_payload)
        collector = HistoricalDataCollector(fetch, store)
        result = collector.collect_bootstrap(gameweek=5, retrieved_at=_FIXED_TIME)
        assert result.snapshot is not None
        assert result.snapshot.payload == bootstrap_payload
        assert result.snapshot.metadata.endpoint == "/bootstrap-static/"
        assert result.snapshot.metadata.gameweek == 5
        assert result.skipped is False
        assert result.error is None
        fetch.assert_called_once_with("/bootstrap-static/")

    def test_collect_idempotent_rerun(
        self, store, bootstrap_payload
    ) -> None:
        fetch = Mock(return_value=bootstrap_payload)
        collector = HistoricalDataCollector(fetch, store)
        result1 = collector.collect_bootstrap(gameweek=5, retrieved_at=_FIXED_TIME)
        result2 = collector.collect_bootstrap(gameweek=5, retrieved_at=_FIXED_TIME)
        assert result1.skipped is False
        assert result2.skipped is True
        assert result2.snapshot is not None
        assert result2.snapshot.metadata.gameweek == 5
        # Second call should NOT have re-fetched
        fetch.assert_called_once_with("/bootstrap-static/")

    def test_collect_force_bypasses_idempotency(
        self, store, bootstrap_payload
    ) -> None:
        fetch = Mock(return_value=bootstrap_payload)
        collector = HistoricalDataCollector(fetch, store)
        collector.collect_bootstrap(gameweek=5, retrieved_at=_FIXED_TIME)
        result = collector.collect_bootstrap(gameweek=5, retrieved_at=_FIXED_TIME, force=True)
        assert result.skipped is False
        assert result.snapshot is not None
        # Should have fetched twice
        assert fetch.call_count == 2

    def test_collect_api_failure_raises(
        self, store
    ) -> None:
        fetch = Mock(side_effect=RuntimeError("API down"))
        collector = HistoricalDataCollector(fetch, store)
        with pytest.raises(HistoricalCollectionError, match="Failed to fetch"):
            collector.collect_bootstrap(gameweek=5, retrieved_at=_FIXED_TIME)

    def test_collect_stores_deadline_time(
        self, store, bootstrap_payload
    ) -> None:
        fetch = Mock(return_value=bootstrap_payload)
        collector = HistoricalDataCollector(fetch, store)
        result = collector.collect_bootstrap(
            gameweek=5,
            deadline_time=_GW5_DEADLINE,
            retrieved_at=_FIXED_TIME,
        )
        assert result.snapshot is not None
        assert result.snapshot.metadata.deadline_time == _GW5_DEADLINE

    def test_collect_without_deadline_time(
        self, store, bootstrap_payload
    ) -> None:
        fetch = Mock(return_value=bootstrap_payload)
        collector = HistoricalDataCollector(fetch, store)
        result = collector.collect_bootstrap(gameweek=5, retrieved_at=_FIXED_TIME)
        assert result.snapshot is not None
        assert result.snapshot.metadata.deadline_time is None

    def test_collect_no_snapshot_written_on_api_failure(
        self, store
    ) -> None:
        fetch = Mock(side_effect=RuntimeError("API down"))
        collector = HistoricalDataCollector(fetch, store)
        with pytest.raises(HistoricalCollectionError):
            collector.collect_bootstrap(gameweek=5, retrieved_at=_FIXED_TIME)
        # No snapshot should have been persisted
        assert store.get("/bootstrap-static/", gameweek=5) is None


# --------------------------------------------------------------------------- #
# collect_fixtures
# --------------------------------------------------------------------------- #


class TestCollectFixtures:
    def test_collect_fixtures_success(self, store, fixtures_payload) -> None:
        fetch = Mock(return_value=fixtures_payload)
        collector = HistoricalDataCollector(fetch, store)
        result = collector.collect_fixtures(gameweek=5, retrieved_at=_FIXED_TIME)
        assert result.snapshot is not None
        assert result.snapshot.payload == fixtures_payload
        assert result.snapshot.metadata.endpoint == "/fixtures/"
        assert result.snapshot.metadata.gameweek == 5
        assert result.skipped is False
        assert result.error is None
        fetch.assert_called_once_with("/fixtures/")

    def test_collect_fixtures_idempotent(self, store, fixtures_payload) -> None:
        fetch = Mock(return_value=fixtures_payload)
        collector = HistoricalDataCollector(fetch, store)
        collector.collect_fixtures(gameweek=5, retrieved_at=_FIXED_TIME)
        result = collector.collect_fixtures(gameweek=5, retrieved_at=_FIXED_TIME)
        assert result.skipped is True
        assert result.snapshot is not None
        fetch.assert_called_once_with("/fixtures/")

    def test_collect_fixtures_api_failure_raises(self, store) -> None:
        fetch = Mock(side_effect=ConnectionError("timeout"))
        collector = HistoricalDataCollector(fetch, store)
        with pytest.raises(HistoricalCollectionError, match="Failed to fetch fixtures"):
            collector.collect_fixtures(gameweek=5, retrieved_at=_FIXED_TIME)


# --------------------------------------------------------------------------- #
# collect_element_summary / collect_element_summaries
# --------------------------------------------------------------------------- #


class TestCollectElementSummary:
    def test_collect_element_summary_success(
        self, store, element_summary_payload
    ) -> None:
        fetch = Mock(return_value=element_summary_payload)
        collector = HistoricalDataCollector(fetch, store)
        result = collector.collect_element_summary(
            42, gameweek=5, retrieved_at=_FIXED_TIME
        )
        assert result.player_id == 42
        assert result.gameweek == 5
        assert result.snapshot is not None
        assert result.snapshot.payload == element_summary_payload
        assert result.snapshot.metadata.gameweek == 5
        assert result.skipped is False
        assert result.error is None
        fetch.assert_called_once_with("/element-summary/42/")

    def test_collect_element_summary_idempotent(
        self, store, element_summary_payload
    ) -> None:
        fetch = Mock(return_value=element_summary_payload)
        collector = HistoricalDataCollector(fetch, store)
        collector.collect_element_summary(42, gameweek=5, retrieved_at=_FIXED_TIME)
        result = collector.collect_element_summary(
            42, gameweek=5, retrieved_at=_FIXED_TIME
        )
        assert result.skipped is True
        assert result.snapshot is not None
        fetch.assert_called_once_with("/element-summary/42/")

    def test_collect_element_summary_api_failure_returns_error(
        self, store
    ) -> None:
        """A player fetch failure is reported, not raised."""
        fetch = Mock(side_effect=ConnectionError("503"))
        collector = HistoricalDataCollector(fetch, store)
        result = collector.collect_element_summary(
            99, gameweek=5, retrieved_at=_FIXED_TIME
        )
        assert result.snapshot is None
        assert result.skipped is False
        assert result.error is not None
        assert "Failed to fetch" in result.error


class TestCollectElementSummaries:
    def test_all_players_succeed(self, store, element_summary_payload) -> None:
        fetch = Mock(return_value=element_summary_payload)
        collector = HistoricalDataCollector(fetch, store)
        results = collector.collect_element_summaries(
            [1, 2, 3], gameweek=5, retrieved_at=_FIXED_TIME
        )
        assert len(results) == 3
        assert all(r.snapshot is not None for r in results)
        assert all(r.skipped is False for r in results)
        assert all(r.error is None for r in results)
        assert fetch.call_count == 3

    def test_partial_failure_one_player_fails(self, store) -> None:
        """One failed player request must not silently imply full success."""
        good_payload = {"history": [], "fixtures": []}
        fetch = Mock(return_value=good_payload)
        fetch.side_effect = [good_payload, ConnectionError("503"), good_payload]
        collector = HistoricalDataCollector(fetch, store)
        results = collector.collect_element_summaries(
            [10, 20, 30], gameweek=5, retrieved_at=_FIXED_TIME
        )
        assert len(results) == 3
        # Player 10 succeeded
        assert results[0].player_id == 10
        assert results[0].snapshot is not None
        assert results[0].error is None
        # Player 20 failed
        assert results[1].player_id == 20
        assert results[1].snapshot is None
        assert results[1].error is not None
        # Player 30 succeeded
        assert results[2].player_id == 30
        assert results[2].snapshot is not None
        assert results[2].error is None
        # Verify we can distinguish success from failure
        succeeded = [r for r in results if r.snapshot is not None]
        failed = [r for r in results if r.snapshot is None and r.error is not None]
        assert len(succeeded) == 2
        assert len(failed) == 1
        assert failed[0].player_id == 20

    def test_idempotent_rerun_skips_all(
        self, store, element_summary_payload
    ) -> None:
        fetch = Mock(return_value=element_summary_payload)
        collector = HistoricalDataCollector(fetch, store)
        collector.collect_element_summaries(
            [1, 2], gameweek=5, retrieved_at=_FIXED_TIME
        )
        results = collector.collect_element_summaries(
            [1, 2], gameweek=5, retrieved_at=_FIXED_TIME
        )
        assert len(results) == 2
        assert all(r.skipped is True for r in results)
        assert all(r.snapshot is not None for r in results)
        # No additional API calls
        assert fetch.call_count == 2

    def test_mixed_skipped_and_new(
        self, store, element_summary_payload
    ) -> None:
        fetch = Mock(return_value=element_summary_payload)
        collector = HistoricalDataCollector(fetch, store)
        # First collect for player 1
        collector.collect_element_summary(1, gameweek=5, retrieved_at=_FIXED_TIME)
        # Now collect for players 1 and 2 (1 is skipped, 2 is new)
        results = collector.collect_element_summaries(
            [1, 2], gameweek=5, retrieved_at=_FIXED_TIME
        )
        assert results[0].skipped is True
        assert results[1].skipped is False
        # Only one NEW fetch within collect_element_summaries (player 2)
        # Total calls = 1 (initial) + 1 (player 2) = 2
        assert fetch.call_count == 2
        assert fetch.call_args_list[-1].args == ("/element-summary/2/",)


# --------------------------------------------------------------------------- #
# Deadline-aware idempotency: collection is only skipped when a pre-deadline
# snapshot exists.  A snapshot collected *after* the deadline must not prevent
# a fresh pre-deadline collection.
# --------------------------------------------------------------------------- #

class TestDeadlineAwareIdempotency:
    """The collector must not skip collection merely because *any* snapshot
    exists for a gameweek -- it must check that one was collected before the
    deadline/cutoff."""

    def test_skip_when_pre_deadline_snapshot_exists(
        self, store, bootstrap_payload
    ) -> None:
        """A pre-deadline snapshot exists; rerun with same deadline is skipped."""
        fetch = Mock(return_value=bootstrap_payload)
        collector = HistoricalDataCollector(fetch, store)
        collector.collect_bootstrap(
            gameweek=5, retrieved_at=_FIXED_TIME, deadline_time=_GW5_DEADLINE
        )
        result = collector.collect_bootstrap(
            gameweek=5, retrieved_at=_POST_DEADLINE_TIME, deadline_time=_GW5_DEADLINE
        )
        assert result.skipped is True
        assert result.snapshot is not None
        assert result.snapshot.metadata.retrieved_at == _FIXED_TIME
        fetch.assert_called_once_with("/bootstrap-static/")

    def test_not_skipped_when_only_post_deadline_exists(
        self, store, bootstrap_payload
    ) -> None:
        """Only a post-deadline snapshot exists; rerun with deadline is NOT skipped.

        This is the key regression test for the incorrect-idempotency-skip bug:
        previously the collector would skip because ``has_snapshot`` found *any*
        GW5 snapshot, but the only one was collected after the deadline.
        """
        fetch = Mock(return_value=bootstrap_payload)
        collector = HistoricalDataCollector(fetch, store)
        collector.collect_bootstrap(
            gameweek=5, retrieved_at=_POST_DEADLINE_TIME, deadline_time=None
        )
        result = collector.collect_bootstrap(
            gameweek=5, retrieved_at=_POST_DEADLINE_TIME, deadline_time=_GW5_DEADLINE
        )
        assert result.skipped is False
        assert result.snapshot is not None
        assert result.snapshot.metadata.retrieved_at == _POST_DEADLINE_TIME
        assert fetch.call_count == 2

    def test_force_bypasses_deadline_check(
        self, store, bootstrap_payload
    ) -> None:
        """force=True bypasses the deadline-aware skip."""
        fetch = Mock(return_value=bootstrap_payload)
        collector = HistoricalDataCollector(fetch, store)
        collector.collect_bootstrap(
            gameweek=5, retrieved_at=_FIXED_TIME, deadline_time=_GW5_DEADLINE
        )
        result = collector.collect_bootstrap(
            gameweek=5, retrieved_at=_POST_DEADLINE_TIME,
            deadline_time=_GW5_DEADLINE, force=True,
        )
        assert result.skipped is False
        assert result.snapshot is not None
        assert fetch.call_count == 2

    def test_force_does_not_destroy_existing_snapshot(
        self, store, bootstrap_payload
    ) -> None:
        """Forced collection creates a new snapshot; the old one is retained."""
        fetch = Mock(return_value=bootstrap_payload)
        collector = HistoricalDataCollector(fetch, store)
        collector.collect_bootstrap(
            gameweek=5, retrieved_at=_FIXED_TIME, deadline_time=_GW5_DEADLINE
        )
        collector.collect_bootstrap(
            gameweek=5, retrieved_at=_POST_DEADLINE_TIME,
            deadline_time=_GW5_DEADLINE, force=True,
        )
        all_snaps = store.get_all("/bootstrap-static/")
        assert len(all_snaps) == 2

    def test_rerun_without_deadline_then_with_deadline(
        self, store, bootstrap_payload
    ) -> None:
        """A snapshot stored without deadline_time metadata; rerun with deadline.

        The pre-deadline snapshot is found via the ``before`` cutoff, so the
        rerun is correctly skipped.
        """
        fetch = Mock(return_value=bootstrap_payload)
        collector = HistoricalDataCollector(fetch, store)
        collector.collect_bootstrap(
            gameweek=5, retrieved_at=_FIXED_TIME
        )
        result = collector.collect_bootstrap(
            gameweek=5, retrieved_at=_POST_DEADLINE_TIME,
            deadline_time=_GW5_DEADLINE,
        )
        assert result.skipped is True
        assert result.snapshot is not None
        assert result.snapshot.metadata.retrieved_at == _FIXED_TIME
        fetch.assert_called_once_with("/bootstrap-static/")

    def test_no_deadline_falls_back_to_retrieved_at(
        self, store, bootstrap_payload
    ) -> None:
        """Without deadline_time, the idempotency cutoff is retrieved_at."""
        fetch = Mock(return_value=bootstrap_payload)
        collector = HistoricalDataCollector(fetch, store)
        collector.collect_bootstrap(gameweek=5, retrieved_at=_FIXED_TIME)
        result = collector.collect_bootstrap(gameweek=5, retrieved_at=_FIXED_TIME)
        assert result.skipped is True
        assert fetch.call_count == 1

    def test_element_summary_deadline_aware_skip(
        self, store, element_summary_payload
    ) -> None:
        """element-summary idempotency also respects the deadline cutoff."""
        fetch = Mock(return_value=element_summary_payload)
        collector = HistoricalDataCollector(fetch, store)
        collector.collect_element_summary(
            1, gameweek=5, retrieved_at=_FIXED_TIME, deadline_time=_GW5_DEADLINE
        )
        result = collector.collect_element_summary(
            1, gameweek=5, retrieved_at=_POST_DEADLINE_TIME, deadline_time=_GW5_DEADLINE
        )
        assert result.skipped is True
        assert result.snapshot is not None
        assert result.snapshot.metadata.retrieved_at == _FIXED_TIME
        fetch.assert_called_once_with("/element-summary/1/")

    def test_element_summary_not_skipped_when_post_deadline_only(
        self, store, element_summary_payload
    ) -> None:
        """element-summary with only a post-deadline snapshot does not skip."""
        fetch = Mock(return_value=element_summary_payload)
        collector = HistoricalDataCollector(fetch, store)
        collector.collect_element_summary(
            50, gameweek=5, retrieved_at=_POST_DEADLINE_TIME
        )
        result = collector.collect_element_summary(
            50, gameweek=5, retrieved_at=_POST_DEADLINE_TIME,
            deadline_time=_GW5_DEADLINE,
        )
        assert result.skipped is False
        assert result.snapshot is not None
        assert fetch.call_count == 2

    def test_multi_player_mixed_deadline_aware(
        self, store, element_summary_payload
    ) -> None:
        """Multi-player collection respects deadline-aware idempotency per player."""
        fetch = Mock(return_value=element_summary_payload)
        collector = HistoricalDataCollector(fetch, store)
        collector.collect_element_summary(
            1, gameweek=5, retrieved_at=_FIXED_TIME, deadline_time=_GW5_DEADLINE
        )
        results = collector.collect_element_summaries(
            [1, 2], gameweek=5,
            retrieved_at=_POST_DEADLINE_TIME, deadline_time=_GW5_DEADLINE,
        )
        assert len(results) == 2
        assert results[0].player_id == 1
        assert results[0].skipped is True
        assert results[1].player_id == 2
        assert results[1].skipped is False
        # fetch called for initial player 1 + player 2 in the rerun;
        # player 1 in the rerun was skipped (pre-deadline snapshot exists)
        assert fetch.call_count == 2
        assert fetch.call_args_list[-1].args == ("/element-summary/2/",)


# --------------------------------------------------------------------------- #
# Collection-time cutoff safety
# (Stage 4I-A.2 regression: cannot reuse a snapshot collected after the
# current collection attempt, even when it was collected before the deadline)
# --------------------------------------------------------------------------- #


class TestCollectionTimeCutoff:
    """Regression: _cutoff must be min(retrieved_at, deadline_time).

    Scenario (the 08:00 / 10:00 / 11:00 bug):

    * Collection attempt at 08:00 UTC.
    * An existing snapshot was retrieved at 10:00 UTC (after the attempt
      but before the 11:00 UTC deadline).
    * The old _cutoff returned deadline_time (11:00) and the 10:00 snapshot
      would pass the before filter, causing an incorrect skip.  The correct
      cutoff is min(08:00, 11:00) = 08:00, which excludes the 10:00 snapshot.
    """

    # ---- collect_bootstrap ----

    def test_bootstrap_not_skipped_when_snapshot_after_attempt(
        self, store, bootstrap_payload
    ) -> None:
        """Snapshot at 10:00, attempt at 08:00, deadline 11:00 - must NOT skip."""
        fetch = Mock(return_value=bootstrap_payload)
        collector = HistoricalDataCollector(fetch, store)
        # Pre-populate a snapshot retrieved at 10:00 (after the 08:00 attempt,
        # before the 11:00 deadline).
        collector.collect_bootstrap(
            gameweek=5, retrieved_at=_SNAPSHOT_10, deadline_time=_FIXED_TIME,
        )
        # Replay: collection at 08:00 with deadline 11:00.
        result = collector.collect_bootstrap(
            gameweek=5, retrieved_at=_ATTEMPT_08, deadline_time=_FIXED_TIME,
        )
        assert result.skipped is False
        assert result.snapshot is not None
        # New snapshot stored with the attempt's timestamp, not the old one.
        assert result.snapshot.metadata.retrieved_at == _ATTEMPT_08
        # API was re-fetched (snapshot was NOT reused).
        assert fetch.call_count == 2
        assert fetch.call_args_list[-1].args == ("/bootstrap-static/",)
        # Old snapshot is still in storage (non-destructive).
        assert len(store.get_all("/bootstrap-static/")) == 2

    def test_bootstrap_skipped_when_snapshot_before_both_attempt_and_deadline(
        self, store, bootstrap_payload
    ) -> None:
        """Snapshot at 07:00, attempt at 08:00, deadline 11:00 - skipped."""
        fetch = Mock(return_value=bootstrap_payload)
        collector = HistoricalDataCollector(fetch, store)
        attempt_minus_hour = datetime(2024, 9, 13, 7, 0, 0, tzinfo=UTC)
        collector.collect_bootstrap(
            gameweek=5, retrieved_at=attempt_minus_hour, deadline_time=_FIXED_TIME,
        )
        result = collector.collect_bootstrap(
            gameweek=5, retrieved_at=_ATTEMPT_08, deadline_time=_FIXED_TIME,
        )
        assert result.skipped is True
        assert result.snapshot is not None
        assert result.snapshot.metadata.retrieved_at == attempt_minus_hour
        assert fetch.call_count == 1

    # ---- collect_fixtures ----

    def test_fixtures_not_skipped_when_snapshot_after_attempt(
        self, store, fixtures_payload
    ) -> None:
        """Same 08:00/10:00/11:00 regression for collect_fixtures."""
        fetch = Mock(return_value=fixtures_payload)
        collector = HistoricalDataCollector(fetch, store)
        collector.collect_fixtures(
            gameweek=5, retrieved_at=_SNAPSHOT_10, deadline_time=_FIXED_TIME,
        )
        result = collector.collect_fixtures(
            gameweek=5, retrieved_at=_ATTEMPT_08, deadline_time=_FIXED_TIME,
        )
        assert result.skipped is False
        assert result.snapshot is not None
        assert result.snapshot.metadata.retrieved_at == _ATTEMPT_08
        assert fetch.call_count == 2
        assert fetch.call_args_list[-1].args == ("/fixtures/",)

    # ---- collect_element_summary ----

    def test_element_summary_not_skipped_when_snapshot_after_attempt(
        self, store, element_summary_payload
    ) -> None:
        """Same 08:00/10:00/11:00 regression for collect_element_summary."""
        fetch = Mock(return_value=element_summary_payload)
        collector = HistoricalDataCollector(fetch, store)
        collector.collect_element_summary(
            1, gameweek=5, retrieved_at=_SNAPSHOT_10, deadline_time=_FIXED_TIME,
        )
        result = collector.collect_element_summary(
            1, gameweek=5, retrieved_at=_ATTEMPT_08, deadline_time=_FIXED_TIME,
        )
        assert result.skipped is False
        assert result.snapshot is not None
        assert result.snapshot.metadata.retrieved_at == _ATTEMPT_08
        assert fetch.call_count == 2
        assert fetch.call_args_list[-1].args == ("/element-summary/1/",)

    def test_element_summary_skipped_when_snapshot_before_attempt_and_deadline(
        self, store, element_summary_payload
    ) -> None:
        """Snapshot before attempt - still correctly skipped."""
        fetch = Mock(return_value=element_summary_payload)
        collector = HistoricalDataCollector(fetch, store)
        collector.collect_element_summary(
            1, gameweek=5, retrieved_at=_SNAPSHOT_10, deadline_time=_FIXED_TIME,
        )
        result = collector.collect_element_summary(
            1, gameweek=5, retrieved_at=_SNAPSHOT_10 + timedelta(hours=1),
            deadline_time=_FIXED_TIME,
        )
        assert result.skipped is True
        assert result.snapshot is not None
        assert result.snapshot.metadata.retrieved_at == _SNAPSHOT_10
        assert fetch.call_count == 1
