"""Historical data collection orchestration.

This module provides :class:`HistoricalDataCollector`, a small orchestrator
that wraps a raw-API fetcher and a
:class:`~fpl_engine.storage.historical.HistoricalSnapshotStore` to collect and
persist immutable historical snapshots for backtesting.

The collector is **not** an always-running scheduler.  It is called
explicitly (manually or by an external scheduler) so that callers control
when collection happens and what player IDs are fetched.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..logging import get_logger
from ..storage.exceptions import CacheError
from ..storage.historical import HistoricalSnapshot, HistoricalSnapshotStore

log = get_logger(__name__)

__all__ = [
    "HistoricalCollectionError",
    "HistoricalCollectionResult",
    "HistoricalDataCollector",
    "PlayerCollectionResult",
]

#: Default base URL for the FPL API (matches :class:`~fpl_engine.FPLClient`).
_DEFAULT_BASE_URL = "https://fantasy.premierleague.com/api"

#: Endpoint paths accepted by the collector.
_BOOTSTRAP_ENDPOINT = "/bootstrap-static/"
_FIXTURES_ENDPOINT = "/fixtures/"

#: Type alias for a callable that fetches raw JSON for an endpoint path.
RawFetcher = Callable[[str], Any]


class HistoricalCollectionError(Exception):
    """Raised when a historical collection step cannot complete."""


@dataclass(frozen=True)
class HistoricalCollectionResult:
    """Outcome of a single-endpoint collection attempt.

    Attributes
    ----------
    endpoint
        The raw endpoint path, e.g. ``"/bootstrap-static/"``.
    gameweek
        The gameweek tag supplied by the caller.
    snapshot
        The stored snapshot, or ``None`` if collection failed.
    skipped
        ``True`` if an existing snapshot for this gameweek -- collected at
        or before the idempotency cutoff (the gameweek ``deadline_time``
        if supplied, otherwise ``retrieved_at``) -- was reused without
        making a new API request.
    error
        Error message if collection failed; ``None`` on success.
    """

    endpoint: str
    gameweek: int
    snapshot: HistoricalSnapshot | None
    skipped: bool
    error: str | None = None


@dataclass(frozen=True)
class PlayerCollectionResult:
    """Outcome of collecting a single player's element-summary.

    Attributes
    ----------
    player_id
        The FPL player (element) ID.
    gameweek
        The gameweek tag supplied by the caller.
    snapshot
        The stored snapshot, or ``None`` if collection failed.
    skipped
        ``True`` if an existing snapshot for this player + gameweek --
        collected at or before the idempotency cutoff (the gameweek
        ``deadline_time`` if supplied, otherwise ``retrieved_at``) --
        was reused without an API call.
    error
        Error message if collection failed; ``None`` on success.
    """

    player_id: int
    gameweek: int
    snapshot: HistoricalSnapshot | None
    skipped: bool
    error: str | None = None


class HistoricalDataCollector:
    """Orchestrates historical snapshot collection for backtesting.

    The collector wraps:

    * A *raw-API fetcher* -- a callable that accepts an endpoint path string
      and returns the raw JSON ``dict``.  In production this is typically
      ``FPLClient._request`` (which already includes retry/backoff); in tests
      it is a ``Mock`` or stub.
    * A :class:`~fpl_engine.storage.historical.HistoricalSnapshotStore`
      for immutable, integrity-verified persistence.

    Collection is **explicit** -- never automatic.  The collector is safe to
    rerun: if a snapshot already exists for an endpoint + gameweek, the
    fetch is skipped and the existing snapshot is returned (``skipped=True``).
    One failed player request does not abort the entire collection.
    """

    def __init__(
        self,
        fetch_raw: RawFetcher,
        store: HistoricalSnapshotStore,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        source: str = "fpl-api",
    ) -> None:
        self._fetch_raw = fetch_raw
        self._store = store
        self._base_url = base_url
        self._source = source

    @property
    def store(self) -> HistoricalSnapshotStore:
        """The underlying snapshot store."""
        return self._store

    def _url_for(self, endpoint: str) -> str:
        """Construct the fully-qualified URL for *endpoint*."""
        return f"{self._base_url}{endpoint}"

    def collect_bootstrap(
        self,
        gameweek: int,
        *,
        deadline_time: datetime | None = None,
        retrieved_at: datetime | None = None,
        force: bool = False,
    ) -> HistoricalCollectionResult:
        """Collect and store a ``/bootstrap-static/`` snapshot.

        Parameters
        ----------
        gameweek
            FPL gameweek tag to associate with the snapshot.
        deadline_time
            Optional official gameweek deadline.  Stored as-is; the
            collector never infers the deadline from the API
            response.
        retrieved_at
            Explicit collection timestamp.  Defaults to ``datetime.now(UTC)``.
        force
            If ``True``, bypass the idempotency check and always fetch.

        Returns
        -------
        HistoricalCollectionResult
            Result with ``skipped=True`` if a snapshot already existed.

        Raises
        ------
        HistoricalCollectionError
            If the API request fails or the snapshot cannot be persisted.
        """
        if retrieved_at is None:
            retrieved_at = datetime.now(UTC)

        cutoff = self._cutoff(deadline_time, retrieved_at)

        if not force and self._store.has_snapshot(
            _BOOTSTRAP_ENDPOINT, gameweek=gameweek, before=cutoff
        ):
            snap = self._store.get(_BOOTSTRAP_ENDPOINT, gameweek=gameweek, before=cutoff)
            log.info(
                "Skipping bootstrap snapshot (already collected)",
                gameweek=gameweek,
            )
            assert snap is not None
            return HistoricalCollectionResult(
                endpoint=_BOOTSTRAP_ENDPOINT,
                gameweek=gameweek,
                snapshot=snap,
                skipped=True,
            )

        try:
            payload = self._fetch_raw(_BOOTSTRAP_ENDPOINT)
        except Exception as exc:
            msg = f"Failed to fetch bootstrap-static: {exc}"
            log.exception(msg, endpoint=_BOOTSTRAP_ENDPOINT)
            raise HistoricalCollectionError(msg) from exc

        try:
            snap = self._store.store(
                _BOOTSTRAP_ENDPOINT,
                payload,
                source=self._source,
                url=self._url_for(_BOOTSTRAP_ENDPOINT),
                retrieved_at=retrieved_at,
                gameweek=gameweek,
                deadline_time=deadline_time,
            )
        except (CacheError, ValueError) as exc:
            msg = f"Failed to persist bootstrap snapshot: {exc}"
            log.exception(msg, endpoint=_BOOTSTRAP_ENDPOINT)
            raise HistoricalCollectionError(msg) from exc

        return HistoricalCollectionResult(
            endpoint=_BOOTSTRAP_ENDPOINT,
            gameweek=gameweek,
            snapshot=snap,
            skipped=False,
        )

    def collect_fixtures(
        self,
        gameweek: int,
        *,
        deadline_time: datetime | None = None,
        retrieved_at: datetime | None = None,
        force: bool = False,
    ) -> HistoricalCollectionResult:
        """Collect and store a ``/fixtures/`` snapshot.

        Parameters and return value follow the same conventions as
        :meth:`collect_bootstrap`.
        """
        if retrieved_at is None:
            retrieved_at = datetime.now(UTC)

        cutoff = self._cutoff(deadline_time, retrieved_at)

        if not force and self._store.has_snapshot(
            _FIXTURES_ENDPOINT, gameweek=gameweek, before=cutoff
        ):
            snap = self._store.get(_FIXTURES_ENDPOINT, gameweek=gameweek, before=cutoff)
            log.info(
                "Skipping fixtures snapshot (already collected)",
                gameweek=gameweek,
            )
            assert snap is not None
            return HistoricalCollectionResult(
                endpoint=_FIXTURES_ENDPOINT,
                gameweek=gameweek,
                snapshot=snap,
                skipped=True,
            )

        try:
            payload = self._fetch_raw(_FIXTURES_ENDPOINT)
        except Exception as exc:
            msg = f"Failed to fetch fixtures: {exc}"
            log.exception(msg, endpoint=_FIXTURES_ENDPOINT)
            raise HistoricalCollectionError(msg) from exc

        try:
            snap = self._store.store(
                _FIXTURES_ENDPOINT,
                payload,
                source=self._source,
                url=self._url_for(_FIXTURES_ENDPOINT),
                retrieved_at=retrieved_at,
                gameweek=gameweek,
                deadline_time=deadline_time,
            )
        except (CacheError, ValueError) as exc:
            msg = f"Failed to persist fixtures snapshot: {exc}"
            log.exception(msg, endpoint=_FIXTURES_ENDPOINT)
            raise HistoricalCollectionError(msg) from exc

        return HistoricalCollectionResult(
            endpoint=_FIXTURES_ENDPOINT,
            gameweek=gameweek,
            snapshot=snap,
            skipped=False,
        )

    @staticmethod
    def _element_summary_endpoint(player_id: int) -> str:
        """Return the API endpoint path for a single player's summary."""
        return f"/element-summary/{player_id}/"

    @staticmethod
    def _cutoff(
        deadline_time: datetime | None, retrieved_at: datetime
    ) -> datetime:
        """Idempotency cutoff for the collection attempt.

        The cutoff is the latest timestamp a reusable snapshot may have been
        collected.  It is the earlier of:

        * ``retrieved_at`` -- the current collection attempt's timestamp
          (so we never reuse a snapshot collected *after* the attempt).
        * ``deadline_time`` -- the official gameweek deadline (so we never
          reuse a post-deadline snapshot for pre-deadline predictions).

        When ``deadline_time`` is ``None``, the cutoff is simply
        ``retrieved_at``.
        """
        if deadline_time is not None:
            return min(retrieved_at, deadline_time)
        return retrieved_at

    def collect_element_summary(
        self,
        player_id: int,
        gameweek: int,
        *,
        deadline_time: datetime | None = None,
        retrieved_at: datetime | None = None,
        force: bool = False,
    ) -> PlayerCollectionResult:
        """Collect and store an ``/element-summary/{player_id}/`` snapshot.

        Parameters
        ----------
        player_id
            FPL player (element) ID whose summary to fetch.
        gameweek
            FPL gameweek tag to associate with the snapshot.
        deadline_time
            Optional official gameweek deadline.
        retrieved_at
            Explicit collection timestamp; defaults to ``datetime.now(UTC)``.
        force
            If ``True``, bypass the idempotency check.

        Returns
        -------
        PlayerCollectionResult
            Result with ``skipped=True`` if a snapshot already existed.
        """
        if retrieved_at is None:
            retrieved_at = datetime.now(UTC)

        endpoint = self._element_summary_endpoint(player_id)
        cutoff = self._cutoff(deadline_time, retrieved_at)

        if not force and self._store.has_snapshot(
            endpoint, gameweek=gameweek, before=cutoff
        ):
            snap = self._store.get(endpoint, gameweek=gameweek, before=cutoff)
            log.info(
                "Skipping element-summary (already collected)",
                player_id=player_id,
                gameweek=gameweek,
            )
            assert snap is not None
            return PlayerCollectionResult(
                player_id=player_id,
                gameweek=gameweek,
                snapshot=snap,
                skipped=True,
            )

        try:
            payload = self._fetch_raw(endpoint)
        except Exception as exc:
            msg = f"Failed to fetch element-summary for player {player_id}: {exc}"
            log.exception(msg, player_id=player_id, endpoint=endpoint)
            return PlayerCollectionResult(
                player_id=player_id,
                gameweek=gameweek,
                snapshot=None,
                skipped=False,
                error=msg,
            )

        try:
            snap = self._store.store(
                endpoint,
                payload,
                source=self._source,
                url=self._url_for(endpoint),
                retrieved_at=retrieved_at,
                gameweek=gameweek,
                deadline_time=deadline_time,
            )
        except (CacheError, ValueError) as exc:
            msg = f"Failed to persist element-summary for player {player_id}: {exc}"
            log.exception(msg, player_id=player_id, endpoint=endpoint)
            return PlayerCollectionResult(
                player_id=player_id,
                gameweek=gameweek,
                snapshot=None,
                skipped=False,
                error=msg,
            )

        return PlayerCollectionResult(
            player_id=player_id,
            gameweek=gameweek,
            snapshot=snap,
            skipped=False,
        )

    def collect_element_summaries(
        self,
        player_ids: list[int],
        gameweek: int,
        *,
        deadline_time: datetime | None = None,
        retrieved_at: datetime | None = None,
        force: bool = False,
    ) -> list[PlayerCollectionResult]:
        """Collect element-summary snapshots for multiple players.

        Iterates over *player_ids* sequentially (no uncontrolled concurrency).
        Each player is handled independently: a failure for one player is
        recorded in its :class:`PlayerCollectionResult` and does not abort the
        collection for remaining players.

        Parameters
        ----------
        player_ids
            List of FPL player (element) IDs to fetch.
        gameweek
            FPL gameweek tag applied to all snapshots.
        deadline_time
            Optional official gameweek deadline.
        retrieved_at
            Explicit collection timestamp; defaults to ``datetime.now(UTC)``.
        force
            If ``True``, bypass idempotency checks for all players.

        Returns
        -------
        list[PlayerCollectionResult]
            One entry per input player ID, in the same order.
        """
        return [
            self.collect_element_summary(
                pid,
                gameweek,
                deadline_time=deadline_time,
                retrieved_at=retrieved_at,
                force=force,
            )
            for pid in player_ids
        ]
