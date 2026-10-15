"""Ingestion coordinator: composes provider, cache, and writer into a pipeline.

The :class:`IngestionCoordinator` bridges the three existing layers:

1. **Provider** (:class:`~fpl_engine.providers.protocols.BootstrapDataProvider`)
   — fetches raw bootstrap-static data from the FPL API.
2. **Cache** (:class:`~fpl_engine.storage.cache.SnapshotCache`)
   — persists and retrieves raw-API snapshots with SHA-256 integrity and
   TTL-based freshness.
3. **Normalizers** (:mod:`fpl_engine.ingestion.bootstrap`)
   — pure functions that map the raw payload into canonical models.
4. **Processed writer** (:class:`~fpl_engine.ingestion.processed.ProcessedWriter`)
   — persists the normalized entities as a versioned JSON bundle.

The coordinator decides *when* to use the cache vs. the API, *when* to fall
back to stale data, and *when* to reject the payload entirely.

HTTP-client lifecycle management (opening/closing connections) is the caller's
responsibility — the coordinator only calls ``provider.get_bootstrap_data()``.
"""
from __future__ import annotations

from datetime import UTC, datetime

import httpx

from ..logging import get_logger
from ..providers.protocols import BootstrapDataProvider
from ..storage import SnapshotCache
from ..storage.exceptions import CacheError
from ..storage.models import Snapshot
from .bootstrap import normalize_gameweeks, normalize_players, normalize_teams
from .exceptions import IngestionCoordinatorError, IngestionError
from .models import IngestionResult
from .processed import ProcessedWriter

__all__ = ["IngestionCoordinator"]

log = get_logger(__name__)

#: All httpx exceptions are treated as *expected* transport/HTTP failures
#: eligible for stale-cache fallback.  Programming errors and
#: :class:`IngestionError` / :class:`CacheError` are NOT caught here.
_PROVIDER_ERRORS: tuple[type[BaseException], ...] = (
    httpx.HTTPStatusError,
    httpx.RequestError,
)

#: Source label embedded in :class:`IngestionResult.source`.
_SOURCE_LIVE = "live"
_SOURCE_FRESH_CACHE = "fresh_cache"
_SOURCE_STALE_CACHE = "stale_cache"



class IngestionCoordinator:
    """Orchestrate the end-to-end bootstrap-data ingestion pipeline.

    Parameters
    ----------
    provider
        Any :class:`BootstrapDataProvider` (e.g. :class:`~fpl_engine.FPLClient`).
        The caller owns its HTTP-client lifecycle.
    cache
        A configured :class:`~fpl_engine.storage.SnapshotCache` for raw
        snapshot persistence / retrieval.
    writer
        A :class:`~fpl_engine.ingestion.processed.ProcessedWriter` for
        processed-data persistence.  If ``None``, processed data is not
        written (useful for dry-run or cache-only scenarios).
    ttl_seconds
        Maximum age (in seconds) for a cached snapshot to be considered
        *fresh*.  Defaults to 3600 (one hour).
    endpoint
        Raw FPL API endpoint path, e.g. ``"/bootstrap-static/"``.
    source
        Human-readable label stored in :class:`SnapshotMetadata.source`
        for newly fetched snapshots (typically ``"fpl-api"``).
    url
        Fully-qualified URL used for provenance in
        :class:`SnapshotMetadata.url` for newly fetched snapshots.
    """

    def __init__(
        self,
        provider: BootstrapDataProvider,
        cache: SnapshotCache,
        writer: ProcessedWriter | None = None,
        *,
        ttl_seconds: float = 3600.0,
        endpoint: str = "/bootstrap-static/",
        source: str = "fpl-api",
        url: str = "https://fantasy.premierleague.com/api/bootstrap-static/",
    ) -> None:
        self._provider = provider
        self._cache = cache
        self._writer = writer
        self._ttl_seconds = ttl_seconds
        self._endpoint = endpoint
        self._source = source
        self._url = url

    def run(
        self,
        *,
        allow_stale_fallback: bool = True,
        write_processed: bool = True,
        now: datetime | None = None,
    ) -> IngestionResult:
        """Execute the ingestion pipeline and return a structured result.

        Parameters
        ----------
        allow_stale_fallback
            When ``True`` (default) and the API request fails, the
            coordinator will use the newest *valid* stale snapshot from the
            cache instead of raising immediately.
        write_processed
            When ``True`` (default) and a writer was provided, the
            coordinator persists the processed bundle to disk.  Set to
            ``False`` for a dry-run that still returns the normalized
            result.
        now
            Override the current time (for deterministic testing).  When
            ``None``, :func:`datetime.now` with UTC timezone is used.

        Raises
        ------
        IngestionCoordinatorError
            If the API request fails and no usable stale snapshot is
            available, if raw-snapshot persistence fails, if
            normalization raises :class:`IngestionError`, if all three
            required sections are empty, or if processed-data persistence
            fails.

        Returns
        -------
        IngestionResult
            The normalized entities, validation reports, and provenance.
        """
        if now is None:
            now = datetime.now(UTC)

        # ---- Step 1: check for a fresh cached snapshot -------------------
        fresh = self._cache.get_cached(
            self._endpoint,
            ttl_seconds=self._ttl_seconds,
            now=now,
        )

        if fresh is not None:
            # Step 2: fresh cache hit — no API request needed.
            log.info(
                "Using fresh cached bootstrap snapshot",
                source=_SOURCE_FRESH_CACHE,
                endpoint=self._endpoint,
            )
            return self._normalize_and_persist(
                fresh,
                is_fresh=True,
                source=_SOURCE_FRESH_CACHE,
                write_processed=write_processed,
            )

        # ---- Step 3: retain newest stale snapshot for potential fallback --
        stale: Snapshot | None = None
        if allow_stale_fallback:
            stale = self._cache.get_cached_any(self._endpoint)

        # ---- Step 4: request bootstrap data from the provider ------------
        try:
            data = self._provider.get_bootstrap_data()
        except _PROVIDER_ERRORS as exc:
            # Step 6: API failure — use stale snapshot if allowed.
            if stale is not None and allow_stale_fallback:
                log.warning(
                    "Bootstrap API request failed; using stale cache fallback",
                    error=str(exc),
                    source=_SOURCE_STALE_CACHE,
                )
                return self._normalize_and_persist(
                    stale,
                    is_fresh=False,
                    source=_SOURCE_STALE_CACHE,
                    write_processed=write_processed,
                )
            # Step 7: no usable data - raise.
            raise IngestionCoordinatorError(  # noqa: TRY003
                f"Bootstrap API request failed and no stale snapshot is "
                f"available for fallback: {exc}"
            ) from exc

        # ---- Step 5: store the raw snapshot with accurate provenance ------
        try:
            snapshot = self._cache.store_raw(
                self._endpoint,
                data,
                source=self._source,
                url=self._url,
                retrieved_at=now,
            )
        except CacheError as exc:
            raise IngestionCoordinatorError(  # noqa: TRY003
                f"Failed to persist raw bootstrap snapshot: {exc}"
            ) from exc

        log.info("Fetched and stored fresh bootstrap snapshot", source=_SOURCE_LIVE)
        return self._normalize_and_persist(
            snapshot,
            is_fresh=True,
            source=_SOURCE_LIVE,
            write_processed=write_processed,
        )

    def _normalize_and_persist(
        self,
        snapshot: Snapshot,
        *,
        is_fresh: bool,
        source: str,
        write_processed: bool,
    ) -> IngestionResult:
        """Normalize the snapshot payload and optionally persist processed data.

        Raises :class:`IngestionCoordinatorError` if normalization fails
        (wrapping :class:`IngestionError`) or if all three required sections
        are empty.  Processed-data persistence failures are also surfaced
        as :class:`IngestionCoordinatorError`.
        """
        data = snapshot.payload

        # ---- Steps 8 & 9: normalize and reject all-empty payloads --------
        try:
            players, player_report = normalize_players(data)
            teams, team_report = normalize_teams(data)
            gameweeks, gameweek_report = normalize_gameweeks(data)
        except IngestionError as exc:
            raise IngestionCoordinatorError(  # noqa: TRY003
                f"Bootstrap payload is structurally unusable: {exc}"
            ) from exc

        if not players and not teams and not gameweeks:
            raise IngestionCoordinatorError(  # noqa: TRY003
                "Bootstrap payload has all three required sections empty "
                "(elements, teams, events)."
            )

        # ---- Step 10: persist only after successful normalization --------
        if write_processed and self._writer is not None:
            try:
                self._writer.write(
                    players,
                    teams,
                    gameweeks,
                    snapshot.metadata,
                )
            except CacheError as exc:
                raise IngestionCoordinatorError(  # noqa: TRY003
                    f"Failed to persist processed data: {exc}"
                ) from exc

        # ---- Step 11: return result with truthful provenance -------------
        return IngestionResult(
            players=players,
            teams=teams,
            gameweeks=gameweeks,
            player_report=player_report,
            team_report=team_report,
            gameweek_report=gameweek_report,
            snapshot_metadata=snapshot.metadata,
            is_fresh=is_fresh,
            source=source,
        )
