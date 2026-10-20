"""Exception types for the data ingestion layer."""
from __future__ import annotations


class IngestionError(Exception):
    """Raised when input is structurally invalid or yields no usable records.

    Used by the normalization layer to signal:
      * payloads that are the wrong type or missing required top-level sections
      * sections whose content is not iterable as expected
      * results where every record was rejected (the dataset is unusable)
    """


class IngestionCoordinatorError(Exception):
    """Raised when the end-to-end ingestion pipeline cannot complete.

    This is distinct from :class:`IngestionError` (originating in the
    normalization layer) and from
    :class:`~fpl_engine.storage.exceptions.CacheError` (originating in the
    raw-snapshot persistence layer).

    The coordinator raises this when:

    * The bootstrap API request failed and no usable stale snapshot is
      available for fallback.
    * The bootstrap payload has all three required sections empty
      (``elements``, ``teams``, ``events`` are all present but empty).
    * Normalization raised :class:`IngestionError` (the payload is
      structurally unusable).
    * Raw-snapshot persistence or processed-data persistence failed.
    """
