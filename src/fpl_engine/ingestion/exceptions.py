"""Exception types for the data ingestion layer."""
from __future__ import annotations


class IngestionError(Exception):
    """Raised when input is structurally invalid or yields no usable records.

    Used by the normalization layer to signal:
      * payloads that are the wrong type or missing required top-level sections
      * sections whose content is not iterable as expected
      * results where every record was rejected (the dataset is unusable)
    """
