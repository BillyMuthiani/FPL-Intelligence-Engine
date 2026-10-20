"""Data models for snapshot metadata and snapshot containers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, ClassVar

from pydantic import BaseModel, field_validator


class SnapshotMetadata(BaseModel):
    """Provenance metadata for a single raw-API snapshot.

    The set of schema versions this code can read is defined by the
    class-level constant :attr:`SUPPORTED_SCHEMA_VERSIONS` and is **not**
    itself serialized into the sidecar file.
    """

    SUPPORTED_SCHEMA_VERSIONS: ClassVar[set[str]] = {"1"}

    schema_version: str
    source: str
    endpoint: str
    url: str
    retrieved_at: datetime
    content_sha256: str
    content_length: int
    content_encoding: str = "utf-8"
    json_serialization: str = "compact-sorted-keys"

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, v: str) -> str:
        if v not in cls.SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(  # noqa: TRY003
                f"Unsupported schema_version: {v!r} "
                f"(supported: {cls.SUPPORTED_SCHEMA_VERSIONS})"
            )
        return v

    @field_validator("content_sha256")
    @classmethod
    def _validate_sha256(cls, v: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", v):
            raise ValueError(  # noqa: TRY003
                "content_sha256 must be exactly 64 lowercase hex characters"
            )
        return v

    @field_validator("content_length")
    @classmethod
    def _validate_content_length(cls, v: int) -> int:
        if v < 0:
            raise ValueError("content_length must be non-negative")  # noqa: TRY003
        return v

    @field_validator("retrieved_at")
    @classmethod
    def _validate_retrieved_at(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("retrieved_at must be timezone-aware")  # noqa: TRY003
        if v.utcoffset() != UTC.utcoffset(None):
            raise ValueError("retrieved_at must be UTC")  # noqa: TRY003
        return v

    @field_validator("endpoint")
    @classmethod
    def _validate_endpoint(cls, v: str) -> str:
        if not v:
            raise ValueError("endpoint must not be empty")  # noqa: TRY003
        if not v.startswith("/"):
            raise ValueError("endpoint must start with '/'")  # noqa: TRY003
        return v

    @field_validator("source", "url")
    @classmethod
    def _validate_non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("must not be empty")  # noqa: TRY003
        return v


@dataclass(frozen=True)
class Snapshot:
    """A validated raw-API snapshot: the parsed payload plus its metadata."""

    payload: dict[str, Any]
    metadata: SnapshotMetadata
