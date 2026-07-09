"""Absolute time base.

Observation units and temporal interactions store time as offset seconds from a
single anchor. The anchor is an optional timezone-aware instant recorded once in
the SourceRecord; when it is None the offsets are their own canonical time (zero
origin). Making the anchor a parameter means the same numeric stream can be read
as pure offsets by default and as absolute wall-clock time the moment an anchor
is supplied, with no reprocessing.
"""

from __future__ import annotations

from datetime import datetime, timedelta


def parse_anchor(value: str | None) -> datetime | None:
    """Parse an ISO 8601 (ideally timezone-aware) anchor string, or None."""
    if not value:
        return None
    return datetime.fromisoformat(value)


def absolute_time(anchor: datetime | None, offset_seconds: float) -> datetime | None:
    """Absolute instant for an offset, or None under a zero-origin (no anchor) run."""
    if anchor is None:
        return None
    return anchor + timedelta(seconds=offset_seconds)
