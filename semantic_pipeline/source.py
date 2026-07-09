"""SourceRecord: metadata about the transient observation source (the video).

Written once per run to source.json. Holds the time anchor and media properties.
The MP4 is transient and never becomes a semantic node; this record exists only so
the observation process is reproducible and so offset times can be resolved to
absolute time via the anchor.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SourceRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    video: str
    time_anchor: str | None = None   # ISO 8601 tz-aware absolute t=0; None = zero origin
    main_language: str | None = None  # detected primary (spoken) language; used to align captions
    duration_s: float
    fps: float
    frame_count: int
    width: int
    height: int
