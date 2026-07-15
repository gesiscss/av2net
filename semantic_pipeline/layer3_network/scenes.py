"""Scene lookup from vision shot intervals.

Scenes are the vision shots produced in Layer 1 (each VisionObservation is one
shot). Layer 3 reads only their `[t_start, t_end)` intervals and ids, as a
structural partition of the timeline, to ground "same scene" co-occurrence. The
vision descriptions are not used here, and no MP4 is opened.

An event is assigned to the shot that contains its midpoint. Events outside every
shot (gaps, or no vision run) get scene id None.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models.observation import VisionObservation


@dataclass(frozen=True)
class SceneIndex:
    # Sorted, non-overlapping shot intervals: (t_start, t_end, scene_id).
    shots: tuple[tuple[float, float, str], ...]

    @staticmethod
    def from_observations(vision: list[VisionObservation]) -> "SceneIndex":
        shots = sorted(
            ((v.t_start, v.t_end, v.observation_id) for v in vision),
            key=lambda s: (s[0], s[1]),
        )
        return SceneIndex(shots=tuple(shots))

    def scene_at(self, t: float) -> str | None:
        for t0, t1, sid in self.shots:
            if t0 <= t < t1:
                return sid
        # Fall back to an inclusive end match (last shot boundary).
        for t0, t1, sid in self.shots:
            if t0 <= t <= t1:
                return sid
        return None

    def scene_of_interval(self, t_start: float, t_end: float) -> str | None:
        return self.scene_at((t_start + t_end) / 2.0)
