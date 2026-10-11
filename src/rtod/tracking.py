"""Multi-object tracking: give each physical object a persistent ID across frames.

Two backends:
- ByteTrack / BoT-SORT via Ultralytics (`YoloDetector.track`) — production.
- `IouTracker` below — a deliberately simple reference tracker. It shows the
  core idea (associate boxes frame-to-frame by IoU) and is fully unit-tested.

What ByteTrack adds on top of IouTracker:
1. Kalman filter: predicts where each track's box moves next frame, and matches
   detections against the *prediction*, not last frame's box -> survives fast motion.
2. Hungarian algorithm: globally optimal assignment instead of greedy.
3. Two-stage association: first match high-confidence boxes, then try to rescue
   unmatched tracks with LOW-confidence boxes (occluded / blurred objects).
   That is ByteTrack's key idea: low-score boxes are evidence, not noise.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .geometry import iou_matrix
from .types import Detection


@dataclass(slots=True)
class _Track:
    track_id: int
    box: tuple[float, float, float, float]
    class_id: int
    hits: int = 1
    missed: int = 0


class IouTracker:
    """Greedy IoU association with a lost-track timeout.

    iou_threshold: minimum IoU to call two boxes the same object.
    max_age:       frames a track may go unmatched before it is deleted
                   (bridges short occlusions / missed detections).
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(self, dets: list[Detection]) -> list[Detection]:
        track_ids = list(self._tracks)
        matched_det: dict[int, int] = {}  # det index -> track id

        if track_ids and dets:
            t_boxes = np.array([self._tracks[t].box for t in track_ids])
            d_boxes = np.array([d.xyxy for d in dets])
            m = iou_matrix(t_boxes, d_boxes)
            # greedy: best IoU pairs first; each track/det used at most once
            pairs = sorted(
                ((m[i, j], i, j) for i in range(len(track_ids)) for j in range(len(dets))),
                reverse=True,
            )
            used_t: set[int] = set()
            for score, i, j in pairs:
                if score < self.iou_threshold:
                    break
                tid = track_ids[i]
                if i in used_t or j in matched_det or self._tracks[tid].class_id != dets[j].class_id:
                    continue
                used_t.add(i)
                matched_det[j] = tid

        out: list[Detection] = []
        seen: set[int] = set()
        for j, d in enumerate(dets):
            tid = matched_det.get(j)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = _Track(tid, d.xyxy, d.class_id)
            else:
                t = self._tracks[tid]
                t.box, t.hits, t.missed = d.xyxy, t.hits + 1, 0
            seen.add(tid)
            out.append(replace(d, track_id=tid))

        for tid in list(self._tracks):
            if tid not in seen:
                self._tracks[tid].missed += 1
                if self._tracks[tid].missed > self.max_age:
                    del self._tracks[tid]
        return out
