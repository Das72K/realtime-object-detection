"""Line-crossing and zone-occupancy counters (the classic paid CV use case).

Both work on *tracked* detections: without a persistent ID you cannot tell
"the same person moved across the line" from "a different person appeared".

Coordinates are pixels. Lines/zones are usually specified normalised (0..1)
on the CLI and converted with `denormalize()` once the frame size is known.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field

from .geometry import Point, point_in_polygon, segments_intersect, side_of_line
from .types import Detection


def anchor_point(d: Detection, anchor: str = "bottom") -> Point:
    """Which point of the box represents the object's position.

    'bottom' (bottom-centre) = where a person/car touches the ground. For a
    camera looking down at a floor or road this is the honest position; the box
    centre floats and makes tall objects "cross" too early.
    """
    return d.bottom_center if anchor == "bottom" else d.center


def denormalize(points: list[Point], width: int, height: int) -> list[Point]:
    return [(x * width, y * height) for x, y in points]


@dataclass(slots=True)
class CountEvent:
    kind: str          # 'line' | 'zone'
    counter: str       # counter name
    event: str         # line: 'in'/'out' ; zone: 'enter'/'exit'
    track_id: int
    class_name: str
    frame: int
    t: float
    dwell_s: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class LineCounter:
    """Counts tracks whose anchor point crosses the segment p1-p2.

    Direction: 'in' = moving from the negative to the positive side of p1->p2.
    For a horizontal line drawn left->right that means moving DOWN the image
    (image y grows downwards). Draw the line the other way round to flip it.

    Two checks, both required:
      1. side_of_line sign flips between frames      (crossed the infinite line)
      2. movement segment intersects p1-p2           (...within the drawn segment)
    `once_per_track` debounces jitter: a box wobbling on the line counts once
    per direction, not every frame.
    """

    name: str
    p1: Point
    p2: Point
    anchor: str = "bottom"
    once_per_track: bool = True
    forget_after: int = 300  # frames without seeing a track before its state is dropped

    count_in: int = 0
    count_out: int = 0
    by_class: Counter = field(default_factory=Counter)  # (class, 'in'/'out') -> n
    _side: dict[int, int] = field(default_factory=dict)
    _pt: dict[int, Point] = field(default_factory=dict)
    _seen: dict[int, int] = field(default_factory=dict)
    _counted: set[tuple[int, str]] = field(default_factory=set)

    def update(self, dets: list[Detection], frame: int, t: float) -> list[CountEvent]:
        events: list[CountEvent] = []
        for d in dets:
            if d.track_id is None:
                continue
            tid = d.track_id
            pt = anchor_point(d, self.anchor)
            s = side_of_line(self.p1, self.p2, pt)
            self._seen[tid] = frame
            if s == 0:  # exactly on the line: wait until it commits to a side
                continue
            prev_s = self._side.get(tid)
            prev_pt = self._pt.get(tid)
            if prev_s is not None and prev_s != s and segments_intersect(prev_pt, pt, self.p1, self.p2):
                direction = "in" if prev_s < 0 < s else "out"
                key = (tid, direction)
                if not (self.once_per_track and key in self._counted):
                    self._counted.add(key)
                    if direction == "in":
                        self.count_in += 1
                    else:
                        self.count_out += 1
                    self.by_class[(d.class_name, direction)] += 1
                    events.append(CountEvent("line", self.name, direction, tid, d.class_name, frame, t))
            self._side[tid] = s
            self._pt[tid] = pt
        self._forget(frame)
        return events

    def _forget(self, frame: int) -> None:
        for tid in [k for k, f in self._seen.items() if frame - f > self.forget_after]:
            self._seen.pop(tid, None)
            self._side.pop(tid, None)
            self._pt.pop(tid, None)

    def summary(self) -> dict:
        classes = sorted({c for c, _ in self.by_class})
        return {
            "p1": self.p1, "p2": self.p2,
            "in": self.count_in, "out": self.count_out, "net": self.count_in - self.count_out,
            "by_class": {c: {"in": self.by_class[(c, "in")], "out": self.by_class[(c, "out")]} for c in classes},
        }


@dataclass
class ZoneCounter:
    """Tracks which IDs are inside a polygon: occupancy, unique entries, dwell time.

    A track that vanishes while inside (occlusion, left the frame through the
    zone edge, tracker lost it) is exited after `grace` frames, so occupancy
    can't get stuck at a phantom count.
    """

    name: str
    polygon: list[Point]
    anchor: str = "bottom"
    grace: int = 30

    unique_ids: set[int] = field(default_factory=set)
    by_class: Counter = field(default_factory=Counter)  # class -> unique entries
    dwell_times: list[float] = field(default_factory=list)
    _inside: dict[int, tuple[float, str]] = field(default_factory=dict)  # id -> (enter_t, class)
    _seen: dict[int, int] = field(default_factory=dict)

    @property
    def occupancy(self) -> int:
        return len(self._inside)

    def update(self, dets: list[Detection], frame: int, t: float) -> list[CountEvent]:
        events: list[CountEvent] = []
        visible: set[int] = set()
        for d in dets:
            if d.track_id is None:
                continue
            tid = d.track_id
            visible.add(tid)
            self._seen[tid] = frame
            inside = point_in_polygon(anchor_point(d, self.anchor), self.polygon)
            if inside and tid not in self._inside:
                self._inside[tid] = (t, d.class_name)
                if tid not in self.unique_ids:
                    self.unique_ids.add(tid)
                    self.by_class[d.class_name] += 1
                events.append(CountEvent("zone", self.name, "enter", tid, d.class_name, frame, t))
            elif not inside and tid in self._inside:
                events.append(self._exit(tid, frame, t))

        for tid in list(self._inside):
            if tid not in visible and frame - self._seen.get(tid, frame) > self.grace:
                events.append(self._exit(tid, frame, t))
        return events

    def _exit(self, tid: int, frame: int, t: float) -> CountEvent:
        enter_t, cls = self._inside.pop(tid)
        dwell = max(0.0, t - enter_t)
        self.dwell_times.append(dwell)
        return CountEvent("zone", self.name, "exit", tid, cls, frame, t, round(dwell, 3))

    def summary(self) -> dict:
        dw = self.dwell_times
        return {
            "polygon": self.polygon,
            "unique_entries": len(self.unique_ids),
            "occupancy_at_end": self.occupancy,
            "mean_dwell_s": round(sum(dw) / len(dw), 2) if dw else None,
            "max_dwell_s": round(max(dw), 2) if dw else None,
            "by_class": dict(self.by_class),
        }


# --------------------------------------------------------------------------- #
# CLI spec parsing:  --line "door:0.1,0.6,0.9,0.6"   --zone "desk:0.2,0.3;0.6,0.3;0.6,0.9;0.2,0.9"
# --------------------------------------------------------------------------- #
def _split_name(spec: str, default: str) -> tuple[str, str]:
    if ":" in spec:
        name, body = spec.split(":", 1)
        return name.strip() or default, body
    return default, spec


def _check_norm(points: list[Point], spec: str) -> list[Point]:
    for x, y in points:
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"Coordinates must be normalised to 0..1 (got {x},{y}) in {spec!r}")
    return points


def parse_line_spec(spec: str, index: int = 0) -> tuple[str, list[Point]]:
    name, body = _split_name(spec, f"line{index + 1}")
    vals = [float(v) for v in body.replace(";", ",").split(",") if v.strip()]
    if len(vals) != 4:
        raise ValueError(f"Line needs 4 numbers x1,y1,x2,y2: {spec!r}")
    return name, _check_norm([(vals[0], vals[1]), (vals[2], vals[3])], spec)


def parse_zone_spec(spec: str, index: int = 0) -> tuple[str, list[Point]]:
    name, body = _split_name(spec, f"zone{index + 1}")
    pts = []
    for pair in body.split(";"):
        if pair.strip():
            x, y = (float(v) for v in pair.split(","))
            pts.append((x, y))
    if len(pts) < 3:
        raise ValueError(f"Zone needs at least 3 points 'x,y;x,y;x,y': {spec!r}")
    return name, _check_norm(pts, spec)
