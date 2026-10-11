"""Streaming CSV export + JSON run summary.

Output folder per run:
    <stem>_detections.csv   one row per tracked object per frame
    <stem>_events.csv       one row per line crossing / zone enter / zone exit
    <stem>_summary.json     totals, per-class counts, latency stats, run config

CSV rows are written as they happen (not buffered to the end), so a crash or
Ctrl+C after an hour of video still leaves usable data on disk.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .counting import CountEvent
from .types import Detection

DET_FIELDS = ["frame", "t", "track_id", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
EVENT_FIELDS = ["frame", "t", "kind", "counter", "event", "track_id", "class_name", "dwell_s"]


class ResultsExporter:
    def __init__(self, out_dir: str | Path, stem: str) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.det_path = self.out_dir / f"{stem}_detections.csv"
        self.event_path = self.out_dir / f"{stem}_events.csv"
        self.summary_path = self.out_dir / f"{stem}_summary.json"
        self._det_f = open(self.det_path, "w", newline="", encoding="utf-8")
        self._evt_f = open(self.event_path, "w", newline="", encoding="utf-8")
        self._det_w = csv.DictWriter(self._det_f, fieldnames=DET_FIELDS)
        self._evt_w = csv.DictWriter(self._evt_f, fieldnames=EVENT_FIELDS)
        self._det_w.writeheader()
        self._evt_w.writeheader()
        self.n_det_rows = 0
        self.n_event_rows = 0

    def log_detections(self, frame: int, t: float, dets: Iterable[Detection]) -> None:
        for d in dets:
            self._det_w.writerow({
                "frame": frame, "t": round(t, 3), "track_id": d.track_id if d.track_id is not None else "",
                "class_id": d.class_id, "class_name": d.class_name, "confidence": round(d.confidence, 4),
                "x1": round(d.x1, 1), "y1": round(d.y1, 1), "x2": round(d.x2, 1), "y2": round(d.y2, 1),
            })
            self.n_det_rows += 1

    def log_events(self, events: Iterable[CountEvent]) -> None:
        for e in events:
            row = e.as_dict()
            row["t"] = round(row["t"], 3)
            row["dwell_s"] = "" if row["dwell_s"] is None else row["dwell_s"]
            self._evt_w.writerow({k: row[k] for k in EVENT_FIELDS})
            self.n_event_rows += 1

    def flush(self) -> None:
        self._det_f.flush()
        self._evt_f.flush()

    def write_summary(self, summary: dict) -> Path:
        self.summary_path.write_text(json.dumps(summary, indent=2, default=_jsonable), encoding="utf-8")
        return self.summary_path

    def close(self) -> None:
        self._det_f.close()
        self._evt_f.close()

    def __enter__(self) -> "ResultsExporter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _jsonable(o):
    if isinstance(o, (set, tuple)):
        return list(o)
    if hasattr(o, "item"):  # numpy scalars
        return o.item()
    return str(o)
