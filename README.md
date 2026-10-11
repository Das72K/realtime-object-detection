# Real-Time Object Detection & Tracking

YOLO (Ultralytics) pipeline for webcam, video and images — boxes, labels, live FPS/latency HUD,
persistent tracking IDs, line/zone counting, PyTorch vs ONNX Runtime benchmarks, CSV/JSON export.

> 🚧 Phase 2 of 4 — detection, tracking (ByteTrack), line/zone counting and CSV/JSON export work.
> Benchmarks (PyTorch vs ONNX Runtime) and the demo GIF land in Phases 3–4.

## Quick start (Windows, PowerShell)

```powershell
py -3 -m venv .venv               # any Python 3.11+
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,onnx]"
python scripts\check_env.py
rtod detect --source data\samples\bus.jpg
rtod detect --source 0            # webcam

# tracking + counting (coords normalised 0..1; click them with scripts\pick_points.py)
rtod track --source data\videos\clip.mp4 --model yolo26m.pt --classes 0 `
  --line "door:0.5,0.95,0.5,0.05" --zone "desk:0.6,0.3;0.95,0.3;0.95,0.95;0.6,0.95"
```

Outputs in `runs/track/`: annotated `.mp4`, `*_detections.csv` (every tracked box),
`*_events.csv` (line crossings, zone enter/exit with dwell time) and `*_summary.json`.

**Line direction:** `in` = crossing toward the side the yellow arrow points to (the right-hand
side of p1→p2 on screen). Swap the two points to flip it.

## Layout

```
src/rtod/
  types.py      Detection + Timing dataclasses (the pipeline's common language)
  geometry.py   IoU, IoU matrix, reference NMS, side-of-line, segment intersection, point-in-polygon
  detector.py   YOLO backend (PyTorch) + ByteTrack/BoT-SORT tracking; ONNX backend in Phase 3
  tracking.py   IouTracker — minimal reference tracker (greedy IoU + lost-track timeout)
  counting.py   LineCounter (direction, debounce), ZoneCounter (occupancy, dwell), spec parsing
  export.py     streaming CSV + JSON run summary
  sources.py    webcam / video / image / folder -> Frame iterator
  overlay.py    boxes, labels, FPS + latency HUD
  metrics.py    EMA FPS meter, latency percentiles, stopwatch
  device.py     CPU-first device selection
  cli.py        `rtod detect ...` / `rtod track ...`
scripts/        check_env, get_samples, record_clip, pick_points, diag_latency
tests/
```

## License

MIT © 2026 Dorin Dascălu
