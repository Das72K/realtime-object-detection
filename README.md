# Real-Time Object Detection & Tracking

YOLO (Ultralytics) pipeline for webcam, video and images — boxes, labels, live FPS/latency HUD,
persistent tracking IDs, line/zone counting, PyTorch vs ONNX Runtime benchmarks, CSV/JSON export.

> 🚧 Phase 1 of 4 — detection on images / video / webcam is working. Tracking, counting,
> benchmarks and the demo GIF land in Phases 2–4.

## Quick start (Windows, PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,onnx]"
python scripts\check_env.py
rtod detect --source data\samples\bus.jpg
rtod detect --source 0            # webcam
```

## Layout

```
src/rtod/
  types.py      Detection + Timing dataclasses (the pipeline's common language)
  geometry.py   IoU, vectorised IoU matrix, reference NMS
  detector.py   YOLO backend (PyTorch); ONNX backend added in Phase 3
  sources.py    webcam / video / image / folder -> Frame iterator
  overlay.py    boxes, labels, FPS + latency HUD
  metrics.py    EMA FPS meter, latency percentiles, stopwatch
  device.py     CPU-first device selection
  cli.py        `rtod detect ...`
scripts/check_env.py
tests/
```

## License

MIT © 2026 Dorin Dascălu
