# Real-Time Object Detection & Tracking

YOLO (Ultralytics) pipeline for webcam, video and images — boxes, labels, live FPS/latency HUD,
persistent tracking IDs, line/zone counting, PyTorch vs ONNX Runtime benchmarks, CSV/JSON export.

> 🚧 Phase 3 of 4 — detection, tracking, counting, export and the PyTorch vs ONNX Runtime
> benchmark work. Results table, plots and the demo GIF are published in Phase 4.

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

## Benchmark

```powershell
# GPU ONNX Runtime: must match PyTorch's CUDA major version (torch cu128 -> CUDA 12 build).
# onnxruntime-gpu >= 1.27 on PyPI is built for CUDA 13 and silently falls back to CPU here.
pip uninstall -y onnxruntime onnxruntime-gpu
pip install "onnxruntime-gpu>=1.21,<1.27"

rtod bench --source data\videos\clip.mp4 --models n,s,m --map      # GPU + CPU, ~10-15 min
```

Writes `results/benchmark.csv`, `benchmark.md`, `environment.json`, `latency_gpu.png`,
`latency_cpu.png`, `tradeoff_gpu.png`. Exported models are cached in `models/`.

| Backend | What runs |
|---|---|
| PyTorch (Ultralytics) | `.pt` weights, eager PyTorch, Ultralytics pre/post-processing |
| ONNX Runtime (Ultralytics) | same network exported to `.onnx`, run by Ultralytics on ONNX Runtime |
| ONNX Runtime (raw) | same `.onnx`, our ~150-line `onnx_backend.py`: letterbox → `session.run` → decode |

Methodology: frames pre-decoded into RAM, identical network input (e.g. 384×640) for every
backend, warmup discarded, 3 repeats → median p50 / p95, parity vs PyTorch boxes, mAP on coco128.

## Layout

```
src/rtod/
  types.py      Detection + Timing dataclasses (the pipeline's common language)
  geometry.py   IoU, IoU matrix, reference NMS, side-of-line, segment intersection, point-in-polygon
  detector.py   Ultralytics backend (.pt or .onnx) + ByteTrack/BoT-SORT tracking
  onnx_backend.py  raw ONNX Runtime backend: letterbox, NCHW tensor, decode (end-to-end + classic/NMS)
  bench.py      benchmark harness: repeats, percentiles, parity, mAP, table, plots
  tracking.py   IouTracker — minimal reference tracker (greedy IoU + lost-track timeout)
  counting.py   LineCounter (direction, debounce), ZoneCounter (occupancy, dwell), spec parsing
  export.py     streaming CSV + JSON run summary
  sources.py    webcam / video / image / folder -> Frame iterator
  overlay.py    boxes, labels, FPS + latency HUD
  metrics.py    EMA FPS meter, latency percentiles, stopwatch
  device.py     CPU-first device selection
  cli.py        `rtod detect ...` / `rtod track ...` / `rtod bench ...`
scripts/        check_env, get_samples, record_clip, pick_points, diag_latency
tests/
```

## License

MIT © 2026 Dorin Dascălu
