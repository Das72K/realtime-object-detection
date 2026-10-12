"""Benchmark: model sizes x backends x devices -> latency, throughput, parity, accuracy.

Methodology (each point exists because skipping it gives wrong numbers):
  * Frames are decoded into RAM first   -> video decoding isn't timed.
  * Same network input for every backend: ONNX is exported at the exact
    rectangle the .pt path letterboxes to (e.g. 384x640 for 16:9) -> equal pixels.
  * Warmup frames are discarded          -> no cold-start (CUDA init, kernel load).
  * Several repeats, report the median    -> one noisy run can't fake a trend.
  * p50 AND p95                          -> typical speed AND worst-case jitter.
  * Parity: ONNX boxes vs PyTorch boxes on the same frames -> export didn't
    silently change the model.
  * Accuracy (mAP) is a property of the WEIGHTS, so it's measured once per model
    on the .pt file; parity proves the ONNX copies behave the same.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .geometry import iou_matrix
from .metrics import Stopwatch
from .types import Detection

BACKEND_LABELS = {
    "torch": "PyTorch (Ultralytics)",
    "onnx": "ONNX Runtime (Ultralytics)",
    "ort": "ONNX Runtime (raw)",
}


@dataclass
class BenchResult:
    model: str
    backend: str
    device: str
    provider: str
    input_hw: str
    frames: int
    repeats: int
    total_p50: float          # median over repeats of the per-repeat p50 (ms)
    total_p95: float
    total_mean: float
    spread_p50: float         # max - min of the repeat p50s: how stable the number is
    pre_p50: float
    inf_p50: float
    post_p50: float
    fps: float                # 1000 / total_mean  (model-only throughput)
    parity_f1: float | None = None     # agreement with PyTorch on the same model/device
    parity_iou: float | None = None
    map50_95: float | None = None
    extra: dict = field(default_factory=dict)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    k = (len(data) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (k - lo)


def summarize(repeats: list[dict[str, list[float]]]) -> dict[str, float]:
    """repeats: [{'total': [...], 'pre': [...], 'inf': [...], 'post': [...]}, ...]"""
    p50s = [percentile(r["total"], 50) for r in repeats]
    p95s = [percentile(r["total"], 95) for r in repeats]
    all_total = [v for r in repeats for v in r["total"]]
    return {
        "total_p50": statistics.median(p50s),
        "total_p95": statistics.median(p95s),
        "total_mean": statistics.fmean(all_total),
        "spread_p50": max(p50s) - min(p50s),
        "pre_p50": statistics.median(percentile(r["pre"], 50) for r in repeats),
        "inf_p50": statistics.median(percentile(r["inf"], 50) for r in repeats),
        "post_p50": statistics.median(percentile(r["post"], 50) for r in repeats),
    }


def parity(ref: list[list[Detection]], test: list[list[Detection]], iou_thr: float = 0.5) -> tuple[float, float]:
    """How closely `test` reproduces `ref`, frame by frame.

    Greedy same-class matching at IoU >= iou_thr. Returns:
      F1 = 2*matches / (N_ref + N_test)   (1.0 = identical sets of boxes)
      mean IoU of matched pairs           (1.0 = pixel-identical boxes)
    """
    matches, n_ref, n_test, ious = 0, 0, 0, []
    for a, b in zip(ref, test):
        n_ref += len(a)
        n_test += len(b)
        if not a or not b:
            continue
        m = iou_matrix(np.array([d.xyxy for d in a]), np.array([d.xyxy for d in b]))
        used_a, used_b = set(), set()
        for i, j in sorted(np.ndindex(m.shape), key=lambda ij: -m[ij]):
            if m[i, j] < iou_thr:
                break
            if i in used_a or j in used_b or a[i].class_id != b[j].class_id:
                continue
            used_a.add(i)
            used_b.add(j)
            matches += 1
            ious.append(float(m[i, j]))
    if n_ref + n_test == 0:
        return 1.0, 1.0
    return 2 * matches / (n_ref + n_test), (statistics.fmean(ious) if ious else 0.0)


def time_detector(det, frames: list[np.ndarray], n_frames: int, repeats: int, warmup: int):
    """Returns (list of per-repeat timing dicts, detections of the first len(frames) frames)."""
    for i in range(warmup):
        det.detect(frames[i % len(frames)])
    runs = []
    captured: list[list[Detection]] = []
    for r in range(repeats):
        rec = {"total": [], "pre": [], "inf": [], "post": []}
        for i in range(n_frames):
            f = frames[i % len(frames)]
            with Stopwatch() as sw:
                dets, timing = det.detect(f)
            rec["total"].append(sw.ms)
            rec["pre"].append(timing.preprocess_ms)
            rec["inf"].append(timing.inference_ms)
            rec["post"].append(timing.postprocess_ms)
            if r == 0 and i < len(frames):
                captured.append(dets)
        runs.append(rec)
    return runs, captured


def load_frames(source: str, max_frames: int = 100) -> list[np.ndarray]:
    from .sources import FrameSource

    frames = []
    with FrameSource(source) as src:
        for f in src:
            frames.append(f.image)
            if len(frames) >= max_frames:
                break
    if not frames:
        raise RuntimeError(f"no frames read from {source}")
    return frames


def export_onnx(pt: str, hw: tuple[int, int], device: str, out_dir: Path) -> Path:
    """Export once, cache as e.g. models/yolo26m_384x640.onnx."""
    from ultralytics import YOLO

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{Path(pt).stem}_{hw[0]}x{hw[1]}.onnx"
    if target.exists():
        return target
    print(f"[bench] exporting {pt} -> {target.name} (static {hw[0]}x{hw[1]}, FP32)")
    produced = Path(YOLO(pt).export(format="onnx", imgsz=list(hw), dynamic=False, simplify=True,
                                    device=device if device != "cpu" else "cpu", verbose=False))
    produced.replace(target)
    return target


def measure_map(pt: str, device: str, data: str = "coco128.yaml") -> float | None:
    from ultralytics import YOLO

    try:
        m = YOLO(pt).val(data=data, imgsz=640, batch=1, device=device, plots=False, verbose=False)
        return float(m.box.map)
    except Exception as e:  # noqa: BLE001 - accuracy is optional, never kill the benchmark
        print(f"[bench] mAP for {pt} failed: {e}")
        return None


def environment() -> dict:
    env = {"python": platform.python_version(), "os": f"{platform.system()} {platform.release()}",
           "cpu": platform.processor()}
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    for mod in ("ultralytics", "onnxruntime", "cv2"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            pass
    return env


def check_ort_gpu() -> None:
    """Fail fast (before 15 minutes of benchmarking) if ONNX Runtime can't use the GPU."""
    try:
        from importlib.metadata import distributions

        names = {d.metadata["Name"].lower() for d in distributions()}
    except Exception:
        names = set()
    both = {"onnxruntime", "onnxruntime-gpu"} <= names
    import onnxruntime as ort

    providers = ort.get_available_providers()
    if both or "CUDAExecutionProvider" not in providers:
        raise SystemExit(
            "[bench] ONNX Runtime GPU is not usable:\n"
            f"  installed: {sorted(n for n in names if n.startswith('onnxruntime'))}\n"
            f"  providers: {providers}\n"
            "  fix:  pip uninstall -y onnxruntime onnxruntime-gpu\n"
            '        pip install "onnxruntime-gpu>=1.21,<1.27"   (CUDA 12 build, matches torch cu12x)'
        )
    print(f"[bench] ONNX Runtime GPU ok: {providers}")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_csv(results: list[BenchResult], path: Path) -> None:
    rows = [{k: v for k, v in asdict(r).items() if k != "extra"} for r in results]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _fmt(v, nd=1):
    return "–" if v is None else f"{v:.{nd}f}"


def markdown_table(results: list[BenchResult]) -> str:
    head = ("| Model | Backend | Device | p50 ms | p95 ms | FPS | pre / inf / post (p50 ms) "
            "| Parity F1 | mAP50-95 |\n|---|---|---|--:|--:|--:|---|--:|--:|")
    lines = [head]
    for r in results:
        lines.append(
            f"| {r.model} | {BACKEND_LABELS.get(r.backend, r.backend)} | {r.device} "
            f"| **{r.total_p50:.1f}** | {r.total_p95:.1f} | {r.fps:.0f} "
            f"| {r.pre_p50:.1f} / {r.inf_p50:.1f} / {r.post_p50:.1f} "
            f"| {_fmt(r.parity_f1, 3)} | {_fmt(r.map50_95, 3)} |"
        )
    return "\n".join(lines)


# Validated categorical slots 1-3 (light surface), fixed order: colour follows the backend.
PALETTE = {"torch": "#2a78d6", "onnx": "#eb6834", "ort": "#1baf7a"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _device_label(device: str) -> str:
    return "CPU" if device == "cpu" else "GPU"


def _legend_on_top(ax, n: int) -> None:
    """Legend in its own row above the plot: never collides with bars, whiskers or labels."""
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=n, loc="lower left",
              bbox_to_anchor=(0, 1.0), borderaxespad=0.3, handlelength=1.2, columnspacing=1.6)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_latency(results: list[BenchResult], path: Path, device: str) -> Path | None:
    """Grouped bars: p50 latency per model, one bar per backend, p95 as a thin whisker."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rs = [r for r in results if r.device == device]
    if not rs:
        return None
    models = list(dict.fromkeys(r.model for r in rs))
    backends = [b for b in PALETTE if any(r.backend == b for r in rs)]
    width = 0.8 / len(backends)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150, facecolor=SURFACE)
    _style(ax)
    for k, b in enumerate(backends):
        xs, p50, p95 = [], [], []
        for i, m in enumerate(models):
            r = next((r for r in rs if r.model == m and r.backend == b), None)
            if r:
                xs.append(i + (k - (len(backends) - 1) / 2) * width)
                p50.append(r.total_p50)
                p95.append(r.total_p95)
        ax.bar(xs, p50, width * 0.9, color=PALETTE[b], label=BACKEND_LABELS[b], zorder=2,
               edgecolor=SURFACE, linewidth=1)
        ax.vlines(xs, p50, p95, color=INK_2, linewidth=1.2, zorder=3)
        ax.scatter(xs, p95, marker="_", s=60, color=INK_2, zorder=3)
        for x, v in zip(xs, p50):
            ax.text(x, v / 2, f"{v:.1f}", ha="center", va="center", fontsize=7.5, color="#ffffff", zorder=4)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, color=INK)
    ax.set_ylabel("latency per frame (ms)", color=INK_2)
    ax.set_title(f"Model latency on {_device_label(device)} — bar = p50, whisker = p95 (lower is better)",
                 color=INK, fontsize=10, loc="left", pad=26)
    _legend_on_top(ax, len(backends))
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_tradeoff(results: list[BenchResult], path: Path, device: str) -> Path | None:
    """Accuracy vs latency: the curve a client actually chooses from."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rs = [r for r in results if r.device == device and r.map50_95 is not None]
    if not rs:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150, facecolor=SURFACE)
    _style(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    for b in PALETTE:
        pts = sorted((r.total_p50, r.map50_95, r.model) for r in rs if r.backend == b)
        if not pts:
            continue
        xs, ys, names = zip(*pts)
        ax.plot(xs, ys, color=PALETTE[b], linewidth=2, zorder=2)
        ax.scatter(xs, ys, s=64, color=PALETTE[b], edgecolor=SURFACE, linewidth=2, zorder=3,
                   label=BACKEND_LABELS[b])
        if b == "torch":
            for x, y, n in pts:
                ax.annotate(n, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8, color=INK)
    ax.set_xlabel("p50 latency (ms)  →  slower", color=INK_2)
    ax.set_ylabel("mAP50-95 (coco128)  →  better", color=INK_2)
    ax.set_title(f"Accuracy vs latency on {_device_label(device)} (up and left is better)",
                 color=INK, fontsize=10, loc="left", pad=26)
    _legend_on_top(ax, sum(any(r.backend == b for r in rs) for b in PALETTE))
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_benchmark(
    source: str,
    models: list[str],
    backends: list[str],
    devices: list[str],
    n_frames: int = 300,
    repeats: int = 3,
    warmup: int = 30,
    out_dir: str = "results",
    with_map: bool = False,
    cpu_frames: int = 100,
) -> list[BenchResult]:
    # Ultralytics "helpfully" pip-installs the CPU `onnxruntime` package when it only
    # finds `onnxruntime-gpu` -> the CPU package overwrites the GPU one mid-benchmark.
    # Must be set before ultralytics is first imported.
    os.environ["YOLO_AUTOINSTALL"] = "False"

    from .detector import DetectorConfig, YoloDetector
    from .onnx_backend import OnnxDetector, make_session, rect_shape

    if {"onnx", "ort"} & set(backends) and any(d != "cpu" for d in devices):
        check_ort_gpu()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = load_frames(source)
    hw = rect_shape(*frames[0].shape[:2])
    print(f"[bench] {len(frames)} frames in RAM ({frames[0].shape[1]}x{frames[0].shape[0]}) "
          f"-> network input {hw[0]}x{hw[1]} for every backend")

    maps: dict[str, float | None] = {}
    if with_map:
        for m in models:
            print(f"[bench] mAP {m} on coco128 ...")
            maps[m] = measure_map(m, devices[0])

    results: list[BenchResult] = []
    gpu_probed = False
    for device in devices:
        n = cpu_frames if device == "cpu" else n_frames
        for m in models:
            onnx_path = export_onnx(m, hw, device, Path("models")) if {"onnx", "ort"} & set(backends) else None
            if onnx_path is not None and device != "cpu" and not gpu_probed:
                # get_available_providers() only lists what ORT was COMPILED with; a real
                # session proves the CUDA DLLs actually load (version mismatch = silent CPU).
                probe = make_session(str(onnx_path), device).get_providers()[0]
                if probe != "CUDAExecutionProvider":
                    raise SystemExit(
                        "[bench] ONNX Runtime could not start on the GPU (it would silently run on CPU).\n"
                        "  Most likely a CUDA version mismatch between onnxruntime-gpu and PyTorch.\n"
                        "  For PyTorch built with CUDA 12.x:\n"
                        "    pip uninstall -y onnxruntime onnxruntime-gpu\n"
                        '    pip install "onnxruntime-gpu>=1.21,<1.27"'
                    )
                print("[bench] ONNX Runtime CUDA session verified")
                gpu_probed = True
            ref_dets = None
            for b in backends:
                if b == "torch":
                    det = YoloDetector(DetectorConfig(weights=m, device=device))
                    provider = "torch-cuda" if device != "cpu" else "torch-cpu"
                elif b == "onnx":
                    det = YoloDetector(DetectorConfig(weights=str(onnx_path), device=device, imgsz=list(hw)))
                    provider = "ultralytics-ort"
                else:
                    det = OnnxDetector(str(onnx_path), device=device)
                    provider = det.provider
                label = f"{Path(m).stem:9s} {b:5s} {device:4s}"
                t0 = time.perf_counter()
                runs, dets = time_detector(det, frames, n, repeats, warmup)
                s = summarize(runs)
                r = BenchResult(
                    model=Path(m).stem, backend=b, device=device, provider=provider,
                    input_hw=f"{hw[0]}x{hw[1]}", frames=n, repeats=repeats,
                    fps=1000.0 / s["total_mean"], map50_95=maps.get(m), **s,
                )
                if b == "torch":
                    ref_dets = dets
                elif ref_dets is not None:
                    r.parity_f1, r.parity_iou = parity(ref_dets, dets)
                results.append(r)
                print(f"[bench] {label} p50 {r.total_p50:6.2f} ms  p95 {r.total_p95:6.2f} ms  "
                      f"±{r.spread_p50:.2f}  parity {_fmt(r.parity_f1, 3)}  ({time.perf_counter() - t0:.0f}s)")
                del det

    write_csv(results, out / "benchmark.csv")
    md = markdown_table(results)
    (out / "benchmark.md").write_text(md + "\n", encoding="utf-8")
    env = environment()
    (out / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    for device in devices:
        tag = "gpu" if device != "cpu" else "cpu"
        plot_latency(results, out / f"latency_{tag}.png", device)
        plot_tradeoff(results, out / f"tradeoff_{tag}.png", device)
    print("\n" + md + "\n")
    print(f"[bench] wrote {out}/benchmark.csv, benchmark.md, environment.json and plots")
    return results
