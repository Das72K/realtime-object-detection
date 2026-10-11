"""Where do the milliseconds go? Isolates each layer of the latency stack.

    python scripts/diag_latency.py                 # yolo26n, 300 iters
    python scripts/diag_latency.py --model yolo26s.pt

Layer A: raw network forward on a GPU tensor (pure compute, synced) — FP32 and FP16
Layer B: full Ultralytics predict() on a 720p numpy frame (pre + inf + post)
Layer C: GPU clocks / power / P-state sampled WHILE A runs (throttling check)
"""

from __future__ import annotations

import argparse
import copy
import platform
import shutil
import statistics
import subprocess
import threading
import time

import numpy as np


def pct(values: list[float], p: float) -> float:
    data = sorted(values)
    k = (len(data) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (k - lo)


def fmt(values: list[float]) -> str:
    return f"p50 {pct(values, 50):6.2f} ms | p95 {pct(values, 95):6.2f} ms | mean {statistics.fmean(values):6.2f} ms"


class GpuSampler(threading.Thread):
    """Polls nvidia-smi every 200 ms while the hot loop runs."""

    QUERY = "clocks.sm,clocks.max.sm,power.draw,power.limit,pstate,temperature.gpu,utilization.gpu"

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.rows: list[list[str]] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={self.QUERY}", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2,
                ).stdout.strip().splitlines()
                if out:
                    self.rows.append([x.strip() for x in out[0].split(",")])
            except Exception:  # noqa: BLE001
                return
            time.sleep(0.2)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--h", type=int, default=384, help="network input height (16:9 letterbox = 384)")
    ap.add_argument("--w", type=int, default=640)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    print("== Versions ==")
    print(f"  python {platform.python_version()} | torch {torch.__version__} (CUDA {torch.version.cuda})")
    if not torch.cuda.is_available():
        print("  CUDA not available — this script is meant for the GPU path.")
        return
    dev = torch.device("cuda:0")
    print(f"  GPU: {torch.cuda.get_device_name(0)} | cudnn {torch.backends.cudnn.version()} "
          f"| cudnn.benchmark={torch.backends.cudnn.benchmark}")

    yolo = YOLO(args.model)
    # Fuse Conv+BatchNorm like predict() does -> same kernel count as the real pipeline
    base = copy.deepcopy(yolo.model).fuse(verbose=False) if hasattr(yolo.model, "fuse") else copy.deepcopy(yolo.model)
    net32 = copy.deepcopy(base).to(dev).eval()
    net16 = copy.deepcopy(base).to(dev).half().eval()
    n_modules = sum(1 for m in base.modules() if not list(m.children()))
    print(f"  model: {args.model} | {sum(p.numel() for p in base.parameters()) / 1e6:.1f} M params "
          f"| {n_modules} leaf layers (~ kernel launches per frame)")
    x32 = torch.rand(1, 3, args.h, args.w, device=dev)
    x16 = x32.half()

    def time_forward(net, x) -> list[float]:
        times = []
        with torch.inference_mode():
            for _ in range(30):  # warmup
                net(x)
            torch.cuda.synchronize()
            for _ in range(args.iters):
                t0 = time.perf_counter()
                net(x)
                torch.cuda.synchronize()  # GPU is async: without sync you time the launch, not the work
                times.append((time.perf_counter() - t0) * 1000)
        return times

    print(f"\n== Layer A: raw forward, input 1x3x{args.h}x{args.w} ==")
    sampler = GpuSampler() if shutil.which("nvidia-smi") else None
    if sampler:
        sampler.start()
    a32 = time_forward(net32, x32)
    a16 = time_forward(net16, x16)
    if sampler:
        sampler.stop()
    print(f"  FP32: {fmt(a32)}")
    print(f"  FP16: {fmt(a16)}")

    print("\n== Layer B: full predict() on a 1280x720 frame ==")
    frame = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
    for _ in range(20):
        yolo.predict(frame, device=0, verbose=False)
    pre, inf, post, wall = [], [], [], []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        r = yolo.predict(frame, device=0, verbose=False)[0]
        wall.append((time.perf_counter() - t0) * 1000)
        pre.append(r.speed["preprocess"])
        inf.append(r.speed["inference"])
        post.append(r.speed["postprocess"])
    print(f"  pre : {fmt(pre)}")
    print(f"  inf : {fmt(inf)}")
    print(f"  post: {fmt(post)}")
    print(f"  wall: {fmt(wall)}   <- includes Python/API overhead around the three stages")

    print("\n== Layer C: GPU state during Layer A ==")
    if sampler and sampler.rows:
        rows = sampler.rows
        sm = [float(r[0]) for r in rows if r[0].replace(".", "").isdigit()]
        pw = [float(r[2]) for r in rows if r[2].replace(".", "").isdigit()]
        print(f"  SM clock: max {max(sm):.0f} MHz of {rows[0][1]} MHz possible" if sm else "  SM clock: n/a")
        print(f"  power: max {max(pw):.1f} W (limit {rows[0][3]} W)" if pw else "  power: n/a")
        def num_max(i: int) -> str:  # compare as numbers: as strings, "7" > "48"
            vals = [float(r[i]) for r in rows if r[i].replace(".", "").isdigit()]
            return f"{max(vals):.0f}" if vals else "n/a"

        print(f"  P-states seen: {sorted({r[4] for r in rows})} | temp max {num_max(5)} C "
              f"| util max {num_max(6)} %")
    else:
        print("  nvidia-smi not available")

    print("\n== Verdict ==")
    a = pct(a32, 50)
    a_half = pct(a16, 50)
    b_inf = pct(inf, 50)
    b_wall = pct(wall, 50)
    util_max = max((float(r[6]) for r in sampler.rows if r[6].isdigit()), default=100.0) if sampler else 100.0
    if a > 8 and (util_max < 50 or a_half >= a * 0.9):
        print(f"  LAUNCH-BOUND: forward {a:.1f} ms but GPU util max {util_max:.0f}% and FP16 ({a_half:.1f} ms) is no faster.")
        print("  The GPU finishes each tiny kernel instantly and waits for the CPU to dispatch the next one.")
        print("  Levers: CPU clocks (Windows power mode = Best performance), keep GPU at P0")
        print("  (NVIDIA Control Panel: 'Prefer maximum performance'), fewer launches (ONNX/TensorRT/CUDA graphs).")
        print("  Test: a bigger model (yolo26s/m) should cost almost the same if this is right.")
    elif a > 12:
        print(f"  Raw GPU compute is slow ({a:.1f} ms). The GPU itself is held back: check that you're on AC")
        print("  power, Windows power mode = Best performance, NVIDIA Control Panel / vendor app = performance mode.")
    elif b_inf > 2 * a + 3:
        print(f"  GPU math is fine ({a:.1f} ms) but predict()'s inference is {b_inf:.1f} ms -> per-call overhead")
        print("  (sync, small-kernel launch latency, Python). Exported engines (ONNX/TensorRT) attack exactly this.")
    elif b_wall > 3 * a + 5:
        print(f"  GPU fine ({a:.1f} ms) but wall time {b_wall:.1f} ms -> CPU-side pre/post-processing dominates.")
    else:
        print(f"  Healthy: forward {a:.1f} ms, full predict {b_wall:.1f} ms. Slowness in rtod is elsewhere")
        print("  (video decode, drawing, imshow, writer, or the source's own frame rate).")


if __name__ == "__main__":
    main()
