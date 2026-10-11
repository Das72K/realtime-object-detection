"""Command line entry point.

    rtod detect --source data/samples/bus.jpg
    rtod detect --source data/samples/traffic.mp4 --model yolo26s.pt
    rtod detect --source 0                      # webcam (run on Windows!)
    rtod detect --source clip.mp4 --no-show     # headless (WSL / CI)

Keys in the window:  q / Esc = quit   space = pause   s = save snapshot
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from .detector import DetectorConfig, YoloDetector
from .device import describe_device
from .metrics import FpsMeter, LatencyStats, Stopwatch
from .overlay import draw_detections, draw_hud
from .sources import FrameSource


def _parse_classes(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [int(x) for x in text.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rtod", description="Real-time object detection")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="run detection on webcam / video / image / folder")
    d.add_argument("--source", required=True, help="webcam index (0), video, image, or folder")
    d.add_argument("--model", default="yolo26n.pt", help="weights: yolo26{n,s,m,l,x}.pt, yolo11n.pt, ...")
    d.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    d.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (NMS models only)")
    d.add_argument("--imgsz", type=int, default=640, help="network input size")
    d.add_argument("--device", default=None, help="cpu | 0 | mps (default: auto)")
    d.add_argument("--classes", default=None, help="comma-separated class IDs, e.g. 0,2 (person,car)")
    d.add_argument("--out", default="runs/detect", help="output folder")
    d.add_argument("--no-save", action="store_true", help="don't write annotated output")
    d.add_argument("--no-show", action="store_true", help="no window (headless / WSL)")
    d.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    d.add_argument("--cam-width", type=int, default=1280)
    d.add_argument("--cam-height", type=int, default=720)
    d.add_argument("--cam-backend", default="auto", choices=["auto", "msmf", "dshow", "any"],
                   help="webcam capture backend (auto tries msmf -> dshow -> any on Windows)")
    return p


def run_detect(args: argparse.Namespace) -> int:
    cfg = DetectorConfig(
        weights=args.model, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
        device=args.device, classes=_parse_classes(args.classes),
    )
    print(f"[rtod] loading {cfg.weights} ...")
    det = YoloDetector(cfg)
    print(f"[rtod] device: {describe_device(det.device)}")

    src = FrameSource(args.source, args.cam_width, args.cam_height, args.cam_backend)
    if src.is_live:
        print(f"[rtod] webcam opened via {src.backend.upper()}")
    out_dir = Path(args.out)
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    fps_meter, model_lat, e2e_lat = FpsMeter(), LatencyStats(), LatencyStats()
    pre_lat, inf_lat, post_lat = LatencyStats(), LatencyStats(), LatencyStats()
    writer: cv2.VideoWriter | None = None
    show = not args.no_show
    model_name = Path(cfg.weights).stem
    paused = False
    n_frames = 0
    t_start = time.perf_counter()

    try:
        for frame in src:
            det.ensure_warm(frame.image)  # untimed, once per input shape
            with Stopwatch() as sw:
                dets, timing = det.detect(frame.image)
                img = draw_detections(frame.image, dets)
            fps = fps_meter.tick() if src.is_stream else 0.0
            model_lat.add(timing.model_ms)
            pre_lat.add(timing.preprocess_ms)
            inf_lat.add(timing.inference_ms)
            post_lat.add(timing.postprocess_ms)
            e2e_lat.add(sw.ms)
            draw_hud(img, fps, timing, len(dets), describe_device(det.device), model_name)
            n_frames += 1

            if not args.no_save:
                if src.is_stream:
                    if writer is None:
                        h, w = img.shape[:2]
                        path = out_dir / f"{src.name}_{model_name}.mp4"
                        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), src.fps, (w, h))
                        print(f"[rtod] writing {path}")
                    writer.write(img)
                else:
                    path = out_dir / f"{frame.name}_{model_name}.jpg"
                    cv2.imwrite(str(path), img)
                    summary = ", ".join(f"{d.class_name} {d.confidence:.2f}" for d in dets) or "nothing"
                    print(f"[rtod] {frame.name}: {len(dets)} objects ({summary}) -> {path}")

            if show:
                cv2.imshow("rtod", img)
                key = cv2.waitKey(0 if not src.is_stream else 1) & 0xFF
                while paused and key not in (ord(" "), ord("q"), 27):
                    key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = not paused
                if key == ord("s"):
                    snap = out_dir / f"snapshot_{int(time.time())}.jpg"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(snap), img)
                    print(f"[rtod] snapshot -> {snap}")

            if args.max_frames and n_frames >= args.max_frames:
                break
    finally:
        src.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    wall = time.perf_counter() - t_start
    print(
        f"[rtod] {n_frames} frames in {wall:.1f}s  |  "
        f"model latency mean {model_lat.mean():.1f} ms, p95 {model_lat.percentile(95):.1f} ms  |  "
        f"end-to-end mean {e2e_lat.mean():.1f} ms"
        + (f"  |  throughput {n_frames / wall:.1f} FPS" if src.is_stream and wall > 0 else "")
    )
    print(
        f"[rtod] split (p50 ms): pre {pre_lat.percentile(50):.1f} | "
        f"inf {inf_lat.percentile(50):.1f} | post {post_lat.percentile(50):.1f}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "detect":
        return run_detect(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
