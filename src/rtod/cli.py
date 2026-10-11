"""Command line entry point.

    rtod detect --source data/samples/bus.jpg
    rtod detect --source 0                                  # webcam (run on Windows!)
    rtod track  --source data/videos/clip.mp4 --model yolo26m.pt \\
                --line "door:0.5,0.1,0.5,0.95" --zone "desk:0.1,0.5;0.4,0.5;0.4,0.95;0.1,0.95"
    rtod track  --source 0 --classes 0 --line "0.5,0,0.5,1"  # people only, vertical line

Coordinates for --line / --zone are normalised (0..1) so they work at any
resolution; get them by clicking with `python scripts/pick_points.py --source ...`.

Keys in the window:  q / Esc = quit   space = pause   s = save snapshot
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from .counting import LineCounter, ZoneCounter, denormalize, parse_line_spec, parse_zone_spec
from .detector import DetectorConfig, YoloDetector
from .device import describe_device
from .export import ResultsExporter
from .metrics import FpsMeter, LatencyStats, Stopwatch
from .overlay import TrailBuffer, draw_detections, draw_hud, draw_line_counter, draw_zone
from .sources import FrameSource
from .tracking import IouTracker


def _parse_classes(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [int(x) for x in text.split(",") if x.strip()]


def _common_args(d: argparse.ArgumentParser, default_conf: float) -> None:
    d.add_argument("--source", required=True, help="webcam index (0), video, image, or folder")
    d.add_argument("--model", default="yolo26n.pt", help="weights: yolo26{n,s,m,l,x}.pt, yolo11n.pt, ...")
    d.add_argument("--conf", type=float, default=default_conf, help="confidence threshold")
    d.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (NMS models only)")
    d.add_argument("--imgsz", type=int, default=640, help="network input size")
    d.add_argument("--device", default=None, help="cpu | 0 | mps (default: auto)")
    d.add_argument("--classes", default=None, help="comma-separated class IDs, e.g. 0,2 (person,car)")
    d.add_argument("--no-save", action="store_true", help="don't write annotated output")
    d.add_argument("--no-show", action="store_true", help="no window (headless / WSL)")
    d.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    d.add_argument("--cam-width", type=int, default=1280)
    d.add_argument("--cam-height", type=int, default=720)
    d.add_argument("--cam-backend", default="auto", choices=["auto", "msmf", "dshow", "any"],
                   help="webcam capture backend (auto tries msmf -> dshow -> any on Windows)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rtod", description="Real-time object detection, tracking & counting")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="detection only (webcam / video / image / folder)")
    _common_args(d, default_conf=0.25)
    d.add_argument("--out", default="runs/detect", help="output folder")

    t = sub.add_parser("track", help="detection + tracking + line/zone counting + CSV/JSON export")
    # low default conf on purpose: ByteTrack's 2nd stage needs the low-score boxes (see tracking.py)
    _common_args(t, default_conf=0.1)
    t.add_argument("--out", default="runs/track", help="output folder")
    t.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "botsort", "iou"],
                   help="bytetrack/botsort = Ultralytics; iou = our simple reference tracker")
    t.add_argument("--line", action="append", default=[], metavar="[NAME:]x1,y1,x2,y2",
                   help="counting line, normalised coords; repeatable")
    t.add_argument("--zone", action="append", default=[], metavar="[NAME:]x,y;x,y;x,y...",
                   help="counting zone polygon, normalised coords; repeatable")
    t.add_argument("--anchor", default="bottom", choices=["bottom", "center"],
                   help="point of the box used for counting (bottom = feet / wheels)")
    t.add_argument("--trail", type=int, default=30, help="trail length in frames (0 = off)")
    t.add_argument("--no-export", action="store_true", help="skip CSV/JSON export")
    return p


def run(args: argparse.Namespace) -> int:
    tracking = args.cmd == "track"
    cfg = DetectorConfig(
        weights=args.model, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
        device=args.device, classes=_parse_classes(args.classes),
        tracker=f"{getattr(args, 'tracker', 'bytetrack')}.yaml",
    )
    print(f"[rtod] loading {cfg.weights} ...")
    det = YoloDetector(cfg)
    print(f"[rtod] device: {describe_device(det.device)}")

    src = FrameSource(args.source, args.cam_width, args.cam_height, args.cam_backend)
    if src.is_live:
        print(f"[rtod] webcam opened via {src.backend.upper()}")
    if tracking and not src.is_stream:
        print("[rtod] note: tracking needs consecutive frames; on single images IDs mean nothing")
    out_dir = Path(args.out)
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase 2 state ------------------------------------------------------
    line_specs = [parse_line_spec(s, i) for i, s in enumerate(getattr(args, "line", []))]
    zone_specs = [parse_zone_spec(s, i) for i, s in enumerate(getattr(args, "zone", []))]
    lines: list[LineCounter] = []
    zones: list[ZoneCounter] = []
    iou_tracker = IouTracker() if tracking and args.tracker == "iou" else None
    trails = TrailBuffer(args.trail, args.anchor) if tracking and args.trail > 0 else None
    model_name = Path(cfg.weights).stem
    exporter = None
    if tracking and not args.no_export:
        exporter = ResultsExporter(out_dir, f"{src.name}_{model_name}")
    flash: dict[str, int] = {}  # line name -> frames left to flash after a crossing

    fps_meter, model_lat, e2e_lat = FpsMeter(), LatencyStats(), LatencyStats()
    pre_lat, inf_lat, post_lat, trk_lat = LatencyStats(), LatencyStats(), LatencyStats(), LatencyStats()
    writer: cv2.VideoWriter | None = None
    show = not args.no_show
    paused = False
    n_frames = 0
    n_events = 0
    t_start = time.perf_counter()

    try:
        for frame in src:
            img = frame.image
            if tracking and not lines and not zones and (line_specs or zone_specs):
                h, w = img.shape[:2]  # first frame: convert normalised specs to pixels
                lines = [LineCounter(n, *denormalize(p, w, h), anchor=args.anchor) for n, p in line_specs]
                zones = [ZoneCounter(n, denormalize(p, w, h), anchor=args.anchor) for n, p in zone_specs]

            det.ensure_warm(img)  # untimed, once per input shape
            # video: position in the file; webcam: wall clock since start
            t = frame.timestamp_s if src.kind == "video" else time.perf_counter() - t_start

            with Stopwatch() as sw:
                if tracking and iou_tracker is None:
                    with Stopwatch() as sw_call:
                        dets, timing = det.track(img)
                    trk_lat.add(max(0.0, sw_call.ms - timing.model_ms))  # tracker + API overhead
                else:
                    dets, timing = det.detect(img)
                    if iou_tracker is not None:
                        with Stopwatch() as sw_trk:
                            dets = iou_tracker.update(dets)
                        trk_lat.add(sw_trk.ms)

                events = []
                for c in (*lines, *zones):
                    events += c.update(dets, frame.index, t)
                for e in events:
                    if e.kind == "line":
                        flash[e.counter] = 6
                        print(f"[rtod] {e.counter}: #{e.track_id} {e.class_name} {e.event.upper()} @ {t:.1f}s")

                for z in zones:
                    draw_zone(img, z)
                if trails is not None:
                    trails.update(dets, frame.index)
                    trails.draw(img, {d.track_id for d in dets if d.track_id is not None})
                draw_detections(img, dets)
                for lc in lines:
                    draw_line_counter(img, lc, flash.get(lc.name, 0) > 0)
            flash = {k: v - 1 for k, v in flash.items() if v > 1}

            if exporter is not None:
                exporter.log_detections(frame.index, t, dets)
                exporter.log_events(events)
            n_events += len(events)

            fps = fps_meter.tick() if src.is_stream else 0.0
            model_lat.add(timing.model_ms)
            pre_lat.add(timing.preprocess_ms)
            inf_lat.add(timing.inference_ms)
            post_lat.add(timing.postprocess_ms)
            e2e_lat.add(sw.ms)
            extra = []
            if tracking:
                n_ids = len({d.track_id for d in dets if d.track_id is not None})
                extra.append(f"tracker {args.tracker}: {n_ids} tracked  |  events {n_events}")
            draw_hud(img, fps, timing, len(dets), describe_device(det.device), model_name, extra)
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
            if exporter is not None and n_frames % 100 == 0:
                exporter.flush()
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
    split = (f"[rtod] split (p50 ms): pre {pre_lat.percentile(50):.1f} | "
             f"inf {inf_lat.percentile(50):.1f} | post {post_lat.percentile(50):.1f}")
    if len(trk_lat):
        split += f" | track {trk_lat.percentile(50):.1f}"
    print(split)

    for lc in lines:
        print(f"[rtod] line '{lc.name}': in {lc.count_in}, out {lc.count_out}, net {lc.count_in - lc.count_out}")
    for z in zones:
        s = z.summary()
        print(f"[rtod] zone '{z.name}': {s['unique_entries']} unique entries, "
              f"mean dwell {s['mean_dwell_s']} s, occupancy at end {s['occupancy_at_end']}")

    if exporter is not None:
        summary = {
            "source": str(args.source), "model": model_name, "device": describe_device(det.device),
            "tracker": args.tracker, "conf": args.conf, "imgsz": args.imgsz, "anchor": args.anchor,
            "frames": n_frames, "wall_s": round(wall, 2),
            "throughput_fps": round(n_frames / wall, 2) if wall > 0 else None,
            "latency_ms": {
                "model_mean": round(model_lat.mean(), 2), "model_p50": round(model_lat.percentile(50), 2),
                "model_p95": round(model_lat.percentile(95), 2), "end_to_end_mean": round(e2e_lat.mean(), 2),
                "tracker_p50": round(trk_lat.percentile(50), 2) if len(trk_lat) else None,
            },
            "lines": {lc.name: lc.summary() for lc in lines},
            "zones": {z.name: z.summary() for z in zones},
            "rows": {"detections": exporter.n_det_rows, "events": exporter.n_event_rows},
        }
        exporter.close()
        path = exporter.write_summary(summary)
        print(f"[rtod] exported -> {exporter.det_path.name}, {exporter.event_path.name}, {path.name} in {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd in ("detect", "track"):
        return run(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
