"""Record a test clip from the webcam into data/videos/ (raw frames, no detection).

    python scripts/record_clip.py                    # 20 s from webcam 0 -> data/videos/clip.mp4
    python scripts/record_clip.py --seconds 40 --out data/videos/room.mp4

Walk in and out of frame a few times: that gives the tracker and the line
counter something to count. Press q to stop early.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rtod.sources import FrameSource  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="0")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--out", default="data/videos/clip.mp4")
    ap.add_argument("--cam-backend", default="auto")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    src = FrameSource(args.cam, 1280, 720, args.cam_backend)
    print(f"recording via {src.backend.upper()} -> {out}  (q to stop)")

    writer = None
    t0 = time.perf_counter()
    n = 0
    try:
        for frame in src:
            if writer is None:
                h, w = frame.image.shape[:2]
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
            writer.write(frame.image)
            n += 1
            elapsed = time.perf_counter() - t0
            preview = frame.image.copy()
            cv2.putText(preview, f"REC {elapsed:4.1f}/{args.seconds:.0f}s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("record", preview)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27) or elapsed >= args.seconds:
                break
    finally:
        src.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    real_fps = n / max(time.perf_counter() - t0, 1e-6)
    print(f"saved {n} frames ({real_fps:.1f} fps actual) -> {out}")


if __name__ == "__main__":
    main()
