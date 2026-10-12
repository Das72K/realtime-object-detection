"""Turn an annotated run video into a README-sized GIF.

    python scripts/make_gif.py runs/track/webcam0_yolo26m_384x640.mp4
    python scripts/make_gif.py clip.mp4 --start 2 --seconds 8 --width 640 --fps 12

Keeps the file small enough for GitHub (aim for < 10 MB): fewer frames per
second, smaller width, and one shared 128-colour palette.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="docs/demo.gif")
    ap.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    ap.add_argument("--seconds", type=float, default=8.0, help="clip length")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--fps", type=float, default=12.0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / args.fps))
    first = int(args.start * src_fps)
    last = first + int(args.seconds * src_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    frames: list[Image.Image] = []
    idx = first
    while idx < last:
        ok, img = cap.read()
        if not ok:
            break
        if (idx - first) % step == 0:
            h, w = img.shape[:2]
            img = cv2.resize(img, (args.width, round(h * args.width / w)), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        idx += 1
    cap.release()
    if not frames:
        raise SystemExit("no frames in the selected range")

    # one palette for all frames -> smaller file, no colour flicker between frames
    palette = frames[len(frames) // 2].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    gif = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gif[0].save(out, save_all=True, append_images=gif[1:], duration=int(1000 * step / src_fps),  # real-time playback speed
                loop=0, optimize=True)
    print(f"{len(gif)} frames -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
