"""Click points on a frame -> get normalised coords for --line / --zone.

    python scripts/pick_points.py --source data/videos/clip.mp4
    python scripts/pick_points.py --source 0                 # webcam snapshot

Left-click to add points.
  l      -> print the last 2 points as a --line argument
  z      -> print all points (3+) as a --zone argument
  c      -> clear points
  q/Esc  -> quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rtod.sources import FrameSource  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--frame", type=int, default=0, help="which frame of a video to show")
    args = ap.parse_args()

    with FrameSource(args.source) as src:
        base = None
        for f in src:
            base = f.image
            if f.index >= args.frame:
                break
    if base is None:
        sys.exit("could not read a frame")

    h, w = base.shape[:2]
    pts: list[tuple[int, int]] = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))

    cv2.namedWindow("pick", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("pick", on_mouse)
    print(__doc__)
    while True:
        img = base.copy()
        for i, p in enumerate(pts):
            cv2.circle(img, p, 6, (0, 255, 255), -1)
            cv2.putText(img, str(i + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if i:
                cv2.line(img, pts[i - 1], p, (0, 255, 255), 2)
        cv2.imshow("pick", img)
        key = cv2.waitKey(30) & 0xFF
        norm = [(round(x / w, 3), round(y / h, 3)) for x, y in pts]
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            pts.clear()
        if key == ord("l") and len(norm) >= 2:
            (x1, y1), (x2, y2) = norm[-2:]
            print(f'--line "line:{x1},{y1},{x2},{y2}"')
        if key == ord("z") and len(norm) >= 3:
            print('--zone "zone:' + ";".join(f"{x},{y}" for x, y in norm) + '"')
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
