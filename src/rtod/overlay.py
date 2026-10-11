"""Drawing: boxes, labels, and the FPS / latency HUD."""

from __future__ import annotations

import colorsys
from collections import deque

import cv2
import numpy as np

from .counting import LineCounter, ZoneCounter, anchor_point
from .types import Detection, Timing

FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for(key: int) -> tuple[int, int, int]:
    """Stable, well-separated BGR colour per class (or per track ID later).

    Golden-ratio hue stepping spreads consecutive IDs far apart on the colour wheel.
    """
    h = (key * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def _label(img: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int], scale: float) -> None:
    thickness = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    y_top = max(0, y - th - base - 4)
    cv2.rectangle(img, (x, y_top), (x + tw + 6, y_top + th + base + 4), color, -1)
    # dark text on light boxes, light text on dark boxes
    luminance = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    text_color = (0, 0, 0) if luminance > 150 else (255, 255, 255)
    cv2.putText(img, text, (x + 3, y_top + th + 2), FONT, scale, text_color, thickness, cv2.LINE_AA)


def draw_detections(img: np.ndarray, dets: list[Detection], scale: float | None = None) -> np.ndarray:
    scale = scale or max(0.4, min(img.shape[:2]) / 1100)
    thickness = max(2, int(round(scale * 3)))
    for d in dets:
        key = d.track_id if d.track_id is not None else d.class_id
        color = color_for(key)
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(img, p1, p2, color, thickness, cv2.LINE_AA)
        prefix = f"#{d.track_id} " if d.track_id is not None else ""
        _label(img, f"{prefix}{d.class_name} {d.confidence:.2f}", p1[0], p1[1], color, scale)
    return img


def draw_hud(
    img: np.ndarray,
    fps: float,
    timing: Timing,
    n_dets: int,
    device: str,
    model: str,
    extra_lines: list[str] | None = None,
) -> np.ndarray:
    lines = [
        f"FPS {fps:5.1f}" if fps > 0 else "FPS   --",
        f"model {timing.model_ms:5.1f} ms  (pre {timing.preprocess_ms:.1f} | inf {timing.inference_ms:.1f} | post {timing.postprocess_ms:.1f})",
        f"{model} @ {device}  |  objects: {n_dets}",
        *(extra_lines or []),
    ]
    scale = max(0.45, min(img.shape[:2]) / 1300)
    thickness = 1 if scale < 0.7 else 2
    line_h = int(28 * scale / 0.6)
    width = max(cv2.getTextSize(t, FONT, scale, thickness)[0][0] for t in lines) + 20
    height = line_h * len(lines) + 12

    # translucent panel
    overlay = img.copy()
    cv2.rectangle(overlay, (8, 8), (8 + width, 8 + height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, dst=img)
    for i, t in enumerate(lines):
        color = (80, 255, 120) if i == 0 else (235, 235, 235)
        cv2.putText(img, t, (18, 8 + line_h * (i + 1)), FONT, scale, color, thickness, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------- #
# Phase 2: tracking trails, counting line, zone
# --------------------------------------------------------------------------- #


class TrailBuffer:
    """Last N anchor points per track ID -> draws motion trails."""

    def __init__(self, length: int = 30, anchor: str = "bottom", forget_after: int = 60) -> None:
        self.length = length
        self.anchor = anchor
        self.forget_after = forget_after
        self._pts: dict[int, deque] = {}
        self._seen: dict[int, int] = {}

    def update(self, dets: list[Detection], frame: int) -> None:
        for d in dets:
            if d.track_id is None:
                continue
            q = self._pts.setdefault(d.track_id, deque(maxlen=self.length))
            x, y = anchor_point(d, self.anchor)
            q.append((int(x), int(y)))
            self._seen[d.track_id] = frame
        for tid in [t for t, f in self._seen.items() if frame - f > self.forget_after]:
            self._pts.pop(tid, None)
            self._seen.pop(tid, None)

    def draw(self, img: np.ndarray, active: set[int]) -> None:
        for tid, q in self._pts.items():
            if tid not in active or len(q) < 2:
                continue
            color = color_for(tid)
            pts = list(q)
            for i in range(1, len(pts)):
                thickness = 1 + int(3 * i / len(pts))  # thicker towards the present
                cv2.line(img, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)


def _ipt(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def draw_line_counter(img: np.ndarray, lc: LineCounter, flash: bool = False) -> None:
    p1, p2 = _ipt(lc.p1), _ipt(lc.p2)
    color = (0, 255, 255) if not flash else (0, 0, 255)
    cv2.line(img, p1, p2, color, 3 if not flash else 6, cv2.LINE_AA)
    for p in (p1, p2):
        cv2.circle(img, p, 6, color, -1, cv2.LINE_AA)
    # small arrow showing the 'in' direction (the positive side of p1->p2)
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    norm = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    nx, ny = -dy / norm, dx / norm  # screen-space right-hand normal = positive side
    tip = (int(mx + nx * 35), int(my + ny * 35))
    cv2.arrowedLine(img, (int(mx), int(my)), tip, color, 2, cv2.LINE_AA, tipLength=0.4)
    scale = max(0.5, min(img.shape[:2]) / 1100)
    _label(img, f"{lc.name}  in {lc.count_in}  out {lc.count_out}", p1[0], max(p1[1] - 8, 20), color, scale)


def draw_zone(img: np.ndarray, zc: ZoneCounter) -> None:
    pts = np.array([_ipt(p) for p in zc.polygon], dtype=np.int32)
    color = (255, 160, 60) if zc.occupancy == 0 else (60, 60, 255)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, dst=img)
    cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
    x, y = pts[:, 0].min(), pts[:, 1].min()
    scale = max(0.5, min(img.shape[:2]) / 1100)
    _label(img, f"{zc.name}  now {zc.occupancy}  total {len(zc.unique_ids)}", int(x), int(y) + 30, color, scale)
