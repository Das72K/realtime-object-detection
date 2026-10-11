"""Drawing: boxes, labels, and the FPS / latency HUD."""

from __future__ import annotations

import colorsys

import cv2
import numpy as np

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
