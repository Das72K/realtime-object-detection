"""Box geometry: IoU and a reference NMS implementation.

These are written from scratch on purpose. Ultralytics does all of this
internally, but the tracker (Phase 2) and the ONNX backend (Phase 3) need IoU
directly, and implementing NMS once is the best way to understand it.
"""

from __future__ import annotations

import numpy as np


def iou(a: np.ndarray | tuple, b: np.ndarray | tuple) -> float:
    """Intersection-over-Union of two boxes in (x1, y1, x2, y2) format.

    IoU = area(A ∩ B) / area(A ∪ B), in [0, 1].
    0 → no overlap, 1 → identical boxes.
    """
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorised pairwise IoU: (N, 4) x (M, 4) -> (N, M)."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list[int]:
    """Greedy Non-Maximum Suppression. Returns indices of kept boxes, best first.

    1. Sort candidates by score (highest first).
    2. Keep the best one.
    3. Discard every remaining box whose IoU with it exceeds `iou_threshold`
       (they're duplicates of the same object).
    4. Repeat with what's left.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        best = int(order[0])
        keep.append(best)
        if order.size == 1:
            break
        overlaps = iou_matrix(boxes[best][None, :], boxes[order[1:]])[0]
        order = order[1:][overlaps <= iou_threshold]
    return keep
