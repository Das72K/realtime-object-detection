"""Box geometry: IoU and a reference NMS implementation.

These are written from scratch on purpose. Ultralytics does all of this
internally, but the tracker and the raw ONNX backend need IoU and NMS
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


# --------------------------------------------------------------------------- #
# 2-D primitives for line / zone counting
# --------------------------------------------------------------------------- #
Point = tuple[float, float]


def cross(o: Point, a: Point, b: Point) -> float:
    """z-component of (a - o) x (b - o).

    > 0 : b is on one side of the directed line o->a
    < 0 : the other side
    = 0 : collinear
    Image coordinates have y pointing DOWN, so on screen a positive value means
    b is on the RIGHT-hand side of the arrow o->a (the mirror image of the
    maths-class convention, where positive means left).
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def side_of_line(p1: Point, p2: Point, pt: Point) -> int:
    """Sign of `pt` relative to the infinite line through p1 -> p2: -1, 0 or +1."""
    c = cross(p1, p2, pt)
    return (c > 0) - (c < 0)


def _on_segment(p: Point, q: Point, r: Point) -> bool:
    """Assuming p, q, r are collinear: is q within the bounding box of segment p-r?"""
    return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])


def segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    """Do closed segments a1-a2 and b1-b2 share at least one point?

    Classic orientation test: the segments cross iff each one's endpoints lie on
    opposite sides of the other one (plus collinear/touching special cases).
    """
    d1 = side_of_line(b1, b2, a1)
    d2 = side_of_line(b1, b2, a2)
    d3 = side_of_line(a1, a2, b1)
    d4 = side_of_line(a1, a2, b2)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True
    if d1 == 0 and _on_segment(b1, a1, b2):
        return True
    if d2 == 0 and _on_segment(b1, a2, b2):
        return True
    if d3 == 0 and _on_segment(a1, b1, a2):
        return True
    if d4 == 0 and _on_segment(a1, b2, a2):
        return True
    return False


def point_in_polygon(pt: Point, polygon: list[Point]) -> bool:
    """Ray casting: shoot a ray to the right, count edge crossings. Odd = inside."""
    x, y = pt
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):  # edge straddles the horizontal through pt
            x_at_y = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_at_y:
                inside = not inside
    return inside
