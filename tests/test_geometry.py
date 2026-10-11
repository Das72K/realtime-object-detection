import numpy as np
import pytest

from rtod.geometry import iou, iou_matrix, nms


def test_iou_identical_boxes_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_touching_edges_is_zero():
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_iou_half_overlap():
    # A = 10x10, B shifted 5 right -> intersection 50, union 150
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_iou_contained_box():
    # small 5x5 inside 10x10 -> 25 / 100
    assert iou((0, 0, 10, 10), (2, 2, 7, 7)) == pytest.approx(0.25)


def test_iou_degenerate_box():
    assert iou((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0


def test_iou_matrix_matches_scalar():
    a = np.array([[0, 0, 10, 10], [5, 5, 15, 15]])
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [5, 0, 15, 10]])
    m = iou_matrix(a, b)
    assert m.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            assert m[i, j] == pytest.approx(iou(a[i], b[j]))


def test_iou_matrix_empty():
    assert iou_matrix(np.zeros((0, 4)), np.array([[0, 0, 1, 1]])).shape == (0, 1)


def test_nms_removes_duplicates_keeps_best():
    boxes = np.array([
        [100, 100, 200, 200],  # person, best
        [105, 102, 205, 198],  # duplicate of the same person
        [300, 300, 400, 400],  # different object
    ])
    scores = np.array([0.9, 0.75, 0.8])
    assert nms(boxes, scores, iou_threshold=0.5) == [0, 2]


def test_nms_threshold_controls_aggressiveness():
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10]])  # IoU = 0.333
    scores = np.array([0.9, 0.8])
    assert nms(boxes, scores, iou_threshold=0.5) == [0, 1]   # lenient: both survive
    assert nms(boxes, scores, iou_threshold=0.3) == [0]      # strict: duplicate removed


def test_nms_empty():
    assert nms(np.zeros((0, 4)), np.zeros(0)) == []
