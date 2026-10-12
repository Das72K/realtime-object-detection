import numpy as np
import pytest

from rtod.onnx_backend import (
    LetterboxInfo, OnnxDetector, decode_classic, decode_end2end, letterbox, rect_shape, scale_boxes, to_tensor,
)


def test_rect_shape_matches_ultralytics_rect_letterbox():
    assert rect_shape(720, 1280) == (384, 640)   # 16:9 webcam
    assert rect_shape(480, 640) == (480, 640)    # 4:3
    assert rect_shape(1080, 810) == (640, 480)   # portrait bus.jpg


def test_letterbox_16x9_into_384x640():
    img = np.full((720, 1280, 3), 200, np.uint8)
    out, info = letterbox(img, (384, 640))
    assert out.shape == (384, 640, 3)
    assert info == LetterboxInfo(0.5, 0, 12)          # 360 rows of image + 12 px grey bars top/bottom
    assert (out[:12] == 114).all() and (out[-12:] == 114).all() and (out[12:372] == 200).all()


def test_letterbox_square_target_pads_left_right_for_portrait():
    out, info = letterbox(np.zeros((640, 320, 3), np.uint8), (640, 640))
    assert info.ratio == 1.0 and info.pad_x == 160 and info.pad_y == 0 and out.shape == (640, 640, 3)


def test_scale_boxes_roundtrip():
    info = LetterboxInfo(0.5, 0, 12)
    orig = np.array([[100.0, 200.0, 300.0, 400.0]])
    net = orig * 0.5 + [0, 12, 0, 12]
    assert np.allclose(scale_boxes(net, info, (720, 1280)), orig)


def test_scale_boxes_clips_to_frame():
    b = scale_boxes(np.array([[-10.0, 0.0, 700.0, 400.0]]), LetterboxInfo(0.5, 0, 12), (720, 1280))
    assert b[0, 0] == 0 and b[0, 2] == 1280 and b[0, 1] == 0 and b[0, 3] == 720


def test_to_tensor_layout_and_scale():
    img = np.zeros((2, 3, 3), np.uint8)
    img[..., 0] = 255  # blue channel in BGR
    x = to_tensor(img)
    assert x.shape == (1, 3, 2, 3) and x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]
    assert x[0, 2].max() == 1.0 and x[0, 0].max() == 0.0  # blue ended up as the LAST channel (RGB)


def test_decode_end2end_filters_conf_and_classes():
    out = np.array([[[0, 0, 10, 10, 0.9, 0], [0, 0, 10, 10, 0.1, 0], [5, 5, 9, 9, 0.8, 2]]], np.float32)
    assert len(decode_end2end(out, 0.25)) == 2
    assert decode_end2end(out, 0.25, classes=[2])[:, 5].tolist() == [2]


def classic_output(rows):
    """rows: (cx, cy, w, h, [scores per class]) -> (1, 4 + C, A)."""
    return np.array([[*r[:4], *r[4]] for r in rows], np.float32).T[None]


def test_decode_classic_class_aware_nms():
    out = classic_output([
        (50, 50, 40, 40, [0.90, 0.0]),   # person
        (52, 50, 40, 40, [0.80, 0.0]),   # duplicate person -> suppressed
        (51, 50, 40, 40, [0.0, 0.70]),   # same place, other class -> kept (class-aware)
        (300, 300, 20, 20, [0.10, 0.0]),  # below conf
    ])
    d = decode_classic(out, conf=0.25, iou=0.45)
    assert sorted(d[:, 5].astype(int).tolist()) == [0, 1]
    assert np.allclose(d[d[:, 5] == 0][0, :4], [30, 30, 70, 70])


def test_decode_classic_empty():
    assert decode_classic(classic_output([(5, 5, 2, 2, [0.01])]), 0.25, 0.45).shape == (0, 6)


class _Inp:
    name, shape, type = "images", [1, 3, 384, 640], "tensor(float)"


class _Meta:
    custom_metadata_map = {"names": "{0: 'person', 2: 'car'}"}


class FakeSession:
    def __init__(self, out):
        self.out, self.last_input = out, None

    def get_inputs(self):
        return [_Inp()]

    def get_modelmeta(self):
        return _Meta()

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _names, feeds):
        self.last_input = feeds["images"]
        return [self.out]


def test_onnx_detector_end_to_end_with_fake_session():
    # network-space box (50, 112, 150, 212) == original (100, 200, 300, 400) for 1280x720
    out = np.array([[[50, 112, 150, 212, 0.9, 2], [0, 0, 5, 5, 0.05, 0]]], np.float32)
    sess = FakeSession(out)
    det = OnnxDetector("fake.onnx", session=sess)
    dets, timing = det.detect(np.zeros((720, 1280, 3), np.uint8))
    assert sess.last_input.shape == (1, 3, 384, 640)
    assert len(dets) == 1 and dets[0].class_name == "car" and dets[0].confidence == pytest.approx(0.9)
    assert dets[0].xyxy == pytest.approx((100, 200, 300, 400))
    assert timing.model_ms >= 0 and det.input_hw == (384, 640)
