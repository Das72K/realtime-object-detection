"""ONNX Runtime backend with our OWN pre- and post-processing.

Why write this when Ultralytics can load .onnx files itself?
  1. To see every step the framework normally hides: letterbox, normalise,
     run the graph, decode, map boxes back to the original frame.
  2. To measure how much of the latency is the *runtime* vs the *Python wrapper*.
     The benchmark runs the same .onnx file through Ultralytics and through this
     thin class; the difference is pure framework overhead.

Supports both output layouts:
  * End-to-end (YOLO26, YOLOv10):  (1, N, 6)  rows = x1, y1, x2, y2, score, class
                                   already de-duplicated -> no NMS
  * Classic (YOLOv8 / YOLO11):     (1, 4 + C, A) rows = cx, cy, w, h, C class scores
                                   for A anchors -> needs NMS (we use geometry.nms)
"""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import nms
from .types import Detection, Timing


# --------------------------------------------------------------------------- #
# Letterbox: resize keeping aspect ratio, pad the rest with grey (114)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class LetterboxInfo:
    ratio: float   # scale applied to the original frame
    pad_x: int     # left padding in network pixels
    pad_y: int     # top padding in network pixels


def letterbox(img: np.ndarray, new_hw: tuple[int, int], color: int = 114) -> tuple[np.ndarray, LetterboxInfo]:
    """Fit `img` into `new_hw` without distortion, centred, padding with grey.

    Mirrors Ultralytics' LetterBox(auto=False, center=True) so our ONNX path sees
    exactly the same pixels as the PyTorch path -> fair benchmark, parity check.
    """
    h, w = img.shape[:2]
    nh, nw = new_hw
    r = min(nh / h, nw / w)
    rh, rw = int(round(h * r)), int(round(w * r))
    dh, dw = (nh - rh) / 2, (nw - rw) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    if (rh, rw) != (h, w):
        img = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    out = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color,) * 3)
    return out, LetterboxInfo(r, left, top)


def rect_shape(h: int, w: int, imgsz: int = 640, stride: int = 32) -> tuple[int, int]:
    """Smallest stride-multiple rectangle Ultralytics' .pt path uses for an (h, w) frame.

    1280x720 -> scale to 640x360 -> pad up to multiples of 32 -> (384, 640).
    """
    r = imgsz / max(h, w)
    nh, nw = round(h * r), round(w * r)
    return int(np.ceil(nh / stride) * stride), int(np.ceil(nw / stride) * stride)


def scale_boxes(boxes: np.ndarray, info: LetterboxInfo, orig_hw: tuple[int, int]) -> np.ndarray:
    """Undo the letterbox: network pixels -> original frame pixels, clipped."""
    b = boxes.astype(np.float64).copy()
    b[:, [0, 2]] = (b[:, [0, 2]] - info.pad_x) / info.ratio
    b[:, [1, 3]] = (b[:, [1, 3]] - info.pad_y) / info.ratio
    h, w = orig_hw
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    return b


def to_tensor(img_bgr: np.ndarray, dtype=np.float32) -> np.ndarray:
    """BGR HWC uint8 -> RGB NCHW float in [0, 1], contiguous."""
    x = img_bgr[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW (views, no copy yet)
    x = np.ascontiguousarray(x, dtype=dtype)
    x /= 255.0
    return x[None]


# --------------------------------------------------------------------------- #
# Output decoding
# --------------------------------------------------------------------------- #
def decode_end2end(out: np.ndarray, conf: float, classes: list[int] | None = None) -> np.ndarray:
    """(1, N, 6) -> (K, 6) [x1, y1, x2, y2, score, cls] in network pixels."""
    d = out[0]
    d = d[d[:, 4] >= conf]
    if classes is not None:
        d = d[np.isin(d[:, 5].astype(int), classes)]
    return d


def decode_classic(
    out: np.ndarray, conf: float, iou: float, classes: list[int] | None = None, max_det: int = 300
) -> np.ndarray:
    """(1, 4 + C, A) -> (K, 6) after confidence filter + class-aware NMS."""
    p = out[0].T                                   # (A, 4 + C)
    scores_all = p[:, 4:]
    cls = scores_all.argmax(1)
    scores = scores_all[np.arange(len(p)), cls]
    keep = scores >= conf
    if classes is not None:
        keep &= np.isin(cls, classes)
    p, cls, scores = p[keep], cls[keep], scores[keep]
    if len(p) == 0:
        return np.zeros((0, 6))
    cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    # Class-aware NMS in ONE call: shift each class's boxes far apart so boxes of
    # different classes can never overlap (a person never suppresses a bicycle).
    offset = cls[:, None] * 7680.0
    idx = nms(boxes + offset, scores, iou)[:max_det]
    return np.concatenate([boxes[idx], scores[idx, None], cls[idx, None].astype(float)], 1)


# --------------------------------------------------------------------------- #
# Session creation (GPU if possible)
# --------------------------------------------------------------------------- #
def make_session(path: str, device: str = "cpu"):
    import onnxruntime as ort

    want_gpu = device not in ("cpu", None)
    if want_gpu:
        try:  # make CUDA/cuDNN DLLs visible: torch's bundled ones, then ORT's own finder
            import torch  # noqa: F401
        except Exception:
            pass
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass
    available = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]
    if want_gpu and "CUDAExecutionProvider" in available:
        dev_id = int(device) if str(device).isdigit() else 0
        providers = [("CUDAExecutionProvider", {"device_id": dev_id}), "CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, sess_options=so, providers=providers)
    if want_gpu and sess.get_providers()[0] != "CUDAExecutionProvider":
        print("[rtod] WARNING: ONNX Runtime fell back to CPU. onnxruntime-gpu must be built for the "
              "same CUDA major version as PyTorch (torch cu12x -> onnxruntime-gpu>=1.21,<1.27) — see README.")
    return sess


class OnnxDetector:
    """Same interface as YoloDetector: detect(frame) -> (dets, Timing)."""

    def __init__(
        self,
        path: str,
        device: str = "cpu",
        conf: float = 0.25,
        iou: float = 0.45,
        classes: list[int] | None = None,
        session=None,  # injectable for tests
    ) -> None:
        self.path = path
        self.device = device
        self.conf, self.iou, self.classes = conf, iou, classes
        self.session = session or make_session(path, device)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_hw = (int(inp.shape[2]), int(inp.shape[3]))
        self.dtype = np.float16 if "float16" in inp.type else np.float32
        meta = self._metadata()
        self.names: dict[int, str] = ast.literal_eval(meta["names"]) if "names" in meta else {}
        self.provider = self.session.get_providers()[0]
        self._warm = False

    def _metadata(self) -> dict:
        try:
            return dict(self.session.get_modelmeta().custom_metadata_map)
        except Exception:
            return {}

    def warmup(self, frame: np.ndarray | None = None, runs: int = 3) -> None:
        frame = np.zeros((*self.input_hw, 3), np.uint8) if frame is None else frame
        for _ in range(runs):
            self.detect(frame)
        self._warm = True

    def ensure_warm(self, frame: np.ndarray) -> None:
        # static input shape: one warmup covers every frame size
        if not self._warm:
            self.warmup(frame)

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], Timing]:
        t0 = time.perf_counter()
        img, info = letterbox(frame, self.input_hw)
        x = to_tensor(img, self.dtype)
        t1 = time.perf_counter()
        out = self.session.run(None, {self.input_name: x})[0]
        t2 = time.perf_counter()
        if out.ndim == 3 and out.shape[2] == 6:
            d = decode_end2end(out, self.conf, self.classes)
        else:
            d = decode_classic(out, self.conf, self.iou, self.classes)
        dets: list[Detection] = []
        if len(d):
            boxes = scale_boxes(d[:, :4], info, frame.shape[:2])
            for b, s, c in zip(boxes, d[:, 4], d[:, 5].astype(int)):
                dets.append(Detection(float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                                      float(s), int(c), self.names.get(int(c), str(c))))
        t3 = time.perf_counter()
        return dets, Timing((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3)
