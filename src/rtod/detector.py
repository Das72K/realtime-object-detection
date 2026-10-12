"""YOLO detector backend (PyTorch via Ultralytics).

onnx_backend.OnnxDetector has the same `detect()` signature, so
the rest of the pipeline never knows which one it is talking to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .device import select_device
from .types import Detection, Timing


@dataclass(slots=True)
class DetectorConfig:
    weights: str = "yolo26n.pt"   # n / s / m / l / x  — size vs accuracy knob
    conf: float = 0.25            # drop detections below this confidence
    iou: float = 0.45             # NMS IoU threshold (ignored by NMS-free YOLO26 heads)
    imgsz: int = 640              # network input size (frame is letterboxed to this)
    device: str | None = None     # None -> auto (CUDA if available, else CPU)
    classes: list[int] | None = None  # e.g. [0] = person only
    half: bool = False            # FP16 — GPU only
    tracker: str = "bytetrack.yaml"  # Ultralytics tracker config (bytetrack.yaml | botsort.yaml)


class YoloDetector:
    def __init__(self, cfg: DetectorConfig | None = None) -> None:
        from ultralytics import YOLO  # imported lazily so unit tests don't need torch

        self.cfg = cfg or DetectorConfig()
        self.device = select_device(self.cfg.device)
        self.model = YOLO(self.cfg.weights)
        self.names: dict[int, str] = dict(self.model.names)
        self._warm_shapes: set[tuple[int, int]] = set()

    def warmup(self, frame: np.ndarray | None = None, runs: int = 3) -> None:
        """Run a few untimed inferences so later timings are steady-state.

        First calls pay one-off costs (CUDA context, lazy kernel loading, cuDNN
        algorithm choice). Those costs are per *network input shape*, and for
        .pt models Ultralytics letterboxes to the smallest stride-32 rectangle:
        16:9 -> 384x640, 4:3 -> 480x640, portrait 3:4 -> 640x480. So warm up on
        a frame with the real aspect ratio, not an arbitrary dummy.
        """
        if frame is None:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for _ in range(runs):
            self._predict(frame)
        self._warm_shapes.add(frame.shape[:2])

    def ensure_warm(self, frame: np.ndarray) -> None:
        """Warm up once per new frame shape (folders of mixed images, new cameras)."""
        if frame.shape[:2] not in self._warm_shapes:
            self.warmup(frame)

    def _kwargs(self) -> dict:
        kwargs = dict(
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            device=self.device,
            classes=self.cfg.classes,
            verbose=False,
        )
        if self.cfg.half and self.device != "cpu":
            kwargs["half"] = True  # only pass when wanted: newer Ultralytics warns on every call otherwise
        return kwargs

    def _predict(self, frame: np.ndarray):
        return self.model.predict(frame, **self._kwargs())[0]

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], Timing]:
        """Run one BGR frame (as read by OpenCV) through the model."""
        return self._to_detections(self._predict(frame))

    def track(self, frame: np.ndarray) -> tuple[list[Detection], Timing]:
        """Detect + associate with the tracker. Call on consecutive frames of ONE stream.

        persist=True keeps the tracker state between calls (otherwise every frame
        would start fresh and IDs would never persist). Create a new detector (or
        call reset_tracker) when switching to a different video.
        """
        result = self.model.track(frame, persist=True, tracker=self.cfg.tracker, **self._kwargs())[0]
        return self._to_detections(result)

    def reset_tracker(self) -> None:
        predictor = getattr(self.model, "predictor", None)
        for t in getattr(predictor, "trackers", []) or []:
            t.reset()

    def _to_detections(self, result) -> tuple[list[Detection], Timing]:
        speed = result.speed or {}
        timing = Timing(
            preprocess_ms=float(speed.get("preprocess", 0.0)),
            inference_ms=float(speed.get("inference", 0.0)),
            postprocess_ms=float(speed.get("postprocess", 0.0)),
        )

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return [], timing

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        ids = getattr(boxes, "id", None)
        ids = ids.cpu().numpy().astype(int) if ids is not None else [None] * len(xyxy)
        dets = [
            Detection(
                x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]),
                confidence=float(c), class_id=int(k), class_name=self.names.get(int(k), str(k)),
                track_id=None if i is None else int(i),
            )
            for b, c, k, i in zip(xyxy, confs, cls, ids)
        ]
        return dets, timing
