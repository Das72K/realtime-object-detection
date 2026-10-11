"""Core data types shared by every stage of the pipeline.

The detector, tracker, counter and exporter all speak `Detection`, so any
backend (PyTorch, ONNX Runtime, ...) only has to produce this one shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Detection:
    """One object in one frame, in pixel coordinates of the ORIGINAL frame."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    track_id: int | None = None  # filled in by the tracker (Phase 2)

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Anchor point for line/zone counting: where the object touches the ground."""
        return ((self.x1 + self.x2) / 2.0, self.y2)


@dataclass(slots=True)
class Timing:
    """Per-frame latency breakdown in milliseconds."""

    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def model_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms
