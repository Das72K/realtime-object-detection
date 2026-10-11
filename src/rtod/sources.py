"""Unified frame input: webcam index, video file, image file, or folder of images."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


@dataclass(slots=True)
class Frame:
    image: np.ndarray
    index: int
    timestamp_s: float  # position in the stream (video) or 0 for images
    name: str           # source name, used for output files


class FrameSource:
    """Iterate frames from any supported source.

    `kind` is one of: 'webcam', 'video', 'image', 'folder'.
    """

    def __init__(
        self,
        source: str | int,
        width: int | None = None,
        height: int | None = None,
        cam_backend: str = "auto",
    ) -> None:
        self.raw = source
        self._cap: cv2.VideoCapture | None = None
        self._images: list[Path] = []
        self.backend = "file"

        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            self.kind = "webcam"
            idx = int(source)
            self.name = f"webcam{idx}"
            self._cap, self.backend = self._open_webcam(idx, width, height, cam_backend)
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(
                    f"Source not found: {path.resolve()}\n"
                    "  -> put a clip there first (see README: 'Getting a test video')."
                )
            self.name = path.stem
            if path.is_dir():
                self.kind = "folder"
                self._images = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
                if not self._images:
                    raise ValueError(f"No images in folder: {path}")
            elif path.suffix.lower() in IMAGE_EXTS:
                self.kind = "image"
                self._images = [path]
            elif path.suffix.lower() in VIDEO_EXTS:
                self.kind = "video"
                self._cap = cv2.VideoCapture(str(path))
            else:
                raise ValueError(f"Unsupported file type: {path.suffix}")

        if self.kind == "video" and not self._cap.isOpened():
            raise RuntimeError(f"Could not open video {source!r} (corrupt file or unsupported codec).")

    @staticmethod
    def _open_webcam(
        idx: int, width: int | None, height: int | None, cam_backend: str
    ) -> tuple[cv2.VideoCapture, str]:
        """Try capture backends in order until one actually delivers a frame.

        On Windows, DirectShow is fast to open but some OpenCV builds/cameras refuse
        it ("can't be used to capture by index"); Media Foundation (MSMF) is the
        modern default. We test-read a frame, because isOpened() can lie.
        """
        backends = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}
        if cam_backend != "auto":
            order = [cam_backend]
        elif platform.system() == "Windows":
            order = ["msmf", "dshow", "any"]
        else:
            order = ["any"]

        tried = []
        for name in order:
            cap = cv2.VideoCapture(idx, backends[name])
            if cap.isOpened():
                if width:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                if height:
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always process the freshest frame
                ok, _ = cap.read()
                if ok:
                    return cap, name
            cap.release()
            tried.append(name)

        hint = ""
        if platform.system() == "Windows":
            hint = (
                "\n  Check: Settings > Privacy & security > Camera > 'Let desktop apps access your camera' = On;"
                "\n         close Teams/Zoom/Camera app (they lock the device);"
                "\n         try another index: --source 1"
            )
        elif "microsoft" in platform.release().lower():
            hint = "\n  WSL has no webcam access: run on Windows, or use a video file."
        raise RuntimeError(f"Could not read webcam {idx} (tried backends: {', '.join(tried)}).{hint}")

    @property
    def is_live(self) -> bool:
        return self.kind == "webcam"

    @property
    def is_stream(self) -> bool:
        return self.kind in {"webcam", "video"}

    @property
    def fps(self) -> float:
        if self._cap is None:
            return 0.0
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 1 else 30.0

    @property
    def frame_count(self) -> int:
        if self.kind == "video" and self._cap is not None:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return len(self._images)

    def __iter__(self) -> Iterator[Frame]:
        if self._cap is not None:
            idx = 0
            while True:
                ok, img = self._cap.read()
                if not ok:
                    break
                ts = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if self.kind == "video" else 0.0
                yield Frame(img, idx, ts, self.name)
                idx += 1
        else:
            for idx, p in enumerate(self._images):
                img = cv2.imread(str(p))
                if img is None:
                    continue
                yield Frame(img, idx, 0.0, p.stem)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
