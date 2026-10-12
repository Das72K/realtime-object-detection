"""Day-1 environment check: Python, OpenCV, torch/CUDA, ONNX Runtime, webcam.

    python scripts/check_env.py            # everything
    python scripts/check_env.py --no-cam   # skip webcam probe (WSL)
"""

from __future__ import annotations

import platform
import sys


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def main() -> None:
    print("== System ==")
    ok(f"{platform.system()} {platform.release()} | Python {sys.version.split()[0]}")
    if "microsoft" in platform.release().lower():
        warn("Running inside WSL — webcams are not available here; use video files or run on Windows.")
    if sys.version_info < (3, 11):
        warn("Python >= 3.11 required")

    print("== OpenCV ==")
    try:
        import cv2

        ok(f"opencv {cv2.__version__}")
        gui = "GTK" in cv2.getBuildInformation() or platform.system() == "Windows"
        (ok if gui else warn)(f"GUI (imshow) support: {'yes' if gui else 'maybe not — use --no-show'}")
    except ImportError:
        warn("opencv-python not installed")

    print("== PyTorch ==")
    try:
        import torch

        ok(f"torch {torch.__version__}")
        if torch.cuda.is_available():
            ok(f"CUDA available: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
        else:
            warn("CUDA not available -> CPU mode (fine; see README for the CUDA wheel if you have an NVIDIA GPU)")
    except ImportError:
        warn("torch not installed")

    print("== Ultralytics ==")
    try:
        import ultralytics

        ok(f"ultralytics {ultralytics.__version__}")
    except ImportError:
        warn("ultralytics not installed")

    print("== ONNX Runtime ==")
    try:
        import onnxruntime as ort

        ok(f"onnxruntime {ort.__version__} providers={ort.get_available_providers()}")
    except ImportError:
        warn("onnxruntime not installed (needed for the ONNX backend and benchmark: pip install -e .[onnx])")

    if "--no-cam" not in sys.argv:
        print("== Webcam ==")
        try:
            import cv2

            backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
            cap = cv2.VideoCapture(0, backend)
            good, frame = cap.read() if cap.isOpened() else (False, None)
            cap.release()
            if good:
                ok(f"webcam 0 delivers frames {frame.shape[1]}x{frame.shape[0]}")
            else:
                warn("webcam 0 not readable")
        except Exception as e:  # noqa: BLE001
            warn(f"webcam probe failed: {e}")


if __name__ == "__main__":
    main()
