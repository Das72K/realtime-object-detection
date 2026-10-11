"""CPU-first device selection: use a GPU only if one is actually usable."""

from __future__ import annotations


def select_device(preferred: str | None = None) -> str:
    """Return an Ultralytics device string: '0' (CUDA), 'mps' (Apple) or 'cpu'."""
    if preferred:
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # torch missing or broken CUDA install -> fall back safely
        pass
    return "cpu"


def describe_device(device: str) -> str:
    if device == "cpu":
        return "CPU"
    try:
        import torch

        if device.isdigit():
            return f"CUDA:{device} ({torch.cuda.get_device_name(int(device))})"
    except Exception:
        pass
    return device
