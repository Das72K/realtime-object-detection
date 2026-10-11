"""Throughput and latency measurement."""

from __future__ import annotations

import time
from collections import deque


class FpsMeter:
    """Smoothed frames-per-second using an exponential moving average.

    Raw 1/dt jumps around every frame; the EMA gives a readable number:
        fps_ema = alpha * fps_now + (1 - alpha) * fps_ema
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._last: float | None = None
        self.fps: float = 0.0

    def tick(self, now: float | None = None) -> float:
        now = time.perf_counter() if now is None else now
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0.0 else self.alpha * inst + (1 - self.alpha) * self.fps
        self._last = now
        return self.fps


class LatencyStats:
    """Rolling window of latencies (ms) -> mean / p50 / p95."""

    def __init__(self, window: int = 300) -> None:
        self._values: deque[float] = deque(maxlen=window)

    def add(self, ms: float) -> None:
        self._values.append(float(ms))

    def __len__(self) -> int:
        return len(self._values)

    def mean(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0

    def percentile(self, p: float) -> float:
        if not self._values:
            return 0.0
        data = sorted(self._values)
        k = (len(data) - 1) * (p / 100.0)
        lo, hi = int(k), min(int(k) + 1, len(data) - 1)
        return data[lo] + (data[hi] - data[lo]) * (k - lo)


class Stopwatch:
    """`with Stopwatch() as sw: ...` then read `sw.ms`."""

    def __enter__(self) -> "Stopwatch":
        self._t0 = time.perf_counter()
        self.ms = 0.0
        return self

    def __exit__(self, *exc) -> None:
        self.ms = (time.perf_counter() - self._t0) * 1000.0
