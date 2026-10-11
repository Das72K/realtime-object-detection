import pytest

from rtod.metrics import FpsMeter, LatencyStats
from rtod.types import Detection, Timing


def test_fps_meter_steady_rate():
    m = FpsMeter(alpha=0.5)
    t = 0.0
    for _ in range(50):
        m.tick(t)
        t += 1 / 30
    assert m.fps == pytest.approx(30.0, rel=1e-3)


def test_fps_meter_first_tick_is_zero():
    assert FpsMeter().tick(1.0) == 0.0


def test_fps_meter_rejects_bad_alpha():
    with pytest.raises(ValueError):
        FpsMeter(alpha=0)


def test_latency_stats_percentiles():
    s = LatencyStats()
    for v in range(1, 101):  # 1..100 ms
        s.add(v)
    assert s.mean() == pytest.approx(50.5)
    assert s.percentile(50) == pytest.approx(50.5)
    assert s.percentile(95) == pytest.approx(95.05)


def test_latency_stats_window_drops_old_values():
    s = LatencyStats(window=3)
    for v in (1000, 1, 2, 3):
        s.add(v)
    assert len(s) == 3 and s.mean() == pytest.approx(2.0)


def test_detection_geometry():
    d = Detection(10, 20, 50, 100, 0.9, 0, "person")
    assert (d.width, d.height, d.area) == (40, 80, 3200)
    assert d.center == (30, 60)
    assert d.bottom_center == (30, 100)


def test_timing_sum():
    assert Timing(1.0, 10.0, 2.0).model_ms == pytest.approx(13.0)
