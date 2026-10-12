import pytest

from rtod.bench import BenchResult, markdown_table, parity, percentile, summarize
from rtod.types import Detection


def d(x, cls=0):
    return Detection(x, 0, x + 10, 10, 0.9, cls, "person" if cls == 0 else "car")


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([], 50) == 0.0


def test_summarize_uses_median_of_repeats():
    runs = [{"total": [10] * 10, "pre": [1] * 10, "inf": [8] * 10, "post": [1] * 10},
            {"total": [12] * 10, "pre": [1] * 10, "inf": [10] * 10, "post": [1] * 10},
            {"total": [50] * 10, "pre": [1] * 10, "inf": [48] * 10, "post": [1] * 10}]  # one noisy repeat
    s = summarize(runs)
    assert s["total_p50"] == 12            # median ignores the outlier repeat
    assert s["spread_p50"] == 40           # ...but the spread exposes it
    assert s["inf_p50"] == 10


def test_parity_identical():
    frames = [[d(0), d(50, 2)], [d(5)]]
    assert parity(frames, frames) == (pytest.approx(1.0), pytest.approx(1.0))


def test_parity_missing_and_class_mismatch():
    ref = [[d(0), d(50)]]
    test = [[d(1, cls=0), d(50, cls=2)]]  # 2nd box: right place, wrong class
    f1, miou = parity(ref, test)
    assert f1 == pytest.approx(0.5)       # 1 match out of (2 + 2) / 2
    assert 0.8 < miou < 1.0


def test_parity_empty_frames():
    assert parity([[]], [[]]) == (1.0, 1.0)


def _result(**kw):
    base = dict(model="yolo26m", backend="ort", device="0", provider="CUDAExecutionProvider",
                input_hw="384x640", frames=300, repeats=3, total_p50=6.2, total_p95=7.9,
                total_mean=6.5, spread_p50=0.3, pre_p50=1.1, inf_p50=4.5, post_p50=0.2, fps=153.8)
    base.update(kw)
    return BenchResult(**base)


def test_markdown_table_row():
    md = markdown_table([_result(parity_f1=0.998, map50_95=None)])
    assert "| yolo26m | ONNX Runtime (raw) | 0 | **6.2** | 7.9 | 154 |" in md
    assert "0.998" in md and "–" in md


def test_plots_render(tmp_path):
    pytest.importorskip("matplotlib")
    from rtod.bench import plot_latency, plot_tradeoff

    rs = [_result(backend=b, model=m, total_p50=p, map50_95=a)
          for b in ("torch", "onnx", "ort") for m, p, a in (("yolo26n", 12, 0.40), ("yolo26m", 13, 0.52))]
    assert plot_latency(rs, tmp_path / "lat.png", "0").stat().st_size > 10_000
    assert plot_tradeoff(rs, tmp_path / "trade.png", "0").stat().st_size > 10_000
    assert plot_latency(rs, tmp_path / "x.png", "cpu") is None
