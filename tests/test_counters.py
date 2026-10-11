import pytest

from rtod.counting import (
    LineCounter, ZoneCounter, anchor_point, denormalize, parse_line_spec, parse_zone_spec,
)
from rtod.types import Detection


def box_at(cx, bottom, tid, cls="person", w=20, h=40):
    """Box whose bottom-centre (the counting anchor) is (cx, bottom)."""
    return Detection(cx - w / 2, bottom - h, cx + w / 2, bottom, 0.9, 0, cls, track_id=tid)


def run_line(lc, path, tid=1, cls="person"):
    events = []
    for f, (x, y) in enumerate(path):
        events += lc.update([box_at(x, y, tid, cls)], f, f / 30)
    return events


def line():
    return LineCounter("door", (0, 100), (200, 100))  # horizontal, left -> right


def test_crossing_down_is_in():
    lc = line()
    ev = run_line(lc, [(100, 80), (100, 95), (100, 110), (100, 130)])
    assert (lc.count_in, lc.count_out) == (1, 0)
    assert len(ev) == 1 and ev[0].event == "in" and ev[0].track_id == 1 and ev[0].frame == 2


def test_crossing_up_is_out():
    lc = line()
    run_line(lc, [(100, 130), (100, 90)])
    assert (lc.count_in, lc.count_out) == (0, 1)


def test_crossing_outside_segment_not_counted():
    lc = line()
    run_line(lc, [(300, 80), (300, 130)])  # crosses y=100 but at x=300 (> segment end 200)
    assert (lc.count_in, lc.count_out) == (0, 0)


def test_approaching_without_crossing():
    lc = line()
    run_line(lc, [(100, 50), (100, 90), (100, 99), (100, 60)])
    assert (lc.count_in, lc.count_out) == (0, 0)


def test_jitter_on_line_counted_once_per_direction():
    lc = line()
    run_line(lc, [(100, 98), (100, 102), (100, 98), (100, 102), (100, 98), (100, 102)])
    assert (lc.count_in, lc.count_out) == (1, 1)


def test_jitter_counts_every_time_without_debounce():
    lc = LineCounter("door", (0, 100), (200, 100), once_per_track=False)
    run_line(lc, [(100, 98), (100, 102), (100, 98), (100, 102)])
    assert (lc.count_in, lc.count_out) == (2, 1)


def test_landing_exactly_on_line_then_crossing():
    lc = line()
    run_line(lc, [(100, 90), (100, 100), (100, 100), (100, 110)])
    assert lc.count_in == 1


def test_multiple_tracks_and_classes():
    lc = line()
    for f, (y1, y2) in enumerate([(80, 130), (120, 120), (120, 80)]):
        lc.update([box_at(50, y1, 1, "person"), box_at(150, y2, 2, "car")], f, f / 30)
    # frame 0: car #2 starts below; frame 1: person crosses down; frame 2: car crosses up
    assert lc.count_in == 1 and lc.count_out == 1
    assert lc.summary()["by_class"] == {"car": {"in": 0, "out": 1}, "person": {"in": 1, "out": 0}}


def test_untracked_detections_ignored():
    lc = line()
    for f, y in enumerate([80, 130]):
        lc.update([Detection(90, y - 40, 110, y, 0.9, 0, "person")], f, 0.0)
    assert lc.count_in == 0


def test_anchor_matters():
    # tall box: bottom crosses, centre does not
    d = Detection(90, 60, 110, 120, 0.9, 0, "person", track_id=1)
    assert anchor_point(d, "bottom") == (100, 120)
    assert anchor_point(d, "center") == (100, 90)


# ---------------------------------------------------------------- zones
ZONE = [(100, 100), (200, 100), (200, 200), (100, 200)]


def test_zone_enter_exit_dwell():
    z = ZoneCounter("desk", ZONE)
    seq = [(50, 150), (150, 150), (150, 160), (150, 170), (250, 150)]
    ev = []
    for f, (x, y) in enumerate(seq):
        ev += z.update([box_at(x, y, 7)], f, t=float(f))
    assert [e.event for e in ev] == ["enter", "exit"]
    assert ev[1].dwell_s == pytest.approx(3.0)  # entered t=1, left t=4
    assert z.occupancy == 0 and len(z.unique_ids) == 1


def test_zone_occupancy_counts_concurrent_tracks():
    z = ZoneCounter("desk", ZONE)
    z.update([box_at(150, 150, 1), box_at(120, 180, 2), box_at(300, 300, 3)], 0, 0.0)
    assert z.occupancy == 2


def test_zone_reentry_is_not_a_new_unique_entry():
    z = ZoneCounter("desk", ZONE)
    for f, x in enumerate([150, 250, 150]):
        z.update([box_at(x, 150, 1)], f, float(f))
    assert len(z.unique_ids) == 1 and len(z.dwell_times) == 1 and z.occupancy == 1


def test_zone_lost_track_times_out():
    z = ZoneCounter("desk", ZONE, grace=5)
    z.update([box_at(150, 150, 1)], 0, 0.0)
    for f in range(1, 7):
        ev = z.update([], f, float(f))
    assert z.occupancy == 0 and ev and ev[0].event == "exit"


# ---------------------------------------------------------------- spec parsing
def test_parse_line_spec_named_and_default():
    assert parse_line_spec("door:0.1,0.5,0.9,0.5") == ("door", [(0.1, 0.5), (0.9, 0.5)])
    assert parse_line_spec("0,0,1,1", index=2)[0] == "line3"


def test_parse_zone_spec():
    name, pts = parse_zone_spec("desk:0.1,0.1;0.5,0.1;0.5,0.5")
    assert name == "desk" and pts == [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5)]


@pytest.mark.parametrize("bad", ["0,0,1", "door:0,0,2,1"])
def test_parse_line_spec_rejects_bad(bad):
    with pytest.raises(ValueError):
        parse_line_spec(bad)


def test_parse_zone_needs_three_points():
    with pytest.raises(ValueError):
        parse_zone_spec("0,0;1,1")


def test_denormalize():
    assert denormalize([(0.5, 0.25)], 1280, 720) == [(640.0, 180.0)]
