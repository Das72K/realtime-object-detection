import csv
import json

from rtod.counting import CountEvent
from rtod.export import ResultsExporter
from rtod.tracking import IouTracker
from rtod.types import Detection


def det(x, y, cls=0, name="person", w=40, h=80):
    return Detection(x, y, x + w, y + h, 0.9, cls, name)


def test_tracker_keeps_id_while_object_moves():
    trk = IouTracker(iou_threshold=0.3)
    ids = [trk.update([det(100 + 5 * f, 100)])[0].track_id for f in range(10)]
    assert set(ids) == {1}


def test_tracker_two_objects_get_distinct_stable_ids():
    trk = IouTracker()
    first = trk.update([det(100, 100), det(400, 100)])
    second = trk.update([det(404, 100), det(104, 100)])  # order swapped in the detector output
    a = {round(d.x1): d.track_id for d in first}
    b = {round(d.x1): d.track_id for d in second}
    assert a[100] == b[104] and a[400] == b[404] and a[100] != a[400]


def test_tracker_new_id_after_large_jump():
    trk = IouTracker()
    a = trk.update([det(100, 100)])[0].track_id
    b = trk.update([det(600, 400)])[0].track_id  # no overlap -> different object
    assert a != b


def test_tracker_bridges_short_gap_then_forgets():
    trk = IouTracker(max_age=3)
    a = trk.update([det(100, 100)])[0].track_id
    for _ in range(2):
        trk.update([])
    assert trk.update([det(102, 100)])[0].track_id == a  # back after 2 missed frames
    for _ in range(5):
        trk.update([])
    assert trk.active_tracks == 0
    assert trk.update([det(102, 100)])[0].track_id != a


def test_tracker_does_not_match_across_classes():
    trk = IouTracker()
    a = trk.update([det(100, 100, 0, "person")])[0].track_id
    b = trk.update([det(100, 100, 2, "car")])[0].track_id
    assert a != b


def test_exporter_writes_csv_and_json(tmp_path):
    with ResultsExporter(tmp_path, "run") as ex:
        ex.log_detections(0, 0.0, [Detection(1, 2, 3, 4, 0.5, 0, "person", track_id=7), det(5, 5)])
        ex.log_events([CountEvent("line", "door", "in", 7, "person", 3, 0.1), CountEvent(
            "zone", "desk", "exit", 7, "person", 9, 0.3, 0.2)])
    ex.write_summary({"lines": {"door": {"in": 1}}, "ids": {1, 2}})

    rows = list(csv.DictReader(open(ex.det_path, encoding="utf-8")))
    assert len(rows) == 2 and rows[0]["track_id"] == "7" and rows[1]["track_id"] == ""
    events = list(csv.DictReader(open(ex.event_path, encoding="utf-8")))
    assert [e["event"] for e in events] == ["in", "exit"] and events[1]["dwell_s"] == "0.2"
    summary = json.loads(ex.summary_path.read_text(encoding="utf-8"))
    assert summary["lines"]["door"]["in"] == 1 and sorted(summary["ids"]) == [1, 2]
