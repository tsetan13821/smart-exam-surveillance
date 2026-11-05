"""PersonTracker landmark handling and session setup helpers."""
import json
from types import SimpleNamespace

import pytest

from sim import mp_result


def mp_with_nose(x, y, w=640, h=480):
    """MediaPipe-like result whose nose tip (landmark 1) is at pixel (x, y)."""
    landmarks = [SimpleNamespace(x=0.5, y=0.5)] * 478
    landmarks = list(landmarks)
    landmarks[1] = SimpleNamespace(x=x / w, y=y / h)
    r = mp_result()
    r["landmarks_object"] = SimpleNamespace(landmark=landmarks)
    return r


def test_tracker_drops_landmarks_of_people_it_did_not_see(clock, logger):
    from app.processors.person_tracker import PersonTracker
    tracker = PersonTracker(logger)
    box = [250, 150, 360, 300]
    persons = tracker.update_tracks({"boxes": [box], "names": ["A"]}, [mp_with_nose(305, 220)], 640, 480, None)
    person = next(iter(persons.values()))
    assert person.mp_result is not None

    clock.advance(0.5)
    persons = tracker.update_tracks({"boxes": [], "names": []}, [], 640, 480, None)
    assert person.id in persons            # still held by the tracker for a few seconds...
    assert person.mp_result is None        # ...but without stale landmarks


def test_load_desk_rois(tmp_path):
    main = pytest.importorskip("main")
    path = tmp_path / "desk_rois.json"
    assert main.load_desk_rois(str(tmp_path / "missing.json")) == {}
    path.write_text("")
    assert main.load_desk_rois(str(path)) == {}
    path.write_text(json.dumps({"Alice": [[0, 0], [10, 0], [10, 10]], "Bad": [[0, 0]]}))
    assert main.load_desk_rois(str(path)) == {"Alice": [[0, 0], [10, 0], [10, 10]]}
