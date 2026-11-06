"""SessionManager lifecycle with a fake engine (no models, no camera)."""
import sqlite3
import threading
import time

import numpy as np
import pytest

import app.utils.database as db
import config
from app.session_manager import (NoActiveSessionError, SessionBusyError, SessionManager,
                                 normalize_exam_context)
from app.session_setup import SessionSetupError


class FakeRegistry:
    STUDENTS = {
        "1": {"name": "Asha", "status": "approved"},
        "2": {"name": "Bilal", "status": "approved"},
        "3": {"name": "Chen", "status": "pending"},
    }

    def get_student(self, student_id):
        data = self.STUDENTS.get(student_id)
        if data is None:
            return None
        return dict(data, student_id=student_id, encodings=[np.ones(512)])


class FakeLogger:
    def __init__(self, *a):
        self.listeners, self.records, self.closed = [], [], False

    def add_listener(self, fn):
        self.listeners.append(fn)

    def _emit(self, entry):
        self.records.append(entry)
        for fn in self.listeners:
            fn(entry)

    def log_activity(self, message):
        self._emit({"level": "INFO", "alert_type": "ACTIVITY", "message": message})

    def log_alert(self, alert_type, message):
        self._emit({"level": "ALERT", "alert_type": alert_type, "message": message})

    def close(self):
        self.closed = True


class FakeEngine:
    """Stands in for ProctoringApp: 'processes' frames until stopped or out of frames."""

    def __init__(self, frames=None):
        self.stop_event = threading.Event()
        self.frames_left = frames
        self.frames = 0
        self.ran = False

    def run(self):
        self.ran = True
        while not self.stop_event.is_set():
            if self.frames_left is not None:
                if self.frames_left == 0:
                    break
                self.frames_left -= 1
            self.frames += 1
            time.sleep(0.005)

    def stop(self):
        self.stop_event.set()

    def get_status(self):
        return {"classification": "Normal", "frames_processed": self.frames}

    def wait_for_jpeg(self, after_seq=0, timeout=1.0):
        return after_seq + 1, b"jpeg"


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                    # sqlite database is a relative path
    engines = []

    def factory(plan, exam_context, logger, source):
        engine = FakeEngine(frames=source if isinstance(source, int) else None)
        engines.append(engine)
        return engine

    m = SessionManager(registry_factory=FakeRegistry, engine_factory=factory, logger_factory=FakeLogger)
    m.engines = engines
    yield m
    m.shutdown(timeout=5)


def wait_state(manager, state, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if manager.status()["state"] == state:
            return manager.status()
        time.sleep(0.01)
    raise AssertionError(f"state never became {state!r}; now {manager.status()['state']!r}")


def session_row(session_id):
    with sqlite3.connect(config.DATABASE_FILE) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()


def test_start_run_stop_records_the_session(manager):
    info = manager.start(["1", "2"], {"is_paper_exam": True})
    assert info["state"] == "starting" and info["student_names"] == ["Asha", "Bilal"]

    status = wait_state(manager, "running")
    assert status["live"]["classification"] == "Normal"
    assert session_row(info["session_id"])["status"] == "running"

    stopped = manager.stop(info["session_id"], wait=True)
    assert stopped["state"] == "ended"
    row = session_row(info["session_id"])
    assert row["status"] == "ended" and row["end_time"]
    assert '"is_paper_exam": true' in row["exam_context"]
    assert manager.status() == {"state": "idle", "session": None, "live": None,
                                "last_session": manager.status()["last_session"]}
    assert manager.status()["last_session"]["state"] == "ended"


def test_only_one_session_at_a_time(manager):
    manager.start(["1"])
    with pytest.raises(SessionBusyError):
        manager.start(["2"])
    manager.stop(wait=True)
    manager.start(["2"])                            # free again once the first has ended


def test_unapproved_and_unknown_students_are_skipped(manager):
    info = manager.start(["1", "3", "99", "1"])
    assert info["student_ids"] == ["1"]
    reasons = {s["id"]: s["reason"] for s in info["skipped_students"]}
    assert "pending" in reasons["3"] and "not registered" in reasons["99"]


def test_session_with_no_valid_students_is_refused(manager):
    with pytest.raises(SessionSetupError):
        manager.start(["3", "99"])
    with pytest.raises(SessionSetupError):
        manager.start([])
    assert manager.status()["state"] == "idle"


def test_invalid_exam_rules_are_refused():
    with pytest.raises(SessionSetupError):
        normalize_exam_context({"allowed_devices": ["smartwatch"]})
    with pytest.raises(SessionSetupError):
        normalize_exam_context({"allowed_devices": "cell phone"})
    assert normalize_exam_context({"allowed_devices": ["laptop", "laptop"]}) == {
        "is_open_book": False, "is_paper_exam": False, "allowed_devices": ["laptop"]}


def test_engine_failure_marks_session_failed_and_frees_the_slot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def broken(plan, ctx, logger, source):
        raise IOError("Cannot open webcam at index 0.")

    m = SessionManager(registry_factory=FakeRegistry, engine_factory=broken, logger_factory=FakeLogger)
    info = m.start(["1"])
    status = wait_state(m, "idle")
    assert status["last_session"]["state"] == "failed"
    assert "Cannot open webcam" in status["last_session"]["error"]
    assert session_row(info["session_id"])["status"] == "failed"


def test_stop_while_models_are_loading(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    release = threading.Event()
    engine = FakeEngine()

    def slow(plan, ctx, logger, source):
        release.wait(5)
        return engine

    m = SessionManager(registry_factory=FakeRegistry, engine_factory=slow, logger_factory=FakeLogger)
    m.start(["1"])
    assert m.stop()["state"] == "stopping"
    release.set()
    wait_state(m, "idle")
    assert m.status()["last_session"]["state"] == "ended"
    assert engine.stop_event.is_set() and engine.frames == 0


def test_session_ends_by_itself_when_the_video_finishes(manager):
    info = manager.start(["1"], source=20)         # fake engine: 20 frames, then the "file" ends
    wait_state(manager, "idle")
    assert manager.status()["last_session"]["state"] == "ended"
    assert session_row(info["session_id"])["status"] == "ended"


def test_listeners_receive_state_changes_and_log_events(manager):
    events = []
    manager.add_listener(events.append)
    manager.start(["1"])
    wait_state(manager, "running")
    manager.stop(wait=True)
    states = [e["session"]["state"] for e in events if e["type"] == "state"]
    assert states == ["starting", "running", "stopping", "ended"]
    assert any(e["type"] == "log" and "starting" in e["message"] for e in events)


def test_stopping_an_unknown_session_is_an_error(manager):
    with pytest.raises(NoActiveSessionError):
        manager.stop()
    manager.start(["1"])
    with pytest.raises(NoActiveSessionError):
        manager.stop("some-other-id")


def test_frames_are_only_served_for_the_running_session(manager):
    info = manager.start(["1"])
    wait_state(manager, "running")
    assert manager.wait_for_jpeg(info["session_id"], 0) == (1, b"jpeg")
    with pytest.raises(NoActiveSessionError):
        manager.wait_for_jpeg("other", 0)


def test_sessions_left_running_by_a_crash_are_marked_failed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db.init_database()
    db.create_session("s1", "[]", "[]")
    assert db.mark_interrupted_sessions() == 1
    assert session_row("s1")["status"] == "failed"
