"""
Simulation helpers for model-free tests.

The proctoring logic is time-based (datetime.now()), so tests replace the
`datetime` class inside each module with one driven by a simulated clock.
That makes a 20-second scenario run in milliseconds and makes every
timing assertion deterministic.
"""
import datetime as _dt
import importlib

T0 = _dt.datetime(2026, 1, 1, 9, 0, 0)

TIME_DEPENDENT_MODULES = [
    "app.processors.liveness_detector",
    "app.processors.cheating_detector",
    "app.processors.proctoring_analyzer",
    "app.processors.person_analyzer",
    "app.processors.person_tracker",
    "app.models.tracked_person",
]


class SimClock:
    def __init__(self):
        self.t = T0

    def advance(self, seconds):
        self.t += _dt.timedelta(seconds=seconds)

    def elapsed(self):
        return (self.t - T0).total_seconds()

    def patch(self, monkeypatch):
        clock = self

        class FakeDateTime(_dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return clock.t

        for name in TIME_DEPENDENT_MODULES:
            monkeypatch.setattr(importlib.import_module(name), "datetime", FakeDateTime)
        return self


class RecordingLogger:
    """Stands in for ActivityLogger: records evidence/alerts in memory, ignores debug logs."""

    def __init__(self, clock):
        self.clock = clock
        self.evidence = []   # (seconds since start, alert_type, message)
        self.alerts = []

    def log_evidence(self, frame, alert_type, message):
        self.evidence.append((round(self.clock.elapsed(), 2), alert_type, message))

    def log_alert(self, alert_type, message):
        self.alerts.append((alert_type, message))

    def evidence_types(self):
        return [e[1] for e in self.evidence]

    def first_evidence(self, alert_type):
        return next((t for t, a, _ in self.evidence if a == alert_type), None)

    def __getattr__(self, name):          # log_activity, log_core_debug, log_liveness_debug, ...
        return lambda *args, **kwargs: None


def mp_result(yaw=0.0, pitch=0.0, ear=0.30, gaze="Center", mar=0.1):
    """A MediaPipe result dict as produced by MediaPipeProcessor.process()."""
    return {
        "yaw": yaw, "pitch": pitch, "pose_uncertain": False,
        "ear": ear, "ear_uncertain": False, "eyes_closed": ear < 0.2,
        "gaze": gaze, "gaze_uncertain": False,
        "mar": mar, "mar_uncertain": False,
    }


def blinking_ear(t, period=3.0, open_ear=0.30, closed_ear=0.12, duration=0.13):
    """EAR of a person who blinks for `duration` seconds every `period` seconds."""
    return closed_ear if (t % period) < duration else open_ear


BOX = [250, 150, 360, 300]           # the student's face box in a 640x480 frame
DESK_ITEM_BOX = [300, 330, 360, 400]  # an object on the desk right below the face


def item(label, box=None):
    return {"label": label, "box": list(box or DESK_ITEM_BOX), "score": 0.9}


class SessionRun:
    """Result of run_session(): status timeline + evidence log."""

    def __init__(self, timeline, logger, analyzer):
        self.timeline = timeline          # [(t, classification)] at each change
        self.logger = logger
        self.analyzer = analyzer

    def first(self, classification):
        return next((t for t, c in self.timeline if c == classification), None)

    def statuses(self):
        return {c for _, c in self.timeline}


def run_session(clock, logger, seconds, frame_fn, *, students=("Test Student",),
                person_name="Test Student", exam_context=None, rois=None,
                present_fn=lambda t: True, fps=30):
    """
    Drive ProctoringAnalyzer.analyze_frame() for `seconds` of simulated video.

    frame_fn(t) -> (mp_result, contraband_items)
    present_fn(t) -> whether the student's face is in view at time t
    """
    from app.models.tracked_person import TrackedPerson
    from app.processors.proctoring_analyzer import ProctoringAnalyzer

    analyzer = ProctoringAnalyzer(logger, list(students), exam_context=exam_context or {},
                                  rois_data=rois)
    person = TrackedPerson("Person-1", person_name, BOX, logger)
    timeline, last = [], None
    for i in range(int(seconds * fps)):
        t = i / fps
        mp, items = frame_fn(t)
        persons = {}
        if present_fn(t):
            person.update(BOX, person_name, mp)
            persons = {"Person-1": person}
        status = analyzer.analyze_frame(None, persons, items, False)
        if status != last:
            timeline.append((round(t, 2), status))
            last = status
        clock.advance(1 / fps)
    return SessionRun(timeline, logger, analyzer)
