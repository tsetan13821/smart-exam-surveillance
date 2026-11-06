import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from sim import SimClock, RecordingLogger  # noqa: E402

FPS = 30


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    """Every test gets its own database and data folders, never the real ones."""
    import config
    monkeypatch.setattr(config, "DATABASE_FILE", str(tmp_path / "test.db"))
    monkeypatch.setattr(config, "STUDENT_DATABASE_FILE", str(tmp_path / "student_encodings.json"))
    monkeypatch.setattr(config, "REGISTRATION_EVIDENCE_DIR", str(tmp_path / "registrations"))
    monkeypatch.setattr(config, "DEMO_VIDEO_DIR", str(tmp_path / "demo_videos"))
    monkeypatch.setattr(config, "PROCTORING_LOG_DIR", str(tmp_path / "proctoring_logs"))
    for name in ("CORE_LOGS_DIR", "MEDIAPIPE_LOGS_DIR", "OBJECT_DETECTOR_LOGS_DIR", "LIVENESS_LOGS_DIR",
                 "SIMILARITY_LOGS_DIR", "HUMAN_ACTION_LOG_DIR"):
        monkeypatch.setattr(config, name, str(tmp_path / "logs" / name.lower()))
    return tmp_path


@pytest.fixture
def clock(monkeypatch):
    return SimClock().patch(monkeypatch)


@pytest.fixture
def logger(clock):
    return RecordingLogger(clock)
