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


@pytest.fixture
def clock(monkeypatch):
    return SimClock().patch(monkeypatch)


@pytest.fixture
def logger(clock):
    return RecordingLogger(clock)
