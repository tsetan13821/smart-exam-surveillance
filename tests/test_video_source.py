"""Video sources: recorded files play at their own frame rate; cameras are opened by index."""
import time

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from app import video_source  # noqa: E402
from app.video_source import CameraSource, FileSource, open_source  # noqa: E402


def make_clip(path, frames=10, fps=20):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(frames):
        writer.write(np.full((48, 64, 3), i * 20, np.uint8))
    writer.release()
    return str(path)


def test_file_plays_at_its_frame_rate_then_ends(tmp_path):
    src = FileSource(make_clip(tmp_path / "clip.mp4", frames=10, fps=20))
    start = time.perf_counter()
    frames = []
    while (f := src.read()) is not None:
        frames.append(f)
    elapsed = time.perf_counter() - start
    src.release()
    assert len(frames) == 10
    assert 0.4 <= elapsed < 1.5                     # 10 frames at 20 FPS ~ 0.45 s, not instant


def test_file_can_loop(tmp_path):
    src = FileSource(make_clip(tmp_path / "clip.mp4", frames=5), loop=True, realtime=False)
    assert all(src.read() is not None for _ in range(12))
    src.release()


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(IOError):
        open_source(str(tmp_path / "nope.mp4"))


def test_digit_source_opens_a_camera(monkeypatch):
    opened = []

    class FakeCapture:
        def __init__(self, index):
            opened.append(index)

        def isOpened(self):
            return True

        def read(self):
            return True, np.zeros((4, 4, 3), np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(video_source.cv2, "VideoCapture", FakeCapture)
    src = open_source("1")
    assert isinstance(src, CameraSource) and opened == [1]
    assert src.read().shape == (4, 4, 3)


def test_camera_that_never_opens_raises(monkeypatch):
    class DeadCapture:
        def __init__(self, index):
            pass

        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(video_source.cv2, "VideoCapture", DeadCapture)
    monkeypatch.setattr(video_source.time, "sleep", lambda s: None)
    with pytest.raises(IOError):
        CameraSource(0, retries=2)
