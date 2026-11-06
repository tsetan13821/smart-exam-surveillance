# app/video_source.py
"""Frame sources for the proctoring engine: a webcam, or a recorded video file.

A video file is played back at its own frame rate (not as fast as possible), so
time-based rules such as liveness and gaze timers behave as they would live.
That makes recorded clips usable for demos and end-to-end tests without a camera.
"""
import os
import time

import cv2


class CameraSource:
    """Webcam frames, mirrored so the proctor view looks like a mirror."""

    def __init__(self, index=0, retries=5, logger=None):
        self.description = f"camera {index}"
        self.capture = None
        for attempt in range(1, retries + 1):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    self.capture = cap
                    if logger:
                        logger.log_activity(f"Webcam opened (attempt {attempt}).")
                    break
                cap.release()
            if logger:
                logger.log_activity(f"Camera {index} not ready (attempt {attempt}/{retries}).")
            time.sleep(1)
        if self.capture is None:
            raise IOError(f"Cannot open webcam at index {index}.")

    def read(self):
        """Returns the next frame, or None if the camera stopped delivering frames."""
        ok, frame = self.capture.read()
        return cv2.flip(frame, 1) if ok and frame is not None else None

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class FileSource:
    """Frames from a video file, delivered at the file's frame rate."""

    def __init__(self, path, loop=False, realtime=True):
        if not os.path.isfile(path):
            raise IOError(f"Video file not found: {path}")
        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            raise IOError(f"Cannot open video file: {path}")
        self.description = f"file {os.path.basename(path)}"
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and 1 <= fps <= 240 else 30.0
        self.loop = loop
        self.realtime = realtime
        self._next_due = None

    def read(self):
        """Returns the next frame (waiting to keep real-time pace), or None at the end."""
        ok, frame = self.capture.read()
        if not ok and self.loop:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
        if not ok or frame is None:
            return None
        if self.realtime:
            now = time.perf_counter()
            if self._next_due is None:
                self._next_due = now
            if self._next_due > now:
                time.sleep(self._next_due - now)
            # If processing fell behind, don't try to catch up with a burst of frames
            self._next_due = max(self._next_due + 1.0 / self.fps, time.perf_counter() - 1.0 / self.fps)
        return frame

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None


def open_source(source=None, loop=False, logger=None):
    """Opens a camera index (int or digit string) or a video file path.

    None means the camera from config.CAMERA_INDEX.
    """
    if source is None:
        import config
        source = getattr(config, "CAMERA_INDEX", 0)
    if isinstance(source, int) or (isinstance(source, str) and source.strip().isdigit()):
        return CameraSource(int(source), logger=logger)
    return FileSource(str(source), loop=loop)
