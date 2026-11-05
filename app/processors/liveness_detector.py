# app/processors/liveness_detector.py
import config as config
import numpy as np
from datetime import datetime
from collections import deque


class LivenessDetector:
    """
    Decides whether a tracked face is a live person or a photo held up to the camera.

    Proof of life is an adaptive blink: the eye aspect ratio (EAR) dropping well
    below this person's own recent open-eye level and then recovering. A photo's
    EAR only wobbles by a few percent from landmark noise, while a blink drops it
    by 50-70%, so the signal holds up under camera noise. Comparing against the
    person's own baseline means narrow eyes, glasses or looking down at a paper
    do not break it the way a fixed EAR threshold does.

    The required drop also adapts to how noisy this face's EAR is: on a small or
    narrow-eyed face, landmark jitter alone can move EAR by 15%, so a dip only
    counts if it is LIVENESS_BLINK_NOISE_K robust standard deviations below the
    baseline (never less than the 30% drop, never more than a 50% drop).

    Signals that a photo can imitate are deliberately NOT used as proof of life:
    head-pose changes (a tilted phone fakes them), optical flow (a shaking phone
    fakes it) and categorical gaze flips (they flicker from landmark noise).

    States (stored on person.liveness_state):
      UNCONFIRMED     new face; must blink within LIVENESS_CONFIRM_SECONDS
      CONFIRMED_LIVE  has blinked; flagged only after LIVENESS_SPOOF_THRESHOLD_SEC
                      without another blink

    Time during which the face has no usable landmarks is not counted: a face the
    system cannot see clearly is "not evaluated" rather than assumed to be a photo.

    The detector also smooths head pose and reports large head movements, which
    PersonAnalyzer counts for the excessive-movement alert.
    """

    def __init__(self, person, logger):
        self.person = person
        self.logger = logger

        self.CONFIRM_SECONDS = getattr(config, "LIVENESS_CONFIRM_SECONDS", 8.0)
        self.TIMEOUT_SECONDS = getattr(config, "LIVENESS_SPOOF_THRESHOLD_SEC", 30.0)
        self.BASELINE_SEC = getattr(config, "LIVENESS_BLINK_BASELINE_SEC", 3.0)
        self.DROP_RATIO = getattr(config, "LIVENESS_BLINK_DROP_RATIO", 0.70)
        self.RECOVER_RATIO = getattr(config, "LIVENESS_BLINK_RECOVER_RATIO", 0.85)
        self.MAX_CLOSED_SEC = getattr(config, "LIVENESS_BLINK_MAX_SEC", 2.0)
        self.MIN_BASELINE_SAMPLES = getattr(config, "LIVENESS_BLINK_MIN_BASELINE_SAMPLES", 5)
        self.NOISE_K = getattr(config, "LIVENESS_BLINK_NOISE_K", 4.0)
        self.MIN_DROP_RATIO = getattr(config, "LIVENESS_BLINK_MIN_DROP_RATIO", 0.50)
        self.POSE_SMOOTHING_FRAMES = getattr(config, "POSE_SMOOTHING_FRAMES", 5)
        self.EXCESSIVE_SWAY_THRESHOLD = getattr(config, "EXCESSIVE_MOVEMENT_SWAY_THRESHOLD", 10.0)

        self.logger.log_liveness_debug(
            f"Liveness: blink within {self.CONFIRM_SECONDS}s to confirm, "
            f"flag after {self.TIMEOUT_SECONDS}s without a blink", self.person.get_logging_name())
        self.reset()

    def reset(self):
        """Clears all state. Called when a new face session begins."""
        self.unproven_sec = 0.0          # evaluated seconds since the last proof of life
        self.last_update = None
        self.is_spoofing = False
        self.ear_history = deque()       # (time, ear) samples while eyes are open
        self.closed_since = None
        self.pose_history = deque(maxlen=self.POSE_SMOOTHING_FRAMES)
        self.movement_reference = None
        self.person.liveness_state = "UNCONFIRMED"

    # ------------------------------------------------------------------ blink
    def _baseline_ear(self, now):
        """(median open-eye EAR, EAR ratio a dip must fall below), or None while warming up."""
        while self.ear_history and (now - self.ear_history[0][0]).total_seconds() > self.BASELINE_SEC:
            self.ear_history.popleft()
        if len(self.ear_history) < self.MIN_BASELINE_SAMPLES:
            return None
        ears = np.array([e for _, e in self.ear_history])
        baseline = float(np.median(ears))
        if baseline <= 0:
            return None
        rel_sd = 1.4826 * float(np.median(np.abs(ears - baseline))) / baseline   # robust (MAD) noise
        drop_ratio = min(self.DROP_RATIO, max(self.MIN_DROP_RATIO, 1.0 - self.NOISE_K * rel_sd))
        return baseline, drop_ratio

    def _detect_blink(self, now, ear):
        """Returns True on the frame where a blink completes (eyes reopen)."""
        base = self._baseline_ear(now)
        if base is None:
            self.ear_history.append((now, ear))
            return False
        baseline, drop_ratio = base

        if self.closed_since is None:
            if ear < drop_ratio * baseline:
                self.closed_since = now          # eyes closing; baseline frozen meanwhile
            else:
                self.ear_history.append((now, ear))
            return False

        closed_for = (now - self.closed_since).total_seconds()
        if ear > self.RECOVER_RATIO * baseline:
            self.closed_since = None
            self.ear_history.append((now, ear))
            return True
        if closed_for > self.MAX_CLOSED_SEC:
            # Eyes stayed low (e.g. looking down at the paper): adopt the new level as
            # the baseline so blinks from this position can be detected.
            self.closed_since = None
            self.ear_history.clear()
            self.ear_history.append((now, ear))
        return False

    # --------------------------------------------------------------- movement
    def _detect_large_movement(self, mp_results):
        if mp_results.get("pose_uncertain", True):
            return False
        self.pose_history.append((mp_results["yaw"], mp_results["pitch"]))
        smoothed = np.median(np.array(self.pose_history), axis=0)
        if self.movement_reference is None:
            self.movement_reference = smoothed      # first reading is the reference, not (0, 0)
            return False
        if np.max(np.abs(smoothed - self.movement_reference)) > self.EXCESSIVE_SWAY_THRESHOLD:
            self.movement_reference = smoothed
            return True
        return False

    # ------------------------------------------------------------------- main
    def check_liveness(self, mp_results):
        """
        Updates liveness from this frame's MediaPipe result.

        Returns:
            is_spoofing (bool): the face is currently judged to be a photo
            large_movement (bool): a large head movement completed this frame
        """
        name = self.person.get_logging_name()
        now = datetime.now()
        dt = 0.0 if self.last_update is None else (now - self.last_update).total_seconds()
        self.last_update = now

        if not mp_results or mp_results.get("ear_uncertain", True):
            # Not evaluated: keep the previous verdict, don't let the timer run.
            return self.is_spoofing, False

        large_movement = self._detect_large_movement(mp_results)
        self.unproven_sec += dt

        if self._detect_blink(now, mp_results["ear"]):
            if self.person.liveness_state != "CONFIRMED_LIVE" or self.is_spoofing:
                self.logger.log_liveness_debug(
                    f"LIVE: blink after {self.unproven_sec:.1f}s (EAR {mp_results['ear']:.3f})", name)
            self.person.liveness_state = "CONFIRMED_LIVE"
            self.unproven_sec = 0.0
            self.is_spoofing = False
            return False, large_movement

        limit = self.CONFIRM_SECONDS if self.person.liveness_state == "UNCONFIRMED" else self.TIMEOUT_SECONDS
        spoofing = self.unproven_sec > limit
        if spoofing and not self.is_spoofing:
            self.logger.log_liveness_debug(
                f"PHOTO SUSPECTED: no blink for {self.unproven_sec:.1f}s "
                f"(state {self.person.liveness_state}, limit {limit}s)", name)
        self.is_spoofing = spoofing
        return spoofing, large_movement
