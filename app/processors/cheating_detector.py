# app/processors/cheating_detector.py
import config as config
from datetime import datetime


class CheatingDetector:
    """
    Pattern-based rule engine. Combines a TrackedPerson's accumulated signals
    (liveness, gaze-away time, devices seen on their desk) with the exam context
    (open-book, allowed devices, paper-based) to decide which alerts are active.
    """

    # alert key -> (status text shown to the proctor, evidence message)
    RULES = {
        "SUSPICIOUS_DEVICE_GAZE": ("Suspicious Activity (Device + Gaze)",
                                   "Device detected while student is looking away."),
        "CONTRABAND_DETECTED": ("Contraband Detected", "Contraband item detected near student."),
        "GAZE_AWAY": ("Sustained Gaze Away", "Student has been looking away from the exam."),
    }

    def __init__(self, logger, exam_context=None):
        self.logger = logger
        self.exam_context = exam_context if exam_context else {}

        self.T_GAZE_ONLY = getattr(config, "RULE_GAZE_ONLY_DURATION", 5.0)
        self.T_GAZE_W_DEVICE = getattr(config, "RULE_GAZE_WITH_DEVICE_DURATION", 3.0)
        self.T_DEVICE = getattr(config, "RULE_DEVICE_DURATION", 5.0)
        self.T_DEVICE_PERSIST = getattr(config, "CHEATING_DEVICE_PERSISTENCE_SEC", 2.0)
        self.allowed_devices = set(self.exam_context.get("allowed_devices", []))

        self.logger.log_core_debug(
            f"CheatingDetector rules: GAZE_ONLY({self.T_GAZE_ONLY}s), GAZE_W_DEV({self.T_GAZE_W_DEVICE}s), "
            f"DEVICE persist {self.T_DEVICE_PERSIST}s / memory {self.T_DEVICE}s, allowed={sorted(self.allowed_devices)}")

    def _persistent_device(self, person, now):
        """
        Returns the label of a non-allowed device that has been seen on this
        student's desk continuously for at least T_DEVICE_PERSIST seconds and
        was last seen within T_DEVICE seconds, or None. A single-frame
        detection (often a false positive) is not enough.
        """
        for label, (first_seen, last_seen) in person.device_sightings.items():
            if label in self.allowed_devices:
                continue
            recent = (now - last_seen).total_seconds() < self.T_DEVICE
            persistent = (last_seen - first_seen).total_seconds() >= self.T_DEVICE_PERSIST
            if recent and persistent:
                return label
        return None

    def update(self, person, frame):
        """Updates person.cheating_state and returns this frame's alert texts."""
        person.cheating_state.clear()
        now = datetime.now()

        if person.is_spoofing:
            person.cheating_state.add("SPOOF_DETECTED")
            self._reset_evidence_flags(person, active=set())
            return ["Image Detected"]

        if person.absent_duration_sec > getattr(config, "RULE_ABSENCE_DURATION", 10.0):
            person.cheating_state.add("STUDENT_MISSING")

        device = self._persistent_device(person, now)
        if device:
            person.cheating_state.add("DEVICE_DETECTED")

        active = set()
        if device and person.gaze_away_duration_sec > self.T_GAZE_W_DEVICE:
            active.add("SUSPICIOUS_DEVICE_GAZE")      # more sensitive when a device is present
        elif device:
            active.add("CONTRABAND_DETECTED")
        if person.gaze_away_duration_sec > self.T_GAZE_ONLY:
            active.add("GAZE_AWAY")
            person.cheating_state.add("GAZE_AWAY")

        for key in active:
            if not person.alert_has_been_logged.get(key, False):
                msg = self.RULES[key][1]
                if key != "GAZE_AWAY" and device:
                    msg = f"{msg} ({device})"
                self.logger.log_evidence(frame, key, f"({person.get_logging_name()}) {msg}",
                                         subject=person.student_name())
                person.alert_has_been_logged[key] = True
        self._reset_evidence_flags(person, active)

        return [self.RULES[k][0] for k in active]

    def _reset_evidence_flags(self, person, active):
        """Re-arms evidence logging for rules whose condition has cleared."""
        for key in self.RULES:
            if key not in active and person.alert_has_been_logged.get(key, False):
                self.logger.log_activity(f"Condition {key} resolved for {person.get_logging_name()}.")
                person.alert_has_been_logged[key] = False
