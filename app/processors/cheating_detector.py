# app/processors/cheating_detector.py
import config as config
from datetime import datetime

class CheatingDetector:
    """
    Analyzes a TrackedPerson's state and accumulated signals
    to determine if any cheating rules are violated.

    This is a "Pattern-Based Rule Engine"
    """

    def __init__(self, logger, exam_context=None):
        self.logger = logger
        self.exam_context = exam_context if exam_context else {}

        # Load thresholds from config
        self.T_SPOOF = getattr(config, "RULE_SPOOF_DURATION", 1.0)
        self.T_ABSENCE = getattr(config, "RULE_ABSENCE_DURATION", 10.0)
        self.T_GAZE_ONLY = getattr(config, "RULE_GAZE_ONLY_DURATION", 7.0)
        self.T_GAZE_W_DEVICE = getattr(config, "RULE_GAZE_WITH_DEVICE_DURATION", 3.0)
        self.T_DEVICE = getattr(config, "RULE_DEVICE_DURATION", 5.0)
        
        # Log loaded thresholds for debugging
        self.logger.log_core_debug(f"CheatingDetector Rules: SPOOF({self.T_SPOOF}s), ABSENCE({self.T_ABSENCE}s), GAZE_ONLY({self.T_GAZE_ONLY}s), GAZE_W_DEV({self.T_GAZE_W_DEVICE}s), DEV_ONLY({self.T_DEVICE}s)")


    def update(self, person, frame):
        """
        Analyzes the person and updates their 'cheating_state' set.
        Returns a list of alerts for this frame.
        """

        new_alerts = []
        person.cheating_state.clear() # Reset state every frame
        now = datetime.now()

        # --- Rule 1: Spoofing (Highest Priority) ---
        # 'is_spoofing' is set by LivenessDetector -> PersonAnalyzer
        # The 'IMAGE_DETECTED' alert timer in person_analyzer.py
        # already handles the duration logic (which uses ALERT_THRESHOLDS_SEC['IMAGE_DETECTED']).
        # We can rely on the person.is_spoofing flag.
        if person.is_spoofing:
            person.cheating_state.add("SPOOF_DETECTED")
            new_alerts.append("Image Detected") # Use existing alert
            # No need to check other rules if it's a spoof
            return new_alerts 

        # --- Rule 2: Absence ---
        if person.absent_duration_sec > self.T_ABSENCE:
            person.cheating_state.add("STUDENT_MISSING")
            # This is handled by global logic, but we flag it
            # The global logic in proctoring_analyzer already logs evidence for this

        # --- Rule 3 & 4: Gaze and Device ---

        # Check for a recent, persistent device
        is_device_present = False
        if person.device_events:
            # Check if the *latest* device event is recent
            latest_event_time, device_label, _ = person.device_events[-1]
            if (now - latest_event_time).total_seconds() < self.T_DEVICE:

                # --- CONTEXT CHECK (from Phase 4, but functional now) ---
                allowed_devices = self.exam_context.get("allowed_devices", [])

                if device_label not in allowed_devices:
                    is_device_present = True
                    person.cheating_state.add("DEVICE_DETECTED")

        # Now, apply gaze rules based on device presence
        if person.gaze_away_duration_sec > self.T_GAZE_ONLY:
            person.cheating_state.add("GAZE_AWAY")

        if is_device_present and person.gaze_away_duration_sec > self.T_GAZE_W_DEVICE:
            # If device is present, we are *more sensitive* to gaze
            person.cheating_state.add("DEVICE_AND_GAZE")

        # --- Consolidate Alerts ---
        if "DEVICE_AND_GAZE" in person.cheating_state:
            new_alerts.append("Suspicious Activity (Device + Gaze)")
            self.log_evidence_once(person, frame, "SUSPICIOUS_DEVICE_GAZE", "Device detected while student is looking away.")
        elif "DEVICE_DETECTED" in person.cheating_state:
            new_alerts.append("Contraband Detected") # Use existing alert
            self.log_evidence_once(person, frame, "CONTRABAND_DETECTED", "Contraband item detected near student.")

        return list(set(new_alerts)) # Return unique alerts

    def log_evidence_once(self, person, frame, alert_key, message):
        """Helper to log evidence only once when an alert is first raised."""

        # We use the existing 'alert_has_been_logged' on the person
        if not person.alert_has_been_logged.get(alert_key, False):
            full_msg = f"({person.get_logging_name()}) {message}"
            self.logger.log_evidence(frame, alert_key, full_msg)
            person.alert_has_been_logged[alert_key] = True
        
        # --- This logic resets the flag when the condition is no longer met ---
        # (This is handled by person_analyzer for 'IMAGE_DETECTED', 
        # but we must handle it for our new alerts)
        
        # Find all keys this helper is responsible for
        cheating_alert_keys = {"SUSPICIOUS_DEVICE_GAZE", "CONTRABAND_DETECTED", "GAZE_AWAY"}
        
        # Check if any *other* cheating alert was logged but is now resolved
        for key in cheating_alert_keys:
            if key != alert_key and person.alert_has_been_logged.get(key, False):
                # This key was logged, but its condition is NOT met this frame, so reset it
                self.logger.log_activity(f"Condition {key} resolved for {person.get_logging_name()}.")
                person.alert_has_been_logged[key] = False