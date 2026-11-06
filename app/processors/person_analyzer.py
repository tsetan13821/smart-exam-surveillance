# app/processors/person_analyzer.py
import config as config
from datetime import datetime, timedelta # <-- ADD TIMEDELTA

"""
This processor handles all analysis logic for a SINGLE person.
It is a helper class. It is called by the ProctoringAnalyzer.
"""

class PersonAnalyzer:
    def __init__(self, logger):
        self.logger = logger

    def analyze_liveness(self, person, frame, image_id_map, next_image_id):
        """
        Runs liveness checks and handles the IMAGE_DETECTED alert logic.
        NOW ALSO tracks head sway events for frequency analysis.
        """
        # --- MODIFIED: Unpack two return values ---
        person.is_spoofing, head_sway_event = person.liveness_detector.check_liveness(person.mp_result)
        
        # --- NEW: Track head sway events ---
        if head_sway_event:
            person.head_sway_timestamps.append(datetime.now())
        
        had_image_id = person.id in image_id_map
        if self._check_person_alert(person, frame, "IMAGE_DETECTED", person.is_spoofing, "Liveness check failed", image_id_map, next_image_id):
            if not had_image_id and person.id in image_id_map:
                next_image_id += 1  # an ID was assigned just now; the next photo gets a new one
            return ["Image Detected"], next_image_id
            
        return [], next_image_id

    def log_behavior(self, person):
        """
        Logs *changes* in a specific person's actions (pose, gaze, etc.).
        This method also logs the raw MediaPipe data with the
        correct person_name, fixing the 'null' log error.
        """
        mp_results = person.mp_result
        log_id = person.get_logging_name()
        
        # --- <<< NEW FIX START >>> ---
        # Log the raw MediaPipe data with the correct person name.
        if mp_results:
            # Create a shallow copy so we don't modify the original object
            loggable_details = mp_results.copy()
            
            # Remove the non-JSON-serializable 'landmarks_object'
            # Use .pop() with a default (None) to prevent errors if key doesn't exist
            loggable_details.pop("landmarks_object", None)
            
            # Now, log the *cleaned* dictionary
            self.logger.log_mediapipe_debug(
                event_type="FRAME_ANALYSIS",
                message="Full MediaPipe results for person.",
                details=loggable_details,  # <-- This is now safe to log
                person_name=log_id  
            )
        # --- <<< NEW FIX END >>> ---

        current_pose_action = "Uncertain"
        current_gaze_action = "Uncertain"
        current_eyes_action = "Uncertain"
        current_mouth_action = "Uncertain"
        details = {}

        if mp_results:
            if not mp_results.get("pose_uncertain", True):
                yaw = mp_results["yaw"]
                pitch = mp_results["pitch"]
                details['pose'] = {"yaw": round(yaw, 2), "pitch": round(pitch, 2)}
                
                if pitch > config.HEAD_POSE_PITCH_DOWN_THRESHOLD: current_pose_action = "Head Down"
                elif pitch < config.HEAD_POSE_PITCH_UP_THRESHOLD: current_pose_action = "Head Up"
                elif yaw > config.HEAD_POSE_YAW_THRESHOLD: current_pose_action = "Head Right"
                elif yaw < -config.HEAD_POSE_YAW_THRESHOLD: current_pose_action = "Head Left"
                else: current_pose_action = "Center"

            if not mp_results.get("gaze_uncertain", True):
                current_gaze_action = mp_results["gaze"]
                details['gaze'] = {"value": current_gaze_action}

            if not mp_results.get("ear_uncertain", True):
                current_eyes_action = "Eyes Closed" if mp_results["eyes_closed"] else "Eyes Open"
                details['eyes'] = {"ear": round(mp_results["ear"], 3)}
                
            if not mp_results.get("mar_uncertain", True):
                is_open = mp_results["mar"] > config.MOUTH_MAR_THRESHOLD
                current_mouth_action = "Mouth Open" if is_open else "Mouth Closed"
                details['mouth'] = {"mar": round(mp_results["mar"], 3)}
        
        if current_pose_action != person.last_action_pose:
            self.logger.log_human_action("pose", f"({log_id}) {current_pose_action}", details.get('pose'))
            person.last_action_pose = current_pose_action

        if current_gaze_action != person.last_action_gaze:
            self.logger.log_human_action("gaze", f"({log_id}) {current_gaze_action}", details.get('gaze'))
            person.last_action_gaze = current_gaze_action
            
        if current_eyes_action != person.last_action_eyes:
            self.logger.log_human_action("eyes", f"({log_id}) {current_eyes_action}", details.get('eyes'))
            person.last_action_eyes = current_eyes_action
            
        if current_mouth_action != person.last_action_mouth:
            self.logger.log_human_action("mouth", f"({log_id}) {current_mouth_action}", details.get('mouth'))
            person.last_action_mouth = current_mouth_action

    # --- NEW METHOD (replaces old pose alert logic) ---
    def analyze_pose_alerts(self, person, frame):
        """
        Checks for head pose-based alerts (peeking, excessive movement).
        """
        active_alerts = []
        
        # --- 1. PEEKING (HEAD TURN) ALERT ---
        peeking_condition = False
        if person.mp_result and not person.mp_result.get("pose_uncertain", True):
            yaw = person.mp_result["yaw"]
            peeking_condition = abs(yaw) > config.HEAD_POSE_YAW_ALERT_THRESHOLD
        
        # This call will handle the alert timer AND take a screenshot 
        # (via log_evidence) if the alert is triggered.
        if self._check_person_alert(person, frame, "HEAD_TURN_ALERT", peeking_condition, "Excessive head turning (peeking)", None, None):
            active_alerts.append("PEEKING")
            
        # --- 2. EXCESSIVE MOVEMENT ALERT ---
        now = datetime.now()
        window_start = now - timedelta(seconds=config.EXCESSIVE_MOVEMENT_TIME_WINDOW_SEC)
        
        # Pop old timestamps
        while person.head_sway_timestamps and person.head_sway_timestamps[0] < window_start:
            person.head_sway_timestamps.popleft()
        
        # Check condition
        movement_count = len(person.head_sway_timestamps)
        movement_condition = movement_count > config.EXCESSIVE_MOVEMENT_COUNT_THRESHOLD

        # This call also handles the timer AND takes a screenshot
        # (via log_evidence) if the alert is triggered.
        if self._check_person_alert(person, frame, "EXCESSIVE_MOVEMENT_ALERT", movement_condition, f"Excessive head movement detected ({movement_count} movements in 60s)", None, None):
            active_alerts.append("EXCESSIVE MOVEMENT")

        return active_alerts

    def _check_person_alert(self, person, frame, alert_key, condition_met, message, image_id_map=None, next_image_id=None):
        """Internal helper to manage alert timers and logging for a person."""
        is_active = False
        # Only IMAGE_DETECTED uses this, but we check other keys in case you add them back
        threshold = config.ALERT_THRESHOLDS_SEC.get(alert_key, 3.0) 
        person_logging_name = person.get_logging_name()
        
        if condition_met:
            if person.alert_start_times[alert_key] is None:
                person.alert_start_times[alert_key] = datetime.now()
            else:
                duration = (datetime.now() - person.alert_start_times[alert_key]).total_seconds()
                if duration >= threshold:
                    is_active = True
                    if not person.alert_has_been_logged[alert_key]:
                        # Special handling for image detection
                        if alert_key == "IMAGE_DETECTED" and image_id_map is not None:
                            if person.id not in image_id_map:
                                image_id_map[person.id] = f"Image-{next_image_id}"
                            
                            image_id = image_id_map[person.id]
                            person_display_name = person.get_display_name() 
                            log_msg = f"{person_display_name} identified as a image. Now tracking as {image_id}."
                            self.logger.log_liveness_debug(log_msg, person_display_name)
                            person.spoof_id = image_id # Note: This is still 'spoof_id' in your TrackedPerson model
                            self.logger.log_evidence(frame, alert_key, log_msg, subject=person.student_name())
                            person.alert_has_been_logged[alert_key] = True
                        
                        # Standard handling for other alerts (if you add any back)
                        elif alert_key != "IMAGE_DETECTED":
                            log_msg = f"({person_logging_name}) {message} (Duration: {duration:.1f}s)"
                            self.logger.log_evidence(frame, alert_key, log_msg, subject=person.student_name())
                            person.alert_has_been_logged[alert_key] = True
        else:
            if person.alert_start_times[alert_key] is not None:
                if person.alert_has_been_logged[alert_key]:
                    log_msg = f"Condition {alert_key} resolved for {person_logging_name}."
                    self.logger.log_activity(log_msg)
                    if alert_key == "IMAGE_DETECTED" and image_id_map is not None:
                        self.logger.log_liveness_debug("Condition resolved.", person_logging_name)
                        if person.id in image_id_map:
                            del image_id_map[person.id]
                        person.spoof_id = None # Note: This is still 'spoof_id' in your TrackedPerson model
                        
            person.alert_start_times[alert_key] = None
            person.alert_has_been_logged[alert_key] = False
            
        return is_active