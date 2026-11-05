# app/processors/proctoring_analyzer.py
import config as config
from datetime import datetime
from app.processors.person_analyzer import PersonAnalyzer
from app.processors.cheating_detector import CheatingDetector
from app.processors.person_tracker import PersonTracker 
import cv2 # <-- NEW IMPORT (Phase 5)
import numpy as np # <-- NEW IMPORT (Phase 5)

"""
This processor orchestrates the analysis of the entire session.
It manages session-level state (e.g., student presence, global alerts)
and uses the PersonAnalyzer to handle per-person logic.
"""

class ProctoringAnalyzer:
    # --- MODIFIED: __init__ signature updated (Phase 5) ---
    def __init__(self, logger, authorized_student_names: list, exam_context=None, rois_data=None):
        self.logger = logger
        
        # --- NEW: Use the PersonAnalyzer helper class ---
        self.person_analyzer = PersonAnalyzer(logger)
        
        # Use a set for fast O(1) lookups
        self.authorized_student_names = set(authorized_student_names)
        print(f"Analyzer: Tracking {len(self.authorized_student_names)} authorized students.")
        
        # --- NEW: Store Desk ROIs (Phase 5) ---
        self.desk_rois = rois_data if rois_data is not None else {}
        if self.desk_rois:
            print(f"Analyzer: Loaded {len(self.desk_rois)} Desk ROIs.")
        else:
            print("Analyzer: No Desk ROIs loaded. Will use 'nearest person' for contraband.")
        # --- END NEW ---
        
        # --- Session-Level State: Per-Student Presence ---
        self.student_presence_state = {}
        for student_name in self.authorized_student_names:
            self.student_presence_state[student_name] = {
                "is_present": False,
                "alert_start_time": None,
                "has_been_logged": False,
                "ever_present": False,
                "last_seen": None
            }
        
        # --- Session-Level State: Global Alerts ---
        self.global_alert_has_been_logged = {
            "UNAUTHORIZED_FACE_DETECTED": False,
        }
        
        # --- Session-Level State: Image ID Tracking ---
        self.image_id_map = {} # Maps person.id -> "Image-X"
        self.next_image_id = 1
        
        # --- Cheating Detector ---
        self.exam_context = exam_context if exam_context is not None else {} 
        self.cheating_detector = CheatingDetector(self.logger, self.exam_context)
        

    def analyze_frame(self, frame, tracked_persons, contraband_items, unauthorized_alert_active):
        """
        Main entry point for the analyzer.
        Orchestrates per-person and session-level analysis.
        """
        
        # --- 1. Prune stale image IDs ---
        current_person_ids = set(tracked_persons.keys())
        stale_image_ids = [pid for pid in self.image_id_map if pid not in current_person_ids]
        for pid in stale_image_ids:
            self.logger.log_liveness_debug(f"Pruning stale image ID mapping for deregistered {pid}", "Analyzer")
            if pid in self.image_id_map:
                del self.image_id_map[pid]
        
        all_person_alerts = []
        global_alerts = []
        
        # --- 2. Get Set of Visible, Live, Authorized Students ---
        visible_authorized_students = set(
            p.name for p in tracked_persons.values()
            if p.name in self.authorized_student_names and not p.is_spoofing
        )
        
        # --- 3. Process Each Tracked Person ---
        for person in tracked_persons.values():
            person.current_alerts = [] # Clear last frame's alerts
            
            # --- 3a. Liveness (delegated) ---
            liveness_alerts, self.next_image_id = self.person_analyzer.analyze_liveness(
                person, frame, self.image_id_map, self.next_image_id
            )
            person.current_alerts.extend(liveness_alerts)

            # --- 3b. Attach Desk ROI (Phase 5) ---
            # Check if this person has an ROI and it's not already attached
            if person.name in self.desk_rois and not hasattr(person, 'desk_roi'):
                try:
                    # Convert list of points to a NumPy array for cv2
                    person.desk_roi = np.array(self.desk_rois[person.name], dtype=np.int32)
                except Exception as e:
                    self.logger.log_alert("ROI_ERROR", f"Failed to create ROI for {person.name}: {e}")
                    # Remove from dict to prevent repeated errors
                    if person.name in self.desk_rois:
                        del self.desk_rois[person.name] 

            # --- 3c. Accumulate Cheating Signals ---
            now = datetime.now()
            
            # (A) Accumulate Gaze
            is_gaze_away = False
            if person.mp_result and not person.mp_result.get("pose_uncertain", True):
                yaw = person.mp_result.get("yaw", 0)
                pitch = person.mp_result.get("pitch", 0)
                
                if (abs(yaw) > config.CHEATING_GAZE_AWAY_THRESHOLD_YAW or 
                    abs(pitch) > config.CHEATING_GAZE_AWAY_THRESHOLD_PITCH):
                    is_gaze_away = True
            
            if is_gaze_away:
                if person.gaze_away_start_time is None:
                    person.gaze_away_start_time = now
                person.gaze_away_duration_sec = (now - person.gaze_away_start_time).total_seconds()
            else:
                person.gaze_away_start_time = None
                person.gaze_away_duration_sec = 0.0
            
            # --- 3d. Run Cheating Detector ---
            cheating_alerts = self.cheating_detector.update(person, frame)
            person.current_alerts.extend(cheating_alerts)

            # --- 3e. Behavior Logging (delegated) ---
            self.person_analyzer.log_behavior(person)
            
            # --- 3f. NEW: Run Rule-Based Pose Alerts ---
            # Only run this for authorized students
            if person.name in self.authorized_student_names:
                pose_alerts = self.person_analyzer.analyze_pose_alerts(person, frame)
                person.current_alerts.extend(pose_alerts)
                    
            all_person_alerts.extend(person.current_alerts)

        # --- 3.5: Assign Contraband (MODIFIED FOR PHASE 5) ---
        if contraband_items:
            now = datetime.now()
            
            # --- Logic 1: Use Desk ROIs (Primary) ---
            if self.desk_rois:
                for item in contraband_items:
                    item_centroid = ( (item['box'][0] + item['box'][2]) // 2, 
                                    (item['box'][1] + item['box'][3]) // 2 )

                    for person in tracked_persons.values():
                        # Check if this person has a desk ROI defined
                        if hasattr(person, 'desk_roi'):
                            # Check if the item's centroid is inside the person's desk polygon
                            is_inside = cv2.pointPolygonTest(person.desk_roi, item_centroid, False)

                            if is_inside >= 0:
                                # This item is ON this person's desk
                                event = (now, item['label'], item['box'])
                                person.device_events.append(event)
                                self.logger.log_core_debug(f"Assigned {item['label']} to {person.get_logging_name()} via Desk ROI")
                                break # Stop checking other persons for this item
            
            # --- Logic 2: Fallback to "Nearest Person" ---
            else:
                for item in contraband_items:
                    item_centroid = ( (item['box'][0] + item['box'][2]) // 2, 
                                    (item['box'][1] + item['box'][3]) // 2 )
                    best_person = None
                    min_dist = getattr(config, "CHEATING_DEVICE_MAX_DISTANCE_PX", 250) 
                    for person in tracked_persons.values():
                        dist = PersonTracker._euclidean_dist(item_centroid, person.centroid)
                        if dist < min_dist:
                            min_dist = dist
                            best_person = person
                    if best_person:
                        event = (now, item['label'], item['box'])
                        best_person.device_events.append(event)
                        self.logger.log_core_debug(f"Assigned {item['label']} to {best_person.get_logging_name()} via Nearest_Person (Fallback)")

        # --- 4. Process Per-Student Missing Alerts (Session Logic) ---
        is_any_student_missing = False
        for student_name in self.authorized_student_names:
            is_present = student_name in visible_authorized_students
            if self._check_student_missing_alert(frame, student_name, is_present):
                is_any_student_missing = True 
        
        if is_any_student_missing:
            global_alerts.append("Student Missing")

        # --- 5. Process Global Alerts (Session Logic) ---
        
        # Check for Unauthorized Faces
        num_unauthorized_faces = 0
        for person in tracked_persons.values():
            if person.name == "Unknown":
                num_unauthorized_faces += 1
        is_unauthorized_present = unauthorized_alert_active or (num_unauthorized_faces > 0)
        
        if self._check_global_alert(frame, "UNAUTHORIZED_FACE_DETECTED", is_unauthorized_present, f"Unauthorized face detected ({num_unauthorized_faces} found)."):
            global_alerts.append("Unauthorized Face")
            
        # --- 6. Final Classification Logic ---
        final_classification = "Normal"
        priority_alert = None
        person_alerts_set = set(all_person_alerts)
        
        if "Image Detected" in person_alerts_set:
            priority_alert = "Image Detected"
        elif "Suspicious Activity (Device + Gaze)" in person_alerts_set:
            priority_alert = "Suspicious Activity (Device + Gaze)"
        elif "Contraband Detected" in person_alerts_set:
            priority_alert = "Contraband Detected"
        # --- NEW ALERT HIERARCHY ---
        elif "PEEKING" in person_alerts_set:
            priority_alert = "Peeking Alert"
        elif "EXCESSIVE MOVEMENT" in person_alerts_set:
            priority_alert = "Excessive Movement"
        # --- END NEW ---
        elif "Sustained Gaze Away" in person_alerts_set:
            priority_alert = "Sustained Gaze Away"
        
        if any(p.is_spoofing for p in tracked_persons.values()):
            final_classification = "Image Detected"
        elif "Student Missing" in global_alerts:
            final_classification = "Student Missing"
        elif "Unauthorized Face" in global_alerts:
            final_classification = "Unauthorized Face"
        elif priority_alert: 
            final_classification = priority_alert
            
        return final_classification

    # --- Session-Level Helper: Student Missing ---
    def _check_student_missing_alert(self, frame, student_name, is_present):
        state = self.student_presence_state[student_name]
        alert_key = "STUDENT_MISSING"
        is_alert_active = False
        
        if is_present:
            if not state["is_present"]:
                if state["ever_present"]: 
                    self.logger.log_evidence(frame, "PRESENCE_RETURNED", f"Student {student_name} has RETURNED.")
                elif not state["ever_present"]:
                    self.logger.log_evidence(frame, "ATTENDANCE_MARKED", f"Student {student_name} marked PRESENT.")
                    state["ever_present"] = True
            
            if state["alert_start_time"] is not None and state["has_been_logged"]:
                self.logger.log_activity(f"Condition {alert_key} resolved for {student_name}.")
                
            state["is_present"] = True
            state["alert_start_time"] = None
            state["has_been_logged"] = False
            state["last_seen"] = datetime.now()
        else:
            if state["is_present"]:
                self.logger.log_evidence(frame, "PRESENCE_LEFT", f"Student {student_name} has LEFT.")
                state["is_present"] = False 
            
            if state["ever_present"]:
                threshold = config.ALERT_THRESHOLDS_SEC.get(alert_key, 10.0)
                
                if state["alert_start_time"] is None:
                    state["alert_start_time"] = datetime.now()
                else:
                    duration = (datetime.now() - state["alert_start_time"]).total_seconds()
                    if duration >= threshold:
                        is_alert_active = True
                        if not state["has_been_logged"]:
                            log_msg = f"Student {student_name} is missing. (Duration: {duration:.1f}s)"
                            self.logger.log_evidence(frame, alert_key, log_msg)
                            state["has_been_logged"] = True
            
            state["is_present"] = False
        
        return is_alert_active

    # --- Session-Level Helper: Global Alerts ---
    def _check_global_alert(self, frame, alert_key, condition_met, message):
        if alert_key not in self.global_alert_has_been_logged:
            return False
        
        is_active = False
        
        if condition_met:
            is_active = True
            if not self.global_alert_has_been_logged[alert_key]:
                self.logger.log_evidence(frame, alert_key, message)
                self.global_alert_has_been_logged[alert_key] = True
        else:
            if self.global_alert_has_been_logged[alert_key]:
                    self.logger.log_activity(f"Condition {alert_key} resolved.")
            self.global_alert_has_been_logged[alert_key] = False
                
        return is_active