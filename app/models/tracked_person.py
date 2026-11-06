# app/models/tracked_person.py
import config as config
from datetime import datetime
from app.processors.liveness_detector import LivenessDetector
from collections import deque # <-- NEW IMPORT 

"""
This file defines the data model for a single tracked person.
It holds the state for that person, including their liveness detector,
alert timers, and last-known actions.
"""

class TrackedPerson:
    def __init__(self, person_id, name, box, logger):
        self.id = person_id
        self.name = name
        self.box = box
        self.centroid = TrackedPerson._get_centroid(box)
        self.logger = logger
        
        # --- <<< CRITICAL FIX: INITIALIZE SPOOF_ID FIRST >>> ---
        self.spoof_id = None 
        # --- <<< END CRITICAL FIX >>> ---
        
        # Each person gets their own LivenessDetector
        # The constructor for LivenessDetector calls get_logging_name(),
        # which is why self.spoof_id MUST be initialized before this line.
        self.liveness_detector = LivenessDetector(person=self, logger=logger)
        
        self.liveness_check_active = True # Active on creation
        self.is_spoofing = False
        self.liveness_state = "UNCONFIRMED"
        
        # --- NEW: Cheating Detector States (Phase 1) ---
        self.gaze_away_start_time = None
        self.gaze_away_duration_sec = 0.0

        self.absent_start_time = None
        self.absent_duration_sec = 0.0
        
        # label -> [first_seen, last_seen] of the current continuous sighting
        # of each device on this student's desk
        self.device_sightings = {}
        # Stores active cheating rule violations (e.g., "SPOOFING", "DEVICE_USE")
        self.cheating_state = set()
        # --- END NEW ---
        
        # --- NEW: For Excessive Movement ---
        self.head_sway_timestamps = deque()
        
        # Each person gets their own state trackers
        self.last_action_pose = "Center"
        self.last_action_gaze = "Center"
        self.last_action_eyes = "Open"
        self.last_action_mouth = "Closed"
        
        # Each person gets their own alert timers
        self.alert_start_times = {
            "IMAGE_DETECTED": None,
            # --- NEW ---
            "HEAD_TURN_ALERT": None,
            "EXCESSIVE_MOVEMENT_ALERT": None
            # --- REMOVED "HEAD_DOWN" and other pose alerts ---
        }
        # This tracks if an alert *has been logged* to avoid spam
        # This dictionary comprehension automatically picks up the new keys
        self.alert_has_been_logged = {k: False for k in self.alert_start_times}
        
        self.last_gaze_direction = "Center"
        self.gaze_change_times = []
        
        # This will be updated by the main loop with the latest data
        self.mp_result = None 
        self.last_seen = datetime.now()
        self.current_alerts = [] # List of active alerts for this person

    def record_device(self, label, now):
        """Records that `label` was seen on this student's desk at `now`."""
        max_gap = getattr(config, "CHEATING_DEVICE_GAP_SEC", 3.0)
        seen = self.device_sightings.get(label)
        if seen and (now - seen[1]).total_seconds() <= max_gap:
            seen[1] = now                       # same continuous sighting
        else:
            self.device_sightings[label] = [now, now]

    @staticmethod
    def _get_centroid(box):
        """Calculates the centroid of a bounding box."""
        x1, y1, x2, y2 = box
        return (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))

    def update(self, box, name, mp_result):
        """Updates the person's state with new data from the current frame."""
        self.box = box
        self.centroid = TrackedPerson._get_centroid(box)
        self.mp_result = mp_result
        
        # --- <<< MODIFICATION START >>> ---
        # Always trust the FaceRecognizer's current assessment.
        # The old logic incorrectly kept the "Student Name" even
        # if the recognizer was now seeing "Unknown" (e.g., a spoof).
        self.name = name
        # --- <<< MODIFICATION END >>> ---
        
        self.last_seen = datetime.now()

    def student_name(self):
        """The recognised student's name, or None for an unknown face."""
        return self.name if self.name and self.name != "Unknown" else None

    def get_display_name(self):
        """Returns the name to be drawn on the box."""
        # --- <<< FIX: Prioritize spoof_id if it exists >>> ---
        if self.spoof_id:
            return self.spoof_id # Returns "Image-1", "Image-2", etc.
        # --- <<< END FIX >>> ---
        
        if self.name != "Unknown":
            return self.name # e.g., "Student Name"
        
        return f"Unknown ({self.id})" # e.g., "Unknown (Person-1)"

    def get_logging_name(self):
        """
        Returns the ID to be used in debug logs, as requested.
        - "Image-X" if confirmed spoof.
        - "Person-X" (the ID) if Unknown.
        - "Student Name" (the name) if known.
        """
        if self.spoof_id:
            return self.spoof_id  # e.g., "Image-1"
        
        if self.name == "Unknown":
            return self.id       # e.g., "Person-1"
            
        return self.name         # e.g., "Student Name"