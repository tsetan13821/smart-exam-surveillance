# app/processors/face_processor.py
import cv2
import config as config
import numpy as np

class FaceProcessor:
    """
    Unified processor for face detection.
    
    --- MODIFICATION (v2) ---
    This class now checks if *any* authorized student is present,
    based on the multi-student FaceRecognizer.
    """
    def __init__(self, face_recognizer):
        print("Initializing Unified Face Processor (State Tracker)...")
        
        self.face_recognizer = face_recognizer # Store the injected recognizer

        # --- Config Parameters ---
        # --- FIX #6: Use getattr for robust config loading ---
        # Use a default of 5 frames if not specified in config.py
        self.detection_interval = getattr(config, "FACE_DETECTION_INTERVAL", 5) 
        # --- END FIX ---

        # --- State Tracking ---
        self.frame_count = 0
        self.face_present = False # General flag if *any* face is present
        
        # --- MODIFIED: Was self.student_present ---
        self.authorized_student_present = False # Specific flag for *any* registered student
        
        self.last_known_boxes = []
        self.last_known_names = []
        self.last_unauthorized_alert_state = False
        
        self.alert_logger = None

    def set_alert_logger(self, alert_logger):
        self.alert_logger = alert_logger

    
    def detect_faces(self, frame):
        """
        Calls the FaceRecognizer to get all face data, updates internal state,
        and returns key information.
        """
        self.frame_count += 1

        default_return = {
            'face_present': self.face_present,
            # --- MODIFIED: Return the correct multi-student flag ---
            'student_present': self.authorized_student_present, 
            'boxes': self.last_known_boxes,
            'names': self.last_known_names,
            'unauthorized_alert_active': self.last_unauthorized_alert_state, 
        }

        if self.frame_count % self.detection_interval != 0:
            return default_return

        # --- IT IS AN INTERVAL FRAME, RUN DETECTION ---
        
        boxes, names = self.face_recognizer.identify_faces(frame)

        # --- NEW: Multi-Student Presence Check ---
        
        # Get the set of authorized names (e.g., {"Alice", "Bob"})
        authorized_names_set = self.face_recognizer.get_authorized_names_set()
        
        # Check if *any* name in the 'names' list (e.g., ["Alice", "Unknown"])
        # is present in the authorized set.
        self.authorized_student_present = any(name in authorized_names_set for name in names)
        
        unauthorized_face_detected = "Unknown" in names
        # --- END NEW ---

        if len(boxes) >= 1:
            self.face_present = True
        else:
            self.face_present = False

        self.last_known_boxes = boxes
        self.last_known_names = names
        self.last_unauthorized_alert_state = unauthorized_face_detected
        
        # Return the results for this *current* interval frame
        return {
            'face_present': self.face_present,
            # --- MODIFIED: Return the correct multi-student flag ---
            'student_present': self.authorized_student_present,
            'boxes': boxes, 
            'names': names,
            'unauthorized_alert_active': unauthorized_face_detected,
        }