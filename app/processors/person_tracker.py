# app/processors/person_tracker.py
import math
from datetime import datetime
from app.models.tracked_person import TrackedPerson

"""
This processor is responsible for stateful person tracking.

It takes raw detections from face_processor and mediapipe_processor
and links them together, maintaining a persistent ID for each
person from frame to frame using centroid tracking.
"""

class PersonTracker:
    def __init__(self, logger):
        self.logger = logger
        
        # --- Configuration ---
        # Max time to track a person without seeing them
        self.MAX_DISAPPEARED_SEC = 5 
        # Max pixels a centroid can move between frames
        self.MAX_CENTROID_DIST = 150 
        
        # --- <<< NEW CONFIGURATION START >>> ---
        # Max pixels a *confirmed spoof* centroid can move
        # A static image shouldn't move much. This gives a small buffer
        # for detection jitter.
        self.MAX_CENTROID_DIST_SPOOF = 30 
        # --- <<< NEW CONFIGURATION END >>> ---
        
        # --- State ---
        self.tracked_persons = {} # Holds TrackedPerson objects, keyed by ID
        self.next_person_id = 1

    # --- Helper 1: Get Nose Tip ---
    def _get_mp_nose_tip(self, mp_result, w, h):
        """Gets the (x,y) pixel coordinate of the nose tip from MediaPipe results."""
        if not mp_result or 'landmarks_object' not in mp_result:
            return None
        try:
            # Index 1 is the nose tip
            lm = mp_result['landmarks_object'].landmark[1] 
            if not (0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0):
                return None
            return (int(lm.x * w), int(lm.y * h))
        except Exception:
            return None

    # --- Helper 2: Check Point in Box ---
    def _is_point_in_box(self, point, box):
        """Checks if an (x,y) point is inside a (x1, y1, x2, y2) box."""
        x, y = point
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    @staticmethod
    def _expand_box(box, frac):
        x1, y1, x2, y2 = box
        dx, dy = (x2 - x1) * frac, (y2 - y1) * frac
        return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)

    # --- Helper 3: Euclidean Distance ---
    @staticmethod
    def _euclidean_dist(p1, p2):
        """Helper to calculate euclidean distance between two (x,y) points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # --- Main Tracking Logic ---
    # --- <<< MODIFICATION: Added 'frame' parameter >>> ---
    def update_tracks(self, face_results, list_of_mp_results, w, h, frame):
        """
        Updates the tracker with new frame data and returns the dictionary
        of currently tracked persons.
        """
        
        # --- 1. Link Face Boxes with MediaPipe Results ---
        linked_faces = [] # (box, name, mp_result, centroid)
        unmatched_mp_results = list(list_of_mp_results)
        
        for box, name in zip(face_results['boxes'], face_results['names']):
            best_match_mp = None
            best_match_idx = -1
            
            # Find an mp_result whose nose tip is inside this box
            for i, mp_res in enumerate(unmatched_mp_results):
                nose_tip = self._get_mp_nose_tip(mp_res, w, h)
                # Face boxes can lag the landmarks by one inference cycle, so
                # allow some margin around the box when linking them.
                if nose_tip and self._is_point_in_box(nose_tip, self._expand_box(box, 0.2)):
                    best_match_mp = mp_res
                    best_match_idx = i
                    break # Found a match
            
            centroid = TrackedPerson._get_centroid(box)
            if best_match_mp:
                # Add to linked list
                linked_faces.append((box, name, best_match_mp, centroid))
                # Remove from pool so it can't be matched twice
                unmatched_mp_results.pop(best_match_idx)
            else:
                # This face box has no landmarks (e.g., too blurry, or MP failed)
                # Still, add it to the tracker so it can be tracked
                linked_faces.append((box, name, None, centroid))

        # --- 2. Handle No Detections ---
        if not linked_faces:
            # No faces detected. Prune any old trackers.
            to_deregister = []
            for person_id, person in self.tracked_persons.items():
                # Not seen this frame: drop the stale landmarks so liveness and
                # gaze aren't judged on a frozen face.
                person.mp_result = None

                # --- NEW ABSENCE LOGIC (Phase 2) ---
                if person.absent_start_time is None:
                    person.absent_start_time = datetime.now()
                person.absent_duration_sec = (datetime.now() - person.absent_start_time).total_seconds()
                # --- END NEW LOGIC ---
                
                if (datetime.now() - person.last_seen).total_seconds() > self.MAX_DISAPPEARED_SEC:
                    to_deregister.append(person_id)
            for person_id in to_deregister:
                self.logger.log_core_debug(f"Deregistering {person_id} (timeout).")
                if person_id in self.tracked_persons:
                    del self.tracked_persons[person_id]
            return self.tracked_persons # Return the (now pruned) list

        # --- 3. Build Centroid Lists ---
        existing_ids = list(self.tracked_persons.keys())
        existing_centroids = [self.tracked_persons[pid].centroid for pid in existing_ids]
        current_centroids = [data[3] for data in linked_faces]
        
        # --- 4. Match Existing Persons to Current Detections ---
        # --- <<< MODIFIED SECTION START >>> ---
        used_existing_indices = set()
        used_current_indices = set()
        distances = []
        
        if existing_centroids:
            for i, exist_c in enumerate(existing_centroids):
                person_id = existing_ids[i]
                person = self.tracked_persons[person_id]
                
                # Determine the correct distance threshold for this person
                is_spoof = (person.spoof_id is not None)
                current_max_dist = self.MAX_CENTROID_DIST_SPOOF if is_spoof else self.MAX_CENTROID_DIST
                
                # Log if we're using the strict threshold
                if is_spoof:
                    self.logger.log_core_debug(f"Tracker: {person.id} ({person.get_logging_name()}) is spoof. Using strict threshold ({current_max_dist}px).")

                for j, current_c in enumerate(current_centroids):
                    dist = self._euclidean_dist(exist_c, current_c)
                    
                    if dist < current_max_dist: # Use the *dynamic* threshold
                        distances.append((dist, i, j))
        # --- <<< MODIFIED SECTION END >>> ---
        
        distances.sort()
        
        for dist, exist_idx, current_idx in distances:
            if exist_idx in used_existing_indices or current_idx in used_current_indices:
                continue
                
            person_id = existing_ids[exist_idx]
            box, name, mp_result, _ = linked_faces[current_idx]
            self.tracked_persons[person_id].update(box, name, mp_result)
            
            # --- NEW ABSENCE RESET (Phase 2) ---
            self.tracked_persons[person_id].absent_start_time = None
            self.tracked_persons[person_id].absent_duration_sec = 0.0
            # --- END NEW RESET ---
            
            used_existing_indices.add(exist_idx)
            used_current_indices.add(current_idx)

        # --- 5. Deregister Old Persons ---
        to_deregister = []
        for i, person_id in enumerate(existing_ids):
            if i not in used_existing_indices:
                person = self.tracked_persons[person_id]
                person.mp_result = None   # not seen this frame: no fresh landmarks

                # --- NEW ABSENCE LOGIC (Phase 2) ---
                if person.absent_start_time is None:
                    person.absent_start_time = datetime.now()
                person.absent_duration_sec = (datetime.now() - person.absent_start_time).total_seconds()
                # --- END NEW LOGIC ---

                if (datetime.now() - person.last_seen).total_seconds() > self.MAX_DISAPPEARED_SEC:
                    to_deregister.append(person_id)
        
        for person_id in to_deregister:
            self.logger.log_core_debug(f"Deregistering {person_id} (disappeared).")
            if person_id in self.tracked_persons:
                del self.tracked_persons[person_id]

        # --- 6. Register New Persons ---
        for i, (box, name, mp_result, centroid) in enumerate(linked_faces):
            if i not in used_current_indices:
                new_id = f"Person-{self.next_person_id}"
                self.next_person_id += 1
                
                self.logger.log_core_debug(f"Registering new person: {new_id} ({name})")
                
                # --- <<< NEW CODE START: Evidence Logging >>> ---
                # Check if the newly registered person is "Unknown"
                if name == "Unknown":
                    # Log screenshot evidence
                    self.logger.log_evidence(frame, "UNAUTHORIZED_PERSON_REGISTERED", f"New unauthorized person detected: {new_id}")
                # --- <<< NEW CODE END >>> ---
                
                new_person = TrackedPerson(new_id, name, box, self.logger)
                new_person.update(box, name, mp_result) # Do initial update
                self.tracked_persons[new_id] = new_person

        return self.tracked_persons