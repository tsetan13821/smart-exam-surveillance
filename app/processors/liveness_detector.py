# app/processors/liveness_detector.py
import config as config
import cv2 
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from app.utils import liveness_helpers 

class LivenessDetector:
    """
    A stateful processor dedicated to detecting static image "spoof" attacks.
    ... (class docstring) ...
    """
    def __init__(self, person, logger):
        self.person = person # Store the person object
        self.logger = logger # Store the logger instance
        
        # Get the initial logging name
        person_name = self.person.get_logging_name()
        self.logger.log_liveness_debug("Initializing Liveness Detector...", person_name)
        
        # --- Configs for 1.4 ---
        self.MP_MISSING_THRESHOLD = getattr(config, "LIVENESS_MP_MISSING_THRESHOLD", 3)
        
        # --- NEW Configs for 2.1 & 2.3 ---
        self.CONFIRM_SECONDS = getattr(config, "LIVENESS_CONFIRM_SECONDS", 3.0)
        
        # Pixel check (2.1) configs
        # Assume 3s @ 30fps = 90 frames. Use a shorter default for safety.
        ssim_frames = getattr(config, "LIVENESS_SSIM_WINDOW_FRAMES", 60)
        flow_frames = getattr(config, "LIVENESS_FLOW_WINDOW_FRAMES", 60)
        self.SSIM_STATIC_THRESHOLD = getattr(config, "LIVENESS_SSIM_STATIC_THRESHOLD", 0.98)
        self.FLOW_STATIC_THRESHOLD = getattr(config, "LIVENESS_FLOW_STATIC_THRESHOLD", 0.4)
        
        # --- NEW: Decoupled Sway Thresholds ---
        self.LIVENESS_SWAY_THRESHOLD = getattr(config, "LIVENESS_HEAD_SWAY_THRESHOLD", 2.5)
        self.EXCESSIVE_SWAY_THRESHOLD = getattr(config, "EXCESSIVE_MOVEMENT_SWAY_THRESHOLD", 5.0)
        self.logger.log_liveness_debug(f"Sway Thresholds: Liveness={self.LIVENESS_SWAY_THRESHOLD} deg, Excessive={self.EXCESSIVE_SWAY_THRESHOLD} deg", person_name)
        # --- END NEW ---
        
        # State for liveness timer
        self.last_live_event_time = datetime.now()
        
        # State for Blink Check
        self.current_blink_state = "open"
        self.blink_closed_frames = 0
        
        # State for Head Sway Check
        self.last_known_liveness_pose = (0.0, 0.0) # (yaw, pitch)
        
        # State for Gaze Shift Check
        self.last_gaze_direction = "Center"
        
        # --- NEW STATE for 1.4 (MP Missing) ---
        self.mp_missing_frames = 0
        
        # --- NEW STATE for 2.1 (Pixel Checks) ---
        self.prev_face_crop_gray = None
        self.ssim_scores = deque(maxlen=ssim_frames)
        self.flow_scores = deque(maxlen=flow_frames)

    def reset(self):
        """
        Resets the liveness timer and all check states.
        Called when a new face session begins.
        """
        person_name = self.person.get_logging_name()
        self.logger.log_liveness_debug("Liveness Detector Reset.", person_name)
        self.last_live_event_time = datetime.now()
        self.current_blink_state = "open"
        self.blink_closed_frames = 0
        self.last_known_liveness_pose = (0.0, 0.0)
        self.last_gaze_direction = "Center"
        
        # --- NEW 1.4 Reset ---
        self.mp_missing_frames = 0
        
        # --- NEW 2.1 Reset ---
        self.prev_face_crop_gray = None
        self.ssim_scores.clear()
        self.flow_scores.clear()
        
        # --- NEW 2.3 Reset ---
        # The person's state is reset when a new TrackedPerson
        # is created. This line is for safety if reset() is
        # called mid-track.
        self.person.liveness_state = "UNCONFIRMED"

    # --- <<< MODIFIED: Updated signature & return value >>> ---
    def check_liveness(self, mp_results, frame, box) -> tuple[bool, bool]:
        """
        Checks the provided MediaPipe results for any "proof of life".
        ... (docstring) ...
        Returns:
            (bool): is_spoofing
            (bool): excessive_sway_event_detected
        """
        
        # --- <<< NEW: Get person's logging name once per frame >>> ---
        person_name = self.person.get_logging_name()
        
        liveness_event_detected = False
        excessive_sway_event_detected = False # <-- NEW
        debug_events = [] # A list to hold all events detected this frame

        if not mp_results:
            # --- MODIFIED for 1.4 ---
            self.mp_missing_frames += 1
            self.logger.log_liveness_debug(f"No MediaPipe results. Missing frames: {self.mp_missing_frames}", person_name)
            # --- END MODIFIED ---
        else:
            # --- NEW 1.4: Reset missing counter ---
            self.mp_missing_frames = 0
            
            # --- 1. LIVENESS EVENT: BLINK (with jitter filter) ---
            if not mp_results.get("ear_uncertain", True): 
                current_ear = mp_results["ear"]
                self.logger.log_liveness_debug(f"Checking Blinks -> EAR: {current_ear:4.2f}", person_name)
                
                if self.current_blink_state == "open":
                    if current_ear < config.REG_LIVENESS_BLINK_EAR:
                        self.blink_closed_frames += 1
                        if self.blink_closed_frames >= config.LIVENESS_BLINK_MIN_FRAMES:
                            self.current_blink_state = "closed" 
                    else:
                        self.blink_closed_frames = 0
                elif self.current_blink_state == "closed":
                    if current_ear > (config.REG_LIVENESS_BLINK_EAR + config.LIVENESS_BLINK_HYSTERESIS):
                        liveness_event_detected = True # BLINK CONFIRMED
                        debug_events.append("Blink")
                        self.current_blink_state = "open"
                        self.blink_closed_frames = 0
            else:
                self.logger.log_liveness_debug("Skipping Blinks (EAR uncertain)", person_name)

            # --- 2. LIVENESS EVENT: HEAD SWAY (DECOUPLED) ---
            if not mp_results.get("pose_uncertain", True):
                current_yaw = mp_results["yaw"]
                current_pitch = mp_results["pitch"]
                
                self.logger.log_liveness_debug(f"Checking Pose   -> YAW: {current_yaw:6.2f} | PITCH: {current_pitch:6.2f}", person_name)
                yaw_delta = abs(current_yaw - self.last_known_liveness_pose[0]) 
                pitch_delta = abs(current_pitch - self.last_known_liveness_pose[1]) 
                
                # --- NEW LOGIC: Check both thresholds ---
                
                # Check for liveness sway (small movement)
                liveness_pose_changed = (yaw_delta > self.LIVENESS_SWAY_THRESHOLD or \
                                         pitch_delta > self.LIVENESS_SWAY_THRESHOLD)
                
                if liveness_pose_changed:
                    liveness_event_detected = True # Resets liveness timer
                    debug_events.append(f"LivenessSway(Y_D:{yaw_delta:.1f},P_D:{pitch_delta:.1f})")
                    self.last_known_liveness_pose = (current_yaw, current_pitch)

                # Check for excessive sway (larger movement)
                excessive_pose_changed = (yaw_delta > self.EXCESSIVE_SWAY_THRESHOLD or \
                                          pitch_delta > self.EXCESSIVE_SWAY_THRESHOLD)
                
                if excessive_pose_changed:
                    excessive_sway_event_detected = True # Sent to person_analyzer for counting
                    debug_events.append(f"ExcessiveSway(Y_D:{yaw_delta:.1f},P_D:{pitch_delta:.1f})")
                    # We also update last_known_liveness_pose here if it wasn't already
                    if not liveness_pose_changed:
                         self.last_known_liveness_pose = (current_yaw, current_pitch)
                
            else:
                self.logger.log_liveness_debug("Skipping Pose (Pose uncertain)", person_name)

            # --- 3. LIVENESS EVENT: GAZE SHIFT ---
            if not mp_results.get("gaze_uncertain", True):
                current_gaze = mp_results["gaze"]
                
                self.logger.log_liveness_debug(f"Checking Gaze   -> GAZE: {current_gaze:<10}", person_name)

                if current_gaze != self.last_gaze_direction and current_gaze != "Uncertain":
                    liveness_event_detected = True # GAZE SHIFT CONFIRMED
                    debug_events.append(f"GazeShift({self.last_gaze_direction}→{current_gaze})")
                    self.last_gaze_direction = current_gaze
            else:
                self.logger.log_liveness_debug("Skipping Gaze (Gaze uncertain)", person_name)
        
        # --- <<< NEW 2.1: Pixel Liveness Checks (Data Gathering) >>> ---
        curr_face_crop_gray = None
        if frame is not None:
            try:
                x1, y1, x2, y2 = [int(b) for b in box]
                pad_x, pad_y = (x2-x1)//4, (y2-y1)//4
                h, w, _ = frame.shape
                x1, y1 = max(0, x1-pad_x), max(0, y1-pad_y)
                x2, y2 = min(w, x2+pad_x), min(h, y2+pad_y)
                
                if y2 > y1 and x2 > x1: 
                    curr_face_crop = frame[y1:y2, x1:x2]
                    curr_face_crop_gray = cv2.cvtColor(curr_face_crop, cv2.COLOR_BGR2GRAY)
                    
                    if self.prev_face_crop_gray is not None:
                        flow_score = liveness_helpers.mean_optical_flow(self.prev_face_crop_gray, curr_face_crop_gray)
                        ssim_score = liveness_helpers.ssim_or_mse(self.prev_face_crop_gray, curr_face_crop_gray)
                        
                        if flow_score is not None: self.flow_scores.append(flow_score)
                        if ssim_score is not None: self.ssim_scores.append(ssim_score)
                        
                    self.prev_face_crop_gray = curr_face_crop_gray
                    
            except Exception as e:
                self.logger.log_liveness_debug(f"Pixel check error: {e}", person_name)
                self.prev_face_crop_gray = None # Reset on error
        # --- <<< END 2.1 >>> ---


        # --- <<< NEW 2.3: Liveness State Machine >>> ---
        
        current_state = self.person.liveness_state
        time_since_last_event = (datetime.now() - self.last_live_event_time).total_seconds()
        is_spoofing = False # Default
        
        avg_flow = 0.0
        if len(self.flow_scores) > 10: 
            avg_flow = np.mean(self.flow_scores)
        
        
        if current_state == "UNCONFIRMED":
            self.logger.log_liveness_debug(f"State: UNCONFIRMED. Waiting for positive liveness. (Time: {time_since_last_event:.2f}s / {self.CONFIRM_SECONDS}s)", person_name)
            
            # Check for positive liveness
            if liveness_event_detected:
                self.person.liveness_state = "CONFIRMED_LIVE"
                self.logger.log_liveness_debug(f"*** LIVENESS CONFIRMED (Event) *** Reasons: {', '.join(debug_events)}", person_name)
                self.last_live_event_time = datetime.now() 
            
            # Check for pixel-based liveness (as a backup)
            elif avg_flow > self.FLOW_STATIC_THRESHOLD:
                self.person.liveness_state = "CONFIRMED_LIVE"
                self.logger.log_liveness_debug(f"*** LIVENESS CONFIRMED (Pixel Flow) *** Avg: {avg_flow:.2f}", person_name)
                self.last_live_event_time = datetime.now()
            
            # Check if confirmation window has *expired*
            elif time_since_last_event > self.CONFIRM_SECONDS:
                is_spoofing = True
                self.logger.log_liveness_debug(f"!!! IMAGE DETECTED (Failed to Confirm) !!! No liveness event in {self.CONFIRM_SECONDS}s", person_name)

        elif current_state == "CONFIRMED_LIVE":
            
            # Check for positive liveness *first*.
            if liveness_event_detected:
                self.last_live_event_time = datetime.now()
                self.logger.log_liveness_debug(f"State: CONFIRMED. *** TIMER RESET (Event) *** Reason(s): {', '.join(debug_events)}", person_name)
                return False, excessive_sway_event_detected # This frame is live
            
            # Check for positive *pixel* liveness.
            if avg_flow > self.FLOW_STATIC_THRESHOLD:
                self.last_live_event_time = datetime.now()
                self.logger.log_liveness_debug(f"State: CONFIRMED. *** TIMER RESET (Flow) *** Avg Flow {avg_flow:.3f} > {self.FLOW_STATIC_THRESHOLD}", person_name)
                return False, excessive_sway_event_detected # This frame is live (pixel motion detected)
            
            # --- IF WE GET HERE: No MP event AND no pixel flow was detected ---
            # Now we are safe to check for *static* conditions.

            # Check for static pixel-based spoof (2.1)
            is_static_ssim = False
            if len(self.ssim_scores) == self.ssim_scores.maxlen:
                avg_ssim = np.mean(self.ssim_scores)
                if avg_ssim > self.SSIM_STATIC_THRESHOLD:
                    is_static_ssim = True
                    self.logger.log_liveness_debug(f"!!! IMAGE DETECTED (SSIM) !!! Avg SSIM {avg_ssim:.3f} > {self.SSIM_STATIC_THRESHOLD}", person_name)
                    
            # Check liveness timer (1.3)
            current_spoof_threshold = config.LIVENESS_SPOOF_THRESHOLD_SEC
            is_spoofing_timer = (time_since_last_event > current_spoof_threshold)
            
            if is_spoofing_timer:
                 self.logger.log_liveness_debug(f"!!! IMAGE DETECTED (Timer) !!! Timer exceeded {current_spoof_threshold}s", person_name)
            
            # --- <<< LOGIC FIX #3 >>> ---
            # The *only* triggers are the timer or high SSIM.
            # Low flow (is_static_flow) is NO LONGER a trigger.
            is_spoofing = is_spoofing_timer or is_static_ssim
            # --- <<< END LOGIC FIX #3 >>> ---

            # --- NEW 1.4: Conservative check for missing MP ---
            if is_spoofing and self.mp_missing_frames > 0:
                if self.mp_missing_frames <= self.MP_MISSING_THRESHOLD:
                    is_spoofing = False # Override spoof detection
                    self.logger.log_liveness_debug(f"!!! SPOOF OVERRIDDEN !!! MP results missing for {self.mp_missing_frames} frames (Threshold: {self.MP_MISSING_THRESHOLD}).", person_name)
                else:
                    self.logger.log_liveness_debug(f"!!! Image CONFIRMED (MP Missing) !!! Timer exceeded and MP missing for {self.mp_missing_frames} frames.", person_name)
        
        return is_spoofing, excessive_sway_event_detected # <-- MODIFIED RETURN