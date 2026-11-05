# app/processors/mediapipe_processor.py  (architecture-preserving update)
import cv2
import numpy as np
import mediapipe as mp
import inspect
import config as config
from datetime import datetime
from collections import deque
import math
import os

class MediaPipeProcessor:
    
    @staticmethod
    def _euclidean_dist(p1, p2):
        """Helper to calculate euclidean distance between two (x,y) points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def set_alert_logger(self, logger):
        self.logger = logger
        if self.logger:
            self.logger.log_mediapipe_debug("CONFIG", f"Initialized MediaPipeProcessor with thresholds",
                                          {"PRESENCE": self.PRESENCE_THRESHOLD, "INFERRED": self.INFERRED_CONF,
                                           "EDGE_MARGIN": self.EDGE_MARGIN, "MAX_ABS_Z": self.MAX_ABS_Z,
                                           "TEMPORAL_WINDOW": self.TEMPORAL_WINDOW})
            self.logger.log_mediapipe_debug("CONFIG", f"EAR_THRESH={self.EAR_THRESHOLD}, MAR_THRESH={self.MAR_THRESHOLD}, GAZE_THRESH={self.GAZE_THRESHOLD}")
            self.logger.log_mediapipe_debug("CONFIG", f"FaceMesh initialized with params: {self.filtered_params}")

    def __init__(self):
        print("Initializing MediaPipe Processor...")

        # --- Config thresholds ---
        self.PRESENCE_THRESHOLD = getattr(config, "MEDIAPIPE_PRESENCE_THRESHOLD", 0.6)
        self.INFERRED_CONF = getattr(config, "MEDIAPIPE_INFERRED_CONFIDENCE", 0.8)
        self.EDGE_MARGIN = getattr(config, "MEDIAPIPE_EDGE_MARGIN", 0.02)
        self.MAX_ABS_Z = getattr(config, "MEDIAPIPE_MAX_ABS_Z", 0.5)
        self.TEMPORAL_WINDOW = getattr(config, "MEDIAPIPE_TEMPORAL_WINDOW", 5)
        
        self.EAR_THRESHOLD = getattr(config, "EYE_EAR_THRESHOLD", 0.2)
        self.MAR_THRESHOLD = getattr(config, "MOUTH_MAR_THRESHOLD", 0.5)
        self.GAZE_THRESHOLD = getattr(config, "EYE_GAZE_THRESHOLD", 10.0) 
        
        # --- Camera / solvePnP parameters ---
        self.CAM_FOCAL_SCALE = getattr(config, "CAMERA_FOCAL_SCALE", 1.2)
        self.CAM_MATRIX = getattr(config, "CAMERA_MATRIX", None)
        self.USE_PNP_REFINE = getattr(config, "USE_PNP_REFINE", True)
        self.DIST_COEFFS = getattr(config, "DIST_COEFFS", None)

        # --- NEW: Optional smoothing ---
        self.ENABLE_SMOOTHING = getattr(config, "GAZE_ENABLE_SMOOTHING", True)
        self.SMOOTHING_ALPHA = float(getattr(config, "GAZE_SMOOTHING_ALPHA", 0.35))
        self._ema = {"lp": None, "rp": None, "hit": None}

        fm_ctor = mp.solutions.face_mesh.FaceMesh.__init__
        fm_params = list(inspect.signature(fm_ctor).parameters.keys())
        want = {
            "static_image_mode": False,
            "max_num_faces": getattr(config, "MEDIAPIPE_MAX_NUM_FACES", 1),
            "min_detection_confidence": getattr(config, "MEDIAPIPE_MIN_DET_CONF", 0.5), 
            "min_tracking_confidence": getattr(config, "MEDIAPIPE_MIN_TRACK_CONF", 0.5),
            "refine_landmarks": True, # REQUIRED for iris/pupil indices
        }
        filtered = {k: v for k, v in want.items() if k in fm_params}
        
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(**filtered)
        self.filtered_params = filtered 

        self.logger = None
        self._mp_stats = {"frames": 0, "inferred_landmarks": 0, "explicit_conf_landmarks": 0}
        self.presence_buffer = deque(maxlen=self.TEMPORAL_WINDOW)

        # Landmarks
        self.HEAD_POSE_LANDMARKS = [1, 152, 263, 33, 287, 57]
        self.LEFT_EYE_LMS = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_LMS = [33, 160, 158, 133, 153, 144]
        self.MOUTH_MAR_LMS_VERT = [13, 14]
        self.MOUTH_MAR_LMS_HORIZ = [78, 308]
        self.NOSE_TIP_INDEX = [1]

        # Pupil landmark indices (iris center-ish)
        self.LEFT_PUPIL_LMS = [473]
        self.RIGHT_PUPIL_LMS = [468]

        # Cache frame size for intrinsics
        self._cached_size = None  # (h, w)

    def preprocess_frame(self, frame_rgb):
        try:
            lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_eq = clahe.apply(l)
            lab_eq = cv2.merge((l_eq, a, b))
            # Contrast equalisation only. A sharpening pass used to follow here,
            # but it amplified camera noise and tripled landmark jitter.
            return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        except Exception:
            return frame_rgb 

    # ----------------------------- Landmarks -----------------------------
    def _get_valid_landmarks(self, face_landmarks, indices, h, w):
        coords = []
        for idx in indices:
            try:
                lm = face_landmarks.landmark[idx]
                presence = getattr(lm, "presence", 0.0)
                visibility = getattr(lm, "visibility", 0.0)
                used = max(presence, visibility)

                if used <= 1e-5:
                    used = self.INFERRED_CONF
                    self._mp_stats["inferred_landmarks"] += 1
                else:
                    self._mp_stats["explicit_conf_landmarks"] += 1

                if abs(lm.z) > self.MAX_ABS_Z:
                    continue

                if not (self.EDGE_MARGIN <= lm.x <= 1 - self.EDGE_MARGIN and
                        self.EDGE_MARGIN <= lm.y <= 1 - self.EDGE_MARGIN):
                    continue

                self.presence_buffer.append(used)
                if len(self.presence_buffer) == self.TEMPORAL_WINDOW:
                    avg_presence = np.mean(self.presence_buffer)
                    used = 0.5 * used + 0.5 * avg_presence 

                if used < self.PRESENCE_THRESHOLD:
                    continue

                # Keep sub-pixel precision. Rounding to whole pixels made head
                # pose jump by several degrees and EAR by ~30% between frames
                # on small faces, which swamped the liveness signals.
                coords.append((lm.x * w, lm.y * h))

            except Exception as e:
                return None, f"Error accessing landmark {idx}: {e}"

        if not coords:
            return None, f"All landmarks invalid (Threshold {self.PRESENCE_THRESHOLD})"
        
        if len(coords) != len(indices):
             return None, f"Not all landmarks were valid. Got {len(coords)}, need {len(indices)}"
             
        return coords, None

    # -------------------------- Camera intrinsics ------------------------
    def _ensure_camera_matrix(self, h, w):
        self._cached_size = (h, w)
        if self.CAM_MATRIX is not None:
            try:
                cm = np.array(self.CAM_MATRIX, dtype=np.float64)
                if cm.shape == (3,3):
                    return cm, (np.zeros((5,1)) if self.DIST_COEFFS is None else np.array(self.DIST_COEFFS, dtype=np.float64).reshape((-1,1)))
            except Exception:
                pass
        f = float(self.CAM_FOCAL_SCALE) * float(max(w, h))
        cam_matrix = np.array([[f, 0, w / 2.0],[0, f, h / 2.0],[0, 0, 1.0]], dtype=np.float64)
        dist = np.zeros((5,1), dtype=np.float64) if self.DIST_COEFFS is None else np.array(self.DIST_COEFFS, dtype=np.float64).reshape((-1,1))
        self.CAM_MATRIX = cam_matrix
        self.DIST_COEFFS = dist
        return cam_matrix, dist

    # ---------------------------- Head pose ------------------------------
    def _calculate_head_pose(self, face_landmarks, h, w):
        face_2d, err = self._get_valid_landmarks(face_landmarks, self.HEAD_POSE_LANDMARKS, h, w)
        if face_2d is None:
            return None, None, None, err

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array([
            [0.0, 0.0, 0.0],          # Nose tip
            [0.0, -330.0, -65.0],     # Chin
            [225.0, 170.0, -135.0],   # Left eye corner
            [-225.0, 170.0, -135.0],  # Right eye corner
            [150.0, -150.0, -125.0],  # Left mouth corner
            [-150.0, -150.0, -125.0]  # Right mouth corner
        ], dtype=np.float64)

        cam_matrix, dist = self._ensure_camera_matrix(h, w)

        try:
            success, rvec, tvec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if not success:
                return None, None, None, "solvePnP returned success=False"

            if self.USE_PNP_REFINE and hasattr(cv2, "solvePnPRefineLM"):
                try:
                    cv2.solvePnPRefineLM(face_3d, face_2d, cam_matrix, dist, rvec, tvec)
                except Exception:
                    pass

            R, _ = cv2.Rodrigues(rvec)
            sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
            singular = sy < 1e-6
            if not singular:
                pitch_rad = math.atan2(-R[2,1], -R[2,2])
                yaw_rad   = math.atan2(-R[2,0], sy)
            else:
                pitch_rad = math.atan2(-R[1,2], R[1,1])
                yaw_rad   = 0.0

            yaw = math.degrees(yaw_rad)
            pitch = math.degrees(pitch_rad)
            return (yaw, pitch), rvec, tvec, None
        except Exception as e:
            return None, None, None, str(e)

    # --------------------------- Eye features ----------------------------
    def _calculate_ear(self, face_landmarks, h, w):
        left_eye_coords, err_l = self._get_valid_landmarks(face_landmarks, self.LEFT_EYE_LMS, h, w)
        if err_l:
            return None, f"Left eye landmarks invalid: {err_l}"
            
        right_eye_coords, err_r = self._get_valid_landmarks(face_landmarks, self.RIGHT_EYE_LMS, h, w)
        if err_r:
            return None, f"Right eye landmarks invalid: {err_r}"
        
        try:
            l_p1, l_p2, l_p3, l_p4, l_p5, l_p6 = left_eye_coords
            l_vert1 = self._euclidean_dist(l_p2, l_p6)
            l_vert2 = self._euclidean_dist(l_p3, l_p5)
            l_horiz = self._euclidean_dist(l_p1, l_p4)
            ear_left = (l_vert1 + l_vert2) / (2.0 * l_horiz)

            r_p1, r_p2, r_p3, r_p4, r_p5, r_p6 = right_eye_coords
            r_vert1 = self._euclidean_dist(r_p2, r_p6)
            r_vert2 = self._euclidean_dist(r_p3, r_p5)
            r_horiz = self._euclidean_dist(r_p1, r_p4)
            ear_right = (r_vert1 + r_vert2) / (2.0 * r_horiz)

            avg_ear = (ear_left + ear_right) / 2.0
            return avg_ear, None
        except Exception as e:
            return None, f"EAR calculation error: {e}"

    def _calculate_mar(self, face_landmarks, h, w):
        mouth_vert_coords, err_v = self._get_valid_landmarks(face_landmarks, self.MOUTH_MAR_LMS_VERT, h, w)
        if err_v:
            return None, f"Vertical mouth landmarks invalid: {err_v}"
            
        mouth_horiz_coords, err_h = self._get_valid_landmarks(face_landmarks, self.MOUTH_MAR_LMS_HORIZ, h, w)
        if err_h:
            return None, f"Horizontal mouth landmarks invalid: {err_h}"
        
        try:
            m_p2, m_p4 = mouth_vert_coords
            vert_dist = self._euclidean_dist(m_p2, m_p4)
            
            m_p1, m_p3 = mouth_horiz_coords
            horiz_dist = self._euclidean_dist(m_p1, m_p3)

            if horiz_dist == 0:
                return None, "MAR horizontal distance is zero."

            mar = vert_dist / horiz_dist
            return mar, None
        except Exception as e:
            return None, f"MAR calculation error: {e}"

    # ----------------------- Simple categorical gaze ---------------------
    def _calculate_gaze(self, face_landmarks, h, w):
        """Calculates the SIMPLE gaze (Left/Right/Center) using nose vs. eye center."""
        nose_tip_coords, err_n = self._get_valid_landmarks(face_landmarks, self.NOSE_TIP_INDEX, h, w)
        if err_n:
            return None, f"Nose tip landmark invalid: {err_n}"

        left_eye_coords, err_l = self._get_valid_landmarks(face_landmarks, self.LEFT_EYE_LMS, h, w)
        if err_l:
            return None, f"Left eye landmarks invalid: {err_l}"
            
        right_eye_coords, err_r = self._get_valid_landmarks(face_landmarks, self.RIGHT_EYE_LMS, h, w)
        if err_r:
            return None, f"Right eye landmarks invalid: {err_r}"

        try:
            nose_p = np.array(nose_tip_coords[0])
            left_c = np.mean(np.array(left_eye_coords), axis=0)
            right_c = np.mean(np.array(right_eye_coords), axis=0)
            
            center_eye_x = (left_c[0] + right_c[0]) / 2.0
            horiz_diff = center_eye_x - nose_p[0]
            
            gaze = "Center"
            if horiz_diff < -self.GAZE_THRESHOLD: 
                gaze = "Gaze Right"
            elif horiz_diff > self.GAZE_THRESHOLD: 
                gaze = "Gaze Left"
            
            return gaze, None
        except Exception as e:
            return None, f"Gaze calculation error: {e}"

    # ----------------- Vector gaze (existing behaviour kept) --------------
    def _calculate_gaze_vector(self, pose_yaw_pitch, face_landmarks, h, w):
        """
        Calculates the 2D pupil coordinates and a 2D gaze vector based on head pose.
        (Kept for backward compatibility)
        """
        left_pupil_coords, err_l = self._get_valid_landmarks(face_landmarks, self.LEFT_PUPIL_LMS, h, w)
        if err_l:
            return None, None, None, f"Left pupil landmark invalid: {err_l}"
            
        right_pupil_coords, err_r = self._get_valid_landmarks(face_landmarks, self.RIGHT_PUPIL_LMS, h, w)
        if err_r:
            return None, None, None, f"Right pupil landmark invalid: {err_r}"
        
        left_pupil_2d = left_pupil_coords[0]
        right_pupil_2d = right_pupil_coords[0]
        
        gaze_vector_2d = None
        if pose_yaw_pitch:
            try:
                yaw_deg, pitch_deg = pose_yaw_pitch
                yaw_rad = math.radians(yaw_deg)
                pitch_rad = math.radians(pitch_deg)
                gaze_x = -math.sin(yaw_rad)
                gaze_y = math.sin(pitch_rad)
                norm = math.hypot(gaze_x, gaze_y)
                gaze_vector_2d = (gaze_x / norm, gaze_y / norm) if norm > 1e-6 else (0.0, 0.0)
            except Exception as e:
                return left_pupil_2d, right_pupil_2d, None, f"Gaze vector math error: {e}"
        else:
             return left_pupil_2d, right_pupil_2d, None, "Head pose is uncertain"

        return left_pupil_2d, right_pupil_2d, gaze_vector_2d, None

    # -------------- NEW: 3D->2D projected gaze hit (non-breaking) --------
    def _calculate_gaze_projection(self, rvec, tvec, face_landmarks, h, w):
        """
        Uses head pose (rvec/tvec) + intrinsics to project a forward gaze ray.
        Returns (left_pupil, right_pupil, hit_point_2d, vector_from_left, err).
        """
        if rvec is None or tvec is None:
            return None, None, None, None, "No rvec/tvec"

        left_pupil_coords, err_l = self._get_valid_landmarks(face_landmarks, self.LEFT_PUPIL_LMS, h, w)
        if err_l:
            return None, None, None, None, f"Left pupil landmark invalid: {err_l}"
        right_pupil_coords, err_r = self._get_valid_landmarks(face_landmarks, self.RIGHT_PUPIL_LMS, h, w)
        if err_r:
            return None, None, None, None, f"Right pupil landmark invalid: {err_r}"

        lp = left_pupil_coords[0]
        rp = right_pupil_coords[0]

        cam_mtx, dist = self._ensure_camera_matrix(h, w)

        try:
            R, _ = cv2.Rodrigues(rvec)
            forward_head = np.array([0.0, 0.0, 1.0], dtype=np.float64).reshape((3,1))
            gaze_cam = (R @ forward_head).reshape(3)

            # --- MODIFIED ---
            distance = float(max(w, h)) * 3.0 # Was 1.8
            gaze_point_cam = gaze_cam * distance + tvec.reshape(3)

            proj, _ = cv2.projectPoints(gaze_point_cam.reshape(1,3), np.zeros((3,1)), np.zeros((3,1)), cam_mtx, dist)
            xh, yh = float(proj[0,0,0]), float(proj[0,0,1])
            hit = (int(round(xh)), int(round(yh)))

            gvx, gvy = (xh - lp[0]), (yh - lp[1])
            n = math.hypot(gvx, gvy)
            gvec = (gvx / n, gvy / n) if n > 1e-6 else (0.0, 0.0)

            return lp, rp, hit, gvec, None
        except Exception as e:
            return lp, rp, None, None, f"Projection error: {e}"

    # ----------------------- Confidence & smoothing ----------------------
    def _estimate_confidence(self, yaw_pitch, lp, rp, h, w):
        if lp is None or rp is None:
            return 0.0
        lp_in = 0 <= lp[0] < w and 0 <= lp[1] < h
        rp_in = 0 <= rp[0] < w and 0 <= rp[1] < h
        if not (lp_in and rp_in):
            return 0.0
        eye_dist = math.hypot(lp[0] - rp[0], lp[1] - rp[1])
        eye_ok = 18.0 <= eye_dist <= (0.6 * w)
        pose_ok = 1.0
        if yaw_pitch is not None:
            yaw, pitch = abs(yaw_pitch[0]), abs(yaw_pitch[1])
            if yaw > 45 or pitch > 35:
                pose_ok = 0.4
        return float(0.6 * (1.0 if eye_ok else 0.3) + 0.4 * pose_ok)

    def _ema_point(self, key, pt):
        if not self.ENABLE_SMOOTHING or pt is None:
            return pt
        a = self.SMOOTHING_ALPHA
        prev = self._ema.get(key)
        pt_arr = np.array(pt, dtype=np.float64)
        if prev is None:
            self._ema[key] = pt_arr
        else:
            self._ema[key] = a * pt_arr + (1 - a) * prev
        sm = self._ema[key]
        return (float(sm[0]), float(sm[1]))

    # ------------------------------- Main --------------------------------
    def process(self, frame_rgb):
        self._mp_stats["frames"] += 1
        h, w, _ = frame_rgb.shape

        # cache/update intrinsics
        self._ensure_camera_matrix(h, w)

        frame_pre = self.preprocess_frame(frame_rgb)

        try:
            results = self.face_mesh.process(frame_pre)
        except Exception as e:
            return [] 

        if not results.multi_face_landmarks:
            return [] 

        all_face_results = []
        
        for face_landmarks in results.multi_face_landmarks:
            
            results_dict = {
                "face_detected": True,
                "yaw": 0.0,
                "pitch": 0.0,
                "pose_uncertain": True,
                "gaze": "Uncertain",      
                "gaze_uncertain": True,   
                "ear": 0.0,
                "eyes_closed": False,
                "ear_uncertain": True,
                "mar": 0.0,
                "mar_uncertain": True,
                "debug_message": "",
                "landmarks_object": face_landmarks,
                
                # existing vector outputs
                "gaze_vector_2d": None,
                "left_pupil_2d": None,
                "right_pupil_2d": None,

                # NEW (non-breaking additions):
                "gaze_proj_2d": None,  # projected hit
                "rvec": None,          # rotation vector
                "tvec": None,          # translation vector
                "confidence": 0.0,     # heuristic confidence
            }

            # Head pose now returns rvec & tvec (signature changed, call site updated here only)
            (pose, rvec, tvec, pose_err) = self._calculate_head_pose(face_landmarks, h, w)
            if pose is None:
                results_dict["debug_message"] += f"Pose failed: {pose_err}; "
            else:
                results_dict["yaw"] = pose[0]
                results_dict["pitch"] = pose[1]
                results_dict["pose_uncertain"] = False
            results_dict["rvec"] = (rvec.reshape(-1).tolist() if rvec is not None else None)
            results_dict["tvec"] = (tvec.reshape(-1).tolist() if tvec is not None else None)

            # EAR / MAR / categorical gaze (unchanged)
            (ear, ear_err) = self._calculate_ear(face_landmarks, h, w)
            if ear is None:
                results_dict["debug_message"] += f"EAR failed: {ear_err}; "
            else:
                results_dict["ear"] = ear
                results_dict["ear_uncertain"] = False
                results_dict["eyes_closed"] = ear < self.EAR_THRESHOLD

            (mar, mar_err) = self._calculate_mar(face_landmarks, h, w)
            if mar is None:
                results_dict["debug_message"] += f"MAR failed: {mar_err}; "
            else:
                results_dict["mar"] = mar
                results_dict["mar_uncertain"] = False

            (gaze, gaze_err) = self._calculate_gaze(face_landmarks, h, w)
            if gaze is None:
                results_dict["debug_message"] += f"Gaze failed: {gaze_err}; "
            else:
                results_dict["gaze"] = gaze
                results_dict["gaze_uncertain"] = False

            # Existing vector gaze
            (l_pupil, r_pupil, gaze_vec, gaze_vec_err) = self._calculate_gaze_vector(
                pose, face_landmarks, h, w
            )
            if gaze_vec_err:
                results_dict["debug_message"] += f"GazeVec failed: {gaze_vec_err}; "
            else:
                # Optional smoothing of pupils
                l_pupil = self._ema_point("lp", l_pupil)
                r_pupil = self._ema_point("rp", r_pupil)

                results_dict["gaze_vector_2d"] = gaze_vec
                results_dict["left_pupil_2d"] = (int(round(l_pupil[0])), int(round(l_pupil[1]))) if l_pupil else None
                results_dict["right_pupil_2d"] = (int(round(r_pupil[0])), int(round(r_pupil[1]))) if r_pupil else None

            # NEW: Projected hit using rvec/tvec (preferred for drawing & testing)
            if rvec is not None and tvec is not None:
                (lp2, rp2, hit, gvec_proj, proj_err) = self._calculate_gaze_projection(rvec, tvec, face_landmarks, h, w)
                if proj_err:
                    results_dict["debug_message"] += f"GazeProj failed: {proj_err}; "
                else:
                    # optional smoothing of hit
                    hit_sm = self._ema_point("hit", hit)
                    results_dict["gaze_proj_2d"] = (int(round(hit_sm[0])), int(round(hit_sm[1]))) if hit_sm else None
                    # if vector not already set, set from projection
                    if results_dict["gaze_vector_2d"] is None and gvec_proj is not None:
                        results_dict["gaze_vector_2d"] = gvec_proj
                    # ensure pupils present
                    if results_dict["left_pupil_2d"] is None and lp2 is not None:
                        results_dict["left_pupil_2d"] = lp2
                    if results_dict["right_pupil_2d"] is None and rp2 is not None:
                        results_dict["right_pupil_2d"] = rp2

            # Confidence last
            results_dict["confidence"] = self._estimate_confidence(
                (results_dict["yaw"], results_dict["pitch"]) if not results_dict["pose_uncertain"] else None,
                results_dict["left_pupil_2d"],
                results_dict["right_pupil_2d"],
                h, w
            )

            all_face_results.append(results_dict)

        return all_face_results 

    def close(self):
        """Cleans up the MediaPipe FaceMesh resource."""
        if self.logger: self.logger.log_mediapipe_debug("CLOSE", "Cleaning up FaceMesh instance...")
        try:
            if hasattr(self, "face_mesh") and hasattr(self.face_mesh, "close"):
                self.face_mesh.close()
                if self.logger: self.logger.log_mediapipe_debug("CLOSE", "FaceMesh closed successfully.")
        except Exception as e:
            if self.logger: self.logger.log_mediapipe_debug("CLOSE_FAIL", str(e))
