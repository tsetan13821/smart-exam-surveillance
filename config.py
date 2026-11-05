"""
============================================================================
                        MASTER CONFIGURATION FILE
============================================================================
This file centralizes all tunable parameters for the proctoring application.
"""

import numpy as np

# --------------------------------------------------------------------------
# 1. CORE APPLICATION & DATABASE
# --------------------------------------------------------------------------

# The index of the webcam (e.g., 0 for built-in, 1 for external)
CAMERA_INDEX = 0

# File path for the main student encoding database
STUDENT_DATABASE_FILE = "student_encodings.json"

# File path for the SQLite database that stores session and alert logs
DATABASE_FILE = "proctoring_session.db"

# --------------------------------------------------------------------------
# 2. LOGGING DIRECTORIES
# --------------------------------------------------------------------------
# These directories will be created automatically if they don't exist.

# Directory for general debug logs from core.py
CORE_LOGS_DIR = "logs/core"

# Directory for MediaPipe debug logs (landmark failures, etc.)
MEDIAPIPE_LOGS_DIR = "logs/mediapipe"

# Directory for object detection debug logs (raw detections)
OBJECT_DETECTOR_LOGS_DIR = "logs/detection"

# Directory for liveness-specific debug logs (timers, spoof checks)
LIVENESS_LOGS_DIR = "logs/liveness"

# Directory for face recognition similarity scores
SIMILARITY_LOGS_DIR = "logs/similarity"

# Directory for detailed human action logs (pose, gaze, etc.)
HUMAN_ACTION_LOG_DIR = "logs/human_actions"


# --------------------------------------------------------------------------
# 3. OBJECT DETECTOR (RT-DETR)
# ----------------------------------q----------------------------------------

# Run object detection every N frames. (10 = ~3 times/sec @ 30fps)
# This saves significant resources.
OBJECT_DETECTION_INTERVAL = 5

# Hugging Face model name for the object detector
HF_MODEL_NAME = "PekingU/rtdetr_r50vd"

# Confidence threshold for an object to be considered "detected"
DETECTION_CONFIDENCE_THRESHOLD = 0.7

# Object classes to detect. RT-DETR is trained on COCO, so only COCO class
# names work here; earphones, calculators and tablets would need a custom model.
CLASSES_TO_DETECT = [
    'person',
    'cell phone',
    'laptop',
    'book',
    'mouse',
    'keyboard',
]

# Subset of detected classes that are considered "contraband". Any of them can
# be allowed for a particular exam from the launcher (exam context).
CONTRABAND_CLASSES = [
    'cell phone',
    'laptop',
    'book',
    'mouse',
    'keyboard',
]


# --------------------------------------------------------------------------
# 4. FACE RECOGNIZER (InsightFace)
# --------------------------------------------------------------------------

# Run face recognition every N frames.
FACE_DETECTION_INTERVAL = 5

# Hugging Face model name for InsightFace (e.g., 'buffalo_l', 'buffalo_s')
INSIGHTFACE_MODEL_NAME = "buffalo_l"

# Detection size for InsightFace. (640, 640) is standard.
INSIGHTFACE_DET_SIZE = (640, 640)

# Similarity threshold (dot product) for matching a face.
# Higher = stricter. 0.55 is a good starting point.
FACE_SIMILARITY_THRESHOLD = 0.50


# --------------------------------------------------------------------------
# 5. MEDIAPIPE PROCESSOR (Face Mesh)
# --------------------------------------------------------------------------

# Max number of faces FaceMesh analyses per frame (head pose, gaze, blinks).
# Must be at least the number of students in view: a face without landmarks
# gets no pose/gaze/liveness analysis. Each extra face costs a few ms per frame.
MEDIAPIPE_MAX_NUM_FACES = 4

# MediaPipe's internal detection and tracking confidence levels
MEDIAPIPE_MIN_DET_CONF = 0.5
MEDIAPIPE_MIN_TRACK_CONF = 0.5

# Landmark confidence thresholds
MEDIAPIPE_PRESENCE_THRESHOLD = 0.6
MEDIAPIPE_INFERRED_CONFIDENCE = 0.8

# --- Head Pose & Gaze ---

# Eye Aspect Ratio (EAR) threshold for blink detection
EYE_EAR_THRESHOLD = 0.2

# Mouth Aspect Ratio (MAR) threshold for detecting an open mouth
MOUTH_MAR_THRESHOLD = 0.5

# Gaze threshold (pixel difference) to detect non-centered gaze
EYE_GAZE_THRESHOLD = 15.0

# Head pose thresholds (in degrees) for logging actions
HEAD_POSE_PITCH_DOWN_THRESHOLD = 15.0  # Looking down
HEAD_POSE_PITCH_UP_THRESHOLD = -10.0 # Looking up
HEAD_POSE_YAW_THRESHOLD = 18.0       # Looking left/right

# --- Camera Parameters (for solvePnP head pose) ---
# Tweak this if head pose yaw/pitch seems incorrect
CAMERA_FOCAL_SCALE = 1.2
# Use a pre-calibrated matrix if available, otherwise set to None
CAMERA_MATRIX = None 
# Use a pre-calibrated distortion coefficent, otherwise set to None
DIST_COEFFS = None
# Use advanced refinement for solvePnP (slower but more stable)
USE_PNP_REFINE = True

# --- Landmark Validation (Advanced) ---
MEDIAPIPE_EDGE_MARGIN = 0.02
MEDIAPIPE_MAX_ABS_Z = 0.5
MEDIAPIPE_TEMPORAL_WINDOW = 5


# --------------------------------------------------------------------------
# 6. LIVENESS & SPOOF DETECTOR
# --------------------------------------------------------------------------

# Proof of life is an adaptive blink (see app/processors/liveness_detector.py).
# Head pose, optical flow and gaze changes are not used: a tilted or shaking
# phone showing a photo imitates them (measured with benchmarks/liveness_benchmark.py).

# Seconds a newly seen face has to blink before it is treated as a photo
LIVENESS_CONFIRM_SECONDS = 8.0

# Seconds a confirmed-live student can go without blinking before being flagged.
# People concentrating on a screen can go 10-20 s between blinks.
LIVENESS_SPOOF_THRESHOLD_SEC = 30.0

# Rolling window (seconds) for the person's own open-eye EAR baseline
LIVENESS_BLINK_BASELINE_SEC = 3.0
# A blink starts when EAR falls below this fraction of the baseline...
LIVENESS_BLINK_DROP_RATIO = 0.70
# ...and completes when it recovers above this fraction
LIVENESS_BLINK_RECOVER_RATIO = 0.85
# Eyes lower than this for longer (e.g. looking down) re-baseline instead of counting
LIVENESS_BLINK_MAX_SEC = 2.0
# Minimum open-eye samples needed before blinks can be detected
LIVENESS_BLINK_MIN_BASELINE_SAMPLES = 5

# Head pose is median-smoothed over this many frames before movement checks
POSE_SMOOTHING_FRAMES = 5


# --------------------------------------------------------------------------
# 7. ALERT THRESHOLDS (in Seconds)
# --------------------------------------------------------------------------
# Duration an alert condition must be met before it is logged as evidence.

ALERT_THRESHOLDS_SEC = {
    # Time an unauthorized person must be present
    "UNAUTHORIZED_FACE_DETECTED": 3.0,
    
    # Time contraband must be present
    "CONTRABAND_DETECTED": 3.0,
    
    # Time a student must be missing before alert
    "STUDENT_MISSING": 10.0,
    
    # Seconds a face must stay judged as a photo before evidence is saved
    # (the liveness verdict itself already needs LIVENESS_CONFIRM_SECONDS without a blink)
    "IMAGE_DETECTED": 2.0,
    
    # (Example of other alerts you could add)
    # "HEAD_DOWN_ALERT": 5.0, # <-- REMOVED
    # "GAZE_AWAY_ALERT": 5.0 # <-- REMOVED
    
    # --- NEW ---
    "HEAD_TURN_ALERT": 5.0,  # Must be peeking for 5 sec
    "EXCESSIVE_MOVEMENT_ALERT": 10.0 # Movement threshold must be met for 10 sec
}


# --------------------------------------------------------------------------
# 8. CHEATING DETECTOR CONFIG
# --------------------------------------------------------------------------

# Gaze angle (degrees) beyond which a student is "looking away"
CHEATING_GAZE_AWAY_THRESHOLD_YAW = 25.0
CHEATING_GAZE_AWAY_THRESHOLD_PITCH = 20.0

# A device must be seen continuously this long before it counts (filters
# one-off false detections)
CHEATING_DEVICE_PERSISTENCE_SEC = 2.0
# Gaps shorter than this between detections still count as one continuous sighting
CHEATING_DEVICE_GAP_SEC = 3.0

# Without desk ROIs (calibrate_rois.py), a device is assigned to the nearest
# student within this many pixels
CHEATING_DEVICE_MAX_DISTANCE_PX = 250

# --- Rule Engine Thresholds (in seconds) ---
RULE_ABSENCE_DURATION = 10.0      # Time a student must be absent
RULE_GAZE_ONLY_DURATION = 5.0     # Looking away this long raises "Sustained Gaze Away"
RULE_GAZE_WITH_DEVICE_DURATION = 3.0 # Shorter time if device is also present
RULE_DEVICE_DURATION = 5.0        # A device stays "present" this long after it was last seen

# --- Presence ---
# A student counts as present only if their face was matched within this many seconds
PRESENCE_GRACE_SEC = 2.0
# Flag an enrolled student who hasn't appeared this long after the session starts
STUDENT_NO_SHOW_SEC = 60.0

# --- Head Pose Alerting ---
# Yaw (left/right) degrees to trigger a "peeking" alert
HEAD_POSE_YAW_ALERT_THRESHOLD = 25.0 

# --- Excessive Movement Alerting ---
# Time window to count movements in
EXCESSIVE_MOVEMENT_TIME_WINDOW_SEC = 60.0 
# Number of movements in the time window to trigger an alert
EXCESSIVE_MOVEMENT_COUNT_THRESHOLD = 20

EXCESSIVE_MOVEMENT_SWAY_THRESHOLD = 10.0