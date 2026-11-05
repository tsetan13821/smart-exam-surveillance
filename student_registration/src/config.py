# core/config.py
import os

# --- File Paths & Config ---
ENCODING_DB_PATH = ".\\..\\student_encodings.json"
REGISTRATION_EVIDENCE_DIR = ".\\..\\registrations_for_review"
os.makedirs(REGISTRATION_EVIDENCE_DIR, exist_ok=True)

# Define the yaw angle requirements for each pose
# NOTE: Positive Yaw = User's head turns to THEIR LEFT (camera's right)
#       Negative Yaw = User's head turns to THEIR RIGHT (camera's left)
POSE_REQUIREMENTS = {
    "front": {
        "range": (-12, 12),
        "message": "Please Face Forward" # Base message
    },
    "left": {
        "range": (30, 50),
        "message": "Please Turn to Your LEFT" # Base message
    },
    "right": {
        "range": (-50, -30),
        "message": "Please Turn to Your RIGHT" # Base message
    }
}
# Number of consecutive frames the pose must be held
DEBOUNCE_FRAMES = 15