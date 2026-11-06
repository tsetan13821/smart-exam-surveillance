# student_registration/src/config.py
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config as _project_config  # noqa: E402  (the project-level config.py)

# --- File Paths & Config ---
# Registration photos go to the folder the proctor console shows them from
REGISTRATION_EVIDENCE_DIR = _project_config.REGISTRATION_EVIDENCE_DIR
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