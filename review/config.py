# config.py

import os # <-- NEW IMPORT

# Central configuration file for the proctoring application.
_CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the project's base directory (one level up from this file)

BASE_DIR = os.path.abspath(os.path.join(_CURRENT_FILE_DIR, '..'))

# --- Database ---
# File path for the main student encoding database
STUDENT_DATABASE_FILE = os.path.join(BASE_DIR, "student_encodings.json")

# File path for the SQLite database that stores session and alert logs
DATABASE_FILE = os.path.join(BASE_DIR, "proctoring_session.db")

# --- NEW: Path for registration evidence (from your snippet) ---
REGISTRATION_EVIDENCE_DIR = os.path.join(BASE_DIR, "registrations_for_review")