# src/database.py
"""Registration storage for the Streamlit portal, backed by the project's SQLite database."""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import app.utils.database as _db  # noqa: E402
from app.utils import students_db  # noqa: E402

_db.init_database()


def get_registration(student_id):
    """{"name", "status"} for a registered student, or None."""
    rec = students_db.get_student(student_id)
    return {"name": rec["name"], "status": rec["status"]} if rec else None


def save_registration(student_id, name, encodings, review_evidence_id):
    """Stores a new (or repeated) registration as 'pending' for proctor review."""
    students_db.save_registration(student_id, name, encodings, review_evidence_id=review_evidence_id)
