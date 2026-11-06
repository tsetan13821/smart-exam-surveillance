# app/utils/student_registry.py
"""Read access to student registrations for the proctoring engine (backed by SQLite)."""
from app.utils import students_db
import app.utils.database as db


class StudentRegistry:
    """Lists students and loads their face embeddings for a session."""

    def __init__(self):
        db.init_database()     # creates tables and imports the legacy JSON registry on first use

    def list_students(self):
        """Returns [{"id", "name", "status"}] for every registered student."""
        return [{"id": s["student_id"], "name": s["name"], "status": s["status"]}
                for s in students_db.list_students()]

    def get_student(self, student_id):
        """A student's data with face embeddings (numpy arrays), or None."""
        rec = students_db.get_student(student_id, with_encodings=True)
        if rec is None or not rec["encodings"]:
            return None
        return {
            "student_id": rec["student_id"],
            "name": rec["name"],
            "encodings": rec["encodings"],
            "status": rec["status"],
        }
