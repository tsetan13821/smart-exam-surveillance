# app/utils/student_registry.py
import json
import os
import config as config
import numpy as np

class StudentRegistry:
    """
    Manages the student encoding database (student_encodings.json).
    Handles loading multiple encodings per student and their status.
    """
    def __init__(self):
        self.db_file = config.STUDENT_DATABASE_FILE

        if not os.path.exists(self.db_file):
            print(f"FATAL: Student database not found at {self.db_file}")
            print("Please run 'student_portal.py' to add at least one student.")
            raise FileNotFoundError(f"Student database not found at {self.db_file}")

        try:
            with open(self.db_file, 'r') as f:
                self.students = json.load(f)
            if not self.students:
                raise Exception("Student database is empty.")
            print(f"Loaded {len(self.students)} student records from {self.db_file}.")
        except Exception as e:
            print(f"FATAL: Could not parse {self.db_file}: {e}")
            raise

    def list_students(self):
        """
        Returns a simple list of student names, IDs, and their status.
        """
        # --- MODIFICATION: Now returns 'status' as well ---
        return [{"id": student_id, "name": data["name"], "status": data.get("status", "pending")}
                for student_id, data in self.students.items()]

    def get_student(self, student_id):
        """
        Gets a single student's data, including all pre-computed encodings.
        """
        student_data = self.students.get(student_id)

        if student_data:
            # --- Load Multiple Encodings ---
            encodings_list = student_data.get("encodings", []) 

            # Convert each encoding from a plain list (JSON)
            # back to a NumPy array for 'face_recognition'
            encodings_np = [np.array(enc) for enc in encodings_list]

            if not encodings_np:
                print(f"Warning: Student {student_id} has no encodings in database.")
                return None

            return {
                "student_id": student_id,
                "name": student_data["name"],
                "encodings": encodings_np,
                "status": student_data.get("status", "pending") # Also return status here
            }

        return None