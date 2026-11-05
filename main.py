# main.py
import uuid
import sys
import sqlite3
import json
import argparse # <-- To parse arguments from launcher
from app.core import ProctoringApp
from app.utils.logger import ActivityLogger
from app.processors.face_recognizer import FaceRecognizer
from app.utils.student_registry import StudentRegistry
import app.utils.database as db

# Define the main log directory
LOG_DIRECTORY = "proctoring_logs"
    
def main():
    """
    Initializes and runs the proctoring application.
    Receives student IDs and exam context from the launcher.
    """
    # --- MODIFIED: Argument Parsing (Phase 4) ---
    parser = argparse.ArgumentParser(description="Proctoring Application Engine")
    parser.add_argument("--student_ids", required=True,
                        help="A JSON string list of student IDs for this session.")
    # --- NEW ---
    parser.add_argument("--exam_context", required=False, default="{}",
                        help="A JSON string of exam rules (e.g., allowed devices).")
    # --- END NEW ---
    args = parser.parse_args()
    
    logger = None
    session_id = str(uuid.uuid4())
    
    try:
        student_ids = json.loads(args.student_ids)
        # --- NEW (Phase 4) ---
        exam_context = json.loads(args.exam_context)
        print(f"Applying Exam Context: {exam_context}")
        # --- END NEW ---
        
        if not student_ids:
            print("FATAL: No student IDs provided from launcher.")
            sys.exit("Exiting: No student IDs.")
            
        print("Session starting with students passed from launcher...")
        
        db.init_database()
        registry = StudentRegistry()
        
        # --- NEW: Load multiple students ---
        authorized_encodings = {}
        authorized_student_names = []
        student_ids_for_db = []
        
        print(f"Loading data for {len(student_ids)} student(s)...")
        
        for student_id in student_ids:
            student_data = registry.get_student(student_id)
            if student_data and student_data.get('encodings'):
                student_name = student_data['name']
                
                # Add student's name to the authorized list
                if student_name not in authorized_student_names:
                    authorized_student_names.append(student_name)
                    
                student_ids_for_db.append(student_id)
                
                # Add this student's encodings to the main dictionary
                # This correctly groups encodings for students with the same name
                if student_name in authorized_encodings:
                    authorized_encodings[student_name].extend(student_data['encodings'])
                else:
                    authorized_encodings[student_name] = student_data['encodings']
                
                print(f"  > Loaded: {student_name} ({student_id})")
            else:
                print(f"  > WARNING: Could not load data or encodings for student ID {student_id}. Skipping.")
        
        if not authorized_encodings:
            print("FATAL: No valid student encodings could be loaded for the selected IDs.")
            sys.exit("Exiting: No valid students.")
            
        # Convert lists to JSON strings for database logging
        student_ids_json = json.dumps(student_ids_for_db)
        student_names_json = json.dumps(authorized_student_names)
        
        # --- NEW: Call updated create_session and Logger ---
        db.create_session(
            session_id=session_id,
            student_ids_json=student_ids_json,
            student_names_json=student_names_json
        )
        
        logger = ActivityLogger(
            session_id=session_id,
            student_ids_json=student_ids_json,
            student_names_json=student_names_json,
            log_dir=LOG_DIRECTORY
        )
        logger.log_activity(f"Application starting for session {session_id}...")
        
        # --- NEW: Instantiate FaceRecognizer with all encodings ---
        face_recognizer = FaceRecognizer(
            logger=logger,
            authorized_encodings=authorized_encodings
        )
        
        # --- MODIFIED: Pass exam_context to ProctoringApp (Phase 4) ---
        app = ProctoringApp(
            logger=logger, 
            face_recognizer=face_recognizer,
            authorized_student_names=authorized_student_names,
            exam_context=exam_context # <-- PASS THE CONTEXT
        )
        app.run()
        
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Missing required file: {e}")
        if logger:
            logger.log_alert("FATAL_ERROR", f"Missing required file: {e}")
    except (sqlite3.Error, Exception) as e:
        print(f"An unexpected error occurred: {e}")
        if logger:
            logger.log_alert("FATAL_ERROR", str(e))
        print("Application will now exit.")
    finally:
        if logger:
            logger.log_activity("Application shutting down.")
            logger.close()

if __name__ == '__main__':
    main()