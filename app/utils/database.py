# app/utils/database.py
import sqlite3
import config as config
from datetime import datetime

def init_database():
    """
    Initializes the SQLite database and creates tables if they don't exist.
    """
    try:
        with sqlite3.connect(config.DATABASE_FILE) as conn:
            cursor = conn.cursor()

            # --- Create 'sessions' Table (MODIFIED) ---
            # Stores student_ids and student_names as JSON text
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                student_ids TEXT NOT NULL,
                student_names TEXT NOT NULL,
                start_time TEXT NOT NULL
            )
            """)

            # --- Create 'alerts' Table (Unchanged) ---
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,      -- e.g., 'INFO', 'ALERT', 'EVIDENCE'
                alert_type TEXT,          -- e.g., 'CONTRABAND_DETECTED', 'ATTENDANCE_LEFT'
                message TEXT NOT NULL,    -- Human-readable description
                evidence_file TEXT,       -- Absolute path to the screenshot, if any
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
            """)

            conn.commit()
        print(f"Database initialized and tables ensured at {config.DATABASE_FILE}")
    except sqlite3.Error as e:
        print(f"FATAL: Error initializing database: {e}")
        raise

def create_session(session_id, student_ids_json, student_names_json):
    """
    Logs the start of a new proctoring session in the database.
    (MODIFIED to accept JSON lists for multi-student sessions)
    """
    try:
        with sqlite3.connect(config.DATABASE_FILE) as conn:
            cursor = conn.cursor()
            
            # --- MODIFIED: Insert new multi-student format ---
            cursor.execute(
                "INSERT INTO sessions (session_id, student_ids, student_names, start_time) VALUES (?, ?, ?, ?)",
                (session_id, student_ids_json, student_names_json, datetime.now().isoformat())
            )
            conn.commit()
        print(f"Database session created with ID: {session_id}")
    except sqlite3.Error as e:
        print(f"FATAL: Error creating database session: {e}")
        raise