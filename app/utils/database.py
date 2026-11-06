# app/utils/database.py
import sqlite3
import config as config
from datetime import datetime

from app.utils import students_db

# Columns added after the first release; existing databases get them on startup.
SESSION_EXTRA_COLUMNS = {
    "end_time": "TEXT",
    "status": "TEXT NOT NULL DEFAULT 'ended'",   # running | ended | failed
    "exam_context": "TEXT NOT NULL DEFAULT '{}'",
    "source": "TEXT",
}


def _add_missing_columns(cursor, table, columns):
    existing = {row[1] for row in cursor.execute(f"PRAGMA table_info({table})")}
    for name, ddl in columns.items():
        if name not in existing:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def init_database():
    """
    Initializes the SQLite database and creates tables if they don't exist.
    """
    try:
        with sqlite3.connect(config.DATABASE_FILE) as conn:
            # WAL lets the API read while the logger thread writes
            conn.execute("PRAGMA journal_mode=WAL")
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

            _add_missing_columns(cursor, "sessions", SESSION_EXTRA_COLUMNS)
            _add_missing_columns(cursor, "alerts", {"subject": "TEXT"})   # student an event is about
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_session ON alerts (session_id, timestamp)")

            students_db.create_schema(conn)
            imported = students_db.import_json_registry(conn)
            if imported:
                print(f"Imported {imported} student registration(s) from {config.STUDENT_DATABASE_FILE}")
            conn.commit()
        print(f"Database initialized and tables ensured at {config.DATABASE_FILE}")
    except sqlite3.Error as e:
        print(f"FATAL: Error initializing database: {e}")
        raise

def create_session(session_id, student_ids_json, student_names_json, exam_context_json="{}", source=None):
    """
    Logs the start of a new proctoring session in the database (status 'running').
    """
    try:
        with sqlite3.connect(config.DATABASE_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (session_id, student_ids, student_names, start_time, status, exam_context, source)"
                " VALUES (?, ?, ?, ?, 'running', ?, ?)",
                (session_id, student_ids_json, student_names_json, datetime.now().isoformat(),
                 exam_context_json, source)
            )
            conn.commit()
        print(f"Database session created with ID: {session_id}")
    except sqlite3.Error as e:
        print(f"FATAL: Error creating database session: {e}")
        raise


def end_session(session_id, status="ended"):
    """Marks a session as finished ('ended') or as having failed to run ('failed')."""
    with sqlite3.connect(config.DATABASE_FILE, timeout=10) as conn:
        conn.execute("UPDATE sessions SET end_time = ?, status = ? WHERE session_id = ?",
                     (datetime.now().isoformat(), status, session_id))
        conn.commit()


def mark_interrupted_sessions():
    """Sessions still 'running' when the server starts were cut off by a crash or restart."""
    with sqlite3.connect(config.DATABASE_FILE, timeout=10) as conn:
        cur = conn.execute("UPDATE sessions SET status = 'failed', end_time = COALESCE(end_time, start_time)"
                           " WHERE status = 'running'")
        conn.commit()
        return cur.rowcount
