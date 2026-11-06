# app/utils/students_db.py
"""Student registrations stored in SQLite.

Replaces student_encodings.json, which every app rewrote in full on each save
(so two apps saving at once could lose a registration). Face embeddings are
stored as float32 blobs, one row per student.
"""
import json
import os
import re
import sqlite3
from datetime import datetime

import numpy as np

import config

STATUSES = ("pending", "approved", "rejected")

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    student_id         TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    status             TEXT NOT NULL DEFAULT 'pending'
                       CHECK (status IN ('pending', 'approved', 'rejected')),
    encodings          BLOB NOT NULL,          -- float32, encoding_count x dim
    encoding_count     INTEGER NOT NULL,
    review_evidence_id TEXT,                   -- registration photos: <id>_front/left/right.jpg
    extracted_id_text  TEXT,
    created_at         TEXT NOT NULL,
    reviewed_at        TEXT,
    reviewed_by        TEXT
);
CREATE INDEX IF NOT EXISTS idx_students_status ON students (status);
"""

_PUBLIC_COLUMNS = ("student_id, name, status, encoding_count, review_evidence_id, extracted_id_text, "
                   "created_at, reviewed_at, reviewed_by")


def connect():
    conn = sqlite3.connect(config.DATABASE_FILE, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _pack(encodings):
    arr = np.asarray([np.asarray(e, dtype=np.float32).ravel() for e in encodings], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError("At least one face embedding is required.")
    return arr.tobytes(), arr.shape[0]


def _unpack(blob, count):
    arr = np.frombuffer(blob, dtype=np.float32)
    return [row.copy() for row in arr.reshape(count, -1)]


def create_schema(conn):
    conn.executescript(SCHEMA)


def import_json_registry(conn, json_path=None):
    """One-time import of the legacy JSON registry into an empty students table.

    Returns the number of students imported. The JSON file is left untouched.
    """
    json_path = json_path or config.STUDENT_DATABASE_FILE
    if conn.execute("SELECT COUNT(*) FROM students").fetchone()[0] > 0 or not os.path.exists(json_path):
        return 0
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return 0
    now = datetime.now().isoformat()
    imported = 0
    for student_id, rec in (data or {}).items():
        encodings = rec.get("encodings") or []
        status = rec.get("status", "pending")
        if not encodings or status not in STATUSES:
            continue
        blob, count = _pack(encodings)
        conn.execute(
            "INSERT OR IGNORE INTO students (student_id, name, status, encodings, encoding_count, "
            "review_evidence_id, extracted_id_text, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (str(student_id), rec.get("name", ""), status, blob, count, rec.get("review_evidence_id"),
             rec.get("extracted_id_text"), now))
        imported += 1
    return imported


def save_registration(student_id, name, encodings, review_evidence_id=None,
                      extracted_id_text="N/A (Live Capture)", status="pending"):
    """Creates or replaces a registration.

    Self-registrations go (back) to 'pending' for a proctor to review; the admin
    command-line tool registers students as 'approved' directly.
    """
    student_id, name = str(student_id).strip(), str(name).strip()
    if not student_id or not name:
        raise ValueError("Student ID and name are required.")
    if status not in STATUSES:
        raise ValueError(f"Unknown status: {status}")
    blob, count = _pack(encodings)
    now = datetime.now().isoformat()
    with connect() as conn:
        conn.execute(
            "INSERT INTO students (student_id, name, status, encodings, encoding_count, review_evidence_id, "
            "extracted_id_text, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(student_id) DO UPDATE SET name = excluded.name, status = excluded.status, "
            "encodings = excluded.encodings, encoding_count = excluded.encoding_count, "
            "review_evidence_id = excluded.review_evidence_id, extracted_id_text = excluded.extracted_id_text, "
            "created_at = excluded.created_at, reviewed_at = NULL, reviewed_by = NULL",
            (student_id, name, status, blob, count, review_evidence_id, extracted_id_text, now))


def add_encodings(student_id, encodings):
    """Appends face embeddings to an existing student. Returns the new count, or None if unknown."""
    rec = get_student(student_id, with_encodings=True)
    if rec is None:
        return None
    blob, count = _pack(list(rec["encodings"]) + list(encodings))
    with connect() as conn:
        conn.execute("UPDATE students SET encodings = ?, encoding_count = ? WHERE student_id = ?",
                     (blob, count, str(student_id)))
    return count


def list_students(status=None):
    """Students without their embeddings, newest first; optionally filtered by status."""
    query = f"SELECT {_PUBLIC_COLUMNS} FROM students"
    params = ()
    if status is not None:
        if status not in STATUSES:
            raise ValueError(f"Unknown status: {status}")
        query += " WHERE status = ?"
        params = (status,)
    with connect() as conn:
        return [dict(r) for r in conn.execute(query + " ORDER BY created_at DESC, student_id", params)]


def get_student(student_id, with_encodings=False):
    columns = _PUBLIC_COLUMNS + (", encodings" if with_encodings else "")
    with connect() as conn:
        row = conn.execute(f"SELECT {columns} FROM students WHERE student_id = ?", (str(student_id),)).fetchone()
    if row is None:
        return None
    rec = dict(row)
    if with_encodings:
        rec["encodings"] = _unpack(rec["encodings"], rec["encoding_count"])
    return rec


def set_status(student_id, status, reviewed_by=None):
    """Approves or rejects a registration. Returns False if the student doesn't exist."""
    if status not in STATUSES:
        raise ValueError(f"Unknown status: {status}")
    with connect() as conn:
        cur = conn.execute(
            "UPDATE students SET status = ?, reviewed_at = ?, reviewed_by = ? WHERE student_id = ?",
            (status, datetime.now().isoformat(), reviewed_by, str(student_id)))
        return cur.rowcount > 0


def registration_photo_path(student, pose):
    """Path of a registration photo (front/left/right), or None if it doesn't exist."""
    evidence_id = student.get("review_evidence_id") or ""
    if pose not in ("front", "left", "right") or not re.fullmatch(r"[A-Za-z0-9-]{1,64}", evidence_id):
        return None
    path = os.path.join(config.REGISTRATION_EVIDENCE_DIR, f"{evidence_id}_{pose}.jpg")
    return path if os.path.isfile(path) else None
