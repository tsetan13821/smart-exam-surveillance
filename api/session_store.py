# api/session_store.py
"""Read-only queries over past sessions and their alerts."""
import json
import sqlite3

import config
from api.events import evidence_url

# Logged as evidence but not misconduct: attendance and breaks.
INFORMATIONAL_TYPES = ("ATTENDANCE_MARKED", "PRESENCE_LEFT", "PRESENCE_RETURNED")
_INFO_SQL = ", ".join(f"'{t}'" for t in INFORMATIONAL_TYPES)
INCIDENT_SQL = f"(a.level = 'EVIDENCE' AND COALESCE(a.alert_type, '') NOT IN ({_INFO_SQL}))"


def connect():
    conn = sqlite3.connect(config.DATABASE_FILE, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _loads(text, default):
    try:
        value = json.loads(text) if text else default
    except (TypeError, json.JSONDecodeError):
        return default
    return value if isinstance(value, type(default)) else default


def _session_row(row):
    return {
        "session_id": row["session_id"],
        "student_ids": [str(s) for s in _loads(row["student_ids"], [])],
        "student_names": [str(s) for s in _loads(row["student_names"], [])],
        "start_time": row["start_time"],
        "end_time": row["end_time"],
        "status": row["status"],
        "exam_context": _loads(row["exam_context"], {}),
        "source": row["source"],
        "incident_count": row["incident_count"] or 0,
        "evidence_count": row["evidence_count"] or 0,
    }


_SESSION_SELECT = f"""
SELECT s.*,
       (SELECT COUNT(*) FROM alerts a WHERE a.session_id = s.session_id AND {INCIDENT_SQL}) AS incident_count,
       (SELECT COUNT(*) FROM alerts a WHERE a.session_id = s.session_id AND a.evidence_file IS NOT NULL)
           AS evidence_count
FROM sessions s
"""


def list_sessions(limit=20, offset=0, status=None, student_id=None):
    where, params = [], []
    if status:
        where.append("s.status = ?")
        params.append(status)
    if student_id:
        # student_ids is a JSON list; match the quoted id exactly
        where.append("EXISTS (SELECT 1 FROM json_each(s.student_ids) j WHERE CAST(j.value AS TEXT) = ?)")
        params.append(str(student_id))
    clause = f" WHERE {' AND '.join(where)}" if where else ""
    with connect() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM sessions s{clause}", params).fetchone()[0]
        rows = conn.execute(f"{_SESSION_SELECT}{clause} ORDER BY s.start_time DESC LIMIT ? OFFSET ?",
                            params + [limit, offset]).fetchall()
    return {"items": [_session_row(r) for r in rows], "total": total, "limit": limit, "offset": offset}


def get_session(session_id):
    with connect() as conn:
        row = conn.execute(f"{_SESSION_SELECT} WHERE s.session_id = ?", (session_id,)).fetchone()
    return _session_row(row) if row else None


def _alert_row(row):
    return {
        "alert_id": row["alert_id"],
        "timestamp": row["timestamp"],
        "level": row["level"],
        "alert_type": row["alert_type"],
        "message": row["message"],
        "subject": row["subject"],
        "evidence_url": evidence_url(row["evidence_file"]),
    }


def list_alerts(session_id, limit=200, offset=0, levels=None, incidents_only=False):
    where, params = ["a.session_id = ?"], [session_id]
    if levels:
        where.append(f"a.level IN ({', '.join('?' for _ in levels)})")
        params.extend(levels)
    if incidents_only:
        where.append(INCIDENT_SQL)
    clause = " WHERE " + " AND ".join(where)
    with connect() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM alerts a{clause}", params).fetchone()[0]
        rows = conn.execute(f"SELECT a.* FROM alerts a{clause} ORDER BY a.timestamp, a.alert_id LIMIT ? OFFSET ?",
                            params + [limit, offset]).fetchall()
    return {"items": [_alert_row(r) for r in rows], "total": total, "limit": limit, "offset": offset}


def all_alerts(session_id):
    with connect() as conn:
        rows = conn.execute("SELECT a.* FROM alerts a WHERE a.session_id = ? ORDER BY a.timestamp, a.alert_id",
                            (session_id,)).fetchall()
    return [_alert_row(r) for r in rows]
