# api/reports.py
"""Builds a session report (attendance, breaks, incidents per student) from its alert log."""
import re
from collections import Counter
from datetime import datetime

from api.session_store import INFORMATIONAL_TYPES

# Older sessions were logged before alerts had a 'subject' column; recover it from the message.
_STUDENT_MSG = re.compile(r"^Student (?P<name>.+?) (?:has LEFT|has RETURNED|marked PRESENT|is missing|has not appeared)")
_PERSON_MSG = re.compile(r"^\((?P<name>[^)]+)\) ")


def subject_of(alert):
    if alert.get("subject"):
        return alert["subject"]
    message = alert.get("message") or ""
    m = _STUDENT_MSG.match(message) or _PERSON_MSG.match(message)
    return m.group("name") if m else None


def _parse(ts):
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def _seconds(a, b):
    a, b = _parse(a), _parse(b)
    return max(0.0, (b - a).total_seconds()) if a and b else None


def build_report(session, alerts):
    """session: a session_store summary; alerts: all of its alerts in time order."""
    end = session.get("end_time") or (alerts[-1]["timestamp"] if alerts else session.get("start_time"))
    students = {}
    for sid, name in zip(session["student_ids"], session["student_names"]):
        students[name] = {"student_id": sid, "name": name, "first_seen": None, "breaks": [],
                          "away_seconds": 0.0, "incidents": Counter(), "flagged_missing": False}

    incidents_by_type = Counter()
    unassigned = Counter()
    system_errors = 0
    for alert in alerts:
        kind = alert.get("alert_type") or ""
        level = alert.get("level")
        if level == "ALERT":
            system_errors += 1
            continue
        if level != "EVIDENCE":
            continue
        who = students.get(subject_of(alert))
        when = alert["timestamp"]
        if kind == "ATTENDANCE_MARKED":
            if who and who["first_seen"] is None:
                who["first_seen"] = when
        elif kind == "PRESENCE_LEFT":
            if who:
                who["breaks"].append({"left": when, "returned": None})
        elif kind == "PRESENCE_RETURNED":
            if who and who["breaks"] and who["breaks"][-1]["returned"] is None:
                who["breaks"][-1]["returned"] = when
        else:
            incidents_by_type[kind] += 1
            if who:
                who["incidents"][kind] += 1
                if kind == "STUDENT_MISSING":
                    who["flagged_missing"] = True
            else:
                unassigned[kind] += 1

    out_students = []
    for s in students.values():
        for b in s["breaks"]:
            secs = _seconds(b["left"], b["returned"] or end)
            b["seconds"] = None if secs is None else round(secs, 1)
            s["away_seconds"] += secs or 0.0
        if s["first_seen"] is None:
            attendance = "absent"
        elif s["breaks"] and s["breaks"][-1]["returned"] is None:
            attendance = "left"
        else:
            attendance = "present"
        out_students.append({
            "student_id": s["student_id"],
            "name": s["name"],
            "attendance": attendance,
            "first_seen": s["first_seen"],
            "breaks": s["breaks"],
            "away_seconds": round(s["away_seconds"], 1),
            "incident_count": sum(s["incidents"].values()),
            "incidents": dict(s["incidents"]),
        })

    duration = _seconds(session.get("start_time"), end)
    return {
        "duration_seconds": None if duration is None else round(duration, 1),
        "students": out_students,
        "incidents_by_type": dict(incidents_by_type.most_common()),
        "unassigned_incidents": dict(unassigned),     # e.g. unknown people, global alerts
        "system_errors": system_errors,
        "informational_types": list(INFORMATIONAL_TYPES),
    }
