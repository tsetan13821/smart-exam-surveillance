# app/session_setup.py
"""Turns the students chosen for an exam into what the engine needs.

Shared by the command-line entry point (main.py) and the web API's SessionManager.
"""
import json
from collections import Counter
from dataclasses import dataclass, field

DESK_ROI_FILE = "desk_rois.json"


class SessionSetupError(ValueError):
    """The requested session can't run (no valid students, bad input)."""


@dataclass
class SessionPlan:
    student_ids: list                      # approved students that will be proctored
    student_names: list
    encodings: dict                        # {name: [embedding, ...]}
    rois: dict                             # {name: [[x, y], ...]} desk polygons
    skipped: list = field(default_factory=list)   # [{"id": ..., "reason": ...}]


def load_desk_rois(path=DESK_ROI_FILE):
    """
    Loads per-student desk polygons saved by calibrate_rois.py
    ({student_name: [[x, y], ...]}). Returns {} if the file is missing,
    empty or invalid, in which case devices go to the nearest student.
    """
    try:
        with open(path, "r") as f:
            rois = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if not isinstance(rois, dict):
        return {}
    return {name: pts for name, pts in rois.items()
            if isinstance(pts, list) and len(pts) >= 3}


def plan_session(student_ids, registry, roi_path=DESK_ROI_FILE):
    """Validates the requested students and loads their embeddings and desk regions.

    Students that are unknown, not approved, or have no embeddings are skipped and
    reported in plan.skipped. Raises SessionSetupError if nobody is left.
    """
    if not isinstance(student_ids, (list, tuple)) or not student_ids:
        raise SessionSetupError("Select at least one student.")

    accepted, skipped = [], []
    for student_id in dict.fromkeys(str(s) for s in student_ids):     # de-duplicate, keep order
        data = registry.get_student(student_id)
        if data is None:
            skipped.append({"id": student_id, "reason": "not registered or has no face data"})
            continue
        if data.get("status") != "approved":
            skipped.append({"id": student_id, "reason": f"registration is {data.get('status')}, not approved"})
            continue
        accepted.append((student_id, data))

    # The engine tracks students by display name, so two students who share a name
    # are told apart by their ID: "Siddhant (1)" and "Siddhant (r)".
    name_counts = Counter(data["name"] for _, data in accepted)
    encodings, names, ids = {}, [], []
    for student_id, data in accepted:
        name = data["name"] if name_counts[data["name"]] == 1 else f"{data['name']} ({student_id})"
        names.append(name)
        ids.append(student_id)
        encodings[name] = list(data["encodings"])

    if not encodings:
        reasons = "; ".join(f"{s['id']}: {s['reason']}" for s in skipped)
        raise SessionSetupError(f"No approved students with face data in this session ({reasons}).")

    rois = {name: pts for name, pts in load_desk_rois(roi_path).items() if name in names}
    return SessionPlan(student_ids=ids, student_names=names, encodings=encodings, rois=rois, skipped=skipped)
