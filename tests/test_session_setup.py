"""Choosing the students for a session."""
import numpy as np
import pytest

from app.session_setup import SessionSetupError, plan_session


class Registry:
    def __init__(self, students):
        self.students = students

    def get_student(self, student_id):
        rec = self.students.get(student_id)
        return None if rec is None else dict(rec, student_id=student_id, encodings=[np.full(4, len(student_id))])


def test_students_sharing_a_name_are_told_apart_by_id(tmp_path):
    reg = Registry({"1": {"name": "Siddhant", "status": "approved"},
                    "r": {"name": "Siddhant", "status": "approved"},
                    "7": {"name": "Asha", "status": "approved"}})
    plan = plan_session(["1", "r", "7"], reg, roi_path=str(tmp_path / "none.json"))
    assert plan.student_ids == ["1", "r", "7"]
    assert plan.student_names == ["Siddhant (1)", "Siddhant (r)", "Asha"]
    assert set(plan.encodings) == set(plan.student_names)      # one face set per student, never merged


def test_unknown_unapproved_and_repeated_ids(tmp_path):
    reg = Registry({"1": {"name": "Asha", "status": "approved"}, "2": {"name": "Bilal", "status": "pending"}})
    plan = plan_session(["1", "1", "2", "9"], reg, roi_path=str(tmp_path / "none.json"))
    assert plan.student_ids == ["1"] and plan.student_names == ["Asha"]
    assert [s["id"] for s in plan.skipped] == ["2", "9"]
    with pytest.raises(SessionSetupError):
        plan_session(["2"], reg, roi_path=str(tmp_path / "none.json"))
