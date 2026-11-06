"""Student registrations in SQLite (replacing student_encodings.json)."""
import json
import os

import numpy as np
import pytest

import app.utils.database as db
import config
from app.utils import students_db
from app.utils.student_registry import StudentRegistry


def emb(seed, n=1):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(512).astype(np.float32) for _ in range(n)]


def test_save_and_read_back_embeddings_exactly():
    db.init_database()
    vectors = emb(1, 3)
    students_db.save_registration("1001", "Asha", vectors, review_evidence_id="abc-123")

    rec = students_db.get_student("1001", with_encodings=True)
    assert rec["name"] == "Asha" and rec["status"] == "pending" and rec["encoding_count"] == 3
    assert all(np.array_equal(a, b) for a, b in zip(rec["encodings"], vectors))
    assert "encodings" not in students_db.get_student("1001")      # not loaded unless asked


def test_reregistering_resets_the_review():
    db.init_database()
    students_db.save_registration("1001", "Asha", emb(1))
    students_db.set_status("1001", "approved", reviewed_by="proctor")
    students_db.save_registration("1001", "Asha K", emb(2))

    rec = students_db.get_student("1001")
    assert rec["name"] == "Asha K" and rec["status"] == "pending" and rec["reviewed_by"] is None


def test_status_changes_and_filtering():
    db.init_database()
    students_db.save_registration("1", "A", emb(1))
    students_db.save_registration("2", "B", emb(2))
    assert students_db.set_status("2", "approved", reviewed_by="p") is True
    assert students_db.set_status("missing", "approved") is False
    with pytest.raises(ValueError):
        students_db.set_status("1", "deleted")

    assert [s["student_id"] for s in students_db.list_students("approved")] == ["2"]
    assert {s["student_id"] for s in students_db.list_students()} == {"1", "2"}


def test_add_encodings_appends():
    db.init_database()
    students_db.save_registration("1", "A", emb(1, 2))
    assert students_db.add_encodings("1", emb(2, 3)) == 5
    assert students_db.add_encodings("nobody", emb(3)) is None


def test_empty_registration_is_refused():
    db.init_database()
    with pytest.raises(ValueError):
        students_db.save_registration("1", "A", [])
    with pytest.raises(ValueError):
        students_db.save_registration(" ", "A", emb(1))


def test_legacy_json_is_imported_once():
    legacy = {
        "1001": {"name": "Asha", "status": "approved", "encodings": [list(map(float, emb(1)[0]))]},
        "1002": {"name": "Bilal", "status": "pending", "encodings": [list(map(float, emb(2)[0]))],
                 "review_evidence_id": "rev-1"},
        "1003": {"name": "No faces", "status": "pending", "encodings": []},
    }
    with open(config.STUDENT_DATABASE_FILE, "w") as f:
        json.dump(legacy, f)

    db.init_database()
    reg = StudentRegistry()
    assert {s["id"] for s in reg.list_students()} == {"1001", "1002"}
    assert reg.get_student("1001")["status"] == "approved"

    students_db.set_status("1002", "approved")
    db.init_database()                                   # a restart must not re-import over changes
    assert students_db.get_student("1002")["status"] == "approved"
    assert os.path.exists(config.STUDENT_DATABASE_FILE)  # the legacy file is left alone


def test_registration_photo_path_is_validated(tmp_path):
    os.makedirs(config.REGISTRATION_EVIDENCE_DIR)
    open(os.path.join(config.REGISTRATION_EVIDENCE_DIR, "abc-123_front.jpg"), "wb").close()

    ok = {"review_evidence_id": "abc-123"}
    assert students_db.registration_photo_path(ok, "front").endswith("abc-123_front.jpg")
    assert students_db.registration_photo_path(ok, "left") is None             # file missing
    assert students_db.registration_photo_path(ok, "back") is None             # not a pose
    assert students_db.registration_photo_path({"review_evidence_id": "../x"}, "front") is None
    assert students_db.registration_photo_path({}, "front") is None
