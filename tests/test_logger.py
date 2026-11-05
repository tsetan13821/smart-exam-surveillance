"""ActivityLogger: evidence screenshots use the annotated frame."""
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def test_evidence_saved_after_annotation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)               # logger and database use relative paths
    import app.utils.database as db
    from app.utils.logger import ActivityLogger

    db.init_database()
    logger = ActivityLogger("s1", '["1"]', '["A"]', log_dir="proctoring_logs")
    try:
        raw = np.zeros((48, 64, 3), np.uint8)
        logger.begin_frame()
        logger.log_evidence(raw, "CONTRABAND_DETECTED", "phone")
        annotated = raw.copy()
        annotated[:, :] = (0, 0, 255)          # stands in for boxes / status drawn later
        logger.end_frame(annotated)

        files = os.listdir(os.path.join("proctoring_logs", "evidence"))
        assert len(files) == 1 and files[0].startswith("CONTRABAND_DETECTED_")
        saved = cv2.imread(os.path.join("proctoring_logs", "evidence", files[0]))
        assert saved[..., 2].mean() > 200      # red: the annotated frame was saved, not the raw one
    finally:
        logger.close()


def test_evidence_outside_a_frame_is_saved_immediately(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import app.utils.database as db
    from app.utils.logger import ActivityLogger

    db.init_database()
    logger = ActivityLogger("s2", '["1"]', '["A"]', log_dir="proctoring_logs")
    try:
        path = logger.log_evidence(np.zeros((8, 8, 3), np.uint8), "ATTENDANCE_MARKED", "present")
        assert path and os.path.exists(path)
    finally:
        logger.close()
