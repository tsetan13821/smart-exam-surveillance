# main.py
"""Command-line entry point: runs one proctoring session in an OpenCV window.

The web API runs sessions through app/session_manager.py instead; both share
app/session_setup.py for choosing students and loading their face data.

Example:
  python main.py --student_ids '["1001"]' --exam_context '{"is_paper_exam": true}'
  python main.py --student_ids '["1001"]' --source benchmarks/clips/live/reading.mp4
"""
import argparse
import json
import sys
import uuid

import app.utils.database as db
import config
from app.core import ProctoringApp
from app.processors.face_recognizer import FaceRecognizer
from app.session_manager import normalize_exam_context
from app.session_setup import SessionSetupError, load_desk_rois, plan_session  # noqa: F401 (re-exported)
from app.utils.logger import ActivityLogger
from app.utils.student_registry import StudentRegistry
from app.video_source import open_source



def main():
    parser = argparse.ArgumentParser(description="Proctoring Application Engine")
    parser.add_argument("--student_ids", required=True,
                        help="A JSON string list of student IDs for this session.")
    parser.add_argument("--exam_context", required=False, default="{}",
                        help="A JSON string of exam rules (e.g., allowed devices).")
    parser.add_argument("--source", default=None,
                        help="Camera index or video file (default: config.CAMERA_INDEX).")
    args = parser.parse_args()

    try:
        student_ids = json.loads(args.student_ids)
        exam_context = normalize_exam_context(json.loads(args.exam_context))
        db.init_database()
        plan = plan_session(student_ids, StudentRegistry())
    except (json.JSONDecodeError, SessionSetupError, FileNotFoundError) as e:
        sys.exit(f"Cannot start session: {e}")

    for s in plan.skipped:
        print(f"  > Skipped {s['id']}: {s['reason']}")
    print(f"Students: {', '.join(plan.student_names)} | Exam rules: {exam_context}")

    session_id = str(uuid.uuid4())
    ids_json, names_json = json.dumps(plan.student_ids), json.dumps(plan.student_names)
    db.create_session(session_id, ids_json, names_json, exam_context_json=json.dumps(exam_context),
                      source=args.source)
    logger = ActivityLogger(session_id, ids_json, names_json, log_dir=config.PROCTORING_LOG_DIR)
    status = "ended"
    try:
        logger.log_activity(f"Application starting for session {session_id}...")
        video = open_source(args.source, logger=logger)
        recognizer = FaceRecognizer(logger=logger, authorized_encodings=plan.encodings)
        app = ProctoringApp(logger=logger, face_recognizer=recognizer,
                            authorized_student_names=plan.student_names, exam_context=exam_context,
                            rois_data=plan.rois, video_source=video, display=True)
        app.run()
    except Exception as e:
        status = "failed"
        print(f"An unexpected error occurred: {e}")
        logger.log_alert("FATAL_ERROR", str(e))
    finally:
        db.end_session(session_id, status)
        logger.log_activity("Application shutting down.")
        logger.close()


if __name__ == '__main__':
    main()
