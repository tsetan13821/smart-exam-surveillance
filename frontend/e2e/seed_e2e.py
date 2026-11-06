"""Creates a fresh database for the end-to-end tests (run before the API starts).

Uses the SMARTSQUAD_* paths from the environment, so the real data is never touched.
Students get random face embeddings: they won't match anyone in a video.
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402

for var in ("SMARTSQUAD_DB", "SMARTSQUAD_REGISTRATIONS_DIR", "SMARTSQUAD_LOG_DIR", "SMARTSQUAD_LEGACY_REGISTRY"):
    if not os.environ.get(var):
        sys.exit(f"{var} must be set: refusing to seed the real data folders.")

for path in (config.DATABASE_FILE, config.DATABASE_FILE + "-wal", config.DATABASE_FILE + "-shm"):
    if os.path.exists(path):
        os.remove(path)
for folder in (config.REGISTRATION_EVIDENCE_DIR, config.PROCTORING_LOG_DIR):
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)

import app.utils.database as db  # noqa: E402
from app.utils import students_db  # noqa: E402

db.init_database()
rng = np.random.default_rng(7)


def embeddings():
    return [rng.standard_normal(512).astype(np.float32) for _ in range(3)]


students_db.save_registration("E2E-001", "Asha Rao", embeddings(), status="approved")
students_db.save_registration("E2E-002", "Bilal Khan", embeddings(), review_evidence_id="e2e-bilal")

for pose, colour in (("front", (90, 140, 40)), ("left", (40, 90, 140)), ("right", (140, 60, 90))):
    img = np.full((240, 320, 3), colour, np.uint8)
    cv2.putText(img, f"Bilal - {pose}", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(config.REGISTRATION_EVIDENCE_DIR, f"e2e-bilal_{pose}.jpg"), img)

print(f"Seeded {config.DATABASE_FILE}")
