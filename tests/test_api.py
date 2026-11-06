"""Web API end to end: real database, registry and logger; a scripted engine instead of the models."""
import os
import sqlite3
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import app.utils.database as db
import config
from api import auth_store
from api.main import create_app
from api.settings import Settings
from app.session_manager import SessionManager
from app.utils import students_db

PASSWORD = "correct-horse-battery"
JPEG = b"\xff\xd8fake-jpeg\xff\xd9"


class ScriptedEngine:
    """Behaves like ProctoringApp from the API's point of view."""

    def __init__(self, logger, script):
        self.logger, self.script = logger, script
        self.stop_event = threading.Event()
        self.jpegs_sent = 0

    def run(self):
        for alert_type, message, subject in self.script.get("evidence", []):
            self.logger.log_evidence(np.zeros((8, 8, 3), np.uint8), alert_type, message, subject=subject)
        self.stop_event.wait(self.script.get("run_seconds", 30))

    def stop(self):
        self.stop_event.set()

    def get_status(self):
        return {"classification": "Normal", "fps": 30.0, "frames_processed": 42,
                "students": [{"name": "Asha", "present": True, "ever_present": True}], "people": []}

    def wait_for_jpeg(self, after_seq=0, timeout=1.0):
        limit = self.script.get("max_jpegs")
        if self.stop_event.is_set():
            return after_seq, None
        if limit is not None and self.jpegs_sent >= limit:
            self.stop()                     # the clip ran out: the session ends by itself
            return after_seq, None
        time.sleep(0.01)
        self.jpegs_sent += 1
        return after_seq + 1, JPEG


@pytest.fixture
def script():
    return {}


@pytest.fixture
def api(tmp_path, script):
    db.init_database()
    students_db.save_registration("1", "Asha", [np.ones(512, np.float32)], review_evidence_id="rev-1",
                                  status="approved")
    students_db.save_registration("2", "Bilal", [np.ones(512, np.float32)], status="approved")
    students_db.save_registration("3", "Chen", [np.ones(512, np.float32)])          # pending
    os.makedirs(config.REGISTRATION_EVIDENCE_DIR)
    with open(os.path.join(config.REGISTRATION_EVIDENCE_DIR, "rev-1_front.jpg"), "wb") as f:
        f.write(JPEG)
    os.makedirs(config.DEMO_VIDEO_DIR)
    open(os.path.join(config.DEMO_VIDEO_DIR, "demo.mp4"), "wb").close()

    sources = []

    def engine_factory(plan, exam_context, logger, source):
        sources.append(source)
        if script.get("load_error"):
            raise RuntimeError(script["load_error"])
        time.sleep(script.get("load_seconds", 0))
        return ScriptedEngine(logger, script)

    manager = SessionManager(engine_factory=engine_factory)
    app = create_app(Settings(demo_video_dir=config.DEMO_VIDEO_DIR,
                              frontend_dist=str(tmp_path / "no-frontend")), manager=manager)
    with TestClient(app) as client:
        auth_store.create_user("proctor", PASSWORD)
        client.sources = sources
        yield client


def login(client, username="proctor", password=PASSWORD):
    return client.post("/api/auth/login", json={"username": username, "password": password})


@pytest.fixture
def authed(api):
    assert login(api).status_code == 200
    return api


def wait_for_state(client, state, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get("/api/sessions/active").json()
        current = body["state"] if body["state"] != "idle" else (body["last_session"] or {}).get("state", "idle")
        if current == state:
            return body
        time.sleep(0.02)
    raise AssertionError(f"session never reached {state!r}: {body}")


def start(client, ids=("1", "2"), **extra):
    return client.post("/api/sessions", json={"student_ids": list(ids), **extra})


# ------------------------------------------------------------------ auth
def test_health_is_public_and_everything_else_needs_login(api):
    assert api.get("/api/health").json() == {"status": "ok"}
    for path in ("/api/auth/me", "/api/students", "/api/sessions", "/api/sessions/active", "/api/sources",
                 "/api/evidence/CONTRABAND_DETECTED_20261106_011509_850481.jpg"):
        assert api.get(path).status_code == 401, path
    assert start(api).status_code == 401


def test_login_sets_a_strict_httponly_cookie_and_logout_revokes_it(api):
    r = login(api)
    assert r.status_code == 200 and r.json() == {"username": "proctor"}
    cookie = r.headers["set-cookie"].lower()
    assert "ss_session=" in cookie and "httponly" in cookie and "samesite=strict" in cookie
    token = api.cookies.get("ss_session")
    assert api.get("/api/auth/me").json() == {"username": "proctor"}

    assert api.post("/api/auth/logout").status_code == 204
    assert api.get("/api/auth/me").status_code == 401
    api.cookies.set("ss_session", token)                 # replaying the old token doesn't work either
    assert api.get("/api/auth/me").status_code == 401


def test_tokens_are_stored_hashed():
    db.init_database()
    auth_store.create_schema()
    uid = auth_store.create_user("someone", PASSWORD)
    token = auth_store.create_login_session(uid, hours=1)
    with sqlite3.connect(config.DATABASE_FILE) as conn:
        stored = [r[0] for r in conn.execute("SELECT token_hash FROM auth_sessions")]
        pw = conn.execute("SELECT password_hash FROM users").fetchone()[0]
    assert token not in stored and PASSWORD not in pw and pw.startswith("scrypt$")


def test_wrong_and_unknown_users_get_the_same_answer(api):
    a, b = login(api, password="wrong-password"), login(api, username="nobody")
    assert a.status_code == b.status_code == 401 and a.json() == b.json()
    assert "ss_session" not in api.cookies


def test_repeated_failures_lock_the_account_for_that_client(api):
    for _ in range(5):
        assert login(api, password="wrong-password").status_code == 401
    locked = login(api)                                   # even the right password is refused for now
    assert locked.status_code == 429 and int(locked.headers["retry-after"]) > 0
    assert login(api, username="someone-else").status_code == 401   # other usernames unaffected


def test_expired_login_is_rejected(api):
    user = auth_store.get_user_by_username("proctor")
    api.cookies.set("ss_session", auth_store.create_login_session(user["user_id"], hours=-1))
    assert api.get("/api/auth/me").status_code == 401


def test_password_change_signs_out_everywhere(authed):
    assert auth_store.set_password("proctor", "another-long-password")
    assert authed.get("/api/auth/me").status_code == 401


def test_cross_site_requests_are_refused(authed):
    evil = {"Origin": "https://evil.example"}
    assert authed.post("/api/sessions", json={"student_ids": ["1"]}, headers=evil).status_code == 403
    assert authed.patch("/api/students/3", json={"status": "approved"}, headers=evil).status_code == 403
    ok = authed.patch("/api/students/3", json={"status": "approved"}, headers={"Origin": "http://localhost:5173"})
    assert ok.status_code == 200


def test_security_headers(authed):
    r = authed.get("/api/students")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["cache-control"] == "no-store"
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]


# -------------------------------------------------------------- students
def test_students_list_filter_and_review(authed):
    assert {s["student_id"] for s in authed.get("/api/students").json()} == {"1", "2", "3"}
    pending = authed.get("/api/students", params={"status": "pending"}).json()
    assert [s["student_id"] for s in pending] == ["3"]
    assert "encodings" not in pending[0] and "review_evidence_id" not in pending[0]
    assert authed.get("/api/students", params={"status": "bogus"}).status_code == 422

    r = authed.patch("/api/students/3", json={"status": "approved"})
    assert r.status_code == 200 and r.json()["status"] == "approved" and r.json()["reviewed_by"] == "proctor"
    assert authed.patch("/api/students/3", json={"status": "deleted"}).status_code == 422
    assert authed.patch("/api/students/3", json={"status": "approved", "x": 1}).status_code == 422
    assert authed.patch("/api/students/999", json={"status": "approved"}).status_code == 404


def test_registration_photos(authed):
    asha = authed.get("/api/students/1").json()
    assert asha["photos"] == ["/api/students/1/photos/front"]
    photo = authed.get(asha["photos"][0])
    assert photo.status_code == 200 and photo.content == JPEG and photo.headers["content-type"] == "image/jpeg"
    assert authed.get("/api/students/1/photos/left").status_code == 404
    assert authed.get("/api/students/1/photos/..%2F..%2Fconfig").status_code in (404, 422)
    assert authed.get("/api/students/2/photos/front").status_code == 404


# -------------------------------------------------------------- sessions
def test_sources_list_the_camera_and_demo_videos(authed):
    assert authed.get("/api/sources").json() == [
        {"type": "camera", "label": f"Camera {config.CAMERA_INDEX}", "index": config.CAMERA_INDEX, "name": None},
        {"type": "video", "label": "Video: demo.mp4", "index": None, "name": "demo.mp4"},
    ]


def test_exam_options_list_the_detectable_devices(authed):
    assert authed.get("/api/exam-options").json() == {"devices": list(config.CONTRABAND_CLASSES)}


def test_only_listed_videos_can_be_opened(authed):
    for name in ("../config.py", "C:/Windows/win.ini", "missing.mp4"):
        r = start(authed, source={"type": "video", "name": name})
        assert r.status_code == 422, name
    assert authed.get("/api/sessions").json()["total"] == 0


@pytest.mark.parametrize("script", [{"evidence": [
    ("ATTENDANCE_MARKED", "Student Asha marked PRESENT.", "Asha"),
    ("CONTRABAND_DETECTED", "(Asha) Contraband: cell phone", "Asha"),
    ("UNAUTHORIZED_FACE_DETECTED", "Unauthorized face detected (1 found).", None),
]}])
def test_session_lifecycle_report_and_evidence(authed, script):
    r = start(authed, exam_context={"is_paper_exam": True, "allowed_devices": ["book"]},
              source={"type": "video", "name": "demo.mp4"})
    assert r.status_code == 202, r.text
    session = r.json()
    sid = session["session_id"]
    assert session["state"] == "starting" and session["student_names"] == ["Asha", "Bilal"]
    assert authed.sources == [os.path.join(config.DEMO_VIDEO_DIR, "demo.mp4")]

    running = wait_for_state(authed, "running")
    assert running["live"]["frames_processed"] == 42
    assert start(authed, ids=["1"]).status_code == 409             # one camera, one session

    assert authed.post("/api/sessions/not-this-one/stop").status_code == 404
    assert authed.post(f"/api/sessions/{sid}/stop").status_code == 202
    wait_for_state(authed, "ended")

    page = authed.get("/api/sessions").json()
    assert page["total"] == 1
    summary = page["items"][0]
    assert summary["status"] == "ended" and summary["end_time"]
    assert summary["incident_count"] == 2 and summary["evidence_count"] == 3
    assert summary["exam_context"] == {"is_open_book": False, "is_paper_exam": True, "allowed_devices": ["book"]}

    detail = authed.get(f"/api/sessions/{sid}").json()
    by_name = {s["name"]: s for s in detail["report"]["students"]}
    assert by_name["Asha"]["attendance"] == "present" and by_name["Asha"]["incidents"] == {"CONTRABAND_DETECTED": 1}
    assert by_name["Bilal"]["attendance"] == "absent"
    assert detail["report"]["unassigned_incidents"] == {"UNAUTHORIZED_FACE_DETECTED": 1}

    incidents = authed.get(f"/api/sessions/{sid}/alerts", params={"incidents_only": True}).json()
    assert [a["alert_type"] for a in incidents["items"]] == ["CONTRABAND_DETECTED", "UNAUTHORIZED_FACE_DETECTED"]
    url = incidents["items"][0]["evidence_url"]
    assert url.startswith("/api/evidence/CONTRABAND_DETECTED_") and "\\" not in url
    image = authed.get(url)
    assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"

    assert authed.get("/api/sessions/unknown").status_code == 404
    assert authed.get("/api/sessions", params={"student_id": "2"}).json()["total"] == 1
    assert authed.get("/api/sessions", params={"student_id": "3"}).json()["total"] == 0


def test_evidence_names_are_validated(authed):
    for name in ("..%2F..%2Fconfig.py", "config.py", "X_20261106_011509_850481.png"):
        assert authed.get(f"/api/evidence/{name}").status_code == 404, name


def test_invalid_session_requests(authed):
    assert start(authed, ids=["3"]).status_code == 422                       # pending only
    assert start(authed, ids=[]).status_code == 422
    assert start(authed, exam_context={"allowed_devices": ["rocket"]}).status_code == 422
    assert start(authed, exam_context={"cheat_mode": True}).status_code == 422
    assert start(authed, source={"type": "camera", "index": -1}).status_code == 422
    r = start(authed, ids=["1", "3", "404"])
    assert r.status_code == 202
    assert {s["id"] for s in r.json()["skipped_students"]} == {"3", "404"}


@pytest.mark.parametrize("script", [{"load_error": "camera 0 could not be opened"}])
def test_engine_failure_is_reported(authed, script):
    sid = start(authed).json()["session_id"]
    failed = wait_for_state(authed, "failed")
    assert "camera 0 could not be opened" in failed["last_session"]["error"]
    assert authed.get(f"/api/sessions/{sid}").json()["status"] == "failed"
    assert start(authed).status_code == 202                                  # the camera is free again


# ----------------------------------------------------------------- video
@pytest.mark.parametrize("script", [{"max_jpegs": 3}])
def test_mjpeg_stream(authed, script):
    sid = start(authed).json()["session_id"]
    wait_for_state(authed, "running")
    r = authed.get(f"/api/sessions/{sid}/stream")       # ends when the scripted clip runs out
    assert r.status_code == 200
    assert r.headers["content-type"] == "multipart/x-mixed-replace; boundary=frame"
    parts = [p for p in r.content.split(b"--frame\r\n") if p]
    assert len(parts) == 3
    head, body = parts[0].split(b"\r\n\r\n", 1)
    assert b"Content-Type: image/jpeg" in head and body == JPEG + b"\r\n"
    wait_for_state(authed, "ended")
    assert authed.get(f"/api/sessions/{sid}/stream").status_code == 404


# ------------------------------------------------------------ websocket
def receive_until(ws, predicate, limit=200):
    seen = []
    for _ in range(limit):
        msg = ws.receive_json()
        seen.append(msg)
        if predicate(msg):
            return msg, seen
    raise AssertionError(f"no matching message in {[m['type'] for m in seen]}")


def test_websocket_requires_login(api):
    with pytest.raises(WebSocketDisconnect) as e:
        with api.websocket_connect("/api/live") as ws:
            ws.receive_json()
    assert e.value.code == 4401


def test_websocket_refuses_other_origins(authed):
    with pytest.raises(WebSocketDisconnect) as e:
        with authed.websocket_connect("/api/live", headers={"origin": "https://evil.example"}) as ws:
            ws.receive_json()
    assert e.value.code == 4403


@pytest.mark.parametrize("script", [{"load_seconds": 0.2, "evidence": [
    ("CONTRABAND_DETECTED", "(Asha) Contraband: cell phone", "Asha")]}])
def test_websocket_streams_state_events_and_status(authed, script):
    with authed.websocket_connect("/api/live", headers={"origin": "http://localhost:5173"}) as ws:
        snapshot = ws.receive_json()
        assert snapshot["type"] == "snapshot" and snapshot["status"]["state"] == "idle"

        sid = start(authed).json()["session_id"]
        _, seen = receive_until(ws, lambda m: m["type"] == "state" and m["session"]["state"] == "running")
        assert any(m["type"] == "state" and m["session"]["state"] == "starting" for m in seen)

        evidence, _ = receive_until(ws, lambda m: m["type"] == "log" and m.get("level") == "EVIDENCE")
        assert evidence["session_id"] == sid and evidence["subject"] == "Asha"
        assert evidence["evidence_url"].startswith("/api/evidence/CONTRABAND_DETECTED_")
        assert "evidence_file" not in evidence                     # no disk paths leave the server

        status, _ = receive_until(ws, lambda m: m["type"] == "status")
        assert status["session"]["session_id"] == sid and status["live"]["classification"] == "Normal"

        authed.post(f"/api/sessions/{sid}/stop")
        receive_until(ws, lambda m: m["type"] == "state" and m["session"]["state"] == "ended")

    # A tab that connects later gets the session's recent alerts in its snapshot
    with authed.websocket_connect("/api/live") as ws:
        assert ws.receive_json()["recent"] == []                    # nothing running now


def test_reconnecting_tab_gets_recent_alerts(authed, script):
    script["evidence"] = [("CONTRABAND_DETECTED", "(Asha) Contraband: cell phone", "Asha")]
    start(authed)
    wait_for_state(authed, "running")
    deadline = time.time() + 5
    while time.time() < deadline:
        with authed.websocket_connect("/api/live") as ws:
            recent = ws.receive_json()["recent"]
        if recent:
            break
        time.sleep(0.05)
    assert recent[-1]["alert_type"] == "CONTRABAND_DETECTED" and recent[-1]["evidence_url"]


# --------------------------------------------------------------- startup
def test_sessions_cut_off_by_a_restart_are_marked_failed(tmp_path):
    db.init_database()
    db.create_session("old", '["1"]', '["Asha"]')
    app = create_app(Settings(frontend_dist=str(tmp_path / "none")),
                     manager=SessionManager(engine_factory=lambda *a: None))
    with TestClient(app):
        pass
    with sqlite3.connect(config.DATABASE_FILE) as conn:
        assert conn.execute("SELECT status FROM sessions WHERE session_id='old'").fetchone()[0] == "failed"


def test_built_frontend_is_served_with_client_side_routes(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>SmartSquad</title>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    app = create_app(Settings(frontend_dist=str(dist)), manager=SessionManager(engine_factory=lambda *a: None))
    with TestClient(app) as client:
        assert "SmartSquad" in client.get("/sessions/abc").text
        assert client.get("/assets/app.js").text == "console.log(1)"
        assert client.get("/api/nope").status_code == 404
        assert client.get("/..%2Fsecret").status_code in (200, 404)     # falls back to index, never escapes
