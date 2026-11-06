# app/session_manager.py
"""Runs proctoring sessions in the background for the web API.

There is one camera, so at most one session runs at a time. A session goes through

    starting  -> models load and the video source opens (several seconds on CPU)
    running   -> the engine analyses frames; status and JPEG frames are published
    stopping  -> a stop was requested; the engine is shutting down
    ended / failed

and the database row is marked 'ended' or 'failed' however the session finishes.
Listeners receive {"type": "state", ...} on every state change and
{"type": "log", ...} for each activity/alert/evidence record the engine logs.
"""
import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime

import config
import app.utils.database as db
from app.session_setup import SessionSetupError, plan_session



class SessionBusyError(RuntimeError):
    """Another session is already starting, running or stopping."""


class NoActiveSessionError(LookupError):
    """There is no running session (or not the one asked for)."""


def normalize_exam_context(ctx):
    """Validates the exam rules sent by a client; returns the cleaned dict."""
    if ctx is None:
        ctx = {}
    if not isinstance(ctx, dict):
        raise SessionSetupError("exam_context must be an object.")
    allowed = ctx.get("allowed_devices", [])
    if not isinstance(allowed, list) or not all(isinstance(d, str) for d in allowed):
        raise SessionSetupError("allowed_devices must be a list of device names.")
    known = set(getattr(config, "CONTRABAND_CLASSES", []))
    unknown = sorted(set(allowed) - known)
    if unknown:
        raise SessionSetupError(f"Unknown device(s): {', '.join(unknown)}. Known: {', '.join(sorted(known))}.")
    return {
        "is_open_book": bool(ctx.get("is_open_book", False)),
        "is_paper_exam": bool(ctx.get("is_paper_exam", False)),
        "allowed_devices": sorted(set(allowed)),
    }


def default_engine_factory(plan, exam_context, logger, source):
    """Opens the video source, loads the models and builds a headless ProctoringApp."""
    from app.core import ProctoringApp
    from app.processors.face_recognizer import FaceRecognizer
    from app.video_source import open_source

    video = open_source(source, logger=logger)      # fail fast if the camera/file is missing
    try:
        recognizer = FaceRecognizer(logger=logger, authorized_encodings=plan.encodings)
        return ProctoringApp(logger=logger, face_recognizer=recognizer,
                             authorized_student_names=plan.student_names, exam_context=exam_context,
                             rois_data=plan.rois, video_source=video, display=False, publish_jpeg=True)
    except Exception:
        video.release()
        raise


@dataclass
class _Session:
    id: str
    plan: object
    exam_context: dict
    source: object
    logger: object
    started_at: str
    state: str = "starting"
    engine: object = None
    thread: threading.Thread = None
    error: str = None
    ended_at: str = None
    stop_requested: bool = False
    stopped_by: str = None
    done: threading.Event = field(default_factory=threading.Event)

    def info(self):
        return {
            "session_id": self.id,
            "state": self.state,
            "student_ids": list(self.plan.student_ids),
            "student_names": list(self.plan.student_names),
            "skipped_students": list(self.plan.skipped),
            "exam_context": dict(self.exam_context),
            "source": None if self.source is None else str(self.source),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "error": self.error,
        }


class SessionManager:
    def __init__(self, registry_factory=None, engine_factory=None, logger_factory=None):
        if registry_factory is None:
            from app.utils.student_registry import StudentRegistry
            registry_factory = StudentRegistry
        if logger_factory is None:
            from app.utils.logger import ActivityLogger

            def logger_factory(session_id, ids_json, names_json):
                return ActivityLogger(session_id, ids_json, names_json, log_dir=config.PROCTORING_LOG_DIR)
        self.registry_factory = registry_factory
        self.engine_factory = engine_factory or default_engine_factory
        self.logger_factory = logger_factory
        self._lock = threading.RLock()
        # Held while a state changes and its event goes out, so listeners see
        # starting -> running -> stopping -> ended in order. Always taken before _lock.
        self._transition_lock = threading.Lock()
        self._active = None
        self._last = None
        self._listeners = []

    # ------------------------------------------------------------- listeners
    def add_listener(self, fn):
        with self._lock:
            self._listeners.append(fn)

    def remove_listener(self, fn):
        with self._lock:
            if fn in self._listeners:
                self._listeners.remove(fn)

    def _emit(self, event):
        with self._lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(event)
            except Exception as e:
                print(f"SESSION_LISTENER_ERROR: {e}")

    def _emit_state(self, info):
        self._emit({"type": "state", "session": info})

    # ----------------------------------------------------------------- start
    def start(self, student_ids, exam_context=None, source=None):
        """Validates the request and starts a session in the background.

        Returns the session info (state 'starting'). Raises SessionBusyError if a
        session is active and SessionSetupError for invalid students or rules.
        """
        exam_context = normalize_exam_context(exam_context)
        with self._transition_lock, self._lock:
            if self._active is not None:
                raise SessionBusyError(f"Session {self._active.id} is {self._active.state}.")
            plan = plan_session(student_ids, self.registry_factory())

            session_id = str(uuid.uuid4())
            ids_json, names_json = json.dumps(plan.student_ids), json.dumps(plan.student_names)
            db.init_database()
            db.create_session(session_id, ids_json, names_json,
                              exam_context_json=json.dumps(exam_context),
                              source=None if source is None else str(source))
            logger = self.logger_factory(session_id, ids_json, names_json)
            session = _Session(id=session_id, plan=plan, exam_context=exam_context, source=source,
                               logger=logger, started_at=datetime.now().isoformat())
            add_listener = getattr(logger, "add_listener", None)
            if add_listener:
                add_listener(lambda entry: self._emit({"type": "log", **entry}))
            logger.log_activity(f"Session {session_id} starting for {', '.join(plan.student_names)}.")
            session.thread = threading.Thread(target=self._run, args=(session,),
                                              name=f"session-{session_id[:8]}", daemon=True)
            self._active = session
            session.thread.start()
            info = session.info()
            self._emit_state(info)
        return info

    def _run(self, session):
        status = "ended"
        try:
            engine = self.engine_factory(session.plan, session.exam_context, session.logger, session.source)
            with self._transition_lock, self._lock:
                session.engine = engine
                if session.stop_requested:
                    engine.stop()               # stopped while the models were loading
                else:
                    session.state = "running"
                    self._emit_state(session.info())
            engine.run()                        # blocks until stopped or the video ends; cleans up
        except Exception as e:
            status = "failed"
            session.error = f"{type(e).__name__}: {e}"
            try:
                session.logger.log_alert("SESSION_FAILED", session.error)
            except Exception:
                pass
        finally:
            try:
                db.end_session(session.id, status)
            except Exception as e:
                print(f"SESSION_DB_ERROR: {e}")
            try:
                session.logger.log_activity(f"Session {status}.")
                session.logger.close()
            except Exception:
                pass
            with self._transition_lock, self._lock:
                session.state = status
                session.ended_at = datetime.now().isoformat()
                self._last = session
                if self._active is session:
                    self._active = None
                self._emit_state(session.info())
            session.done.set()

    # ------------------------------------------------------------------ stop
    def stop(self, session_id=None, wait=False, timeout=30.0):
        """Requests the active session to stop. Returns its info."""
        with self._transition_lock, self._lock:
            session = self._active
            if session is None or (session_id is not None and session.id != session_id):
                raise NoActiveSessionError("No such running session.")
            info = session.info()
            if session.state in ("starting", "running"):
                session.stop_requested = True
                session.state = "stopping"
                if session.engine is not None:
                    session.engine.stop()
                info = session.info()
                self._emit_state(info)
        if wait:
            session.done.wait(timeout)
            info = session.info()
        return info

    def shutdown(self, timeout=30.0):
        """Stops any active session and waits for it (used when the server exits)."""
        try:
            self.stop(wait=True, timeout=timeout)
        except NoActiveSessionError:
            pass

    # ----------------------------------------------------------------- query
    def active_session_id(self):
        with self._lock:
            return self._active.id if self._active else None

    def status(self):
        """State of the active session (with live engine status), or 'idle'."""
        with self._lock:
            session = self._active
            if session is None:
                last = self._last.info() if self._last else None
                return {"state": "idle", "session": None, "last_session": last, "live": None}
            info, engine = session.info(), session.engine
        live = engine.get_status() if engine is not None and info["state"] == "running" else None
        return {"state": info["state"], "session": info, "last_session": None, "live": live}

    def wait_for_jpeg(self, session_id, after_seq=0, timeout=1.0):
        """Next annotated JPEG of the given running session: (seq, bytes or None)."""
        with self._lock:
            session = self._active
            if session is None or session.id != session_id:
                raise NoActiveSessionError("Session is not running.")
            engine = session.engine
        if engine is None:                  # still loading models
            session.done.wait(min(timeout, 0.2))
            return after_seq, None
        return engine.wait_for_jpeg(after_seq, timeout)
