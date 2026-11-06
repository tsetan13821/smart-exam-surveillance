# api/routes/sessions.py
import os
import re
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse

import config
from api import reports, session_store
from api.deps import current_user, get_manager, get_settings
from api.schemas import (ActiveStatus, AlertPage, SessionDetail, SessionInfo, SessionPage, SourceOption,
                         StartSessionRequest)
from app.session_manager import NoActiveSessionError

router = APIRouter(prefix="/api", tags=["sessions"], dependencies=[Depends(current_user)])

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
EVIDENCE_NAME = re.compile(r"[A-Z_]{1,64}_\d{8}_\d{6}_\d{6}\.jpg")
BOUNDARY = "frame"


# ----------------------------------------------------------------- sources
def _demo_videos(settings):
    folder = settings.demo_video_dir
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder)
                  if f.lower().endswith(VIDEO_EXTENSIONS) and os.path.isfile(os.path.join(folder, f)))


@router.get("/exam-options")
def exam_options():
    """Devices the object detector knows, which a proctor can allow for an exam."""
    return {"devices": list(config.CONTRABAND_CLASSES)}


@router.get("/sources", response_model=List[SourceOption])
def list_sources(settings=Depends(get_settings)):
    options = []
    if settings.camera_enabled:
        options.append({"type": "camera", "index": config.CAMERA_INDEX, "label": f"Camera {config.CAMERA_INDEX}"})
    options += [{"type": "video", "name": name, "label": f"Video: {name}"} for name in _demo_videos(settings)]
    return options


def _resolve_source(spec, settings):
    """Turns the client's source choice into what the engine opens: a camera index or a file path."""
    if spec is None or spec.type == "camera":
        if not settings.camera_enabled:
            raise HTTPException(422, "No camera is available on this server; choose a video.")
        return None if spec is None else spec.index
    if spec.name not in _demo_videos(settings):       # only files listed by /api/sources, no paths
        raise HTTPException(422, f"Unknown video '{spec.name}'.")
    return os.path.join(settings.demo_video_dir, spec.name)


# ---------------------------------------------------------------- sessions
@router.get("/sessions", response_model=SessionPage)
def list_sessions(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
                  status_filter: Optional[Literal["running", "ended", "failed"]] = Query(None, alias="status"),
                  student_id: Optional[str] = Query(None, max_length=64)):
    return session_store.list_sessions(limit, offset, status_filter, student_id)


@router.get("/sessions/active", response_model=ActiveStatus)
def active_session(manager=Depends(get_manager)):
    return manager.status()


@router.post("/sessions", response_model=SessionInfo, status_code=status.HTTP_202_ACCEPTED)
def start_session(body: StartSessionRequest, manager=Depends(get_manager), settings=Depends(get_settings)):
    """Starts a session in the background. Watch /api/live for it to reach 'running'."""
    source = _resolve_source(body.source, settings)
    return manager.start(body.student_ids, body.exam_context.model_dump(), source)


@router.post("/sessions/{session_id}/stop", response_model=SessionInfo, status_code=status.HTTP_202_ACCEPTED)
def stop_session(session_id: str, manager=Depends(get_manager)):
    return manager.stop(session_id)


@router.get("/sessions/{session_id}", response_model=SessionDetail)
def get_session(session_id: str):
    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found.")
    session["report"] = reports.build_report(session, session_store.all_alerts(session_id))
    return session


@router.get("/sessions/{session_id}/alerts", response_model=AlertPage)
def list_alerts(session_id: str, limit: int = Query(200, ge=1, le=1000), offset: int = Query(0, ge=0),
                level: Optional[List[Literal["INFO", "ALERT", "EVIDENCE"]]] = Query(None),
                incidents_only: bool = False):
    if session_store.get_session(session_id) is None:
        raise HTTPException(404, "Session not found.")
    return session_store.list_alerts(session_id, limit, offset, level, incidents_only)


@router.get("/sessions/{session_id}/stream")
async def stream(session_id: str, request: Request, manager=Depends(get_manager)):
    """Annotated video of the running session as MJPEG (use it as an <img> src)."""
    if manager.active_session_id() != session_id:
        raise HTTPException(404, "Session is not running.")

    async def frames():
        seq = 0
        while not await request.is_disconnected():
            try:
                seq, jpeg = await run_in_threadpool(manager.wait_for_jpeg, session_id, seq, 1.0)
            except NoActiveSessionError:
                return
            if jpeg is not None:
                yield (f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n"
                       .encode("ascii") + jpeg + b"\r\n")

    return StreamingResponse(frames(), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------- evidence
@router.get("/evidence/{filename}", tags=["evidence"])
def get_evidence(filename: str):
    if not EVIDENCE_NAME.fullmatch(filename):
        raise HTTPException(404, "Evidence not found.")
    path = os.path.join(config.PROCTORING_LOG_DIR, "evidence", filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "Evidence not found.")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})
