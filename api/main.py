# api/main.py
"""FastAPI application: proctor login, students, sessions, live video and events.

Run with `python -m api` (one worker: sessions and the camera live in this process).
"""
import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import app.utils.database as db
from api import auth_store
from api.deps import origin_allowed
from api.events import EventHub
from api.routes import auth, live, sessions, students
from api.security import LoginThrottle
from api.settings import Settings
from app.session_manager import NoActiveSessionError, SessionBusyError, SessionManager
from app.session_setup import SessionSetupError

UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
CSP = ("default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
       "connect-src 'self' ws: wss:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'")


def _bootstrap_admin():
    """Docker/demo convenience: create the first proctor from environment variables."""
    username, password = os.environ.get("SMARTSQUAD_ADMIN_USER"), os.environ.get("SMARTSQUAD_ADMIN_PASSWORD")
    if username and password and auth_store.count_users() == 0:
        auth_store.create_user(username, password)
        print(f"Created proctor account '{username}'.")


def create_app(settings=None, manager=None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        await run_in_threadpool(db.init_database)
        await run_in_threadpool(auth_store.create_schema)
        await run_in_threadpool(_bootstrap_admin)
        interrupted = await run_in_threadpool(db.mark_interrupted_sessions)
        if interrupted:
            print(f"Marked {interrupted} interrupted session(s) as failed.")
        app.state.hub.bind(asyncio.get_running_loop())
        app.state.manager.add_listener(app.state.hub.publish)
        try:
            yield
        finally:
            app.state.manager.remove_listener(app.state.hub.publish)
            await run_in_threadpool(app.state.manager.shutdown)

    app = FastAPI(title="SmartSquad Proctoring API", version="2.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.manager = manager or SessionManager()
    app.state.hub = EventHub()
    app.state.login_throttle = LoginThrottle(settings.max_login_failures, settings.login_lockout_seconds)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        # Cross-site request forgery: state-changing requests must come from our own pages.
        if request.method in UNSAFE_METHODS and not origin_allowed(request.headers.get("origin"), settings):
            return JSONResponse({"detail": "Cross-origin request refused."}, status_code=403)
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("X-Frame-Options", "DENY")
        if not request.url.path.startswith(("/docs", "/redoc")):      # Swagger UI loads from a CDN
            response.headers.setdefault("Content-Security-Policy", CSP)
        if request.url.path.startswith("/api/") and "Cache-Control" not in response.headers:
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(SessionSetupError)
    async def _setup_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(SessionBusyError)
    async def _busy(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(NoActiveSessionError)
    async def _not_running(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.get("/api/health", tags=["meta"])
    def health():
        return {"status": "ok"}

    app.include_router(auth.router)
    app.include_router(students.router)
    app.include_router(sessions.router)
    app.include_router(live.router)
    _mount_frontend(app, settings.frontend_dist)
    return app


def _mount_frontend(app, dist):
    """Serves the built React app (frontend/dist) with client-side routing, if it has been built."""
    index = os.path.join(dist, "index.html")
    if not os.path.isfile(index):
        return
    assets = os.path.join(dist, "assets")
    if os.path.isdir(assets):
        app.mount("/assets", StaticFiles(directory=assets), name="assets")
    root = os.path.realpath(dist)

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        candidate = os.path.realpath(os.path.join(root, path))
        if path and candidate.startswith(root + os.sep) and os.path.isfile(candidate):
            return FileResponse(candidate)          # favicon, robots.txt, ...
        return FileResponse(index, headers={"Cache-Control": "no-cache"})
