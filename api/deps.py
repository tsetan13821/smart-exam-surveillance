# api/deps.py
"""Shared request dependencies: the logged-in proctor and app-wide services."""
from fastapi import HTTPException, Request, status

from api import auth_store


def get_settings(request: Request):
    return request.app.state.settings


def get_manager(request: Request):
    return request.app.state.manager


def current_user(request: Request):
    """The logged-in proctor; 401 if the session cookie is missing, expired or revoked.

    Declared as a plain function so FastAPI runs the SQLite lookup in its thread pool.
    """
    token = request.cookies.get(request.app.state.settings.cookie_name)
    user = auth_store.user_for_token(token)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not logged in.")
    return user


def origin_allowed(origin, settings):
    """Requests without an Origin (curl, scripts) rely on the SameSite cookie; browsers always send one."""
    return origin is None or origin.rstrip("/") in settings.allowed_origins
