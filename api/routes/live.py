# api/routes/live.py
"""WS /api/live: one socket per browser tab for everything that happens in real time.

Server -> client messages:
  {"type": "snapshot", "status": {...}, "recent": [...]}   once, on connect
  {"type": "state",    "session": {...}}                   session state changed
  {"type": "log",      "level", "alert_type", "message", "subject", "evidence_url", ...}
  {"type": "status",   ...}                                twice a second while a session runs
  {"type": "ping"}                                         keep-alive when idle
"""
import asyncio
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

from api import auth_store, session_store
from api.deps import origin_allowed

router = APIRouter()

STATUS_INTERVAL = 0.5
PING_INTERVAL = 15.0
RECENT_EVENTS = 50
CLOSE_UNAUTHORIZED = 4401
CLOSE_FORBIDDEN = 4403


async def _recent_events(session_id):
    if session_id is None:
        return []
    page = await run_in_threadpool(session_store.list_alerts, session_id, 10_000, 0, ["ALERT", "EVIDENCE"])
    items = page["items"][-RECENT_EVENTS:]
    return [dict(item, type="log", session_id=session_id) for item in items]


@router.websocket("/api/live")
async def live(ws: WebSocket):
    app = ws.app
    settings, manager, hub = app.state.settings, app.state.manager, app.state.hub

    if not origin_allowed(ws.headers.get("origin"), settings):
        await ws.close(code=CLOSE_FORBIDDEN)
        return
    user = await run_in_threadpool(auth_store.user_for_token, ws.cookies.get(settings.cookie_name))
    await ws.accept()
    if user is None:
        await ws.close(code=CLOSE_UNAUTHORIZED, reason="Not logged in.")
        return

    queue = hub.subscribe()
    # The client never needs to send anything; reading lets us notice a disconnect straight away.
    receiver = asyncio.create_task(_drain(ws))
    try:
        status = manager.status()
        await ws.send_json({"type": "snapshot", "status": status,
                            "recent": await _recent_events(manager.active_session_id())})
        last_status = last_ping = time.monotonic()
        while not receiver.done():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=STATUS_INTERVAL)
                await ws.send_json(event)
            except asyncio.TimeoutError:
                pass
            now = time.monotonic()
            if now - last_status >= STATUS_INTERVAL:
                last_status = now
                status = manager.status()
                if status["state"] == "running":
                    await ws.send_json({"type": "status", **status})
                    last_ping = now
            if now - last_ping >= PING_INTERVAL:
                last_ping = now
                await ws.send_json({"type": "ping"})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.unsubscribe(queue)
        receiver.cancel()


async def _drain(ws):
    try:
        while True:
            await ws.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        return
