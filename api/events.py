# api/events.py
"""Bridges events from the engine's threads to WebSocket clients on the asyncio loop."""
import asyncio
import os
import threading


def evidence_url(path):
    """Public URL of an evidence screenshot (only its file name is exposed, never the disk path)."""
    if not path:
        return None
    return f"/api/evidence/{os.path.basename(str(path).replace(chr(92), '/'))}"


def to_client_event(event):
    """Makes an engine/session event safe to send to the browser."""
    if event.get("type") != "log":
        return event
    out = {k: v for k, v in event.items() if k != "evidence_file"}
    out["evidence_url"] = evidence_url(event.get("evidence_file"))
    return out


class EventHub:
    """Fan-out of events to any number of subscribers.

    publish() may be called from any thread. Each subscriber gets its own bounded
    queue; a slow client loses its oldest events rather than holding up the engine.
    """

    def __init__(self, max_queue=500):
        self.max_queue = max_queue
        self._loop = None
        self._subscribers = set()
        self._lock = threading.Lock()

    def bind(self, loop):
        self._loop = loop

    def subscribe(self):
        q = asyncio.Queue(maxsize=self.max_queue)
        with self._lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            self._subscribers.discard(q)

    @property
    def subscriber_count(self):
        with self._lock:
            return len(self._subscribers)

    def publish(self, event):
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        event = to_client_event(event)
        try:
            loop.call_soon_threadsafe(self._fanout, event)
        except RuntimeError:        # loop shut down between the check and the call
            pass

    def _fanout(self, event):
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(event)
