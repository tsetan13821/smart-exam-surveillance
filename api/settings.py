# api/settings.py
import os
from dataclasses import dataclass, field

import config


def _default_origins():
    # The API serving the built frontend (on whatever port `python -m api` uses) and the Vite dev server
    ports = {8000, 5173, int(os.environ.get("SMARTSQUAD_PORT", "8000"))}
    return [f"http://{host}:{port}" for port in sorted(ports) for host in ("localhost", "127.0.0.1")]


def _env_list(name, default):
    raw = os.environ.get(name)
    return [v.strip().rstrip("/") for v in raw.split(",") if v.strip()] if raw else default()


@dataclass
class Settings:
    # Browser origins allowed to make state-changing requests and open the live socket
    allowed_origins: list = field(default_factory=lambda: _env_list("SMARTSQUAD_ALLOWED_ORIGINS", _default_origins))
    cookie_name: str = "ss_session"
    # Set SMARTSQUAD_COOKIE_SECURE=1 when served over HTTPS
    cookie_secure: bool = field(default_factory=lambda: os.environ.get("SMARTSQUAD_COOKIE_SECURE") == "1")
    session_hours: float = 12.0
    max_login_failures: int = 5
    login_lockout_seconds: int = 60
    demo_video_dir: str = field(default_factory=lambda: config.DEMO_VIDEO_DIR)
    frontend_dist: str = field(default_factory=lambda: os.path.join(config.BASE_DIR, "frontend", "dist"))
