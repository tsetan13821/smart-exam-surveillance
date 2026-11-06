# api/auth_store.py
"""Proctor accounts and login sessions in SQLite."""
import re
import sqlite3
from datetime import datetime, timedelta

import config
from api.security import MIN_PASSWORD_LENGTH, hash_password, new_session_token, token_digest

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS auth_sessions (
    token_hash TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users (user_id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_user ON auth_sessions (user_id);
"""

USERNAME_RE = re.compile(r"[A-Za-z0-9_.-]{3,32}")


def connect():
    conn = sqlite3.connect(config.DATABASE_FILE, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_schema():
    with connect() as conn:
        conn.executescript(SCHEMA)


def create_user(username, password):
    if not USERNAME_RE.fullmatch(username or ""):
        raise ValueError("Username must be 3-32 letters, digits, '.', '_' or '-'.")
    if len(password or "") < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    try:
        with connect() as conn:
            cur = conn.execute("INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                               (username, hash_password(password), datetime.now().isoformat()))
            return cur.lastrowid
    except sqlite3.IntegrityError:
        raise ValueError(f"User '{username}' already exists.")


def set_password(username, password):
    if len(password or "") < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    with connect() as conn:
        user = conn.execute("SELECT user_id FROM users WHERE username = ?", (username,)).fetchone()
        if user is None:
            return False
        conn.execute("UPDATE users SET password_hash = ? WHERE user_id = ?",
                     (hash_password(password), user["user_id"]))
        conn.execute("DELETE FROM auth_sessions WHERE user_id = ?", (user["user_id"],))   # log out everywhere
        return True


def get_user_by_username(username):
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    return dict(row) if row else None


def count_users():
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def create_login_session(user_id, hours):
    token = new_session_token()
    now = datetime.now()
    with connect() as conn:
        conn.execute("DELETE FROM auth_sessions WHERE expires_at < ?", (now.isoformat(),))
        conn.execute("INSERT INTO auth_sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                     (token_digest(token), user_id, now.isoformat(), (now + timedelta(hours=hours)).isoformat()))
    return token


def user_for_token(token):
    """The logged-in user for a session token, or None if missing/expired/revoked."""
    if not token:
        return None
    with connect() as conn:
        row = conn.execute(
            "SELECT u.user_id, u.username, s.expires_at FROM auth_sessions s JOIN users u USING (user_id) "
            "WHERE s.token_hash = ?", (token_digest(token),)).fetchone()
    if row is None or row["expires_at"] < datetime.now().isoformat():
        return None
    return {"user_id": row["user_id"], "username": row["username"]}


def delete_login_session(token):
    if token:
        with connect() as conn:
            conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (token_digest(token),))
