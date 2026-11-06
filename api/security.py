# api/security.py
"""Password hashing, session tokens and login throttling (standard library only)."""
import base64
import hashlib
import hmac
import secrets
import threading
import time
from collections import defaultdict, deque

_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1}
MIN_PASSWORD_LENGTH = 8


def _b64(data):
    return base64.b64encode(data).decode("ascii")


def hash_password(password):
    """scrypt with a random salt, stored as scrypt$n$r$p$salt$hash."""
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, dklen=32, **_SCRYPT)
    return f"scrypt${_SCRYPT['n']}${_SCRYPT['r']}${_SCRYPT['p']}${_b64(salt)}${_b64(dk)}"


def verify_password(password, stored):
    try:
        algo, n, r, p, salt_b64, hash_b64 = stored.split("$")
        if algo != "scrypt":
            return False
        expected = base64.b64decode(hash_b64)
        dk = hashlib.scrypt(password.encode("utf-8"), salt=base64.b64decode(salt_b64),
                            n=int(n), r=int(r), p=int(p), dklen=len(expected))
        return hmac.compare_digest(dk, expected)
    except (ValueError, TypeError):
        return False


# A fixed hash to check against when the username doesn't exist, so a login attempt
# takes the same time whether or not the account exists.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


def verify_password_or_dummy(password, stored):
    if stored is None:
        verify_password(password, _DUMMY_HASH)
        return False
    return verify_password(password, stored)


def new_session_token():
    return secrets.token_urlsafe(32)


def token_digest(token):
    """Only the SHA-256 of a session token is stored, so a leaked database can't be used to log in."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class LoginThrottle:
    """Locks a username + client address after repeated failed logins."""

    def __init__(self, max_failures=5, window_seconds=60):
        self.max_failures = max_failures
        self.window = window_seconds
        self._failures = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, key, now):
        q = self._failures[key]
        while q and now - q[0] > self.window:
            q.popleft()
        return q

    def retry_after(self, key):
        """Seconds until this key may try again (0 if not locked)."""
        now = time.monotonic()
        with self._lock:
            q = self._prune(key, now)
            if len(q) < self.max_failures:
                return 0
            return max(1, int(self.window - (now - q[0])) + 1)

    def record_failure(self, key):
        with self._lock:
            self._failures[key].append(time.monotonic())

    def reset(self, key):
        with self._lock:
            self._failures.pop(key, None)
