"""
sessions.py

One private Engine (agent + training run) per visitor, so a public deployment
never mixes two people's experiments. Framework-free and unit-tested.

Limits are tunable with environment variables:
    FLL_MAX_SESSIONS   concurrent visitors kept in memory      (default 100)
    FLL_MAX_TRAINING   training jobs allowed at the same time  (default 3)
    FLL_SESSION_TTL    idle seconds before a session is freed  (default 1800)
"""

import os
import re
import secrets
import threading
import time

from engine import Engine

_SID = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


class Busy(Exception):
    """The server is at capacity."""


class SessionStore:
    def __init__(self, max_sessions=None, ttl=None, max_training=None, clock=time.monotonic):
        env = os.environ.get
        self.max_sessions = int(max_sessions if max_sessions is not None else env("FLL_MAX_SESSIONS", 100))
        self.ttl = float(ttl if ttl is not None else env("FLL_SESSION_TTL", 1800))
        self.max_training = int(max_training if max_training is not None else env("FLL_MAX_TRAINING", 3))
        self._clock = clock
        self._lock = threading.Lock()
        self._items = {}          # sid -> [Engine, last_seen]
        self._last_sweep = clock()

    @staticmethod
    def new_id():
        return secrets.token_urlsafe(18)

    @staticmethod
    def valid(sid):
        return bool(sid) and bool(_SID.match(sid))

    def __len__(self):
        return len(self._items)

    def get(self, sid):
        """Return the visitor's engine, creating it on first use."""
        if not self.valid(sid):
            raise KeyError("invalid session id")
        now = self._clock()
        victims = []
        try:
            with self._lock:
                if now - self._last_sweep >= 30:
                    self._last_sweep = now
                    victims = self._pop_expired(now)
                item = self._items.get(sid)
                if item is None:
                    if len(self._items) >= self.max_sessions:
                        victims += self._pop_expired(now, ttl=0)  # last resort: drop the idlest
                        if len(self._items) >= self.max_sessions:
                            raise Busy("The lab is full right now. Please try again in a few minutes.")
                    item = self._items[sid] = [Engine(), now]
                item[1] = now
                return item[0]
        finally:
            for engine in victims:
                engine.shutdown()

    def touch(self, sid):
        with self._lock:
            item = self._items.get(sid)
            if item:
                item[1] = self._clock()

    def training_count(self):
        with self._lock:
            return sum(1 for engine, _ in self._items.values() if engine.is_training)

    def _pop_expired(self, now, ttl=None):
        """Remove idle sessions. ttl=0 means: free the single longest-idle, non-training one."""
        if ttl == 0:
            idle = [(seen, sid) for sid, (eng, seen) in self._items.items() if not eng.is_training]
            if not idle:
                return []
            _, sid = min(idle)
            return [self._items.pop(sid)[0]]
        limit = self.ttl if ttl is None else ttl
        dead = [sid for sid, (_, seen) in self._items.items() if now - seen > limit]
        return [self._items.pop(sid)[0] for sid in dead]
