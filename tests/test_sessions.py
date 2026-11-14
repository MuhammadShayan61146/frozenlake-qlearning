"""Session isolation, capacity and expiry (no web stack needed)."""
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from engine import TrainParams  # noqa: E402
from sessions import Busy, SessionStore  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def sid(n):
    return f"session-{n:02d}-abcdefghij"


def test_ids_are_validated_and_generated_safely():
    assert SessionStore.valid(SessionStore.new_id())
    assert not SessionStore.valid(None) and not SessionStore.valid("short") and not SessionStore.valid("../" * 10)
    with pytest.raises(KeyError):
        SessionStore().get("bad id")


def test_each_visitor_gets_a_private_engine():
    store = SessionStore()
    a, b = store.get(sid(1)), store.get(sid(2))
    assert a is not b and store.get(sid(1)) is a
    a.start_training(TrainParams(episodes=300, pace=0))
    end = time.time() + 20
    while json.loads(a.current())["status"] != "done" and time.time() < end:
        time.sleep(0.05)
    assert json.loads(a.current())["episode"] == 300
    assert json.loads(b.current())["episode"] == 0   # untouched by A's run


def test_capacity_is_enforced():
    store = SessionStore(max_sessions=2, max_training=1)
    store.get(sid(1)); store.get(sid(2))
    # both sessions are fresh and idle-eligible, so the idlest is recycled instead of refusing
    store.get(sid(3))
    assert len(store) == 2


def test_busy_when_every_session_is_training():
    store = SessionStore(max_sessions=1)
    eng = store.get(sid(1))
    eng.start_training(TrainParams(episodes=100_000, pace=500))
    try:
        with pytest.raises(Busy):
            store.get(sid(2))
        assert store.training_count() == 1
    finally:
        eng.stop_training()


def test_idle_sessions_expire_and_are_stopped():
    clock = Clock()
    store = SessionStore(ttl=60, clock=clock)
    old = store.get(sid(1))
    old.start_training(TrainParams(episodes=100_000, pace=500))
    clock.t += 120                      # idle past the TTL
    store.get(sid(2))                   # triggers the sweep
    assert len(store) == 1
    end = time.time() + 10
    while json.loads(old.current())["status"] == "training" and time.time() < end:
        time.sleep(0.05)
    assert json.loads(old.current())["status"] == "stopped"
