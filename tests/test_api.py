"""HTTP-level tests. Need the web stack: pip install -r requirements-dev.txt"""
import os
import sys

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient  # noqa: E402

from backend import app  # noqa: E402

client = TestClient(app)
client.get("/")  # visiting the page issues the session cookie


def test_config_and_state():
    assert client.get("/api/config").json()["maps"] == ["4x4", "8x8"]
    assert client.get("/api/state").json()["status"] in {"idle", "training", "done", "stopped"}


def test_validation_rejects_bad_hyperparameters():
    assert client.post("/api/train", json={"alpha": 5}).status_code == 422
    assert client.post("/api/train", json={"map_name": "12x12"}).status_code == 422


def test_play_flow():
    client.post("/api/reset", json={"slippery": False})
    assert client.post("/api/play/reset").json()["state"] == 0
    step = client.post("/api/play/step", json={"action": 2}).json()
    assert step["state"] == 1 and step["slipped"] is False


def test_dashboard_is_served():
    assert "Frozen Lake Lab" in client.get("/").text


def test_requests_without_a_session_are_rejected():
    fresh = TestClient(app)
    assert fresh.post("/api/play/reset").status_code == 401
