"""``X-API-Key`` auth: a no-op by default, enforced once ``Settings.api_key`` is set.

Builds a dedicated app instance (rather than reusing the shared ``client``
fixture) so ``API_KEY`` can be toggled without affecting other tests, which
rely on auth staying disabled.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app


@pytest.fixture
def keyed_client():
    os.environ["API_KEY"] = "secret-123"
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as test_client:
            yield test_client
    finally:
        os.environ.pop("API_KEY", None)
        get_settings.cache_clear()


def test_trigger_without_key_is_401(keyed_client):
    resp = keyed_client.post(
        "/api/trigger",
        json={"gym_name": "spothub", "action": "email.reply", "session_id": "s1", "params": {}},
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_trigger_with_wrong_key_is_401(keyed_client):
    resp = keyed_client.post(
        "/api/trigger",
        json={"gym_name": "spothub", "action": "email.reply", "session_id": "s1", "params": {}},
        headers={"X-API-Key": "nope"},
    )
    assert resp.status_code == 401


def test_trigger_with_correct_key_passes_auth(keyed_client):
    resp = keyed_client.post(
        "/api/trigger",
        json={
            "gym_name": "spothub",
            "action": "email.reply",
            "session_id": "s1",
            "params": {"contact_id": "1"},
            "dry_run": True,  # avoids needing a gym mock; just proves auth passed
        },
        headers={"X-API-Key": "secret-123"},
    )
    assert resp.status_code == 200


def test_gyms_introspection_also_requires_key(keyed_client):
    assert keyed_client.get("/api/gyms").status_code == 401
    assert keyed_client.get("/api/gyms", headers={"X-API-Key": "secret-123"}).status_code == 200


def test_health_never_requires_key(keyed_client):
    assert keyed_client.get("/api/health").status_code == 200
