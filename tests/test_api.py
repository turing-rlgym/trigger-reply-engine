"""Introspection, validation, and dry-run behavior (no gym calls needed)."""

from __future__ import annotations


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_list_gyms(client):
    gyms = {g["name"]: g for g in client.get("/api/gyms").json()}
    assert set(gyms) == {"spothub", "sales_explorer"}
    assert "chat.reply" in gyms["sales_explorer"]["actions"]


def test_action_schema(client):
    schema = client.get("/api/gyms/sales_explorer/actions/chat.reply/schema").json()
    assert schema["generates"] == ["content"]
    assert schema["inputs"]["conversation_id"]["required"] is True
    assert schema["inputs"]["content"]["required"] is False


def test_unknown_gym_404(client):
    resp = client.post(
        "/api/trigger",
        json={"gym_name": "nope", "action": "chat.reply", "session_id": "s1", "params": {}},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "unknown_gym"


def test_unknown_action_404(client):
    resp = client.post(
        "/api/trigger",
        json={"gym_name": "sales_explorer", "action": "email.reply", "session_id": "s1"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "unknown_action"


def test_missing_required_params_422(client):
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "chat.reply",
            "session_id": "s1",
            "params": {},
        },
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "param_validation_error"
    joined = " ".join(body["error"]["details"])
    assert "conversation_id" in joined and "sender" in joined


def test_dry_run_builds_request_without_calling_gym(client):
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "chat.reply",
            "session_id": "sess-1",
            "params": {"conversation_id": "c1", "sender": "lead-9"},
            "dry_run": True,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    previews = body["request_previews"]
    assert len(previews) == 1
    preview = previews[0]
    assert preview["method"] == "POST"
    assert preview["url"] == "http://sales.test/api/conversations/c1/messages"
    assert preview["json"]["sender"] == "lead-9"
    # Generation is skipped on dry-run, so the generated body stays empty.
    assert preview["json"]["content"] == ""
    assert body["generated"] == {}
