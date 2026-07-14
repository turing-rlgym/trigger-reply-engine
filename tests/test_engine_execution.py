"""End-to-end trigger execution against a mocked gym (respx).

Asserts the three things that matter at the boundary: the session cookie is
attached, the generated reply is filled in, and the gym response/errors are
normalized into the uniform envelope.

Uses respx's ``respx_mock`` fixture (listed first so patching is active before
the app's HTTP client is built by the ``client`` fixture).
"""

from __future__ import annotations

import asyncio
import json

import httpx

from app.schemas import TriggerRequest

SALES_URL = "http://sales.test"
SPOTHUB_URL = "http://spothub.test"


def test_chat_reply_sends_cookie_and_generates_content(respx_mock, client):
    route = respx_mock.post(f"{SALES_URL}/api/conversations/c1/messages").mock(
        return_value=httpx.Response(200, json={"id": "m5", "content": "ok"})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "chat.reply",
            "session_id": "sess-abc",
            "params": {"conversation_id": "c1", "sender": "lead-9"},
            "seed": 1,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 1  # single-request action back-compat: still a list
    assert body["results"][0]["gym_status"] == 200
    assert body["results"][0]["body"]["id"] == "m5"
    assert body["generated"]["content"]  # non-empty (stub-filled)

    assert route.called
    sent = route.calls.last.request
    assert "user_id=sess-abc" in sent.headers.get("cookie", "")
    payload = json.loads(sent.content)
    assert payload["sender"] == "lead-9"
    assert payload["isRead"] is False
    assert payload["content"]  # the generated reply was sent through


def test_chat_send_posts_message_conversation_and_timeline(respx_mock, client):
    """chat.send mirrors the real UI's 3-call sequence and shares one
    correlation_id (as both the message id and the timeline's messageId) and
    one timestamp across all three steps."""
    message_route = respx_mock.post(f"{SALES_URL}/api/conversations/lead-1-1/messages").mock(
        return_value=httpx.Response(200, json={"id": "m-created"})
    )
    conversation_route = respx_mock.patch(f"{SALES_URL}/api/conversations/lead-1-1").mock(
        return_value=httpx.Response(200, json={"id": "lead-1-1"})
    )
    timeline_route = respx_mock.post(f"{SALES_URL}/api/timeline/lead-1-1").mock(
        return_value=httpx.Response(201, json={"id": "timeline-lead-1-1-message_sent-abc123"})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "chat.send",
            "session_id": "sess-send",
            "params": {"lead_id": "lead-1-1", "actor": "John Doe"},
            "seed": 4,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert [r["step"] for r in body["results"]] == ["message", "conversation", "timeline"]
    assert body["results"][2]["gym_status"] == 201
    assert body["generated"]["content"]  # non-empty (stub-filled)

    assert message_route.called and conversation_route.called and timeline_route.called
    for route in (message_route, conversation_route, timeline_route):
        assert "user_id=sess-send" in route.calls.last.request.headers.get("cookie", "")

    message_payload = json.loads(message_route.calls.last.request.content)
    timeline_payload = json.loads(timeline_route.calls.last.request.content)
    assert message_payload["content"]  # the generated content was sent through
    assert message_payload["isRead"] is True
    assert timeline_payload["type"] == "message_sent"
    assert timeline_payload["actor"] == "John Doe"
    assert timeline_payload["metadata"]["messagePreview"] == message_payload["content"]
    assert timeline_payload["metadata"]["isInMail"] is False  # a real bool, not the string "False"
    # message id and timeline's messageId share one correlation_id.
    assert message_payload["id"] == timeline_payload["metadata"]["messageId"]

    conversation_payload = json.loads(conversation_route.calls.last.request.content)
    assert conversation_payload == {
        "unreadCount": 0,
        "archived": False,
        "lastUpdated": message_payload["sentAt"],
    }


def test_chat_send_step_override_deep_merges_conversation_body(respx_mock, client):
    """params.steps.<name> lets a caller add fields to a step without the
    manifest needing to template every possible field."""
    respx_mock.post(f"{SALES_URL}/api/conversations/lead-1-1/messages").mock(
        return_value=httpx.Response(200, json={"id": "m-created"})
    )
    conversation_route = respx_mock.patch(f"{SALES_URL}/api/conversations/lead-1-1").mock(
        return_value=httpx.Response(200, json={"id": "lead-1-1"})
    )
    respx_mock.post(f"{SALES_URL}/api/timeline/lead-1-1").mock(
        return_value=httpx.Response(201, json={"id": "t1"})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "chat.send",
            "session_id": "sess-send",
            "params": {
                "lead_id": "lead-1-1",
                "steps": {"conversation": {"contactName": "Anna Rogers"}},
            },
        },
    )
    assert resp.status_code == 200
    conversation_payload = json.loads(conversation_route.calls.last.request.content)
    assert conversation_payload["contactName"] == "Anna Rogers"
    # The override is additive: manifest-templated fields survive the merge.
    assert conversation_payload["unreadCount"] == 0
    assert conversation_payload["archived"] is False


def test_chat_send_partial_failure_stops_and_reports_completed_step(respx_mock, client):
    """When a step fails, later steps must not run, and the error must name
    the failed step while listing the results of the steps that already
    succeeded (no rollback — matches the real UI's fire-and-forget calls)."""
    respx_mock.post(f"{SALES_URL}/api/conversations/lead-1-1/messages").mock(
        return_value=httpx.Response(200, json={"id": "m-created"})
    )
    respx_mock.patch(f"{SALES_URL}/api/conversations/lead-1-1").mock(
        return_value=httpx.Response(500, json={"message": "boom"})
    )
    timeline_route = respx_mock.post(f"{SALES_URL}/api/timeline/lead-1-1")

    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "chat.send",
            "session_id": "sess-send",
            "params": {"lead_id": "lead-1-1"},
        },
    )
    assert resp.status_code == 502
    err = resp.json()["error"]
    assert err["details"]["step"] == "conversation"
    assert err["details"]["step_index"] == 1
    completed = err["details"]["completed"]
    assert len(completed) == 1
    assert completed[0]["step"] == "message"
    assert not timeline_route.called


def test_chat_send_dry_run_returns_three_previews(respx_mock, client):
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "chat.send",
            "session_id": "sess-send",
            "params": {"lead_id": "lead-1-1"},
            "dry_run": True,
        },
    )
    assert resp.status_code == 200
    previews = resp.json()["request_previews"]
    assert [p["step"] for p in previews] == ["message", "conversation", "timeline"]


def test_email_reply_renders_properties(respx_mock, client):
    route = respx_mock.post(f"{SPOTHUB_URL}/crm/v3/objects/emails").mock(
        return_value=httpx.Response(201, json={"id": "e51", "properties": {}})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "spothub",
            "action": "email.reply",
            "session_id": "sess-1",
            "params": {"contact_id": "123"},
            "seed": 2,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["generated"]["description"]

    props = json.loads(route.calls.last.request.content)["properties"]
    assert props["contactId"] == "123"
    assert props["title"] == "Re: your email"
    assert props["description"]  # generated body
    assert props["contacted"] is True


def test_gym_error_is_normalized_to_502(respx_mock, client):
    respx_mock.post(f"{SALES_URL}/api/conversations/cX/messages").mock(
        return_value=httpx.Response(404, json={"message": "Unknown session"})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "chat.reply",
            "session_id": "expired",
            "params": {"conversation_id": "cX", "sender": "lead-1"},
        },
    )
    assert resp.status_code == 502
    err = resp.json()["error"]
    assert err["code"] == "gym_error"
    assert err["details"]["gym_status"] == 404


def test_idempotency_key_replays_without_recalling_gym(respx_mock, client):
    route = respx_mock.post(f"{SALES_URL}/api/conversations/c9/messages").mock(
        return_value=httpx.Response(200, json={"id": "m9"})
    )
    payload = {
        "gym_name": "sales_explorer",
        "action": "chat.reply",
        "session_id": "sess-idem",
        "params": {"conversation_id": "c9", "sender": "lead-2"},
        "idempotency_key": "abc-123",
        "seed": 3,
    }
    first = client.post("/api/trigger", json=payload).json()
    second = client.post("/api/trigger", json=payload).json()

    assert route.call_count == 1  # second call served from the idempotency cache
    assert first["trigger_id"] == second["trigger_id"]


def test_idempotency_key_reused_across_actions_does_not_cross_contaminate(respx_mock, client):
    """The same idempotency_key must not let one action's cached response leak
    into a different action (the cache key must be scoped per-action)."""
    reply_route = respx_mock.post(f"{SPOTHUB_URL}/crm/v3/objects/emails").mock(
        return_value=httpx.Response(201, json={"id": "e1"})
    )
    unsub_route = respx_mock.patch(f"{SPOTHUB_URL}/crm/v3/objects/contacts/c1").mock(
        return_value=httpx.Response(200, json={"id": "c1"})
    )
    shared_key = "shared-key-1"

    reply_resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "spothub",
            "action": "email.reply",
            "session_id": "sess-shared",
            "params": {"contact_id": "c1"},
            "idempotency_key": shared_key,
            "seed": 4,
        },
    )
    unsub_resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "spothub",
            "action": "email.unsubscribe",
            "session_id": "sess-shared",
            "params": {"contact_id": "c1"},
            "idempotency_key": shared_key,
        },
    )

    assert reply_resp.status_code == 200
    assert unsub_resp.status_code == 200
    assert reply_route.call_count == 1
    assert unsub_route.call_count == 1  # not short-circuited by email.reply's cached result
    assert reply_resp.json()["action"] == "email.reply"
    assert unsub_resp.json()["action"] == "email.unsubscribe"
    assert reply_resp.json()["trigger_id"] != unsub_resp.json()["trigger_id"]


def test_dry_run_is_not_cached_and_does_not_block_live_call(respx_mock, client):
    """A dry-run preview sharing an idempotency_key with a later live call (a
    natural preview-then-commit pattern) must not make the live call replay
    the side-effect-free dry-run response instead of actually hitting the gym.
    """
    route = respx_mock.post(f"{SALES_URL}/api/conversations/cD/messages").mock(
        return_value=httpx.Response(200, json={"id": "mD"})
    )
    base_payload = {
        "gym_name": "sales_explorer",
        "action": "chat.reply",
        "session_id": "sess-preview",
        "params": {"conversation_id": "cD", "sender": "lead-9"},
        "idempotency_key": "preview-then-commit",
    }

    preview = client.post("/api/trigger", json={**base_payload, "dry_run": True})
    live = client.post("/api/trigger", json=base_payload)

    assert preview.status_code == 200
    assert preview.json()["dry_run"] is True
    assert live.status_code == 200
    assert live.json()["dry_run"] is False
    assert route.call_count == 1  # the live call actually hit the gym


def test_post_read_timeout_is_not_retried(respx_mock, client):
    """A mid-flight timeout on a non-idempotent method must not be resent —
    the gym may have already committed the write; retrying risks a duplicate."""
    route = respx_mock.post(f"{SALES_URL}/api/conversations/cT/messages").mock(
        side_effect=httpx.ReadTimeout("simulated timeout")
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "chat.reply",
            "session_id": "sess-timeout",
            "params": {"conversation_id": "cT", "sender": "lead-9"},
        },
    )
    assert resp.status_code == 504
    assert resp.json()["error"]["code"] == "gym_unavailable"
    assert route.call_count == 1  # no retry attempted


async def test_concurrent_idempotent_requests_call_gym_once(respx_mock, client):
    """Two concurrent requests sharing an idempotency key must not race the gym."""
    route = respx_mock.post(f"{SALES_URL}/api/conversations/c7/messages").mock(
        return_value=httpx.Response(200, json={"id": "m7"})
    )
    req_kwargs = {
        "gym_name": "sales_explorer",
        "action": "chat.reply",
        "session_id": "sess-concurrent",
        "params": {"conversation_id": "c7", "sender": "lead-3"},
        "idempotency_key": "concurrent-1",
        "seed": 5,
    }
    engine = client.app.state.engine
    first, second = await asyncio.gather(
        engine.trigger(TriggerRequest(**req_kwargs)),
        engine.trigger(TriggerRequest(**req_kwargs)),
    )

    assert route.call_count == 1
    assert first.trigger_id == second.trigger_id
