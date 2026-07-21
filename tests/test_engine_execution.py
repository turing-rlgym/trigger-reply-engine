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
import re

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


# ── new Sales Explorer actions: read-modify-write (response-piping) ──────────


def test_leads_save_hydrates_then_puts_merged_set_and_excludes_hydrate_from_results(
    respx_mock, client
):
    """leads.save's hydrate GET is internal: piped into the save PUT body but
    excluded from the caller-facing results, even though it was called."""
    hydrate_route = respx_mock.get(f"{SALES_URL}/api/lead-lists").mock(
        return_value=httpx.Response(
            200,
            json={
                "leadLists": [],
                "leadListMapping": {},
                "allSavedLeads": [{"id": "lead-9-9", "name": "Bob"}],
            },
        )
    )
    save_route = respx_mock.put(f"{SALES_URL}/api/saved-leads").mock(
        return_value=httpx.Response(200, json={"allSavedLeads": []})
    )
    timeline_route = respx_mock.post(f"{SALES_URL}/api/timeline/lead-1-1").mock(
        return_value=httpx.Response(201, json={"id": "t1"})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "leads.save",
            "session_id": "sess-save",
            "params": {"lead_id": "lead-1-1", "lead_name": "Anna", "actor": "You"},
        },
    )
    assert resp.status_code == 200
    assert [r["step"] for r in resp.json()["results"]] == ["save", "timeline"]
    assert hydrate_route.called

    save_payload = json.loads(save_route.calls.last.request.content)
    assert save_payload["leads"] == [
        {"id": "lead-9-9", "name": "Bob"},
        {"id": "lead-1-1", "name": "Anna"},
    ]
    timeline_payload = json.loads(timeline_route.calls.last.request.content)
    assert timeline_payload["type"] == "lead_saved"
    assert timeline_payload["actor"] == "You"

    for route in (hydrate_route, save_route, timeline_route):
        assert "user_id=sess-save" in route.calls.last.request.headers.get("cookie", "")


def test_leads_save_is_idempotent_when_lead_already_saved(respx_mock, client):
    respx_mock.get(f"{SALES_URL}/api/lead-lists").mock(
        return_value=httpx.Response(
            200,
            json={
                "leadLists": [],
                "leadListMapping": {},
                "allSavedLeads": [{"id": "lead-1-1", "name": "Anna"}],
            },
        )
    )
    save_route = respx_mock.put(f"{SALES_URL}/api/saved-leads").mock(
        return_value=httpx.Response(200, json={"allSavedLeads": []})
    )
    respx_mock.post(f"{SALES_URL}/api/timeline/lead-1-1").mock(
        return_value=httpx.Response(201, json={"id": "t1"})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "leads.save",
            "session_id": "sess-save",
            "params": {"lead_id": "lead-1-1", "lead_name": "Anna"},
        },
    )
    assert resp.status_code == 200
    save_payload = json.loads(save_route.calls.last.request.content)
    assert save_payload["leads"] == [{"id": "lead-1-1", "name": "Anna"}]  # not duplicated


def test_leads_save_hydrate_failure_reports_step_and_empty_completed(respx_mock, client):
    """The internal hydrate step still surfaces by name on failure, even
    though a *successful* hydrate would never appear in results."""
    respx_mock.get(f"{SALES_URL}/api/lead-lists").mock(
        return_value=httpx.Response(500, json={"message": "boom"})
    )
    save_route = respx_mock.put(f"{SALES_URL}/api/saved-leads")
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "leads.save",
            "session_id": "sess-save",
            "params": {"lead_id": "lead-1-1"},
        },
    )
    assert resp.status_code == 502
    err = resp.json()["error"]
    assert err["details"]["step"] == "hydrate"
    # Nothing succeeded before the failing first step, so `completed` (only
    # ever set when non-empty — see UpstreamError) is absent entirely.
    assert "completed" not in err["details"]
    assert not save_route.called


def test_leads_save_dry_run_excludes_internal_hydrate_from_previews(respx_mock, client):
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "leads.save",
            "session_id": "sess-save",
            "params": {"lead_id": "lead-1-1", "lead_name": "Anna"},
            "dry_run": True,
        },
    )
    assert resp.status_code == 200
    previews = resp.json()["request_previews"]
    assert [p["step"] for p in previews] == ["save", "timeline"]
    # dry_run makes no calls, so the piped hydrate response is empty -> the
    # filter treats the prior set as empty (documented limitation).
    assert previews[0]["json"]["leads"] == [{"id": "lead-1-1", "name": "Anna"}]


def test_accounts_star_hydrates_then_puts_merged_starred_set(respx_mock, client):
    hydrate_route = respx_mock.get(f"{SALES_URL}/api/account-lists").mock(
        return_value=httpx.Response(
            200,
            json={
                "lists": [],
                "accountListMapping": {},
                "allSavedAccounts": [],
                "starredAccounts": [{"id": 9, "name": "Old Co"}],
                "crmStatusOverrides": {},
            },
        )
    )
    star_route = respx_mock.put(f"{SALES_URL}/api/starred-accounts").mock(
        return_value=httpx.Response(200, json={"starredAccounts": []})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "accounts.star",
            "session_id": "sess-star",
            "params": {"account_id": 42, "account_name": "Acme"},
        },
    )
    assert resp.status_code == 200
    assert [r["step"] for r in resp.json()["results"]] == ["star"]  # hydrate excluded
    assert hydrate_route.called
    star_payload = json.loads(star_route.calls.last.request.content)
    assert star_payload["accounts"] == [
        {"id": 9, "name": "Old Co"},
        {"id": 42, "name": "Acme"},
    ]


def test_accounts_set_crm_status_upserts_override_without_dropping_others(respx_mock, client):
    respx_mock.get(f"{SALES_URL}/api/account-lists").mock(
        return_value=httpx.Response(
            200,
            json={
                "lists": [],
                "accountListMapping": {},
                "allSavedAccounts": [],
                "starredAccounts": [],
                "crmStatusOverrides": {"3": "Lead"},
            },
        )
    )
    crm_route = respx_mock.put(f"{SALES_URL}/api/account-crm-overrides").mock(
        return_value=httpx.Response(200, json={"crmStatusOverrides": {}})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "accounts.set_crm_status",
            "session_id": "sess-crm",
            "params": {"account_id": 7, "status": "Customer"},
        },
    )
    assert resp.status_code == 200
    crm_payload = json.loads(crm_route.calls.last.request.content)
    # account_id (int, via `| string`) becomes a dict-key string; the
    # existing override for a different account survives the upsert.
    assert crm_payload["overrides"] == {"3": "Lead", "7": "Customer"}


# ── new Sales Explorer actions: simple (single-call, chat.send-style) writes ─


def test_alerts_bookmark_patches_is_bookmarked(respx_mock, client):
    route = respx_mock.patch(f"{SALES_URL}/api/alerts/5").mock(
        return_value=httpx.Response(200, json={"id": 5, "is_bookmarked": 1})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "alerts.bookmark",
            "session_id": "sess-alert",
            "params": {"alert_id": 5},
        },
    )
    assert resp.status_code == 200
    payload = json.loads(route.calls.last.request.content)
    assert payload == {"is_bookmarked": 1}  # default is_bookmarked=1, a real int


def test_alerts_dismiss_deletes(respx_mock, client):
    route = respx_mock.delete(f"{SALES_URL}/api/alerts/5").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "alerts.dismiss",
            "session_id": "sess-alert",
            "params": {"alert_id": 5},
        },
    )
    assert resp.status_code == 200
    assert route.called


def test_smartlinks_create_generates_title_and_posts(respx_mock, client):
    route = respx_mock.post(f"{SALES_URL}/api/smart-links").mock(
        return_value=httpx.Response(201, json={"id": "smartlink-1", "title": "x"})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "smartlinks.create",
            "session_id": "sess-link",
            "params": {"url": "https://acme.com/deck"},
            "seed": 7,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["generated"]["title"]  # non-empty (stub-filled)

    payload = json.loads(route.calls.last.request.content)
    assert payload["title"]  # the generated title was sent through
    assert payload["items"][0]["url"] == "https://acme.com/deck"
    assert payload["allowDownload"] is False  # a real bool, not the string "False"


def test_leads_view_records_view(respx_mock, client):
    route = respx_mock.post(f"{SALES_URL}/api/lead-ui-views/lead-1-1").mock(
        return_value=httpx.Response(200, json={"leadId": "lead-1-1", "viewCount": 1})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "leads.view",
            "session_id": "sess-view",
            "params": {"lead_id": "lead-1-1", "lead_name": "Anna"},
        },
    )
    assert resp.status_code == 200
    payload = json.loads(route.calls.last.request.content)
    assert payload["leadName"] == "Anna"
    assert payload["source"] == "panel"


def test_leads_unlock_spends_credit(respx_mock, client):
    route = respx_mock.post(f"{SALES_URL}/api/lead-unlocks/lead-1-1/unlock").mock(
        return_value=httpx.Response(
            200, json={"success": True, "creditsRemaining": 4, "unlockedLeads": []}
        )
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "leads.unlock",
            "session_id": "sess-unlock",
            "params": {"lead_id": "lead-1-1", "lead_name": "Anna"},
        },
    )
    assert resp.status_code == 200
    assert route.called
    payload = json.loads(route.calls.last.request.content)
    assert payload == {"leadName": "Anna"}



# ── new Sales Explorer actions: full action-coverage batch ───────────────────


def test_chat_send_inmail_pipes_hydrated_credits_and_decrements(respx_mock, client):
    respx_mock.get(f"{SALES_URL}/api/inbox").mock(
        return_value=httpx.Response(200, json={"inMailCredits": 7})
    )
    respx_mock.post(f"{SALES_URL}/api/conversations/lead-1-1/messages").mock(
        return_value=httpx.Response(200, json={"id": "m-created"})
    )
    respx_mock.patch(f"{SALES_URL}/api/conversations/lead-1-1").mock(
        return_value=httpx.Response(200, json={"id": "lead-1-1"})
    )
    credits_route = respx_mock.patch(f"{SALES_URL}/api/inbox/metadata").mock(
        return_value=httpx.Response(200, json={"inMailCredits": 6})
    )
    respx_mock.post(f"{SALES_URL}/api/timeline/lead-1-1").mock(
        return_value=httpx.Response(201, json={"id": "t1"})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "chat.send_inmail",
            "session_id": "sess-inmail",
            "params": {"lead_id": "lead-1-1"},
        },
    )
    assert resp.status_code == 200
    # hydrate is internal: executed, but excluded from caller-facing results.
    assert [r["step"] for r in resp.json()["results"]] == [
        "message",
        "conversation",
        "credits",
        "timeline",
    ]
    credits_payload = json.loads(credits_route.calls.last.request.content)
    assert credits_payload == {"inMailCredits": 6}


def test_chat_send_inmail_defaults_credits_to_50_when_hydrate_omits_it(respx_mock, client):
    respx_mock.get(f"{SALES_URL}/api/inbox").mock(return_value=httpx.Response(200, json={}))
    respx_mock.post(f"{SALES_URL}/api/conversations/lead-1-1/messages").mock(
        return_value=httpx.Response(200, json={"id": "m-created"})
    )
    respx_mock.patch(f"{SALES_URL}/api/conversations/lead-1-1").mock(
        return_value=httpx.Response(200, json={"id": "lead-1-1"})
    )
    credits_route = respx_mock.patch(f"{SALES_URL}/api/inbox/metadata").mock(
        return_value=httpx.Response(200, json={"inMailCredits": 49})
    )
    respx_mock.post(f"{SALES_URL}/api/timeline/lead-1-1").mock(
        return_value=httpx.Response(201, json={"id": "t1"})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "chat.send_inmail",
            "session_id": "sess-inmail",
            "params": {"lead_id": "lead-1-1"},
        },
    )
    assert resp.status_code == 200
    assert json.loads(credits_route.calls.last.request.content) == {"inMailCredits": 49}


def test_chat_send_inmail_floors_credits_at_zero_instead_of_going_negative(respx_mock, client):
    respx_mock.get(f"{SALES_URL}/api/inbox").mock(
        return_value=httpx.Response(200, json={"inMailCredits": 0})
    )
    respx_mock.post(f"{SALES_URL}/api/conversations/lead-1-1/messages").mock(
        return_value=httpx.Response(200, json={"id": "m-created"})
    )
    respx_mock.patch(f"{SALES_URL}/api/conversations/lead-1-1").mock(
        return_value=httpx.Response(200, json={"id": "lead-1-1"})
    )
    credits_route = respx_mock.patch(f"{SALES_URL}/api/inbox/metadata").mock(
        return_value=httpx.Response(200, json={"inMailCredits": 0})
    )
    respx_mock.post(f"{SALES_URL}/api/timeline/lead-1-1").mock(
        return_value=httpx.Response(201, json={"id": "t1"})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "chat.send_inmail",
            "session_id": "sess-inmail",
            "params": {"lead_id": "lead-1-1"},
        },
    )
    assert resp.status_code == 200
    # floored at 0 (would be -1 without the max(...) guard)
    assert json.loads(credits_route.calls.last.request.content) == {"inMailCredits": 0}


def test_chat_mark_read_and_unread(respx_mock, client):
    read_route = respx_mock.patch(f"{SALES_URL}/api/conversations/lead-2-1").mock(
        return_value=httpx.Response(200, json={"id": "lead-2-1"})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "chat.mark_read",
            "session_id": "sess-read",
            "params": {"lead_id": "lead-2-1"},
        },
    )
    assert resp.status_code == 200
    assert json.loads(read_route.calls.last.request.content) == {"unreadCount": 0}

    unread_route = respx_mock.patch(f"{SALES_URL}/api/conversations/lead-2-1").mock(
        return_value=httpx.Response(200, json={"id": "lead-2-1"})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "chat.mark_unread",
            "session_id": "sess-read",
            "params": {"lead_id": "lead-2-1"},
        },
    )
    assert resp.status_code == 200
    assert json.loads(unread_route.calls.last.request.content) == {"unreadCount": 1}


def test_chat_add_contact_upserts_messaging_lead(respx_mock, client):
    route = respx_mock.put(f"{SALES_URL}/api/messaging-leads/lead-1-1").mock(
        return_value=httpx.Response(200, json={"ok": True, "leadId": "lead-1-1"})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "chat.add_contact",
            "session_id": "sess-contact",
            "params": {
                "lead_id": "lead-1-1",
                "lead_name": "Anna Rogers",
                "email": "anna@acme.com",
                "first_name": "Anna",
                "last_name": "Rogers",
            },
        },
    )
    assert resp.status_code == 200
    assert route.called
    assert "user_id=sess-contact" in route.calls.last.request.headers.get("cookie", "")

    payload = json.loads(route.calls.last.request.content)
    assert payload["name"] == "Anna Rogers"
    assert payload["archived"] is False  # a real bool, not the string "False"
    assert payload["personalDetails"]["firstName"] == "Anna"
    assert payload["personalDetails"]["lastName"] == "Rogers"


def test_leads_connect_generates_personal_message(respx_mock, client):
    route = respx_mock.post(f"{SALES_URL}/api/lead-connections").mock(
        return_value=httpx.Response(201, json={"leadId": "lead-1-1", "status": "pending"})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "leads.connect",
            "session_id": "sess-connect",
            "params": {"lead_id": "lead-1-1", "lead_name": "Anna"},
            "seed": 9,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["generated"]["personal_message"]
    payload = json.loads(route.calls.last.request.content)
    assert payload["status"] == "pending"
    assert payload["personalMessage"]


def test_notes_create_pipes_server_assigned_note_id_into_timeline(respx_mock, client):
    respx_mock.post(f"{SALES_URL}/api/notes/lead-1-1").mock(
        return_value=httpx.Response(201, json={"id": "note-99", "content": "hi"})
    )
    timeline_route = respx_mock.post(f"{SALES_URL}/api/timeline/lead-1-1").mock(
        return_value=httpx.Response(201, json={"id": "t1"})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "notes.create",
            "session_id": "sess-note",
            "params": {"lead_id": "lead-1-1", "content": "Great call today."},
        },
    )
    assert resp.status_code == 200
    timeline_payload = json.loads(timeline_route.calls.last.request.content)
    assert timeline_payload["type"] == "note_added"
    assert timeline_payload["metadata"]["noteId"] == "note-99"
    assert timeline_payload["metadata"]["notePreview"] == "Great call today."


def test_notes_create_account_does_not_log_timeline(respx_mock, client):
    note_route = respx_mock.post(f"{SALES_URL}/api/notes/account-42").mock(
        return_value=httpx.Response(201, json={"id": "note-100", "content": "hi"})
    )
    # Registered so the test would fail loudly (unexpected-call error) if
    # notes.create_account ever grew a timeline step — not because one
    # should exist for accounts (it shouldn't; see the action's description).
    timeline_route = respx_mock.post(f"{SALES_URL}/api/timeline/account-42").mock(
        return_value=httpx.Response(201, json={"id": "t1"})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "notes.create_account",
            "session_id": "sess-note",
            "params": {"account_id": 42, "content": "Great call today."},
        },
    )
    assert resp.status_code == 200
    assert note_route.called
    assert not timeline_route.called
    body = resp.json()
    assert len(body["results"]) == 1
    payload = json.loads(note_route.calls.last.request.content)
    assert payload["content"] == "Great call today."
    assert payload["isPrivate"] is False
    assert payload["entityName"] == ""


def test_savedsearches_create_generates_name(respx_mock, client):
    route = respx_mock.post(f"{SALES_URL}/api/saved-searches").mock(
        return_value=httpx.Response(201, json={"id": "search-1"})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "savedsearches.create",
            "session_id": "sess-search",
            "params": {},
            "seed": 2,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["generated"]["name"]
    payload = json.loads(route.calls.last.request.content)
    assert payload["type"] == "Lead"
    assert payload["filters"] == {}


def test_smartlinks_view_records_duration(respx_mock, client):
    route = respx_mock.post(f"{SALES_URL}/api/smart-links/link-1/view").mock(
        return_value=httpx.Response(200, json={"id": "link-1", "viewCount": 2})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "smartlinks.view",
            "session_id": "sess-link",
            "params": {"link_id": "link-1", "duration_seconds": 45},
        },
    )
    assert resp.status_code == 200
    assert json.loads(route.calls.last.request.content) == {"durationSeconds": 45}


def test_leads_set_account_sends_null_for_omitted_original_account(respx_mock, client):
    route = respx_mock.post(f"{SALES_URL}/api/lead-account-overrides/lead-1-1").mock(
        return_value=httpx.Response(200, json={"leadId": "lead-1-1"})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "leads.set_account",
            "session_id": "sess-override",
            "params": {"lead_id": "lead-1-1", "account_id": 5, "account_name": "Acme"},
        },
    )
    assert resp.status_code == 200
    payload = json.loads(route.calls.last.request.content)
    assert payload["currentAccountId"] == 5
    assert payload["originalAccountId"] is None  # real JSON null, not ""
    assert payload["originalAccountName"] is None


def test_company_visit_records_iso_timestamp(respx_mock, client):
    route = respx_mock.post(f"{SALES_URL}/api/company-visits/acct-1").mock(
        return_value=httpx.Response(200, json={"companyId": "acct-1", "visitCount": 1})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "company.visit",
            "session_id": "sess-visit",
            "params": {"company_id": "acct-1", "company_name": "Acme"},
        },
    )
    assert resp.status_code == 200
    payload = json.loads(route.calls.last.request.content)
    assert payload["companyName"] == "Acme"
    assert payload["lastVisitDate"]


def test_leads_unsave_rejects_lead_and_logs_timeline(respx_mock, client):
    respx_mock.get(f"{SALES_URL}/api/lead-lists").mock(
        return_value=httpx.Response(
            200,
            json={
                "leadLists": [],
                "leadListMapping": {},
                "allSavedLeads": [
                    {"id": "lead-1-1", "name": "Anna"},
                    {"id": "lead-9-9", "name": "Bob"},
                ],
            },
        )
    )
    save_route = respx_mock.put(f"{SALES_URL}/api/saved-leads").mock(
        return_value=httpx.Response(200, json={"allSavedLeads": []})
    )
    timeline_route = respx_mock.post(f"{SALES_URL}/api/timeline/lead-1-1").mock(
        return_value=httpx.Response(201, json={"id": "t1"})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "leads.unsave",
            "session_id": "sess-unsave",
            "params": {"lead_id": "lead-1-1"},
        },
    )
    assert resp.status_code == 200
    assert [r["step"] for r in resp.json()["results"]] == ["save", "timeline"]  # hydrate excluded
    save_payload = json.loads(save_route.calls.last.request.content)
    assert save_payload["leads"] == [{"id": "lead-9-9", "name": "Bob"}]
    timeline_payload = json.loads(timeline_route.calls.last.request.content)
    assert timeline_payload["type"] == "lead_unsaved"


def test_leads_remove_from_list_rejects_lead_from_members(respx_mock, client):
    respx_mock.get(f"{SALES_URL}/api/lead-lists").mock(
        return_value=httpx.Response(
            200,
            json={
                "leadLists": [],
                "leadListMapping": {"list-1": [{"id": "lead-1-1", "name": "Anna"}]},
                "allSavedLeads": [],
            },
        )
    )
    members_route = respx_mock.put(f"{SALES_URL}/api/lead-lists/list-1/members").mock(
        return_value=httpx.Response(200, json={"listId": "list-1", "leadCount": 0})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "leads.remove_from_list",
            "session_id": "sess-remove",
            "params": {"list_id": "list-1", "lead_id": "lead-1-1"},
        },
    )
    assert resp.status_code == 200
    assert json.loads(members_route.calls.last.request.content)["leads"] == []


def test_accounts_unstar_rejects_account(respx_mock, client):
    respx_mock.get(f"{SALES_URL}/api/account-lists").mock(
        return_value=httpx.Response(
            200,
            json={
                "lists": [],
                "accountListMapping": {},
                "allSavedAccounts": [],
                "starredAccounts": [{"id": 42, "name": "Acme"}, {"id": 7, "name": "Zed"}],
                "crmStatusOverrides": {},
            },
        )
    )
    unstar_route = respx_mock.put(f"{SALES_URL}/api/starred-accounts").mock(
        return_value=httpx.Response(200, json={"starredAccounts": []})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "accounts.unstar",
            "session_id": "sess-unstar",
            "params": {"account_id": 42},
        },
    )
    assert resp.status_code == 200
    assert json.loads(unstar_route.calls.last.request.content)["accounts"] == [
        {"id": 7, "name": "Zed"}
    ]


def test_leads_copy_list_creates_list_and_copies_members(respx_mock, client):
    respx_mock.get(f"{SALES_URL}/api/lead-lists").mock(
        return_value=httpx.Response(
            200,
            json={
                "leadLists": [],
                "leadListMapping": {"list-1": [{"id": "lead-1-1", "name": "Anna"}]},
                "allSavedLeads": [],
            },
        )
    )
    create_route = respx_mock.post(f"{SALES_URL}/api/lead-lists").mock(
        return_value=httpx.Response(201, json={"id": "new-list"})
    )
    members_route = respx_mock.put(re.compile(rf"{SALES_URL}/api/lead-lists/.+/members")).mock(
        return_value=httpx.Response(200, json={"leadCount": 1})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "leads.copy_list",
            "session_id": "sess-copy",
            "params": {"source_list_id": "list-1", "list_name": "Copy of list"},
        },
    )
    assert resp.status_code == 200
    create_payload = json.loads(create_route.calls.last.request.content)
    assert create_payload["name"] == "Copy of list"
    members_payload = json.loads(members_route.calls.last.request.content)
    assert members_payload["leads"] == [{"id": "lead-1-1", "name": "Anna"}]
    # The new list's id (from `create`'s correlation_id) addresses the members call.
    assert create_payload["id"] in members_route.calls.last.request.url.path


def test_settings_set_app_notifications_overrides_one_category_and_hydrates_others(
    respx_mock, client
):
    """Only the leads category is supplied; the other two must come from the
    hydrated current state rather than being wiped to []."""
    respx_mock.get(f"{SALES_URL}/api/user-state/settings-app-notifications").mock(
        return_value=httpx.Response(
            200,
            json={
                "savedLeadsNotifications": [{"id": "l1", "enabled": True}],
                "savedAccountsNotifications": [{"id": "a1", "enabled": True}],
                "sharedActivityNotifications": [{"id": "s1", "enabled": False}],
            },
        )
    )
    put_route = respx_mock.put(f"{SALES_URL}/api/user-state/settings-app-notifications").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "settings.set_app_notifications",
            "session_id": "sess-settings",
            "params": {"saved_leads_notifications": [{"id": "l1", "enabled": False}]},
        },
    )
    assert resp.status_code == 200
    payload = json.loads(put_route.calls.last.request.content)
    assert payload["savedLeadsNotifications"] == [{"id": "l1", "enabled": False}]
    assert payload["savedAccountsNotifications"] == [{"id": "a1", "enabled": True}]
    assert payload["sharedActivityNotifications"] == [{"id": "s1", "enabled": False}]


def test_settings_set_privacy_defaults_when_omitted(respx_mock, client):
    route = respx_mock.put(f"{SALES_URL}/api/user-state/settings-privacy").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    resp = client.post(
        "/api/trigger",
        json={
            "gym_name": "sales_explorer",
            "action": "settings.set_privacy",
            "session_id": "sess-settings",
            "params": {},
        },
    )
    assert resp.status_code == 200
    assert json.loads(route.calls.last.request.content) == {
        "profileViewingOption": "name-headline",
        "linkedInActivitiesEnabled": True,
    }


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
