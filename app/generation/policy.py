"""Deterministic reaction content — the "world model" defaults.

This is intentionally simple in v1: given a world action, produce plausible
content for the fields the manifest wants generated. It is fully deterministic
when a ``seed`` is supplied (via ``make_rng``), which is what keeps episodes
reproducible. The LLM generator (``llm.py``) layers on top; this also serves as
the fallback when the LLM is disabled or errors.

Extension point: a richer policy could decide *whether* and *how* a prospect
reacts (reply vs. decline vs. unsubscribe) from prior state; today the caller
names the action and this fills the text.
"""

from __future__ import annotations

import random
from typing import Any

# Canned options per action. Multiple options let a seeded RNG add variety
# across episodes without any nondeterminism within a seed.
#
# Keyed by bare action name (not (gym_name, action)): an intentional, accepted
# coupling for this deterministic fallback stub — the real LLM generation
# path (llm.py) is already gym-scoped via GenerationRequest.gym_name. Two
# gyms reusing the same action name would share a pool here, which is fine
# for a stub whose only job is plausible-looking placeholder text.
_TEMPLATES: dict[str, list[str]] = {
    "email.reply": [
        "Thanks for reaching out. This is interesting — could you share a bit more detail?",
        "Appreciate the note. We're evaluating options this quarter; let's set up a quick call.",
        "Thanks for the follow-up. Could you send over pricing and suggested next steps?",
    ],
    "email.out_of_office": [
        "I'm out of the office with limited email access and will reply on my return.",
    ],
    "email.unsubscribe": [
        "Please remove me from this list. Thanks.",
    ],
    "chat.reply": [
        "Thanks for the message! Happy to connect and learn more.",
        "Appreciate you reaching out — what did you have in mind?",
        "Thanks for connecting. Could you share how this would help our team?",
    ],
    "chat.send": [
        "Hi — following up on our last conversation, do you have a few minutes this week?",
        "Hi, wanted to reach out and see if this is still a priority for your team.",
        "Hello! Circling back — happy to share more details whenever works for you.",
    ],
    "chat.send_inmail": [
        "Hi — I came across your profile and thought I'd reach out directly. "
        "Would you be open to a quick call?",
        "Hello! I help teams like yours with this exact problem — happy to "
        "share more if it's useful.",
        "Hi there, reaching out via InMail since we're not yet connected. "
        "Worth a quick chat?",
    ],
    "notes.create": [
        "Followed up after the demo — seemed genuinely interested, wants pricing next.",
        "Mentioned budget is approved for this quarter; timing looks good.",
        "Key stakeholder — loop in before finalizing the proposal.",
    ],
    "notes.update": [
        "Updated after our latest call — moving to next stage, wants a formal proposal.",
        "Correcting earlier note: budget owner is actually someone else on the team.",
    ],
    "leads.connect": [
        "Hi, I'd love to connect and learn more about what you're working on.",
        "Came across your profile and would like to add you to my network.",
        "Hi — we work with similar companies in your space; would be great to connect.",
    ],
    "savedsearches.create": [
        "VPs of Sales — Enterprise SaaS",
        "Recently funded startups",
        "Decision makers in FinTech",
        "Q3 outbound target list",
    ],
    "meeting.decline": [
        "Thanks for the invite, but I'll have to pass for now. Let's revisit next quarter.",
    ],
    "meeting.accept": [
        "Sounds good — that time works for me. Looking forward to it.",
    ],
    "smartlinks.create": [
        "Product Overview Deck",
        "Case Study: Q3 Rollout",
        "Pricing & Packaging",
        "Onboarding Guide",
        "ROI Calculator",
    ],
}
# smartlinks.update edits the same kind of title as smartlinks.create — same pool.
_TEMPLATES["smartlinks.update"] = _TEMPLATES["smartlinks.create"]
# notes.create_account is the same kind of note text as notes.create — same pool.
_TEMPLATES["notes.create_account"] = _TEMPLATES["notes.create"]

_DEFAULT = ["Thanks for your message — I'll follow up shortly."]


def make_rng(seed: int | None) -> random.Random:
    """Deterministic RNG when ``seed`` is given; otherwise process-random."""
    return random.Random(seed)


def canned_text(action: str, field: str, params: dict[str, Any], rng: random.Random) -> str:
    """Produce deterministic content for ``field`` of ``action``.

    An action with its own pool (``_TEMPLATES[action]``, e.g.
    ``smartlinks.create``'s link titles) always uses it — checked first so a
    field named ``subject``/``title`` doesn't fall through to the reply-echo
    heuristic below just because of its name. Only an action *without* a pool
    falls back to that heuristic for those two field names (e.g. a
    hypothetical email-reply subject echoing the incoming subject as "Re:
    ..."), then to a generic default for anything else.
    """
    options = _TEMPLATES.get(action)
    if options is not None:
        return rng.choice(options)

    if field in ("subject", "title"):
        incoming = params.get("subject") or params.get("title") or "your message"
        base = str(incoming)
        return base if base.lower().startswith("re:") else f"Re: {base}"

    return rng.choice(_DEFAULT)
