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
    "meeting.decline": [
        "Thanks for the invite, but I'll have to pass for now. Let's revisit next quarter.",
    ],
    "meeting.accept": [
        "Sounds good — that time works for me. Looking forward to it.",
    ],
}

_DEFAULT = ["Thanks for your message — I'll follow up shortly."]


def make_rng(seed: int | None) -> random.Random:
    """Deterministic RNG when ``seed`` is given; otherwise process-random."""
    return random.Random(seed)


def canned_text(action: str, field: str, params: dict[str, Any], rng: random.Random) -> str:
    """Produce deterministic content for ``field`` of ``action``.

    A couple of fields are special-cased (e.g. an email ``subject`` echoes the
    incoming subject as a reply); everything else picks from the action's pool.
    """
    if field in ("subject", "title"):
        incoming = params.get("subject") or params.get("title") or "your message"
        base = str(incoming)
        return base if base.lower().startswith("re:") else f"Re: {base}"

    options = _TEMPLATES.get(action, _DEFAULT)
    return rng.choice(options)
