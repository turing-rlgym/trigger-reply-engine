"""Session binding + resilient execution of a built request.

Session binding is the crux of correctness: every gym isolates state by a
session cookie (``user_id``), and a request without the right cookie hits the
wrong database or 404s. ``SessionContext`` carries that identity and
``execute`` attaches it as a cookie to every call.

Retries cover only transient failures (network errors, timeouts, 5xx), and are
method-aware to avoid duplicate writes: a full retry policy applies to
idempotent methods (GET/HEAD/OPTIONS/PUT/DELETE), but a non-idempotent method
(POST/PATCH) is only retried for connect-phase failures — i.e. when the
request provably never reached the gym. A read-timeout or 5xx on a POST means
the gym may already have committed the write, so retrying could create a
duplicate; that failure is surfaced instead of blindly resent. A 4xx —
notably a gym 404 for an unknown/expired session — is returned as-is so the
engine can surface it rather than silently retrying.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx

from app.adapters.base import BuiltRequest
from app.errors import UpstreamUnavailable

# Methods where re-sending an identical request is safe even if the gym
# already processed a prior attempt (per HTTP semantics; POST/PATCH excluded).
IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE"})


def _is_connect_phase_error(exc: Exception) -> bool:
    """True if ``exc`` happened before the request reached the gym.

    A connect-phase failure means the gym never saw the request, so retrying
    is safe regardless of method.
    """
    return isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout))


@dataclass(frozen=True)
class SessionContext:
    """The gym session a trigger acts under."""

    user_id: str
    cookie_name: str = "user_id"

    def cookie_header(self) -> str:
        return f"{self.cookie_name}={self.user_id}"


async def execute(
    client: httpx.AsyncClient,
    built: BuiltRequest,
    session: SessionContext,
    *,
    max_retries: int,
    backoff: float,
) -> httpx.Response:
    """Send ``built`` with the session cookie, retrying transient failures.

    Raises ``UpstreamUnavailable`` if the gym can't be reached after retries.
    Returns the response otherwise (including 4xx/5xx — the engine classifies it).
    """
    # Attach the session cookie as an explicit header (per-request ``cookies=``
    # is deprecated in httpx and has ambiguous persistence semantics).
    headers = dict(built.headers or {})
    headers["Cookie"] = session.cookie_header()

    idempotent = built.method in IDEMPOTENT_METHODS

    last_response: httpx.Response | None = None
    for attempt in range(max_retries + 1):
        try:
            response = await client.request(
                built.method,
                built.url,
                json=built.json,
                params=built.query,
                headers=headers,
            )
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            can_retry = idempotent or _is_connect_phase_error(exc)
            if can_retry and attempt < max_retries:
                await asyncio.sleep(backoff * (2**attempt))
                continue
            raise UpstreamUnavailable(
                f"Could not reach gym at {built.url}: {exc}"
            ) from exc

        # Retry server errors only for idempotent methods; a non-idempotent
        # method may have already been applied by the gym, so surface the
        # 5xx instead of risking a duplicate write.
        if response.status_code >= 500 and idempotent and attempt < max_retries:
            last_response = response
            await asyncio.sleep(backoff * (2**attempt))
            continue
        return response

    # Retries on 5xx exhausted — return the last server-error response.
    assert last_response is not None  # loop always sets it before exhausting
    return last_response
