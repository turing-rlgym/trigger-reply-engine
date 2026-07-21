"""FastAPI dependencies: hand routers the singletons built in the lifespan.

The ``Engine`` and ``Registry`` are constructed once at startup and stored on
``app.state`` (see ``app.main``); these resolve them per request.
"""

from __future__ import annotations

import secrets

from fastapi import Header, Request

from app.engine import Engine
from app.errors import AuthError
from app.registry import Registry


def get_engine(request: Request) -> Engine:
    return request.app.state.engine


def get_registry(request: Request) -> Registry:
    return request.app.state.registry


def require_api_key(
    request: Request, x_api_key: str | None = Header(default=None, alias="X-API-Key")
) -> None:
    """Require a matching ``X-API-Key`` header when ``Settings.api_key`` is set.

    A no-op when the setting is empty (the local/dev default) — see
    ``.env.example``. Without this, any network-reachable deployment is an
    open write proxy into every gym session.
    """
    expected = request.app.state.settings.api_key
    if not expected:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise AuthError("Missing or invalid API key")
