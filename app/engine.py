"""The engine: orchestrates one trigger end to end.

Pipeline: resolve gym/action -> validate + fill params -> generate the
``generates`` fields (skipped on ``dry_run``) -> render the request -> execute
against the gym with the session cookie -> normalize the response. Errors are
raised as ``EngineError`` subclasses and rendered uniformly by ``app.main``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
import weakref
from collections import OrderedDict
from typing import Any

import httpx

from app.adapters.base import GymAdapter
from app.config import Settings
from app.errors import UpstreamError
from app.generation.llm import GenerationRequest, Generator
from app.registry import Registry
from app.schemas import TriggerRequest, TriggerResponse
from app.session import SessionContext, execute

logger = logging.getLogger("trigger_reply_engine.engine")


class _TTLCache:
    """Bounded in-memory TTL cache for idempotent replays (single-instance).

    Backed by an ``OrderedDict`` in insertion order so ``set`` can cheaply
    sweep expired entries from the front and evict the oldest entry once
    ``max_size`` is exceeded — without either, a steady stream of one-off
    idempotency keys would grow the cache forever even though each entry
    individually has a TTL (an entry only expired lazily, on a later ``get``
    for that same key, which never happens for a key used once).
    """

    def __init__(self, ttl: float, max_size: int = 10_000) -> None:
        self._ttl = ttl
        self._max_size = max_size
        self._store: OrderedDict[str, tuple[float, TriggerResponse]] = OrderedDict()

    def get(self, key: str) -> TriggerResponse | None:
        item = self._store.get(key)
        if item is None:
            return None
        stamped, value = item
        if time.monotonic() - stamped > self._ttl:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: TriggerResponse) -> None:
        self._sweep_expired()
        self._store[key] = (time.monotonic(), value)
        self._store.move_to_end(key)
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def _sweep_expired(self) -> None:
        now = time.monotonic()
        while self._store:
            _, (stamped, _) = next(iter(self._store.items()))
            if now - stamped <= self._ttl:
                break
            self._store.popitem(last=False)


def _parse_body(response: httpx.Response) -> Any:
    """Best-effort decode of a gym response body (JSON, else raw text)."""
    try:
        return response.json()
    except ValueError:
        text = response.text
        return {"raw": text} if text else None


def _mask(session_id: str) -> str:
    """A short, non-reversible digest for logs (the raw id is a session token)."""
    return hashlib.sha256(session_id.encode()).hexdigest()[:10]


class Engine:
    def __init__(
        self,
        *,
        registry: Registry,
        generator: Generator,
        client: httpx.AsyncClient,
        settings: Settings,
    ) -> None:
        self.registry = registry
        self._generator = generator
        self._client = client
        self._settings = settings
        self._idempotency = _TTLCache(settings.idempotency_ttl, settings.idempotency_max_entries)
        # Serializes concurrent requests sharing an idempotency key so only
        # one actually calls the gym; the rest wait and get the cached result.
        # A ``WeakValueDictionary`` means a key's lock is dropped automatically
        # once no request is holding it — no separate cleanup needed.
        self._idempotency_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    async def trigger(self, req: TriggerRequest) -> TriggerResponse:
        trigger_id = uuid.uuid4().hex
        # Passed as `extra=` on every log call below so trigger_id/episode_id/
        # gym/action are queryable JSON fields (see app.logutil) rather than
        # baked into a free-text message.
        log_ctx = {
            "trigger_id": trigger_id,
            "episode_id": req.episode_id,
            "gym": req.gym_name,
            "action": req.action,
            "session": _mask(req.session_id),
            "dry_run": req.dry_run,
        }
        logger.info("trigger start", extra=log_ctx)

        cache_key = self._cache_key(req)
        if cache_key is None:
            return await self._run(req, trigger_id, log_ctx)

        lock = self._lock_for(cache_key)
        async with lock:
            cached = self._idempotency.get(cache_key)
            if cached is not None:
                logger.info("trigger idempotent-replay", extra=log_ctx)
                return cached
            out = await self._run(req, trigger_id, log_ctx)
            self._idempotency.set(cache_key, out)
            return out

    def _lock_for(self, key: str) -> asyncio.Lock:
        lock = self._idempotency_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._idempotency_locks[key] = lock
        return lock

    async def _run(
        self, req: TriggerRequest, trigger_id: str, log_ctx: dict[str, Any]
    ) -> TriggerResponse:
        """The actual trigger work, run once per (non-replayed) request."""
        adapter = self.registry.get(req.gym_name)  # UnknownGymError -> 404
        # ``steps`` is a reserved top-level params key: per-step JSON body
        # overrides keyed by step name (see ManifestAdapter.build_requests),
        # not a semantic input any action's manifest should declare.
        overrides = req.params.get("steps") or {}
        semantic = {k: v for k, v in req.params.items() if k != "steps"}

        filled = adapter.validate_and_fill(req.action, semantic)  # 404 / 422

        generated = await self._generate(adapter, req, filled)
        filled.update(generated)

        built = adapter.build_requests(req.action, filled, overrides)

        if req.dry_run:
            logger.info("trigger dry-run", extra=log_ctx)
            return TriggerResponse(
                gym_name=req.gym_name,
                action=req.action,
                trigger_id=trigger_id,
                dry_run=True,
                generated=generated,
                request_previews=[b.preview() for b in built],
            )

        session = SessionContext(user_id=req.session_id, cookie_name=adapter.cookie_name)
        results: list[dict[str, Any]] = []
        for index, request in enumerate(built):
            response = await execute(
                self._client,
                request,
                session,
                max_retries=self._settings.max_retries,
                backoff=self._settings.retry_backoff,
            )
            body = _parse_body(response)
            if response.status_code >= 400:
                logger.warning(
                    "trigger gym-error",
                    extra={
                        **log_ctx,
                        "gym_status": response.status_code,
                        "step": request.name,
                        "step_index": index,
                    },
                )
                raise UpstreamError(
                    response.status_code,
                    body,
                    step_name=request.name,
                    step_index=index,
                    completed=results,
                )
            results.append(
                {
                    "step": request.name,
                    "method": request.method,
                    "gym_status": response.status_code,
                    "body": body,
                }
            )

        out = TriggerResponse(
            gym_name=req.gym_name,
            action=req.action,
            trigger_id=trigger_id,
            generated=generated,
            results=results,
        )
        logger.info(
            "trigger ok", extra={**log_ctx, "gym_status": [r["gym_status"] for r in results]}
        )
        return out

    # ── helpers ────────────────────────────────────────────────────────────────

    async def _generate(
        self, adapter: GymAdapter, req: TriggerRequest, filled: dict[str, Any]
    ) -> dict[str, str]:
        """Fill the action's ``generates`` fields that the caller left empty.

        Skipped entirely on ``dry_run`` (no LLM/gym side effects). The
        configured ``Generator`` is responsible for covering every requested
        field (``TemplateStub`` always does; ``FallbackGenerator`` backfills
        from it on partial/failed LLM output) — the engine trusts that
        contract rather than re-implementing its own backfill.
        """
        generates = adapter.generates_for(req.action)
        missing = [f for f in generates if f not in filled or filled[f] in (None, "")]
        if not missing or req.dry_run:
            return {}

        gen_req = GenerationRequest(
            gym_name=req.gym_name,
            action=req.action,
            fields=missing,
            description=adapter.action(req.action).description,
            params=filled,
            seed=req.seed,
        )
        return await self._generator.generate(gen_req)

    @staticmethod
    def _cache_key(req: TriggerRequest) -> str | None:
        # dry_run has no side effect to dedupe, so it's deliberately excluded
        # from caching entirely (not just from the key) — otherwise a
        # preview-then-commit pattern sharing one idempotency_key would let
        # the live call replay the dry-run's response and never hit the gym.
        if not req.idempotency_key or req.dry_run:
            return None
        # `action` is part of the key: two different actions must never share
        # a cache slot even if a caller (incorrectly) reuses the same
        # idempotency_key for both.
        return f"{req.gym_name}:{req.action}:{req.session_id}:{req.idempotency_key}"
