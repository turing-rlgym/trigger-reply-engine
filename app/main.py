"""FastAPI application entrypoint.

Boot: the lifespan loads all gym manifests into a ``Registry`` once, builds the
reply ``Generator`` (LLM or deterministic stub) and a shared outbound HTTP
client, and assembles the ``Engine``. These live on ``app.state`` and are handed
to routers via ``app.deps``. Every route (business + meta) lives under ``/api``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.deps import require_api_key
from app.engine import Engine
from app.errors import EngineError
from app.generation.llm import build_generator
from app.logutil import configure_logging
from app.registry import Registry
from app.routers import gyms, health, trigger

configure_logging()
logger = logging.getLogger("trigger_reply_engine")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    registry = Registry.load(settings.manifests_dir)
    logger.info(
        "Loaded %d gym manifest(s) from %s: %s",
        len(registry.gyms()),
        settings.manifests_dir,
        ", ".join(registry.gyms()) or "(none)",
    )
    if settings.use_llm and not settings.openai_api_key:
        logger.warning(
            "USE_LLM is set but OPENAI_API_KEY is empty; falling back to the template stub."
        )

    generator = build_generator(settings)
    client = httpx.AsyncClient(timeout=settings.request_timeout)
    engine = Engine(registry=registry, generator=generator, client=client, settings=settings)

    app.state.settings = settings
    app.state.registry = registry
    app.state.engine = engine
    try:
        yield
    finally:
        await client.aclose()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Trigger-Reply-Engine",
        version="0.1.0",
        description=(
            "World Simulator: a manifest-driven service that generates realistic "
            "reactions (email replies, chat replies, unsubscribes, …) and applies "
            "them to a target gym via its API."
        ),
        lifespan=lifespan,
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    if settings.cors_origins == "*":
        origins = ["*"]
    else:
        origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(EngineError)
    async def _engine_error_handler(_request: Request, exc: EngineError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status": "error",
                "error": {"code": exc.code, "message": exc.message, "details": exc.details},
            },
        )

    # Every route lives under /api. Business routes additionally require the
    # API key (a no-op until Settings.api_key is set — see app.deps).
    app.include_router(health.router, prefix="/api")
    app.include_router(trigger.router, prefix="/api", dependencies=[Depends(require_api_key)])
    app.include_router(gyms.router, prefix="/api", dependencies=[Depends(require_api_key)])
    return app


app = create_app()
