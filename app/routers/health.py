"""Liveness endpoint. Mounted under ``/api`` (see ``app.main``) so the final
path is ``/api/health``, matching the gym ecosystem convention and the
container HEALTHCHECK / Cloud Run probe."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["meta"])


@router.get("/health", summary="Liveness probe")
async def health() -> dict[str, str]:
    return {"status": "ok"}
