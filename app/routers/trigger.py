"""``POST /api/trigger`` — the single entrypoint that simulates one world reaction.

Mounted under ``/api`` (see ``app.main``). Input:
``{gym_name, action, session_id, params, ...}``. Output: a uniform
``TriggerResponse``. All orchestration lives in the engine; this router is a
thin HTTP shell.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.deps import get_engine
from app.engine import Engine
from app.schemas import TriggerRequest, TriggerResponse

router = APIRouter(tags=["trigger"])


@router.post(
    "/trigger",
    response_model=TriggerResponse,
    summary="Simulate one world reaction against a target gym",
)
async def post_trigger(
    req: TriggerRequest,
    engine: Engine = Depends(get_engine),
) -> TriggerResponse:
    return await engine.trigger(req)
