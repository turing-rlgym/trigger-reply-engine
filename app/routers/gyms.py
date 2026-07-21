"""Introspection — lets callers (and agents) discover what's supported.

Mounted under ``/api`` (see ``app.main``): ``GET /api/gyms`` -> gyms;
``GET /api/gyms/{name}/actions`` -> that gym's world actions;
``GET /api/gyms/{name}/actions/{action}/schema`` -> the action's inputs,
required fields, and which fields are LLM-generated. This is the runtime,
self-describing counterpart to the manifest files.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.deps import get_registry
from app.registry import Registry
from app.schemas import ActionSchemaResponse, ActionSummary, GymSummary

router = APIRouter(tags=["introspection"])


@router.get("/gyms", response_model=list[GymSummary], summary="List available gyms")
async def list_gyms(registry: Registry = Depends(get_registry)) -> list[GymSummary]:
    summaries: list[GymSummary] = []
    for name in registry.gyms():
        adapter = registry.get(name)
        summaries.append(
            GymSummary(name=name, base_url=adapter.base_url, actions=sorted(adapter.actions()))
        )
    return summaries


@router.get(
    "/gyms/{gym_name}/actions",
    response_model=list[ActionSummary],
    summary="List a gym's supported world actions",
)
async def list_actions(
    gym_name: str, registry: Registry = Depends(get_registry)
) -> list[ActionSummary]:
    adapter = registry.get(gym_name)  # UnknownGymError -> 404
    return [
        ActionSummary(action=name, description=spec.description, generates=spec.generates)
        for name, spec in adapter.actions().items()
    ]


@router.get(
    "/gyms/{gym_name}/actions/{action}/schema",
    response_model=ActionSchemaResponse,
    summary="Describe an action's inputs, required fields, and generated fields",
)
async def action_schema(
    gym_name: str, action: str, registry: Registry = Depends(get_registry)
) -> ActionSchemaResponse:
    adapter = registry.get(gym_name)  # UnknownGymError -> 404
    spec = adapter.action(action)  # UnknownActionError -> 404
    return ActionSchemaResponse(
        gym_name=gym_name,
        action=action,
        description=spec.description,
        generates=spec.generates,
        inputs=spec.inputs,
    )
