"""The pluggable reply Generator.

``Generator`` is the interface the engine depends on. Two implementations:

* ``TemplateStub`` — deterministic, no network; the default and the fallback.
* ``OpenAIGenerator`` — GPT-backed; used only when ``USE_LLM`` is on and a key
  is present. Wrapped by ``FallbackGenerator`` so any LLM error/timeout quietly
  degrades to the stub instead of failing the trigger.

Every generator returns ``{field: text}`` covering exactly the requested
``fields`` (the manifest's ``generates`` list). The engine validates that
before rendering, so a missing field never silently produces an empty body.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Protocol

from app.config import Settings
from app.generation import policy

logger = logging.getLogger("trigger_reply_engine.generation")


@dataclass
class GenerationRequest:
    """Everything a generator needs to fill an action's ``generates`` fields."""

    gym_name: str
    action: str
    fields: list[str]
    description: str = ""
    params: dict[str, Any] = dataclass_field(default_factory=dict)
    seed: int | None = None


class Generator(Protocol):
    async def generate(self, req: GenerationRequest) -> dict[str, str]: ...


class TemplateStub:
    """Deterministic, offline generator built on ``policy``."""

    async def generate(self, req: GenerationRequest) -> dict[str, str]:
        rng = policy.make_rng(req.seed)
        return {f: policy.canned_text(req.action, f, req.params, rng) for f in req.fields}


class OpenAIGenerator:
    """GPT-backed generator. Produces short, realistic content per field."""

    def __init__(self, settings: Settings) -> None:
        # Imported lazily: `openai` is an optional extra ([llm], see
        # pyproject.toml) so the default install/image stays lean.
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "USE_LLM=1 but the 'openai' package isn't installed. Install the "
                "'llm' extra (pip install '.[llm]') or set USE_LLM=0."
            ) from exc

        self._client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=settings.llm_timeout)
        self._model = settings.openai_model

    async def generate(self, req: GenerationRequest) -> dict[str, str]:
        system = (
            "You simulate a realistic sales prospect reacting inside a CRM / "
            "outreach tool. Reply ONLY with a compact JSON object mapping each "
            "requested field name to a short, natural, professional value. No prose."
        )
        user = json.dumps(
            {
                "action": req.action,
                "description": req.description,
                "fields_to_fill": req.fields,
                "context": req.params,
            },
            default=str,
        )
        completion = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.7,
            seed=req.seed,
        )
        content = completion.choices[0].message.content or "{}"
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("LLM did not return a JSON object")
        # Keep only requested fields; coerce to str.
        return {f: str(data[f]) for f in req.fields if f in data}


class FallbackGenerator:
    """Try the primary generator; on any error, degrade to the stub."""

    def __init__(self, primary: Generator, fallback: Generator) -> None:
        self._primary = primary
        self._fallback = fallback

    async def generate(self, req: GenerationRequest) -> dict[str, str]:
        try:
            result = await self._primary.generate(req)
        except Exception as exc:  # noqa: BLE001 — any LLM failure must degrade, not crash
            logger.warning("LLM generation failed (%s); using deterministic fallback.", exc)
            return await self._fallback.generate(req)
        # If the LLM omitted fields, backfill them deterministically.
        missing = [f for f in req.fields if f not in result or not result[f]]
        if missing:
            backfill = await self._fallback.generate(
                GenerationRequest(
                    gym_name=req.gym_name,
                    action=req.action,
                    fields=missing,
                    description=req.description,
                    params=req.params,
                    seed=req.seed,
                )
            )
            result = {**backfill, **{k: v for k, v in result.items() if v}}
        return result


def build_generator(settings: Settings) -> Generator:
    """Pick the generator based on config. Stub unless LLM is enabled + keyed."""
    stub = TemplateStub()
    if settings.use_llm and settings.openai_api_key:
        return FallbackGenerator(OpenAIGenerator(settings), stub)
    return stub
