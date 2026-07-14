"""The adapter layer: how a gym manifest becomes one or more HTTP requests.

``ManifestAdapter`` is the default, config-only adapter — it validates params
against the action's input schema and renders each of the action's steps
(path/json/query/headers) with Jinja2. It needs no per-gym code, which is what
lets a new gym — however many calls one of its actions needs — be added by
dropping a YAML file.

Every action resolves to an ordered list of steps (``ActionModel.resolved_steps``
— a single-``request`` action is just a list of one). All steps in one action
share a render context: the caller's params, any generated fields, and a
handful of engine-owned helpers (``now``, ``today``, ``correlation_id``)
computed *once* per ``build_requests`` call so, e.g., a message id referenced
by two different steps is bit-identical across them rather than independently
regenerated.

For the rare gym whose request can't be expressed with templating, subclass
``ManifestAdapter`` and override ``build_requests`` (the escape hatch); the
``GymAdapter`` protocol is what the engine actually depends on.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from urllib.parse import quote

from jinja2 import ChainableUndefined, Environment, Template

from app.errors import ParamValidationError, UnknownActionError
from app.schemas import ActionModel, InputSpec, ManifestModel, StepSpec

# Undefined template vars render as empty strings (and chain safely) rather than
# raising — a missing optional/generated field should not blow up rendering.
_JINJA = Environment(undefined=ChainableUndefined, autoescape=False)

# One structural check per declared ``InputSpec.type``. ``bool`` is excluded
# from ``integer``/``number`` since Python's ``bool`` is a subclass of ``int``
# and a manifest declaring ``type: integer`` almost certainly doesn't want
# ``true``/``false`` silently accepted.
_TYPE_CHECKS: dict[str, Any] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
}


def _type_error(name: str, ispec: InputSpec, value: Any) -> str | None:
    check = _TYPE_CHECKS.get(ispec.type)
    if check is None or check(value):
        return None
    return f"Parameter '{name}' must be of type '{ispec.type}' (got {type(value).__name__})"


def _url_quote(value: Any) -> str:
    return "" if value is None else quote(str(value), safe="")


# Path segments must not let a param value inject extra segments or query
# string (e.g. a ``conversation_id`` of ``"c1/../admin"`` or ``"c1?x=1"``).
# ``finalize`` runs on the result of each ``{{ }}`` expression before it's
# stitched into the surrounding literal text, so this percent-encodes only
# the dynamic parts — the manifest's literal slashes are untouched.
_JINJA_PATH = Environment(undefined=ChainableUndefined, autoescape=False, finalize=_url_quote)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _deep_merge(base: Any, override: Any) -> Any:
    """Recursively merge ``override`` onto ``base``; ``override`` wins on leaves.

    Dicts are merged key-by-key (recursively); anything else — including
    lists — is replaced wholesale by ``override`` when present. Lists aren't
    merged element-wise since there's no reliable key to align entries on.
    """
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = _deep_merge(base.get(key), value) if key in base else value
        return merged
    return base if override is None else override


def _compile(value: Any, env: Environment) -> Any:
    """Recursively pre-compile Jinja templates embedded in a manifest structure.

    Mirrors the shape-walk in ``_render_compiled``, but runs once (at adapter
    construction, since manifests don't change at runtime) instead of
    re-parsing every string leaf's template on every ``/trigger`` call.
    String leaves become compiled ``Template`` objects; dict/list are walked;
    other scalars (bool/int/float/None) pass through unchanged.
    """
    if isinstance(value, str):
        return env.from_string(value)
    if isinstance(value, dict):
        return {k: _compile(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [_compile(v, env) for v in value]
    return value


def _render_compiled(value: Any, ctx: dict[str, Any]) -> Any:
    """Render a structure previously produced by ``_compile`` against ``ctx``."""
    if isinstance(value, Template):
        return value.render(**ctx)
    if isinstance(value, dict):
        return {k: _render_compiled(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [_render_compiled(v, ctx) for v in value]
    return value


@dataclass(frozen=True)
class _CompiledStep:
    """Pre-compiled Jinja templates for one step of an action."""

    name: str
    method: str
    path: Any
    json_body: Any
    query: Any
    headers: Any

    @classmethod
    def from_spec(cls, spec: StepSpec) -> _CompiledStep:
        return cls(
            name=spec.name,
            method=spec.method,
            path=_compile(spec.path, _JINJA_PATH),
            json_body=_compile(spec.json_body, _JINJA) if spec.json_body is not None else None,
            query=_compile(spec.query, _JINJA) if spec.query is not None else None,
            headers=_compile(spec.headers, _JINJA) if spec.headers is not None else None,
        )


@dataclass
class BuiltRequest:
    """A fully-resolved outbound request (before the session cookie is attached)."""

    name: str
    method: str
    url: str
    json: dict[str, Any] | None = None
    query: dict[str, Any] | None = None
    headers: dict[str, str] | None = field(default=None)

    def preview(self) -> dict[str, Any]:
        """A JSON-serializable view for ``dry_run`` responses / logs."""
        return {
            "step": self.name,
            "method": self.method,
            "url": self.url,
            "json": self.json,
            "query": self.query,
            "headers": self.headers,
        }


@runtime_checkable
class GymAdapter(Protocol):
    """What the engine depends on. ``ManifestAdapter`` is the default impl."""

    name: str
    base_url: str
    cookie_name: str

    def actions(self) -> dict[str, ActionModel]: ...

    def action(self, name: str) -> ActionModel: ...

    def validate_and_fill(self, action: str, params: dict[str, Any]) -> dict[str, Any]: ...

    def generates_for(self, action: str) -> list[str]: ...

    def build_requests(
        self,
        action: str,
        params: dict[str, Any],
        overrides: dict[str, Any] | None = None,
    ) -> list[BuiltRequest]: ...


class ManifestAdapter:
    """Default adapter driven entirely by a validated manifest."""

    def __init__(self, manifest: ManifestModel) -> None:
        self._manifest = manifest
        self.name = manifest.name
        self.base_url = manifest.base_url
        self.cookie_name = manifest.session.cookie
        # Compiled once here (see _CompiledStep) rather than re-parsing every
        # action's Jinja templates on every /trigger call. Keyed by action
        # name -> its resolved (single- or multi-step) list of steps.
        self._compiled: dict[str, list[_CompiledStep]] = {
            action_name: [_CompiledStep.from_spec(step) for step in spec.resolved_steps]
            for action_name, spec in manifest.actions.items()
        }

    # ── introspection ─────────────────────────────────────────────────────────

    def actions(self) -> dict[str, ActionModel]:
        return self._manifest.actions

    def action(self, name: str) -> ActionModel:
        spec = self._manifest.actions.get(name)
        if spec is None:
            raise UnknownActionError(
                f"Gym '{self.name}' has no action '{name}'",
                details={"available": sorted(self._manifest.actions)},
            )
        return spec

    def generates_for(self, action: str) -> list[str]:
        return list(self.action(action).generates)

    # ── validation + rendering ─────────────────────────────────────────────────

    def validate_and_fill(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Apply declared defaults, enforce required params, and check types.

        Returns a new params dict; raises ``ParamValidationError`` listing every
        failure at once (better DX than failing one at a time).
        """
        spec = self.action(action)
        filled = dict(params)
        errors: list[str] = []
        for name, ispec in spec.inputs.items():
            missing = name not in filled or filled[name] in (None, "")
            if missing:
                if ispec.default is not None:
                    filled[name] = ispec.default
                elif ispec.required:
                    errors.append(f"Missing required parameter: '{name}'")
                continue
            type_error = _type_error(name, ispec, filled[name])
            if type_error:
                errors.append(type_error)
        if errors:
            raise ParamValidationError(errors)
        return filled

    def build_requests(
        self,
        action: str,
        params: dict[str, Any],
        overrides: dict[str, Any] | None = None,
    ) -> list[BuiltRequest]:
        """Render every step of ``action`` into a fully-resolved request list.

        ``overrides`` (from ``params.steps.<name>`` at the API layer, split out
        before this is called) are deep-merged onto each step's rendered JSON
        body, keyed by step name — the escape hatch for fields the manifest
        doesn't template. ``now``/``today``/``correlation_id`` are computed once
        so every step in this call sees the identical values.
        """
        self.action(action)  # UnknownActionError -> 404
        steps = self._compiled[action]
        ctx: dict[str, Any] = {
            **params,
            "now": _now_iso(),
            "today": _today(),
            "correlation_id": uuid.uuid4().hex,
        }
        overrides = overrides or {}

        def _maybe_render(value: Any) -> Any:
            return _render_compiled(value, ctx) if value is not None else None

        built = []
        for step in steps:
            path = _render_compiled(step.path, ctx)
            url = self.base_url.rstrip("/") + "/" + str(path).lstrip("/")
            json_body = _maybe_render(step.json_body)
            step_override = overrides.get(step.name)
            if step_override is not None:
                json_body = _deep_merge(json_body, step_override)
            built.append(
                BuiltRequest(
                    name=step.name,
                    method=step.method,
                    url=url,
                    json=json_body,
                    query=_maybe_render(step.query),
                    headers=_maybe_render(step.headers),
                )
            )
        return built
