"""The adapter layer: how a gym manifest becomes one or more HTTP requests.

``ManifestAdapter`` is the default, config-only adapter — it validates params
against the action's input schema and renders each of the action's steps
(path/json/query/headers) with Jinja2. It needs no per-gym code, which is what
lets a new gym — however many calls one of its actions needs — be added by
dropping a YAML file.

Every action resolves to an ordered list of steps (``ActionModel.resolved_steps``
— a single-``request`` action is just a list of one). All steps in one action
share a render context (``ctx``): the caller's params, any generated fields,
engine-owned helpers (``now``, ``today``, ``correlation_id``), and — the
response-piping primitive — ``steps``, a dict of prior steps' parsed response
bodies keyed by step name. ``ctx``/``now``/``today``/``correlation_id`` are
computed once per trigger (by ``app.engine._run``, which owns the per-step
build+execute loop) so, e.g., a message id referenced by two different steps
is bit-identical across them rather than independently regenerated, and a
later step can read an earlier step's response (e.g. a "hydrate" ``GET`` of
the caller's current saved-items list, mutated and ``PUT`` back — the
read-modify-write shape needed when a gym's write API replaces a whole
collection instead of appending one item).

``json`` bodies are rendered with a Jinja native-typed environment (a tweaked
``NativeEnvironment``, see ``_TypedNativeEnvironment``): a leaf that is
*purely* one expression (e.g. ``"{{ steps.hydrate.allSavedLeads |
append_unique(item) }}"``) yields the real Python value (a list, int, dict —
not its ``str()``) rather than a string, which is what lets a templated leaf
become a real JSON list/int. A leaf that mixes literal text with an
expression (e.g. ``"smartlink-{{ correlation_id }}"``) still renders to a
string, so existing manifests (which only ever template string params into
string leaves, or use literal bools/ints/lists) are unaffected. Unlike stock
``NativeEnvironment``, a lone *string* value is never coerced further (see
``_native_concat_no_literal_eval``) — a string-typed param stays a string
even if its value happens to look like a Python literal (e.g. a numeric list
id). ``path``/``query``/``headers`` keep the plain (string-rendering)
environment.

Three small filters are registered on the JSON environment for the
read-modify-write case: ``append_unique(seq, item, key=None)`` (idempotent
list append), ``upsert(mapping, key, value)`` (dict key set), and
``reject_by(seq, key, value)`` (drop dict entries matching a key) — see
``_append_unique``/``_upsert``/``_reject_by`` below. The first two back the
"save"/"add to list" actions; ``reject_by`` backs their inverses
("unsave"/"remove from list"/"unstar").

For the rare gym whose request can't be expressed with templating, subclass
``ManifestAdapter`` and override ``build_step`` (the escape hatch); the
``GymAdapter`` protocol is what the engine actually depends on.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import chain, islice
from types import GeneratorType
from typing import Any, Protocol, runtime_checkable
from urllib.parse import quote

from jinja2 import ChainableUndefined, Environment, Template, Undefined
from jinja2.nativetypes import NativeEnvironment, NativeTemplate

from app.errors import ParamValidationError, UnknownActionError
from app.schemas import ActionModel, InputSpec, ManifestModel, StepSpec

# Undefined template vars render as empty strings (and chain safely) rather than
# raising — a missing optional/generated field should not blow up rendering.
_JINJA = Environment(undefined=ChainableUndefined, autoescape=False)


def _append_unique(seq: Any, item: Any, key: str | None = None) -> list[Any]:
    """Jinja filter: append ``item`` to ``seq`` unless an equivalent entry exists.

    ``seq``/``item`` may be Jinja ``Undefined`` (an un-piped/empty prior step,
    e.g. during ``dry_run``) — treated as absent rather than raising. With
    ``key``, "equivalent" means an existing dict entry whose ``[key]`` equals
    ``item[key]`` (e.g. matching on a lead/account id regardless of other
    fields like a display name); without it, exact equality is used. This is
    the read-modify-write primitive for gyms whose write API replaces a whole
    collection instead of appending one item.
    """
    result = list(seq) if seq else []
    if key is not None:
        duplicate = any(
            isinstance(entry, dict) and entry.get(key) == item.get(key) for entry in result
        )
    else:
        duplicate = item in result
    if not duplicate:
        result.append(item)
    return result


def _upsert(mapping: Any, key: Any, value: Any) -> dict[str, Any]:
    """Jinja filter: set ``mapping[str(key)] = value``, returning a new dict.

    ``mapping`` may be Jinja ``Undefined`` (treated as ``{}``). Keys are
    coerced to ``str`` since JSON object keys are always strings (e.g. an
    integer account id used as a CRM-status-override key).
    """
    return {**(mapping or {}), str(key): value}


def _reject_by(seq: Any, key: str, value: Any) -> list[Any]:
    """Jinja filter: drop entries of ``seq`` whose ``[key]`` equals ``value``.

    The inverse of ``append_unique``'s ``key=`` matching — the read-modify-
    write primitive for "unsave"/"remove from list"/"unstar" actions, whose
    gym API replaces a whole collection rather than exposing a "delete one"
    endpoint. ``seq`` may be Jinja ``Undefined`` (an un-piped prior step)
    — treated as ``[]``. Non-dict entries are kept as-is (can't match a
    ``key`` lookup, so there's nothing to reject).
    """
    return [
        entry for entry in (seq or []) if not (isinstance(entry, dict) and entry.get(key) == value)
    ]


def _finalize_json(value: Any) -> Any:
    """Collapse an un-piped/un-filled ``Undefined`` leaf to ``""``.

    Without this, a pure-expression leaf on a param the caller omitted (no
    default, e.g. an optional field with no ``generates``/default — unlike
    ``append_unique``/``upsert``, which already return a concrete list/dict
    even when their input is undefined) would render as the raw
    ``Undefined`` sentinel object under native typing instead of the empty
    string the old string environment produced — and that sentinel isn't
    JSON-serializable, so it would blow up at request-send time rather than
    just rendering blank. Only affects leaves still ``Undefined`` after any
    filters ran; a defined value (including a filter's output) passes through
    untouched.
    """
    return "" if isinstance(value, Undefined) else value


def _native_concat_no_literal_eval(values: Any) -> Any:
    """Like ``jinja2.nativetypes.native_concat``, minus its ``ast.literal_eval``
    pass on a lone string result.

    Stock ``native_concat`` feeds any single-node *string* output through
    ``literal_eval`` — so a plain string param that happens to look like a
    Python literal (a numeric id such as ``"123"``, or one of the epoch-ms
    list ids this domain uses, e.g. ``"1710000000000"``) would silently
    become an ``int`` rather than staying the ``str`` its ``InputSpec``
    declares. We only want native typing for values that are *already*
    non-string (an ``integer``/``boolean``/``array``/``object`` param, or a
    filter's return value) — never a string -> other-type coercion — so a
    lone string is returned as-is instead.
    """
    head = list(islice(values, 2))
    if not head:
        return None
    if len(head) == 1:
        # A lone node is returned as-is: native typing for an already-non-string
        # value (int/bool/list/dict param or filter output), and no str->other
        # coercion for a string (unlike stock native_concat's literal_eval).
        return head[0]
    if isinstance(values, GeneratorType):
        values = chain(head, values)
    return "".join(str(v) for v in values)


class _TypedNativeEnvironment(NativeEnvironment):
    """``NativeEnvironment`` without the literal-eval-a-lone-string surprise."""

    concat = staticmethod(_native_concat_no_literal_eval)


class _TypedNativeTemplate(NativeTemplate):
    environment_class = _TypedNativeEnvironment


_TypedNativeEnvironment.template_class = _TypedNativeTemplate

# Native rendering: a leaf that is *purely* one `{{ expression }}` yields the
# expression's real Python value (list/dict/int/bool) instead of `str()` of
# it — what makes a templated leaf like the two filters above produce actual
# JSON structure. A leaf mixing literal text with an expression still renders
# to a string (see module docstring), so this only changes behavior for
# leaves that were already single opaque expressions.
_JINJA_JSON = _TypedNativeEnvironment(
    undefined=ChainableUndefined, autoescape=False, finalize=_finalize_json
)
_JINJA_JSON.filters["append_unique"] = _append_unique
_JINJA_JSON.filters["upsert"] = _upsert
_JINJA_JSON.filters["reject_by"] = _reject_by

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


def now_iso() -> str:
    """Current UTC instant, ISO-8601 — the ``now`` render-context helper.

    Public (unlike most of this module) because ``app.engine._run`` computes
    the shared ``{now, today, correlation_id}`` context once per trigger and
    needs the same helper ``build_requests`` uses for its own (piping-less)
    ``ctx``.
    """
    return datetime.now(UTC).isoformat()


def today() -> str:
    """Current UTC date, ``YYYY-MM-DD`` — the ``today`` render-context helper."""
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
    internal: bool

    @classmethod
    def from_spec(cls, spec: StepSpec) -> _CompiledStep:
        return cls(
            name=spec.name,
            method=spec.method,
            path=_compile(spec.path, _JINJA_PATH),
            # Native env: see module docstring + _JINJA_JSON.
            json_body=_compile(spec.json_body, _JINJA_JSON) if spec.json_body is not None else None,
            query=_compile(spec.query, _JINJA) if spec.query is not None else None,
            headers=_compile(spec.headers, _JINJA) if spec.headers is not None else None,
            internal=spec.internal,
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
    # A step whose only purpose is to feed a later step via response-piping
    # (see StepSpec.internal) — still a real request, just omitted from the
    # caller-facing results/previews by app.engine.
    internal: bool = False

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

    def build_step(
        self,
        action: str,
        index: int,
        ctx: dict[str, Any],
        override: dict[str, Any] | None = None,
    ) -> BuiltRequest: ...

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

    def build_step(
        self,
        action: str,
        index: int,
        ctx: dict[str, Any],
        override: dict[str, Any] | None = None,
    ) -> BuiltRequest:
        """Render one step (``resolved_steps[index]``) of ``action`` against ``ctx``.

        ``ctx`` is the step's full render context — the caller's params,
        generated fields, engine-owned helpers (``now``/``today``/
        ``correlation_id``), and ``steps`` (prior steps' parsed response
        bodies, keyed by name; see module docstring). ``override`` (a single
        step's ``params.steps.<name>`` body override, resolved by the caller)
        is deep-merged onto the rendered JSON body — the escape hatch for
        fields the manifest doesn't template. Building one step at a time
        (rather than the whole action up front) is what lets ``app.engine``
        execute a step and fold its response into ``ctx["steps"]`` before
        rendering the next.
        """
        self.action(action)  # UnknownActionError -> 404
        step = self._compiled[action][index]

        def _maybe_render(value: Any) -> Any:
            return _render_compiled(value, ctx) if value is not None else None

        path = _render_compiled(step.path, ctx)
        url = self.base_url.rstrip("/") + "/" + str(path).lstrip("/")
        json_body = _maybe_render(step.json_body)
        if override is not None:
            json_body = _deep_merge(json_body, override)
        return BuiltRequest(
            name=step.name,
            method=step.method,
            url=url,
            json=json_body,
            query=_maybe_render(step.query),
            headers=_maybe_render(step.headers),
            internal=step.internal,
        )

    def build_requests(
        self,
        action: str,
        params: dict[str, Any],
        overrides: dict[str, Any] | None = None,
    ) -> list[BuiltRequest]:
        """Render every step of ``action`` up front, with no response-piping.

        A thin back-compat wrapper around ``build_step`` for callers that want
        an action's full request list without ``app.engine``'s per-step
        build+execute loop (existing adapter tests; any external
        ``ManifestAdapter`` subclass written against the old signature). Every
        step renders against the same ``ctx``, including an always-empty
        ``steps`` — a step that templates ``{{ steps.<name>... }}`` renders
        against ``{}`` (the two merge filters tolerate this; see module
        docstring), so this only behaves differently from the engine's own
        loop for actions that rely on response-piping.
        """
        self.action(action)  # UnknownActionError -> 404
        overrides = overrides or {}
        ctx: dict[str, Any] = {
            **params,
            "now": now_iso(),
            "today": today(),
            "correlation_id": uuid.uuid4().hex,
            "steps": {},
        }
        return [
            self.build_step(action, index, ctx, overrides.get(step.name))
            for index, step in enumerate(self._compiled[action])
        ]
