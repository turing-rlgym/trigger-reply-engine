"""Pydantic models: the manifest schema (how a gym is described) and the public
API schema (``/trigger`` request/response + introspection payloads).

The manifest models double as validation for the YAML files: a malformed
manifest fails fast at load time with a clear error.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ── Manifest schema ───────────────────────────────────────────────────────────


class InputSpec(BaseModel):
    """One input parameter of an action."""

    type: Literal["string", "integer", "number", "boolean", "object", "array"] = "string"
    required: bool = False
    default: Any | None = None
    description: str = ""


class RequestSpec(BaseModel):
    """How an action maps to a concrete HTTP request against the gym.

    ``path``/``json``/``query``/``headers`` values are Jinja2 templates rendered
    with the (validated + generated) params plus helpers (``now``, ``today``,
    ``correlation_id``).
    """

    model_config = ConfigDict(populate_by_name=True)

    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    path: str
    # ``json`` in YAML -> ``json_body`` here (avoids clashing with BaseModel.json()).
    json_body: dict[str, Any] | None = Field(default=None, alias="json")
    query: dict[str, Any] | None = None
    headers: dict[str, str] | None = None


class StepSpec(RequestSpec):
    """One HTTP call within a multi-step action.

    ``name`` identifies the step for: result/preview labeling, per-step caller
    overrides (``params.steps.<name>``, see ``app.engine``), response-piping
    (``{{ steps.<name>.<field> }}`` in a later step's template, see
    ``app.adapters.base``), and error reporting on partial failure. Left
    blank for an auto-generated ``step-<index>`` name (see
    ``ActionModel.resolved_steps``).
    """

    name: str = ""
    # A step that exists purely to read state for a later step to template
    # against (e.g. a "hydrate" GET before a full-collection PUT-replace) —
    # still executed and piped into ``steps``, but omitted from the caller-
    # facing ``results``/``request_previews`` since it has no side effect the
    # caller asked for.
    internal: bool = False


class ActionModel(BaseModel):
    """A single world action a gym supports (e.g. ``chat.reply``).

    An action is either a single ``request`` (the common case — one HTTP call)
    or an ordered list of ``steps`` (for actions that mirror a UI flow issuing
    several calls, e.g. creating a message *and* patching the conversation
    *and* logging a timeline activity). Exactly one of the two must be given;
    ``resolved_steps`` normalizes either into a uniform list so the rest of the
    engine/adapter never has to special-case single- vs. multi-step actions.
    """

    description: str = ""
    # Fields the reply Generator fills when the caller omits them.
    generates: list[str] = []
    inputs: dict[str, InputSpec] = {}
    request: RequestSpec | None = None
    steps: list[StepSpec] = []

    @model_validator(mode="after")
    def _exactly_one_of_request_or_steps(self) -> ActionModel:
        if bool(self.request) == bool(self.steps):
            raise ValueError("An action must declare exactly one of 'request' or 'steps'")
        return self

    @property
    def resolved_steps(self) -> list[StepSpec]:
        """The action's steps, normalized (single ``request`` -> one named step)."""
        if self.request is not None:
            return [StepSpec(name="request", **self.request.model_dump(by_alias=True))]
        return [
            step if step.name else step.model_copy(update={"name": f"step-{i}"})
            for i, step in enumerate(self.steps)
        ]


class SessionSpec(BaseModel):
    """How the gym carries session identity. All current gyms use a cookie."""

    cookie: str = "user_id"


class ManifestModel(BaseModel):
    """A complete gym manifest — the unit of extensibility.

    ``base_url`` is optional: it is the *default* used when the gym's
    convention-derived override env var (see ``app.registry.base_url_env_var``)
    is unset. A manifest may omit it entirely and rely purely on that env var;
    the registry warns (but does not fail) if neither resolves to a usable URL.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    base_url: str = ""
    session: SessionSpec = SessionSpec()
    actions: dict[str, ActionModel] = {}


# ── Public API schema ─────────────────────────────────────────────────────────


class TriggerRequest(BaseModel):
    """Body of ``POST /trigger`` — one simulated world reaction."""

    gym_name: str = Field(..., description="Target gym (must have a loaded manifest).")
    action: str = Field(..., description="World action, e.g. 'chat.reply' or 'email.reply'.")
    session_id: str = Field(
        ...,
        description="Gym session id; sent as the gym's session cookie so the write "
        "lands in the correct isolated database. Required.",
    )
    params: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = Field(
        default=None,
        description="If set, a repeated call returns the first result instead of re-applying.",
    )
    dry_run: bool = Field(
        default=False,
        description="Validate + build the request without calling the LLM or the gym.",
    )
    seed: int | None = Field(default=None, description="Seeds deterministic reply generation.")
    episode_id: str | None = Field(default=None, description="Correlation id echoed into logs.")


class TriggerResponse(BaseModel):
    """Result of a trigger. Uniform across all gyms.

    ``results``/``request_previews`` are always lists, one entry per step of
    the action (see ``ActionModel.resolved_steps``) — length 1 for an
    ordinary single-request action, length N for a multi-step one. Each
    ``results`` entry is ``{"step", "method", "gym_status", "body"}``.
    """

    status: Literal["ok"] = "ok"
    gym_name: str
    action: str
    trigger_id: str
    dry_run: bool = False
    generated: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    request_previews: list[dict[str, Any]] | None = None


# ── Introspection payloads ────────────────────────────────────────────────────


class GymSummary(BaseModel):
    name: str
    base_url: str
    actions: list[str]


class ActionSummary(BaseModel):
    action: str
    description: str
    generates: list[str]


class ActionSchemaResponse(BaseModel):
    gym_name: str
    action: str
    description: str
    generates: list[str]
    inputs: dict[str, InputSpec]
