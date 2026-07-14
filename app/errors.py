"""Domain errors + their normalized HTTP mapping.

The engine and adapters raise these; ``app.main`` registers handlers that render
them in one consistent envelope: ``{"status": "error", "error": {...}}``. Keeping
the mapping in one place is what makes error behavior uniform across every gym.
"""

from __future__ import annotations

from typing import Any


class EngineError(Exception):
    """Base class for all trigger-engine domain errors."""

    status_code: int = 500
    code: str = "engine_error"

    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class AuthError(EngineError):
    """Missing/incorrect ``X-API-Key`` (only raised when ``Settings.api_key`` is set)."""

    status_code = 401
    code = "unauthorized"


class UnknownGymError(EngineError):
    status_code = 404
    code = "unknown_gym"


class UnknownActionError(EngineError):
    status_code = 404
    code = "unknown_action"


class ParamValidationError(EngineError):
    status_code = 422
    code = "param_validation_error"

    def __init__(self, errors: list[str]) -> None:
        super().__init__("Parameter validation failed", details=errors)
        self.errors = errors


class UpstreamError(EngineError):
    """The target gym returned a non-2xx response (surfaced, not swallowed).

    For a multi-step action, ``step_name``/``step_index`` identify which step
    failed and ``completed`` lists the ``results`` of the steps that already
    succeeded (there's no rollback — matches the real UI's fire-and-forget
    multi-call flows — so the caller needs to know what already landed).
    """

    status_code = 502
    code = "gym_error"

    def __init__(
        self,
        gym_status: int,
        body: Any,
        *,
        step_name: str | None = None,
        step_index: int | None = None,
        completed: list[dict[str, Any]] | None = None,
    ) -> None:
        details: dict[str, Any] = {"gym_status": gym_status, "gym_body": body}
        if step_name is not None:
            details["step"] = step_name
            details["step_index"] = step_index
        if completed:
            details["completed"] = completed
        super().__init__(f"Target gym returned {gym_status}", details=details)
        self.gym_status = gym_status
        self.body = body
        self.step_name = step_name
        self.step_index = step_index
        self.completed = completed or []


class UpstreamUnavailable(EngineError):
    """The target gym could not be reached (network error / timeout)."""

    status_code = 504
    code = "gym_unavailable"
