"""The gym registry: loads every ``manifests/*.yaml`` into an adapter map.

This is the extensibility seam. Adding a gym means adding a YAML file here — no
code change. Manifests are loaded once at startup (see ``app.main`` lifespan);
a malformed manifest raises immediately with a path-qualified error.

Two independent ``${ENV}`` mechanisms exist:

1. A per-gym **override convention**: every gym automatically gets an override
   env var derived from its ``name`` (see ``base_url_env_var``), e.g. the gym
   ``spothub`` is overridable via ``SPOTHUB_BASE_URL``. No manifest authoring is
   needed to opt into this — it is purely name-derived — so a brand-new gym
   works out of the box with zero code changes and can still be redirected (to
   a local/dev deployment, say) without editing its manifest.
2. A generic ``${VAR}`` **expansion escape hatch** (any manifest field, not just
   ``base_url``) for anything the naming convention doesn't cover.

The manifest's own ``base_url`` is the *default* used when the convention env
var is unset — so a gym is fully self-contained (works with zero configuration)
while still being redirectable per-deployment.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from app.adapters.base import GymAdapter, ManifestAdapter
from app.errors import UnknownGymError
from app.schemas import ManifestModel

logger = logging.getLogger("trigger_reply_engine.registry")

_NON_ALNUM = re.compile(r"[^A-Z0-9]")


def base_url_env_var(gym_name: str) -> str:
    """The override env var a gym's base URL is read from, by convention.

    Purely name-derived (``spothub`` -> ``SPOTHUB_BASE_URL``,
    ``sales_explorer`` -> ``SALES_EXPLORER_BASE_URL``), so every gym gets one
    automatically — no manifest field or code change required to add a gym.
    """
    return _NON_ALNUM.sub("_", gym_name.upper()) + "_BASE_URL"


def _expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``$VAR`` in every string leaf."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _resolve_base_url(gym_name: str, manifest_default: str) -> str:
    """Override env var (by convention) if set/non-empty, else the manifest default."""
    override = os.environ.get(base_url_env_var(gym_name))
    return override if override else manifest_default


class Registry:
    """Immutable-after-load map of ``gym_name -> GymAdapter``."""

    def __init__(self, adapters: dict[str, GymAdapter]) -> None:
        self._adapters = adapters

    @classmethod
    def load(cls, manifests_dir: str | Path) -> Registry:
        directory = Path(manifests_dir)
        adapters: dict[str, GymAdapter] = {}

        if not directory.exists():
            logger.warning("Manifests dir %s does not exist; starting with no gyms.", directory)
            return cls(adapters)

        paths = sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")])
        for path in paths:
            manifest = cls._load_one(path)
            if manifest.name in adapters:
                raise ValueError(
                    f"Duplicate gym name '{manifest.name}' (second definition in {path.name})"
                )

            manifest.base_url = _resolve_base_url(manifest.name, manifest.base_url)
            if not manifest.base_url or "${" in manifest.base_url:
                env_var = base_url_env_var(manifest.name)
                logger.warning(
                    "Gym '%s' has no usable base_url (got %r). Set %s or give the "
                    "manifest a literal base_url default.",
                    manifest.name,
                    manifest.base_url,
                    env_var,
                )

            adapters[manifest.name] = ManifestAdapter(manifest)

        return cls(adapters)

    @staticmethod
    def _load_one(path: Path) -> ManifestModel:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Manifest {path.name} is not valid YAML: {exc}") from exc
        try:
            return ManifestModel.model_validate(_expand_env(raw))
        except ValidationError as exc:
            raise ValueError(f"Manifest {path.name} failed validation: {exc}") from exc

    # ── lookups ────────────────────────────────────────────────────────────────

    def get(self, gym_name: str) -> GymAdapter:
        adapter = self._adapters.get(gym_name)
        if adapter is None:
            raise UnknownGymError(
                f"Unknown gym '{gym_name}'", details={"available": self.gyms()}
            )
        return adapter

    def gyms(self) -> list[str]:
        return sorted(self._adapters)

    def __len__(self) -> int:
        return len(self._adapters)
