"""Application configuration via pydantic-settings.

All values are overridable via environment variables (see ``.env.example`` and
``docker-compose.yml``). Gym base URLs are deliberately NOT settings fields
here: each gym gets an override env var derived from its manifest ``name`` by
convention (``app.registry.base_url_env_var``, e.g. ``SPOTHUB_BASE_URL``), and
the manifest's own ``base_url`` is the default used when that env var is unset.
This is what lets a brand-new gym work with zero code changes — see
``app.registry``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = parent of the ``app`` package. Used as the default location of the
# ``manifests/`` directory for local (source) runs. In the container this is
# overridden via ``MANIFESTS_DIR`` (see the Dockerfile), mirroring forcesales'
# ``SEED_DIR`` pattern so the data dir is decoupled from the installed package.
REPO_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Runtime configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Directory holding gym manifest YAML files.
    manifests_dir: Path = REPO_ROOT / "manifests"

    # ── LLM (reply generation) ────────────────────────────────────────────────
    # When ``use_llm`` is false (default) or the key is empty, the engine uses the
    # deterministic template stub so the service runs with no network/keys.
    use_llm: bool = False
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    llm_timeout: float = 20.0

    # ── Outbound HTTP (calls to the target gym) ───────────────────────────────
    request_timeout: float = 15.0
    max_retries: int = 2
    retry_backoff: float = 0.5

    # ── Idempotency ───────────────────────────────────────────────────────────
    # Seconds a completed trigger keyed by ``idempotency_key`` is remembered.
    # In-memory only (single-instance); a shared store would be needed to scale out.
    idempotency_ttl: float = 900.0
    # Upper bound on the number of remembered keys (oldest evicted past this),
    # so a stream of one-off idempotency keys can't grow memory unbounded.
    idempotency_max_entries: int = 10_000

    # CORS origins ("*" or a comma-separated list).
    cors_origins: str = "*"

    # ── Auth ───────────────────────────────────────────────────────────────────
    # Shared secret required on the ``X-API-Key`` header for /trigger and the
    # introspection routes. Empty (default) disables auth — fine for local dev,
    # but should be set before deploying anywhere reachable off-box.
    api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
