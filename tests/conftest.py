"""Shared test fixtures.

Gym base-URL env vars are set *before* the app is imported so the manifests'
``${...}`` references resolve to fixed, mockable hosts. The LLM is left off so
generation uses the deterministic stub (no network/keys in tests).
"""

from __future__ import annotations

import os

os.environ["SPOTHUB_BASE_URL"] = "http://spothub.test"
os.environ["SALES_EXPLORER_BASE_URL"] = "http://sales.test"
os.environ["USE_LLM"] = "0"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

SPOTHUB_URL = "http://spothub.test"
SALES_URL = "http://sales.test"


@pytest.fixture
def client():
    # Entering the context runs the lifespan: loads manifests, builds the engine.
    with TestClient(app) as test_client:
        yield test_client
