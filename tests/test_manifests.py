"""Manifest loading + validation, and base_url resolution (convention + escape hatch)."""

from __future__ import annotations

import os

import pytest

from app.config import REPO_ROOT
from app.registry import Registry, base_url_env_var


def test_bundled_manifests_load():
    registry = Registry.load(REPO_ROOT / "manifests")
    assert registry.gyms() == ["sales_explorer", "spothub"]

    spothub = registry.get("spothub")
    assert "email.reply" in spothub.actions()
    assert spothub.generates_for("email.reply") == ["description"]
    # SPOTHUB_BASE_URL is set by conftest, so it overrides the manifest default
    # by convention (see base_url_env_var).
    assert spothub.base_url == "http://spothub.test"


def test_base_url_env_var_convention():
    assert base_url_env_var("spothub") == "SPOTHUB_BASE_URL"
    assert base_url_env_var("sales_explorer") == "SALES_EXPLORER_BASE_URL"
    # Any punctuation in a future gym name still maps to a clean env var.
    assert base_url_env_var("my-new.gym") == "MY_NEW_GYM_BASE_URL"


def test_base_url_default_from_manifest(tmp_path):
    """No env var set -> the manifest's literal base_url is used as-is."""
    (tmp_path / "demo.yaml").write_text(
        "name: demo\nbase_url: https://demo.example.com\nactions: {}\n",
        encoding="utf-8",
    )
    assert "DEMO_BASE_URL" not in os.environ
    registry = Registry.load(tmp_path)
    assert registry.get("demo").base_url == "https://demo.example.com"


def test_base_url_env_override_by_convention(tmp_path):
    """Setting <NAME>_BASE_URL overrides the manifest default — no code change."""
    (tmp_path / "demo.yaml").write_text(
        "name: demo\nbase_url: https://demo.example.com\nactions: {}\n",
        encoding="utf-8",
    )
    os.environ["DEMO_BASE_URL"] = "http://overridden.test"
    try:
        registry = Registry.load(tmp_path)
    finally:
        del os.environ["DEMO_BASE_URL"]
    assert registry.get("demo").base_url == "http://overridden.test"


def test_generic_env_expansion_escape_hatch(tmp_path):
    """A non-convention ${VAR} in base_url is still expanded (the escape hatch)."""
    (tmp_path / "demo.yaml").write_text(
        "name: demo\nbase_url: ${CUSTOM_TARGET}\nactions: {}\n",
        encoding="utf-8",
    )
    os.environ["CUSTOM_TARGET"] = "http://custom.test"
    try:
        registry = Registry.load(tmp_path)
    finally:
        del os.environ["CUSTOM_TARGET"]
    assert registry.get("demo").base_url == "http://custom.test"


def test_invalid_manifest_raises(tmp_path):
    # Missing the required `name` field (base_url is optional now).
    (tmp_path / "broken.yaml").write_text("base_url: http://x.test\n", encoding="utf-8")
    with pytest.raises(ValueError, match="broken.yaml"):
        Registry.load(tmp_path)


def test_duplicate_gym_name_raises(tmp_path):
    body = "base_url: http://x.test\nactions: {}\n"
    (tmp_path / "a.yaml").write_text(f"name: dup\n{body}", encoding="utf-8")
    (tmp_path / "b.yaml").write_text(f"name: dup\n{body}", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate gym name"):
        Registry.load(tmp_path)
