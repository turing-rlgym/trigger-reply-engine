"""Unit tests for ManifestAdapter: declared-type validation and path encoding.

Uses a small ad hoc manifest (not the bundled ones) since neither spothub nor
sales_explorer currently declares an ``integer``-typed input.
"""

from __future__ import annotations

import pytest

from app.adapters.base import ManifestAdapter
from app.errors import ParamValidationError
from app.schemas import ActionModel, InputSpec, ManifestModel, RequestSpec, StepSpec


def _adapter() -> ManifestAdapter:
    manifest = ManifestModel(
        name="demo",
        base_url="https://demo.example.com",
        actions={
            "widget.create": ActionModel(
                description="Create a widget",
                inputs={
                    "widget_id": InputSpec(type="string", required=True),
                    "count": InputSpec(type="integer", required=True),
                },
                request=RequestSpec(
                    method="POST",
                    path="/widgets/{{ widget_id }}",
                    json_body={"count": "{{ count }}"},
                ),
            )
        },
    )
    return ManifestAdapter(manifest)


def test_integer_type_rejects_non_numeric_string():
    adapter = _adapter()
    with pytest.raises(ParamValidationError) as exc_info:
        adapter.validate_and_fill("widget.create", {"widget_id": "w1", "count": "five"})
    assert any("count" in e and "integer" in e for e in exc_info.value.errors)


def test_integer_type_rejects_bool():
    # bool is a subclass of int in Python; a manifest declaring `integer`
    # almost certainly doesn't mean to accept true/false.
    adapter = _adapter()
    with pytest.raises(ParamValidationError):
        adapter.validate_and_fill("widget.create", {"widget_id": "w1", "count": True})


def test_integer_type_accepts_int():
    adapter = _adapter()
    filled = adapter.validate_and_fill("widget.create", {"widget_id": "w1", "count": 5})
    assert filled["count"] == 5


def test_path_segment_is_url_encoded():
    """A param value can't inject extra path segments or escape into a query string."""
    adapter = _adapter()
    filled = adapter.validate_and_fill("widget.create", {"widget_id": "w1/../admin", "count": 3})
    built = adapter.build_requests("widget.create", filled)
    assert len(built) == 1
    assert built[0].url == "https://demo.example.com/widgets/w1%2F..%2Fadmin"


def _multistep_adapter() -> ManifestAdapter:
    manifest = ManifestModel(
        name="demo",
        base_url="https://demo.example.com",
        actions={
            "widget.create": ActionModel(
                description="Create a widget in two calls",
                inputs={"widget_id": InputSpec(type="string", required=True)},
                steps=[
                    StepSpec(
                        name="create",
                        method="POST",
                        path="/widgets",
                        json_body={"id": "{{ widget_id }}", "corr": "{{ correlation_id }}"},
                    ),
                    StepSpec(
                        name="activate",
                        method="PATCH",
                        path="/widgets/{{ widget_id }}",
                        json_body={
                            "active": True,
                            "corr": "{{ correlation_id }}",
                            "at": "{{ now }}",
                        },
                    ),
                ],
            )
        },
    )
    return ManifestAdapter(manifest)


def test_multistep_shares_one_correlation_id_and_now_across_steps():
    adapter = _multistep_adapter()
    filled = adapter.validate_and_fill("widget.create", {"widget_id": "w1"})
    built = adapter.build_requests("widget.create", filled)
    assert [b.name for b in built] == ["create", "activate"]
    assert built[0].json["corr"] == built[1].json["corr"]
    assert built[1].json["at"]  # rendered, non-empty


def test_multistep_per_step_override_deep_merges_onto_rendered_body():
    adapter = _multistep_adapter()
    filled = adapter.validate_and_fill("widget.create", {"widget_id": "w1"})
    built = adapter.build_requests(
        "widget.create", filled, overrides={"activate": {"note": "manual"}}
    )
    activate = built[1]
    assert activate.json["note"] == "manual"
    # Manifest-templated fields survive the merge (override is additive, not replacing).
    assert activate.json["active"] is True
