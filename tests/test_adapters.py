"""Unit tests for ManifestAdapter: declared-type validation, path encoding,
native JSON rendering (incl. the ``append_unique``/``upsert``/``reject_by``
response-piping filters), and ``build_step``.

Uses small ad hoc manifests (not the bundled ones) to isolate each behavior.
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


# ── native JSON rendering ──────────────────────────────────────────────────────


def _typed_adapter() -> ManifestAdapter:
    manifest = ManifestModel(
        name="demo",
        base_url="https://demo.example.com",
        actions={
            "widget.tag": ActionModel(
                description="Native-typed leaves",
                inputs={
                    "count": InputSpec(type="integer", required=True),
                    "enabled": InputSpec(type="boolean", required=True),
                    "label": InputSpec(type="string", required=True),
                    "note": InputSpec(type="string", required=False),
                },
                request=RequestSpec(
                    method="POST",
                    path="/widgets",
                    json_body={
                        "count": "{{ count }}",
                        "enabled": "{{ enabled }}",
                        "label": "{{ label }}",
                        "note": "{{ note }}",
                        "greeting": "hi {{ label }}",
                    },
                ),
            )
        },
    )
    return ManifestAdapter(manifest)


def test_native_rendering_yields_real_types_for_typed_params():
    """A pure-expression leaf on an integer/boolean param yields that type
    (not its ``str()``) — what makes append_unique/upsert produce real JSON
    structure rather than a stringified one."""
    adapter = _typed_adapter()
    filled = adapter.validate_and_fill(
        "widget.tag", {"count": 5, "enabled": True, "label": "widget-1"}
    )
    built = adapter.build_requests("widget.tag", filled)[0]
    assert built.json["count"] == 5 and isinstance(built.json["count"], int)
    assert built.json["enabled"] is True


def test_native_rendering_never_coerces_a_string_typed_param():
    """A string param whose value happens to look like a Python literal (a
    numeric id, as this domain's client-generated list/message ids often are)
    must stay a string — stock Jinja NativeEnvironment would literal_eval a
    lone-string output and silently turn it into an int."""
    adapter = _typed_adapter()
    filled = adapter.validate_and_fill(
        "widget.tag", {"count": 1, "enabled": False, "label": "1710000000000"}
    )
    built = adapter.build_requests("widget.tag", filled)[0]
    assert built.json["label"] == "1710000000000"
    assert isinstance(built.json["label"], str)
    # A leaf mixing literal text with an expression is unaffected either way.
    assert built.json["greeting"] == "hi 1710000000000"


def test_native_rendering_undefined_optional_leaf_is_empty_string():
    """An optional param the caller omits (no default) must render as ``""``,
    not leak Jinja's ``Undefined`` sentinel (which isn't JSON-serializable)
    into the body — the old string environment did this implicitly; native
    rendering needs the explicit ``finalize`` hook to keep doing it."""
    adapter = _typed_adapter()
    filled = adapter.validate_and_fill(
        "widget.tag", {"count": 1, "enabled": False, "label": "w1"}
    )
    assert "note" not in filled  # no default, not provided -> stays absent
    built = adapter.build_requests("widget.tag", filled)[0]
    assert built.json["note"] == ""


# ── response-piping filters (append_unique / upsert) ──────────────────────────


def _read_modify_write_adapter() -> ManifestAdapter:
    manifest = ManifestModel(
        name="demo",
        base_url="https://demo.example.com",
        actions={
            "list.sync": ActionModel(
                description="Read-modify-write via response piping",
                inputs={"item_id": InputSpec(type="string", required=True)},
                steps=[
                    StepSpec(name="hydrate", internal=True, method="GET", path="/elements"),
                    StepSpec(
                        name="write",
                        method="PUT",
                        path="/elements",
                        # Deliberately not named "items"/"keys"/etc.: those
                        # collide with dict methods under Jinja's `.attr`
                        # (getattr-first) resolution, same as in Python.
                        json_body={
                            "elements": (
                                "{{ steps.hydrate.elements | "
                                "append_unique({'id': item_id}, key='id') }}"
                            )
                        },
                    ),
                ],
            ),
            "crm.set_status": ActionModel(
                description="Upsert a CRM status override",
                inputs={
                    "account_id": InputSpec(type="integer", required=True),
                    "status": InputSpec(type="string", required=True),
                },
                request=RequestSpec(
                    method="PUT",
                    path="/overrides",
                    json_body={
                        "overrides": (
                            "{{ steps.hydrate.overrides | "
                            "upsert(account_id | string, status) }}"
                        )
                    },
                ),
            ),
            "list.remove": ActionModel(
                description="Read-modify-write removal via reject_by",
                inputs={"item_id": InputSpec(type="string", required=True)},
                steps=[
                    StepSpec(name="hydrate", internal=True, method="GET", path="/elements"),
                    StepSpec(
                        name="write",
                        method="PUT",
                        path="/elements",
                        json_body={
                            "elements": "{{ steps.hydrate.elements | reject_by('id', item_id) }}"
                        },
                    ),
                ],
            ),
        },
    )
    return ManifestAdapter(manifest)


def test_build_step_resolved_steps_flag_hydrate_internal():
    adapter = _read_modify_write_adapter()
    filled = adapter.validate_and_fill("list.sync", {"item_id": "x2"})
    ctx = {**filled, "steps": {}}
    hydrate = adapter.build_step("list.sync", 0, ctx)
    write = adapter.build_step("list.sync", 1, ctx)
    assert hydrate.internal is True
    assert write.internal is False


def test_append_unique_with_no_prior_response_treats_it_as_empty():
    """No piped response yet (e.g. the back-compat build_requests wrapper,
    or a dry_run preview) renders against an empty `steps` — the filter
    treats that as an empty collection rather than raising."""
    adapter = _read_modify_write_adapter()
    filled = adapter.validate_and_fill("list.sync", {"item_id": "x2"})
    built = adapter.build_requests("list.sync", filled)
    assert built[1].json["elements"] == [{"id": "x2"}]


def test_append_unique_merges_and_dedupes_piped_response_by_key():
    adapter = _read_modify_write_adapter()
    filled = adapter.validate_and_fill("list.sync", {"item_id": "x2"})

    piped = {**filled, "steps": {"hydrate": {"elements": [{"id": "x1"}]}}}
    built = adapter.build_step("list.sync", 1, piped)
    assert built.json["elements"] == [{"id": "x1"}, {"id": "x2"}]

    # Idempotent: re-adding an item already present in the piped response
    # does not duplicate it.
    already_there = {**filled, "steps": {"hydrate": {"elements": [{"id": "x2"}]}}}
    built_dup = adapter.build_step("list.sync", 1, already_there)
    assert built_dup.json["elements"] == [{"id": "x2"}]


def test_upsert_sets_key_on_piped_dict_without_dropping_other_keys():
    adapter = _read_modify_write_adapter()
    filled = adapter.validate_and_fill(
        "crm.set_status", {"account_id": 7, "status": "Customer"}
    )
    piped = {**filled, "steps": {"hydrate": {"overrides": {"3": "Lead"}}}}
    built = adapter.build_step("crm.set_status", 0, piped)
    assert built.json["overrides"] == {"3": "Lead", "7": "Customer"}


def test_reject_by_drops_matching_entry_from_piped_response():
    adapter = _read_modify_write_adapter()
    filled = adapter.validate_and_fill("list.remove", {"item_id": "x2"})
    piped = {
        **filled,
        "steps": {"hydrate": {"elements": [{"id": "x1"}, {"id": "x2"}, {"id": "x3"}]}},
    }
    built = adapter.build_step("list.remove", 1, piped)
    assert built.json["elements"] == [{"id": "x1"}, {"id": "x3"}]


def test_reject_by_with_no_prior_response_treats_it_as_empty():
    """An Undefined piped seq (no hydrate response yet, e.g. build_requests'
    back-compat wrapper or a dry_run preview) must not raise — there's
    nothing to reject from, so the result is an empty list."""
    adapter = _read_modify_write_adapter()
    filled = adapter.validate_and_fill("list.remove", {"item_id": "x2"})
    built = adapter.build_requests("list.remove", filled)
    assert built[1].json["elements"] == []


def test_reject_by_is_a_noop_when_the_item_is_not_present():
    adapter = _read_modify_write_adapter()
    filled = adapter.validate_and_fill("list.remove", {"item_id": "missing"})
    piped = {**filled, "steps": {"hydrate": {"elements": [{"id": "x1"}]}}}
    built = adapter.build_step("list.remove", 1, piped)
    assert built.json["elements"] == [{"id": "x1"}]
