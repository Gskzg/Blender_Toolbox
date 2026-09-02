"""Protocol contract tests that do not require a Blender installation."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from blender_toolbox.addon import ExecutorError, _artifact_destination  # noqa: E402
from blender_toolbox.protocol import (  # noqa: E402
    MAX_SEED,
    ActionRequest,
    ProtocolError,
    ToolSpec,
    get_tool_spec,
    request_fingerprint,
    tool_registry,
    validate_action_args,
)


def _request(*, action: str, args: dict[str, object]) -> dict[str, object]:
    return {
        "session_id": "session-test",
        "episode_id": "episode-test",
        "step_id": 0,
        "action": action,
        "args": args,
    }


def test_registry_required_args_match_input_schema() -> None:
    """Clients and the executor must see the same required arguments."""
    for entry in tool_registry():
        assert entry["input_schema"]["required"] == entry["required_args"], entry["name"]

        spec = get_tool_spec(entry["name"])
        assert list(spec.input_schema["required"]) == list(spec.required_args), entry["name"]


def test_required_normalization_preserves_explicit_schema_constraints() -> None:
    spec = ToolSpec(
        name="test.custom",
        description="test",
        required_args=("target",),
        input_schema={
            "type": "object",
            "properties": {
                "target": {"type": "string", "minLength": 1},
                "nested": {"type": "object", "required": ["value"]},
            },
            # A stale top-level value is normalized from required_args.
            "required": ["legacy"],
            "additionalProperties": False,
        },
    )

    assert spec.input_schema["required"] == ["target"]
    assert spec.input_schema["properties"]["target"]["minLength"] == 1
    assert spec.input_schema["properties"]["nested"]["required"] == ["value"]
    assert spec.as_dict()["input_schema"]["required"] == ["target"]


def test_tool_spec_schema_trees_are_isolated_from_callers() -> None:
    source_input = {
        "type": "object",
        "properties": {"nested": {"type": "object", "properties": {"value": {"type": "string"}}}},
    }
    source_output = {"type": "object", "properties": {"result": {"type": "array", "items": {"type": "string"}}}}
    spec = ToolSpec(
        name="test.deep_copy",
        description="test",
        input_schema=source_input,
        output_schema=source_output,
    )

    # Mutating the constructor input must not alter the frozen spec.
    source_input["properties"]["nested"]["properties"]["value"]["type"] = "integer"
    source_output["properties"]["result"]["items"]["type"] = "number"
    assert spec.input_schema["properties"]["nested"]["properties"]["value"]["type"] == "string"
    assert spec.output_schema["properties"]["result"]["items"]["type"] == "string"

    # Nor may a registry/listing consumer mutate the spec through as_dict().
    advertised = spec.as_dict()
    advertised["input_schema"]["properties"]["nested"]["properties"]["value"]["type"] = "boolean"
    advertised["output_schema"]["properties"]["result"]["items"]["type"] = "integer"
    assert spec.input_schema["properties"]["nested"]["properties"]["value"]["type"] == "string"
    assert spec.output_schema["properties"]["result"]["items"]["type"] == "string"


def test_registry_schema_listing_isolated_from_mutation() -> None:
    listing = tool_registry()
    entry = next(item for item in listing if item["name"] == "object.create")
    entry["input_schema"]["properties"]["kind"]["enum"].append("evil")
    entry["output_schema"]["properties"]["uuid"]["minLength"] = 999
    fresh = get_tool_spec("object.create").as_dict()
    assert "evil" not in fresh["input_schema"]["properties"]["kind"]["enum"]
    assert fresh["output_schema"]["properties"]["uuid"]["minLength"] == 1


def test_action_request_rejects_missing_required_argument() -> None:
    with pytest.raises(ProtocolError, match="missing required args"):
        ActionRequest.from_dict(_request(action="artifact.export_glb", args={}))


@pytest.mark.parametrize("request_id", ["", None, 0, False])
def test_action_request_rejects_explicit_invalid_request_id(request_id: object) -> None:
    with pytest.raises(ProtocolError, match="request_id must be a non-empty string"):
        ActionRequest.from_dict({**_request(action="model.plan", args={}), "request_id": request_id})


def test_action_request_accepts_valid_required_arguments() -> None:
    request = ActionRequest.from_dict(_request(action="artifact.export_glb", args={"path": "/tmp/scene.glb"}))

    assert request.action == "artifact.export_glb"
    assert request.args == {"path": "/tmp/scene.glb"}


def test_particles_scatter_requires_instance_reference() -> None:
    with pytest.raises(ProtocolError, match="missing required args"):
        ActionRequest.from_dict(_request(action="particles.scatter", args={"target": "mesh-1"}))


def test_high_frequency_edit_and_boolean_modifier_actions_are_discoverable() -> None:
    for action in (
        "mesh.duplicate_region",
        "mesh.extrude_individual",
        "mesh.inset_individual",
        "mesh.bridge_edge_loops",
        "mesh.loop_cut",
    ):
        assert get_tool_spec(action).mutating is True
    modifier_schema = get_tool_spec("geometry.add_modifier").input_schema
    assert "BOOLEAN" in modifier_schema["properties"]["modifier_type"]["enum"]


def test_artifact_root_policy_rejects_escape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLENDER_TOOLBOX_ARTIFACT_ROOTS", str(tmp_path))
    with pytest.raises(ExecutorError, match="outside configured artifact roots"):
        _artifact_destination(str(tmp_path.parent / "escape.glb"), ".glb", "export")


@pytest.mark.parametrize("seed", [-1, MAX_SEED + 1])
def test_action_request_rejects_out_of_range_episode_seed(seed: int) -> None:
    with pytest.raises(ProtocolError, match="seed must be between"):
        ActionRequest.from_dict({**_request(action="model.plan", args={}), "seed": seed})


def test_quality_first_actions_are_discoverable() -> None:
    plan = get_tool_spec("model.plan")
    audit = get_tool_spec("inspect.quality")
    assert plan.mutating is False
    assert audit.mutating is False
    assert "quality_profile" in get_tool_spec("session.open").input_schema["properties"]
    assert "quality_contract" in get_tool_spec("session.open").input_schema["properties"]


def test_sections_schema_accepts_custom_profiles_and_section_frames() -> None:
    """Custom continuous carriers stay discoverable without a domain recipe."""
    validate_action_args(
        "mesh.from_sections",
        {
            "profile": "custom",
            "profile_points": [[-1, -1], [1, -1], [1, 1], [-1, 1]],
            "rotation_euler": [0.0, 0.1, -0.2],
            "center_offset": [0.2, -0.1],
            "sections": [
                {
                    "x": 0.0,
                    "width": 1.0,
                    "height": 1.0,
                    "rotation_x": 0.25,
                    "center": [0.1, 0.2],
                    "profile_points": [
                        {"y": -1, "z": -1},
                        {"y": 1, "z": -1},
                        {"y": 1, "z": 1},
                        {"y": -1, "z": 1},
                    ],
                },
                {"x": 1.0, "width": 1.0, "height": 1.0},
                {"x": 2.0, "width": 1.0, "height": 1.0},
            ],
        },
    )


def test_sections_schema_rejects_unknown_profile_name() -> None:
    with pytest.raises(ProtocolError, match="profile must be one of"):
        validate_action_args(
            "mesh.from_sections",
            {
                "profile": "vehicle_specific",
                "sections": [
                    {"x": 0.0, "width": 1.0, "height": 1.0},
                    {"x": 1.0, "width": 1.0, "height": 1.0},
                    {"x": 2.0, "width": 1.0, "height": 1.0},
                ],
            },
        )


def test_action_request_rejects_non_json_values_even_in_extra_fields() -> None:
    with pytest.raises(ProtocolError, match="finite numbers"):
        ActionRequest.from_dict(_request(action="model.plan", args={"task_spec": {"weight": math.nan}}))

    with pytest.raises(ProtocolError, match="object keys must be strings"):
        ActionRequest.from_dict(_request(action="model.plan", args={"task_spec": {1: "invalid"}}))

    with pytest.raises(ProtocolError, match="idempotency_key"):
        ActionRequest.from_dict({**_request(action="model.plan", args={}), "idempotency_key": 7})


def test_request_fingerprint_is_canonical_and_excludes_transport_ids() -> None:
    first = ActionRequest.from_dict(
        {
            **_request(action="object.create", args={"name": "Body", "kind": "cube"}),
            "request_id": "req-first",
            "idempotency_key": "retry-key",
            "expected_revision": 3,
            "seed": 17,
        }
    )
    reordered = ActionRequest.from_dict(
        {
            **_request(action="object.create", args={"kind": "cube", "name": "Body"}),
            "request_id": "req-retry",
            "idempotency_key": "retry-key",
            "expected_revision": 3,
            "seed": 17,
        }
    )
    changed = ActionRequest.from_dict(
        {
            **_request(action="object.create", args={"kind": "cube", "name": "Other"}),
            "request_id": "req-other",
            "idempotency_key": "retry-key",
            "expected_revision": 3,
            "seed": 17,
        }
    )

    assert request_fingerprint(first) == request_fingerprint(reordered)
    assert request_fingerprint(first) != request_fingerprint(changed)
