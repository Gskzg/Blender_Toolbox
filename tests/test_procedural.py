"""Pure-Python tests for the Toolbox procedural recipe IR."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from blender_toolbox.addon import ExecutorError, _inject_request_seed, _parameters_sample  # noqa: E402
from blender_toolbox.procedural import (  # noqa: E402
    PROCEDURAL_RECIPE_SCHEMA,
    RECIPE_SCHEMA_VERSION,
    RecipeError,
    normalize_recipe,
    recipe_hash,
)
from blender_toolbox.protocol import ActionRequest, ProtocolError, validate_action_args  # noqa: E402


def _recipe(*, reverse: bool = False) -> dict[str, object]:
    nodes = [
        {"id": "output", "type": "GeometryNodeJoinGeometry", "inputs": {}},
        {"id": "source", "type": "GeometryNodeMeshCube", "inputs": {"Size": 1.0}},
    ]
    links = [
        {"from_node": "source", "from_socket": "Mesh", "to_node": "output", "to_socket": "Geometry"},
        {"from_node": "GroupInput", "from_socket": "Geometry", "to_node": "output", "to_socket": "Geometry"},
    ]
    if reverse:
        nodes.reverse()
        links.reverse()
    return {
        "schema_version": RECIPE_SCHEMA_VERSION,
        "name": "cube_recipe",
        "kind": "geometry_nodes",
        "interface": [
            {"name": "Geometry", "in_out": "OUTPUT", "socket_type": "NodeSocketGeometry"},
            {"name": "Geometry", "in_out": "INPUT", "socket_type": "NodeSocketGeometry"},
        ],
        "nodes": nodes,
        "links": links,
    }


def test_normalization_is_order_independent_and_hashed() -> None:
    first = normalize_recipe(_recipe())
    second = normalize_recipe(_recipe(reverse=True))

    assert first.as_dict() == second.as_dict()
    assert first.graph_hash == second.graph_hash == recipe_hash(first)
    assert [node.id for node in first.nodes] == ["output", "source"]
    assert first.as_dict()["interface"][0]["in_out"] == "INPUT"


def test_normalization_allows_reserved_group_endpoints() -> None:
    normalized = normalize_recipe(_recipe())
    assert normalized.links[0].from_node == "GroupInput"

    implicit = normalize_recipe({"nodes": [{"id": "source", "type": "GeometryNodeMeshCube"}]})
    assert [item["in_out"] for item in implicit.as_dict()["interface"]] == ["INPUT", "OUTPUT"]


def test_recipe_preserves_explicit_attributes_and_multi_input_order() -> None:
    recipe = normalize_recipe({
        "kind": "geometry_nodes",
        "interface": [
            {"name": "Geometry", "in_out": "OUTPUT", "socket_type": "NodeSocketGeometry", "index": 1},
            {"name": "Geometry", "in_out": "INPUT", "socket_type": "NodeSocketGeometry", "index": 0},
        ],
        "nodes": [
            {"id": "join", "type": "GeometryNodeJoinGeometry", "attributes": {}},
            {"id": "a", "type": "GeometryNodeMeshCube", "attributes": {"label_mode": "NONE"}},
        ],
        "links": [
            {"from_node": "a", "from_socket": "Mesh", "to_node": "join", "to_socket": "Geometry", "order": 1},
        ],
    })
    assert recipe.nodes[0].id == "a"
    assert recipe.nodes[0].as_dict()["attributes"] == {"label_mode": "NONE"}
    assert recipe.links[0].order == 1
    assert recipe.as_dict()["interface"][0]["index"] == 0


def test_policy_allowlist_is_checked_before_execution() -> None:
    with pytest.raises(RecipeError, match="not allowlisted") as error:
        normalize_recipe(_recipe(), allowed_node_types={"GeometryNodeMeshCube"})
    assert error.value.code == "policy_denied"


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda value: value["nodes"].append({"id": "source", "type": "GeometryNodeMeshCube"}), "duplicate"),
        (lambda value: value["links"].append(dict(value["links"][0])), "duplicate"),
        (lambda value: value["links"].__setitem__(0, {"from_node": "missing", "from_socket": "Mesh", "to_node": "output", "to_socket": "Geometry"}), "unknown node"),
        (lambda value: value["nodes"].append({"id": "GroupInput", "type": "GeometryNodeMeshCube"}), "reserved"),
    ],
)
def test_normalization_rejects_malformed_graph(mutator, message: str) -> None:
    value = _recipe()
    mutator(value)
    with pytest.raises(RecipeError, match=message):
        normalize_recipe(value)


def test_normalization_rejects_cycles_and_non_finite_values() -> None:
    value = _recipe()
    value["links"] = [
        {"from_node": "source", "from_socket": "Mesh", "to_node": "output", "to_socket": "Geometry"},
        {"from_node": "output", "from_socket": "Geometry", "to_node": "source", "to_socket": "Mesh"},
    ]
    with pytest.raises(RecipeError, match="cycle"):
        normalize_recipe(value)

    value = _recipe()
    value["nodes"][1]["inputs"]["Size"] = math.nan
    with pytest.raises(RecipeError, match="finite"):
        normalize_recipe(value)


def test_schema_requires_nodes_and_is_strict() -> None:
    assert PROCEDURAL_RECIPE_SCHEMA["required"] == ["nodes"]
    assert PROCEDURAL_RECIPE_SCHEMA["additionalProperties"] is False
    assert PROCEDURAL_RECIPE_SCHEMA["properties"]["nodes"]["maxItems"] == 256


def test_protocol_exposes_recipe_action_with_nested_required_fields() -> None:
    validate_action_args("geometry_nodes.apply_recipe", {"target": "obj-1", "recipe": _recipe()})
    with pytest.raises(ProtocolError, match="missing required"):
        validate_action_args("geometry_nodes.apply_recipe", {"target": "obj-1", "recipe": {}})


def test_action_request_canonicalizes_recipe_before_transport() -> None:
    raw = {
        "session_id": "session-test",
        "episode_id": "episode-test",
        "step_id": 0,
        "action": "geometry_nodes.apply_recipe",
        "args": {"target": "obj-1", "recipe": _recipe(reverse=True)},
    }
    request = ActionRequest.from_dict(raw)
    canonical = request.args["recipe"]
    assert canonical["nodes"][0]["id"] == "output"
    assert canonical["links"][0]["from_node"] == "GroupInput"

    material = ActionRequest.from_dict(
        {
            "session_id": "session-test",
            "episode_id": "episode-test",
            "step_id": 0,
            "action": "material.apply_recipe",
            "args": {
                "name": "Material",
                "recipe": {
                    "kind": "material",
                    "name": "Material",
                    "nodes": [{"id": "output", "type": "ShaderNodeOutputMaterial"}],
                },
            },
        }
    )
    assert material.args["recipe"]["kind"] == "material"


def test_parameter_sampling_is_seeded_and_hashable() -> None:
    args = {"distribution": "triangular", "seed": 23, "count": 8, "low": 0.1, "high": 0.9, "mode": 0.4}
    first = _parameters_sample(args)
    second = _parameters_sample(dict(args))

    assert first == second
    assert first["hash"].startswith("sha256:")
    assert len(first["values"]) == 8
    assert all(0.1 <= value <= 0.9 for value in first["values"])

    integer = _parameters_sample({"distribution": "integer", "seed": 5, "count": 10, "low": 2, "high": 4})
    assert all(value in {2, 3, 4} for value in integer["values"])

    with pytest.raises(ExecutorError, match="positive low/high"):
        _parameters_sample({"distribution": "log_uniform", "low": 0, "high": 1})


def test_episode_seed_fills_random_action_seed_without_overriding_explicit_seed() -> None:
    request = ActionRequest.from_dict(
        {
            "session_id": "session",
            "episode_id": "episode",
            "step_id": 1,
            "action": "parameters.sample",
            "seed": 91,
            "args": {"distribution": "uniform"},
        }
    )
    assert _inject_request_seed("parameters.sample", request.args, request)["seed"] == 91
    assert _inject_request_seed("parameters.sample", {"distribution": "uniform", "seed": 7}, request)["seed"] == 7
