#!/usr/bin/env python3
"""Exercise the native typed procedural recipe actions inside Blender."""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import bpy
except ImportError:  # pragma: no cover - intended for Blender's Python.
    print("smoke_recipe.py must run with Blender's Python", file=sys.stderr)
    raise SystemExit(2)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blender_toolbox.addon import ToolboxExecutor
from blender_toolbox.protocol import SCHEMA_VERSION


def main() -> int:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    executor = ToolboxExecutor()
    session = "recipe-session"
    episode = "recipe-episode"
    step = 0

    def call(action: str, args: dict) -> dict:
        nonlocal step
        payload = {
            "schema_version": SCHEMA_VERSION,
            "request_id": f"req-{step}",
            "session_id": session,
            "episode_id": episode,
            "step_id": step,
            "expected_revision": executor.revision,
            "action": action,
            "args": args,
        }
        step += 1
        return executor.execute(payload)

    opened = call("session.open", {"mode": "new", "reset": True})
    assert opened["ok"], opened
    created = call("object.create", {"kind": "cube", "name": "RecipeTarget", "id": "recipe.target"})
    assert created["ok"], created
    target = created["result"]["ref"]

    geometry_recipe = {
        "schema_version": "blender_toolbox.procedural_recipe.v1",
        "kind": "geometry_nodes",
        "name": "RecipeGeometry",
        "interface": [
            {"name": "Geometry", "in_out": "INPUT", "socket_type": "NodeSocketGeometry"},
            {"name": "Geometry", "in_out": "OUTPUT", "socket_type": "NodeSocketGeometry"},
        ],
        "nodes": [
            {"id": "cube", "type": "GeometryNodeMeshCube", "inputs": {"Size": [1.0, 1.0, 1.0]}},
            {"id": "join", "type": "GeometryNodeJoinGeometry"},
        ],
        "links": [
            {"from_node": "cube", "from_socket": "Mesh", "to_node": "join", "to_socket": "Geometry"},
            {"from_node": "join", "from_socket": "Geometry", "to_node": "GroupOutput", "to_socket": "Geometry"},
        ],
    }
    applied = call("geometry_nodes.apply_recipe", {"target": target, "recipe": geometry_recipe})
    assert applied["ok"], applied
    assert applied["result"]["recipe_hash"].startswith("sha256:"), applied
    before_graph = call("inspect.geometry_nodes", {"target": target})
    assert before_graph["ok"], before_graph

    invalid_geometry = dict(geometry_recipe)
    invalid_geometry["links"] = [
        {"from_node": "cube", "from_socket": "missing", "to_node": "join", "to_socket": "Geometry"},
    ]
    failed_geometry = call("geometry_nodes.apply_recipe", {"target": target, "recipe": invalid_geometry})
    assert not failed_geometry["ok"], failed_geometry
    after_graph = call("inspect.geometry_nodes", {"target": target})
    assert after_graph["ok"], after_graph
    assert after_graph["result"] == before_graph["result"], (before_graph, after_graph)

    material_recipe = {
        "schema_version": "blender_toolbox.procedural_recipe.v1",
        "kind": "material",
        "name": "RecipeMaterial",
        "nodes": [
            {"id": "principled", "type": "ShaderNodeBsdfPrincipled", "inputs": {"Roughness": 0.25}},
            {"id": "output", "type": "ShaderNodeOutputMaterial"},
        ],
        "links": [
            {"from_node": "principled", "from_socket": "BSDF", "to_node": "output", "to_socket": "Surface"},
        ],
    }
    applied_material = call("material.apply_recipe", {"name": "RecipeMaterial", "recipe": material_recipe})
    assert applied_material["ok"], applied_material
    assert applied_material["result"]["graph_hash"].startswith("sha256:"), applied_material
    material_nodes = sorted(node.name for node in bpy.data.materials["RecipeMaterial"].node_tree.nodes)

    invalid_material = dict(material_recipe)
    invalid_material["links"] = [
        {"from_node": "principled", "from_socket": "missing", "to_node": "output", "to_socket": "Surface"},
    ]
    failed_material = call("material.apply_recipe", {"name": "RecipeMaterial", "recipe": invalid_material})
    assert not failed_material["ok"], failed_material
    assert sorted(node.name for node in bpy.data.materials["RecipeMaterial"].node_tree.nodes) == material_nodes

    print(json.dumps({"status": "ok", "revision": executor.revision, "target": target}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
