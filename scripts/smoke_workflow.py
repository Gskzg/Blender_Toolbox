#!/usr/bin/env python3
"""Headless smoke test for the task-facing workflow helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import bpy
except ImportError:  # pragma: no cover
    print("smoke_workflow.py must run with Blender's Python", file=sys.stderr)
    raise SystemExit(2)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blender_toolbox.addon import ToolboxExecutor
from blender_toolbox.protocol import SCHEMA_VERSION, tool_registry


def main() -> int:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    executor = ToolboxExecutor()
    session = "workflow-session"
    episode = "workflow-episode"
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
        response = executor.execute(payload)
        assert response.get("ok"), response
        return response

    def call_raw(action: str, args: dict) -> dict:
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

    opened = call("session.open", {
        "mode": "new",
        "reset": True,
        "profile": "vehicle",
        "include_capabilities": True,
        "include_examples": False,
    })
    assert opened["result"]["contract"]["up_axis"] == "Z", opened
    assert opened["result"]["capabilities"]["selected_workflow"]["name"] == "vehicle", opened
    created = call("object.create_batch", {
        "objects": [
            {"kind": "cube", "id": "vehicle.body", "name": "Body", "scale": [2, 1, 0.4], "semantic_tags": ["vehicle", "body"]},
            {"kind": "cylinder", "id": "vehicle.wheel.front_left", "name": "Wheel_FL", "vertices": 32, "semantic_tags": ["vehicle", "wheel", "tire"]},
        ],
    })
    assert created["result"]["successful"] == 2, created
    body_ref = "vehicle.body"
    wheel_ref = "vehicle.wheel.front_left"
    call("material.create", {"name": "Blue", "base_color": [0.03, 0.12, 0.6, 1], "metallic": 0.5, "roughness": 0.25})
    assigned = call("material.assign_batch", {"assignments": [{"target": body_ref, "material": "Blue"}, {"target": wheel_ref, "material": "Blue"}]})
    assert assigned["result"]["successful"] == 2, assigned
    stacked = call("geometry.modifier_stack", {
        "target": body_ref,
        "modifiers": [{"modifier_type": "BEVEL", "name": "BodyBevel", "properties": {"width": 0.1, "segments": 3}, "apply": True}],
    })
    assert stacked["result"]["successful"] == 1, stacked
    mesh_count = len(bpy.data.meshes)
    unnamed_stack = call("geometry.modifier_stack", {
        "target": wheel_ref,
        "modifiers": [{"modifier_type": "BEVEL"}],
    })
    assert unnamed_stack["result"]["modifiers"][0]["added"]["name"] != "None", unnamed_stack
    assert len(bpy.data.meshes) == mesh_count, (mesh_count, len(bpy.data.meshes))
    failed_stack = call_raw("geometry.modifier_stack", {
        "target": wheel_ref,
        "atomic": True,
        "modifiers": [
            {"modifier_type": "BEVEL", "name": "Temporary"},
            {"modifier_type": "NOT_A_MODIFIER"},
        ],
    })
    assert failed_stack.get("ok") is False, failed_stack
    assert len(bpy.data.meshes) == mesh_count, (mesh_count, len(bpy.data.meshes))
    inspected = call("inspect.batch", {"targets": [body_ref, wheel_ref]})
    assert inspected["result"]["count"] == 2, inspected
    capabilities = call("toolbox.capabilities", {"profile": "vehicle"})
    assert capabilities["result"]["registry_count"] == len(tool_registry()), capabilities
    assert "examples" not in capabilities["result"]["selected_workflow"], capabilities
    described = call("workflow.describe", {"name": "vehicle"})
    assert described["result"]["workflow"]["name"] == "vehicle", described
    assert "examples" not in described["result"]["workflow"], described
    print(json.dumps({"status": "ok", "registry": len(tool_registry()), "objects": inspected["result"]["count"], "revision": executor.revision}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
