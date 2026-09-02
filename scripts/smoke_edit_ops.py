#!/usr/bin/env python3
"""Headless smoke for high-frequency mesh edit actions and modifiers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from blender_toolbox.addon import ToolboxExecutor
from blender_toolbox.protocol import SCHEMA_VERSION, TOOL_SPECS


def main() -> int:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    executor = ToolboxExecutor()
    step = 0

    def call(action: str, args: dict) -> dict:
        nonlocal step
        payload = {"schema_version": SCHEMA_VERSION, "request_id": f"r{step}", "session_id": "s", "episode_id": "e", "step_id": step, "expected_revision": executor.revision, "action": action, "args": args}
        step += 1
        response = executor.execute(payload)
        assert response["ok"], response
        return response["result"]

    call("session.open", {"mode": "new", "reset": True, "quality_profile": "advisory"})
    target = call("object.create", {"kind": "cube", "id": "cube"})["ref"]
    call("mesh.duplicate_region", {"target": target, "selection": {"face_indices": [0]}, "offset": [0, 0, 1]})
    call("mesh.extrude_individual", {"target": target, "selection": {"face_indices": [1]}, "distance": 0.1})
    call("mesh.inset_individual", {"target": target, "selection": {"face_indices": [0]}, "thickness": 0.05})
    call("mesh.loop_cut", {"target": target, "selection": {"edge_indices": [0, 1, 2, 3]}, "cuts": 1})
    call("geometry.add_modifier", {"target": target, "modifier_type": "BEVEL", "properties": {"width": 0.02, "segments": 2}})
    assert "mesh.duplicate_region" in TOOL_SPECS and "mesh.loop_cut" in TOOL_SPECS
    print(json.dumps({"status": "ok", "objects": len(bpy.context.scene.objects), "revision": executor.revision}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
