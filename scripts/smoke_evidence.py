#!/usr/bin/env python3
"""Regression smoke for opt-in rendered evidence lifecycle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import bpy
except ImportError:  # pragma: no cover
    raise SystemExit("smoke_evidence.py must run with Blender's Python")

ROOT = Path(__file__).resolve().parent
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from blender_toolbox.addon import ToolboxExecutor
from blender_toolbox.protocol import SCHEMA_VERSION


def main() -> int:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    executor = ToolboxExecutor()
    step = 0

    def call(action: str, args: dict, *, expect_ok: bool = True) -> dict:
        nonlocal step
        payload = {
            "schema_version": SCHEMA_VERSION,
            "request_id": f"evidence-smoke-{step}",
            "session_id": "evidence-smoke-session",
            "episode_id": "evidence-smoke-episode",
            "step_id": step,
            "expected_revision": executor.revision,
            "action": action,
            "args": args,
        }
        step += 1
        response = executor.execute(payload)
        assert bool(response.get("ok")) is expect_ok, response
        return response

    contract = {
        "identity": {"asset": "render evidence probe"},
        "scale": {"units": "m", "dimensions": [2, 2, 2]},
        "representation": {"kind": "control_mesh", "carrier": "control_mesh", "primary_refs": ["carrier"]},
        "primary_refs": ["carrier"],
        "reference_views": ["front", "side", "top"],
        "evidence": {"require_render": True, "min_views": 3},
    }
    call("session.open", {"mode": "new", "reset": True, "quality_profile": "quality_first", "quality_contract": contract})
    call("mesh.from_sections", {"id": "carrier", "sections": [{"x": -1, "width": 1, "height": 1}, {"x": 0, "width": 1.2, "height": 1.2}, {"x": 1, "width": 1, "height": 1}], "segments": 24, "cap_ends": True, "role": "primary", "representation": "section_stack"})
    missing = call("verify.run", {}, expect_ok=True)
    assert missing["result"]["gate"] is False
    assert any(item.get("reason") == "render_evidence_missing" for item in missing["result"]["quality"]["stages"]["evidence"]["failures"])
    call("render.views", {"output_dir": "/tmp/blender_toolbox_evidence_smoke", "resolution": 64, "views": [{"name": "front", "location": [3, -3, 2]}, {"name": "side", "location": [3, 0, 2]}, {"name": "top", "location": [0, 0, 4]}]})
    verified = call("verify.run", {})
    assert verified["result"]["gate"] is True, verified
    assert verified["state"].get("render_evidence", {}).get("views") == ["front", "side", "top"], verified
    print(json.dumps({"status": "ok", "missing_gate": missing["result"]["gate"], "verified_gate": verified["result"]["gate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
