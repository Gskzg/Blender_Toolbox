#!/usr/bin/env python3
"""Blender smoke test for the domain-neutral quality-first contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import bpy
except ImportError:  # pragma: no cover
    print("smoke_quality.py must run with Blender's Python", file=sys.stderr)
    raise SystemExit(2)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blender_toolbox.addon import ToolboxExecutor
from blender_toolbox.protocol import SCHEMA_VERSION


def main() -> int:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    executor = ToolboxExecutor()
    session_id = "quality-smoke-session"
    episode_id = "quality-smoke-episode"
    step = 0

    def call(action: str, args: dict) -> dict:
        nonlocal step
        payload = {
            "schema_version": SCHEMA_VERSION,
            "request_id": f"quality-smoke-{step}",
            "session_id": session_id,
            "episode_id": episode_id,
            "step_id": step,
            "expected_revision": executor.revision,
            "action": action,
            "args": args,
        }
        step += 1
        return executor.execute(payload)

    quality = {
        "enforce": True,
        "min_quality": 0.78,
        "representation": {"kind": "mixed", "primary_refs": ["asset.body"]},
        "secondary_refs": ["asset.detail"],
        "reference_views": ["front", "three_quarter", "side"],
        "feature_scales": [0.05],
        "technical": {"require_topology": True, "strict_topology": True},
    }
    opened = call("session.open", {"mode": "new", "reset": True, "quality_profile": "quality_first", "task_spec": {"quality": quality}})
    assert opened.get("ok"), opened
    assert opened["result"]["quality_contract"]["enforce"] is True, opened
    planned = call("model.plan", {"intent": "a continuous manufactured object", "continuous_envelope": True, "task_spec": {"quality": quality}})
    assert planned.get("ok"), planned
    assert planned["result"]["representation"]["kind"] == "mixed", planned

    primitive = call("object.create", {"kind": "cube", "id": "asset.body", "name": "Primitive_Body", "role": "primary", "semantic_tags": ["asset", "body"]})
    assert primitive.get("ok"), primitive
    rejected = call("verify.run", {})
    assert rejected.get("ok"), rejected
    rejected_quality = rejected["result"]["quality"]
    assert rejected_quality["gate"] is False, rejected
    assert rejected_quality["first_failure"] in {"primary", "evidence", "structure"}, rejected
    assert rejected_quality["repair_action"], rejected
    weakened = call("verify.run", {"quality": {"enforce": False}})
    assert weakened.get("ok"), weakened
    assert weakened["result"]["quality"]["gate"] is False, weakened

    opened = call("session.open", {"mode": "new", "reset": True, "quality_profile": "quality_first", "task_spec": {"quality": quality}})
    assert opened.get("ok"), opened
    carrier = call("mesh.from_sections", {
        "id": "asset.body",
        "name": "Carrier",
        "sections": [
            {"x": -1.5, "width": 1.0, "height": 0.7, "z": 0.5},
            {"x": 0.0, "width": 1.6, "height": 1.0, "z": 0.55},
            {"x": 1.5, "width": 1.0, "height": 0.7, "z": 0.5},
        ],
        "segments": 32,
        "profile": "ellipse",
        "cap_ends": True,
        "smooth_shading": True,
        "role": "primary",
        "representation": "section_stack",
        "semantic_tags": ["asset", "body"],
    })
    assert carrier.get("ok"), carrier
    detail = call("object.create", {"kind": "cube", "id": "asset.detail", "name": "Secondary_Detail", "role": "secondary", "semantic_tags": ["asset", "detail"]})
    assert detail.get("ok"), detail
    audit = call("inspect.quality", {"targets": ["asset.body", "asset.detail"], "include_contacts": True})
    assert audit.get("ok"), audit
    assert audit["result"]["audit_version"] == "quality_audit.v1", audit
    assert audit["result"]["count"] == 2, audit
    verified = call("verify.run", {})
    assert verified.get("ok"), verified
    assert verified["result"]["gate"] is True, verified
    assert verified["result"]["quality"]["gate"] is True, verified
    exported = call("artifact.export_glb", {"path": "/tmp/blender_toolbox_quality_smoke.glb"})
    assert exported.get("ok"), exported
    changed = call("object.transform", {"target": "asset.body", "location_delta": [0.1, 0.0, 0.0]})
    assert changed.get("ok"), changed
    stale = call("artifact.export_glb", {"path": "/tmp/blender_toolbox_quality_smoke_stale.glb"})
    assert stale.get("ok") is False, stale
    assert stale.get("error", {}).get("code") == "precondition_failed", stale
    print(json.dumps({"status": "ok", "primitive_gate": rejected_quality["gate"], "quality": verified["result"]["quality"]["score"], "stale_export": stale["error"]["code"]}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
