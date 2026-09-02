#!/usr/bin/env python3
"""Generic quality-first smoke coverage for non-vehicle assets.

The test intentionally exercises two different carrier families instead of a
domain recipe: a manufactured product enclosure built from a section stack and
an organic curve carrier.  It is a regression test for the *contract* and its
adaptive stage semantics, not a visual benchmark for either asset.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

try:
    import bpy
except ImportError:  # pragma: no cover
    print("smoke_quality_generic.py must run with Blender's Python", file=sys.stderr)
    raise SystemExit(2)

ROOT = Path(__file__).resolve().parent
ROOT_STRING = str(ROOT)
# Blender may preload another ``blender_toolbox`` package (for example from a
# benchmark working directory).  Move the bundled runtime to the front rather
# than merely checking membership, otherwise this smoke can silently exercise
# a stale registry.
if ROOT_STRING in sys.path:
    sys.path.remove(ROOT_STRING)
sys.path.insert(0, ROOT_STRING)

from blender_toolbox.addon import ToolboxExecutor
from blender_toolbox.protocol import SCHEMA_VERSION

SESSION_ID = "generic-quality-smoke-session"
EPISODE_ID = "generic-quality-smoke-episode"
VIEWS = ["front", "three_quarter", "side"]


def main() -> int:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    executor = ToolboxExecutor()
    step = 0

    def call(action: str, args: dict) -> dict:
        nonlocal step
        payload = {
            "schema_version": SCHEMA_VERSION,
            "request_id": f"generic-quality-smoke-{step}",
            "session_id": SESSION_ID,
            "episode_id": EPISODE_ID,
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
            "request_id": f"generic-quality-smoke-{step}",
            "session_id": SESSION_ID,
            "episode_id": EPISODE_ID,
            "step_id": step,
            "expected_revision": executor.revision,
            "action": action,
            "args": args,
        }
        step += 1
        return executor.execute(payload)

    # A profile-less workflow query must route to the generic safety workflow.
    described_default = call("workflow.describe", {})
    assert described_default["result"]["workflow"]["name"] == "quality_first", described_default

    prop_quality = {
        "enforce": True,
        "min_quality": 0.78,
        "min_primary_vertices": 64,
        "identity": {"asset": "product enclosure", "symmetry": "bilateral"},
        "scale": {"units": "m", "dimensions": [2.4, 1.4, 0.9]},
        "representation": {
            "kind": "mixed",
            "carrier": "section_stack",
            "primary_refs": ["prop.enclosure"],
        },
        "primary_refs": ["prop.enclosure"],
        "reference_views": VIEWS,
        # Deliberately omit feature_scales/detail_regions and secondary refs:
        # these stages should remain visible as unknown, not lower the score.
        "technical": {"require_topology": True, "strict_topology": True},
    }
    prop_task = {
        "intent": "manufactured product enclosure",
        "asset_type": "product",
        "silhouette_views": VIEWS,
        "quality": prop_quality,
    }

    opened = call(
        "session.open",
        {
            "mode": "new",
            "reset": True,
            "profile": "general",
            "quality_profile": "quality_first",
            "task_spec": prop_task,
            "include_capabilities": True,
            "include_examples": False,
        },
    )
    opened_result = opened["result"]
    assert opened_result["profile"] == "general", opened
    assert opened_result["quality_contract"]["enforce"] is True, opened
    selected_workflow = opened_result["capabilities"]["selected_workflow"]
    assert selected_workflow["name"] == "general", opened
    # The general profile may inherit the quality-first bar, but it must not
    # smuggle in vehicle-specific semantic requirements.
    assert not selected_workflow["recommended_contract"].get("required_tags"), selected_workflow
    assert "vehicle" not in selected_workflow["recommended_contract"].get("required_tags", []), selected_workflow
    assert "quality_plan" in opened_result, opened
    assert "quality_contract_template" in opened_result["quality_plan"], opened
    assert opened_result["quality_plan"]["stages"] == ["structure", "primary", "technical", "evidence"], opened

    planned = call("model.plan", {"intent": prop_task["intent"], "continuous_envelope": True})
    plan = planned["result"]
    assert "quality_contract_template" in plan, planned
    assert plan["representation"]["kind"] == "mixed", planned
    assert plan["stages"] == ["structure", "primary", "technical", "evidence"], planned

    # Deliberately author the identity carrier as a primitive.  The quality
    # gate must reject it before any amount of secondary/beauty work can hide
    # the representation error.
    primitive = call(
        "object.create",
        {
            "kind": "cube",
            "id": "prop.enclosure",
            "name": "Primitive_Enclosure",
            "role": "primary",
            "semantic_tags": ["prop", "enclosure", "primary"],
        },
    )
    assert primitive["result"]["ref"] == "prop.enclosure", primitive
    rejected = call("verify.run", {})
    rejected_quality = rejected["result"]["quality"]
    assert rejected_quality["enforced"] is True, rejected
    assert rejected_quality["gate"] is False, rejected
    assert rejected_quality["first_failure"] in {"primary", "evidence", "structure"}, rejected
    assert rejected_quality["repair_action"], rejected

    # Re-open the same generic task and use a continuous section carrier.
    opened = call(
        "session.open",
        {
            "mode": "new",
            "reset": True,
            "profile": "general",
            "quality_profile": "quality_first",
            "task_spec": prop_task,
        },
    )
    carrier = call(
        "mesh.from_sections",
        {
            "id": "prop.enclosure",
            "name": "Product_Enclosure",
            "sections": [
                {"x": -1.2, "width": 0.52, "height": 0.34, "z": 0.42},
                {"x": -0.8, "width": 0.68, "height": 0.46, "z": 0.46},
                {"x": 0.0, "width": 0.74, "height": 0.52, "z": 0.48},
                {"x": 0.8, "width": 0.68, "height": 0.46, "z": 0.46},
                {"x": 1.2, "width": 0.50, "height": 0.32, "z": 0.42},
            ],
            "segments": 24,
            "profile": "superellipse",
            "power": 4.0,
            "cap_ends": True,
            "smooth_shading": True,
            "role": "primary",
            "representation": "section_stack",
            "quality_stage": "primary",
            "semantic_tags": ["prop", "product", "enclosure", "primary"],
        },
    )
    assert carrier["result"]["ref"] == "prop.enclosure", carrier
    assert carrier["result"]["topology"]["watertight"] is True, carrier
    assert carrier["result"]["vertices"] >= prop_quality["min_primary_vertices"], carrier

    audit = call("inspect.quality", {"targets": ["prop.enclosure"], "include_contacts": False})
    audit_result = audit["result"]
    assert audit_result["audit_version"] == "quality_audit.v1", audit
    assert audit_result["objects"][0]["representation"] == "section_stack", audit
    assert audit_result["gate"] is True, audit

    verified_prop = call("verify.run", {})
    prop_result = verified_prop["result"]
    assert prop_result["gate"] is True, verified_prop
    prop_quality_result = prop_result["quality"]
    assert prop_quality_result["gate"] is True, verified_prop
    assert prop_quality_result["score"] >= prop_quality["min_quality"], verified_prop
    assert prop_quality_result["stages"]["technical"]["status"] == "pass", verified_prop
    assert prop_quality_result["stages"]["evidence"]["status"] == "pass", verified_prop
    assert prop_quality_result["stages"]["secondary"]["status"] == "unknown", verified_prop
    assert prop_quality_result["stages"]["tertiary"]["status"] == "unknown", verified_prop
    assert prop_quality_result["unknown_stage_count"] >= 2, verified_prop

    export_path = "/tmp/blender_toolbox_quality_generic_prop.glb"
    exported = call("artifact.export_glb", {"path": export_path})
    # macOS resolves /tmp to /private/tmp in Blender's response.
    assert Path(exported["result"]["path"]).resolve() == Path(export_path).resolve(), exported
    changed = call("object.transform", {"target": "prop.enclosure", "location_delta": [0.03, 0.0, 0.0]})
    assert changed["ok"], changed
    stale = call_raw("artifact.export_glb", {"path": "/tmp/blender_toolbox_quality_generic_prop_stale.glb"})
    assert stale.get("ok") is False, stale
    assert stale.get("error", {}).get("code") == "precondition_failed", stale

    # Organic/curve carrier: no mesh topology is invented just to satisfy a
    # manufactured-asset path.  The same baseline contract remains enforced,
    # while topology is correctly scoped to the declared primary (the curve).
    curve_quality = {
        "enforce": True,
        "min_quality": 0.78,
        "min_primary_vertices": 24,
        "identity": {"asset": "organic tendril", "symmetry": "intentional asymmetry"},
        "scale": {"units": "m", "dimensions": [2.8, 1.2, 1.6]},
        "representation": {"kind": "curve", "carrier": "curve", "primary_refs": ["organic.tendril"]},
        "primary_refs": ["organic.tendril"],
        "reference_views": VIEWS,
        "technical": {"require_topology": True},
    }
    curve_task = {
        "intent": "organic tendril study",
        "asset_type": "organic",
        "silhouette_views": VIEWS,
        "quality": curve_quality,
    }
    opened = call(
        "session.open",
        {
            "mode": "new",
            "reset": True,
            "profile": "general",
            "quality_profile": "quality_first",
            "task_spec": curve_task,
            "include_capabilities": True,
        },
    )
    assert opened["result"]["quality_plan"]["representation"]["kind"] == "curve", opened
    curve_plan = call("model.plan", {"intent": curve_task["intent"], "representation": curve_quality["representation"]})
    assert curve_plan["result"]["representation"]["kind"] == "curve", curve_plan
    assert curve_plan["result"]["stages"] == ["structure", "primary", "technical", "evidence"], curve_plan

    points = [
        [
            -1.35 + 2.7 * index / 71.0,
            0.28 * math.sin(index / 8.0) + 0.08 * math.sin(index / 2.7),
            0.35 + 0.9 * index / 71.0 + 0.12 * math.sin(index / 6.0),
        ]
        for index in range(72)
    ]
    curve = call(
        "curve.create",
        {
            "id": "organic.tendril",
            "name": "Organic_Tendril",
            "points": points,
            "bezier": True,
            "resolution": 8,
            "bevel_depth": 0.08,
            "bevel_resolution": 3,
            "role": "primary",
            "representation": "curve",
            "quality_stage": "primary",
            "semantic_tags": ["organic", "tendril", "primary"],
        },
    )
    assert curve["result"]["ref"] == "organic.tendril", curve
    curve_audit = call("inspect.quality", {"targets": "organic.tendril", "include_contacts": False})
    assert curve_audit["result"]["gate"] is True, curve_audit
    assert curve_audit["result"]["objects"][0]["representation"] == "curve", curve_audit
    verified_curve = call("verify.run", {})
    curve_result = verified_curve["result"]
    assert curve_result["gate"] is True, verified_curve
    curve_quality_result = curve_result["quality"]
    assert curve_quality_result["gate"] is True, verified_curve
    assert curve_quality_result["score"] >= curve_quality["min_quality"], verified_curve
    assert curve_quality_result["stages"]["technical"]["status"] == "pass", verified_curve
    assert curve_quality_result["stages"]["evidence"]["status"] == "pass", verified_curve
    assert curve_quality_result["stages"]["secondary"]["status"] == "unknown", verified_curve
    assert curve_quality_result["stages"]["tertiary"]["status"] == "unknown", verified_curve

    print(
        json.dumps(
            {
                "status": "ok",
                "workflow_default": described_default["result"]["workflow"]["name"],
                "profile": "general",
                "prop_quality": prop_quality_result["score"],
                "curve_quality": curve_quality_result["score"],
                "optional_unknown": curve_quality_result["unknown_stage_count"],
                "stale_export": stale["error"]["code"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
