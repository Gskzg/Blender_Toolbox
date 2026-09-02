#!/usr/bin/env python3
"""Headless smoke test for the structured cross-section loft action."""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import bpy
except ImportError:  # pragma: no cover
    print("smoke_sections.py must run with Blender's Python", file=sys.stderr)
    raise SystemExit(2)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blender_toolbox.addon import ToolboxExecutor
from blender_toolbox.protocol import SCHEMA_VERSION


def main() -> int:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    executor = ToolboxExecutor()
    session = "sections-session"
    episode = "sections-episode"
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

    call("session.open", {"mode": "new", "reset": True, "profile": "vehicle"})

    # Required section fields and enum constraints are checked before Blender
    # dispatch.  A malformed request must not advance the scene revision.
    revision_before_schema_error = executor.revision
    invalid_schema = call_raw("mesh.from_sections", {
        "sections": [
            {"x": 0.0, "width": 1.0},
            {"x": 1.0, "width": 1.0, "height": 1.0},
            {"x": 2.0, "width": 1.0, "height": 1.0},
        ],
    })
    assert invalid_schema.get("ok") is False, invalid_schema
    assert invalid_schema.get("error", {}).get("code") == "invalid_args", invalid_schema
    assert executor.revision == revision_before_schema_error, invalid_schema

    ellipse = call("mesh.from_sections", {
        "id": "vehicle.body.ellipse",
        "name": "Body_Ellipse",
        "profile": "ellipse",
        "segments": 12,
        "sections": [
            {"x": -1.5, "width": 1.2, "height": 0.5, "z": 0.7},
            {"x": 0.0, "width": 1.8, "height": 0.8, "z": 0.75},
            {"x": 1.5, "width": 1.1, "height": 0.45, "z": 0.7},
        ],
        "semantic_tags": ["vehicle", "body", "ellipse"],
    })
    ellipse_result = ellipse["result"]
    assert ellipse_result["ref"] == "vehicle.body.ellipse", ellipse
    assert ellipse_result["profile"] == "ellipse", ellipse
    assert ellipse_result["vertices"] == 3 * 12, ellipse
    assert ellipse_result["faces"] == 2 * 12 + 2, ellipse
    assert ellipse_result["topology"]["watertight"] is True, ellipse
    assert ellipse_result["topology"]["boundary_edges"] == 0, ellipse

    ellipse_obj = next(obj for obj in bpy.context.scene.objects if obj.get("blender_toolbox_ref") == "vehicle.body.ellipse")
    assert all(poly.use_smooth for poly in ellipse_obj.data.polygons), "smooth shading was not applied"

    created = call("mesh.from_sections", {
        "id": "vehicle.body.envelope",
        "name": "Body_Envelope",
        "profile": "superellipse",
        "power": 4.0,
        "segments": 24,
        "sections": [
            {"x": -2.5, "width": 1.20, "height": 0.42, "z": 0.62},
            {"x": -1.8, "width": 1.92, "height": 0.76, "z": 0.66},
            {"x": 0.0, "width": 2.04, "height": 0.82, "z": 0.67},
            {"x": 1.7, "width": 1.82, "height": 0.70, "z": 0.67},
            {"x": 2.5, "width": 1.15, "height": 0.38, "z": 0.64},
        ],
        "semantic_tags": ["vehicle", "body", "primary", "envelope"],
    })
    result = created["result"]
    assert result["ref"] == "vehicle.body.envelope", created
    assert result["topology"]["watertight"] is True, created
    assert result["topology"]["boundary_edges"] == 0, created
    assert result["topology"]["nonmanifold_edges"] == 0, created
    assert result["vertices"] == 5 * 24, created
    assert result["faces"] == 4 * 24 + 2, created
    assert result["profile"] == "superellipse", created
    assert result["power"] == 4.0, created

    # Custom profiles are domain-neutral: a normalized closed Y/Z loop is
    # resampled to the requested ring resolution, while per-section roll and
    # center offsets carry intentional twist/asymmetry without primitive
    # helper geometry.  The legacy z/width/height fields remain in use.
    custom = call("mesh.from_sections", {
        "id": "generic.custom.loft",
        "name": "Generic_Custom_Loft",
        "profile": "custom",
        "profile_points": [
            [-1.0, -0.55], [-0.35, -1.0], [0.72, -0.82],
            [1.0, 0.05], [0.54, 1.0], [-0.62, 0.78],
        ],
        "segments": 16,
        "center_offset": [0.08, 0.12],
        "offset_x": 0.0,
        "sections": [
            {"x": -1.2, "width": 1.0, "height": 0.8},
            {"x": 0.0, "width": 1.5, "height": 1.1, "rotation_x": 0.32,
             "center_offset": [0.12, -0.06]},
            {"x": 1.3, "width": 0.9, "height": 0.65, "rotation_euler": [0.0, 0.0, -0.18]},
        ],
        "semantic_tags": ["generic", "custom_profile", "primary"],
    })
    custom_result = custom["result"]
    assert custom_result["profile"] == "custom", custom
    assert custom_result["profile_points"] == 16, custom
    assert custom_result["vertices"] == 3 * 16, custom
    assert custom_result["topology"]["watertight"] is True, custom
    custom_obj = next(obj for obj in bpy.context.scene.objects if obj.get("blender_toolbox_ref") == "generic.custom.loft")
    # The center offsets and section roll must affect authored coordinates,
    # rather than merely being accepted and dropped at the protocol edge.
    custom_y = [float(vertex.co.y) for vertex in custom_obj.data.vertices]
    custom_z = [float(vertex.co.z) for vertex in custom_obj.data.vertices]
    assert max(custom_y) > 0.75 and min(custom_y) < -0.35, custom_y
    assert max(custom_z) > 0.65 and min(custom_z) < -0.35, custom_z

    invalid_custom = call_raw("mesh.from_sections", {
        "id": "generic.invalid.custom",
        "profile": "custom",
        "segments": 8,
        "sections": [
            {"x": 0.0, "width": 1.0, "height": 1.0},
            {"x": 1.0, "width": 1.0, "height": 1.0},
            {"x": 2.0, "width": 1.0, "height": 1.0},
        ],
    })
    assert invalid_custom.get("ok") is False, invalid_custom
    assert invalid_custom.get("error", {}).get("code") == "invalid_args", invalid_custom

    topology = call("inspect.topology", {"target": "vehicle.body.envelope"})
    assert topology["result"]["watertight"] is True, topology
    assert topology["result"]["boundary_edges"] == 0, topology

    inspected = call("inspect.batch", {"targets": "vehicle.body.envelope", "detail": "compact"})
    assert inspected["result"]["count"] == 1, inspected
    assert inspected["result"]["objects"][0]["semantic_tags"] == ["vehicle", "body", "primary", "envelope"], inspected

    open_loft = call("mesh.from_sections", {
        "id": "vehicle.body.open",
        "segments": 8,
        "cap_ends": False,
        "sections": [
            {"x": -1.0, "width": 1.0, "height": 0.5},
            {"x": 0.0, "width": 1.2, "height": 0.6},
            {"x": 1.0, "width": 0.9, "height": 0.4},
        ],
    })
    assert open_loft["result"]["vertices"] == 3 * 8, open_loft
    assert open_loft["result"]["faces"] == 2 * 8, open_loft
    assert open_loft["result"]["topology"]["watertight"] is False, open_loft
    assert open_loft["result"]["topology"]["boundary_edges"] == 16, open_loft

    # The documented maximum is a vertex budget, and caps reuse ring
    # vertices.  Exercise the exact upper bound to guard against regressions.
    budget_loft = call("mesh.from_sections", {
        "id": "vehicle.body.budget",
        "segments": 256,
        "sections": [
            {"x": float(index), "width": 1.0, "height": 1.0}
            for index in range(128)
        ],
    })
    assert budget_loft["result"]["vertices"] == 128 * 256, budget_loft
    assert budget_loft["result"]["topology"]["watertight"] is True, budget_loft
    call("object.delete", {"targets": "vehicle.body.budget"})

    # Duplicate stable references are preflighted before any mesh datablock is
    # created, so the failed action leaves both the scene and revision intact.
    object_count_before_duplicate = len(bpy.context.scene.objects)
    mesh_count_before_duplicate = len(bpy.data.meshes)
    revision_before_duplicate = executor.revision
    duplicate = call_raw("mesh.from_sections", {
        "id": "vehicle.body.envelope",
        "sections": [
            {"x": 0.0, "width": 1.0, "height": 1.0},
            {"x": 1.0, "width": 1.0, "height": 1.0},
            {"x": 2.0, "width": 1.0, "height": 1.0},
        ],
    })
    assert duplicate.get("ok") is False, duplicate
    assert duplicate.get("error", {}).get("code") == "conflict", duplicate
    assert executor.revision == revision_before_duplicate, duplicate
    assert len(bpy.context.scene.objects) == object_count_before_duplicate, duplicate
    assert len(bpy.data.meshes) == mesh_count_before_duplicate, duplicate
    still_present = call("inspect.batch", {"targets": "vehicle.body.envelope"})
    assert still_present["result"]["count"] == 1, still_present

    failed = call_raw("mesh.from_sections", {
        "id": "vehicle.invalid",
        "sections": [
            {"x": 1.0, "width": 1.0, "height": 1.0},
            {"x": 0.0, "width": 1.0, "height": 1.0},
            {"x": 2.0, "width": 1.0, "height": 1.0},
        ],
    })
    assert failed.get("ok") is False, failed
    assert failed.get("error", {}).get("code") == "invalid_args", failed
    missing = call("inspect.batch", {"targets": "vehicle.invalid", "strict": False})
    assert missing["result"]["count"] == 0, missing

    print(json.dumps({
        "status": "ok",
        "revision": executor.revision,
        "ellipse_vertices": ellipse_result["vertices"],
        "vertices": result["vertices"],
        "faces": result["faces"],
        "watertight": result["topology"]["watertight"],
        "open_boundary_edges": open_loft["result"]["topology"]["boundary_edges"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
