#!/usr/bin/env python3
"""Exercise the batch inspection and transform actions inside Blender.

Run with the bundled Blender executable, for example::

    blender --background --factory-startup --python smoke_batch.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import bpy
except ImportError:  # pragma: no cover - this script is intended for Blender.
    print("smoke_batch.py must run with Blender's Python (use --background --python)", file=sys.stderr)
    raise SystemExit(2)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blender_toolbox import addon as addon_module
from blender_toolbox.addon import ToolboxExecutor
from blender_toolbox.protocol import SCHEMA_VERSION


def main() -> int:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    summary_calls = [0]
    original_scene_summary = addon_module.scene_summary

    def counting_scene_summary(*args: object, **kwargs: object) -> dict:
        summary_calls[0] += 1
        return original_scene_summary(*args, **kwargs)

    # The executor's state cache should avoid a second full census for the
    # post-observation of non-mutating actions and refresh after mutations.
    addon_module.scene_summary = counting_scene_summary
    executor = ToolboxExecutor()
    session = "smoke-session"
    episode = "smoke-episode"
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

    try:
        call("session.create", {})
        assert summary_calls[0] == 1, summary_calls
        call("inspect.scene", {})
        assert summary_calls[0] == 2, summary_calls
        body = call("object.create", {"kind": "cube", "name": "Body", "semantic_tags": ["vehicle", "body"]})
        assert summary_calls[0] == 3, summary_calls
    finally:
        addon_module.scene_summary = original_scene_summary
    wheel = call("object.create", {"kind": "cube", "name": "Wheel", "semantic_tags": ["vehicle", "wheel"]})
    body_uuid = body["result"]["uuid"]
    wheel_uuid = wheel["result"]["uuid"]

    before_revision = executor.revision
    transformed = call("object.transform_batch", {
        "transforms": [
            {"target": body_uuid, "location_delta": [1, 0, 0]},
            {"target": wheel_uuid, "location": [2, 0, 0], "scale": [0.5, 0.5, 0.5]},
        ],
    })
    assert transformed["revision"] == before_revision + 1, transformed
    assert transformed["result"]["committed"] is True, transformed
    assert transformed["result"]["successful"] == 2, transformed

    inspected = call("inspect.batch", {"targets": [body_uuid, wheel_uuid]})
    assert inspected["result"]["count"] == 2, inspected
    assert inspected["result"]["missing"] == [], inspected
    compact_keys = set(inspected["result"]["objects"][0])
    assert compact_keys == {
        "uuid", "ref", "name", "type", "collections", "location", "scale",
        "aabb", "mesh", "materials", "semantic_tags", "origin", "role",
        "representation", "quality_stage",
    }, compact_keys
    full = call("inspect.batch", {"targets": body_uuid, "detail": "full"})
    assert "geometry_hash" in full["result"]["objects"][0], full
    selected_fields = call("inspect.batch", {"targets": body_uuid, "fields": ["aabb", "rotation_euler"]})
    assert set(selected_fields["result"]["objects"][0]) == {"uuid", "ref", "name", "type", "aabb", "rotation_euler"}, selected_fields
    body_match = next(item for item in inspected["result"]["objects"] if item["uuid"] == body_uuid)
    assert body_match["location"][0] == 1.0, body_match

    filtered = call("inspect.batch", {"query": {"semantic_tag": "wheel"}})
    assert filtered["result"]["count"] == 1, filtered
    assert filtered["result"]["objects"][0]["uuid"] == wheel_uuid, filtered

    missing = call("inspect.batch", {"targets": [body_uuid, "missing-object"], "strict": False})
    assert missing["result"]["count"] == 1, missing
    assert missing["result"]["missing"] == ["missing-object"], missing

    rollback_revision = executor.revision
    rollback = call("object.transform_batch", {
        "transforms": [
            {"target": body_uuid, "location_delta": [10, 0, 0]},
            {"target": "missing-object", "location_delta": [1, 0, 0]},
        ],
        "atomic": True,
        "stop_on_error": False,
    })
    assert rollback["result"]["rolled_back"] is True, rollback
    assert rollback["revision"] == rollback_revision, rollback
    after_rollback = call("inspect.batch", {"targets": body_uuid})
    body_after = after_rollback["result"]["objects"][0]
    assert body_after["location"][0] == 1.0, body_after

    duplicate = call_raw("object.create_batch", {
        "objects": [
            {"kind": "cube", "id": "vehicle.body"},
            {"kind": "cube", "id": "vehicle.body"},
        ],
    })
    assert duplicate.get("ok") is False, duplicate
    unchanged = call("inspect.batch", {"query": {"semantic_tag": "vehicle"}})
    assert unchanged["result"]["count"] == 2, unchanged
    partial = call_raw("object.create_batch", {
        "objects": [
            {"kind": "cube", "id": "vehicle.partial"},
            {"kind": "not_a_primitive", "id": "vehicle.invalid"},
        ],
    })
    assert partial.get("ok") is False, partial
    removed_partial = call("inspect.batch", {"targets": ["vehicle.partial", "vehicle.invalid"], "strict": False})
    assert removed_partial["result"]["count"] == 0, removed_partial

    print(json.dumps({
        "status": "ok",
        "revision": executor.revision,
        "inspected": inspected["result"]["count"],
        "filtered": filtered["result"]["count"],
        "rollback": rollback["result"]["rolled_back"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
