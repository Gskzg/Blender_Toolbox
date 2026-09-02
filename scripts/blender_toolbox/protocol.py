"""Versioned, transport-neutral Blender Toolbox protocol.

This module intentionally has no Blender dependency.  It is the contract
between an LLM-facing orchestrator and the Blender addon executor.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

SCHEMA_VERSION = "blender_toolbox.v1"
MAX_IPC_MESSAGE_BYTES = 8 * 1024 * 1024
MAX_SEED = 2_147_483_647
DEFAULT_MAX_PROJECT_POINTS = 4096
MAX_PROJECT_POINTS = 262144
_ACTION_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")


class ProtocolError(ValueError):
    """A client request or tool response violates the public protocol."""

    def __init__(self, message: str, code: str = "invalid_request") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    mutating: bool = False
    deterministic: bool = True
    needs_verifier: bool = False
    estimated_cost: str = "low"
    training_allowed: bool = True
    coordinate_dump: bool = False
    preconditions: tuple[str, ...] = ()
    required_args: tuple[str, ...] = ()
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # ``ToolSpec`` is frozen, but its schema trees are mutable JSON.  Deep
        # copy them at construction and normalize the top-level required list
        # so registry consumers cannot mutate global validation state.
        object.__setattr__(self, "required_args", tuple(self.required_args or ()))
        object.__setattr__(self, "preconditions", tuple(self.preconditions or ()))
        input_schema = copy.deepcopy(self.input_schema) if self.input_schema else {"type": "object"}
        input_schema["required"] = list(self.required_args)
        object.__setattr__(self, "input_schema", input_schema)
        object.__setattr__(self, "output_schema", copy.deepcopy(self.output_schema) if self.output_schema else {"type": "object"})

    def as_dict(self) -> Dict[str, Any]:
        input_schema = copy.deepcopy(self.input_schema) if self.input_schema else {
            "type": "object",
            "required": list(self.required_args),
            "additionalProperties": True,
        }
        output_schema = copy.deepcopy(self.output_schema) if self.output_schema else {"type": "object"}
        return {
            "name": self.name,
            "description": self.description,
            "mutating": self.mutating,
            "deterministic": self.deterministic,
            "needs_verifier": self.needs_verifier,
            "estimated_cost": self.estimated_cost,
            "training_allowed": self.training_allowed,
            "coordinate_dump": self.coordinate_dump,
            "preconditions": list(self.preconditions),
            "required_args": list(self.required_args),
            "input_schema": input_schema,
            "output_schema": output_schema,
            "schema_version": SCHEMA_VERSION,
        }


def _spec(
    name: str,
    description: str,
    *,
    mutating: bool = False,
    deterministic: bool = True,
    needs_verifier: bool = False,
    estimated_cost: str = "low",
    training_allowed: bool = True,
    coordinate_dump: bool = False,
    preconditions: tuple[str, ...] = (),
    required_args: tuple[str, ...] = (),
    input_schema: Optional[Dict[str, Any]] = None,
    output_schema: Optional[Dict[str, Any]] = None,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        mutating=mutating,
        deterministic=deterministic,
        needs_verifier=needs_verifier,
        estimated_cost=estimated_cost,
        training_allowed=training_allowed,
        coordinate_dump=coordinate_dump,
        preconditions=preconditions,
        required_args=required_args,
        input_schema=input_schema or {
            "type": "object",
            "required": list(required_args),
            "additionalProperties": True,
        },
        output_schema=output_schema or {"type": "object"},
    )


def _vec3_schema(*, minimum: Optional[float] = None, maximum: Optional[float] = None) -> Dict[str, Any]:
    item: Dict[str, Any] = {"type": "number"}
    if minimum is not None:
        item["minimum"] = minimum
    if maximum is not None:
        item["maximum"] = maximum
    return {"type": "array", "minItems": 3, "maxItems": 3, "items": item}


def _string_schema(*, min_length: Optional[int] = None, max_length: Optional[int] = None) -> Dict[str, Any]:
    schema: Dict[str, Any] = {"type": "string"}
    if min_length is not None:
        schema["minLength"] = min_length
    if max_length is not None:
        schema["maxLength"] = max_length
    return schema


def _object_schema(
    properties: Mapping[str, Any],
    *,
    required: tuple[str, ...] = (),
    additional_properties: bool = True,
) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": additional_properties,
    }


def _index_array_schema(*, max_items: int = 200000) -> Dict[str, Any]:
    return {
        "type": "array",
        "maxItems": max_items,
        "items": {"type": "integer", "minimum": 0},
    }


def _selection_schema() -> Dict[str, Any]:
    """Local mesh selection description shared by edit tools.

    Indices refer to the current mesh datablock.  A spatial selection uses
    object-local coordinates, which makes the same action deterministic after
    replay and independent of viewport state.
    """
    return _object_schema({
        "vertex_indices": _index_array_schema(),
        "edge_indices": _index_array_schema(),
        "face_indices": _index_array_schema(),
        "center": _VEC3,
        "radius": {"type": "number", "exclusiveMinimum": 0},
        "box_min": _VEC3,
        "box_max": _VEC3,
        "vertex_group": _string_schema(min_length=1, max_length=255),
        "vertex_group_min_weight": {"type": "number", "minimum": 0, "maximum": 1},
        "region_handle": _string_schema(min_length=1, max_length=255),
    })


def _symmetry_schema() -> Dict[str, Any]:
    return _object_schema({
        "x": {"type": "boolean"},
        "y": {"type": "boolean"},
        "z": {"type": "boolean"},
    })


def _points_schema(*, max_items: int = 4096) -> Dict[str, Any]:
    return {"type": "array", "minItems": 2, "maxItems": max_items, "items": _VEC3}


def _node_list_schema(*, max_items: int = 256) -> Dict[str, Any]:
    return {"type": "array", "maxItems": max_items, "items": {"type": "object"}}


def _link_list_schema(*, max_items: int = 1024) -> Dict[str, Any]:
    return {"type": "array", "maxItems": max_items, "items": {"type": "object"}}


_PRIMITIVE_KINDS = ["cube", "uv_sphere", "sphere", "ico_sphere", "cylinder", "cone", "torus", "plane", "grid", "circle", "monkey"]
_MODIFIER_TYPES = ["BEVEL", "SUBSURF", "DECIMATE", "SOLIDIFY", "ARRAY", "MIRROR", "REMESH", "SMOOTH", "DISPLACE", "LATTICE", "SHRINKWRAP", "CAST", "SIMPLE_DEFORM", "CORRECTIVE_SMOOTH", "WEIGHTED_NORMAL", "WIREFRAME", "SKIN", "SCREW", "WELD", "EDGE_SPLIT", "CURVE", "LAPLACIANSMOOTH", "BOOLEAN"]
_OPERATION_VALUES = ["UNION", "DIFFERENCE", "INTERSECT", "union", "difference", "intersect"]
_AXIS_VALUES = ["POS_X", "NEG_X", "POS_Y", "NEG_Y", "POS_Z", "NEG_Z"]
_LOCAL_AXIS_VALUES = ["X", "-X", "Y", "-Y", "Z", "-Z"]
_COORDINATE_SPACES = ["WORLD", "LOCAL", "PARENT"]
_VEC3 = _vec3_schema()
_UUID_RESULT = _object_schema({"uuid": _string_schema(min_length=1), "name": _string_schema(min_length=1)}, required=("uuid", "name"))
_PATH_RESULT = _object_schema({"path": _string_schema(min_length=1), "exists": {"type": "boolean"}}, required=("path", "exists"))
_SELECTION = _selection_schema()
_COORDINATE_FRAME = _object_schema({
    "space": {"type": "string", "enum": _COORDINATE_SPACES},
    "up_axis": {"type": "string", "enum": _AXIS_VALUES},
    "front_axis": {"type": "string", "enum": _AXIS_VALUES},
    "units": {"type": "string", "enum": ["meters", "centimeters", "millimeters"]},
    "origin": {"type": "string", "enum": ["world_origin", "asset_origin", "custom"]},
    "custom_origin": _VEC3,
}, required=("space",))

# Cross-section profiles are expressed in the section's local Y/Z plane.  A
# compact two-number point is the canonical form, while a three-number point
# is accepted for clients that reuse generic vector data (the X component is
# ignored by the executor).  Named ``{y, z}`` points remain useful when
# authoring readable payloads.
_PROFILE_POINT = {
    "oneOf": [
        {"type": "array", "minItems": 2, "maxItems": 3, "items": {"type": "number"}},
        _object_schema({"y": {"type": "number"}, "z": {"type": "number"}}, required=("y", "z")),
    ]
}
_RELATION_VALUES = ["attached", "surface_contact", "supported_by", "disjoint", "overlap_free"]
_RELATION = _object_schema({
    "a": _string_schema(min_length=1),
    "b": _string_schema(min_length=1),
    "relation": {"type": "string", "enum": _RELATION_VALUES},
    "units": {"type": "string", "enum": ["meters", "centimeters", "millimeters"]},
    "max_gap": {"type": "number", "minimum": 0, "maximum": 1000},
    "max_penetration": {"type": "number", "minimum": 0, "maximum": 1000},
    "min_overlap": {"type": "number", "minimum": 0, "maximum": 1000},
    "normal_tolerance_degrees": {"type": "number", "minimum": 0, "maximum": 180},
}, required=("a", "b", "relation"))
_VISUAL_CHECKLIST = _object_schema({
    "floating": {"type": "boolean"},
    "overlap": {"type": "boolean"},
    "alignment": {"type": "boolean"},
    "surface_contact": {"type": "boolean"},
    "framing": {"type": "boolean"},
    "proportion": {"type": "boolean"},
}, required=("floating", "overlap", "alignment", "surface_contact", "framing", "proportion"), additional_properties=True)

try:
    from .procedural import PROCEDURAL_RECIPE_SCHEMA, RECIPE_SCHEMA_VERSION, normalize_recipe
except ImportError:  # pragma: no cover
    PROCEDURAL_RECIPE_SCHEMA = {"type": "object", "required": ["nodes"]}
    RECIPE_SCHEMA_VERSION = "blender_toolbox.procedural_recipe.v1"
    normalize_recipe = None

_SECTION = _object_schema({
    "x": {"type": "number"}, "width": {"type": "number", "exclusiveMinimum": 0},
    "height": {"type": "number", "exclusiveMinimum": 0}, "z": {"type": "number"},
    "rotation_x": {"type": "number"}, "center": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}},
    "profile_points": {"type": "array", "minItems": 3, "maxItems": 512, "items": _PROFILE_POINT},
}, required=("x", "width", "height"))
_BATCH_OBJECT = _object_schema({
    "kind": {"type": "string", "enum": _PRIMITIVE_KINDS}, "name": _string_schema(min_length=1, max_length=255),
    "id": _string_schema(min_length=1, max_length=255), "ref": _string_schema(min_length=1, max_length=255),
    "location": _VEC3, "scale": _vec3_schema(minimum=0.000001), "rotation_euler": _VEC3, "coordinate_frame": _COORDINATE_FRAME,
    "segments": {"type": "integer", "minimum": 3, "maximum": 512}, "rings": {"type": "integer", "minimum": 3, "maximum": 256},
    "subdivisions": {"type": "integer", "minimum": 1, "maximum": 6}, "vertices": {"type": "integer", "minimum": 3, "maximum": 512},
    "semantic_tags": {"type": "array", "items": _string_schema(min_length=1), "maxItems": 64},
})

# Compound actions keep the low-level Toolbox protocol explicit while letting
# callers submit a bounded product-level workflow in one revision.  The
# executor expands these steps internally and returns the same per-step
# results in the parent response, so the operation remains inspectable.
_BATCH_STEP = _object_schema({
    "action": _string_schema(min_length=1, max_length=255),
    "args": {"type": "object"},
    "label": _string_schema(min_length=1, max_length=255),
}, required=("action",))
_OBJECT_NAME_ARRAY = {"type": "array", "maxItems": 4096, "items": _string_schema(min_length=1, max_length=255)}
_DECLARED_OBJECT_ARRAY = {"type": "array", "maxItems": 4096, "items": _string_schema(min_length=1, max_length=255)}
_PARENT_CONDITION = _object_schema({
    "child": _string_schema(min_length=1, max_length=255),
    "parent": _string_schema(min_length=1, max_length=255),
}, required=("child", "parent"))
_TAG_CONDITION = _object_schema({
    "object": _string_schema(min_length=1, max_length=255),
    "tags": _OBJECT_NAME_ARRAY,
}, required=("object", "tags"))
_POSTCONDITIONS = _object_schema({
    "objects_exist": _OBJECT_NAME_ARRAY,
    "objects_absent": _OBJECT_NAME_ARRAY,
    "parent_of": {"type": "array", "maxItems": 4096, "items": _PARENT_CONDITION},
    "semantic_tags": {"type": "array", "maxItems": 4096, "items": _TAG_CONDITION},
})


_TOOL_SPECS = (
    _spec("session.create", "Create or attach to a toolbox session.", output_schema=_object_schema({"session": _string_schema(min_length=1)}, required=("session",))),
    _spec("session.open", "Open or resume a profiled Toolbox session and freeze its quality contract.", mutating=True, required_args=(), input_schema=_object_schema({
        "mode": {"type": "string", "enum": ["new", "resume"]}, "reset": {"type": "boolean"},
        "profile": _string_schema(min_length=1), "quality_profile": {"type": "string", "enum": ["advisory", "quality_first", "structural", "production", "organic", "strict"]},
        "quality_contract": {"type": "object"}, "task_spec": {"type": "object"},
        "include_capabilities": {"type": "boolean"}, "include_examples": {"type": "boolean"}, "include_scene": {"type": "boolean"},
        "scene_detail": {"type": "string", "enum": ["compact", "full"]},
    })),
    _spec("session.reset", "Clear the current scene and reset revision.", mutating=True),
    _spec("session.close", "Close a toolbox session."),
    _spec("scene.reset", "Remove all scene objects and orphaned data.", mutating=True),
    _spec("scene.census", "Return a compact scene census.", output_schema={"type": "object"}),
    _spec("model.plan", "Return a read-only representation and quality-contract plan before authoring.", input_schema=_object_schema({
        "intent": _string_schema(min_length=1, max_length=2000), "task_spec": {"type": "object"},
    })),
    _spec("scene.camera_create", "Create or update a deterministic inspection camera.", mutating=True, required_args=("name",), output_schema=_UUID_RESULT, input_schema=_object_schema({
        "name": _string_schema(min_length=1, max_length=255), "location": _VEC3, "target": _VEC3,
        "coordinate_frame": _COORDINATE_FRAME,
        "camera_type": {"type": "string", "enum": ["PERSP", "ORTHO", "PANO"]}, "orthographic_scale": {"type": "number", "exclusiveMinimum": 0, "maximum": 100000},
        "lens": {"type": "number", "minimum": 1, "maximum": 300}, "clip_start": {"type": "number", "exclusiveMinimum": 0},
        "clip_end": {"type": "number", "exclusiveMinimum": 0}, "shift_x": {"type": "number", "minimum": -10, "maximum": 10}, "shift_y": {"type": "number", "minimum": -10, "maximum": 10},
        "dof_target": _string_schema(min_length=1), "dof_fstop": {"type": "number", "exclusiveMinimum": 0, "maximum": 128},
    })),
    _spec("scene.light_create", "Create or update a typed deterministic light.", mutating=True, required_args=("name", "light_type"), output_schema=_UUID_RESULT, input_schema=_object_schema({
        "name": _string_schema(min_length=1, max_length=255), "light_type": {"type": "string", "enum": ["POINT", "SUN", "SPOT", "AREA"]},
        "location": _VEC3, "target": _VEC3, "coordinate_frame": _COORDINATE_FRAME, "energy": {"type": "number", "minimum": 0, "maximum": 10000000},
        "color": {"type": "array", "minItems": 3, "maxItems": 4, "items": {"type": "number", "minimum": 0, "maximum": 1}},
        "size": {"type": "number", "minimum": 0, "maximum": 10000}, "size_y": {"type": "number", "minimum": 0, "maximum": 10000}, "spot_size": {"type": "number", "minimum": 0.01, "maximum": 3.14159}, "spot_blend": {"type": "number", "minimum": 0, "maximum": 1}, "shadow_soft_size": {"type": "number", "minimum": 0, "maximum": 10000},
    })),
    _spec("scene.set_camera", "Set the active scene camera.", mutating=True, required_args=("target",), input_schema=_object_schema({"target": _string_schema(min_length=1)})),
    _spec("scene.coordinate_system", "Declare the scene coordinate contract used by all world-space vectors.", mutating=True, required_args=("units", "up_axis", "front_axis", "handedness", "origin"), input_schema=_object_schema({
        "units": {"type": "string", "enum": ["meters", "centimeters", "millimeters"]},
        "up_axis": {"type": "string", "enum": _AXIS_VALUES},
        "front_axis": {"type": "string", "enum": _AXIS_VALUES},
        "handedness": {"type": "string", "enum": ["right"]},
        "origin": {"type": "string", "enum": ["world_origin", "asset_origin", "custom"]},
        "custom_origin": _VEC3,
    })),
    _spec("scene.set_render_settings", "Set bounded render and animation output settings.", mutating=True, input_schema=_object_schema({
        "engine": {"type": "string", "enum": ["BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES"]},
        "resolution_x": {"type": "integer", "minimum": 16, "maximum": 16384}, "resolution_y": {"type": "integer", "minimum": 16, "maximum": 16384},
        "resolution_percentage": {"type": "integer", "minimum": 1, "maximum": 100}, "fps": {"type": "number", "minimum": 1, "maximum": 240},
        "frame_start": {"type": "integer", "minimum": -100000, "maximum": 100000}, "frame_end": {"type": "integer", "minimum": -100000, "maximum": 100000},
    })),
    _spec("object.create", "Create a named primitive object.", mutating=True, required_args=("kind",), output_schema=_UUID_RESULT, input_schema=_object_schema({
        "kind": {"type": "string", "enum": _PRIMITIVE_KINDS},
        "name": _string_schema(min_length=1, max_length=255),
        "location": _VEC3,
        "coordinate_frame": _COORDINATE_FRAME,
        "scale": _vec3_schema(minimum=0.000001),
        "segments": {"type": "integer", "minimum": 3, "maximum": 512},
        "rings": {"type": "integer", "minimum": 3, "maximum": 256},
        "subdivisions": {"type": "integer", "minimum": 1, "maximum": 6},
        "vertices": {"type": "integer", "minimum": 3, "maximum": 512},
        "major_segments": {"type": "integer", "minimum": 3, "maximum": 512},
        "minor_segments": {"type": "integer", "minimum": 3, "maximum": 256},
        "x_subdivisions": {"type": "integer", "minimum": 2, "maximum": 512},
        "y_subdivisions": {"type": "integer", "minimum": 2, "maximum": 512},
        "fill_type": {"type": "string", "enum": ["NOTHING", "NGON", "TRIFAN"]},
        "semantic_tags": {"type": "array", "items": _string_schema(min_length=1), "maxItems": 64},
    })),
    _spec("object.parent_set", "Set an explicit parent while preserving the child's world transform.", mutating=True, required_args=("child",), input_schema=_object_schema({
        "child": _string_schema(min_length=1), "parent": _string_schema(min_length=1), "clear_parent": {"type": "boolean"},
    })),
    _spec("object.surface_snap", "Snap a specified local contact point to a target mesh surface along a declared direction.", mutating=True, needs_verifier=True, required_args=("target", "surface"), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "surface": _string_schema(min_length=1), "contact_point": _VEC3,
        "direction": _VEC3, "max_distance": {"type": "number", "exclusiveMinimum": 0, "maximum": 10000},
        "offset": {"type": "number", "minimum": -1000, "maximum": 1000},
        "align_axis": {"type": "string", "enum": _LOCAL_AXIS_VALUES}, "align_to_normal": {"type": "boolean"},
        "search_both_directions": {"type": "boolean"}, "coordinate_frame": _COORDINATE_FRAME,
    })),
    _spec("assembly.anchor_create", "Create a named local anchor point and normal on a parent object.", mutating=True, required_args=("parent", "name", "position"), input_schema=_object_schema({
        "parent": _string_schema(min_length=1), "name": _string_schema(min_length=1, max_length=255), "position": _VEC3, "coordinate_frame": _COORDINATE_FRAME,
        "normal": _VEC3, "semantic_tags": {"type": "array", "items": _string_schema(min_length=1), "maxItems": 32},
    })),
    _spec("assembly.attach", "Attach a child to a named anchor using a contact point and clearance.", mutating=True, needs_verifier=True, required_args=("child", "parent", "anchor"), input_schema=_object_schema({
        "child": _string_schema(min_length=1), "parent": _string_schema(min_length=1), "anchor": _string_schema(min_length=1),
        "contact_point": _VEC3, "clearance": {"type": "number", "minimum": -1000, "maximum": 1000},
        "align_axis": {"type": "string", "enum": _LOCAL_AXIS_VALUES}, "align_to_normal": {"type": "boolean"}, "coordinate_frame": _COORDINATE_FRAME,
    })),
    _spec("object.duplicate", "Duplicate an object while preserving semantic tags.", mutating=True, required_args=("target",), output_schema=_UUID_RESULT, input_schema=_object_schema({
        "target": _string_schema(min_length=1), "name": _string_schema(min_length=1, max_length=255),
        "linked_data": {"type": "boolean"}, "location_delta": _VEC3, "coordinate_frame": _COORDINATE_FRAME,
    })),
    _spec("object.join", "Join mesh objects into the first target object.", mutating=True, needs_verifier=True, required_args=("targets",), input_schema=_object_schema({
        "targets": {"type": "array", "minItems": 2, "maxItems": 256, "items": _string_schema(min_length=1)},
        "name": _string_schema(min_length=1, max_length=255),
    })),
    _spec("object.delete", "Delete one or more objects.", mutating=True, required_args=("targets",), preconditions=("each target resolves to an existing object",), output_schema=_object_schema({"deleted": {"type": "array", "items": _string_schema(min_length=1)}}, required=("deleted",)), input_schema=_object_schema({
        "targets": {"oneOf": [_string_schema(min_length=1), {"type": "array", "minItems": 1, "items": _string_schema(min_length=1)}]},
    })),
    _spec("object.transform", "Set or offset an object's transform.", mutating=True, required_args=("target",), preconditions=("target resolves to an existing object",), output_schema=_UUID_RESULT, input_schema=_object_schema({
        "target": _string_schema(min_length=1),
        "location": _VEC3,
        "location_delta": _VEC3,
        "rotation_euler": _VEC3,
        "scale": _vec3_schema(minimum=0.000001),
        "semantic_tags": {"type": "array", "items": _string_schema(min_length=1), "maxItems": 64},
        "coordinate_frame": _COORDINATE_FRAME,
    })),
    _spec("curve.create", "Create a polyline or Bezier curve.", mutating=True, required_args=("points",), output_schema=_UUID_RESULT, input_schema=_object_schema({
        "points": {"type": "array", "minItems": 2, "maxItems": 10000, "items": _VEC3},
        "name": _string_schema(min_length=1, max_length=255),
        "resolution": {"type": "integer", "minimum": 1, "maximum": 128},
        "bevel_depth": {"type": "number", "minimum": 0, "maximum": 1000},
        "bevel_resolution": {"type": "integer", "minimum": 0, "maximum": 32},
        "bezier": {"type": "boolean"},
        "semantic_tags": {"type": "array", "items": _string_schema(min_length=1), "maxItems": 64},
        "coordinate_frame": _COORDINATE_FRAME,
    })),
    _spec("hair.create_strands", "Create deterministic multi-strand hair or fur curves.", mutating=True, estimated_cost="medium", required_args=("strands",), output_schema=_UUID_RESULT, input_schema=_object_schema({
        "name": _string_schema(min_length=1, max_length=255), "strands": {"type": "array", "minItems": 1, "maxItems": 10000, "items": _points_schema(max_items=256)},
        "radii": {"type": "array", "maxItems": 10000, "items": {"type": "array", "minItems": 2, "maxItems": 256, "items": {"type": "number", "exclusiveMinimum": 0, "maximum": 1000}}},
        "bevel_depth": {"type": "number", "minimum": 0, "maximum": 10}, "bevel_resolution": {"type": "integer", "minimum": 0, "maximum": 8},
        "resolution": {"type": "integer", "minimum": 1, "maximum": 32}, "cyclic": {"type": "boolean"},
        "semantic_tags": {"type": "array", "items": _string_schema(min_length=1), "maxItems": 64}, "coordinate_frame": _COORDINATE_FRAME,
    })),
    _spec("hair.convert_to_mesh", "Convert deterministic hair curves to a mesh.", mutating=True, needs_verifier=True, required_args=("target",), input_schema=_object_schema({"target": _string_schema(min_length=1)})),
    _spec("particles.scatter", "Scatter deterministic instances or points on a mesh using Geometry Nodes.", mutating=True, needs_verifier=True, estimated_cost="medium", required_args=("target", "instance"), output_schema=_object_schema({"target": _string_schema(min_length=1), "instance": _string_schema(min_length=1), "modifier": _string_schema(min_length=1), "node_group": _string_schema(min_length=1), "density": {"type": "number"}, "seed": {"type": "integer"}, "realize_instances": {"type": "boolean"}}, required=("target", "instance", "modifier", "node_group")), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "instance": _string_schema(min_length=1), "name": _string_schema(min_length=1, max_length=255),
        "density": {"type": "number", "minimum": 0, "maximum": 1000000}, "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647},
        "scale": _vec3_schema(minimum=0), "keep_surface": {"type": "boolean"}, "realize_instances": {"type": "boolean"},
    })),
    _spec("mesh.subdivide", "Subdivide selected mesh edges for controlled detail density.", mutating=True, needs_verifier=True, required_args=("target",), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "selection": _SELECTION,
        "cuts": {"type": "integer", "minimum": 1, "maximum": 8},
        "smooth": {"type": "number", "minimum": 0, "maximum": 1},
    })),
    _spec("mesh.transform_selection", "Transform a local mesh region without changing object identity.", mutating=True, required_args=("target",), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "selection": _SELECTION,
        "translation": _VEC3, "rotation_euler": _VEC3,
        "scale": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number", "minimum": -1000, "maximum": 1000}},
        "pivot": _VEC3,
    })),
    _spec("mesh.extrude_region", "Extrude a selected face region along a vector or its average normal.", mutating=True, needs_verifier=True, required_args=("target",), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "selection": _SELECTION,
        "offset": _VEC3, "distance": {"type": "number", "minimum": -1000, "maximum": 1000},
        "scale": {"type": "number", "minimum": 0.001, "maximum": 1000},
    })),
    _spec("mesh.inset_region", "Inset selected faces with an optional depth.", mutating=True, needs_verifier=True, required_args=("target",), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "selection": _SELECTION,
        "thickness": {"type": "number", "exclusiveMinimum": 0, "maximum": 1000},
        "depth": {"type": "number", "minimum": -1000, "maximum": 1000},
    })),
    _spec("mesh.bevel", "Bevel selected edges with bounded width and segments.", mutating=True, needs_verifier=True, required_args=("target",), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "selection": _SELECTION,
        "width": {"type": "number", "exclusiveMinimum": 0, "maximum": 1000},
        "segments": {"type": "integer", "minimum": 1, "maximum": 32},
        "profile": {"type": "number", "minimum": 0, "maximum": 1},
    })),
    _spec("mesh.merge_by_distance", "Merge nearby selected vertices in object-local space.", mutating=True, needs_verifier=True, required_args=("target",), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "selection": _SELECTION,
        "distance": {"type": "number", "exclusiveMinimum": 0, "maximum": 1000},
    })),
    _spec("mesh.recalculate_normals", "Recalculate selected mesh face normals consistently.", mutating=True, required_args=("target",), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "selection": _SELECTION, "inside": {"type": "boolean"},
    })),
    _spec("mesh.delete_region", "Delete an explicitly selected mesh region.", mutating=True, needs_verifier=True, required_args=("target", "mode"), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "selection": _SELECTION,
        "mode": {"type": "string", "enum": ["VERTS", "EDGES", "FACES", "ONLY_FACE"]},
    })),
    _spec("mesh.dissolve_region", "Dissolve selected mesh elements while preserving surrounding shape.", mutating=True, needs_verifier=True, required_args=("target", "mode"), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "selection": _SELECTION,
        "mode": {"type": "string", "enum": ["VERTS", "EDGES", "FACES"]},
        "use_face_split": {"type": "boolean"}, "use_boundary_tear": {"type": "boolean"},
    })),
    _spec("mesh.fill_holes", "Fill selected or all boundary loops with faces.", mutating=True, needs_verifier=True, required_args=("target",), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "selection": _SELECTION,
        "sides": {"type": "integer", "minimum": 3, "maximum": 10000},
    })),
    _spec("mesh.triangulate", "Triangulate selected faces for explicit export topology.", mutating=True, needs_verifier=True, required_args=("target",), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "selection": _SELECTION,
    })),
    _spec("mesh.shade_smooth", "Set smooth or flat shading without changing geometry.", mutating=True, required_args=("target",), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "smooth": {"type": "boolean"},
    })),
    _spec("mesh.vertex_group_assign", "Assign a spatially selected region to a named vertex group.", mutating=True, required_args=("target", "name"), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "name": _string_schema(min_length=1, max_length=255),
        "selection": _SELECTION, "weight": {"type": "number", "minimum": 0, "maximum": 1},
        "mode": {"type": "string", "enum": ["REPLACE", "ADD", "SUBTRACT"]},
    })),
    _spec("sculpt.stroke", "Apply a deterministic local sculpt brush stroke to a dense mesh.", mutating=True, needs_verifier=True, estimated_cost="medium", required_args=("target", "points", "radius", "strength", "mode"), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "points": _points_schema(),
        "radius": {"type": "number", "exclusiveMinimum": 0, "maximum": 1000},
        "strength": {"type": "number", "minimum": -10, "maximum": 10},
        "mode": {"type": "string", "enum": ["draw", "inflate", "crease", "grab", "smooth", "flatten", "pinch", "relax"]},
        "direction": _VEC3, "offset": _VEC3,
        "falloff": {"type": "string", "enum": ["smooth", "linear", "constant"]},
        "symmetry": _symmetry_schema(), "front_facing_only": {"type": "boolean"},
    })),
    _spec("sculpt.multires", "Add or raise a Multiresolution detail carrier deterministically.", mutating=True, needs_verifier=True, estimated_cost="high", required_args=("target",), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "levels": {"type": "integer", "minimum": 1, "maximum": 5},
        "sculpt_level": {"type": "integer", "minimum": 0, "maximum": 5}, "render_level": {"type": "integer", "minimum": 0, "maximum": 5},
    })),
    _spec("geometry.boolean", "Apply a declared boolean operation.", mutating=True, needs_verifier=True, required_args=("target", "cutter", "operation"), preconditions=("target and cutter resolve to mesh objects",), output_schema=_object_schema({"target": _string_schema(min_length=1), "operation": {"type": "string", "enum": ["UNION", "DIFFERENCE", "INTERSECT"]}}, required=("target", "operation")), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "cutter": _string_schema(min_length=1),
        "operation": {"type": "string", "enum": _OPERATION_VALUES}, "delete_cutter": {"type": "boolean"},
    })),
    _spec("geometry.add_modifier", "Add a supported Blender modifier.", mutating=True, required_args=("target", "modifier_type"), output_schema=_object_schema({"name": _string_schema(min_length=1), "type": _string_schema(min_length=1)}, required=("name", "type")), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "modifier_type": {"type": "string", "enum": _MODIFIER_TYPES + [v.lower() for v in _MODIFIER_TYPES]},
        "name": _string_schema(min_length=1, max_length=255), "properties": {"type": "object"},
    })),
    _spec("geometry.apply_modifier", "Apply a named modifier.", mutating=True, needs_verifier=True, required_args=("target", "modifier"), preconditions=("target resolves to an object containing the named modifier",), output_schema=_object_schema({"applied": _string_schema(min_length=1)}, required=("applied",)), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "modifier": _string_schema(min_length=1, max_length=255),
    })),
    _spec("geometry.remesh_voxel", "Remesh a mesh at a declared voxel size for sculpt-ready topology.", mutating=True, needs_verifier=True, estimated_cost="high", preconditions=("target has no active Multires modifier",), required_args=("target", "voxel_size"), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "voxel_size": {"type": "number", "exclusiveMinimum": 0.00001, "maximum": 1000},
        "adaptivity": {"type": "number", "minimum": 0, "maximum": 1}, "smooth_shading": {"type": "boolean"},
    })),
    _spec("geometry.shrinkwrap", "Project a mesh onto a validated surface object.", mutating=True, needs_verifier=True, required_args=("target", "surface"), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "surface": _string_schema(min_length=1),
        "offset": {"type": "number", "minimum": -1000, "maximum": 1000},
        "method": {"type": "string", "enum": ["NEAREST_SURFACEPOINT", "PROJECT", "NEAREST_VERTEX"]},
        "axis": {"type": "string", "enum": ["POS_X", "NEG_X", "POS_Y", "NEG_Y", "POS_Z", "NEG_Z"]},
    })),
    _spec("material.create", "Create a Principled BSDF material.", mutating=True, required_args=("name",), output_schema=_object_schema({"name": _string_schema(min_length=1)}, required=("name",)), input_schema=_object_schema({
        "name": _string_schema(min_length=1, max_length=255),
        "base_color": {"type": "array", "minItems": 3, "maxItems": 4, "items": {"type": "number", "minimum": 0, "maximum": 1}},
        "roughness": {"type": "number", "minimum": 0, "maximum": 1},
        "metallic": {"type": "number", "minimum": 0, "maximum": 1},
    })),
    _spec("material.assign", "Assign a material to an object.", mutating=True, required_args=("target", "material"), preconditions=("target and material exist",), output_schema=_object_schema({"target": _string_schema(min_length=1), "material": _string_schema(min_length=1)}, required=("target", "material")), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "material": _string_schema(min_length=1, max_length=255),
    })),
    _spec("material.node_graph", "Build an allowlisted procedural material node graph.", mutating=True, estimated_cost="medium", required_args=("name", "nodes"), input_schema=_object_schema({
        "name": _string_schema(min_length=1, max_length=255), "nodes": _node_list_schema(), "links": _link_list_schema(), "clear": {"type": "boolean"},
    })),
    _spec("material.set_input", "Set one allowlisted material node input value.", mutating=True, required_args=("material", "node", "socket", "value"), input_schema=_object_schema({
        "material": _string_schema(min_length=1), "node": _string_schema(min_length=1), "socket": _string_schema(min_length=1), "value": {},
    })),
    _spec("inspect.scene", "Return the current structured scene observation."),
    _spec("inspect.quality", "Return a compact authoritative quality audit for the current revision.", input_schema=_object_schema({
        "targets": {"type": "array", "maxItems": 4096, "items": _string_schema(min_length=1)},
        "include_contacts": {"type": "boolean"}, "quality_stage": {"type": "string", "enum": ["structure", "primary", "secondary", "tertiary", "technical", "evidence"]},
    })),
    _spec("inspect.object", "Return one object's structured observation.", required_args=("target",), input_schema=_object_schema({"target": _string_schema(min_length=1)})),
    _spec("inspect.topology", "Return mesh topology diagnostics.", needs_verifier=True, required_args=("target",), input_schema=_object_schema({"target": _string_schema(min_length=1)})),
    _spec("inspect.measure", "Return dimensions and basic geometric measures.", required_args=("target",), input_schema=_object_schema({"target": _string_schema(min_length=1)})),
    _spec("inspect.relationships", "Audit declared or all pairwise spatial relationships in world coordinates.", input_schema=_object_schema({
        "relations": {"type": "array", "maxItems": 512, "items": _RELATION},
        "targets": {"type": "array", "maxItems": 256, "items": _string_schema(min_length=1)},
        "include_disjoint": {"type": "boolean"},
        "audit_spatial": {"type": "boolean"}, "max_pair_gap": {"type": "number", "minimum": 0, "maximum": 1000},
    })),
    _spec("inspect.uv", "Return UV layer and coverage diagnostics.", required_args=("target",), input_schema=_object_schema({"target": _string_schema(min_length=1)})),
    _spec("inspect.material", "Return a compact material node graph description.", required_args=("target",), input_schema=_object_schema({"target": _string_schema(min_length=1)})),
    _spec("inspect.armature", "Return armature bone names, hierarchy and transforms.", required_args=("target",), input_schema=_object_schema({"target": _string_schema(min_length=1)})),
    _spec("inspect.animation", "Return animation, shape key and frame information for an object.", required_args=("target",), output_schema={"type": "object"}, input_schema=_object_schema({"target": _string_schema(min_length=1)})),
    _spec("inspect.geometry_nodes", "Return Geometry Nodes graphs attached to an object.", required_args=("target",), output_schema={"type": "object"}, input_schema=_object_schema({"target": _string_schema(min_length=1)})),
    _spec("inspect.landmarks", "Return named landmark empties and semantic tags.", output_schema={"type": "object"}, input_schema=_object_schema({"semantic_tag": _string_schema(min_length=1)})),
    _spec("uv.unwrap", "Unwrap a mesh with a declared deterministic method.", mutating=True, needs_verifier=True, required_args=("target", "method"), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "method": {"type": "string", "enum": ["ANGLE_BASED", "CONFORMAL", "SMART_PROJECT", "CUBE_PROJECT", "CYLINDER_PROJECT", "SPHERE_PROJECT"]},
        "margin": {"type": "number", "minimum": 0, "maximum": 1}, "selection": _SELECTION,
    })),
    _spec("uv.pack", "Pack UV islands inside the 0-1 tile.", mutating=True, required_args=("target",), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "margin": {"type": "number", "minimum": 0, "maximum": 1}, "rotate": {"type": "boolean"},
    })),
    _spec("uv.mark_seams", "Mark an explicit local edge selection as UV seams.", mutating=True, required_args=("target",), output_schema={"type": "object"}, input_schema=_object_schema({
        "target": _string_schema(min_length=1), "selection": _SELECTION,
    })),
    _spec("uv.clear_seams", "Clear UV seams on an explicit local edge selection.", mutating=True, required_args=("target",), output_schema={"type": "object"}, input_schema=_object_schema({
        "target": _string_schema(min_length=1), "selection": _SELECTION,
    })),
    _spec("uv.project", "Project UVs from a declared axis or camera-like direction.", mutating=True, required_args=("target", "method"), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "method": {"type": "string", "enum": ["CUBE", "CYLINDER", "SPHERE", "VIEW"]},
        "axis": {"type": "string", "enum": ["POS_X", "NEG_X", "POS_Y", "NEG_Y", "POS_Z", "NEG_Z"]}, "scale_to_bounds": {"type": "boolean"}, "selection": _SELECTION,
    })),
    _spec("rig.create_armature", "Create a named armature from explicit bone landmarks.", mutating=True, required_args=("name", "bones"), output_schema=_UUID_RESULT, input_schema=_object_schema({
        "name": _string_schema(min_length=1, max_length=255), "bones": {"type": "array", "minItems": 1, "maxItems": 512, "items": _object_schema({
            "name": _string_schema(min_length=1, max_length=255), "head": _VEC3, "tail": _VEC3, "parent": _string_schema(min_length=1), "use_connect": {"type": "boolean"},
        })}, "location": _VEC3, "semantic_tags": {"type": "array", "items": _string_schema(min_length=1), "maxItems": 64}, "coordinate_frame": _COORDINATE_FRAME,
    })),
    _spec("rig.bind", "Bind a mesh to an armature with an explicit modifier and weight policy.", mutating=True, needs_verifier=True, required_args=("target", "armature"), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "armature": _string_schema(min_length=1), "weights": {"type": "string", "enum": ["empty", "envelopes", "automatic"]},
    })),
    _spec("rig.pose", "Set named pose bone transforms at a frame.", mutating=True, required_args=("armature", "bones"), input_schema=_object_schema({
        "armature": _string_schema(min_length=1), "frame": {"type": "integer", "minimum": -100000, "maximum": 100000},
        "bones": {"type": "array", "minItems": 1, "maxItems": 512, "items": _object_schema({"name": _string_schema(min_length=1), "rotation_euler": _VEC3, "location": _VEC3, "scale": _VEC3})},
        "keyframe": {"type": "boolean"},
    })),
    _spec("rig.add_constraint", "Add a bounded constraint to a named pose bone.", mutating=True, required_args=("armature", "bone", "constraint_type"), output_schema={"type": "object"}, input_schema=_object_schema({
        "armature": _string_schema(min_length=1), "bone": _string_schema(min_length=1),
        "constraint_type": {"type": "string", "enum": ["COPY_LOCATION", "COPY_ROTATION", "COPY_SCALE", "COPY_TRANSFORMS", "IK", "DAMPED_TRACK", "LIMIT_ROTATION"]},
        "target": _string_schema(min_length=1), "subtarget": _string_schema(min_length=1), "influence": {"type": "number", "minimum": 0, "maximum": 1},
        "chain_count": {"type": "integer", "minimum": 0, "maximum": 256},
        "use_limit_x": {"type": "boolean"}, "use_limit_y": {"type": "boolean"}, "use_limit_z": {"type": "boolean"},
        "min_x": {"type": "number", "minimum": -1000, "maximum": 1000}, "max_x": {"type": "number", "minimum": -1000, "maximum": 1000},
        "min_y": {"type": "number", "minimum": -1000, "maximum": 1000}, "max_y": {"type": "number", "minimum": -1000, "maximum": 1000},
        "min_z": {"type": "number", "minimum": -1000, "maximum": 1000}, "max_z": {"type": "number", "minimum": -1000, "maximum": 1000},
    })),
    _spec("animation.keyframe_transform", "Keyframe a controlled object transform.", mutating=True, required_args=("target", "frame"), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "frame": {"type": "integer", "minimum": -100000, "maximum": 100000},
        "location": _VEC3, "rotation_euler": _VEC3, "scale": _VEC3, "interpolation": {"type": "string", "enum": ["CONSTANT", "LINEAR", "BEZIER"]},
    })),
    _spec("animation.set_range", "Set the scene frame range and playback rate.", mutating=True, input_schema=_object_schema({
        "frame_start": {"type": "integer", "minimum": -100000, "maximum": 100000}, "frame_end": {"type": "integer", "minimum": -100000, "maximum": 100000},
        "fps": {"type": "number", "minimum": 1, "maximum": 240},
    })),
    _spec("landmark.create", "Create a stable named facial or anatomical landmark.", mutating=True, required_args=("name", "location"), output_schema=_UUID_RESULT, input_schema=_object_schema({
        "name": _string_schema(min_length=1, max_length=255), "location": _VEC3, "parent": _string_schema(min_length=1), "semantic_tags": {"type": "array", "items": _string_schema(min_length=1), "maxItems": 32}, "coordinate_frame": _COORDINATE_FRAME,
    })),
    _spec("landmark.create_set", "Create or update a deterministic set of named landmarks.", mutating=True, needs_verifier=False, required_args=("landmarks",), output_schema={"type": "object"}, input_schema=_object_schema({
        "landmarks": {"type": "array", "minItems": 1, "maxItems": 512, "items": _object_schema({
            "name": _string_schema(min_length=1, max_length=255), "location": _VEC3, "parent": _string_schema(min_length=1), "coordinate_frame": _COORDINATE_FRAME,
            "semantic_tags": {"type": "array", "items": _string_schema(min_length=1), "maxItems": 32},
        })}, "coordinate_frame": _COORDINATE_FRAME,
    })),
    _spec("face.curve_from_landmarks", "Create a curve through named landmark objects.", mutating=True, required_args=("name", "landmarks"), output_schema=_UUID_RESULT, input_schema=_object_schema({
        "name": _string_schema(min_length=1, max_length=255), "landmarks": {"type": "array", "minItems": 2, "maxItems": 256, "items": _string_schema(min_length=1)},
        "bezier": {"type": "boolean"}, "cyclic": {"type": "boolean"}, "bevel_depth": {"type": "number", "minimum": 0, "maximum": 10},
        "bevel_resolution": {"type": "integer", "minimum": 0, "maximum": 16}, "surface": _string_schema(min_length=1), "offset": {"type": "number", "minimum": -100, "maximum": 100},
        "semantic_tags": {"type": "array", "items": _string_schema(min_length=1), "maxItems": 32},
    })),
    _spec("face.curve_network_from_landmarks", "Create multiple semantic facial curves from landmark sequences.", mutating=True, estimated_cost="medium", required_args=("curves",), output_schema={"type": "object"}, input_schema=_object_schema({
        "curves": {"type": "array", "minItems": 1, "maxItems": 256, "items": _object_schema({
            "name": _string_schema(min_length=1, max_length=255), "landmarks": {"type": "array", "minItems": 2, "maxItems": 256, "items": _string_schema(min_length=1)},
            "bezier": {"type": "boolean"}, "cyclic": {"type": "boolean"}, "bevel_depth": {"type": "number", "minimum": 0, "maximum": 10},
            "bevel_resolution": {"type": "integer", "minimum": 0, "maximum": 16}, "surface": _string_schema(min_length=1), "offset": {"type": "number", "minimum": -100, "maximum": 100},
            "semantic_tags": {"type": "array", "items": _string_schema(min_length=1), "maxItems": 32},
        })},
    })),
    _spec("face.sculpt_landmarks", "Apply a deterministic sculpt stroke through named landmarks.", mutating=True, needs_verifier=True, estimated_cost="medium", required_args=("target", "landmarks", "radius", "strength", "mode"), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "landmarks": {"type": "array", "minItems": 2, "maxItems": 256, "items": _string_schema(min_length=1)},
        "radius": {"type": "number", "exclusiveMinimum": 0, "maximum": 1000}, "strength": {"type": "number", "minimum": -10, "maximum": 10},
        "mode": {"type": "string", "enum": ["draw", "inflate", "crease", "grab", "smooth", "flatten", "pinch", "relax"]},
        "direction": _VEC3, "offset": _VEC3, "falloff": {"type": "string", "enum": ["smooth", "linear", "constant"]}, "symmetry": _symmetry_schema(),
    })),
    _spec("face.shape_key_landmarks", "Create a deterministic facial expression shape key from landmark offsets.", mutating=True, needs_verifier=True, estimated_cost="medium", required_args=("target", "name", "landmarks", "offsets", "radius"), output_schema={"type": "object"}, input_schema=_object_schema({
        "target": _string_schema(min_length=1), "name": _string_schema(min_length=1, max_length=255),
        "landmarks": {"type": "array", "minItems": 1, "maxItems": 256, "items": _string_schema(min_length=1)},
        "offsets": {"type": "array", "minItems": 1, "maxItems": 256, "items": _VEC3},
        "radius": {"type": "number", "exclusiveMinimum": 0, "maximum": 1000},
        "strength": {"type": "number", "minimum": -10, "maximum": 10}, "symmetry": _symmetry_schema(),
    })),
    _spec("animation.keyframe_shape_key", "Keyframe a mesh shape key value for facial animation.", mutating=True, required_args=("target", "shape_key", "frame", "value"), output_schema={"type": "object"}, input_schema=_object_schema({
        "target": _string_schema(min_length=1), "shape_key": _string_schema(min_length=1),
        "frame": {"type": "integer", "minimum": -100000, "maximum": 100000}, "value": {"type": "number", "minimum": 0, "maximum": 1},
        "interpolation": {"type": "string", "enum": ["CONSTANT", "LINEAR", "BEZIER"]},
    })),
    _spec("geometry_nodes.create", "Create an allowlisted Geometry Nodes graph on an object.", mutating=True, estimated_cost="medium", required_args=("target", "nodes"), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "name": _string_schema(min_length=1, max_length=255), "nodes": _node_list_schema(), "links": _link_list_schema(),
    })),
    _spec("geometry_nodes.set_input", "Set one Geometry Nodes input value by node id and socket.", mutating=True, required_args=("target", "node", "socket", "value"), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "node": _string_schema(min_length=1), "socket": _string_schema(min_length=1), "value": {},
    })),
    _spec("inspect.mesh_region", "Inspect selected local mesh counts and bounds.", required_args=("target",), input_schema=_object_schema({
        "target": _string_schema(min_length=1), "selection": _SELECTION,
    })),
    _spec("parameters.sample", "Sample deterministic procedural parameters from a declared distribution.", required_args=("distribution",), input_schema=_object_schema({
        "distribution": {"type": "string", "enum": ["uniform", "normal", "log_uniform", "triangular", "integer"]},
        "seed": {"type": "integer", "minimum": 0, "maximum": MAX_SEED}, "count": {"type": "integer", "minimum": 1, "maximum": 4096},
        "low": {"type": "number"}, "high": {"type": "number"}, "mean": {"type": "number"}, "std": {"type": "number", "exclusiveMinimum": 0}, "mode": {"type": "number"}, "name": _string_schema(min_length=1, max_length=255),
    })),
    _spec("verify.run", "Run the configured verifier and return a scorecard.", needs_verifier=True, estimated_cost="medium", input_schema=_object_schema({
        "expect_shells": {"type": "integer", "minimum": 1}, "expect_genus": {"type": "integer", "minimum": 0},
        "require_closed": {"type": "boolean"}, "required_tags": {"type": "array", "items": _string_schema(min_length=1)},
        "feature_sizes": {"type": "array", "items": {"type": "number", "exclusiveMinimum": 0}},
        "voxel": {"type": "number", "exclusiveMinimum": 0}, "verifier_path": _string_schema(min_length=1),
        "measure_path": _string_schema(min_length=1), "silhouette_path": _string_schema(min_length=1),
        "task_spec": {"type": "object"}, "task_spec_path": _string_schema(min_length=1),
        "relations": {"type": "array", "maxItems": 512, "items": _RELATION},
        "spatial_contract": {"type": "object"}, "audit_spatial": {"type": "boolean"},
        "max_pair_gap": {"type": "number", "minimum": 0, "maximum": 1000},
        "audit_scope": {"type": "string", "enum": ["scene", "targets"]},
        "targets": {"type": "array", "maxItems": 4096, "items": _string_schema(min_length=1)},
        "include_disjoint": {"type": "boolean"},
        "allow_default_coordinates": {"type": "boolean"},
        "quality_profile": {"type": "string", "enum": ["structural", "production", "organic", "strict"]},
        "completion_gate": {"type": "boolean"},
        "required_views": {"type": "array", "maxItems": 64, "items": _string_schema(min_length=1)},
        "required_review_stages": {"type": "array", "maxItems": 16, "items": {"type": "string", "enum": ["structure", "primary", "secondary", "tertiary", "technical", "evidence"]}},
        "required_evidence_types": {"type": "array", "maxItems": 16, "items": {"type": "string", "enum": ["beauty", "clay", "silhouette", "closeup", "source_detail", "reference"]}},
        "min_visual_views": {"type": "integer", "minimum": 1, "maximum": 64},
        "min_visual_score": {"type": "number", "minimum": 0, "maximum": 1},
        "quality_stage": {"type": "string", "enum": ["structure", "primary", "secondary", "tertiary", "technical", "evidence"]},
        "require_visual_review": {"type": "boolean"},
    })),
    _spec("render.views", "Render declared orthographic inspection views.", deterministic=True, estimated_cost="high", input_schema=_object_schema({
        "output_dir": _string_schema(min_length=1), "resolution": {"type": "integer", "minimum": 16, "maximum": 4096},
        "engine": {"type": "string", "enum": ["BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES"]},
        "coordinate_frame": _COORDINATE_FRAME,
        "target": _VEC3,
        "views": {"type": "array", "maxItems": 32, "items": _object_schema({"name": _string_schema(min_length=1), "location": _VEC3, "target": _VEC3, "coordinate_frame": _COORDINATE_FRAME, "evidence_type": {"type": "string", "enum": ["beauty", "clay", "silhouette", "closeup", "source_detail", "reference"]}})},
        "evidence_types": {"type": "array", "maxItems": 16, "items": {"type": "string", "enum": ["beauty", "clay", "silhouette", "closeup", "source_detail", "reference"]}},
        "quality_stage": {"type": "string", "enum": ["structure", "primary", "secondary", "tertiary", "technical", "evidence"]},
    })),
    _spec("evidence.visual_review", "Record a checklist-based visual review against the current render revision.", input_schema=_object_schema({
        "revision": {"type": "integer", "minimum": 0}, "views": {"type": "array", "minItems": 1, "maxItems": 32, "items": _string_schema(min_length=1)},
        "targets": {"type": "array", "maxItems": 4096, "items": _string_schema(min_length=1)},
        "quality_stage": {"type": "string", "enum": ["structure", "primary", "secondary", "tertiary", "technical", "evidence"]},
        "passed": {"type": "boolean"}, "review_mode": {"type": "string", "enum": ["standard", "critical"]},
        "reviewer": _string_schema(min_length=1, max_length=255), "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "scores": {"type": "object"}, "blockers": {"type": "array", "maxItems": 128, "items": _string_schema(min_length=1)},
        "render_hashes": {"type": "object"}, "reference_views": {"type": "array", "maxItems": 64, "items": _string_schema(min_length=1)},
        "anti_slop_checks": {"type": "object"},
        "anti_slop_evidence": {"type": "object"},
        "checklist": _VISUAL_CHECKLIST, "findings": {"type": "array", "maxItems": 128, "items": _string_schema(min_length=1)},
    }, required=("revision", "views", "quality_stage", "passed", "checklist"))),
    _spec("artifact.save_checkpoint", "Save an immutable Blender checkpoint.", estimated_cost="medium", required_args=("path",), output_schema=_PATH_RESULT, input_schema=_object_schema({"path": _string_schema(min_length=1)})),
    _spec("artifact.export_glb", "Export the current scene as GLB after optional current-revision quality, visual, and completion gates.", estimated_cost="medium", required_args=("path",), output_schema=_PATH_RESULT, input_schema=_object_schema({
        "path": _string_schema(min_length=1), "require_quality": {"type": "boolean"}, "require_visual_review": {"type": "boolean"}, "require_completion": {"type": "boolean"},
        "required_views": {"type": "array", "maxItems": 64, "items": _string_schema(min_length=1)},
        "required_review_stages": {"type": "array", "maxItems": 16, "items": {"type": "string", "enum": ["structure", "primary", "secondary", "tertiary", "technical", "evidence"]}},
        "required_evidence_types": {"type": "array", "maxItems": 16, "items": {"type": "string", "enum": ["beauty", "clay", "silhouette", "closeup", "source_detail", "reference"]}},
        "min_visual_views": {"type": "integer", "minimum": 1, "maximum": 64},
        "min_visual_score": {"type": "number", "minimum": 0, "maximum": 1},
    })),
    _spec("workflow.batch", "Apply a bounded group of Toolbox actions as one atomic, traceable workflow.", mutating=True, needs_verifier=True, estimated_cost="medium", required_args=("intent", "steps"), output_schema=_object_schema({
        "intent": _string_schema(min_length=1), "steps": {"type": "array"}, "completed": {"type": "integer"},
        "rolled_back": {"type": "boolean"}, "created": _OBJECT_NAME_ARRAY, "modified": _OBJECT_NAME_ARRAY,
        "deleted": _OBJECT_NAME_ARRAY, "declarations": {"type": "object"},
        "verify_after": {"type": ["object", "null"]},
    }), input_schema=_object_schema({
        "intent": _string_schema(min_length=1, max_length=2000),
        "steps": {"type": "array", "minItems": 1, "maxItems": 256, "items": _BATCH_STEP},
        "creates": _OBJECT_NAME_ARRAY,
        "modifies": _OBJECT_NAME_ARRAY,
        "deletes": _OBJECT_NAME_ARRAY,
        "transaction": {"type": "boolean"},
        "rollback_on_error": {"type": "boolean"},
        "strict_declarations": {"type": "boolean"},
        "verify_after": {"type": "object"},
    })),
    _spec(
        "mesh.from_pydata",
        "Create a mesh from explicit vertices and faces; advanced use only.",
        mutating=True,
        needs_verifier=True,
        training_allowed=False,
        coordinate_dump=True,
        required_args=("vertices", "faces"), input_schema=_object_schema({
            "vertices": {"type": "array", "minItems": 3, "maxItems": 200000, "items": _VEC3},
            "faces": {"type": "array", "minItems": 1, "maxItems": 200000, "items": {"type": "array", "minItems": 3, "maxItems": 32, "items": {"type": "integer", "minimum": 0}}},
            "name": _string_schema(min_length=1, max_length=255), "semantic_tags": {"type": "array", "items": _string_schema(min_length=1)}, "coordinate_frame": _COORDINATE_FRAME,
        }),
    ),
    _spec(
        "run_python",
        "Restricted escape hatch for a reviewed Python fragment.",
        mutating=True,
        deterministic=False,
        needs_verifier=True,
        estimated_cost="medium",
        training_allowed=False,
        required_args=("source",), input_schema=_object_schema({
            "source": _string_schema(min_length=1, max_length=8000),
            "timeout_ms": {"type": "integer", "minimum": 1, "maximum": 10000},
            "max_result_chars": {"type": "integer", "minimum": 128, "maximum": 65536},
        }),
    ),
    _spec(
        "bpy.apply",
        "Execute an explicitly declared Blender Python asset and validate its scene delta.",
        mutating=True,
        deterministic=False,
        needs_verifier=True,
        estimated_cost="high",
        training_allowed=False,
        required_args=("purpose", "creates", "modifies"),
        input_schema=_object_schema({
            "purpose": _string_schema(min_length=1, max_length=2000),
            "source_path": _string_schema(min_length=1, max_length=4096),
            "source": _string_schema(min_length=1, max_length=200000),
            "source_sha256": _string_schema(min_length=10, max_length=128),
            "creates": _DECLARED_OBJECT_ARRAY,
            "modifies": _DECLARED_OBJECT_ARRAY,
            "deletes": _OBJECT_NAME_ARRAY,
            "postconditions": _POSTCONDITIONS,
            "transaction": {"type": "boolean"},
            "rollback_on_error": {"type": "boolean"},
            "strict_declarations": {"type": "boolean"},
            "timeout_ms": {"type": "integer", "minimum": 1, "maximum": 120000},
            "max_result_chars": {"type": "integer", "minimum": 128, "maximum": 65536},
            "seed": {"type": "integer", "minimum": 0, "maximum": MAX_SEED},
        }),
    ),
)


# The canonical executor exposes these compound and advanced actions.  Keep
# their schemas in the same registry as the foundational actions so MCP
# discovery, request validation, and replay all share one contract.
_EXTRA_TOOL_SPECS = (
    _spec("object.create_batch", "Create bounded primitives atomically.", mutating=True, required_args=("objects",), input_schema=_object_schema({"objects": {"type": "array", "minItems": 1, "maxItems": 256, "items": _BATCH_OBJECT}, "atomic": {"type": "boolean"}, "stop_on_error": {"type": "boolean"}})),
    _spec("object.transform_batch", "Apply bounded transforms atomically.", mutating=True, required_args=("transforms",), input_schema=_object_schema({"transforms": {"type": "array", "minItems": 1, "maxItems": 256, "items": {"type": "object"}}, "atomic": {"type": "boolean"}, "stop_on_error": {"type": "boolean"}})),
    _spec("material.assign_batch", "Assign materials to multiple objects atomically.", mutating=True, required_args=("assignments",), input_schema=_object_schema({"assignments": {"type": "array", "minItems": 1, "maxItems": 512, "items": {"type": "object"}}, "atomic": {"type": "boolean"}})),
    _spec("geometry.modifier_stack", "Build an ordered modifier stack atomically.", mutating=True, required_args=("target", "modifiers"), input_schema=_object_schema({"target": _string_schema(min_length=1), "modifiers": {"type": "array", "minItems": 1, "maxItems": 64, "items": {"type": "object"}}, "apply": {"type": "boolean"}, "atomic": {"type": "boolean"}})),
    _spec("mesh.from_sections", "Create a deterministic continuous section-stack carrier.", mutating=True, needs_verifier=True, required_args=("sections",), input_schema=_object_schema({"sections": {"type": "array", "minItems": 3, "maxItems": 128, "items": _SECTION}, "name": _string_schema(min_length=1), "id": _string_schema(min_length=1), "ref": _string_schema(min_length=1), "segments": {"type": "integer", "minimum": 8, "maximum": 256}, "profile": {"type": "string", "enum": ["ellipse", "superellipse", "custom"]}, "profile_points": {"type": "array", "minItems": 3, "maxItems": 512, "items": _PROFILE_POINT}, "rotation_euler": _VEC3, "center_offset": {"type": "array", "minItems": 2, "maxItems": 3, "items": {"type": "number"}}, "coordinate_frame": _COORDINATE_FRAME, "power": {"type": "number", "exclusiveMinimum": 0, "maximum": 16}, "cap_ends": {"type": "boolean"}, "smooth_shading": {"type": "boolean"}, "semantic_tags": {"type": "array", "items": _string_schema(min_length=1)}})),
    _spec("geometry_nodes.apply_recipe", "Apply a validated Geometry Nodes recipe.", mutating=True, needs_verifier=True, required_args=("target", "recipe"), output_schema=_object_schema({"target": _string_schema(min_length=1), "recipe_hash": _string_schema(min_length=1)}, required=("target", "recipe_hash")), input_schema=_object_schema({"target": _string_schema(min_length=1), "recipe": PROCEDURAL_RECIPE_SCHEMA})),
    _spec("material.apply_recipe", "Create or replace a validated material recipe.", mutating=True, needs_verifier=True, required_args=("name", "recipe"), output_schema=_object_schema({"name": _string_schema(min_length=1), "recipe_hash": _string_schema(min_length=1)}, required=("name", "recipe_hash")), input_schema=_object_schema({"name": _string_schema(min_length=1), "recipe": PROCEDURAL_RECIPE_SCHEMA})),
    _spec("landmark.project_to_surface", "Project a landmark onto a target surface.", mutating=True, needs_verifier=True, required_args=("landmark", "target"), input_schema=_object_schema({"landmark": _string_schema(min_length=1), "target": _string_schema(min_length=1), "direction": _VEC3, "max_distance": {"type": "number", "exclusiveMinimum": 0}, "offset": {"type": "number"}, "coordinate_frame": _COORDINATE_FRAME})),
    _spec("inspect.batch", "Return a compact scene census and selected diagnostics.", input_schema=_object_schema({"query": {"type": "object"}, "limit": {"type": "integer", "minimum": 1, "maximum": 4096}, "detail": {"type": "string", "enum": ["compact", "full"]}, "fields": {"type": "array", "items": _string_schema(min_length=1)}})),
    _spec("inspect.sculpt_quality", "Inspect sculpt density and stage quality.", input_schema=_object_schema({"target": _string_schema(min_length=1), "selection": _SELECTION})),
    _spec("toolbox.capabilities", "Return capability catalog and quality bar.", input_schema=_object_schema({"profile": _string_schema(min_length=1), "include_examples": {"type": "boolean"}})),
    _spec("workflow.describe", "Describe a named workflow profile.", input_schema=_object_schema({"profile": _string_schema(min_length=1), "include_examples": {"type": "boolean"}})),
    _spec("collection.group_objects", "Group objects into a named collection.", mutating=True, required_args=("targets",), input_schema=_object_schema({"targets": {"type": "array", "minItems": 1, "maxItems": 512, "items": _string_schema(min_length=1)}, "name": _string_schema(min_length=1), "exclusive": {"type": "boolean"}, "include_children": {"type": "boolean"}})),
    _spec("curve.subdivide", "Subdivide every spline of a curve.", mutating=True, needs_verifier=True, required_args=("target",), input_schema=_object_schema({"target": _string_schema(min_length=1), "cuts": {"type": "integer", "minimum": 1, "maximum": 32}})),
    _spec("object.convert", "Convert an object to a supported Blender type.", mutating=True, required_args=("target",), input_schema=_object_schema({"target": _string_schema(min_length=1), "type": {"type": "string"}})),
    _spec("object.transform_apply", "Apply selected object transforms.", mutating=True, required_args=("target",), input_schema=_object_schema({"target": _string_schema(min_length=1), "location": {"type": "boolean"}, "rotation": {"type": "boolean"}, "scale": {"type": "boolean"}})),
    _spec("mesh.attribute_write", "Write a bounded mesh attribute field.", mutating=True, required_args=("target", "name", "domain", "data_type", "values"), input_schema=_object_schema({"target": _string_schema(min_length=1), "name": _string_schema(min_length=1), "domain": {"type": "string"}, "data_type": {"type": "string"}, "values": {"type": "array"}})),
    _spec("mesh.attribute_read", "Read a bounded mesh attribute field.", required_args=("target", "name"), input_schema=_object_schema({"target": _string_schema(min_length=1), "name": _string_schema(min_length=1), "sample_limit": {"type": "integer", "minimum": 1, "maximum": 4096}})),
    _spec("mesh.geometry_query", "Query mesh geometry in a declared space.", required_args=("target", "field"), input_schema=_object_schema({"target": _string_schema(min_length=1), "field": _string_schema(min_length=1), "space": {"type": "string"}, "sample_limit": {"type": "integer", "minimum": 1, "maximum": 4096}})),
    _spec("mesh.region_define", "Define a named bounded mesh region.", mutating=True, required_args=("target", "name"), input_schema=_object_schema({"target": _string_schema(min_length=1), "name": _string_schema(min_length=1), "selection": _SELECTION})),
    _spec("mesh.region_to_loop", "Convert a named mesh region to a boundary loop.", required_args=("target",), input_schema=_object_schema({"target": _string_schema(min_length=1), "selection": _SELECTION})),
    _spec("mesh.repair", "Repair bounded mesh topology issues.", mutating=True, needs_verifier=True, required_args=("target",), input_schema=_object_schema({"target": _string_schema(min_length=1), "merge_distance": {"type": "number", "minimum": 0}, "fill_holes": {"type": "boolean"}})),
    _spec("mesh.separate", "Separate a selected mesh region.", mutating=True, needs_verifier=True, required_args=("target",), input_schema=_object_schema({"target": _string_schema(min_length=1), "selection": _SELECTION, "mode": {"type": "string"}})),
    _spec("mesh.subdivide_adaptive", "Subdivide a mesh to a target edge length.", mutating=True, needs_verifier=True, required_args=("target",), input_schema=_object_schema({"target": _string_schema(min_length=1), "target_edge_length": {"type": "number", "exclusiveMinimum": 0}, "selection": _SELECTION})),
    _spec("mesh.symmetrize", "Symmetrize a mesh across a declared axis.", mutating=True, needs_verifier=True, required_args=("target", "direction"), input_schema=_object_schema({"target": _string_schema(min_length=1), "direction": {"type": "string"}, "selection": _SELECTION})),
    _spec("mesh.duplicate_region", "Duplicate a selected mesh region.", mutating=True, needs_verifier=True, required_args=("target",), input_schema=_object_schema({"target": _string_schema(min_length=1), "selection": _SELECTION}, additional_properties=True)),
    _spec("mesh.extrude_individual", "Extrude selected faces individually.", mutating=True, needs_verifier=True, required_args=("target",), input_schema=_object_schema({"target": _string_schema(min_length=1), "selection": _SELECTION, "distance": {"type": "number"}})),
    _spec("mesh.inset_individual", "Inset selected faces individually.", mutating=True, needs_verifier=True, required_args=("target",), input_schema=_object_schema({"target": _string_schema(min_length=1), "selection": _SELECTION, "thickness": {"type": "number", "exclusiveMinimum": 0}})),
    _spec("mesh.bridge_edge_loops", "Bridge two selected edge loops.", mutating=True, needs_verifier=True, required_args=("target",), input_schema=_object_schema({"target": _string_schema(min_length=1), "selection": _SELECTION})),
    _spec("mesh.loop_cut", "Insert a bounded loop cut.", mutating=True, needs_verifier=True, required_args=("target",), input_schema=_object_schema({"target": _string_schema(min_length=1), "cuts": {"type": "integer", "minimum": 1, "maximum": 32}, "selection": _SELECTION})),
    _spec("mesh.cut_plane", "Cut mesh geometry with a bounded plane.", mutating=True, needs_verifier=True, required_args=("target", "point", "normal"), input_schema=_object_schema({"target": _string_schema(min_length=1), "point": _VEC3, "normal": _VEC3, "side": {"type": "string"}, "distance": {"type": "number", "minimum": 0}, "cap": {"type": "boolean"}, "coordinate_frame": _COORDINATE_FRAME})),
    _spec("mesh.cut_curve", "Cut mesh geometry along a bounded curve.", mutating=True, needs_verifier=True, required_args=("target", "points"), input_schema=_object_schema({"target": _string_schema(min_length=1), "points": _points_schema(), "depth": {"type": "number"}})),
    _spec("sculpt.stroke_batch", "Apply bounded projected sculpt strokes.", mutating=True, needs_verifier=True, required_args=("strokes",), input_schema=_object_schema({"strokes": {"type": "array", "minItems": 1, "maxItems": 256, "items": {"type": "object"}}, "atomic": {"type": "boolean"}})),
    _spec("sculpt.surface_patch_batch", "Apply bounded surface patches.", mutating=True, needs_verifier=True, required_args=("patches",), input_schema=_object_schema({"target": _string_schema(min_length=1), "patches": {"type": "array", "minItems": 1, "maxItems": 256, "items": {"type": "object"}}, "atomic": {"type": "boolean"}})),
    _spec("sculpt.region_deform_batch", "Apply bounded region deformations.", mutating=True, needs_verifier=True, required_args=("deformations",), input_schema=_object_schema({"target": _string_schema(min_length=1), "deformations": {"type": "array", "minItems": 1, "maxItems": 256, "items": {"type": "object"}}, "atomic": {"type": "boolean"}})),
    _spec("sculpt.surface_prepare", "Prepare a mesh carrier for sculpting.", mutating=True, needs_verifier=True, required_args=("target",), input_schema=_object_schema({"target": _string_schema(min_length=1), "voxel_size": {"type": "number", "exclusiveMinimum": 0}, "levels": {"type": "integer", "minimum": 1, "maximum": 8}})),
    _spec("sculpt.materialize_multires", "Materialize a sculpt multiresolution surface.", mutating=True, needs_verifier=True, required_args=("target",), input_schema=_object_schema({"target": _string_schema(min_length=1), "levels": {"type": "integer", "minimum": 1, "maximum": 8}})),
)

TOOL_SPECS: Dict[str, ToolSpec] = {spec.name: spec for spec in (*_TOOL_SPECS, *_EXTRA_TOOL_SPECS)}


def _schema_type_ok(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _validate_schema(value: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    """Small dependency-free JSON Schema subset used at the protocol boundary."""
    if "oneOf" in schema:
        errors = []
        for candidate in schema["oneOf"]:
            try:
                _validate_schema(value, candidate, path)
                break
            except ProtocolError as exc:
                errors.append(str(exc))
        else:
            raise ProtocolError(f"{path} does not match any allowed shape", "invalid_args")
        return
    expected = schema.get("type")
    if expected:
        types = expected if isinstance(expected, list) else [expected]
        if not any(_schema_type_ok(value, item) for item in types):
            raise ProtocolError(f"{path} must be {','.join(types)}", "invalid_args")
    if "enum" in schema and value not in schema["enum"]:
        raise ProtocolError(f"{path} must be one of {schema['enum']}", "invalid_args")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            raise ProtocolError(f"{path} must contain at least {schema['minLength']} characters", "invalid_args")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            raise ProtocolError(f"{path} exceeds maximum length {schema['maxLength']}", "invalid_args")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ProtocolError(f"{path} must be finite", "invalid_args")
        if "minimum" in schema and value < schema["minimum"]:
            raise ProtocolError(f"{path} must be >= {schema['minimum']}", "invalid_args")
        if "maximum" in schema and value > schema["maximum"]:
            raise ProtocolError(f"{path} must be <= {schema['maximum']}", "invalid_args")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise ProtocolError(f"{path} must be > {schema['exclusiveMinimum']}", "invalid_args")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            raise ProtocolError(f"{path} must be < {schema['exclusiveMaximum']}", "invalid_args")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            raise ProtocolError(f"{path} must contain at least {schema['minItems']} items", "invalid_args")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise ProtocolError(f"{path} exceeds maximum item count {schema['maxItems']}", "invalid_args")
        if "items" in schema:
            for index, item in enumerate(value):
                _validate_schema(item, schema["items"], f"{path}[{index}]")
    if isinstance(value, Mapping):
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise ProtocolError(f"{path} missing required keys: {missing}", "invalid_args")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(properties), key=str)
            if unknown:
                raise ProtocolError(f"{path} contains unknown keys: {unknown}", "invalid_args")
        for key, child_schema in properties.items():
            if key in value:
                _validate_schema(value[key], child_schema, f"{path}.{key}")


def validate_action_args(action: str, args: Mapping[str, Any]) -> None:
    spec = get_tool_spec(action)
    _validate_schema(args, spec.input_schema or {"type": "object"})
    frame = args.get("coordinate_frame") if isinstance(args, Mapping) else None
    if isinstance(frame, Mapping):
        up_axis = frame.get("up_axis")
        front_axis = frame.get("front_axis")
        if up_axis is not None and front_axis is not None and str(up_axis)[-1:] == str(front_axis)[-1:]:
            raise ProtocolError("coordinate_frame up_axis and front_axis must be perpendicular", "invalid_args")
    if action == "scene.coordinate_system":
        up_axis = args.get("up_axis")
        front_axis = args.get("front_axis")
        if up_axis is not None and front_axis is not None and str(up_axis)[-1:] == str(front_axis)[-1:]:
            raise ProtocolError("scene.coordinate_system up_axis and front_axis must be perpendicular", "invalid_args")
        if args.get("origin") == "custom" and "custom_origin" not in args:
            raise ProtocolError("scene.coordinate_system custom origin requires custom_origin", "invalid_args")
    if action == "workflow.batch":
        forbidden = {"workflow.batch", "bpy.apply", "run_python", "session.create", "session.reset", "session.close", "scene.reset",
                     "render.views", "evidence.visual_review", "verify.run", "artifact.save_checkpoint", "artifact.export_glb"}
        for index, step in enumerate(args.get("steps") or ()):
            if not isinstance(step, Mapping):
                raise ProtocolError(f"$.steps[{index}] must be an object", "invalid_args")
            child_action = step.get("action")
            if not isinstance(child_action, str):
                raise ProtocolError(f"$.steps[{index}].action must be a string", "invalid_args")
            if child_action in forbidden:
                raise ProtocolError(f"workflow.batch cannot contain {child_action}", "invalid_args")
            try:
                child_spec = get_tool_spec(child_action)
            except ProtocolError as exc:
                raise ProtocolError(f"invalid workflow step {index}: {exc}", "invalid_args") from exc
            if not child_spec.mutating:
                raise ProtocolError(f"workflow.batch step {index} must be mutating: {child_action}", "invalid_args")
            child_args = step.get("args") or {}
            if not isinstance(child_args, Mapping):
                raise ProtocolError(f"$.steps[{index}].args must be an object", "invalid_args")
            try:
                validate_action_args(child_action, child_args)
            except ProtocolError as exc:
                raise ProtocolError(f"invalid workflow step {index}: {exc}", "invalid_args") from exc
    if action == "bpy.apply":
        if not args.get("source_path") and not args.get("source"):
            raise ProtocolError("bpy.apply requires source_path or source", "invalid_args")
        if not args.get("purpose"):
            raise ProtocolError("bpy.apply requires an explicit purpose", "invalid_args")


def get_tool_spec(name: str) -> ToolSpec:
    try:
        return TOOL_SPECS[name]
    except KeyError as exc:
        raise ProtocolError(f"unknown action: {name!r}", "unknown_action") from exc


def tool_registry() -> list[Dict[str, Any]]:
    return [TOOL_SPECS[name].as_dict() for name in sorted(TOOL_SPECS)]


def canonical_json(value: Any) -> str:
    """Serialize protocol values deterministically without lossy coercion."""
    _validate_json_value(value)
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"value is not valid JSON: {exc}", "invalid_json") from exc


def _validate_json_value(value: Any, path: str = "$", *, _stack: Optional[set[int]] = None) -> None:
    """Reject non-JSON values before an MCP/IPC boundary.

    Kept as a small public-private compatibility helper because older cached
    adapters import it directly.  It deliberately does not impose action
    schemas; :func:`validate_action_args` remains responsible for those.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolError(f"{path} must contain finite numbers", "invalid_json")
        return
    if isinstance(value, list):
        stack = _stack if _stack is not None else set()
        marker = id(value)
        if marker in stack:
            raise ProtocolError(f"{path} contains a cyclic reference", "invalid_json")
        stack.add(marker)
        try:
            for index, item in enumerate(value):
                _validate_json_value(item, f"{path}[{index}]", _stack=stack)
        finally:
            stack.remove(marker)
        return
    if isinstance(value, Mapping):
        stack = _stack if _stack is not None else set()
        marker = id(value)
        if marker in stack:
            raise ProtocolError(f"{path} contains a cyclic reference", "invalid_json")
        stack.add(marker)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ProtocolError(f"{path} object keys must be strings", "invalid_json")
                _validate_json_value(item, f"{path}.{key}", _stack=stack)
        finally:
            stack.remove(marker)
        return
    raise ProtocolError(f"{path} contains a non-JSON value", "invalid_json")


def content_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def request_fingerprint(request: Mapping[str, Any]) -> str:
    """Stable hash for deduplicating a canonical action request."""
    if isinstance(request, ActionRequest):
        request = request.as_dict()
    if not isinstance(request, Mapping):
        raise ProtocolError("request fingerprint input must be an object", "invalid_request")
    payload = {key: value for key, value in request.items() if key not in {"request_id", "idempotency_key"}}
    return content_hash(payload)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass(frozen=True)
class ActionRequest:
    request_id: str
    session_id: str
    episode_id: str
    step_id: int
    action: str
    args: Dict[str, Any] = field(default_factory=dict)
    expected_revision: Optional[int] = None
    idempotency_key: Optional[str] = None
    seed: Optional[int] = None
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ActionRequest":
        if not isinstance(raw, Mapping):
            raise ProtocolError("request must be a JSON object")
        # Validate the complete envelope before any recursive schema walk so
        # cyclic/non-finite provider values fail deterministically.
        _validate_json_value(raw)
        version = raw.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ProtocolError(f"unsupported schema_version: {version!r}", "unsupported_version")
        if "request_id" in raw:
            request_id = raw.get("request_id")
        else:
            request_id = new_id("req")
        if not isinstance(request_id, str) or not request_id:
            raise ProtocolError("request_id must be a non-empty string")
        session_id = raw.get("session_id")
        episode_id = raw.get("episode_id")
        action = raw.get("action")
        if not isinstance(session_id, str) or not session_id:
            raise ProtocolError("session_id must be a non-empty string")
        if not isinstance(episode_id, str) or not episode_id:
            raise ProtocolError("episode_id must be a non-empty string")
        if not isinstance(action, str) or not _ACTION_RE.match(action):
            raise ProtocolError("action must be a dotted lower-case name")
        spec = get_tool_spec(action)
        args = raw.get("args", {})
        if not isinstance(args, dict):
            raise ProtocolError("args must be an object")
        missing = [key for key in spec.required_args if key not in args]
        if missing:
            raise ProtocolError(f"missing required args for {action}: {missing}")
        validate_action_args(action, args)
        step_id = raw.get("step_id")
        if not isinstance(step_id, int) or isinstance(step_id, bool) or step_id < 0:
            raise ProtocolError("step_id must be a non-negative integer")
        expected_revision = raw.get("expected_revision")
        if expected_revision is not None and (
            not isinstance(expected_revision, int) or isinstance(expected_revision, bool) or expected_revision < 0
        ):
            raise ProtocolError("expected_revision must be a non-negative integer or null")
        seed = raw.get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ProtocolError("seed must be an integer or null")
        if seed is not None and not 0 <= seed <= MAX_SEED:
            raise ProtocolError(f"seed must be between 0 and {MAX_SEED}", "invalid_args")
        idempotency_key = raw.get("idempotency_key")
        if idempotency_key is not None and (not isinstance(idempotency_key, str) or not idempotency_key.strip()):
            raise ProtocolError("idempotency_key must be a non-empty string or null", "invalid_request")
        if action in {"geometry_nodes.apply_recipe", "material.apply_recipe"} and normalize_recipe is not None:
            normalized = normalize_recipe(args["recipe"])
            args = dict(args)
            args["recipe"] = normalized.as_dict()
        return cls(
            request_id=str(request_id),
            session_id=session_id,
            episode_id=episode_id,
            step_id=step_id,
            action=action,
            args=args,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            seed=seed,
            schema_version=version,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "expected_revision": self.expected_revision,
            "idempotency_key": self.idempotency_key,
            "action": self.action,
            "args": self.args,
            "seed": self.seed,
        }


@dataclass
class ActionResponse:
    request_id: str
    ok: bool
    revision: int
    result: Any = None
    error: Optional[Dict[str, Any]] = None
    state: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: list[Dict[str, Any]] = field(default_factory=list)
    duration_ms: int = 0
    schema_version: str = SCHEMA_VERSION

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "ok": bool(self.ok),
            "revision": int(self.revision),
            "result": self.result,
            "error": self.error,
            "state": self.state,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "duration_ms": int(self.duration_ms),
        }


def error_payload(code: str, message: str, *, retryable: bool = False, details: Any = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"code": code, "message": message, "retryable": bool(retryable)}
    if details is not None:
        payload["details"] = details
    return payload


def response_from_error(request_id: str, revision: int, exc: Exception, *, code: Optional[str] = None) -> ActionResponse:
    error_code = code or getattr(exc, "code", None) or "execution_error"
    details = getattr(exc, "details", None)
    return ActionResponse(
        request_id=request_id,
        ok=False,
        revision=revision,
        error=error_payload(error_code, str(exc), retryable=error_code in {"busy", "timeout", "blender_unavailable"}, details=details),
    )
