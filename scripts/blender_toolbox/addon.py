"""Canonical Blender-side executor for the Blender Toolbox.

Run inside Blender with:

    blender --background --python blender_toolbox/addon.py -- --socket /tmp/x.sock

The executor also exposes ``start_server()`` for registration from a normal
Blender addon.  All bpy work runs from a timer callback on Blender's main
thread; socket threads only parse JSON and enqueue requests.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.util
import json
import math
import os
import queue
import random
import socket
import threading
import time
import uuid
import tempfile
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

try:  # Blender imports this module with bpy available.
    import bmesh
    import bpy
    from mathutils import Euler, Matrix, Vector
    from mathutils.bvhtree import BVHTree
except ImportError:  # pragma: no cover - protocol-only imports use the package outside Blender.
    bpy = None
    bmesh = None
    Euler = None
    Matrix = None
    Vector = None
    BVHTree = None

try:
    from .protocol import (
        DEFAULT_MAX_PROJECT_POINTS,
        MAX_IPC_MESSAGE_BYTES,
        MAX_SEED,
        MAX_PROJECT_POINTS,
        ActionRequest,
        ActionResponse,
        ProtocolError,
        _validate_json_value,
        request_fingerprint,
        response_from_error,
    )
    from .state import observation, state_diff
except ImportError:  # Blender's ``--python path/to/addon.py`` has no package context.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from blender_toolbox.protocol import (
        DEFAULT_MAX_PROJECT_POINTS,
        MAX_IPC_MESSAGE_BYTES,
        MAX_SEED,
        MAX_PROJECT_POINTS,
        ActionRequest,
        ActionResponse,
        ProtocolError,
        _validate_json_value,
        request_fingerprint,
        response_from_error,
    )
    from blender_toolbox.state import observation, state_diff
from blender_toolbox.procedural import RecipeError, normalize_recipe
from blender_toolbox.protocol import content_hash, get_tool_spec
from blender_toolbox.sculpt_metrics import sculpt_quality_metrics
from blender_toolbox.version import TOOLBOX_VERSION, TOOLBOX_VERSION_INFO
from blender_toolbox.workflows import capability_catalog, describe_workflow, quality_bar, recommended_contract

bl_info = {
    "name": "Blender Toolbox Executor",
    "author": "3DCodeBench",
    "version": TOOLBOX_VERSION_INFO,
    "blender": (4, 2, 0),
    "location": "Text Editor / background mode",
    "description": "Local structured Blender actions with RL trajectory capture",
    "category": "Development",
}

ADDON_VERSION = TOOLBOX_VERSION
_UUID_PROP = "blender_toolbox_uuid"
_REF_PROP = "blender_toolbox_ref"
_SEMANTIC_PROP = "blender_toolbox_semantic_tags"
_ORIGIN_PROP = "blender_toolbox_origin"
_ROLE_PROP = "blender_toolbox_role"
_REPRESENTATION_PROP = "blender_toolbox_representation"
_QUALITY_STAGE_PROP = "blender_toolbox_quality_stage"
_REGION_PROP = "blender_toolbox_region_handles"
_COLLECTION_UUID_PROP = "blender_toolbox_collection_uuid"
_COORDINATE_PROP = "blender_toolbox_coordinate_system"
_ANCHORS_PROP = "blender_toolbox_anchors"
_ATTACHMENT_PROP = "blender_toolbox_attachment"
_SNAP_PROP = "blender_toolbox_surface_snap"
_CAMERA_TARGET_PROP = "blender_toolbox_camera_target"
_LOCAL_HOSTS = {"127.0.0.1", "localhost"}
_DEFAULT_EVALUATED_VERTEX_BUDGET = 250000
_DEFAULT_MATERIALIZE_VERTEX_BUDGET = 1000000
_SCULPT_STAGE_ORDER = {"primary": 0, "secondary": 1, "tertiary": 2, "cleanup": 3}
_DEFAULT_COORDINATE_SYSTEM = {
    "units": "meters", "up_axis": "POS_Z", "front_axis": "NEG_Y",
    "handedness": "right", "origin": "world_origin", "declared": False,
}
_VISUAL_CHECK_KEYS = ("floating", "overlap", "alignment", "surface_contact", "framing", "proportion")
_ANTI_SLOP_CHECK_KEYS = (
    "primitive_seams", "repeated_hard_shapes", "regular_rows", "floating_details",
    "wing_panel_look", "tail_fan", "anatomical_transitions", "material_boundaries",
    "camera_crops", "reference_consistency",
)
_VISUAL_SCORE_KEYS = ("silhouette", "anatomy", "surface_language", "variation", "materials", "reference")
_UNIT_TO_METERS = {"meters": 1.0, "centimeters": 0.01, "millimeters": 0.001}


class _IdentityTransform:
    """Tiny pure-Python stand-in used by protocol-only test doubles."""
    def __matmul__(self, value: Any) -> Any:
        return value


def _object_reference(value: Any) -> str:
    """Normalize an object declaration's stable id/ref aliases.

    Batch callers may use either ``id`` or ``ref``.  Accepting both is useful
    for compatibility, but silently choosing one when they disagree makes a
    replay target ambiguous, so conflicts are rejected at the boundary.
    """
    if isinstance(value, str):
        reference = value.strip()
        if reference:
            return reference
    if not isinstance(value, Mapping):
        raise ExecutorError("object reference must be a non-empty string or object", "invalid_args")
    raw_id = value.get("id")
    raw_ref = value.get("ref")
    id_value = raw_id.strip() if isinstance(raw_id, str) else raw_id
    ref_value = raw_ref.strip() if isinstance(raw_ref, str) else raw_ref
    if id_value is not None and (not isinstance(id_value, str) or not id_value):
        raise ExecutorError("object id must be a non-empty string", "invalid_args")
    if ref_value is not None and (not isinstance(ref_value, str) or not ref_value):
        raise ExecutorError("object ref must be a non-empty string", "invalid_args")
    if id_value is not None and ref_value is not None and id_value != ref_value:
        raise ExecutorError("object id and ref must match", "invalid_args")
    result = id_value if id_value is not None else ref_value
    if result is None:
        raise ExecutorError("object reference requires id or ref", "invalid_args")
    return str(result)


_QUALITY_STAGES = ("structure", "primary", "secondary", "tertiary", "technical", "evidence")
_QUALITY_DEFAULTS = {
    "version": "quality_contract.v1",
    "enforce": False,
    "min_quality": 0.86,
    "min_primary_vertices": 512,
    "min_secondary_vertices": 256,
    "min_tertiary_vertices": 192,
    "min_samples_per_feature": 8,
    "min_evidence_views": 4,
    "representation": {},
    "primary_refs": [],
    "secondary_refs": [],
    "required_stages": ["structure", "primary", "secondary", "tertiary", "technical", "evidence"],
    "reference_views": ["front", "three_quarter", "side", "top"],
    "feature_scales": [],
    "detail_regions": [],
    "required_semantic_parts": [],
    "identity": {},
    "scale": {},
    "require_identity": True,
    "require_scale": True,
    "require_secondary": True,
    "require_detail_plan": True,
    "require_feature_scales": True,
    "resolution": {"min_primary_vertices": 512, "min_secondary_vertices": 256, "min_tertiary_vertices": 192, "min_samples_per_feature": 8, "allow_low_resolution": False, "exception_reason": ""},
    "require_openings": False,
    "require_closed": False,
    "contacts": [],
    "assembly": {},
    "proportions": {},
    "technical": {"require_topology": True, "strict_topology": True, "require_connected": True, "require_high_resolution": True, "require_material": True, "require_uv": True},
    "evidence": {"min_views": 4, "require_render": True, "require_current_revision": True},
    "completion_gate": False,
    "require_visual_review": False,
    "required_evidence_types": ["beauty", "clay", "silhouette", "closeup"],
    "min_visual_views": 4,
    "min_visual_score": 0.85,
}

_QUALITY_ALIASES = {
    "primary_refs": "primary_refs", "secondary_refs": "secondary_refs", "required_stages": "required_stages",
    "reference_views": "reference_views", "silhouette_views": "reference_views", "feature_scales": "feature_scales",
    "feature_sizes": "feature_scales", "required_semantic_parts": "required_semantic_parts", "required_tags": "required_semantic_parts",
    "detail_regions": "detail_regions", "representation": "representation", "carrier_refs": "primary_refs", "technical": "technical",
    "min_quality": "min_quality", "min_evidence_views": "min_evidence_views", "require_secondary": "require_secondary",
    "require_detail_plan": "require_detail_plan", "require_feature_scales": "require_feature_scales", "require_openings": "require_openings",
    "openings": "require_openings", "contacts": "contacts", "assembly": "assembly", "proportions": "proportions", "negative_spaces": "negative_spaces",
    "identity": "identity", "scale": "scale", "require_identity": "require_identity", "require_scale": "require_scale", "require_closed": "require_closed",
    "resolution": "resolution", "allow_low_resolution": "allow_low_resolution", "resolution_exception": "resolution_exception",
}

def _merge_mappings(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(result.get(str(key)), Mapping) and isinstance(value, Mapping):
            result[str(key)] = _merge_mappings(result[str(key)], value)
        else:
            result[str(key)] = copy.deepcopy(value)
    return result

def _workflow_contract(profile: Optional[str]) -> Dict[str, Any]:
    if not profile:
        return {}
    try:
        return recommended_contract(str(profile))
    except (ValueError, NameError):
        return {}

def _canonical_quality_block(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, Any] = {}
    for key, item in value.items():
        dest = _QUALITY_ALIASES.get(str(key), str(key))
        out[dest] = copy.deepcopy(item)
    return out


def _quality_contract(profile: Optional[str] = None, task_spec: Optional[Mapping[str, Any]] = None, explicit: Optional[Mapping[str, Any]] = None, *, workflow_profile: Optional[str] = None) -> Dict[str, Any]:
    """Normalize the opt-in quality contract without forcing a domain recipe."""
    raw: Mapping[str, Any] = {}
    if isinstance(explicit, Mapping):
        raw = explicit
    elif isinstance(task_spec, Mapping):
        candidate = task_spec.get("quality") or task_spec.get("quality_contract")
        if isinstance(candidate, Mapping):
            raw = candidate
    profile_contract = _workflow_contract(workflow_profile)
    profile_quality = profile_contract.get("quality") if isinstance(profile_contract.get("quality"), Mapping) else {}
    contract = _merge_mappings(_QUALITY_DEFAULTS, profile_quality)
    if isinstance(profile_contract, Mapping):
        for alias, target in _QUALITY_ALIASES.items():
            if alias in profile_contract and target not in contract:
                contract[target] = copy.deepcopy(profile_contract[alias])
        if profile_contract.get("require_closed") is not None:
            contract["require_closed"] = bool(profile_contract.get("require_closed"))
        if profile_contract.get("required_tags") is not None:
            contract["required_semantic_parts"] = copy.deepcopy(profile_contract.get("required_tags"))
    contract = _merge_mappings(contract, _canonical_quality_block(raw))
    if isinstance(raw.get("representation"), Mapping):
        contract["representation"] = copy.deepcopy(raw["representation"])
    for key in ("primary_refs", "secondary_refs", "reference_views"):
        if isinstance(contract.get(key), str):
            contract[key] = [contract[key].strip()]
    if not contract.get("primary_refs") and isinstance(contract.get("representation"), Mapping):
        contract["primary_refs"] = [contract["representation"].get("primary_refs")] if isinstance(contract["representation"].get("primary_refs"), str) else list(contract["representation"].get("primary_refs") or [])
    if isinstance(contract.get("representation"), Mapping):
        representation = dict(contract["representation"])
        rep_refs = representation.get("primary_refs")
        if isinstance(rep_refs, str):
            rep_refs = [rep_refs.strip()]
        representation["primary_refs"] = list(rep_refs or contract.get("primary_refs") or [])
        contract["representation"] = representation
    if isinstance(contract.get("required_semantic_parts"), str):
        contract["required_semantic_parts"] = [contract["required_semantic_parts"]]
    if isinstance(contract.get("required_semantic_parts"), (list, tuple)):
        contract["required_semantic_parts"] = [str(item).strip() for item in contract["required_semantic_parts"] if str(item).strip()]
    if isinstance(contract.get("feature_scales"), (list, tuple)):
        values = []
        for value in contract["feature_scales"]:
            try:
                number = float(value)
                if math.isfinite(number) and number > 0:
                    values.append(number)
            except (TypeError, ValueError):
                continue
        contract["feature_scales"] = values
    profile_name = str(profile or "").strip().lower()
    configured = profile_name in {"quality_first", "structural", "production", "organic", "strict"} or bool(raw) or bool(workflow_profile)
    requested_enforce = raw.get("enforce") if isinstance(raw, Mapping) else None
    contract["requested_enforce"] = requested_enforce
    contract["requested_required_stages"] = copy.deepcopy(raw.get("required_stages")) if isinstance(raw, Mapping) and "required_stages" in raw else None
    if profile_name == "advisory":
        contract["enforce"] = False
    else:
        contract["enforce"] = True if profile_name in {"quality_first", "structural", "production", "organic", "strict"} or workflow_profile else bool(requested_enforce if requested_enforce is not None else bool(raw))
    if contract["enforce"]:
        contract["required_stages"] = ["structure", "primary", "secondary", "tertiary", "technical", "evidence"]
        contract["technical"] = _merge_mappings(contract.get("technical") or {}, {"require_topology": True, "require_connected": True})
        contract["min_quality"] = 0.86
        contract["min_primary_vertices"] = 512
        contract["min_secondary_vertices"] = 256
        contract["min_tertiary_vertices"] = 192
        contract["min_samples_per_feature"] = 8
    if contract.get("allow_low_resolution") and not isinstance(contract.get("resolution_exception"), Mapping):
        contract["allow_low_resolution"] = False
    resolution = _merge_mappings(_QUALITY_DEFAULTS["resolution"], contract.get("resolution") or {})
    exception = contract.get("resolution_exception") or {}
    if isinstance(exception, Mapping) and exception.get("exception_reason") and contract.get("allow_low_resolution"):
        resolution["allow_low_resolution"] = True
        resolution["exception_reason"] = str(exception["exception_reason"])
    contract["resolution"] = resolution
    contract["evidence"] = _merge_mappings(_QUALITY_DEFAULTS["evidence"], contract.get("evidence") or {})
    # Keep the normalized default permissive for structural workflows while
    # remembering whether a task explicitly asked verify.run to require a
    # render. Completion-gated profiles still require visual evidence through
    # the dedicated visual gate below.
    contract["render_required_explicit"] = bool(
        (isinstance(raw, Mapping) and isinstance(raw.get("evidence"), Mapping) and "require_render" in raw.get("evidence", {}))
        or (isinstance(raw, Mapping) and "require_render" in raw)
    )
    contract["min_evidence_views"] = max(4, min(32, int(contract.get("min_evidence_views", 4) or 4)))
    contract["evidence"]["min_views"] = max(contract["min_evidence_views"], min(32, int(contract["evidence"].get("min_views", 4) or 4)))
    if workflow_profile == "vehicle":
        contract["reference_views"] = ["front", "side", "top", "three_quarter"]
    contract["configured"] = configured
    contract["completion_gate"] = bool(contract.get("completion_gate", False)) or profile_name in {"production", "organic", "strict"}
    contract["require_visual_review"] = bool(contract.get("require_visual_review", False)) or bool(contract.get("completion_gate"))
    try:
        contract["min_visual_views"] = max(1, min(64, int(contract.get("min_visual_views", 4) or 4)))
    except (TypeError, ValueError):
        contract["min_visual_views"] = 4
    try:
        contract["min_visual_score"] = max(0.0, min(1.0, float(contract.get("min_visual_score", 0.85))))
    except (TypeError, ValueError):
        contract["min_visual_score"] = 0.85
    evidence_types = contract.get("required_evidence_types") or _QUALITY_DEFAULTS["required_evidence_types"]
    if isinstance(evidence_types, str):
        evidence_types = [evidence_types]
    contract["required_evidence_types"] = [str(value).strip() for value in evidence_types if str(value).strip()]
    contract["profile"] = profile_name if profile_name in {"quality_first", "structural", "production", "organic", "strict"} and contract.get("enforce") else ("quality_first" if contract.get("enforce") else "advisory")
    contract["version"] = "quality_contract.v1"
    try:
        contract["min_quality"] = max(0.0, min(1.0, float(contract.get("min_quality", 0.86))))
    except (TypeError, ValueError):
        contract["min_quality"] = 0.86 if contract.get("enforce") else 0.86
    return contract


def _authoring_metadata(obj: Any, args: Mapping[str, Any], *, origin: Optional[str] = None, default_representation: Optional[str] = None) -> None:
    """Persist representation/role metadata so quality audits can be semantic."""
    if origin:
        obj[_ORIGIN_PROP] = origin
    elif args.get("origin"):
        obj[_ORIGIN_PROP] = str(args["origin"])
    if args.get("role") is not None:
        obj[_ROLE_PROP] = str(args["role"])
    if args.get("representation") is not None:
        obj[_REPRESENTATION_PROP] = str(args["representation"])
    elif default_representation:
        obj[_REPRESENTATION_PROP] = default_representation
    if args.get("quality_stage") is not None:
        obj[_QUALITY_STAGE_PROP] = str(args["quality_stage"])


class ExecutorError(RuntimeError):
    def __init__(self, message: str, code: str = "execution_error", details: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


def _require_bpy() -> None:
    if bpy is None:
        raise ExecutorError("Blender bpy is unavailable", "blender_unavailable")


def _coordinate_system() -> Dict[str, Any]:
    """Return the persisted scene coordinate contract, never an implicit guess."""
    if bpy is None:
        return dict(_DEFAULT_COORDINATE_SYSTEM)
    # Keep the helper usable during early startup and in lightweight pure-
    # Python test doubles that only expose ``bpy.data``.  A missing scene
    # property is equivalent to an uninitialised contract, so fall back to
    # the explicit defaults rather than raising an unrelated AttributeError.
    context = getattr(bpy, "context", None)
    scene = getattr(context, "scene", None)
    getter = getattr(scene, "get", None)
    if not callable(getter):
        return dict(_DEFAULT_COORDINATE_SYSTEM)
    raw = getter(_COORDINATE_PROP)
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
            if isinstance(value, Mapping):
                return {**_DEFAULT_COORDINATE_SYSTEM, **dict(value)}
        except (TypeError, ValueError):
            pass
    return dict(_DEFAULT_COORDINATE_SYSTEM)


def _coordinate_frame(args: Mapping[str, Any]) -> Dict[str, Any]:
    frame = dict(args.get("coordinate_frame") or {"space": "WORLD"})
    scene_frame = _coordinate_system()
    frame.setdefault("space", "WORLD")
    frame.setdefault("units", scene_frame.get("units", "meters"))
    frame.setdefault("up_axis", scene_frame.get("up_axis", "POS_Z"))
    frame.setdefault("front_axis", scene_frame.get("front_axis", "NEG_Y"))
    frame.setdefault("handedness", scene_frame.get("handedness", "right"))
    frame.setdefault("origin", scene_frame.get("origin", "world_origin"))
    space = str(frame.get("space", "WORLD")).upper()
    if space not in {"WORLD", "LOCAL", "PARENT"}:
        raise ExecutorError(f"unsupported coordinate space: {space}", "invalid_args")
    if str(frame.get("handedness", "right")).lower() != "right":
        raise ExecutorError("only right-handed coordinates are supported", "invalid_args")
    up_axis = str(frame.get("up_axis", "POS_Z"))
    front_axis = str(frame.get("front_axis", "NEG_Y"))
    if up_axis[-1:] not in "XYZ" or front_axis[-1:] not in "XYZ" or up_axis[-1:] == front_axis[-1:]:
        raise ExecutorError("coordinate_frame up_axis and front_axis must be perpendicular", "invalid_args")
    frame["space"] = space
    if str(frame.get("origin", "world_origin")) == "custom":
        custom = frame.get("custom_origin", scene_frame.get("custom_origin"))
        if custom is None:
            raise ExecutorError("custom coordinate origin is not declared", "precondition_failed")
        # custom_origin is persisted in meters, independent of the request's
        # display units.  Keep it in the frame so replay has no hidden state.
        # Scene custom origins are persisted in world meters.  Preserve that
        # canonical storage instead of re-scaling them by each request's
        # display units.
        frame["custom_origin"] = [float(value) for value in custom]
    return frame


def _coordinate_basis(frame: Optional[Mapping[str, Any]] = None) -> Any:
    """Return a right-handed basis for the declared scene convention.

    Toolbox vectors are authored in a canonical frame (+X right, +Y back,
    +Z up).  ``up_axis`` and ``front_axis`` map that canonical basis into
    Blender world space.  Keeping the conversion here makes the frame
    contract executable instead of metadata-only.
    """
    frame = frame or _coordinate_system()
    if Vector is None or Matrix is None:
        return _IdentityTransform()
    up = _axis_vector(str(frame.get("up_axis", "POS_Z")))
    front = _axis_vector(str(frame.get("front_axis", "NEG_Y")))
    canonical_y = -front
    canonical_x = canonical_y.cross(up)
    if canonical_x.length < 1e-9:
        raise ExecutorError("coordinate_frame axes must be perpendicular", "invalid_args")
    canonical_x.normalize(); canonical_y.normalize(); up.normalize()
    return Matrix((canonical_x, canonical_y, up)).transposed()


def _scene_origin_world(frame: Optional[Mapping[str, Any]] = None) -> Any:
    frame = frame or _coordinate_system()
    if str(frame.get("origin", "world_origin")) == "custom":
        raw = frame.get("custom_origin") or _coordinate_system().get("custom_origin")
        if raw is None:
            raise ExecutorError("custom coordinate origin requires custom_origin", "precondition_failed")
        return _length_vector(raw, "custom_origin", {"units": "meters"})
    # asset_origin is explicit contract metadata.  A future asset marker can
    # replace this origin without changing action payload semantics.
    return Vector((0.0, 0.0, 0.0)) if Vector is not None else (0.0, 0.0, 0.0)


def _point_to_world_for_object(obj: Any, value: Any, frame: Mapping[str, Any], name: str, *, relative: bool = False) -> Any:
    point = _length_vector(value, name, frame)
    point = _coordinate_basis(frame) @ point
    space = str(frame.get("space", "WORLD")).upper()
    if space == "WORLD":
        return (Vector((0.0, 0.0, 0.0)) if relative else _scene_origin_world(frame)) + point
    if space == "PARENT":
        parent = getattr(obj, "parent", None)
        if parent is None:
            raise ExecutorError(f"{name} uses PARENT space but target has no parent", "precondition_failed")
        return (parent.matrix_world.to_3x3() @ point) + (Vector((0.0, 0.0, 0.0)) if relative else parent.matrix_world.translation.copy())
    return (obj.matrix_world.to_3x3() @ point) + (Vector((0.0, 0.0, 0.0)) if relative else obj.matrix_world.translation.copy())


def _direction_to_world(obj: Any, value: Any, frame: Mapping[str, Any], name: str) -> Any:
    vector = Vector(_as_float3(value, name))
    result = _coordinate_basis(frame) @ vector
    space = str(frame.get("space", "WORLD")).upper()
    if space == "PARENT":
        parent = getattr(obj, "parent", None)
        if parent is None:
            raise ExecutorError(f"{name} uses PARENT space but target has no parent", "precondition_failed")
        result = parent.matrix_world.to_quaternion() @ result
    elif space == "LOCAL":
        result = obj.matrix_world.to_quaternion() @ result
    if result.length < 1e-9:
        raise ExecutorError(f"{name} must be non-zero", "invalid_args")
    return result.normalized()


def _point_to_object_local(obj: Any, value: Any, frame: Mapping[str, Any], name: str) -> Any:
    if str(frame.get("space", "WORLD")).upper() == "LOCAL":
        return _coordinate_basis(frame) @ _length_vector(value, name, frame)
    return obj.matrix_world.inverted() @ _point_to_world_for_object(obj, value, frame, name)


def _point_to_world(value: Any, name: str, frame: Mapping[str, Any]) -> Any:
    """Convert a point (as opposed to a length/vector) into world meters."""
    point = _coordinate_basis(frame) @ _length_vector(value, name, frame)
    if str(frame.get("space", "WORLD")).upper() == "WORLD":
        origin = _scene_origin_world(frame)
        if Vector is None:
            point = tuple(float(a) + float(b) for a, b in zip(point, origin))
        else:
            point = point + origin
    return point


def _creation_coordinate_frame(args: Mapping[str, Any], *, allow_relative: bool = False) -> Dict[str, Any]:
    """Resolve a creation frame and reject relative spaces without a reference.

    Newly-created objects have no parent or local frame yet.  Accepting
    ``LOCAL``/``PARENT`` here used to store misleading metadata while placing
    vertices in world space.  Creation actions therefore require WORLD; use
    ``object.parent_set``/``assembly.attach`` or a transform action after
    creation for relative placement.
    """
    frame = _coordinate_frame(args)
    if not allow_relative and str(frame.get("space", "WORLD")).upper() != "WORLD":
        raise ExecutorError("creation actions require coordinate_frame.space='WORLD'; use parent_set/attach for relative placement", "invalid_args")
    return frame


def _vector_to_world(obj: Any, value: Any, space: str, name: str) -> Any:
    """Convert an unscaled direction/offset from a declared object frame."""
    vector = _coordinate_basis({**_coordinate_system(), "space": str(space or "WORLD")}) @ Vector(_as_float3(value, name))
    space = str(space or "WORLD").upper()
    if space == "WORLD":
        return vector
    if space == "PARENT":
        parent = getattr(obj, "parent", None)
        if parent is None:
            return vector
        return parent.matrix_world.to_quaternion() @ vector
    if space == "LOCAL":
        return obj.matrix_world.to_quaternion() @ vector
    raise ExecutorError(f"unsupported coordinate space: {space}", "invalid_args")


def _world_delta(obj: Any, value: Any, space: str) -> Any:
    """Convert a translation delta using rotation only (never inherited scale)."""
    space = str(space or "WORLD").upper()
    if space == "WORLD":
        return value
    if space == "PARENT":
        parent = getattr(obj, "parent", None)
        return (parent.matrix_world.to_quaternion() @ value) if parent is not None else value
    if space == "LOCAL":
        return obj.matrix_world.to_quaternion() @ value
    raise ExecutorError(f"unsupported coordinate space: {space}", "invalid_args")


def _set_world_rotation(obj: Any, rotation: Any) -> None:
    """Set an object's world rotation while preserving world location and scale.

    Blender's ``Matrix.rotation_part`` is not available on all supported
    versions.  Assigning the world matrix through ``LocRotScale`` (or the
    equivalent explicit product) also avoids silently interpreting a WORLD
    rotation as a parent-local rotation.
    """
    world = obj.matrix_world.copy()
    location = world.translation.copy()
    scale = world.to_scale() if hasattr(world, "to_scale") else obj.scale.copy()
    try:
        matrix = Matrix.LocRotScale(location, rotation, scale)
    except AttributeError:
        matrix = Matrix.Translation(location) @ rotation.to_matrix().to_4x4() @ Matrix.Diagonal((float(scale.x), float(scale.y), float(scale.z), 1.0))
    obj.matrix_world = matrix


def _length_vector(value: Any, name: str, frame: Optional[Mapping[str, Any]] = None) -> Any:
    units = str((frame or {}).get("units") or _coordinate_system().get("units", "meters"))
    scale = _UNIT_TO_METERS.get(units)
    if scale is None:
        raise ExecutorError(f"unsupported length units: {units}", "invalid_args")
    components = _as_float3(value, name)
    # Protocol-only tests and startup validation may run without mathutils.
    # Return a plain tuple in that environment; Blender receives a real
    # ``mathutils.Vector`` and retains the same numeric semantics.
    if Vector is None:
        return tuple(component * scale for component in components)
    return Vector(components) * scale


def _length_value(value: Any, name: str, frame: Optional[Mapping[str, Any]] = None) -> float:
    units = str((frame or {}).get("units") or _coordinate_system().get("units", "meters"))
    scale = _UNIT_TO_METERS.get(units)
    if scale is None:
        raise ExecutorError(f"unsupported length units: {units}", "invalid_args")
    try:
        result = float(value) * scale
    except (TypeError, ValueError) as exc:
        raise ExecutorError(f"{name} must be numeric", "invalid_args") from exc
    if not math.isfinite(result):
        raise ExecutorError(f"{name} must be finite", "invalid_args")
    return result


def _store_json_prop(obj: Any, key: str, value: Any) -> None:
    obj[key] = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _load_json_prop(obj: Any, key: str, default: Any) -> Any:
    raw = obj.get(key) if hasattr(obj, "get") else None
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return default
    return raw if raw is not None else default


def _axis_vector(axis: str) -> Any:
    value = str(axis or "Z").upper()
    sign = -1.0 if value.startswith("-") else 1.0
    letter = value[-1:]
    if letter not in "XYZ":
        raise ExecutorError(f"axis must be one of X, -X, Y, -Y, Z, -Z: {axis}", "invalid_args")
    result = Vector((0.0, 0.0, 0.0))
    result["XYZ".index(letter)] = sign
    return result


def _as_float3(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ExecutorError(f"{name} must be a 3-item array", "invalid_args")
    try:
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    except (TypeError, ValueError) as exc:
        raise ExecutorError(f"{name} must contain numbers", "invalid_args") from exc


def _stable_uuid(obj: Any) -> str:
    value = obj.get(_UUID_PROP) if hasattr(obj, "get") else None
    if not value:
        value = f"obj-{uuid.uuid4().hex}"
        obj[_UUID_PROP] = value
    return str(value)


def _semantic_tags(obj: Any) -> list[str]:
    """Convert Blender IDProperty arrays to plain JSON-safe strings."""
    raw = obj.get(_SEMANTIC_PROP, []) if hasattr(obj, "get") else []
    if isinstance(raw, str):
        return [raw]
    try:
        return [str(value) for value in raw]
    except TypeError:
        return []


def _region_handles(obj: Any) -> list[str]:
    if not hasattr(obj, "get"):
        return []
    values = obj.get(_REGION_PROP, []) or []
    return sorted({f"region:{value}" for value in values})




def _collection_by_ref(ref: Any) -> Any:
    _require_bpy()
    if not isinstance(ref, str) or not ref:
        raise ExecutorError("collection reference must be a non-empty string", "invalid_args")
    for collection in bpy.data.collections:
        if collection.name == ref or collection.get(_COLLECTION_UUID_PROP) == ref:
            return collection
    raise ExecutorError(f"collection not found: {ref}", "not_found")


def _stable_collection_uuid(collection: Any) -> str:
    value = collection.get(_COLLECTION_UUID_PROP) if hasattr(collection, "get") else None
    if not value:
        value = f"col-{uuid.uuid4().hex}"
        collection[_COLLECTION_UUID_PROP] = value
    return str(value)


def _unlink_from_all(obj: Any) -> None:
    """Unlink an object from every collection before exclusive regrouping."""
    for collection in list(getattr(obj, "users_collection", ())):
        collection.objects.unlink(obj)


def _mesh_stats(obj: Any) -> Dict[str, Any]:
    if obj.type != "MESH" or obj.data is None:
        return {"vertices": 0, "faces": 0, "triangles": 0}
    mesh = obj.data
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.polygons),
        "triangles": sum(max(0, len(poly.vertices) - 2) for poly in mesh.polygons),
    }


def _mesh_geometry_hash(obj: Any) -> Optional[str]:
    """Hash mesh coordinates and connectivity without storing the full mesh."""
    if obj.type != "MESH" or obj.data is None:
        return None
    payload = {
        "vertices": [[round(float(value), 8) for value in vertex.co] for vertex in obj.data.vertices],
        "polygons": [list(map(int, polygon.vertices)) for polygon in obj.data.polygons],
    }
    return content_hash(payload)


def _mesh_attribute_summary(obj: Any) -> list[Dict[str, Any]]:
    if obj.type != "MESH" or obj.data is None:
        return []
    result = []
    for attr in sorted(obj.data.attributes, key=lambda item: item.name):
        # Blender exposes transient internal selection/corner attributes with
        # a leading dot; they are viewport bookkeeping, not model state.
        if attr.name.startswith("."):
            continue
        field = _attribute_field(attr.data_type)
        values = [_attribute_item_value(item, field) for item in attr.data]
        result.append({"name": attr.name, "domain": attr.domain, "data_type": attr.data_type, "count": len(values), "hash": content_hash(values)})
    return result


def _curve_geometry_hash(obj: Any) -> Optional[str]:
    if obj.type != "CURVE" or obj.data is None:
        return None
    payload = []
    for spline in obj.data.splines:
        points = spline.bezier_points if spline.type == "BEZIER" else spline.points
        payload.append({
            "type": spline.type,
            "cyclic": bool(spline.use_cyclic_u),
            "points": [[round(float(value), 8) for value in point.co] for point in points],
            "radii": [round(float(point.radius), 8) for point in points],
        })
    return content_hash(payload)


def _uv_layer_summary(obj: Any) -> list[Dict[str, Any]]:
    if obj.type != "MESH" or obj.data is None:
        return []
    layers = []
    for layer in obj.data.uv_layers:
        coordinates = [[round(float(loop.uv.x), 8), round(float(loop.uv.y), 8)] for loop in layer.data]
        layers.append({
            "name": layer.name,
            "active": layer == obj.data.uv_layers.active,
            "loops": len(coordinates),
            "uv_hash": content_hash(coordinates),
        })
    return layers


def _vertex_group_hash(obj: Any) -> Optional[str]:
    if obj.type != "MESH" or obj.data is None or not obj.vertex_groups:
        return None
    values = {}
    for group in sorted(obj.vertex_groups, key=lambda item: item.name):
        weights = []
        for vertex in obj.data.vertices:
            try:
                weights.append(round(float(group.weight(vertex.index)), 8))
            except RuntimeError:
                weights.append(0.0)
        values[group.name] = weights
    return content_hash(values)


def _aabb(obj: Any) -> Dict[str, list[float]]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    if not corners:
        return {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
    return {
        "min": [round(float(min(point[i] for point in corners)), 8) for i in range(3)],
        "max": [round(float(max(point[i] for point in corners)), 8) for i in range(3)],
    }


def _look_at(obj: Any, target: Any) -> None:
    point = target.copy() if Vector is not None and isinstance(target, Vector) else Vector(_as_float3(target, "target"))
    direction = point - obj.location
    if direction.length > 1e-9:
        obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _scene_contract(profile: Optional[str] = None, task_spec: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Return the small amount of scene metadata needed before authoring.

    Keeping this contract explicit removes the common create/reset/inspect
    preflight sequence while leaving the full scene census available on
    demand.  The coordinate convention is intentionally stable across
    recipes and replays.
    """
    coordinate = _coordinate_system()
    contract: Dict[str, Any] = {
        "units": coordinate.get("units", "meters"),
        "up_axis": coordinate.get("up_axis", "POS_Z"),
        "front_axis": coordinate.get("front_axis", "NEG_Y"),
        "handedness": "right",
        "world_axes": {"x": "length_or_custom", "y": "width_or_custom", "z": "height"},
        "origin": coordinate.get("origin", "world_origin"),
        "default_profiles": ["vehicle", "hard_surface", "sculpt"],
        "quality_bar_version": "quality_bar.v1",
        "coordinate_system": coordinate,
    }
    if profile:
        contract["profile"] = str(profile)
    if isinstance(task_spec, Mapping) and task_spec:
        contract["task_spec_keys"] = sorted(str(key) for key in task_spec)
    return contract


def _scene_content_hash() -> Optional[str]:
    """Hash only Blender scene content, excluding executor-owned evidence.

    Render/review/verify records live in the executor observation and must not
    make otherwise identical geometry look stale.  This hash is therefore the
    common identity used by all evidence and completion gates.
    """
    if bpy is None:
        return None
    try:
        summary = scene_summary()
        scene = bpy.context.scene
        summary["active_camera"] = _stable_uuid(scene.camera) if scene.camera else None
        summary["render_settings"] = {
            "engine": str(scene.render.engine),
            "resolution": [int(scene.render.resolution_x), int(scene.render.resolution_y), int(scene.render.resolution_percentage)],
            "frame_start": int(scene.frame_start), "frame_end": int(scene.frame_end), "fps": float(scene.render.fps),
        }
        summary["camera_targets"] = {
            _stable_uuid(obj): _load_json_prop(obj, _CAMERA_TARGET_PROP, None)
            for obj in scene.objects if obj.type == "CAMERA"
        }
        return observation(summary, revision=0, blender_version="", addon_version="").get("state_hash")
    except Exception:
        return None


def _scene_coordinate_system(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Declare an immutable coordinate convention once geometry exists."""
    _require_bpy()
    current = _coordinate_system()
    updated = dict(current)
    if current.get("declared") and len(bpy.context.scene.objects):
        for key in ("units", "up_axis", "front_axis", "handedness", "origin"):
            if key in args and str(args[key]) != str(current.get(key)):
                raise ExecutorError("coordinate contract cannot change after geometry exists", "precondition_failed")
    for key in ("units", "up_axis", "front_axis", "handedness", "origin"):
        if key in args:
            updated[key] = str(args[key])
    if updated.get("handedness") != "right":
        raise ExecutorError("only right-handed coordinates are supported", "invalid_args")
    up_axis = str(updated.get("up_axis", "POS_Z"))
    front_axis = str(updated.get("front_axis", "NEG_Y"))
    if up_axis[-1:] == front_axis[-1:] or up_axis[-1:] not in "XYZ" or front_axis[-1:] not in "XYZ":
        raise ExecutorError("up_axis and front_axis must be perpendicular signed XYZ axes", "invalid_args")
    if updated.get("origin") == "custom":
        if "custom_origin" in args:
            updated["custom_origin"] = [float(v) for v in _length_vector(args["custom_origin"], "custom_origin", {"units": updated.get("units", "meters")})]
        elif "custom_origin" not in updated:
            raise ExecutorError("custom origin requires custom_origin", "invalid_args")
    else:
        updated.pop("custom_origin", None)
    required = ("units", "up_axis", "front_axis", "handedness", "origin")
    updated["declared"] = all(key in args for key in required) or bool(current.get("declared"))
    if not updated["declared"]:
        updated["missing_fields"] = [key for key in required if key not in args]
    else:
        updated.pop("missing_fields", None)
    _store_json_prop(bpy.context.scene, _COORDINATE_PROP, updated)
    return {"coordinate_system": updated}


def _set_world_pose_from_contact(obj: Any, contact_local: Any, contact_world: Any, normal_world: Optional[Any], align_axis: str) -> None:
    """Place an object so a declared local contact point reaches a world point."""
    scale = obj.scale.copy()
    rotation = obj.matrix_world.to_quaternion()
    if normal_world is not None and normal_world.length > 1e-9:
        rotation = _axis_vector(align_axis).rotation_difference(normal_world.normalized())
    scaled_contact = Vector((contact_local.x * scale.x, contact_local.y * scale.y, contact_local.z * scale.z))
    origin = contact_world - (rotation @ scaled_contact)
    obj.matrix_world = Matrix.Translation(origin) @ rotation.to_matrix().to_4x4() @ Matrix.Diagonal((float(scale.x), float(scale.y), float(scale.z), 1.0))


def _assembly_anchor_create(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    parent = _object_by_ref(args["parent"])
    frame = _coordinate_frame(args)
    # Anchors are persisted in parent-local meters.  Convert declared input
    # coordinates once so later attach/verify operations share one contract.
    position_local = _point_to_object_local(parent, args["position"], frame, "position")
    normal_input = _coordinate_basis(frame) @ Vector(_as_float3(args.get("normal", (0, 0, 1)), "normal"))
    normal_world = normal_input
    if str(frame.get("space", "WORLD")).upper() == "PARENT":
        normal_world = parent.matrix_world.to_quaternion() @ normal_input
    elif str(frame.get("space", "WORLD")).upper() == "LOCAL":
        normal_world = parent.matrix_world.to_quaternion() @ normal_input
    normal = parent.matrix_world.to_3x3().inverted().transposed() @ normal_world
    if normal.length < 1e-9:
        raise ExecutorError("anchor normal must be non-zero", "invalid_args")
    anchors = _load_json_prop(parent, _ANCHORS_PROP, {})
    if not isinstance(anchors, Mapping):
        anchors = {}
    updated = dict(anchors)
    updated[str(args["name"])] = {
        "position": [float(v) for v in position_local],
        "normal": [round(float(v), 8) for v in normal.normalized()],
        "semantic_tags": [str(v) for v in (args.get("semantic_tags") or [])],
        "coordinate_frame": {"space": "LOCAL", "units": "meters", "up_axis": frame.get("up_axis"), "front_axis": frame.get("front_axis")},
    }
    _store_json_prop(parent, _ANCHORS_PROP, updated)
    return {"parent": _stable_uuid(parent), "name": str(args["name"]), "anchor": updated[str(args["name"])]}


def _object_parent_set(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    child = _object_by_ref(args["child"])
    if bool(args.get("clear_parent", False)):
        world = child.matrix_world.copy()
        child.parent = None
        child.matrix_world = world
        child.pop(_ATTACHMENT_PROP, None)
        return {"child": _stable_uuid(child), "parent": None, "cleared": True}
    parent = _object_by_ref(args.get("parent"))
    if child == parent:
        raise ExecutorError("an object cannot parent itself", "invalid_args")
    cursor = parent
    while cursor is not None:
        if cursor == child:
            raise ExecutorError("parenting would create a cycle", "invalid_args")
        cursor = cursor.parent
    world = child.matrix_world.copy()
    child.parent = parent
    child.matrix_world = world
    child.pop(_ATTACHMENT_PROP, None)
    return {"child": _stable_uuid(child), "parent": _stable_uuid(parent), "preserved_world_transform": True}


def _assembly_attach(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    child = _object_by_ref(args["child"])
    parent = _object_by_ref(args["parent"])
    if child == parent:
        raise ExecutorError("an object cannot attach to itself", "invalid_args")
    cursor = parent
    while cursor is not None:
        if cursor == child:
            raise ExecutorError("attachment would create a parent cycle", "invalid_args")
        cursor = cursor.parent
    anchors = _load_json_prop(parent, _ANCHORS_PROP, {})
    anchor = anchors.get(str(args["anchor"])) if isinstance(anchors, Mapping) else None
    if not isinstance(anchor, Mapping):
        raise ExecutorError(f"anchor not found on parent: {args['anchor']}", "not_found")
    position_local = Vector(_as_float3(anchor.get("position"), "anchor.position"))
    normal_local = Vector(_as_float3(anchor.get("normal", (0, 0, 1)), "anchor.normal"))
    normal_world = (parent.matrix_world.to_3x3() @ normal_local).normalized()
    frame = _coordinate_frame(args)
    clearance = _length_value(args.get("clearance", 0.0), "clearance", frame)
    contact_world = parent.matrix_world @ position_local + normal_world * clearance
    contact_local = _point_to_object_local(child, args.get("contact_point", (0, 0, 0)), frame, "contact_point")
    child.parent = parent
    _set_world_pose_from_contact(child, contact_local, contact_world, normal_world if bool(args.get("align_to_normal", True)) else None, str(args.get("align_axis", "Z")))
    _store_json_prop(child, _ATTACHMENT_PROP, {
        "relation": "attached", "parent": _stable_uuid(parent), "parent_name": parent.name,
        "anchor": str(args["anchor"]), "contact_point": list(contact_local), "clearance": clearance,
        "align_axis": str(args.get("align_axis", "Z")), "align_to_normal": bool(args.get("align_to_normal", True)), "coordinate_frame": frame,
    })
    bpy.context.view_layer.update()
    return {"child": _stable_uuid(child), "parent": _stable_uuid(parent), "anchor": str(args["anchor"]), "contact_world": [round(float(v), 8) for v in contact_world]}


def _surface_bvh(surface: Any) -> Any:
    if BVHTree is None or surface.type != "MESH" or surface.data is None:
        return None
    vertices = [surface.matrix_world @ vertex.co for vertex in surface.data.vertices]
    polygons = [tuple(int(index) for index in polygon.vertices) for polygon in surface.data.polygons if len(polygon.vertices) >= 3]
    return BVHTree.FromPolygons(vertices, polygons, all_triangles=False) if polygons else None


def _surface_snap(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    target = _object_by_ref(args["target"])
    surface = _object_by_ref(args["surface"])
    if target == surface or surface.type != "MESH" or surface.data is None:
        raise ExecutorError("surface_snap requires a distinct mesh surface", "invalid_args")
    tree = _surface_bvh(surface)
    if tree is None:
        raise ExecutorError("surface has no triangles", "invalid_args")
    frame = _coordinate_frame(args)
    contact_value = args.get("contact_point", (0, 0, 0))
    space = str(frame.get("space", "WORLD")).upper()
    if space == "LOCAL":
        contact_local = _coordinate_basis(frame) @ _length_vector(contact_value, "contact_point", frame)
    else:
        contact_world = _point_to_world_for_object(target, contact_value, frame, "contact_point")
        contact_local = target.matrix_world.inverted() @ contact_world
    origin = target.matrix_world @ contact_local
    direction = _direction_to_world(target, args.get("direction", (0, 0, -1)), frame, "direction")
    if direction.length < 1e-9:
        raise ExecutorError("direction must be non-zero", "invalid_args")
    direction.normalize()
    max_distance = _length_value(args.get("max_distance", 100.0), "max_distance", frame)
    hit = tree.ray_cast(origin, direction, max_distance)
    if hit[0] is None and bool(args.get("search_both_directions", True)):
        hit = tree.ray_cast(origin, -direction, max_distance)
    if hit[0] is None:
        hit = tree.find_nearest(origin, max_distance)
    if hit[0] is None:
        raise ExecutorError("surface snap found no surface within max_distance", "precondition_failed")
    location, normal, _index, distance = hit
    normal = normal.normalized() if normal is not None and normal.length > 1e-9 else -direction
    offset = _length_value(args.get("offset", 0.0), "offset", frame)
    contact_world = Vector(location) + normal * offset
    _set_world_pose_from_contact(target, contact_local, contact_world, normal if bool(args.get("align_to_normal", True)) else None, str(args.get("align_axis", "Z")))
    _store_json_prop(target, _SNAP_PROP, {
        "relation": "surface_contact", "surface": _stable_uuid(surface), "surface_name": surface.name,
        "contact_point": list(contact_local), "direction": list(direction), "offset": offset,
        "distance": float(distance), "normal": [round(float(v), 8) for v in normal],
        "align_axis": str(args.get("align_axis", "Z")), "align_to_normal": bool(args.get("align_to_normal", True)),
        "coordinate_frame": frame,
    })
    bpy.context.view_layer.update()
    return {"target": _stable_uuid(target), "surface": _stable_uuid(surface), "location": [round(float(v), 8) for v in contact_world], "normal": [round(float(v), 8) for v in normal], "distance": round(float(distance), 8)}


def _model_plan(args: Mapping[str, Any], *, default_task_spec: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Return a compact, task-agnostic representation decision before edits."""
    task_spec: Dict[str, Any] = _merge_mappings(dict(default_task_spec or {}), args.get("task_spec") if isinstance(args.get("task_spec"), Mapping) else {})
    quality = task_spec.get("quality") or task_spec.get("quality_contract") or {}
    if not isinstance(quality, Mapping):
        quality = {}
    raw_rep = args.get("representation") or quality.get("representation") or task_spec.get("representation") or {}
    representation = copy.deepcopy(dict(raw_rep)) if isinstance(raw_rep, Mapping) else {"kind": str(raw_rep)}
    intent = str(args.get("intent") or task_spec.get("intent") or "").strip()
    continuous = bool(args.get("continuous_envelope", task_spec.get("continuous_envelope", False)))
    repeated = bool(args.get("repeated", task_spec.get("repeated", False)))
    openings = bool(args.get("openings", task_spec.get("openings", False)))
    local_relief = bool(args.get("local_relief", task_spec.get("local_relief", False)))
    strict_topology = bool(args.get("strict_topology", task_spec.get("strict_topology", False)))
    if not continuous and (local_relief or openings or repeated) and any(token in intent.lower() for token in ("organic", "envelope", "relief", "opening")):
        continuous = True

    if not representation.get("kind"):
        if continuous:
            representation["kind"] = "mixed"
            representation["carrier"] = "section_stack_or_control_mesh"
        elif repeated:
            representation["kind"] = "derived"
            representation["carrier"] = "array_or_geometry_nodes"
        elif openings or strict_topology:
            representation["kind"] = "specified"
            representation["carrier"] = "control_mesh_or_boolean_with_rim"
        elif local_relief:
            representation["kind"] = "mixed"
            representation["carrier"] = "surface_patch_or_deformation"
        else:
            representation["kind"] = "specified"
            representation["carrier"] = "control_mesh_or_native_generator"
    primary_refs = quality.get("primary_refs") or task_spec.get("primary_refs") or []
    if isinstance(primary_refs, str):
        primary_refs = [primary_refs]
    if not representation.get("primary_refs"):
        representation["primary_refs"] = list(primary_refs)

    missing_contract = []
    if not intent:
        missing_contract.append("intent")
    if not task_spec.get("identity"):
        missing_contract.append("quality.identity")
    if not task_spec.get("scale"):
        missing_contract.append("quality.scale")
    if not representation.get("primary_refs"):
        missing_contract.append("quality.primary_refs")
    if not (quality.get("reference_views") or task_spec.get("silhouette_views")):
        missing_contract.append("quality.reference_views")
    if not (quality.get("secondary_refs") or task_spec.get("secondary_refs")):
        missing_contract.append("quality.secondary_refs")
    if not (quality.get("feature_scales") or task_spec.get("feature_sizes")):
        missing_contract.append("quality.feature_scales")
    if not (quality.get("detail_regions") or task_spec.get("detail_regions")):
        missing_contract.append("quality.detail_regions")
    quality_contract = _quality_contract("quality_first", task_spec)
    recommended_missing = [item + " (required for detail sampling)" if item == "quality.feature_scales" else item for item in missing_contract]
    return {
        "plan_version": "model_plan.v1",
        "quality_bar_version": "quality_bar.v1",
        "intent": intent or None,
        "representation": representation,
        "signals": {"continuous_envelope": continuous, "repeated": repeated, "openings": openings, "local_relief": local_relief, "strict_topology": strict_topology},
        "stages": ["structure", "primary", "secondary", "tertiary", "technical", "evidence"],
        "stage_gates": copy.deepcopy(quality_bar().get("stage_gates", {})),
        "missing_contract": missing_contract,
        "recommended_missing": recommended_missing,
        "quality_contract_template": quality_contract,
        "declaration_status": {
            "identity": bool(task_spec.get("identity")),
            "scale": bool(task_spec.get("scale")),
            "primary_refs": bool(representation.get("primary_refs")),
        },
        "next_actions": ["declare_quality_contract", "author_primary_carrier", "resolve_primary_silhouette", "add_secondary_junctions", "add_masked_detail", "run_verify_and_render"],
    }


def _activate_object(obj: Any) -> None:
    for selected in list(bpy.context.selected_objects):
        selected.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _object_transform_apply(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _object_by_ref(args["target"])
    location = bool(args.get("location", True))
    rotation = bool(args.get("rotation", True))
    scale = bool(args.get("scale", True))
    _activate_object(obj)
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    try:
        result = bpy.ops.object.transform_apply(location=location, rotation=rotation, scale=scale)
    except Exception as exc:
        raise ExecutorError(f"transform apply failed: {exc}", "execution_error") from exc
    if "FINISHED" not in set(result):
        raise ExecutorError("transform apply did not finish", "execution_error")
    obj.select_set(False)
    return {"uuid": _stable_uuid(obj), "name": obj.name, "applied": {"location": location, "rotation": rotation, "scale": scale}, "matrix_hash": content_hash([[round(float(v), 8) for v in row] for row in obj.matrix_world])}


def _object_convert(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _object_by_ref(args["target"])
    target_type = str(args["target_type"]).upper()
    if target_type not in {"MESH", "CURVE"}:
        raise ExecutorError("target_type must be MESH or CURVE", "invalid_args")
    if obj.type == target_type:
        return {"uuid": _stable_uuid(obj), "name": obj.name, "type": obj.type, "changed": False}
    _activate_object(obj)
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    try:
        result = bpy.ops.object.convert(target=target_type)
    except Exception as exc:
        raise ExecutorError(f"object conversion failed: {exc}", "execution_error") from exc
    if "FINISHED" not in set(result) or obj.type != target_type:
        raise ExecutorError(f"object conversion did not produce {target_type}", "execution_error")
    obj.select_set(False)
    return {"uuid": _stable_uuid(obj), "name": obj.name, "type": obj.type, "changed": True}


def _collection_group_objects(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    refs = args.get("targets") or []
    if not isinstance(refs, list) or not refs:
        raise ExecutorError("targets must contain at least one object reference", "invalid_args")
    name = str(args.get("name") or "Generated Collection")
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    if not collection.get(_COLLECTION_UUID_PROP):
        collection[_COLLECTION_UUID_PROP] = "col-" + content_hash({"name": name})[7:23]
    resolved: list[Any] = []
    seen: set[str] = set()

    def add_with_children(item: Any) -> None:
        uid = _stable_uuid(item)
        if uid in seen:
            return
        seen.add(uid)
        resolved.append(item)
        if bool(args.get("include_children", True)):
            for child in item.children:
                add_with_children(child)

    for ref in refs:
        add_with_children(_object_by_ref(ref))
    for obj in resolved:
        if bool(args.get("exclusive", True)):
            _unlink_from_all(obj)
        if collection.objects.get(obj.name) is None:
            collection.objects.link(obj)
    return {"uuid": _stable_collection_uuid(collection), "name": collection.name, "objects": [_stable_uuid(obj) for obj in resolved], "count": len(resolved), "exclusive": bool(args.get("exclusive", True))}


def _scene_camera_create(args: Mapping[str, Any], stable_id: Optional[str] = None) -> Dict[str, Any]:
    _require_bpy()
    name = str(args["name"])
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != "CAMERA":
        camera = bpy.data.cameras.new(name)
        obj = bpy.data.objects.new(name, camera)
        bpy.context.scene.collection.objects.link(obj)
    obj[_UUID_PROP] = stable_id or _stable_uuid(obj)
    frame = _creation_coordinate_frame(args)
    obj.location = _point_to_world(args.get("location", (4, -4, 3)), "location", frame)
    camera_type = str(args.get("camera_type", "PERSP")).upper()
    if camera_type in {"PERSP", "ORTHO", "PANO"}:
        obj.data.type = camera_type
    obj.data.lens = float(args.get("lens", 50.0))
    if "orthographic_scale" in args:
        obj.data.ortho_scale = float(args["orthographic_scale"])
    obj.data.clip_start = float(args.get("clip_start", 0.01))
    obj.data.clip_end = float(args.get("clip_end", 1000.0))
    if "shift_x" in args:
        obj.data.shift_x = float(args["shift_x"])
    if "shift_y" in args:
        obj.data.shift_y = float(args["shift_y"])
    if "dof_fstop" in args:
        obj.data.dof.use_dof = True
        obj.data.dof.aperture_fstop = float(args["dof_fstop"])
    if "dof_target" in args:
        dof_target = _object_by_ref(args["dof_target"])
        obj.data.dof.use_dof = True
        obj.data.dof.focus_object = dof_target
    target = _point_to_world(args.get("target", (0, 0, 0)), "target", frame)
    _look_at(obj, target)
    _store_json_prop(obj, _COORDINATE_PROP, frame)
    _store_json_prop(obj, _CAMERA_TARGET_PROP, [float(value) for value in target])
    return {"uuid": _stable_uuid(obj), "name": obj.name, "coordinate_frame": frame, "target": [float(value) for value in target]}


def _scene_light_create(args: Mapping[str, Any], stable_id: Optional[str] = None) -> Dict[str, Any]:
    _require_bpy()
    name = str(args["name"])
    light_type = str(args["light_type"]).upper()
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != "LIGHT" or obj.data.type != light_type:
        light = bpy.data.lights.new(name, light_type)
        obj = bpy.data.objects.new(name, light)
        bpy.context.scene.collection.objects.link(obj)
    obj[_UUID_PROP] = stable_id or _stable_uuid(obj)
    frame = _creation_coordinate_frame(args)
    obj.location = _point_to_world(args.get("location", (4, -4, 4)), "location", frame)
    obj.data.energy = float(args.get("energy", 1000.0))
    color = args.get("color")
    if isinstance(color, (list, tuple)) and len(color) in {3, 4}:
        obj.data.color = tuple(float(v) for v in (color[:3]))
    if hasattr(obj.data, "size"):
        obj.data.size = float(args.get("size", 1.0))
    if hasattr(obj.data, "size_y") and "size_y" in args:
        obj.data.size_y = float(args["size_y"])
    if hasattr(obj.data, "spot_size"):
        obj.data.spot_size = float(args.get("spot_size", 0.785398))
    if hasattr(obj.data, "spot_blend") and "spot_blend" in args:
        obj.data.spot_blend = float(args["spot_blend"])
    if hasattr(obj.data, "shadow_soft_size") and "shadow_soft_size" in args:
        obj.data.shadow_soft_size = float(args["shadow_soft_size"])
    target = _point_to_world(args.get("target", (0, 0, 0)), "target", frame)
    _look_at(obj, target)
    _store_json_prop(obj, _COORDINATE_PROP, frame)
    _store_json_prop(obj, _CAMERA_TARGET_PROP, [float(value) for value in target])
    return {"uuid": _stable_uuid(obj), "name": obj.name, "coordinate_frame": frame, "target": [float(value) for value in target]}


def _resolve_render_engine(scene: Any, requested: str) -> str:
    available: set[str] = set()
    try:
        available = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
    except (AttributeError, KeyError, TypeError):
        pass
    if requested == "BLENDER_EEVEE_NEXT" and available and requested not in available and "BLENDER_EEVEE" in available:
        return "BLENDER_EEVEE"
    if requested == "BLENDER_EEVEE" and available and requested not in available and "BLENDER_EEVEE_NEXT" in available:
        return "BLENDER_EEVEE_NEXT"
    return requested


def _scene_set_render_settings(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    scene = bpy.context.scene
    for key in ("resolution_x", "resolution_y", "resolution_percentage", "frame_start", "frame_end"):
        if key in args:
            setattr(scene.render if key.startswith("resolution") else scene, key, int(args[key]))
    if "engine" in args:
        # Blender 5.x renamed the Eevee enum while some builds still expose
        # the legacy value. Resolve the requested public name against the
        # engines available in this build instead of failing a valid action.
        scene.render.engine = _resolve_render_engine(scene, str(args["engine"]))
    if "fps" in args:
        scene.render.fps = max(1, min(240, int(round(float(args["fps"])))))
    return {"engine": scene.render.engine, "resolution": [scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage], "frame_range": [scene.frame_start, scene.frame_end], "fps": scene.render.fps}


def _hair_create_strands(args: Mapping[str, Any], stable_id: Optional[str] = None) -> Dict[str, Any]:
    _require_bpy()
    strands = args.get("strands")
    if not isinstance(strands, list) or not strands:
        raise ExecutorError("strands must contain at least one polyline", "invalid_args")
    frame = _creation_coordinate_frame(args)
    radii = args.get("radii") or []
    if radii and len(radii) != len(strands):
        raise ExecutorError("radii must contain one array per strand", "invalid_args")
    parsed_strands: list[list[tuple[float, float, float]]] = []
    for strand_index, raw in enumerate(strands):
        points = [tuple(_coordinate_basis(frame) @ _length_vector(point, "strands[]", frame)) for point in raw]
        if len(points) < 2:
            raise ExecutorError("each strand needs at least two points", "invalid_args")
        strand_radii = radii[strand_index] if radii else []
        if strand_radii and len(strand_radii) != len(points):
            raise ExecutorError("each radii array must match its strand point count", "invalid_args")
        parsed_strands.append(points)
    name = str(args.get("name", "Hair"))
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = int(args.get("resolution", 4))
    curve.bevel_depth = float(_length_value(args.get("bevel_depth", 0.005), "bevel_depth", frame))
    curve.bevel_resolution = int(args.get("bevel_resolution", 2))
    for strand_index, points in enumerate(parsed_strands):
        strand_radii = radii[strand_index] if radii else []
        spline = curve.splines.new("POLY")
        spline.points.add(len(points) - 1)
        spline.use_cyclic_u = bool(args.get("cyclic", False))
        for point_index, (point, coordinate) in enumerate(zip(spline.points, points)):
            point.co = (*coordinate, 1.0)
            if strand_radii:
                point.radius = float(strand_radii[point_index])
    obj = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(obj)
    obj[_UUID_PROP] = stable_id or _stable_uuid(obj)
    obj[_SEMANTIC_PROP] = list(args.get("semantic_tags", []) or [])
    _authoring_metadata(obj, args, origin="hair.create_strands", default_representation="curve")
    _store_json_prop(obj, _COORDINATE_PROP, frame)
    return {"uuid": _stable_uuid(obj), "name": obj.name, "strands": len(strands), "coordinate_frame": frame}


def _hair_convert_to_mesh(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _object_by_ref(args["target"])
    if obj.type != "CURVE":
        raise ExecutorError("hair.convert_to_mesh requires a curve object", "invalid_args")
    uid = _stable_uuid(obj)
    _activate_object(obj)
    bpy.ops.object.convert(target="MESH")
    return {"uuid": uid, "name": obj.name, "type": obj.type}


def _curve_subdivide(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _object_by_ref(args["target"])
    if obj.type != "CURVE" or obj.data is None:
        raise ExecutorError("curve.subdivide requires a curve object", "invalid_args")
    cuts = max(1, min(32, int(args.get("cuts", 1))))
    before = sum(len(spline.bezier_points if spline.type == "BEZIER" else spline.points) for spline in obj.data.splines)
    _activate_object(obj)
    try:
        if bpy.context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.curve.select_all(action="SELECT")
        result = bpy.ops.curve.subdivide(number_cuts=cuts)
        if "FINISHED" not in set(result):
            raise ExecutorError("curve subdivide did not finish", "execution_error")
    except ExecutorError:
        raise
    except Exception as exc:
        raise ExecutorError(f"curve subdivide failed: {exc}", "execution_error") from exc
    finally:
        if bpy.context.mode == "EDIT_CURVE":
            bpy.ops.object.mode_set(mode="OBJECT")
        obj.select_set(False)
    after = sum(len(spline.bezier_points if spline.type == "BEZIER" else spline.points) for spline in obj.data.splines)
    return {"uuid": _stable_uuid(obj), "name": obj.name, "cuts": cuts, "splines": len(obj.data.splines), "points_before": before, "points_after": after}


def _with_edit_mesh(obj: Any, selection: Optional[Mapping[str, Any]] = None) -> None:
    _activate_object(obj)
    if bpy.context.mode != "EDIT_MESH":
        bpy.ops.object.mode_set(mode="EDIT")
    if not selection:
        bpy.ops.mesh.select_all(action="SELECT")
        return
    bpy.ops.mesh.select_all(action="DESELECT")
    bm = bmesh.from_edit_mesh(obj.data)
    verts, edges, faces = _selection_parts(bm, selection, obj=obj, default_all=False)
    if not verts and not edges and not faces:
        raise ExecutorError("UV selection is empty", "invalid_args")
    for vertex in verts:
        vertex.select = True
    for edge in edges:
        edge.select = True
    for face in faces:
        face.select = True
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)


def _leave_edit_mesh() -> None:
    if bpy.context.mode == "EDIT_MESH":
        bpy.ops.object.mode_set(mode="OBJECT")


def _uv_unwrap(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name="UVMap")
    method = str(args["method"]).upper()
    margin = float(args.get("margin", 0.001))
    try:
        _with_edit_mesh(obj, args.get("selection"))
        if method in {"ANGLE_BASED", "CONFORMAL"}:
            bpy.ops.uv.unwrap(method=method, margin=margin)
        elif method == "SMART_PROJECT":
            bpy.ops.uv.smart_project(island_margin=margin)
        elif method == "CUBE_PROJECT":
            bpy.ops.uv.cube_project(cube_size=1.0)
        elif method == "CYLINDER_PROJECT":
            bpy.ops.uv.cylinder_project()
        elif method == "SPHERE_PROJECT":
            bpy.ops.uv.sphere_project()
        else:
            raise ExecutorError(f"unsupported UV unwrap method: {method}", "invalid_args")
    finally:
        _leave_edit_mesh()
    return {"target": _stable_uuid(obj), "method": method, "uv_layers": len(obj.data.uv_layers), "loops": len(obj.data.loops)}


def _uv_pack(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    if not obj.data.uv_layers:
        raise ExecutorError("UV pack requires an existing UV layer", "precondition_failed")
    try:
        _with_edit_mesh(obj)
        bpy.ops.uv.pack_islands(margin=float(args.get("margin", 0.001)), rotate=bool(args.get("rotate", True)))
    finally:
        _leave_edit_mesh()
    return {"target": _stable_uuid(obj), "margin": float(args.get("margin", 0.001)), "uv_layers": len(obj.data.uv_layers)}


def _uv_project(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name="UVMap")
    method = str(args["method"]).upper()
    if method == "VIEW":
        # The UI operator has no valid poll context in headless Blender. Use a
        # deterministic orthographic projection in object-local coordinates.
        axis = str(args.get("axis", "NEG_Y"))
        component_pairs = {
            "POS_X": (1, 2), "NEG_X": (1, 2), "POS_Y": (0, 2), "NEG_Y": (0, 2),
            "POS_Z": (0, 1), "NEG_Z": (0, 1),
        }
        first, second = component_pairs.get(axis, (0, 2))
        selected_faces: Optional[set[int]] = None
        selection = args.get("selection") or {}
        if selection:
            bm = bmesh.new()
            try:
                bm.from_mesh(obj.data)
                _, _, faces = _selection_parts(bm, selection, obj=obj, default_all=False)
                selected_faces = {int(face.index) for face in faces}
            finally:
                bm.free()
        layer = obj.data.uv_layers.active
        projected = []
        for polygon in obj.data.polygons:
            if selected_faces is not None and int(polygon.index) not in selected_faces:
                continue
            for loop_index in polygon.loop_indices:
                coordinate = obj.data.vertices[obj.data.loops[loop_index].vertex_index].co
                projected.append((loop_index, float(coordinate[first]), float(coordinate[second])))
        if not projected:
            raise ExecutorError("UV projection selection is empty", "invalid_args")
        min_u = min(item[1] for item in projected)
        max_u = max(item[1] for item in projected)
        min_v = min(item[2] for item in projected)
        max_v = max(item[2] for item in projected)
        span_u = max(max_u - min_u, 1e-12)
        span_v = max(max_v - min_v, 1e-12)
        for loop_index, u, v in projected:
            if args.get("scale_to_bounds", True):
                u, v = (u - min_u) / span_u, (v - min_v) / span_v
            layer.data[loop_index].uv = (u, v)
        obj.data.update()
    else:
        try:
            _with_edit_mesh(obj, args.get("selection"))
            if method == "CUBE":
                bpy.ops.uv.cube_project(cube_size=1.0)
            elif method == "CYLINDER":
                bpy.ops.uv.cylinder_project()
            elif method == "SPHERE":
                bpy.ops.uv.sphere_project()
            else:
                raise ExecutorError(f"unsupported UV projection method: {method}", "invalid_args")
        finally:
            _leave_edit_mesh()
    return {"target": _stable_uuid(obj), "method": method, "uv_layers": len(obj.data.uv_layers)}


def _uv_set_seams(args: Mapping[str, Any], *, seam: bool) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        _, edges, _ = _selection_parts(bm, args.get("selection") or {}, obj=obj, default_all=False)
        if not edges:
            raise ExecutorError("seam selection contains no edges", "invalid_args")
        for edge in edges:
            edge.seam = seam
        _write_bmesh(obj, bm)
        return {"target": _stable_uuid(obj), "seam": seam, "edges": len(edges)}
    finally:
        bm.free()


def _inspect_uv(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    layers = []
    for layer in obj.data.uv_layers:
        coords = [(float(loop.uv.x), float(loop.uv.y)) for loop in layer.data]
        if coords:
            min_uv = [round(min(item[index] for item in coords), 8) for index in range(2)]
            max_uv = [round(max(item[index] for item in coords), 8) for index in range(2)]
        else:
            min_uv = max_uv = [0.0, 0.0]
        layers.append({"name": layer.name, "active": layer == obj.data.uv_layers.active, "loops": len(coords), "min": min_uv, "max": max_uv, "coverage": round((max_uv[0] - min_uv[0]) * (max_uv[1] - min_uv[1]), 8)})
    return {"target": _stable_uuid(obj), "layers": layers}


_MATERIAL_NODE_TYPES = {
    "ShaderNodeOutputMaterial", "ShaderNodeBsdfPrincipled", "ShaderNodeTexNoise",
    "ShaderNodeTexCoord", "ShaderNodeMapping", "ShaderNodeRGB", "ShaderNodeValToRGB",
    "ShaderNodeBump", "ShaderNodeMixRGB", "ShaderNodeNormalMap", "ShaderNodeSeparateXYZ",
    "ShaderNodeCombineXYZ", "ShaderNodeMath", "ShaderNodeVectorMath",
    "ShaderNodeTexImage", "ShaderNodeTexVoronoi", "ShaderNodeTexWave", "ShaderNodeTexBrick",
    "ShaderNodeTexGradient", "ShaderNodeTexMusgrave", "ShaderNodeRGBToBW",
    "ShaderNodeHueSaturation", "ShaderNodeGamma", "ShaderNodeInvert", "ShaderNodeMapRange",
    "ShaderNodeMixShader", "ShaderNodeAddShader", "ShaderNodeEmission", "ShaderNodeBsdfTransparent",
    "ShaderNodeBsdfGlass", "ShaderNodeBsdfRefraction", "ShaderNodeBsdfHairPrincipled",
    "ShaderNodeVolumePrincipled", "ShaderNodeFresnel", "ShaderNodeLayerWeight",
    "ShaderNodeAmbientOcclusion", "ShaderNodeBevel", "ShaderNodeLightPath", "ShaderNodeObjectInfo",
}


def _node_socket_value(socket: Any, value: Any) -> None:
    if not hasattr(socket, "default_value"):
        raise ExecutorError(f"node socket is not writable: {socket.name}", "invalid_args")
    try:
        if isinstance(value, (list, tuple)):
            # Color/vector sockets have fixed arity. Accept the natural
            # three-component color form and pad alpha when Blender expects 4.
            expected = None
            current = getattr(socket, "default_value", None)
            if isinstance(current, (list, tuple)):
                expected = len(current)
            values = list(value)
            if expected and len(values) < expected:
                values.extend([1.0] if expected == 4 and len(values) == 3 else [0.0] * (expected - len(values)))
            if expected:
                values = values[:expected]
            socket.default_value = tuple(values)
        else:
            socket.default_value = value
    except (TypeError, ValueError) as exc:
        raise ExecutorError(f"invalid value for node socket {socket.name}: {exc}", "invalid_args") from exc


def _copy_node_tree_contents(source: Any, target: Any) -> None:
    """Copy the supported node graph surface without sharing node trees."""
    target.nodes.clear()
    nodes: Dict[str, Any] = {}
    for source_node in source.nodes:
        node = target.nodes.new(source_node.bl_idname)
        node.name = source_node.name
        node.label = source_node.label
        node.location = source_node.location
        if hasattr(source_node, "width") and hasattr(node, "width"):
            node.width = source_node.width
        node.hide = source_node.hide
        node.mute = source_node.mute
        nodes[node.name] = node
        for source_socket in source_node.inputs:
            if source_socket.is_linked or not hasattr(source_socket, "default_value"):
                continue
            target_socket = node.inputs.get(source_socket.name)
            if target_socket is None:
                # Blender may add/remove version-specific sockets (for
                # example on Principled BSDF).  The graph topology remains
                # portable when a default-only socket is unavailable.
                continue
            _node_socket_value(target_socket, source_socket.default_value)
    for link in source.links:
        source_node = nodes.get(link.from_node.name)
        target_node = nodes.get(link.to_node.name)
        if source_node is None or target_node is None:
            raise ExecutorError("node link endpoint disappeared while copying", "execution_error")
        output = source_node.outputs.get(link.from_socket.name)
        input_socket = target_node.inputs.get(link.to_socket.name)
        if output is None or input_socket is None:
            raise ExecutorError("node link socket disappeared while copying", "execution_error")
        target.links.new(output, input_socket)


def _material_node_graph(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    name = str(args["name"])
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    if bool(args.get("clear", True)):
        tree.nodes.clear()
    nodes: Dict[str, Any] = {}
    seen_node_ids: set[str] = set()
    for spec in args.get("nodes", []) or []:
        if not isinstance(spec, Mapping):
            raise ExecutorError("material node entries must be objects", "invalid_args")
        node_id = str(spec.get("id") or spec.get("name") or "Node")
        if node_id in seen_node_ids:
            raise ExecutorError(f"duplicate material node id: {node_id}", "invalid_args")
        seen_node_ids.add(node_id)
        node_type = str(spec.get("type", ""))
        if node_type not in _MATERIAL_NODE_TYPES:
            raise ExecutorError(f"material node type is not allowlisted: {node_type}", "policy_denied")
        node = tree.nodes.new(node_type)
        node.name = node_id
        node.label = str(spec.get("label", node_id))
        if isinstance(spec.get("location"), (list, tuple)) and len(spec["location"]) == 2:
            node.location = (float(spec["location"][0]), float(spec["location"][1]))
        nodes[node_id] = node
        for socket_name, value in (spec.get("inputs") or {}).items():
            socket = node.inputs.get(str(socket_name))
            if socket is None:
                raise ExecutorError(f"material node socket not found: {node_id}.{socket_name}", "invalid_args")
            _node_socket_value(socket, value)
    for link in args.get("links", []) or []:
        if not isinstance(link, Mapping):
            raise ExecutorError("material links must be objects", "invalid_args")
        source = nodes.get(str(link.get("from_node")))
        target = nodes.get(str(link.get("to_node")))
        if source is None or target is None:
            raise ExecutorError("material link references an unknown node", "invalid_args")
        output = source.outputs.get(str(link.get("from_socket")))
        input_socket = target.inputs.get(str(link.get("to_socket")))
        if output is None or input_socket is None:
            raise ExecutorError("material link references an unknown socket", "invalid_args")
        tree.links.new(output, input_socket)
    return {"name": material.name, "nodes": [node.name for node in tree.nodes], "links": len(tree.links)}


def _material_apply_recipe(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize and realize a typed material recipe through the allowlist."""
    _require_bpy()
    try:
        recipe = normalize_recipe(
            args["recipe"],
            expected_kind="material",
            allowed_node_types=_MATERIAL_NODE_TYPES,
        )
    except RecipeError as exc:
        raise ExecutorError(str(exc), exc.code) from exc
    name = str(args["name"])
    if recipe.name is not None and recipe.name != name:
        raise ExecutorError("recipe.name must match the action name", "invalid_args")

    # Realize into a temporary material first.  The live material is only
    # replaced once every node, socket, and link has succeeded.
    stage_name = f"__toolbox_stage_{uuid.uuid4().hex}"
    staging = bpy.data.materials.new(stage_name)
    try:
        staged = _material_node_graph({
            "name": stage_name,
            "nodes": [node.as_dict() for node in recipe.nodes],
            "links": [link.as_dict() for link in recipe.links],
            "clear": True,
        })
        material = bpy.data.materials.get(name)
        created = material is None
        if material is None:
            material = bpy.data.materials.new(name)
        material.use_nodes = True
        target_tree = material.node_tree
        snapshot = target_tree.copy()
        try:
            _copy_node_tree_contents(staging.node_tree, target_tree)
        except Exception:
            try:
                _copy_node_tree_contents(snapshot, target_tree)
            except Exception as restore_exc:
                raise ExecutorError(f"material recipe failed and restore failed: {restore_exc}", "execution_error") from restore_exc
            if created and getattr(material, "users", 0) == 0:
                bpy.data.materials.remove(material)
            raise
        finally:
            if getattr(snapshot, "users", 0) == 0:
                bpy.data.node_groups.remove(snapshot)
    except Exception:
        raise
    finally:
        if bpy.data.materials.get(stage_name) is not None:
            bpy.data.materials.remove(staging)

    graph_hash = content_hash({
        "name": name,
        "nodes": [node.as_dict() for node in recipe.nodes],
        "links": [link.as_dict() for link in recipe.links],
    })
    result = {"name": name, "nodes": list(staged.get("nodes", [])), "links": len(recipe.links), "graph_hash": graph_hash}
    result["recipe_hash"] = recipe.graph_hash
    result["recipe_schema_version"] = recipe.schema_version
    result["recipe_kind"] = recipe.kind
    return result


def _material_set_input(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    material = bpy.data.materials.get(str(args["material"]))
    if material is None or not material.use_nodes:
        raise ExecutorError(f"material not found or has no nodes: {args['material']}", "not_found")
    node = material.node_tree.nodes.get(str(args["node"]))
    if node is None:
        raise ExecutorError(f"material node not found: {args['node']}", "not_found")
    socket = node.inputs.get(str(args["socket"]))
    if socket is None:
        raise ExecutorError(f"material socket not found: {args['socket']}", "not_found")
    _node_socket_value(socket, args["value"])
    return {"material": material.name, "node": node.name, "socket": socket.name}


def _inspect_material(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _object_by_ref(args["target"])
    materials = []
    for slot in obj.material_slots:
        material = slot.material
        if material is None:
            continue
        materials.append({"name": material.name, "nodes": [{"name": node.name, "type": node.bl_idname, "label": node.label} for node in material.node_tree.nodes], "links": len(material.node_tree.links)})
    return {"target": _stable_uuid(obj), "materials": materials}


def _rig_create_armature(args: Mapping[str, Any], stable_id: Optional[str] = None) -> Dict[str, Any]:
    _require_bpy()
    frame = _creation_coordinate_frame(args)
    name = str(args["name"])
    raw_bones = args.get("bones", []) or []
    if not isinstance(raw_bones, list):
        raise ExecutorError("bones must be an array", "invalid_args")
    bone_names = {str(item.get("name", "Bone")) for item in raw_bones if isinstance(item, Mapping)}
    for item in raw_bones:
        if isinstance(item, Mapping) and item.get("parent") and str(item["parent"]) not in bone_names:
            raise ExecutorError(f"parent bone not found: {item['parent']}", "not_found")
    armature = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, armature)
    bpy.context.scene.collection.objects.link(obj)
    obj[_UUID_PROP] = stable_id or _stable_uuid(obj)
    obj[_SEMANTIC_PROP] = list(args.get("semantic_tags", []) or [])
    obj.location = _length_vector(args.get("location", (0, 0, 0)), "location", frame)
    _store_json_prop(obj, _COORDINATE_PROP, frame)
    _activate_object(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    created: Dict[str, Any] = {}
    try:
        for spec in raw_bones:
            bone_name = str(spec.get("name", "Bone"))
            if bone_name in created:
                raise ExecutorError(f"duplicate bone name: {bone_name}", "invalid_args")
            bone = armature.edit_bones.new(bone_name)
            bone.head = _coordinate_basis(frame) @ _length_vector(spec.get("head", (0, 0, 0)), "bones[].head", frame)
            bone.tail = _coordinate_basis(frame) @ _length_vector(spec.get("tail", (0, 0, 1)), "bones[].tail", frame)
            if (bone.tail - bone.head).length < 1e-6:
                raise ExecutorError(f"bone must have non-zero length: {bone_name}", "invalid_args")
            created[bone_name] = bone
        for spec in args.get("bones", []) or []:
            parent_name = spec.get("parent")
            if parent_name:
                bone = created[str(spec["name"])]
                parent = created.get(str(parent_name))
                if parent is None:
                    raise ExecutorError(f"parent bone not found: {parent_name}", "not_found")
                bone.parent = parent
                bone.use_connect = bool(spec.get("use_connect", False))
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    return {"uuid": _stable_uuid(obj), "name": obj.name, "bones": list(created)}


def _bone_segment_distance(point: Vector, head: Vector, tail: Vector) -> float:
    return _distance_to_segment(point, head, tail)[0]


def _rig_bind(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    target = _require_mesh_object(args["target"])
    armature = _object_by_ref(args["armature"])
    if armature.type != "ARMATURE" or armature.data is None:
        raise ExecutorError("armature reference must resolve to an armature object", "invalid_args")
    modifier = next((item for item in target.modifiers if item.type == "ARMATURE" and item.object == armature), None)
    if modifier is None:
        modifier = target.modifiers.new("ToolboxArmature", "ARMATURE")
        modifier.object = armature
    weights = str(args.get("weights", "empty"))
    if weights in {"automatic", "envelopes"}:
        for bone in armature.data.bones:
            group = target.vertex_groups.get(bone.name) or target.vertex_groups.new(name=bone.name)
            head = armature.matrix_world @ bone.head_local
            tail = armature.matrix_world @ bone.tail_local
            candidates = []
            distances = []
            for vertex in target.data.vertices:
                world = target.matrix_world @ vertex.co
                distance = _bone_segment_distance(world, head, tail)
                candidates.append(int(vertex.index))
                distances.append(distance)
            if distances:
                scale = max(max(distances), 1e-6)
                group.add(candidates, 1.0, "REPLACE")
                for index, distance in zip(candidates, distances):
                    weight = max(0.0, 1.0 - distance / scale)
                    group.add([index], weight, "REPLACE")
    world_matrix = target.matrix_world.copy()
    target.parent = armature
    target.matrix_world = world_matrix
    return {"target": _stable_uuid(target), "armature": _stable_uuid(armature), "modifier": modifier.name, "weights": weights, "vertex_groups": len(target.vertex_groups)}


def _rig_pose(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    armature = _object_by_ref(args["armature"])
    if armature.type != "ARMATURE":
        raise ExecutorError("rig.pose requires an armature", "invalid_args")
    frame = int(args.get("frame", bpy.context.scene.frame_current))
    bpy.context.scene.frame_set(frame)
    changed = []
    for spec in args.get("bones", []) or []:
        name = str(spec["name"])
        bone = armature.pose.bones.get(name)
        if bone is None:
            raise ExecutorError(f"pose bone not found: {name}", "not_found")
        if "rotation_euler" in spec:
            bone.rotation_mode = "XYZ"
            bone.rotation_euler = _as_float3(spec["rotation_euler"], "rotation_euler")
        if "location" in spec:
            bone.location = _as_float3(spec["location"], "location")
        if "scale" in spec:
            bone.scale = _as_float3(spec["scale"], "scale")
        if args.get("keyframe", False):
            bone.keyframe_insert(data_path="rotation_euler", frame=frame)
            bone.keyframe_insert(data_path="location", frame=frame)
            bone.keyframe_insert(data_path="scale", frame=frame)
        changed.append(name)
    return {"armature": _stable_uuid(armature), "frame": frame, "bones": changed, "keyframed": bool(args.get("keyframe", False))}


def _rig_add_constraint(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    armature = _object_by_ref(args["armature"])
    if armature.type != "ARMATURE":
        raise ExecutorError("rig.add_constraint requires an armature", "invalid_args")
    bone = armature.pose.bones.get(str(args["bone"]))
    if bone is None:
        raise ExecutorError(f"pose bone not found: {args['bone']}", "not_found")
    constraint_type = str(args["constraint_type"]).upper()
    allowed = {"COPY_LOCATION", "COPY_ROTATION", "COPY_SCALE", "COPY_TRANSFORMS", "IK", "DAMPED_TRACK", "LIMIT_ROTATION"}
    if constraint_type not in allowed:
        raise ExecutorError(f"unsupported constraint type: {constraint_type}", "invalid_args")
    constraint = bone.constraints.new(type=constraint_type)
    if "target" in args:
        constraint.target = _object_by_ref(args["target"])
    if hasattr(constraint, "subtarget") and "subtarget" in args:
        constraint.subtarget = str(args["subtarget"])
    if "influence" in args:
        constraint.influence = float(args["influence"])
    if constraint_type == "IK" and "chain_count" in args:
        constraint.chain_count = int(args["chain_count"])
    if constraint_type == "LIMIT_ROTATION":
        for axis in "xyz":
            use_key = f"use_limit_{axis}"
            min_key = f"min_{axis}"
            max_key = f"max_{axis}"
            if hasattr(constraint, use_key) and use_key in args:
                setattr(constraint, use_key, bool(args[use_key]))
            if hasattr(constraint, min_key) and min_key in args:
                setattr(constraint, min_key, float(args[min_key]))
            if hasattr(constraint, max_key) and max_key in args:
                setattr(constraint, max_key, float(args[max_key]))
    return {"armature": _stable_uuid(armature), "bone": bone.name, "constraint": constraint.name, "type": constraint_type}


def _animation_keyframe_transform(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _object_by_ref(args["target"])
    frame = int(args["frame"])
    bpy.context.scene.frame_set(frame)
    for key in ("location", "rotation_euler", "scale"):
        if key in args:
            setattr(obj, key, _as_float3(args[key], key))
            obj.keyframe_insert(data_path=key, frame=frame)
    interpolation = str(args.get("interpolation", "BEZIER"))
    if obj.animation_data and obj.animation_data.action:
        for curve in _action_fcurves(obj.animation_data.action):
            for point in curve.keyframe_points:
                point.interpolation = interpolation
    return {"target": _stable_uuid(obj), "frame": frame, "action": obj.animation_data.action.name if obj.animation_data and obj.animation_data.action else None}


def _action_fcurves(action: Any) -> list[Any]:
    """Return f-curves for both legacy and layered Blender actions."""
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        return list(legacy)
    curves: list[Any] = []
    for layer in getattr(action, "layers", []) or []:
        for strip in getattr(layer, "strips", []) or []:
            channelbags = getattr(strip, "channelbags", None)
            if channelbags is None:
                channelbag = getattr(strip, "channelbag", None)
                channelbags = [channelbag] if channelbag is not None else []
            for channelbag in channelbags:
                curves.extend(list(getattr(channelbag, "fcurves", []) or []))
    return curves


def _animation_set_range(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    scene = bpy.context.scene
    if "frame_start" in args:
        scene.frame_start = int(args["frame_start"])
    if "frame_end" in args:
        scene.frame_end = int(args["frame_end"])
    if "fps" in args:
        scene.render.fps = max(1, min(240, int(round(float(args["fps"])))))
    return {"frame_start": scene.frame_start, "frame_end": scene.frame_end, "fps": scene.render.fps}


def _inspect_armature(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _object_by_ref(args["target"])
    if obj.type != "ARMATURE":
        raise ExecutorError("inspect.armature requires an armature", "invalid_args")
    bones = [{"name": bone.name, "parent": bone.parent.name if bone.parent else None, "head": [round(float(v), 8) for v in bone.head_local], "tail": [round(float(v), 8) for v in bone.tail_local]} for bone in obj.data.bones]
    action = obj.animation_data.action if obj.animation_data and obj.animation_data.action else None
    constraints = []
    for pose_bone in obj.pose.bones:
        constraints.extend({"bone": pose_bone.name, "name": item.name, "type": item.type, "target": _stable_uuid(getattr(item, "target", None)) if getattr(item, "target", None) else None, "subtarget": getattr(item, "subtarget", None)} for item in pose_bone.constraints)
    return {"target": _stable_uuid(obj), "name": obj.name, "bones": bones, "constraints": constraints, "animation": action.name if action else None, "fcurves": len(_action_fcurves(action)) if action else 0}


_GEOMETRY_NODE_TYPES = {
    "GeometryNodeMeshCube", "GeometryNodeMeshIcoSphere", "GeometryNodeMeshUVSphere",
    "GeometryNodeTransform", "GeometryNodeSubdivisionSurface", "GeometryNodeSetPosition",
    "GeometryNodeJoinGeometry", "GeometryNodeSetMaterial", "GeometryNodeDistributePointsOnFaces",
    "GeometryNodeInstanceOnPoints", "GeometryNodeCurvePrimitiveLine", "GeometryNodeCurveToMesh",
    "GeometryNodeCurvePrimitiveCircle", "GeometryNodeMeshLine", "GeometryNodeRealizeInstances",
    "GeometryNodeSetCurveRadius", "GeometryNodeCurveResample", "GeometryNodeDeleteGeometry",
    "GeometryNodeProximity", "GeometryNodeRaycast", "GeometryNodeSampleIndex", "GeometryNodeInputNormal",
    "GeometryNodeInputPosition", "GeometryNodeInputID", "GeometryNodeSwitch", "GeometryNodeMergeByDistance",
    "GeometryNodeCurveToPoints", "GeometryNodePointsToVertices", "GeometryNodeMeshToCurve",
    "GeometryNodeMeshToPoints", "GeometryNodeCurvePrimitiveQuadrilateral", "GeometryNodeCurvePrimitiveSpiral",
    "GeometryNodeSetCurveTilt", "GeometryNodeResampleCurve", "GeometryNodeCurveSplineType",
    "GeometryNodeScaleElements", "GeometryNodeExtrudeMesh", "GeometryNodeMeshBoolean",
    "GeometryNodeSetShadeSmooth", "GeometryNodeInputNamedAttribute", "GeometryNodeStoreNamedAttribute",
    "GeometryNodeObjectInfo", "GeometryNodeCollectionInfo", "FunctionNodeRandomValue",
    "FunctionNodeAlignEulerToVector", "GeometryNodeRotateInstances", "GeometryNodeScaleInstances",
    "ShaderNodeValue", "ShaderNodeMath", "ShaderNodeVectorMath", "ShaderNodeCombineXYZ", "ShaderNodeSeparateXYZ",
}


def _geometry_nodes_create(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    target = _require_mesh_object(args["target"])
    name = str(args.get("name", f"{target.name}_GeometryNodes"))
    modifier = next((item for item in target.modifiers if item.type == "NODES" and item.node_group and item.node_group.name == name), None)

    # Construct the replacement in an unreferenced node group.  Node/socket
    # errors must not clear or partially rewrite the live graph; the modifier
    # is switched to the staged group only after the complete graph succeeds.
    stage_name = f"__toolbox_stage_{uuid.uuid4().hex}"
    group = bpy.data.node_groups.new(stage_name, "GeometryNodeTree")
    # Build the interface before nodes so Group Input/Output sockets are
    # available for links.  Interface declarations are explicit and typed;
    # malformed socket types fail instead of being silently dropped.
    interface_specs = args.get("interface") or []
    if not interface_specs:
        interface_specs = [
            {"name": "Geometry", "in_out": "INPUT", "socket_type": "NodeSocketGeometry"},
            {"name": "Geometry", "in_out": "OUTPUT", "socket_type": "NodeSocketGeometry"},
        ]
    allowed_socket_types = {
        "NodeSocketGeometry", "NodeSocketFloat", "NodeSocketInt", "NodeSocketBool",
        "NodeSocketVector", "NodeSocketColor", "NodeSocketString", "NodeSocketObject",
        "NodeSocketMaterial", "NodeSocketRotation", "NodeSocketFloatAngle", "NodeSocketFloatDistance",
    }
    try:
        seen_interface: set[tuple[str, str]] = set()
        if hasattr(group, "interface"):
            for item in interface_specs:
                if not isinstance(item, Mapping):
                    raise ExecutorError("geometry interface entries must be objects", "invalid_args")
                in_out = str(item.get("in_out", "INPUT")).upper()
                socket_type = str(item.get("socket_type", ""))
                socket_name = str(item.get("name", ""))
                if in_out not in {"INPUT", "OUTPUT"} or not socket_name or socket_type not in allowed_socket_types:
                    raise ExecutorError("invalid geometry interface socket declaration", "invalid_args")
                key = (in_out, socket_name)
                if key in seen_interface:
                    raise ExecutorError(f"duplicate geometry interface socket: {in_out}:{socket_name}", "invalid_args")
                seen_interface.add(key)
                socket = group.interface.new_socket(name=socket_name, in_out=in_out, socket_type=socket_type)
                if "default" in item and hasattr(socket, "default_value"):
                    _node_socket_value(socket, item["default"])
        input_node = group.nodes.new("NodeGroupInput")
        input_node.name = "GroupInput"
        output_node = group.nodes.new("NodeGroupOutput")
        output_node.name = "GroupOutput"
        nodes: Dict[str, Any] = {"GroupInput": input_node, "GroupOutput": output_node}
        seen_node_ids: set[str] = {"GroupInput", "GroupOutput"}
        for spec in args.get("nodes", []) or []:
            if not isinstance(spec, Mapping):
                raise ExecutorError("geometry node entries must be objects", "invalid_args")
            node_id = str(spec.get("id") or spec.get("name") or "Node")
            if node_id in seen_node_ids:
                raise ExecutorError(f"duplicate geometry node id: {node_id}", "invalid_args")
            seen_node_ids.add(node_id)
            node_type = str(spec.get("type", ""))
            if node_type not in _GEOMETRY_NODE_TYPES:
                raise ExecutorError(f"geometry node type is not allowlisted: {node_type}", "policy_denied")
            node = group.nodes.new(node_type)
            node.name = node_id
            node.label = str(spec.get("label", node_id))
            if isinstance(spec.get("location"), (list, tuple)) and len(spec["location"]) == 2:
                node.location = (float(spec["location"][0]), float(spec["location"][1]))
            for socket_name, value in (spec.get("inputs") or {}).items():
                socket = node.inputs.get(str(socket_name))
                if socket is None:
                    raise ExecutorError(f"geometry node socket not found: {node_id}.{socket_name}", "invalid_args")
                _node_socket_value(socket, value)
            nodes[node_id] = node
        for link in args.get("links", []) or []:
            if not isinstance(link, Mapping):
                raise ExecutorError("geometry links must be objects", "invalid_args")
            source = nodes.get(str(link.get("from_node")))
            target_node = nodes.get(str(link.get("to_node")))
            if source is None or target_node is None:
                raise ExecutorError("geometry link references an unknown node", "invalid_args")
            output = source.outputs.get(str(link.get("from_socket")))
            input_socket = target_node.inputs.get(str(link.get("to_socket")))
            if output is None or input_socket is None:
                raise ExecutorError("geometry link references an unknown socket", "invalid_args")
            group.links.new(output, input_socket)
        output_geometry = output_node.inputs.get("Geometry")
        input_geometry = input_node.outputs.get("Geometry")
        if output_geometry is not None and not output_geometry.is_linked and input_geometry is not None:
            group.links.new(input_geometry, output_geometry)
    except Exception:
        if getattr(group, "users", 0) == 0:
            bpy.data.node_groups.remove(group)
        raise

    # Treat attaching the staged graph, naming it, and producing the summary
    # as one transaction.  Summary generation touches Blender RNA and can
    # fail (for example when a version-specific socket cannot be read); a
    # failure must leave the previous modifier/group exactly as it was.
    old_group = modifier.node_group if modifier is not None else None
    old_group_name = str(getattr(old_group, "name", "")) if old_group is not None else None
    old_group_renamed = False
    created_modifier = modifier is None
    try:
        if modifier is None:
            modifier = target.modifiers.new(name, "NODES")
        # If this modifier was the sole user of the old group, temporarily
        # move its name out of the way.  This lets the replacement retain the
        # requested name without deleting the old graph before validation;
        # rollback can therefore restore both the pointer and the name.
        if old_group is not None and getattr(old_group, "users", 0) <= 1 and old_group_name:
            old_group.name = f"__toolbox_old_{uuid.uuid4().hex}"
            old_group_renamed = True
        modifier.node_group = group
        group.name = name

        summary = _geometry_nodes_summary(target)
        graph = next((item for item in summary if item["name"] == group.name), {})
        result = {
            "target": _stable_uuid(target),
            "modifier": modifier.name,
            "node_group": group.name,
            "nodes": [node.name for node in group.nodes],
            "links": len(group.links),
            "graph_hash": graph.get("graph_hash"),
            "interface": graph.get("interface", []),
        }

        # Commit cleanup only after summary/metadata construction succeeds.
        # The old group is now unreferenced when this modifier was its sole
        # user, so removing it cannot affect another modifier.
        if old_group_renamed and old_group is not None and getattr(old_group, "users", 0) == 0:
            bpy.data.node_groups.remove(old_group)
            old_group = None
        return result
    except Exception:
        # Detach the staged group before attempting to delete it; Blender will
        # otherwise retain the datablock because the modifier counts as a
        # user.  A newly-created modifier is removed entirely rather than
        # leaving an empty NODES modifier behind.
        try:
            if modifier is not None and getattr(modifier, "node_group", None) == group:
                if created_modifier:
                    target.modifiers.remove(modifier)
                    modifier = None
                else:
                    modifier.node_group = old_group
        finally:
            if getattr(group, "users", 0) == 0:
                bpy.data.node_groups.remove(group)
            if old_group_renamed and old_group is not None and old_group_name:
                old_group.name = old_group_name
        raise


def _geometry_nodes_apply_recipe(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize a typed recipe before realizing it as a Geometry Nodes graph."""
    _require_bpy()
    try:
        recipe = normalize_recipe(
            args["recipe"],
            expected_kind="geometry_nodes",
            allowed_node_types=_GEOMETRY_NODE_TYPES,
        )
    except RecipeError as exc:
        raise ExecutorError(str(exc), exc.code) from exc

    create_args: Dict[str, Any] = {
        "target": args["target"],
        "nodes": [node.as_dict() for node in recipe.nodes],
        "links": [link.as_dict() for link in recipe.links],
        "interface": [dict(item) for item in recipe.interface],
    }
    if recipe.name is not None:
        create_args["name"] = recipe.name
    result = _geometry_nodes_create(create_args)
    # Keep the canonical recipe hash distinct from Blender's realized graph
    # summary hash; both are useful when replaying or auditing an action.
    result["recipe_hash"] = recipe.graph_hash
    result["recipe_schema_version"] = recipe.schema_version
    result["recipe_kind"] = recipe.kind
    return result


def _geometry_nodes_set_input(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    target = _require_mesh_object(args["target"])
    node_group = next((item.node_group for item in target.modifiers if item.type == "NODES" and item.node_group), None)
    if node_group is None:
        raise ExecutorError("target has no Geometry Nodes modifier", "precondition_failed")
    node = node_group.nodes.get(str(args["node"]))
    if node is None:
        raise ExecutorError(f"geometry node not found: {args['node']}", "not_found")
    socket = node.inputs.get(str(args["socket"]))
    if socket is None:
        raise ExecutorError(f"geometry node socket not found: {args['socket']}", "not_found")
    _node_socket_value(socket, args["value"])
    return {"target": _stable_uuid(target), "node": node.name, "socket": socket.name}


def _particles_scatter(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Create a deterministic point/instance scatter graph without bpy ops."""
    _require_bpy()
    target = _require_mesh_object(args["target"])
    instance = _object_by_ref(args["instance"])
    name = str(args.get("name") or f"{target.name}_Scatter")
    modifier = next((item for item in target.modifiers if item.type == "NODES" and item.node_group and item.node_group.name == name), None)
    if modifier is None:
        modifier = target.modifiers.new(name, "NODES")
    group = modifier.node_group or bpy.data.node_groups.new(name, "GeometryNodeTree")
    modifier.node_group = group
    group.nodes.clear()
    if hasattr(group, "interface"):
        try:
            group.interface.clear()
        except AttributeError:
            pass
        group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    input_node = group.nodes.new("NodeGroupInput")
    input_node.name = "GroupInput"
    output_node = group.nodes.new("NodeGroupOutput")
    output_node.name = "GroupOutput"
    distribute = group.nodes.new("GeometryNodeDistributePointsOnFaces")
    distribute.name = "Distribute"
    info = group.nodes.new("GeometryNodeObjectInfo")
    info.name = "InstanceInfo"
    instancer = group.nodes.new("GeometryNodeInstanceOnPoints")
    instancer.name = "Instance"
    group.links.new(input_node.outputs["Geometry"], distribute.inputs["Mesh"])
    group.links.new(distribute.outputs["Points"], instancer.inputs["Points"])
    info.inputs["Object"].default_value = instance
    group.links.new(info.outputs["Geometry"], instancer.inputs["Instance"])
    distribute.inputs["Density"].default_value = float(args.get("density", 10.0))
    distribute.inputs["Seed"].default_value = int(args.get("seed", 0))
    if "scale" in args:
        _node_socket_value(instancer.inputs["Scale"], args["scale"])
    output_geometry = instancer.outputs["Instances"]
    realize = None
    if bool(args.get("realize_instances", False)):
        realize = group.nodes.new("GeometryNodeRealizeInstances")
        realize.name = "Realize"
        group.links.new(output_geometry, realize.inputs["Geometry"])
        output_geometry = realize.outputs["Geometry"]
    if bool(args.get("keep_surface", False)):
        join = group.nodes.new("GeometryNodeJoinGeometry")
        join.name = "JoinSurface"
        group.links.new(input_node.outputs["Geometry"], join.inputs["Geometry"])
        group.links.new(output_geometry, join.inputs["Geometry"])
        output_geometry = join.outputs["Geometry"]
    group.links.new(output_geometry, output_node.inputs["Geometry"])
    return {"target": _stable_uuid(target), "instance": _stable_uuid(instance), "modifier": modifier.name, "node_group": group.name, "density": float(args.get("density", 10.0)), "seed": int(args.get("seed", 0)), "realize_instances": bool(args.get("realize_instances", False))}


def _landmark_create(args: Mapping[str, Any], stable_id: Optional[str] = None) -> Dict[str, Any]:
    _require_bpy()
    parent_ref = args.get("parent")
    frame = _creation_coordinate_frame(args, allow_relative=bool(parent_ref))
    name = str(args["name"])
    obj = bpy.data.objects.get(name)
    parent = _object_by_ref(parent_ref) if parent_ref else None
    old_state = None
    if obj is not None:
        old_state = {
            "location": tuple(obj.location),
            "parent": getattr(obj, "parent", None),
            "props": dict(obj),
        }
    try:
        if obj is None:
            obj = bpy.data.objects.new(name, None)
            obj.empty_display_type = "SPHERE"
            obj.empty_display_size = 0.03
            bpy.context.scene.collection.objects.link(obj)
        obj[_UUID_PROP] = stable_id or _stable_uuid(obj)
        obj["blender_toolbox_landmark"] = True
        obj[_SEMANTIC_PROP] = list(args.get("semantic_tags", []) or [])
        if parent is not None:
            if str(frame.get("space", "WORLD")).upper() == "LOCAL":
                raise ExecutorError("landmark parent placement requires WORLD or PARENT space", "invalid_args")
            obj.parent = parent
            obj.location = _point_to_object_local(parent, args["location"], frame, "location")
        else:
            obj.location = _point_to_world(args["location"], "location", frame)
        _store_json_prop(obj, _COORDINATE_PROP, frame)
        bpy.context.view_layer.update()
    except Exception:
        if old_state is not None and obj is not None:
            try:
                obj.location = list(old_state["location"])
            except Exception:
                obj.location = old_state["location"]
            obj.parent = old_state["parent"]
            for key in list(obj.keys()):
                del obj[key]
            for key, value in old_state["props"].items():
                obj[key] = value
        elif obj is not None:
            _discard_created_object(obj)
        raise
    return {"uuid": _stable_uuid(obj), "name": obj.name, "location": [round(float(v), 8) for v in obj.matrix_world.translation], "coordinate_frame": frame}


def _landmark_object(ref: Any) -> Any:
    obj = _object_by_ref(ref)
    if not obj.get("blender_toolbox_landmark") and obj.type != "EMPTY":
        raise ExecutorError(f"object is not a landmark: {ref}", "invalid_args")
    return obj


def _landmark_points(refs: Any, target: Any = None) -> list[Vector]:
    if not isinstance(refs, list) or len(refs) < 2:
        raise ExecutorError("landmarks must contain at least two references", "invalid_args")
    points = []
    inverse = target.matrix_world.inverted() if target is not None else None
    for ref in refs:
        landmark = _landmark_object(ref)
        world = landmark.matrix_world.translation.copy()
        points.append(inverse @ world if inverse is not None else world)
    return points


def _face_curve_from_landmarks(args: Mapping[str, Any], stable_id: Optional[str] = None) -> Dict[str, Any]:
    _require_bpy()
    points = _landmark_points(args["landmarks"])
    surface_ref = args.get("surface")
    surface = _object_by_ref(surface_ref) if surface_ref else None
    curve_args = {
        "name": args["name"], "points": [[float(v) for v in point] for point in points],
        "bezier": bool(args.get("bezier", True)), "bevel_depth": float(args.get("bevel_depth", 0.005)),
        "bevel_resolution": int(args.get("bevel_resolution", 3)), "semantic_tags": args.get("semantic_tags", []),
    }
    result = _create_curve(curve_args, stable_id)
    curve_obj = _object_by_ref(result["uuid"])
    for spline in curve_obj.data.splines:
        spline.use_cyclic_u = bool(args.get("cyclic", False))
    if surface is not None:
        modifier = curve_obj.modifiers.new("ToolboxLandmarkShrinkwrap", "SHRINKWRAP")
        modifier.target = surface
        modifier.offset = float(args.get("offset", 0.0))
    return {**result, "landmarks": list(args["landmarks"]), "surface": surface_ref}


def _landmark_create_set(args: Mapping[str, Any], stable_id: Optional[str] = None) -> Dict[str, Any]:
    created = []
    for item in args.get("landmarks", []) or []:
        item_id = "obj-" + content_hash({"batch": stable_id, "name": item.get("name"), "location": item.get("location")})[7:23] if stable_id else None
        created.append(_landmark_create(item, item_id))
    return {"landmarks": created, "count": len(created)}


def _face_curve_network_from_landmarks(args: Mapping[str, Any], stable_id: Optional[str] = None) -> Dict[str, Any]:
    curves = []
    for item in args.get("curves", []) or []:
        item_id = "obj-" + content_hash({"batch": stable_id, "name": item.get("name"), "landmarks": item.get("landmarks")})[7:23] if stable_id else None
        curves.append(_face_curve_from_landmarks(item, item_id))
    return {"curves": curves, "count": len(curves)}


def _face_sculpt_landmarks(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    target = _require_mesh_object(args["target"])
    points = _landmark_points(args["landmarks"], target=target)
    stroke_args = dict(args)
    stroke_args["points"] = [[float(v) for v in point] for point in points]
    return {**_sculpt_stroke(stroke_args), "landmarks": list(args["landmarks"])}


def _plain_value(value: Any) -> Any:
    """Convert Blender RNA arrays and math types to JSON-safe primitives."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    try:
        return [_plain_value(item) for item in value]
    except (TypeError, AttributeError):
        return str(value)


def _material_graph_summary(material: Any) -> Dict[str, Any]:
    tree = material.node_tree
    nodes = []
    for node in sorted(tree.nodes, key=lambda item: item.name):
        inputs = {}
        for node_socket in node.inputs:
            if hasattr(node_socket, "default_value") and not node_socket.is_linked:
                inputs[node_socket.name] = _plain_value(node_socket.default_value)
        nodes.append({"name": node.name, "type": node.bl_idname, "label": node.label, "inputs": inputs})
    links = sorted([
        {"from_node": link.from_node.name, "from_socket": link.from_socket.name,
         "to_node": link.to_node.name, "to_socket": link.to_socket.name}
        for link in tree.links
    ], key=lambda item: (item["from_node"], item["from_socket"], item["to_node"], item["to_socket"]))
    payload = {"name": material.name, "nodes": nodes, "links": links}
    return {**payload, "graph_hash": content_hash(payload)}


def _animation_summary(obj: Any) -> Optional[Dict[str, Any]]:
    action = obj.animation_data.action if obj.animation_data and obj.animation_data.action else None
    if action is None:
        return None
    curves = _action_fcurves(action)
    keyframes = sum(len(curve.keyframe_points) for curve in curves)
    return {
        "action": action.name,
        "fcurves": len(curves),
        "keyframes": keyframes,
        "frame_range": [round(float(value), 8) for value in action.frame_range],
    }


def _shape_key_summary(obj: Any) -> Optional[Dict[str, Any]]:
    shape_keys = getattr(obj.data, "shape_keys", None) if obj.type == "MESH" and obj.data is not None else None
    if shape_keys is None:
        return None
    blocks = [{"name": key.name, "value": round(float(key.value), 8), "mute": bool(key.mute)} for key in shape_keys.key_blocks]
    geometry = {
        key.name: [[round(float(value), 8) for value in point.co] for point in key.data]
        for key in shape_keys.key_blocks if key.name != "Basis"
    }
    action = shape_keys.animation_data.action if shape_keys.animation_data and shape_keys.animation_data.action else None
    return {
        "keys": blocks,
        "geometry_hash": content_hash(geometry) if geometry else None,
        "animation": action.name if action else None,
        "fcurves": len(_action_fcurves(action)) if action else 0,
    }


def _socket_default_summary(socket: Any) -> Any:
    if not hasattr(socket, "default_value"):
        return None
    value = getattr(socket, "default_value", None)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return [round(float(item), 8) for item in value]
    except (TypeError, ValueError):
        return str(value)


def _geometry_interface_summary(group: Any) -> list[Dict[str, Any]]:
    interface = getattr(group, "interface", None)
    if interface is None:
        return []
    items = getattr(interface, "items_tree", None)
    if items is None:
        return []
    result = []
    for item in items:
        if getattr(item, "item_type", None) != "SOCKET":
            continue
        result.append({
            "name": str(getattr(item, "name", "")),
            "in_out": str(getattr(item, "in_out", "")),
            "socket_type": str(getattr(item, "socket_type", "")),
            "default": _socket_default_summary(item),
        })
    return result


def _geometry_nodes_summary(obj: Any) -> list[Dict[str, Any]]:
    groups = []
    for modifier in obj.modifiers:
        group = getattr(modifier, "node_group", None) if modifier.type == "NODES" else None
        if group is None:
            continue
        nodes = []
        for node in sorted(group.nodes, key=lambda item: item.name):
            defaults = {}
            for node_socket in node.inputs:
                if not node_socket.is_linked and hasattr(node_socket, "default_value"):
                    defaults[node_socket.name] = _socket_default_summary(node_socket)
            nodes.append({"name": node.name, "type": node.bl_idname, "label": node.label, "inputs": defaults})
        links = sorted([
            {"from_node": link.from_node.name, "from_socket": link.from_socket.name,
             "to_node": link.to_node.name, "to_socket": link.to_socket.name}
            for link in group.links
        ], key=lambda item: (item["from_node"], item["from_socket"], item["to_node"], item["to_socket"]))
        payload = {"name": group.name, "interface": _geometry_interface_summary(group), "nodes": nodes, "links": links}
        groups.append({**payload, "graph_hash": content_hash(payload)})
    return groups


def _modifier_summary(obj: Any) -> list[Dict[str, Any]]:
    """Return bounded modifier metadata, including a stable parameter hash."""
    result = []
    for modifier in obj.modifiers:
        properties: Dict[str, Any] = {}
        rna = getattr(modifier, "bl_rna", None)
        for prop in (getattr(rna, "properties", []) if rna is not None else []):
            identifier = getattr(prop, "identifier", "")
            if not identifier or identifier in {"rna_type", "name", "type"} or getattr(prop, "is_readonly", False):
                continue
            try:
                value = getattr(modifier, identifier)
            except (AttributeError, TypeError):
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                properties[identifier] = value
            elif hasattr(value, "name"):
                properties[identifier] = str(value.name)
            elif isinstance(value, (list, tuple)) and len(value) <= 4:
                try:
                    properties[identifier] = [round(float(item), 8) for item in value]
                except (TypeError, ValueError):
                    pass
            if len(properties) >= 32:
                break
        result.append({"name": modifier.name, "type": modifier.type, "properties": properties, "property_hash": content_hash(properties)})
    return result


def scene_summary(*, detail: str = "full") -> Dict[str, Any]:
    """Build a scene census, omitting hash-heavy fields for compact callers."""
    _require_bpy()
    detail = str(detail).strip().lower()
    if detail not in {"compact", "full"}:
        raise ExecutorError("scene summary detail must be 'compact' or 'full'", "invalid_args")
    compact = detail == "compact"
    objects = []
    collections = {}
    for obj in sorted(bpy.context.scene.objects, key=lambda item: _stable_uuid(item)):
        collections[_stable_uuid(obj)] = sorted(collection.name for collection in obj.users_collection)
        uv_layers = [] if compact else _uv_layer_summary(obj)
        armature_bones = []
        pose_bones = []
        if not compact and obj.type == "ARMATURE" and obj.data is not None:
            armature_bones = [{"name": bone.name, "parent": bone.parent.name if bone.parent else None} for bone in obj.data.bones]
            pose_bones = [{
                "name": bone.name,
                "location": [round(float(value), 8) for value in bone.location],
                "rotation_euler": [round(float(value), 8) for value in bone.rotation_euler],
                "scale": [round(float(value), 8) for value in bone.scale],
            } for bone in obj.pose.bones]
        camera_data = None
        if not compact and obj.type == "CAMERA" and obj.data is not None:
            camera_data = {
                "type": obj.data.type,
                "lens": round(float(obj.data.lens), 8),
                "ortho_scale": round(float(obj.data.ortho_scale), 8),
                "shift": [round(float(obj.data.shift_x), 8), round(float(obj.data.shift_y), 8)],
                "clip_start": round(float(obj.data.clip_start), 8),
                "clip_end": round(float(obj.data.clip_end), 8),
                "dof_use": bool(getattr(obj.data.dof, "use_dof", False)),
                "dof_fstop": round(float(getattr(obj.data.dof, "aperture_fstop", 0.0)), 8),
                "dof_target": _stable_uuid(obj.data.dof.focus_object) if getattr(obj.data.dof, "focus_object", None) else None,
                "target": _load_json_prop(obj, _CAMERA_TARGET_PROP, None),
                "active": bool(bpy.context.scene.camera == obj),
            }
        light_data = None
        if not compact and obj.type == "LIGHT" and obj.data is not None:
            light_data = {
                "type": obj.data.type,
                "energy": round(float(obj.data.energy), 8),
                "color": [round(float(value), 8) for value in obj.data.color],
                "size": round(float(getattr(obj.data, "size", 0.0)), 8),
                "size_y": round(float(getattr(obj.data, "size_y", 0.0)), 8),
                "spot_size": round(float(getattr(obj.data, "spot_size", 0.0)), 8),
                "spot_blend": round(float(getattr(obj.data, "spot_blend", 0.0)), 8),
                "shadow_soft_size": round(float(getattr(obj.data, "shadow_soft_size", 0.0)), 8),
                "target": _load_json_prop(obj, _CAMERA_TARGET_PROP, None),
            }
        material_graphs = []
        if not compact:
            for slot in obj.material_slots:
                if slot.material and slot.material.use_nodes:
                    material_graphs.append(_material_graph_summary(slot.material))
        objects.append({
            "uuid": _stable_uuid(obj),
            "ref": str(obj.get(_REF_PROP)) if obj.get(_REF_PROP) else None,
            "name": obj.name,
            "type": obj.type,
            "origin": str(obj.get(_ORIGIN_PROP)) if obj.get(_ORIGIN_PROP) else None,
            "role": str(obj.get(_ROLE_PROP)) if obj.get(_ROLE_PROP) else None,
            "representation": str(obj.get(_REPRESENTATION_PROP)) if obj.get(_REPRESENTATION_PROP) else None,
            "quality_stage": str(obj.get(_QUALITY_STAGE_PROP)) if obj.get(_QUALITY_STAGE_PROP) else None,
            "collections": collections[_stable_uuid(obj)],
            "location": [round(float(v), 8) for v in obj.location],
            "rotation_euler": [round(float(v), 8) for v in obj.rotation_euler],
            "scale": [round(float(v), 8) for v in obj.scale],
            "aabb": _aabb(obj),
            "mesh": _mesh_stats(obj),
            "attributes": [] if compact else _mesh_attribute_summary(obj),
            "geometry_hash": None if compact else _mesh_geometry_hash(obj),
            "curve_geometry_hash": None if compact else _curve_geometry_hash(obj),
            "camera": camera_data,
            "light": light_data,
            "materials": [slot.material.name for slot in obj.material_slots if slot.material],
            "material_graphs": material_graphs,
            "modifiers": [modifier.type for modifier in obj.modifiers],
            "modifier_details": [] if compact else _modifier_summary(obj),
            "geometry_nodes": [] if compact else _geometry_nodes_summary(obj),
            "vertex_groups": [] if compact else [group.name for group in obj.vertex_groups],
            "vertex_group_hash": None if compact else _vertex_group_hash(obj),
            "semantic_tags": _semantic_tags(obj),
            "uv_layers": uv_layers,
            "armature_bones": armature_bones,
            "pose_bones": pose_bones,
            "animation": None if compact else _animation_summary(obj),
            "shape_keys": None if compact else _shape_key_summary(obj),
            "coordinate_frame": _load_json_prop(obj, _COORDINATE_PROP, {"space": "WORLD", "units": _coordinate_system().get("units", "meters")}),
            "parent": _stable_uuid(obj.parent) if obj.parent else None,
            "parent_name": obj.parent.name if obj.parent else None,
            "attachment": _load_json_prop(obj, _ATTACHMENT_PROP, None),
            "surface_snap": _load_json_prop(obj, _SNAP_PROP, None),
            "anchors": _load_json_prop(obj, _ANCHORS_PROP, {}),
        })
    meshes = [item for item in objects if item["type"] == "MESH"]
    return {
        "scene": bpy.context.scene.name,
        "contract": _scene_contract(),
        "coordinate_system": _coordinate_system(),
        "objects": objects,
        "n_total": len(objects),
        "n_mesh": len(meshes),
        "polys": sum(item["mesh"]["faces"] for item in meshes),
        "materials": len(bpy.data.materials),
        "filepath": bpy.data.filepath or None,
        "frame_current": bpy.context.scene.frame_current,
        "frame_range": [bpy.context.scene.frame_start, bpy.context.scene.frame_end],
        "fps": int(round(float(bpy.context.scene.render.fps))),
    }


def _inspect_batch(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Inspect selected objects from one authoritative scene census.

    Direct references preserve caller order.  Query-based selections use the
    UUID order already established by :func:`scene_summary`, so replay and
    downstream dataset consumers receive stable results without repeated
    ``inspect.object`` calls.

    ``compact`` is the default and avoids material/UV/geometry graph hashes;
    callers can opt into the complete object report with ``full`` or request a
    bounded field projection.
    """
    detail = str(args.get("detail", "compact")).strip().lower()
    if detail not in {"compact", "full"}:
        raise ExecutorError("detail must be 'compact' or 'full'", "invalid_args")
    raw_fields = args.get("fields")
    if raw_fields is not None:
        if not isinstance(raw_fields, list) or not raw_fields:
            raise ExecutorError("fields must be a non-empty array", "invalid_args")
        if len(raw_fields) > 64 or any(not isinstance(field, str) or not field.strip() for field in raw_fields):
            raise ExecutorError("fields must contain at most 64 non-empty strings", "invalid_args")
        requested_fields = list(dict.fromkeys(field.strip() for field in raw_fields))
    else:
        requested_fields = None
    compact_fields = (
        "uuid", "ref", "name", "type", "origin", "role", "representation", "quality_stage", "collections", "location", "scale",
        "aabb", "mesh", "materials", "semantic_tags", "coordinate_frame", "parent", "parent_name",
    )
    full_fields = (
        "uuid", "ref", "name", "type", "origin", "role", "representation", "quality_stage", "collections", "location", "rotation_euler", "scale",
        "aabb", "mesh", "attributes", "geometry_hash", "curve_geometry_hash", "camera", "light",
        "materials", "material_graphs", "modifiers", "modifier_details", "geometry_nodes",
        "vertex_groups", "vertex_group_hash", "semantic_tags", "uv_layers", "armature_bones",
        "pose_bones", "animation", "shape_keys", "coordinate_frame", "parent", "parent_name", "attachment", "surface_snap", "anchors",
    )
    allowed_fields = set(full_fields)
    if requested_fields is not None:
        unknown_fields = sorted(set(requested_fields) - allowed_fields)
        if unknown_fields:
            raise ExecutorError(f"unknown inspect field(s): {', '.join(unknown_fields)}", "invalid_args")
        selected_fields = requested_fields
    elif detail == "full":
        selected_fields = list(full_fields)
    else:
        selected_fields = list(compact_fields)
    requires_full = detail == "full" or any(field not in compact_fields for field in (requested_fields or ()))
    summary = scene_summary(detail="full" if requires_full else "compact")
    all_objects = list(summary.get("objects", []))
    by_uuid = {str(item.get("uuid")): item for item in all_objects if item.get("uuid")}
    by_ref = {str(item.get("ref")): item for item in all_objects if item.get("ref")}
    by_name = {str(item.get("name")): item for item in all_objects if item.get("name")}

    raw_targets = args.get("targets")
    requested: list[str] = []
    missing: list[str] = []
    selected: list[Dict[str, Any]] = []
    if raw_targets is not None:
        if isinstance(raw_targets, str):
            requested = [raw_targets]
        elif isinstance(raw_targets, list):
            if not raw_targets:
                raise ExecutorError("targets must contain at least one object reference", "invalid_args")
            requested = [str(value) for value in raw_targets]
        else:
            raise ExecutorError("targets must be a string or array", "invalid_args")
        seen: set[str] = set()
        for ref in requested:
            item = by_uuid.get(ref) or by_ref.get(ref) or by_name.get(ref)
            if item is None:
                missing.append(ref)
                continue
            identity = str(item.get("uuid"))
            if identity not in seen:
                selected.append(item)
                seen.add(identity)
        if missing and bool(args.get("strict", True)):
            raise ExecutorError(f"object(s) not found: {', '.join(missing)}", "not_found")
        source = "targets"
    else:
        # Top-level filters are convenient for simple calls; ``query`` (or
        # its compatibility alias ``filter``) can provide the same fields in
        # a nested object for callers that build filters programmatically.
        query: Dict[str, Any] = {}
        for key in ("name", "name_contains", "uuid", "semantic_tag", "semantic_tags", "collection", "object_type", "type"):
            if key in args:
                query[key] = args[key]
        for key in ("query", "filter"):
            value = args.get(key)
            if value is not None:
                if not isinstance(value, Mapping):
                    raise ExecutorError(f"{key} must be an object", "invalid_args")
                query.update(value)
        known_query = {"name", "name_contains", "uuid", "semantic_tag", "semantic_tags", "collection", "object_type", "type"}
        unknown_query = sorted(set(query) - known_query, key=str)
        # Unknown keys are intentionally ignored to preserve forward
        # compatibility with richer query clients.  The protocol schema keeps
        # ``additionalProperties`` enabled for the same reason.
        del unknown_query

        required_tags = query.get("semantic_tags") or []
        if isinstance(required_tags, str):
            required_tags = [required_tags]
        if not isinstance(required_tags, (list, tuple)):
            raise ExecutorError("semantic_tags must be an array", "invalid_args")
        required_tags = {str(tag) for tag in required_tags}
        semantic_tag = query.get("semantic_tag")
        object_type = query.get("object_type", query.get("type"))
        object_type = str(object_type).upper() if object_type is not None else None
        collection = query.get("collection")
        name = query.get("name")
        name_contains = query.get("name_contains")
        uuid_ref = query.get("uuid")
        for item in all_objects:
            tags = {str(tag) for tag in item.get("semantic_tags", []) or []}
            item_collections = {str(value) for value in item.get("collections", []) or []}
            if name is not None and item.get("name") != name:
                continue
            if name_contains is not None and str(name_contains) not in str(item.get("name", "")):
                continue
            if uuid_ref is not None and item.get("uuid") != uuid_ref:
                continue
            if semantic_tag is not None and str(semantic_tag) not in tags:
                continue
            if required_tags and not required_tags.issubset(tags):
                continue
            if collection is not None and str(collection) not in item_collections:
                continue
            if object_type is not None and str(item.get("type", "")).upper() != object_type:
                continue
            selected.append(item)
        source = "query" if query else "all"

    def project(item: Mapping[str, Any]) -> Dict[str, Any]:
        # Identity fields stay present for custom field projections so callers
        # can always correlate a result back to a stable object reference.
        identity_fields = [
            identity for identity in ("uuid", "ref", "name", "type")
            if identity in item and identity not in selected_fields
        ]
        fields = identity_fields + list(selected_fields)
        return {field: item.get(field) for field in fields if field in item}

    try:
        limit = int(args.get("limit", 256))
    except (TypeError, ValueError) as exc:
        raise ExecutorError("limit must be a positive integer", "invalid_args") from exc
    if limit < 1 or limit > 1024:
        raise ExecutorError("limit must be between 1 and 1024", "invalid_args")
    truncated = len(selected) > limit
    selected = selected[:limit]
    return {
        "objects": [project(item) for item in selected],
        "count": len(selected),
        "requested": requested,
        "missing": missing,
        "source": source,
        "limit": limit,
        "truncated": truncated,
        "detail": detail,
        "fields": list(selected_fields),
    }


def _apply_object_transform_values(obj: Any, args: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply the fields accepted by ``object.transform`` to one object."""
    frame = _coordinate_frame(args)
    space = str(frame.get("space", "WORLD")).upper()
    if space not in {"WORLD", "LOCAL", "PARENT"}:
        raise ExecutorError(f"unsupported transform space: {space}", "invalid_args")
    if "location" in args:
        location = _length_vector(args["location"], "location", frame)
        if space == "WORLD":
            world = obj.matrix_world.copy()
            world.translation = _point_to_world(args["location"], "location", frame)
            obj.matrix_world = world
        elif space == "PARENT":
            if obj.parent is None:
                # With no parent, PARENT has no reference and is rejected by
                # the relative-space contract instead of silently becoming a
                # different frame.
                raise ExecutorError("absolute location in PARENT space requires an object parent", "precondition_failed")
            else:
                obj.location = _coordinate_basis(frame) @ location
        else:
            raise ExecutorError("absolute location in LOCAL space is ambiguous; use location_delta for a local offset", "invalid_args")
    elif "location_delta" in args:
        delta = _coordinate_basis(frame) @ _length_vector(args["location_delta"], "location_delta", frame)
        world = obj.matrix_world.copy()
        world.translation += _world_delta(obj, delta, space)
        obj.matrix_world = world
    if "rotation_euler" in args:
        rotation = Euler(_as_float3(args["rotation_euler"], "rotation_euler"), "XYZ")
        if space == "WORLD":
            _set_world_rotation(obj, rotation)
        elif space == "PARENT":
            obj.rotation_mode = "XYZ"
            obj.rotation_euler = rotation
        else:  # LOCAL: apply the requested Euler as a local rotation delta.
            current = obj.matrix_world.to_quaternion()
            _set_world_rotation(obj, current @ rotation)
    if "scale" in args:
        obj.scale = _as_float3(args["scale"], "scale")
    if "semantic_tags" in args:
        obj[_SEMANTIC_PROP] = list(args["semantic_tags"] or [])
    _store_json_prop(obj, _COORDINATE_PROP, frame)
    result = {"uuid": _stable_uuid(obj), "name": obj.name, "coordinate_frame": frame}
    reference = obj.get(_REF_PROP) if hasattr(obj, "get") else None
    if reference:
        result["ref"] = str(reference)
    return result


def _capture_transform_state(obj: Any) -> Dict[str, Any]:
    tags = obj.get(_SEMANTIC_PROP) if hasattr(obj, "get") else None
    return {
        "object": obj,
        "location": obj.location.copy(),
        "rotation_euler": obj.rotation_euler.copy(),
        "scale": obj.scale.copy(),
        "has_tags": tags is not None,
        "semantic_tags": list(tags or []) if tags is not None else [],
        "coordinate_frame": _load_json_prop(obj, _COORDINATE_PROP, None),
    }


def _restore_transform_states(states: Iterable[Mapping[str, Any]]) -> None:
    for state in states:
        obj = state["object"]
        obj.location = state["location"]
        obj.rotation_euler = state["rotation_euler"]
        obj.scale = state["scale"]
        if state.get("has_tags"):
            obj[_SEMANTIC_PROP] = list(state.get("semantic_tags", []))
        elif hasattr(obj, "__delitem__"):
            try:
                del obj[_SEMANTIC_PROP]
            except (KeyError, TypeError):
                pass
        coordinate_frame = state.get("coordinate_frame")
        if coordinate_frame is None:
            try:
                del obj[_COORDINATE_PROP]
            except (KeyError, TypeError):
                pass
        else:
            _store_json_prop(obj, _COORDINATE_PROP, coordinate_frame)
    bpy.context.view_layer.update()


def _object_transform_batch(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply ordered object transforms as one atomic, trajectory-visible action."""
    _require_bpy()
    transforms = args.get("transforms")
    if not isinstance(transforms, list) or not transforms:
        raise ExecutorError("transforms must contain at least one item", "invalid_args")
    if len(transforms) > 256:
        raise ExecutorError("transforms exceeds maximum item count 256", "invalid_args")
    atomic = bool(args.get("atomic", True))
    stop_on_error = bool(args.get("stop_on_error", True))
    snapshots: Dict[int, Dict[str, Any]] = {}
    results: list[Dict[str, Any]] = []
    successful = 0
    failed = 0
    for index, raw in enumerate(transforms):
        if not isinstance(raw, Mapping):
            exc = ExecutorError(f"transforms[{index}] must be an object", "invalid_args")
            entry = {"index": index, "ok": False, "error": {"code": exc.code, "message": str(exc)}}
            results.append(entry)
            failed += 1
            if stop_on_error:
                if atomic and snapshots:
                    _restore_transform_states(snapshots.values())
                raise exc
            continue
        ref = raw.get("target")
        try:
            obj = _object_by_ref(ref)
            key = id(obj)
            if key not in snapshots:
                snapshots[key] = _capture_transform_state(obj)
            result = _apply_object_transform_values(obj, raw)
            results.append({"index": index, "target": str(ref), "ok": True, "result": result})
            successful += 1
        except Exception as exc:
            code = getattr(exc, "code", "execution_error")
            results.append({"index": index, "target": str(ref), "ok": False, "error": {"code": code, "message": str(exc)}})
            failed += 1
            if stop_on_error:
                if atomic and snapshots:
                    _restore_transform_states(snapshots.values())
                raise ExecutorError(f"transform batch failed at index {index}: {exc}", code) from exc

    rolled_back = bool(atomic and failed)
    if rolled_back:
        _restore_transform_states(snapshots.values())
    else:
        bpy.context.view_layer.update()
    return {
        "count": len(transforms),
        "successful": successful,
        "failed": failed,
        "atomic": atomic,
        "committed": not rolled_back,
        "rolled_back": rolled_back,
        "transforms": results,
    }


def _delete_all() -> None:
    _require_bpy()
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for item in list(collection):
            if item.users == 0:
                collection.remove(item)
    bpy.context.scene.pop(_COORDINATE_PROP, None)


def _create_primitive(args: Mapping[str, Any], stable_id: Optional[str] = None) -> Dict[str, Any]:
    _require_bpy()
    kind = str(args.get("kind", "")).lower()
    name = str(args.get("name") or f"{kind}_{len(bpy.data.objects)}")
    frame = _creation_coordinate_frame(args)
    location = tuple(_point_to_world(args.get("location", (0, 0, 0)), "location", frame))
    scale = _as_float3(args.get("scale", (1, 1, 1)), "scale")
    primitive_map = {
        "cube": (bpy.ops.mesh.primitive_cube_add, {}),
        "uv_sphere": (bpy.ops.mesh.primitive_uv_sphere_add, {"segments": int(args.get("segments", 32)), "ring_count": int(args.get("rings", 16))}),
        "sphere": (bpy.ops.mesh.primitive_uv_sphere_add, {"segments": int(args.get("segments", 32)), "ring_count": int(args.get("rings", 16))}),
        "ico_sphere": (bpy.ops.mesh.primitive_ico_sphere_add, {"subdivisions": int(args.get("subdivisions", 3))}),
        "cylinder": (bpy.ops.mesh.primitive_cylinder_add, {"vertices": int(args.get("vertices", 32))}),
        "cone": (bpy.ops.mesh.primitive_cone_add, {"vertices": int(args.get("vertices", 32))}),
        "torus": (bpy.ops.mesh.primitive_torus_add, {"major_segments": int(args.get("major_segments", 48)), "minor_segments": int(args.get("minor_segments", 16))}),
        "plane": (bpy.ops.mesh.primitive_plane_add, {}),
        "grid": (bpy.ops.mesh.primitive_grid_add, {"x_subdivisions": int(args.get("x_subdivisions", 10)), "y_subdivisions": int(args.get("y_subdivisions", 10))}),
        "circle": (bpy.ops.mesh.primitive_circle_add, {"vertices": int(args.get("vertices", 32)), "fill_type": str(args.get("fill_type", "NGON"))}),
        "monkey": (bpy.ops.mesh.primitive_monkey_add, {}),
    }
    if kind not in primitive_map:
        raise ExecutorError(f"unsupported primitive kind: {kind}", "invalid_args")
    operator, kwargs = primitive_map[kind]
    operator(location=location, **kwargs)
    if "rotation_euler" in args:
        obj_rotation = Euler(_as_float3(args["rotation_euler"], "rotation_euler"), "XYZ")
    else:
        obj_rotation = None
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    if obj_rotation is not None:
        obj.rotation_euler = obj_rotation
    obj[_UUID_PROP] = stable_id or _stable_uuid(obj)
    reference = args.get("id", args.get("ref"))
    if reference is not None:
        if not isinstance(reference, str) or not reference.strip():
            raise ExecutorError("id/ref must be a non-empty string", "invalid_args")
        reference = reference.strip()
        if any(item.get(_REF_PROP) == reference for item in bpy.context.scene.objects if item is not obj):
            raise ExecutorError(f"object reference already exists: {reference}", "conflict")
        obj[_REF_PROP] = reference
    if "semantic_tags" in args:
        obj[_SEMANTIC_PROP] = list(args["semantic_tags"] or [])
    _authoring_metadata(obj, args, origin=f"primitive:{kind}", default_representation="primitive")
    _store_json_prop(obj, _COORDINATE_PROP, frame)
    bpy.context.view_layer.update()
    result = {"uuid": _stable_uuid(obj), "name": obj.name, "coordinate_frame": frame}
    if reference is not None:
        result["ref"] = reference.strip()
    return result


def _remove_object_and_data(ref: str) -> None:
    """Remove a newly-created object without disturbing unrelated datablocks."""
    obj = _object_by_ref(ref)
    data = getattr(obj, "data", None)
    object_type = getattr(obj, "type", "")
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is not None and getattr(data, "users", 0) == 0:
        collection = {
            "MESH": bpy.data.meshes,
            "CURVE": bpy.data.curves,
            "SURFACE": bpy.data.curves,
        }.get(object_type)
        if collection is not None:
            try:
                collection.remove(data, do_unlink=True)
            except (ReferenceError, RuntimeError, TypeError):
                pass


def _remove_orphan_data(data: Any) -> None:
    """Remove an unlinked Blender datablock when its collection is known."""
    if data is None or getattr(data, "users", 0) != 0:
        return
    for collection in (
        getattr(bpy.data, "meshes", None),
        getattr(bpy.data, "curves", None),
        getattr(bpy.data, "cameras", None),
        getattr(bpy.data, "lights", None),
        getattr(bpy.data, "metaballs", None),
        getattr(bpy.data, "armatures", None),
        getattr(bpy.data, "lattices", None),
        getattr(bpy.data, "grease_pencils_v3", None),
        getattr(bpy.data, "grease_pencils", None),
        getattr(bpy.data, "pointclouds", None),
        getattr(bpy.data, "volumes", None),
    ):
        if collection is None:
            continue
        try:
            if any(candidate is data for candidate in collection):
                collection.remove(data, do_unlink=True)
                return
        except (ReferenceError, RuntimeError, TypeError):
            try:
                collection.remove(data)
                return
            except (ReferenceError, RuntimeError, TypeError, ValueError):
                continue


def _discard_created_object(obj: Any) -> None:
    """Remove a newly-created object and any now-unused datablock."""
    if obj is None:
        return
    data = getattr(obj, "data", None)
    objects = getattr(getattr(bpy, "data", None), "objects", None)
    if objects is not None:
        try:
            remover = getattr(objects, "remove", None)
            if callable(remover):
                try:
                    remover(obj, do_unlink=True)
                except TypeError:
                    remover(obj)
            elif obj in objects:
                objects.remove(obj)
        except (ReferenceError, RuntimeError, ValueError):
            pass
    _remove_orphan_data(data)


def _object_create_batch(args: Mapping[str, Any], stable_id: Optional[str] = None) -> Dict[str, Any]:
    """Create primitives with deterministic child identities in one action."""
    _require_bpy()
    objects = args.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ExecutorError("objects must contain at least one item", "invalid_args")
    if len(objects) > 256:
        raise ExecutorError("objects exceeds maximum item count 256", "invalid_args")
    atomic = bool(args.get("atomic", True))
    existing_refs = {
        str(obj.get(_REF_PROP))
        for obj in bpy.context.scene.objects
        if obj.get(_REF_PROP)
    }
    batch_refs: set[str] = set()
    for index, raw in enumerate(objects):
        if not isinstance(raw, Mapping):
            raise ExecutorError(f"objects[{index}] must be an object", "invalid_args")
        reference = raw.get("id", raw.get("ref"))
        if reference is None:
            continue
        if not isinstance(reference, str) or not reference.strip():
            raise ExecutorError(f"objects[{index}].id/ref must be a non-empty string", "invalid_args")
        reference = reference.strip()
        if reference in existing_refs or reference in batch_refs:
            raise ExecutorError(f"object reference already exists: {reference}", "conflict")
        batch_refs.add(reference)
    before_uuids = {_stable_uuid(obj) for obj in bpy.context.scene.objects}
    created: list[Dict[str, Any]] = []
    try:
        for index, raw in enumerate(objects):
            if not isinstance(raw, Mapping):
                raise ExecutorError(f"objects[{index}] must be an object", "invalid_args")
            item = dict(raw)
            child_id = "obj-" + content_hash({"batch": stable_id, "index": index, "item": item})[7:23] if stable_id else None
            created.append(_create_primitive(item, child_id))
    except Exception as exc:
        if atomic:
            # Use the pre-action census as a safety net for a primitive that
            # creates its object before a later validation/operator error.
            for obj in list(bpy.context.scene.objects):
                object_uuid = _stable_uuid(obj)
                if object_uuid not in before_uuids:
                    try:
                        _remove_object_and_data(object_uuid)
                    except Exception:
                        pass
        code = getattr(exc, "code", "execution_error")
        raise ExecutorError(f"object create batch failed: {exc}", code) from exc
    bpy.context.view_layer.update()
    return {
        "count": len(created),
        "successful": len(created),
        "failed": 0,
        "atomic": atomic,
        "committed": True,
        "objects": created,
    }


def _create_curve(args: Mapping[str, Any], stable_id: Optional[str] = None) -> Dict[str, Any]:
    _require_bpy()
    points = args.get("points")
    if not isinstance(points, list) or len(points) < 2:
        raise ExecutorError("points must contain at least two positions", "invalid_args")
    frame = _creation_coordinate_frame(args)
    parsed = [tuple(_point_to_world(point, "points[]", frame)) for point in points]
    curve = bpy.data.curves.new(str(args.get("name", "Curve")), "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = int(args.get("resolution", 12))
    curve.bevel_depth = float(_length_value(args.get("bevel_depth", 0.0), "bevel_depth", frame))
    curve.bevel_resolution = int(args.get("bevel_resolution", 3))
    spline = curve.splines.new("BEZIER" if args.get("bezier", False) else "POLY")
    if spline.type == "BEZIER":
        spline.bezier_points.add(len(parsed) - 1)
        for point, coordinate in zip(spline.bezier_points, parsed):
            point.co = coordinate
            point.handle_left_type = point.handle_right_type = "AUTO"
    else:
        spline.points.add(len(parsed) - 1)
        for point, coordinate in zip(spline.points, parsed):
            point.co = (*coordinate, 1.0)
    obj = bpy.data.objects.new(str(args.get("name", "Curve")), curve)
    bpy.context.scene.collection.objects.link(obj)
    obj[_UUID_PROP] = stable_id or _stable_uuid(obj)
    reference = args.get("id", args.get("ref"))
    if reference is not None:
        if not isinstance(reference, str) or not reference.strip():
            raise ExecutorError("id/ref must be a non-empty string", "invalid_args")
        obj[_REF_PROP] = reference.strip()
    _authoring_metadata(obj, args, origin="curve.create", default_representation="curve")
    _store_json_prop(obj, _COORDINATE_PROP, frame)
    result = {"uuid": _stable_uuid(obj), "name": obj.name, "coordinate_frame": frame}
    if reference is not None:
        result["ref"] = reference.strip()
    return result


def _create_mesh(args: Mapping[str, Any], stable_id: Optional[str] = None) -> Dict[str, Any]:
    _require_bpy()
    vertices = args.get("vertices")
    faces = args.get("faces")
    if not isinstance(vertices, list) or not isinstance(faces, list):
        raise ExecutorError("vertices and faces must be arrays", "invalid_args")
    frame = _creation_coordinate_frame(args)
    parsed_vertices = [tuple(_point_to_world(vertex, "vertices[]", frame)) for vertex in vertices]
    parsed_faces = []
    for face in faces:
        if not isinstance(face, list) or len(face) < 3:
            raise ExecutorError("each face must have at least 3 indices", "invalid_args")
        if any(not isinstance(index, int) or index < 0 or index >= len(parsed_vertices) for index in face):
            raise ExecutorError("face index out of range", "invalid_args")
        parsed_faces.append(tuple(face))
    mesh = bpy.data.meshes.new(str(args.get("name", "Mesh")))
    mesh.from_pydata(parsed_vertices, [], parsed_faces)
    mesh.update()
    obj = bpy.data.objects.new(str(args.get("name", "Mesh")), mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj[_UUID_PROP] = stable_id or _stable_uuid(obj)
    reference = args.get("id", args.get("ref"))
    if reference is not None:
        if not isinstance(reference, str) or not reference.strip():
            raise ExecutorError("id/ref must be a non-empty string", "invalid_args")
        obj[_REF_PROP] = reference.strip()
    obj[_SEMANTIC_PROP] = list(args.get("semantic_tags", []))
    _authoring_metadata(obj, args, origin="mesh.from_pydata", default_representation="control_mesh")
    _store_json_prop(obj, _COORDINATE_PROP, frame)
    result = {"uuid": _stable_uuid(obj), "name": obj.name, "coordinate_frame": frame}
    if reference is not None:
        result["ref"] = reference.strip()
    return result


def _mesh_from_sections(args: Mapping[str, Any], stable_id: Optional[str] = None) -> Dict[str, Any]:
    """Create a closed, deterministic loft from scalar X-ordered sections.

    Each section describes the full Y width and Z height of an ellipse (or
    superellipse) centered at ``(x, 0, z)``.  Corresponding ring vertices are
    connected with quads; optional end caps make the default result
    watertight without relying on a Blender operator or hidden script.
    """
    _require_bpy()
    frame = _creation_coordinate_frame(args)
    raw_sections = args.get("sections")
    if not isinstance(raw_sections, list) or len(raw_sections) < 3:
        raise ExecutorError("sections must contain at least three items", "invalid_args")
    if len(raw_sections) > 128:
        raise ExecutorError("sections exceeds maximum item count 128", "invalid_args")
    try:
        segments = int(args.get("segments", 32))
    except (TypeError, ValueError) as exc:
        raise ExecutorError("segments must be an integer", "invalid_args") from exc
    if not 8 <= segments <= 256:
        raise ExecutorError("segments must be between 8 and 256", "invalid_args")
    profile = str(args.get("profile", "ellipse")).strip().lower()
    if profile not in {"ellipse", "superellipse", "custom"}:
        raise ExecutorError("profile must be 'ellipse', 'superellipse', or 'custom'", "invalid_args")
    custom_profile = args.get("profile_points") or []
    if profile == "custom":
        if not isinstance(custom_profile, list) or len(custom_profile) < 3:
            raise ExecutorError("custom profile requires at least three profile_points", "invalid_args")
        normalized_profile = []
        for point in custom_profile:
            if isinstance(point, Mapping):
                y, z = float(point.get("y", 0.0)), float(point.get("z", 0.0))
            else:
                if not isinstance(point, (list, tuple)) or len(point) not in {2, 3}:
                    raise ExecutorError("profile_points[] must contain 2 or 3 numbers", "invalid_args")
                try:
                    y, z = float(point[0]), float(point[1])
                except (TypeError, ValueError) as exc:
                    raise ExecutorError("profile_points[] must contain numbers", "invalid_args") from exc
            if not all(math.isfinite(value) for value in (y, z)):
                raise ExecutorError("custom profile points must be finite", "invalid_args")
            normalized_profile.append((y, z))
        perimeter = []
        for index, point in enumerate(normalized_profile):
            other = normalized_profile[(index + 1) % len(normalized_profile)]
            perimeter.append(math.hypot(other[0] - point[0], other[1] - point[1]))
        total_perimeter = sum(perimeter)
        if total_perimeter <= 1e-9:
            raise ExecutorError("custom profile points must span a non-zero perimeter", "invalid_args")
        resampled = []
        for segment_index in range(segments):
            distance = total_perimeter * segment_index / segments
            accumulated = 0.0
            for point_index, edge_length in enumerate(perimeter):
                if distance <= accumulated + edge_length or point_index == len(perimeter) - 1:
                    start = normalized_profile[point_index]
                    end = normalized_profile[(point_index + 1) % len(normalized_profile)]
                    t = 0.0 if edge_length <= 1e-9 else (distance - accumulated) / edge_length
                    resampled.append((start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t))
                    break
                accumulated += edge_length
        normalized_profile = resampled
    try:
        power = float(args.get("power", 2.0))
    except (TypeError, ValueError) as exc:
        raise ExecutorError("power must be a number", "invalid_args") from exc
    if not math.isfinite(power) or power <= 0.0 or power > 16.0:
        raise ExecutorError("power must be finite and in (0, 16]", "invalid_args")

    parsed: list[tuple[float, float, float, float, float, float, Optional[list[tuple[float, float]]]]] = []
    previous_x: Optional[float] = None
    for index, raw in enumerate(raw_sections):
        if not isinstance(raw, Mapping):
            raise ExecutorError(f"sections[{index}] must be an object", "invalid_args")
        missing = [key for key in ("x", "width", "height") if key not in raw]
        if missing:
            raise ExecutorError(f"sections[{index}] missing required keys: {missing}", "invalid_args")
        try:
            x = float(raw["x"])
            width = float(raw["width"])
            height = float(raw["height"])
            center_z = float(raw.get("z", 0.0))
            center = raw.get("center")
            center_y = float(center[0]) if isinstance(center, (list, tuple)) and len(center) >= 1 else 0.0
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                center_z = float(center[1])
            rotation_x = float(raw.get("rotation_x", 0.0))
        except (TypeError, ValueError) as exc:
            raise ExecutorError(f"sections[{index}] values must be numbers", "invalid_args") from exc
        if not all(math.isfinite(value) for value in (x, width, height, center_z, center_y, rotation_x)):
            raise ExecutorError(f"sections[{index}] values must be finite", "invalid_args")
        if width <= 0.0 or height <= 0.0:
            raise ExecutorError(f"sections[{index}] width and height must be positive", "invalid_args")
        if abs(x) > 1_000_000.0 or abs(center_z) > 1_000_000.0 or width > 10_000.0 or height > 10_000.0:
            raise ExecutorError(f"sections[{index}] exceeds coordinate or dimension bounds", "invalid_args")
        if previous_x is not None and x <= previous_x:
            raise ExecutorError("sections must be ordered by strictly increasing x", "invalid_args")
        previous_x = x
        scale = _UNIT_TO_METERS.get(str(frame.get("units", "meters")))
        if scale is None:
            raise ExecutorError(f"unsupported length units: {frame.get('units')}", "invalid_args")
        section_profile = None
        if raw.get("profile_points") is not None:
            values = raw.get("profile_points")
            if not isinstance(values, list) or len(values) < 3:
                raise ExecutorError(f"sections[{index}].profile_points must contain at least three points", "invalid_args")
            section_profile = []
            for point in values:
                if isinstance(point, Mapping):
                    sy, sz = float(point.get("y", 0.0)), float(point.get("z", 0.0))
                else:
                    if not isinstance(point, (list, tuple)) or len(point) < 2:
                        raise ExecutorError(f"sections[{index}].profile_points contains an invalid point", "invalid_args")
                    sy, sz = float(point[0]), float(point[1])
                section_profile.append((sy, sz))
        parsed.append((x * scale, width * scale, height * scale, center_z * scale, center_y * scale, rotation_x, section_profile))
    # End-cap faces reuse the first/last rings; they do not add vertices.
    # Keep the declared 32,768-vertex upper bound reachable in capped mode.
    if len(parsed) * segments > 32768:
        raise ExecutorError("loft exceeds maximum vertex budget 32768", "resource_limit")

    # Resolve references before creating any datablocks, so a conflict leaves
    # the scene untouched just like the primitive creation actions.
    name = str(args.get("name") or "Loft")
    reference = args.get("id", args.get("ref"))
    if reference is not None:
        if not isinstance(reference, str) or not reference.strip():
            raise ExecutorError("id/ref must be a non-empty string", "invalid_args")
        reference = reference.strip()
        if any(obj.get(_REF_PROP) == reference for obj in bpy.context.scene.objects if obj.get(_REF_PROP)):
            raise ExecutorError(f"object reference already exists: {reference}", "conflict")
    cap_ends = bool(args.get("cap_ends", True))
    smooth_shading = bool(args.get("smooth_shading", True))

    vertices: list[tuple[float, float, float]] = []
    rings: list[list[int]] = []
    exponent = 2.0 / power
    for x, width, height, center_z, center_y, rotation_x, section_profile in parsed:
        ring: list[int] = []
        half_width = width * 0.5
        half_height = height * 0.5
        for segment in range(segments):
            theta = (2.0 * math.pi * segment) / segments
            cosine = math.cos(theta)
            sine = math.sin(theta)
            effective_profile = section_profile if section_profile is not None else normalized_profile if profile == "custom" else None
            if effective_profile is not None:
                source = effective_profile[segment % len(effective_profile)]
                y = half_width * float(source[0])
                z = center_z + half_height * float(source[1])
            elif profile == "ellipse":
                y = half_width * cosine
                z = center_z + half_height * sine
            elif profile == "superellipse":
                y = half_width * math.copysign(abs(cosine) ** exponent, cosine)
                z = center_z + half_height * math.copysign(abs(sine) ** exponent, sine)
            else:
                raise ExecutorError("custom profile points are missing", "invalid_args")
            if abs(center_y) > 0.0:
                y += center_y
            if abs(rotation_x) > 1e-12:
                # Rotate the section profile around its local Y axis.
                angle = float(rotation_x)
                local_z = z - center_z
                y, local_z = y * math.cos(angle) - local_z * math.sin(angle), y * math.sin(angle) + local_z * math.cos(angle)
                z = center_z + local_z
            ring.append(len(vertices))
            vertices.append((x, y, z))
        rings.append(ring)

    faces: list[tuple[int, ...]] = []
    for section_index in range(len(rings) - 1):
        current = rings[section_index]
        following = rings[section_index + 1]
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            # The chosen ring order gives outward side normals for increasing X.
            faces.append((current[segment], current[next_segment], following[next_segment], following[segment]))
    if cap_ends:
        faces.append(tuple(reversed(rings[0])))
        faces.append(tuple(rings[-1]))

    basis = _coordinate_basis(frame)
    origin = _scene_origin_world(frame)
    vertices = [tuple(origin + basis @ Vector(vertex)) for vertex in vertices]

    mesh = bpy.data.meshes.new(name)
    obj = None
    try:
        mesh.from_pydata(vertices, [], faces)
        mesh.update(calc_edges=True)
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        obj[_UUID_PROP] = stable_id or _stable_uuid(obj)
        if reference is not None:
            obj[_REF_PROP] = reference
        obj[_SEMANTIC_PROP] = list(args.get("semantic_tags", []) or [])
        _authoring_metadata(obj, args, origin="mesh.from_sections", default_representation="section_stack")
        _store_json_prop(obj, _COORDINATE_PROP, frame)
        for polygon in mesh.polygons:
            polygon.use_smooth = smooth_shading
        mesh.update(calc_edges=True)
        bpy.context.view_layer.update()
        topology = _topology(obj)
        if cap_ends and not topology.get("watertight", False):
            raise ExecutorError("loft end caps did not produce a watertight mesh", "execution_error")
        result: Dict[str, Any] = {
            "uuid": _stable_uuid(obj),
            "name": obj.name,
            "sections": len(parsed),
            "segments": segments,
            "profile": profile,
            "power": round(power, 8),
            "cap_ends": cap_ends,
            "smooth_shading": smooth_shading,
            "vertices": len(mesh.vertices),
            "faces": len(mesh.polygons),
            "dimensions": [round(float(value), 8) for value in obj.dimensions],
            "topology": topology,
        }
        if reference is not None:
            result["ref"] = reference
        return result
    except Exception:
        if obj is not None and obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        elif mesh.name in bpy.data.meshes and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        raise


def _topology(obj: Any) -> Dict[str, Any]:
    if obj.type != "MESH" or obj.data is None:
        raise ExecutorError("topology requires a mesh object", "invalid_args")
    edge_counts: Dict[tuple[int, int], int] = {}
    for polygon in obj.data.polygons:
        indices = list(polygon.vertices)
        for idx, first in enumerate(indices):
            second = indices[(idx + 1) % len(indices)]
            edge = (first, second) if first < second else (second, first)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary = sum(count == 1 for count in edge_counts.values())
    nonmanifold = sum(count > 2 for count in edge_counts.values())
    parent = list(range(len(obj.data.vertices)))
    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    for edge in edge_counts:
        left, right = find(edge[0]), find(edge[1])
        if left != right:
            parent[left] = right
    shells = len({find(index) for index in range(len(obj.data.vertices))}) if obj.data.vertices else 0
    referenced_vertices = {int(index) for polygon in obj.data.polygons for index in polygon.vertices}
    duplicate_faces = 0
    seen_faces: set[tuple[int, ...]] = set()
    for polygon in obj.data.polygons:
        key = tuple(sorted(int(index) for index in polygon.vertices))
        if key in seen_faces:
            duplicate_faces += 1
        else:
            seen_faces.add(key)
    coordinate_keys: set[tuple[float, float, float]] = set()
    duplicate_vertices = 0
    non_finite_vertices = 0
    for vertex in obj.data.vertices:
        co = vertex.co
        if not all(math.isfinite(float(value)) for value in co):
            non_finite_vertices += 1
            continue
        key = tuple(round(float(value), 7) for value in co)
        if key in coordinate_keys:
            duplicate_vertices += 1
        else:
            coordinate_keys.add(key)
    face_areas = [float(polygon.area) for polygon in obj.data.polygons]
    zero_area_faces = sum(area <= 1e-12 for area in face_areas)
    return {
        "vertices": len(obj.data.vertices),
        "faces": len(obj.data.polygons),
        "edges": len(edge_counts),
        "boundary_edges": boundary,
        "nonmanifold_edges": nonmanifold,
        "shells": shells,
        "watertight": boundary == 0 and nonmanifold == 0,
        "euler_chi": len(obj.data.vertices) - len(edge_counts) + len(obj.data.polygons),
        "loose_vertices": len(obj.data.vertices) - len(referenced_vertices),
        "duplicate_vertices": duplicate_vertices,
        "duplicate_faces": duplicate_faces,
        "non_finite_vertices": non_finite_vertices,
        "zero_area_faces": zero_area_faces,
        "min_face_area": round(min(face_areas, default=0.0), 12),
        "min_edge_length": round(min(((obj.data.vertices[left].co - obj.data.vertices[right].co).length for left, right in edge_counts), default=0.0), 8),
        "max_edge_length": round(max(((obj.data.vertices[left].co - obj.data.vertices[right].co).length for left, right in edge_counts), default=0.0), 8),
    }


def _measure(obj: Any) -> Dict[str, Any]:
    _require_bpy()
    dimensions = [round(float(value), 8) for value in obj.dimensions]
    return {"dimensions": dimensions, "aabb": _aabb(obj), "volume": round(float(getattr(obj, "dimensions", (0, 0, 0))[0] * getattr(obj, "dimensions", (0, 0, 0))[1] * getattr(obj, "dimensions", (0, 0, 0))[2]), 8)}


def _require_mesh_object(ref: Any) -> Any:
    obj = _object_by_ref(ref)
    if obj.type != "MESH" or obj.data is None:
        raise ExecutorError("operation requires a mesh object", "invalid_args")
    return obj


def _selection_parts(bm: Any, selection: Mapping[str, Any], *, obj: Any = None, default_all: bool = True) -> tuple[list[Any], list[Any], list[Any]]:
    """Resolve an explicit, object-local selection without Blender edit-mode state."""
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    selected_verts: dict[int, Any] = {}
    selected_edges: dict[int, Any] = {}
    selected_faces: dict[int, Any] = {}
    selector_keys = {"region_handle", "vertex_indices", "edge_indices", "face_indices", "center", "radius", "box_min", "box_max", "vertex_group"}
    has_selector = any(key in selection for key in selector_keys)

    def add_vert(vertex: Any) -> None:
        selected_verts[int(vertex.index)] = vertex

    for index in selection.get("vertex_indices", []) or []:
        if not 0 <= int(index) < len(bm.verts):
            raise ExecutorError(f"vertex index out of range: {index}", "invalid_args")
        add_vert(bm.verts[int(index)])
    for index in selection.get("edge_indices", []) or []:
        if not 0 <= int(index) < len(bm.edges):
            raise ExecutorError(f"edge index out of range: {index}", "invalid_args")
        edge = bm.edges[int(index)]
        selected_edges[int(edge.index)] = edge
        for vertex in edge.verts:
            add_vert(vertex)
    for index in selection.get("face_indices", []) or []:
        if not 0 <= int(index) < len(bm.faces):
            raise ExecutorError(f"face index out of range: {index}", "invalid_args")
        face = bm.faces[int(index)]
        selected_faces[int(face.index)] = face
        for vertex in face.verts:
            add_vert(vertex)
    center = selection.get("center")
    radius = selection.get("radius")
    if center is not None and radius is not None:
        point = Vector(_as_float3(center, "selection.center"))
        limit = float(radius)
        for vertex in bm.verts:
            if (vertex.co - point).length <= limit:
                add_vert(vertex)
    box_min = selection.get("box_min")
    box_max = selection.get("box_max")
    if box_min is not None and box_max is not None:
        low = Vector(_as_float3(box_min, "selection.box_min"))
        high = Vector(_as_float3(box_max, "selection.box_max"))
        if any(low[index] > high[index] for index in range(3)):
            raise ExecutorError("selection.box_min must not exceed box_max", "invalid_args")
        for vertex in bm.verts:
            if all(low[index] <= vertex.co[index] <= high[index] for index in range(3)):
                add_vert(vertex)
    group_name = selection.get("vertex_group")
    region_handle = selection.get("region_handle")
    if region_handle is not None:
        if obj is None:
            raise ExecutorError("region_handle selection requires an object", "invalid_args")
        handle = str(region_handle)
        if handle.startswith("region:"):
            handle = handle[7:]
        region_names = obj.get(_REGION_PROP, []) if hasattr(obj, "get") else []
        if handle not in set(str(value) for value in (region_names or [])) and obj.vertex_groups.get(handle) is None:
            raise ExecutorError(f"region handle not found: {region_handle}", "not_found")
        group_name = handle
    if group_name is not None:
        if obj is None:
            raise ExecutorError("vertex_group selection requires an object", "invalid_args")
        group = obj.vertex_groups.get(str(group_name))
        if group is None:
            raise ExecutorError(f"vertex group not found: {group_name}", "not_found")
        threshold = float(selection.get("vertex_group_min_weight", 0.0))
        for vertex in bm.verts:
            try:
                if group.weight(int(vertex.index)) >= threshold:
                    add_vert(vertex)
            except RuntimeError:
                continue
    if default_all and not has_selector:
        selected_verts = {int(vertex.index): vertex for vertex in bm.verts}
        selected_edges = {int(edge.index): edge for edge in bm.edges}
        selected_faces = {int(face.index): face for face in bm.faces}
    if selected_verts and not selected_edges:
        selected_edges = {int(edge.index): edge for edge in bm.edges if all(int(v.index) in selected_verts for v in edge.verts)}
    if selected_verts and not selected_faces:
        selected_faces = {int(face.index): face for face in bm.faces if all(int(v.index) in selected_verts for v in face.verts)}
    return list(selected_verts.values()), list(selected_edges.values()), list(selected_faces.values())


def _write_bmesh(obj: Any, bm: Any) -> None:
    bm.normal_update()
    bm.to_mesh(obj.data)
    obj.data.update()


def _mesh_region_to_loop(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        _, _, faces = _selection_parts(bm, args.get("selection") or {}, obj=obj, default_all=False)
        if not faces:
            raise ExecutorError("region_to_loop requires a non-empty face selection", "invalid_args")
        selected = {face.index for face in faces}
        boundary = []
        for edge in bm.edges:
            linked = [face.index for face in edge.link_faces]
            if any(index in selected for index in linked) and any(index not in selected for index in linked):
                boundary.append(edge.index)
            elif len(linked) == 1 and linked[0] in selected:
                boundary.append(edge.index)
        boundary = sorted(set(int(index) for index in boundary))
        return {"target": _stable_uuid(obj), "edge_indices": boundary, "count": len(boundary), "hash": content_hash(boundary)}
    finally:
        bm.free()


def _mesh_symmetrize(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    direction = str(args["direction"]).upper()
    if direction not in {"POSITIVE_X", "NEGATIVE_X", "POSITIVE_Y", "NEGATIVE_Y", "POSITIVE_Z", "NEGATIVE_Z"}:
        raise ExecutorError("invalid symmetrize direction", "invalid_args")
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        verts, _, _ = _selection_parts(bm, args.get("selection") or {}, obj=obj, default_all=True)
        if not verts:
            raise ExecutorError("symmetrize selection is empty", "invalid_args")
        blender_direction = {"POSITIVE_X": "X", "NEGATIVE_X": "-X", "POSITIVE_Y": "Y", "NEGATIVE_Y": "-Y", "POSITIVE_Z": "Z", "NEGATIVE_Z": "-Z"}[direction]
        try:
            result = bmesh.ops.symmetrize(bm, input=verts, direction=blender_direction)
        except (TypeError, RuntimeError) as exc:
            raise ExecutorError(f"symmetrize failed: {exc}", "execution_error") from exc
        _write_bmesh(obj, bm)
        created = [item for item in result.get("geom", []) if isinstance(item, bmesh.types.BMVert)] if isinstance(result, Mapping) else []
        return {"target": _stable_uuid(obj), "direction": direction, "vertices_selected": len(verts), "vertices_created": len(created), "vertices": len(obj.data.vertices), "faces": len(obj.data.polygons)}
    finally:
        bm.free()


def _mesh_separate(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    mode = str(args["mode"]).upper()
    if mode not in {"SELECTED", "LOOSE"}:
        raise ExecutorError("separate mode must be SELECTED or LOOSE", "invalid_args")
    source_uuid = _stable_uuid(obj)
    before = {item.as_pointer() for item in bpy.context.scene.objects if item.type == "MESH"}
    prefix = str(args.get("name_prefix") or f"{obj.name}_part")
    try:
        if mode == "LOOSE":
            _with_edit_mesh(obj)
        else:
            _with_edit_mesh(obj, args.get("selection") or {})
            bm = bmesh.from_edit_mesh(obj.data)
            if not any(face.select for face in bm.faces):
                raise ExecutorError("separate selection contains no faces", "invalid_args")
        result = bpy.ops.mesh.separate(type=mode)
        if "FINISHED" not in set(result):
            raise ExecutorError("mesh separate did not finish", "execution_error")
    finally:
        _leave_edit_mesh()
    created = [item for item in bpy.context.scene.objects if item.type == "MESH" and item.as_pointer() not in before]
    for index, item in enumerate(sorted(created, key=lambda value: value.name)):
        # Blender copies IDProperties when separating; always assign a fresh
        # deterministic UUID so source and part cannot alias each other.
        item[_UUID_PROP] = "obj-" + content_hash({"source": source_uuid, "name": item.name, "index": index})[7:23]
        if len(created) > 1 or item.name != prefix:
            item.name = f"{prefix}_{index + 1:03d}"
    return {"target": source_uuid, "mode": mode, "objects": [{"uuid": _stable_uuid(item), "name": item.name, "vertices": len(item.data.vertices), "faces": len(item.data.polygons)} for item in sorted(created, key=lambda value: value.name)], "count": len(created)}


def _attribute_field(data_type: str) -> str:
    return {"FLOAT": "value", "INT": "value", "BOOLEAN": "value", "FLOAT_VECTOR": "vector", "FLOAT2": "vector", "INT2": "value", "INT32_2D": "value", "FLOAT_COLOR": "color", "BYTE_COLOR": "color"}.get(data_type, "value")


def _attribute_item_value(item: Any, field: str) -> Any:
    value = getattr(item, field)
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        return [round(float(part), 8) for part in value]
    except (TypeError, ValueError):
        return str(value)


def _mesh_attribute_write(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    name = str(args["name"])
    domain = str(args["domain"]).upper()
    data = args.get("data")
    if domain not in {"POINT", "EDGE", "FACE", "CORNER"}:
        raise ExecutorError("invalid attribute domain", "invalid_args")
    counts = {"POINT": len(obj.data.vertices), "EDGE": len(obj.data.edges), "FACE": len(obj.data.polygons), "CORNER": len(obj.data.loops)}
    count = counts[domain]
    data_type = str(args.get("data_type") or "").upper()
    if data_type == "INT2":
        # Blender 4.x/5.x exposes this logical type as INT32_2D.
        data_type = "INT32_2D"
    if isinstance(data, (bool, int, float)):
        rows = [data] * count
    elif isinstance(data, list):
        rows = data
        if len(rows) != count:
            raise ExecutorError(f"attribute data length {len(rows)} does not match {domain} count {count}", "invalid_args")
    else:
        raise ExecutorError("attribute data must be a scalar or array", "invalid_args")
    if not data_type:
        first = rows[0] if rows else 0.0
        if isinstance(first, bool):
            data_type = "BOOLEAN"
        elif isinstance(first, int) and not isinstance(first, bool):
            data_type = "INT"
        elif isinstance(first, (list, tuple)):
            data_type = {2: "FLOAT2", 3: "FLOAT_VECTOR", 4: "FLOAT_COLOR"}.get(len(first), "")
        else:
            data_type = "FLOAT"
    supported = {"FLOAT", "INT", "BOOLEAN", "FLOAT_VECTOR", "FLOAT2", "INT32_2D", "FLOAT_COLOR", "BYTE_COLOR"}
    if data_type not in supported:
        raise ExecutorError(f"unsupported attribute data type: {data_type}", "invalid_args")
    if name in obj.data.attributes and not bool(args.get("overwrite", False)):
        raise ExecutorError(f"attribute already exists: {name}", "conflict")
    existing = obj.data.attributes.get(name)
    if existing is not None and (existing.domain != domain or existing.data_type != data_type):
        if bool(args.get("overwrite", False)):
            obj.data.attributes.remove(existing)
            existing = None
        else:
            raise ExecutorError("existing attribute domain or type differs", "invalid_args")
    attr = existing or obj.data.attributes.new(name=name, type=data_type, domain=domain)
    field = _attribute_field(data_type)
    flat = []
    dimension = {"FLOAT_VECTOR": 3, "FLOAT2": 2, "INT32_2D": 2, "FLOAT_COLOR": 4, "BYTE_COLOR": 4}.get(data_type, 1)
    for row in rows:
        if dimension == 1:
            if isinstance(row, (list, tuple)):
                raise ExecutorError(f"scalar attribute {name} expects scalar values", "invalid_args")
            flat.append(bool(row) if data_type == "BOOLEAN" else (int(row) if data_type == "INT" else float(row)))
        else:
            if not isinstance(row, (list, tuple)) or len(row) != dimension:
                raise ExecutorError(f"attribute {name} expects {dimension}-component values", "invalid_args")
            flat.extend(float(value) if data_type not in {"INT32_2D"} else int(value) for value in row)
    try:
        attr.data.foreach_set(field, flat)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ExecutorError(f"failed to write attribute {name}: {exc}", "invalid_args") from exc
    obj.data.update()
    return {"target": _stable_uuid(obj), "name": name, "domain": domain, "data_type": data_type, "count": count, "hash": content_hash(rows)}


def _mesh_attribute_read(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    name = str(args["name"])
    attr = obj.data.attributes.get(name)
    if attr is None:
        raise ExecutorError(f"attribute not found: {name}", "not_found")
    requested_domain = args.get("domain")
    if requested_domain and str(requested_domain).upper() != attr.domain:
        raise ExecutorError(f"attribute {name} has domain {attr.domain}", "invalid_args")
    field = _attribute_field(attr.data_type)
    values = [_attribute_item_value(item, field) for item in attr.data]
    sample_limit = int(args.get("sample_limit", 64))
    result = {"target": _stable_uuid(obj), "name": name, "domain": attr.domain, "data_type": attr.data_type, "count": len(values), "hash": content_hash(values), "sample": values[:sample_limit]}
    if bool(args.get("include_data", False)):
        if len(values) > 4096:
            raise ExecutorError("include_data is limited to 4096 values; request a sample instead", "invalid_args")
        result["data"] = values
    return result


def _mesh_geometry_query(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    field = str(args["field"])
    space = str(args.get("space", "LOCAL")).upper()
    if space not in {"LOCAL", "WORLD"}:
        raise ExecutorError("space must be LOCAL or WORLD", "invalid_args")
    verts = [vertex.co.copy() for vertex in obj.data.vertices]
    transform_point = (lambda value: obj.matrix_world @ value) if space == "WORLD" else (lambda value: value)
    transform_normal = obj.matrix_world.to_3x3().inverted().transposed() if space == "WORLD" else None
    values: list[Any]
    if field == "vertex_positions":
        values = [[round(float(v), 8) for v in transform_point(co)] for co in verts]
    elif field == "edge_indices":
        values = [[int(edge.vertices[0]), int(edge.vertices[1])] for edge in obj.data.edges]
    elif field in {"edge_centers", "edge_lengths", "edge_directions"}:
        values = []
        for edge in obj.data.edges:
            a, b = verts[int(edge.vertices[0])], verts[int(edge.vertices[1])]
            if field == "edge_centers":
                values.append([round(float(v), 8) for v in transform_point((a + b) * 0.5)])
            elif field == "edge_lengths":
                left, right = transform_point(a), transform_point(b)
                values.append(round(float((left - right).length), 8))
            else:
                direction = (b - a).normalized() if (b - a).length > 1e-12 else Vector((0.0, 0.0, 0.0))
                if transform_normal is not None:
                    direction = (obj.matrix_world.to_3x3() @ direction).normalized()
                values.append([round(float(v), 8) for v in direction])
    elif field in {"polygon_centers", "polygon_normals", "polygon_areas", "polygon_vertex_indices"}:
        values = []
        for polygon in obj.data.polygons:
            if field == "polygon_centers":
                values.append([round(float(v), 8) for v in transform_point(polygon.center)])
            elif field == "polygon_normals":
                normal = polygon.normal.copy()
                if transform_normal is not None:
                    normal = (transform_normal @ normal).normalized()
                values.append([round(float(v), 8) for v in normal])
            elif field == "polygon_areas":
                if space == "WORLD":
                    points = [transform_point(verts[int(index)]) for index in polygon.vertices]
                    area = 0.0
                    if len(points) >= 3:
                        origin = points[0]
                        for index in range(1, len(points) - 1):
                            area += 0.5 * (points[index] - origin).cross(points[index + 1] - origin).length
                    values.append(round(float(area), 8))
                else:
                    values.append(round(float(polygon.area), 8))
            else:
                values.append([int(index) for index in polygon.vertices])
    elif field == "loop_vertex_indices":
        values = [int(loop.vertex_index) for loop in obj.data.loops]
    elif field == "bbox":
        if space == "WORLD":
            return {"target": _stable_uuid(obj), "field": field, "space": space, "value": _aabb(obj)}
        coords = [list(map(float, vertex.co)) for vertex in obj.data.vertices]
        if not coords:
            value = {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
        else:
            value = {"min": [round(min(row[i] for row in coords), 8) for i in range(3)], "max": [round(max(row[i] for row in coords), 8) for i in range(3)]}
        return {"target": _stable_uuid(obj), "field": field, "space": space, "value": value}
    else:
        raise ExecutorError(f"unsupported geometry field: {field}", "invalid_args")
    sample_limit = int(args.get("sample_limit", 64))
    return {"target": _stable_uuid(obj), "field": field, "space": space, "count": len(values), "hash": content_hash(values), "sample": values[:sample_limit]}


def _mesh_subdivide(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        _, edges, _ = _selection_parts(bm, args.get("selection") or {}, obj=obj)
        if not edges:
            raise ExecutorError("selection contains no edges", "invalid_args")
        cuts = int(args.get("cuts", 1))
        result = bmesh.ops.subdivide_edges(bm, edges=edges, cuts=cuts, use_grid_fill=True)
        smooth = float(args.get("smooth", 0.0))
        if smooth > 0:
            verts = [item for item in result.get("geom_split", []) if isinstance(item, bmesh.types.BMVert)]
            if verts:
                bmesh.ops.smooth_vert(bm, verts=verts, factor=smooth, use_axis_x=True, use_axis_y=True, use_axis_z=True)
        _write_bmesh(obj, bm)
        return {"target": _stable_uuid(obj), "cuts": cuts, "vertices": len(obj.data.vertices), "faces": len(obj.data.polygons), "region_handles": _region_handles(obj)}
    finally:
        bm.free()


def _mesh_subdivide_adaptive(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Refine long selected edges in bounded passes toward a target length."""
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    target_length = float(args["target_edge_length"])
    max_passes = int(args.get("max_passes", 4))
    max_cuts = int(args.get("max_cuts", 4))
    selection = args.get("selection") or {}
    numeric_selection = any(selection.get(key) for key in ("vertex_indices", "edge_indices", "face_indices"))
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bm.edges.ensure_lookup_table()
        initial_edges = len(bm.edges)
        initial_max = max((edge.calc_length() for edge in bm.edges), default=0.0)
        passes = 0
        split_edges = 0
        for _ in range(max_passes):
            _, edges, _ = _selection_parts(bm, selection, obj=obj, default_all=True)
            candidates = [edge for edge in edges if edge.calc_length() > target_length * (1.0 + 1e-6)]
            if not candidates:
                break
            longest = max(edge.calc_length() for edge in candidates)
            cuts = max(1, min(max_cuts, int(math.ceil(longest / target_length)) - 1))
            result = bmesh.ops.subdivide_edges(bm, edges=candidates, cuts=cuts, use_grid_fill=True)
            split_edges += len(candidates)
            passes += 1
            if args.get("smooth", 0.0) > 0:
                new_verts = [item for item in result.get("geom_split", []) if isinstance(item, bmesh.types.BMVert)]
                if new_verts:
                    bmesh.ops.smooth_vert(bm, verts=new_verts, factor=float(args["smooth"]), use_axis_x=True, use_axis_y=True, use_axis_z=True)
            # Numeric indices are tied to the old datablock and cannot safely
            # be re-used after the first topology-changing pass.
            if numeric_selection:
                break
        _write_bmesh(obj, bm)
        bm.edges.ensure_lookup_table()
        _, selected_after, _ = _selection_parts(bm, selection if not numeric_selection else {}, obj=obj, default_all=True)
        final_max = max((edge.calc_length() for edge in selected_after), default=0.0)
        return {
            "target": _stable_uuid(obj), "target_edge_length": target_length,
            "passes": passes, "edges_split": split_edges, "edges_before": initial_edges,
            "edges_after": len(obj.data.edges), "max_edge_before": round(initial_max, 8),
            "max_edge_after": round(final_max, 8), "numeric_selection_limited": numeric_selection, "region_handles": _region_handles(obj),
        }
    finally:
        bm.free()


def _mesh_transform_selection(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        verts, _, _ = _selection_parts(bm, args.get("selection") or {}, obj=obj)
        if not verts:
            raise ExecutorError("selection contains no vertices", "invalid_args")
        pivot = Vector(_as_float3(args.get("pivot", (0, 0, 0)), "pivot"))
        translation = Vector(_as_float3(args.get("translation", (0, 0, 0)), "translation"))
        scale = _as_float3(args.get("scale", (1, 1, 1)), "scale")
        rotation = Euler(_as_float3(args.get("rotation_euler", (0, 0, 0)), "rotation_euler"), "XYZ").to_matrix()
        for vertex in verts:
            local = vertex.co - pivot
            local = rotation @ Vector((local.x * scale[0], local.y * scale[1], local.z * scale[2]))
            vertex.co = pivot + local + translation
        _write_bmesh(obj, bm)
        return {"target": _stable_uuid(obj), "vertices_affected": len(verts)}
    finally:
        bm.free()


def _mesh_extrude_region(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        _, _, faces = _selection_parts(bm, args.get("selection") or {}, obj=obj, default_all=False)
        if not faces:
            raise ExecutorError("extrude_region requires selected faces", "invalid_args")
        result = bmesh.ops.extrude_face_region(bm, geom=faces)
        new_verts = [item for item in result.get("geom", []) if isinstance(item, bmesh.types.BMVert)]
        offset_value = args.get("offset")
        if offset_value is not None:
            offset = Vector(_as_float3(offset_value, "offset"))
        else:
            normal = sum((face.normal for face in faces), Vector((0, 0, 0)))
            normal.normalize()
            offset = normal * float(args.get("distance", 0.0))
        pivot = sum((vertex.co for vertex in new_verts), Vector((0, 0, 0))) / max(1, len(new_verts))
        factor = float(args.get("scale", 1.0))
        for vertex in new_verts:
            vertex.co = pivot + (vertex.co - pivot) * factor + offset
        _write_bmesh(obj, bm)
        return {"target": _stable_uuid(obj), "faces_extruded": len(faces), "vertices_created": len(new_verts)}
    finally:
        bm.free()


def _mesh_inset_region(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        _, _, faces = _selection_parts(bm, args.get("selection") or {}, obj=obj, default_all=False)
        if not faces:
            raise ExecutorError("inset_region requires selected faces", "invalid_args")
        result = bmesh.ops.inset_region(
            bm, faces=faces, thickness=float(args.get("thickness", 0.01)),
            depth=float(args.get("depth", 0.0)), use_boundary=True, use_even_offset=True,
        )
        _write_bmesh(obj, bm)
        return {"target": _stable_uuid(obj), "faces_inset": len(faces), "faces_created": len(result.get("faces", []))}
    finally:
        bm.free()


def _mesh_extrude_individual(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Extrude each selected face separately, preserving semantic intent."""
    # Blender's bmesh discrete extrusion API is version-sensitive; the
    # region operation remains deterministic and is a safe fallback for
    # callers that request the individual variant.
    result = _mesh_extrude_region(args)
    result["mode"] = "individual_compat"
    return result


def _mesh_inset_individual(args: Mapping[str, Any]) -> Dict[str, Any]:
    result = _mesh_inset_region(args)
    result["mode"] = "individual_compat"
    return result


def _mesh_bridge_edge_loops(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        _, edges, _ = _selection_parts(bm, args.get("selection") or {}, obj=obj, default_all=False)
        if len(edges) < 2:
            raise ExecutorError("bridge_edge_loops requires at least two selected edges", "invalid_args")
        result = bmesh.ops.bridge_loops(bm, edges=edges)
        _write_bmesh(obj, bm)
        return {"target": _stable_uuid(obj), "edges_selected": len(edges), "faces_created": len(result.get("faces", [])) if isinstance(result, Mapping) else 0}
    finally:
        bm.free()


def _mesh_loop_cut(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    cuts = max(1, min(32, int(args.get("cuts", 1))))
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        _, edges, _ = _selection_parts(bm, args.get("selection") or {}, obj=obj, default_all=True)
        if not edges:
            raise ExecutorError("loop_cut selection is empty", "invalid_args")
        result = bmesh.ops.subdivide_edges(bm, edges=edges, cuts=cuts, use_grid_fill=True)
        _write_bmesh(obj, bm)
        return {"target": _stable_uuid(obj), "cuts": cuts, "edges_selected": len(edges), "geometry_created": len(result.get("geom_split", []))}
    finally:
        bm.free()


def _mesh_duplicate_region(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Duplicate a selected region as a separate object when Blender permits."""
    _require_bpy()
    source = _require_mesh_object(args["target"])
    bm = bmesh.new()
    try:
        bm.from_mesh(source.data)
        _, _, faces = _selection_parts(bm, args.get("selection") or {}, obj=source, default_all=False)
        if not faces:
            raise ExecutorError("duplicate_region requires selected faces", "invalid_args")
        selected_faces = set(faces)
        selected_verts = {vert for face in selected_faces for vert in face.verts}
        new_mesh = bpy.data.meshes.new(str(args.get("name") or f"{source.name}_region"))
        new_bm = bmesh.new()
        vert_map = {vert: new_bm.verts.new(vert.co.copy()) for vert in selected_verts}
        new_bm.verts.ensure_lookup_table()
        for face in faces:
            try:
                new_bm.faces.new([vert_map[vert] for vert in face.verts])
            except ValueError:
                pass
        new_bm.to_mesh(new_mesh)
        new_bm.free()
        new_mesh.update()
        obj = bpy.data.objects.new(str(args.get("name") or f"{source.name}_region"), new_mesh)
        bpy.context.scene.collection.objects.link(obj)
        obj[_UUID_PROP] = _stable_uuid(obj)
        obj[_SEMANTIC_PROP] = list(_semantic_tags(source))
        _store_json_prop(obj, _COORDINATE_PROP, _load_json_prop(source, _COORDINATE_PROP, {"space": "WORLD", "units": _coordinate_system().get("units", "meters")}))
        return {"target": _stable_uuid(source), "region": _stable_uuid(obj), "name": obj.name, "faces_duplicated": len(faces), "vertices_duplicated": len(selected_verts)}
    finally:
        bm.free()


def _mesh_bevel(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        _, edges, _ = _selection_parts(bm, args.get("selection") or {}, obj=obj, default_all=False)
        if not edges:
            raise ExecutorError("bevel requires selected edges", "invalid_args")
        bmesh.ops.bevel(
            bm, geom=edges, offset=float(args.get("width", 0.01)),
            segments=int(args.get("segments", 2)), profile=float(args.get("profile", 0.5)), affect="EDGES",
        )
        _write_bmesh(obj, bm)
        return {"target": _stable_uuid(obj), "edges_beveled": len(edges)}
    finally:
        bm.free()


def _mesh_merge_by_distance(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        verts, _, _ = _selection_parts(bm, args.get("selection") or {}, obj=obj, default_all=True)
        if not verts:
            raise ExecutorError("merge_by_distance requires selected vertices", "invalid_args")
        result = bmesh.ops.remove_doubles(bm, verts=verts, dist=float(args.get("distance", 0.0001)))
        _write_bmesh(obj, bm)
        return {"target": _stable_uuid(obj), "vertices_merged": len(result.get("targetmap", {})) if isinstance(result, Mapping) else 0}
    finally:
        bm.free()


def _mesh_recalculate_normals(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        _, _, faces = _selection_parts(bm, args.get("selection") or {}, obj=obj, default_all=True)
        bmesh.ops.recalc_face_normals(bm, faces=faces)
        if args.get("inside", False):
            bmesh.ops.reverse_faces(bm, faces=faces)
        _write_bmesh(obj, bm)
        return {"target": _stable_uuid(obj), "faces": len(faces), "inside": bool(args.get("inside", False))}
    finally:
        bm.free()


def _mesh_delete_region(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        verts, edges, faces = _selection_parts(bm, args.get("selection") or {}, obj=obj, default_all=False)
        mode = str(args["mode"]).upper()
        if mode == "VERTS":
            geom = verts
        elif mode == "EDGES":
            geom = edges
        elif mode in {"FACES", "ONLY_FACE"}:
            geom = faces
        else:
            raise ExecutorError(f"unsupported delete mode: {mode}", "invalid_args")
        if not geom:
            raise ExecutorError("delete_region selection is empty", "invalid_args")
        bmesh.ops.delete(bm, geom=geom, context=mode)
        _write_bmesh(obj, bm)
        return {"target": _stable_uuid(obj), "mode": mode, "deleted": len(geom)}
    finally:
        bm.free()


def _mesh_dissolve_region(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        verts, edges, faces = _selection_parts(bm, args.get("selection") or {}, obj=obj, default_all=False)
        mode = str(args["mode"]).upper()
        if mode == "VERTS":
            geom = verts
            op = bmesh.ops.dissolve_verts
            geom_key = "verts"
        elif mode == "EDGES":
            geom = edges
            op = bmesh.ops.dissolve_edges
            geom_key = "edges"
        elif mode == "FACES":
            geom = faces
            op = bmesh.ops.dissolve_faces
            geom_key = "faces"
        else:
            raise ExecutorError(f"unsupported dissolve mode: {mode}", "invalid_args")
        if not geom:
            raise ExecutorError("dissolve_region selection is empty", "invalid_args")
        kwargs: Dict[str, Any] = {"use_face_split": bool(args.get("use_face_split", False))}
        op(bm, **{geom_key: geom}, **kwargs)
        _write_bmesh(obj, bm)
        return {"target": _stable_uuid(obj), "mode": mode, "dissolved": len(geom)}
    finally:
        bm.free()


def _mesh_fill_holes(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        _, edges, _ = _selection_parts(bm, args.get("selection") or {}, obj=obj, default_all=False)
        boundary = [edge for edge in (edges or list(bm.edges)) if edge.is_boundary]
        if not boundary:
            raise ExecutorError("fill_holes found no boundary edges", "precondition_failed")
        result = bmesh.ops.holes_fill(bm, edges=boundary, sides=int(args.get("sides", 0)))
        _write_bmesh(obj, bm)
        return {"target": _stable_uuid(obj), "boundary_edges": len(boundary), "faces_created": len(result.get("geom", [])) if isinstance(result, Mapping) else 0}
    finally:
        bm.free()


def _mesh_cut_plane(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Bisect a mesh with an explicitly framed plane and optionally cap it."""
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    # ``point``/``normal`` are canonical.  Keep the old aliases readable for
    # replay compatibility, but never silently accept a missing plane.
    point_value = args.get("point", args.get("plane_co"))
    normal_value = args.get("normal", args.get("plane_no"))
    if point_value is None or normal_value is None:
        raise ExecutorError("mesh.cut_plane requires point and normal", "invalid_args")
    frame = _coordinate_frame(args)
    point = _point_to_world_for_object(obj, point_value, frame, "point")
    normal = _direction_to_world(obj, normal_value, frame, "normal")
    plane_co = obj.matrix_world.inverted() @ point
    plane_no = obj.matrix_world.inverted().to_3x3() @ normal
    if plane_no.length < 1e-9:
        raise ExecutorError("normal must be non-zero", "invalid_args")
    plane_no.normalize()
    side = str(args.get("side", "BOTH")).upper()
    tolerance = _length_value(args.get("distance", 1e-5), "distance", frame)
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
        result = bmesh.ops.bisect_plane(
            bm, geom=geom, plane_co=plane_co, plane_no=plane_no,
            dist=tolerance, use_snap_center=False, clear_outer=False, clear_inner=False,
        )
        cut_edges = [item for item in result.get("geom_cut", []) if isinstance(item, bmesh.types.BMEdge)]
        if side not in {"BOTH", "POSITIVE", "NEGATIVE"}:
            raise ExecutorError(f"unsupported cut side: {side}", "invalid_args")
        deleted = 0
        if side != "BOTH":
            keep_positive = side == "POSITIVE"
            to_delete = []
            for vertex in bm.verts:
                signed = (vertex.co - plane_co).dot(plane_no)
                if (keep_positive and signed < -tolerance) or ((not keep_positive) and signed > tolerance):
                    to_delete.append(vertex)
            deleted = len(to_delete)
            if to_delete:
                bmesh.ops.delete(bm, geom=to_delete, context="VERTS")
        capped = 0
        if bool(args.get("cap", True)) and side != "BOTH":
            boundary = [edge for edge in bm.edges if edge.is_boundary and all(abs((v.co - plane_co).dot(plane_no)) <= tolerance * 4.0 for v in edge.verts)]
            if boundary:
                fill = bmesh.ops.holes_fill(bm, edges=boundary, sides=0)
                capped = len(fill.get("geom", [])) if isinstance(fill, Mapping) else 0
        _write_bmesh(obj, bm)
        if args.get("semantic_tags"):
            obj[_SEMANTIC_PROP] = sorted(set(_semantic_tags(obj)) | set(str(tag) for tag in args["semantic_tags"]))
        topology = _topology(obj)
        return {
            "target": _stable_uuid(obj), "side": side, "cut_edges": len(cut_edges),
            "deleted_vertices": deleted, "faces_capped": capped, "topology": topology,
        }
    finally:
        bm.free()


def _mesh_cut_curve(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Subtract or combine a deterministic beveled polyline cutter."""
    _require_bpy()
    target = _require_mesh_object(args["target"])
    points = [Vector(_as_float3(point, "points[]")) for point in args.get("points", [])]
    if len(points) < 2:
        raise ExecutorError("mesh.cut_curve requires at least two points", "invalid_args")
    width = float(args.get("width", 0.01))
    depth = float(args.get("depth", max(width, 0.02)))
    curve_data = bpy.data.curves.new("ToolboxCutCurve", "CURVE")
    curve_data.dimensions = "3D"
    curve_data.resolution_u = 1
    curve_data.bevel_depth = width * 0.5
    curve_data.bevel_resolution = 1
    curve_data.extrude = depth * 0.5
    curve_data.use_fill_caps = True
    spline = curve_data.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for item, point in zip(spline.points, points):
        item.co = (point.x, point.y, point.z, 1.0)
    cutter_curve = bpy.data.objects.new("ToolboxCutCurve", curve_data)
    bpy.context.scene.collection.objects.link(cutter_curve)
    cutter_curve.matrix_world = target.matrix_world.copy()
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = cutter_curve.evaluated_get(depsgraph)
        cutter_mesh = bpy.data.meshes.new_from_object(evaluated, depsgraph=depsgraph)
        cutter = bpy.data.objects.new("ToolboxCutCurveMesh", cutter_mesh)
        bpy.context.scene.collection.objects.link(cutter)
        cutter.matrix_world = target.matrix_world.copy()
        operation = str(args.get("operation", "DIFFERENCE")).upper()
        result = _boolean({"target": _stable_uuid(target), "cutter": _stable_uuid(cutter), "operation": operation, "delete_cutter": True})
        if args.get("semantic_tags"):
            target[_SEMANTIC_PROP] = sorted(set(_semantic_tags(target)) | set(str(tag) for tag in args["semantic_tags"]))
        return {**result, "width": width, "depth": depth, "points": len(points), "topology": _topology(target)}
    finally:
        if cutter_curve.name in bpy.data.objects:
            bpy.data.objects.remove(cutter_curve, do_unlink=True)


def _mesh_repair(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Run bounded mesh cleanup operations and return post-repair topology."""
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        merged = 0
        if "merge_distance" in args:
            result = bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=float(args["merge_distance"]))
            merged = len(result.get("targetmap", {})) if isinstance(result, Mapping) else 0
        if bool(args.get("recalculate_normals", True)):
            bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        filled = 0
        if bool(args.get("fill_holes", False)):
            boundary = [edge for edge in bm.edges if edge.is_boundary]
            if boundary:
                result = bmesh.ops.holes_fill(bm, edges=boundary, sides=0)
                filled = len(result.get("geom", [])) if isinstance(result, Mapping) else 0
        _write_bmesh(obj, bm)
        return {"target": _stable_uuid(obj), "vertices_merged": merged, "faces_filled": filled, "topology": _topology(obj)}
    finally:
        bm.free()


def _mesh_triangulate(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        _, _, faces = _selection_parts(bm, args.get("selection") or {}, obj=obj, default_all=True)
        result = bmesh.ops.triangulate(bm, faces=faces, quad_method="BEAUTY", ngon_method="BEAUTY")
        _write_bmesh(obj, bm)
        return {"target": _stable_uuid(obj), "faces_input": len(faces), "faces_created": len(result.get("faces", []))}
    finally:
        bm.free()


def _mesh_shade_smooth(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    smooth = bool(args.get("smooth", True))
    for polygon in obj.data.polygons:
        polygon.use_smooth = smooth
    obj.data.update()
    return {"target": _stable_uuid(obj), "smooth": smooth}


def _mesh_vertex_group_assign(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        verts, _, _ = _selection_parts(bm, args.get("selection") or {}, obj=obj, default_all=False)
        if not verts:
            raise ExecutorError("vertex_group_assign requires selected vertices", "invalid_args")
        indices = [int(vertex.index) for vertex in verts]
    finally:
        bm.free()
    group = obj.vertex_groups.get(str(args["name"])) or obj.vertex_groups.new(name=str(args["name"]))
    group.add(indices, float(args.get("weight", 1.0)), str(args.get("mode", "REPLACE")))
    handles = set(str(value) for value in (obj.get(_REGION_PROP, []) or []))
    handles.add(group.name)
    obj[_REGION_PROP] = sorted(handles)
    return {"target": _stable_uuid(obj), "name": group.name, "region_handle": f"region:{group.name}", "vertices_assigned": len(indices), "weight": float(args.get("weight", 1.0))}


def _mesh_region_define(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Define a reusable region as a named vertex group handle."""
    payload = dict(args)
    payload["name"] = str(args["name"])
    result = _mesh_vertex_group_assign(payload)
    return {**result, "region_handle": f"region:{payload['name']}"}


def _sculpt_points(args: Mapping[str, Any]) -> list[Vector]:
    return [Vector(_as_float3(point, "points[]")) for point in args.get("points", [])]




def _distance_to_segment(point: Vector, start: Vector, end: Vector) -> tuple[float, float]:
    delta = end - start
    denominator = delta.length_squared
    if denominator < 1e-12:
        return (point - start).length, 0.0
    t = max(0.0, min(1.0, (point - start).dot(delta) / denominator))
    return (point - (start + delta * t)).length, t


def _distance_to_polyline(point: Vector, points: list[Vector]) -> tuple[float, int, float]:
    best = (float("inf"), 0, 0.0)
    for index in range(len(points) - 1):
        distance, t = _distance_to_segment(point, points[index], points[index + 1])
        if distance < best[0]:
            best = (distance, index, t)
    return best






def _sculpt_ridge(args: Mapping[str, Any]) -> Dict[str, Any]:
    return _sculpt_path_relief(args, mode="ridge", amplitude=float(args["height"]))


def _sculpt_groove(args: Mapping[str, Any]) -> Dict[str, Any]:
    return _sculpt_path_relief(args, mode="groove", amplitude=-abs(float(args["depth"])))


def _sculpt_muscle(args: Mapping[str, Any]) -> Dict[str, Any]:
    return _sculpt_path_relief(args, mode="muscle", amplitude=abs(float(args["height"])))




def _animation_keyframe_shape_key(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    target = _require_mesh_object(args["target"])
    shape_keys = target.data.shape_keys
    if shape_keys is None or shape_keys.key_blocks.get(str(args["shape_key"])) is None:
        raise ExecutorError(f"shape key not found: {args['shape_key']}", "not_found")
    key = shape_keys.key_blocks[str(args["shape_key"])]
    frame = int(args["frame"])
    bpy.context.scene.frame_set(frame)
    key.value = float(args["value"])
    key.keyframe_insert(data_path="value", frame=frame)
    interpolation = str(args.get("interpolation", "BEZIER"))
    action = shape_keys.animation_data.action if shape_keys.animation_data and shape_keys.animation_data.action else None
    if action:
        for curve in _action_fcurves(action):
            for point in curve.keyframe_points:
                point.interpolation = interpolation
    return {"target": _stable_uuid(target), "shape_key": key.name, "frame": frame, "value": key.value, "action": action.name if action else None}




def _geometry_remesh_voxel(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    if any(item.type == "MULTIRES" for item in obj.modifiers):
        raise ExecutorError("voxel remesh requires applying or removing Multires first", "precondition_failed")
    voxel_size = float(args["voxel_size"])
    previous_handles = [str(value) for value in (obj.get(_REGION_PROP, []) or [])]
    obj.data.remesh_voxel_size = voxel_size
    obj.data.remesh_voxel_adaptivity = float(args.get("adaptivity", 0.0))
    bpy.context.view_layer.objects.active = obj
    for item in bpy.context.selected_objects:
        item.select_set(False)
    obj.select_set(True)
    try:
        bpy.ops.object.voxel_remesh()
    except Exception as exc:
        raise ExecutorError(f"voxel remesh failed: {exc}", "execution_error") from exc
    if args.get("smooth_shading", True):
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
    surviving = {group.name for group in obj.vertex_groups}
    invalidated = [name for name in previous_handles if name not in surviving]
    if invalidated:
        obj[_REGION_PROP] = sorted(surviving & set(previous_handles))
    return {"target": _stable_uuid(obj), "voxel_size": voxel_size, "vertices": len(obj.data.vertices), "faces": len(obj.data.polygons), "invalidated_region_handles": [f"region:{name}" for name in invalidated]}


def _geometry_shrinkwrap(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    target = _require_mesh_object(args["target"])
    surface = _object_by_ref(args["surface"])
    if target == surface:
        raise ExecutorError("shrinkwrap target and surface must differ", "invalid_args")
    modifier = target.modifiers.new("ToolboxShrinkwrap", "SHRINKWRAP")
    modifier.target = surface
    modifier.wrap_method = str(args.get("method", "NEAREST_SURFACEPOINT"))
    modifier.offset = float(args.get("offset", 0.0))
    axis = str(args.get("axis", "POS_Z"))
    if modifier.wrap_method == "PROJECT":
        modifier.use_project_x = axis in {"POS_X", "NEG_X"}
        modifier.use_project_y = axis in {"POS_Y", "NEG_Y"}
        modifier.use_project_z = axis in {"POS_Z", "NEG_Z"}
        modifier.use_negative_direction = axis.startswith("NEG_")
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    try:
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    finally:
        target.select_set(False)
    return {"target": _stable_uuid(target), "surface": _stable_uuid(surface), "method": modifier.wrap_method, "offset": modifier.offset}


def _inspect_mesh_region(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        verts, edges, faces = _selection_parts(bm, args.get("selection") or {}, obj=obj, default_all=False)
        if verts:
            coords = [vertex.co for vertex in verts]
            bounds = {
                "min": [round(float(min(coordinate[i] for coordinate in coords)), 8) for i in range(3)],
                "max": [round(float(max(coordinate[i] for coordinate in coords)), 8) for i in range(3)],
            }
        else:
            bounds = None
        return {"target": _stable_uuid(obj), "vertices": len(verts), "edges": len(edges), "faces": len(faces), "aabb_local": bounds, "detail": _region_detail_stats(obj, args.get("selection") or {})}
    finally:
        bm.free()


def _material_create(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    name = str(args["name"])
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    color = args.get("base_color", (0.8, 0.8, 0.8, 1.0))
    if principled and isinstance(color, (list, tuple)) and len(color) in {3, 4}:
        principled.inputs["Base Color"].default_value = tuple(float(v) for v in (color if len(color) == 4 else (*color, 1.0)))
    if principled and "roughness" in args:
        principled.inputs["Roughness"].default_value = float(args["roughness"])
    if principled and "metallic" in args:
        principled.inputs["Metallic"].default_value = float(args["metallic"])
    return {"name": material.name}


def _material_assign_batch(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Assign materials with a single preflight and optional rollback."""
    _require_bpy()
    assignments = args.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ExecutorError("assignments must contain at least one item", "invalid_args")
    if len(assignments) > 512:
        raise ExecutorError("assignments exceeds maximum item count 512", "invalid_args")
    atomic = bool(args.get("atomic", True))
    resolved: list[tuple[Any, Any, str]] = []
    for index, raw in enumerate(assignments):
        if not isinstance(raw, Mapping):
            raise ExecutorError(f"assignments[{index}] must be an object", "invalid_args")
        target = _object_by_ref(raw.get("target"))
        material_name = str(raw.get("material", ""))
        material = bpy.data.materials.get(material_name)
        if material is None:
            raise ExecutorError(f"material not found: {material_name}", "not_found")
        if getattr(target, "data", None) is None or not hasattr(target.data, "materials"):
            raise ExecutorError(f"target has no material slots: {raw.get('target')}", "invalid_args")
        resolved.append((target, material, str(raw.get("target"))))
    snapshots = {
        id(target): [slot.material for slot in target.material_slots]
        for target, _, _ in resolved
    }
    results: list[Dict[str, Any]] = []
    try:
        for index, (target, material, ref) in enumerate(resolved):
            if len(target.data.materials) == 0:
                target.data.materials.append(material)
            else:
                target.data.materials[0] = material
            results.append({"index": index, "target": _stable_uuid(target), "name": target.name, "material": material.name, "ok": True})
    except Exception as exc:
        if atomic:
            for target, _, _ in resolved:
                if getattr(target, "data", None) is None or not hasattr(target.data, "materials"):
                    continue
                for slot_index in range(len(target.data.materials) - 1, -1, -1):
                    target.data.materials.pop(index=slot_index)
                for material in snapshots[id(target)]:
                    target.data.materials.append(material)
        code = getattr(exc, "code", "execution_error")
        raise ExecutorError(f"material assignment batch failed: {exc}", code) from exc
    return {
        "count": len(results),
        "successful": len(results),
        "failed": 0,
        "atomic": atomic,
        "committed": True,
        "assignments": results,
    }


def _apply_modifier(args: Mapping[str, Any], apply: bool = False) -> Dict[str, Any]:
    _require_bpy()
    obj = _object_by_ref(args["target"])
    modifier_name = args.get("modifier")
    if apply:
        if not modifier_name:
            raise ExecutorError("modifier is required", "invalid_args")
        modifier = obj.modifiers.get(str(modifier_name))
        if modifier is None:
            raise ExecutorError(f"modifier not found: {modifier_name}", "not_found")
        applied_name = modifier.name
        _activate_object(obj)
        try:
            result = bpy.ops.object.modifier_apply(modifier=modifier.name)
        except Exception as exc:
            obj.select_set(False)
            raise ExecutorError(f"modifier apply failed: {exc}", "execution_error") from exc
        obj.select_set(False)
        if "FINISHED" not in set(result) or obj.modifiers.get(str(modifier_name)) is not None:
            raise ExecutorError("modifier apply did not remove the modifier", "execution_error")
        return {"applied": applied_name}
    modifier_type = str(args["modifier_type"]).upper()
    allowed = {"BEVEL", "SUBSURF", "DECIMATE", "SOLIDIFY", "ARRAY", "MIRROR", "REMESH", "SMOOTH", "DISPLACE", "LATTICE", "SHRINKWRAP", "CAST", "SIMPLE_DEFORM", "CORRECTIVE_SMOOTH", "WEIGHTED_NORMAL", "WIREFRAME", "SKIN", "SCREW", "WELD", "EDGE_SPLIT", "CURVE", "LAPLACIANSMOOTH"}
    if modifier_type not in allowed:
        raise ExecutorError(f"unsupported modifier type: {modifier_type}", "invalid_args")
    requested_name = args.get("name")
    modifier_name = str(requested_name).strip() if requested_name is not None and str(requested_name).strip() else modifier_type.title()
    modifier = obj.modifiers.new(modifier_name, modifier_type)
    properties = args.get("properties", {})
    if not isinstance(properties, Mapping):
        raise ExecutorError("modifier properties must be an object", "invalid_args")
    for key, value in properties.items():
        key = str(key)
        if key.startswith("id:"):
            try:
                modifier[key[3:]] = value
            except (TypeError, ValueError) as exc:
                raise ExecutorError(f"invalid modifier custom property: {key}", "invalid_args") from exc
            continue
        if key in {"object", "target", "origin"} and isinstance(value, str):
            value = _object_by_ref(value)
        if key == "collection" and isinstance(value, str):
            value = _collection_by_ref(value)
        if not hasattr(modifier, key):
            raise ExecutorError(f"unknown modifier property: {key}", "invalid_args")
        try:
            setattr(modifier, key, value)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ExecutorError(f"invalid modifier property: {key}", "invalid_args") from exc
    return {"name": modifier.name, "type": modifier.type, "properties": dict(properties), "property_hash": content_hash(dict(properties))}


def _geometry_modifier_stack(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Build an ordered modifier stack without one round trip per modifier."""
    _require_bpy()
    target = _object_by_ref(args.get("target"))
    modifiers = args.get("modifiers")
    if not isinstance(modifiers, list) or not modifiers:
        raise ExecutorError("modifiers must contain at least one item", "invalid_args")
    if len(modifiers) > 64:
        raise ExecutorError("modifiers exceeds maximum item count 64", "invalid_args")
    default_apply = bool(args.get("apply", False))
    atomic = bool(args.get("atomic", True))
    original_data = getattr(target, "data", None).copy() if atomic and getattr(target, "data", None) is not None else None
    original_modifier_names = {modifier.name for modifier in target.modifiers}
    results: list[Dict[str, Any]] = []
    try:
        for index, raw in enumerate(modifiers):
            if not isinstance(raw, Mapping):
                raise ExecutorError(f"modifiers[{index}] must be an object", "invalid_args")
            add_args = {
                "target": _stable_uuid(target),
                "modifier_type": raw.get("modifier_type"),
                "properties": raw.get("properties") or {},
            }
            if raw.get("name") is not None:
                add_args["name"] = raw.get("name")
            added = _apply_modifier(add_args)
            entry: Dict[str, Any] = {"index": index, "added": added, "ok": True}
            if bool(raw.get("apply", default_apply)):
                applied = _apply_modifier({"target": _stable_uuid(target), "modifier": added["name"]}, apply=True)
                entry["applied"] = applied
            results.append(entry)
    except Exception as exc:
        if atomic:
            # Applying a modifier mutates the mesh datablock.  Restore the
            # captured copy and remove only modifiers created by this action.
            if original_data is not None:
                current_data = target.data
                target.data = original_data
                if current_data is not original_data:
                    _remove_orphan_data(current_data)
            for modifier in list(target.modifiers):
                if modifier.name not in original_modifier_names:
                    target.modifiers.remove(modifier)
        code = getattr(exc, "code", "execution_error")
        raise ExecutorError(f"modifier stack failed: {exc}", code) from exc
    finally:
        if original_data is not None and target.data is not original_data:
            _remove_orphan_data(original_data)
    return {
        "target": _stable_uuid(target),
        "count": len(results),
        "successful": len(results),
        "failed": 0,
        "apply": default_apply,
        "atomic": atomic,
        "committed": True,
        "modifiers": results,
    }


def _boolean(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    target = _require_mesh_object(args["target"])
    cutter = _require_mesh_object(args["cutter"])
    operation = str(args["operation"]).upper()
    if operation not in {"UNION", "DIFFERENCE", "INTERSECT"}:
        raise ExecutorError("operation must be UNION, DIFFERENCE, or INTERSECT", "invalid_args")
    _activate_object(target)
    modifier = target.modifiers.new("ToolboxBoolean", "BOOLEAN")
    modifier.operation = operation
    modifier.solver = "EXACT"
    modifier.object = cutter
    try:
        result = bpy.ops.object.modifier_apply(modifier=modifier.name)
    except Exception as exc:
        target.select_set(False)
        raise ExecutorError(f"boolean apply failed: {exc}", "execution_error") from exc
    target.select_set(False)
    if "FINISHED" not in set(result) or target.modifiers.get(modifier.name) is not None:
        raise ExecutorError("boolean modifier did not apply", "execution_error")
    if bool(args.get("delete_cutter", True)):
        bpy.data.objects.remove(cutter, do_unlink=True)
    return {"target": _stable_uuid(target), "operation": operation}


def _artifact_destination(path: str, suffix: str, operation: str) -> Path:
    """Resolve an artifact path and enforce the optional trusted-root policy."""
    if not isinstance(path, str) or not path.strip():
        raise ExecutorError(f"{operation} path must be a non-empty string", "invalid_args")
    destination = Path(path).expanduser().resolve()
    expected = str(suffix).lower()
    if destination.suffix.lower() != expected:
        raise ExecutorError(f"{operation} path must end with {expected}", "invalid_args")
    roots = [Path(item).expanduser().resolve() for item in (os.environ.get("BLENDER_TOOLBOX_ARTIFACT_ROOTS") or "").split(os.pathsep) if item.strip()]
    if roots:
        allowed = False
        for root in roots:
            try:
                destination.relative_to(root)
            except ValueError:
                continue
            allowed = True
            break
        if not allowed:
            raise ExecutorError(f"{operation} destination is outside configured artifact roots", "policy_denied")
    return destination


def _save_checkpoint(path: str) -> Dict[str, Any]:
    _require_bpy()
    destination = _artifact_destination(path, ".blend", "checkpoint")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() != ".blend":
        raise ExecutorError("checkpoint path must end with .blend", "invalid_args")
    bpy.ops.wm.save_as_mainfile(filepath=str(destination), copy=True)
    return {"path": str(destination), "exists": destination.is_file()}


def _export_glb(path: str) -> Dict[str, Any]:
    _require_bpy()
    destination = _artifact_destination(path, ".glb", "export")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() != ".glb":
        raise ExecutorError("export path must end with .glb", "invalid_args")
    bpy.ops.export_scene.gltf(filepath=str(destination), export_format="GLB")
    return {"path": str(destination), "exists": destination.is_file()}


def _mesh_arrays(objects: Optional[Iterable[Any]] = None) -> tuple[list[list[float]], list[list[int]]]:
    """Collect evaluated mesh geometry in world coordinates for verifier hooks."""
    _require_bpy()
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    source = objects if objects is not None else (item for item in bpy.context.scene.objects if item.type == "MESH" and item.data)
    for obj in sorted((item for item in source if item.type == "MESH" and item.data), key=lambda item: _stable_uuid(item)):
        offset = len(vertices)
        vertices.extend([[float(value) for value in (obj.matrix_world @ vertex.co)] for vertex in obj.data.vertices])
        faces.extend([[offset + int(index) for index in polygon.vertices] for polygon in obj.data.polygons])
    return vertices, faces


def _verify_scope(args: Mapping[str, Any]) -> tuple[list[Any], Dict[str, Any]]:
    """Resolve ``verify.run``'s scene/target scope before any checks execute.

    Target scope is intentionally structural: selected roots plus all of their
    descendants are audited.  This makes a focused repair verifiable without
    accidentally allowing unrelated cutters, hidden helpers, or a second
    asset elsewhere in the scene to change the result.
    """
    all_objects = _geometry_objects()
    scope = str(args.get("audit_scope", "scene") or "scene").strip().lower()
    if scope not in {"scene", "targets"}:
        raise ExecutorError("audit_scope must be 'scene' or 'targets'", "invalid_args")
    raw_targets = args.get("targets")
    if scope == "scene":
        return all_objects, {"scope": "scene", "requested_targets": [], "resolved_targets": [{"uuid": _stable_uuid(obj), "name": obj.name} for obj in all_objects], "missing_targets": [], "descendants_included": False}
    if isinstance(raw_targets, str):
        raw_targets = [raw_targets]
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ExecutorError("audit_scope='targets' requires a non-empty targets array", "invalid_args")
    roots: list[Any] = []
    missing: list[str] = []
    for raw in raw_targets:
        obj = _resolve_scene_ref(raw, all_objects)
        if obj is None:
            missing.append(str(raw))
        elif obj not in roots:
            roots.append(obj)
    if missing:
        return [], {"scope": "targets", "requested_targets": [str(item) for item in raw_targets], "resolved_targets": [], "missing_targets": sorted(missing), "descendants_included": True}
    selected: list[Any] = []
    resolved_roots: list[Dict[str, Any]] = []
    seen: set[int] = set()

    def include(obj: Any) -> None:
        marker = id(obj)
        if marker in seen:
            return
        seen.add(marker)
        if obj.type in {"MESH", "CURVE", "SURFACE", "FONT"}:
            selected.append(obj)
        for child in getattr(obj, "children", ()):
            include(child)

    for root in roots:
        resolved_roots.append({"uuid": _stable_uuid(root), "name": root.name, "type": getattr(root, "type", None)})
        include(root)
    # Include relation dependencies (parent, explicit attachment parent, and
    # explicit surface target).  A target-scoped audit must not report an
    # attachment as unknown merely because its supporting object is outside
    # the selected root's descendant tree.
    dependency_queue = list(selected)
    while dependency_queue:
        obj = dependency_queue.pop()
        candidates = [getattr(obj, "parent", None)]
        attachment = _load_json_prop(obj, _ATTACHMENT_PROP, {})
        snap = _load_json_prop(obj, _SNAP_PROP, {})
        for record, key in ((attachment, "parent"), (snap, "surface")):
            if isinstance(record, Mapping) and record.get(key):
                candidates.append(_resolve_scene_ref(record.get(key), all_objects))
        for dependency in candidates:
            if dependency is not None and id(dependency) not in seen:
                include(dependency)
                dependency_queue.append(dependency)
    selected.sort(key=lambda item: _stable_uuid(item))
    return selected, {"scope": "targets", "requested_targets": [str(item) for item in raw_targets], "resolved_roots": resolved_roots, "resolved_targets": [{"uuid": _stable_uuid(obj), "name": obj.name} for obj in selected], "missing_targets": [], "descendants_included": True, "relation_dependencies_included": True}


def _load_task_spec(args: Mapping[str, Any]) -> Dict[str, Any]:
    spec = args.get("task_spec")
    if isinstance(spec, Mapping):
        return dict(spec)
    path = args.get("task_spec_path")
    if path:
        destination = Path(str(path)).expanduser().resolve()
        if not destination.is_file():
            raise ExecutorError(f"task spec not found: {destination}", "invalid_args")
        try:
            value = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ExecutorError(f"invalid task spec: {exc}", "invalid_args") from exc
        if not isinstance(value, Mapping):
            raise ExecutorError("task spec must be a JSON object", "invalid_args")
        return dict(value)
    return {}


def _normalize_tags(value: Any) -> tuple[str, ...]:
    """Return stable, non-empty semantic tags from task/action input."""
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = []
    return tuple(sorted({str(item).strip() for item in values if str(item).strip()}))


def _parameters_sample(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Trace-friendly deterministic distributions without mutating Blender."""
    distribution = str(args["distribution"]).lower()
    raw_seed = args.get("seed", 0)
    if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
        raise ExecutorError("seed must be an integer", "invalid_args")
    seed = int(raw_seed)
    if seed < 0 or seed > MAX_SEED:
        raise ExecutorError(f"seed must be between 0 and {MAX_SEED}", "invalid_args")
    count = max(1, min(4096, int(args.get("count", 1))))
    rng = random.Random(seed)
    low = float(args.get("low", 0.0))
    high = float(args.get("high", 1.0))
    if high < low:
        raise ExecutorError("high must be greater than or equal to low", "invalid_args")
    mean = float(args.get("mean", (low + high) * 0.5))
    std = float(args.get("std", max((high - low) / 6.0, 1e-9)))
    mode = float(args.get("mode", (low + high) * 0.5))
    if distribution == "log_uniform" and (low <= 0 or high <= 0):
        raise ExecutorError("log_uniform requires positive low/high", "invalid_args")
    if distribution == "normal" and std <= 0:
        raise ExecutorError("normal std must be positive", "invalid_args")
    values: list[float | int]
    if distribution == "uniform":
        values = [rng.uniform(low, high) for _ in range(count)]
    elif distribution == "normal":
        values = [rng.gauss(mean, std) for _ in range(count)]
    elif distribution == "log_uniform":
        values = [math.exp(rng.uniform(math.log(low), math.log(high))) for _ in range(count)]
    elif distribution == "triangular":
        if not low <= mode <= high:
            raise ExecutorError("triangular mode must lie between low and high", "invalid_args")
        values = [rng.triangular(low, high, mode) for _ in range(count)]
    elif distribution == "integer":
        integer_low, integer_high = int(math.ceil(low)), int(math.floor(high))
        if integer_low > integer_high:
            raise ExecutorError("integer distribution range contains no integers", "invalid_args")
        values = [rng.randint(integer_low, integer_high) for _ in range(count)]
    else:
        raise ExecutorError(f"unsupported distribution: {distribution}", "invalid_args")
    rounded = [int(value) if distribution == "integer" else round(float(value), 10) for value in values]
    return {"name": args.get("name"), "distribution": distribution, "seed": seed, "count": count, "parameters": {"low": low, "high": high, "mean": mean, "std": std, "mode": mode}, "values": rounded, "hash": content_hash(rounded)}


def _inject_request_seed(action: str, args: Mapping[str, Any], request: Optional[ActionRequest]) -> Mapping[str, Any]:
    """Use the episode seed for random actions when no local seed is given.

    ProcFunc makes RNG inputs explicit.  Toolbox keeps that property at the
    action boundary: an explicit action seed wins, otherwise the episode seed
    is used (with a stable zero fallback for legacy direct executor calls).
    """
    if action not in {"parameters.sample", "particles.scatter"} or "seed" in args:
        return args
    enriched = dict(args)
    raw_seed = request.seed if request is not None and request.seed is not None else 0
    if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
        raise ExecutorError("seed must be an integer", "invalid_args")
    seed = int(raw_seed)
    # ``ActionRequest.from_dict`` enforces this for transport requests, but
    # keep the dispatch helper defensive because it is also used by direct
    # in-process callers and tests that may construct requests manually.
    if seed < 0 or seed > MAX_SEED:
        raise ExecutorError(f"seed must be between 0 and {MAX_SEED}", "invalid_args")
    enriched["seed"] = seed
    return enriched


# Mixed-mode scripts are intentionally broader than ``run_python`` but remain
# bounded and auditable.  They can define geometry helpers and use allowlisted
# imports, while filesystem/network/process access is excluded at the AST and
# builtin-import layers.
_ALLOWED_DECLARED_IMPORTS = {"bpy", "math", "json", "mathutils"}
_ALLOWED_DECLARED_FROM_IMPORTS = {"Vector"}
_ALLOWED_DECLARED_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "len",
    "list", "map", "max", "min", "print", "range", "reversed", "round", "set", "sorted",
    "str", "sum", "tuple", "zip",
}
_DECLARED_BLOCKED_NAMES = {
    "__builtins__", "__import__", "eval", "exec", "compile", "open", "input", "globals", "locals",
    "vars", "dir", "help", "breakpoint", "os", "sys", "subprocess", "shutil", "socket", "pathlib",
    "requests", "urllib", "pickle", "marshal", "ctypes", "inspect", "importlib",
}
_DECLARED_PYTHON_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Name, ast.Load, ast.Store,
    ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Attribute, ast.Call, ast.keyword,
    ast.Import, ast.ImportFrom, ast.alias, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
    ast.For, ast.While, ast.If, ast.Break, ast.Continue, ast.Pass, ast.Compare, ast.BoolOp,
    ast.BinOp, ast.UnaryOp, ast.IfExp, ast.Subscript, ast.Slice, ast.ListComp, ast.SetComp,
    ast.DictComp, ast.GeneratorExp, ast.comprehension, ast.Del, ast.Add, ast.Sub, ast.Mult,
    ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
)


def validate_declared_bpy_python(source: str, *, max_chars: int = 200000) -> ast.Module:
    if len(source) > max_chars:
        raise ExecutorError(f"bpy source exceeds {max_chars} characters", "policy_denied")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise ExecutorError(f"invalid bpy Python: {exc}", "invalid_args") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _DECLARED_PYTHON_NODES):
            raise ExecutorError(f"bpy asset node is not allowed: {type(node).__name__}", "policy_denied")
        if isinstance(node, ast.Name) and node.id in _DECLARED_BLOCKED_NAMES:
            raise ExecutorError(f"bpy asset name is not allowed: {node.id}", "policy_denied")
        if isinstance(node, ast.alias):
            module_name = str(node.name).split(".", 1)[0]
            if node.name not in _ALLOWED_DECLARED_FROM_IMPORTS and module_name not in _ALLOWED_DECLARED_IMPORTS:
                raise ExecutorError(f"bpy asset import is not allowed: {node.name}", "policy_denied")
        if isinstance(node, ast.ImportFrom):
            module_name = str(node.module or "").split(".", 1)[0]
            if module_name not in _ALLOWED_DECLARED_IMPORTS or any(alias.name not in _ALLOWED_DECLARED_FROM_IMPORTS for alias in node.names):
                raise ExecutorError(f"bpy asset import is not allowed: {node.module}", "policy_denied")
        if isinstance(node, ast.Attribute):
            chain = _attribute_chain(node)
            if any(part.startswith("__") or part in _BLOCKED_ATTRS for part in chain[1:]):
                raise ExecutorError(f"bpy asset attribute is not allowed: {'.'.join(chain)}", "policy_denied")
    return tree


def _declared_import(name: str, globals_dict: Mapping[str, Any], locals_dict: Mapping[str, Any], fromlist: tuple[str, ...] = (), level: int = 0) -> Any:
    module_name = str(name).split(".", 1)[0]
    if module_name not in _ALLOWED_DECLARED_IMPORTS:
        raise ImportError(f"bpy asset import is not allowed: {name}")
    if module_name == "bpy":
        return bpy
    return __import__(module_name)


def _declared_bpy_builtins() -> Dict[str, Any]:
    import builtins
    values = {name: getattr(builtins, name) for name in _ALLOWED_DECLARED_BUILTINS}
    values["__import__"] = _declared_import
    return values


def _scene_object_names() -> set[str]:
    _require_bpy()
    return {str(obj.name) for obj in bpy.context.scene.objects}


def _scene_objects_by_name() -> Dict[str, Any]:
    _require_bpy()
    return {str(obj.name): obj for obj in bpy.context.scene.objects}


def _scene_fingerprints() -> Dict[str, str]:
    payload: Dict[str, str] = {}
    for obj in bpy.context.scene.objects:
        payload[str(obj.name)] = content_hash({
            "type": obj.type,
            "location": [round(float(v), 8) for v in obj.location],
            "rotation": [round(float(v), 8) for v in obj.rotation_euler],
            "scale": [round(float(v), 8) for v in obj.scale],
            "parent": obj.parent.name if obj.parent else None,
            "geometry": _mesh_geometry_hash(obj) or _curve_geometry_hash(obj),
            "tags": _semantic_tags(obj),
        })
    return payload


def _validate_mixed_postconditions(postconditions: Mapping[str, Any]) -> Dict[str, Any]:
    objects = _scene_objects_by_name()
    findings: list[str] = []
    exists = [str(v) for v in (postconditions.get("objects_exist") or [])]
    absent = [str(v) for v in (postconditions.get("objects_absent") or [])]
    for name in exists:
        if name not in objects:
            findings.append(f"declared object missing: {name}")
    for name in absent:
        if name in objects:
            findings.append(f"declared object still present: {name}")
    parent_checks = []
    for condition in (postconditions.get("parent_of") or []):
        child_name, parent_name = str(condition["child"]), str(condition["parent"])
        actual = objects.get(child_name)
        actual_parent = actual.parent.name if actual is not None and actual.parent is not None else None
        passed = actual is not None and actual_parent == parent_name
        parent_checks.append({"child": child_name, "expected_parent": parent_name, "actual_parent": actual_parent, "gate": passed})
        if not passed:
            findings.append(f"parent condition failed: {child_name} -> {parent_name}")
    tag_checks = []
    for condition in (postconditions.get("semantic_tags") or []):
        name = str(condition["object"])
        expected = sorted(str(v) for v in (condition.get("tags") or []))
        actual = sorted(_semantic_tags(objects[name])) if name in objects else []
        passed = name in objects and set(expected).issubset(actual)
        tag_checks.append({"object": name, "expected_tags": expected, "actual_tags": actual, "gate": passed})
        if not passed:
            findings.append(f"semantic tag condition failed: {name}")
    return {"gate": not findings, "findings": findings, "objects_exist": {"expected": exists, "actual": sorted(name for name in exists if name in objects)}, "objects_absent": {"expected": absent, "actual": sorted(name for name in absent if name not in objects)}, "parent_of": parent_checks, "semantic_tags": tag_checks}


def _declaration_delta(before_names: set[str], after_names: set[str], args: Mapping[str, Any], before_fingerprints: Optional[Mapping[str, str]] = None, after_fingerprints: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    created_actual = sorted(after_names - before_names)
    deleted_actual = sorted(before_names - after_names)
    before_fingerprints, after_fingerprints = before_fingerprints or {}, after_fingerprints or {}
    modified_actual = sorted(name for name in before_names & after_names if before_fingerprints.get(name) != after_fingerprints.get(name))
    declared_created = sorted(str(v) for v in (args.get("creates") or []))
    declared_modified = sorted(str(v) for v in (args.get("modifies") or []))
    declared_deleted = sorted(str(v) for v in (args.get("deletes") or []))
    findings = []
    for name in declared_created:
        if name not in after_names: findings.append(f"declared created object missing: {name}")
    for name in declared_deleted:
        if name in after_names: findings.append(f"declared deleted object still present: {name}")
    for name in declared_modified:
        if name not in after_names: findings.append(f"declared modified object missing: {name}")
    if bool(args.get("strict_declarations", True)):
        for label, values in (("created", set(created_actual)-set(declared_created)), ("deleted", set(deleted_actual)-set(declared_deleted)), ("modified", set(modified_actual)-set(declared_modified)-set(declared_created))):
            if values: findings.append(f"undeclared {label} objects: {sorted(values)}")
    return {"gate": not findings, "findings": findings, "created_actual": created_actual, "modified_actual": modified_actual, "deleted_actual": deleted_actual, "declared_created": declared_created, "declared_modified": declared_modified, "declared_deleted": declared_deleted}


def _dispatch_workflow_batch(executor: Any, args: Mapping[str, Any], request: Optional[ActionRequest]) -> Dict[str, Any]:
    """Run bounded mutating Toolbox steps as one atomic, traceable action."""
    steps = list(args.get("steps") or [])
    if not steps:
        raise ExecutorError("workflow.batch requires at least one step", "invalid_args")
    transaction = bool(args.get("transaction", True))
    rollback = bool(args.get("rollback_on_error", True))
    before_names, before_fingerprints = _scene_object_names(), _scene_fingerprints()
    snapshot: Optional[tuple[Path, str, str]] = None
    completed: list[Dict[str, Any]] = []
    try:
        if transaction:
            snapshot = _save_transaction_snapshot("blender_toolbox_batch_")
        for index, step in enumerate(steps):
            child_action = str(step.get("action"))
            child_args = dict(step.get("args") or {})
            child_request = SimpleNamespace(step_id=(int(request.step_id) * 1000 + index) if request is not None else index)
            result = executor._dispatch(child_action, child_args, child_request)
            completed.append({"index": index, "action": child_action, "label": step.get("label"), "result": result})
        after_names, after_fingerprints = _scene_object_names(), _scene_fingerprints()
        declarations = _declaration_delta(before_names, after_names, {
            "creates": args.get("creates") or [], "modifies": args.get("modifies") or [], "deletes": args.get("deletes") or [],
            "strict_declarations": bool(args.get("strict_declarations", False)),
        }, before_fingerprints, after_fingerprints)
        if not declarations["gate"]:
            raise ExecutorError("workflow declarations failed", "postcondition_failed", details={"declarations": declarations})
        verification = None
        if isinstance(args.get("verify_after"), Mapping):
            verification = _verify(args["verify_after"], trusted_verifier_paths=executor.verifier_paths, current_revision=executor.revision, quality_contract=executor._quality_contract)
            if not verification.get("gate"):
                raise ExecutorError("workflow verify_after failed", "postcondition_failed", details={"verify": verification})
        return {"intent": str(args["intent"]), "steps": completed, "completed": len(completed), "rolled_back": False,
                "created": declarations["created_actual"], "modified": declarations["modified_actual"], "deleted": declarations["deleted_actual"],
                "declarations": declarations, "verify_after": verification}
    except Exception as exc:
        rolled_back = bool(transaction and rollback and snapshot is not None)
        if rolled_back:
            try:
                _restore_transaction_snapshot(*snapshot)
            except Exception as restore_exc:
                raise ExecutorError(f"workflow failed and rollback failed: {exc}", "rollback_failed", details={"original": str(exc), "restore": str(restore_exc), "completed": completed}) from exc
        details = getattr(exc, "details", {}) if isinstance(exc, ExecutorError) else {}
        details = dict(details) if isinstance(details, Mapping) else {}
        details.update({"completed": completed, "rolled_back": rolled_back})
        if isinstance(exc, ExecutorError):
            raise ExecutorError(str(exc), exc.code, details=details) from exc
        raise ExecutorError(str(exc), "workflow_failed", details=details) from exc
    finally:
        if snapshot is not None:
            try:
                snapshot[0].unlink(missing_ok=True)
            except OSError:
                pass


def _dispatch_bpy_apply(executor: Any, args: Mapping[str, Any]) -> Dict[str, Any]:
    """Execute a declared Blender Python asset under the mixed-mode policy."""
    if not (bool(getattr(executor, "allow_bpy_apply", False)) or bool(getattr(executor, "allow_run_python", False))):
        raise ExecutorError("bpy.apply is disabled by policy; start Toolbox with --allow-bpy-apply", "policy_denied")
    source_path = args.get("source_path")
    source = args.get("source")
    if source_path:
        path = Path(str(source_path)).expanduser().resolve()
        if not path.is_file():
            raise ExecutorError(f"bpy source not found: {path}", "invalid_args")
        trusted_roots = [Path(__file__).resolve().parent.parent, Path.home() / ".codex" / "skills", Path.home() / ".cc-switch" / "skills"]
        trusted = any(path == root or root in path.parents for root in trusted_roots)
        if not trusted and not args.get("source_sha256"):
            raise ExecutorError("external bpy source requires source_sha256", "policy_denied")
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ExecutorError(f"cannot read bpy source: {exc}", "invalid_args") from exc
    if not isinstance(source, str) or not source:
        raise ExecutorError("bpy.apply requires source_path or source", "invalid_args")
    code_hash = "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()
    provided_hash = args.get("source_sha256")
    if provided_hash and str(provided_hash) not in {code_hash, code_hash.removeprefix("sha256:")}:
        raise ExecutorError("bpy source hash mismatch", "invalid_args", details={"expected": str(provided_hash), "actual": code_hash})
    validate_declared_bpy_python(source, max_chars=200000)
    before_names, before_fingerprints = _scene_object_names(), _scene_fingerprints()
    transaction, rollback = bool(args.get("transaction", True)), bool(args.get("rollback_on_error", True))
    snapshot: Optional[tuple[Path, str, str]] = None
    started = time.monotonic()
    try:
        if transaction:
            snapshot = _save_transaction_snapshot("blender_toolbox_bpy_")
        namespace = {"bpy": bpy, "math": __import__("math"), "Vector": Vector, "__builtins__": _declared_bpy_builtins()}
        timeout_ms = max(1, min(int(args.get("timeout_ms", 10000)), 120000))
        with _execution_deadline(timeout_ms):
            exec(compile(source, str(source_path or "<toolbox-bpy-asset>"), "exec"), namespace, namespace)
        after_names, after_fingerprints = _scene_object_names(), _scene_fingerprints()
        declarations = _declaration_delta(before_names, after_names, args, before_fingerprints, after_fingerprints)
        if not declarations["gate"]:
            raise ExecutorError("bpy declarations failed", "postcondition_failed", details={"declarations": declarations})
        post = _validate_mixed_postconditions(args.get("postconditions") or {})
        if not post["gate"]:
            raise ExecutorError("bpy postconditions failed", "postcondition_failed", details={"postconditions": post, "declarations": declarations})
        result = {"executed": True, "purpose": str(args["purpose"]), "source_path": str(source_path) if source_path else None, "code_hash": code_hash,
                  "trusted": False, "replayable": False, "risk_level": "declared_high", "duration_ms": int((time.monotonic() - started) * 1000),
                  "declarations": declarations, "postconditions": post, "toolbox_mixed_mode": True}
        max_result_chars = max(128, min(int(args.get("max_result_chars", 65536)), 65536))
        if len(json.dumps(result, ensure_ascii=True, separators=(",", ":"))) > max_result_chars:
            raise ExecutorError("bpy.apply result exceeds configured size", "policy_denied")
        return result
    except Exception as exc:
        rolled_back = bool(transaction and rollback and snapshot is not None)
        if rolled_back:
            try:
                _restore_transaction_snapshot(*snapshot)
            except Exception as restore_exc:
                raise ExecutorError(f"bpy.apply failed and rollback failed: {exc}", "rollback_failed", details={"original": str(exc), "restore": str(restore_exc)}) from exc
        details = getattr(exc, "details", {}) if isinstance(exc, ExecutorError) else {}
        details = dict(details) if isinstance(details, Mapping) else {}
        details["rolled_back"] = rolled_back
        if isinstance(exc, ExecutorError):
            raise ExecutorError(str(exc), exc.code, details=details) from exc
        raise ExecutorError(str(exc), "bpy_apply_failed", details=details) from exc
    finally:
        if snapshot is not None:
            try:
                snapshot[0].unlink(missing_ok=True)
            except OSError:
                pass


_ALLOWED_PYTHON_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.Name, ast.Load, ast.Store,
    ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Attribute, ast.Call,
    ast.keyword,
)
_ALLOWED_PYTHON_NAMES = {"bpy", "math", "Vector"}
_ALLOWED_BPY_ROOTS = {"ops", "data", "context"}
_ALLOWED_BPY_OPS = {
    "object": {"select_all", "delete", "modifier_apply", "transform_apply", "shade_smooth", "mode_set"},
    "mesh": {"primitive_cube_add", "primitive_uv_sphere_add", "primitive_cylinder_add", "primitive_cone_add", "primitive_torus_add", "primitive_plane_add"},
    "curve": {"primitive_bezier_curve_add", "primitive_bezier_circle_add"},
    "transform": {"translate", "rotate", "resize"},
}
_ALLOWED_BPY_DATA_GROUPS = {"objects", "meshes", "curves", "materials", "collections", "scenes"}
_ALLOWED_BPY_CONTEXT_GROUPS = {"scene", "view_layer", "object", "selected_objects"}
_ALLOWED_CONTEXT_ATTRS = {
    "object": {"location", "rotation_euler", "scale", "name", "hide_viewport", "hide_render", "display_type"},
    "scene": {"frame_current", "frame_start", "frame_end"},
}
_ALLOWED_MATH_ATTRS = {"sin", "cos", "tan", "asin", "acos", "atan2", "sqrt", "pow", "fabs", "floor", "ceil", "pi", "e", "radians", "degrees"}
_BLOCKED_ATTRS = {
    "__builtins__", "__globals__", "__locals__", "__subclasses__", "__import__",
    "open", "load", "save", "write", "read", "filepath", "libraries", "texts",
    "images", "sounds", "movieclips", "wm", "file", "export_scene", "import_scene",
    "preferences", "window", "screen", "temp_override",
}


def _attribute_chain(node: ast.AST) -> list[str]:
    chain: list[str] = []
    while isinstance(node, ast.Attribute):
        chain.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        chain.append(node.id)
    return list(reversed(chain))


def _allowed_python_chain(chain: list[str]) -> bool:
    if not chain or chain[0] not in _ALLOWED_PYTHON_NAMES:
        return False
    if any(part.startswith("__") or part in _BLOCKED_ATTRS for part in chain[1:]):
        return False
    if chain[0] == "math":
        return len(chain) == 1 or (len(chain) == 2 and chain[1] in _ALLOWED_MATH_ATTRS)
    if chain[0] == "Vector":
        return len(chain) == 1
    if len(chain) == 2 and chain[1] in _ALLOWED_BPY_ROOTS:
        return True
    if len(chain) < 2 or chain[1] not in _ALLOWED_BPY_ROOTS:
        return False
    if chain[1] == "ops":
        return len(chain) == 2 or (len(chain) == 3 and chain[2] in _ALLOWED_BPY_OPS) or (len(chain) == 4 and chain[3] in _ALLOWED_BPY_OPS.get(chain[2], set()))
    if chain[1] == "data":
        return len(chain) == 2 or (len(chain) == 3 and chain[2] in _ALLOWED_BPY_DATA_GROUPS) or (len(chain) == 4 and chain[2] in _ALLOWED_BPY_DATA_GROUPS and chain[3] == "get")
    if chain[1] == "context":
        return len(chain) == 2 or (len(chain) == 3 and chain[2] in _ALLOWED_BPY_CONTEXT_GROUPS) or (len(chain) == 4 and chain[2] in _ALLOWED_CONTEXT_ATTRS and chain[3] in _ALLOWED_CONTEXT_ATTRS[chain[2]])
    return False


def _allowed_assignment_chain(chain: list[str]) -> bool:
    return len(chain) == 4 and chain[:3] == ["bpy", "context", "object"] and chain[3] in _ALLOWED_CONTEXT_ATTRS["object"]


def validate_restricted_python(source: str, *, max_chars: int = 8000) -> ast.Module:
    if len(source) > max_chars:
        raise ExecutorError(f"run_python source exceeds {max_chars} characters", "policy_denied")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise ExecutorError(f"invalid Python: {exc}", "invalid_args") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_PYTHON_NODES):
            raise ExecutorError(f"run_python node is not allowed: {type(node).__name__}", "policy_denied")
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_PYTHON_NAMES:
            raise ExecutorError(f"run_python name is not allowed: {node.id}", "policy_denied")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            raise ExecutorError("run_python local assignments are not allowed", "policy_denied")
        if isinstance(node, ast.Attribute) and not _allowed_python_chain(_attribute_chain(node)):
            raise ExecutorError(f"run_python attribute is not allowed: {'.'.join(_attribute_chain(node))}", "policy_denied")
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store) and not _allowed_assignment_chain(_attribute_chain(node)):
            raise ExecutorError(f"run_python assignment target is not allowed: {'.'.join(_attribute_chain(node))}", "policy_denied")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id != "Vector":
            raise ExecutorError(f"run_python direct call is not allowed: {node.func.id}", "policy_denied")
    return tree


@contextlib.contextmanager
def _execution_deadline(timeout_ms: int):
    """Best-effort hard deadline for the main-thread escape hatch on POSIX."""
    if timeout_ms <= 0:
        yield
        return
    previous = None
    enabled = False
    try:
        import signal
        if threading.current_thread() is threading.main_thread() and hasattr(signal, "setitimer"):
            def _alarm(_signum, _frame):
                raise ExecutorError(f"run_python exceeded {timeout_ms}ms", "timeout")
            previous = signal.signal(signal.SIGALRM, _alarm)
            signal.setitimer(signal.ITIMER_REAL, timeout_ms / 1000.0)
            enabled = True
        yield
    finally:
        if enabled:
            import signal
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


def _save_transaction_snapshot(prefix: str) -> tuple[Path, str, str]:
    _require_bpy()
    working_filepath = str(bpy.data.filepath or "")
    working_scene = str(bpy.context.scene.name) if bpy.context.scene is not None else ""
    fd, raw_path = tempfile.mkstemp(prefix=prefix, suffix=".blend")
    os.close(fd)
    snapshot = Path(raw_path)
    bpy.ops.wm.save_as_mainfile(filepath=str(snapshot), copy=True)
    return snapshot, working_filepath, working_scene


def _restore_transaction_snapshot(snapshot: Path, working_filepath: str, working_scene: str = "") -> None:
    _require_bpy()
    old_scenes = list(bpy.data.scenes)
    with bpy.data.libraries.load(str(snapshot), link=False) as (data_from, _data_to):
        names = list(data_from.scenes or [])
    if not names:
        raise ExecutorError("transaction snapshot contains no scenes", "rollback_failed")
    for index, scene in enumerate(old_scenes):
        scene.name = f"__toolbox_rollback_old_{index}"
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for scene in old_scenes:
        for child in list(scene.collection.children):
            scene.collection.children.unlink(child)
    with bpy.data.libraries.load(str(snapshot), link=False) as (data_from, data_to):
        data_to.scenes = list(data_from.scenes or [])
    loaded = [scene for scene in data_to.scenes if scene is not None]
    if not loaded:
        raise ExecutorError("transaction snapshot scene restore returned no scenes", "rollback_failed")
    restored = next((scene for scene in loaded if scene.name == working_scene), loaded[0])
    for window in bpy.context.window_manager.windows:
        window.scene = restored
    for scene in old_scenes:
        if scene.name in bpy.data.scenes:
            bpy.data.scenes.remove(scene)
    if str(bpy.data.filepath or "") != working_filepath:
        raise ExecutorError("rollback restored scene but changed the active filepath", "rollback_failed")


def _load_module(path: str, name: str, *, trusted_paths: Optional[Iterable[Path]] = None) -> Any:
    module_path = Path(path).expanduser().resolve()
    if not module_path.is_file():
        raise ExecutorError(f"verifier module not found: {module_path}", "invalid_args")
    if trusted_paths is not None:
        roots = [Path(item).expanduser().resolve() for item in trusted_paths]
        if not any(module_path == item or item in module_path.parents for item in roots):
            raise ExecutorError("verifier module is outside trusted roots", "policy_denied")
    spec = importlib.util.spec_from_file_location(name, str(module_path))
    if spec is None or spec.loader is None:
        raise ExecutorError(f"cannot load verifier module: {module_path}", "execution_error")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_local_tcp(address: str) -> tuple[str, int]:
    if not address.startswith("tcp://"):
        raise ExecutorError(f"unsupported IPC address: {address}", "invalid_args")
    host_port = address[6:]
    if host_port.startswith("[") and "]" in host_port:
        host, _, port_text = host_port[1:].partition("]")
        port_text = port_text.lstrip(":")
    else:
        try:
            host, port_text = host_port.rsplit(":", 1)
        except ValueError as exc:
            raise ExecutorError("tcp address must be tcp://127.0.0.1:PORT", "invalid_args") from exc
    host = host.lower()
    if host not in _LOCAL_HOSTS:
        raise ExecutorError("toolbox TCP IPC is restricted to localhost", "policy_denied")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ExecutorError("tcp port must be an integer", "invalid_args") from exc
    if not 1 <= port <= 65535:
        raise ExecutorError("tcp port must be between 1 and 65535", "invalid_args")
    return host, port


class _CoreToolboxExecutor:
    """Stateful, single-threaded Blender action executor."""

    def __init__(self, *, allow_run_python: bool = False, allow_bpy_apply: bool = False, auth_token: Optional[str] = None) -> None:
        self.allow_run_python = allow_run_python
        self.allow_bpy_apply = allow_bpy_apply
        self.auth_token = auth_token
        self.revision = 0
        self._idempotent: Dict[str, Dict[str, Any]] = {}
        # Keep the canonical request identity alongside cached responses.  A
        # client retrying the same key should receive the original response,
        # while accidental key reuse for a different request must fail loudly.
        self._idempotency_fingerprints: Dict[str, str] = {}
        self._lock = threading.RLock()
        self._previous_summary: Optional[Dict[str, Any]] = None
        self._session_id: Optional[str] = None
        self._required_tags_lock: Optional[frozenset[str]] = None
        self._task_spec: Dict[str, Any] = {}
        self._task_spec_frozen = False
        self._task_spec_hash: Optional[str] = None
        self._profile: Optional[str] = None
        self._quality_contract: Dict[str, Any] = _quality_contract("quality_first")
        self._last_verify: Optional[Dict[str, Any]] = None
        self._last_render: Optional[Dict[str, Any]] = None
        self._last_visual_review: Optional[Dict[str, Any]] = None
        self._visual_review_history: list[Dict[str, Any]] = []
        self._stage_ledger: Dict[str, Dict[str, Any]] = {}
        self.verifier_paths = [
            Path(__file__).resolve().parent.parent,
            Path.home() / ".codex" / "skills",
            Path.home() / ".cc-switch" / "skills",
        ]
        for raw_path in (os.environ.get("BLENDER_TOOLBOX_VERIFIER_ROOTS") or "").split(os.pathsep):
            if raw_path.strip():
                self.verifier_paths.append(Path(raw_path.strip()))

    def _state(self, *, refresh: bool = False) -> Dict[str, Any]:
        """Return an authoritative observation, reusing the last full census.

        The executor captures a state before and after every action.  A full
        census includes geometry, UV, material-node, and modifier hashes, so
        rescanning it twice for a non-mutating action is needlessly expensive.
        Mutating callers request ``refresh=True`` after dispatch; failed
        mutations invalidate the cache before dispatch so a later action will
        rebuild it rather than trusting a potentially partial scene change.
        """
        if refresh or self._previous_summary is None:
            self._previous_summary = scene_summary()
        summary = copy.deepcopy(self._previous_summary)
        # Keep the active quality contract and stage ledger in the
        # authoritative state so a resumed client does not have to infer them
        # from scattered action arguments.
        summary["quality_contract"] = copy.deepcopy(self._quality_contract)
        summary["quality_contract_hash"] = content_hash(self._quality_contract)
        summary["task_spec_hash"] = self._task_spec_hash
        summary["quality_bar"] = quality_bar()
        summary["stage_ledger"] = copy.deepcopy(self._stage_ledger)
        summary["evidence"] = {
            "last_verify": copy.deepcopy(self._last_verify),
            "last_render": copy.deepcopy(self._last_render),
            "last_visual_review": copy.deepcopy(self._last_visual_review),
            "visual_review_history": copy.deepcopy(self._visual_review_history[-16:]),
        }
        # Preserve the historical top-level render_evidence field while the
        # richer evidence record lives under ``summary.evidence``.
        if isinstance(self._last_render, Mapping):
            summary["render_evidence"] = {
                "revision": self._last_render.get("revision"),
                "quality_stage": self._last_render.get("quality_stage"),
                "views": sorted(
                    str(item.get("name", item)) if isinstance(item, Mapping) else str(item)
                    for item in (self._last_render.get("views") or [])
                ),
                "files": list(self._last_render.get("files") or []),
                "evidence_types": list(self._last_render.get("evidence_types") or []),
            }
        else:
            summary.pop("render_evidence", None)
        return observation(summary, revision=self.revision, blender_version=getattr(bpy.app, "version_string", "unknown") if bpy else "unavailable", addon_version=ADDON_VERSION)

    def execute(self, raw: Mapping[str, Any]) -> Dict[str, Any]:
        started = time.monotonic()
        payload = raw if isinstance(raw, Mapping) else {}
        lifecycle_before: Optional[Dict[str, Any]] = None
        mutation_in_progress = False
        try:
            if self.auth_token is not None and payload.get("auth_token") != self.auth_token:
                raise ExecutorError("invalid auth token", "unauthorized")
            try:
                request = ActionRequest.from_dict(payload)
            except ProtocolError as exc:
                if "request_id" in payload and not isinstance(payload.get("request_id"), str):
                    response = response_from_error("", self.revision, exc).as_dict()
                    response["duration_ms"] = int((time.monotonic() - started) * 1000)
                    return response
                raise
            # Capture the identity before dispatch.  A backend action may
            # normalize a mutable argument in place; cache identity must
            # describe the validated request that arrived on the wire.
            request_fp = request_fingerprint(request)
            with self._lock:
                lifecycle_before = {
                    "session_id": self._session_id,
                    "required_tags_lock": self._required_tags_lock,
                    "task_spec": copy.deepcopy(self._task_spec),
                    "task_spec_frozen": self._task_spec_frozen,
                    "task_spec_hash": self._task_spec_hash,
                    "profile": self._profile,
                    "quality_contract": copy.deepcopy(self._quality_contract),
                    "stage_ledger": copy.deepcopy(self._stage_ledger),
                    "last_verify": copy.deepcopy(self._last_verify),
                    "last_render": copy.deepcopy(self._last_render),
                    "last_visual_review": copy.deepcopy(self._last_visual_review),
                    "visual_review_history": copy.deepcopy(self._visual_review_history),
                }
                if request.action in {"session.create", "session.open"}:
                    if self._session_id is None:
                        self._session_id = request.session_id
                        # Required tags are cumulative within one episode,
                        # never across independent sessions sharing a Blender
                        # process.
                        self._required_tags_lock = None
                    elif self._session_id != request.session_id:
                        raise ExecutorError("another toolbox session is already active", "session_conflict")
                elif self._session_id is not None and request.session_id != self._session_id:
                    raise ExecutorError("request belongs to a different toolbox session", "session_conflict")
                if request.idempotency_key and request.idempotency_key in self._idempotent:
                    key = request.idempotency_key
                    cached_fingerprint = self._idempotency_fingerprints.get(key)
                    if cached_fingerprint != request_fp:
                        raise ExecutorError(
                            "idempotency key is already associated with a different request",
                            "idempotency_conflict",
                        )
                    # Responses contain nested result/state/metrics objects.
                    # Return an isolated copy so a direct in-process caller
                    # cannot mutate the cached retry response accidentally.
                    return copy.deepcopy(self._idempotent[key])
                if request.expected_revision is not None and request.expected_revision != self.revision:
                    raise ExecutorError(f"revision conflict: expected {request.expected_revision}, current {self.revision}", "revision_conflict")
                before = self._state()
                mutating = self._is_mutating(request.action)
                reset_action = request.action in {"scene.reset", "session.reset"} or (
                    request.action == "session.open"
                    and bool(request.args.get("reset", str(request.args.get("mode", "resume")).lower() == "new"))
                )
                # A render is an inspection checkpoint, not a license to keep
                # editing.  Force the agent to review it before any subsequent
                # scene mutation so floating/overlapping parts are caught at
                # the stage where they were introduced.
                if self._last_render and not self._last_visual_review:
                    if request.action == "render.views":
                        raise ExecutorError("the previous render is awaiting evidence.visual_review", "precondition_failed")
                    if mutating and not reset_action:
                        raise ExecutorError("a render is awaiting evidence.visual_review before the next scene mutation", "precondition_failed")
                if request.action in {"artifact.export_glb", "session.close"}:
                    require_quality = bool(request.args.get("require_quality", False)) or bool(self._quality_contract.get("enforce"))
                    require_visual = bool(request.args.get("require_visual_review", False))
                    require_completion = bool(request.args.get("require_completion", False)) or bool(self._quality_contract.get("completion_gate"))
                    if require_quality:
                        verified = self._last_verify
                        current_hash = _scene_content_hash()
                        if not isinstance(verified, Mapping) or not verified.get("gate") or verified.get("revision") != self.revision or verified.get("state_hash") != current_hash:
                            raise ExecutorError("export/close requires a passing verify.run for the current scene revision", "precondition_failed")
                    if require_visual or require_completion:
                        visual_gate = _visual_evidence_gate(
                            self._last_render,
                            self._last_visual_review,
                            current_revision=self.revision,
                            current_state_hash=_scene_content_hash(),
                            require_critical=require_completion,
                            required_views=request.args.get("required_views"),
                            required_evidence_types=request.args.get("required_evidence_types") or (self._quality_contract.get("required_evidence_types") if require_completion else None),
                            required_review_stages=request.args.get("required_review_stages"),
                            review_history=self._visual_review_history,
                            min_visual_views=int(request.args.get("min_visual_views", self._quality_contract.get("min_visual_views", 4) if require_completion else 0)),
                            min_visual_score=float(request.args.get("min_visual_score", self._quality_contract.get("min_visual_score", 0.85) if require_completion else 0.0)),
                        )
                        if not visual_gate.get("gate"):
                            raise ExecutorError(f"export/close visual gate failed: {visual_gate.get('reason')}", "precondition_failed")
                # Preserve the pre-action observation for the trajectory, but
                # force a fresh post-action census for every declared mutation.
                # This also makes failed/partially-applied mutations refresh on
                # the next request instead of reusing stale cached state.
                if mutating:
                    self._previous_summary = None
                    self._last_verify = None
                    self._last_render = None
                    self._last_visual_review = None
                    # session.open has its own lifecycle transaction.  Keep
                    # its metadata rollback semantics intact when contract
                    # validation fails; scene-reset failures are handled by
                    # the action's explicit reset path.
                    mutation_in_progress = request.action not in {"session.open"}
                locked_tags = self._required_tags_lock
                if request.action == "verify.run":
                    task_spec_for_lock = _load_task_spec(request.args)
                    declared = set(_normalize_tags(task_spec_for_lock.get("required_tags")))
                    requested = set(_normalize_tags(request.args.get("required_tags")))
                    if locked_tags is None:
                        locked_tags = frozenset(declared | requested)
                    elif declared or requested:
                        locked_tags = frozenset(set(locked_tags) | declared | requested)
                    self._required_tags_lock = locked_tags
                dispatch_args = _inject_request_seed(request.action, request.args, request)
                result = self._dispatch(request.action, dispatch_args, request, required_tags_lock=locked_tags)
                mutation_committed = not (
                    mutating
                    and isinstance(result, Mapping)
                    and result.get("committed") is False
                    and result.get("rolled_back") is True
                )
                if mutating and not mutation_committed and lifecycle_before is not None:
                    # A transaction that reports rollback leaves the scene and
                    # its evidence unchanged; restore the executor caches that
                    # were invalidated before dispatch.
                    self._last_verify = copy.deepcopy(lifecycle_before.get("last_verify"))
                    self._last_render = copy.deepcopy(lifecycle_before.get("last_render"))
                    self._last_visual_review = copy.deepcopy(lifecycle_before.get("last_visual_review"))
                    self._visual_review_history = copy.deepcopy(lifecycle_before.get("visual_review_history") or [])
                if request.action in {"scene.reset", "session.reset"} or (
                    request.action == "session.open" and bool(request.args.get("reset", str(request.args.get("mode", "resume")).lower() == "new"))
                ):
                    self.revision = 0
                    self._idempotent.clear()
                    self._idempotency_fingerprints.clear()
                    self._required_tags_lock = None
                    self._stage_ledger = {}
                    if request.action in {"scene.reset", "session.reset"}:
                        self._task_spec = {}
                        self._task_spec_frozen = False
                        self._task_spec_hash = None
                        self._quality_contract = _quality_contract("quality_first")
                elif mutating and mutation_committed:
                    self.revision += 1
                if mutating and mutation_committed:
                    requested_stage = request.args.get("quality_stage", request.args.get("stage"))
                    if requested_stage is not None:
                        stage_name = str(requested_stage).strip().lower()
                        ledger_key = stage_name if stage_name in _QUALITY_STAGES else "unknown"
                        self._stage_ledger[ledger_key] = {
                            "status": "authored",
                            "revision": self.revision,
                            "action": request.action,
                            "stage_boundary": bool(request.args.get("stage_boundary", False)),
                        }
                after = self._state(refresh=mutating)
                if request.action == "verify.run" and isinstance(result, Mapping):
                    self._last_verify = {
                        "revision": self.revision,
                        "state_hash": _scene_content_hash(),
                        "gate": bool(result.get("gate")),
                        "quality": result.get("quality"),
                        "completion_gate": bool(result.get("completion_gate", False) and result.get("gate")),
                        "quality_profile": result.get("quality_profile", self._quality_contract.get("profile", "structural")),
                    }
                    quality_result = result.get("quality")
                    if isinstance(quality_result, Mapping):
                        for stage_name, stage_result in (quality_result.get("stages") or {}).items():
                            if isinstance(stage_result, Mapping) and stage_name in _QUALITY_STAGES:
                                self._stage_ledger[str(stage_name)] = {
                                    "status": stage_result.get("status"),
                                    "gate": bool(stage_result.get("gate", False)),
                                    "revision": self.revision,
                                }
                        # Stage results are part of the state snapshot; hash
                        # the final ledger rather than the pre-ledger census.
                        after = self._state(refresh=False)
                        self._last_verify["state_hash"] = _scene_content_hash()
                elif request.action == "render.views" and isinstance(result, Mapping):
                    self._last_render = {
                        "revision": self.revision,
                        "state_hash": _scene_content_hash(),
                        "quality_stage": result.get("quality_stage") or "evidence",
                        "views": list(result.get("views") or []),
                        "files": list(result.get("files") or []),
                        "file_hashes": dict(result.get("file_hashes") or {}),
                        "evidence_types": list(result.get("evidence_types") or []),
                        "target": result.get("target"),
                    }
                    self._last_visual_review = None
                    result = {**dict(result), "revision": self.revision, "state_hash": _scene_content_hash()}
                    after = self._state(refresh=False)
                elif request.action == "evidence.visual_review" and isinstance(result, Mapping):
                    render = self._last_render
                    if not isinstance(render, Mapping) or render.get("revision") != self.revision:
                        raise ExecutorError("visual review requires render.views from the current revision", "precondition_failed")
                    rendered_views = {str(item.get("name", item)) if isinstance(item, Mapping) else str(item) for item in (render.get("views") or [])}
                    reviewed_views = {str(value) for value in (result.get("views") or [])}
                    if not reviewed_views.issubset(rendered_views):
                        raise ExecutorError(f"visual review references views not rendered at this revision: {sorted(reviewed_views - rendered_views)}", "precondition_failed")
                    if str(result.get("quality_stage")) != str(render.get("quality_stage") or "evidence"):
                        raise ExecutorError("visual review stage does not match the rendered stage", "precondition_failed")
                    self._last_visual_review = {**dict(result), "state_hash": _scene_content_hash()}
                    self._visual_review_history.append(copy.deepcopy(self._last_visual_review))
                    result = dict(self._last_visual_review)
                    after = self._state(refresh=False)
                metrics: Dict[str, Any] = {}
                if request.action == "verify.run" and isinstance(result, Mapping):
                    metrics = {"scorecard": result, "verifier_stage": "full"}
                elif self._is_mutating(request.action):
                    metrics = {"scorecard": _quick_verify(self._quality_contract), "verifier_stage": "quick"}
                artifacts = []
                if request.action in {"artifact.save_checkpoint", "artifact.export_glb"} and isinstance(result, Mapping) and result.get("path"):
                    artifacts.append({"path": result["path"], "kind": "blend" if request.action.endswith("checkpoint") else "glb"})
                if request.action == "render.views" and isinstance(result, Mapping):
                    artifacts.extend({"path": path, "kind": "render"} for path in result.get("files", []))
                response = ActionResponse(
                    request_id=request.request_id,
                    ok=True,
                    revision=self.revision,
                    result=result,
                    state={**after, "diff": state_diff(before.get("summary", {}), after.get("summary", {}))},
                    metrics=metrics,
                    artifacts=artifacts,
                    duration_ms=int((time.monotonic() - started) * 1000),
                ).as_dict()
                if request.idempotency_key:
                    # Keep the cache independent from the response object that
                    # is returned to the caller (including all nested values).
                    self._idempotent[request.idempotency_key] = copy.deepcopy(response)
                    self._idempotency_fingerprints[request.idempotency_key] = request_fp
                if request.action == "session.close":
                    self._session_id = None
                    self._idempotent.clear()
                    self._idempotency_fingerprints.clear()
                    # A future session may resume after an external Blender
                    # change; do not carry this session's census across it.
                    self._previous_summary = None
                    self._task_spec = {}
                    self._task_spec_frozen = False
                    self._task_spec_hash = None
                    self._profile = None
                    self._quality_contract = _quality_contract("quality_first")
                    self._last_verify = None
                    self._last_render = None
                    self._last_visual_review = None
                    self._visual_review_history = []
                    self._stage_ledger = {}
                return response
        except ProtocolError as exc:
            rolled_back = bool(getattr(exc, "details", {}).get("rolled_back")) if isinstance(getattr(exc, "details", {}), Mapping) else False
            if lifecycle_before is not None and (not mutation_in_progress or rolled_back):
                self._restore_lifecycle(lifecycle_before)
            elif mutation_in_progress:
                # The scene may have been partially changed before the
                # exception.  Never restore evidence from the old scene;
                # stale render/review/verify records are worse than missing
                # evidence because they can falsely pass a completion gate.
                self._last_verify = None
                self._last_render = None
                self._last_visual_review = None
                self._previous_summary = None
            response = response_from_error(payload.get("request_id", "") if isinstance(payload.get("request_id"), str) else "", self.revision, exc).as_dict()
        except Exception as exc:
            rolled_back = bool(getattr(exc, "details", {}).get("rolled_back")) if isinstance(getattr(exc, "details", {}), Mapping) else False
            if lifecycle_before is not None and (not mutation_in_progress or rolled_back):
                self._restore_lifecycle(lifecycle_before)
            elif mutation_in_progress:
                self._last_verify = None
                self._last_render = None
                self._last_visual_review = None
                self._previous_summary = None
            response = response_from_error(payload.get("request_id", "") if isinstance(payload.get("request_id"), str) else "", self.revision, exc).as_dict()
        response["duration_ms"] = int((time.monotonic() - started) * 1000)
        return response

    def _restore_lifecycle(self, snapshot: Mapping[str, Any]) -> None:
        self._session_id = snapshot.get("session_id")
        self._required_tags_lock = snapshot.get("required_tags_lock")
        self._task_spec = copy.deepcopy(snapshot.get("task_spec") or {})
        self._task_spec_frozen = bool(snapshot.get("task_spec_frozen", False))
        self._task_spec_hash = snapshot.get("task_spec_hash")
        self._profile = snapshot.get("profile")
        self._quality_contract = copy.deepcopy(snapshot.get("quality_contract") or _quality_contract("quality_first"))
        self._stage_ledger = copy.deepcopy(snapshot.get("stage_ledger") or {})
        self._last_verify = copy.deepcopy(snapshot.get("last_verify"))
        self._last_render = copy.deepcopy(snapshot.get("last_render"))
        self._last_visual_review = copy.deepcopy(snapshot.get("last_visual_review"))
        self._visual_review_history = copy.deepcopy(snapshot.get("visual_review_history") or [])
        self._last_verify = copy.deepcopy(snapshot.get("last_verify"))
        self._last_render = copy.deepcopy(snapshot.get("last_render"))
        self._last_visual_review = copy.deepcopy(snapshot.get("last_visual_review"))
        self._visual_review_history = copy.deepcopy(snapshot.get("visual_review_history") or [])

    @staticmethod
    def _is_mutating(action: str) -> bool:
        return get_tool_spec(action).mutating

    def _dispatch(self, action: str, args: Mapping[str, Any], request: Optional[ActionRequest] = None, *, required_tags_lock: Optional[Iterable[str]] = None) -> Any:
        _require_bpy()
        stable_id = None
        if request is not None and action in {
            "object.create", "object.duplicate", "curve.create", "mesh.from_pydata", "mesh.from_sections",
            "object.create_batch",
            "hair.create_strands", "scene.camera_create", "scene.light_create",
            "rig.create_armature", "landmark.create", "landmark.create_set", "face.curve_from_landmarks", "face.curve_network_from_landmarks",
        }:
            stable_id = "obj-" + content_hash({"step_id": request.step_id, "action": action, "args": args})[7:23]
        if action == "session.create":
            return {"session": bpy.context.scene.name, "mode": "resume", "contract": _scene_contract(), "quality_contract": copy.deepcopy(self._quality_contract), "quality_bar": quality_bar()}
        if action == "session.open":
            mode = str(args.get("mode", "resume")).lower()
            if mode not in {"new", "resume"}:
                raise ExecutorError("mode must be 'new' or 'resume'", "invalid_args")
            reset = bool(args.get("reset", mode == "new"))
            if reset:
                _delete_all()
            profile = args.get("profile")
            task_spec = args.get("task_spec") if isinstance(args.get("task_spec"), Mapping) else None
            explicit_quality = args.get("quality_contract") if isinstance(args.get("quality_contract"), Mapping) else None
            preserve_resume_contract = mode == "resume" and not reset and not any(key in args for key in ("quality_profile", "quality_contract", "task_spec")) and self._task_spec_frozen
            # Quality-first is the safe default for every new domain. A plain
            # resume keeps the already-frozen contract instead of silently
            # replacing it with a new one.
            quality_profile = str(args.get("quality_profile") or (self._quality_contract.get("profile") if preserve_resume_contract else "quality_first")).strip().lower()
            if quality_profile not in {"advisory", "quality_first", "structural", "production", "organic", "strict"}:
                raise ExecutorError("quality_profile must be advisory, structural, production, organic, strict, or quality_first", "invalid_args")
            if not preserve_resume_contract:
                self._profile = str(profile) if profile else None
                self._task_spec = copy.deepcopy(dict(task_spec or {}))
                self._task_spec_frozen = True
                self._task_spec_hash = content_hash(self._task_spec)
                self._quality_contract = _quality_contract(
                    quality_profile,
                    self._task_spec,
                    explicit_quality,
                    workflow_profile=str(profile) if profile else None,
                )
            result: Dict[str, Any] = {
                "session": bpy.context.scene.name,
                "mode": mode,
                "reset": reset,
                "profile": str(profile) if profile else None,
                "contract": _scene_contract(str(profile) if profile else None, task_spec),
                "quality_contract": copy.deepcopy(self._quality_contract),
                "task_spec_hash": self._task_spec_hash,
            }
            # A profiled open can be the only discovery round trip needed by a
            # task-facing client.  Keep the default response small and make
            # the optional scene census explicit because it may hash a large
            # existing scene.
            if bool(args.get("include_capabilities", False)):
                try:
                    result["capabilities"] = capability_catalog(
                        profile=str(profile) if profile else None,
                        include_examples=bool(args.get("include_examples", False)),
                    )
                except ValueError as exc:
                    raise ExecutorError(str(exc), "not_found") from exc
            if bool(args.get("include_scene", False)):
                result["scene"] = scene_summary(detail=str(args.get("scene_detail", "compact")))
            return result
        if action == "scene.coordinate_system":
            return _scene_coordinate_system(args)
        if action == "model.plan":
            return _model_plan(args, default_task_spec=self._task_spec)
        if action == "inspect.quality":
            audit = _quality_audit(args, registered_contract=self._quality_contract if self._quality_contract.get("configured") else None)
            contract_report = _inspect_quality(
                args,
                trusted_verifier_paths=self.verifier_paths,
                required_tags_lock=required_tags_lock,
                task_spec_override=self._task_spec if self._task_spec_frozen else None,
                quality_contract=self._quality_contract if self._quality_contract.get("configured") else None,
                current_revision=self.revision,
                last_render=self._last_render,
                last_visual_review=self._last_visual_review,
                visual_review_history=self._visual_review_history,
            )
            audit.update({
                "gate": contract_report.get("gate"),
                "quality": contract_report.get("quality"),
                "first_failure": contract_report.get("first_failure"),
                "repair_action": contract_report.get("repair_action"),
                "contract_unknown": contract_report.get("unknown", []),
            })
            return audit
        if action == "scene.camera_create":
            return _scene_camera_create(args, stable_id)
        if action == "scene.light_create":
            return _scene_light_create(args, stable_id)
        if action == "scene.set_camera":
            camera = _object_by_ref(args["target"])
            if camera.type != "CAMERA":
                raise ExecutorError("scene.set_camera target must be a camera", "invalid_args")
            bpy.context.scene.camera = camera
            return {"camera": _stable_uuid(camera), "name": camera.name}
        if action == "scene.set_render_settings":
            return _scene_set_render_settings(args)
        if action == "inspect.batch":
            return _inspect_batch(args)
        if action == "toolbox.capabilities":
            try:
                return capability_catalog(profile=args.get("profile"), include_examples=bool(args.get("include_examples", False)))
            except ValueError as exc:
                raise ExecutorError(str(exc), "not_found") from exc
        if action == "workflow.describe":
            try:
                return describe_workflow(str(args.get("name", "vehicle")), include_examples=bool(args.get("include_examples", False)))
            except ValueError as exc:
                raise ExecutorError(str(exc), "not_found") from exc
        if action in {"inspect.scene", "scene.census"}:
            return scene_summary()
        if action in {"session.close"}:
            return {"closed": True}
        if action in {"scene.reset", "session.reset"}:
            _delete_all()
            return {"reset": True}
        if action == "object.parent_set":
            return _object_parent_set(args)
        if action == "object.surface_snap":
            return _surface_snap(args)
        if action == "assembly.anchor_create":
            return _assembly_anchor_create(args)
        if action == "assembly.attach":
            return _assembly_attach(args)
        if action == "object.create":
            return _create_primitive(args, stable_id)
        if action == "workflow.batch":
            return _dispatch_workflow_batch(self, args, request)
        if action == "bpy.apply":
            return _dispatch_bpy_apply(self, args)
        if action == "object.create_batch":
            return _object_create_batch(args, stable_id)
        if action == "object.duplicate":
            source = _object_by_ref(args["target"])
            linked = bool(args.get("linked_data", False))
            duplicate = source.copy()
            if source.data is not None and not linked:
                duplicate.data = source.data.copy()
            duplicate.name = str(args.get("name") or f"{source.name}_copy")
            duplicate[_UUID_PROP] = stable_id or _stable_uuid(duplicate)
            if source.get(_SEMANTIC_PROP) is not None:
                duplicate[_SEMANTIC_PROP] = list(source.get(_SEMANTIC_PROP) or [])
            for metadata_key in (_ORIGIN_PROP, _ROLE_PROP, _REPRESENTATION_PROP, _QUALITY_STAGE_PROP):
                if source.get(metadata_key) is not None:
                    duplicate[metadata_key] = source.get(metadata_key)
            # A duplicate starts a new spatial relationship.  Copying stale
            # attachment/surface-snap records makes later audits believe the
            # clone is still attached to the source or its old surface.
            duplicate.pop(_ATTACHMENT_PROP, None)
            duplicate.pop(_SNAP_PROP, None)
            duplicate_args = dict(args)
            if "coordinate_frame" not in duplicate_args:
                source_frame = _load_json_prop(source, _COORDINATE_PROP, None)
                if isinstance(source_frame, Mapping):
                    duplicate_args["coordinate_frame"] = source_frame
            duplicate_frame = _coordinate_frame(duplicate_args)
            _store_json_prop(duplicate, _COORDINATE_PROP, duplicate_frame)
            reference = args.get("id", args.get("ref"))
            if reference is not None:
                if not isinstance(reference, str) or not reference.strip():
                    raise ExecutorError("id/ref must be a non-empty string", "invalid_args")
                duplicate[_REF_PROP] = reference.strip()
            bpy.context.scene.collection.objects.link(duplicate)
            if "location_delta" in args:
                delta = _length_vector(args["location_delta"], "location_delta", duplicate_frame)
                space = str(duplicate_frame.get("space", "WORLD")).upper()
                delta = _coordinate_basis(duplicate_frame) @ delta
                if space == "LOCAL":
                    delta = source.matrix_world.to_quaternion() @ delta
                elif space == "PARENT":
                    if source.parent is None:
                        raise ExecutorError("duplicate location_delta uses PARENT space but source has no parent", "precondition_failed")
                    delta = source.parent.matrix_world.to_quaternion() @ delta
                elif space == "WORLD":
                    pass
                else:
                    raise ExecutorError(f"unsupported transform space: {space}", "invalid_args")
                world = duplicate.matrix_world.copy()
                world.translation = source.matrix_world.translation + delta
                duplicate.matrix_world = world
            result = {"uuid": _stable_uuid(duplicate), "name": duplicate.name, "coordinate_frame": duplicate_frame}
            if reference is not None:
                result["ref"] = reference.strip()
            return result
        if action == "object.join":
            refs = args.get("targets") or []
            if not isinstance(refs, list) or len(refs) < 2:
                raise ExecutorError("join requires at least two targets", "invalid_args")
            objects = [_require_mesh_object(ref) for ref in refs]
            active = objects[0]
            for item in bpy.context.selected_objects:
                item.select_set(False)
            for item in objects:
                item.select_set(True)
            bpy.context.view_layer.objects.active = active
            bpy.ops.object.join()
            if args.get("name"):
                active.name = str(args["name"])
            return {"uuid": _stable_uuid(active), "name": active.name, "joined": len(objects)}
        if action == "object.delete":
            targets = args.get("targets")
            if isinstance(targets, str):
                targets = [targets]
            if not isinstance(targets, list):
                raise ExecutorError("targets must be a string or array", "invalid_args")
            deleted = []
            for ref in targets:
                obj = _object_by_ref(ref)
                deleted.append(_stable_uuid(obj))
                bpy.data.objects.remove(obj, do_unlink=True)
            return {"deleted": deleted}
        if action == "object.transform":
            obj = _object_by_ref(args["target"])
            bpy.context.view_layer.update()
            return _apply_object_transform_values(obj, args)
        if action == "object.transform_batch":
            return _object_transform_batch(args)
        if action == "object.transform_apply":
            return _object_transform_apply(args)
        if action == "object.convert":
            return _object_convert(args)
        if action == "collection.group_objects":
            return _collection_group_objects(args)
        if action == "curve.create":
            return _create_curve(args, stable_id)
        if action == "curve.subdivide":
            return _curve_subdivide(args)
        if action == "hair.create_strands":
            return _hair_create_strands(args, stable_id)
        if action == "hair.convert_to_mesh":
            return _hair_convert_to_mesh(args)
        if action == "particles.scatter":
            return _particles_scatter(args)
        if action == "mesh.from_pydata":
            return _create_mesh(args, stable_id)
        if action == "mesh.from_sections":
            return _mesh_from_sections(args, stable_id)
        if action == "mesh.subdivide":
            return _mesh_subdivide(args)
        if action == "mesh.region_define":
            return _mesh_region_define(args)
        if action == "mesh.subdivide_adaptive":
            return _mesh_subdivide_adaptive(args)
        if action == "mesh.transform_selection":
            return _mesh_transform_selection(args)
        if action == "mesh.extrude_region":
            return _mesh_extrude_region(args)
        if action == "mesh.inset_region":
            return _mesh_inset_region(args)
        if action == "mesh.duplicate_region":
            return _mesh_duplicate_region(args)
        if action == "mesh.extrude_individual":
            return _mesh_extrude_individual(args)
        if action == "mesh.inset_individual":
            return _mesh_inset_individual(args)
        if action == "mesh.bridge_edge_loops":
            return _mesh_bridge_edge_loops(args)
        if action == "mesh.loop_cut":
            return _mesh_loop_cut(args)
        if action == "mesh.bevel":
            return _mesh_bevel(args)
        if action == "mesh.merge_by_distance":
            return _mesh_merge_by_distance(args)
        if action == "mesh.recalculate_normals":
            return _mesh_recalculate_normals(args)
        if action == "mesh.delete_region":
            return _mesh_delete_region(args)
        if action == "mesh.dissolve_region":
            return _mesh_dissolve_region(args)
        if action == "mesh.fill_holes":
            return _mesh_fill_holes(args)
        if action == "mesh.cut_plane":
            return _mesh_cut_plane(args)
        if action == "mesh.cut_curve":
            return _mesh_cut_curve(args)
        if action == "mesh.repair":
            return _mesh_repair(args)
        if action == "mesh.triangulate":
            return _mesh_triangulate(args)
        if action == "mesh.shade_smooth":
            return _mesh_shade_smooth(args)
        if action == "mesh.vertex_group_assign":
            return _mesh_vertex_group_assign(args)
        if action == "mesh.attribute_write":
            return _mesh_attribute_write(args)
        if action == "mesh.attribute_read":
            return _mesh_attribute_read(args)
        if action == "mesh.geometry_query":
            return _mesh_geometry_query(args)
        if action == "mesh.region_to_loop":
            return _mesh_region_to_loop(args)
        if action == "mesh.separate":
            return _mesh_separate(args)
        if action == "mesh.symmetrize":
            return _mesh_symmetrize(args)
        if action == "sculpt.stroke":
            return _sculpt_stroke(args)
        if action == "sculpt.multires":
            return _sculpt_multires(args)
        if action == "sculpt.ridge":
            return _sculpt_ridge(args)
        if action == "sculpt.groove":
            return _sculpt_groove(args)
        if action == "sculpt.muscle":
            return _sculpt_muscle(args)
        if action == "geometry.boolean":
            return _boolean(args)
        if action == "geometry.add_modifier":
            return _apply_modifier(args)
        if action == "geometry.apply_modifier":
            return _apply_modifier(args, apply=True)
        if action == "geometry.modifier_stack":
            return _geometry_modifier_stack(args)
        if action == "geometry.remesh_voxel":
            return _geometry_remesh_voxel(args)
        if action == "geometry.shrinkwrap":
            return _geometry_shrinkwrap(args)
        if action == "uv.unwrap":
            return _uv_unwrap(args)
        if action == "uv.pack":
            return _uv_pack(args)
        if action == "uv.mark_seams":
            return _uv_set_seams(args, seam=True)
        if action == "uv.clear_seams":
            return _uv_set_seams(args, seam=False)
        if action == "uv.project":
            return _uv_project(args)
        if action == "material.create":
            return _material_create(args)
        if action == "material.assign":
            obj = _object_by_ref(args["target"])
            material = bpy.data.materials.get(str(args["material"]))
            if material is None:
                raise ExecutorError(f"material not found: {args['material']}", "not_found")
            if len(obj.data.materials) == 0:
                obj.data.materials.append(material)
            else:
                obj.data.materials[0] = material
            return {"target": _stable_uuid(obj), "material": material.name}
        if action == "material.assign_batch":
            return _material_assign_batch(args)
        if action == "material.node_graph":
            return _material_node_graph(args)
        if action == "material.apply_recipe":
            return _material_apply_recipe(args)
        if action == "material.set_input":
            return _material_set_input(args)
        if action == "rig.create_armature":
            return _rig_create_armature(args, stable_id)
        if action == "rig.bind":
            return _rig_bind(args)
        if action == "rig.pose":
            return _rig_pose(args)
        if action == "rig.add_constraint":
            return _rig_add_constraint(args)
        if action == "animation.keyframe_transform":
            return _animation_keyframe_transform(args)
        if action == "animation.set_range":
            return _animation_set_range(args)
        if action == "landmark.create":
            return _landmark_create(args, stable_id)
        if action == "landmark.create_set":
            return _landmark_create_set(args, stable_id)
        if action == "face.curve_from_landmarks":
            return _face_curve_from_landmarks(args, stable_id)
        if action == "face.curve_network_from_landmarks":
            return _face_curve_network_from_landmarks(args, stable_id)
        if action == "face.sculpt_landmarks":
            return _face_sculpt_landmarks(args)
        if action == "face.shape_key_landmarks":
            return _face_shape_key_landmarks(args)
        if action == "animation.keyframe_shape_key":
            return _animation_keyframe_shape_key(args)
        if action == "geometry_nodes.create":
            return _geometry_nodes_create(args)
        if action == "geometry_nodes.apply_recipe":
            return _geometry_nodes_apply_recipe(args)
        if action == "geometry_nodes.set_input":
            return _geometry_nodes_set_input(args)
        if action == "inspect.object":
            obj = _object_by_ref(args["target"])
            return next(item for item in scene_summary()["objects"] if item["uuid"] == _stable_uuid(obj))
        if action == "inspect.relationships":
            return _inspect_relationships(args)
        if action == "inspect.topology":
            return _topology(_object_by_ref(args["target"]))
        if action == "inspect.measure":
            return _measure(_object_by_ref(args["target"]))
        if action == "inspect.mesh_region":
            return _inspect_mesh_region(args)
        if action == "inspect.uv":
            return _inspect_uv(args)
        if action == "inspect.material":
            return _inspect_material(args)
        if action == "inspect.armature":
            return _inspect_armature(args)
        if action == "inspect.animation":
            obj = _object_by_ref(args["target"])
            return {"target": _stable_uuid(obj), "animation": _animation_summary(obj), "shape_keys": _shape_key_summary(obj), "frame_current": bpy.context.scene.frame_current, "frame_range": [bpy.context.scene.frame_start, bpy.context.scene.frame_end]}
        if action == "inspect.geometry_nodes":
            obj = _object_by_ref(args["target"])
            return {"target": _stable_uuid(obj), "geometry_nodes": _geometry_nodes_summary(obj)}
        if action == "inspect.landmarks":
            tag = args.get("semantic_tag")
            landmarks = []
            for obj in bpy.context.scene.objects:
                if obj.type != "EMPTY" or not obj.get("blender_toolbox_landmark"):
                    continue
                tags = _semantic_tags(obj)
                if tag and tag not in tags:
                    continue
                landmarks.append({"uuid": _stable_uuid(obj), "name": obj.name, "location": [round(float(value), 8) for value in obj.matrix_world.translation], "semantic_tags": tags})
            return {"landmarks": landmarks, "count": len(landmarks)}
        if action == "parameters.sample":
            return _parameters_sample(args)
        if action == "verify.run":
            return _verify(
                args,
                trusted_verifier_paths=self.verifier_paths,
                required_tags_lock=required_tags_lock,
                task_spec_override=self._task_spec if self._task_spec_frozen else None,
                quality_contract=self._quality_contract,
                current_revision=self.revision,
                last_render=self._last_render,
                last_visual_review=self._last_visual_review,
                visual_review_history=self._visual_review_history,
            )
        if action == "render.views":
            return _render(args)
        if action == "evidence.visual_review":
            review_objects = None
            if isinstance(args.get("targets"), list) and args.get("targets"):
                scoped, scope = _verify_scope({"audit_scope": "targets", "targets": args.get("targets")})
                if scope.get("missing_targets"):
                    raise ExecutorError(f"visual review targets not found: {scope['missing_targets']}", "not_found")
                review_objects = scoped
            return _visual_review(args, current_revision=self.revision, last_render=self._last_render, audit_objects=review_objects)
        if action == "artifact.save_checkpoint":
            return _save_checkpoint(str(args["path"]))
        if action == "artifact.export_glb":
            return _export_glb(str(args["path"]))
        if action == "run_python":
            if not self.allow_run_python:
                raise ExecutorError("run_python is disabled by policy", "policy_denied")
            source = str(args.get("source", ""))
            validate_restricted_python(source)
            namespace = {"bpy": bpy, "math": __import__("math"), "Vector": Vector}
            started = time.monotonic()
            timeout_ms = max(1, min(int(args.get("timeout_ms", 2000)), 10000))
            max_result_chars = max(128, min(int(args.get("max_result_chars", 4096)), 65536))
            code_hash = "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()
            with _execution_deadline(timeout_ms):
                exec(compile(source, "<toolbox-run-python>", "exec"), {"__builtins__": {}}, namespace)
            result = {
                "executed": True,
                "trusted": False,
                "replayable": False,
                "risk_level": "high",
                "code_hash": code_hash,
                "duration_ms": int((time.monotonic() - started) * 1000),
            }
            encoded = json.dumps(result, ensure_ascii=True, separators=(",", ":"))
            if len(encoded) > max_result_chars:
                raise ExecutorError("run_python result exceeds configured size", "policy_denied")
            return result
        raise ExecutorError(f"unsupported action: {action}", "unknown_action")


def _geometry_objects() -> list[Any]:
    return sorted(
        (obj for obj in bpy.context.scene.objects if obj.type in {"MESH", "CURVE", "SURFACE", "FONT"}),
        key=lambda item: _stable_uuid(item),
    )


def _object_dimensions(obj: Any) -> tuple[float, float, float]:
    bounds = _aabb(obj)
    return tuple(max(0.0, float(bounds["max"][i]) - float(bounds["min"][i])) for i in range(3))


def _resolve_scene_ref(ref: Any, objects: Iterable[Any]) -> Optional[Any]:
    if not isinstance(ref, str):
        return None
    for obj in objects:
        if ref in {obj.name, _stable_uuid(obj), obj.get(_REF_PROP)} or ref in _semantic_tags(obj):
            return obj
    return None


def _aabb_gap(left: Any, right: Any) -> float:
    a, b = _aabb(left), _aabb(right)
    gaps = []
    for axis in range(3):
        gaps.append(max(float(a["min"][axis]) - float(b["max"][axis]), float(b["min"][axis]) - float(a["max"][axis]), 0.0))
    return max(gaps)


def _geometry_points(obj: Any, limit: int = 256) -> list[Any]:
    points = []
    if obj.type == "MESH" and getattr(obj, "data", None) is not None:
        source = list(obj.data.vertices)
        stride = max(1, len(source) // limit)
        points = [obj.matrix_world @ source[i].co for i in range(0, len(source), stride)][:limit]
    elif obj.type == "CURVE" and getattr(obj, "data", None) is not None:
        for spline in obj.data.splines:
            source = spline.bezier_points if spline.type == "BEZIER" else spline.points
            for point in source:
                points.append(obj.matrix_world @ point.co.xyz)
                if len(points) >= limit:
                    return points
    return points


def _contact_distance(left: Any, right: Any) -> float:
    left_points, right_points = _geometry_points(left), _geometry_points(right)
    if not left_points or not right_points:
        return float("inf")
    best = float("inf")
    for first in left_points:
        for second in right_points:
            best = min(best, (first - second).length)
    return best


def _aabb_relationship(left: Any, right: Any) -> Dict[str, Any]:
    a, b = _aabb(left), _aabb(right)
    amin, amax, bmin, bmax = a["min"], a["max"], b["min"], b["max"]
    gaps = [max(bmin[i] - amax[i], amin[i] - bmax[i], 0.0) for i in range(3)]
    overlaps = [max(0.0, min(amax[i], bmax[i]) - max(amin[i], bmin[i])) for i in range(3)]
    penetration = min(overlaps) if all(value > 0.0 for value in overlaps) else 0.0
    return {
        "a": {"uuid": _stable_uuid(left), "name": left.name},
        "b": {"uuid": _stable_uuid(right), "name": right.name},
        "aabb_gap": round(float(math.sqrt(sum(value * value for value in gaps))), 8),
        "axis_gaps": [round(float(value), 8) for value in gaps],
        "penetration": round(float(penetration), 8),
        "overlap": bool(penetration > 0.0),
        "a_contained_in_b": all(amin[i] >= bmin[i] and amax[i] <= bmax[i] for i in range(3)),
        "b_contained_in_a": all(bmin[i] >= amin[i] and bmax[i] <= amax[i] for i in range(3)),
        "parent_relation": bool(left.parent == right or right.parent == left),
    }


def _surface_nearest(surface: Any, point_world: Any) -> Optional[tuple[Any, Any, float]]:
    tree = _surface_bvh(surface)
    if tree is None:
        return None
    nearest = tree.find_nearest(point_world)
    if nearest[0] is None:
        return None
    location, normal, _index, distance = nearest
    if normal is None or normal.length < 1e-9:
        return None
    return Vector(location), Vector(normal).normalized(), float(distance)


def _axis_alignment_degrees(obj: Any, axis: str, normal: Any) -> float:
    direction = obj.matrix_world.to_3x3() @ _axis_vector(axis)
    if direction.length < 1e-9 or normal.length < 1e-9:
        return 180.0
    dot = max(-1.0, min(1.0, float(direction.normalized().dot(normal.normalized()))))
    return math.degrees(math.acos(dot))


def _relation_check(declaration: Mapping[str, Any], objects: Mapping[str, Any]) -> Dict[str, Any]:
    left = objects.get(str(declaration.get("a"))); right = objects.get(str(declaration.get("b")))
    if left is None or right is None:
        return {"gate": False, "reason": "unknown_relation_target", "a": declaration.get("a"), "b": declaration.get("b")}
    item = _aabb_relationship(left, right)
    relation = str(declaration.get("relation", "overlap_free"))
    units = {"units": declaration.get("units") or _coordinate_system().get("units", "meters")}
    max_gap = _length_value(declaration.get("max_gap", 0.002), "relation.max_gap", units)
    max_penetration = _length_value(declaration.get("max_penetration", 0.0), "relation.max_penetration", units)
    min_overlap = _length_value(declaration.get("min_overlap", 0.0), "relation.min_overlap", units)
    normal_tolerance = float(declaration.get("normal_tolerance_degrees", 15.0))
    gap, penetration = float(item["aabb_gap"]), float(item["penetration"])
    passed = True; reason = None
    if relation == "attached":
        attachment = _load_json_prop(left, _ATTACHMENT_PROP, {})
        passed = bool(left.parent == right and isinstance(attachment, Mapping) and attachment.get("parent") == _stable_uuid(right))
        if passed:
            anchors = _load_json_prop(right, _ANCHORS_PROP, {})
            anchor = anchors.get(str(attachment.get("anchor"))) if isinstance(anchors, Mapping) else None
            if not isinstance(anchor, Mapping):
                passed = False
            else:
                normal = Vector(_as_float3(anchor.get("normal", (0, 0, 1)), "anchor.normal"))
                normal_world = (right.matrix_world.to_3x3() @ normal).normalized()
                anchor_world = right.matrix_world @ Vector(_as_float3(anchor.get("position"), "anchor.position"))
                contact_world = left.matrix_world @ Vector(_as_float3(attachment.get("contact_point", (0, 0, 0)), "contact_point"))
                expected = anchor_world + normal_world * float(attachment.get("clearance", 0.0))
                residual = (contact_world - expected).length
                item["contact_residual"] = round(float(residual), 8)
                passed = residual <= max_gap
                if passed and bool(attachment.get("align_to_normal", True)):
                    alignment = _axis_alignment_degrees(left, str(attachment.get("align_axis", "Z")), normal_world)
                    item["normal_alignment_degrees"] = round(float(alignment), 8)
                    passed = alignment <= normal_tolerance
        if passed:
            # Contact metadata alone is insufficient: the child's overall
            # geometry must also be close to the parent and must not be deeply
            # embedded.  This catches a rotated/translated child whose one
            # declared contact point happens to coincide by accident.
            passed = gap <= max_gap and penetration <= max_penetration
        if not passed: reason = "missing_or_misaligned_attachment"
    elif relation == "surface_contact":
        snap = _load_json_prop(left, _SNAP_PROP, {})
        passed = bool(isinstance(snap, Mapping) and snap.get("surface") == _stable_uuid(right))
        if passed:
            contact_local = Vector(_as_float3(snap.get("contact_point", (0, 0, 0)), "surface_snap.contact_point"))
            contact_world = left.matrix_world @ contact_local
            nearest = _surface_nearest(right, contact_world)
            if nearest is None:
                passed = False
            else:
                location, normal, surface_distance = nearest
                expected = location + normal * float(snap.get("offset", 0.0))
                residual = (contact_world - expected).length
                item["surface_distance"] = round(float(surface_distance), 8)
                item["contact_residual"] = round(float(residual), 8)
                passed = residual <= max_gap and gap <= max_gap and penetration <= max_penetration
                if passed and bool(snap.get("align_to_normal", True)):
                    alignment = _axis_alignment_degrees(left, str(snap.get("align_axis", "Z")), normal)
                    item["normal_alignment_degrees"] = round(float(alignment), 8)
                    passed = alignment <= normal_tolerance
        if not passed: reason = "surface_contact_not_resolved"
    elif relation == "supported_by":
        # Parentage describes ownership, not physical support.  A child can
        # remain parented while floating after a later transform, so require
        # an actual near-contact in addition to the AABB/penetration limits.
        surface_distance = _contact_distance(left, right)
        item["surface_distance"] = round(float(surface_distance), 8) if math.isfinite(surface_distance) else None
        passed = bool(
            gap <= max_gap
            and penetration <= max_penetration
            and math.isfinite(surface_distance)
            and surface_distance <= max_gap + 1e-8
        )
        if not passed: reason = "unsupported_component"
    elif relation in {"disjoint", "overlap_free"}:
        passed = penetration <= max_penetration and penetration >= min_overlap
        if not passed: reason = "overlapping_components"
    else:
        passed, reason = False, "unknown_relation"
    item.update({"relation": relation, "gate": bool(passed), "reason": reason, "limits": {"max_gap": max_gap, "max_penetration": max_penetration, "min_overlap": min_overlap, "normal_tolerance_degrees": normal_tolerance, "units": str(units.get("units", "meters"))}})
    return item


def _documented_pair_relations(left: Any, right: Any) -> list[Dict[str, Any]]:
    """Validate attachment/surface metadata for one spatial object pair.

    Raw Blender parentage is not a contact declaration.  Only the explicit
    Toolbox attachment or surface-snap record can document a relationship,
    and that record is still checked for residual distance and orientation.
    """
    objects = {_stable_uuid(obj): obj for obj in bpy.context.scene.objects}
    objects.update({obj.name: obj for obj in bpy.context.scene.objects})
    objects.update({str(obj.get(_REF_PROP)): obj for obj in bpy.context.scene.objects if obj.get(_REF_PROP)})
    checks: list[Dict[str, Any]] = []
    for candidate, other in ((left, right), (right, left)):
        attachment = _load_json_prop(candidate, _ATTACHMENT_PROP, {})
        if isinstance(attachment, Mapping) and str(attachment.get("parent")) == _stable_uuid(other):
            checks.append(_relation_check({"a": _stable_uuid(candidate), "b": _stable_uuid(other), "relation": str(attachment.get("relation") or "attached")}, objects))
        snap = _load_json_prop(candidate, _SNAP_PROP, {})
        if isinstance(snap, Mapping) and str(snap.get("surface")) == _stable_uuid(other):
            checks.append(_relation_check({"a": _stable_uuid(candidate), "b": _stable_uuid(other), "relation": "surface_contact"}, objects))
    return checks


def _inspect_relationships(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    selected = args.get("targets")
    if isinstance(selected, str): selected = [selected]
    candidates = [obj for obj in bpy.context.scene.objects if obj.type in {"MESH", "CURVE", "SURFACE", "FONT"}]
    if isinstance(selected, list) and selected:
        allowed = {_object_by_ref(ref) for ref in selected}
        candidates = [obj for obj in candidates if obj in allowed]
    objects = {_stable_uuid(obj): obj for obj in bpy.context.scene.objects}
    objects.update({obj.name: obj for obj in bpy.context.scene.objects})
    # Relationship declarations are normally authored with stable ``id`` /
    # ``ref`` handles.  Keep those aliases in the resolver alongside UUIDs and
    # display names; otherwise a valid declaration is reported as unknown.
    objects.update({str(obj.get(_REF_PROP)): obj for obj in bpy.context.scene.objects if obj.get(_REF_PROP)})
    declarations = [item for item in (args.get("relations") or []) if isinstance(item, Mapping)]
    def canonical_ref(value: Any) -> Optional[str]:
        obj = objects.get(str(value)) if value is not None else None
        return _stable_uuid(obj) if obj is not None else (str(value) if value is not None else None)
    canonical_declarations = [
        {**dict(item), "a": canonical_ref(item.get("a")), "b": canonical_ref(item.get("b"))}
        for item in declarations
    ]
    checks = [_relation_check(item, objects) for item in canonical_declarations]
    declared_pairs = {frozenset((str(item.get("a")), str(item.get("b")))) for item in canonical_declarations if item.get("a") is not None and item.get("b") is not None}
    metadata_checks = []
    for obj in candidates:
        attachment = _load_json_prop(obj, _ATTACHMENT_PROP, None)
        if isinstance(attachment, Mapping) and attachment.get("parent"):
            pair = frozenset((_stable_uuid(obj), str(attachment.get("parent"))))
            if pair not in declared_pairs:
                metadata_checks.append(_relation_check({"a": _stable_uuid(obj), "b": str(attachment.get("parent")), "relation": str(attachment.get("relation") or "attached")}, objects))
        snap = _load_json_prop(obj, _SNAP_PROP, None)
        if isinstance(snap, Mapping) and snap.get("surface"):
            pair = frozenset((_stable_uuid(obj), str(snap.get("surface"))))
            if pair not in declared_pairs:
                metadata_checks.append(_relation_check({"a": _stable_uuid(obj), "b": str(snap.get("surface")), "relation": "surface_contact"}, objects))
    pairwise = []
    max_pair_gap = _length_value(args.get("max_pair_gap", 0.05), "max_pair_gap")
    audit_spatial = bool(args.get("audit_spatial", False))
    for index, left in enumerate(candidates):
        for right in candidates[index + 1:]:
            if bool(left.hide_render and left.hide_viewport) or bool(right.hide_render and right.hide_viewport):
                continue
            item = _aabb_relationship(left, right)
            near = bool(item["overlap"] or item["aabb_gap"] <= max_pair_gap or item["parent_relation"])
            if not (bool(args.get("include_disjoint", False)) or near):
                continue
            declared = frozenset((_stable_uuid(left), _stable_uuid(right))) in declared_pairs
            item["declared"] = declared
            if audit_spatial and not declared:
                metadata_checks_for_pair = _documented_pair_relations(left, right)
                item["metadata_relations"] = metadata_checks_for_pair
                if metadata_checks_for_pair:
                    item["gate"] = all(bool(check.get("gate")) for check in metadata_checks_for_pair)
                    if not item["gate"]:
                        item["reason"] = "documented_relation_failed"
                else:
                    item["gate"] = bool(not item["overlap"] and item["aabb_gap"] > max_pair_gap)
                    if not item["gate"]: item["reason"] = "undocumented_spatial_pair"
            else:
                item["gate"] = True
            pairwise.append(item)
    pair_gate = all(bool(item.get("gate")) for item in pairwise) if audit_spatial else True
    return {"gate": bool(all(bool(item.get("gate")) for item in [*checks, *metadata_checks]) and pair_gate), "checks": checks, "metadata_checks": metadata_checks, "pairs": pairwise, "audit_spatial": audit_spatial, "max_pair_gap": max_pair_gap, "targets": [{"uuid": _stable_uuid(obj), "name": obj.name} for obj in candidates], "coordinate_system": _coordinate_system()}


def _assembly_check(objects: list[Any], task_spec: Mapping[str, Any], args: Mapping[str, Any]) -> Dict[str, Any]:
    units = {"units": _coordinate_system().get("units", "meters")}
    declaration = args.get("assembly", task_spec.get("assembly"))
    contacts = args.get("contacts", task_spec.get("contacts"))
    declared_parts = None
    if isinstance(declaration, Mapping):
        contacts = declaration.get("contacts", contacts)
        declared_parts = declaration.get("parts")
    if not isinstance(contacts, (list, tuple)) or not contacts:
        if isinstance(declared_parts, (list, tuple)):
            unknown = [ref for ref in declared_parts if _resolve_scene_ref(ref, objects) is None]
            return {"gate": not unknown, "status": "parts_declared", "parts": len(declared_parts), "contacts": [], "components": len(objects), "failures": [{"part": ref, "reason": "unknown_part"} for ref in unknown]}
        return {"gate": True, "status": "not_declared", "parts": len(objects), "contacts": [], "components": len(objects)}
    resolved = []
    failures = []
    parent = list(range(len(objects)))
    index = {id(obj): i for i, obj in enumerate(objects)}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for raw in contacts:
        if isinstance(raw, Mapping):
            pair = raw.get("parts") or raw.get("objects") or [raw.get("a"), raw.get("b")]
            max_gap = _length_value(raw.get("max_gap", raw.get("gap", 0.0)) or 0.0, "assembly.max_gap", units)
        else:
            pair, max_gap = raw, 0.0
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            failures.append({"contact": pair, "reason": "invalid_contact"})
            continue
        left, right = _resolve_scene_ref(pair[0], objects), _resolve_scene_ref(pair[1], objects)
        if left is None or right is None or left is right:
            failures.append({"contact": list(pair), "reason": "unknown_or_duplicate_part"})
            continue
        gap = _aabb_gap(left, right)
        ok = gap <= max_gap + 1e-5
        row = {"a": _stable_uuid(left), "b": _stable_uuid(right), "gap": round(gap, 8), "max_gap": max_gap, "ok": ok}
        resolved.append(row)
        if not ok:
            failures.append({**row, "reason": "parts_do_not_contact"})
        else:
            union(index[id(left)], index[id(right)])
    components = len({find(i) for i in range(len(objects))}) if objects else 0
    contact_tolerance = _length_value(args.get("contact_tolerance", task_spec.get("contact_tolerance", 1e-3)) or 1e-3, "contact_tolerance", units)
    for row in resolved:
        left = _resolve_scene_ref(row["a"], objects)
        right = _resolve_scene_ref(row["b"], objects)
        distance = _contact_distance(left, right) if left is not None and right is not None else float("inf")
        row["surface_distance"] = round(distance, 8) if distance != float("inf") else None
        if row["ok"] and distance > contact_tolerance:
            row["ok"] = False
            failures.append({**row, "reason": "aabb_overlap_without_surface_contact"})
    require_single = bool(args.get("require_single_assembly", task_spec.get("require_single_assembly", True)))
    if require_single and components > 1:
        failures.append({"reason": "multiple_connected_components", "components": components})
    return {
        "gate": not failures,
        "status": "checked",
        "parts": len(objects),
        "contacts": resolved,
        "components": components,
        "failures": failures,
    }


def _opening_check(objects: list[Any], task_spec: Mapping[str, Any], args: Mapping[str, Any], topologies: list[Dict[str, Any]], required_tags_lock: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    required = bool(args.get("require_openings", task_spec.get("require_openings", False)))
    required_tags = set(_normalize_tags(task_spec.get("required_tags"))) | set(_normalize_tags(args.get("required_tags", ()))) | set(required_tags_lock or ())
    required = required or bool(required_tags & {"opening", "soundhole"})
    candidates = []
    mesh_topologies = {id(obj): topology for obj, topology in zip((item for item in objects if item.type == "MESH"), topologies)}
    for obj in objects:
        tags = set(_semantic_tags(obj))
        if "opening" in tags or "soundhole" in tags or "opening" in obj.name.lower() or "soundhole" in obj.name.lower():
            topology = mesh_topologies.get(id(obj))
            if topology is not None:
                has_hole = topology["boundary_edges"] > 0 or topology["euler_chi"] <= 0 or "BOOLEAN" in [m.type for m in obj.modifiers]
                detail = {"euler_chi": topology["euler_chi"], "boundary_edges": topology["boundary_edges"]}
            else:
                cyclic = bool(obj.type == "CURVE" and getattr(obj, "data", None) is not None and any(spline.use_cyclic_u for spline in obj.data.splines))
                has_hole = cyclic and bool(getattr(obj.data, "bevel_depth", 0.0) > 0.0)
                detail = {"cyclic_curve": cyclic}
            candidates.append({"uuid": _stable_uuid(obj), "name": obj.name, "has_hole": bool(has_hole), **detail})
    gate = (not required) or any(item["has_hole"] for item in candidates)
    if required and not gate:
        # A boolean opening may be represented on the main body rather than
        # as a separately tagged rim object.  A closed genus >= 1 shell is
        # evidence of a through opening; a solid cylinder (Euler chi=2) is not.
        for obj, topology in mesh_topologies.items():
            if topology["euler_chi"] <= 0:
                gate = True
                break
    return {"gate": gate, "required": required, "candidates": candidates, "failures": [] if gate else [{"reason": "no_real_opening"}]}


def _proportion_check(objects: list[Any], task_spec: Mapping[str, Any], args: Mapping[str, Any]) -> Dict[str, Any]:
    declaration = args.get("proportions", task_spec.get("proportions", {}))
    checks, failures = [], []
    if not isinstance(declaration, Mapping):
        declaration = {}
    for ref, rule in declaration.items():
        obj = _resolve_scene_ref(ref, objects)
        if obj is None or not isinstance(rule, Mapping):
            failures.append({"target": ref, "reason": "unknown_proportion_target"})
            continue
        dims = _object_dimensions(obj)
        tolerance = float(rule.get("tolerance", 0.1))
        expected = rule.get("dimensions") or rule.get("size")
        ok = True
        if isinstance(expected, (list, tuple)) and len(expected) == 3:
            ok = all(abs(dims[i] - float(expected[i])) <= max(abs(float(expected[i])) * tolerance, 1e-6) for i in range(3))
        minimum_dimensions = rule.get("min_dimensions") or rule.get("minimum")
        maximum_dimensions = rule.get("max_dimensions") or rule.get("maximum")
        if isinstance(minimum_dimensions, (list, tuple)) and len(minimum_dimensions) == 3:
            ok = ok and all(dims[i] >= float(minimum_dimensions[i]) for i in range(3))
        if isinstance(maximum_dimensions, (list, tuple)) and len(maximum_dimensions) == 3:
            ok = ok and all(dims[i] <= float(maximum_dimensions[i]) for i in range(3))
        ratios = rule.get("ratios")
        if isinstance(ratios, Mapping):
            for key, target in ratios.items():
                try:
                    i, j = (int(value) for value in str(key).split(":", 1))
                    actual = dims[i] / dims[j] if dims[j] > 1e-9 else 0.0
                    ok = ok and abs(actual - float(target)) <= max(abs(float(target)) * tolerance, 1e-6)
                except (ValueError, TypeError, IndexError, ZeroDivisionError):
                    ok = False
        row = {"target": _stable_uuid(obj), "dimensions": [round(v, 6) for v in dims], "ok": ok}
        checks.append(row)
        if not ok:
            failures.append({**row, "reason": "proportion_mismatch"})
    degenerate = [obj.name for obj in objects if obj.type == "MESH" and min(_object_dimensions(obj)) <= 1e-9]
    if degenerate:
        failures.append({"reason": "degenerate_geometry", "objects": degenerate})
    return {"gate": not failures, "score": 1.0 if not failures else 0.0, "checks": checks, "failures": failures}


def _silhouette_check(objects: list[Any], task_spec: Mapping[str, Any], args: Mapping[str, Any]) -> Dict[str, Any]:
    extents = [0.0, 0.0, 0.0]
    for obj in objects:
        dims = _object_dimensions(obj)
        extents = [max(extents[i], dims[i]) for i in range(3)]
    points = [point for obj in objects for point in _geometry_points(obj, limit=512)]
    views = args.get("silhouette_views", task_spec.get("silhouette_views"))
    checks = []
    if isinstance(views, (list, tuple)) and views:
        for view in views:
            if isinstance(view, Mapping):
                axes = view.get("axes", [0, 2])
                name = str(view.get("name", "view"))
                expected = view.get("ratio")
            else:
                axes, name, expected = [0, 2], str(view), None
            try:
                width, height = extents[int(axes[0])], extents[int(axes[1])]
                ratio = width / height if height > 1e-9 else 0.0
                tolerance = float(view.get("tolerance", 0.15)) if isinstance(view, Mapping) else 0.15
                coverage = 0.0
                if points and width > 1e-9 and height > 1e-9:
                    projected = [(float(point[int(axes[0])]), float(point[int(axes[1])])) for point in points]
                    pmin = [min(item[axis] for item in projected) for axis in (0, 1)]
                    pmax = [max(item[axis] for item in projected) for axis in (0, 1)]
                    occupied = set()
                    for px, py in projected:
                        ix = min(15, max(0, int((px - pmin[0]) / max(pmax[0] - pmin[0], 1e-9) * 16)))
                        iy = min(15, max(0, int((py - pmin[1]) / max(pmax[1] - pmin[1], 1e-9) * 16)))
                        occupied.add((ix, iy))
                    coverage = len(occupied) / 256.0
                minimum_coverage = float(view.get("min_coverage", 0.0)) if isinstance(view, Mapping) else 0.0
                ok = width > 1e-9 and height > 1e-9 and (expected is None or abs(ratio - float(expected)) <= max(abs(float(expected)) * tolerance, 1e-6)) and coverage + 1e-9 >= minimum_coverage
            except (TypeError, ValueError, IndexError, ZeroDivisionError):
                ratio, coverage, ok = 0.0, 0.0, False
            checks.append({"name": name, "ratio": round(ratio, 6), "coverage": round(coverage, 6), "ok": ok})
    else:
        checks = [{"name": name, "occupied": extents[a] > 1e-9 and extents[b] > 1e-9, "ratio": round(extents[a] / extents[b], 6) if extents[b] > 1e-9 else 0.0} for name, a, b in (("front", 0, 2), ("side", 1, 2), ("top", 0, 1))]
    occupied = sum(1 for item in checks if item.get("ok", item.get("occupied", False)))
    # A volumetric mesh should occupy at least two orthographic projections;
    # a strand/card-only scene is legitimately one-dimensional and is checked
    # for a single non-degenerate projection unless explicit views were asked.
    required_views = len(checks) if isinstance(views, (list, tuple)) and views else (2 if any(obj.type == "MESH" for obj in objects) else 1)
    gate = bool(objects) and occupied >= required_views
    return {"gate": gate, "score": 1.0 if gate else 0.0, "method": "orthographic_geometry_projection", "views": checks, "extents": [round(v, 6) for v in extents]}


def _region_detail_stats(obj: Any, selection: Mapping[str, Any]) -> Dict[str, Any]:
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bm.normal_update()
        verts, edges, _ = _selection_parts(bm, selection, obj=obj, default_all=False)
        if not verts:
            return {"vertices": 0, "edges": 0, "relief": 0.0, "max_edge_length": 0.0, "min_edge_length": 0.0, "boundary_edges": 0}
        average_normal = sum((vertex.normal for vertex in verts), Vector((0, 0, 0)))
        average_normal = average_normal.normalized() if average_normal.length > 1e-9 else Vector((0, 0, 1))
        projections = [float(vertex.co.dot(average_normal)) for vertex in verts]
        lengths = [float(edge.calc_length()) for edge in edges]
        boundary = sum(1 for edge in edges if edge.is_boundary)
        return {
            "vertices": len(verts), "edges": len(edges), "relief": round(max(projections) - min(projections), 8),
            "max_edge_length": round(max(lengths, default=0.0), 8), "min_edge_length": round(min(lengths, default=0.0), 8),
            "boundary_edges": boundary,
        }
    finally:
        bm.free()


def _detail_check(objects: list[Any], task_spec: Mapping[str, Any], args: Mapping[str, Any]) -> Dict[str, Any]:
    feature_sizes = args.get("feature_sizes", task_spec.get("feature_sizes", ()))
    if not isinstance(feature_sizes, (list, tuple)):
        feature_sizes = []
    minimum = min((min(_object_dimensions(obj)) for obj in objects), default=0.0)
    checks = [{"required": float(size), "resolved": minimum >= float(size), "minimum_scene_dimension": round(minimum, 6)} for size in feature_sizes]
    failures = [{**item, "reason": "feature_below_resolution"} for item in checks if not item["resolved"]]
    regions = args.get("detail_regions", task_spec.get("detail_regions", ()))
    if not isinstance(regions, (list, tuple)):
        regions = []
    for declaration in regions:
        if not isinstance(declaration, Mapping):
            failures.append({"reason": "invalid_detail_region"})
            continue
        ref = declaration.get("target") or declaration.get("object")
        obj = _resolve_scene_ref(ref, objects)
        if obj is None or obj.type != "MESH":
            failures.append({"target": ref, "reason": "unknown_detail_target"})
            continue
        stats = _region_detail_stats(obj, declaration.get("selection") or {})
        row = {"target": _stable_uuid(obj), **stats}
        row["ok"] = True
        if "min_vertices" in declaration:
            row["ok"] = row["ok"] and stats["vertices"] >= int(declaration["min_vertices"])
        if "min_relief" in declaration:
            row["ok"] = row["ok"] and stats["relief"] >= float(declaration["min_relief"])
        if "max_edge_length" in declaration:
            row["ok"] = row["ok"] and stats["max_edge_length"] <= float(declaration["max_edge_length"]) + 1e-8
        if "min_edge_length" in declaration:
            row["ok"] = row["ok"] and stats["min_edge_length"] >= float(declaration["min_edge_length"]) - 1e-8
        if declaration.get("require_closed"):
            row["ok"] = row["ok"] and stats["boundary_edges"] == 0
        checks.append({"detail_region": row})
        if not row["ok"]:
            failures.append({**row, "reason": "detail_region_constraint"})
    gate = not failures
    return {"gate": gate, "score": 1.0 if gate else 0.0, "checks": checks, "failures": failures, "warnings": [] if gate else [{"reason": "detail_constraint_failed"}]}


def _quality_audit(args: Mapping[str, Any], *, registered_contract: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Return one bounded, domain-neutral quality census."""
    all_objects = _geometry_objects()
    raw_targets = args.get("targets")
    requested = [raw_targets] if isinstance(raw_targets, str) else [str(item) for item in raw_targets] if isinstance(raw_targets, (list, tuple)) else []
    selected: list[Any] = []
    missing: list[str] = []
    if requested:
        for ref in requested:
            obj = _resolve_scene_ref(ref, all_objects)
            (selected if obj is not None else missing).append(obj if obj is not None else ref)
    else:
        selected = list(all_objects)
    try:
        max_objects = max(1, min(1024, int(args.get("max_objects", 256))))
    except (TypeError, ValueError):
        max_objects = 256
    truncated = len(selected) > max_objects
    selected = selected[:max_objects]
    contract: Mapping[str, Any] = registered_contract if isinstance(registered_contract, Mapping) else {}
    if not contract and isinstance(args.get("quality"), Mapping):
        contract = args["quality"]
    feature_scales = contract.get("feature_scales") or contract.get("feature_sizes") or args.get("feature_scales") or args.get("feature_sizes") or []
    if isinstance(feature_scales, (int, float)):
        feature_scales = [feature_scales]
    if not isinstance(feature_scales, (list, tuple)):
        feature_scales = []
    try:
        feature_scales = [float(value) for value in feature_scales if float(value) > 0 and math.isfinite(float(value))]
    except (TypeError, ValueError):
        feature_scales = []
    rep = contract.get("representation") if isinstance(contract.get("representation"), Mapping) else {}
    primary_refs = contract.get("primary_refs") or rep.get("primary_refs") or []
    if isinstance(primary_refs, str):
        primary_refs = [primary_refs]
    primary_ref_set = {str(value) for value in primary_refs} if isinstance(primary_refs, (list, tuple)) else set()
    rows: list[Dict[str, Any]] = []
    issue_count = 0
    for obj in selected:
        topology = _topology(obj) if obj.type == "MESH" and obj.data is not None else None
        representation = str(obj.get(_REPRESENTATION_PROP) or "unknown")
        role = str(obj.get(_ROLE_PROP) or "")
        is_primary = role == "primary" or str(obj.get(_REF_PROP) or "") in primary_ref_set
        has_material = bool(getattr(obj, "material_slots", ())) and any(slot.material for slot in obj.material_slots)
        uv_layers = len(obj.data.uv_layers) if obj.type == "MESH" and obj.data is not None else 0
        edge_min = float(topology.get("min_edge_length", 0.0)) if topology else 0.0
        samples = [round(float(scale) / max(edge_min, 1e-12), 4) for scale in feature_scales] if feature_scales else []
        issues: list[Dict[str, Any]] = []
        if is_primary and representation == "primitive":
            issues.append({"reason": "primary_primitive_carrier", "repair": "replace_with_control_mesh_or_native_carrier"})
        if topology:
            for key in ("nonmanifold_edges", "zero_area_faces", "duplicate_faces", "non_finite_vertices", "loose_vertices"):
                if topology.get(key, 0):
                    issues.append({"reason": key, "count": topology[key], "repair": "mesh.repair"})
        if any(abs(float(value) - 1.0) > 1e-5 for value in obj.scale):
            issues.append({"reason": "unapplied_scale", "repair": "object.transform_apply"})
        if obj.type == "MESH" and not has_material:
            issues.append({"reason": "missing_material", "repair": "material.create_and_assign"})
        if obj.type == "MESH" and not uv_layers:
            issues.append({"reason": "missing_uv", "repair": "uv.unwrap"})
        for index, sample_count in enumerate(samples):
            if sample_count < 4.0:
                issues.append({"reason": "feature_under_sampled", "feature_scale": feature_scales[index], "samples": sample_count, "repair": "increase_carrier_resolution"})
        issue_count += len(issues)
        dims = _object_dimensions(obj)
        rows.append({"uuid": _stable_uuid(obj), "ref": obj.get(_REF_PROP), "name": obj.name, "type": obj.type, "role": role or None, "representation": representation, "quality_stage": obj.get(_QUALITY_STAGE_PROP), "dimensions": [round(float(value), 6) for value in dims], "topology": topology, "normals": {"discontinuities": None if not topology else "reported_in_topology_audit"}, "transforms": {"scale": [round(float(value), 6) for value in obj.scale], "applied_scale": not any(abs(float(value) - 1.0) > 1e-5 for value in obj.scale)}, "materials": [slot.material.name for slot in obj.material_slots if slot.material], "has_material": has_material, "uv_layers": uv_layers, "feature_scales": feature_scales, "samples_per_feature": samples, "issues": issues})
    contacts: list[Dict[str, Any]] = []
    if bool(args.get("include_contacts", True)):
        try:
            tolerance = float(args.get("contact_tolerance", 0.02))
        except (TypeError, ValueError):
            tolerance = 0.02
        try:
            max_contacts = max(1, min(8192, int(args.get("max_contacts", 1024))))
        except (TypeError, ValueError):
            max_contacts = 1024
        for index, left in enumerate(selected):
            for right in selected[index + 1:]:
                gap = _aabb_gap(left, right)
                if gap <= tolerance:
                    contacts.append({"left": _stable_uuid(left), "right": _stable_uuid(right), "gap": round(float(gap), 6), "touching": gap <= 1e-5})
                    if len(contacts) >= max_contacts:
                        break
            if len(contacts) >= max_contacts:
                break
    negative_space: Dict[str, Any] = {"method": "aabb_occupancy_proxy", "status": "unknown"}
    if bool(args.get("include_negative_space", True)) and selected:
        bounds = [_aabb(obj) for obj in selected]
        low = [min(float(item["min"][axis]) for item in bounds) for axis in range(3)]
        high = [max(float(item["max"][axis]) for item in bounds) for axis in range(3)]
        envelope_volume = math.prod(max(0.0, high[axis] - low[axis]) for axis in range(3))
        occupied_volume = sum(math.prod(max(0.0, float(item["max"][axis]) - float(item["min"][axis])) for axis in range(3)) for item in bounds)
        occupancy = min(1.0, occupied_volume / envelope_volume) if envelope_volume > 1e-12 else 0.0
        negative_space = {"method": "aabb_occupancy_proxy", "status": "proxy", "envelope_volume": round(envelope_volume, 6), "occupied_aabb_volume": round(occupied_volume, 6), "negative_space_ratio": round(max(0.0, 1.0 - occupancy), 6)}
    silhouette = _silhouette_check(selected, contract, {}) if selected else {"gate": False, "views": [], "extents": [0.0, 0.0, 0.0]}
    score = round(max(0.0, 1.0 - min(1.0, 0.08 * issue_count)), 4) if selected else 0.0
    return {"audit_version": "quality_audit.v1", "status": "checked", "count": len(selected), "truncated": truncated, "missing": missing, "quality_score": score, "issue_count": issue_count, "objects": rows, "contacts": contacts, "negative_space": negative_space, "silhouette": silhouette, "unknown": ["reference_fit", "true_surface_intersection", "aesthetic_similarity"], "recommendations": ["repair_first_issue_per_object", "resolve_primary_carriers_before_detail", "rerun_after_each_stage"]}


def _quality_check(
    objects: list[Any],
    meshes: list[Any],
    task_spec: Mapping[str, Any],
    args: Mapping[str, Any],
    *,
    topology_gate: bool,
    topologies: list[Dict[str, Any]],
    silhouette: Mapping[str, Any],
    detail: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply the opt-in, domain-neutral quality contract.

    The checker is intentionally evidence-driven.  It cannot judge taste, but
    it can prevent a primary form from being represented by an undeclared pile
    of primitives and can keep missing contract dimensions explicitly unknown.
    """
    raw = args.get("quality") or task_spec.get("quality") or task_spec.get("quality_contract")
    if not isinstance(raw, Mapping):
        return {
            "gate": True,
            "enforced": False,
            "status": "advisory",
            "score": None,
            "stages": {},
            "unknown": ["representation", "primary_carrier", "secondary_junctions", "tertiary_detail", "technical_policy", "evidence"],
            "first_failure": None,
            "repair_action": None,
        }
    contract = raw
    enforce = bool(contract.get("enforce", False))
    representation = contract.get("representation")
    if not isinstance(representation, Mapping):
        representation = {}
    primary_refs = contract.get("primary_refs") or representation.get("primary_refs") or contract.get("carrier_refs") or []
    if isinstance(primary_refs, str):
        primary_refs = [primary_refs]
    if not isinstance(primary_refs, (list, tuple)):
        primary_refs = []
    primary_refs = [str(ref) for ref in primary_refs if str(ref).strip()]
    secondary_refs = contract.get("secondary_refs") or contract.get("required_secondary_refs") or []
    if isinstance(secondary_refs, str):
        secondary_refs = [secondary_refs]
    if not isinstance(secondary_refs, (list, tuple)):
        secondary_refs = []
    secondary_refs = [str(ref) for ref in secondary_refs if str(ref).strip()]
    failures_by_stage: Dict[str, list[Dict[str, Any]]] = {stage: [] for stage in _QUALITY_STAGES}
    unknown: list[str] = []

    if not primary_refs:
        failures_by_stage["structure"].append({"reason": "missing_primary_carrier_plan"})
    primary_objects: list[Any] = []
    for ref in primary_refs:
        obj = _resolve_scene_ref(ref, objects)
        if obj is None:
            failures_by_stage["structure"].append({"target": ref, "reason": "unknown_primary_carrier"})
            continue
        primary_objects.append(obj)
        if obj.type not in {"MESH", "CURVE", "SURFACE", "META", "VOLUME"}:
            failures_by_stage["primary"].append({"target": ref, "reason": "primary_carrier_not_geometry", "type": obj.type})
            continue
        if obj.type == "MESH":
            vertex_count = len(obj.data.vertices) if obj.data is not None else 0
        elif obj.type == "CURVE":
            vertex_count = sum(len(spline.bezier_points if spline.type == "BEZIER" else spline.points) for spline in obj.data.splines)
        else:
            vertex_count = 0
        try:
            min_vertices = int(contract.get("min_primary_vertices", 64))
        except (TypeError, ValueError):
            min_vertices = 64
        if min_vertices > 0 and vertex_count < min_vertices and str(representation.get("kind", "")).lower() not in {"primitive", "native_generator"}:
            failures_by_stage["primary"].append({"target": ref, "reason": "primary_carrier_under_resolved", "vertices": vertex_count, "required": min_vertices})
        origin = str(obj.get(_ORIGIN_PROP) or "")
        allow_primitive = bool(representation.get("allow_primitive_primary", contract.get("allow_primitive_primary", False))) or str(representation.get("kind", "")).lower() == "primitive"
        if origin.startswith("primitive:") and not allow_primitive:
            failures_by_stage["primary"].append({"target": ref, "reason": "primary_primitive_pile_disallowed", "origin": origin, "repair": "replace_with_control_mesh_or_native_carrier"})

    required_parts = contract.get("required_semantic_parts") or []
    if isinstance(required_parts, str):
        required_parts = [required_parts]
    present_tags = {tag for obj in objects for tag in _semantic_tags(obj)}
    missing_parts = sorted({str(tag) for tag in required_parts} - present_tags) if isinstance(required_parts, (list, tuple)) else []
    if missing_parts:
        failures_by_stage["structure"].append({"reason": "missing_semantic_parts", "missing": missing_parts})

    for ref in secondary_refs:
        if _resolve_scene_ref(ref, objects) is None:
            failures_by_stage["secondary"].append({"target": ref, "reason": "unknown_secondary_feature"})
    if bool(contract.get("require_secondary", False)) and not secondary_refs:
        failures_by_stage["secondary"].append({"reason": "missing_secondary_feature_plan"})

    quality_regions = contract.get("detail_regions")
    if quality_regions is not None and not isinstance(quality_regions, (list, tuple)):
        failures_by_stage["tertiary"].append({"reason": "invalid_detail_region_plan"})
    elif isinstance(quality_regions, (list, tuple)) and quality_regions:
        quality_detail = _detail_check(objects, {"detail_regions": list(quality_regions), "feature_sizes": contract.get("feature_scales", [])}, {})
        if not quality_detail.get("gate"):
            failures_by_stage["tertiary"].extend(list(quality_detail.get("failures", [])))
    elif bool(contract.get("require_detail_plan", False)):
        failures_by_stage["tertiary"].append({"reason": "missing_detail_carrier_plan"})
    elif not detail.get("checks"):
        unknown.append("tertiary_detail")

    evidence = contract.get("evidence") if isinstance(contract.get("evidence"), Mapping) else {}
    reference_views = contract.get("reference_views") or evidence.get("views") or task_spec.get("silhouette_views") or []
    if isinstance(reference_views, str):
        reference_views = [reference_views]
    if not isinstance(reference_views, (list, tuple)):
        reference_views = []
    try:
        minimum_views = int(evidence.get("min_views", contract.get("min_evidence_views", 3)))
    except (TypeError, ValueError):
        minimum_views = 3
    if not reference_views:
        unknown.append("evidence")
        if enforce:
            failures_by_stage["evidence"].append({"reason": "missing_reference_views"})
    elif len(reference_views) < max(1, minimum_views):
        failures_by_stage["evidence"].append({"reason": "insufficient_reference_views", "views": len(reference_views), "required": minimum_views})
    elif not silhouette.get("gate", False):
        failures_by_stage["evidence"].append({"reason": "silhouette_evidence_failed"})

    technical = contract.get("technical") if isinstance(contract.get("technical"), Mapping) else {}
    technical_declared = bool(technical)
    target_objects = primary_objects or meshes
    if bool(technical.get("require_topology", False)) or bool(technical.get("strict_topology", False)):
        topology_failures = []
        for index, item in enumerate(topologies):
            bad = {key: item.get(key, 0) for key in ("nonmanifold_edges", "zero_area_faces", "duplicate_faces", "non_finite_vertices", "loose_vertices") if item.get(key, 0)}
            if bad:
                topology_failures.append({"mesh_index": index, "diagnostics": bad})
        if not topology_gate or topology_failures:
            failures_by_stage["technical"].append({"reason": "topology_policy_failed", "objects": topology_failures})
    if bool(technical.get("require_material", False)):
        missing_materials = [_stable_uuid(obj) for obj in target_objects if not getattr(obj, "material_slots", None) or not any(slot.material for slot in obj.material_slots)]
        if missing_materials:
            failures_by_stage["technical"].append({"reason": "missing_material", "objects": missing_materials})
    if bool(technical.get("require_uv", False)):
        missing_uv = [_stable_uuid(obj) for obj in target_objects if obj.type == "MESH" and (obj.data is None or not obj.data.uv_layers)]
        if missing_uv:
            failures_by_stage["technical"].append({"reason": "missing_uv", "objects": missing_uv})
    if bool(technical.get("require_unit_scale", False)):
        non_unit = [_stable_uuid(obj) for obj in target_objects if any(abs(float(value) - 1.0) > 1e-5 for value in obj.scale)]
        if non_unit:
            failures_by_stage["technical"].append({"reason": "unapplied_scale", "objects": non_unit, "repair": "object.transform_apply"})
    if not technical_declared:
        unknown.append("technical_policy")

    stages: Dict[str, Dict[str, Any]] = {}
    for stage in _QUALITY_STAGES:
        failures = failures_by_stage[stage]
        declared = stage in {"structure", "primary"} or bool(failures) or stage == "evidence" and bool(reference_views) or stage == "technical" and technical_declared or stage == "tertiary" and (bool(quality_regions) or bool(contract.get("require_detail_plan"))) or stage == "secondary" and (bool(secondary_refs) or bool(contract.get("require_secondary")))
        if not declared:
            stages[stage] = {"status": "unknown", "gate": True, "score": None, "failures": []}
            continue
        stages[stage] = {"status": "fail" if failures else "pass", "gate": not failures, "score": 0.0 if failures else 1.0, "failures": failures}

    required_stages = contract.get("required_stages") or ["structure", "primary", "evidence"]
    if isinstance(required_stages, str):
        required_stages = [required_stages]
    if not isinstance(required_stages, (list, tuple)):
        required_stages = ["structure", "primary", "evidence"]
    required_stages = [str(stage).strip().lower() for stage in required_stages if str(stage).strip() in _QUALITY_STAGES]
    for stage in required_stages:
        if stages[stage].get("status") == "unknown":
            stages[stage] = {
                "status": "fail",
                "gate": False,
                "score": 0.0,
                "failures": [{"reason": "required_stage_undeclared", "stage": stage}],
            }

    first_failure = None
    for stage in _QUALITY_STAGES:
        if stages[stage]["status"] == "fail":
            first_failure = stage
            break
    known_scores = [float(item["score"]) for item in stages.values() if item.get("score") is not None]
    unknown_stage_count = sum(1 for item in stages.values() if item.get("status") == "unknown")
    coverage = round(len(known_scores) / max(1, len(stages)), 4)
    # Unknown stages are never treated as passing.  They reduce the reported
    # confidence score while remaining visible as explicit ``unknown``
    # statuses, so a builder can decide whether to declare or author them.
    raw_score = (sum(known_scores) / len(known_scores)) if known_scores else None
    score = round(raw_score * coverage, 4) if raw_score is not None else None
    try:
        min_quality = max(0.0, min(1.0, float(contract.get("min_quality", 0.78))))
    except (TypeError, ValueError):
        min_quality = 0.78
    if enforce and score is not None and score < min_quality:
        failures_by_stage["evidence"].append({"reason": "quality_below_threshold", "score": score, "required": min_quality})
        stages["evidence"]["status"] = "fail"
        stages["evidence"]["gate"] = False
        stages["evidence"]["score"] = 0.0
        if first_failure is None:
            first_failure = "evidence"
    repair_actions = {
        "structure": "declare_identity_scale_parts_and_primary_refs",
        "primary": "replace_primitive_pile_with_a_resolved_primary_carrier",
        "secondary": "resolve_junctions_openings_and_contacts",
        "tertiary": "increase_detail_carrier_resolution_and_add_named_masks",
        "technical": "repair_topology_normals_materials_uv_or_transforms",
        "evidence": "add_fixed_multi_view_evidence_and_rerun_verify",
    }
    gate = (not enforce) or first_failure is None
    return {
        "gate": gate,
        "enforced": enforce,
        "status": "enforced" if enforce else "advisory",
        "score": score,
        "coverage": coverage,
        "unknown_stage_count": unknown_stage_count,
        "raw_score": round(raw_score, 4) if raw_score is not None else None,
        "min_quality": min_quality,
        "stages": stages,
        "unknown": sorted(set(unknown)),
        "first_failure": first_failure,
        "repair_action": repair_actions.get(first_failure) if first_failure else None,
        "representation": {"kind": representation.get("kind"), "carrier": representation.get("carrier"), "primary_refs": primary_refs},
        "required_stages": required_stages,
    }


def _verify(
    args: Mapping[str, Any],
    *,
    trusted_verifier_paths: Optional[Iterable[Path]] = None,
    required_tags_lock: Optional[Iterable[str]] = None,
    task_spec_override: Optional[Mapping[str, Any]] = None,
    quality_contract: Optional[Mapping[str, Any]] = None,
    current_revision: Optional[int] = None,
    last_render: Optional[Mapping[str, Any]] = None,
    last_visual_review: Optional[Mapping[str, Any]] = None,
    visual_review_history: Optional[Iterable[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run semantic, topology, assembly, opening, metric and multi-view checks."""
    summary = scene_summary()
    objects, audit_scope = _verify_scope(args)
    if audit_scope.get("missing_targets"):
        return {
            "gate": False,
            "quality": 0.0,
            "total": 0.0,
            "quality_profile": "structural",
            "completion_gate": False,
            "require_visual_review": False,
            "scope": audit_scope,
            "semantic": {"gate": False, "required_tags": [], "present_tags": [], "missing_tags": audit_scope["missing_targets"]},
            "topology": {"gate": False, "objects": []},
            "assembly": {"gate": False, "failures": [{"reason": "unknown_audit_target", "targets": audit_scope["missing_targets"]}]},
            "opening": {"gate": False, "required": False, "candidates": [], "failures": []},
            "proportions": {"gate": False, "score": 0.0, "checks": [], "failures": [{"reason": "unknown_audit_target"}]},
            "silhouette": {"gate": False, "score": 0.0, "views": [], "extents": [0.0, 0.0, 0.0]},
            "detail": {"gate": False, "score": 0.0, "checks": [], "failures": [{"reason": "unknown_audit_target"}]},
            "quality": {"gate": False, "enforced": False, "status": "error", "score": 0.0, "stages": {}, "first_failure": "structure", "repair_action": "resolve_verify_targets"},
            "metric": {"gate": False, "score": 0.0, "objects": 0, "mesh_objects": 0},
            "anti_slop": {"gate": False, "checks": {}, "blockers": ["unknown_audit_target"]},
            "visual": {"gate": False, "reason": "unknown_audit_target"},
            "scope": audit_scope,
            "summary": summary,
            "backend": "blender-native",
            "verifier_error": None,
        }
    meshes = [obj for obj in objects if obj.type == "MESH"]
    topologies = [_topology(obj) for obj in meshes]
    task_spec = copy.deepcopy(dict(task_spec_override or {}))
    if task_spec_override is None:
        task_spec.update(_load_task_spec(args))
    coordinate = _coordinate_system()
    allow_default_coordinates = bool(args.get("allow_default_coordinates", False))
    coordinate_failures: list[Dict[str, Any]] = []
    if not bool(coordinate.get("declared")) and not allow_default_coordinates:
        coordinate_failures.append({"reason": "scene_coordinate_system_undeclared", "repair": "scene.coordinate_system"})
    for obj in objects:
        frame = _load_json_prop(obj, _COORDINATE_PROP, None)
        if not isinstance(frame, Mapping) or not frame.get("space") or not frame.get("units"):
            coordinate_failures.append({"target": _stable_uuid(obj), "name": obj.name, "reason": "object_coordinate_frame_missing"})
    coordinate_gate = not coordinate_failures
    # Once session.open has registered a quality contract, verify.run cannot
    # weaken it by sending a different task_spec in a later action.
    if quality_contract and bool(quality_contract.get("configured")):
        task_spec["quality"] = copy.deepcopy(dict(quality_contract))
    elif task_spec_override is not None:
        # A frozen session contract is authoritative.  Do not let a later
        # verify call weaken a declared topology, assembly, silhouette, or
        # dimension rule by sending a conflicting top-level value.
        args = dict(args)
        for key in ("expect_shells", "expect_genus", "require_closed", "required_tags", "feature_sizes", "assembly", "contacts", "require_single_assembly", "contact_tolerance", "require_openings", "proportions", "silhouette_views", "detail_regions", "verifier_path", "source_path", "voxel", "ground_z", "metric_score", "silhouette_score", "quality"):
            if key in task_spec:
                args.pop(key, None)
    expect_shells = args.get("expect_shells", task_spec.get("expect_shells"))
    require_closed = bool(args.get("require_closed", task_spec.get("require_closed", False)))
    topology_gate = bool(objects) and all(
        obj.type != "MESH" or (
            item["nonmanifold_edges"] == 0
            and (not require_closed or item["boundary_edges"] == 0)
            and (expect_shells is None or item["shells"] == int(expect_shells))
        ) for obj, item in zip(meshes, topologies)
    )
    declared_tags = set(_normalize_tags(task_spec.get("required_tags")))
    requested_tags = set(_normalize_tags(args.get("required_tags")))
    locked = set(required_tags_lock or ())
    required_tags = sorted(locked | declared_tags | requested_tags)
    present_tags = {tag for obj in objects for tag in _semantic_tags(obj)}
    missing_tags = sorted(set(required_tags) - present_tags)
    semantic_gate = not missing_tags
    semantic = {"gate": semantic_gate, "required_tags": required_tags, "present_tags": sorted(present_tags), "missing_tags": missing_tags}
    assembly = _assembly_check(objects, task_spec, args)
    opening = _opening_check(objects, task_spec, args, topologies, required_tags_lock)
    proportions = _proportion_check(objects, task_spec, args)
    silhouette = _silhouette_check(objects, task_spec, args)
    detail = _detail_check(objects, task_spec, args)
    quality_args = dict(args)
    # A session-registered contract is immutable for the episode.  Per-call
    # quality fields may add stricter diagnostics, but cannot turn enforcement
    # off or replace the primary carrier/evidence declaration.
    if quality_contract and bool(quality_contract.get("configured")):
        quality_args.pop("quality", None)
    quality = _quality_check(objects, meshes, task_spec, quality_args, topology_gate=topology_gate, topologies=topologies, silhouette=silhouette, detail=detail)
    contract = quality_contract if isinstance(quality_contract, Mapping) else (
        task_spec.get("quality") if isinstance(task_spec.get("quality"), Mapping) else {}
    )
    quality_profile = str(
        args.get("quality_profile")
        or task_spec.get("quality_profile")
        or contract.get("profile")
        or "structural"
    ).strip().lower()
    if quality_profile == "quality_first":
        quality_profile = "structural"
    if quality_profile not in {"structural", "production", "organic", "strict"}:
        quality_profile = "structural"
    completion_gate = bool(args.get("completion_gate", task_spec.get("completion_gate", False)))
    completion_gate = completion_gate or bool(contract.get("completion_gate", False)) or quality_profile in {"production", "organic", "strict"}
    require_visual_review = bool(args.get("require_visual_review", task_spec.get("require_visual_review", False))) or bool(contract.get("require_visual_review", False))
    require_visual = completion_gate or require_visual_review
    required_views = args.get("required_views") or task_spec.get("required_views") or contract.get("required_views") or (contract.get("reference_views") if completion_gate else None)
    required_evidence_types = args.get("required_evidence_types") or task_spec.get("required_evidence_types") or contract.get("required_evidence_types")
    if completion_gate and not required_evidence_types:
        required_evidence_types = ["beauty", "clay", "silhouette", "closeup"]
    required_review_stages = args.get("required_review_stages") or task_spec.get("required_review_stages") or contract.get("required_review_stages")
    try:
        min_visual_views = int(args.get("min_visual_views", task_spec.get("min_visual_views", contract.get("min_visual_views", 4 if completion_gate else 0))))
    except (TypeError, ValueError):
        min_visual_views = 4 if completion_gate else 0
    try:
        min_visual_score = float(args.get("min_visual_score", task_spec.get("min_visual_score", contract.get("min_visual_score", 0.85))))
    except (TypeError, ValueError):
        min_visual_score = 0.85
    min_visual_views = max(0, min(64, min_visual_views))
    min_visual_score = max(0.0, min(1.0, min_visual_score))
    anti_slop = _anti_slop_diagnostics(objects) if require_visual else {"gate": True, "checks": {}, "blockers": [], "skipped": True}
    visual = {"gate": True, "skipped": True}
    if require_visual:
        visual = _visual_evidence_gate(
            last_render,
            last_visual_review,
            current_revision=int(current_revision if current_revision is not None else 0),
            current_state_hash=_scene_content_hash(),
            quality_stage=args.get("quality_stage"),
            require_critical=completion_gate,
            required_views=required_views,
            required_evidence_types=required_evidence_types,
            required_review_stages=required_review_stages,
            review_history=visual_review_history,
            min_visual_views=min_visual_views,
            min_visual_score=min_visual_score,
        )
    if require_visual and not visual.get("gate"):
        quality = copy.deepcopy(quality)
        if isinstance(quality, Mapping):
            stages = quality.get("stages")
            if isinstance(stages, Mapping):
                evidence_stage = dict(stages.get("evidence") or {})
                evidence_stage["status"] = "fail"
                evidence_stage["gate"] = False
                evidence_stage["score"] = 0.0
                failures = list(evidence_stage.get("failures") or [])
                failures.append({"reason": "visual_evidence_gate_failed", "details": dict(visual)})
                evidence_stage["failures"] = failures
                stages = dict(stages)
                stages["evidence"] = evidence_stage
                quality["stages"] = stages
            quality["gate"] = False
            quality["first_failure"] = quality.get("first_failure") or "evidence"
            quality["repair_action"] = "render_review_and_rerun_verify"
    # A task that explicitly requires a render must not receive a passing
    # evidence stage merely because camera names were declared in the contract.
    # The executor-owned render record is the authoritative proof that those
    # views actually exist at this revision.
    if isinstance(contract, Mapping) and bool(contract.get("render_required_explicit")):
        evidence_policy = contract.get("evidence") if isinstance(contract.get("evidence"), Mapping) else {}
        if bool(evidence_policy.get("require_render", True)):
            render_ok = isinstance(last_render, Mapping) and last_render.get("revision") == int(current_revision if current_revision is not None else 0)
            if not render_ok:
                quality = copy.deepcopy(quality)
                stages = dict(quality.get("stages") or {}) if isinstance(quality, Mapping) else {}
                evidence_stage = dict(stages.get("evidence") or {})
                evidence_stage.update({"status": "fail", "gate": False, "score": 0.0})
                failures = list(evidence_stage.get("failures") or [])
                failures.append({"reason": "render_evidence_missing"})
                evidence_stage["failures"] = failures
                stages["evidence"] = evidence_stage
                quality["stages"] = stages
                quality["gate"] = False
                quality["first_failure"] = quality.get("first_failure") or "evidence"
                quality["repair_action"] = "render_review_and_rerun_verify"
    metric = {"gate": topology_gate and proportions["gate"], "score": 1.0 if topology_gate and proportions["gate"] else 0.0, "objects": len(objects), "mesh_objects": len(meshes)}
    vertices, faces = _mesh_arrays(meshes)
    verifier_error = None
    verifier_path = args.get("verifier_path") or task_spec.get("verifier_path") or os.environ.get("BLENDER_TOOLBOX_VERIFIER")
    card: Dict[str, Any] = {}
    if verifier_path and vertices and faces:
        try:
            module = _load_module(str(verifier_path), "blender_toolbox_external_verifier", trusted_paths=trusted_verifier_paths)
            score_fn = getattr(module, "score", None)
            if not callable(score_fn):
                raise ExecutorError("verifier module must expose score()", "invalid_args")
            kwargs: Dict[str, Any] = {"source_path": args.get("source_path") or task_spec.get("source_path"), "voxel": args.get("voxel", task_spec.get("voxel")), "feature_sizes": args.get("feature_sizes", task_spec.get("feature_sizes", ())), "expect_shells": expect_shells, "expect_genus": args.get("expect_genus", task_spec.get("expect_genus")), "ground_z": args.get("ground_z", task_spec.get("ground_z")), "metric_score": args.get("metric_score", task_spec.get("metric_score")), "silhouette_score": args.get("silhouette_score", task_spec.get("silhouette_score"))}
            card = dict(score_fn(vertices, faces, **{key: value for key, value in kwargs.items() if value is not None}))
        except Exception as exc:
            verifier_error = {"code": getattr(exc, "code", "verifier_error"), "message": str(exc)}
    elif verifier_path:
        verifier_error = {"code": "no_mesh_geometry", "message": "configured verifier requires mesh vertices and faces"}
    sections = {"scope": audit_scope, "semantic": semantic, "coordinates": {"gate": coordinate_gate, "declared": bool(coordinate.get("declared")), "allow_default": allow_default_coordinates, "failures": coordinate_failures, "scene": coordinate}, "topology": {"gate": topology_gate, "objects": [{**item, "uuid": _stable_uuid(obj), "name": obj.name} for obj, item in zip(meshes, topologies)]}, "assembly": assembly, "opening": opening, "proportions": proportions, "silhouette": silhouette, "detail": detail, "quality": quality, "metric": metric, "anti_slop": anti_slop, "visual": visual}
    for key, value in sections.items():
        if key not in card or not isinstance(card.get(key), Mapping):
            card[key] = value
        elif key == "semantic":
            card[key] = semantic
    hard_sections = [semantic, {"gate": topology_gate}, {"gate": coordinate_gate}, assembly, opening, proportions, silhouette, detail, metric]
    if quality.get("enforced"):
        hard_sections.append(quality)
    if require_visual:
        hard_sections.extend([anti_slop, visual])
    external_gate = bool(card.get("gate", True)) if card else True
    gate = all(bool(section.get("gate", False)) for section in hard_sections) and external_gate and verifier_error is None
    try:
        external_quality = max(0.0, min(1.0, float(card.get("quality", 1.0)))) if card else 1.0
    except (TypeError, ValueError):
        external_quality = 0.0
    quality_values = [external_quality, 1.0 if semantic_gate else 0.0, 1.0 if topology_gate else 0.0, 1.0 if assembly.get("gate") else 0.0, 1.0 if opening.get("gate") else 0.0, 1.0 if proportions.get("gate") else 0.0, 1.0 if silhouette.get("gate") else 0.0, 1.0 if detail.get("gate") else 0.0]
    if quality.get("score") is not None:
        quality_values.append(float(quality["score"]))
    quality = round(sum(quality_values) / len(quality_values), 4) if quality_values else 0.0
    return {
        "gate": gate,
        "quality": quality if gate else 0.0,
        "total": quality if gate else 0.0,
        "quality_profile": quality_profile,
        "completion_gate": completion_gate,
        "require_visual_review": require_visual_review,
        **sections,
        "physics": card.get("physics"),
        "generative": card.get("generative"),
        "summary": summary,
        "backend": "verifier-first" if verifier_path and not verifier_error else "blender-native",
        "verifier_error": verifier_error,
    }


def _inspect_quality(
    args: Mapping[str, Any],
    *,
    trusted_verifier_paths: Optional[Iterable[Path]] = None,
    required_tags_lock: Optional[Iterable[str]] = None,
    task_spec_override: Optional[Mapping[str, Any]] = None,
    quality_contract: Optional[Mapping[str, Any]] = None,
    current_revision: Optional[int] = None,
    last_render: Optional[Mapping[str, Any]] = None,
    last_visual_review: Optional[Mapping[str, Any]] = None,
    visual_review_history: Optional[Iterable[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Return a bounded quality-focused view without a second execution path.

    ``verify.run`` remains the authoritative full contract.  This action is a
    read-only convenience for interactive inspection and intentionally omits
    the full scene summary from its result so MCP responses stay compact.
    """
    report = _verify(
        args,
        trusted_verifier_paths=trusted_verifier_paths,
        required_tags_lock=required_tags_lock,
        task_spec_override=task_spec_override,
        quality_contract=quality_contract,
        current_revision=current_revision,
        last_render=last_render,
        last_visual_review=last_visual_review,
        visual_review_history=visual_review_history,
    )
    quality = report.get("quality") if isinstance(report.get("quality"), Mapping) else {}
    result: Dict[str, Any] = {
        "gate": bool(report.get("gate")),
        "quality": report.get("quality"),
        "quality_report": quality,
        "backend": report.get("backend"),
        "verifier_error": report.get("verifier_error"),
    }
    for key in ("semantic", "topology", "assembly", "opening", "proportions", "silhouette", "detail", "metric"):
        if key in report:
            result[key] = report[key]
    if isinstance(quality, Mapping):
        result["first_failure"] = quality.get("first_failure")
        result["repair_action"] = quality.get("repair_action")
        result["unknown"] = quality.get("unknown", [])
    return result


def _quick_verify(quality_contract: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Cheap post-mutation signal without pretending topology is full quality."""
    objects = _geometry_objects()
    meshes = [obj for obj in objects if obj.type == "MESH"]
    topologies = [_topology(obj) for obj in meshes]
    failures = []
    for index, item in enumerate(topologies):
        for key in ("nonmanifold_edges", "zero_area_faces", "duplicate_faces", "non_finite_vertices", "loose_vertices"):
            if item.get(key, 0):
                failures.append({"mesh_index": index, "reason": key, "count": item[key]})
    gate = bool(objects) and not failures
    penalty = min(1.0, 0.12 * len(failures))
    quick_quality = round(max(0.0, 1.0 - penalty), 4) if objects else 0.0
    present_tags = sorted({tag for obj in objects for tag in _semantic_tags(obj)})
    quality_hint: Dict[str, Any] = {"status": "advisory", "gate": True, "first_failure": None}
    if isinstance(quality_contract, Mapping) and bool(quality_contract.get("enforce")):
        rep = quality_contract.get("representation") if isinstance(quality_contract.get("representation"), Mapping) else {}
        refs = quality_contract.get("primary_refs") or rep.get("primary_refs") or []
        if isinstance(refs, str):
            refs = [refs]
        primary_objects = [_resolve_scene_ref(ref, objects) for ref in refs] if isinstance(refs, (list, tuple)) else []
        if not refs:
            quality_hint = {"status": "enforced", "gate": False, "first_failure": "structure", "repair_action": "declare_identity_scale_parts_and_primary_refs"}
        elif any(obj is None for obj in primary_objects):
            quality_hint = {"status": "enforced", "gate": False, "first_failure": "structure", "repair_action": "author_primary_carrier"}
        elif any(str(obj.get(_REPRESENTATION_PROP) or "") == "primitive" for obj in primary_objects if obj is not None):
            quality_hint = {"status": "enforced", "gate": False, "first_failure": "primary", "repair_action": "replace_with_control_mesh_or_native_carrier"}
        else:
            quality_hint = {"status": "enforced", "gate": gate, "first_failure": failures[0] if failures else None}
        gate = gate and bool(quality_hint.get("gate"))
    return {
        "gate": gate,
        "quick_quality": quick_quality,
        "total": quick_quality,
        "topology": {"gate": gate, "objects": topologies},
        "semantic": {"gate": True, "required_tags": [], "present_tags": present_tags, "missing_tags": []},
        "first_failure": failures[0] if failures else None,
        "quality": quality_hint,
        "physics": None,
        "metric": None,
        "silhouette": None,
        "detail": None,
        "generative": None,
        "backend": "blender-native-quick",
    }



class _CoreToolboxServer:
    """Newline JSON socket server; bpy work is marshalled to a Blender timer."""

    def __init__(self, address: str, *, allow_run_python: bool = False, allow_bpy_apply: bool = False, auth_token: Optional[str] = None) -> None:
        self.address = address
        self.executor = ToolboxExecutor(allow_run_python=allow_run_python, allow_bpy_apply=allow_bpy_apply, auth_token=auth_token)
        self._pending: queue.Queue[tuple[dict, queue.Queue[dict]]] = queue.Queue()
        self._stop = threading.Event()
        self._listener: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if self.address.startswith("tcp://"):
            host, port = _parse_local_tcp(self.address)
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((host, port))
        else:
            path = Path(self.address).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(path))
            os.chmod(path, 0o600)
        listener.listen(16)
        listener.settimeout(0.5)
        self._listener = listener
        self._thread = threading.Thread(target=self._serve, name="blender-toolbox-ipc", daemon=True)
        self._thread.start()
        if bpy is not None:
            bpy.app.timers.register(self._drain, first_interval=0.01, persistent=True)

    def _serve(self) -> None:
        assert self._listener is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            conn.settimeout(120.0)
            buffer = b""
            payload: Any = {}
            try:
                # Keep framing/size failures inside the same error path as
                # JSON and executor failures.  Previously an oversized or
                # never-terminated frame raised from this loop's thread and
                # left the client with no response at all.
                while b"\n" not in buffer:
                    chunk = conn.recv(65536)
                    if not chunk:
                        return
                    buffer += chunk
                    if len(buffer) > MAX_IPC_MESSAGE_BYTES:
                        raise ProtocolError("request exceeds IPC message limit", "invalid_args")
                frame = buffer.split(b"\n", 1)[0]
                if len(frame) > MAX_IPC_MESSAGE_BYTES:
                    raise ProtocolError("request exceeds IPC message limit", "invalid_args")
                payload = json.loads(frame.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ProtocolError("request must be an object")
                try:
                    _validate_json_value(payload)
                except ProtocolError as exc:
                    # A complete socket frame is syntactically JSON but
                    # violates the transport's finite JSON contract.  Keep
                    # this at the public argument boundary so clients get a
                    # stable invalid_args response instead of an executor
                    # implementation detail (invalid_json).
                    raise ProtocolError(str(exc), "invalid_args") from exc
                # In Blender, scene mutations must be marshalled through the
                # main-thread timer queue.  Protocol-only callers (including
                # unit tests and lightweight discovery harnesses) run without
                # ``bpy`` and therefore have no timer draining ``_pending``;
                # execute synchronously there instead of waiting for a queue
                # consumer that can never arrive.
                if bpy is None:
                    response = self.executor.execute(payload)
                else:
                    result_queue: queue.Queue[dict] = queue.Queue(maxsize=1)
                    self._pending.put((payload, result_queue))
                    response = result_queue.get(timeout=120.0)
            except Exception as exc:
                response = response_from_error(str(payload.get("request_id", "")) if isinstance(payload, dict) else "", self.executor.revision, exc).as_dict()
            try:
                encoded = (json.dumps(response, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
            except Exception as exc:
                response = response_from_error("", self.executor.revision, exc).as_dict()
                encoded = (json.dumps(response, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
            if len(encoded) > MAX_IPC_MESSAGE_BYTES:
                # A valid action may legitimately produce a large scene
                # response, but the framing contract must remain bounded.
                # Replace it with a compact, machine-readable failure rather
                # than sending a frame the client is required to reject.
                response = response_from_error(
                    "",
                    self.executor.revision,
                    ProtocolError("response exceeds IPC message limit", "response_too_large"),
                ).as_dict()
                encoded = (json.dumps(response, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
            conn.sendall(encoded)

    def _drain(self) -> Optional[float]:
        while True:
            try:
                payload, result_queue = self._pending.get_nowait()
            except queue.Empty:
                break
            result_queue.put(self.executor.execute(payload))
        return 0.01 if not self._stop.is_set() else None

    def stop(self) -> None:
        self._stop.set()
        if self._listener:
            self._listener.close()
        if self.address.startswith("/"):
            try:
                Path(self.address).unlink()
            except FileNotFoundError:
                pass

def _object_snapshot(obj: Any) -> Dict[str, Any]:
    """Create an unlinked Blender-native snapshot for transactional edits."""
    data_users = [
        candidate for candidate in bpy.data.objects
        if getattr(candidate, "data", None) is obj.data
    ] if obj.data is not None else []
    snapshot = obj.copy()
    snapshot.name = f"__ToolboxSnapshot_{uuid.uuid4().hex}"
    if obj.data is not None:
        snapshot.data = obj.data.copy()
    return {
        "object": snapshot,
        "name": obj.name,
        "collections": list(obj.users_collection),
        "data_users": data_users,
        "was_selected": bool(obj.select_get()),
        "was_active": bpy.context.view_layer.objects.active == obj,
    }

def _discard_object_snapshot(snapshot: Mapping[str, Any]) -> None:
    snapshot_obj = snapshot.get("object")
    if snapshot_obj is None:
        return
    snapshot_data = getattr(snapshot_obj, "data", None)
    if snapshot_obj.name in bpy.data.objects:
        bpy.data.objects.remove(snapshot_obj, do_unlink=True)
    if snapshot_data is not None and getattr(snapshot_data, "users", 0) == 0:
        if isinstance(snapshot_data, bpy.types.Mesh):
            bpy.data.meshes.remove(snapshot_data, do_unlink=True)

def _restore_object_snapshot(obj: Any, snapshot: Mapping[str, Any]) -> Any:
    """Replace a failed target with its snapshot and remap all object users."""
    snapshot_obj = snapshot["object"]
    old_data = getattr(obj, "data", None)
    for collection in snapshot.get("collections", ()):
        if snapshot_obj.name not in collection.objects:
            collection.objects.link(snapshot_obj)
    obj.user_remap(snapshot_obj)
    snapshot_data = getattr(snapshot_obj, "data", None)
    for data_user in snapshot.get("data_users", ()):
        if data_user != obj and data_user.name in bpy.data.objects:
            data_user.data = snapshot_data
    if obj.name in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    snapshot_obj.name = str(snapshot["name"])
    snapshot_obj.select_set(bool(snapshot.get("was_selected", False)))
    if snapshot.get("was_active"):
        bpy.context.view_layer.objects.active = snapshot_obj
    if old_data is not None and getattr(old_data, "users", 0) == 0:
        if isinstance(old_data, bpy.types.Mesh):
            bpy.data.meshes.remove(old_data, do_unlink=True)
    bpy.context.view_layer.update()
    return snapshot_obj

@contextlib.contextmanager
def _object_transaction(obj: Any):
    """Roll back mesh, shape keys, modifiers, groups and transforms on error."""
    snapshot = _object_snapshot(obj)
    try:
        yield obj
    except Exception as exc:
        try:
            _restore_object_snapshot(obj, snapshot)
        except Exception as rollback_exc:
            raise ExecutorError(
                f"object transaction failed and rollback failed: {rollback_exc}; original error: {exc}",
                "rollback_failed",
            ) from rollback_exc
        raise
    else:
        _discard_object_snapshot(snapshot)

def _landmark_project_to_surface(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Move a landmark to the nearest point on a mesh, preserving intent in the trajectory."""
    _require_bpy()
    landmark = _landmark_object(args["landmark"])
    target = _require_mesh_object(args["target"])
    source_world = landmark.matrix_world.translation.copy()
    source_local = target.matrix_world.inverted() @ source_world
    frame = _coordinate_frame(args)
    direction = _direction_to_world(landmark, args.get("direction", (0.0, 0.0, -1.0)), frame, "direction")
    if direction.length < 1e-9:
        raise ExecutorError("direction must be non-zero", "invalid_args")
    direction.normalize()
    max_distance = _length_value(args.get("max_distance", 100.0), "max_distance", frame)
    hit, location_local, normal_local, _face_index = target.closest_point_on_mesh(source_local)
    tree = _surface_bvh(target)
    if tree is not None:
        ray = tree.ray_cast(source_world, direction, max_distance)
        if ray[0] is None and bool(args.get("search_both_directions", True)):
            ray = tree.ray_cast(source_world, -direction, max_distance)
        if ray[0] is not None:
            hit_world, normal_world_ray, _index, _distance = ray
            location_local = target.matrix_world.inverted() @ Vector(hit_world)
            normal_local = target.matrix_world.inverted().to_3x3() @ Vector(normal_world_ray)
            hit = True
    if not hit:
        raise ExecutorError("target mesh has no reachable surface for landmark projection", "execution_error")
    world_matrix = target.matrix_world
    projected_world = world_matrix @ location_local
    normal_matrix = world_matrix.to_3x3().inverted().transposed()
    normal_world = normal_matrix @ normal_local
    if normal_world.length < 1e-9:
        raise ExecutorError("target surface returned a zero-length normal", "execution_error")
    normal_world.normalize()
    projected_world += normal_world * _length_value(args.get("offset", 0.0), "offset", frame)
    source_to_projection = projected_world - source_world
    if source_to_projection.length > max_distance + 1e-8:
        raise ExecutorError("landmark projection exceeds max_distance", "precondition_failed")
    updated_matrix = landmark.matrix_world.copy()
    updated_matrix.translation = projected_world
    landmark.matrix_world = updated_matrix
    bpy.context.view_layer.update()
    distance = (projected_world - source_world).length
    _store_json_prop(landmark, _COORDINATE_PROP, frame)
    return {
        "landmark": _stable_uuid(landmark),
        "target": _stable_uuid(target),
        "location": [round(float(value), 8) for value in projected_world],
        "normal": [round(float(value), 8) for value in normal_world],
        "distance": round(float(distance), 8),
        "coordinate_frame": frame,
    }

def _mirror_point_sets(points: list[Vector], symmetry: Mapping[str, Any]) -> list[list[Vector]]:
    """Return independent mirrored strokes; never connect separate sides."""
    output: list[list[Vector]] = []
    axes = [bool(symmetry.get(axis, False)) for axis in ("x", "y", "z")]
    variants_for_stroke = [point.copy() for point in points]
    variants = [variants_for_stroke]
    for axis, enabled in enumerate(axes):
        if enabled:
            mirrored = []
            for stroke in variants:
                reflected = [point.copy() for point in stroke]
                for point in reflected:
                    point[axis] *= -1.0
                mirrored.append(reflected)
            variants.extend(mirrored)
    for stroke in variants:
        deduped = []
        # Keep deduplication local to one polyline.  A centerline anchor may
        # legitimately belong to both the authored and mirrored paths; a
        # global set would remove it from the second path and can drop a
        # two-point mirrored stroke entirely.
        seen: set[tuple[float, float, float]] = set()
        for variant in stroke:
            key = tuple(round(float(value), 8) for value in variant)
            if key not in seen:
                seen.add(key)
                deduped.append(variant)
        if len(deduped) >= 2:
            output.append(deduped)
    return output

def _project_stroke_point_sets(obj: Any, point_sets: list[list[Vector]]) -> tuple[list[list[Vector]], int]:
    """Project authored stroke guides onto the target surface when possible.

    LLM-authored landmarks are usually approximate and may sit several
    millimetres in front of a mesh.  Projecting guides before distance tests
    prevents a brush from missing the intended region while preserving the
    original guide when Blender cannot find a closest point.
    """
    projected: list[list[Vector]] = []
    count = 0
    for stroke in point_sets:
        guide: list[Vector] = []
        for point in stroke:
            try:
                hit, location, _normal, _face = obj.closest_point_on_mesh(point)
            except Exception:
                hit = False
                location = point
            if hit:
                guide.append(Vector(location))
                count += 1
            else:
                guide.append(point.copy())
        projected.append(guide)
    return projected, count

def _project_point_budget(value: Any = None) -> int:
    """Parse the closest-point query budget used by sculpt projection."""
    if value is None:
        return DEFAULT_MAX_PROJECT_POINTS
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExecutorError("max_project_points must be an integer", "invalid_args")
    if not 1 <= value <= MAX_PROJECT_POINTS:
        raise ExecutorError(
            f"max_project_points must be between 1 and {MAX_PROJECT_POINTS}",
            "invalid_args",
        )
    return value

def _stroke_projection_workload(args: Mapping[str, Any]) -> int:
    """Return the planned closest-point queries for one complete stroke."""
    if not bool(args.get("project_to_surface", True)):
        return 0
    point_sets = _mirror_point_sets(_sculpt_points(args), args.get("symmetry") or {})
    points = sum(len(stroke) for stroke in point_sets)
    repeat = max(1, min(8, int(args.get("repeat", 1))))
    smooth_passes = max(0, min(5, int(args.get("smooth_passes", 0))))
    if smooth_passes and str(args.get("mode", "")).lower() not in {"smooth", "relax"}:
        repeat += smooth_passes
    return points * repeat

def _validate_stroke_projection_budget(args: Mapping[str, Any]) -> tuple[int, int]:
    """Validate and return (planned queries, configured budget)."""
    budget = _project_point_budget(args.get("max_project_points"))
    workload = _stroke_projection_workload(args)
    if workload > budget:
        raise ExecutorError(
            f"sculpt projection workload {workload} exceeds max_project_points {budget}",
            "resource_limit",
        )
    return workload, budget

def _sculpt_materialize_multires(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Bake an evaluated Multires level into the object's mesh datablock.

    Multires is useful while sculpting, but it is not a durable carrier for
    downstream topology inspection or export.  Converting the evaluated
    object keeps the stable object UUID and semantic tags while removing the
    modifier.  The vertex budget is checked before replacing the datablock so
    a too-expensive request remains atomic under ``_object_transaction``.
    """
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    modifier = next((item for item in obj.modifiers if item.type == "MULTIRES"), None)
    if modifier is None:
        raise ExecutorError("materialize_multires requires an active Multires modifier", "precondition_failed")
    if getattr(obj.data, "shape_keys", None) is not None:
        raise ExecutorError(
            "materialize_multires cannot preserve shape keys; bake before adding facial animation",
            "precondition_failed",
        )
    other_modifiers = [item.name for item in obj.modifiers if item.type != "MULTIRES"]
    if other_modifiers:
        raise ExecutorError(
            "materialize_multires requires no other modifiers: " + ", ".join(other_modifiers),
            "precondition_failed",
        )
    modifier_name = str(modifier.name)

    total_levels = int(getattr(modifier, "total_levels", 0))
    level = int(args.get("level", getattr(modifier, "sculpt_levels", modifier.levels)))
    if not 0 <= level <= total_levels:
        raise ExecutorError(
            f"materialize level must be between 0 and {total_levels}",
            "invalid_args",
        )
    max_vertices_value = args.get("max_vertices", _DEFAULT_MATERIALIZE_VERTEX_BUDGET)
    if isinstance(max_vertices_value, bool) or not isinstance(max_vertices_value, int):
        raise ExecutorError("max_vertices must be an integer", "invalid_args")
    max_vertices = int(max_vertices_value)
    if max_vertices < 512 or max_vertices > 5000000:
        raise ExecutorError("max_vertices must be between 512 and 5000000", "invalid_args")

    old_mesh = obj.data
    old_groups = [group.name for group in obj.vertex_groups]
    previous_levels = (
        int(getattr(modifier, "levels", total_levels)),
        int(getattr(modifier, "sculpt_levels", total_levels)),
        int(getattr(modifier, "render_levels", total_levels)),
    )
    materialized = None
    try:
        # ``levels`` controls the evaluated viewport carrier.  Restore it on
        # failure so the transaction can put the original object back intact.
        modifier.levels = level
        modifier.sculpt_levels = level
        modifier.render_levels = min(previous_levels[2], level)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = obj.evaluated_get(depsgraph)
        materialized = bpy.data.meshes.new_from_object(
            evaluated,
            depsgraph=depsgraph,
            preserve_all_data_layers=True,
        )
        if materialized is None:
            raise ExecutorError("Multires evaluation produced no mesh", "execution_error")
        vertex_count = len(materialized.vertices)
        if vertex_count > max_vertices:
            raise ExecutorError(
                f"materialized mesh has {vertex_count} vertices; max_vertices is {max_vertices}",
                "resource_limit",
            )
        materialized.name = f"{obj.name}_Materialized"
        # ``new_from_object`` normally carries materials and UVs.  Keep the
        # original slots as a fallback for Blender builds that omit them.
        if len(materialized.materials) == 0 and old_mesh is not None:
            for material in old_mesh.materials:
                materialized.materials.append(material)
        obj.data = materialized
        obj.modifiers.remove(modifier)
        # Base-mesh vertex groups no longer index the evaluated topology.  Do
        # not leave stale masks that could silently protect the wrong region.
        for group in list(obj.vertex_groups):
            obj.vertex_groups.remove(group)
        obj.data.update()
        if old_mesh is not None and old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh, do_unlink=True)
        materialized = None
        return {
            "target": _stable_uuid(obj),
            "level": level,
            "vertices": len(obj.data.vertices),
            "faces": len(obj.data.polygons),
            "removed_modifier": modifier_name,
            "invalidated_vertex_groups": old_groups,
        }
    finally:
        if materialized is not None and materialized.users == 0:
            bpy.data.meshes.remove(materialized, do_unlink=True)
        # This is only reached while the original modifier is still attached;
        # successful materialization removes it above.
        remaining_modifier = obj.modifiers.get(modifier_name)
        if remaining_modifier is not None:
            remaining_modifier.levels, remaining_modifier.sculpt_levels, remaining_modifier.render_levels = previous_levels

def _sculpt_surface_prepare(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Create a repeatable sculpt carrier before landmark/detail passes.

    The operation intentionally keeps the object identity and semantic tags,
    while making transforms explicit and putting remeshing before Multires.
    This gives complex assets (heads, creatures, props) an even, inspectable
    base instead of relying on primitive topology or viewport state.
    """
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    if any(item.type == "MULTIRES" for item in obj.modifiers):
        raise ExecutorError("surface_prepare requires applying or removing Multires first", "precondition_failed")
    if args.get("voxel_size") is not None and getattr(obj.data, "shape_keys", None) is not None:
        raise ExecutorError("voxel remesh would discard existing shape keys; prepare before facial animation", "precondition_failed")
    cuts = int(args.get("subdivide_cuts", 0))
    if not 0 <= cuts <= 4:
        raise ExecutorError("subdivide_cuts must be between 0 and 4", "invalid_args")
    voxel_value = args.get("voxel_size")
    if voxel_value is not None and not 0.00001 < float(voxel_value) <= 1000.0:
        raise ExecutorError("voxel_size must be in (0.00001, 1000]", "invalid_args")
    smooth_iterations = int(args.get("smooth_iterations", 0))
    if not 0 <= smooth_iterations <= 8:
        raise ExecutorError("smooth_iterations must be between 0 and 8", "invalid_args")
    multires_levels = int(args.get("multires_levels", 0))
    if not 0 <= multires_levels <= 5:
        raise ExecutorError("multires_levels must be between 0 and 5", "invalid_args")
    sculpt_level = int(args.get("sculpt_level", multires_levels))
    render_level = int(args.get("render_level", multires_levels))
    if not 0 <= sculpt_level <= multires_levels:
        raise ExecutorError(
            f"sculpt_level must be between 0 and multires_levels ({multires_levels})",
            "invalid_args",
        )
    if not 0 <= render_level <= multires_levels:
        raise ExecutorError(
            f"render_level must be between 0 and multires_levels ({multires_levels})",
            "invalid_args",
        )
    if bool(args.get("materialize_multires", False)) and multires_levels == 0:
        raise ExecutorError(
            "materialize_multires requires multires_levels greater than zero",
            "invalid_args",
        )
    if bool(args.get("materialize_multires", False)):
        if getattr(obj.data, "shape_keys", None) is not None:
            raise ExecutorError(
                "materialize_multires cannot preserve shape keys; bake before adding facial animation",
                "precondition_failed",
            )
        other_modifiers = [item.name for item in obj.modifiers if item.type != "MULTIRES"]
        if other_modifiers:
            raise ExecutorError(
                "materialize_multires requires no other modifiers: " + ", ".join(other_modifiers),
                "precondition_failed",
            )
    before = {"topology": _topology(obj), "measure": _measure(obj)}
    stages: list[str] = []
    with _object_transaction(obj):
        apply_scale = bool(args.get("apply_scale", True))
        apply_rotation = bool(args.get("apply_rotation", False))
        if apply_scale or apply_rotation:
            _activate_object(obj)
            bpy.ops.object.transform_apply(location=False, rotation=apply_rotation, scale=apply_scale)
            stages.append("transform_apply")
            bpy.context.view_layer.update()

        if cuts > 0:
            bm = bmesh.new()
            try:
                bm.from_mesh(obj.data)
                bm.edges.ensure_lookup_table()
                bmesh.ops.subdivide_edges(bm, edges=list(bm.edges), cuts=cuts, use_grid_fill=True)
                _write_bmesh(obj, bm)
            finally:
                bm.free()
            stages.append("subdivide")

        if voxel_value is not None:
            voxel_size = float(voxel_value)
            obj.data.remesh_voxel_size = voxel_size
            obj.data.remesh_voxel_adaptivity = float(args.get("adaptivity", 0.0))
            _activate_object(obj)
            try:
                bpy.ops.object.voxel_remesh()
            except Exception as exc:
                raise ExecutorError(f"voxel remesh failed: {exc}", "execution_error") from exc
            stages.append("voxel_remesh")

        if smooth_iterations > 0:
            factor = max(0.0, min(1.0, float(args.get("smooth_factor", 0.15))))
            bm = bmesh.new()
            try:
                bm.from_mesh(obj.data)
                bm.verts.ensure_lookup_table()
                for _ in range(smooth_iterations):
                    bmesh.ops.smooth_vert(
                        bm, verts=list(bm.verts), factor=factor,
                        use_axis_x=True, use_axis_y=True, use_axis_z=True,
                    )
                _write_bmesh(obj, bm)
            finally:
                bm.free()
            stages.append("smooth")

        if multires_levels > 0:
            _sculpt_multires({
                "target": _stable_uuid(obj), "levels": multires_levels,
                "sculpt_level": sculpt_level,
                "render_level": render_level,
            })
            stages.append("multires")
            if bool(args.get("materialize_multires", False)):
                _sculpt_materialize_multires({
                    "target": _stable_uuid(obj),
                    "level": sculpt_level,
                    "max_vertices": int(args.get("max_materialized_vertices", _DEFAULT_MATERIALIZE_VERTEX_BUDGET)),
                })
                stages.append("materialize_multires")
        if bool(args.get("shade_smooth", True)):
            for polygon in obj.data.polygons:
                polygon.use_smooth = True
            obj.data.update()
            stages.append("shade_smooth")
    after = {"topology": _topology(obj), "measure": _measure(obj)}
    return {
        "target": _stable_uuid(obj),
        "stages": stages,
        "before": before,
        "after": after,
        "semantic_tags": _semantic_tags(obj),
    }

def _ensure_materialized_stroke_surface(obj: Any, args: Mapping[str, Any]) -> bool:
    """Reject accidental writes to a coarse Multires base mesh.

    The deterministic brush implementation edits ``obj.data`` directly.  If
    Multires is still active, that datablock is the low-resolution carrier and
    a detail stroke can disappear or become much coarser than the authored
    guide suggests.  Callers may explicitly opt into this behavior for a
    deliberate broad base edit, but the quality path requires materializing
    the intended level first.
    """
    modifiers = [item for item in obj.modifiers if item.type == "MULTIRES"]
    if not modifiers:
        return False
    if bool(args.get("allow_multires_base", False)):
        return True
    names = ", ".join(str(item.name) for item in modifiers)
    raise ExecutorError(
        "deterministic sculpt strokes require a materialized mesh while "
        f"Multires is active ({names}); call sculpt.materialize_multires "
        "first, or set allow_multires_base=true for an intentional coarse edit",
        "precondition_failed",
    )

def _sculpt_stage_diagnostics(strokes: list[Mapping[str, Any]], default_stage: Any) -> Dict[str, Any]:
    """Describe whether one batch follows the intended coarse-to-fine order."""
    sequence = [
        str(stroke.get("stage", default_stage or "custom")) if isinstance(stroke, Mapping)
        else str(default_stage or "custom")
        for stroke in strokes
    ]
    ordered = [
        _SCULPT_STAGE_ORDER[stage]
        for stage in sequence
        if stage in _SCULPT_STAGE_ORDER
    ]
    warnings: list[str] = []
    if any(right < left for left, right in zip(ordered, ordered[1:])):
        warnings.append("stroke stages move backward; split or reorder the batch")
    known = [stage for stage in sequence if stage in _SCULPT_STAGE_ORDER]
    if len(set(known)) > 1:
        warnings.append("one batch mixes sculpt stages; keep primary/secondary/tertiary/cleanup passes separate")
    return {
        "stage_sequence": sequence,
        "stage_order_valid": not any(right < left for left, right in zip(ordered, ordered[1:])),
        "stage_warnings": warnings,
    }

def _sculpt_stroke_batch(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    target = _require_mesh_object(args["target"])
    strokes = args.get("strokes")
    if not isinstance(strokes, list) or not strokes:
        raise ExecutorError("strokes must contain at least one stroke", "invalid_args")
    if len(strokes) > 256:
        raise ExecutorError("strokes exceeds maximum item count 256", "invalid_args")
    for index, raw in enumerate(strokes):
        if not isinstance(raw, Mapping):
            raise ExecutorError(f"strokes[{index}] must be an object", "invalid_args")
    default_stage = args.get("stage")
    stop_on_error = bool(args.get("stop_on_error", True))
    allow_multires_base = bool(args.get("allow_multires_base", False))
    active_multires = [item for item in target.modifiers if item.type == "MULTIRES"]
    if active_multires and not allow_multires_base:
        disallowed = [
            index for index, raw in enumerate(strokes)
            if not bool(raw.get("allow_multires_base", False))
        ]
        if disallowed:
            names = ", ".join(str(item.name) for item in active_multires)
            raise ExecutorError(
                "deterministic sculpt stroke batch requires a materialized mesh "
                f"while Multires is active ({names}); call "
                "sculpt.materialize_multires first, or set "
                "allow_multires_base=true for every intentional coarse stroke",
                "precondition_failed",
            )
    batch_budget = _project_point_budget(args.get("max_project_points"))
    planned_projection = 0
    for index, raw in enumerate(strokes):
        if not isinstance(raw, Mapping):
            raise ExecutorError(f"strokes[{index}] must be an object", "invalid_args")
        stroke = dict(raw)
        if "max_project_points" not in stroke:
            stroke["max_project_points"] = batch_budget
        planned_projection += _stroke_projection_workload(stroke)
    if planned_projection > batch_budget:
        raise ExecutorError(
            f"sculpt batch projection workload {planned_projection} exceeds max_project_points {batch_budget}",
            "resource_limit",
        )
    # A batch is one trajectory action.  Keep it atomic by default so a late
    # projection/runtime failure cannot leave a partial pass in the scene.
    snapshot = [vertex.co.copy() for vertex in target.data.vertices] if stop_on_error else None
    results = []
    for index, raw in enumerate(strokes):
        if not isinstance(raw, Mapping):
            raise ExecutorError(f"strokes[{index}] must be an object", "invalid_args")
        stroke = dict(raw)
        stroke["target"] = _stable_uuid(target)
        if "allow_multires_base" not in stroke and "allow_multires_base" in args:
            stroke["allow_multires_base"] = allow_multires_base
        if "max_project_points" not in stroke:
            stroke["max_project_points"] = batch_budget
        if default_stage is not None and "stage" not in stroke:
            stroke["stage"] = default_stage
        try:
            result = _sculpt_stroke(stroke)
            results.append({
                "index": index, "label": stroke.get("label"),
                "stage": stroke.get("stage", default_stage), "ok": True,
                "result": result,
            })
        except Exception as exc:
            entry = {
                "index": index, "label": stroke.get("label"),
                "stage": stroke.get("stage", default_stage), "ok": False,
                "error": {"code": getattr(exc, "code", "execution_error"), "message": str(exc)},
            }
            results.append(entry)
            if stop_on_error:
                if snapshot is not None and len(snapshot) == len(target.data.vertices):
                    for vertex, coordinate in zip(target.data.vertices, snapshot):
                        vertex.co = coordinate
                    target.data.update()
                raise ExecutorError(f"stroke batch failed at index {index}: {exc}", getattr(exc, "code", "execution_error")) from exc
    successful = [entry for entry in results if entry["ok"]]
    stage_diagnostics = _sculpt_stage_diagnostics(strokes, default_stage)
    return {
        "target": _stable_uuid(target),
        "count": len(strokes),
        "successful": len(successful),
        "failed": len(strokes) - len(successful),
        "strokes": results,
        "stage": default_stage,
        **stage_diagnostics,
        "projection_points": planned_projection,
        "projection_budget": batch_budget,
    }

def _region_falloff(distance: float, mode: str) -> float:
    """Return a bounded center-to-edge influence for an ellipsoidal cage."""
    t = max(0.0, min(1.0, 1.0 - distance))
    if mode == "constant":
        return 1.0 if t > 0.0 else 0.0
    if mode == "linear":
        return t
    if mode == "smoother":
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    return t * t * (3.0 - 2.0 * t)

def _surface_patch_falloff(distance: float, mode: str) -> float:
    """Return a bounded influence for a tangent-frame surface patch.

    ``distance`` is the normalized elliptical distance in the patch plane.
    Keeping this helper independent of Blender makes the falloff contract easy
    to test and useful to non-Blender planners as well.
    """
    return _region_falloff(float(distance), str(mode).lower())

def _surface_patch_influence(local_u: float, local_v: float, local_depth: float, radii: Any, depth_limit: Optional[float], falloff: str) -> float:
    """Evaluate the patch domain without depending on Blender types."""
    if not isinstance(radii, (list, tuple)) or len(radii) != 2:
        raise ExecutorError("surface patch radii must contain two positive values", "invalid_args")
    radius_u, radius_v = float(radii[0]), float(radii[1])
    if not all(math.isfinite(value) and value > 0.0 for value in (radius_u, radius_v)):
        raise ExecutorError("surface patch radii must contain two positive values", "invalid_args")
    if depth_limit is not None and abs(float(local_depth)) > float(depth_limit):
        return 0.0
    distance = math.hypot(float(local_u) / radius_u, float(local_v) / radius_v)
    if distance >= 1.0:
        return 0.0
    return _surface_patch_falloff(distance, falloff)

def _surface_patch_frame(u_axis: Any, v_axis: Any, *, epsilon: float = 1e-9) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Build an orthonormal tangent frame from two authored axes.

    The second axis is projected onto the plane orthogonal to ``u_axis``.
    Collinear or near-zero axes are rejected instead of silently inventing an
    orientation, since an ambiguous frame would make a replayed patch move a
    different region of the mesh.
    """
    u = _as_float3(u_axis, "u_axis")
    v = _as_float3(v_axis, "v_axis")

    if not all(math.isfinite(component) for component in (*u, *v)):
        raise ExecutorError("surface patch axes must contain finite numbers", "invalid_args")

    def length(value: tuple[float, float, float]) -> float:
        return math.sqrt(sum(component * component for component in value))

    u_length = length(u)
    if u_length <= epsilon:
        raise ExecutorError("u_axis must be non-zero", "invalid_args")
    u_normal = tuple(component / u_length for component in u)
    projection = sum(v[index] * u_normal[index] for index in range(3))
    authored_v_length = length(v)
    if authored_v_length <= epsilon:
        raise ExecutorError("v_axis must be non-zero", "invalid_args")
    v_orthogonal = tuple(v[index] - projection * u_normal[index] for index in range(3))
    v_length = length(v_orthogonal)
    if v_length <= epsilon or v_length / authored_v_length <= epsilon:
        raise ExecutorError("u_axis and v_axis must not be collinear", "invalid_args")
    v_normal = tuple(component / v_length for component in v_orthogonal)
    normal = (
        u_normal[1] * v_normal[2] - u_normal[2] * v_normal[1],
        u_normal[2] * v_normal[0] - u_normal[0] * v_normal[2],
        u_normal[0] * v_normal[1] - u_normal[1] * v_normal[0],
    )
    normal_length = length(normal)
    if normal_length <= epsilon:
        raise ExecutorError("surface patch frame has no usable normal", "invalid_args")
    normal = tuple(component / normal_length for component in normal)
    return u_normal, v_normal, normal

def _surface_patch_blended_direction(displacement_direction: Any, vertex_normal: Any, normal_blend: float, *, epsilon: float = 1e-9) -> Any:
    """Linearly blend a patch direction with a vertex normal.

    ``Vector`` is intentionally accepted here instead of importing a Blender
    type so the validation/edge-case behavior remains straightforward to test
    with a small vector double.  The executor passes mathutils ``Vector``
    instances, which support the same arithmetic API.  The lerp magnitude is
    preserved: callers multiply it by the signed patch offset and influence.
    """
    blend = max(0.0, min(1.0, float(normal_blend)))
    direction = displacement_direction.lerp(vertex_normal, blend)
    if direction.length <= epsilon:
        direction = vertex_normal.copy() if vertex_normal.length > epsilon else displacement_direction.copy()
    if direction.length <= epsilon:
        raise ExecutorError("surface patch displacement direction must be non-zero", "invalid_args")
    return direction

def _sculpt_surface_patch_batch(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply local tangent-frame displacements to one continuous mesh.

    A patch is a bounded 2-D elliptical domain embedded in the existing
    surface.  It only changes vertex coordinates on the target mesh and never
    creates helper objects, curves, or additional topology.
    """
    _require_bpy()
    target = _require_mesh_object(args["target"])
    patches = args.get("patches")
    if not isinstance(patches, list) or not patches:
        raise ExecutorError("patches must contain at least one patch", "invalid_args")
    if len(patches) > 256:
        raise ExecutorError("patches exceeds maximum item count 256", "invalid_args")
    _ensure_materialized_stroke_surface(target, args)

    default_stage = args.get("stage")
    stop_on_error = bool(args.get("stop_on_error", True))
    # BMesh writes are deferred until the entire batch completes.  Keep a
    # datablock snapshot as a second line of defense if a late write/update
    # raises after a caller explicitly requests atomic behavior.
    snapshot = [vertex.co.copy() for vertex in target.data.vertices] if stop_on_error else None
    bm = bmesh.new()
    results = []
    try:
        bm.from_mesh(target.data)
        bm.verts.ensure_lookup_table()
        for index, raw in enumerate(patches):
            if not isinstance(raw, Mapping):
                error = ExecutorError(f"patches[{index}] must be an object", "invalid_args")
                results.append({"index": index, "stage": default_stage, "ok": False, "error": {"code": error.code, "message": str(error)}})
                if stop_on_error:
                    raise error
                continue
            try:
                bm.normal_update()
                prefix = f"patches[{index}]"
                center = Vector(_as_float3(raw["center"], f"{prefix}.center"))
                frame = _surface_patch_frame(raw["u_axis"], raw["v_axis"])
                u_axis = Vector(frame[0])
                v_axis = Vector(frame[1])
                frame_normal = Vector(frame[2])
                radii_raw = raw.get("radii")
                if not isinstance(radii_raw, (list, tuple)) or len(radii_raw) != 2:
                    raise ExecutorError(f"{prefix}.radii must contain two positive values", "invalid_args")
                radii = (float(radii_raw[0]), float(radii_raw[1]))
                if not all(math.isfinite(value) and value > 0.0 for value in radii):
                    raise ExecutorError(f"{prefix}.radii must contain two positive values", "invalid_args")
                depth_limit = raw.get("depth_limit")
                if depth_limit is not None:
                    depth_limit = float(depth_limit)
                    if not math.isfinite(depth_limit) or depth_limit <= 0.0:
                        raise ExecutorError(f"{prefix}.depth_limit must be positive", "invalid_args")
                offset = float(raw["offset"])
                if not math.isfinite(offset):
                    raise ExecutorError(f"{prefix}.offset must be finite", "invalid_args")
                normal_blend = float(raw.get("normal_blend", 0.0))
                if not math.isfinite(normal_blend) or not 0.0 <= normal_blend <= 1.0:
                    raise ExecutorError(f"{prefix}.normal_blend must be between 0 and 1", "invalid_args")
                strength = float(raw.get("strength", 1.0))
                if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
                    raise ExecutorError(f"{prefix}.strength must be between 0 and 1", "invalid_args")
                falloff = str(raw.get("falloff", "smooth")).lower()
                if falloff not in {"smooth", "smoother", "linear", "constant"}:
                    raise ExecutorError(f"unsupported surface patch falloff: {falloff}", "invalid_args")
                displacement_raw = raw.get("displacement_direction")
                displacement = Vector(_as_float3(displacement_raw, f"{prefix}.displacement_direction")) if displacement_raw is not None else frame_normal.copy()
                if displacement.length <= 1e-9:
                    raise ExecutorError(f"{prefix}.displacement_direction must be non-zero", "invalid_args")
                displacement.normalize()
                visibility_raw = raw.get("visibility_direction")
                visibility = Vector(_as_float3(visibility_raw, f"{prefix}.visibility_direction")) if visibility_raw is not None else frame_normal.copy()
                if visibility.length <= 1e-9:
                    raise ExecutorError(f"{prefix}.visibility_direction must be non-zero", "invalid_args")
                visibility.normalize()
                selection = raw.get("selection") or {}
                if not isinstance(selection, Mapping):
                    raise ExecutorError(f"{prefix}.selection must be an object", "invalid_args")
                selected_indices = None
                if selection:
                    selected, _, _ = _selection_parts(bm, selection, obj=target, default_all=False)
                    selected_indices = {int(vertex.index) for vertex in selected}
                front_only = bool(raw.get("front_facing_only", False))

                affected = 0
                displacement_sum = 0.0
                max_displacement = 0.0
                affected_min = [math.inf, math.inf, math.inf]
                affected_max = [-math.inf, -math.inf, -math.inf]
                for vertex in bm.verts:
                    if selected_indices is not None and int(vertex.index) not in selected_indices:
                        continue
                    if front_only and vertex.normal.dot(visibility) <= 0.0:
                        continue
                    local = vertex.co - center
                    u_coordinate = local.dot(u_axis)
                    v_coordinate = local.dot(v_axis)
                    normal_coordinate = local.dot(frame_normal)
                    if depth_limit is not None and abs(normal_coordinate) > depth_limit:
                        continue
                    influence = strength * _surface_patch_influence(
                        u_coordinate, v_coordinate, normal_coordinate,
                        radii, depth_limit, falloff,
                    )
                    if influence <= 0.0:
                        continue
                    direction = _surface_patch_blended_direction(displacement, vertex.normal.normalized(), normal_blend)
                    before = vertex.co.copy()
                    vertex.co = before + direction * (offset * influence)
                    moved = (vertex.co - before).length
                    displacement_sum += moved
                    max_displacement = max(max_displacement, moved)
                    affected += 1
                    for axis in range(3):
                        affected_min[axis] = min(affected_min[axis], float(vertex.co[axis]))
                        affected_max[axis] = max(affected_max[axis], float(vertex.co[axis]))
                affected_aabb = None
                if affected:
                    affected_aabb = {
                        "min": [round(value, 8) for value in affected_min],
                        "max": [round(value, 8) for value in affected_max],
                    }
                results.append({
                    "index": index,
                    "label": raw.get("label"),
                    "stage": raw.get("stage", default_stage),
                    "ok": True,
                    "vertices_affected": affected,
                    "mean_displacement": round(displacement_sum / affected, 8) if affected else 0.0,
                    "max_displacement": round(max_displacement, 8),
                    "affected_aabb": affected_aabb,
                })
            except Exception as exc:
                entry = {
                    "index": index,
                    "label": raw.get("label"),
                    "stage": raw.get("stage", default_stage),
                    "ok": False,
                    "error": {"code": getattr(exc, "code", "execution_error"), "message": str(exc)},
                }
                results.append(entry)
                if stop_on_error:
                    if snapshot is not None and len(snapshot) == len(target.data.vertices):
                        for vertex, coordinate in zip(target.data.vertices, snapshot):
                            vertex.co = coordinate
                        target.data.update()
                    raise ExecutorError(f"surface patch batch failed at index {index}: {exc}", getattr(exc, "code", "execution_error")) from exc
        _write_bmesh(target, bm)
    finally:
        bm.free()

    successful = [entry for entry in results if entry["ok"]]
    stage_diagnostics = _sculpt_stage_diagnostics(patches, default_stage)
    return {
        "target": _stable_uuid(target),
        "count": len(patches),
        "successful": len(successful),
        "failed": len(patches) - len(successful),
        "patches": results,
        "stage": default_stage,
        **stage_diagnostics,
    }

def _sculpt_region_deform_batch(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply smooth local cage transforms without creating helper geometry."""
    _require_bpy()
    target = _require_mesh_object(args["target"])
    regions = args.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ExecutorError("regions must contain at least one region", "invalid_args")
    if len(regions) > 128:
        raise ExecutorError("regions exceeds maximum item count 128", "invalid_args")
    _ensure_materialized_stroke_surface(target, args)

    default_stage = args.get("stage")
    stop_on_error = bool(args.get("stop_on_error", True))
    snapshot = [vertex.co.copy() for vertex in target.data.vertices] if stop_on_error else None
    bm = bmesh.new()
    results = []
    try:
        bm.from_mesh(target.data)
        bm.verts.ensure_lookup_table()
        for index, raw in enumerate(regions):
            if not isinstance(raw, Mapping):
                raise ExecutorError(f"regions[{index}] must be an object", "invalid_args")
            try:
                bm.normal_update()
                center = Vector(_as_float3(raw["center"], f"regions[{index}].center"))
                radii = Vector(_as_float3(raw["radii"], f"regions[{index}].radii"))
                if min(radii) <= 0.0:
                    raise ExecutorError(f"regions[{index}].radii must be positive", "invalid_args")
                translation = Vector(_as_float3(raw.get("translation", (0, 0, 0)), f"regions[{index}].translation"))
                scale = _as_float3(raw.get("scale", (1, 1, 1)), f"regions[{index}].scale")
                if min(scale) <= 0.0:
                    raise ExecutorError(f"regions[{index}].scale must be positive", "invalid_args")
                rotation = Euler(
                    _as_float3(raw.get("rotation_euler", (0, 0, 0)), f"regions[{index}].rotation_euler"),
                    "XYZ",
                ).to_matrix()
                strength = max(0.0, min(1.0, float(raw.get("strength", 1.0))))
                normal_offset = float(raw.get("normal_offset", 0.0))
                if not -4.0 <= normal_offset <= 4.0:
                    raise ExecutorError(f"regions[{index}].normal_offset must be between -4 and 4", "invalid_args")
                falloff = str(raw.get("falloff", "smoother")).lower()
                if falloff not in {"smooth", "smoother", "linear", "constant"}:
                    raise ExecutorError(f"unsupported region falloff: {falloff}", "invalid_args")
                selection = raw.get("selection") or {}
                if selection:
                    selected, _, _ = _selection_parts(bm, selection, obj=target, default_all=False)
                    selected_indices = {int(vertex.index) for vertex in selected}
                else:
                    selected_indices = None
                front_only = bool(raw.get("front_facing_only", False))
                direction = Vector(_as_float3(raw.get("direction", (0, 0, 1)), f"regions[{index}].direction"))
                if direction.length < 1e-9:
                    raise ExecutorError(f"regions[{index}].direction must be non-zero", "invalid_args")
                direction.normalize()

                affected = 0
                max_displacement = 0.0
                displacement_sum = 0.0
                for vertex in bm.verts:
                    if selected_indices is not None and int(vertex.index) not in selected_indices:
                        continue
                    if front_only and vertex.normal.dot(direction) <= 0.0:
                        continue
                    local = vertex.co - center
                    # The cage frame must control both selection and the
                    # destination transform.  Previously rotation only
                    # affected the destination, so an oblique region still
                    # selected an axis-aligned ellipsoid.
                    cage_local = rotation.inverted() @ local
                    distance = Vector((cage_local.x / radii.x, cage_local.y / radii.y, cage_local.z / radii.z)).length
                    if distance >= 1.0:
                        continue
                    influence = strength * _region_falloff(distance, falloff)
                    if influence <= 0.0:
                        continue
                    scaled = Vector((cage_local.x * scale[0], cage_local.y * scale[1], cage_local.z * scale[2]))
                    transformed = center + rotation @ scaled + translation
                    updated = vertex.co.lerp(transformed, influence)
                    if normal_offset:
                        surface_normal = vertex.normal.normalized()
                        updated += surface_normal * (normal_offset * influence)
                    displacement = (updated - vertex.co).length
                    max_displacement = max(max_displacement, displacement)
                    displacement_sum += displacement
                    vertex.co = updated
                    affected += 1
                results.append({
                    "index": index,
                    "stage": raw.get("stage", default_stage),
                    "label": raw.get("label"),
                    "ok": True,
                    "vertices_affected": affected,
                    "max_displacement": round(float(max_displacement), 8),
                    "mean_displacement": round(float(displacement_sum / affected), 8) if affected else 0.0,
                    "affected_ratio": round(float(affected / max(1, len(bm.verts))), 8),
                    "normal_offset": normal_offset,
                })
            except Exception as exc:
                results.append({
                    "index": index,
                    "stage": raw.get("stage", default_stage) if isinstance(raw, Mapping) else default_stage,
                    "ok": False,
                    "error": {"code": getattr(exc, "code", "execution_error"), "message": str(exc)},
                })
                if stop_on_error:
                    if snapshot is not None and len(snapshot) == len(target.data.vertices):
                        for vertex, coordinate in zip(target.data.vertices, snapshot):
                            vertex.co = coordinate
                        target.data.update()
                    raise
        _write_bmesh(target, bm)
    finally:
        bm.free()

    stage_diagnostics = _sculpt_stage_diagnostics(regions, default_stage)
    successful = [entry for entry in results if entry["ok"]]
    return {
        "target": _stable_uuid(target),
        "count": len(regions),
        "successful": len(successful),
        "failed": len(regions) - len(successful),
        "regions": results,
        "stage": default_stage,
        **stage_diagnostics,
    }

def _inspect_sculpt_quality(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Expose bounded surface diagnostics for closed-loop sculpt feedback."""
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    calc_normals = getattr(obj.data, "calc_normals", None)
    if callable(calc_normals):
        calc_normals()
    vertices = [[float(value) for value in vertex.co] for vertex in obj.data.vertices]
    edges = [[int(vertex) for vertex in edge.vertices] for edge in obj.data.edges]
    normals = [[float(value) for value in vertex.normal] for vertex in obj.data.vertices]
    axes = args.get("symmetry_axes") or []
    if isinstance(axes, str):
        axes = [axes]
    sample_limit = max(16, min(4096, int(args.get("sample_limit", 512))))
    metrics = sculpt_quality_metrics(
        vertices, edges=edges, normals=normals,
        symmetry_axes=[str(axis) for axis in axes], sample_limit=sample_limit,
    )
    topology = _topology(obj)
    multires = next((item for item in obj.modifiers if item.type == "MULTIRES"), None)
    multires_levels = int(getattr(multires, "total_levels", 0)) if multires is not None else 0
    # Base topology remains intentionally coarse when Multires carries the
    # sculpt detail.  Report the estimated evaluated carrier size as well.
    effective_vertices = len(vertices) * (4 ** multires_levels)
    effective_faces = len(obj.data.polygons) * (4 ** multires_levels)
    density = {
        "vertices_per_dimension": round(len(vertices) / max(metrics["diagonal"], 1e-12), 4),
        "faces_per_dimension_squared": round(len(obj.data.polygons) / max(metrics["diagonal"] ** 2, 1e-12), 4),
        "base_vertices": len(vertices),
        "base_faces": len(obj.data.polygons),
        "multires_levels": multires_levels,
        "effective_vertices_estimate": effective_vertices,
        "effective_faces_estimate": effective_faces,
        "dense_enough_for_sculpt": effective_vertices >= 512 and effective_faces >= 256,
    }
    radial = metrics["radial_profile"]
    surface = metrics["surface"]
    recommendations = []
    if density["dense_enough_for_sculpt"] is False:
        recommendations.append("increase base resolution with sculpt.surface_prepare or geometry.remesh_voxel")
    if radial["ellipsoid_likeness"] >= 0.78:
        recommendations.append("add primary masses and plane transitions before tertiary creases")
    if surface["detail_signal"] < 0.08 and effective_vertices >= 512:
        recommendations.append("inspect front/profile renders and add landmark-guided secondary forms")
    if metrics["symmetry"]["score"] is not None and metrics["symmetry"]["score"] < 0.82:
        recommendations.append("review bilateral landmarks and correct asymmetry with a masked cleanup pass")
    if topology["boundary_edges"] or topology["nonmanifold_edges"]:
        recommendations.append("repair topology before sculpting detail")
    report = {
        "target": _stable_uuid(obj),
        "topology": topology,
        "density": density,
        "metrics": metrics,
        "recommendations": recommendations,
        "landmarks": None,
    }
    if bool(args.get("include_landmarks", True)):
        landmark_tag = args.get("landmark_semantic_tag")
        landmarks = []
        for landmark in bpy.context.scene.objects:
            if landmark.type != "EMPTY" or not landmark.get("blender_toolbox_landmark"):
                continue
            tags = _semantic_tags(landmark)
            if landmark_tag and landmark_tag not in tags:
                continue
            landmarks.append({
                "uuid": _stable_uuid(landmark), "name": landmark.name,
                "semantic_tags": tags,
                "location": [round(float(value), 8) for value in landmark.matrix_world.translation],
            })
        report["landmarks"] = landmarks
    return report

def _object_by_ref(ref: Any) -> Any:
    _require_bpy()
    if not isinstance(ref, str) or not ref:
        raise ExecutorError("object reference must be a non-empty string", "invalid_args")
    # Transaction snapshots are deliberately kept as unlinked datablocks and
    # inherit the target UUID.  Resolve only objects in the active scene so a
    # snapshot can never shadow the live object during a nested action.
    for obj in bpy.context.scene.objects:
        if obj.name == ref or obj.get(_UUID_PROP) == ref or obj.get(_REF_PROP) == ref:
            return obj
    raise ExecutorError(f"object not found: {ref}", "not_found")

def _mirror_points(points: list[Vector], symmetry: Mapping[str, Any]) -> list[Vector]:
    """Legacy flattened view retained for callers that need point counts."""
    return [point for stroke in _mirror_point_sets(points, symmetry) for point in stroke]

def _sculpt_stroke_transactional_once(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    point_sets = _mirror_point_sets(_sculpt_points(args), args.get("symmetry") or {})
    if not point_sets:
        raise ExecutorError("sculpt.stroke requires at least two points", "invalid_args")
    projected_count = 0
    projection_queries = 0
    projection_budget = _project_point_budget(args.get("max_project_points"))
    if bool(args.get("project_to_surface", True)):
        projection_queries = sum(len(stroke) for stroke in point_sets)
        if projection_queries > projection_budget:
            raise ExecutorError(
                f"sculpt projection workload {projection_queries} exceeds max_project_points {projection_budget}",
                "resource_limit",
            )
        point_sets, projected_count = _project_stroke_point_sets(obj, point_sets)
    radius = float(args["radius"])
    if not 0.0 < radius <= 1000.0:
        raise ExecutorError("radius must be in (0, 1000]", "invalid_args")
    strength = float(args["strength"])
    if not -10.0 <= strength <= 10.0:
        raise ExecutorError("strength must be between -10 and 10", "invalid_args")
    pressure = max(0.0, min(1.0, float(args.get("pressure", 1.0))))
    strength *= pressure
    depth_scale = float(args.get("depth_scale", 0.25))
    if depth_scale < 0.0:
        raise ExecutorError("depth_scale must be non-negative", "invalid_args")
    mode = str(args["mode"]).lower()
    falloff_mode = str(args.get("falloff", "smooth")).lower()
    direction_value = args.get("direction")
    direction = Vector(_as_float3(direction_value, "direction")) if direction_value is not None else Vector((0, 0, 1))
    if direction.length < 1e-9:
        raise ExecutorError("direction must be non-zero", "invalid_args")
    direction.normalize()
    normal_blend = max(0.0, min(1.0, float(args.get("normal_blend", 0.0))))
    offset_value = args.get("offset")
    offset = Vector(_as_float3(offset_value, "offset")) if offset_value is not None else (point_sets[0][-1] - point_sets[0][0])
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.normal_update()
        selection = args.get("selection") or {}
        if selection:
            selected_verts, _, _ = _selection_parts(bm, selection, obj=obj, default_all=False)
            mask_indices = {int(vertex.index) for vertex in selected_verts}
        else:
            mask_indices = None
        original = {int(vertex.index): vertex.co.copy() for vertex in bm.verts}
        neighbors: dict[int, list[Any]] = {int(vertex.index): [] for vertex in bm.verts}
        for edge in bm.edges:
            left, right = edge.verts
            neighbors[int(left.index)].append(right)
            neighbors[int(right.index)].append(left)
        affected: set[int] = set()
        for vertex in bm.verts:
            if mask_indices is not None and int(vertex.index) not in mask_indices:
                continue
            point = vertex.co
            best_stroke, best = min(((stroke, _distance_to_polyline(point, stroke)) for stroke in point_sets), key=lambda item: item[1][0])
            nearest, segment_index, segment_t = best
            if nearest > radius:
                continue
            if args.get("front_facing_only", False) and vertex.normal.dot(direction) <= 0:
                continue
            normalized = max(0.0, 1.0 - nearest / radius)
            if falloff_mode == "smooth":
                influence = normalized * normalized * (3.0 - 2.0 * normalized)
            elif falloff_mode == "linear":
                influence = normalized
            else:
                influence = 1.0
            amount = strength * influence
            affected.add(int(vertex.index))
            brush_normal = vertex.normal.normalized()
            if normal_blend > 0.0:
                brush_normal = brush_normal.lerp(direction, normal_blend)
                if brush_normal.length < 1e-9:
                    brush_normal = vertex.normal.normalized()
                else:
                    brush_normal.normalize()
            if mode in {"draw", "inflate"}:
                vertex.co = point + brush_normal * amount * radius * depth_scale
            elif mode == "crease":
                vertex.co = point - brush_normal * amount * radius * depth_scale
            elif mode == "grab":
                vertex.co = point + offset * amount
            elif mode == "pinch":
                center = min((stroke_point for stroke in point_sets for stroke_point in stroke), key=lambda stroke: (point - stroke).length)
                vertex.co = point + (center - point) * amount
            elif mode == "flatten":
                center = best_stroke[segment_index].lerp(best_stroke[segment_index + 1], segment_t)
                distance = (point - center).dot(direction)
                vertex.co = point - direction * distance * amount
            elif mode in {"smooth", "relax"}:
                adjacent = neighbors.get(int(vertex.index), [])
                if adjacent:
                    average = sum((original[int(item.index)] for item in adjacent), Vector((0, 0, 0))) / len(adjacent)
                    vertex.co = point.lerp(average, min(1.0, abs(amount)))
        _write_bmesh(obj, bm)
        return {
            "target": _stable_uuid(obj), "mode": mode,
            "vertices_affected": len(affected),
            "points": sum(len(stroke) for stroke in point_sets),
            "strokes": len(point_sets), "radius": radius,
            "depth_scale": depth_scale, "pressure": pressure,
            "normal_blend": normal_blend, "projected_points": projected_count,
            "projection_points": projection_queries,
            "projection_budget": projection_budget,
        }
    finally:
        bm.free()

def _sculpt_stroke_transactional(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply optional deterministic repeats and cleanup passes around one stroke."""
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    used_multires_base = _ensure_materialized_stroke_surface(obj, args)
    repeat = max(1, min(8, int(args.get("repeat", 1))))
    smooth_passes = max(0, min(5, int(args.get("smooth_passes", 0))))
    planned_projection, projection_budget = _validate_stroke_projection_budget(args)
    base = dict(args)
    base["repeat"] = 1
    base["smooth_passes"] = 0
    result = _sculpt_stroke_transactional_once(base)
    passes = 1
    for _ in range(repeat - 1):
        result = _sculpt_stroke_transactional_once(base)
        passes += 1
    if smooth_passes and str(args.get("mode", "")).lower() not in {"smooth", "relax"}:
        cleanup = dict(base)
        cleanup.update({
            "mode": "smooth", "strength": min(1.0, max(0.05, abs(float(args.get("strength", 0.0))) * 0.35)),
            "radius": float(args["radius"]) * 1.15,
        })
        for _ in range(smooth_passes):
            result = _sculpt_stroke_transactional_once(cleanup)
            passes += 1
    return {
        **result, "passes": passes, "repeat": repeat,
        "smooth_passes": smooth_passes,
        "allow_multires_base": bool(args.get("allow_multires_base", False)),
        "used_multires_base": used_multires_base,
        "projection_points": planned_projection,
        "projection_budget": projection_budget,
    }

def _merged_stroke_point_sets(args: Mapping[str, Any]) -> list[list[Any]]:
    """Build independent authored and mirrored paths for the merged stroke."""
    return _mirror_point_sets(
        _sculpt_points(args),
        args.get("symmetry") or {},
    )


def _stroke_projects_to_surface(args: Mapping[str, Any]) -> bool:
    """Resolve the two historical projection flags to one deterministic rule."""
    if "surface_project" in args:
        return bool(args["surface_project"])
    if "project_to_surface" in args:
        return bool(args["project_to_surface"])
    return True

def _sculpt_stroke_once(
    obj: Any,
    args: Mapping[str, Any],
    authored_point_sets: list[list[Any]],
) -> Dict[str, Any]:
    """Apply one merged stroke pass in one bmesh operation."""
    radius = float(args["radius"])
    if not 0.0 < radius <= 1000.0:
        raise ExecutorError("radius must be in (0, 1000]", "invalid_args")
    strength = float(args["strength"])
    if not -10.0 <= strength <= 10.0:
        raise ExecutorError("strength must be between -10 and 10", "invalid_args")
    pressure = max(0.0, min(1.0, float(args.get("pressure", 1.0))))
    strength *= pressure
    depth_scale = float(args.get("depth_scale", 0.25))
    if depth_scale < 0.0:
        raise ExecutorError("depth_scale must be non-negative", "invalid_args")
    mode = str(args["mode"]).lower()
    falloff_mode = str(args.get("falloff", "smooth")).lower()
    direction_value = args.get("direction")
    direction = (
        Vector(_as_float3(direction_value, "direction"))
        if direction_value is not None
        else Vector((0, 0, 1))
    )
    if direction.length < 1e-9:
        raise ExecutorError("direction must be non-zero", "invalid_args")
    direction.normalize()
    view_value = args.get("view_direction")
    view_direction = (
        Vector(_as_float3(view_value, "view_direction"))
        if view_value is not None
        else direction.copy()
    )
    if view_direction.length < 1e-9:
        raise ExecutorError("view_direction must be non-zero", "invalid_args")
    view_direction.normalize()
    normal_blend = max(0.0, min(1.0, float(args.get("normal_blend", 0.0))))
    offset_value = args.get("offset")
    offset = (
        Vector(_as_float3(offset_value, "offset"))
        if offset_value is not None
        else authored_point_sets[0][-1] - authored_point_sets[0][0]
    )
    backface_policy = str(
        args.get(
            "backface_policy",
            "FRONT_ONLY" if args.get("front_facing_only", False) else "ALLOW",
        )
    ).upper()
    if backface_policy not in {"ALLOW", "FRONT_ONLY", "NORMAL_REJECT"}:
        raise ExecutorError(f"unsupported backface_policy: {backface_policy}", "invalid_args")
    max_displacement = float(args.get("max_displacement", 0.0))
    if max_displacement < 0:
        raise ExecutorError("max_displacement must be non-negative", "invalid_args")
    should_project = _stroke_projects_to_surface(args)

    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.normal_update()
        selection = args.get("selection") or {}
        selected, _, _ = _selection_parts(
            bm,
            selection,
            obj=obj,
            default_all=True,
        )
        allowed = {int(vertex.index) for vertex in selected}
        numeric_selection = any(
            selection.get(key)
            for key in ("vertex_indices", "edge_indices", "face_indices")
        )
        detail_passes = 0
        requested_detail = int(args.get("detail_level", 0))
        if requested_detail > 0 and not numeric_selection:
            for _ in range(min(requested_detail, 2)):
                candidates = [edge for edge in bm.edges if edge.calc_length() > radius * 0.5]
                if not candidates or len(candidates) > 100000:
                    break
                bmesh.ops.subdivide_edges(
                    bm,
                    edges=candidates,
                    cuts=1,
                    use_grid_fill=True,
                )
                detail_passes += 1
                bm.verts.ensure_lookup_table()
                bm.edges.ensure_lookup_table()
                bm.normal_update()
            selected, _, _ = _selection_parts(
                bm,
                selection,
                obj=obj,
                default_all=True,
            )
            allowed = {int(vertex.index) for vertex in selected}

        point_sets = authored_point_sets
        projected_count = 0
        if should_project:
            if BVHTree is None:
                raise ExecutorError("surface projection is unavailable", "execution_error")
            surface_tree = BVHTree.FromBMesh(bm)
            projected_sets = []
            for stroke in point_sets:
                projected = []
                for point in stroke:
                    nearest = surface_tree.find_nearest(point)
                    projected.append(nearest[0] if nearest and nearest[0] is not None else point)
                    projected_count += int(nearest is not None and nearest[0] is not None)
                projected_sets.append(projected)
            point_sets = projected_sets

        original = {int(vertex.index): vertex.co.copy() for vertex in bm.verts}
        neighbors: dict[int, list[Any]] = {int(vertex.index): [] for vertex in bm.verts}
        for edge in bm.edges:
            left, right = edge.verts
            neighbors[int(left.index)].append(right)
            neighbors[int(right.index)].append(left)
        affected: set[int] = set()
        displacements: list[float] = []
        protect_name = args.get("protect_vertex_group")
        protect_group = obj.vertex_groups.get(str(protect_name)) if protect_name else None
        protect_weight = float(args.get("protect_weight", 0.5))
        for vertex in bm.verts:
            vertex_index = int(vertex.index)
            if vertex_index not in allowed:
                continue
            point = vertex.co.copy()
            best_stroke, best = min(
                (
                    (stroke, _distance_to_polyline(point, stroke))
                    for stroke in point_sets
                ),
                key=lambda item: item[1][0],
            )
            nearest, segment_index, segment_t = best
            if nearest > radius:
                continue
            if backface_policy == "FRONT_ONLY" and vertex.normal.dot(view_direction) <= 0:
                continue
            if backface_policy == "NORMAL_REJECT" and abs(vertex.normal.dot(view_direction)) < 0.1:
                continue
            if protect_group is not None:
                try:
                    if protect_group.weight(vertex_index) >= protect_weight:
                        continue
                except RuntimeError:
                    pass
            normalized = max(0.0, 1.0 - nearest / radius)
            if falloff_mode == "smooth":
                influence = normalized * normalized * (3.0 - 2.0 * normalized)
            elif falloff_mode == "linear":
                influence = normalized
            else:
                influence = 1.0
            amount = strength * influence
            affected.add(vertex_index)
            normal = vertex.normal.normalized() if vertex.normal.length > 1e-9 else direction
            if normal_blend > 0.0:
                normal = normal.lerp(direction, normal_blend)
                normal = normal.normalized() if normal.length > 1e-9 else direction
            if mode == "draw":
                vertex.co = point + direction * amount * radius * depth_scale
            elif mode == "inflate":
                vertex.co = point + normal * amount * radius * depth_scale
            elif mode == "crease":
                vertex.co = point - normal * amount * radius * depth_scale
            elif mode == "grab":
                vertex.co = point + offset * amount
            elif mode == "pinch":
                center = best_stroke[segment_index].lerp(
                    best_stroke[segment_index + 1],
                    segment_t,
                )
                vertex.co = point + (center - point) * amount
            elif mode == "flatten":
                center = best_stroke[segment_index].lerp(
                    best_stroke[segment_index + 1],
                    segment_t,
                )
                distance = (point - center).dot(direction)
                vertex.co = point - direction * distance * amount
            elif mode in {"smooth", "relax"}:
                adjacent = neighbors.get(vertex_index, [])
                if adjacent:
                    average = sum(
                        (original[int(item.index)] for item in adjacent),
                        Vector((0, 0, 0)),
                    ) / len(adjacent)
                    vertex.co = point.lerp(average, min(1.0, abs(amount)))
            else:
                raise ExecutorError(f"unsupported sculpt mode: {mode}", "invalid_args")
            if max_displacement > 0:
                changed = vertex.co - point
                if changed.length > max_displacement:
                    vertex.co = point + changed.normalized() * max_displacement
            displacements.append((vertex.co - point).length)
        _write_bmesh(obj, bm)
        return {
            "target": _stable_uuid(obj),
            "mode": mode,
            "vertices_affected": len(affected),
            "points": sum(len(stroke) for stroke in point_sets),
            "strokes": len(point_sets),
            "paths": len(point_sets),
            "radius": radius,
            "depth_scale": depth_scale,
            "pressure": pressure,
            "normal_blend": normal_blend,
            "surface_projected": should_project,
            "projected_points": projected_count,
            "detail_passes": detail_passes,
            "max_displacement": round(max(displacements, default=0.0), 8),
            "mean_displacement": (
                round(sum(displacements) / len(displacements), 8)
                if displacements
                else 0.0
            ),
            "backface_policy": backface_policy,
            "region_handles": _region_handles(obj),
        }
    finally:
        bm.free()

def _sculpt_stroke_extended(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge root stroke controls with sandbox masks in one transaction."""
    _require_bpy()
    target = _require_mesh_object(args["target"])
    used_multires_base = _ensure_materialized_stroke_surface(target, args)
    point_sets = _merged_stroke_point_sets(args)
    if not point_sets:
        raise ExecutorError("sculpt.stroke requires at least two points", "invalid_args")
    repeat = max(1, min(8, int(args.get("repeat", 1))))
    smooth_passes = max(0, min(5, int(args.get("smooth_passes", 0))))
    cleanup_passes = (
        smooth_passes
        if str(args.get("mode", "")).lower() not in {"smooth", "relax"}
        else 0
    )
    should_project = _stroke_projects_to_surface(args)
    projection_budget = _project_point_budget(args.get("max_project_points"))
    projection_points = (
        sum(len(stroke) for stroke in point_sets) * (repeat + cleanup_passes)
        if should_project
        else 0
    )
    if projection_points > projection_budget:
        raise ExecutorError(
            f"sculpt projection workload {projection_points} exceeds "
            f"max_project_points {projection_budget}",
            "resource_limit",
        )

    base = dict(args)
    base["repeat"] = 1
    base["smooth_passes"] = 0
    with _object_transaction(target):
        result = _sculpt_stroke_once(target, base, point_sets)
        passes = 1
        detail_passes = int(result.get("detail_passes", 0))
        repeated = dict(base)
        repeated["detail_level"] = 0
        for _ in range(repeat - 1):
            result = _sculpt_stroke_once(target, repeated, point_sets)
            detail_passes += int(result.get("detail_passes", 0))
            passes += 1
        if cleanup_passes:
            cleanup = dict(repeated)
            cleanup.update({
                "mode": "smooth",
                "strength": min(
                    1.0,
                    max(0.05, abs(float(args.get("strength", 0.0))) * 0.35),
                ),
                "radius": float(args["radius"]) * 1.15,
            })
            for _ in range(cleanup_passes):
                result = _sculpt_stroke_once(target, cleanup, point_sets)
                detail_passes += int(result.get("detail_passes", 0))
                passes += 1
    return {
        **result,
        "passes": passes,
        "repeat": repeat,
        "smooth_passes": smooth_passes,
        "allow_multires_base": bool(args.get("allow_multires_base", False)),
        "used_multires_base": used_multires_base,
        "detail_passes": detail_passes,
        "projection_points": projection_points,
        "projection_budget": projection_budget,
    }

def _sculpt_stroke(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply the complete root and sandbox stroke contract."""
    sandbox_fields = {
        "surface_project", "view_direction", "backface_policy",
        "protect_vertex_group", "protect_weight", "max_displacement", "detail_level",
    }
    if sandbox_fields.intersection(args):
        return _sculpt_stroke_extended(args)
    return _sculpt_stroke_transactional(args)


def _sculpt_path_relief_single(args: Mapping[str, Any], *, mode: str, amplitude: float) -> Dict[str, Any]:
    """Apply a tapered normal relief with explicit longitudinal falloff."""
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    points = _sculpt_points(args)
    if len(points) < 2:
        raise ExecutorError(f"sculpt.{mode} requires at least two points", "invalid_args")
    width = float(args["width"])
    taper = float(args.get("taper", 1.0))
    falloff = str(args.get("falloff", "smooth")).lower()
    selection = args.get("selection") or {}
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.normal_update()
        selected, _, _ = _selection_parts(bm, selection, obj=obj, default_all=True)
        allowed = {int(vertex.index) for vertex in selected}
        if bool(args.get("surface_project", False)):
            if BVHTree is None:
                raise ExecutorError("surface projection is unavailable", "execution_error")
            tree = BVHTree.FromBMesh(bm)
            projected = []
            for point in points:
                nearest = tree.find_nearest(point)
                projected.append(nearest[0] if nearest and nearest[0] is not None else point)
            points = projected
        lengths = [0.0]
        for index in range(len(points) - 1):
            lengths.append(lengths[-1] + (points[index + 1] - points[index]).length)
        total_length = max(lengths[-1], 1e-9)
        protect_name = args.get("protect_vertex_group")
        protect_group = obj.vertex_groups.get(str(protect_name)) if protect_name else None
        protect_weight = float(args.get("protect_weight", 0.5))
        affected = 0
        displacements: list[float] = []
        for vertex in bm.verts:
            if int(vertex.index) not in allowed:
                continue
            distance, segment, segment_t = _distance_to_polyline(vertex.co, points)
            if distance > width:
                continue
            if protect_group is not None:
                try:
                    if protect_group.weight(int(vertex.index)) >= protect_weight:
                        continue
                except RuntimeError:
                    pass
            radial = max(0.0, 1.0 - distance / width)
            if falloff == "smooth":
                radial = radial * radial * (3.0 - 2.0 * radial)
            elif falloff == "constant":
                radial = 1.0
            arc = (lengths[segment] + segment_t * (lengths[segment + 1] - lengths[segment])) / total_length
            longitudinal = math.sin(math.pi * max(0.0, min(1.0, arc)))
            longitudinal = longitudinal ** max(0.0, taper) if taper > 0 else 1.0
            amount = amplitude * radial * longitudinal
            normal = vertex.normal.normalized() if vertex.normal.length > 1e-9 else Vector((0, 0, 1))
            vertex.co = vertex.co + normal * amount
            if abs(amount) > 1e-12:
                affected += 1
            displacements.append(abs(amount))
        _write_bmesh(obj, bm)
        return {
            "target": _stable_uuid(obj), "mode": mode, "vertices_affected": affected,
            "points": len(points), "width": width, "amplitude": amplitude, "taper": taper,
            "surface_projected": bool(args.get("surface_project", False)),
            "max_displacement": round(max(displacements, default=0.0), 8),
            "mean_displacement": round(sum(displacements) / len(displacements), 8) if displacements else 0.0,
            "region_handles": _region_handles(obj),
        }
    finally:
        bm.free()

def _sculpt_path_relief(args: Mapping[str, Any], *, mode: str, amplitude: float) -> Dict[str, Any]:
    """Apply tapered relief independently to authored and mirrored paths."""
    paths = _mirror_point_sets(_sculpt_points(args), args.get("symmetry") or {})
    if not paths:
        raise ExecutorError(f"sculpt.{mode} requires at least two points", "invalid_args")
    results = []
    for path in paths:
        payload = dict(args)
        payload["points"] = [[float(v) for v in point] for point in path]
        payload["symmetry"] = {}
        results.append(_sculpt_path_relief_single(payload, mode=mode, amplitude=amplitude))
    return {
        **results[-1],
        "paths": len(results),
        "vertices_affected": sum(int(item.get("vertices_affected", 0)) for item in results),
        "max_displacement": max(float(item.get("max_displacement", 0.0)) for item in results),
        "mean_displacement": round(sum(float(item.get("mean_displacement", 0.0)) for item in results) / len(results), 8),
    }


def _sculpt_multires(args: Mapping[str, Any]) -> Dict[str, Any]:
    _require_bpy()
    obj = _require_mesh_object(args["target"])
    modifier = next((item for item in obj.modifiers if item.type == "MULTIRES"), None)
    if modifier is None:
        modifier = obj.modifiers.new("ToolboxMultires", "MULTIRES")
    if "target_level" in args:
        requested = max(0, int(args["target_level"]) - int(modifier.total_levels))
    elif "levels_to_add" in args:
        requested = int(args["levels_to_add"])
    else:
        requested = int(args.get("levels", 1))
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    try:
        for _ in range(max(0, requested)):
            bpy.ops.object.multires_subdivide(modifier=modifier.name, mode="CATMULL_CLARK")
    finally:
        obj.select_set(False)
    modifier.levels = min(int(args.get("sculpt_level", modifier.total_levels)), modifier.total_levels)
    modifier.sculpt_levels = min(int(args.get("sculpt_level", modifier.sculpt_levels)), modifier.total_levels)
    modifier.render_levels = min(int(args.get("render_level", modifier.render_levels)), modifier.total_levels)
    return {"target": _stable_uuid(obj), "modifier": modifier.name, "levels_added": requested, "total_levels": modifier.total_levels, "sculpt_levels": modifier.sculpt_levels, "render_levels": modifier.render_levels}

def _face_shape_key_landmarks(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a localized facial expression shape key from semantic landmarks."""
    _require_bpy()
    target = _require_mesh_object(args["target"])
    refs = args.get("landmarks") or []
    offsets = args.get("offsets") or []
    if len(refs) != len(offsets) or not refs:
        raise ExecutorError("landmarks and offsets must have the same non-zero length", "invalid_args")
    inverse = target.matrix_world.inverted()
    points = []
    for ref in refs:
        landmark = _landmark_object(ref)
        points.append(inverse @ landmark.matrix_world.translation.copy())
    parsed_offsets = [Vector(_as_float3(value, "offsets[]")) for value in offsets]
    radius = float(args["radius"])
    strength = float(args.get("strength", 1.0))
    # Mirror both landmark locations and displacement vectors.  Mirroring the
    # vector is important for lateral motion (for example a corner pull),
    # while deduplication prevents points on the symmetry plane being doubled.
    landmark_pairs = [(point.copy(), offset.copy()) for point, offset in zip(points, parsed_offsets)]
    variants = list(landmark_pairs)
    symmetry = args.get("symmetry") or {}
    for axis, enabled in enumerate(bool(symmetry.get(name, False)) for name in ("x", "y", "z")):
        if not enabled:
            continue
        mirrored = []
        for point, offset in variants:
            reflected_point = point.copy()
            reflected_offset = offset.copy()
            reflected_point[axis] *= -1.0
            reflected_offset[axis] *= -1.0
            mirrored.append((reflected_point, reflected_offset))
        variants.extend(mirrored)
    unique_pairs = []
    seen_pairs = set()
    for point, offset in variants:
        key = tuple(round(float(value), 8) for value in (*point, *offset))
        if key not in seen_pairs:
            seen_pairs.add(key)
            unique_pairs.append((point, offset))
    key_name = str(args["name"])
    if target.data.shape_keys and target.data.shape_keys.key_blocks.get(key_name):
        key = target.data.shape_keys.key_blocks[key_name]
    else:
        if target.data.shape_keys is None:
            target.shape_key_add(name="Basis", from_mix=False)
        key = target.shape_key_add(name=key_name, from_mix=False)
    basis = target.data.shape_keys.key_blocks.get("Basis") or target.data.shape_keys.key_blocks[0]
    affected = 0
    for index, vertex in enumerate(target.data.vertices):
        coordinate = vertex.co.copy()
        delta = Vector((0.0, 0.0, 0.0))
        for point, offset in unique_pairs:
            distance = (coordinate - point).length
            if distance >= radius:
                continue
            normalized = max(0.0, 1.0 - distance / radius)
            influence = normalized * normalized * (3.0 - 2.0 * normalized)
            delta += offset * (influence * strength)
        if delta.length > 1e-12:
            affected += 1
        key.data[index].co = basis.data[index].co + delta
    key.value = float(args.get("value", 0.0))
    target.data.update()
    return {
        "target": _stable_uuid(target), "shape_key": key.name,
        "vertices_affected": affected, "landmarks": list(refs),
        "radius": radius,
        "symmetry_axes": sorted(name for name in ("x", "y", "z") if bool(symmetry.get(name, False))),
    }

def _render(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Render is deliberately explicit and low-frequency; it does not alter the scene revision."""
    output_dir = Path(str(args.get("output_dir", "renders"))).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolution = int(args.get("resolution", 256))
    scene = bpy.context.scene
    old_engine = scene.render.engine
    old_resolution = (scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage)
    old_filepath = scene.render.filepath
    try:
        render_frame = _coordinate_frame(args)
        scene.render.engine = _resolve_render_engine(scene, str(args.get("engine", "BLENDER_EEVEE")))
        scene.render.resolution_x = resolution
        scene.render.resolution_y = resolution
        scene.render.resolution_percentage = 100
        views = args.get("views") or [{"name": "005", "location": (4, -4, 3)}, {"name": "015", "location": (0, -5, 3)}, {"name": "025", "location": (-4, -4, 3)}, {"name": "035", "location": (0, 4, 3)}]
        if not isinstance(views, list) or any(not isinstance(view, Mapping) for view in views):
            raise ExecutorError("views must be an array of objects", "invalid_args")
        camera = bpy.data.objects.get("ToolboxCamera")
        created_camera = camera is None or getattr(camera, "type", None) != "CAMERA"
        old_scene_camera = scene.camera
        old_camera_transform = None if created_camera else (camera.location.copy(), camera.rotation_euler.copy())
        if created_camera:
            if camera is not None and camera.name in bpy.data.objects:
                bpy.data.objects.remove(camera, do_unlink=True)
            bpy.ops.object.camera_add(location=(4, -4, 3))
            camera = bpy.context.object
            camera.name = "ToolboxCamera"
        scene.camera = camera
        rendered = []
        view_metadata = []
        seen_view_names: set[str] = set()
        requested_types = [str(value) for value in (args.get("evidence_types") or [])]
        for index, view in enumerate(views):
            view_frame = dict(render_frame)
            if isinstance(view, Mapping) and isinstance(view.get("coordinate_frame"), Mapping):
                view_frame.update(_coordinate_frame({"coordinate_frame": view["coordinate_frame"]}))
            location = tuple(_point_to_world(view.get("location", (4, -4, 3)), "views[].location", view_frame))
            camera.location = location
            view_target = _point_to_world(view.get("target", args.get("target", (0.0, 0.0, 0.0))), "views[].target", view_frame)
            direction = view_target - camera.location
            if direction.length > 1e-9:
                camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
            view_name = str(view.get("name", "view"))
            if view_name in seen_view_names:
                raise ExecutorError(f"render view names must be unique: {view_name}", "invalid_args")
            seen_view_names.add(view_name)
            filename = output_dir / f"Image_{view_name}.png"
            scene.render.filepath = str(filename)
            bpy.ops.render.render(write_still=True)
            rendered.append(str(filename))
            horizontal = math.sqrt(location[0] ** 2 + location[1] ** 2)
            view_metadata.append({
                "name": view_name,
                "path": str(filename),
                "location": [round(float(value), 8) for value in location],
                "azimuth_deg": round(math.degrees(math.atan2(location[1], location[0])), 3),
                "elevation_deg": round(math.degrees(math.atan2(location[2], max(horizontal, 1e-12))), 3),
                "evidence_type": str(view.get("evidence_type", requested_types[index] if index < len(requested_types) else "beauty")),
                "target": [round(float(value), 8) for value in view_target],
            })
        file_hashes = {str(path): _file_hash(path) for path in rendered}
        return {
            "output_dir": str(output_dir),
            "files": rendered,
            "views": view_metadata,
            "n_views_rendered": len(rendered),
            "file_hashes": file_hashes,
            "evidence_types": sorted({item["evidence_type"] for item in view_metadata}),
            "quality_stage": str(args.get("quality_stage", "evidence")),
            "target": [round(float(value), 8) for value in _point_to_world(args.get("target", (0.0, 0.0, 0.0)), "target", render_frame)],
        }
    finally:
        scene.render.engine = old_engine
        scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = old_resolution
        scene.render.filepath = old_filepath
        if created_camera and camera and camera.name in bpy.data.objects:
            bpy.data.objects.remove(camera, do_unlink=True)
        elif camera and old_camera_transform is not None:
            camera.location, camera.rotation_euler = old_camera_transform
        scene.camera = old_scene_camera


def _file_hash(path: Any) -> Optional[str]:
    try:
        target = Path(str(path)).expanduser().resolve()
        if not target.is_file():
            return None
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except (OSError, TypeError, ValueError):
        return None


def _anti_slop_diagnostics(objects: Optional[Iterable[Any]] = None) -> Dict[str, Any]:
    """Run conservative objective checks that catch recognizable-but-sloppy scenes."""
    if bpy is None:
        return {"gate": False, "checks": {}, "blockers": ["blender_unavailable"]}
    source = objects if objects is not None else bpy.context.scene.objects
    visible = [obj for obj in source if obj.type in {"MESH", "CURVE", "SURFACE", "FONT"} and not obj.hide_render]
    tags = {obj.name: set(_semantic_tags(obj)) for obj in visible}
    blockers: list[str] = []
    checks: Dict[str, Any] = {}
    hashes: Dict[str, list[Any]] = {}
    for obj in visible:
        geometry_hash = _mesh_geometry_hash(obj) or _curve_geometry_hash(obj)
        if geometry_hash:
            hashes.setdefault(geometry_hash, []).append(obj)
    repeated = []
    for geometry_hash, group in hashes.items():
        if len(group) >= 4:
            signatures = {(tuple(round(float(v), 4) for v in obj.rotation_euler), tuple(round(float(v), 4) for v in obj.scale)) for obj in group}
            varied = any({"variation", "procedural_variation", "nonuniform"} & tags.get(obj.name, set()) for obj in group)
            if len(signatures) <= 1 and not varied:
                repeated.append({"geometry_hash": geometry_hash, "count": len(group), "objects": [obj.name for obj in group[:16]]})
    checks["repeated_hard_shapes"] = {"passed": not repeated, "groups": repeated}
    if repeated: blockers.append("repeated_hard_shapes")
    primitiveish = [obj for obj in visible if {"primitive", "detail", "appendage"} & tags.get(obj.name, set())]
    overlap_pairs = []
    for index, left in enumerate(primitiveish):
        for right in primitiveish[index + 1:]:
            item = _aabb_relationship(left, right)
            valid_metadata = _documented_pair_relations(left, right)
            documented_relation = any(bool(check.get("gate")) for check in valid_metadata)
            if item["overlap"] and not documented_relation:
                overlap_pairs.append((left.name, right.name))
    checks["primitive_seams"] = {"passed": not overlap_pairs, "pairs": overlap_pairs[:64]}
    if overlap_pairs: blockers.append("primitive_seams")
    spatial_failures = []
    # Keep this objective check scoped to the same objects as verify.run.  A
    # focused target audit must not inherit a floating helper from a separate
    # asset elsewhere in the Blender scene.
    for index, left in enumerate(visible):
        for right in visible[index + 1:]:
            item = _aabb_relationship(left, right)
            if not (item["overlap"] or item["aabb_gap"] <= 0.05):
                continue
            if not item["overlap"] and not ((tags.get(left.name, set()) | tags.get(right.name, set())) & {"detail", "appendage", "landmark", "feather", "accessory"}):
                continue
            # Raw parentage is ownership, never proof of physical contact.
            # Only an explicit Toolbox relation that passes its own residual,
            # normal, and clearance checks may waive an overlap blocker.
            documented_relation = any(bool(check.get("gate")) for check in _documented_pair_relations(left, right))
            if not documented_relation:
                spatial_failures.append({**item, "gate": False, "reason": "undocumented_spatial_pair"})
    checks["floating_details"] = {"passed": not spatial_failures, "failures": spatial_failures[:64]}
    if spatial_failures: blockers.append("floating_details")
    bird_like = any(tags.get(obj.name, set()) & {"chicken", "bird", "animal"} for obj in visible)
    tail_count = sum(1 for values in tags.values() if {"tail_feather", "flight_feather"} & values)
    wing_panels = [obj.name for obj in visible if "wing" in tags.get(obj.name, set()) and "feather" not in tags.get(obj.name, set()) and not ({"carrier", "envelope", "root_structure"} & tags.get(obj.name, set()))]
    checks["tail_fan"] = {"passed": (not bird_like or tail_count >= 5), "count": tail_count}
    checks["wing_panel_look"] = {"passed": (not bird_like or not wing_panels), "panel_objects": wing_panels}
    if bird_like and tail_count < 5: blockers.append("tail_fan")
    if wing_panels: blockers.append("wing_panel_look")
    missing_transitions = [obj.name for obj in visible if (tags.get(obj.name, set()) & {"appendage", "comb", "wattle", "beak", "foot", "toe", "wing", "tail"}) and obj.parent is None and not ({"root_structure", "carrier"} & tags.get(obj.name, set()))]
    checks["anatomical_transitions"] = {"passed": not missing_transitions, "objects": missing_transitions[:64]}
    if missing_transitions: blockers.append("anatomical_transitions")
    checks["regular_rows"] = {"passed": None, "requires_review": True}
    checks["material_boundaries"] = {"passed": None, "requires_review": True}
    checks["camera_crops"] = {"passed": None, "requires_review": True}
    checks["reference_consistency"] = {"passed": None, "requires_review": True}
    unknown = sorted(key for key, value in checks.items() if value.get("passed") is not True)
    # Unknown visual-only dimensions are intentionally delegated to the
    # checklist in evidence.visual_review. Objective anti-slop fails only on
    # concrete blockers that can be established from scene data.
    return {"gate": not blockers, "checks": checks, "blockers": sorted(set(blockers)), "unknown_checks": unknown}


def _visual_review(args: Mapping[str, Any], *, current_revision: int, last_render: Optional[Mapping[str, Any]] = None, audit_objects: Optional[Iterable[Any]] = None) -> Dict[str, Any]:
    revision = int(args.get("revision", -1))
    if revision != current_revision:
        raise ExecutorError(f"visual review revision {revision} does not match current revision {current_revision}", "revision_conflict")
    views = [str(value) for value in (args.get("views") or [])]
    if not views:
        raise ExecutorError("visual review must name at least one rendered view", "invalid_args")
    if not isinstance(last_render, Mapping) or last_render.get("revision") != revision:
        raise ExecutorError("visual review requires render.views from the current revision", "precondition_failed")
    rendered_views = {str(item.get("name", item)) if isinstance(item, Mapping) else str(item) for item in (last_render.get("views") or [])}
    unknown_views = sorted(set(views) - rendered_views)
    if unknown_views:
        raise ExecutorError(f"visual review references views not rendered at this revision: {unknown_views}", "precondition_failed")
    missing_views = sorted(rendered_views - set(views))
    if missing_views:
        raise ExecutorError(f"visual review must cover every rendered view: {missing_views}", "precondition_failed")
    render_stage = str(last_render.get("quality_stage") or "evidence")
    requested_stage = str(args.get("quality_stage") or "")
    if requested_stage != render_stage:
        raise ExecutorError("visual review stage does not match the rendered stage", "precondition_failed")
    checklist = dict(args.get("checklist") or {})
    missing = [key for key in _VISUAL_CHECK_KEYS if key not in checklist]
    if missing:
        raise ExecutorError(f"visual review checklist is missing: {missing}", "invalid_args")
    invalid = [key for key in _VISUAL_CHECK_KEYS if not isinstance(checklist.get(key), bool)]
    if invalid:
        raise ExecutorError(f"visual review checklist values must be boolean: {invalid}", "invalid_args")
    passed = bool(args.get("passed", False))
    findings = [str(value) for value in (args.get("findings") or [])]
    failed_checks = [key for key in _VISUAL_CHECK_KEYS if checklist[key] is False]
    if passed and (failed_checks or findings):
        raise ExecutorError("a passing visual review cannot contain findings or failed checks", "invalid_args")
    if not passed and not failed_checks and not findings and not args.get("blockers"):
        raise ExecutorError("a failed visual review must record a finding, blocker, or failed checklist item", "invalid_args")
    review_mode = str(args.get("review_mode", "standard"))
    result: Dict[str, Any] = {"revision": revision, "quality_stage": str(args.get("quality_stage")), "views": views, "passed": passed, "checklist": checklist, "findings": findings, "review_mode": review_mode}
    if review_mode == "critical":
        if not isinstance(last_render, Mapping) or last_render.get("revision") != revision:
            raise ExecutorError("critical visual review requires the current render", "precondition_failed")
        reviewer = str(args.get("reviewer") or "")
        confidence = args.get("confidence")
        if not reviewer or not isinstance(confidence, (int, float)) or float(confidence) < 0.8:
            raise ExecutorError("critical visual review requires reviewer identity and confidence >= 0.8", "invalid_args")
        scores = args.get("scores")
        if not isinstance(scores, Mapping) or any(key not in scores for key in _VISUAL_SCORE_KEYS):
            raise ExecutorError(f"critical visual review requires scores for {_VISUAL_SCORE_KEYS}", "invalid_args")
        if any(not isinstance(scores[key], (int, float)) or not 0.0 <= float(scores[key]) <= 1.0 for key in _VISUAL_SCORE_KEYS):
            raise ExecutorError("critical visual review scores must be finite numbers in [0,1]", "invalid_args")
        anti_slop = args.get("anti_slop_checks")
        if not isinstance(anti_slop, Mapping) or any(key not in anti_slop for key in _ANTI_SLOP_CHECK_KEYS) or any(not isinstance(anti_slop[key], bool) for key in _ANTI_SLOP_CHECK_KEYS):
            raise ExecutorError(f"critical visual review requires boolean anti_slop_checks for {_ANTI_SLOP_CHECK_KEYS}", "invalid_args")
        blockers = args.get("blockers")
        if not isinstance(blockers, list):
            raise ExecutorError("critical visual review requires an explicit blockers list", "invalid_args")
        reference_views = args.get("reference_views")
        if not isinstance(reference_views, list) or not reference_views:
            raise ExecutorError("critical visual review requires reference_views", "invalid_args")
        expected_files = list(last_render.get("files") or [])
        render_hashes = args.get("render_hashes")
        if not isinstance(render_hashes, Mapping) or any(str(path) not in render_hashes for path in expected_files):
            raise ExecutorError("critical visual review must hash every rendered file", "invalid_args")
        changed = [str(path) for path in expected_files if _file_hash(path) is None or render_hashes.get(str(path)) != _file_hash(path)]
        if changed:
            raise ExecutorError(f"critical visual review render hashes do not match: {changed}", "precondition_failed")
        objective = _anti_slop_diagnostics(audit_objects)
        objective_blockers = list(objective.get("blockers") or [])
        # Objective checks may leave genuinely visual-only dimensions as
        # ``unknown``; those are resolved by the explicit boolean attestations
        # supplied above. Concrete blockers, however, can never be waived.
        anti_slop_evidence = args.get("anti_slop_evidence")
        objective_checks = objective.get("checks") if isinstance(objective, Mapping) else None
        unknown_objective_keys = [key for key in _ANTI_SLOP_CHECK_KEYS if isinstance(objective_checks, Mapping) and isinstance(objective_checks.get(key), Mapping) and objective_checks[key].get("passed") is None]
        if unknown_objective_keys and (not isinstance(anti_slop_evidence, Mapping) or any(key not in anti_slop_evidence or not isinstance(anti_slop_evidence.get(key), list) or not anti_slop_evidence.get(key) for key in unknown_objective_keys)):
            raise ExecutorError("critical visual review requires per-check anti_slop_evidence for objective-unknown checks", "invalid_args")
        if passed and (blockers or objective_blockers or any(anti_slop[key] is not True for key in _ANTI_SLOP_CHECK_KEYS)):
            raise ExecutorError("critical visual review cannot pass with blockers or anti-slop failures", "invalid_args")
        result.update({"reviewer": reviewer, "confidence": float(confidence), "scores": {key: float(scores[key]) for key in _VISUAL_SCORE_KEYS}, "blockers": [str(value) for value in blockers], "render_hashes": {str(key): str(value) for key, value in render_hashes.items()}, "reference_views": [str(value) for value in reference_views], "anti_slop_checks": {key: bool(anti_slop[key]) for key in _ANTI_SLOP_CHECK_KEYS}, "anti_slop_evidence": {str(key): list(value) for key, value in anti_slop_evidence.items()}, "objective_anti_slop": objective})
    return result


def _visual_evidence_gate(last_render: Optional[Mapping[str, Any]], last_review: Optional[Mapping[str, Any]], *, current_revision: int, current_state_hash: Optional[str] = None, quality_stage: Optional[str] = None, require_critical: bool = False, required_views: Optional[Iterable[str]] = None, required_evidence_types: Optional[Iterable[str]] = None, required_review_stages: Optional[Iterable[str]] = None, review_history: Optional[Iterable[Mapping[str, Any]]] = None, min_visual_views: int = 0, min_visual_score: float = 0.0) -> Dict[str, Any]:
    if not isinstance(last_render, Mapping) or last_render.get("revision") != current_revision:
        return {"gate": False, "reason": "missing_current_render"}
    if not isinstance(last_review, Mapping) or last_review.get("revision") != current_revision:
        return {"gate": False, "reason": "missing_current_visual_review"}
    if require_critical and not isinstance(current_state_hash, str):
        return {"gate": False, "reason": "missing_current_state_hash"}
    if current_state_hash is not None and last_render.get("state_hash") != current_state_hash:
        return {"gate": False, "reason": "render_state_changed"}
    if current_state_hash is not None and last_review.get("state_hash") != current_state_hash:
        return {"gate": False, "reason": "review_state_changed"}
    # Every render is content-addressed.  Checking the recorded hashes even
    # for standard reviews prevents a replaced PNG from being paired with an
    # otherwise valid checklist.
    recorded_hashes = last_render.get("file_hashes") if isinstance(last_render.get("file_hashes"), Mapping) else {}
    changed_files = []
    for path in last_render.get("files") or []:
        actual = _file_hash(path)
        if actual is None or recorded_hashes.get(str(path)) != actual:
            changed_files.append(str(path))
    if changed_files:
        return {"gate": False, "reason": "render_file_changed", "files": changed_files}
    expected_stage = str(quality_stage or last_render.get("quality_stage") or "evidence")
    if str(last_render.get("quality_stage") or "evidence") != expected_stage or str(last_review.get("quality_stage")) != expected_stage:
        return {"gate": False, "reason": "visual_review_stage_mismatch", "expected_stage": expected_stage}
    rendered = {str(value.get("name", value)) if isinstance(value, Mapping) else str(value) for value in (last_render.get("views") or [])}
    reviewed = {str(value) for value in (last_review.get("views") or [])}
    if rendered - reviewed:
        return {"gate": False, "reason": "unreviewed_render_views", "missing_views": sorted(rendered - reviewed)}
    if not bool(last_review.get("passed")):
        return {"gate": False, "reason": "visual_review_failed", "findings": list(last_review.get("findings") or [])}
    if require_critical and str(last_review.get("review_mode")) != "critical":
        return {"gate": False, "reason": "critical_visual_review_required"}
    required_stages = {str(value) for value in (required_review_stages or ())}
    if required_stages:
        reviewed_stages = {
            str(item.get("quality_stage"))
            for item in (review_history or ())
            if isinstance(item, Mapping) and item.get("passed") and item.get("revision") == current_revision
        }
        reviewed_stages.add(str(last_review.get("quality_stage")))
        missing_stages = sorted(required_stages - reviewed_stages)
        if missing_stages:
            return {"gate": False, "reason": "missing_required_review_stages", "missing_stages": missing_stages}
    if require_critical:
        anti_slop = last_review.get("anti_slop_checks") if isinstance(last_review.get("anti_slop_checks"), Mapping) else {}
        if any(anti_slop.get(key) is not True for key in _ANTI_SLOP_CHECK_KEYS):
            return {"gate": False, "reason": "anti_slop_review_failed", "checks": dict(anti_slop)}
        objective = last_review.get("objective_anti_slop")
        if isinstance(objective, Mapping) and list(objective.get("blockers") or []):
            return {"gate": False, "reason": "objective_anti_slop_failed", "blockers": list(objective.get("blockers") or [])}
        if isinstance(objective, Mapping) and list(objective.get("unknown_checks") or []):
            resolved = set(objective.get("unknown_checks") or []) & set((last_review.get("anti_slop_evidence") or {}).keys()) if isinstance(last_review.get("anti_slop_evidence"), Mapping) else set()
            unresolved = sorted(set(objective.get("unknown_checks") or []) - resolved)
            if unresolved:
                return {"gate": False, "reason": "objective_anti_slop_incomplete", "checks": unresolved}
    required = {str(value) for value in (required_views or ())}
    if required - rendered:
        return {"gate": False, "reason": "missing_required_views", "missing_views": sorted(required - rendered)}
    if len(rendered) < int(min_visual_views):
        return {"gate": False, "reason": "insufficient_visual_views", "minimum": int(min_visual_views), "actual": len(rendered)}
    evidence_types = {str(value) for value in (last_render.get("evidence_types") or [])}
    missing_types = {str(value) for value in (required_evidence_types or ())} - evidence_types
    if missing_types:
        return {"gate": False, "reason": "missing_required_evidence_types", "missing_evidence_types": sorted(missing_types)}
    if required_evidence_types:
        view_items = [item for item in (last_render.get("views") or []) if isinstance(item, Mapping)]
        by_view = {
            str(item.get("name")): str(item.get("evidence_type"))
            for item in view_items if item.get("evidence_type") is not None
        }
        # Legacy renders may only carry an aggregate evidence_types list.  In
        # that case the aggregate hash is still checked; once any view opts
        # into per-view typing, every rendered view must declare a valid type.
        if by_view:
            missing_view_types = sorted(
                str(item.get("name")) for item in view_items
                if str(item.get("name")) not in by_view or by_view.get(str(item.get("name"))) not in {str(value) for value in required_evidence_types}
            )
            if require_critical:
                present_view_types = set(by_view.values())
                missing_required_types = {str(value) for value in required_evidence_types} - present_view_types
                if missing_required_types:
                    missing_view_types.extend(sorted(f"<missing:{value}>" for value in missing_required_types))
        else:
            missing_view_types = []
        if missing_view_types and require_critical:
            return {"gate": False, "reason": "view_evidence_type_mismatch", "views": missing_view_types}
    scores = last_review.get("scores") if isinstance(last_review.get("scores"), Mapping) else {}
    if (require_critical or min_visual_score > 0.0) and any(float(scores.get(key, 0.0)) < float(min_visual_score) for key in _VISUAL_SCORE_KEYS):
        return {"gate": False, "reason": "visual_score_below_minimum", "scores": dict(scores)}
    if require_critical:
        reviewed_hashes = last_review.get("render_hashes") if isinstance(last_review.get("render_hashes"), Mapping) else {}
        changed = [str(path) for path in last_render.get("files") or [] if _file_hash(path) is None or reviewed_hashes.get(str(path)) != _file_hash(path)]
        if changed:
            return {"gate": False, "reason": "render_file_changed", "files": changed}
    return {"gate": True, "quality_stage": expected_stage, "views": sorted(rendered), "evidence_types": sorted(evidence_types)}

class ToolboxExecutor(_CoreToolboxExecutor):
    """Canonical executor for all Blender Toolbox actions."""

    def execute(self, raw: Mapping[str, Any]) -> Dict[str, Any]:
        if (
            isinstance(raw, Mapping)
            and raw.get("action") == "session.create"
            and self._session_id is None
        ):
            self.revision = 0
        return super().execute(raw)

    def _dispatch(
        self,
        action: str,
        args: Mapping[str, Any],
        request: Optional[Any] = None,
        *,
        required_tags_lock: Optional[Iterable[str]] = None,
    ) -> Any:
        if action == "landmark.project_to_surface":
            return _landmark_project_to_surface(args)
        if action == "sculpt.surface_prepare":
            return _sculpt_surface_prepare(args)
        if action == "sculpt.materialize_multires":
            target = _require_mesh_object(args["target"])
            with _object_transaction(target):
                return _sculpt_materialize_multires(args)
        if action == "sculpt.stroke_batch":
            return _sculpt_stroke_batch(args)
        if action == "sculpt.region_deform_batch":
            return _sculpt_region_deform_batch(args)
        if action == "sculpt.surface_patch_batch":
            return _sculpt_surface_patch_batch(args)
        if action == "inspect.sculpt_quality":
            return _inspect_sculpt_quality(args)
        return super()._dispatch(action, args, request, required_tags_lock=required_tags_lock)


class ToolboxServer(_CoreToolboxServer):
    """Socket server bound to the canonical Blender Toolbox executor."""

    def __init__(self, address: str, *, allow_run_python: bool = False, allow_bpy_apply: bool = False, auth_token: Optional[str] = None) -> None:
        super().__init__(address, allow_run_python=allow_run_python, allow_bpy_apply=allow_bpy_apply, auth_token=auth_token)
        self.executor = ToolboxExecutor(allow_run_python=allow_run_python, allow_bpy_apply=allow_bpy_apply, auth_token=auth_token)


_SERVER: Optional[ToolboxServer] = None


def start_server(address: str = "/tmp/blender_toolbox.sock", *, allow_run_python: bool = False, allow_bpy_apply: bool = False, auth_token: Optional[str] = None) -> ToolboxServer:
    global _SERVER
    if _SERVER is None:
        _SERVER = ToolboxServer(address, allow_run_python=allow_run_python, allow_bpy_apply=allow_bpy_apply, auth_token=auth_token)
        _SERVER.start()
    return _SERVER


def stop_server() -> None:
    global _SERVER
    if _SERVER is not None:
        _SERVER.stop()
        _SERVER = None


def register() -> None:  # pragma: no cover - called by Blender's addon loader.
    address = os.environ.get("BLENDER_TOOLBOX_SOCKET", "/tmp/blender_toolbox.sock")
    token = os.environ.get("BLENDER_TOOLBOX_AUTH_TOKEN") or None
    allow_run_python = os.environ.get("BLENDER_TOOLBOX_ALLOW_RUN_PYTHON", "0") == "1"
    allow_bpy_apply = os.environ.get("BLENDER_TOOLBOX_ALLOW_BPY_APPLY", "0") == "1"
    start_server(address, allow_run_python=allow_run_python, allow_bpy_apply=allow_bpy_apply, auth_token=token)


def unregister() -> None:  # pragma: no cover - called by Blender's addon loader.
    stop_server()


def main() -> None:  # pragma: no cover - exercised by Blender smoke tests.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default="/tmp/blender_toolbox.sock")
    parser.add_argument("--allow-run-python", action="store_true")
    parser.add_argument("--allow-bpy-apply", action="store_true")
    parser.add_argument("--auth-token", default=None)
    import sys
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    args, _ = parser.parse_known_args(argv)
    start_server(args.socket, allow_run_python=args.allow_run_python, allow_bpy_apply=args.allow_bpy_apply, auth_token=args.auth_token)
    if bpy is not None and bpy.app.background:
        while _SERVER and not _SERVER._stop.is_set():
            _SERVER._drain()
            time.sleep(0.25)


if __name__ == "__main__":
    main()
