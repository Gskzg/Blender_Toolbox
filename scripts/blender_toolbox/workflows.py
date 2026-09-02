"""Small, deterministic workflow catalog for task-facing Toolbox clients.

The catalog is intentionally data-only.  It helps a model choose a sensible
subset of the 100+ atomic actions without hiding any scene mutation behind an
untracked script.  Builder implementations may use these profiles as a
starting contract and still emit the canonical actions from ``protocol``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Optional

_COORDINATE_CONTRACT = {
    "units": "meters",
    "up_axis": "Z",
    "handedness": "right",
    "origin": "world_origin",
    "default_axes": {
        "x": "asset_length_or_custom",
        "y": "asset_width_or_custom",
        "z": "asset_height",
    },
}


# This is a decision aid shared by every domain profile.  It deliberately
# describes quality gates rather than a fixed modeling recipe: a vehicle,
# prop, character, or environment can choose a different carrier while still
# being held to the same evidence and representation standard.
QUALITY_BAR = {
    "version": "quality_bar.v2",
    "intent": "production_quality",
    "scope": "domain_neutral",
    "decision_order": [
        "identity_and_scale",
        "representation_and_carrier",
        "primary_silhouette_and_negative_space",
        "secondary_junctions_and_openings",
        "tertiary_detail_on_resolved_carriers",
        "technical_readiness",
        "multi_view_evidence",
    ],
    "representation_rules": [
        {
            "question": "Does this part define the identity or silhouette?",
            "preferred": ["control_mesh", "section_stack", "curve", "sdf", "deformation"],
            "avoid": "unrelated_primitive_pile",
        },
        {
            "question": "Is this repeated or rule-driven?",
            "preferred": ["array", "geometry_nodes", "instancing", "parameterized_transform"],
        },
        {
            "question": "Does it need a real opening, rim, or contact?",
            "preferred": ["boolean_with_rim", "topology_edit", "explicit_thickness"],
        },
        {
            "question": "Does it need local relief or a continuous plane change?",
            "preferred": ["surface_patch", "sculpt", "landmark_driven_deformation"],
        },
    ],
    "stage_gates": {
        "structure": ["semantic_parts", "scale", "symmetry_or_intentional_asymmetry"],
        "primary": ["carrier_declared", "silhouette_resolved", "negative_space_resolved"],
        "secondary": ["junctions_and_openings_resolved", "contacts_declared"],
        "tertiary": ["detail_carrier_has_resolution", "detail_is_masked_or_named"],
        "technical": ["topology_policy", "connected_carriers", "normals", "materials_and_uv_policy"],
        "evidence": ["fixed_multi_view_checks", "verification_matches_current_revision"],
    },
    # Quality-first is now a production bar.  Secondary junctions and
    # tertiary surface work are required unless a task explicitly declares a
    # documented exception (for example a deliberately low-poly asset).
    "stage_policy": {
        "baseline": ["structure", "primary", "secondary", "tertiary", "technical", "evidence"],
        "optional": [],
        "secondary_when": ["secondary_refs", "require_secondary", "contacts", "openings"],
        "tertiary_when": ["detail_regions", "require_detail_plan", "feature_scales"],
    },
    "declaration_contract": {
        "baseline_declarations": ["identity", "scale", "representation", "primary_refs", "reference_views", "technical"],
        "recommended": ["negative_spaces", "secondary_refs", "feature_scales", "detail_regions", "landmarks", "materials"],
        "unknown_is_not_pass": True,
    },
    "resolution_policy": {
        "min_samples_per_feature": 8,
        "carrier_first": True,
        "detail_requires_resolved_primary": True,
        "min_primary_vertices": 512,
        "min_secondary_vertices": 256,
        "min_tertiary_vertices": 192,
        "allow_low_resolution_if_declared": True,
        "low_resolution_exception_requires_reason": True,
        "detail_region_requires_named_measurement": True,
    },
    "connectivity_policy": {
        "require_connected_carriers": True,
        "default_expected_shells": 1,
        "intentional_multi_shell_requires_explicit_expected_shells": True,
    },
    "anti_slop": {
        "primary_primitive_pile": "fail_when_disallowed_by_contract",
        "beauty_render_only": "insufficient_evidence",
        "detail_before_primary": "repair_primary_before_detail",
        "unknown_contract_dimension": "report_unknown_instead_of_passing",
    },
    "defaults": {
        "min_evidence_views": 4,
        "min_samples_per_feature": 8,
        "min_primary_vertices": 512,
        "min_secondary_vertices": 256,
        "min_tertiary_vertices": 192,
        "min_quality": 0.86,
        "required_stages": ["structure", "primary", "secondary", "tertiary", "technical", "evidence"],
        "require_secondary": True,
        "require_detail_plan": True,
        "require_feature_scales": True,
        "require_high_resolution": True,
        "require_connected": True,
        "require_render": True,
    },
}


def quality_bar() -> Dict[str, Any]:
    """Return a defensive copy for session/bootstrap responses."""
    return deepcopy(QUALITY_BAR)


def _merge_contracts(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep-merge workflow contract fragments without sharing mutable data.

    Workflow profiles can inherit a generic quality-first contract and then
    add domain-specific requirements.  A shallow ``dict.update`` would drop
    nested ``quality`` fields (or, worse, return references into the catalog),
    so mappings are merged recursively while scalar/list values from the more
    specific profile replace the inherited value.
    """
    result: Dict[str, Any] = deepcopy(dict(base))
    for key, value in overlay.items():
        name = str(key)
        if isinstance(result.get(name), Mapping) and isinstance(value, Mapping):
            result[name] = _merge_contracts(result[name], value)  # type: ignore[arg-type]
        else:
            result[name] = deepcopy(value)
    return result


_WORKFLOWS: Dict[str, Dict[str, Any]] = {
    "quality_first": {
        "name": "quality_first",
        "title": "Quality-first modeling",
        "description": "Choose a representation before geometry, resolve dense primary and junction forms, complete named surface-detail passes, and require current multi-view evidence before export.",
        "representation": {"primary": "decision_tree", "carrier": "task_declared", "repeated": "derived", "detail": "carrier_resolution_first"},
        "phases": [
            {"name": "open", "actions": ["session.open"], "args": {"include_capabilities": True, "include_examples": False}},
            {"name": "plan", "actions": ["model.plan", "inspect.batch"]},
            {"name": "primary", "actions": ["mesh.from_sections", "mesh.from_pydata", "curve.create", "object.create_batch", "geometry.modifier_stack"]},
            {"name": "secondary", "actions": ["geometry.boolean", "mesh.inset_region", "mesh.extrude_region", "mesh.repair", "object.transform_batch"]},
            {"name": "tertiary", "actions": ["sculpt.surface_patch_batch", "sculpt.stroke_batch", "mesh.attribute_write", "material.assign_batch"]},
            {"name": "technical", "actions": ["mesh.recalculate_normals", "uv.unwrap", "inspect.topology", "inspect.uv", "inspect.material"]},
            {"name": "evidence", "actions": ["inspect.batch", "inspect.quality", "inspect.measure", "verify.run", "render.views"]},
            {"name": "export", "actions": ["artifact.export_glb", "session.close"]},
        ],
        "recommended_contract": {
            "quality": {
                "enforce": True,
                "min_quality": 0.86,
                "min_primary_vertices": 512,
                "min_secondary_vertices": 256,
                "min_tertiary_vertices": 192,
                "min_samples_per_feature": 8,
                "require_secondary": True,
                "require_detail_plan": True,
                "require_feature_scales": True,
                "resolution": {
                    "min_primary_vertices": 512,
                    "min_secondary_vertices": 256,
                    "min_tertiary_vertices": 192,
                    "min_samples_per_feature": 8,
                    "allow_low_resolution": False,
                    "exception_reason": "",
                },
                "technical": {"require_topology": True, "strict_topology": True, "require_high_resolution": True, "require_material": True, "require_uv": True},
                "evidence": {"min_views": 4, "require_render": True, "require_current_revision": True},
                "representation": {
                    "kind": "task_declared",
                    "carrier_options": ["control_mesh", "section_stack", "curve", "sdf", "deformation", "native_generator"],
                    "primary_refs": [],
                },
                "reference_views": ["front", "three_quarter", "side", "top"],
                "required_stages": ["structure", "primary", "secondary", "tertiary", "technical", "evidence"],
                "min_evidence_views": 4,
            },
            "require_closed": False,
            "silhouette_views": ["front", "three_quarter", "side", "top"],
        },
    },
    "general": {
        "name": "general",
        "title": "General asset authoring",
        "description": "A domain-neutral quality-first route for props, characters, environments, architecture, products, and other assets.",
        "base_workflow": "quality_first",
        "representation": {
            "primary": "decision_tree",
            "continuous": ["control_mesh", "section_stack", "curve", "sdf", "deformation"],
            "repeated": ["array", "geometry_nodes", "instancing"],
            "detail": ["surface_patch", "sculpt", "landmark_driven_deformation"],
        },
        "phases": [
            {"name": "open", "actions": ["session.open"], "args": {"include_capabilities": True, "include_examples": False}},
            {"name": "plan", "actions": ["model.plan", "inspect.batch"]},
            {"name": "primary", "actions": ["mesh.from_sections", "mesh.from_pydata", "curve.create", "geometry_nodes.apply_recipe", "object.create_batch"]},
            {"name": "secondary", "actions": ["geometry.boolean", "mesh.inset_region", "mesh.extrude_region", "geometry.modifier_stack"]},
            {"name": "tertiary", "actions": ["sculpt.surface_patch_batch", "sculpt.stroke_batch", "material.assign_batch"]},
            {"name": "technical", "actions": ["mesh.recalculate_normals", "uv.unwrap", "inspect.topology", "inspect.uv", "inspect.material"]},
            {"name": "evidence", "actions": ["inspect.quality", "inspect.measure", "verify.run", "render.views"]},
            {"name": "export", "actions": ["artifact.export_glb", "session.close"]},
        ],
        "recommended_contract": {
            "quality": {
                "enforce": True,
                "required_stages": ["structure", "primary", "secondary", "tertiary", "technical", "evidence"],
            },
        },
    },
    "vehicle": {
        "name": "vehicle",
        "title": "Parametric vehicle authoring",
        "description": "Build a semantic vehicle assembly with batched transforms, materials, modifiers, and staged verification.",
        "base_workflow": "quality_first",
        "representation": {"primary": "mixed", "body": "section_or_surface_envelope", "repeated": "derived", "carrier_action": "mesh.from_sections"},
        "phases": [
            {"name": "open", "actions": ["session.open"], "args": {"include_capabilities": True, "include_examples": False}},
            {"name": "blockout", "actions": ["mesh.from_sections", "object.create_batch", "object.transform_batch", "geometry.modifier_stack"]},
            {"name": "parts", "actions": ["object.duplicate", "object.transform_batch", "material.assign_batch"]},
            {"name": "inspect", "actions": ["inspect.batch", "inspect.topology", "inspect.measure"]},
            {"name": "verify", "actions": ["verify.run", "render.views"]},
            {"name": "export", "actions": ["artifact.export_glb", "session.close"]},
        ],
        "recommended_contract": {
            "required_tags": ["vehicle", "body", "wheel", "tire", "window", "light"],
            "require_closed": True,
            "require_openings": False,
            "silhouette_views": ["front", "side", "top", "three_quarter"],
            "assembly_checks": ["wheel_contact", "body_attachment", "left_right_symmetry"],
        },
        "examples": [
            {
                "action": "session.open",
                "args": {"mode": "new", "reset": True, "profile": "vehicle"},
            },
            {
                "action": "mesh.from_sections",
                "args": {
                    "id": "vehicle.body.shell",
                    "sections": [
                        {"x": -2.6, "width": 1.7, "height": 0.8, "z": 0.75},
                        {"x": -1.2, "width": 2.0, "height": 1.0, "z": 0.7},
                        {"x": 0.8, "width": 2.0, "height": 1.0, "z": 0.7},
                        {"x": 2.6, "width": 1.6, "height": 0.75, "z": 0.72},
                    ],
                    "segments": 32,
                    "profile": "superellipse",
                    "power": 4.0,
                    "cap_ends": True,
                    "smooth_shading": True,
                    "semantic_tags": ["vehicle", "body"],
                },
            },
            {
                "action": "object.create_batch",
                "args": {
                    "objects": [
                        {"kind": "cube", "id": "vehicle.body", "name": "Body", "semantic_tags": ["vehicle", "body"]},
                        {"kind": "cylinder", "id": "vehicle.wheel.front_left", "name": "Wheel_FL", "vertices": 48, "semantic_tags": ["vehicle", "wheel", "tire"]},
                    ],
                    "atomic": True,
                },
            },
            {
                "action": "inspect.batch",
                "args": {"query": {"semantic_tag": "vehicle"}, "limit": 256},
            },
        ],
    },
    "hard_surface": {
        "name": "hard_surface",
        "title": "Hard-surface authoring",
        "description": "Use explicit topology, modifier stacks, openings, repairs, and compact inspection for manufactured forms.",
        "base_workflow": "quality_first",
        "representation": {"primary": "specified_or_mixed", "preferred": ["section_stack", "control_mesh", "boolean_with_rim"]},
        "phases": [
            {"name": "blockout", "actions": ["object.create_batch", "geometry.modifier_stack"]},
            {"name": "topology", "actions": ["mesh.inset_region", "mesh.extrude_region", "geometry.boolean", "mesh.repair"]},
            {"name": "surface", "actions": ["mesh.bevel", "mesh.recalculate_normals", "uv.unwrap", "material.node_graph"]},
            {"name": "inspect", "actions": ["inspect.batch", "inspect.topology", "inspect.uv", "verify.run"]},
        ],
        "recommended_contract": {"require_closed": True, "require_openings": True, "feature_sizes": [0.01]},
    },
    "sculpt": {
        "name": "sculpt",
        "title": "Stage-gated sculpting",
        "description": "Prepare a surface, make bounded primary/secondary/tertiary passes, and inspect density and symmetry after each pass.",
        "base_workflow": "quality_first",
        "representation": {"primary": "deformation_or_multires", "detail": "surface_patch_or_stroke"},
        "phases": [
            {"name": "prepare", "actions": ["sculpt.surface_prepare", "inspect.sculpt_quality"]},
            {"name": "primary", "actions": ["sculpt.region_deform_batch", "sculpt.stroke_batch"]},
            {"name": "secondary", "actions": ["landmark.project_to_surface", "sculpt.surface_patch_batch"]},
            {"name": "cleanup", "actions": ["sculpt.materialize_multires", "mesh.recalculate_normals", "inspect.sculpt_quality"]},
        ],
        "recommended_contract": {"require_closed": True, "symmetry_axes": ["x"], "detail_stages": ["primary", "secondary", "tertiary", "cleanup"]},
    },
}


def recommended_contract(name: str) -> Dict[str, Any]:
    """Return the effective recommended contract for a workflow profile.

    Domain profiles may inherit ``base_workflow`` (for example, ``vehicle``
    inherits ``quality_first``).  Callers should receive the complete,
    deterministic contract rather than only the leaf fragment.  The returned
    value is always a deep copy and is therefore safe for callers to augment.
    """
    key = str(name or "").strip().lower()
    if key not in _WORKFLOWS:
        available = ", ".join(sorted(_WORKFLOWS))
        raise ValueError(f"unknown workflow profile {name!r}; available workflows: {available}")

    visiting: set[str] = set()

    def resolve(profile_name: str) -> Dict[str, Any]:
        if profile_name in visiting:
            raise ValueError(f"workflow profile inheritance cycle at {profile_name!r}")
        visiting.add(profile_name)
        profile = _WORKFLOWS[profile_name]
        base_name = profile.get("base_workflow")
        result = resolve(str(base_name).strip().lower()) if base_name else {}
        result = _merge_contracts(result, profile.get("recommended_contract", {}))
        visiting.remove(profile_name)
        return result

    return resolve(key)


def _summary(profile: Mapping[str, Any], *, include_examples: bool) -> Dict[str, Any]:
    result = {
        "name": profile["name"],
        "title": profile["title"],
        "description": profile["description"],
        "representation": deepcopy(profile.get("representation", {})),
        "phases": deepcopy(profile.get("phases", [])),
        # Include inherited quality requirements in discovery responses.  This
        # keeps ``session.open(profile=...)`` and ``workflow.describe`` in
        # agreement about the contract a builder is expected to satisfy.
        "recommended_contract": recommended_contract(str(profile["name"])),
        "base_workflow": profile.get("base_workflow"),
        "quality_bar": quality_bar(),
    }
    if include_examples and profile.get("examples"):
        result["examples"] = deepcopy(profile["examples"])
    return result


def describe_workflow(name: str, *, include_examples: bool = False) -> Dict[str, Any]:
    key = str(name or "").strip().lower()
    if key not in _WORKFLOWS:
        available = ", ".join(sorted(_WORKFLOWS))
        raise ValueError(f"unknown workflow {name!r}; available workflows: {available}")
    return {"schema": "blender_toolbox.workflow.v1", "coordinate_contract": deepcopy(_COORDINATE_CONTRACT), "workflow": _summary(_WORKFLOWS[key], include_examples=include_examples)}


def capability_catalog(*, profile: Optional[str] = None, include_examples: bool = False) -> Dict[str, Any]:
    selected = str(profile).strip().lower() if profile else None
    if selected and selected not in _WORKFLOWS:
        available = ", ".join(sorted(_WORKFLOWS))
        raise ValueError(f"unknown workflow profile {profile!r}; available workflows: {available}")
    workflows = [_summary(_WORKFLOWS[key], include_examples=include_examples) for key in sorted(_WORKFLOWS)]
    result: Dict[str, Any] = {
        "schema": "blender_toolbox.capabilities.v1",
        "coordinate_contract": deepcopy(_COORDINATE_CONTRACT),
        "quality_bar": quality_bar(),
        "atomic_layers": {
            "scene": ["session.open", "scene.camera_create", "scene.light_create", "render.views"],
            "semantic": ["object.create_batch", "object.transform_batch", "geometry.modifier_stack", "material.assign_batch"],
            "topology": ["mesh.from_sections", "mesh.from_pydata", "mesh.inset_region", "mesh.extrude_region", "geometry.boolean", "mesh.repair", "geometry_nodes.apply_recipe"],
            "surface": ["sculpt.stroke_batch", "sculpt.surface_patch_batch", "geometry.remesh_voxel", "material.apply_recipe"],
            "inspection": ["inspect.batch", "inspect.quality", "inspect.topology", "inspect.measure", "verify.run"],
            "artifacts": ["artifact.save_checkpoint", "artifact.export_glb", "session.close"],
        },
        "workflows": workflows,
        "registry_count": _registry_count(),
    }
    if selected:
        result["selected_workflow"] = _summary(_WORKFLOWS[selected], include_examples=include_examples)
    return result


def _registry_count() -> int:
    # Imported lazily to keep this catalog usable during protocol bootstrap.
    try:
        from .protocol import TOOL_SPECS
        return len(TOOL_SPECS)
    except Exception:
        return 0


__all__ = ["capability_catalog", "describe_workflow", "recommended_contract"]
