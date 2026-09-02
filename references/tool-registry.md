# Toolbox Registry

The bundled protocol exposes 123 Toolbox actions. MCP exposes those actions
plus `trajectory.record_external`, for 124 tools total. The live MCP
`tools/list` response and `blender_toolbox.protocol.tool_registry()` remain the
authoritative schemas and limits.

## Lifecycle And Scene

`session.create`, `session.open`, `session.reset`, `session.close`, `scene.reset`,
`scene.census`, `scene.camera_create`, `scene.light_create`,
`scene.set_camera`, `scene.set_render_settings`, `inspect.scene`,
`inspect.object`, `render.views`, `model.plan`.

## Objects, Collections, Curves, And Parameters

`object.create`, `object.create_batch`, `object.delete`, `object.duplicate`,
`object.join`, `object.transform`, `object.transform_batch`,
`object.transform_apply`, `object.convert`,
`collection.group_objects`, `curve.create`, `curve.subdivide`,
`parameters.sample`.

## Mesh Topology, Regions, And Attributes

`mesh.from_pydata`, `mesh.from_sections`, `mesh.subdivide`, `mesh.extrude_region`,
`mesh.duplicate_region`, `mesh.extrude_individual`, `mesh.inset_individual`,
`mesh.bridge_edge_loops`, `mesh.loop_cut`, `mesh.subdivide_adaptive`, `mesh.inset_region`, `mesh.bevel`,
`mesh.transform_selection`, `mesh.merge_by_distance`,
`mesh.vertex_group_assign`, `mesh.region_define`, `mesh.region_to_loop`,
`mesh.recalculate_normals`, `mesh.shade_smooth`, `mesh.delete_region`,
`mesh.dissolve_region`, `mesh.fill_holes`, `mesh.triangulate`,
`mesh.cut_plane`, `mesh.cut_curve`, `mesh.repair`, `mesh.separate`,
`mesh.symmetrize`, `mesh.attribute_write`, `mesh.attribute_read`,
`mesh.geometry_query`.

## Geometry And Sculpting

`geometry.remesh_voxel`, `geometry.modifier_stack`,
`geometry.boolean`, `geometry.add_modifier`, `geometry.apply_modifier`,
`geometry.shrinkwrap`, `sculpt.surface_prepare`, `sculpt.multires`,
`sculpt.materialize_multires`, `sculpt.stroke`, `sculpt.stroke_batch`,
`sculpt.region_deform_batch`, `sculpt.surface_patch_batch`, `sculpt.ridge`,
`sculpt.groove`, `sculpt.muscle`.

## Materials, UV, Nodes, Hair, And Particles

`material.create`, `material.assign`, `material.assign_batch`, `material.node_graph`,
`material.apply_recipe`, `material.set_input`, `uv.mark_seams`, `uv.clear_seams`, `uv.unwrap`,
`uv.project`, `uv.pack`, `geometry_nodes.create`,
`geometry_nodes.apply_recipe`, `geometry_nodes.set_input`,
`hair.create_strands`, `hair.convert_to_mesh`,
`particles.scatter`.

## Landmarks, Faces, Rigging, And Animation

`landmark.create`, `landmark.create_set`, `landmark.project_to_surface`,
`face.curve_from_landmarks`, `face.curve_network_from_landmarks`,
`face.sculpt_landmarks`, `face.shape_key_landmarks`,
`rig.create_armature`, `rig.bind`, `rig.pose`, `rig.add_constraint`,
`animation.keyframe_transform`, `animation.keyframe_shape_key`,
`animation.set_range`.

## Inspection

`inspect.topology`, `inspect.measure`, `inspect.mesh_region`, `inspect.uv`,
`inspect.material`,
`inspect.geometry_nodes`, `inspect.landmarks`, `inspect.armature`,
`inspect.animation`, `inspect.sculpt_quality`, `inspect.quality`.

## Verification And Artifacts

`verify.run`, `artifact.save_checkpoint`, `artifact.export_glb`, and the
restricted `run_python` escape hatch. Use the
live schema for exact argument names; never infer an action from a similar
Blender operator.

## MCP Meta-tool

`trajectory.record_external` records a diagnostic call made through a separate
MCP server. It is not a Blender mutation action and is intentionally absent
from the Toolbox protocol registry.

## Task-facing helpers

`session.open` accepts `mode: "new"|"resume"`, an optional `profile`,
`quality_profile`, `quality_contract`, and `task_spec`. It is the compact
lifecycle entry point and can reset a scene before the first modeling action.
Quality-first is the default for new episodes; `profile: "general"` selects
the domain-neutral route, while manufactured/organic/architectural profiles
are optional routing hints. Use `quality_profile: "advisory"` only for
legacy/partial assets. Set `include_capabilities: true` to return the selected
workflow, quality bar, coordinate contract, and a matching `quality_plan` in
this same response; `include_scene: true` is available when a compact or full
census is required.

`object.create_batch`, `object.transform_batch`, `material.assign_batch`, and
`geometry.modifier_stack` group repeated work into one deterministic event.
They validate references before mutation where possible and report
`committed`/`rolled_back` status. Use `id` or `ref` on created objects and pass
those stable references to later actions.

`inspect.batch` performs one scene census and supports direct `targets` or a
semantic `query`. The default is compact output; use `detail: "full"` or
`fields: ["geometry_hash", "uv_layers"]` for larger diagnostics. Missing
targets can be reported without failing the action with `strict: false`.

`model.plan` chooses a representation family from task signals before any
mutation. Its production `stages` and `quality_contract_template` cover
structure, primary, secondary, tertiary, technical, and evidence. High-
resolution density, feature sampling, and connected carrier surfaces are
enforced unless a documented exception is declared. Intentional multi-shell
carriers must declare `technical.expected_shells`. `inspect.quality` combines a bounded per-object
technical census with the active contract report; use it for repair guidance,
then use `verify.run` as the authoritative gate. Quality-first keeps all six
stages required; advisory inspection may report undeclared stages as `unknown`.

Quality-first contracts set `quality.evidence.require_render: true` and list
four expected names in `reference_views`. The executor records `render.views`
provenance (revision, scene hash, camera targets, evidence types, and file
hashes), blocks further mutation until the current render is reviewed, and
checks that the review belongs to the current revision.

`mesh.from_sections` creates a deterministic X-ordered loft from typed
`{x, width, height, z}` sections. The legacy `ellipse` and `superellipse`
profiles remain available; set `profile: "custom"` (or omit `profile`) and
provide a normalized closed `profile_points` polyline in local `[y, z]`
coordinates for an arbitrary section shape. Points are resampled by perimeter
to `segments` and may be overridden per section. A scalar `rotation`/`rotation_x`
is a roll in radians around the loft axis; `rotation_euler` accepts XYZ radians
for a full section-frame rotation. `center_offset` (or its `offset`/`center`
alias) accepts `[y, z]` or `[x, y, z]` and can be supplied globally or per
section; the existing `z` field remains the compact center-height field
(`center_z` is an explicit alias). Scalar `offset_x`, `offset_y`, and
`offset_z` aliases are also available when constructing payloads without a
vector field.
`cap_ends: true` (the default) produces a watertight carrier suitable for
secondary parts and verification. Custom points should normally stay within
the normalized `[-1, 1]` envelope; the runtime permits a bounded `[-4, 4]`
range for intentional overhangs.

`toolbox.capabilities` returns the coordinate contract, action layers, and
workflow profiles. `workflow.describe` returns one profile with phases and a
starter example when `include_examples: true` is requested. Both default to
compact metadata and are read-only; a profiled `session.open` can include the
same catalog in its first response.
