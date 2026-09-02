# Advanced Authoring

Read this reference for sculpt-heavy work, explicit topology edits, or
domain-specific material/UV/rigging operations. Simple assets can stay at the
semantic and primary levels in `SKILL.md`; the same quality contract applies to
manufactured, organic, architectural, and environmental assets.

## Stage-Gated Sculpting

## Quality-First Planning (All Domains)

Use this section for any asset whose first-pass silhouette, construction, or
detail fidelity matters; it is not specific to sculpting or to a vehicle.

1. Register a `quality_contract` at `session.open`. Keep identity, scale,
   primary and secondary carrier refs, required semantic parts, four evidence
   views, named detail regions, feature scales, and technical policy in the
   contract. The runtime freezes the contract for the episode
   and exposes its hash in authoritative state.
2. Call `model.plan` when the representation is uncertain. Treat its decision
   tree as a routing aid: continuous identity forms need a continuous carrier;
   repeated forms need a derived/instanced carrier; openings need real
   thickness and topology; local relief needs a resolved surface carrier.
3. Label authored objects with `role`, `representation`, and
   `quality_stage`. This lets `inspect.quality` distinguish a legitimate
   primitive detail from a primitive pile standing in for the primary form.
   Joined carriers must be one connected shell by default; use
   `technical.expected_shells` only for an intentional multi-shell design.
4. Run `inspect.quality` after each meaningful stage. It is diagnostic and
   bounded: topology defects, unapplied transforms, missing materials/UVs,
   feature undersampling, contacts, and a conservative negative-space proxy
   are reported together. It does not replace `verify.run`.
5. Run `verify.run` after all six production stages are authored. Read
   `quality.first_failure` and apply its `repair_action` before progressing.
   Quality-first keeps structure, primary, secondary, tertiary, technical, and
   evidence stages in the required-stage contract.

Set `quality.evidence.require_render: true` and declare four view names in
`reference_views` for quality-first work. The runtime records `render.views`
provenance (revision, names, camera targets, evidence types, scene hash, and
files), rejects missing or stale renders during the next `verify.run`, and
blocks subsequent scene mutations until `evidence.visual_review` covers every
rendered view.

`quality_profile: "advisory"` is available for legacy inspection or partial
assets. It does not unlock a quality-first export gate. New authored episodes
should leave the default `quality_first` profile enabled.

Treat a complex sculpt as measurable passes rather than one long brush list.
The same protocol applies to faces, creatures, reliefs, hard-surface
ornament, and any continuous form whose identity depends on local planes.

### Contract And Preparation

For a reference-driven or high-risk sculpt, record object-local axes, intended
symmetry axes, primary bounding-box ratios, named landmarks or supports, and
four fixed views: front, three-quarter, profile, and rear. Do not create those
views for a simple blockout unless they answer a real comparison question.

Call `sculpt.surface_prepare` after blockout. Apply scale and rotation, remesh
before adding Multires, and choose `voxel_size` from the smallest primary
feature that must survive. Add only as many Multires levels as the checked
proportions require. If the final mesh must be inspectable or exported without
a modifier, materialize the selected level with an explicit vertex budget.
Materialization invalidates base-topology vertex groups; recreate masks after
the bake and before masked strokes.

Immediately call `inspect.sculpt_quality` with the intended `symmetry_axes`, a
bounded `sample_limit`, and `include_landmarks: true`. A sparse or non-manifold
base, missing landmarks, or an untouched ellipsoid warning blocks tertiary
detail until repaired.

### Passes

Use one `sculpt.stroke_batch` per coherent pass and label every stroke with
`stage`: `primary`, `secondary`, `tertiary`, or `cleanup`.

- **Primary**: establish the non-elliptical envelope, major masses, silhouette,
  profile, and negative spaces. Prefer `sculpt.region_deform_batch` for broad
  organic changes. Its smooth ellipsoidal cages can translate, scale, rotate,
  and apply a bounded signed `normal_offset` without creating helper geometry.
- **Secondary**: resolve planes, sockets, transitions, and attachments. Project
  authored landmarks with `landmark.project_to_surface`. Use bilateral
  symmetry only for genuinely bilateral regions; use a local selection for
  intentional asymmetry. For a bounded plane or shallow relief on the same
  carrier, use `sculpt.surface_patch_batch` with tangent axes, in-plane radii,
  optional `depth_limit`, `normal_blend`, and a semantic label.
- **Tertiary**: add restrained relief with short narrow strokes or bounded
  surface patches. Use `crease` for recesses and `draw`/`inflate` for rims and
  folds. Keep `pressure`, falloff, visibility, depth, and vertex-group masks
  explicit. Materialize Multires before deterministic mesh strokes unless an
  intentional coarse edit sets `allow_multires_base: true`.
- **Cleanup**: use low-strength `smooth` or `relax` only on transition masks,
  recalculate normals, repair holes or non-manifold edges, and inspect the same
  landmarks again. Cleanup must not erase a required crease or flatten the
  primary silhouette.

After each pass, answer: did the intended region change, did protected regions
stay within the mask, and does the shape still agree in the checked views? Read
affected counts, displacement, AABBs, topology, and landmark reports before
starting the next pass.

## Operation Layers

Keep the modeling idea at the highest useful layer and descend only when the
shape requires it.

1. **Semantic structure**: `object.create`, `curve.create`,
   `object.duplicate`, `object.join`, transforms, booleans, modifiers, and
   materials.
2. **Topology**: `mesh.from_sections`, `mesh.subdivide`, `mesh.extrude_region`,
   `mesh.duplicate_region`, `mesh.extrude_individual`, `mesh.inset_individual`,
   `mesh.bridge_edge_loops`, `mesh.loop_cut`, `mesh.inset_region`, `mesh.bevel`, `mesh.transform_selection`,
   `mesh.merge_by_distance`, `mesh.vertex_group_assign`,
   `mesh.recalculate_normals`, `mesh.delete_region`, `mesh.dissolve_region`,
   `mesh.fill_holes`, and `mesh.triangulate`. Use `mesh.region_define` and
   returned `region_handle` values for reusable selections; use attribute
   read/write, geometry queries, cuts, repair, separation, adaptive
   subdivision, or symmetrization when the topology workflow requires them.
   Region operations require a non-empty explicit selection.

   For a continuous carrier, `mesh.from_sections` can loft more than an
   ellipse: provide a normalized, ordered closed `profile_points` loop in the
   local Y/Z plane and let the runtime resample it to the requested ring
   resolution. Profiles can be overridden per section. Use section-level
   `rotation`/`rotation_euler` and `center_offset` (or scalar `offset_x/y/z`)
   to carry intentional twist,
   lean, or asymmetric placement through the loft; these controls are generic
   and should describe the asset's structure rather than encode a vehicle-only
recipe. Keep the baseline profile within `[-1, 1]` unless a bounded
   overhang in `[-4, 4]` is deliberate.
3. **Surface detail**: `geometry.remesh_voxel`, `sculpt.multires`,
   `sculpt.materialize_multires`, `sculpt.stroke`, `sculpt.stroke_batch`,
   `sculpt.region_deform_batch`, and `sculpt.surface_patch_batch`. Use
   `sculpt.ridge`, `sculpt.groove`, and `sculpt.muscle` for tapered procedural
   relief paths.

Use `mesh.vertex_group_assign` for stable supports such as `jaw_mask`,
`eye_socket_mask`, or `mouth_ring`. Do not send an unbounded coordinate dump
through MCP; use declared selections and inspection actions instead.

`geometry.add_modifier` resolves object references from Toolbox UUIDs. For
typed, reusable Geometry Nodes graphs, prefer `geometry_nodes.apply_recipe`:
its recipe IR canonicalizes node/link order, rejects unknown references and
cycles, bounds graph size, preserves explicit node attributes and multi-input
link order, and returns a stable recipe hash. The Blender executor still
applies its node/attribute allowlists and socket checks, so unknown properties,
nodes, links, or sockets should be corrected at the action boundary rather
than worked around with arbitrary Python.

For mode-sensitive nodes, pass constant selectors under `attributes`, for
example `{"type":"ShaderNodeMath","attributes":{"operation":"MULTIPLY"}}`.
For multi-input sockets, add an integer `order` to each link; the adapter
accounts for Blender's reverse insertion behavior. Interface `index` values
are likewise available when declaration order is semantically significant.

## Domain Notes

- **UV**: mark semantic seams with `uv.mark_seams`/`uv.clear_seams`, then use
  `uv.unwrap`, `uv.project`, and `uv.pack`. Inspect bounds and coverage.
- **Materials**: create and assign a base material, then use the allowlisted
  graph through `material.node_graph` or the hashable `material.apply_recipe`,
  and single-input changes through `material.set_input`. Inspect the graph hash.
- **Hair and scatter**: use deterministic `hair.create_strands` polylines with
  optional radii, or seeded `particles.scatter` on a declared target. Curves
  count as semantic evidence; convert to mesh only when downstream topology or
  export requires it.
- **Rigging and animation**: create explicit bones, bind, add bounded
  constraints, then keyframe transforms or landmark-driven shape keys. Inspect
  armature and animation state.
- **Landmarks and faces**: create named landmarks or a set, inspect them,
  project approximate positions to the surface, and use semantic curve
  networks or localized sculpt strokes. Preserve names in action arguments.
- **Camera and render**: use structured camera/light/render actions. Render
  `render.views` only at stage boundaries or for a targeted repair, with stable
  locations and framing.
