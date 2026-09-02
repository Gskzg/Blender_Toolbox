---
name: blender-toolbox
description: "Build, sculpt, inspect, verify, replay, and export Blender scenes through one canonical structured Toolbox MCP runtime. Use when Codex must perform deterministic Blender modeling through declared actions, combine procedural geometry and attributes with advanced sculpt workflows, capture trajectories and checkpoints, or run the 3DCodeBench Blender harness."
---

# Blender Toolbox

Use this skill as the single execution contract for a structured Blender
episode. It provides one canonical runtime under `scripts`: 123 public
Toolbox actions, plus the MCP-only `trajectory.record_external` meta-tool
(124 tools through the stdio MCP adapter).
Keep the bundled `blender_toolbox` and `trajectory` packages together and use
the supplied launchers so the harness resolves this skill as its only Toolbox
implementation.

## Fast path

For a new modeling task, establish the quality contract before the first
geometry mutation. The goal is a strong first pass, not a late discovery that
the chosen representation cannot carry the intended form.

1. Call `session.open` with `mode: "new"`, `reset: true`,
   `quality_profile: "quality_first"`, and a domain profile when one applies.
   Include a `task_spec.quality` contract with identity/scale, primary and
   secondary carrier refs, representation kind, four evidence views, named
   detail regions, feature scales, and technical policy. A high-resolution
   carrier is the default; a low-resolution exception must include a written
   `resolution_exception.exception_reason`. Set
   `include_capabilities: true` to receive the domain-neutral
   quality bar, selected workflow, coordinate contract, and an optional plan
   template in the same response. `model.plan` is a compact read-only
   decision tree when the carrier is not obvious; `ToolboxSession.bootstrap()`
   is the equivalent one-call client helper.
2. Choose the representation before authoring geometry. Use a continuous
   carrier (`section_stack`, control mesh, curve, SDF, or deformation) for an
   identity-defining envelope; reserve primitives for genuinely derived
   repeated parts or intentionally primitive objects. Record `role`,
   `representation`, and `quality_stage` metadata when those distinctions
   matter.
3. Build repeated secondary parts with `object.create_batch`, apply transforms
   with `object.transform_batch`, assign shared materials with
   `material.assign_batch`, and add/apply ordered modifiers with
   `geometry.modifier_stack`. Each batch is one deterministic trajectory event
   with atomic rollback enabled by default.
4. For a continuous envelope (manufactured, organic, architectural, or
   environmental), use a structured carrier such as `mesh.from_sections`, a
   control mesh, a deformation cage, or an SDF recipe before adding secondary
   parts. The carrier should expose the controls needed to change the
   silhouette, depth, junctions, and negative spaces together; do not build an
   identity-defining form from a long list of unrelated primitive edits.
5. Use `inspect.batch` for one scene census. Its default `detail: "compact"`
   response contains identity, transforms, bounds, mesh counts, materials,
   semantic tags, and collections. Request `detail: "full"` or an explicit
   `fields` list only when a dependent edit needs hashes, UVs, node graphs, or
   other expensive diagnostics.

The quality contract is domain-neutral and production-first. Its non-negotiable
gates cover structure, primary carrier, secondary junctions, tertiary detail,
technical readiness, and four-view evidence. High-resolution density, feature
sampling, connected carrier surfaces, materials, UVs, and strict topology are
enforced by default. A
low-resolution exception is accepted only when explicitly documented with a
reason. An explicitly required stage that has no evidence fails. `verify.run` reports
`quality.stages`, `first_failure`, and a concrete `repair_action`; fix the
earliest failed layer before adding detail.
`artifact.export_glb` and `session.close` require a passing `verify.run` at the
current revision whenever the active contract is enforced. Production,
organic, and strict profiles additionally enable the completion gate: a
critical visual review, current render hashes, required evidence types, and
minimum visual scores are mandatory.

For reusable Geometry Nodes or material graphs, use the typed
`geometry_nodes.apply_recipe` or `material.apply_recipe` action. Recipes are
JSON-only, bounded, hashable, and checked against the Toolbox node allowlist
before Blender realization; they are not arbitrary Python programs.

Stable object references (`id`/`ref`) are preferred over Blender display names.
The batch create action preflights duplicate references before mutating the
scene, and all batch actions return per-item results plus commit/rollback
status. This keeps the modeling loop short without hiding individual changes
from replay or verification.

The executor caches the last authoritative full census by revision. Routine
non-mutating actions reuse that snapshot; explicit `inspect.*` calls still
perform the requested compact or full diagnostic pass. This keeps trajectory
state hashes lossless while avoiding duplicate scans during setup.

## Choose a representation

Classify each important region before building it:

- **Derived**: repeated or rule-driven form. Use curves, modifiers, Geometry
  Nodes, deterministic distributions, or parameterized transforms.
- **Mixed**: a generated structure with supplied landmarks. Build the shared
  envelope first, then use named landmarks, curves, masks, or local sculpting.
- **Specified**: a unique continuous shape. Use a control mesh, cross-section
  stack, deformation cage, multires mesh, or landmark-driven surface instead of
  unrelated primitives.

  A representation is adequate only when its controls can change silhouette,
  depth, semantic regions, junctions, openings, and local landmarks without
  unrelated coordinate edits. If improving fidelity requires adding coordinates
  one by one, change the representation before adding more polygons.

  Prefer native structured generators first, structured direct meshes second,
  SDF fields when smooth unions or robust CSG are central, and deformation
  representations when many landmarks must move together. For sampled fields,
  resolve the narrowest important feature with roughly four or more samples.
  Resolution cannot restore a structural degree of freedom that the
  representation does not have.

  Use this quality ladder and do not advance while an earlier level fails:

1. **Structure**: axes, proportions, semantic parts, contacts, openings,
   landmarks, and intended topology.
2. **Primary**: envelope, major masses, silhouette, depth profile, and negative
   spaces.
3. **Secondary**: junctions, rims, plane changes, attachments, and transitions.
4. **Tertiary**: only details that the current carrier can resolve.
5. **Verify**: neutral multi-view geometry, topology, measurements, and semantic
   requirements.

## Action contract

Emit exactly one JSON action per turn, for example:

```json
{"action":"object.create","args":{"kind":"cube","name":"body"},"done":false}
```

Use only action names and argument shapes exposed by the Toolbox registry.
Never emit Blender Python, an unbounded `bpy` expression, Markdown in place of
an action, or hidden reasoning. The executor validates arguments, serializes
actions, checks revisions, and returns the authoritative result.

Keep the UUID returned by each creation action. Do not guess Blender names or
reuse stale numeric indices after a topology change. Selections are evaluated
in object-local coordinates and should use a bounded spatial selection or a
named vertex group.

## Structured modeling loop

1. Open or resume the session with `session.open`; it can reset the scene and
   declare the profile/quality contract in the same action. Inspect the initial
   scene only when the resume path requires it.
2. Run `model.plan` when the carrier is not obvious, then lock the identity,
   scale, primary refs, negative spaces, detail scales, and evidence views
   before the first primary mutation.
3. Build semantic primary structure with a declared carrier, then use
   `object.create_batch` only for derived or secondary parts. Give objects
   stable IDs and semantic tags.
4. Inspect after meaningful mutations. Prefer one `inspect.batch` census, then
   use `inspect.topology`,
   `inspect.measure`, `inspect.mesh_region`, and the relevant domain inspection
   before making a dependent edit.
5. Resolve secondary and tertiary structure with explicit masks and bounded
   operations. Repair the earliest failing level when an inspection disagrees.
6. Run `verify.run`, render a small set of stable inspection views, immediately
   inspect every image, and submit `evidence.visual_review`; fix any failed
   gate before export.
7. Export GLB only after verification passes at the same revision. Make
   `session.close` the final Toolbox action and require the runtime to report a
   verified final checkpoint.

   Every mutation is recorded with before/after observations, state hashes and
   diffs, the result, verifier data, reward, and a checkpoint when policy
   requires it. The trajectory is the source for replay and offline-RL export;
   unlogged side effects do not count as part of the model.

## Advanced authoring

For sculpt-heavy work, explicit topology edits, UV/material nodes, rigging,
landmarks, or animation, read [authoring.md](references/authoring.md). It
contains the stage-gated sculpt protocol and operation-layer details while
keeping the common quality-first entry path short.

## Verification contract

Verification is a structural and representation contract, not a beauty score.
For a quality-first episode, declare it before the first primary pass and
inspect the returned diagnostics after each meaningful stage.

- `required_tags` are cumulative for a session. A later action may add a tag but
  may not remove or weaken a task-declared requirement. Tags on MESH and CURVE
  objects are valid semantic evidence.
- For multiple parts, declare `assembly.parts` and `contacts` with an allowed
  gap. AABB overlap is only a precheck; sampled surface or volume proximity is
  required to prove contact and connected components.
- For an opening, use a real boolean or topological hole and rim. A closed
  cylinder with a dark material is not an opening.
- Add `proportions`, `silhouette_views`, and `feature_sizes` whenever the task
  has measurable dimensions, multiple views, or small details. Use a
  `detail_regions` selection with density, relief, edge-length, and closure
  constraints when local relief matters.
- Add `quality` with `enforce: true`, `representation`, `primary_refs`,
  `secondary_refs`, four `reference_views`, `feature_scales`, named
  `detail_regions`, and the applicable `technical` policy. The production
  quality score treats missing required declarations as failures; inspect
  `quality.first_failure` and follow its `repair_action`.
- Require the topology appropriate to the task. Check non-finite vertices,
  invalid indices, zero-area faces, boundaries, non-manifold edges, shells,
  genus, and intended open or closed surfaces.

  When verification fails, classify the first failure as missing/disconnected
  structure, silhouette/proportion, alignment, or local detail. Map it to one
  bounded repair batch and re-inspect; never hide a structural failure with a
  material, smooth shading, subdivision, or a beauty render.

## Visual feedback

Render views are evidence for a specific question. Compare matching views by
the returned filename and camera azimuth/elevation metadata; never assume that
the first image is the front view. Use the same cameras across iterations and
keep at least one view held out when a reference fit is being evaluated. A
single attractive three-quarter render does not prove depth, profile, or
connectivity.
If image evidence is part of the task contract, set
`quality.evidence.require_render: true`; the runtime records rendered view
names and revision, and `verify.run` rejects missing or stale renders. Leave it
off for geometry-only checks so rendering remains an explicit, low-frequency
inspection action.

## MCP, trajectory, and recovery

MCP is a transport adapter, not a second modeling authority. Route every scene
mutation through the directly declared `blender_toolbox` server so revisions,
idempotency, checkpoints, rewards, and replay remain coherent. A separate
Blender MCP may be read-only; immediately mirror required calls with the
`trajectory.record_external` meta-tool and mark them diagnostic.

Honor `expected_revision` and idempotency keys. Treat timeouts, revision
conflicts, Blender crashes, and invalid actions as observable failures. Inspect
or reset before continuing, and keep the failed event in the trajectory.

`run_python` and `mesh.from_pydata` are escape hatches: they are untrusted or
non-replayable and excluded from high-quality RL export by default. Prefer
structured actions, typed fields, named masks, and deterministic parameters.

## Mandatory visual checkpoint lifecycle

`render.views` is a checkpoint for the current scene revision. The executor
records rendered view names, camera targets, evidence types, file hashes, and
the scene-content hash. A new render supersedes the previous review. Until
`evidence.visual_review` succeeds for every rendered view, the next scene
mutation and another render are rejected; visual review is therefore an
immediate feedback loop instead of a late final pass. Reviews must use the
current revision and stage and cannot reference unrendered views.

Critical reviews require reviewer identity, confidence >= 0.8, six numeric
scores, every anti-slop boolean check, explicit blockers, reference views, and
hashes for every rendered file. Concrete objective blockers, failed checks,
stale hashes, missing views/evidence types, and scores below the configured
minimum fail the gate. Unknown dimensions remain visible in diagnostics and
must be resolved by the checklist; they are never silently treated as a pass.

For deployments that expose artifact export over a shared service, set
`BLENDER_TOOLBOX_ARTIFACT_ROOTS` to one or more colon-separated trusted
directories; resolved `.blend`/`.glb` destinations are then rejected when they
escape those roots (including via symlinks).

Before claiming completion, inspect the final scene, run `verify.run`, export
only after its gates pass, and close the session. The final session action must
produce a verified `final.blend` checkpoint; a GLB or a successful-looking
render alone is not completion evidence.

For runtime launch, action registry details, and trajectory/export commands,
read only the relevant bundled references:

- [runtime.md](references/runtime.md)
- [tool-registry.md](references/tool-registry.md)
- [trajectory.md](references/trajectory.md)

The runtime defaults to `topology_terminal` checkpoints. For long episodes,
`every_n` (configured with `checkpoint_interval`) bounds recovery distance,
while `stage` saves only successful mutating actions carrying the optional
`stage_boundary: true` hint. Checkpoint policy changes cadence only; it never
changes the action or trajectory contract.
