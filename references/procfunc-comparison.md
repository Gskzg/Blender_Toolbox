# ProcFunc Reference Audit

`procfunc-main` is a design reference only. The Toolbox runtime must remain
usable when that directory is removed; no Toolbox package imports ProcFunc and
the package discovery rules intentionally exclude it.

`procfunc-main/LICENSE.md` is BSD-3-Clause (Princeton University, 2026). No
ProcFunc source or manifest is embedded in Toolbox; if code/data is copied in a
future offline tool, retain the license notice and attribution separately.

## Surface Comparison

| Area | ProcFunc | Toolbox | Decision |
| --- | --- | --- | --- |
| Execution model | Python functions are traced into a compute graph and later realized by Blender | Validated JSON actions are executed by one Blender adapter | Keep Toolbox execution as the authority |
| Procedural graphs | Typed node bindings and a code generator for geometry, shader, compositor, and texture graphs | 123 declared actions, with allowlisted Geometry Nodes/material graph actions | Add a bounded JSON recipe IR; do not expose arbitrary Python graphs |
| Blender coverage | `nodes/manifest.json` contains 630 node records (625 named records / about 483 unique `pf.nodes.*` names) across geometry, shader, compositor, and texture domains | Geometry/material recipes use explicit, Blender-5.1-smoke-tested allowlists | Expand allowlists only with target-version smoke coverage and a compatibility policy |
| Operators | Large operator metadata manifest; only the named `pf.ops` surface is executable | Structured actions wrap the high-value operations and preserve stable refs | Do not copy the manifest wholesale |
| Randomness | Trace-aware random/distribution primitives | Seeded `parameters.sample` and seeded action arguments | Keep seed in the protocol and trajectory fingerprint |
| State and recovery | Primarily a graph/tracing library | Revision conflicts, idempotency, auth, transactions, checkpoints, replay, rewards, and verification | These Toolbox guarantees take precedence |

## Adopted

- `geometry_nodes.apply_recipe` and `material.apply_recipe` accept a bounded,
  JSON-only typed graph. Recipes are canonicalized at the protocol boundary,
  checked for finite values, duplicate references, cycles, size limits, and
  node allowlists, and carry a stable SHA-256 recipe hash. Node attributes
  (`operation`, `data_type`, `mode`, etc.) are now explicitly allowlisted and
  applied in dependency order; optional interface indices and multi-input link
  order preserve graph semantics instead of silently falling back to Blender
  defaults.
- Blender graph realization is staged. A malformed node, socket, or link leaves
  the previous live graph/material intact.
- Seeded parameter distributions are deterministic and return their sampled
  values plus a content hash, making them usable in a replayable authoring
  trajectory.
- A small set of high-value typed graph nodes inspired by ProcFunc's node
  metadata is supported (`FunctionNodeInput*`, `FunctionNodeCompare`, curve
  sampling/fillet/arc, and common shader color/vector nodes). The list is
  probed against the target Blender build; the ProcFunc manifest is not copied
  as a compatibility promise.
- Action schemas and MCP discovery are generated from the same registry. The
  canonical registry has 123 actions; stdio MCP adds only the diagnostic
  `trajectory.record_external` meta-tool (124 tools total).
- Reusing an idempotency key with a different canonical request is rejected as
  `idempotency_conflict` instead of silently replaying the wrong result.

## Deliberately Deferred

- The complete ProcFunc Python tracing/transpilation stack is not copied. It is
  tightly coupled to a particular `bpy` version and would introduce a second
  execution authority, untrusted code paths, and another replay contract.
- Compositor, texture, and world graph actions are not inferred from the
  manifest. They need explicit Toolbox schemas, allowlists, Blender-version
  smoke tests, and verifier coverage before becoming public actions.
- ProcFunc's full node/operator manifest is not exposed through MCP. Public
  surface area is added only when the action has stable references, bounded
  inputs, deterministic behavior, and transactional failure handling.
- Per-node socket/data-type schemas, contextual node remapping, and graph
  export/diff remain a read-only/offline follow-up. Runtime recipes continue
  to require explicit allowlisted node types and Blender socket validation.

When the reference is no longer needed, remove only `procfunc-main/` after a
final `rg` check confirms there are no runtime imports or packaging references.
