# Blender Toolbox

Blender Toolbox is a local, structured action interface for Blender. It is
designed for LLM agents that need detailed modeling control without exposing
arbitrary `bpy` execution. The canonical protocol is independent of MCP;
`blender_toolbox.mcp_server` is an optional transport adapter for MCP hosts.

The repository also includes `trajectory/`, a standalone environment-agnostic
trajectory and offline-RL data layer. It records one structured action per
step together with observations, state hashes, diffs, verifier data, rewards,
errors, artifacts, and replay metadata. Hidden chain-of-thought is never
stored.

## Install

Python 3.9 or newer is required. Blender 4.2 or newer is required for the
executor (the action protocol and trajectory package can be tested without
Blender).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

## Start Blender

Run the executor in a local background Blender process:

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background \
  --python blender_toolbox/addon.py -- \
  --socket /tmp/blender_toolbox.sock
```

The socket is local-only. On Windows, use a random-token localhost socket.
Set `BLENDER_TOOLBOX_SOCKET` to change the default address. The restricted
`run_python` escape hatch is disabled unless
`BLENDER_TOOLBOX_ALLOW_RUN_PYTHON=1` is explicitly set.

For the mixed workflow, enable the reviewed/declared Blender-Python asset
path explicitly:

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background \
  --python blender_toolbox/addon.py -- \
  --socket /tmp/blender_toolbox.sock --allow-bpy-apply
# equivalent environment switch: BLENDER_TOOLBOX_ALLOW_BPY_APPLY=1
```

## Call the toolbox directly

```python
from blender_toolbox.client import LocalIPCClient, ToolboxSession

session = ToolboxSession(
    LocalIPCClient("/tmp/blender_toolbox.sock"),
    "episodes/coffee-table/trajectory",
    task_id="CoffeeTable_seed0",
    seed=7,
)
session.start()
body = session.step("object.create", {
    "kind": "cube", "name": "body", "scale": [2, 1, 0.2],
})
session.step("verify.run", {"quality_profile": "production", "completion_gate": True,
                             "required_evidence_types": ["beauty", "clay", "silhouette", "closeup"],
                             "min_visual_score": 0.85})
session.step("artifact.export_glb", {"path": "episodes/coffee-table/final.glb",
                                      "require_quality": True, "require_completion": True}, done=True)
session.close()
```

`verify.run gate=true` is a structural gate, not a claim that the asset is
finished. A completion export requires current critical visual evidence. The
critical review records renderer file hashes, reviewer identity, confidence,
numeric visual scores, reference views, and anti-slop blockers; it rejects
stale renders and self-declared `passed=true` evidence. Use `audit_scope` and
`targets` to verify an assembly without technical cutters or reusable source
meshes.

Every step is revision-checked and idempotent. Topology-changing actions can
write immutable `.blend` checkpoints according to the session policy.

## Batch and mixed-mode workflows

Use `workflow.batch` when several ordinary Toolbox mutations form one
meaningful operation. The children are validated before execution, recorded in
the parent trajectory event, and advance the scene revision only once. By
default the batch is transactional: an action error, declaration mismatch, or
optional `verify_after` failure restores the pre-batch scene.

```python
session.batch(
    "创建底座并调整比例",
    [
        {"action": "object.create", "args": {
            "kind": "cube", "name": "Base", "scale": [2, 2, 0.25],
            "semantic_tags": ["base"],
        }},
        {"action": "object.transform", "args": {
            "target": "Base", "location": [0, 0, 0.25],
        }},
    ],
    creates=["Base"], modifies=["Base"],
    verify_after={"required_tags": ["base"]},
)
```

Use `bpy.apply` only for a bounded complex asset (for example a custom blade,
spiral rack, or a procedural housing) that is awkward to express as primitive
actions. Every script must state its `purpose`, declare `creates` and
`modifies` (and `deletes` when applicable), and should provide a
`source_sha256` for scripts outside the configured roots. Optional declarative
postconditions check object existence, absence, parenting, and semantic tags.
The executor validates a restricted AST, blocks files/network/process APIs,
records the source hash and timing, and rolls back on failure. Mixed Python
assets remain explicitly `trusted=false` and `replayable=false`; the trajectory
still retains the intent, declarations, source hash, observed delta, and
postcondition result for audit and review.

```python
session.apply_bpy(
    "生成带刀片的螺旋架",
    source_path="assets/spiral_blade.py",
    source_sha256="<sha256>",
    creates=["SpiralRack", "Blade_0", "Blade_1"],
    modifies=["MotorHub"],
    postconditions={
        "objects_exist": ["SpiralRack", "Blade_0", "Blade_1"],
        "parent_of": [{"child": "Blade_0", "parent": "SpiralRack"}],
    },
)
```

Keep the high-level sequence in Toolbox actions and descend to `bpy.apply`
only for the shape that needs it. Follow with `inspect.*`, `verify.run`, and
visual review at the appropriate quality stage; do not treat a successful
script execution as proof that spatial relationships or appearance are sound.

## Use from an agent through MCP

The MCP server exposes the same registry and forwards calls to one
`ToolboxSession`; it does not change the action schema or trajectory format:

```bash
python3 -m blender_toolbox.mcp_server \
  --socket /tmp/blender_toolbox.sock \
  --trajectory-dir episodes/coffee-table/trajectory \
  --task-id CoffeeTable_seed0
```

For Codex, register that stdio server once:

```bash
codex mcp add blender-toolbox -- \
  python3 -m blender_toolbox.mcp_server \
  --socket /tmp/blender_toolbox.sock \
  --trajectory-dir episodes/coffee-table/trajectory \
  --task-id CoffeeTable_seed0
```

The agent should load modeling knowledge first, then the verifier guidance,
then `skills/creative/blender-toolbox/SKILL.md`. The skill is an LLM-facing
policy: emit exactly one JSON action per turn, use semantic and topology tools
before local mesh/sculpt operations, inspect after meaningful changes, verify,
render, and export. All scene mutations should go through Toolbox. A separate
Blender MCP can remain read-only; mirror any call that must be retained with
the MCP-only `trajectory.record_external` tool.

The bundled skill describes the full registry, including UVs, material node
graphs, rigging, animation, hair, Geometry Nodes, camera/light setup,
landmark-driven facial curves, sculpt strokes, topology repair, checkpoints,
replay, and training-data restrictions.

## Trajectories and RL export

An episode directory contains:

```text
episode/
  manifest.json
  events.jsonl
  states/<state-hash>.json
  checkpoints/
  artifacts/
```

Export only completed, trusted, replayable events by default:

```bash
python3 scripts/export_trajectory_dataset.py \
  episodes/coffee-table/trajectory \
  -o data/trajectory_events.jsonl
```

Replay against a live executor and compare revision, state hash, topology and
verifier results:

```bash
python3 scripts/replay_toolbox.py \
  episodes/coffee-table/trajectory \
  --socket /tmp/blender_toolbox.sock
```

`trajectory` has no Blender dependency and can be reused by other simulators.
Its exporter excludes failed, truncated, untrusted, coordinate-dump and
non-replayable actions unless explicit diagnostic flags are provided.

## Package layout

- `blender_toolbox/`: protocol, 79-tool registry, Blender addon executor,
  local IPC client, optional MCP adapter, Gymnasium-shaped adapter, and
  Toolbox-side compatibility exports.
- `trajectory/`: generic crash-resilient JSONL writer, recorder, replay,
  reward, state hashing, and offline-RL filtering.
- `skills/creative/blender-toolbox/SKILL.md`: agent usage policy.
- `docs/agent_toolbox_workflow.md`: Codex, MCP, and benchmark integration.
- `scripts/`: trajectory export and replay helpers.
- `tests/`: protocol, executor-policy, trajectory, MCP, and adapter tests.

The legacy complete-script benchmark runner is intentionally not included in
this package. It can consume this package as a dependency and remains a
separate baseline in the benchmark repository.
