# Trajectory And Export

Each `ToolboxSession` writes an episode directory containing:

```text
trajectory/
  manifest.json
  events.jsonl
  states/<content-hash>.json
  checkpoints/
  artifacts/
```

An action event retains the canonical request, before/after state references,
state hashes, compact diff, executor result, verifier data, reward, and
replay metadata. Complete state snapshots stay in `states/`; compact MCP
responses are not a substitute for those snapshots.

Use the bundled helpers from the skill `scripts/` directory:

```bash
python3 scripts/export_trajectory_dataset.py \
  /path/to/episode/trajectory -o /path/to/training.jsonl

python3 scripts/replay_toolbox.py \
  /path/to/episode/trajectory --socket /tmp/blender_toolbox.sock
```

The exporter excludes incomplete, failed, untrusted, coordinate-dump, and
non-replayable events by default. `run_python` and `mesh.from_pydata` are
explicit exceptions and should only be included for diagnostics.

The final action sequence is `verify.run`, optional render and GLB export, then
`session.close`. A completed episode requires the verifier gate and a verified
final Blender checkpoint; a GLB or a render alone is not completion evidence.
