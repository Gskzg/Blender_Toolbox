# Bundled Runtime

This skill carries the complete canonical Toolbox and trajectory runtime in
`scripts/`. The two Python packages must remain siblings:

```text
scripts/
  blender_toolbox/
  trajectory/
  start_blender_toolbox.sh
  start_mcp_server.sh
  replay_toolbox.py
  export_trajectory_dataset.py
  smoke_fake.py
  smoke_batch.py
  smoke_quality.py
  smoke_quality_generic.py
  smoke_evidence.py
  smoke_recipe.py
  smoke_sections.py
  smoke_workflow.py
```

The Blender executor is `scripts/blender_toolbox/addon.py`. Start it from a
launcher or terminal with a local socket. From the skill root, the bundled
launcher is the shortest path:

```bash
BLENDER_TOOLBOX_SOCKET=/tmp/blender_toolbox.sock \
  scripts/start_blender_toolbox.sh
```

The equivalent direct command is:

```bash
BLENDER_BIN=/Applications/Blender.app/Contents/MacOS/Blender
"$BLENDER_BIN" --background --factory-startup \
  --python /path/to/blender-toolbox/scripts/blender_toolbox/addon.py -- \
  --socket /tmp/blender_toolbox.sock
```

The stdio MCP adapter uses the same runtime directory as its working
directory, so `blender_toolbox` and `trajectory` resolve without installation:

```bash
BLENDER_TOOLBOX_SOCKET=/tmp/blender_toolbox.sock \
  scripts/start_mcp_server.sh /path/to/episode/trajectory Example_seed0
```

The equivalent direct command is:

```bash
cd /path/to/blender-toolbox/scripts
python3 -m blender_toolbox.mcp_server \
  --socket /tmp/blender_toolbox.sock \
  --trajectory-dir /path/to/episode/trajectory \
  --task-id Example_seed0
```

The socket accepts Unix paths and localhost TCP addresses only. The executor
uses `BLENDER_TOOLBOX_SOCKET` and `BLENDER_TOOLBOX_AUTH_TOKEN` when launched
as an installed addon. `run_python` stays disabled unless
`BLENDER_TOOLBOX_ALLOW_RUN_PYTHON=1` is explicitly set.

The bundled `protocol.py` owns the complete static registry and the bundled
`addon.py` owns the complete executor. There are no alternate Toolbox runtime
modules in this skill. The registry contains 123 Toolbox actions. The MCP
adapter adds `trajectory.record_external`, so a healthy `tools/list` response
contains 124 tools. `quality_first` is the generic base workflow and
`general` is its domain-neutral profile; manufactured, organic, architectural,
and environmental profiles may add routing hints, but none may replace or
weaken the frozen quality contract.

At `session.open`, a new/reset episode freezes the quality-first contract
before geometry is authored. The production baseline is structure, primary,
secondary, tertiary, technical, and evidence. High-resolution density, feature
sampling, and single-shell connectivity for declared carriers are enforced
unless a documented exception is present. Intentional multi-shell carriers
must declare `technical.expected_shells`. `model.plan` and the `quality_plan` returned by
`session.open(include_capabilities=true)` expose the same stage list and
contract template, so a client can choose a suitable carrier without a
vehicle-specific preflight. Quality-first keeps all six stages required;
advisory inspection may still report undeclared stages as `unknown`.

Image evidence is required for quality-first via `quality.evidence.require_render`.
`render.views` stores view names, camera targets, evidence types, file hashes,
and the scene revision as runtime provenance; `verify.run` rejects missing,
mismatched, or stale renders when the contract requests visual evidence.
After a render, the executor blocks the next scene mutation until
`evidence.visual_review` covers the current views. Render provenance is kept
outside the geometry state hash, so a non-mutating render does not invalidate a
previously verified geometry state.

Use the bundle as one unit. Do not mix it with the root, sandbox, or an
installed Toolbox package. For direct 3DCodeBench runs, configure exactly this
skill path:

```yaml
toolbox_skill_paths:
  - skills/creative/blender-toolbox
```

The repository's `configs/toolbox.yaml` already selects it, and
`core/toolbox_runner.py` puts its `scripts/` directory first for Toolbox mode.
For another harness, set both MCP `cwd` and the leading `PYTHONPATH` entry to
this skill's absolute `scripts/` directory. Confirm the loaded module path and
the 123/124 registry counts before a benchmark run.

`ToolboxSession` supports five checkpoint policies: `topology_terminal` (the
default), `every_action`, `every_n`, `stage`, and `none`. Set `checkpoint_interval`
with `every_n`; for `stage`, mark a successful mutating action with
`stage_boundary: true`. The stage policy is evaluated from the action's
declared mutability, so it also works with task-facing batch actions. For
interactive authoring, prefer `stage` or `every_n` to keep checkpoint I/O out
of every small edit; retain `topology_terminal` when terminal recovery is the
primary requirement.
