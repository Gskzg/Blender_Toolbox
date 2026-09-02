#!/usr/bin/env python3
"""Exercise the sandbox trajectory path without starting Blender or Codex."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blender_toolbox.client import ToolboxSession
from trajectory.dataset import export_training_events
from trajectory.state import state_hash
from trajectory.storage import TrajectoryReader


class FakeExecutor:
    """Small deterministic response double for protocol/trajectory smoke tests."""

    def __init__(self) -> None:
        self.revision = 0
        self.objects: list[dict[str, object]] = []

    def _state(self) -> dict[str, object]:
        summary = {"objects": list(self.objects)}
        return {
            "revision": self.revision,
            "summary": summary,
            "state_hash": state_hash(summary),
            "diff": {},
        }

    def action(self, **request: object) -> dict[str, object]:
        name = str(request["action"])
        if name in {"session.create", "session.reset"}:
            self.revision = 0
            self.objects = []
        elif name == "object.create":
            args = request.get("args") or {}
            self.objects.append({"uuid": f"obj-{len(self.objects) + 1}", "kind": args.get("kind", "cube")})
            self.revision += 1
        elif name == "artifact.save_checkpoint":
            path = Path(str((request.get("args") or {}).get("path")))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake-blend-checkpoint")
            return {"ok": True, "revision": self.revision, "state": self._state(), "artifacts": [{"path": str(path)}]}
        elif name == "verify.run":
            return {
                "ok": True,
                "revision": self.revision,
                "state": self._state(),
                "metrics": {"scorecard": {"gate": True, "metric": {"score": 1.0}}},
                "result": {"gate": True, "metric": {"score": 1.0}},
            }
        elif name == "session.close":
            return {"ok": True, "revision": self.revision, "state": self._state()}
        else:
            return {"ok": True, "revision": self.revision, "state": self._state()}
        return {"ok": True, "revision": self.revision, "state": self._state(), "metrics": {}, "artifacts": []}


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="blender-toolbox-smoke-") as tmp:
        root = Path(tmp) / "episode"
        session = ToolboxSession(FakeExecutor(), root, task_id="smoke", seed=7)
        session.start()
        session.step("object.create", {"kind": "cube", "name": "body"})
        session.step("verify.run", {})
        session.close()
        reader = TrajectoryReader(root)
        exported = Path(tmp) / "training.jsonl"
        count = export_training_events([root], exported)
        manifest = reader.manifest
        events = reader.events()
        assert manifest["status"] == "complete", manifest
        assert len(events) == 4, events
        assert count == 4, count
        assert (root / "checkpoints" / "step-000001.blend").is_file()
        assert (root / "checkpoints" / "final.blend").is_file()
        print(json.dumps({"status": manifest["status"], "events": len(events), "exported": count}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
