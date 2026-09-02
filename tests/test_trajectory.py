"""Regression tests for crash recovery and replay invariants."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from blender_toolbox.client import ToolboxClientError, ToolboxSession  # noqa: E402
from blender_toolbox.protocol import MAX_SEED  # noqa: E402
from trajectory.replay import replay_episode  # noqa: E402
from trajectory.state import state_hash  # noqa: E402
from trajectory.storage import TrajectoryReader, TrajectoryWriter, verified_final_checkpoint  # noqa: E402


def _observation(revision: int, state_hash: str) -> dict[str, object]:
    return {"revision": revision, "state_hash": state_hash, "summary": {"objects": []}}


def test_blender_completion_requires_an_intact_final_checkpoint(tmp_path: Path) -> None:
    writer = TrajectoryWriter(tmp_path, {"environment": "blender", "final_checkpoint_required": True})
    writer.finish(status="complete")
    assert writer.manifest["status"] == "error"

    checkpoint = tmp_path / "checkpoints" / "final.blend"
    checkpoint.write_bytes(b"deterministic-blend")
    digest = "sha256:" + hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    writer.update_manifest(
        status="running",
        final_checkpoint_ref="checkpoints/final.blend",
        final_artifact_hashes={"checkpoints/final.blend": digest},
    )
    writer.finish(status="complete")

    assert writer.manifest["status"] == "complete"
    assert verified_final_checkpoint(tmp_path, writer.manifest)


def test_recovery_discards_only_a_truncated_jsonl_tail(tmp_path: Path) -> None:
    writer = TrajectoryWriter(tmp_path)
    writer.append({"event_type": "diagnostic", "value": 1})
    with writer.events_path.open("ab") as handle:
        handle.write(b'{"event_type":"partial"')

    recovered = TrajectoryWriter(tmp_path)
    assert recovered.manifest["event_count"] == 1
    assert recovered.manifest["status"] == "truncated"
    assert recovered.events_path.read_text(encoding="utf-8").count("\n") == 1
    assert TrajectoryReader(tmp_path).events()[0]["value"] == 1


def test_append_rolls_back_event_when_manifest_commit_fails(tmp_path: Path, monkeypatch) -> None:
    writer = TrajectoryWriter(tmp_path)
    writer.append({"event_type": "diagnostic", "value": 1})
    original_events = writer.events_path.read_bytes()
    original_manifest = dict(writer.manifest)

    import trajectory.storage as storage

    real_dump = storage._json_dump

    def fail_manifest(path, value):
        if path == writer.manifest_path:
            raise OSError("manifest disk full")
        return real_dump(path, value)

    monkeypatch.setattr(storage, "_json_dump", fail_manifest)
    with pytest.raises(OSError, match="manifest disk full"):
        writer.append({"event_type": "diagnostic", "value": 2})
    assert writer.events_path.read_bytes() == original_events
    assert writer.manifest == original_manifest


def test_replay_and_dataset_reject_corrupt_event_suffix(tmp_path: Path) -> None:
    writer = TrajectoryWriter(tmp_path)
    writer.append({"event_type": "diagnostic", "value": 1})
    with writer.events_path.open("ab") as handle:
        handle.write(b'{"event_type":"broken"')
    reader = TrajectoryReader(tmp_path)
    assert len(reader.events()) == 1
    assert reader.events_intact is False
    report = replay_episode(tmp_path.as_posix(), lambda _action: {"ok": True})
    assert report.ok is False
    assert any(item["kind"] == "trajectory_events_corrupt" for item in report.mismatches)


def test_replay_reports_revision_and_state_hash_mismatches(tmp_path: Path) -> None:
    writer = TrajectoryWriter(tmp_path)
    before = _observation(0, "sha256:before")
    after = _observation(1, "sha256:after")
    writer.append_action(
        episode_id="episode",
        step_id=0,
        task_id="task",
        task_spec_hash=None,
        observation_before=before,
        action={"name": "object.create", "replayable": True},
        result={"ok": True},
        observation_after=after,
        verifier=None,
        reward={},
    )
    writer.finish(status="complete")

    report = replay_episode(
        str(tmp_path),
        lambda action: {"ok": True, "revision": 2, "state": {"state_hash": "sha256:wrong"}},
    )
    assert report.ok is False
    assert {item["kind"] for item in report.mismatches} == {"revision", "state_hash"}


def test_truncated_episodes_are_not_replayable(tmp_path: Path) -> None:
    writer = TrajectoryWriter(tmp_path)
    writer.finish(status="truncated")

    report = replay_episode(str(tmp_path), lambda action: {"ok": True})
    assert report.ok is False
    assert report.mismatches == [{"kind": "trajectory_truncated", "status": "truncated"}]


@pytest.mark.parametrize("seed", [-1, MAX_SEED + 1, True])
def test_session_rejects_invalid_seed_before_writing_manifest(tmp_path: Path, seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        ToolboxSession(object(), tmp_path / "invalid", task_id="seed", seed=seed)  # type: ignore[arg-type]


def test_failed_session_start_does_not_leave_started_flag_set(tmp_path: Path) -> None:
    class FailedClient:
        def action(self, **_request):
            summary = {"objects": []}
            return {
                "ok": False,
                "revision": 0,
                "error": {"code": "execution_error", "message": "open failed"},
                "state": {"revision": 0, "summary": summary, "state_hash": state_hash(summary)},
            }

    session = ToolboxSession(FailedClient(), tmp_path / "episode", task_id="failed-start")
    with pytest.raises(ToolboxClientError, match="open failed"):
        session.start()
    assert session._started is False


def test_session_records_canonical_recipe_and_stage_boundary_metadata(tmp_path: Path) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.revision = 0

        def action(self, **request):
            action = request["action"]
            args = request.get("args") or {}
            if action == "artifact.save_checkpoint":
                path = Path(str(args["path"]))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"checkpoint")
                artifacts = [{"path": str(path), "kind": "blend"}]
            else:
                artifacts = []
            summary = {"objects": []}
            state = {"revision": self.revision, "summary": summary, "state_hash": state_hash(summary)}
            return {"ok": True, "revision": self.revision, "state": state, "artifacts": artifacts}

    session = ToolboxSession(FakeClient(), tmp_path / "episode", task_id="canonical", checkpoint_policy="stage")
    session.start()
    session.step(
        "material.apply_recipe",
        {
            "name": "Material",
            "stage_boundary": True,
            "recipe": {
                "kind": "material",
                "nodes": [
                    {"id": "z", "type": "ShaderNodeOutputMaterial"},
                    {"id": "a", "type": "ShaderNodeBsdfPrincipled"},
                ],
            },
        },
    )
    session.recorder.finish(status="aborted")

    actions = [event for event in TrajectoryReader(tmp_path / "episode").events() if event["event_type"] == "action"]
    recipe_action = actions[1]["action"]
    assert [node["id"] for node in recipe_action["args"]["recipe"]["nodes"]] == ["a", "z"]
    assert "stage_boundary" not in recipe_action["args"]
    assert recipe_action["stage_boundary"] is True
    assert actions[1]["checkpoint_ref"] == "checkpoints/step-000001.blend"
