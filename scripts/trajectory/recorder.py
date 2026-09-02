"""Generic episode recorder middleware.

It accepts already-produced observations and executor responses.  The recorder
does not know how an action is executed, which makes it reusable for Blender,
CAD, browser and simulator environments.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .reward import compute_reward
from .storage import TrajectoryWriter


class EpisodeRecorder:
    def __init__(self, root: str | Path, manifest: Optional[Mapping[str, Any]] = None, *, writer: Optional[TrajectoryWriter] = None) -> None:
        self.writer = writer or TrajectoryWriter(root, manifest)
        self.previous_scorecard: Optional[Mapping[str, Any]] = None

    def update_manifest(self, **fields: Any) -> None:
        self.writer.update_manifest(**fields)

    def record_event(self, event: Mapping[str, Any]) -> Dict[str, Any]:
        return self.writer.append(event)

    def record_artifacts(self, artifacts: Any) -> None:
        if not artifacts:
            return
        # Build candidate collections off to the side.  Mutating the writer's
        # live manifest before ``update_manifest`` validates/serializes it
        # defeats the writer's transactional guarantee: a malformed artifact
        # (for example one containing NaN or an arbitrary Python object) would
        # remain in memory even though persistence failed.
        existing_artifacts = self.writer.manifest.get("artifacts", [])
        manifest_artifacts = list(existing_artifacts) if isinstance(existing_artifacts, list) else []
        existing_hashes = self.writer.manifest.get("final_artifact_hashes", {})
        hashes = dict(existing_hashes) if isinstance(existing_hashes, Mapping) else {}
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                continue
            candidate_artifact = dict(artifact)
            manifest_artifacts.append(candidate_artifact)
            path = Path(str(candidate_artifact.get("path", ""))).expanduser().resolve()
            if path.is_file():
                digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
                hashes[str(path)] = digest
                try:
                    hashes[str(path.relative_to(self.writer.root.resolve()))] = digest
                except ValueError:
                    pass
        self.writer.update_manifest(
            artifacts=manifest_artifacts,
            final_artifact_hashes=hashes,
        )

    def record_action(
        self,
        *,
        episode_id: str,
        step_id: int,
        task_id: Optional[str],
        task_spec_hash: Optional[str],
        observation_before: Mapping[str, Any],
        action: Mapping[str, Any],
        response: Mapping[str, Any],
        observation_after: Mapping[str, Any],
        verifier: Optional[Mapping[str, Any]] = None,
        reward: Optional[Mapping[str, Any]] = None,
        checkpoint_ref: Optional[str] = None,
        done: bool = False,
        assistant_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        scorecard = verifier
        if scorecard is None:
            metrics = response.get("metrics")
            if isinstance(metrics, Mapping):
                scorecard = metrics.get("scorecard")
        if scorecard is None and action.get("name") == "verify.run":
            candidate = response.get("result")
            if isinstance(candidate, Mapping):
                scorecard = candidate
        computed_reward = dict(reward or compute_reward(
            previous_scorecard=self.previous_scorecard,
            scorecard=scorecard,
            action_success=bool(response.get("ok")),
            error_code=(response.get("error") or {}).get("code"),
        ))
        event = self.writer.append_action(
            episode_id=episode_id,
            step_id=step_id,
            task_id=task_id,
            task_spec_hash=task_spec_hash,
            observation_before=observation_before,
            action=action,
            result=response,
            observation_after=observation_after,
            verifier=scorecard,
            reward=computed_reward,
            checkpoint_ref=checkpoint_ref,
            done=done,
            assistant_text=assistant_text,
        )
        self.previous_scorecard = scorecard
        self.writer.update_manifest(final_state_hash=observation_after.get("state_hash"))
        return {"event": event, "reward": computed_reward, "verifier": scorecard}

    def finish(self, *, status: str = "complete", final_artifacts: Any = None, replay: Optional[Mapping[str, Any]] = None) -> None:
        self.writer.finish(status=status, final_artifacts=final_artifacts, replay=replay)
