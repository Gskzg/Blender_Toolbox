"""Environment-agnostic deterministic replay reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

from .storage import TrajectoryReader, verified_final_checkpoint


@dataclass
class ReplayReport:
    ok: bool
    steps: int = 0
    mismatches: list[Dict[str, Any]] = field(default_factory=list)
    non_replayable: list[int] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "steps": self.steps, "mismatches": self.mismatches, "non_replayable": self.non_replayable}


def replay_episode(
    root: str,
    apply_action: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    *,
    reset: Optional[Callable[[], Any]] = None,
    compare_scores: bool = True,
    float_tolerance: float = 1e-5,
) -> ReplayReport:
    reader = TrajectoryReader(root)
    manifest = reader.manifest
    report = ReplayReport(ok=manifest.get("status") != "truncated")
    if manifest.get("status") == "truncated":
        report.mismatches.append({"kind": "trajectory_truncated", "status": manifest.get("status")})
    if manifest.get("status") == "complete" and not verified_final_checkpoint(root, manifest):
        report.ok = False
        report.mismatches.append({"kind": "final_checkpoint_missing"})
    if reset is not None:
        reset()
    try:
        events = reader.events()
    except (OSError, UnicodeDecodeError, ValueError, TypeError) as exc:
        report.ok = False
        report.mismatches.append({"kind": "trajectory_events_unreadable", "error": str(exc)})
        events = []
    if not reader.events_intact:
        report.ok = False
        report.mismatches.append({"kind": "trajectory_events_corrupt", "error": reader.last_events_error})
    for event in events:
        if event.get("event_type") != "action":
            continue
        step = int(event.get("step_id", report.steps))
        action = event.get("action") or {}
        if action.get("name") == "run_python" or action.get("coordinate_dump") or action.get("replayable") is False:
            report.non_replayable.append(step)
            continue
        response = dict(apply_action(action))
        report.steps += 1
        if response.get("ok") is False:
            report.ok = False
            report.mismatches.append({"step_id": step, "kind": "execution_error", "error": response.get("error")})
            continue
        expected_revision = (event.get("observation_after") or {}).get("revision")
        actual_revision = response.get("revision")
        if expected_revision is not None and actual_revision is not None and int(expected_revision) != int(actual_revision):
            report.ok = False
            report.mismatches.append({"step_id": step, "kind": "revision", "expected": expected_revision, "actual": actual_revision})
        expected_hash = (event.get("observation_after") or {}).get("state_hash")
        actual = response.get("state") or response.get("observation_after") or {}
        actual_hash = actual.get("state_hash") if isinstance(actual, Mapping) else None
        if expected_hash and actual_hash and expected_hash != actual_hash:
            report.ok = False
            report.mismatches.append({"step_id": step, "kind": "state_hash", "expected": expected_hash, "actual": actual_hash})
        if compare_scores:
            expected_quality = (event.get("reward") or {}).get("quality")
            actual_quality = (response.get("reward") or {}).get("quality")
            if expected_quality is not None and actual_quality is not None:
                try:
                    if abs(float(expected_quality) - float(actual_quality)) > float_tolerance:
                        report.ok = False
                        report.mismatches.append({"step_id": step, "kind": "quality", "expected": expected_quality, "actual": actual_quality})
                except (TypeError, ValueError):
                    report.ok = False
                    report.mismatches.append({"step_id": step, "kind": "quality_non_numeric"})
    if report.non_replayable or manifest.get("contains_non_replayable") or manifest.get("contains_untrusted_actions"):
        report.ok = False
    return report
