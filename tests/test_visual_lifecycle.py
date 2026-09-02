"""Pure-Python regression tests for the visual evidence lifecycle."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from blender_toolbox.addon import (  # noqa: E402
    ExecutorError,
    _CoreToolboxExecutor,
    _VISUAL_SCORE_KEYS,
    _ANTI_SLOP_CHECK_KEYS,
    _visual_evidence_gate,
    _visual_review,
)
from blender_toolbox.protocol import SCHEMA_VERSION  # noqa: E402


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _render_and_review(tmp_path: Path) -> tuple[dict, dict, str]:
    image = tmp_path / "front.png"
    image.write_bytes(b"render")
    digest = _hash(image)
    render = {
        "revision": 2,
        "state_hash": "sha256:scene",
        "quality_stage": "evidence",
        "views": [{"name": "front"}, {"name": "side"}, {"name": "top"}, {"name": "closeup"}],
        "files": [str(image)],
        "file_hashes": {str(image): digest},
        "evidence_types": ["beauty", "clay", "silhouette", "closeup"],
    }
    review = {
        "revision": 2,
        "state_hash": "sha256:scene",
        "quality_stage": "evidence",
        "views": ["front", "side", "top", "closeup"],
        "passed": True,
        "review_mode": "critical",
        "scores": {key: 0.9 for key in _VISUAL_SCORE_KEYS},
        "anti_slop_checks": {key: True for key in _ANTI_SLOP_CHECK_KEYS},
        "anti_slop_evidence": {key: ["front"] for key in ("regular_rows", "material_boundaries", "camera_crops", "reference_consistency")},
        "blockers": [],
        "render_hashes": {str(image): digest},
        "objective_anti_slop": {"gate": True, "blockers": []},
    }
    return render, review, str(image)


def test_visual_gate_requires_current_render_and_review() -> None:
    missing = _visual_evidence_gate(None, None, current_revision=0, require_critical=True)
    assert missing == {"gate": False, "reason": "missing_current_render"}


def test_visual_review_rejects_unrendered_view(tmp_path: Path) -> None:
    render, _, _ = _render_and_review(tmp_path)
    args = {
        "revision": 2,
        "quality_stage": "evidence",
        "views": ["not-rendered"],
        "passed": False,
        "checklist": {key: True for key in ("floating", "overlap", "alignment", "surface_contact", "framing", "proportion")},
        "findings": ["wrong view"],
    }
    with pytest.raises(ExecutorError, match="not rendered"):
        _visual_review(args, current_revision=2, last_render=render)


def test_critical_gate_checks_anti_slop_and_render_hash(tmp_path: Path) -> None:
    render, review, image = _render_and_review(tmp_path)
    passed = _visual_evidence_gate(
        render,
        review,
        current_revision=2,
        current_state_hash="sha256:scene",
        require_critical=True,
        required_evidence_types=["beauty", "clay", "silhouette", "closeup"],
        min_visual_views=4,
        min_visual_score=0.85,
    )
    assert passed["gate"] is True

    review["anti_slop_checks"]["primitive_seams"] = False
    failed = _visual_evidence_gate(render, review, current_revision=2, current_state_hash="sha256:scene", require_critical=True)
    assert failed["reason"] == "anti_slop_review_failed"

    review["anti_slop_checks"]["primitive_seams"] = True
    Path(image).write_bytes(b"changed")
    stale = _visual_evidence_gate(render, review, current_revision=2, current_state_hash="sha256:scene", require_critical=True)
    assert stale["reason"] == "render_file_changed"


def test_visual_gate_rejects_missing_state_hash_when_scene_hash_is_known(tmp_path: Path) -> None:
    render, review, _ = _render_and_review(tmp_path)
    render.pop("state_hash")
    failed = _visual_evidence_gate(render, review, current_revision=2, current_state_hash="sha256:scene")
    assert failed["reason"] == "render_state_changed"

    render, review, _ = _render_and_review(tmp_path)
    review.pop("state_hash")
    failed = _visual_evidence_gate(render, review, current_revision=2, current_state_hash="sha256:scene")
    assert failed["reason"] == "review_state_changed"


def test_critical_gate_requires_evidence_type_on_each_rendered_view(tmp_path: Path) -> None:
    render, review, _ = _render_and_review(tmp_path)
    render["views"] = [
        {"name": "front", "evidence_type": "beauty"},
        {"name": "side", "evidence_type": "beauty"},
        {"name": "top", "evidence_type": "silhouette"},
        {"name": "closeup", "evidence_type": "closeup"},
    ]
    failed = _visual_evidence_gate(
        render,
        review,
        current_revision=2,
        current_state_hash="sha256:scene",
        require_critical=True,
        required_evidence_types=["beauty", "clay", "silhouette", "closeup"],
    )
    assert failed["reason"] == "view_evidence_type_mismatch"


def test_standard_gate_rejects_missing_state_or_file_hash(tmp_path: Path) -> None:
    render, review, _ = _render_and_review(tmp_path)
    render.pop("state_hash")
    failed = _visual_evidence_gate(render, review, current_revision=2, current_state_hash="sha256:scene")
    assert failed["reason"] == "render_state_changed"
    render, review, _ = _render_and_review(tmp_path)
    render["file_hashes"] = {}
    failed = _visual_evidence_gate(render, review, current_revision=2, current_state_hash="sha256:scene")
    assert failed["reason"] == "render_file_changed"


def test_mutation_is_blocked_until_render_is_reviewed(monkeypatch: pytest.MonkeyPatch) -> None:
    executor = _CoreToolboxExecutor()
    executor._session_id = "session-test"
    executor._last_render = {"revision": 0, "views": [{"name": "front"}]}
    monkeypatch.setattr(executor, "_state", lambda *, refresh=False: {"summary": {}, "state_hash": "sha256:state"})
    called = False

    def dispatch(*args, **kwargs):
        nonlocal called
        called = True
        return {"ok": True}

    monkeypatch.setattr(executor, "_dispatch", dispatch)
    response = executor.execute(
        {
            "schema_version": SCHEMA_VERSION,
            "session_id": "session-test",
            "episode_id": "episode-test",
            "step_id": 0,
            "action": "object.transform",
            "args": {"target": "body", "location_delta": [0.1, 0.0, 0.0]},
        }
    )
    assert response["ok"] is False
    assert response["error"]["code"] == "precondition_failed"
    assert called is False
