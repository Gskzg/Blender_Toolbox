"""Regression tests for strict coordinate and protocol contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from blender_toolbox.addon import ExecutorError, _creation_coordinate_frame  # noqa: E402
from blender_toolbox.protocol import ProtocolError, validate_action_args  # noqa: E402


def test_relative_creation_frame_is_rejected_without_reference() -> None:
    with pytest.raises(ExecutorError, match="creation actions require"):
        _creation_coordinate_frame({"coordinate_frame": {"space": "LOCAL"}})


def test_cut_plane_uses_canonical_point_and_normal() -> None:
    validate_action_args("mesh.cut_plane", {"target": "body", "point": [0, 0, 0], "normal": [0, 0, 1]})


def test_workflow_batch_rejects_read_only_steps() -> None:
    with pytest.raises(ProtocolError, match="must be mutating"):
        validate_action_args("workflow.batch", {"intent": "bad", "steps": [{"action": "inspect.scene"}]})
