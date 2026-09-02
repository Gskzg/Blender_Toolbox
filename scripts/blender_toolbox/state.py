"""Backward-compatible Blender Toolbox observation helpers.

The implementation lives in the standalone :mod:`trajectory` package.  This
module keeps the historical import path and observation schema available to
the benchmark and to older recorded episodes.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trajectory.state import canonical_state, state_diff, state_hash
from trajectory.state import observation as _generic_observation

from .version import TOOLBOX_VERSION


def observation(
    summary: Mapping[str, Any],
    *,
    revision: int,
    blender_version: str = "unknown",
    addon_version: str = TOOLBOX_VERSION,
) -> Dict[str, Any]:
    payload = _generic_observation(
        summary,
        revision=revision,
        blender_version=blender_version,
        addon_version=addon_version,
    )
    # This is intentionally the old value.  New generic trajectories use
    # trajectory.observation.v1; Blender responses remain readable by clients
    # released before the standalone package was introduced.
    payload["schema_version"] = "blender_toolbox.observation.v1"
    return payload


__all__ = ["canonical_state", "observation", "state_diff", "state_hash"]
