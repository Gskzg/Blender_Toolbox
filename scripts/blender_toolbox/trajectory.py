"""Compatibility trajectory storage for existing Blender Toolbox users.

New code should import from :mod:`trajectory`.  The wrapper preserves the
historic manifest and event schema labels while using the same crash-resilient
implementation and state deduplication.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from trajectory.storage import TrajectoryReader as _TrajectoryReader
from trajectory.storage import TrajectoryWriter as _TrajectoryWriter


class TrajectoryWriter(_TrajectoryWriter):
    """Old import path with old schema labels, backed by generic storage."""

    def __init__(self, root: str, manifest: Optional[Mapping[str, Any]] = None) -> None:
        values = dict(manifest or {})
        values.setdefault("schema_version", "blender_toolbox.trajectory.v1")
        values.setdefault("toolbox_schema_version", "blender_toolbox.v1")
        super().__init__(root, values)
        # A generic manifest may have been loaded from disk and won precedence
        # during migration. Keep the compatibility label when this class is
        # explicitly requested.
        self.manifest["schema_version"] = "blender_toolbox.trajectory.v1"
        self.update_manifest(schema_version="blender_toolbox.trajectory.v1")

    def append(self, event: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(event)
        payload.setdefault("event_schema_version", "blender_toolbox.event.v1")
        return super().append(payload)


class TrajectoryReader(_TrajectoryReader):
    """Reader alias retained for old benchmark imports."""


__all__ = ["TrajectoryReader", "TrajectoryWriter"]
