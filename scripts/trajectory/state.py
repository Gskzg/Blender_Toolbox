"""Canonical observations and compact JSON state diffs."""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Dict, Mapping

from .storage import canonical_json, normalize_json_value


def _normal(value: Any) -> Any:
    if isinstance(value, float):
        # ``normalize_json_value`` has already rejected non-finite values;
        # retain the explicit check here so this helper remains safe when it
        # is called directly in tests or by downstream users.
        import math

        if not math.isfinite(value):
            raise ValueError("state must contain finite numbers")
        return round(value, 8)
    if isinstance(value, (list, tuple)):
        return [_normal(item) for item in value]
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for key in sorted(value, key=str):
            if not isinstance(key, str):
                raise TypeError("state object keys must be strings")
            normalized[key] = _normal(value[key])
        return normalized
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError(f"state contains a non-JSON value: {type(value).__name__}")


def canonical_state(summary: Mapping[str, Any]) -> Dict[str, Any]:
    # Normalize first to reject non-JSON values, non-string keys, cycles, and
    # non-finite numbers before copying/rounding the state tree.
    normalized = normalize_json_value(summary, "$")
    if not isinstance(normalized, dict):
        raise TypeError("state summary must be an object")
    return _normal(copy.deepcopy(normalized))


def state_hash(summary: Mapping[str, Any]) -> str:
    payload = canonical_json(canonical_state(summary))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _diff(before: Any, after: Any, path: str = "") -> list[Dict[str, Any]]:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        changes: list[Dict[str, Any]] = []
        keys = sorted(set(before) | set(after), key=str)
        for key in keys:
            child = f"{path}/{key}" if path else f"/{key}"
            if key not in before:
                changes.append({"op": "add", "path": child, "value": after[key]})
            elif key not in after:
                changes.append({"op": "remove", "path": child})
            else:
                changes.extend(_diff(before[key], after[key], child))
        return changes
    if isinstance(before, list) and isinstance(after, list):
        return [] if before == after else [{"op": "replace", "path": path or "/", "value": after}]
    return [] if before == after else [{"op": "replace", "path": path or "/", "value": after}]


def state_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[Dict[str, Any]]:
    return _diff(canonical_state(before), canonical_state(after))


def observation(
    summary: Mapping[str, Any],
    *,
    revision: int,
    blender_version: str = "unknown",
    addon_version: str = "1.0.0",
) -> Dict[str, Any]:
    canonical = canonical_state(summary)
    payload = {
        "schema_version": "trajectory.observation.v1",
        "revision": int(revision),
        "blender_version": blender_version,
        "addon_version": addon_version,
        "summary": canonical,
        "state_hash": state_hash(canonical),
    }
    normalize_json_value(payload, "$.observation")
    return payload
