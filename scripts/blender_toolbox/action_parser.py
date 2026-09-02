"""Strict parser for one LLM-produced Toolbox action."""

from __future__ import annotations

import json
import re
from typing import Any

from .protocol import ProtocolError, _validate_json_value

_FENCE_RE = re.compile(r"^\s*```(?:json|python)?\s*|\s*```\s*$", re.IGNORECASE | re.DOTALL)


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {token}")


def parse_action(text: str) -> dict[str, Any]:
    """Parse one JSON action and reject scripts or non-object responses."""
    candidate = _FENCE_RE.sub("", text or "").strip()
    try:
        value = json.loads(candidate, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise ProtocolError("provider response is not valid JSON", "invalid_action_json")
        try:
            value = json.loads(candidate[start : end + 1], parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProtocolError(f"provider response is not valid JSON: {exc}", "invalid_action_json") from exc
    if not isinstance(value, dict):
        raise ProtocolError("toolbox response must be an object", "invalid_action_json")
    try:
        _validate_json_value(value)
    except ProtocolError as exc:
        raise ProtocolError(str(exc), "invalid_action_json") from exc
    action = value.get("action") or value.get("name")
    args = value.get("args", {})
    if not isinstance(action, str) or not isinstance(args, dict):
        raise ProtocolError("toolbox response requires string action and object args", "invalid_action_json")
    return {"action": action, "args": args, "done": bool(value.get("done", False))}
