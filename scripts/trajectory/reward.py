"""Environment-agnostic verifier-first reward shaping."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

DEFAULT_WEIGHTS = {"metric": 0.30, "silhouette": 0.30, "detail": 0.15, "generative": 0.25}


def _score(value: Any) -> Optional[float]:
    if isinstance(value, Mapping):
        value = value.get("score")
    if value is None:
        return None
    try:
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return max(0.0, min(1.0, numeric))
    except (TypeError, ValueError):
        return None


def scorecard_quality(scorecard: Optional[Mapping[str, Any]]) -> float:
    if not scorecard:
        return 0.0
    hard_gate = 1.0
    for key in ("semantic", "topology", "physics", "assembly"):
        section = scorecard.get(key)
        if isinstance(section, Mapping) and section.get("gate") is False:
            hard_gate = 0.0
    if scorecard.get("gate") is False:
        hard_gate = 0.0
    weighted = 0.0
    used = 0.0
    for key, weight in DEFAULT_WEIGHTS.items():
        value = _score(scorecard.get(key))
        if value is not None:
            weighted += weight * value
            used += weight
    if used == 0.0:
        values = [_score(scorecard.get(key)) for key in ("topology", "physics", "assembly")]
        values = [value for value in values if value is not None]
        weighted = sum(values) / len(values) if values else (_score(scorecard.get("quality", scorecard.get("total"))) or 0.0)
    else:
        weighted /= used
    return round(hard_gate * weighted, 6)


def compute_reward(
    *,
    previous_scorecard: Optional[Mapping[str, Any]],
    scorecard: Optional[Mapping[str, Any]],
    action_success: bool,
    error_code: Optional[str] = None,
    timeout_cost: float = 0.10,
    invalid_action_cost: float = 0.05,
    delta_weight: float = 1.0,
) -> dict[str, float]:
    previous = scorecard_quality(previous_scorecard)
    current = scorecard_quality(scorecard)
    delta = current - previous
    action_signal = 1.0 if action_success else 0.0
    is_timeout = error_code in {"timeout", "blender_timeout"}
    penalty = timeout_cost if (not action_success and is_timeout) else (invalid_action_cost if not action_success else 0.0)
    return {
        "action_success": round(action_signal, 6),
        "previous_quality": round(previous, 6),
        "quality": round(current, 6),
        "score_delta": round(delta, 6),
        "timeout_cost": round(timeout_cost if is_timeout else 0.0, 6),
        "invalid_action_cost": round(invalid_action_cost if (not action_success and not is_timeout) else 0.0, 6),
        "total": round(action_signal + delta_weight * delta - penalty, 6),
    }
