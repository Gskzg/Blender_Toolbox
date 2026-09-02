"""Gymnasium-shaped adapter for :class:`ToolboxSession`.

The adapter intentionally does not import gymnasium.  It keeps the standard
reset/step return contract so projects may wrap it in their own space/action
definitions without adding a hard dependency to the toolbox runtime.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .client import ToolboxSession


class ToolboxEnv:
    """Expose a toolbox episode as a Gymnasium-style environment."""

    metadata = {"render_modes": []}

    def __init__(self, session: ToolboxSession, *, max_steps: int = 64) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.session = session
        self.max_steps = int(max_steps)
        self.steps = 0
        self.started = False
        self.terminated = False
        self.successful_termination = False
        self.truncated = False

    @staticmethod
    def _observation(session: ToolboxSession) -> dict[str, Any]:
        return dict(session.observation)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None) -> tuple[dict[str, Any], dict[str, Any]]:
        del options
        if seed is not None:
            self.session.seed = int(seed)
        if self.started:
            outcome = self.session.reset()
            response = outcome.get("response", {})
        else:
            response = self.session.start()
            self.started = True
        self.steps = 0
        self.terminated = False
        self.successful_termination = False
        self.truncated = False
        observation = self._observation(self.session)
        return observation, {"response": response, "revision": observation.get("revision", 0)}

    def step(self, action: Mapping[str, Any] | tuple[str, Mapping[str, Any]]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if not self.started:
            raise RuntimeError("reset() must be called before step()")
        if self.terminated:
            raise RuntimeError("episode is done; call reset() before step()")
        if isinstance(action, Mapping):
            name = action.get("action") or action.get("name")
            args = action.get("args", {})
            requested_done = bool(action.get("done", False))
        elif isinstance(action, tuple) and len(action) == 2:
            name, args = action
            requested_done = False
        else:
            raise TypeError("action must be a mapping or (name, args) tuple")
        if not isinstance(name, str) or not isinstance(args, Mapping):
            raise TypeError("action requires a string name and object args")
        self.steps += 1
        # Export is an observation/artifact operation, not an episode
        # terminator.  Only an explicit done flag (normally session.close)
        # may finish the trajectory and trigger the final checkpoint gate.
        outcome = self.session.step(name, args, done=requested_done)
        response = outcome.get("response", {})
        ok = bool(response.get("ok"))
        terminated = bool(requested_done)
        truncated = self.steps >= self.max_steps and not terminated
        if not ok:
            terminated = True
        self.terminated = terminated or truncated
        self.successful_termination = bool(terminated and ok and not truncated)
        self.truncated = truncated
        reward = float((outcome.get("reward") or {}).get("total", 0.0))
        info = {
            "response": response,
            "event": outcome.get("event"),
            "reward": outcome.get("reward", {}),
            "revision": response.get("revision"),
        }
        return self._observation(self.session), reward, terminated, truncated, info

    def close(self) -> None:
        if self.started:
            status = "complete" if self.successful_termination else ("truncated" if self.truncated else "aborted")
            self.session.close(status=status)
            self.started = False
