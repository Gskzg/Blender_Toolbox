#!/usr/bin/env python3
"""Replay a recorded Blender Toolbox episode against a live socket."""

from __future__ import annotations

import argparse
from pathlib import Path

from blender_toolbox.client import LocalIPCClient
from blender_toolbox.replay import replay_episode
from trajectory.storage import TrajectoryReader


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--socket", default="/tmp/blender_toolbox.sock")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    # Route manifest parsing through the strict trajectory reader so replay
    # cannot silently accept non-standard NaN/Infinity values.
    manifest = TrajectoryReader(args.trajectory).manifest
    client = LocalIPCClient(args.socket, timeout=args.timeout)
    session_id = f"replay-{manifest.get('episode_id', 'episode')}"
    episode_id = f"replay-{manifest.get('episode_id', 'episode')}"
    revision = 0
    step = 0

    def reset() -> None:
        nonlocal revision, step
        response = client.action(
            session_id=session_id, episode_id=episode_id, step_id=step,
            action="session.reset", args={}, expected_revision=revision,
            idempotency_key=f"{episode_id}:reset", seed=manifest.get("seed"),
        )
        revision = int(response.get("revision", 0))
        step = 0

    def apply_action(action: dict) -> dict:
        nonlocal revision, step
        name = action["name"]
        expected = action.get("expected_revision", revision)
        key = action.get("idempotency_key") or f"{episode_id}:{step}:{name}"
        response = client.action(
            session_id=session_id, episode_id=episode_id, step_id=step,
            action=name, args=action.get("args", {}), expected_revision=expected,
            idempotency_key=key, seed=action.get("seed", manifest.get("seed")),
        )
        revision = int(response.get("revision", revision))
        step += 1
        return response

    report = replay_episode(str(args.trajectory), apply_action, reset=reset)
    import json

    print(json.dumps(report.as_dict(), indent=2, sort_keys=True, allow_nan=False))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
