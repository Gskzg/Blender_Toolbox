"""Standalone, environment-agnostic trajectory infrastructure.

The package knows about episodes, observations, rewards, artifacts and replay
reports, but has no Blender or benchmark dependency.  Environment adapters
provide snapshots/checkpoints and choose the action protocol.
"""

from .dataset import export_training_events, iter_training_events
from .recorder import EpisodeRecorder
from .replay import ReplayReport, replay_episode
from .reward import compute_reward, scorecard_quality
from .state import canonical_state, observation, state_diff, state_hash
from .storage import TrajectoryReader, TrajectoryWriter, verified_final_checkpoint

__all__ = [
    "EpisodeRecorder",
    "ReplayReport",
    "TrajectoryReader",
    "TrajectoryWriter",
    "verified_final_checkpoint",
    "canonical_state",
    "compute_reward",
    "export_training_events",
    "iter_training_events",
    "observation",
    "replay_episode",
    "scorecard_quality",
    "state_diff",
    "state_hash",
]
