"""Offline-RL trajectory filtering and export."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .storage import TrajectoryReader, canonical_json, verified_final_checkpoint


def iter_training_events(
    roots: Iterable[str | Path],
    *,
    require_replayable: bool = True,
    include_truncated: bool = False,
    include_untrusted: bool = False,
    include_errors: bool = False,
) -> Iterator[dict[str, Any]]:
    for root in roots:
        reader = TrajectoryReader(root)
        manifest = reader.manifest
        if not include_truncated and manifest.get("status") != "complete":
            continue
        if manifest.get("status") == "complete" and not verified_final_checkpoint(root, manifest):
            continue
        if require_replayable and (
            manifest.get("contains_non_replayable") or manifest.get("contains_untrusted_actions")
        ):
            continue
        try:
            events = reader.events()
        except (OSError, UnicodeDecodeError, ValueError, TypeError):
            continue
        # A valid prefix of a corrupted JSONL stream is not a replay-safe
        # training episode.  Keep export fail-closed even when callers opt in
        # to ``include_truncated`` for diagnostics.
        if not reader.events_intact:
            continue
        for event in events:
            if event.get("event_type") != "action":
                continue
            action = event.get("action") or {}
            if not include_errors:
                result = event.get("result") or {}
                if isinstance(result, Mapping) and result.get("ok") is False:
                    continue
                if event.get("event_type") in {"action_error", "llm_output_error"}:
                    continue
            if not include_untrusted and (action.get("training_allowed") is False or action.get("trusted") is False):
                continue
            if require_replayable and (
                action.get("name") == "run_python"
                or action.get("name") == "mesh.from_pydata"
                or action.get("coordinate_dump") is True
                or action.get("replayable") is False
            ):
                continue
            yield {"manifest": manifest, "event": event}


def export_training_events(
    roots: Iterable[str | Path],
    output: str | Path,
    *,
    require_replayable: bool = True,
    include_truncated: bool = False,
    include_untrusted: bool = False,
    include_errors: bool = False,
) -> int:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for item in iter_training_events(
            roots,
            require_replayable=require_replayable,
            include_truncated=include_truncated,
            include_untrusted=include_untrusted,
            include_errors=include_errors,
        ):
            # Use the same strict serializer as trajectory storage.  This
            # rejects NaN/Infinity, non-string keys, and arbitrary objects
            # instead of silently emitting a non-standard training record.
            handle.write(canonical_json(item) + "\n")
            count += 1
    return count
