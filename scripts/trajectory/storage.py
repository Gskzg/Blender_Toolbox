"""Crash-resilient JSONL trajectory storage with content-addressed states."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _reject_json_constant(token: str) -> None:
    """Reject the non-standard JSON constants accepted by ``json`` by default."""
    raise ValueError(f"non-finite JSON constant is not allowed: {token}")


def normalize_json_value(value: Any, path: str = "$", *, _stack: Optional[set[int]] = None) -> Any:
    """Return a builtin JSON tree, rejecting lossy or non-deterministic values.

    Python's :mod:`json` encoder is deliberately permissive: it emits
    ``NaN``/``Infinity`` unless ``allow_nan`` is disabled, silently coerces
    non-string mapping keys, and can be instructed to stringify arbitrary
    objects through ``default=str``.  Trajectory files are a replay contract,
    so none of those coercions are safe.  Normalize tuples and generic
    ``Mapping`` implementations to builtin containers while rejecting every
    other value and cyclic container graph.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain finite numbers")
        return value

    if _stack is None:
        _stack = set()
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in _stack:
            raise ValueError(f"{path} contains a cyclic reference")
        _stack.add(marker)
        try:
            return [normalize_json_value(item, f"{path}[{index}]", _stack=_stack) for index, item in enumerate(value)]
        finally:
            _stack.remove(marker)
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in _stack:
            raise ValueError(f"{path} contains a cyclic reference")
        _stack.add(marker)
        try:
            normalized: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"{path} object keys must be strings")
                normalized[key] = normalize_json_value(item, f"{path}.{key}", _stack=_stack)
            return normalized
        finally:
            _stack.remove(marker)
    raise TypeError(f"{path} contains a non-JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize a deterministic, strict JSON value for trajectory hashing."""
    normalized = normalize_json_value(value)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def strict_json_loads(text: str, path: str = "$") -> Any:
    """Parse JSON and reject overflowed/non-finite values as well as literals."""
    value = json.loads(text, parse_constant=_reject_json_constant)
    return normalize_json_value(value, path)


def content_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def verified_final_checkpoint(root: str | Path, manifest: Mapping[str, Any]) -> bool:
    """Return whether a Blender completion has an intact final checkpoint."""
    if manifest.get("environment") != "blender" and not manifest.get("final_checkpoint_required"):
        return True
    ref = manifest.get("final_checkpoint_ref")
    if not isinstance(ref, str) or not ref or Path(ref).name != "final.blend":
        return False
    root_path = Path(root).resolve()
    path = (root_path / ref).resolve()
    # A manifest is persisted data and may be edited independently of the
    # writer.  Never allow ``../final.blend`` (or a symlink escaping the
    # trajectory directory) to satisfy the completion invariant.
    try:
        path.relative_to(root_path)
    except ValueError:
        return False
    if not path.is_file():
        return False
    hashes = manifest.get("final_artifact_hashes") or {}
    expected = hashes.get(ref) or hashes.get(str(path))
    if not isinstance(expected, str):
        return False
    return expected == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_json_value(value)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(normalized, handle, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


class TrajectoryWriter:
    """Write one generic episode; no environment-specific imports."""

    def __init__(self, root: str | Path, manifest: Optional[Mapping[str, Any]] = None) -> None:
        self.root = Path(root)
        self.states_dir = self.root / "states"
        self.checkpoints_dir = self.root / "checkpoints"
        self.artifacts_dir = self.root / "artifacts"
        self.events_path = self.root / "events.jsonl"
        self.manifest_path = self.root / "manifest.json"
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.states_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.artifacts_dir.mkdir(exist_ok=True)
        base = {
            "schema_version": "trajectory.manifest.v1",
            "event_count": 0,
            "status": "running",
            "contains_untrusted_actions": False,
            "contains_non_replayable": False,
            "truncated": False,
            "created_at": utc_timestamp(),
        }
        if manifest:
            base.update(dict(manifest))
        if self.manifest_path.exists():
            try:
                existing = strict_json_loads(self.manifest_path.read_text(encoding="utf-8"), "$.manifest")
                base = {**base, **existing}
                if manifest:
                    base.update(dict(manifest))
            except (OSError, ValueError):
                pass
        self.manifest = base
        self._recover_events()
        if self.manifest.get("status") == "complete" and not verified_final_checkpoint(self.root, self.manifest):
            self.manifest["status"] = "error"
            self.manifest["completion_error"] = {
                "code": "final_checkpoint_required",
                "message": "cannot mark trajectory complete without a verified final.blend checkpoint",
            }
        _json_dump(self.manifest_path, self.manifest)

    def _recover_events(self) -> None:
        if not self.events_path.exists():
            self.manifest["event_count"] = 0
            return
        try:
            raw = self.events_path.read_bytes()
            # Keep only complete, valid JSONL records.  A malformed line with
            # a newline is just as unrecoverable as an unterminated tail; the
            # previous implementation counted records before it but left the
            # bad bytes in place, causing every later reader to stop forever.
            valid_end = 0
            count = 0
            corrupted = False
            for line in raw.splitlines(keepends=True):
                has_newline = line.endswith((b"\n", b"\r"))
                body = line.rstrip(b"\r\n")
                if not body.strip():
                    if has_newline:
                        valid_end += len(line)
                        continue
                    break
                if not has_newline:
                    corrupted = True
                    break
                try:
                    parsed = strict_json_loads(body.decode("utf-8"), "$.event")
                    if not isinstance(parsed, dict):
                        raise ValueError("trajectory event must be an object")
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    corrupted = True
                    break
                count += 1
                valid_end += len(line)
            if valid_end != len(raw):
                try:
                    self.events_path.write_bytes(raw[:valid_end])
                except OSError:
                    # Do not silently continue with an event stream whose
                    # invalid tail could not be quarantined.
                    corrupted = True
            self.manifest["event_count"] = count
            if corrupted:
                # A recovered prefix is useful for diagnostics, but it is not
                # a complete/replay-safe episode.  Mark it explicitly so
                # replay and dataset export fail closed instead of training on
                # an unindexed or corrupted suffix.
                self.manifest["status"] = "truncated"
                self.manifest["truncated"] = True
                self.manifest["recovery_error"] = {
                    "code": "trajectory_events_corrupt",
                    "message": "event log contained an invalid or truncated record; only the valid prefix was recovered",
                }
        except (OSError, UnicodeDecodeError):
            self.manifest["status"] = "error"
            self.manifest["recovery_error"] = {
                "code": "trajectory_events_unreadable",
                "message": "event log could not be read during recovery",
            }
            return

    def update_manifest(self, **fields: Any) -> None:
        with self._lock:
            candidate = dict(self.manifest)
            candidate.update(fields)
            if candidate.get("status") == "complete" and not verified_final_checkpoint(self.root, candidate):
                candidate["status"] = "error"
                candidate["completion_error"] = {
                    "code": "final_checkpoint_required",
                    "message": "cannot mark trajectory complete without a verified final.blend checkpoint",
                }
            # Serialize the candidate before mutating the in-memory manifest.
            # A rejected NaN/object must not poison subsequent writes after
            # the caller catches the strict JSON error.
            _json_dump(self.manifest_path, candidate)
            self.manifest.clear()
            self.manifest.update(candidate)

    def write_state(self, summary: Mapping[str, Any], state_hash: Optional[str] = None) -> str:
        normalized = normalize_json_value(summary)
        if not isinstance(normalized, dict):
            raise TypeError("state summary must be an object")
        canonical = normalized
        computed = content_hash(canonical)
        # Older synthetic callers sometimes supplied labels such as
        # ``sha256:before``.  Those are not trusted as filenames; use the
        # computed digest.  For a properly formed digest, fail closed on a
        # mismatch so a corrupted observation cannot masquerade as a valid
        # content-addressed state.
        if state_hash is not None:
            if not isinstance(state_hash, str):
                raise TypeError("state_hash must be a string or None")
            if _SHA256_RE.fullmatch(state_hash) and state_hash != computed:
                raise ValueError("state_hash does not match the canonical state summary")
        digest = computed
        path = self.states_dir / f"{digest.split(':', 1)[-1]}.json"
        if not path.exists():
            _json_dump(path, canonical)
        return str(path.relative_to(self.root))

    def append(self, event: Mapping[str, Any]) -> Dict[str, Any]:
        with self._lock:
            payload = dict(event)
            payload.setdefault("event_schema_version", "trajectory.event.v1")
            payload.setdefault("timestamp", utc_timestamp())
            # Validate before removing sensitive reasoning fields as well: a
            # caller must not smuggle NaN or arbitrary objects through a key
            # that is later stripped from the persisted event.
            normalize_json_value(payload)
            for key in ("thinking", "reasoning", "chain_of_thought", "hidden_reasoning"):
                payload.pop(key, None)
            encoded = canonical_json(payload)
            candidate_manifest = dict(self.manifest)
            candidate_manifest["event_count"] = int(self.manifest.get("event_count", 0)) + 1
            # Validate the next manifest before appending the event, so an
            # invalid pre-existing field cannot create an unindexed JSONL
            # record when strict serialization fails afterward.
            normalize_json_value(candidate_manifest)
            encoded_bytes = (encoded + "\n").encode("utf-8")
            with self.events_path.open("ab") as handle:
                start_offset = handle.tell()
                handle.write(encoded_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                _json_dump(self.manifest_path, candidate_manifest)
            except Exception:
                # Event and manifest are a single append transaction.  If the
                # manifest cannot be committed (disk full, permission race,
                # injected serializer failure), remove the fsynced event so
                # event_count and the JSONL stream cannot diverge.
                try:
                    with self.events_path.open("r+b") as handle:
                        handle.truncate(start_offset)
                        handle.flush()
                        os.fsync(handle.fileno())
                except Exception as rollback_exc:
                    raise OSError(
                        f"trajectory manifest commit failed and event rollback failed: {rollback_exc}"
                    ) from rollback_exc
                raise
            self.manifest.clear()
            self.manifest.update(candidate_manifest)
            return payload

    def append_action(
        self,
        *,
        episode_id: str,
        step_id: int,
        task_id: Optional[str],
        task_spec_hash: Optional[str],
        observation_before: Mapping[str, Any],
        action: Mapping[str, Any],
        result: Mapping[str, Any],
        observation_after: Mapping[str, Any],
        verifier: Optional[Mapping[str, Any]],
        reward: Mapping[str, Any],
        checkpoint_ref: Optional[str] = None,
        done: bool = False,
        assistant_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        before_summary = observation_before.get("summary", observation_before)
        after_summary = observation_after.get("summary", observation_after)
        before_ref = self.write_state(before_summary, observation_before.get("state_hash"))
        after_ref = self.write_state(after_summary, observation_after.get("state_hash"))
        event: Dict[str, Any] = {
            "event_type": "action",
            "episode_id": episode_id,
            "step_id": step_id,
            "task_id": task_id,
            "task_spec_hash": task_spec_hash,
            "observation_before": dict(observation_before),
            "observation_after": dict(observation_after),
            "observation_before_ref": before_ref,
            "observation_after_ref": after_ref,
            "action": dict(action),
            "result": dict(result),
            "verifier": None if verifier is None else dict(verifier),
            "reward": dict(reward),
            "checkpoint_ref": checkpoint_ref,
            "done": bool(done),
        }
        if assistant_text:
            event["assistant_text"] = assistant_text[:4000]
        return self.append(event)

    def finish(self, *, status: str = "complete", final_artifacts: Optional[Iterable[str]] = None, replay: Optional[Mapping[str, Any]] = None) -> None:
        with self._lock:
            if status not in {"running", "complete", "error", "truncated", "aborted"}:
                raise ValueError(f"invalid trajectory status: {status}")
            candidate = dict(self.manifest)
            # Blender episodes always require an intact terminal checkpoint.
            # The explicit flag is kept for generic callers, but an
            # environment label must not be enough to bypass the invariant.
            if status == "complete" and (
                candidate.get("environment") == "blender"
                or candidate.get("final_checkpoint_required")
            ):
                if not verified_final_checkpoint(self.root, candidate):
                    status = "error"
                    candidate["completion_error"] = {
                        "code": "final_checkpoint_required",
                        "message": "cannot mark trajectory complete without a verified final.blend checkpoint",
                    }
            candidate["status"] = status
            candidate["truncated"] = status == "truncated"
            candidate["finished_at"] = utc_timestamp()
            if final_artifacts is not None:
                candidate["final_artifacts"] = list(final_artifacts)
            if replay is not None:
                candidate["replay"] = dict(replay)
            # Validate/write first, then commit the in-memory view.  This
            # keeps a failed strict serialization (e.g. NaN in replay data)
            # from leaving a half-finished manifest behind.
            _json_dump(self.manifest_path, candidate)
            self.manifest.clear()
            self.manifest.update(candidate)


class TrajectoryReader:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.last_events_intact = True
        self.last_events_error: Optional[str] = None

    @property
    def manifest(self) -> Dict[str, Any]:
        value = strict_json_loads((self.root / "manifest.json").read_text(encoding="utf-8"), "$.manifest")
        if not isinstance(value, dict):
            raise TypeError("trajectory manifest must be an object")
        return value

    def events(self) -> list[Dict[str, Any]]:
        path = self.root / "events.jsonl"
        self.last_events_intact = True
        self.last_events_error = None
        if not path.exists():
            return []
        events: list[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = strict_json_loads(line, "$.event")
                if not isinstance(value, dict):
                    raise TypeError("trajectory event must be an object")
                events.append(value)
            except (json.JSONDecodeError, ValueError, TypeError):
                self.last_events_intact = False
                self.last_events_error = "event log contains an invalid or truncated record"
                break
        return events

    @property
    def events_intact(self) -> bool:
        """Whether the most recently read event stream was fully valid."""
        return self.last_events_intact
