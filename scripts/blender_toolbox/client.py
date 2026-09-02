"""Local IPC client and episode-aware toolbox orchestrator."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from trajectory import EpisodeRecorder
from trajectory.state import state_diff

from .protocol import ActionRequest, ActionResponse, MAX_IPC_MESSAGE_BYTES, ProtocolError, SCHEMA_VERSION, _validate_json_value, get_tool_spec, new_id
from .procedural import normalize_recipe


class ToolboxClientError(RuntimeError):
    pass


def _parse_local_tcp(address: str) -> tuple[str, int]:
    if not address.startswith("tcp://"):
        raise ToolboxClientError(f"unsupported IPC address: {address}")
    host_port = address[6:]
    if host_port.startswith("[") and "]" in host_port:
        host, _, port_text = host_port[1:].partition("]")
        port_text = port_text.lstrip(":")
    else:
        try:
            host, port_text = host_port.rsplit(":", 1)
        except ValueError as exc:
            raise ToolboxClientError("tcp address must be tcp://127.0.0.1:PORT") from exc
    host = host.lower()
    if host not in {"127.0.0.1", "localhost"}:
        raise ToolboxClientError("toolbox TCP IPC is restricted to localhost")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ToolboxClientError("tcp port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ToolboxClientError("tcp port must be between 1 and 65535")
    return host, port


class LocalIPCClient:
    """Synchronous newline-delimited JSON client for the addon socket."""

    def __init__(self, address: str, *, timeout: float = 60.0, token: Optional[str] = None) -> None:
        self.address = address
        self.timeout = timeout
        self.token = token

    def _connect(self) -> socket.socket:
        if self.address.startswith("tcp://"):
            host, port = _parse_local_tcp(self.address)
            sock = socket.create_connection((host, port), timeout=self.timeout)
        else:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect(self.address)
        return sock

    def request(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        body = dict(payload)
        body.setdefault("schema_version", SCHEMA_VERSION)
        if self.token:
            body["auth_token"] = self.token
        try:
            _validate_json_value(body)
        except ProtocolError as exc:
            raise ToolboxClientError(f"strict JSON request validation failed: {exc}") from exc
        encoded_request = (json.dumps(body, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        if len(encoded_request) > MAX_IPC_MESSAGE_BYTES:
            raise ToolboxClientError("strict JSON request exceeds IPC message limit")
        try:
            with self._connect() as sock:
                sock.sendall(encoded_request)
                chunks = []
                while True:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    if b"\n" in chunk:
                        break
        except OSError as exc:
            raise ToolboxClientError(str(exc)) from exc
        raw = b"".join(chunks).split(b"\n", 1)[0]
        if not raw:
            raise ToolboxClientError("empty response from Blender toolbox")
        try:
            response = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ToolboxClientError("invalid JSON response from Blender toolbox") from exc
        if not isinstance(response, dict):
            raise ToolboxClientError("toolbox response must be an object")
        try:
            _validate_json_value(response)
        except ProtocolError as exc:
            raise ToolboxClientError(f"invalid JSON response from Blender toolbox: {exc}") from exc
        return response

    def action(self, *, session_id: str, episode_id: str, step_id: int, action: str, args: Optional[Mapping[str, Any]] = None, expected_revision: Optional[int] = None, idempotency_key: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
        request = ActionRequest(
            request_id=new_id("req"),
            session_id=session_id,
            episode_id=episode_id,
            step_id=step_id,
            action=action,
            args=dict(args or {}),
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            seed=seed,
        )
        # Validate before opening the socket so malformed LLM output never
        # reaches Blender.
        ActionRequest.from_dict(request.as_dict())
        return self.request(request.as_dict())


class ToolboxSession:
    """Run actions while producing an RL-ready trajectory."""

    def __init__(
        self,
        client: LocalIPCClient,
        trajectory_dir: str | Path,
        *,
        task_id: str,
        task_spec_hash: Optional[str] = None,
        seed: Optional[int] = None,
        session_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        checkpoint_policy: str = "topology_terminal",
        checkpoint_interval: int = 10,
    ) -> None:
        self.client = client
        self.task_id = task_id
        self.task_spec_hash = task_spec_hash
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2_147_483_647):
            raise ValueError("seed must be an integer between 0 and 2147483647")
        if not isinstance(checkpoint_interval, int) or isinstance(checkpoint_interval, bool) or checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be a positive integer")
        self.seed = seed
        self.session_id = session_id or new_id("sess")
        self.episode_id = episode_id or new_id("ep")
        self.revision = 0
        self.step_id = 0
        self.previous_scorecard: Optional[Mapping[str, Any]] = None
        self.checkpoint_policy = checkpoint_policy
        self.checkpoint_interval = checkpoint_interval
        self._started = False
        self._final_checkpoint_saved = False
        self.last_render_evidence: Optional[Mapping[str, Any]] = None
        self.last_visual_review: Optional[Mapping[str, Any]] = None
        self.visual_review_history: list[Mapping[str, Any]] = []
        self.last_verify: Optional[Mapping[str, Any]] = None
        self.recorder = EpisodeRecorder(
            trajectory_dir,
            {
                "trajectory_schema_version": "trajectory.manifest.v1",
                "environment": "blender",
                "toolbox_schema_version": SCHEMA_VERSION,
                "task_id": task_id,
                "task_spec_hash": task_spec_hash,
                "session_id": self.session_id,
                "episode_id": self.episode_id,
                "seed": seed,
                "checkpoint_policy": checkpoint_policy,
                "initial_state_hash": None,
                "initial_scene_hash": None,
                "final_state_hash": None,
                "final_artifact_hashes": {},
                "contains_untrusted_actions": False,
                "contains_non_replayable": False,
            },
        )
        # Compatibility surface: callers may still use session.writer to add
        # prompt and provider metadata or append diagnostic events.
        self.writer = self.recorder.writer
        self._observation: Dict[str, Any] = {}

    @property
    def observation(self) -> Dict[str, Any]:
        return dict(self._observation)

    def _call(self, action: str, args: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        response = self.client.action(
            session_id=self.session_id,
            episode_id=self.episode_id,
            step_id=self.step_id,
            action=action,
            args=args,
            expected_revision=self.revision,
            idempotency_key=f"{self.episode_id}:{self.step_id}:{action}",
            seed=self.seed,
        )
        if isinstance(response.get("revision"), int):
            self.revision = int(response["revision"])
        result = response.get("result")
        if action == "render.views" and isinstance(result, Mapping) and response.get("ok"):
            self.last_render_evidence = {"revision": self.revision, "quality_stage": result.get("quality_stage"), "views": list(result.get("views") or []), "files": list(result.get("files") or []), "file_hashes": dict(result.get("file_hashes") or {}), "evidence_types": list(result.get("evidence_types") or []), "target": result.get("target")}
            self.last_visual_review = None
        elif action == "evidence.visual_review" and isinstance(result, Mapping) and response.get("ok"):
            self.last_visual_review = dict(result)
            self.visual_review_history.append(dict(result))
        elif action == "verify.run" and isinstance(result, Mapping) and response.get("ok"):
            self.last_verify = {"revision": self.revision, "gate": bool(result.get("gate")), "quality": result.get("quality"), "completion_gate": bool(result.get("completion_gate", False)), "quality_profile": result.get("quality_profile", "structural")}
        elif response.get("ok") and get_tool_spec(action).mutating:
            # Reset all evidence after scene mutations.  The executor remains
            # authoritative; this local cache is only for client ergonomics.
            self.last_render_evidence = None
            self.last_visual_review = None
            self.last_verify = None
            if action in {"scene.reset", "session.reset"}:
                self.visual_review_history.clear()
        return response

    def start(self) -> Dict[str, Any]:
        if self._started:
            return {
                "schema_version": SCHEMA_VERSION,
                "request_id": new_id("req"),
                "ok": True,
                "revision": self.revision,
                "result": {"session": self._observation.get("summary", {}).get("scene", "")},
                "state": self._observation,
                "metrics": {},
                "artifacts": [],
            }
        try:
            outcome = self.step("session.create", {}, done=False)
            response = outcome["response"]
            state = response.get("state") or {}
            self.recorder.update_manifest(
                initial_state_hash=state.get("state_hash"),
                initial_scene_hash=state.get("state_hash"),
                blender_version=state.get("blender_version"),
                addon_version=state.get("addon_version"),
            )
        except Exception as exc:
            self.recorder.record_event({
                "event_type": "episode_start_error",
                "episode_id": self.episode_id,
                "task_id": self.task_id,
                "error": {
                    "code": getattr(exc, "code", None) or "toolbox_transport_error",
                    "message": str(exc),
                },
            })
            self.recorder.finish(status="error")
            self._started = False
            raise
        if not response.get("ok", True):
            self._started = False
            error = response.get("error") or {}
            raise ToolboxClientError(str(error.get("message") or "toolbox session start failed"))
        self._started = True
        return response

    def step(self, action: str, args: Optional[Mapping[str, Any]] = None, *, assistant_text: Optional[str] = None, verifier: Optional[Mapping[str, Any]] = None, done: bool = False) -> Dict[str, Any]:
        try:
            spec = get_tool_spec(action)
        except Exception:
            spec = None
        before = self._observation
        request_args = dict(args or {})
        stage_boundary = request_args.pop("stage_boundary", None)
        if action in {"geometry_nodes.apply_recipe", "material.apply_recipe"} and isinstance(request_args.get("recipe"), Mapping):
            request_args["recipe"] = normalize_recipe(request_args["recipe"]).as_dict()
        if action == "session.close":
            self._save_final_checkpoint()
        try:
            response = self._call(action, request_args)
        except Exception as exc:
            # Preserve transport, timeout, and client-validation failures as
            # ordinary action events. This makes a crashed Blender process or
            # malformed provider action visible to trajectory consumers.
            error_code = getattr(exc, "code", None) or (
                "timeout" if isinstance(exc, TimeoutError) else "toolbox_transport_error"
            )
            response = {
                "schema_version": SCHEMA_VERSION,
                "request_id": new_id("req"),
                "ok": False,
                "revision": self.revision,
                "result": None,
                "error": {"code": error_code, "message": str(exc), "retryable": error_code in {"timeout", "blender_unavailable", "toolbox_transport_error"}},
                "state": before,
                "metrics": {},
                "artifacts": [],
                "duration_ms": 0,
            }
        raw_after = response.get("state")
        after = dict(raw_after) if isinstance(raw_after, Mapping) else dict(before)
        # Some test doubles and non-Blender adapters return a state without the
        # executor's diff. Compute it at the trajectory boundary so every
        # action has the same compact before/after signal.
        if "diff" not in after:
            before_summary = before.get("summary", before) if isinstance(before, Mapping) else before
            after_summary = after.get("summary", after) if isinstance(after, Mapping) else after
            if isinstance(before_summary, Mapping) and isinstance(after_summary, Mapping):
                after["diff"] = state_diff(before_summary, after_summary)
                response = dict(response)
                response["state"] = after
        action_record = {
            "name": action,
            "args": request_args,
            "expected_revision": before.get("revision", self.revision) if isinstance(before, Mapping) else self.revision,
            "idempotency_key": f"{self.episode_id}:{self.step_id}:{action}",
            "seed": self.seed,
            "mutating": bool(spec and spec.mutating),
            "coordinate_dump": bool(spec and spec.coordinate_dump),
            "deterministic": bool(spec and spec.deterministic),
            "training_allowed": bool(spec and spec.training_allowed),
            "trusted": bool(spec and spec.training_allowed and action != "run_python"),
            "replayable": bool(spec and spec.deterministic and not spec.coordinate_dump and action != "run_python"),
        }
        if stage_boundary is not None:
            if not isinstance(stage_boundary, bool):
                raise ValueError("stage_boundary must be a boolean")
            action_record["stage_boundary"] = stage_boundary
        if not action_record["trusted"]:
            self.recorder.update_manifest(
                contains_untrusted_actions=True,
                contains_non_replayable=True,
            )
        if action == "run_python":
            self.recorder.update_manifest(contains_run_python=True)
        checkpoint_ref = None
        checkpoint_action = action in {
            "scene.reset", "session.reset", "object.create", "object.delete",
            "object.duplicate", "object.join", "curve.create", "geometry.boolean",
            "geometry.apply_modifier", "geometry.remesh_voxel", "geometry.shrinkwrap",
            "mesh.from_pydata", "mesh.subdivide", "mesh.extrude_region", "mesh.inset_region",
            "mesh.bevel", "mesh.merge_by_distance", "mesh.delete_region", "mesh.dissolve_region",
            "mesh.fill_holes", "mesh.triangulate", "sculpt.multires",
            "hair.convert_to_mesh", "face.shape_key_landmarks",
            "particles.scatter", "geometry_nodes.create", "geometry_nodes.set_input",
            "rig.create_armature", "rig.bind", "rig.add_constraint", "face.curve_from_landmarks", "face.curve_network_from_landmarks",
            "workflow.batch", "bpy.apply",
        }
        if self.checkpoint_policy == "every_action" and spec and spec.mutating:
            checkpoint_action = True
        if self.checkpoint_policy == "every_n" and spec and spec.mutating and self.step_id % self.checkpoint_interval == 0:
            checkpoint_action = True
        if self.checkpoint_policy == "stage" and stage_boundary is True and spec and spec.mutating:
            checkpoint_action = True
        if checkpoint_action and response.get("ok") and self.checkpoint_policy in {"topology_terminal", "every_action", "every_n", "stage"}:
            checkpoint_path = self.writer.checkpoints_dir / f"step-{self.step_id:06d}.blend"
            try:
                checkpoint = self._call("artifact.save_checkpoint", {"path": str(checkpoint_path)})
                if checkpoint.get("ok"):
                    checkpoint_ref = str(checkpoint_path.relative_to(self.writer.root))
                    self.recorder.record_artifacts(checkpoint.get("artifacts"))
                else:
                    response = dict(response)
                    response["checkpoint_error"] = checkpoint.get("error") or {"code": "checkpoint_failed"}
            except Exception as exc:
                # The modeling action already succeeded. Preserve that action
                # and attach checkpoint failure metadata instead of dropping
                # the event when Blender exits during the follow-up save.
                response = dict(response)
                response["checkpoint_error"] = {
                    "code": getattr(exc, "code", None) or "checkpoint_transport_error",
                    "message": str(exc),
                }
        recorded = self.recorder.record_action(
            episode_id=self.episode_id,
            step_id=self.step_id,
            task_id=self.task_id,
            task_spec_hash=self.task_spec_hash,
            observation_before=before,
            action=action_record,
            response=response,
            observation_after=after,
            verifier=verifier,
            checkpoint_ref=checkpoint_ref,
            done=done,
            assistant_text=assistant_text,
        )
        self.recorder.record_artifacts(response.get("artifacts"))
        self._observation = after
        self.previous_scorecard = recorded.get("verifier")
        self.recorder.update_manifest(final_state_hash=after.get("state_hash"))
        self.step_id += 1
        if done:
            # Persist the terminal scene before marking the manifest complete;
            # close() intentionally skips work for already-finished episodes.
            self._save_final_checkpoint()
            self.recorder.finish(status="complete")
        return {"response": response, "event": recorded["event"], "reward": recorded["reward"], "observation": after}

    def reset(self) -> Dict[str, Any]:
        """Reset the scene within the same transport session and start a fresh revision."""
        outcome = self.step("session.reset", {}, done=False)
        self.previous_scorecard = None
        self.recorder.previous_scorecard = None
        return outcome

    def batch(
        self,
        intent: str,
        steps: list[Mapping[str, Any]],
        *,
        creates: Optional[list[str]] = None,
        modifies: Optional[list[str]] = None,
        deletes: Optional[list[str]] = None,
        verify_after: Optional[Mapping[str, Any]] = None,
        strict_declarations: bool = False,
        transaction: bool = True,
        rollback_on_error: bool = True,
        assistant_text: Optional[str] = None,
        done: bool = False,
    ) -> Dict[str, Any]:
        """Apply a bounded product-level workflow as one trace event."""
        args: Dict[str, Any] = {
            "intent": intent,
            "steps": [dict(step) for step in steps],
            "creates": list(creates or []),
            "modifies": list(modifies or []),
            "deletes": list(deletes or []),
            "rollback_on_error": bool(rollback_on_error),
            "transaction": bool(transaction),
            "strict_declarations": bool(strict_declarations),
        }
        if verify_after is not None:
            args["verify_after"] = dict(verify_after)
        return self.step("workflow.batch", args, assistant_text=assistant_text, done=done)

    def apply_bpy(
        self,
        purpose: str,
        *,
        creates: list[str],
        modifies: list[str],
        source_path: Optional[str] = None,
        source: Optional[str] = None,
        source_sha256: Optional[str] = None,
        deletes: Optional[list[str]] = None,
        postconditions: Optional[Mapping[str, Any]] = None,
        strict_declarations: bool = True,
        transaction: bool = True,
        rollback_on_error: bool = True,
        timeout_ms: Optional[int] = None,
        max_result_chars: Optional[int] = None,
        assistant_text: Optional[str] = None,
        done: bool = False,
    ) -> Dict[str, Any]:
        """Apply a declared mixed-mode bpy asset and record its intent."""
        args: Dict[str, Any] = {
            "purpose": purpose,
            "creates": list(creates),
            "modifies": list(modifies),
            "deletes": list(deletes or []),
            "strict_declarations": bool(strict_declarations),
            "rollback_on_error": bool(rollback_on_error),
            "transaction": bool(transaction),
        }
        if source_path is not None:
            args["source_path"] = source_path
        if source is not None:
            args["source"] = source
        if source_sha256 is not None:
            args["source_sha256"] = source_sha256
        if postconditions is not None:
            args["postconditions"] = dict(postconditions)
        if timeout_ms is not None:
            args["timeout_ms"] = int(timeout_ms)
        if max_result_chars is not None:
            args["max_result_chars"] = int(max_result_chars)
        return self.step("bpy.apply", args, assistant_text=assistant_text, done=done)

    def close(self, *, status: str = "complete") -> None:
        self._save_final_checkpoint()
        try:
            self._call("session.close", {})
        finally:
            self.recorder.finish(status=status)

    def _save_final_checkpoint(self) -> None:
        if self.checkpoint_policy not in {"topology_terminal", "every_action"}:
            return
        if self._final_checkpoint_saved:
            return
        if self.writer.manifest.get("status") != "running":
            return
        checkpoint_path = self.writer.checkpoints_dir / "final.blend"
        try:
            checkpoint = self._call("artifact.save_checkpoint", {"path": str(checkpoint_path)})
            if checkpoint.get("ok"):
                self.recorder.record_artifacts(checkpoint.get("artifacts"))
                self.recorder.update_manifest(final_checkpoint_ref=str(checkpoint_path.relative_to(self.writer.root)))
                self._final_checkpoint_saved = True
            else:
                self.recorder.update_manifest(final_checkpoint_error=checkpoint.get("error") or {"code": "checkpoint_failed"})
        except Exception as exc:
            self.recorder.update_manifest(final_checkpoint_error={
                "code": getattr(exc, "code", None) or "checkpoint_transport_error",
                "message": str(exc),
            })
