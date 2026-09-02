"""Optional MCP-shaped adapter for the transport-neutral Toolbox protocol.

The core package deliberately does not depend on an MCP SDK.  This small
adapter is enough for MCP hosts to discover tools and forward calls while
keeping the canonical request/response contract in :mod:`protocol`.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional

from .protocol import ActionRequest, ProtocolError, _validate_json_value, get_tool_spec, new_id


class MCPAdapter:
    """Expose a Toolbox client/session through MCP-compatible dictionaries.

    ``target`` may be a :class:`ToolboxSession` (preferred), a
    :class:`LocalIPCClient`, or any callable accepting an ActionRequest-shaped
    mapping and returning a response mapping.  No MCP runtime is imported.
    """

    def __init__(
        self,
        target: Any,
        *,
        session_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.target = target
        self.session_id = session_id or getattr(target, "session_id", new_id("mcp-sess"))
        self.episode_id = episode_id or getattr(target, "episode_id", new_id("mcp-ep"))
        self.seed = seed if seed is not None else getattr(target, "seed", None)
        self._step_id = 0
        self._revision = 0

    def list_tools(self) -> list[Dict[str, Any]]:
        """Return the MCP ``tools/list`` result payload (without envelope)."""
        from .protocol import tool_registry

        tools = []
        for spec in tool_registry():
            tools.append(
                {
                    "name": spec["name"],
                    "description": spec["description"],
                    "inputSchema": spec.get("input_schema") or {"type": "object"},
                    "outputSchema": spec.get("output_schema") or {"type": "object"},
                    "x-blender-toolbox": spec,
                }
            )
        return tools

    def tools_list(self) -> Dict[str, Any]:
        """Return the complete MCP-style result object."""
        return {"tools": self.list_tools()}

    def to_action_request(self, name: str, arguments: Optional[Mapping[str, Any]] = None) -> ActionRequest:
        """Convert an MCP tool name/arguments pair into a validated request."""
        get_tool_spec(name)
        request_args = dict(arguments or {})
        # ``stage_boundary`` is a ToolboxSession checkpoint hint, not a
        # Blender action argument.  Keep it out of strict action schemas.
        request_args.pop("stage_boundary", None)
        request = ActionRequest(
            request_id=new_id("req"),
            session_id=self.session_id,
            episode_id=self.episode_id,
            step_id=self._step_id,
            action=name,
            args=request_args,
            expected_revision=self._revision,
            idempotency_key=f"{self.episode_id}:{self._step_id}:{name}",
            seed=self.seed,
        )
        return ActionRequest.from_dict(request.as_dict())

    def _request(self, name: str, arguments: Mapping[str, Any], *, done: bool = False) -> Mapping[str, Any]:
        # Validate once at the adapter boundary so all backends, including a
        # test callable, observe exactly the same protocol contract.
        get_tool_spec(name)
        args = dict(arguments)
        stage_boundary = args.pop("stage_boundary") if "stage_boundary" in args else None
        if stage_boundary is not None and not isinstance(stage_boundary, bool):
            raise ValueError("stage_boundary must be a boolean")
        request = self.to_action_request(name, args)
        # ActionRequest.from_dict canonicalizes procedural recipes and nested
        # JSON values. Forward the canonical payload to every target type so
        # session/action targets observe the same representation as IPC and
        # callable targets.
        args = dict(request.args)
        if hasattr(self.target, "step"):
            try:
                step_kwargs = {"done": done}
                if stage_boundary is not None:
                    step_kwargs["stage_boundary"] = stage_boundary
                outcome = self.target.step(name, args, **step_kwargs)
            except TypeError as exc:
                # Keep compatibility with lightweight test doubles and older
                # adapters that do not expose the terminal flag yet.
                if "unexpected keyword argument" not in str(exc) and "positional argument" not in str(exc):
                    raise
                try:
                    outcome = self.target.step(name, args, done=done)
                except TypeError as fallback_exc:
                    if "unexpected keyword argument" not in str(fallback_exc) and "positional argument" not in str(fallback_exc):
                        raise
                    outcome = self.target.step(name, args)
            response = outcome.get("response", outcome) if isinstance(outcome, Mapping) else outcome
        elif hasattr(self.target, "action"):
            response = self.target.action(
                session_id=self.session_id,
                episode_id=self.episode_id,
                step_id=self._step_id,
                action=name,
                args=args,
                expected_revision=self._revision,
                idempotency_key=f"{self.episode_id}:{self._step_id}:{name}",
                seed=self.seed,
            )
        elif callable(self.target):
            response = self.target(request.as_dict())
        else:
            raise TypeError("MCPAdapter target must provide step(), action(), or be callable")
        if not isinstance(response, Mapping):
            raise TypeError("toolbox response must be an object")
        if isinstance(response.get("revision"), int):
            self._revision = int(response["revision"])
        self._step_id += 1
        return response

    @staticmethod
    def format_response(name: str, response: Mapping[str, Any]) -> Dict[str, Any]:
        """Format a model-facing MCP result without repeating the full scene.

        The recorder receives the original response before this function is
        called, so state snapshots and replay fidelity remain lossless while
        the LLM sees only hashes, diffs, and the result for its requested
        target.
        """
        try:
            structured = _compact_response(name, response)
            text = _json_text(structured)
        except Exception as exc:
            # A direct/test backend can return a non-JSON value even though
            # the socket executor normally guarantees one.  Keep the MCP
            # adapter's response contract intact instead of propagating a
            # serialization exception into the stdio loop.
            code = getattr(exc, "code", "response_not_json")
            structured = {"ok": False, "error": {"code": code, "message": str(exc)}}
            text = _json_text(structured)
        ok = bool(structured.get("ok"))
        return {
            "content": [{"type": "text", "text": text}],
            "isError": not ok,
            "structuredContent": structured,
        }

    def call_tool(
        self, name: str, arguments: Optional[Mapping[str, Any]] = None, *, done: bool = False
    ) -> Dict[str, Any]:
        """Return an MCP ``tools/call`` compatible result object."""
        try:
            response = self._request(name, arguments or {}, done=done)
        except Exception as exc:
            code = getattr(exc, "code", "toolbox_error")
            return {
                "content": [{"type": "text", "text": str(exc)}],
                "isError": True,
                "structuredContent": {"ok": False, "error": {"code": code, "message": str(exc)}},
            }
        return self.format_response(name, response)


def _compact_scorecard(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    compact: Dict[str, Any] = {}
    for key in ("gate", "quality", "total", "backend", "verifier_stage", "verifier_error", "quality_profile", "completion_gate", "require_visual_review"):
        if key in value:
            compact[key] = value[key]
    for key in (
        "semantic",
        "topology",
        "assembly",
        "opening",
        "proportions",
        "silhouette",
        "detail",
        "metric",
        "anti_slop",
        "visual",
        "physics",
        "generative",
    ):
        section = value.get(key)
        if not isinstance(section, Mapping):
            if section is not None:
                compact[key] = section
            continue
        item = {}
        for field in (
            "gate",
            "score",
            "quality",
            "total",
            "status",
            "method",
            "required_tags",
            "missing_tags",
            "present_tags",
            "failures",
            "warnings",
            "views",
            "checks",
            "parts",
            "contacts",
        ):
            if field in section:
                candidate = section[field]
                encoded = _json_text(candidate if isinstance(candidate, Mapping) else {"value": candidate})
                item[field] = (
                    {"elided": True, "count": len(candidate)}
                    if len(encoded) > 4000 and isinstance(candidate, (list, tuple, dict))
                    else candidate
                )
        compact[key] = item
    return compact


def _compact_result(name: str, result: Any) -> Any:
    if not isinstance(result, Mapping):
        return result
    # Inspection of one object is explicitly a target response and is kept
    # intact.  Full scene inspections are represented by a census only.
    if name in {"inspect.scene", "scene.census"}:
        objects = result.get("objects") or []
        return {
            "scene": result.get("scene"),
            "n_total": result.get("n_total"),
            "n_mesh": result.get("n_mesh"),
            "polys": result.get("polys"),
            "object_uuids": [item.get("uuid") for item in objects if isinstance(item, Mapping) and item.get("uuid")],
        }
    if name == "verify.run":
        return _compact_scorecard(result)
    payload = dict(result)
    # Do not let an auxiliary tool reintroduce the scene snapshot under a
    # nested result key.
    payload.pop("summary", None)
    return payload


def _compact_response(name: str, response: Mapping[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in (
        "schema_version",
        "request_id",
        "ok",
        "revision",
        "duration_ms",
        "checkpoint_error",
        "final_checkpoint_ref",
        "final_checkpoint_verified",
    ):
        if key in response:
            compact[key] = response[key]
    if "error" in response and response.get("error") is not None:
        compact["error"] = response["error"]
    if "result" in response:
        compact["result"] = _compact_result(name, response.get("result"))
    state = response.get("state")
    if isinstance(state, Mapping):
        compact["state"] = {
            "schema_version": state.get("schema_version"),
            "revision": state.get("revision", response.get("revision")),
            "state_hash": state.get("state_hash"),
            "diff": _compact_diff(state.get("diff", [])),
        }
    metrics = response.get("metrics")
    if isinstance(metrics, Mapping):
        compact["metrics"] = {
            key: (_compact_scorecard(value) if key == "scorecard" else value)
            for key, value in metrics.items()
            if key != "summary"
        }
    if response.get("artifacts"):
        compact["artifacts"] = response["artifacts"]
    return compact


def _compact_diff(value: Any) -> Any:
    """Keep state changes useful without embedding a replacement scene."""
    if not isinstance(value, list):
        return value
    compact = []
    for change in value:
        if not isinstance(change, Mapping):
            continue
        item = {key: change[key] for key in change if key != "value"}
        if "value" in change:
            raw = change["value"]
            if isinstance(raw, list) and any(isinstance(entry, Mapping) and entry.get("uuid") for entry in raw):
                item["value"] = {
                    "count": len(raw),
                    "uuids": [entry.get("uuid") for entry in raw if isinstance(entry, Mapping) and entry.get("uuid")],
                }
            else:
                try:
                    encoded = _json_text(raw if isinstance(raw, Mapping) else {"value": raw})
                except Exception:
                    encoded = str(raw)
                if len(encoded) > 2000:
                    item["value"] = {"elided": True, "type": type(raw).__name__, "size": len(encoded)}
                else:
                    item["value"] = raw
        compact.append(item)
    return compact


def record_external_mcp_call(
    recorder: Any,
    *,
    server: str,
    name: str,
    arguments: Optional[Mapping[str, Any]],
    result: Any,
    assistant_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Record a separate MCP call in the current episode."""
    if not isinstance(server, str) or not server:
        raise ProtocolError("server must be a non-empty string", "invalid_args")
    if not isinstance(name, str) or not name:
        raise ProtocolError("tool must be a non-empty string", "invalid_args")
    if arguments is not None and not isinstance(arguments, Mapping):
        raise ProtocolError("arguments must be an object", "invalid_args")
    if assistant_text is not None and not isinstance(assistant_text, str):
        raise ProtocolError("assistant_text must be a string", "invalid_args")
    _validate_json_value(arguments or {}, "$.arguments")
    _validate_json_value(result, "$.result")
    event: Dict[str, Any] = {
        "event_type": "mcp_call",
        "server": server,
        "tool": name,
        "arguments": dict(arguments or {}),
        "result": result,
    }
    if assistant_text:
        event["assistant_text"] = assistant_text[:4000]
    return recorder.record_event(event)


def _json_text(value: Mapping[str, Any]) -> str:
    # MCP text is itself JSON consumed by model hosts.  Never stringify
    # arbitrary Python objects or emit non-standard NaN/Infinity tokens.
    _validate_json_value(value)
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


__all__ = ["MCPAdapter", "record_external_mcp_call"]
