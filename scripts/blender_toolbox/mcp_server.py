"""Minimal stdio MCP server backed by a Blender Toolbox session.

This adapter intentionally has no MCP SDK dependency. It supports the common
JSON-RPC methods used by MCP hosts (`initialize`, `tools/list`, and
`tools/call`) and uses Content-Length framing, while also accepting newline
JSON for simple local harnesses.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .client import LocalIPCClient, ToolboxSession
from .mcp_adapter import MCPAdapter, _compact_result, record_external_mcp_call
from .protocol import MAX_IPC_MESSAGE_BYTES, _validate_json_value
from .version import TOOLBOX_VERSION

_MCP_PROTOCOL_DEFAULT = "2024-11-05"
_MCP_PROTOCOL_VERSIONS = {
    # Keep the adapter conservative: these are protocol revisions commonly
    # sent by MCP hosts, and unknown future revisions are negotiated down to
    # the server's baseline rather than echoed blindly.
    "2024-11-05",
    "2025-03-26",
    "2025-06-18",
    "2025-11-25",
}

# This is intentionally an MCP-layer meta-tool rather than a Toolbox action.
# Keeping it out of protocol.tool_registry prevents external calls from being
# validated, replayed, or treated as deterministic Blender mutations.
EXTERNAL_RECORD_TOOL = {
    "name": "trajectory.record_external",
    "description": "Record a call made to another MCP server in this episode.",
    "inputSchema": {
        "type": "object",
        "required": ["server", "tool", "result"],
        "properties": {
            "server": {"type": "string", "minLength": 1},
            "tool": {"type": "string", "minLength": 1},
            "arguments": {"type": "object"},
            "result": {},
            "assistant_text": {"type": "string"},
        },
        "additionalProperties": False,
    },
    "outputSchema": {
        "type": "object",
        "required": ["recorded"],
        "properties": {"recorded": {"type": "boolean"}},
    },
    "x-blender-toolbox": {
        "event_type": "mcp_call",
        "replayable": False,
        "training_allowed": False,
    },
}


def _reject_json_constant(token: str) -> None:
    """Reject non-standard NaN/Infinity tokens in MCP input frames."""
    raise ValueError(f"non-finite JSON constant is not allowed: {token}")


class MCPFrameError(ValueError):
    """A malformed MCP frame, with whether reading can safely continue."""

    def __init__(self, message: str, *, recoverable: bool = False) -> None:
        super().__init__(message)
        self.recoverable = recoverable


class ToolboxMCPServer:
    def __init__(
        self,
        socket_address: str,
        trajectory_dir: str | Path,
        *,
        task_id: str = "mcp_episode",
        seed: Optional[int] = None,
        timeout: float = 120.0,
        auth_token: Optional[str] = None,
        checkpoint_policy: str = "topology_terminal",
        checkpoint_interval: int = 10,
    ) -> None:
        self.session = ToolboxSession(
            LocalIPCClient(socket_address, timeout=timeout, token=auth_token),
            trajectory_dir,
            task_id=task_id,
            seed=seed,
            checkpoint_policy=checkpoint_policy,
            checkpoint_interval=checkpoint_interval,
        )
        self.adapter = MCPAdapter(self.session)
        self.started = False
        self.initialized = False

    def handle(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(request, dict):
            return self._error(None, -32600, "JSON-RPC request must be an object")
        try:
            _validate_json_value(request, "$")
        except Exception as exc:
            # Direct in-process callers can bypass _read_message; preserve
            # the same strict JSON contract as parsed stdio frames.
            return self._error(_safe_jsonrpc_id(request), -32602, str(exc))
        method = request.get("method")
        request_id = request.get("id")
        if method == "notifications/initialized":
            self.initialized = True
            return None
        if method == "ping":
            return self._result(request_id, {})
        if method == "initialize":
            params = request.get("params", {})
            if params is None:
                params = {}
            if not isinstance(params, dict):
                return self._error(request_id, -32602, "initialize params must be an object")
            protocol_version = params.get("protocolVersion", _MCP_PROTOCOL_DEFAULT)
            if not isinstance(protocol_version, str) or not protocol_version.strip() or len(protocol_version) > 64:
                return self._error(request_id, -32602, "initialize protocolVersion must be a non-empty string")
            capabilities = params.get("capabilities", {})
            if not isinstance(capabilities, dict):
                return self._error(request_id, -32602, "initialize capabilities must be an object")
            client_info = params.get("clientInfo", {})
            if not isinstance(client_info, dict):
                return self._error(request_id, -32602, "initialize clientInfo must be an object")
            for key in ("name", "version"):
                if key in client_info and (not isinstance(client_info[key], str) or not client_info[key].strip()):
                    return self._error(request_id, -32602, f"initialize clientInfo.{key} must be a non-empty string")
            # Only mark the transport initialized after all fields pass
            # validation.  A malformed handshake must be safely retryable.
            self.initialized = True
            return self._result(request_id, {
                "protocolVersion": protocol_version if protocol_version in _MCP_PROTOCOL_VERSIONS else _MCP_PROTOCOL_DEFAULT,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "blender-toolbox", "version": TOOLBOX_VERSION},
            })
        if method == "tools/list":
            payload = self.adapter.tools_list()
            # Expose the recording hook to the model without adding it to the
            # canonical action registry.
            payload = dict(payload)
            payload["tools"] = list(payload.get("tools", [])) + [dict(EXTERNAL_RECORD_TOOL)]
            return self._result(request_id, payload)
        if method == "tools/call":
            params = request.get("params", {})
            if params is None:
                params = {}
            if not isinstance(params, dict):
                return self._error(request_id, -32602, "tools/call params must be an object")
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if not isinstance(name, str) or not isinstance(arguments, dict):
                return self._error(request_id, -32602, "tools/call requires name and object arguments")
            # Validate the tool name before auto-opening a Toolbox episode.
            # Unknown calls should be reported as ordinary MCP tool errors,
            # but must not create a trajectory/session as a side effect.
            known_names = {item["name"] for item in self.adapter.list_tools()}
            if name not in known_names and name != EXTERNAL_RECORD_TOOL["name"]:
                return self._result(request_id, self.adapter.call_tool(name, arguments))
            if self.session.writer.manifest.get("status") != "running" and name != "trajectory.record_external":
                return self._error(request_id, -32002, "toolbox episode is already finished")
            # Explicit session lifecycle actions must use ToolboxSession.start(),
            # which records the action and captures the initial state hashes.
            # Do not forward them through adapter.step a second time.
            if name in {"session.create", "session.open"} and not self.started:
                if name == "session.open":
                    response = self.session.start(
                        mode=str(arguments.get("mode", "resume")),
                        reset=bool(arguments.get("reset", False)),
                        profile=arguments.get("profile"),
                        quality_profile=arguments.get("quality_profile"),
                        quality_contract=arguments.get("quality_contract"),
                        task_spec=arguments.get("task_spec"),
                        include_capabilities=bool(arguments.get("include_capabilities", False)),
                        include_examples=bool(arguments.get("include_examples", False)),
                        include_scene=bool(arguments.get("include_scene", False)),
                        scene_detail=str(arguments.get("scene_detail", "compact")),
                    )
                else:
                    response = self.session.start()
                self.started = True
                self.adapter._step_id = self.session.step_id
                self.adapter._revision = self.session.revision
                return self._result(request_id, self.adapter.format_response(name, response))
            if name in {"session.create", "session.open"} and self.started:
                return self._error(request_id, -32001, "toolbox session is already started")
            if not self.started and name not in {"trajectory.record_external", "session.create", "session.open"}:
                self.session.start()
                self.started = True
                self.adapter._step_id = self.session.step_id
                self.adapter._revision = self.session.revision
            if name == "trajectory.record_external":
                missing = [key for key in ("server", "tool", "result") if key not in arguments]
                if missing:
                    return self._result(request_id, {
                        "content": [{"type": "text", "text": f"missing required keys: {missing}"}],
                        "isError": True,
                        "structuredContent": {
                            "ok": False,
                            "error": {"code": "invalid_args", "message": f"missing required keys: {missing}"},
                        },
                    })
                unknown = sorted(set(arguments) - {"server", "tool", "arguments", "result", "assistant_text"})
                if unknown:
                    return self._result(request_id, {
                        "content": [{"type": "text", "text": f"unknown keys: {unknown}"}],
                        "isError": True,
                        "structuredContent": {
                            "ok": False,
                            "error": {"code": "invalid_args", "message": f"unknown keys: {unknown}"},
                        },
                    })
                if not isinstance(arguments["server"], str) or not arguments["server"]:
                    return self._result(request_id, {
                        "content": [{"type": "text", "text": "server must be a non-empty string"}],
                        "isError": True,
                        "structuredContent": {
                            "ok": False,
                            "error": {"code": "invalid_args", "message": "server must be a non-empty string"},
                        },
                    })
                if not isinstance(arguments["tool"], str) or not arguments["tool"]:
                    return self._result(request_id, {
                        "content": [{"type": "text", "text": "tool must be a non-empty string"}],
                        "isError": True,
                        "structuredContent": {
                            "ok": False,
                            "error": {"code": "invalid_args", "message": "tool must be a non-empty string"},
                        },
                    })
                if "arguments" in arguments and not isinstance(arguments["arguments"], dict):
                    return self._result(request_id, {
                        "content": [{"type": "text", "text": "arguments must be an object"}],
                        "isError": True,
                        "structuredContent": {
                            "ok": False,
                            "error": {"code": "invalid_args", "message": "arguments must be an object"},
                        },
                    })
                if "assistant_text" in arguments and not isinstance(arguments["assistant_text"], str):
                    return self._result(request_id, {
                        "content": [{"type": "text", "text": "assistant_text must be a string"}],
                        "isError": True,
                        "structuredContent": {
                            "ok": False,
                            "error": {"code": "invalid_args", "message": "assistant_text must be a string"},
                        },
                    })
                event = record_external_mcp_call(
                    self.session.recorder,
                    server=arguments["server"],
                    name=arguments["tool"],
                    arguments=arguments.get("arguments") or {},
                    result=arguments["result"],
                    assistant_text=arguments.get("assistant_text"),
                )
                compact_event = dict(event)
                compact_event["result"] = _compact_result(arguments["tool"], arguments["result"])
                structured = {"recorded": True, "event_type": "mcp_call", "event": compact_event}
                return self._result(request_id, {
                    "content": [{
                        "type": "text",
                        "text": _strict_json_text(structured),
                    }],
                    "isError": False,
                    "structuredContent": structured,
                })
            result = self.adapter.call_tool(name, arguments, done=name == "session.close")
            if name == "session.close":
                self.started = False
            return self._result(request_id, result)
        if method == "trajectory/event":
            # External Blender MCP calls can be mirrored into the same episode
            # without pretending they were Toolbox actions.
            event = request.get("params") or {}
            if not isinstance(event, dict):
                return self._error(request_id, -32602, "trajectory/event requires an object")
            record_external_mcp_call(
                self.session.recorder,
                server=event.get("server", "external"),
                name=event.get("tool", event.get("name", "unknown")),
                arguments=event.get("arguments"),
                result=event.get("result"),
                assistant_text=event.get("assistant_text"),
            )
            return self._result(request_id, {"recorded": True})
        return self._error(request_id, -32601, f"method not found: {method}")

    @staticmethod
    def _result(request_id: Any, result: Any) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    def close(self) -> None:
        if self.session.writer.manifest.get("status") != "running":
            return
        if self.started:
            try:
                self.session.close(status="aborted")
            except Exception:
                # The stdio adapter may be used with a mocked session.start
                # and no live Blender socket. Closing must still finalize the
                # trajectory rather than masking the original MCP result.
                self.session.recorder.finish(status="aborted")
        else:
            # An episode may contain only auxiliary MCP observations. Finish
            # that trajectory even though no Blender session was opened.
            self.session.recorder.finish(status="aborted")


def _read_message(stream: Any) -> Optional[Dict[str, Any]]:
    first = _readline_limited(stream)
    if not first:
        return None
    if isinstance(first, str):
        first = first.encode("utf-8")
    if not isinstance(first, (bytes, bytearray)):
        raise MCPFrameError("MCP input must be bytes", recoverable=False)
    first = bytes(first)
    if len(first) > MAX_IPC_MESSAGE_BYTES:
        raise MCPFrameError("message exceeds IPC message limit", recoverable=False)
    if first.lower().startswith(b"content-length:"):
        try:
            length = int(first.split(b":", 1)[1].strip())
        except (IndexError, ValueError):
            raise MCPFrameError("invalid Content-Length header", recoverable=False)
        header_bytes = len(first)
        while True:
            header = _readline_limited(stream)
            if not header:
                raise MCPFrameError("truncated MCP headers", recoverable=False)
            if isinstance(header, str):
                header = header.encode("utf-8")
            if not isinstance(header, (bytes, bytearray)):
                raise MCPFrameError("MCP header must be bytes", recoverable=False)
            header = bytes(header)
            header_bytes += len(header)
            if header_bytes > MAX_IPC_MESSAGE_BYTES:
                raise MCPFrameError("MCP headers exceed IPC message limit", recoverable=False)
            if header in {b"\r\n", b"\n"}:
                break
        if length < 0 or length > MAX_IPC_MESSAGE_BYTES:
            raise MCPFrameError("Content-Length exceeds IPC message limit", recoverable=False)
        if header_bytes + length > MAX_IPC_MESSAGE_BYTES:
            raise MCPFrameError("MCP frame exceeds IPC message limit", recoverable=False)
        if header_bytes + length > MAX_IPC_MESSAGE_BYTES:
            raise MCPFrameError("MCP frame exceeds IPC message limit", recoverable=False)
        body = stream.read(length)
        if isinstance(body, str):
            body = body.encode("utf-8")
        if not isinstance(body, (bytes, bytearray)) or len(body) != length:
            raise MCPFrameError("truncated MCP message body", recoverable=False)
        body = bytes(body)
    else:
        body = first.strip()
        if len(body) > MAX_IPC_MESSAGE_BYTES:
            raise MCPFrameError("message exceeds IPC message limit", recoverable=False)
    if not body:
        return None
    try:
        value = json.loads(body.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        # A complete newline/body frame was consumed, so the next frame can
        # still be read safely after the JSON-RPC parse error is reported.
        raise MCPFrameError(f"invalid JSON message: {exc}", recoverable=True) from exc
    if not isinstance(value, dict):
        raise MCPFrameError("JSON-RPC request must be an object", recoverable=True)
    try:
        _validate_json_value(value)
    except Exception as exc:
        raise MCPFrameError(f"invalid JSON message: {exc}", recoverable=True) from exc
    return value


def _readline_limited(stream: Any) -> Any:
    """Read at most one protocol frame-sized line from a byte stream."""
    try:
        # Buffered stdin supports the size argument, preventing a malicious
        # unterminated header from being allocated without a bound.
        return stream.readline(MAX_IPC_MESSAGE_BYTES + 1)
    except TypeError:
        # Keep compatibility with tiny test doubles and stream wrappers that
        # only expose ``readline()``.
        return stream.readline()


def _write_message(stream: Any, value: Dict[str, Any]) -> None:
    """Write one bounded MCP frame.

    A tool may return an unexpectedly large or non-JSON value (notably an
    external MCP result).  Do not let that terminate the stdio server: emit a
    compact JSON-RPC error frame instead of violating the input/output size
    contract.
    """
    try:
        _validate_json_value(value)
        body = json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        value = {
            "jsonrpc": "2.0",
            "id": _safe_jsonrpc_id(value),
            "error": {"code": -32603, "message": "response is not JSON serializable"},
        }
        body = json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(body) > MAX_IPC_MESSAGE_BYTES:
        value = {
            "jsonrpc": "2.0",
            "id": _safe_jsonrpc_id(value),
            "error": {"code": -32003, "message": "response exceeds MCP message limit"},
        }
        body = json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    stream.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
    stream.flush()


def _safe_jsonrpc_id(value: Any) -> str | int | float | None:
    """Keep error correlation IDs small and JSON-RPC-compatible."""
    candidate = value.get("id") if isinstance(value, dict) else None
    if isinstance(candidate, str):
        return candidate[:256]
    if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
        try:
            if math.isfinite(float(candidate)) and len(str(candidate)) <= 128:
                return candidate
        except (TypeError, ValueError, OverflowError):
            pass
    return None


def _strict_json_text(value: Any) -> str:
    """Encode model-facing JSON text without lossy coercions."""
    _validate_json_value(value)
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default="/tmp/blender_toolbox.sock")
    parser.add_argument("--trajectory-dir", required=True, type=Path)
    parser.add_argument("--task-id", default="mcp_episode")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--auth-token", default=None)
    parser.add_argument(
        "--checkpoint-policy",
        choices=["topology_terminal", "every_action", "every_n", "stage", "none"],
        default="topology_terminal",
        help="checkpoint cadence; every_n uses --checkpoint-interval and stage uses stage_boundary=true",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="checkpoint every N mutating actions when using every_n")
    args = parser.parse_args(argv)
    server = ToolboxMCPServer(
        args.socket,
        args.trajectory_dir,
        task_id=args.task_id,
        seed=args.seed,
        timeout=args.timeout,
        auth_token=args.auth_token,
        checkpoint_policy=args.checkpoint_policy,
        checkpoint_interval=args.checkpoint_interval,
    )
    try:
        while True:
            try:
                request = _read_message(sys.stdin.buffer)
            except MCPFrameError as exc:
                # Parsing/framing failures are protocol errors, not process
                # failures.  Emit an error frame before terminating an
                # unrecoverable stream (or continue after a consumed line).
                try:
                    _write_message(sys.stdout.buffer, server._error(None, -32700, str(exc)))
                except Exception:
                    break
                if not exc.recoverable:
                    break
                continue
            except Exception as exc:
                try:
                    _write_message(sys.stdout.buffer, server._error(None, -32700, str(exc)))
                except Exception:
                    break
                break
            if request is None:
                break
            try:
                response = server.handle(request)
            except Exception as exc:
                response = server._error(request.get("id"), -32000, str(exc))
            if response is not None:
                _write_message(sys.stdout.buffer, response)
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
