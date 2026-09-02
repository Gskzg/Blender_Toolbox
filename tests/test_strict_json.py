"""Strict JSON boundary regression tests."""

from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from blender_toolbox import addon as addon_module  # noqa: E402
from blender_toolbox import mcp_server  # noqa: E402
from blender_toolbox.action_parser import parse_action  # noqa: E402
from blender_toolbox.client import LocalIPCClient, ToolboxClientError  # noqa: E402
from blender_toolbox.mcp_adapter import MCPAdapter  # noqa: E402
from blender_toolbox.protocol import ActionRequest, ProtocolError, canonical_json  # noqa: E402
from trajectory.reward import scorecard_quality  # noqa: E402
from trajectory.state import state_hash  # noqa: E402
from trajectory.storage import TrajectoryReader, TrajectoryWriter  # noqa: E402
from trajectory.storage import canonical_json as trajectory_json  # noqa: E402


class _FakeConnection:
    def __init__(self, *chunks: bytes) -> None:
        self.chunks = list(chunks)
        self.sent: list[bytes] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def settimeout(self, _timeout: float) -> None:
        return None

    def recv(self, _size: int) -> bytes:
        return self.chunks.pop(0) if self.chunks else b""

    def sendall(self, payload: bytes) -> None:
        self.sent.append(payload)


def test_protocol_rejects_nonfinite_and_non_json_values_in_full_envelope() -> None:
    request = {
        "session_id": "session",
        "episode_id": "episode",
        "step_id": 0,
        "action": "model.plan",
        "args": {},
        "diagnostic": math.inf,
    }
    with pytest.raises(ProtocolError, match="finite"):
        ActionRequest.from_dict(request)

    with pytest.raises(ProtocolError, match="non-JSON"):
        ActionRequest.from_dict({**request, "diagnostic": object()})

    with pytest.raises(ProtocolError, match="cyclic"):
        cyclic: list[object] = []
        cyclic.append(cyclic)
        canonical_json(cyclic)


def test_addon_socket_rejects_nonfinite_json_frame() -> None:
    connection = _FakeConnection(b'{"action":"model.plan","args":{"x":NaN}}\n')
    server = addon_module._CoreToolboxServer("/tmp/blender-toolbox-strict-json.sock")
    server._handle(connection)
    response = json.loads(connection.sent[0].split(b"\n", 1)[0].decode("utf-8"))
    assert response["ok"] is False
    assert response["error"]["code"] == "invalid_args"

    overflow = _FakeConnection(b'{"action":"model.plan","args":{"x":1e9999}}\n')
    server._handle(overflow)
    overflow_response = json.loads(overflow.sent[0].split(b"\n", 1)[0].decode("utf-8"))
    assert overflow_response["ok"] is False
    assert overflow_response["error"]["code"] == "invalid_args"


def test_mcp_reader_rejects_nonfinite_frame_but_can_continue() -> None:
    stream = io.BytesIO(
        b'{"jsonrpc":"2.0","id":1,"params":{"value":NaN}}\n'
        b'{"jsonrpc":"2.0","id":2,"params":{"value":1e9999}}\n'
        b'{"jsonrpc":"2.0","id":3}\n'
    )
    with pytest.raises(mcp_server.MCPFrameError, match="non-finite") as error:
        mcp_server._read_message(stream)
    assert error.value.recoverable is True
    with pytest.raises(mcp_server.MCPFrameError, match="finite"):
        mcp_server._read_message(stream)
    assert mcp_server._read_message(stream)["id"] == 3


def test_local_client_rejects_nonfinite_request_before_connect(monkeypatch) -> None:
    client = LocalIPCClient("/tmp/unused")
    connected = False

    def fail_connect():
        nonlocal connected
        connected = True
        raise AssertionError("invalid payload must be rejected before connecting")

    monkeypatch.setattr(client, "_connect", fail_connect)
    with pytest.raises(ToolboxClientError, match="strict JSON"):
        client.request({"args": {"value": math.nan}})
    assert connected is False


def test_local_client_rejects_nonfinite_response(monkeypatch) -> None:
    connection = _FakeConnection(b'{"ok":true,"value":Infinity}\n')
    client = LocalIPCClient("/tmp/unused")
    monkeypatch.setattr(client, "_connect", lambda: connection)
    with pytest.raises(ToolboxClientError, match="invalid JSON response"):
        client.request({"action": "model.plan", "args": {}})


def test_trajectory_storage_and_state_reject_lossy_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite"):
        trajectory_json({"value": math.nan})
    with pytest.raises(TypeError, match="non-JSON"):
        trajectory_json({"value": object()})
    with pytest.raises(TypeError, match="keys must be strings"):
        trajectory_json({1: "not a JSON object key"})
    with pytest.raises(ValueError, match="finite"):
        state_hash({"value": -math.inf})

    writer = TrajectoryWriter(tmp_path)
    with pytest.raises(ValueError, match="finite"):
        writer.append({"event_type": "diagnostic", "value": math.inf})
    with pytest.raises(TypeError, match="non-JSON"):
        writer.append({"event_type": "diagnostic", "thinking": object()})

    original_manifest = dict(writer.manifest)
    with pytest.raises(ValueError, match="finite"):
        writer.update_manifest(untrusted_metric=math.nan)
    assert writer.manifest == original_manifest
    assert "untrusted_metric" not in json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    with pytest.raises(TypeError, match="non-JSON"):
        writer.finish(status="aborted", replay={"callback": object()})
    assert writer.manifest == original_manifest

    # Overflowed exponents become ``inf`` in Python's JSON decoder even when
    # literal NaN/Infinity tokens are disallowed; readers must reject those
    # too rather than exposing a non-finite training state.
    (tmp_path / "events.jsonl").write_text('{"event_type":"diagnostic","value":1e9999}\n', encoding="utf-8")
    assert TrajectoryReader(tmp_path).events() == []


def test_action_parser_rejects_nonfinite_provider_output() -> None:
    with pytest.raises(ProtocolError, match="valid JSON"):
        parse_action('{"action":"model.plan","args":{"score":NaN}}')


def test_mcp_adapter_formats_non_json_backend_as_structured_error() -> None:
    response = MCPAdapter.format_response("model.plan", {"ok": True, "result": {"score": math.nan}})
    assert response["isError"] is True
    assert response["structuredContent"]["ok"] is False


def test_reward_does_not_turn_nonfinite_quality_into_a_passing_score() -> None:
    assert scorecard_quality({"quality": math.nan}) == 0.0
    assert math.isfinite(scorecard_quality({"quality": math.inf}))
