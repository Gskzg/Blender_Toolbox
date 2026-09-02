"""IPC framing and size-limit tests that do not require Blender."""

from __future__ import annotations

import io
import json
import math
import socket
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from blender_toolbox import addon as addon_module  # noqa: E402
from blender_toolbox import mcp_server  # noqa: E402


class _FakeConnection:
    def __init__(self, *chunks: bytes, timeout_exc: BaseException | None = None) -> None:
        self.chunks = list(chunks)
        self.timeout_exc = timeout_exc
        self.sent: list[bytes] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def settimeout(self, _timeout: float) -> None:
        return None

    def recv(self, _size: int) -> bytes:
        if self.timeout_exc is not None:
            exc, self.timeout_exc = self.timeout_exc, None
            raise exc
        return self.chunks.pop(0) if self.chunks else b""

    def sendall(self, payload: bytes) -> None:
        self.sent.append(payload)


def _response_from_connection(connection: _FakeConnection) -> dict[str, object]:
    assert len(connection.sent) == 1
    frame = connection.sent[0].split(b"\n", 1)[0]
    return json.loads(frame.decode("utf-8"))


def test_addon_socket_returns_error_for_oversized_unterminated_frame(monkeypatch) -> None:
    monkeypatch.setattr(addon_module, "MAX_IPC_MESSAGE_BYTES", 512)
    connection = _FakeConnection(b"x" * 513)
    server = addon_module._CoreToolboxServer("/tmp/blender-toolbox-ipc-test.sock")

    server._handle(connection)

    response = _response_from_connection(connection)
    assert response["ok"] is False
    assert response["error"]["code"] == "invalid_args"


def test_addon_socket_returns_error_for_read_timeout() -> None:
    connection = _FakeConnection(timeout_exc=socket.timeout("frame read timed out"))
    server = addon_module._CoreToolboxServer("/tmp/blender-toolbox-ipc-test.sock")

    server._handle(connection)

    response = _response_from_connection(connection)
    assert response["ok"] is False


def test_mcp_write_message_replaces_oversized_response(monkeypatch) -> None:
    monkeypatch.setattr(mcp_server, "MAX_IPC_MESSAGE_BYTES", 256)
    stream = io.BytesIO()

    mcp_server._write_message(stream, {"jsonrpc": "2.0", "id": 7, "result": {"blob": "x" * 512}})

    payload = stream.getvalue().split(b"\r\n\r\n", 1)[1]
    response = json.loads(payload.decode("utf-8"))
    assert len(payload) <= 256
    assert response["error"]["code"] == -32003
    assert response["id"] == 7


def test_mcp_write_message_replaces_non_json_response() -> None:
    stream = io.BytesIO()

    mcp_server._write_message(stream, {"jsonrpc": "2.0", "id": 8, "result": {"bad": object()}})

    payload = stream.getvalue().split(b"\r\n\r\n", 1)[1]
    response = json.loads(payload.decode("utf-8"))
    assert response["error"]["code"] == -32603


def test_mcp_write_message_replaces_nan_and_unsafe_error_id() -> None:
    stream = io.BytesIO()

    mcp_server._write_message(stream, {"jsonrpc": "2.0", "id": object(), "result": {"bad": math.nan}})

    payload = stream.getvalue().split(b"\r\n\r\n", 1)[1]
    response = json.loads(payload.decode("utf-8"))
    assert response["error"]["code"] == -32603
    assert response["id"] is None


def test_mcp_write_message_replaces_nan_error_id() -> None:
    stream = io.BytesIO()

    mcp_server._write_message(stream, {"jsonrpc": "2.0", "id": math.nan, "result": {"ok": True}})

    payload = stream.getvalue().split(b"\r\n\r\n", 1)[1]
    response = json.loads(payload.decode("utf-8"))
    assert response["error"]["code"] == -32603
    assert response["id"] is None


def test_mcp_reader_rejects_truncated_content_length_body() -> None:
    body = b'{"jsonrpc":"2.0"}'
    stream = io.BytesIO(b"Content-Length: 64\r\n\r\n" + body)

    with pytest.raises(mcp_server.MCPFrameError, match="truncated MCP message body") as error:
        mcp_server._read_message(stream)
    assert error.value.recoverable is False


def test_mcp_reader_allows_next_frame_after_parse_error() -> None:
    stream = io.BytesIO(b"{not-json}\n{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}\n")

    with pytest.raises(mcp_server.MCPFrameError, match="invalid JSON message") as error:
        mcp_server._read_message(stream)
    assert error.value.recoverable is True
    assert mcp_server._read_message(stream)["id"] == 1


def test_mcp_reader_bounds_header_accumulation(monkeypatch) -> None:
    monkeypatch.setattr(mcp_server, "MAX_IPC_MESSAGE_BYTES", 32)
    stream = io.BytesIO(b"Content-Length: 1\r\nX-Header: too-long\r\n\r\n1")

    with pytest.raises(mcp_server.MCPFrameError, match="headers exceed IPC message limit"):
        mcp_server._read_message(stream)


class _BinaryInput:
    def __init__(self, payload: bytes) -> None:
        self.buffer = io.BytesIO(payload)


class _BinaryOutput:
    def __init__(self) -> None:
        self.buffer = io.BytesIO()


def _output_frames(output: _BinaryOutput) -> list[dict[str, object]]:
    raw = output.buffer.getvalue()
    frames = []
    cursor = 0
    while cursor < len(raw):
        marker = raw.find(b"\r\n\r\n", cursor)
        assert marker >= 0
        header = raw[cursor:marker]
        length = int(header.split(b":", 1)[1].strip())
        start = marker + 4
        body = raw[start : start + length]
        frames.append(json.loads(body.decode("utf-8")))
        cursor = start + length
    return frames


def test_mcp_main_reports_parse_error_and_continues(monkeypatch, tmp_path: Path) -> None:
    class FakeServer:
        def __init__(self, *_args, **_kwargs) -> None:
            self.closed = False

        @staticmethod
        def _error(request_id, code, message):
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

        @staticmethod
        def handle(request):
            return {"jsonrpc": "2.0", "id": request.get("id"), "result": {"ok": True}}

        def close(self):
            self.closed = True

    input_stream = _BinaryInput(b"{bad}\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}\n")
    output_stream = _BinaryOutput()
    monkeypatch.setattr(mcp_server, "ToolboxMCPServer", FakeServer)
    monkeypatch.setattr(mcp_server.sys, "stdin", input_stream)
    monkeypatch.setattr(mcp_server.sys, "stdout", output_stream)

    assert mcp_server.main(["--trajectory-dir", str(tmp_path)]) == 0
    frames = _output_frames(output_stream)
    assert frames[0]["error"]["code"] == -32700
    assert frames[1]["result"] == {"ok": True}


def test_mcp_main_reports_truncated_body_before_exit(monkeypatch, tmp_path: Path) -> None:
    class FakeServer:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        @staticmethod
        def _error(request_id, code, message):
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

        def close(self):
            return None

    input_stream = _BinaryInput(b"Content-Length: 20\r\n\r\n{}")
    output_stream = _BinaryOutput()
    monkeypatch.setattr(mcp_server, "ToolboxMCPServer", FakeServer)
    monkeypatch.setattr(mcp_server.sys, "stdin", input_stream)
    monkeypatch.setattr(mcp_server.sys, "stdout", output_stream)

    assert mcp_server.main(["--trajectory-dir", str(tmp_path)]) == 0
    frames = _output_frames(output_stream)
    assert len(frames) == 1
    assert frames[0]["error"]["code"] == -32700
