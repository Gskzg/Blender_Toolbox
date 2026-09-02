"""MCP discovery and recipe-schema contract tests without Blender."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from blender_toolbox import addon as addon_module  # noqa: E402
from blender_toolbox.mcp_adapter import MCPAdapter  # noqa: E402
from blender_toolbox.mcp_server import EXTERNAL_RECORD_TOOL, ToolboxMCPServer  # noqa: E402
from blender_toolbox.procedural import PROCEDURAL_RECIPE_SCHEMA, RECIPE_SCHEMA_VERSION  # noqa: E402
from blender_toolbox.protocol import get_tool_spec, tool_registry  # noqa: E402


def _geometry_recipe(*, reverse: bool = False) -> dict[str, object]:
    nodes = [
        {"id": "join", "type": "GeometryNodeJoinGeometry"},
        {"id": "cube", "type": "GeometryNodeMeshCube", "inputs": {"Size": [1, 1, 1]}},
    ]
    if reverse:
        nodes.reverse()
    return {
        "kind": "geometry_nodes",
        "nodes": nodes,
        "links": [
            {"from_node": "cube", "from_socket": "Mesh", "to_node": "join", "to_socket": "Geometry"},
        ],
    }


def test_adapter_tools_list_mirrors_canonical_registry_and_excludes_meta_tool() -> None:
    adapter = MCPAdapter(lambda _request: {"ok": True, "revision": 0})
    exposed = adapter.list_tools()
    registry = tool_registry()

    assert len(exposed) == len(registry)
    assert len({item["name"] for item in exposed}) == len(exposed)
    assert "trajectory.record_external" not in {item["name"] for item in exposed}

    by_name = {item["name"]: item for item in exposed}
    for spec in registry:
        item = by_name[spec["name"]]
        assert item["inputSchema"] == spec["input_schema"]
        assert item["outputSchema"] == spec["output_schema"]
        assert item["x-blender-toolbox"] == spec


def test_stdio_server_tools_list_adds_exactly_one_meta_tool(tmp_path: Path) -> None:
    server = ToolboxMCPServer("/tmp/toolbox-mcp-test.sock", tmp_path / "episode")
    try:
        response = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
        assert response is not None
        tools = response["result"]["tools"]
        assert len(tools) == len(tool_registry()) + 1
        assert sum(item["name"] == EXTERNAL_RECORD_TOOL["name"] for item in tools) == 1
        assert next(item for item in tools if item["name"] == EXTERNAL_RECORD_TOOL["name"]) == EXTERNAL_RECORD_TOOL
    finally:
        server.close()


def test_initialize_rejects_non_object_or_malformed_params(tmp_path: Path) -> None:
    server = ToolboxMCPServer("/tmp/toolbox-mcp-test.sock", tmp_path / "episode")
    try:
        response = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": []})
        assert response is not None
        assert response["error"]["code"] == -32602
        assert server.initialized is False

        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "initialize",
                "params": {"protocolVersion": 7, "capabilities": {}, "clientInfo": {}},
            }
        )
        assert response is not None
        assert response["error"]["code"] == -32602
        assert server.initialized is False

        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "initialize",
                "params": {"protocolVersion": "future-version", "capabilities": {}, "clientInfo": {"name": "host", "version": "1"}},
            }
        )
        assert response is not None
        assert response["result"]["protocolVersion"] == "2024-11-05"
        assert server.initialized is True
    finally:
        server.close()


def test_unknown_tool_does_not_auto_start_episode(tmp_path: Path) -> None:
    server = ToolboxMCPServer("/tmp/toolbox-mcp-test.sock", tmp_path / "episode")
    try:
        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "unknown.tool", "arguments": {}},
            }
        )
        assert response is not None
        assert response["result"]["isError"] is True
        assert server.started is False
        assert server.session.writer.manifest["event_count"] == 0
    finally:
        server.close()


def test_recipe_actions_expose_strict_nested_schema_and_version() -> None:
    adapter = MCPAdapter(lambda _request: {"ok": True, "revision": 0})
    for action, outer_required, outer_name in (
        ("geometry_nodes.apply_recipe", ["target", "recipe"], "target"),
        ("material.apply_recipe", ["name", "recipe"], "name"),
    ):
        spec = get_tool_spec(action).as_dict()
        listed = next(item for item in adapter.list_tools() if item["name"] == action)
        assert spec["input_schema"]["required"] == outer_required
        assert listed["inputSchema"] == spec["input_schema"]
        recipe_schema = listed["inputSchema"]["properties"]["recipe"]
        assert recipe_schema == PROCEDURAL_RECIPE_SCHEMA
        assert recipe_schema["required"] == ["nodes"]
        assert recipe_schema["additionalProperties"] is False
        assert recipe_schema["properties"]["schema_version"]["enum"] == [RECIPE_SCHEMA_VERSION]
        assert spec["output_schema"]["required"]
        assert outer_name in spec["input_schema"]["properties"]


def test_adapter_forwards_canonical_recipe_to_action_targets() -> None:
    class Target:
        def __init__(self) -> None:
            self.received: dict[str, object] | None = None

        def action(self, **kwargs):
            self.received = kwargs
            return {"ok": True, "revision": 1}

    target = Target()
    adapter = MCPAdapter(target)
    adapter.call_tool("geometry_nodes.apply_recipe", {"target": "obj-1", "recipe": _geometry_recipe(reverse=True)})

    assert target.received is not None
    sent_recipe = target.received["args"]["recipe"]
    canonical = adapter.to_action_request("geometry_nodes.apply_recipe", {"target": "obj-1", "recipe": _geometry_recipe()}).args["recipe"]
    assert sent_recipe == canonical
    assert sent_recipe["schema_version"] == RECIPE_SCHEMA_VERSION
    assert sent_recipe["interface"][0]["in_out"] == "INPUT"


def test_adapter_keeps_stage_boundary_out_of_strict_action_args() -> None:
    class Target:
        def __init__(self) -> None:
            self.received: dict[str, object] | None = None

        def step(self, name, args, *, done=False, stage_boundary=None):
            self.received = {
                "name": name,
                "args": args,
                "done": done,
                "stage_boundary": stage_boundary,
            }
            return {"response": {"ok": True, "revision": 0}}

    target = Target()
    adapter = MCPAdapter(target)
    adapter.call_tool(
        "material.apply_recipe",
        {
            "name": "Material",
            "stage_boundary": True,
            "recipe": {
                "kind": "material",
                "nodes": [{"id": "output", "type": "ShaderNodeOutputMaterial"}],
            },
        },
    )

    assert target.received is not None
    assert target.received["stage_boundary"] is True
    assert "stage_boundary" not in target.received["args"]


def test_stdio_session_open_forwards_quality_and_catalog_options(tmp_path: Path) -> None:
    server = ToolboxMCPServer("/tmp/toolbox-mcp-test.sock", tmp_path / "episode")
    captured: dict[str, object] = {}

    def fake_start(**kwargs):
        captured.update(kwargs)
        return {
            "ok": True,
            "revision": 0,
            "state": {"schema_version": "blender_toolbox.observation.v1", "revision": 0, "summary": {}, "state_hash": "sha256:test"},
            "result": {"session": "Scene"},
        }

    server.session.start = fake_start
    try:
        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "session.open",
                    "arguments": {
                        "mode": "new",
                        "reset": True,
                        "quality_profile": "quality_first",
                        "quality_contract": {"min_quality": 0.9},
                        "include_capabilities": True,
                        "include_examples": True,
                        "include_scene": True,
                        "scene_detail": "full",
                    },
                },
            }
        )
        assert response is not None
        assert response["result"]["structuredContent"]["ok"] is True
        assert captured == {
            "mode": "new",
            "reset": True,
            "profile": None,
            "quality_profile": "quality_first",
            "quality_contract": {"min_quality": 0.9},
            "task_spec": None,
            "include_capabilities": True,
            "include_examples": True,
            "include_scene": True,
            "scene_detail": "full",
        }
    finally:
        server.close()


def test_quality_inspection_returns_compact_authoritative_report(monkeypatch) -> None:
    monkeypatch.setattr(
        addon_module,
        "_verify",
        lambda *args, **kwargs: {
            "gate": True,
            "quality": {"first_failure": None, "repair_action": None, "unknown": []},
            "semantic": {"gate": True},
            "topology": {"gate": True},
            "summary": {"objects": [{"large": "payload"}]},
            "backend": "blender-native",
            "verifier_error": None,
        },
    )
    report = addon_module._inspect_quality({})
    assert report["gate"] is True
    assert report["quality_report"]["first_failure"] is None
    assert "summary" not in report
