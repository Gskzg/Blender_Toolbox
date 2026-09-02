"""Transport-neutral procedural graph recipes.

This module is intentionally independent of Blender and third-party
procedural runtimes.  A recipe is a small, deterministic graph IR that can be
validated at the Toolbox protocol boundary and realized by a Blender adapter.
The IR borrows the useful typed-node ideas from procedural graph systems while
keeping identity, policy, and execution in the Toolbox runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional

RECIPE_SCHEMA_VERSION = "blender_toolbox.procedural_recipe.v1"
MAX_RECIPE_BYTES = 1_000_000
MAX_RECIPE_NODES = 256
MAX_RECIPE_LINKS = 1024
MAX_RECIPE_INTERFACE = 64
MAX_RECIPE_METADATA_KEYS = 64
MAX_RECIPE_NODE_ATTRIBUTES = 32
MAX_RECIPE_ATTRIBUTE_KEY_LENGTH = 64
MAX_RECIPE_SOCKET_INDEX = 4096

_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")
_TYPE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")
_SOCKET_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_. -]{0,254}$")
_ATTRIBUTE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_RECIPE_KINDS = frozenset({"geometry_nodes", "material"})
_RESERVED_NODE_IDS = frozenset({"GroupInput", "GroupOutput"})


class RecipeError(ValueError):
    """A procedural recipe is malformed or violates an adapter policy."""

    def __init__(self, message: str, code: str = "invalid_args") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RecipeNode:
    """Canonical node description in a procedural graph."""

    id: str
    type: str
    label: str
    location: Optional[tuple[float, float]]
    inputs: Mapping[str, Any]
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Frozen dataclasses do not freeze nested mappings.  Normalize and
        # recursively freeze the input tree so callers cannot mutate a recipe
        # after its hash has been recorded.
        object.__setattr__(self, "inputs", _freeze_json(_safe_json(self.inputs, "node.inputs")))
        object.__setattr__(self, "attributes", _freeze_json(_safe_json(self.attributes, "node.attributes")))

    @property
    def attrs(self) -> Mapping[str, Any]:
        """Compatibility alias for ProcFunc's ``attrs`` terminology."""
        return self.attributes

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "inputs": _thaw_json(self.inputs),
        }
        if self.attributes:
            value["attributes"] = _thaw_json(self.attributes)
        if self.location is not None:
            value["location"] = [self.location[0], self.location[1]]
        return value


@dataclass(frozen=True)
class RecipeLink:
    """Canonical data-flow edge between two node sockets."""

    from_node: str
    from_socket: str
    to_node: str
    to_socket: str
    order: Optional[int] = None
    from_socket_index: Optional[int] = None
    to_socket_index: Optional[int] = None

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "from_node": self.from_node,
            "from_socket": self.from_socket,
            "to_node": self.to_node,
            "to_socket": self.to_socket,
        }
        if self.order is not None:
            value["order"] = self.order
        if self.from_socket_index is not None:
            value["from_socket_index"] = self.from_socket_index
        if self.to_socket_index is not None:
            value["to_socket_index"] = self.to_socket_index
        return value


@dataclass(frozen=True)
class ProceduralRecipe:
    """Immutable canonical recipe suitable for hashing and persistence."""

    name: Optional[str]
    kind: str
    interface: tuple[Mapping[str, Any], ...]
    nodes: tuple[RecipeNode, ...]
    links: tuple[RecipeLink, ...]
    metadata: Mapping[str, Any]
    schema_version: str = RECIPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "interface", tuple(_freeze_json(_safe_json(item, "recipe.interface")) for item in self.interface))
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "links", tuple(self.links))
        object.__setattr__(self, "metadata", _freeze_json(_safe_json(self.metadata, "recipe.metadata")))

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "interface": [_thaw_json(item) for item in self.interface],
            "nodes": [node.as_dict() for node in self.nodes],
            "links": [link.as_dict() for link in self.links],
        }
        if self.name is not None:
            value["name"] = self.name
        if self.metadata:
            value["metadata"] = _thaw_json(self.metadata)
        return value

    @property
    def graph_hash(self) -> str:
        return recipe_hash(self.as_dict())


def _fail(message: str, code: str = "invalid_args") -> None:
    raise RecipeError(message, code)


def _safe_json(value: Any, path: str = "$", *, _stack: Optional[set[int]] = None) -> Any:
    """Copy JSON-compatible values while rejecting NaN, infinity, and keys."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(f"{path} must contain finite numbers")
        return value
    if _stack is None:
        _stack = set()
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in _stack:
            _fail(f"{path} contains a cyclic reference")
        _stack.add(marker)
        try:
            return [_safe_json(item, f"{path}[{index}]", _stack=_stack) for index, item in enumerate(value)]
        finally:
            _stack.remove(marker)
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in _stack:
            _fail(f"{path} contains a cyclic reference")
        _stack.add(marker)
        try:
            copied: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    _fail(f"{path} object keys must be strings")
                copied[key] = _safe_json(item, f"{path}.{key}", _stack=_stack)
            return {key: copied[key] for key in sorted(copied)}
        finally:
            _stack.remove(marker)
    _fail(f"{path} contains a non-JSON value: {type(value).__name__}")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


def _string(value: Any, field: str, *, pattern: Optional[re.Pattern[str]] = None, max_length: int = 255) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{field} must be a non-empty string")
    value = value.strip()
    if len(value) > max_length:
        _fail(f"{field} exceeds maximum length {max_length}")
    if pattern is not None and pattern.fullmatch(value) is None:
        _fail(f"{field} contains unsupported characters")
    return value


def _interface(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        _fail("recipe.interface must be an array")
    if len(raw) > MAX_RECIPE_INTERFACE:
        _fail(f"recipe.interface exceeds maximum item count {MAX_RECIPE_INTERFACE}")
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    allowed = {"name", "in_out", "socket_type", "default", "index"}
    has_explicit_index = any(isinstance(item, Mapping) and "index" in item for item in raw)
    for index, item in enumerate(raw):
        path = f"recipe.interface[{index}]"
        if not isinstance(item, Mapping):
            _fail(f"{path} must be an object")
        unknown = sorted(set(item) - allowed, key=str)
        if unknown:
            _fail(f"{path} contains unknown keys: {unknown}")
        name = _string(item.get("name"), f"{path}.name", pattern=_SOCKET_RE)
        in_out = _string(item.get("in_out"), f"{path}.in_out", max_length=6).upper()
        if in_out not in {"INPUT", "OUTPUT"}:
            _fail(f"{path}.in_out must be INPUT or OUTPUT")
        socket_type = _string(item.get("socket_type"), f"{path}.socket_type", pattern=_TYPE_RE)
        key = (in_out, name)
        if key in seen:
            _fail(f"duplicate recipe interface socket: {in_out}:{name}")
        seen.add(key)
        normalized: dict[str, Any] = {"name": name, "in_out": in_out, "socket_type": socket_type}
        if "default" in item:
            normalized["default"] = _safe_json(item["default"], f"{path}.default")
        if "index" in item:
            index_value = item["index"]
            if isinstance(index_value, bool) or not isinstance(index_value, int) or not 0 <= index_value < MAX_RECIPE_INTERFACE:
                _fail(f"{path}.index must be an integer between 0 and {MAX_RECIPE_INTERFACE - 1}")
            normalized["index"] = index_value
        result.append(normalized)
    if has_explicit_index:
        if any("index" not in item for item in result):
            _fail("recipe.interface.index must be specified for every socket when ordering is explicit")
        indices = [int(item["index"]) for item in result]
        if len(set(indices)) != len(indices):
            _fail("recipe.interface contains duplicate index values")
        return sorted(result, key=lambda item: (int(item["index"]), item["in_out"], item["name"], item["socket_type"]))
    return sorted(result, key=lambda item: (item["in_out"], item["name"], item["socket_type"]))


def _nodes(raw: Any, *, allowed_node_types: Optional[Iterable[str]]) -> list[RecipeNode]:
    if not isinstance(raw, (list, tuple)):
        _fail("recipe.nodes must be an array")
    if not raw:
        _fail("recipe.nodes must contain at least one node")
    if len(raw) > MAX_RECIPE_NODES:
        _fail(f"recipe.nodes exceeds maximum item count {MAX_RECIPE_NODES}")
    allowed = set(allowed_node_types) if allowed_node_types is not None else None
    result: list[RecipeNode] = []
    seen: set[str] = set()
    allowed_keys = {"id", "type", "label", "location", "inputs", "attributes", "attrs"}
    for index, item in enumerate(raw):
        path = f"recipe.nodes[{index}]"
        if not isinstance(item, Mapping):
            _fail(f"{path} must be an object")
        unknown = sorted(set(item) - allowed_keys, key=str)
        if unknown:
            _fail(f"{path} contains unknown keys: {unknown}")
        node_id = _string(item.get("id"), f"{path}.id", pattern=_ID_RE, max_length=128)
        if node_id in _RESERVED_NODE_IDS:
            _fail(f"{path}.id is reserved: {node_id}")
        if node_id in seen:
            _fail(f"duplicate recipe node id: {node_id}")
        seen.add(node_id)
        node_type = _string(item.get("type"), f"{path}.type", pattern=_TYPE_RE, max_length=128)
        if allowed is not None and node_type not in allowed:
            _fail(f"recipe node type is not allowlisted: {node_type}", "policy_denied")
        label = _string(item.get("label", node_id), f"{path}.label", max_length=255)
        location_raw = item.get("location")
        location: Optional[tuple[float, float]] = None
        if location_raw is not None:
            if not isinstance(location_raw, (list, tuple)) or len(location_raw) != 2:
                _fail(f"{path}.location must be a 2-item array")
            coords = []
            for coord in location_raw:
                if isinstance(coord, bool) or not isinstance(coord, (int, float)) or not math.isfinite(float(coord)):
                    _fail(f"{path}.location must contain finite numbers")
                coords.append(float(coord))
            location = (coords[0], coords[1])
        inputs_raw = item.get("inputs", {})
        if not isinstance(inputs_raw, Mapping):
            _fail(f"{path}.inputs must be an object")
        inputs = _safe_json(inputs_raw, f"{path}.inputs")
        if "attributes" in item and "attrs" in item:
            _fail(f"{path} must use only one of attributes or attrs")
        attributes_raw = item.get("attributes", item.get("attrs", {}))
        if not isinstance(attributes_raw, Mapping):
            _fail(f"{path}.attributes must be an object")
        if len(attributes_raw) > MAX_RECIPE_NODE_ATTRIBUTES:
            _fail(f"{path}.attributes exceeds maximum key count {MAX_RECIPE_NODE_ATTRIBUTES}")
        attributes: dict[str, Any] = {}
        for key, value in attributes_raw.items():
            if not isinstance(key, str) or _ATTRIBUTE_RE.fullmatch(key) is None:
                _fail(f"{path}.attributes keys must be identifier strings")
            attributes[key] = _safe_json(value, f"{path}.attributes.{key}")
        result.append(RecipeNode(node_id, node_type, label, location, inputs, attributes))
    return sorted(result, key=lambda node: node.id)


def _links(raw: Any, node_ids: set[str], *, allow_group_nodes: bool) -> list[RecipeLink]:
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        _fail("recipe.links must be an array")
    if len(raw) > MAX_RECIPE_LINKS:
        _fail(f"recipe.links exceeds maximum item count {MAX_RECIPE_LINKS}")
    result: list[RecipeLink] = []
    seen: set[tuple[str, str, str, str]] = set()
    seen_target_orders: set[tuple[str, str, int]] = set()
    allowed_keys = {"from_node", "from_socket", "to_node", "to_socket", "order", "from_socket_index", "to_socket_index"}
    known_ids = node_ids | (_RESERVED_NODE_IDS if allow_group_nodes else set())
    for index, item in enumerate(raw):
        path = f"recipe.links[{index}]"
        if not isinstance(item, Mapping):
            _fail(f"{path} must be an object")
        unknown = sorted(set(item) - allowed_keys, key=str)
        if unknown:
            _fail(f"{path} contains unknown keys: {unknown}")
        from_node = _string(item.get("from_node"), f"{path}.from_node", pattern=_ID_RE, max_length=128)
        to_node = _string(item.get("to_node"), f"{path}.to_node", pattern=_ID_RE, max_length=128)
        if from_node not in known_ids or to_node not in known_ids:
            _fail(f"{path} references an unknown node")
        from_socket = _string(item.get("from_socket"), f"{path}.from_socket", pattern=_SOCKET_RE)
        to_socket = _string(item.get("to_socket"), f"{path}.to_socket", pattern=_SOCKET_RE)
        key = (from_node, from_socket, to_node, to_socket)
        if key in seen:
            _fail(f"duplicate recipe link: {key}")
        seen.add(key)
        order_value = item.get("order")
        if order_value is not None and (isinstance(order_value, bool) or not isinstance(order_value, int) or not 0 <= order_value < MAX_RECIPE_LINKS):
            _fail(f"{path}.order must be an integer between 0 and {MAX_RECIPE_LINKS - 1}")
        from_index = item.get("from_socket_index")
        to_index = item.get("to_socket_index")
        for field_name, index_value in (("from_socket_index", from_index), ("to_socket_index", to_index)):
            if index_value is not None and (isinstance(index_value, bool) or not isinstance(index_value, int) or not 0 <= index_value <= MAX_RECIPE_SOCKET_INDEX):
                _fail(f"{path}.{field_name} must be an integer between 0 and {MAX_RECIPE_SOCKET_INDEX}")
        if order_value is not None:
            target_key = (to_node, to_socket, int(order_value))
            if target_key in seen_target_orders:
                _fail(f"duplicate recipe multi-input order: {target_key}")
            seen_target_orders.add(target_key)
        result.append(RecipeLink(from_node, from_socket, to_node, to_socket, order_value, from_index, to_index))
    if any(link.order is not None for link in result):
        return sorted(result, key=lambda link: (
            link.to_node, link.to_socket,
            link.order is None, link.order if link.order is not None else 0,
            link.from_node, link.from_socket,
            link.from_socket_index if link.from_socket_index is not None else -1,
            link.to_socket_index if link.to_socket_index is not None else -1,
        ))
    return sorted(result, key=lambda link: (link.from_node, link.from_socket, link.to_node, link.to_socket))


def _reject_cycles(nodes: Iterable[RecipeNode], links: Iterable[RecipeLink]) -> None:
    """Reject data-flow cycles before Blender receives a graph."""
    node_ids = {node.id for node in nodes}
    outgoing: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for link in links:
        if link.from_node not in node_ids or link.to_node not in node_ids:
            continue
        if link.to_node not in outgoing[link.from_node]:
            outgoing[link.from_node].add(link.to_node)
            indegree[link.to_node] += 1
    queue = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        current = queue.pop(0)
        visited += 1
        for target in sorted(outgoing[current]):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
        queue.sort()
    if visited != len(node_ids):
        _fail("recipe graph contains a data-flow cycle")


def normalize_recipe(
    raw: Mapping[str, Any],
    *,
    expected_kind: Optional[str] = None,
    allowed_node_types: Optional[Iterable[str]] = None,
) -> ProceduralRecipe:
    """Validate and return a canonical immutable procedural recipe.

    Ordering of nodes, interface sockets, links, and mapping keys is
    normalized so equivalent recipes produce the same ``graph_hash``.
    """
    if not isinstance(raw, Mapping):
        _fail("recipe must be an object")
    allowed_root = {"schema_version", "name", "kind", "interface", "nodes", "links", "metadata"}
    unknown = sorted(set(raw) - allowed_root, key=str)
    if unknown:
        _fail(f"recipe contains unknown keys: {unknown}")
    schema_version = raw.get("schema_version", RECIPE_SCHEMA_VERSION)
    if schema_version != RECIPE_SCHEMA_VERSION:
        _fail(f"unsupported recipe schema_version: {schema_version!r}", "unsupported_version")
    kind = _string(raw.get("kind", expected_kind or "geometry_nodes"), "recipe.kind", max_length=32)
    if kind not in _RECIPE_KINDS:
        _fail(f"recipe.kind must be one of {sorted(_RECIPE_KINDS)}")
    if expected_kind is not None and kind != expected_kind:
        _fail(f"recipe.kind must be {expected_kind}")
    name_raw = raw.get("name")
    name = None if name_raw is None else _string(name_raw, "recipe.name", max_length=255)
    interface = _interface(raw.get("interface"))
    if kind == "geometry_nodes" and not interface:
        # Match Blender's implicit pass-through group interface so omitted
        # defaults and an explicit Geometry input/output recipe hash alike.
        interface = _interface(
            [
                {"name": "Geometry", "in_out": "INPUT", "socket_type": "NodeSocketGeometry"},
                {"name": "Geometry", "in_out": "OUTPUT", "socket_type": "NodeSocketGeometry"},
            ]
        )
    nodes = _nodes(raw.get("nodes"), allowed_node_types=allowed_node_types)
    links = _links(
        raw.get("links"),
        {node.id for node in nodes},
        allow_group_nodes=kind == "geometry_nodes",
    )
    if kind == "geometry_nodes":
        for link in links:
            if link.from_node == "GroupOutput":
                _fail("GroupOutput cannot be a recipe link source")
            if link.to_node == "GroupInput":
                _fail("GroupInput cannot be a recipe link target")
    _reject_cycles(nodes, links)
    metadata_raw = raw.get("metadata", {})
    if not isinstance(metadata_raw, Mapping):
        _fail("recipe.metadata must be an object")
    if len(metadata_raw) > MAX_RECIPE_METADATA_KEYS:
        _fail(f"recipe.metadata exceeds maximum key count {MAX_RECIPE_METADATA_KEYS}")
    metadata = _safe_json(metadata_raw, "recipe.metadata")
    result = ProceduralRecipe(name, kind, tuple(interface), tuple(nodes), tuple(links), metadata)
    encoded = canonical_recipe_json(result.as_dict())
    if len(encoded.encode("utf-8")) > MAX_RECIPE_BYTES:
        _fail(f"recipe exceeds maximum serialized size {MAX_RECIPE_BYTES} bytes")
    return result


def validate_recipe(
    raw: Mapping[str, Any],
    *,
    expected_kind: Optional[str] = None,
    allowed_node_types: Optional[Iterable[str]] = None,
) -> None:
    """Validate a recipe and discard its normalized representation."""
    normalize_recipe(raw, expected_kind=expected_kind, allowed_node_types=allowed_node_types)


def canonical_recipe_json(value: Any) -> str:
    """Serialize recipe values with deterministic JSON rules."""
    return json.dumps(_safe_json(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def recipe_hash(recipe: Mapping[str, Any] | ProceduralRecipe) -> str:
    """Return a stable SHA-256 hash for a normalized recipe."""
    value = recipe.as_dict() if isinstance(recipe, ProceduralRecipe) else recipe
    return "sha256:" + hashlib.sha256(canonical_recipe_json(value).encode("utf-8")).hexdigest()


PROCEDURAL_RECIPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "schema_version": {"type": "string", "enum": [RECIPE_SCHEMA_VERSION]},
        "name": {"type": "string", "minLength": 1, "maxLength": 255},
        "kind": {"type": "string", "enum": sorted(_RECIPE_KINDS)},
        "interface": {
            "type": "array",
            "maxItems": MAX_RECIPE_INTERFACE,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1, "maxLength": 255},
                    "in_out": {"type": "string", "enum": ["INPUT", "OUTPUT"]},
                    "socket_type": {"type": "string", "minLength": 1, "maxLength": 128},
                    "default": {},
                    "index": {"type": "integer", "minimum": 0, "maximum": MAX_RECIPE_INTERFACE - 1},
                },
                "required": ["name", "in_out", "socket_type"],
                "additionalProperties": False,
            },
        },
        "nodes": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_RECIPE_NODES,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "minLength": 1, "maxLength": 128},
                    "type": {"type": "string", "minLength": 1, "maxLength": 128},
                    "label": {"type": "string", "minLength": 1, "maxLength": 255},
                    "location": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}},
                    "inputs": {"type": "object"},
                    "attributes": {"type": "object", "maxProperties": MAX_RECIPE_NODE_ATTRIBUTES},
                    "attrs": {"type": "object", "maxProperties": MAX_RECIPE_NODE_ATTRIBUTES},
                },
                "required": ["id", "type"],
                "additionalProperties": False,
            },
        },
        "links": {
            "type": "array",
            "maxItems": MAX_RECIPE_LINKS,
            "items": {
                "type": "object",
                "properties": {
                    "from_node": {"type": "string", "minLength": 1, "maxLength": 128},
                    "from_socket": {"type": "string", "minLength": 1, "maxLength": 255},
                    "to_node": {"type": "string", "minLength": 1, "maxLength": 128},
                    "to_socket": {"type": "string", "minLength": 1, "maxLength": 255},
                    "order": {"type": "integer", "minimum": 0, "maximum": MAX_RECIPE_LINKS - 1},
                    "from_socket_index": {"type": "integer", "minimum": 0, "maximum": MAX_RECIPE_SOCKET_INDEX},
                    "to_socket_index": {"type": "integer", "minimum": 0, "maximum": MAX_RECIPE_SOCKET_INDEX},
                },
                "required": ["from_node", "from_socket", "to_node", "to_socket"],
                "additionalProperties": False,
            },
        },
        "metadata": {"type": "object"},
    },
    "required": ["nodes"],
    "additionalProperties": False,
}


__all__ = [
    "MAX_RECIPE_BYTES",
    "MAX_RECIPE_INTERFACE",
    "MAX_RECIPE_LINKS",
    "MAX_RECIPE_NODES",
    "MAX_RECIPE_NODE_ATTRIBUTES",
    "MAX_RECIPE_ATTRIBUTE_KEY_LENGTH",
    "MAX_RECIPE_SOCKET_INDEX",
    "PROCEDURAL_RECIPE_SCHEMA",
    "RECIPE_SCHEMA_VERSION",
    "ProceduralRecipe",
    "RecipeError",
    "RecipeLink",
    "RecipeNode",
    "canonical_recipe_json",
    "normalize_recipe",
    "recipe_hash",
    "validate_recipe",
]
