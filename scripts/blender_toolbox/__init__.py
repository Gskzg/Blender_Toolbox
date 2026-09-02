"""Protocol and client utilities for the Blender Toolbox.

The package is deliberately usable outside Blender.  The Blender-specific
executor lives in :mod:`blender_toolbox.addon` and is only imported by a
Blender process.
"""

from trajectory import EpisodeRecorder

from .gym_adapter import ToolboxEnv
from .mcp_adapter import MCPAdapter, record_external_mcp_call
from .procedural import (
    MAX_RECIPE_ATTRIBUTE_KEY_LENGTH,
    MAX_RECIPE_NODE_ATTRIBUTES,
    MAX_RECIPE_SOCKET_INDEX,
    PROCEDURAL_RECIPE_SCHEMA,
    RECIPE_SCHEMA_VERSION,
    ProceduralRecipe,
    RecipeError,
    RecipeLink,
    RecipeNode,
    canonical_recipe_json,
    normalize_recipe,
    recipe_hash,
    validate_recipe,
)
from .protocol import (
    MAX_SEED,
    SCHEMA_VERSION,
    ActionRequest,
    ActionResponse,
    ProtocolError,
    ToolSpec,
    get_tool_spec,
    request_fingerprint,
    tool_registry,
    validate_action_args,
)
from .reward import compute_reward, scorecard_quality
from .sculpt_metrics import sculpt_quality_metrics
from .trajectory import TrajectoryReader, TrajectoryWriter
from .version import TOOLBOX_VERSION, TOOLBOX_VERSION_INFO
from .workflows import capability_catalog, describe_workflow


def __getattr__(name: str):
    if name == "ToolboxMCPServer":
        from .mcp_server import ToolboxMCPServer

        return ToolboxMCPServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SCHEMA_VERSION",
    "MAX_SEED",
    "ActionRequest",
    "ActionResponse",
    "ProtocolError",
    "ToolSpec",
    "TrajectoryReader",
    "TrajectoryWriter",
    "ToolboxEnv",
    "MCPAdapter",
    "record_external_mcp_call",
    "PROCEDURAL_RECIPE_SCHEMA",
    "MAX_RECIPE_ATTRIBUTE_KEY_LENGTH",
    "MAX_RECIPE_NODE_ATTRIBUTES",
    "MAX_RECIPE_SOCKET_INDEX",
    "RECIPE_SCHEMA_VERSION",
    "ProceduralRecipe",
    "RecipeError",
    "RecipeLink",
    "RecipeNode",
    "canonical_recipe_json",
    "normalize_recipe",
    "recipe_hash",
    "validate_recipe",
    "ToolboxMCPServer",
    "EpisodeRecorder",
    "compute_reward",
    "get_tool_spec",
    "request_fingerprint",
    "validate_action_args",
    "scorecard_quality",
    "tool_registry",
    "sculpt_quality_metrics",
    "capability_catalog",
    "describe_workflow",
    "TOOLBOX_VERSION",
    "TOOLBOX_VERSION_INFO",
]
