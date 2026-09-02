#!/usr/bin/env bash
# Start the bundled stdio MCP adapter for an already-running executor.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOLBOX_RUNTIME="${TOOLBOX_RUNTIME_DIR:-$SCRIPT_DIR}"
SOCKET="${BLENDER_TOOLBOX_SOCKET:-/tmp/blender_toolbox.sock}"
TRAJECTORY_DIR="${1:?usage: start_mcp_server.sh <trajectory-dir> [task-id]}"
TASK_ID="${2:-mcp_episode}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

case "$TRAJECTORY_DIR" in
  /*) ;;
  *) TRAJECTORY_DIR="$(pwd)/$TRAJECTORY_DIR" ;;
esac

if [ ! -f "$TOOLBOX_RUNTIME/blender_toolbox/mcp_server.py" ]; then
  echo "Bundled Toolbox runtime not found: $TOOLBOX_RUNTIME/blender_toolbox/mcp_server.py" >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$TOOLBOX_RUNTIME${PYTHONPATH:+:$PYTHONPATH}"
cd "$TOOLBOX_RUNTIME"
exec "$PYTHON_BIN" -m blender_toolbox.mcp_server \
  --socket "$SOCKET" \
  --trajectory-dir "$TRAJECTORY_DIR" \
  --task-id "$TASK_ID"
