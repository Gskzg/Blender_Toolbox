#!/usr/bin/env bash
# Start the bundled Blender Toolbox executor.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOLBOX_RUNTIME="${TOOLBOX_RUNTIME_DIR:-$SCRIPT_DIR}"
BLENDER_BIN="${BLENDER_BIN:-/Applications/Blender.app/Contents/MacOS/Blender}"
SOCKET="${BLENDER_TOOLBOX_SOCKET:-/tmp/blender_toolbox.sock}"

if [ ! -x "$BLENDER_BIN" ]; then
  echo "Blender executable not found: $BLENDER_BIN" >&2
  exit 1
fi
if [ ! -f "$TOOLBOX_RUNTIME/blender_toolbox/addon.py" ]; then
  echo "Bundled Toolbox runtime not found: $TOOLBOX_RUNTIME/blender_toolbox/addon.py" >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1

exec "$BLENDER_BIN" --background --factory-startup \
  --python "$TOOLBOX_RUNTIME/blender_toolbox/addon.py" -- \
  --socket "$SOCKET"
