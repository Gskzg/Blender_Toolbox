#!/usr/bin/env python3
"""Export replayable, trusted trajectory events as offline-RL JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the script usable as a direct executable inside the skill bundle. The
# bundled ``trajectory`` package is a sibling of this file's directory.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trajectory.dataset import export_training_events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", type=Path, help="Episode directories containing manifest.json")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Output JSONL path")
    parser.add_argument("--allow-truncated", action="store_true", help="Include episodes not marked complete")
    parser.add_argument("--allow-untrusted", action="store_true", help="Include run_python/untrusted events")
    parser.add_argument("--allow-non-replayable", action="store_true", help="Include actions without deterministic replay")
    parser.add_argument("--include-errors", action="store_true", help="Include failed action events")
    args = parser.parse_args()
    count = export_training_events(
        args.episodes,
        args.output,
        require_replayable=not args.allow_non_replayable,
        include_truncated=args.allow_truncated,
        include_untrusted=args.allow_untrusted,
        include_errors=args.include_errors,
    )
    print(f"exported {count} events to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
