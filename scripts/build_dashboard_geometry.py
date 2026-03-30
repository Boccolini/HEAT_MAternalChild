from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heat_mnch.dashboard_geometry import build_dashboard_geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Observa Infancia dashboard geometry assets.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tolerance", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    result = build_dashboard_geometry(
        root=root,
        source_zip=Path(args.source),
        output_dir=Path(args.output_dir),
        tolerance=args.tolerance,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
