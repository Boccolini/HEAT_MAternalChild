from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heat_mnch.dashboard_build import build_dashboard_marts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Observa Infancia dashboard marts.")
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = build_dashboard_marts(
        root=root,
        start_year=args.start_year,
        end_year=args.end_year,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
