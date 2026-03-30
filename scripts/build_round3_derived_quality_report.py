from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC = ROOT / "src"
cleaned_path: list[str] = []
for entry in sys.path:
    candidate = Path(entry or ".").resolve()
    if candidate == SCRIPT_DIR:
        continue
    cleaned_path.append(entry)
sys.path[:] = cleaned_path
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from heat_mnch.round3_derived_quality_report import build_round3_derived_quality_report


def main() -> None:
    outputs = build_round3_derived_quality_report(ROOT)
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
