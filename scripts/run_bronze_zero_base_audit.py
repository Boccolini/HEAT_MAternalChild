import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heat_mnch.bronze_audit import build_parser, run_bronze_zero_base_audit


if __name__ == "__main__":
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    payload = run_bronze_zero_base_audit(
        root,
        include_datasets=args.datasets,
        exclude_datasets=args.exclude_datasets,
    )
    summary = {
        "generated_at": payload["generated_at"],
        "scope": payload["scope"],
        "matrix_path": payload["matrix_path"],
        "report_path": payload["report_path"],
        "promotion_queue_path": payload["promotion_queue_path"],
        "details_path": payload["details_path"],
        "ready_dataset_years": payload["ready_dataset_years"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
