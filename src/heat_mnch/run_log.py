from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_run_log(root: Path, label: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / "artifacts" / "run_logs" / f"{timestamp}_{label}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event_type": "run_initialized",
                    "label": label,
                    "timestamp": utc_now(),
                }
            )
        )
        handle.write("\n")
    return path


def append_run_event(
    log_path: Path,
    phase: str,
    dataset: str,
    year: int | None,
    command: str,
    decision: str,
    status: str,
) -> None:
    event = {
        "event_type": "run_event",
        "timestamp": utc_now(),
        "phase": phase,
        "dataset": dataset,
        "year": year,
        "command": command,
        "decision": decision,
        "status": status,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event))
        handle.write("\n")
