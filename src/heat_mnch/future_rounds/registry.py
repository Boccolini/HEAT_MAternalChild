from __future__ import annotations

from pathlib import Path


LRH_AMERICAS_ROUND2_RELATIVE_ROOT = Path("artifacts/analysis/lrh_americas_round2_validation")


def lrh_americas_round2_root(root: Path) -> Path:
    return root / LRH_AMERICAS_ROUND2_RELATIVE_ROOT


def ensure_lrh_americas_round2_directory_structure(root: Path) -> dict[str, Path]:
    base = lrh_americas_round2_root(root)
    directories = {
        "root": base,
        "runtime_root": base / "_runtime",
        "logs_root": base / "_runtime" / "logs",
        "freeze_root": base / "freeze",
        "phase1_root": base / "phase1_temporal_validation",
        "phase2_root": base / "phase2_dlnm",
        "phase2_input_root": base / "phase2_dlnm" / "input",
        "phase3_root": base / "phase3_case_crossover",
        "phase3_input_root": base / "phase3_case_crossover" / "input",
        "phase4_root": base / "phase4_resynthesis",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories
