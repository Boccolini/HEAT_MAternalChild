from __future__ import annotations

from pathlib import Path


def _pending(stage: str, root: Path, message: str) -> dict[str, object]:
    return {
        "status": "PENDING_IMPLEMENTATION",
        "stage": stage,
        "message": message,
        "root": str(root),
    }


def preterm_stage_not_implemented(root: Path) -> dict[str, object]:
    return _pending(
        "track_b_stage_b2_preterm_models",
        root,
        "Track B preterm flagship models are scaffolded but not implemented yet.",
    )


def stillbirth_stage_not_implemented(root: Path) -> dict[str, object]:
    return _pending(
        "track_b_stage_b3_stillbirth_models",
        root,
        "Track B stillbirth flagship models are scaffolded but not implemented yet.",
    )


def neonatal_stage_not_implemented(root: Path) -> dict[str, object]:
    return _pending(
        "track_b_stage_b4_neonatal_models",
        root,
        "Track B neonatal acute models are scaffolded but not implemented yet.",
    )


def burden_stage_not_implemented(root: Path) -> dict[str, object]:
    return _pending(
        "track_b_stage_b5_burden",
        root,
        "Track B burden estimation is scaffolded but not implemented yet.",
    )


def inequity_stage_not_implemented(root: Path) -> dict[str, object]:
    return _pending(
        "track_b_stage_b6_inequity",
        root,
        "Track B inequity and vulnerability analyses are scaffolded but not implemented yet.",
    )


def robustness_stage_not_implemented(root: Path) -> dict[str, object]:
    return _pending(
        "track_b_stage_b7_robustness",
        root,
        "Track B robustness package is scaffolded but not implemented yet.",
    )


def paper_package_stage_not_implemented(root: Path) -> dict[str, object]:
    return _pending(
        "track_b_stage_b8_paper_package",
        root,
        "Track B paper package is scaffolded but not implemented yet.",
    )
