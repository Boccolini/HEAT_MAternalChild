from __future__ import annotations

from pathlib import Path


TRACK_B_RELATIVE_ROOT = Path("artifacts/track_b")
TRACK_B_EVENT_LEVEL_RELATIVE_ROOT = TRACK_B_RELATIVE_ROOT / "event_level"
TRACK_B_MODELS_RELATIVE_ROOT = TRACK_B_RELATIVE_ROOT / "models"
TRACK_B_BURDEN_RELATIVE_ROOT = TRACK_B_RELATIVE_ROOT / "burden"
TRACK_B_INEQUITY_RELATIVE_ROOT = TRACK_B_RELATIVE_ROOT / "inequity"
TRACK_B_ROBUSTNESS_RELATIVE_ROOT = TRACK_B_RELATIVE_ROOT / "robustness"
TRACK_B_PUBLICATION_RELATIVE_ROOT = Path("reports/publications/track_b_lanplh_flagship_perinatal")


def track_b_root(root: Path) -> Path:
    return root / TRACK_B_RELATIVE_ROOT


def track_b_event_level_root(root: Path) -> Path:
    return root / TRACK_B_EVENT_LEVEL_RELATIVE_ROOT


def track_b_models_root(root: Path) -> Path:
    return root / TRACK_B_MODELS_RELATIVE_ROOT


def track_b_burden_root(root: Path) -> Path:
    return root / TRACK_B_BURDEN_RELATIVE_ROOT


def track_b_inequity_root(root: Path) -> Path:
    return root / TRACK_B_INEQUITY_RELATIVE_ROOT


def track_b_robustness_root(root: Path) -> Path:
    return root / TRACK_B_ROBUSTNESS_RELATIVE_ROOT


def track_b_publication_root(root: Path) -> Path:
    return root / TRACK_B_PUBLICATION_RELATIVE_ROOT


def ensure_track_b_directory_structure(root: Path) -> dict[str, str]:
    directories = {
        "track_b_root": track_b_root(root),
        "event_level_root": track_b_event_level_root(root),
        "models_root": track_b_models_root(root),
        "burden_root": track_b_burden_root(root),
        "inequity_root": track_b_inequity_root(root),
        "robustness_root": track_b_robustness_root(root),
        "publication_root": track_b_publication_root(root),
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return {name: str(path.relative_to(root)) for name, path in directories.items()}
