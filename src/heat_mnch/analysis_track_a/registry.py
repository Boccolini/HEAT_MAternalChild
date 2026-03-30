from __future__ import annotations

from pathlib import Path


TRACK_A_RELATIVE_ROOT = Path("artifacts/track_a")
TRACK_A_METADATA_RELATIVE_ROOT = TRACK_A_RELATIVE_ROOT / "metadata"
TRACK_A_PUBLICATION_RELATIVE_ROOT = Path("reports/publications/track_a_framework_validation")


def track_a_root(root: Path) -> Path:
    return root / TRACK_A_RELATIVE_ROOT


def track_a_metadata_root(root: Path) -> Path:
    return root / TRACK_A_METADATA_RELATIVE_ROOT


def track_a_publication_root(root: Path) -> Path:
    return root / TRACK_A_PUBLICATION_RELATIVE_ROOT


def ensure_track_a_directory_structure(root: Path) -> dict[str, str]:
    directories = {
        "track_a_root": track_a_root(root),
        "metadata_root": track_a_metadata_root(root),
        "publication_root": track_a_publication_root(root),
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return {name: str(path.relative_to(root)) for name, path in directories.items()}
