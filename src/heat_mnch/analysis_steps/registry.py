from __future__ import annotations

from pathlib import Path


ANALYSIS_STEP0_RELATIVE_ROOT = Path("artifacts/analysis/step0")
ANALYSIS_STEP1_RELATIVE_ROOT = Path("artifacts/analysis/step1")
ANALYSIS_STEP2_RELATIVE_ROOT = Path("artifacts/analysis/step2")
ANALYSIS_STEP3_RELATIVE_ROOT = Path("artifacts/analysis/step3")
ANALYSIS_STEP4_RELATIVE_ROOT = Path("artifacts/analysis/step4")
ANALYSIS_STEP5_RELATIVE_ROOT = Path("artifacts/analysis/step5")
ANALYSIS_STEP6_RELATIVE_ROOT = Path("artifacts/analysis/step6")
ANALYSIS_STEP7_RELATIVE_ROOT = Path("artifacts/analysis/step7")
ANALYSIS_STEP7_RUNTIME_RELATIVE_ROOT = Path("artifacts/analysis/step7/_runtime")
ANALYSIS_SHARED_MARTS_RELATIVE_ROOT = Path("artifacts/analysis/shared_marts")
INCREMENTAL_PUBLICATION_RELATIVE_ROOT = Path("reports/publications/incremental_analysis_round")


def analysis_step0_root(root: Path) -> Path:
    return root / ANALYSIS_STEP0_RELATIVE_ROOT


def analysis_step1_root(root: Path) -> Path:
    return root / ANALYSIS_STEP1_RELATIVE_ROOT


def analysis_step2_root(root: Path) -> Path:
    return root / ANALYSIS_STEP2_RELATIVE_ROOT


def analysis_step3_root(root: Path) -> Path:
    return root / ANALYSIS_STEP3_RELATIVE_ROOT


def analysis_step4_root(root: Path) -> Path:
    return root / ANALYSIS_STEP4_RELATIVE_ROOT


def analysis_step5_root(root: Path) -> Path:
    return root / ANALYSIS_STEP5_RELATIVE_ROOT


def analysis_step6_root(root: Path) -> Path:
    return root / ANALYSIS_STEP6_RELATIVE_ROOT


def analysis_step7_root(root: Path) -> Path:
    return root / ANALYSIS_STEP7_RELATIVE_ROOT


def analysis_step7_runtime_root(root: Path) -> Path:
    return root / ANALYSIS_STEP7_RUNTIME_RELATIVE_ROOT


def analysis_shared_marts_root(root: Path) -> Path:
    return root / ANALYSIS_SHARED_MARTS_RELATIVE_ROOT


def incremental_publication_root(root: Path) -> Path:
    return root / INCREMENTAL_PUBLICATION_RELATIVE_ROOT


def ensure_step0_directory_structure(root: Path) -> dict[str, str]:
    directories = {
        "artifacts_root": analysis_step0_root(root),
        "publication_root": incremental_publication_root(root),
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return {name: str(path.relative_to(root)) for name, path in directories.items()}


def ensure_step1_directory_structure(root: Path) -> dict[str, str]:
    directories = {
        "artifacts_root": analysis_step1_root(root),
        "shared_marts_root": analysis_shared_marts_root(root),
        "publication_root": incremental_publication_root(root),
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return {name: str(path.relative_to(root)) for name, path in directories.items()}


def ensure_step2_directory_structure(root: Path) -> dict[str, str]:
    directories = {
        "artifacts_root": analysis_step2_root(root),
        "shared_marts_root": analysis_shared_marts_root(root),
        "publication_root": incremental_publication_root(root),
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return {name: str(path.relative_to(root)) for name, path in directories.items()}


def ensure_step3_directory_structure(root: Path) -> dict[str, str]:
    directories = {
        "artifacts_root": analysis_step3_root(root),
        "shared_marts_root": analysis_shared_marts_root(root),
        "publication_root": incremental_publication_root(root),
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return {name: str(path.relative_to(root)) for name, path in directories.items()}


def ensure_step4_directory_structure(root: Path) -> dict[str, str]:
    directories = {
        "artifacts_root": analysis_step4_root(root),
        "shared_marts_root": analysis_shared_marts_root(root),
        "publication_root": incremental_publication_root(root),
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return {name: str(path.relative_to(root)) for name, path in directories.items()}


def ensure_step5_directory_structure(root: Path) -> dict[str, str]:
    directories = {
        "artifacts_root": analysis_step5_root(root),
        "shared_marts_root": analysis_shared_marts_root(root),
        "publication_root": incremental_publication_root(root),
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return {name: str(path.relative_to(root)) for name, path in directories.items()}


def ensure_step6_directory_structure(root: Path) -> dict[str, str]:
    directories = {
        "artifacts_root": analysis_step6_root(root),
        "shared_marts_root": analysis_shared_marts_root(root),
        "publication_root": incremental_publication_root(root),
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return {name: str(path.relative_to(root)) for name, path in directories.items()}


def ensure_step7_directory_structure(root: Path) -> dict[str, str]:
    directories = {
        "artifacts_root": analysis_step7_root(root),
        "runtime_root": analysis_step7_runtime_root(root),
        "shared_marts_root": analysis_shared_marts_root(root),
        "publication_root": incremental_publication_root(root),
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return {name: str(path.relative_to(root)) for name, path in directories.items()}
