from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from pathlib import Path


YEAR_PATTERN = re.compile(r"(20[0-9]{2})")
YEARMONTH_PATTERN = re.compile(r"(\d{2})(\d{2})$")
TWODIGIT_YEAR_PATTERN = re.compile(r"(\d{2})$")


def detect_year(path: Path) -> int | None:
    for part in reversed(path.parts[:-1]):
        match = YEAR_PATTERN.search(part)
        if match:
            return int(match.group(1))
    stem = path.stem.upper()
    yearmonth_match = YEARMONTH_PATTERN.search(stem)
    if yearmonth_match:
        year_fragment = int(yearmonth_match.group(1))
        month_fragment = int(yearmonth_match.group(2))
        if 1 <= month_fragment <= 12:
            return 2000 + year_fragment
    year_match = TWODIGIT_YEAR_PATTERN.search(stem)
    if year_match:
        year_fragment = int(year_match.group(1))
        return 2000 + year_fragment if year_fragment <= 79 else 1900 + year_fragment
    filename_match = YEAR_PATTERN.search(path.name)
    if filename_match:
        return int(filename_match.group(1))
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_rows(path: Path) -> int | None:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if not suffixes:
        return None
    if suffixes[-1] == ".parquet":
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    if suffixes[-1] != ".csv" and suffixes[-2:] != [".csv", ".gz"]:
        return None
    opener = gzip.open if suffixes[-2:] == [".csv", ".gz"] else Path.open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def build_manifest(root: Path, layer: str, dataset: str) -> dict[str, object]:
    dataset_dir = root / "data" / layer / dataset
    entries: list[dict[str, object]] = []
    if dataset_dir.exists():
        for path in sorted(item for item in dataset_dir.rglob("*") if item.is_file() and item.name != ".DS_Store"):
            stats = path.stat()
            entries.append(
                {
                    "path": str(path.relative_to(root)),
                    "size_bytes": stats.st_size,
                    "sha256": sha256_file(path),
                    "modified_at": stats.st_mtime,
                    "row_count": count_rows(path),
                    "year": detect_year(path),
                }
            )
    manifest = {
        "layer": layer,
        "dataset": dataset,
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "entry_count": len(entries),
        "entries": entries,
    }
    output_path = root / "artifacts" / "manifests" / f"{layer}_{dataset}.manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest
