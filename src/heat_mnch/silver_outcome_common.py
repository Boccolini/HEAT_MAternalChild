from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .manifest import count_rows


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def verify_parquet_file(path: Path, expected_rows: int, expected_columns: list[str]) -> None:
    parquet_file = pq.ParquetFile(path)
    if int(parquet_file.metadata.num_rows) != int(expected_rows):
        raise IOError(f"Unexpected parquet row count for {path}: {parquet_file.metadata.num_rows} != {expected_rows}")
    schema_names = parquet_file.schema_arrow.names
    if list(schema_names) != list(expected_columns):
        raise IOError(f"Unexpected parquet schema for {path}: {schema_names} != {expected_columns}")
    sample = pd.read_parquet(path, columns=expected_columns[: min(4, len(expected_columns))])
    if sample is None:
        raise IOError(f"Unable to read parquet verification sample for {path}")


def write_parquet_chunks(frames: list[pd.DataFrame], path: Path) -> int:
    if path.parent.exists():
        shutil.rmtree(path.parent)
    ensure_parent(path)
    writer: pq.ParquetWriter | None = None
    expected_columns: list[str] | None = None
    total_rows = 0
    try:
        for frame in frames:
            if frame.empty:
                continue
            total_rows += len(frame)
            expected_columns = list(frame.columns)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if expected_columns is None:
        raise ValueError(f"No data was written to parquet path {path}")
    verify_parquet_file(path, total_rows, expected_columns)
    return total_rows


class ParquetChunkWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.writer: pq.ParquetWriter | None = None
        self.expected_columns: list[str] | None = None
        self.total_rows = 0
        if path.parent.exists():
            shutil.rmtree(path.parent)
        ensure_parent(path)

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        self.total_rows += len(frame)
        if self.expected_columns is None:
            self.expected_columns = list(frame.columns)
        elif list(frame.columns) != self.expected_columns:
            raise ValueError(f"Chunk parquet schema drift detected for {self.path}")
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="snappy")
        self.writer.write_table(table)

    def close(self) -> tuple[int, list[str]]:
        if self.writer is not None:
            self.writer.close()
        if self.expected_columns is None:
            raise ValueError(f"No chunks were written to {self.path}")
        verify_parquet_file(self.path, self.total_rows, self.expected_columns)
        return self.total_rows, self.expected_columns


def sync_verified_parquet(staging_file: Path, final_file: Path, expected_rows: int, expected_columns: list[str]) -> None:
    verify_parquet_file(staging_file, expected_rows, expected_columns)
    temp_dir = final_file.parent.parent / f".{final_file.parent.name}.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / final_file.name
    shutil.copy2(staging_file, temp_file)
    verify_parquet_file(temp_file, expected_rows, expected_columns)
    if final_file.parent.exists():
        shutil.rmtree(final_file.parent)
    temp_dir.rename(final_file.parent)


def official_total_from_bronze(root: Path, dataset: str, year: int) -> int:
    manifest_path = root / "artifacts" / "manifests" / f"bronze_{dataset}.manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_path = f"data/bronze/{dataset}/{year}/{dataset}_{year}_source_preserving.csv.gz"
        entry = next((item for item in manifest.get("entries", []) if item.get("path") == expected_path), None)
        if entry and entry.get("row_count") is not None:
            return int(entry["row_count"])
    fallback_path = root / "data" / "bronze" / dataset / str(year) / f"{dataset}_{year}_source_preserving.csv.gz"
    row_count = count_rows(fallback_path)
    if row_count is None:
        raise ValueError(f"Unable to determine official bronze total for {dataset} {year}")
    return int(row_count)
