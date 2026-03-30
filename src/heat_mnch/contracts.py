from __future__ import annotations

import csv
from pathlib import Path


def read_contract_columns(schema_path: Path) -> list[str]:
    with schema_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row["column_name"] for row in reader]


def read_csv_header(data_path: Path) -> list[str]:
    with data_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def validate_csv_contract(data_path: Path, schema_path: Path) -> dict[str, object]:
    expected = read_contract_columns(schema_path)
    observed = read_csv_header(data_path)
    return {
        "data_path": str(data_path),
        "schema_path": str(schema_path),
        "valid": observed == expected,
        "expected_columns": expected,
        "observed_columns": observed,
        "missing_columns": [column for column in expected if column not in observed],
        "unexpected_columns": [column for column in observed if column not in expected],
    }
