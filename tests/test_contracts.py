from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from heat_mnch.contracts import validate_csv_contract


class ContractValidationTest(unittest.TestCase):
    def test_validate_csv_contract_accepts_exact_header_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            schema_path = root / "schema.csv"
            data_path = root / "data.csv"

            with schema_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["column_name", "data_type", "required", "description"])
                writer.writerow(["A", "string", "true", "First column"])
                writer.writerow(["B", "integer", "true", "Second column"])

            with data_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["A", "B"])
                writer.writerow(["x", "1"])

            result = validate_csv_contract(data_path, schema_path)
            self.assertTrue(result["valid"])

    def test_validate_csv_contract_rejects_wrong_header_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            schema_path = root / "schema.csv"
            data_path = root / "data.csv"

            with schema_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["column_name", "data_type", "required", "description"])
                writer.writerow(["A", "string", "true", "First column"])
                writer.writerow(["B", "integer", "true", "Second column"])

            with data_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["B", "A"])
                writer.writerow(["1", "x"])

            result = validate_csv_contract(data_path, schema_path)
            self.assertFalse(result["valid"])


if __name__ == "__main__":
    unittest.main()
