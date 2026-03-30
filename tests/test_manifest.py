from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from heat_mnch.manifest import build_manifest, detect_year


class ManifestBuildTest(unittest.TestCase):
    def test_build_manifest_counts_rows_and_writes_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "data" / "raw" / "sinasc").mkdir(parents=True)
            (root / "artifacts" / "manifests").mkdir(parents=True)
            csv_path = root / "data" / "raw" / "sinasc" / "sinasc_2020.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "value"])
                writer.writerow(["1", "a"])
                writer.writerow(["2", "b"])

            manifest = build_manifest(root, "raw", "sinasc")
            self.assertEqual(manifest["entry_count"], 1)
            self.assertEqual(manifest["entries"][0]["row_count"], 2)
            self.assertEqual(manifest["entries"][0]["year"], 2020)

            artifact_path = root / "artifacts" / "manifests" / "raw_sinasc.manifest.json"
            self.assertTrue(artifact_path.exists())
            saved = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["entry_count"], 1)

    def test_build_manifest_counts_parquet_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parquet_dir = root / "data" / "silver" / "climate" / "YEAR=2020"
            parquet_dir.mkdir(parents=True)
            parquet_path = parquet_dir / "climate_canonical_daily_2020.parquet"
            pd.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]}).to_parquet(parquet_path, index=False)

            manifest = build_manifest(root, "silver", "climate")
            self.assertEqual(manifest["entry_count"], 1)
            self.assertEqual(manifest["entries"][0]["row_count"], 3)
            self.assertEqual(manifest["entries"][0]["year"], 2020)

    def test_detect_year_supports_datasus_two_digit_patterns(self) -> None:
        self.assertEqual(detect_year(Path("DOFET24.dbc")), 2024)
        self.assertEqual(detect_year(Path("RDSP2401.dbc")), 2024)

    def test_detect_year_prefers_directory_year_before_filename_digits(self) -> None:
        self.assertEqual(detect_year(Path("data/raw/sih/2020/RDAC2001.dbc")), 2020)


if __name__ == "__main__":
    unittest.main()
