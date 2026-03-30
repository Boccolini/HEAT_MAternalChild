from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from heat_mnch.gold_municipality_context import (
    ALTITUDE_SOURCE_LABEL,
    ATLAS_SOURCE_LABEL,
    COASTAL_SOURCE_LABEL,
    URBAN_RURAL_SOURCE_LABEL,
    assemble_context_frame,
    build_gold_municipality_context,
    build_static_export,
    load_canonical_spine,
    validate_context_frame,
)


def _write_dummy_file(path: Path, content: str = "dummy\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class GoldMunicipalityContextBuildTest(unittest.TestCase):
    def _write_spine(self, root: Path) -> None:
        metadata_dir = root / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"COD_MUNICIPIO": "1100015", "NM_MUNICIPIO": "Alta Floresta D'Oeste", "UF": "RO"},
                {"COD_MUNICIPIO": "1100023", "NM_MUNICIPIO": "Ariquemes", "UF": "RO"},
            ]
        ).to_csv(metadata_dir / "era5_municipio_grid_lookup.csv", index=False)

    def _fake_source_payloads(self, root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict, dict, dict]:
        reference_dir = root / "artifacts" / "reference" / "municipality_context"
        urban_cache = _write_dummy_file(reference_dir / "urban_rural_source.zip")
        lookup_cache = reference_dir / "atlas_lookup.csv"
        pd.DataFrame(
            [
                {"atlas_id": 1, "MUNICIPALITY_CODE_IBGE7": "1100015"},
            ]
        ).to_csv(lookup_cache, index=False)
        gini_cache = _write_dummy_file(reference_dir / "atlas_gini_2010.csv", "MUNICIPALITY_CODE_IBGE7,GINI_RAW\n1100015,0.55\n")
        idh_cache = _write_dummy_file(reference_dir / "atlas_idh_2010.csv", "MUNICIPALITY_CODE_IBGE7,IDH_RAW\n1100015,0.71\n")
        coastal_cache = _write_dummy_file(reference_dir / "coastal_2021.xls")
        altitude_cache = _write_dummy_file(reference_dir / "altitude.csv", "geocode,datetime,Altitude-mean\n110001,2000,250\n110002,2000,125\n")

        urban_rural_frame = pd.DataFrame(
            [
                {"MUNICIPALITY_CODE_IBGE7": "1100015", "URBAN_POP": 80, "RURAL_POP": 20, "SOURCE_TOTAL_POP_VALIDATION": 100},
                {"MUNICIPALITY_CODE_IBGE7": "1100023", "URBAN_POP": 50, "RURAL_POP": 50, "SOURCE_TOTAL_POP_VALIDATION": 100},
            ]
        )
        gini_frame = pd.DataFrame([{"MUNICIPALITY_CODE_IBGE7": "1100015", "GINI_RAW": 0.55}])
        idh_frame = pd.DataFrame([{"MUNICIPALITY_CODE_IBGE7": "1100015", "IDH_RAW": 0.71}])
        coastal_frame = pd.DataFrame([{"MUNICIPALITY_CODE_IBGE7": "1100015", "COASTAL_FLAG": 1}])
        altitude_frame = pd.DataFrame(
            [
                {"municipality_code": "110001", "ALTITUDE_MEAN_M": 250.0},
                {"municipality_code": "110002", "ALTITUDE_MEAN_M": 125.0},
            ]
        )
        urban_metadata = {
            "source": URBAN_RURAL_SOURCE_LABEL,
            "source_year": 2022,
            "source_url": "https://example.org/urban_rural",
            "cache_path": urban_cache,
            "reconciliation_nonzero_count": 0,
            "reconciliation_max_abs_diff": 0.0,
        }
        atlas_metadata = {
            "source": ATLAS_SOURCE_LABEL,
            "source_year": 2010,
            "lookup_cache_path": lookup_cache,
            "gini_cache_path": gini_cache,
            "idh_cache_path": idh_cache,
            "lookup_code_count": 1,
            "gini_observed_count": 1,
            "idh_observed_count": 1,
        }
        coastal_metadata = {
            "source": COASTAL_SOURCE_LABEL,
            "source_url": "https://example.org/coastal",
            "cache_path": coastal_cache,
            "coastal_count": 1,
        }
        altitude_metadata = {
            "source": ALTITUDE_SOURCE_LABEL,
            "source_url": "https://example.org/altitude",
            "cache_path": altitude_cache,
            "observed_code_count": 2,
            "source_year_distribution": {"2000": 2},
        }
        return (
            urban_rural_frame,
            gini_frame,
            idh_frame,
            coastal_frame,
            altitude_frame,
            urban_metadata,
            atlas_metadata,
            coastal_metadata,
            altitude_metadata,
        )

    def test_build_gold_municipality_context_writes_outputs_and_flags_missing_atlas_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_root = root / "runtime"
            log_path = root / "artifacts" / "run_logs" / "municipality_context_test.jsonl"
            self._write_spine(root)
            (
                urban_rural_frame,
                gini_frame,
                idh_frame,
                coastal_frame,
                altitude_frame,
                urban_metadata,
                atlas_metadata,
                coastal_metadata,
                altitude_metadata,
            ) = self._fake_source_payloads(root)

            with (
                patch("heat_mnch.gold_municipality_context.load_urban_rural_context", return_value=(urban_rural_frame, urban_metadata)),
                patch("heat_mnch.gold_municipality_context.load_atlas_context", return_value=(gini_frame, idh_frame, atlas_metadata)),
                patch("heat_mnch.gold_municipality_context.load_coastal_context", return_value=(coastal_frame, coastal_metadata)),
                patch("heat_mnch.gold_municipality_context.load_altitude_context", return_value=(altitude_frame, altitude_metadata)),
            ):
                report = build_gold_municipality_context(
                    root=root,
                    runtime_root=runtime_root,
                    start_year=2020,
                    end_year=2024,
                    log_path=log_path,
                    command="test-build-gold-municipality-context",
                    force=False,
                )

            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["municipalities_covered"], 2)
            self.assertEqual(report["municipality_years_covered"], 10)

            partition_path = root / "data" / "gold" / "gold_municipality_context" / "YEAR=2020" / "gold_municipality_context_2020.parquet"
            self.assertTrue(partition_path.exists())
            output_2020 = pd.read_parquet(partition_path)
            self.assertEqual(output_2020.shape[0], 2)
            self.assertEqual(output_2020.columns.tolist()[0:3], ["municipality_code", "MUNICIPALITY_CODE_IBGE7", "year"])

            row_missing_atlas = output_2020.loc[output_2020["municipality_code"].eq("110002")].iloc[0]
            self.assertTrue(pd.isna(row_missing_atlas["GINI_RAW"]))
            self.assertEqual(row_missing_atlas["GINI_EXPANSION_METHOD"], "missing_source")
            self.assertIn("GINI", row_missing_atlas["JOIN_STATUS_FLAG"])
            self.assertIn("IDH", row_missing_atlas["JOIN_STATUS_FLAG"])
            self.assertEqual(row_missing_atlas["COASTAL_FLAG"], 0)

            row_observed_2022 = pd.read_parquet(
                root / "data" / "gold" / "gold_municipality_context" / "YEAR=2022" / "gold_municipality_context_2022.parquet"
            ).loc[lambda frame: frame["municipality_code"].eq("110001")].iloc[0]
            self.assertEqual(row_observed_2022["URBAN_RURAL_EXPANSION_METHOD"], "observed_source_year")

            static_export = pd.read_parquet(root / "data" / "gold" / "gold_municipality_context_static" / "gold_municipality_context_static.parquet")
            self.assertEqual(static_export.shape[0], 2)
            self.assertNotIn("year", static_export.columns)
            self.assertEqual(static_export.loc[static_export["municipality_code"].eq("110001"), "URBAN_RURAL_EXPANSION_METHOD"].iloc[0], "piecewise_constant_from_2022")

            manifest_path = root / "artifacts" / "manifests" / "municipality_context_manifest.json"
            mini_report_path = root / "artifacts" / "status" / "municipality_context_mini_report.md"
            self.assertTrue(manifest_path.exists())
            self.assertTrue(mini_report_path.exists())

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["municipality_years_covered"], 10)
            self.assertEqual(manifest["internal_validation"]["duplicates"], 0)
            self.assertEqual(manifest["external_validation"]["atlas_missing_from_canonical_spine_count"], 1)

    def test_assemble_context_frame_and_validation_detect_duplicate_and_missing_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_spine(root)
            spine = load_canonical_spine(root)
            context_frame = assemble_context_frame(
                spine=spine,
                years=[2020, 2021],
                urban_rural_frame=pd.DataFrame(
                    [
                        {"MUNICIPALITY_CODE_IBGE7": "1100015", "URBAN_POP": 80, "RURAL_POP": 20, "SOURCE_TOTAL_POP_VALIDATION": 100},
                        {"MUNICIPALITY_CODE_IBGE7": "1100023", "URBAN_POP": 50, "RURAL_POP": 50, "SOURCE_TOTAL_POP_VALIDATION": 100},
                    ]
                ),
                gini_frame=pd.DataFrame([{"MUNICIPALITY_CODE_IBGE7": "1100015", "GINI_RAW": 0.55}]),
                idh_frame=pd.DataFrame([{"MUNICIPALITY_CODE_IBGE7": "1100015", "IDH_RAW": 0.71}]),
                altitude_frame=pd.DataFrame(
                    [
                        {"municipality_code": "110001", "ALTITUDE_MEAN_M": 250.0},
                        {"municipality_code": "110002", "ALTITUDE_MEAN_M": 125.0},
                    ]
                ),
                coastal_frame=pd.DataFrame([{"MUNICIPALITY_CODE_IBGE7": "1100015", "COASTAL_FLAG": 1}]),
            )
            broken = context_frame.loc[~((context_frame["municipality_code"].eq("110002")) & (context_frame["year"].eq(2021)))].copy()
            broken = pd.concat([broken, broken.iloc[[0]]], ignore_index=True)

            validation = validate_context_frame(broken, spine, [2020, 2021])
            self.assertEqual(validation["duplicates"], 1)
            self.assertEqual(validation["missing_municipality_years"], 1)
            self.assertFalse(validation["all_checks_passed"])

    def test_build_static_export_uses_first_target_year_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_spine(root)
            spine = load_canonical_spine(root)
            context_frame = assemble_context_frame(
                spine=spine,
                years=[2020, 2022],
                urban_rural_frame=pd.DataFrame(
                    [
                        {"MUNICIPALITY_CODE_IBGE7": "1100015", "URBAN_POP": 80, "RURAL_POP": 20, "SOURCE_TOTAL_POP_VALIDATION": 100},
                        {"MUNICIPALITY_CODE_IBGE7": "1100023", "URBAN_POP": 50, "RURAL_POP": 50, "SOURCE_TOTAL_POP_VALIDATION": 100},
                    ]
                ),
                gini_frame=pd.DataFrame([{"MUNICIPALITY_CODE_IBGE7": "1100015", "GINI_RAW": 0.55}]),
                idh_frame=pd.DataFrame([{"MUNICIPALITY_CODE_IBGE7": "1100015", "IDH_RAW": 0.71}]),
                altitude_frame=pd.DataFrame(
                    [
                        {"municipality_code": "110001", "ALTITUDE_MEAN_M": 250.0},
                        {"municipality_code": "110002", "ALTITUDE_MEAN_M": 125.0},
                    ]
                ),
                coastal_frame=pd.DataFrame([{"MUNICIPALITY_CODE_IBGE7": "1100015", "COASTAL_FLAG": 1}]),
            )
            static_frame = build_static_export(context_frame, [2020, 2022])
            self.assertEqual(static_frame.shape[0], 2)
            self.assertEqual(static_frame.loc[static_frame["municipality_code"].eq("110001"), "URBAN_RURAL_EXPANSION_METHOD"].iloc[0], "piecewise_constant_from_2022")


if __name__ == "__main__":
    unittest.main()
