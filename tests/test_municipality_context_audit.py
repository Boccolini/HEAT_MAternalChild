from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from heat_mnch.cli import main
from heat_mnch.municipality_context_audit import (
    EXPECTED_YEARS,
    audit_gold_municipality_context,
    coverage_by_year,
    load_context_municipality_level,
    load_context_panel,
)


REQUIRED_PANEL_COLUMNS = [
    "municipality_code",
    "MUNICIPALITY_CODE_IBGE7",
    "year",
    "URBAN_POP",
    "RURAL_POP",
    "PROP_URBAN",
    "PROP_RURAL",
    "GINI_RAW",
    "GINI_SOURCE_YEAR",
    "GINI_EXPANDED",
    "GINI_EXPANSION_METHOD",
    "IDH_RAW",
    "IDH_SOURCE_YEAR",
    "IDH_EXPANDED",
    "IDH_EXPANSION_METHOD",
    "ALTITUDE_MEAN_M",
    "COASTAL_FLAG",
    "URBAN_RURAL_SOURCE",
    "URBAN_RURAL_SOURCE_YEAR",
    "URBAN_RURAL_EXPANSION_METHOD",
    "ALTITUDE_SOURCE",
    "COASTAL_SOURCE",
    "STATIC_CONTEXT_FLAG",
    "MUNICIPAL_BOUNDARY_REFERENCE_YEAR",
    "JOIN_STATUS_FLAG",
]


def _panel_row(
    municipality_code: str,
    municipality_code_ibge7: str,
    year: int,
    *,
    urban_pop: int,
    rural_pop: int,
    gini_raw: float | None,
    idh_raw: float | None,
    altitude_mean_m: float,
    coastal_flag: int,
) -> dict[str, object]:
    total = urban_pop + rural_pop
    return {
        "municipality_code": municipality_code,
        "MUNICIPALITY_CODE_IBGE7": municipality_code_ibge7,
        "year": year,
        "URBAN_POP": urban_pop,
        "RURAL_POP": rural_pop,
        "PROP_URBAN": urban_pop / total,
        "PROP_RURAL": rural_pop / total,
        "GINI_RAW": gini_raw,
        "GINI_SOURCE_YEAR": 2010 if gini_raw is not None else pd.NA,
        "GINI_EXPANDED": gini_raw,
        "GINI_EXPANSION_METHOD": "piecewise_constant_from_2010" if gini_raw is not None else "missing_source",
        "IDH_RAW": idh_raw,
        "IDH_SOURCE_YEAR": 2010 if idh_raw is not None else pd.NA,
        "IDH_EXPANDED": idh_raw,
        "IDH_EXPANSION_METHOD": "piecewise_constant_from_2010" if idh_raw is not None else "missing_source",
        "ALTITUDE_MEAN_M": altitude_mean_m,
        "COASTAL_FLAG": coastal_flag,
        "URBAN_RURAL_SOURCE": "IBGE_CENSO_2022_SETOR_AGREGADO_FTP",
        "URBAN_RURAL_SOURCE_YEAR": 2022,
        "URBAN_RURAL_EXPANSION_METHOD": "observed_source_year" if year == 2022 else "piecewise_constant_from_2022",
        "ALTITUDE_SOURCE": "OSF_ALTITUDE",
        "COASTAL_SOURCE": "IBGE_COASTAL",
        "STATIC_CONTEXT_FLAG": 1,
        "MUNICIPAL_BOUNDARY_REFERENCE_YEAR": 2022,
        "JOIN_STATUS_FLAG": "OK" if gini_raw is not None and idh_raw is not None else "MISSING_GINI|IDH",
    }


class MunicipalityContextAuditTest(unittest.TestCase):
    def _write_manifest(self, root: Path) -> None:
        manifest_path = root / "artifacts" / "manifests" / "municipality_context_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps({"dataset": "gold_municipality_context", "municipalities_covered": 2, "municipality_years_covered": 10}),
            encoding="utf-8",
        )

    def _write_panel(self, root: Path, *, missing_structural: bool = True) -> pd.DataFrame:
        rows = []
        for year in EXPECTED_YEARS:
            rows.append(
                _panel_row(
                    "110001",
                    "1100015",
                    year,
                    urban_pop=80,
                    rural_pop=20,
                    gini_raw=0.55,
                    idh_raw=0.71,
                    altitude_mean_m=250.0,
                    coastal_flag=1,
                )
            )
            rows.append(
                _panel_row(
                    "110002",
                    "1100023",
                    year,
                    urban_pop=50,
                    rural_pop=50,
                    gini_raw=None if missing_structural else 0.53,
                    idh_raw=None if missing_structural else 0.69,
                    altitude_mean_m=125.0,
                    coastal_flag=0,
                )
            )
        frame = pd.DataFrame(rows, columns=REQUIRED_PANEL_COLUMNS)
        for year, group in frame.groupby("year", sort=True):
            path = root / "data" / "gold" / "gold_municipality_context" / f"YEAR={year}" / f"gold_municipality_context_{year}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            group.to_parquet(path, index=False)
        self._write_manifest(root)
        return frame

    def test_load_context_panel_reads_all_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_panel(root)
            panel = load_context_panel(root)
            self.assertEqual(panel.shape[0], 10)
            self.assertEqual(sorted(panel["year"].unique().tolist()), EXPECTED_YEARS)
            self.assertEqual(panel.columns.tolist(), REQUIRED_PANEL_COLUMNS)

    def test_load_context_municipality_level_rejects_non_invariant_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            panel = self._write_panel(root)
            panel.loc[panel["municipality_code"].eq("110001") & panel["year"].eq(2024), "GINI_EXPANDED"] = 0.60
            with self.assertRaisesRegex(ValueError, "non-invariant fields"):
                load_context_municipality_level(panel)

    def test_coverage_by_year_detects_gaps_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            panel = self._write_panel(root)
            panel = panel.loc[~(panel["municipality_code"].eq("110002") & panel["year"].eq(2024))].copy()
            panel = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
            coverage = coverage_by_year(panel)
            coverage_2024 = coverage.loc[coverage["year"].eq(2024)].iloc[0]
            coverage_2020 = coverage.loc[coverage["year"].eq(2020)].iloc[0]
            self.assertEqual(int(coverage_2024["missing_rows"]), 1)
            self.assertEqual(int(coverage_2020["duplicates_n"]), 1)

    def test_audit_gold_municipality_context_writes_md_csv_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "artifacts" / "status" / "municipality_context_audit"
            self._write_panel(root)

            report = audit_gold_municipality_context(root, output_dir=output_dir)

            self.assertEqual(report["status"], "PASS")
            self.assertTrue((root / report["report_path"]).exists())
            self.assertTrue((root / report["csv_outputs"]["coverage_by_year"]).exists())
            self.assertTrue((root / report["csv_outputs"]["missing_structural_municipalities"]).exists())
            for plot_path in report["plot_paths"]:
                absolute = root / plot_path
                self.assertTrue(absolute.exists())
                self.assertGreater(absolute.stat().st_size, 0)

            missing_frame = pd.read_csv(root / report["csv_outputs"]["missing_structural_municipalities"], dtype={"municipality_code": str})
            self.assertEqual(missing_frame["municipality_code"].tolist(), ["110002"])

            report_text = (root / report["report_path"]).read_text(encoding="utf-8")
            self.assertIn("Municipality Context Audit", report_text)
            self.assertIn("Missingness is concentrated in `GINI_*` and `IDH_*`", report_text)

    def test_cli_smoke_runs_audit_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "custom_audit_output"
            self._write_panel(root, missing_structural=False)
            stdout = io.StringIO()
            with patch("heat_mnch.cli.root_path", return_value=root), redirect_stdout(stdout):
                exit_code = main(["audit-gold-municipality-context", "--output-dir", str(output_dir)])
            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["status"], "PASS")
            self.assertTrue((root / payload["report_path"]).exists())
            self.assertTrue((root / payload["csv_outputs"]["frequencies_overall"]).exists())


if __name__ == "__main__":
    unittest.main()
