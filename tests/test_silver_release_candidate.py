from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from heat_mnch.silver_release_candidate import (
    CHILD_AGE_SCOPES,
    POP_UNDER5_FIELDS,
    POP_WRA_15_49_FIELDS,
    SIH_CHILD_FLAG_COLUMNS,
    SIM_CHILD_FLAG_COLUMNS,
    UNDER5_DATASET_NAME,
    WRA_15_49_DATASET_NAME,
    _existing_silver_validation_rows,
    _build_child_sih_daily_view,
    _build_child_sim_daily_view,
    _load_wra_frame_from_popsvs,
    _population_partition_file,
    _validate_population_frame,
)


def _sim_child_row(
    municipality_code: str,
    date: str,
    *,
    age_0_27d: int = 0,
    age_28_364d: int = 0,
    age_1_4y: int = 0,
    age_u5: int = 0,
    direct_heat_underlying: int = 0,
) -> dict[str, object]:
    row = {
        "MUNICIPALITY_CODE": municipality_code,
        "DATE": date,
        "YEAR": 2020,
        "MONTH": 1,
        "AGE_GROUP_0_27D": age_0_27d,
        "AGE_GROUP_28_364D": age_28_364d,
        "AGE_GROUP_1_4Y": age_1_4y,
        "AGE_GROUP_U5": age_u5,
    }
    for column in SIM_CHILD_FLAG_COLUMNS:
        row[column] = 1 if column == "UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_FLAG" and direct_heat_underlying else 0
    return row


def _sih_child_row(
    municipality_code: str,
    date: str,
    *,
    competence_year: int = 2020,
    competence_month: int = 1,
    year: int = 2020,
    age_0_27d: int = 0,
    age_28_364d: int = 0,
    age_1_4y: int = 0,
    age_u5: int = 0,
    direct_heat_principal: int = 0,
) -> dict[str, object]:
    row = {
        "MUNICIPALITY_CODE": municipality_code,
        "DATE": date,
        "YEAR": year,
        "MONTH": 1,
        "COMPETENCE_YEAR": competence_year,
        "COMPETENCE_MONTH": competence_month,
        "AGE_GROUP_0_27D": age_0_27d,
        "AGE_GROUP_28_364D": age_28_364d,
        "AGE_GROUP_1_4Y": age_1_4y,
        "AGE_GROUP_U5": age_u5,
    }
    for column in SIH_CHILD_FLAG_COLUMNS:
        row[column] = 1 if column == "PRINCIPAL_DIAGNOSIS_DIRECT_HEAT_ANY_FLAG" and direct_heat_principal else 0
    return row


class SilverReleaseCandidatePopulationTest(unittest.TestCase):
    def test_population_validation_fails_when_municipality_year_is_missing(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "SOURCE_POPULATION": UNDER5_DATASET_NAME,
                    "BRONZE_SOURCE_SYSTEM": "datasus_popsvs",
                    "BRONZE_SOURCE_FILE": "pop_under5_2024.csv.gz",
                    "BRONZE_SOURCE_PATH": "data/bronze/population/under5/2024/pop_under5_2024_source_preserving.csv.gz",
                    "BRONZE_SOURCE_MEMBER": "",
                    "BRONZE_YEAR": 2024,
                    "BRONZE_INGESTED_AT": "2026-03-19T00:00:00+00:00",
                    "MUNICIPALITY_CODE_IBGE7": "1100015",
                    "MUNICIPALITY_CODE": "110001",
                    "YEAR": 2024,
                    "DENOMINATOR_NAME": "POP_UNDER5",
                    "DENOMINATOR_SCOPE": "AGE_U5",
                    "POP_UNDER5": 100,
                    "QC_STATUS": "PASS",
                }
            ]
        )
        with patch(
            "heat_mnch.silver_release_candidate.population_derivative_expected_codes",
            return_value={"1100015", "1100023"},
        ):
            validation = _validate_population_frame(Path("/tmp"), 2024, frame, "POP_UNDER5", POP_UNDER5_FIELDS)
        self.assertFalse(validation["all_checks_passed"])
        self.assertEqual(validation["missing_code_count"], 1)
        self.assertEqual(validation["missing_codes_sample"], ["1100023"])
        self.assertTrue(validation["duplicates_zero"])

    def test_population_validation_honors_explicit_expected_code_exclusion(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "SOURCE_POPULATION": UNDER5_DATASET_NAME,
                    "BRONZE_SOURCE_SYSTEM": "datasus_popsvs",
                    "BRONZE_SOURCE_FILE": "pop_under5_2024.csv.gz",
                    "BRONZE_SOURCE_PATH": "data/bronze/population/under5/2024/pop_under5_2024_source_preserving.csv.gz",
                    "BRONZE_SOURCE_MEMBER": "",
                    "BRONZE_YEAR": 2024,
                    "BRONZE_INGESTED_AT": "2026-03-19T00:00:00+00:00",
                    "MUNICIPALITY_CODE_IBGE7": "1100015",
                    "MUNICIPALITY_CODE": "110001",
                    "YEAR": 2024,
                    "DENOMINATOR_NAME": "POP_UNDER5",
                    "DENOMINATOR_SCOPE": "AGE_U5",
                    "POP_UNDER5": 100,
                    "QC_STATUS": "PASS",
                }
            ]
        )
        with patch(
            "heat_mnch.silver_release_candidate.population_derivative_expected_codes",
            return_value={"1100015"},
        ):
            validation = _validate_population_frame(Path("/tmp"), 2024, frame, "POP_UNDER5", POP_UNDER5_FIELDS)
        self.assertTrue(validation["all_checks_passed"])
        self.assertEqual(validation["missing_code_count"], 0)

    def test_wra_15_49_comes_from_female_popsvs_ages_15_to_49_only(self) -> None:
        root = Path("/tmp")
        source_frame = pd.DataFrame(
            [
                {"COD_MUN": "1100015", "ANO_NUM": 2020, "SEXO_NUM": 2, "IDADE_NUM": 14, "POP_NUM": 99},
                {"COD_MUN": "1100015", "ANO_NUM": 2020, "SEXO_NUM": 2, "IDADE_NUM": 15, "POP_NUM": 100},
                {"COD_MUN": "1100015", "ANO_NUM": 2020, "SEXO_NUM": 2, "IDADE_NUM": 49, "POP_NUM": 200},
                {"COD_MUN": "1100015", "ANO_NUM": 2020, "SEXO_NUM": 2, "IDADE_NUM": 50, "POP_NUM": 300},
                {"COD_MUN": "1100015", "ANO_NUM": 2020, "SEXO_NUM": 1, "IDADE_NUM": 20, "POP_NUM": 400},
                {"COD_MUN": "1100023", "ANO_NUM": 2020, "SEXO_NUM": 2, "IDADE_NUM": 30, "POP_NUM": 50},
            ]
        )
        metadata = {
            "BRONZE_SOURCE_SYSTEM": "datasus_popsvs",
            "BRONZE_SOURCE_FILE": "datasus_popsvs_population_2020_source_preserving.csv.gz",
            "BRONZE_SOURCE_PATH": "data/bronze/population/datasus_popsvs/2020/datasus_popsvs_population_2020_source_preserving.csv.gz",
            "BRONZE_SOURCE_MEMBER": "",
            "BRONZE_INGESTED_AT": "2026-03-19T00:00:00+00:00",
        }
        with patch(
            "heat_mnch.silver_release_candidate.load_datasus_popsvs_canonical_frame",
            return_value=(source_frame, metadata, root / "data" / "bronze" / "population" / "datasus_popsvs" / "2020" / "datasus_popsvs_population_2020_source_preserving.csv.gz"),
        ):
            frame = _load_wra_frame_from_popsvs(root, 2020, WRA_15_49_DATASET_NAME, "POP_WRA_15_49", 15, 49)
        self.assertEqual(frame.columns.tolist(), POP_WRA_15_49_FIELDS)
        self.assertEqual(frame["MUNICIPALITY_CODE_IBGE7"].tolist(), ["1100015", "1100023"])
        self.assertEqual(frame["MUNICIPALITY_CODE"].tolist(), ["110001", "110002"])
        self.assertEqual(frame["POP_WRA_15_49"].astype(int).tolist(), [300, 50])

    def test_existing_climate_validation_report_is_treated_as_pass_without_all_checks_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            validation_dir = root / "artifacts" / "validation_reports" / "silver" / "climate"
            validation_dir.mkdir(parents=True, exist_ok=True)
            (root / "artifacts" / "manifests").mkdir(parents=True, exist_ok=True)
            (root / "artifacts" / "manifests" / "silver_climate.manifest.json").write_text("{}", encoding="utf-8")
            (validation_dir / "2020.json").write_text(
                pd.Series(
                    {
                        "zero_missing_ok": True,
                        "dense_panel_ok": True,
                        "calendar_coverage_ok": True,
                        "municipality_coverage_ok": True,
                        "partition_integrity_ok": True,
                        "key_uniqueness_ok": True,
                        "provenance_ok": True,
                        "row_level_provenance_ok": True,
                        "observed_rows": 10,
                    }
                ).to_json(),
                encoding="utf-8",
            )

            rows = _existing_silver_validation_rows(root, 2020, 2020)
            climate_row = next(row for row in rows if row["dataset"] == "silver_climate")
            self.assertEqual(climate_row["status"], "PASS")
            self.assertTrue(climate_row["missing_critical_ok"])


class SilverReleaseCandidateAnalyticalViewTest(unittest.TestCase):
    def _write_sim_individual(self, root: Path, rows: list[dict[str, object]]) -> None:
        path = root / "data" / "silver" / "sim" / "silver_sim_individual" / "YEAR=2020" / "silver_sim_individual_2020.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)

    def _write_sih_individual(self, root: Path, rows: list[dict[str, object]]) -> None:
        path = root / "data" / "silver" / "sih" / "silver_sih_individual" / "YEAR=2020" / "silver_sih_individual_2020.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)

    def _write_under5_lookup(self, root: Path) -> None:
        path = _population_partition_file(root, UNDER5_DATASET_NAME, 2020)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"MUNICIPALITY_CODE": "110001", "YEAR": 2020, "POP_UNDER5": 1234},
                {"MUNICIPALITY_CODE": "110002", "YEAR": 2020, "POP_UNDER5": 4321},
            ]
        ).to_parquet(path, index=False)

    def test_child_sim_daily_view_filters_residual_codes_and_keeps_under5_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_root = root / "runtime"
            self._write_under5_lookup(root)
            self._write_sim_individual(
                root,
                [
                    _sim_child_row("110001", "2020-01-01", age_0_27d=1, age_u5=1, direct_heat_underlying=1),
                    _sim_child_row("110001", "2020-01-02", age_28_364d=1, age_u5=1),
                    _sim_child_row("110001", "2020-01-03", age_1_4y=1, age_u5=1),
                    _sim_child_row("290000", "2020-01-04", age_0_27d=1, age_u5=1, direct_heat_underlying=1),
                ],
            )

            validation, partition_path = _build_child_sim_daily_view(root, runtime_root, 2020)

            self.assertTrue(validation["all_checks_passed"])
            self.assertEqual(validation["excluded_nonmunicipal_rows"], 1)
            self.assertTrue(partition_path.endswith("silver_child_u5_sim_daily_2020.parquet"))

            output = pd.read_parquet(root / partition_path)
            self.assertEqual(sorted(output["AGE_SCOPE"].unique().tolist()), sorted([scope for scope, _ in CHILD_AGE_SCOPES]))
            self.assertNotIn("290000", output["MUNICIPALITY_CODE"].tolist())
            self.assertEqual(
                int(output.loc[output["AGE_SCOPE"].eq("AGE_U5"), "UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_TOTAL"].sum()),
                1,
            )
            self.assertTrue(output.loc[output["AGE_SCOPE"].eq("AGE_U5"), "DENOMINATOR_VALUE"].notna().all())
            self.assertTrue(output.loc[~output["AGE_SCOPE"].eq("AGE_U5"), "DENOMINATOR_VALUE"].isna().all())

    def test_child_sim_daily_view_blocks_without_under5_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_root = root / "runtime"

            validation, partition_path = _build_child_sim_daily_view(root, runtime_root, 2020)

            self.assertFalse(validation["all_checks_passed"])
            self.assertEqual(partition_path, "")
            self.assertIn("required denominator partition missing", validation["blocking_issue"])

    def test_child_sih_daily_view_uses_competence_year_for_denominator_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_root = root / "runtime"
            self._write_under5_lookup(root)
            self._write_sih_individual(
                root,
                [
                    _sih_child_row("110001", "2019-12-29", year=2019, competence_year=2020, age_0_27d=1, age_u5=1, direct_heat_principal=1),
                    _sih_child_row("110001", "2019-12-30", year=2019, competence_year=2020, age_28_364d=1, age_u5=1),
                    _sih_child_row("110001", "2019-12-31", year=2019, competence_year=2020, age_1_4y=1, age_u5=1),
                ],
            )

            validation, partition_path = _build_child_sih_daily_view(root, runtime_root, 2020)

            self.assertTrue(validation["all_checks_passed"])
            self.assertEqual(validation["denominator_missing_count"], 0)
            self.assertTrue(partition_path.endswith("silver_child_u5_sih_daily_2020.parquet"))
            output = pd.read_parquet(root / partition_path)
            self.assertTrue(output["POP_UNDER5"].notna().all())
            self.assertEqual(
                int(output.loc[output["AGE_SCOPE"].eq("AGE_U5"), "PRINCIPAL_DIAGNOSIS_DIRECT_HEAT_ANY_TOTAL"].sum()),
                1,
            )


if __name__ == "__main__":
    unittest.main()
