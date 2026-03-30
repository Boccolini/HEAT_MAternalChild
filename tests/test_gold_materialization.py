from __future__ import annotations

import math
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from heat_mnch.gold_materialization import (
    build_gold_analysis_child_u5_year,
    build_gold_analysis_wra_15_45_sens_year,
    build_gold_mortality_year,
    build_gold_panel_year,
    build_gold_perinatal_year,
)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _with_sinasc_stage1_fields(
    frame: pd.DataFrame,
    *,
    birth_weight_g: list[float | None],
    birth_weight_reported: list[int],
    low_birth_weight_lt2500: list[int | None],
    preterm_low_birth_weight: list[int | None],
    prenatal_visits_8plus: list[int | None],
) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["BIRTH_WEIGHT_G"] = pd.array(birth_weight_g, dtype="Float64")
    enriched["BIRTH_WEIGHT_REPORTED_FLAG"] = pd.array(birth_weight_reported, dtype="Int8")
    enriched["LOW_BIRTH_WEIGHT_LT2500_FLAG"] = pd.array(low_birth_weight_lt2500, dtype="Int8")
    enriched["PRETERM_LOW_BIRTH_WEIGHT_FLAG"] = pd.array(preterm_low_birth_weight, dtype="Int8")
    enriched["PRENATAL_VISITS_8PLUS_FLAG"] = pd.array(prenatal_visits_8plus, dtype="Int8")
    return enriched


def _with_aph_totals(
    frame: pd.DataFrame,
    *,
    aph_total: list[int],
    aph_urgent: list[int],
    aph_icu: list[int],
    aph_death: list[int],
) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["APH_ANTEPARTUM_TOTAL"] = aph_total
    enriched["APH_ANTEPARTUM_URGENT_TOTAL"] = aph_urgent
    enriched["APH_ANTEPARTUM_ICU_TOTAL"] = aph_icu
    enriched["APH_ANTEPARTUM_DEATH_TOTAL"] = aph_death
    return enriched


class GoldMaterializationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.runtime_root = self.root / "runtime"
        (self.root / "schemas").mkdir(parents=True, exist_ok=True)
        repo_root = Path(__file__).resolve().parents[1]
        for filename in (
            "gold_mnch_perinatal_diario_municipio.schema.csv",
            "gold_mnch_mortality_diario_municipio.schema.csv",
            "gold_mnch_panel_diario_municipio.schema.csv",
        ):
            shutil.copy2(repo_root / "schemas" / filename, self.root / "schemas" / filename)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_common_climate(self) -> None:
        climate = pd.DataFrame(
            [
                {
                    "COD_MUNICIPIO": "123456",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "TMAX_C": 34.5,
                    "TMIN_C": 24.0,
                    "TMEAN_C": 29.0,
                    "RH_MIN_PCT": 42.0,
                    "RH_MEAN_PCT": 55.0,
                    "HEATWAVE_LEN_P95": 2,
                    "HEAT_INDEX_C": 36.0,
                    "FONTE_CLIMA_FINAL": "mixed",
                    "FONTE_TMAX_C_FINAL": "inmet",
                    "FONTE_TMIN_C_FINAL": "era5",
                    "FONTE_TMEAN_C_FINAL": "mixed",
                    "FONTE_RH_MIN_PCT_FINAL": "era5",
                    "FONTE_RH_MEAN_PCT_FINAL": "era5",
                    "IMPUTADO_POR_FALLBACK": 0,
                    "IMPUTADO_TMAX_C": 0,
                    "IMPUTADO_TMIN_C": 0,
                    "IMPUTADO_TMEAN_C": 0,
                    "IMPUTADO_RH_MIN_PCT": 0,
                    "IMPUTADO_RH_MEAN_PCT": 0,
                    "CLIMATE_UPSTREAM_QUARANTINE_FLAG": 1,
                    "CLIMATE_RECORD_ANOMALY_FLAG": 0,
                    "CLIMATE_CORRECTION_PATH": "upstream_station_quarantine_to_era5",
                    "HEAT_INDEX_METHOD": "approx",
                    "HEAT_INDEX_SOURCE_PAIR": "mixed",
                    "HEAT_INDEX_INPUTS_ZERO_MISSING": 1,
                },
                {
                    "COD_MUNICIPIO": "123456",
                    "DATE": "2024-01-02",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "TMAX_C": 35.0,
                    "TMIN_C": 25.0,
                    "TMEAN_C": 30.0,
                    "RH_MIN_PCT": 44.0,
                    "RH_MEAN_PCT": 57.0,
                    "HEATWAVE_LEN_P95": 3,
                    "HEAT_INDEX_C": 37.5,
                    "FONTE_CLIMA_FINAL": "mixed",
                    "FONTE_TMAX_C_FINAL": "era5",
                    "FONTE_TMIN_C_FINAL": "era5",
                    "FONTE_TMEAN_C_FINAL": "era5",
                    "FONTE_RH_MIN_PCT_FINAL": "era5",
                    "FONTE_RH_MEAN_PCT_FINAL": "era5",
                    "IMPUTADO_POR_FALLBACK": 0,
                    "IMPUTADO_TMAX_C": 0,
                    "IMPUTADO_TMIN_C": 0,
                    "IMPUTADO_TMEAN_C": 0,
                    "IMPUTADO_RH_MIN_PCT": 0,
                    "IMPUTADO_RH_MEAN_PCT": 0,
                    "CLIMATE_UPSTREAM_QUARANTINE_FLAG": 0,
                    "CLIMATE_RECORD_ANOMALY_FLAG": 1,
                    "CLIMATE_CORRECTION_PATH": "record_anomaly_neighbor_imputation",
                    "HEAT_INDEX_METHOD": "approx",
                    "HEAT_INDEX_SOURCE_PAIR": "mixed",
                    "HEAT_INDEX_INPUTS_ZERO_MISSING": 1,
                },
            ]
        )
        _write_parquet(self.root / "data" / "silver" / "climate" / "YEAR=2024" / "climate_canonical_daily_2024.parquet", climate)

    def test_build_gold_perinatal_year_uses_daily_total_births_and_event_level_sinasc(self) -> None:
        self._write_common_climate()
        sinasc_event = pd.DataFrame(
            [
                {
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "GA_WEEKS_CORRECTED_FINAL": 36.0,
                    "GA_DAYS_FINAL": 254.0,
                    "GA_VALID_FOR_ANALYSIS_FLAG": 1,
                    "PRETERM_LT37_FLAG": 1,
                    "PRETERM_LT32_FLAG": 0,
                    "PRETERM_DAYS_SHORT_OF_37W_FINAL": 5.0,
                    "PRENATAL_VISITS_COUNT": 7.0,
                    "PRENATAL_VISITS_6PLUS_FLAG": 1,
                    "PRENATAL_VISITS_SOURCE": "SINASC_CONSPRENAT",
                    "PRENATAL_VISITS_RAW_VALUE": "7",
                    "PRENATAL_VISITS_MISSING_FLAG": 0,
                    "SOURCE_SINASC": "silver_sinasc_event",
                },
                {
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "GA_WEEKS_CORRECTED_FINAL": 38.0,
                    "GA_DAYS_FINAL": 266.0,
                    "GA_VALID_FOR_ANALYSIS_FLAG": 1,
                    "PRETERM_LT37_FLAG": 0,
                    "PRETERM_LT32_FLAG": 0,
                    "PRETERM_DAYS_SHORT_OF_37W_FINAL": 0.0,
                    "PRENATAL_VISITS_COUNT": 5.0,
                    "PRENATAL_VISITS_6PLUS_FLAG": 0,
                    "PRENATAL_VISITS_SOURCE": "SINASC_CONSPRENAT",
                    "PRENATAL_VISITS_RAW_VALUE": "5",
                    "PRENATAL_VISITS_MISSING_FLAG": 0,
                    "SOURCE_SINASC": "silver_sinasc_event",
                },
                {
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-02",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "GA_WEEKS_CORRECTED_FINAL": 31.0,
                    "GA_DAYS_FINAL": 220.0,
                    "GA_VALID_FOR_ANALYSIS_FLAG": 1,
                    "PRETERM_LT37_FLAG": 1,
                    "PRETERM_LT32_FLAG": 1,
                    "PRETERM_DAYS_SHORT_OF_37W_FINAL": 39.0,
                    "PRENATAL_VISITS_COUNT": pd.NA,
                    "PRENATAL_VISITS_6PLUS_FLAG": 1,
                    "PRENATAL_VISITS_SOURCE": "SINASC_CONSULTAS_BAND",
                    "PRENATAL_VISITS_RAW_VALUE": "4",
                    "PRENATAL_VISITS_MISSING_FLAG": 0,
                    "SOURCE_SINASC": "silver_sinasc_event",
                },
            ]
        )
        sinasc_event = _with_sinasc_stage1_fields(
            sinasc_event,
            birth_weight_g=[2400.0, 2600.0, None],
            birth_weight_reported=[1, 1, 0],
            low_birth_weight_lt2500=[1, 0, None],
            preterm_low_birth_weight=[1, 0, None],
            prenatal_visits_8plus=[0, 0, None],
        )
        _write_parquet(
            self.root / "data" / "silver" / "sinasc" / "silver_sinasc_event" / "YEAR=2024" / "silver_sinasc_event_2024.parquet",
            sinasc_event,
        )
        stillbirth = pd.DataFrame(
            [
                {
                    "MUNICIPALITY_CODE": "123456",
                    "EVENT_DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "LIVE_BIRTHS_TOTAL": 2,
                    "STILLBIRTH_TOTAL_GTE28": 1,
                    "STILLBIRTH_TOTAL_GTE22_SENSITIVITY": 1,
                    "TOTAL_BIRTHS_PRIMARY": 3,
                    "TOTAL_BIRTHS_SENSITIVITY_GTE22": 3,
                    "OFFSET_LOG_TOTAL_BIRTHS_PRIMARY": math.log(3),
                    "OFFSET_LOG_TOTAL_BIRTHS_SENSITIVITY_GTE22": math.log(3),
                    "STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_PRIMARY": 1000.0 / 3.0,
                    "STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_SENSITIVITY_GTE22": 1000.0 / 3.0,
                    "SOURCE_STILLBIRTH": "silver_stillbirth_daily",
                    "QC_STATUS": "PASS",
                },
                {
                    "MUNICIPALITY_CODE": "123456",
                    "EVENT_DATE": "2024-01-02",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "LIVE_BIRTHS_TOTAL": 1,
                    "STILLBIRTH_TOTAL_GTE28": 0,
                    "STILLBIRTH_TOTAL_GTE22_SENSITIVITY": 0,
                    "TOTAL_BIRTHS_PRIMARY": 1,
                    "TOTAL_BIRTHS_SENSITIVITY_GTE22": 1,
                    "OFFSET_LOG_TOTAL_BIRTHS_PRIMARY": 0.0,
                    "OFFSET_LOG_TOTAL_BIRTHS_SENSITIVITY_GTE22": 0.0,
                    "STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_PRIMARY": 0.0,
                    "STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_SENSITIVITY_GTE22": 0.0,
                    "SOURCE_STILLBIRTH": "silver_stillbirth_daily",
                    "QC_STATUS": "PASS",
                },
            ]
        )
        _write_parquet(
            self.root
            / "data"
            / "silver"
            / "stillbirth"
            / "silver_stillbirth_daily"
            / "YEAR=2024"
            / "silver_stillbirth_daily_2024.parquet",
            stillbirth,
        )
        stillbirth_individual = pd.DataFrame(
            [
                {
                    "MUNICIPALITY_CODE": "123456",
                    "EVENT_DATE": "2024-01-01",
                    "STILLBIRTH_GTE28_PRIMARY_FLAG": 1,
                    "STILLBIRTH_GTE22_SENSITIVITY_FLAG": 1,
                    "SOURCE_STILLBIRTH": "silver_stillbirth_individual",
                    "QC_STATUS": "PASS",
                }
            ]
        )
        _write_parquet(
            self.root
            / "data"
            / "silver"
            / "stillbirth"
            / "silver_stillbirth_individual"
            / "YEAR=2024"
            / "silver_stillbirth_individual_2024.parquet",
            stillbirth_individual,
        )
        sih = pd.DataFrame(
            [
                {
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "OBSTETRIC_ADMISSIONS_TOTAL": 1,
                    "OBSTETRIC_URGENT_TOTAL": 1,
                    "OBSTETRIC_ICU_TOTAL": 0,
                    "HDP_ANY_TOTAL": 0,
                    "HDP_SEVERE_TOTAL": 0,
                    "HDP_GESTATIONAL_HTN_TOTAL": 0,
                    "HDP_PREECLAMPSIA_TOTAL": 0,
                    "HDP_ECLAMPSIA_TOTAL": 0,
                    "PRETERM_LABOR_TOTAL": 0,
                    "PRETERM_LABOR_URGENT_TOTAL": 0,
                    "PRETERM_LABOR_ICU_TOTAL": 0,
                    "PRETERM_LABOR_DEATH_TOTAL": 0,
                    "GDM_GESTATIONAL_TOTAL": 0,
                    "GDM_GESTATIONAL_URGENT_TOTAL": 0,
                    "GDM_GESTATIONAL_ICU_TOTAL": 0,
                    "GDM_GESTATIONAL_DEATH_TOTAL": 0,
                    "DIABETES_PREGNANCY_BROAD_TOTAL": 0,
                },
                {
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "OBSTETRIC_ADMISSIONS_TOTAL": 5,
                    "OBSTETRIC_URGENT_TOTAL": 4,
                    "OBSTETRIC_ICU_TOTAL": 1,
                    "HDP_ANY_TOTAL": 2,
                    "HDP_SEVERE_TOTAL": 1,
                    "HDP_GESTATIONAL_HTN_TOTAL": 1,
                    "HDP_PREECLAMPSIA_TOTAL": 1,
                    "HDP_ECLAMPSIA_TOTAL": 0,
                    "PRETERM_LABOR_TOTAL": 1,
                    "PRETERM_LABOR_URGENT_TOTAL": 1,
                    "PRETERM_LABOR_ICU_TOTAL": 0,
                    "PRETERM_LABOR_DEATH_TOTAL": 0,
                    "GDM_GESTATIONAL_TOTAL": 1,
                    "GDM_GESTATIONAL_URGENT_TOTAL": 1,
                    "GDM_GESTATIONAL_ICU_TOTAL": 0,
                    "GDM_GESTATIONAL_DEATH_TOTAL": 0,
                    "DIABETES_PREGNANCY_BROAD_TOTAL": 1,
                },
                {
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-02",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "OBSTETRIC_ADMISSIONS_TOTAL": 2,
                    "OBSTETRIC_URGENT_TOTAL": 2,
                    "OBSTETRIC_ICU_TOTAL": 0,
                    "HDP_ANY_TOTAL": 1,
                    "HDP_SEVERE_TOTAL": 0,
                    "HDP_GESTATIONAL_HTN_TOTAL": 1,
                    "HDP_PREECLAMPSIA_TOTAL": 0,
                    "HDP_ECLAMPSIA_TOTAL": 0,
                    "PRETERM_LABOR_TOTAL": 0,
                    "PRETERM_LABOR_URGENT_TOTAL": 0,
                    "PRETERM_LABOR_ICU_TOTAL": 0,
                    "PRETERM_LABOR_DEATH_TOTAL": 0,
                    "GDM_GESTATIONAL_TOTAL": 1,
                    "GDM_GESTATIONAL_URGENT_TOTAL": 0,
                    "GDM_GESTATIONAL_ICU_TOTAL": 0,
                    "GDM_GESTATIONAL_DEATH_TOTAL": 0,
                    "DIABETES_PREGNANCY_BROAD_TOTAL": 1,
                },
            ]
        )
        sih = _with_aph_totals(
            sih,
            aph_total=[0, 1, 0],
            aph_urgent=[0, 1, 0],
            aph_icu=[0, 0, 0],
            aph_death=[0, 0, 0],
        )
        _write_parquet(
            self.root / "data" / "silver" / "sih" / "silver_sih_daily" / "YEAR=2024" / "silver_sih_daily_2024.parquet",
            sih,
        )

        result = build_gold_perinatal_year(self.root, self.runtime_root, 2024)
        output = pd.read_parquet(result["output_path"])

        self.assertEqual(len(output), 2)
        self.assertEqual(int(output.loc[output["DATE"] == "2024-01-01", "TOTAL_BIRTHS_PRIMARY"].iloc[0]), 3)
        self.assertEqual(int(output.loc[output["DATE"] == "2024-01-02", "TOTAL_BIRTHS_PRIMARY"].iloc[0]), 1)
        self.assertEqual(float(output.loc[output["DATE"] == "2024-01-01", "OFFSET_LOG_TOTAL_BIRTHS_PRIMARY"].iloc[0]), math.log(3))
        self.assertEqual(float(output.loc[output["DATE"] == "2024-01-02", "OFFSET_LOG_TOTAL_BIRTHS_PRIMARY"].iloc[0]), 0.0)
        self.assertTrue((output["DENOMINATOR_ZERO_FLAG"] == 0).all())
        self.assertEqual(int(output["PRETERM_FLAG"].sum()), 2)
        self.assertEqual(int(output["VERY_PRETERM_FLAG"].sum()), 1)
        self.assertEqual(int(output["LOW_BIRTH_WEIGHT_LT2500_TOTAL"].sum()), 1)
        self.assertEqual(int(output["PRETERM_LOW_BIRTH_WEIGHT_TOTAL"].sum()), 1)
        self.assertEqual(int(output["PRENATAL_VISITS_8PLUS_FLAG"].sum()), 0)
        self.assertEqual(float(output.loc[output["DATE"] == "2024-01-02", "GEST_DAYS_LOST_BELOW_37W"].iloc[0]), 39.0)
        self.assertEqual(str(output.loc[output["DATE"] == "2024-01-01", "PRENATAL_VISITS_SOURCE"].iloc[0]), "SINASC_CONSPRENAT")
        self.assertEqual(int(output.loc[output["DATE"] == "2024-01-01", "OBSTETRIC_URGENT_TOTAL"].iloc[0]), 5)
        self.assertEqual(int(output.loc[output["DATE"] == "2024-01-01", "OBSTETRIC_ADMISSIONS_TOTAL"].iloc[0]), 6)
        self.assertEqual(int(output.loc[output["DATE"] == "2024-01-01", "APH_ANTEPARTUM_TOTAL"].iloc[0]), 1)
        self.assertEqual(float(output.loc[output["DATE"] == "2024-01-02", "HEAT_INDEX_APPROX_C"].iloc[0]), 37.5)
        self.assertEqual(str(output.loc[output["DATE"] == "2024-01-01", "CLIMATE_CORRECTION_PATH"].iloc[0]), "upstream_station_quarantine_to_era5")
        self.assertEqual(int(output.loc[output["DATE"] == "2024-01-02", "CLIMATE_RECORD_ANOMALY_FLAG"].iloc[0]), 1)

    def test_build_gold_perinatal_year_preserves_stillbirth_rows_without_climate_match(self) -> None:
        self._write_common_climate()
        sinasc_event = pd.DataFrame(
            [
                {
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "GA_WEEKS_CORRECTED_FINAL": 38.0,
                    "GA_DAYS_FINAL": 266.0,
                    "GA_VALID_FOR_ANALYSIS_FLAG": 1,
                    "PRETERM_LT37_FLAG": 0,
                    "PRETERM_LT32_FLAG": 0,
                    "PRETERM_DAYS_SHORT_OF_37W_FINAL": 0.0,
                    "PRENATAL_VISITS_COUNT": 6.0,
                    "PRENATAL_VISITS_6PLUS_FLAG": 1,
                    "PRENATAL_VISITS_SOURCE": "SINASC_CONSPRENAT",
                    "PRENATAL_VISITS_RAW_VALUE": "6",
                    "PRENATAL_VISITS_MISSING_FLAG": 0,
                    "SOURCE_SINASC": "silver_sinasc_event",
                }
            ]
        )
        sinasc_event = _with_sinasc_stage1_fields(
            sinasc_event,
            birth_weight_g=[3100.0],
            birth_weight_reported=[1],
            low_birth_weight_lt2500=[0],
            preterm_low_birth_weight=[0],
            prenatal_visits_8plus=[0],
        )
        _write_parquet(
            self.root / "data" / "silver" / "sinasc" / "silver_sinasc_event" / "YEAR=2024" / "silver_sinasc_event_2024.parquet",
            sinasc_event,
        )
        stillbirth = pd.DataFrame(
            [
                {
                    "MUNICIPALITY_CODE": "654321",
                    "EVENT_DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "LIVE_BIRTHS_TOTAL": 0,
                    "STILLBIRTH_TOTAL_GTE28": 1,
                    "STILLBIRTH_TOTAL_GTE22_SENSITIVITY": 1,
                    "TOTAL_BIRTHS_PRIMARY": 1,
                    "TOTAL_BIRTHS_SENSITIVITY_GTE22": 1,
                    "OFFSET_LOG_TOTAL_BIRTHS_PRIMARY": 0.0,
                    "OFFSET_LOG_TOTAL_BIRTHS_SENSITIVITY_GTE22": 0.0,
                    "STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_PRIMARY": 1000.0,
                    "STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_SENSITIVITY_GTE22": 1000.0,
                    "SOURCE_STILLBIRTH": "silver_stillbirth_daily",
                    "QC_STATUS": "PASS",
                }
            ]
        )
        _write_parquet(
            self.root
            / "data"
            / "silver"
            / "stillbirth"
            / "silver_stillbirth_daily"
            / "YEAR=2024"
            / "silver_stillbirth_daily_2024.parquet",
            stillbirth,
        )
        stillbirth_individual = pd.DataFrame(
            [
                {
                    "MUNICIPALITY_CODE": "654321",
                    "EVENT_DATE": "2024-01-01",
                    "STILLBIRTH_GTE28_PRIMARY_FLAG": 1,
                    "STILLBIRTH_GTE22_SENSITIVITY_FLAG": 1,
                    "SOURCE_STILLBIRTH": "silver_stillbirth_individual",
                    "QC_STATUS": "PASS",
                }
            ]
        )
        _write_parquet(
            self.root
            / "data"
            / "silver"
            / "stillbirth"
            / "silver_stillbirth_individual"
            / "YEAR=2024"
            / "silver_stillbirth_individual_2024.parquet",
            stillbirth_individual,
        )
        sih = pd.DataFrame(
            [
                {
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "OBSTETRIC_ADMISSIONS_TOTAL": 1,
                    "OBSTETRIC_URGENT_TOTAL": 1,
                    "OBSTETRIC_ICU_TOTAL": 0,
                    "HDP_ANY_TOTAL": 1,
                    "HDP_SEVERE_TOTAL": 0,
                    "HDP_GESTATIONAL_HTN_TOTAL": 1,
                    "HDP_PREECLAMPSIA_TOTAL": 0,
                    "HDP_ECLAMPSIA_TOTAL": 0,
                    "PRETERM_LABOR_TOTAL": 0,
                    "PRETERM_LABOR_URGENT_TOTAL": 0,
                    "PRETERM_LABOR_ICU_TOTAL": 0,
                    "PRETERM_LABOR_DEATH_TOTAL": 0,
                    "GDM_GESTATIONAL_TOTAL": 0,
                    "GDM_GESTATIONAL_URGENT_TOTAL": 0,
                    "GDM_GESTATIONAL_ICU_TOTAL": 0,
                    "GDM_GESTATIONAL_DEATH_TOTAL": 0,
                    "DIABETES_PREGNANCY_BROAD_TOTAL": 0,
                }
            ]
        )
        sih = _with_aph_totals(
            sih,
            aph_total=[0],
            aph_urgent=[0],
            aph_icu=[0],
            aph_death=[0],
        )
        _write_parquet(
            self.root / "data" / "silver" / "sih" / "silver_sih_daily" / "YEAR=2024" / "silver_sih_daily_2024.parquet",
            sih,
        )

        result = build_gold_perinatal_year(self.root, self.runtime_root, 2024)
        output = pd.read_parquet(result["output_path"])

        self.assertEqual(len(output), 3)
        unmatched = output.loc[output["MUNICIPALITY_CODE"] == "654321"].iloc[0]
        self.assertEqual(int(unmatched["STILLBIRTH_TOTAL_PRIMARY_GTE28"]), 1)
        self.assertEqual(int(unmatched["TOTAL_BIRTHS_PRIMARY"]), 1)
        self.assertTrue(pd.isna(unmatched["TMAX_C"]))
        self.assertEqual(str(unmatched["FONTE_CLIMA_FINAL"]), "NO_CLIMATE_BACKBONE_MATCH")
        self.assertEqual(int(result["perinatal_summary"]["climate_backbone_unmatched_rows"]), 1)

    def test_build_gold_perinatal_year_excludes_placeholder_municipality_codes(self) -> None:
        self._write_common_climate()
        sinasc_event = pd.DataFrame(
            [
                {
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "GA_WEEKS_CORRECTED_FINAL": 38.0,
                    "GA_DAYS_FINAL": 266.0,
                    "GA_VALID_FOR_ANALYSIS_FLAG": 1,
                    "PRETERM_LT37_FLAG": 0,
                    "PRETERM_LT32_FLAG": 0,
                    "PRETERM_DAYS_SHORT_OF_37W_FINAL": 0.0,
                    "PRENATAL_VISITS_COUNT": 6.0,
                    "PRENATAL_VISITS_6PLUS_FLAG": 1,
                    "PRENATAL_VISITS_SOURCE": "SINASC_CONSPRENAT",
                    "PRENATAL_VISITS_RAW_VALUE": "6",
                    "PRENATAL_VISITS_MISSING_FLAG": 0,
                    "SOURCE_SINASC": "silver_sinasc_event",
                },
                {
                    "MUNICIPALITY_CODE": "130000",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "GA_WEEKS_CORRECTED_FINAL": 35.0,
                    "GA_DAYS_FINAL": 245.0,
                    "GA_VALID_FOR_ANALYSIS_FLAG": 1,
                    "PRETERM_LT37_FLAG": 1,
                    "PRETERM_LT32_FLAG": 0,
                    "PRETERM_DAYS_SHORT_OF_37W_FINAL": 14.0,
                    "PRENATAL_VISITS_COUNT": 4.0,
                    "PRENATAL_VISITS_6PLUS_FLAG": 0,
                    "PRENATAL_VISITS_SOURCE": "SINASC_CONSPRENAT",
                    "PRENATAL_VISITS_RAW_VALUE": "4",
                    "PRENATAL_VISITS_MISSING_FLAG": 0,
                    "SOURCE_SINASC": "silver_sinasc_event",
                },
            ]
        )
        sinasc_event = _with_sinasc_stage1_fields(
            sinasc_event,
            birth_weight_g=[3100.0, 2400.0],
            birth_weight_reported=[1, 1],
            low_birth_weight_lt2500=[0, 1],
            preterm_low_birth_weight=[0, 1],
            prenatal_visits_8plus=[0, 0],
        )
        _write_parquet(
            self.root / "data" / "silver" / "sinasc" / "silver_sinasc_event" / "YEAR=2024" / "silver_sinasc_event_2024.parquet",
            sinasc_event,
        )
        stillbirth = pd.DataFrame(
            [
                {
                    "MUNICIPALITY_CODE": "123456",
                    "EVENT_DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "LIVE_BIRTHS_TOTAL": 1,
                    "STILLBIRTH_TOTAL_GTE28": 1,
                    "STILLBIRTH_TOTAL_GTE22_SENSITIVITY": 1,
                    "TOTAL_BIRTHS_PRIMARY": 2,
                    "TOTAL_BIRTHS_SENSITIVITY_GTE22": 2,
                    "OFFSET_LOG_TOTAL_BIRTHS_PRIMARY": math.log(2),
                    "OFFSET_LOG_TOTAL_BIRTHS_SENSITIVITY_GTE22": math.log(2),
                    "STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_PRIMARY": 500.0,
                    "STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_SENSITIVITY_GTE22": 500.0,
                    "SOURCE_STILLBIRTH": "silver_stillbirth_daily",
                    "QC_STATUS": "PASS",
                },
                {
                    "MUNICIPALITY_CODE": "130000",
                    "EVENT_DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "LIVE_BIRTHS_TOTAL": 0,
                    "STILLBIRTH_TOTAL_GTE28": 1,
                    "STILLBIRTH_TOTAL_GTE22_SENSITIVITY": 1,
                    "TOTAL_BIRTHS_PRIMARY": 1,
                    "TOTAL_BIRTHS_SENSITIVITY_GTE22": 1,
                    "OFFSET_LOG_TOTAL_BIRTHS_PRIMARY": 0.0,
                    "OFFSET_LOG_TOTAL_BIRTHS_SENSITIVITY_GTE22": 0.0,
                    "STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_PRIMARY": 1000.0,
                    "STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_SENSITIVITY_GTE22": 1000.0,
                    "SOURCE_STILLBIRTH": "silver_stillbirth_daily",
                    "QC_STATUS": "PASS",
                },
            ]
        )
        _write_parquet(
            self.root
            / "data"
            / "silver"
            / "stillbirth"
            / "silver_stillbirth_daily"
            / "YEAR=2024"
            / "silver_stillbirth_daily_2024.parquet",
            stillbirth,
        )
        stillbirth_individual = pd.DataFrame(
            [
                {
                    "MUNICIPALITY_CODE": "123456",
                    "EVENT_DATE": "2024-01-01",
                    "STILLBIRTH_GTE28_PRIMARY_FLAG": 1,
                    "STILLBIRTH_GTE22_SENSITIVITY_FLAG": 1,
                },
                {
                    "MUNICIPALITY_CODE": "130000",
                    "EVENT_DATE": "2024-01-01",
                    "STILLBIRTH_GTE28_PRIMARY_FLAG": 1,
                    "STILLBIRTH_GTE22_SENSITIVITY_FLAG": 1,
                },
            ]
        )
        _write_parquet(
            self.root
            / "data"
            / "silver"
            / "stillbirth"
            / "silver_stillbirth_individual"
            / "YEAR=2024"
            / "silver_stillbirth_individual_2024.parquet",
            stillbirth_individual,
        )
        sih = pd.DataFrame(
            [
                {
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "OBSTETRIC_ADMISSIONS_TOTAL": 2,
                    "OBSTETRIC_URGENT_TOTAL": 2,
                    "OBSTETRIC_ICU_TOTAL": 0,
                    "HDP_ANY_TOTAL": 1,
                    "HDP_SEVERE_TOTAL": 0,
                    "HDP_GESTATIONAL_HTN_TOTAL": 1,
                    "HDP_PREECLAMPSIA_TOTAL": 0,
                    "HDP_ECLAMPSIA_TOTAL": 0,
                    "PRETERM_LABOR_TOTAL": 0,
                    "PRETERM_LABOR_URGENT_TOTAL": 0,
                    "PRETERM_LABOR_ICU_TOTAL": 0,
                    "PRETERM_LABOR_DEATH_TOTAL": 0,
                    "GDM_GESTATIONAL_TOTAL": 0,
                    "GDM_GESTATIONAL_URGENT_TOTAL": 0,
                    "GDM_GESTATIONAL_ICU_TOTAL": 0,
                    "GDM_GESTATIONAL_DEATH_TOTAL": 0,
                    "DIABETES_PREGNANCY_BROAD_TOTAL": 0,
                },
                {
                    "MUNICIPALITY_CODE": "130000",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "OBSTETRIC_ADMISSIONS_TOTAL": 3,
                    "OBSTETRIC_URGENT_TOTAL": 3,
                    "OBSTETRIC_ICU_TOTAL": 0,
                    "HDP_ANY_TOTAL": 2,
                    "HDP_SEVERE_TOTAL": 1,
                    "HDP_GESTATIONAL_HTN_TOTAL": 1,
                    "HDP_PREECLAMPSIA_TOTAL": 1,
                    "HDP_ECLAMPSIA_TOTAL": 0,
                    "PRETERM_LABOR_TOTAL": 1,
                    "PRETERM_LABOR_URGENT_TOTAL": 1,
                    "PRETERM_LABOR_ICU_TOTAL": 0,
                    "PRETERM_LABOR_DEATH_TOTAL": 0,
                    "GDM_GESTATIONAL_TOTAL": 1,
                    "GDM_GESTATIONAL_URGENT_TOTAL": 1,
                    "GDM_GESTATIONAL_ICU_TOTAL": 0,
                    "GDM_GESTATIONAL_DEATH_TOTAL": 0,
                    "DIABETES_PREGNANCY_BROAD_TOTAL": 1,
                },
            ]
        )
        sih = _with_aph_totals(
            sih,
            aph_total=[0, 0],
            aph_urgent=[0, 0],
            aph_icu=[0, 0],
            aph_death=[0, 0],
        )
        _write_parquet(
            self.root / "data" / "silver" / "sih" / "silver_sih_daily" / "YEAR=2024" / "silver_sih_daily_2024.parquet",
            sih,
        )

        result = build_gold_perinatal_year(self.root, self.runtime_root, 2024)
        output = pd.read_parquet(result["output_path"])

        self.assertEqual(output["MUNICIPALITY_CODE"].astype(str).unique().tolist(), ["123456"])
        self.assertEqual(int(result["perinatal_summary"]["excluded_placeholder_rows"]), 4)
        self.assertIn("excluded_placeholder_rows=4", result["external_validation_summary"])

    def test_build_gold_mortality_year_rolls_up_infant_and_annual_live_births(self) -> None:
        self._write_common_climate()
        cause_defaults = {
            "UNDERLYING_BASIC_CAUSE_CHILD_HEAT_THERMAL_PRIMARY_TOTAL": 0,
            "UNDERLYING_BASIC_CAUSE_CHILD_DEHYDRATION_PRIMARY_TOTAL": 0,
            "ANY_MENTION_CHILD_FLUID_ELECTROLYTE_SENSITIVITY_TOTAL": 0,
            "UNDERLYING_BASIC_CAUSE_CHILD_DIARRHEA_GI_PRIMARY_TOTAL": 0,
            "UNDERLYING_BASIC_CAUSE_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL": 0,
            "ANY_MENTION_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL": 0,
            "ANY_MENTION_CHILD_RENAL_ACUTE_SENSITIVITY_TOTAL": 0,
            "ANY_MENTION_CHILD_MALNUTRITION_SLOW_MOVING_SENSITIVITY_TOTAL": 0,
            "ANY_MENTION_EXCESSIVE_NATURAL_HEAT_X30_TOTAL": 0,
            "UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_TOTAL": 0,
            "ANY_MENTION_DIRECT_HEAT_ANY_TOTAL": 0,
            "UNDERLYING_BASIC_CAUSE_INDIRECT_HEAT_ANY_TOTAL": 0,
            "ANY_MENTION_INDIRECT_HEAT_ANY_TOTAL": 0,
            "UNDERLYING_BASIC_CAUSE_SLOW_MOVING_HEAT_ANY_TOTAL": 0,
            "ANY_MENTION_SLOW_MOVING_HEAT_ANY_TOTAL": 0,
            "POP_UNDER5": 100,
            "DENOMINATOR_NAME": "POP_UNDER5",
            "DENOMINATOR_VALUE": 100,
            "SOURCE_ANALYTICAL_VIEW": "silver_child_u5_sim_daily",
            "SOURCE_SIM": "silver_sim_individual",
            "QC_STATUS": "PASS",
        }
        child_sim = pd.DataFrame(
            [
                {"AGE_SCOPE": "AGE_0_27D", "MUNICIPALITY_CODE": "123456", "DATE": "2024-01-01", "YEAR": 2024, "MONTH": 1, "DEATHS_TOTAL": 1, **cause_defaults},
                {"AGE_SCOPE": "AGE_28_364D", "MUNICIPALITY_CODE": "123456", "DATE": "2024-01-01", "YEAR": 2024, "MONTH": 1, "DEATHS_TOTAL": 2, **cause_defaults},
                {
                    "AGE_SCOPE": "AGE_U5",
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "DEATHS_TOTAL": 3,
                    **{
                        **cause_defaults,
                        "UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_TOTAL": 1,
                        "ANY_MENTION_DIRECT_HEAT_ANY_TOTAL": 2,
                        "UNDERLYING_BASIC_CAUSE_INDIRECT_HEAT_ANY_TOTAL": 1,
                        "ANY_MENTION_INDIRECT_HEAT_ANY_TOTAL": 1,
                        "ANY_MENTION_EXCESSIVE_NATURAL_HEAT_X30_TOTAL": 1,
                    },
                },
                {"AGE_SCOPE": "AGE_28_364D", "MUNICIPALITY_CODE": "123456", "DATE": "2024-01-02", "YEAR": 2024, "MONTH": 1, "DEATHS_TOTAL": 1, **cause_defaults},
                {"AGE_SCOPE": "AGE_U5", "MUNICIPALITY_CODE": "123456", "DATE": "2024-01-02", "YEAR": 2024, "MONTH": 1, "DEATHS_TOTAL": 4, **cause_defaults},
            ]
        )
        _write_parquet(
            self.root
            / "data"
            / "silver"
            / "analytical"
            / "silver_child_u5_sim_daily"
            / "YEAR=2024"
            / "silver_child_u5_sim_daily_2024.parquet",
            child_sim,
        )
        sim_individual = pd.DataFrame(
            [
                {"MUNICIPALITY_CODE": "123456", "DATE": "2024-01-01", "YEAR": 2024, "MONTH": 1, "AGE_GROUP_0_27D": 1, "AGE_GROUP_28_364D": 0, "AGE_GROUP_U5": 1, "SOURCE_SIM": "silver_sim_individual", "QC_STATUS": "PASS"},
                {"MUNICIPALITY_CODE": "123456", "DATE": "2024-01-01", "YEAR": 2024, "MONTH": 1, "AGE_GROUP_0_27D": 0, "AGE_GROUP_28_364D": 1, "AGE_GROUP_U5": 1, "SOURCE_SIM": "silver_sim_individual", "QC_STATUS": "PASS"},
                {"MUNICIPALITY_CODE": "123456", "DATE": "2024-01-01", "YEAR": 2024, "MONTH": 1, "AGE_GROUP_0_27D": 0, "AGE_GROUP_28_364D": 1, "AGE_GROUP_U5": 1, "SOURCE_SIM": "silver_sim_individual", "QC_STATUS": "PASS"},
                {"MUNICIPALITY_CODE": "123456", "DATE": "2024-01-02", "YEAR": 2024, "MONTH": 1, "AGE_GROUP_0_27D": 0, "AGE_GROUP_28_364D": 1, "AGE_GROUP_U5": 1, "SOURCE_SIM": "silver_sim_individual", "QC_STATUS": "PASS"},
                {"MUNICIPALITY_CODE": "123456", "DATE": "2024-01-02", "YEAR": 2024, "MONTH": 1, "AGE_GROUP_0_27D": 0, "AGE_GROUP_28_364D": 0, "AGE_GROUP_U5": 1, "SOURCE_SIM": "silver_sim_individual", "QC_STATUS": "PASS"},
                {"MUNICIPALITY_CODE": "123456", "DATE": "2024-01-02", "YEAR": 2024, "MONTH": 1, "AGE_GROUP_0_27D": 0, "AGE_GROUP_28_364D": 0, "AGE_GROUP_U5": 1, "SOURCE_SIM": "silver_sim_individual", "QC_STATUS": "PASS"},
                {"MUNICIPALITY_CODE": "123456", "DATE": "2024-01-02", "YEAR": 2024, "MONTH": 1, "AGE_GROUP_0_27D": 0, "AGE_GROUP_28_364D": 0, "AGE_GROUP_U5": 1, "SOURCE_SIM": "silver_sim_individual", "QC_STATUS": "PASS"},
            ]
        )
        _write_parquet(
            self.root / "data" / "silver" / "sim" / "silver_sim_individual" / "YEAR=2024" / "silver_sim_individual_2024.parquet",
            sim_individual,
        )
        child_sih = pd.DataFrame(
            [
                {
                    "AGE_SCOPE": "AGE_U5",
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "ADMISSIONS_TOTAL": 5,
                    "PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL": 1,
                    "PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_TOTAL": 2,
                    "PRINCIPAL_DIAGNOSIS_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL": 3,
                    "ANY_DIAGNOSIS_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL": 4,
                },
                {
                    "AGE_SCOPE": "AGE_U5",
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-02",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "ADMISSIONS_TOTAL": 6,
                    "PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL": 2,
                    "PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_TOTAL": 1,
                    "PRINCIPAL_DIAGNOSIS_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL": 0,
                    "ANY_DIAGNOSIS_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL": 1,
                },
            ]
        )
        _write_parquet(
            self.root
            / "data"
            / "silver"
            / "analytical"
            / "silver_child_u5_sih_daily"
            / "YEAR=2024"
            / "silver_child_u5_sih_daily_2024.parquet",
            child_sih,
        )
        sinasc = pd.DataFrame(
            [
                {"MUNICIPALITY_CODE": "123456", "DATE": "2024-01-01", "YEAR": 2024, "LIVE_BIRTHS_TOTAL": 10},
                {"MUNICIPALITY_CODE": "123456", "DATE": "2024-01-02", "YEAR": 2024, "LIVE_BIRTHS_TOTAL": 15},
            ]
        )
        _write_parquet(
            self.root / "data" / "silver" / "sinasc" / "silver_sinasc_daily" / "YEAR=2024" / "silver_sinasc_daily_2024.parquet",
            sinasc,
        )
        population = pd.DataFrame([{"MUNICIPALITY_CODE": "123456", "YEAR": 2024, "POP_UNDER5": 100}])
        _write_parquet(
            self.root / "data" / "silver" / "population" / "silver_pop_under5" / "YEAR=2024" / "silver_pop_under5_2024.parquet",
            population,
        )

        result = build_gold_mortality_year(self.root, self.runtime_root, 2024)
        output = pd.read_parquet(result["output_path"])

        self.assertEqual(len(output), 2)
        self.assertEqual(int(output.loc[output["DATE"] == "2024-01-01", "INFANT_DEATHS_TOTAL"].iloc[0]), 3)
        self.assertEqual(int(output.loc[output["DATE"] == "2024-01-01", "NEONATAL_DEATHS_TOTAL"].iloc[0]), 1)
        self.assertTrue((output["DEATHS_NEONATAL"] == output["NEONATAL_DEATHS_TOTAL"]).all())
        self.assertTrue((output["DEATHS_U1"] == output["INFANT_DEATHS_TOTAL"]).all())
        self.assertTrue((output["LIVE_BIRTHS_TOTAL_ANNUAL"] == 25).all())
        self.assertTrue((output["POP_U5"] == 100).all())
        self.assertTrue((output["OFFSET_LOG_LIVE_BIRTHS"] == math.log(25)).all())
        self.assertTrue((output["OFFSET_LOG_UNDER5_POPULATION"] == math.log(100)).all())
        self.assertTrue((output["DENOMINATOR_ZERO_FLAG"] == 0).all())
        self.assertTrue((output["RH_MIN_PCT"] == pd.Series([42.0, 44.0])).all())
        self.assertEqual(int(output["UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_TOTAL"].sum()), 1)
        self.assertEqual(int(output["ANY_MENTION_EXCESSIVE_NATURAL_HEAT_X30_TOTAL"].sum()), 1)
        self.assertEqual(int(output["PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL"].sum()), 3)
        self.assertEqual(int(output["PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_TOTAL"].sum()), 3)
        self.assertEqual(int(output["PRINCIPAL_DIAGNOSIS_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL"].sum()), 3)
        self.assertEqual(int(output["ANY_DIAGNOSIS_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL"].sum()), 5)
        self.assertTrue(output["OFFSET_VALUE"].isna().all())

    def test_build_gold_mortality_year_excludes_placeholder_municipality_codes(self) -> None:
        self._write_common_climate()
        cause_defaults = {
            "UNDERLYING_BASIC_CAUSE_CHILD_HEAT_THERMAL_PRIMARY_TOTAL": 0,
            "UNDERLYING_BASIC_CAUSE_CHILD_DEHYDRATION_PRIMARY_TOTAL": 0,
            "ANY_MENTION_CHILD_FLUID_ELECTROLYTE_SENSITIVITY_TOTAL": 0,
            "UNDERLYING_BASIC_CAUSE_CHILD_DIARRHEA_GI_PRIMARY_TOTAL": 0,
            "UNDERLYING_BASIC_CAUSE_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL": 0,
            "ANY_MENTION_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL": 0,
            "ANY_MENTION_CHILD_RENAL_ACUTE_SENSITIVITY_TOTAL": 0,
            "ANY_MENTION_CHILD_MALNUTRITION_SLOW_MOVING_SENSITIVITY_TOTAL": 0,
            "ANY_MENTION_EXCESSIVE_NATURAL_HEAT_X30_TOTAL": 0,
            "UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_TOTAL": 0,
            "ANY_MENTION_DIRECT_HEAT_ANY_TOTAL": 0,
            "UNDERLYING_BASIC_CAUSE_INDIRECT_HEAT_ANY_TOTAL": 0,
            "ANY_MENTION_INDIRECT_HEAT_ANY_TOTAL": 0,
            "UNDERLYING_BASIC_CAUSE_SLOW_MOVING_HEAT_ANY_TOTAL": 0,
            "ANY_MENTION_SLOW_MOVING_HEAT_ANY_TOTAL": 0,
            "POP_UNDER5": 100,
            "DENOMINATOR_NAME": "POP_UNDER5",
            "DENOMINATOR_VALUE": 100,
            "SOURCE_ANALYTICAL_VIEW": "silver_child_u5_sim_daily",
            "SOURCE_SIM": "silver_sim_individual",
            "QC_STATUS": "PASS",
        }
        child_sim = pd.DataFrame(
            [
                {"AGE_SCOPE": "AGE_0_27D", "MUNICIPALITY_CODE": "123456", "DATE": "2024-01-01", "YEAR": 2024, "MONTH": 1, "DEATHS_TOTAL": 1, **cause_defaults},
                {"AGE_SCOPE": "AGE_U5", "MUNICIPALITY_CODE": "123456", "DATE": "2024-01-01", "YEAR": 2024, "MONTH": 1, "DEATHS_TOTAL": 1, **cause_defaults},
                {"AGE_SCOPE": "AGE_0_27D", "MUNICIPALITY_CODE": "130000", "DATE": "2024-01-01", "YEAR": 2024, "MONTH": 1, "DEATHS_TOTAL": 1, **cause_defaults},
                {"AGE_SCOPE": "AGE_U5", "MUNICIPALITY_CODE": "130000", "DATE": "2024-01-01", "YEAR": 2024, "MONTH": 1, "DEATHS_TOTAL": 1, **cause_defaults},
            ]
        )
        _write_parquet(
            self.root
            / "data"
            / "silver"
            / "analytical"
            / "silver_child_u5_sim_daily"
            / "YEAR=2024"
            / "silver_child_u5_sim_daily_2024.parquet",
            child_sim,
        )
        sim_individual = pd.DataFrame(
            [
                {"MUNICIPALITY_CODE": "123456", "DATE": "2024-01-01", "YEAR": 2024, "MONTH": 1, "AGE_GROUP_0_27D": 1, "AGE_GROUP_28_364D": 0, "AGE_GROUP_U5": 1, "SOURCE_SIM": "silver_sim_individual", "QC_STATUS": "PASS"},
                {"MUNICIPALITY_CODE": "130000", "DATE": "2024-01-01", "YEAR": 2024, "MONTH": 1, "AGE_GROUP_0_27D": 1, "AGE_GROUP_28_364D": 0, "AGE_GROUP_U5": 1, "SOURCE_SIM": "silver_sim_individual", "QC_STATUS": "PASS"},
            ]
        )
        _write_parquet(
            self.root / "data" / "silver" / "sim" / "silver_sim_individual" / "YEAR=2024" / "silver_sim_individual_2024.parquet",
            sim_individual,
        )
        child_sih = pd.DataFrame(
            [
                {
                    "AGE_SCOPE": "AGE_U5",
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "ADMISSIONS_TOTAL": 2,
                    "PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL": 1,
                    "PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_TOTAL": 0,
                    "PRINCIPAL_DIAGNOSIS_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL": 1,
                    "ANY_DIAGNOSIS_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL": 1,
                },
                {
                    "AGE_SCOPE": "AGE_U5",
                    "MUNICIPALITY_CODE": "130000",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "ADMISSIONS_TOTAL": 3,
                    "PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL": 1,
                    "PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_TOTAL": 1,
                    "PRINCIPAL_DIAGNOSIS_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL": 1,
                    "ANY_DIAGNOSIS_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL": 1,
                },
            ]
        )
        _write_parquet(
            self.root
            / "data"
            / "silver"
            / "analytical"
            / "silver_child_u5_sih_daily"
            / "YEAR=2024"
            / "silver_child_u5_sih_daily_2024.parquet",
            child_sih,
        )
        sinasc = pd.DataFrame(
            [
                {"MUNICIPALITY_CODE": "123456", "DATE": "2024-01-01", "YEAR": 2024, "LIVE_BIRTHS_TOTAL": 10},
                {"MUNICIPALITY_CODE": "130000", "DATE": "2024-01-01", "YEAR": 2024, "LIVE_BIRTHS_TOTAL": 8},
            ]
        )
        _write_parquet(
            self.root / "data" / "silver" / "sinasc" / "silver_sinasc_daily" / "YEAR=2024" / "silver_sinasc_daily_2024.parquet",
            sinasc,
        )
        population = pd.DataFrame(
            [
                {"MUNICIPALITY_CODE": "123456", "YEAR": 2024, "POP_UNDER5": 100},
                {"MUNICIPALITY_CODE": "130000", "YEAR": 2024, "POP_UNDER5": 90},
            ]
        )
        _write_parquet(
            self.root / "data" / "silver" / "population" / "silver_pop_under5" / "YEAR=2024" / "silver_pop_under5_2024.parquet",
            population,
        )

        result = build_gold_mortality_year(self.root, self.runtime_root, 2024)
        output = pd.read_parquet(result["output_path"])

        self.assertEqual(output["MUNICIPALITY_CODE"].astype(str).unique().tolist(), ["123456"])
        self.assertEqual(int(result["mortality_summary"]["excluded_placeholder_rows"]), 6)
        self.assertIn("excluded_placeholder_rows=6", result["external_validation_summary"])

    def test_build_gold_panel_year_harmonizes_canonical_gold_inputs(self) -> None:
        perinatal = pd.DataFrame(
            [
                {
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "TMAX_C": 34.5,
                    "TMIN_C": 24.0,
                    "TMEAN_C": 29.0,
                    "RH_MIN_PCT": 42.0,
                    "RH_MEAN_PCT": 55.0,
                    "HEATWAVE_LEN_P95": 2,
                    "HEAT_INDEX_APPROX_C": 36.0,
                    "FONTE_CLIMA_FINAL": "mixed",
                    "FONTE_TMAX_C_FINAL": "inmet",
                    "FONTE_TMIN_C_FINAL": "era5",
                    "FONTE_TMEAN_C_FINAL": "mixed",
                    "FONTE_RH_MIN_PCT_FINAL": "era5",
                    "FONTE_RH_MEAN_PCT_FINAL": "era5",
                    "IMPUTADO_POR_FALLBACK": 0,
                    "IMPUTADO_TMAX_C": 0,
                    "IMPUTADO_TMIN_C": 0,
                    "IMPUTADO_TMEAN_C": 0,
                    "IMPUTADO_RH_MIN_PCT": 0,
                    "IMPUTADO_RH_MEAN_PCT": 0,
                    "CLIMATE_UPSTREAM_QUARANTINE_FLAG": 1,
                    "CLIMATE_RECORD_ANOMALY_FLAG": 0,
                    "CLIMATE_CORRECTION_PATH": "upstream_station_quarantine_to_era5",
                    "HEAT_INDEX_METHOD": "approx",
                    "HEAT_INDEX_SOURCE_PAIR": "mixed",
                    "HEAT_INDEX_INPUTS_ZERO_MISSING": 1,
                    "PRETERM_FLAG": 1,
                    "STILLBIRTH_TOTAL_PRIMARY_GTE28": 0,
                    "LIVE_BIRTHS_TOTAL": 10,
                    "TOTAL_BIRTHS_PRIMARY": 10,
                    "LOW_BIRTH_WEIGHT_LT2500_TOTAL": 1,
                    "PRETERM_LOW_BIRTH_WEIGHT_TOTAL": 1,
                    "PRENATAL_VISITS_8PLUS_FLAG": 0,
                    "OBSTETRIC_ADMISSIONS_TOTAL": 4,
                    "HDP_ANY_TOTAL": 2,
                    "HDP_SEVERE_TOTAL": 1,
                    "PRETERM_LABOR_TOTAL": 1,
                    "APH_ANTEPARTUM_TOTAL": 1,
                    "GDM_GESTATIONAL_TOTAL": 1,
                    "DENOMINATOR_ZERO_FLAG": 0,
                    "DENOMINATOR_MISSING_FLAG": 0,
                    "RATE_NOT_COMPUTABLE_FLAG": 0,
                    "OFFSET_VALID_FLAG": 1,
                    "QC_STATUS": "PASS",
                },
                {
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-02",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "TMAX_C": 35.0,
                    "TMIN_C": 25.0,
                    "TMEAN_C": 30.0,
                    "RH_MIN_PCT": 44.0,
                    "RH_MEAN_PCT": 57.0,
                    "HEATWAVE_LEN_P95": 3,
                    "HEAT_INDEX_APPROX_C": 37.5,
                    "FONTE_CLIMA_FINAL": "mixed",
                    "FONTE_TMAX_C_FINAL": "era5",
                    "FONTE_TMIN_C_FINAL": "era5",
                    "FONTE_TMEAN_C_FINAL": "era5",
                    "FONTE_RH_MIN_PCT_FINAL": "era5",
                    "FONTE_RH_MEAN_PCT_FINAL": "era5",
                    "IMPUTADO_POR_FALLBACK": 0,
                    "IMPUTADO_TMAX_C": 0,
                    "IMPUTADO_TMIN_C": 0,
                    "IMPUTADO_TMEAN_C": 0,
                    "IMPUTADO_RH_MIN_PCT": 0,
                    "IMPUTADO_RH_MEAN_PCT": 0,
                    "CLIMATE_UPSTREAM_QUARANTINE_FLAG": 0,
                    "CLIMATE_RECORD_ANOMALY_FLAG": 1,
                    "CLIMATE_CORRECTION_PATH": "record_anomaly_neighbor_imputation",
                    "HEAT_INDEX_METHOD": "approx",
                    "HEAT_INDEX_SOURCE_PAIR": "mixed",
                    "HEAT_INDEX_INPUTS_ZERO_MISSING": 1,
                    "PRETERM_FLAG": 0,
                    "STILLBIRTH_TOTAL_PRIMARY_GTE28": 1,
                    "LIVE_BIRTHS_TOTAL": 12,
                    "TOTAL_BIRTHS_PRIMARY": 13,
                    "LOW_BIRTH_WEIGHT_LT2500_TOTAL": 0,
                    "PRETERM_LOW_BIRTH_WEIGHT_TOTAL": 0,
                    "PRENATAL_VISITS_8PLUS_FLAG": 1,
                    "OBSTETRIC_ADMISSIONS_TOTAL": 5,
                    "HDP_ANY_TOTAL": 1,
                    "HDP_SEVERE_TOTAL": 0,
                    "PRETERM_LABOR_TOTAL": 0,
                    "APH_ANTEPARTUM_TOTAL": 0,
                    "GDM_GESTATIONAL_TOTAL": 1,
                    "DENOMINATOR_ZERO_FLAG": 0,
                    "DENOMINATOR_MISSING_FLAG": 0,
                    "RATE_NOT_COMPUTABLE_FLAG": 0,
                    "OFFSET_VALID_FLAG": 1,
                    "QC_STATUS": "PASS",
                },
            ]
        )
        _write_parquet(
            self.root / "data" / "gold" / "gold_mnch_perinatal" / "YEAR=2024" / "gold_mnch_perinatal_2024.parquet",
            perinatal,
        )

        mortality = pd.DataFrame(
            [
                {
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "DEATHS_NEONATAL": 1,
                    "DEATHS_U1": 2,
                    "DEATHS_U5": 3,
                    "LIVE_BIRTHS_TOTAL_ANNUAL": 22,
                    "POP_U5": 100,
                    "ACUTE_CARE_PROXY_TOTAL": 6,
                    "DENOMINATOR_ZERO_FLAG": 0,
                    "DENOMINATOR_MISSING_FLAG": 0,
                    "RATE_NOT_COMPUTABLE_FLAG": 0,
                    "OFFSET_VALID_FLAG": 1,
                    "ACUTE_CARE_PROXY_LABEL": "AGE_U5_SIH_ADMISSIONS_PROXY",
                    "QC_STATUS": "PASS",
                },
                {
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-02",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "DEATHS_NEONATAL": 0,
                    "DEATHS_U1": 1,
                    "DEATHS_U5": 2,
                    "LIVE_BIRTHS_TOTAL_ANNUAL": 22,
                    "POP_U5": 100,
                    "ACUTE_CARE_PROXY_TOTAL": 5,
                    "DENOMINATOR_ZERO_FLAG": 0,
                    "DENOMINATOR_MISSING_FLAG": 0,
                    "RATE_NOT_COMPUTABLE_FLAG": 0,
                    "OFFSET_VALID_FLAG": 1,
                    "ACUTE_CARE_PROXY_LABEL": "AGE_U5_SIH_ADMISSIONS_PROXY",
                    "QC_STATUS": "PASS",
                },
            ]
        )
        _write_parquet(
            self.root / "data" / "gold" / "gold_mnch_mortality" / "YEAR=2024" / "gold_mnch_mortality_2024.parquet",
            mortality,
        )

        wra = pd.DataFrame([{"MUNICIPALITY_CODE": "123456", "YEAR": 2024, "POP_WRA_15_49": 200}])
        _write_parquet(
            self.root / "data" / "silver" / "population" / "silver_pop_wra_15_49" / "YEAR=2024" / "silver_pop_wra_15_49_2024.parquet",
            wra,
        )

        result = build_gold_panel_year(self.root, self.runtime_root, 2024)
        output = pd.read_parquet(result["output_path"])

        self.assertEqual(len(output), 2)
        self.assertEqual(int(output["PRETERM_BIRTHS_CANONICAL_TOTAL"].sum()), 1)
        self.assertEqual(int(output["STILLBIRTH_TOTAL_PRIMARY_GTE28"].sum()), 1)
        self.assertTrue((output["REF_LIVE_BIRTHS_TOTAL_ANNUAL"] == 22).all())
        self.assertTrue((output["REF_POP_U5"] == 100).all())
        self.assertTrue((output["REF_POP_WRA_15_49"] == 200).all())
        self.assertTrue((output["DENOMINATOR_ZERO_FLAG"] == 0).all())
        self.assertTrue((output["DENOMINATOR_MISSING_FLAG"] == 0).all())
        self.assertTrue((output["RATE_NOT_COMPUTABLE_FLAG"] == 0).all())
        self.assertTrue((output["OFFSET_VALID_FLAG"] == 1).all())
        self.assertNotIn("WRA_15_49_OFFSET_LOG", output.columns)
        self.assertNotIn("QC_STATUS_x", output.columns)
        self.assertNotIn("QC_STATUS_y", output.columns)
        self.assertEqual(
            str(output.loc[output["DATE"] == "2024-01-01", "PANEL_ROLE_NOTE"].iloc[0]),
            "MUNICIPALITY_DAY_COMPARATIVE_MART_CANONICAL_ENDPOINTS_REMAIN_IN_SOURCE_GOLD_TABLES",
        )
        self.assertEqual(int(result["panel_summary"]["total_municipality_days"]), 2)
        self.assertEqual(int(result["panel_summary"]["municipalities_with_full_daily_coverage"]), 0)
        self.assertTrue(all(result["panel_summary"]["zero_missing_confirmation_required_climate_linkage_fields"].values()))

    def test_build_gold_analysis_child_u5_year_adds_explicit_age_logic(self) -> None:
        mortality = pd.DataFrame(
            [
                {
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "DEATHS_NEONATAL": 1,
                    "DEATHS_U1": 3,
                    "DEATHS_U5": 5,
                    "NEONATAL_DEATHS_TOTAL": 1,
                    "POSTNEONATAL_DEATHS_TOTAL": 2,
                    "INFANT_DEATHS_TOTAL": 3,
                    "UNDER5_DEATHS_TOTAL": 5,
                    "LIVE_BIRTHS_TOTAL_ANNUAL": 100,
                    "POP_U5": 500,
                    "ACUTE_CARE_PROXY_TOTAL": 7,
                    "PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL": 2,
                    "PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_TOTAL": 1,
                    "PRINCIPAL_DIAGNOSIS_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL": 3,
                    "ANY_DIAGNOSIS_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL": 4,
                    "TMAX_C": 33.0,
                    "RH_MEAN_PCT": 50.0,
                    "HEATWAVE_LEN_P95": 1,
                    "HEAT_INDEX_APPROX_C": 34.0,
                    "DENOMINATOR_ZERO_FLAG": 0,
                    "DENOMINATOR_MISSING_FLAG": 0,
                    "RATE_NOT_COMPUTABLE_FLAG": 0,
                    "OFFSET_VALID_FLAG": 1,
                    "QC_STATUS": "PASS",
                }
            ]
        )
        _write_parquet(
            self.root / "data" / "gold" / "gold_mnch_mortality" / "YEAR=2024" / "gold_mnch_mortality_2024.parquet",
            mortality,
        )

        result = build_gold_analysis_child_u5_year(self.root, self.runtime_root, 2024)
        output = pd.read_parquet(result["output_path"])

        self.assertEqual(int(output["DEATHS_0_27D"].sum()), 1)
        self.assertEqual(int(output["DEATHS_28_364D"].sum()), 2)
        self.assertEqual(int(output["DEATHS_1_4Y"].sum()), 2)
        self.assertEqual(int(output["DEATHS_UNDER5_TOTAL"].sum()), 5)
        self.assertEqual(int(output["PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL"].sum()), 2)
        self.assertEqual(int(output["ANY_DIAGNOSIS_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL"].sum()), 4)
        self.assertIn("0_59_MONTHS", str(output["CHILD_VIEW_AGE_LOGIC_NOTE"].iloc[0]))
        self.assertTrue(all(result["derived_validation_flags"].values()))

    def test_build_gold_analysis_wra_15_45_sens_year_labels_sensitivity(self) -> None:
        perinatal = pd.DataFrame(
            [
                {
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "PRETERM_FLAG": 1,
                    "OBSTETRIC_ADMISSIONS_TOTAL": 4,
                    "HDP_ANY_TOTAL": 2,
                    "PRETERM_LABOR_TOTAL": 1,
                    "GDM_GESTATIONAL_TOTAL": 1,
                    "DENOMINATOR_ZERO_FLAG": 0,
                    "DENOMINATOR_MISSING_FLAG": 0,
                    "RATE_NOT_COMPUTABLE_FLAG": 0,
                    "OFFSET_VALID_FLAG": 1,
                    "TMAX_C": 34.0,
                    "RH_MEAN_PCT": 52.0,
                    "HEATWAVE_LEN_P95": 2,
                    "HEAT_INDEX_APPROX_C": 35.0,
                    "QC_STATUS": "PASS",
                }
            ]
        )
        _write_parquet(
            self.root / "data" / "gold" / "gold_mnch_perinatal" / "YEAR=2024" / "gold_mnch_perinatal_2024.parquet",
            perinatal,
        )
        pop = pd.DataFrame([{"MUNICIPALITY_CODE": "123456", "YEAR": 2024, "POP_WRA_15_45_SENSITIVITY": 150}])
        _write_parquet(
            self.root
            / "data"
            / "silver"
            / "population"
            / "silver_pop_wra_15_45_sensitivity"
            / "YEAR=2024"
            / "silver_pop_wra_15_45_sensitivity_2024.parquet",
            pop,
        )

        result = build_gold_analysis_wra_15_45_sens_year(self.root, self.runtime_root, 2024)
        output = pd.read_parquet(result["output_path"])

        self.assertEqual(int(output["SENSITIVITY_ONLY_FLAG"].iloc[0]), 1)
        self.assertEqual(str(output["WRA_AGE_SCOPE_NOTE"].iloc[0]), "WOMEN_15_45_SENSITIVITY_ONLY_VIEW")
        self.assertEqual(int(output["POP_WRA_15_45_SENSITIVITY"].iloc[0]), 150)
        self.assertTrue(all(result["derived_validation_flags"].values()))

    def test_build_gold_analysis_wra_15_45_sens_year_excludes_placeholder_municipality_codes(self) -> None:
        perinatal = pd.DataFrame(
            [
                {
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "PRETERM_FLAG": 1,
                    "OBSTETRIC_ADMISSIONS_TOTAL": 4,
                    "HDP_ANY_TOTAL": 2,
                    "PRETERM_LABOR_TOTAL": 1,
                    "GDM_GESTATIONAL_TOTAL": 1,
                    "DENOMINATOR_ZERO_FLAG": 0,
                    "DENOMINATOR_MISSING_FLAG": 0,
                    "RATE_NOT_COMPUTABLE_FLAG": 0,
                    "OFFSET_VALID_FLAG": 1,
                    "TMAX_C": 34.0,
                    "RH_MEAN_PCT": 52.0,
                    "HEATWAVE_LEN_P95": 2,
                    "HEAT_INDEX_APPROX_C": 35.0,
                    "QC_STATUS": "PASS",
                },
                {
                    "MUNICIPALITY_CODE": "130000",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "PRETERM_FLAG": 1,
                    "OBSTETRIC_ADMISSIONS_TOTAL": 2,
                    "HDP_ANY_TOTAL": 1,
                    "PRETERM_LABOR_TOTAL": 0,
                    "GDM_GESTATIONAL_TOTAL": 1,
                    "DENOMINATOR_ZERO_FLAG": 0,
                    "DENOMINATOR_MISSING_FLAG": 0,
                    "RATE_NOT_COMPUTABLE_FLAG": 0,
                    "OFFSET_VALID_FLAG": 1,
                    "TMAX_C": 35.0,
                    "RH_MEAN_PCT": 55.0,
                    "HEATWAVE_LEN_P95": 1,
                    "HEAT_INDEX_APPROX_C": 36.0,
                    "QC_STATUS": "PASS",
                },
            ]
        )
        _write_parquet(
            self.root / "data" / "gold" / "gold_mnch_perinatal" / "YEAR=2024" / "gold_mnch_perinatal_2024.parquet",
            perinatal,
        )
        pop = pd.DataFrame(
            [
                {"MUNICIPALITY_CODE": "123456", "YEAR": 2024, "POP_WRA_15_45_SENSITIVITY": 150},
                {"MUNICIPALITY_CODE": "130000", "YEAR": 2024, "POP_WRA_15_45_SENSITIVITY": 120},
            ]
        )
        _write_parquet(
            self.root
            / "data"
            / "silver"
            / "population"
            / "silver_pop_wra_15_45_sensitivity"
            / "YEAR=2024"
            / "silver_pop_wra_15_45_sensitivity_2024.parquet",
            pop,
        )

        result = build_gold_analysis_wra_15_45_sens_year(self.root, self.runtime_root, 2024)
        output = pd.read_parquet(result["output_path"])

        self.assertEqual(output["MUNICIPALITY_CODE"].astype(str).unique().tolist(), ["123456"])
        self.assertEqual(int(result["derived_summary"]["excluded_placeholder_rows"]), 2)
        self.assertIn("excluded_placeholder_rows=2", result["external_validation_summary"])


if __name__ == "__main__":
    unittest.main()
