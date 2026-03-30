from __future__ import annotations

import argparse
import os
from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from math import exp, log
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from heat_mnch.analysis_steps.aliases import validate_public_output_columns
from heat_mnch.analysis_steps.registry import analysis_shared_marts_root
from heat_mnch.run_log import append_run_event, init_run_log, utc_now


ROUND10_5_RESULTS_DIR = Path("artifacts/analysis/step9_rebuild/results/round10_5")
ROUND10_5_REGISTRY_DIR = Path("artifacts/analysis/step9_rebuild/registries")
ROUND10_5_REVIEW_DIR = Path("artifacts/analysis/step9_rebuild/reviews")

SHARED_MART_INVENTORY_PATH = Path("artifacts/analysis/shared_marts/analysis_mart_inventory.csv")
SHARED_MART_DICT_PATH = Path("artifacts/analysis/shared_marts/shared_mart_variable_dictionary.csv")
SHARED_MART_CROSSWALK_PATH = Path("artifacts/analysis/shared_marts/shared_mart_source_crosswalk.csv")

ROUND10_SYNTHESIS_REGISTRY_PATH = Path(
    "artifacts/analysis/step9_rebuild/registries/round10_domain_synthesis_registry.csv"
)
ROUND10_LINKED_SYNTHESIS_PATH = Path(
    "artifacts/analysis/step9_rebuild/results/round10/round10_execution_linked_synthesis_table.csv"
)
ROUND10_LEDGER_PATH = Path(
    "artifacts/analysis/step9_rebuild/registries/round10_execution_ledger.csv"
)
ROUND8_TRANSLATION_ELIGIBILITY_PATH = Path(
    "artifacts/analysis/step9_rebuild/registries/round8_translation_eligibility_registry.csv"
)
ROUND_REGISTRY_PATH = Path("artifacts/analysis/step9_rebuild/registries/round_registry.csv")
GATE_LOG_PATH = Path("artifacts/analysis/step9_rebuild/gates/gate_verdict_log.csv")
REFERENCE_SUFFICIENCY_PATH = Path(
    "artifacts/analysis/step9_rebuild/references/round_reference_sufficiency_log.csv"
)

ROUND10_MEMO_PATH = Path(
    "reports/publications/incremental_analysis_round/step9_rebuild/12_round10_scientific_synthesis.md"
)
ROUND10_5_MEMO_PATH = Path(
    "reports/publications/incremental_analysis_round/step9_rebuild/12a_round10_5_flagship_execution_surface_materialization.md"
)
ROUND11_MEMO_PATH = Path(
    "reports/publications/incremental_analysis_round/step9_rebuild/13_round11_journal_package.md"
)

ROUND10_FLAGSHIP_SYNTHESIS_MEMO_PATH = Path(
    "artifacts/analysis/step9_rebuild/reviews/round10_flagship_synthesis.md"
)
ROUND10_5_PRETERM_ENGINE_MEMO_PATH = Path(
    "artifacts/analysis/step9_rebuild/reviews/round10_5_preterm_engine_validation.md"
)

SINASC_GLOB = "data/silver/sinasc/silver_sinasc_event/**/*.parquet"
STILLBIRTH_GLOB = "data/silver/stillbirth/silver_stillbirth_individual/**/*.parquet"
SIM_GLOB = "data/silver/sim/silver_sim_individual/**/*.parquet"
CLIMATE_GLOB = "data/gold/gold_mnch_perinatal/**/*.parquet"
CONTEXT_PATH = "data/gold/gold_municipality_context_static/gold_municipality_context_static.parquet"

GESTATIONAL_MART_NAME = "flagship_gestational_weekly_person_period_mart"
STILLBIRTH_MART_NAME = "flagship_stillbirth_fetuses_at_risk_weekly_mart"
NEONATAL_MART_NAME = "flagship_neonatal_bridge_execution_mart"

FLAGSHIP_LEFT_TRUNCATION_START = date(2020, 1, 1)
FLAGSHIP_RIGHT_TRUNCATION_END = date(2024, 12, 31)
PRIMARY_EXPOSURE_BIN_DECIMALS = 1
STATE_PREFIXES = (
    "11", "12", "13", "14", "15", "16", "17",
    "21", "22", "23", "24", "25", "26", "27", "28", "29",
    "31", "32", "33", "35",
    "41", "42", "43",
    "50", "51", "52", "53",
)

MACROREGION_CASE_SQL = """
CASE
    WHEN SUBSTR(CAST(municipality_code AS VARCHAR), 1, 2) IN ('11','12','13','14','15','16','17') THEN 'North'
    WHEN SUBSTR(CAST(municipality_code AS VARCHAR), 1, 2) IN ('21','22','23','24','25','26','27','28','29') THEN 'Northeast'
    WHEN SUBSTR(CAST(municipality_code AS VARCHAR), 1, 2) IN ('31','32','33','35') THEN 'Southeast'
    WHEN SUBSTR(CAST(municipality_code AS VARCHAR), 1, 2) IN ('41','42','43') THEN 'South'
    WHEN SUBSTR(CAST(municipality_code AS VARCHAR), 1, 2) IN ('50','51','52','53') THEN 'Center-West'
    ELSE 'Unknown'
END
"""


@dataclass(frozen=True)
class MartMetadataRow:
    mart_name: str
    academic_variable_name: str
    academic_label: str
    description: str
    units: str
    public_role: str
    outcome_family: str
    exposure_family: str
    provenance_only: bool
    source_dataset: str
    source_columns: str
    canonical_field_reference: str
    derivation_rule: str


@dataclass(frozen=True)
class ReplaySpec:
    domain_id: str
    mart_name: str
    primary_estimator_id: str
    outcome_column: str
    exposure_column: str


REPLAY_SPECS: tuple[ReplaySpec, ...] = (
    ReplaySpec(
        domain_id="key_secondary_stillbirth",
        mart_name=STILLBIRTH_MART_NAME,
        primary_estimator_id="e003_grouped_cause_specific_hazard",
        outcome_column="stillbirth_event_count",
        exposure_column="tmax_weekly_mean",
    ),
    ReplaySpec(
        domain_id="key_secondary_neonatal",
        mart_name=NEONATAL_MART_NAME,
        primary_estimator_id="e004_piecewise_poisson_bridge",
        outcome_column="neonatal_death_count",
        exposure_column="tmax_exposure_value",
    ),
)


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _mart_partition_path(root: Path, mart_name: str, year: int) -> Path:
    return analysis_shared_marts_root(root) / mart_name / f"YEAR={year}" / f"{mart_name}_{year}.parquet"


def _state_month_partition_path(root: Path, mart_name: str, year: int, month: int, state_prefix: str) -> Path:
    return (
        analysis_shared_marts_root(root)
        / mart_name
        / f"YEAR={year}"
        / f"{mart_name}_{year}_UF{state_prefix}_M{month:02d}.parquet"
    )


def _month_partition_path(root: Path, mart_name: str, year: int, month: int) -> Path:
    return (
        analysis_shared_marts_root(root)
        / mart_name
        / f"YEAR={year}"
        / f"{mart_name}_{year}_M{month:02d}.parquet"
    )


def _con() -> duckdb.DuckDBPyConnection:
    temp_dir = Path.cwd() / "artifacts" / "tmp" / "duckdb" / f"pid_{os.getpid()}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA threads=1")
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET memory_limit = '3GB'")
    con.execute("PRAGMA max_temp_directory_size='5GiB'")
    con.execute(f"SET temp_directory = '{temp_dir}'")
    return con


def _safe_log(value: float) -> float | None:
    if value is None or pd.isna(value) or value <= 0:
        return None
    return float(log(float(value)))


def _context_view_sql() -> str:
    return f"""
    WITH context_raw AS (
        SELECT
            CAST(MUNICIPALITY_CODE_IBGE7 AS BIGINT) AS municipality_code,
            CAST(IDH_EXPANDED AS DOUBLE) AS idh_expanded,
            CAST(GINI_EXPANDED AS DOUBLE) AS gini_expanded
        FROM read_parquet('{CONTEXT_PATH}')
        WHERE MUNICIPALITY_CODE_IBGE7 IS NOT NULL
    ),
    context_z AS (
        SELECT
            municipality_code,
            idh_expanded,
            gini_expanded,
            (idh_expanded - AVG(idh_expanded) OVER ())
                / NULLIF(STDDEV_POP(idh_expanded) OVER (), 0) AS idh_z,
            (gini_expanded - AVG(gini_expanded) OVER ())
                / NULLIF(STDDEV_POP(gini_expanded) OVER (), 0) AS gini_z
        FROM context_raw
    )
    SELECT * FROM context_z
    """


def _write_parquet_query(con: duckdb.DuckDBPyConnection, query: str, output_path: Path) -> None:
    _ensure_parent(output_path)
    con.execute(f"COPY ({query}) TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")


def _parquet_shape(path: Path) -> tuple[int, int, list[str]]:
    parquet_file = pq.ParquetFile(path)
    return parquet_file.metadata.num_rows, len(parquet_file.schema.names), parquet_file.schema.names


def _parquet_dataset_shape(paths: list[Path]) -> tuple[int, int, list[str]]:
    if not paths:
        raise ValueError("No parquet files were materialized")
    total_rows = 0
    schema_names: list[str] | None = None
    for path in paths:
        row_count, column_count, columns = _parquet_shape(path)
        total_rows += row_count
        if schema_names is None:
            schema_names = columns
        elif columns != schema_names:
            raise ValueError(f"Inconsistent schema across parquet partitions for {paths[0].parent}")
    return total_rows, column_count, schema_names or []


def _gestational_build_query(
    year: int,
    delivery_month: int | None = None,
    state_prefix: str | None = None,
) -> str:
    climate_year_start = max(2020, year - 1)
    month_filter_sql = f"AND MONTH = {delivery_month}" if delivery_month is not None else ""
    state_filter_sql = (
        f"AND SUBSTR(CAST(CODMUNRES AS VARCHAR), 1, 2) = '{state_prefix}'"
        if state_prefix is not None
        else ""
    )
    climate_state_filter_sql = (
        f"AND SUBSTR(CAST(MUNICIPALITY_CODE AS VARCHAR), 1, 2) = '{state_prefix}'"
        if state_prefix is not None
        else ""
    )
    return f"""
    WITH context_z AS ({_context_view_sql()}),
    climate AS (
        SELECT
            CAST(SUBSTR(CAST(MUNICIPALITY_CODE AS VARCHAR), 1, 6) AS BIGINT) AS municipality_code,
            DATE::DATE AS climate_date,
            CAST(AVG(TMAX_C) AS DOUBLE) AS tmax_c,
            CAST(AVG(HEAT_INDEX_APPROX_C) AS DOUBLE) AS heat_index_approx_c,
            CAST(AVG(HEATWAVE_LEN_P95) AS DOUBLE) AS heatwave_len_p95,
            CAST(AVG(RH_MEAN_PCT) AS DOUBLE) AS rh_mean_pct,
            STRING_AGG(DISTINCT CAST(FONTE_TMAX_C_FINAL AS VARCHAR), '|' ORDER BY CAST(FONTE_TMAX_C_FINAL AS VARCHAR)) AS fonte_tmax_c_final,
            STRING_AGG(DISTINCT CAST(FONTE_RH_MEAN_PCT_FINAL AS VARCHAR), '|' ORDER BY CAST(FONTE_RH_MEAN_PCT_FINAL AS VARCHAR)) AS fonte_rh_mean_pct_final,
            STRING_AGG(DISTINCT CAST(CLIMATE_CORRECTION_PATH AS VARCHAR), '|' ORDER BY CAST(CLIMATE_CORRECTION_PATH AS VARCHAR)) AS climate_correction_path,
            MAX(CAST(IMPUTADO_TMAX_C AS INTEGER)) AS imputado_tmax_c
        FROM read_parquet('{CLIMATE_GLOB}', union_by_name=true)
        WHERE YEAR BETWEEN {climate_year_start} AND {year}
          {climate_state_filter_sql}
        GROUP BY 1, 2
    ),
    births AS (
        SELECT
            CAST(SINASC_EVENT_ID AS VARCHAR) AS pregnancy_id,
            DATE::DATE AS delivery_date,
            CAST(YEAR AS INTEGER) AS birth_record_year,
            CAST(CODMUNRES AS BIGINT) AS municipality_code,
            SUBSTR(CAST(CODMUNRES AS VARCHAR), 1, 2) AS state_code,
            CAST(GA_DAYS_FINAL AS INTEGER) AS ga_days_final,
            CAST(GA_WEEKS_COMPLETED_FINAL AS INTEGER) AS ga_weeks_completed_final,
            CAST(GA_DAYS_APPROX_FLAG AS INTEGER) AS ga_days_approx_flag,
            CAST(PRETERM_LT37_FLAG AS INTEGER) AS preterm_lt37_flag,
            CAST(PRETERM_DAYS_SHORT_OF_37W_FINAL AS INTEGER) AS gestational_days_lost_below_37w,
            CAST(MATERNAL_AGE_YEARS AS INTEGER) AS maternal_age_years,
            CAST(MATERNAL_SCHOOLING_CODE_FINAL AS VARCHAR) AS maternal_schooling_code,
            CAST(PARITY_TOTAL_PRIOR AS INTEGER) AS parity_total_prior,
            CAST(PLURALITY_CODE AS VARCHAR) AS plurality_code,
            CAST(NEWBORN_SEX_CODE AS VARCHAR) AS newborn_sex_code,
            CAST(MATERNAL_RACE_COLOR_CODE AS VARCHAR) AS maternal_race_color_code,
            (DATE::DATE - CAST(GA_DAYS_FINAL AS INTEGER) * INTERVAL 1 DAY) AS conception_date,
            ((DATE::DATE - CAST(GA_DAYS_FINAL AS INTEGER) * INTERVAL 1 DAY) + 22 * INTERVAL 7 DAY) AS risk_start_date
        FROM read_parquet('{SINASC_GLOB}', union_by_name=true)
        WHERE YEAR = {year}
          {month_filter_sql}
          {state_filter_sql}
          AND COALESCE(GA_VALID_FOR_ANALYSIS_FLAG, 0) = 1
          AND COALESCE(MULTIPLE_GESTATION_FLAG, 0) = 0
          AND CODMUNRES IS NOT NULL
          AND GA_WEEKS_COMPLETED_FINAL >= 22
    ),
    eligible AS (
        SELECT *
        FROM births
        WHERE risk_start_date >= DATE '2020-01-01'
    ),
    expanded AS (
        SELECT
            e.*,
            gs.week_completed AS gestational_week_completed,
            conception_date + gs.week_completed * INTERVAL 7 DAY AS week_start_date,
            CASE
                WHEN preterm_lt37_flag = 1 AND gs.week_completed = LEAST(ga_weeks_completed_final, 36)
                    THEN delivery_date
                ELSE conception_date + gs.week_completed * INTERVAL 7 DAY + INTERVAL 6 DAY
            END AS week_end_date,
            CASE
                WHEN gs.week_completed = LEAST(ga_weeks_completed_final, 36) THEN 1
                ELSE 0
            END AS terminal_week_flag,
            CASE
                WHEN preterm_lt37_flag = 1 AND gs.week_completed = LEAST(ga_weeks_completed_final, 36)
                    THEN 1
                ELSE 0
            END AS preterm_event_flag,
            CASE
                WHEN preterm_lt37_flag = 0 AND gs.week_completed = 36 THEN 1
                ELSE 0
            END AS censor_at_36_flag
        FROM eligible e
        CROSS JOIN range(22, LEAST(ga_weeks_completed_final, 36) + 1) AS gs(week_completed)
    )
    SELECT
        pregnancy_id,
        birth_record_year,
        delivery_date,
        conception_date,
        e.municipality_code AS municipality_code,
        e.state_code AS state_code,
        {MACROREGION_CASE_SQL.replace('municipality_code', 'e.municipality_code')} AS macroregion,
        CASE
            WHEN birth_record_year IN (2020, 2021) THEN '2020_2021'
            ELSE '2022_2024'
        END AS subperiod,
        gestational_week_completed,
        week_start_date,
        week_end_date,
        terminal_week_flag,
        preterm_event_flag,
        censor_at_36_flag,
        ga_days_final,
        ga_weeks_completed_final,
        ga_days_approx_flag,
        CAST(AVG(c.tmax_c) AS DOUBLE) AS tmax_weekly_mean,
        CAST(AVG(c.heat_index_approx_c) AS DOUBLE) AS approximate_heat_index_weekly_mean,
        CAST(AVG(c.heatwave_len_p95) AS DOUBLE) AS heatwave_duration_weekly_mean,
        CAST(AVG(c.rh_mean_pct) AS DOUBLE) AS rh_mean_pct_weekly_mean,
        COUNT(c.climate_date) AS climate_days_observed_count,
        STRING_AGG(DISTINCT c.fonte_tmax_c_final, '|' ORDER BY c.fonte_tmax_c_final) AS temperature_source_set,
        STRING_AGG(DISTINCT c.fonte_rh_mean_pct_final, '|' ORDER BY c.fonte_rh_mean_pct_final) AS humidity_source_set,
        STRING_AGG(DISTINCT c.climate_correction_path, '|' ORDER BY c.climate_correction_path) AS climate_correction_path_set,
        MAX(COALESCE(c.imputado_tmax_c, 0)) AS any_temperature_imputation_flag,
        maternal_age_years,
        maternal_schooling_code,
        parity_total_prior,
        plurality_code,
        newborn_sex_code,
        maternal_race_color_code,
        gestational_days_lost_below_37w,
        CAST(ctx.idh_z AS DOUBLE) AS idh_z,
        CAST(ctx.gini_z AS DOUBLE) AS gini_z
    FROM expanded e
    LEFT JOIN climate c
        ON c.municipality_code = e.municipality_code
       AND c.climate_date BETWEEN e.week_start_date AND e.week_end_date
    LEFT JOIN context_z ctx
        ON ctx.municipality_code = e.municipality_code
    GROUP BY ALL
    """


def _stillbirth_build_query(
    year: int,
    source_month: int | None = None,
    state_prefix: str | None = None,
) -> str:
    climate_year_start = max(2020, year - 1)
    month_filter_sql = f"AND MONTH = {source_month}" if source_month is not None else ""
    state_filter_sql = (
        f"AND SUBSTR(CAST(CODMUNRES AS VARCHAR), 1, 2) = '{state_prefix}'"
        if state_prefix is not None
        else ""
    )
    climate_state_filter_sql = (
        f"AND SUBSTR(CAST(MUNICIPALITY_CODE AS VARCHAR), 1, 2) = '{state_prefix}'"
        if state_prefix is not None
        else ""
    )
    return f"""
    WITH context_z AS ({_context_view_sql()}),
    climate AS (
        SELECT
            CAST(SUBSTR(CAST(MUNICIPALITY_CODE AS VARCHAR), 1, 6) AS BIGINT) AS municipality_code,
            DATE::DATE AS climate_date,
            CAST(AVG(TMAX_C) AS DOUBLE) AS tmax_c,
            CAST(AVG(HEAT_INDEX_APPROX_C) AS DOUBLE) AS heat_index_approx_c,
            CAST(AVG(HEATWAVE_LEN_P95) AS DOUBLE) AS heatwave_len_p95,
            CAST(AVG(RH_MEAN_PCT) AS DOUBLE) AS rh_mean_pct,
            STRING_AGG(DISTINCT CAST(FONTE_TMAX_C_FINAL AS VARCHAR), '|' ORDER BY CAST(FONTE_TMAX_C_FINAL AS VARCHAR)) AS fonte_tmax_c_final,
            STRING_AGG(DISTINCT CAST(CLIMATE_CORRECTION_PATH AS VARCHAR), '|' ORDER BY CAST(CLIMATE_CORRECTION_PATH AS VARCHAR)) AS climate_correction_path,
            MAX(CAST(IMPUTADO_TMAX_C AS INTEGER)) AS imputado_tmax_c
        FROM read_parquet('{CLIMATE_GLOB}', union_by_name=true)
        WHERE YEAR BETWEEN {climate_year_start} AND {year}
          {climate_state_filter_sql}
        GROUP BY 1, 2
    ),
    live_births AS (
        SELECT
            'live_birth' AS event_type,
            DATE::DATE AS event_date,
            CAST(YEAR AS INTEGER) AS source_year,
            CAST(CODMUNRES AS BIGINT) AS municipality_code,
            CAST(GA_WEEKS_COMPLETED_FINAL AS INTEGER) AS event_week_completed,
            CAST(MATERNAL_AGE_YEARS AS INTEGER) AS maternal_age_years,
            COALESCE(CAST(MATERNAL_SCHOOLING_CODE_FINAL AS VARCHAR), 'unknown') AS maternal_schooling_code,
            COALESCE(CAST(PLURALITY_CODE AS VARCHAR), 'unknown') AS plurality_code,
            COALESCE(CAST(NEWBORN_SEX_CODE AS VARCHAR), 'unknown') AS fetal_sex_code,
            'live_birth_day_derived_week' AS timing_quality_class
        FROM read_parquet('{SINASC_GLOB}', union_by_name=true)
        WHERE YEAR = {year}
          {month_filter_sql}
          {state_filter_sql}
          AND COALESCE(GA_VALID_FOR_ANALYSIS_FLAG, 0) = 1
          AND CODMUNRES IS NOT NULL
          AND GA_WEEKS_COMPLETED_FINAL >= 22
          AND MATERNAL_AGE_YEARS IS NOT NULL
    ),
    stillbirths AS (
        SELECT
            'stillbirth' AS event_type,
            EVENT_DATE::DATE AS event_date,
            CAST(YEAR AS INTEGER) AS source_year,
            CAST(CODMUNRES AS BIGINT) AS municipality_code,
            CAST(GA_WEEKS_COMPLETED_FINAL AS INTEGER) AS event_week_completed,
            CAST(MATERNAL_AGE_YEARS AS INTEGER) AS maternal_age_years,
            COALESCE(CAST(MATERNAL_SCHOOLING_CODE_FINAL AS VARCHAR), 'unknown') AS maternal_schooling_code,
            COALESCE(CAST(PLURALITY_CODE AS VARCHAR), 'unknown') AS plurality_code,
            COALESCE(CAST(FETAL_SEX_CODE AS VARCHAR), 'unknown') AS fetal_sex_code,
            'native_week_qualified' AS timing_quality_class
        FROM read_parquet('{STILLBIRTH_GLOB}', union_by_name=true)
        WHERE YEAR = {year}
          {month_filter_sql}
          {state_filter_sql}
          AND COALESCE(GA_VALID_FOR_ANALYSIS_FLAG, 0) = 1
          AND CODMUNRES IS NOT NULL
          AND GA_WEEKS_COMPLETED_FINAL >= 22
          AND MATERNAL_AGE_YEARS IS NOT NULL
    ),
    all_events AS (
        SELECT * FROM live_births
        UNION ALL
        SELECT * FROM stillbirths
    ),
    thresholds AS (
        SELECT * FROM (VALUES
            ('primary_gte28', 28),
            ('sensitivity_gte22', 22)
        ) AS t(threshold_scenario, threshold_week)
    ),
    eligible AS (
        SELECT
            a.*,
            t.threshold_scenario,
            t.threshold_week,
            event_date - (event_week_completed - t.threshold_week + 1) * INTERVAL 7 DAY AS risk_start_date
        FROM all_events a
        CROSS JOIN thresholds t
        WHERE event_week_completed >= t.threshold_week
          AND (event_date - (event_week_completed - t.threshold_week + 1) * INTERVAL 7 DAY) >= DATE '2020-01-01'
    ),
    expanded AS (
        SELECT
            e.*,
            gs.week_completed AS gestational_week_completed,
            event_date - (event_week_completed - gs.week_completed + 1) * INTERVAL 7 DAY AS analysis_week_start_date,
            event_date - (event_week_completed - gs.week_completed) * INTERVAL 7 DAY - INTERVAL 1 DAY AS analysis_week_end_date,
            CASE WHEN gs.week_completed = event_week_completed THEN 1 ELSE 0 END AS terminal_week_flag
        FROM eligible e
        CROSS JOIN range(threshold_week, event_week_completed + 1) AS gs(week_completed)
    ),
    weekly_aggregate AS (
        SELECT
            threshold_scenario,
            source_year,
            municipality_code,
            SUBSTR(CAST(municipality_code AS VARCHAR), 1, 2) AS state_code,
            {MACROREGION_CASE_SQL} AS macroregion,
            CASE
                WHEN source_year IN (2020, 2021) THEN '2020_2021'
                ELSE '2022_2024'
            END AS subperiod,
            analysis_week_start_date,
            analysis_week_end_date,
            gestational_week_completed,
            maternal_age_years,
            maternal_schooling_code,
            plurality_code,
            fetal_sex_code,
            timing_quality_class,
            COUNT(*) AS pregnancies_at_risk_count,
            SUM(CASE WHEN terminal_week_flag = 1 AND event_type = 'stillbirth' THEN 1 ELSE 0 END) AS stillbirth_event_count,
            SUM(CASE WHEN terminal_week_flag = 1 AND event_type = 'live_birth' THEN 1 ELSE 0 END) AS live_birth_competing_event_count
        FROM expanded
        GROUP BY ALL
    )
    SELECT
        threshold_scenario,
        source_year,
        w.municipality_code AS municipality_code,
        w.state_code AS state_code,
        w.macroregion AS macroregion,
        w.subperiod AS subperiod,
        w.analysis_week_start_date AS analysis_week_start_date,
        w.analysis_week_end_date AS analysis_week_end_date,
        w.gestational_week_completed AS gestational_week_completed,
        w.maternal_age_years AS maternal_age_years,
        w.maternal_schooling_code AS maternal_schooling_code,
        w.plurality_code AS plurality_code,
        w.fetal_sex_code AS fetal_sex_code,
        w.timing_quality_class AS timing_quality_class,
        w.pregnancies_at_risk_count AS pregnancies_at_risk_count,
        w.stillbirth_event_count AS stillbirth_event_count,
        w.live_birth_competing_event_count AS live_birth_competing_event_count,
        (stillbirth_event_count + live_birth_competing_event_count) AS births_based_total_births_count,
        CAST(AVG(c.tmax_c) AS DOUBLE) AS tmax_weekly_mean,
        CAST(AVG(c.heat_index_approx_c) AS DOUBLE) AS approximate_heat_index_weekly_mean,
        CAST(AVG(c.heatwave_len_p95) AS DOUBLE) AS heatwave_duration_weekly_mean,
        CAST(AVG(c.rh_mean_pct) AS DOUBLE) AS rh_mean_pct_weekly_mean,
        COUNT(c.climate_date) AS climate_days_observed_count,
        STRING_AGG(DISTINCT c.fonte_tmax_c_final, '|' ORDER BY c.fonte_tmax_c_final) AS temperature_source_set,
        STRING_AGG(DISTINCT c.climate_correction_path, '|' ORDER BY c.climate_correction_path) AS climate_correction_path_set,
        MAX(COALESCE(c.imputado_tmax_c, 0)) AS any_temperature_imputation_flag,
        CAST(ctx.idh_z AS DOUBLE) AS idh_z,
        CAST(ctx.gini_z AS DOUBLE) AS gini_z
    FROM weekly_aggregate w
    LEFT JOIN climate c
        ON c.municipality_code = w.municipality_code
       AND c.climate_date BETWEEN w.analysis_week_start_date AND w.analysis_week_end_date
    LEFT JOIN context_z ctx
        ON ctx.municipality_code = w.municipality_code
    GROUP BY ALL
    """


def _neonatal_build_query(
    year: int,
    birth_month: int | None = None,
    state_prefix: str | None = None,
) -> str:
    climate_year_end = min(year + 1, 2024)
    if birth_month is not None:
        climate_start = date(year, birth_month, 1)
        if birth_month == 12:
            climate_end = FLAGSHIP_RIGHT_TRUNCATION_END
        else:
            next_month = birth_month + 1
            climate_end = date(year, next_month, monthrange(year, next_month)[1])
        climate_date_filter_sql = (
            f"AND DATE::DATE BETWEEN DATE '{climate_start.isoformat()}' "
            f"AND DATE '{climate_end.isoformat()}'"
        )
    else:
        climate_date_filter_sql = ""
    month_filter_sql = f"AND MONTH = {birth_month}" if birth_month is not None else ""
    birth_month_from_dtnasc_filter_sql = (
        f"AND EXTRACT('month' FROM TRY_CAST(DTNASC AS DATE)) = {birth_month}"
        if birth_month is not None
        else ""
    )
    birth_state_filter_sql = (
        f"AND SUBSTR(CAST(CODMUNNASC AS VARCHAR), 1, 2) = '{state_prefix}'"
        if state_prefix is not None
        else ""
    )
    death_state_filter_sql = (
        f"AND SUBSTR(CAST(CODMUNOCOR AS VARCHAR), 1, 2) = '{state_prefix}'"
        if state_prefix is not None
        else ""
    )
    climate_state_filter_sql = (
        f"AND SUBSTR(CAST(MUNICIPALITY_CODE AS VARCHAR), 1, 2) = '{state_prefix}'"
        if state_prefix is not None
        else ""
    )
    return f"""
    WITH context_z AS ({_context_view_sql()}),
    climate AS (
        SELECT
            CAST(SUBSTR(CAST(MUNICIPALITY_CODE AS VARCHAR), 1, 6) AS BIGINT) AS municipality_code,
            DATE::DATE AS climate_date,
            CAST(AVG(TMAX_C) AS DOUBLE) AS tmax_c,
            CAST(AVG(HEAT_INDEX_APPROX_C) AS DOUBLE) AS heat_index_approx_c,
            CAST(AVG(HEATWAVE_LEN_P95) AS DOUBLE) AS heatwave_len_p95,
            CAST(AVG(RH_MEAN_PCT) AS DOUBLE) AS rh_mean_pct,
            STRING_AGG(DISTINCT CAST(FONTE_TMAX_C_FINAL AS VARCHAR), '|' ORDER BY CAST(FONTE_TMAX_C_FINAL AS VARCHAR)) AS fonte_tmax_c_final,
            STRING_AGG(DISTINCT CAST(CLIMATE_CORRECTION_PATH AS VARCHAR), '|' ORDER BY CAST(CLIMATE_CORRECTION_PATH AS VARCHAR)) AS climate_correction_path,
            MAX(CAST(IMPUTADO_TMAX_C AS INTEGER)) AS imputado_tmax_c
        FROM read_parquet('{CLIMATE_GLOB}', union_by_name=true)
        WHERE YEAR BETWEEN {year} AND {climate_year_end}
          {climate_state_filter_sql}
          {climate_date_filter_sql}
        GROUP BY 1, 2
    ),
    birth_cohorts AS (
        SELECT
            CAST(SUBSTR(CAST(MUNICIPALITY_CODE AS VARCHAR), 1, 6) AS BIGINT) AS municipality_code_occurrence,
            DATE::DATE AS birth_cohort_date,
            CAST(YEAR AS INTEGER) AS birth_cohort_year,
            CAST(SUM(LIVE_BIRTHS_TOTAL) AS BIGINT) AS births_at_risk_count
        FROM read_parquet('{CLIMATE_GLOB}', union_by_name=true)
        WHERE YEAR = {year}
          {month_filter_sql}
          {climate_state_filter_sql}
          AND LIVE_BIRTHS_TOTAL IS NOT NULL
        GROUP BY 1, 2, 3
    ),
    exact_cells AS (
        SELECT
            b.municipality_code_occurrence,
            b.birth_cohort_date,
            b.birth_cohort_year,
            gs.day_of_life AS day_of_life_value,
            CAST(gs.day_of_life AS VARCHAR) AS day_of_life_cell,
            1 AS exact_day_flag,
            0 AS grouped_timing_flag,
            1 AS grouped_window_complete_flag,
            b.birth_cohort_date + gs.day_of_life * INTERVAL 1 DAY AS analysis_date,
            NULL::DATE AS grouped_window_end_date,
            b.births_at_risk_count
        FROM birth_cohorts b
        CROSS JOIN range(0, 28) AS gs(day_of_life)
        WHERE b.birth_cohort_date + gs.day_of_life * INTERVAL 1 DAY <= DATE '2024-12-31'
    ),
    grouped_cells AS (
        SELECT
            b.municipality_code_occurrence,
            b.birth_cohort_date,
            b.birth_cohort_year,
            NULL::INTEGER AS day_of_life_value,
            'grouped_0_27_nonexact' AS day_of_life_cell,
            0 AS exact_day_flag,
            1 AS grouped_timing_flag,
            CASE
                WHEN b.birth_cohort_date + 27 * INTERVAL 1 DAY <= DATE '2024-12-31' THEN 1
                ELSE 0
            END AS grouped_window_complete_flag,
            NULL::DATE AS analysis_date,
            LEAST(b.birth_cohort_date + 27 * INTERVAL 1 DAY, DATE '2024-12-31') AS grouped_window_end_date,
            b.births_at_risk_count
        FROM birth_cohorts b
    ),
    risk_set AS (
        SELECT * FROM exact_cells
        UNION ALL
        SELECT * FROM grouped_cells
    ),
    deaths_raw AS (
        SELECT
            CAST(CODMUNOCOR AS BIGINT) AS municipality_code_occurrence,
            TRY_CAST(DTNASC AS DATE) AS birth_cohort_date,
            CAST(EXTRACT('year' FROM TRY_CAST(DTNASC AS DATE)) AS INTEGER) AS birth_cohort_year,
            DATE::DATE AS death_date,
            CAST(AGE_UNIT_CODE AS INTEGER) AS age_unit_code,
            CAST(AGE_VALUE AS INTEGER) AS age_value
        FROM read_parquet('{SIM_GLOB}', union_by_name=true)
        WHERE YEAR BETWEEN {year} AND {min(year + 1, 2024)}
          AND COALESCE(AGE_GROUP_0_27D, 0) = 1
          AND CODMUNOCOR IS NOT NULL
          AND DTNASC IS NOT NULL
          {death_state_filter_sql}
          AND TRY_CAST(DTNASC AS DATE) BETWEEN DATE '{year}-01-01' AND DATE '{year}-12-31'
          {birth_month_from_dtnasc_filter_sql}
    ),
    exact_deaths AS (
        SELECT
            municipality_code_occurrence,
            birth_cohort_date,
            birth_cohort_year,
            CAST(age_value AS VARCHAR) AS day_of_life_cell,
            COUNT(*) AS neonatal_death_count,
            COUNT(*) AS exact_timing_death_count,
            0 AS grouped_timing_death_count
        FROM deaths_raw
        WHERE age_unit_code = 2
          AND age_value BETWEEN 0 AND 27
        GROUP BY ALL
    ),
    grouped_deaths AS (
        SELECT
            municipality_code_occurrence,
            birth_cohort_date,
            birth_cohort_year,
            'grouped_0_27_nonexact' AS day_of_life_cell,
            COUNT(*) AS neonatal_death_count,
            0 AS exact_timing_death_count,
            COUNT(*) AS grouped_timing_death_count
        FROM deaths_raw
        WHERE age_unit_code IN (0, 1)
        GROUP BY ALL
    ),
    deaths AS (
        SELECT * FROM exact_deaths
        UNION ALL
        SELECT * FROM grouped_deaths
    ),
    exact_merged AS (
        SELECT
            r.*,
            COALESCE(d.neonatal_death_count, 0) AS neonatal_death_count,
            COALESCE(d.exact_timing_death_count, 0) AS exact_timing_death_count,
            COALESCE(d.grouped_timing_death_count, 0) AS grouped_timing_death_count
        FROM exact_cells r
        LEFT JOIN exact_deaths d
          ON d.municipality_code_occurrence = r.municipality_code_occurrence
         AND d.birth_cohort_date = r.birth_cohort_date
         AND d.day_of_life_cell = r.day_of_life_cell
    ),
    grouped_merged AS (
        SELECT
            r.*,
            COALESCE(d.neonatal_death_count, 0) AS neonatal_death_count,
            COALESCE(d.exact_timing_death_count, 0) AS exact_timing_death_count,
            COALESCE(d.grouped_timing_death_count, 0) AS grouped_timing_death_count
        FROM grouped_cells r
        LEFT JOIN grouped_deaths d
          ON d.municipality_code_occurrence = r.municipality_code_occurrence
         AND d.birth_cohort_date = r.birth_cohort_date
         AND d.day_of_life_cell = r.day_of_life_cell
    ),
    exact_output AS (
        SELECT
            m.birth_cohort_year AS birth_cohort_year,
            m.municipality_code_occurrence AS municipality_code_occurrence,
            SUBSTR(CAST(m.municipality_code_occurrence AS VARCHAR), 1, 2) AS state_code,
            CASE
                WHEN m.birth_cohort_year IN (2020, 2021) THEN '2020_2021'
                ELSE '2022_2024'
            END AS subperiod,
            m.birth_cohort_date AS birth_cohort_date,
            m.day_of_life_cell AS day_of_life_cell,
            m.exact_day_flag AS exact_day_flag,
            m.grouped_timing_flag AS grouped_timing_flag,
            m.grouped_window_complete_flag AS grouped_window_complete_flag,
            m.analysis_date AS analysis_date,
            m.grouped_window_end_date AS grouped_window_end_date,
            m.births_at_risk_count AS births_at_risk_count,
            m.neonatal_death_count AS neonatal_death_count,
            m.exact_timing_death_count AS exact_timing_death_count,
            m.grouped_timing_death_count AS grouped_timing_death_count,
            CASE
                WHEN m.neonatal_death_count > 0
                    THEN m.exact_timing_death_count::DOUBLE / m.neonatal_death_count::DOUBLE
                ELSE NULL
            END AS exact_timing_share,
            {MACROREGION_CASE_SQL.replace('municipality_code', 'm.municipality_code_occurrence')} AS macroregion,
            CAST(MAX(c.tmax_c) AS DOUBLE) AS tmax_exposure_value,
            CAST(MAX(c.heat_index_approx_c) AS DOUBLE) AS approximate_heat_index_exposure_value,
            CAST(MAX(c.heatwave_len_p95) AS DOUBLE) AS heatwave_duration_exposure_value,
            CAST(MAX(c.rh_mean_pct) AS DOUBLE) AS rh_mean_pct_exposure_value,
            COUNT(c.climate_date) AS climate_days_observed_count,
            STRING_AGG(DISTINCT c.fonte_tmax_c_final, '|' ORDER BY c.fonte_tmax_c_final) AS temperature_source_set,
            STRING_AGG(DISTINCT c.climate_correction_path, '|' ORDER BY c.climate_correction_path) AS climate_correction_path_set,
            MAX(COALESCE(c.imputado_tmax_c, 0)) AS any_temperature_imputation_flag,
            CAST(ctx.idh_z AS DOUBLE) AS idh_z,
            CAST(ctx.gini_z AS DOUBLE) AS gini_z
        FROM exact_merged m
        LEFT JOIN climate c
          ON c.municipality_code = m.municipality_code_occurrence
         AND c.climate_date = m.analysis_date
        LEFT JOIN context_z ctx
          ON ctx.municipality_code = m.municipality_code_occurrence
        GROUP BY ALL
    ),
    grouped_output AS (
        SELECT
            m.birth_cohort_year AS birth_cohort_year,
            m.municipality_code_occurrence AS municipality_code_occurrence,
            SUBSTR(CAST(m.municipality_code_occurrence AS VARCHAR), 1, 2) AS state_code,
            CASE
                WHEN m.birth_cohort_year IN (2020, 2021) THEN '2020_2021'
                ELSE '2022_2024'
            END AS subperiod,
            m.birth_cohort_date AS birth_cohort_date,
            m.day_of_life_cell AS day_of_life_cell,
            m.exact_day_flag AS exact_day_flag,
            m.grouped_timing_flag AS grouped_timing_flag,
            m.grouped_window_complete_flag AS grouped_window_complete_flag,
            m.analysis_date AS analysis_date,
            m.grouped_window_end_date AS grouped_window_end_date,
            m.births_at_risk_count AS births_at_risk_count,
            m.neonatal_death_count AS neonatal_death_count,
            m.exact_timing_death_count AS exact_timing_death_count,
            m.grouped_timing_death_count AS grouped_timing_death_count,
            CASE
                WHEN m.neonatal_death_count > 0
                    THEN m.exact_timing_death_count::DOUBLE / m.neonatal_death_count::DOUBLE
                ELSE NULL
            END AS exact_timing_share,
            {MACROREGION_CASE_SQL.replace('municipality_code', 'm.municipality_code_occurrence')} AS macroregion,
            CAST(AVG(c.tmax_c) AS DOUBLE) AS tmax_exposure_value,
            CAST(AVG(c.heat_index_approx_c) AS DOUBLE) AS approximate_heat_index_exposure_value,
            CAST(AVG(c.heatwave_len_p95) AS DOUBLE) AS heatwave_duration_exposure_value,
            CAST(AVG(c.rh_mean_pct) AS DOUBLE) AS rh_mean_pct_exposure_value,
            COUNT(c.climate_date) AS climate_days_observed_count,
            STRING_AGG(DISTINCT c.fonte_tmax_c_final, '|' ORDER BY c.fonte_tmax_c_final) AS temperature_source_set,
            STRING_AGG(DISTINCT c.climate_correction_path, '|' ORDER BY c.climate_correction_path) AS climate_correction_path_set,
            MAX(COALESCE(c.imputado_tmax_c, 0)) AS any_temperature_imputation_flag,
            CAST(ctx.idh_z AS DOUBLE) AS idh_z,
            CAST(ctx.gini_z AS DOUBLE) AS gini_z
        FROM grouped_merged m
        LEFT JOIN climate c
          ON c.municipality_code = m.municipality_code_occurrence
         AND c.climate_date BETWEEN m.birth_cohort_date AND m.grouped_window_end_date
        LEFT JOIN context_z ctx
          ON ctx.municipality_code = m.municipality_code_occurrence
        GROUP BY ALL
    )
    SELECT * FROM exact_output
    UNION ALL
    SELECT * FROM grouped_output
    """


def _mart_inventory_row(
    root: Path,
    mart_name: str,
    partition_label: str,
    path: Path,
    row_count: int,
    column_count: int,
) -> dict[str, Any]:
    return {
        "mart_name": mart_name,
        "partition_label": partition_label,
        "row_count": int(row_count),
        "column_count": int(column_count),
        "output_path": _relative(root, path),
        "live_birth_rows_backfilled": 0,
        "offset_rows_backfilled": 0,
    }


def _gestational_metadata() -> list[MartMetadataRow]:
    return [
        MartMetadataRow(GESTATIONAL_MART_NAME, "pregnancy_id", "Pregnancy identifier", "Unique Step 9 pregnancy identifier carried from SINASC event records.", "id", "identifier", "flagship_gestational", "", False, "silver_sinasc_event", "SINASC_EVENT_ID", "silver_sinasc_event.SINASC_EVENT_ID", "Direct carry-forward from the frozen SINASC event surface."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "birth_record_year", "Birth record year", "Delivery year used to partition the flagship gestational execution mart.", "year", "time", "flagship_gestational", "", False, "silver_sinasc_event", "YEAR", "silver_sinasc_event.YEAR", "Direct carry-forward from the frozen SINASC event surface."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "delivery_date", "Delivery date", "Observed delivery date for the live birth record.", "date", "time", "flagship_gestational", "", False, "silver_sinasc_event", "DATE", "silver_sinasc_event.DATE", "Direct carry-forward from the frozen SINASC event surface."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "conception_date", "Estimated conception date", "Estimated conception date derived as delivery date minus final gestational age in days.", "date", "derived_time", "flagship_gestational", "", False, "silver_sinasc_event", "DATE|GA_DAYS_FINAL", "derived.conception_date", "Derived as DATE minus GA_DAYS_FINAL under the frozen Round 3B timing contract."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "municipality_code", "Residence municipality code", "Maternal residence municipality code used as the flagship municipality axis.", "code", "identifier", "flagship_gestational", "", False, "silver_sinasc_event", "CODMUNRES", "silver_sinasc_event.CODMUNRES", "Direct carry-forward from the frozen SINASC event surface."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "state_code", "State code", "Two-digit state code derived from the residence municipality.", "code", "derived_identifier", "flagship_gestational", "", False, "silver_sinasc_event", "CODMUNRES", "derived.state_code", "Derived from the first two digits of the residence municipality code."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "macroregion", "Macroregion", "Brazil macroregion derived from the residence municipality code.", "label", "derived_stratifier", "flagship_gestational", "", False, "silver_sinasc_event", "CODMUNRES", "derived.macroregion", "Derived from the municipality state prefix under the frozen Round 8 heterogeneity split."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "subperiod", "Subperiod", "Frozen Round 8 subperiod split based on the birth record year.", "label", "derived_stratifier", "flagship_gestational", "", False, "silver_sinasc_event", "YEAR", "derived.subperiod", "Derived as 2020_2021 versus 2022_2024 from the birth record year."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "gestational_week_completed", "Completed gestational week at interval start", "Completed gestational week defining the person-period interval start.", "weeks", "time", "flagship_gestational", "", False, "silver_sinasc_event", "GA_WEEKS_COMPLETED_FINAL", "derived.gestational_week_completed", "Expanded into one row per pregnancy-week from 22 through the terminal observed week or censoring at 36."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "week_start_date", "Person-period week start date", "Observed start date of the gestational-week interval.", "date", "time", "flagship_gestational", "", False, "silver_sinasc_event", "DATE|GA_DAYS_FINAL", "derived.week_start_date", "Derived from the estimated conception date plus completed gestational week times seven days."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "week_end_date", "Person-period week end date", "Observed end date of the gestational-week interval with terminal weeks truncated at delivery.", "date", "time", "flagship_gestational", "", False, "silver_sinasc_event", "DATE|GA_DAYS_FINAL", "derived.week_end_date", "Derived from the interval start plus six days except in the terminal preterm week, which ends on the delivery date."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "terminal_week_flag", "Terminal week flag", "Indicator that the row is the terminal observed week for the pregnancy in the primary preterm risk set.", "binary flag", "quality_flag", "flagship_gestational", "", False, "silver_sinasc_event", "PRETERM_LT37_FLAG|GA_WEEKS_COMPLETED_FINAL", "derived.terminal_week_flag", "Derived inside the gestational person-period expansion."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "preterm_event_flag", "Preterm event flag", "Indicator that the delivery occurred during the current gestational-week interval before 37 completed weeks.", "binary flag", "outcome", "flagship_gestational", "", False, "silver_sinasc_event", "PRETERM_LT37_FLAG|GA_WEEKS_COMPLETED_FINAL", "derived.preterm_event_flag", "Derived inside the gestational person-period expansion."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "censor_at_36_flag", "Administrative censoring at 36 weeks flag", "Indicator that the pregnancy remains ongoing through week 36 and is censored from the primary preterm risk set without an event.", "binary flag", "quality_flag", "flagship_gestational", "", False, "silver_sinasc_event", "PRETERM_LT37_FLAG", "derived.censor_at_36_flag", "Derived inside the gestational person-period expansion."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "ga_days_final", "Final gestational age in days", "Final gestational age in days carried from the Round 3A rebuild.", "days", "covariate", "flagship_gestational", "", False, "silver_sinasc_event", "GA_DAYS_FINAL", "silver_sinasc_event.GA_DAYS_FINAL", "Direct carry-forward from the frozen SINASC event surface."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "ga_weeks_completed_final", "Final completed gestational weeks", "Completed gestational weeks carried from the Round 3A rebuild.", "weeks", "covariate", "flagship_gestational", "", False, "silver_sinasc_event", "GA_WEEKS_COMPLETED_FINAL", "silver_sinasc_event.GA_WEEKS_COMPLETED_FINAL", "Direct carry-forward from the frozen SINASC event surface."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "ga_days_approx_flag", "Approximate gestational age flag", "Indicator that gestational age in days was approximated rather than exact.", "binary flag", "quality_flag", "flagship_gestational", "", False, "silver_sinasc_event", "GA_DAYS_APPROX_FLAG", "silver_sinasc_event.GA_DAYS_APPROX_FLAG", "Direct carry-forward from the frozen SINASC event surface."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "tmax_weekly_mean", "Weekly maximum temperature mean", "Mean daily maximum temperature across the gestational-week interval.", "degrees Celsius", "exposure", "flagship_gestational", "family_1_tmax", False, "gold_mnch_perinatal", "TMAX_C", "gold_mnch_perinatal.TMAX_C", "Weekly mean over municipality-day climate rows spanning the gestational-week interval."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "approximate_heat_index_weekly_mean", "Weekly approximate heat index mean", "Mean approximate heat index across the gestational-week interval.", "degrees Celsius", "derived_exposure", "flagship_gestational", "family_4_heat_index_sensitivity", False, "gold_mnch_perinatal", "HEAT_INDEX_APPROX_C", "gold_mnch_perinatal.HEAT_INDEX_APPROX_C", "Weekly mean over municipality-day climate rows spanning the gestational-week interval."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "heatwave_duration_weekly_mean", "Weekly heatwave duration mean", "Mean heatwave duration across the gestational-week interval.", "days", "derived_exposure", "flagship_gestational", "family_3_heatwave_duration", False, "gold_mnch_perinatal", "HEATWAVE_LEN_P95", "gold_mnch_perinatal.HEATWAVE_LEN_P95", "Weekly mean over municipality-day climate rows spanning the gestational-week interval."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "rh_mean_pct_weekly_mean", "Weekly mean relative humidity", "Mean relative humidity across the gestational-week interval.", "percent", "covariate", "flagship_gestational", "", False, "gold_mnch_perinatal", "RH_MEAN_PCT", "gold_mnch_perinatal.RH_MEAN_PCT", "Weekly mean over municipality-day climate rows spanning the gestational-week interval."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "climate_days_observed_count", "Observed climate days in interval", "Number of municipality-day climate rows observed inside the gestational-week interval.", "days", "quality_flag", "flagship_gestational", "", False, "gold_mnch_perinatal", "DATE", "derived.climate_days_observed_count", "Count of municipality-day climate rows contributing to the interval summary."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "temperature_source_set", "Temperature source set", "Distinct temperature-source labels contributing to the weekly interval summary.", "source labels", "provenance", "flagship_gestational", "", True, "gold_mnch_perinatal", "FONTE_TMAX_C_FINAL", "gold_mnch_perinatal.FONTE_TMAX_C_FINAL", "Distinct source labels aggregated across the interval."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "humidity_source_set", "Humidity source set", "Distinct humidity-source labels contributing to the weekly interval summary.", "source labels", "provenance", "flagship_gestational", "", True, "gold_mnch_perinatal", "FONTE_RH_MEAN_PCT_FINAL", "gold_mnch_perinatal.FONTE_RH_MEAN_PCT_FINAL", "Distinct source labels aggregated across the interval."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "climate_correction_path_set", "Climate correction-path set", "Distinct climate-correction path labels contributing to the weekly interval summary.", "source labels", "provenance", "flagship_gestational", "", True, "gold_mnch_perinatal", "CLIMATE_CORRECTION_PATH", "gold_mnch_perinatal.CLIMATE_CORRECTION_PATH", "Distinct correction-path labels aggregated across the interval."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "any_temperature_imputation_flag", "Any temperature imputation flag", "Indicator that at least one day in the weekly interval used temperature fallback imputation.", "binary flag", "quality_flag", "flagship_gestational", "", False, "gold_mnch_perinatal", "IMPUTADO_TMAX_C", "gold_mnch_perinatal.IMPUTADO_TMAX_C", "Maximum of the daily temperature-imputation indicator across the interval."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "maternal_age_years", "Maternal age in years", "Maternal age in years carried into the flagship person-period surface.", "years", "covariate", "flagship_gestational", "", False, "silver_sinasc_event", "MATERNAL_AGE_YEARS", "silver_sinasc_event.MATERNAL_AGE_YEARS", "Direct carry-forward from the frozen SINASC event surface."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "maternal_schooling_code", "Maternal schooling code", "Final maternal schooling code carried into the flagship person-period surface.", "code", "covariate", "flagship_gestational", "", False, "silver_sinasc_event", "MATERNAL_SCHOOLING_CODE_FINAL", "silver_sinasc_event.MATERNAL_SCHOOLING_CODE_FINAL", "Direct carry-forward from the frozen SINASC event surface."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "parity_total_prior", "Total prior parity", "Total prior live births carried into the flagship person-period surface.", "count", "covariate", "flagship_gestational", "", False, "silver_sinasc_event", "PARITY_TOTAL_PRIOR", "silver_sinasc_event.PARITY_TOTAL_PRIOR", "Direct carry-forward from the frozen SINASC event surface."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "plurality_code", "Plurality code", "Plurality code carried into the flagship person-period surface.", "code", "covariate", "flagship_gestational", "", False, "silver_sinasc_event", "PLURALITY_CODE", "silver_sinasc_event.PLURALITY_CODE", "Direct carry-forward from the frozen SINASC event surface."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "newborn_sex_code", "Newborn sex code", "Newborn sex code carried into the flagship person-period surface.", "code", "covariate", "flagship_gestational", "", False, "silver_sinasc_event", "NEWBORN_SEX_CODE", "silver_sinasc_event.NEWBORN_SEX_CODE", "Direct carry-forward from the frozen SINASC event surface."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "maternal_race_color_code", "Maternal race or skin-colour code", "Maternal race or skin-colour code carried into the flagship person-period surface when available.", "code", "covariate", "flagship_gestational", "", False, "silver_sinasc_event", "MATERNAL_RACE_COLOR_CODE", "silver_sinasc_event.MATERNAL_RACE_COLOR_CODE", "Direct carry-forward from the frozen SINASC event surface."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "gestational_days_lost_below_37w", "Gestational days lost below 37 weeks", "Methods-facing gestational severity field retained for future E006 execution without rebuilding sources.", "days", "descriptor", "flagship_gestational", "", False, "silver_sinasc_event", "PRETERM_DAYS_SHORT_OF_37W_FINAL", "silver_sinasc_event.PRETERM_DAYS_SHORT_OF_37W_FINAL", "Direct carry-forward from the frozen SINASC event surface."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "idh_z", "Standardized HDI", "Municipality-level standardized HDI modifier joined from the static context table.", "z score", "modifier", "flagship_gestational", "", False, "gold_municipality_context_static", "IDH_EXPANDED", "gold_municipality_context_static.IDH_EXPANDED", "Z-score standardized over the full municipality context universe."),
        MartMetadataRow(GESTATIONAL_MART_NAME, "gini_z", "Standardized Gini", "Municipality-level standardized Gini modifier joined from the static context table.", "z score", "modifier", "flagship_gestational", "", False, "gold_municipality_context_static", "GINI_EXPANDED", "gold_municipality_context_static.GINI_EXPANDED", "Z-score standardized over the full municipality context universe."),
    ]


def _stillbirth_metadata() -> list[MartMetadataRow]:
    return [
        MartMetadataRow(STILLBIRTH_MART_NAME, "threshold_scenario", "Stillbirth threshold scenario", "Frozen stillbirth threshold scenario for the grouped fetuses-at-risk mart.", "label", "identifier", "flagship_stillbirth", "", False, "derived", "STILLBIRTH_GTE28_PRIMARY_FLAG|STILLBIRTH_GTE22_SENSITIVITY_FLAG", "derived.threshold_scenario", "Derived as primary_gte28 or sensitivity_gte22 under the frozen Round 3B and Round 5 contracts."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "source_year", "Source event year", "Event year used to partition the grouped stillbirth mart.", "year", "time", "flagship_stillbirth", "", False, "silver_stillbirth_individual|silver_sinasc_event", "YEAR", "shared.YEAR", "Direct carry-forward from the source event year."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "municipality_code", "Residence municipality code", "Maternal residence municipality code used as the stillbirth municipality axis.", "code", "identifier", "flagship_stillbirth", "", False, "silver_stillbirth_individual|silver_sinasc_event", "CODMUNRES", "shared.CODMUNRES", "Direct carry-forward from the frozen individual record surfaces."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "state_code", "State code", "Two-digit state code derived from the residence municipality.", "code", "derived_identifier", "flagship_stillbirth", "", False, "derived", "CODMUNRES", "derived.state_code", "Derived from the first two digits of the residence municipality code."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "macroregion", "Macroregion", "Brazil macroregion derived from the residence municipality code.", "label", "derived_stratifier", "flagship_stillbirth", "", False, "derived", "CODMUNRES", "derived.macroregion", "Derived from the municipality state prefix under the frozen Round 8 heterogeneity split."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "subperiod", "Subperiod", "Frozen Round 8 subperiod split based on the source event year.", "label", "derived_stratifier", "flagship_stillbirth", "", False, "derived", "YEAR", "derived.subperiod", "Derived as 2020_2021 versus 2022_2024 from the source event year."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "analysis_week_start_date", "Analysis week start date", "Week-qualified start date for the grouped fetuses-at-risk interval.", "date", "time", "flagship_stillbirth", "", False, "silver_stillbirth_individual|silver_sinasc_event", "EVENT_DATE|DATE|GA_WEEKS_COMPLETED_FINAL", "derived.analysis_week_start_date", "Derived by stepping backward in whole weeks from the event date under the frozen native-timing rule."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "analysis_week_end_date", "Analysis week end date", "Week-qualified end date for the grouped fetuses-at-risk interval.", "date", "time", "flagship_stillbirth", "", False, "silver_stillbirth_individual|silver_sinasc_event", "EVENT_DATE|DATE|GA_WEEKS_COMPLETED_FINAL", "derived.analysis_week_end_date", "Derived by stepping backward in whole weeks from the event date under the frozen native-timing rule."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "gestational_week_completed", "Completed gestational week", "Completed gestational week defining the grouped fetuses-at-risk interval.", "weeks", "time", "flagship_stillbirth", "", False, "silver_stillbirth_individual|silver_sinasc_event", "GA_WEEKS_COMPLETED_FINAL", "shared.GA_WEEKS_COMPLETED_FINAL", "Expanded into grouped weekly risk-set cells."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "maternal_age_years", "Maternal age in years", "Maternal age in years retained as the only continuous covariate in the stillbirth grouped model surface.", "years", "covariate", "flagship_stillbirth", "", False, "silver_stillbirth_individual|silver_sinasc_event", "MATERNAL_AGE_YEARS", "shared.MATERNAL_AGE_YEARS", "Direct carry-forward with complete-case exclusion for missing age."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "maternal_schooling_code", "Maternal schooling code", "Maternal schooling code retained in the stillbirth grouped model surface.", "code", "covariate", "flagship_stillbirth", "", False, "silver_stillbirth_individual|silver_sinasc_event", "MATERNAL_SCHOOLING_CODE_FINAL", "shared.MATERNAL_SCHOOLING_CODE_FINAL", "Direct carry-forward with explicit unknown category when missing."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "plurality_code", "Plurality code", "Plurality code retained in the stillbirth grouped model surface.", "code", "covariate", "flagship_stillbirth", "", False, "silver_stillbirth_individual|silver_sinasc_event", "PLURALITY_CODE", "shared.PLURALITY_CODE", "Direct carry-forward with explicit unknown category when missing."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "fetal_sex_code", "Fetal sex code", "Fetal sex code retained in the stillbirth grouped model surface.", "code", "covariate", "flagship_stillbirth", "", False, "silver_stillbirth_individual|silver_sinasc_event", "FETAL_SEX_CODE|NEWBORN_SEX_CODE", "shared.FETAL_SEX_CODE", "Direct carry-forward with explicit unknown category when needed."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "timing_quality_class", "Timing quality class", "Explicit timing-quality class distinguishing native week-qualified fetal-death records from live-birth competing events.", "label", "quality_flag", "flagship_stillbirth", "", False, "derived", "EVENT_DATE|DATE|GA_WEEKS_COMPLETED_FINAL", "derived.timing_quality_class", "Derived from source-system type under the frozen native-timing rule."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "pregnancies_at_risk_count", "Pregnancies at risk count", "Count of pregnancies contributing to the grouped fetuses-at-risk cell.", "pregnancies", "denominator", "flagship_stillbirth", "", False, "derived", "live_births|stillbirths", "derived.pregnancies_at_risk_count", "Grouped count after weekly risk-set expansion."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "stillbirth_event_count", "Stillbirth event count", "Count of stillbirth terminal events in the grouped cell.", "events", "outcome", "flagship_stillbirth", "", False, "derived", "stillbirths", "derived.stillbirth_event_count", "Grouped terminal-event count after weekly risk-set expansion."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "live_birth_competing_event_count", "Live-birth competing event count", "Count of competing live-birth terminal events in the grouped cell.", "events", "outcome", "flagship_stillbirth", "", False, "derived", "live_births", "derived.live_birth_competing_event_count", "Grouped terminal-event count after weekly risk-set expansion."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "births_based_total_births_count", "Births-based total births count", "Grouped total births count used only for the Round 8 births-based denominator fallback repeat.", "events", "robustness_only", "flagship_stillbirth", "", False, "derived", "stillbirth_event_count|live_birth_competing_event_count", "derived.births_based_total_births_count", "Derived as stillbirth_event_count plus live_birth_competing_event_count."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "tmax_weekly_mean", "Weekly maximum temperature mean", "Mean daily maximum temperature across the week-qualified stillbirth interval.", "degrees Celsius", "exposure", "flagship_stillbirth", "family_1_tmax", False, "gold_mnch_perinatal", "TMAX_C", "gold_mnch_perinatal.TMAX_C", "Weekly mean over municipality-day climate rows spanning the stillbirth interval."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "approximate_heat_index_weekly_mean", "Weekly approximate heat index mean", "Mean approximate heat index across the week-qualified stillbirth interval.", "degrees Celsius", "derived_exposure", "flagship_stillbirth", "family_4_heat_index_sensitivity", False, "gold_mnch_perinatal", "HEAT_INDEX_APPROX_C", "gold_mnch_perinatal.HEAT_INDEX_APPROX_C", "Weekly mean over municipality-day climate rows spanning the stillbirth interval."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "heatwave_duration_weekly_mean", "Weekly heatwave duration mean", "Mean heatwave duration across the week-qualified stillbirth interval.", "days", "derived_exposure", "flagship_stillbirth", "family_3_heatwave_duration", False, "gold_mnch_perinatal", "HEATWAVE_LEN_P95", "gold_mnch_perinatal.HEATWAVE_LEN_P95", "Weekly mean over municipality-day climate rows spanning the stillbirth interval."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "rh_mean_pct_weekly_mean", "Weekly mean relative humidity", "Mean relative humidity across the week-qualified stillbirth interval.", "percent", "covariate", "flagship_stillbirth", "", False, "gold_mnch_perinatal", "RH_MEAN_PCT", "gold_mnch_perinatal.RH_MEAN_PCT", "Weekly mean over municipality-day climate rows spanning the stillbirth interval."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "climate_days_observed_count", "Observed climate days in interval", "Number of municipality-day climate rows observed inside the stillbirth interval.", "days", "quality_flag", "flagship_stillbirth", "", False, "gold_mnch_perinatal", "DATE", "derived.climate_days_observed_count", "Count of municipality-day climate rows contributing to the interval summary."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "temperature_source_set", "Temperature source set", "Distinct temperature-source labels contributing to the weekly interval summary.", "source labels", "provenance", "flagship_stillbirth", "", True, "gold_mnch_perinatal", "FONTE_TMAX_C_FINAL", "gold_mnch_perinatal.FONTE_TMAX_C_FINAL", "Distinct source labels aggregated across the interval."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "climate_correction_path_set", "Climate correction-path set", "Distinct climate-correction path labels contributing to the weekly interval summary.", "source labels", "provenance", "flagship_stillbirth", "", True, "gold_mnch_perinatal", "CLIMATE_CORRECTION_PATH", "gold_mnch_perinatal.CLIMATE_CORRECTION_PATH", "Distinct correction-path labels aggregated across the interval."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "any_temperature_imputation_flag", "Any temperature imputation flag", "Indicator that at least one day in the stillbirth interval used temperature fallback imputation.", "binary flag", "quality_flag", "flagship_stillbirth", "", False, "gold_mnch_perinatal", "IMPUTADO_TMAX_C", "gold_mnch_perinatal.IMPUTADO_TMAX_C", "Maximum of the daily temperature-imputation indicator across the interval."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "idh_z", "Standardized HDI", "Municipality-level standardized HDI modifier joined from the static context table.", "z score", "modifier", "flagship_stillbirth", "", False, "gold_municipality_context_static", "IDH_EXPANDED", "gold_municipality_context_static.IDH_EXPANDED", "Z-score standardized over the full municipality context universe."),
        MartMetadataRow(STILLBIRTH_MART_NAME, "gini_z", "Standardized Gini", "Municipality-level standardized Gini modifier joined from the static context table.", "z score", "modifier", "flagship_stillbirth", "", False, "gold_municipality_context_static", "GINI_EXPANDED", "gold_municipality_context_static.GINI_EXPANDED", "Z-score standardized over the full municipality context universe."),
    ]


def _neonatal_metadata() -> list[MartMetadataRow]:
    return [
        MartMetadataRow(NEONATAL_MART_NAME, "birth_cohort_year", "Birth cohort year", "Birth-cohort year used to partition the grouped neonatal bridge execution mart.", "year", "time", "flagship_neonatal_bridge", "", False, "silver_sinasc_event|silver_sim_individual", "YEAR|DTNASC", "shared.birth_cohort_year", "Derived from the birth-cohort date."),
        MartMetadataRow(NEONATAL_MART_NAME, "municipality_code_occurrence", "Occurrence municipality code", "Occurrence municipality code used as the neonatal bridge municipality axis.", "code", "identifier", "flagship_neonatal_bridge", "", False, "silver_sim_individual|silver_sinasc_event", "CODMUNOCOR|CODMUNNASC", "shared.municipality_code_occurrence", "Death occurrence municipality on the SIM side aligned to birth occurrence municipality on the SINASC side."),
        MartMetadataRow(NEONATAL_MART_NAME, "state_code", "State code", "Two-digit state code derived from the occurrence municipality.", "code", "derived_identifier", "flagship_neonatal_bridge", "", False, "derived", "CODMUNOCOR|CODMUNNASC", "derived.state_code", "Derived from the first two digits of the occurrence municipality code."),
        MartMetadataRow(NEONATAL_MART_NAME, "subperiod", "Subperiod", "Frozen Round 8 subperiod split based on the birth-cohort year.", "label", "derived_stratifier", "flagship_neonatal_bridge", "", False, "derived", "DTNASC", "derived.subperiod", "Derived as 2020_2021 versus 2022_2024 from the birth-cohort year."),
        MartMetadataRow(NEONATAL_MART_NAME, "birth_cohort_date", "Birth cohort date", "Birth cohort date carried by the grouped neonatal bridge mart.", "date", "time", "flagship_neonatal_bridge", "", False, "silver_sinasc_event|silver_sim_individual", "DATE|DTNASC", "shared.birth_cohort_date", "Birth date used as the cohort anchor in the bridge design."),
        MartMetadataRow(NEONATAL_MART_NAME, "day_of_life_cell", "Day-of-life cell", "Day-of-life cell preserving either exact days 0 through 27 or the grouped nonexact neonatal cell.", "label", "time", "flagship_neonatal_bridge", "", False, "derived", "AGE_UNIT_CODE|AGE_VALUE", "derived.day_of_life_cell", "Derived from exact neonatal age coding or grouped nonexact timing."),
        MartMetadataRow(NEONATAL_MART_NAME, "exact_day_flag", "Exact day flag", "Indicator that the row uses an exact day-of-life cell.", "binary flag", "quality_flag", "flagship_neonatal_bridge", "", False, "derived", "AGE_UNIT_CODE|AGE_VALUE", "derived.exact_day_flag", "Derived from the age-field support for exact neonatal day of life."),
        MartMetadataRow(NEONATAL_MART_NAME, "grouped_timing_flag", "Grouped timing flag", "Indicator that the row preserves grouped neonatal timing rather than an exact day.", "binary flag", "quality_flag", "flagship_neonatal_bridge", "", False, "derived", "AGE_UNIT_CODE|AGE_VALUE", "derived.grouped_timing_flag", "Derived from nonexact neonatal age coding."),
        MartMetadataRow(NEONATAL_MART_NAME, "grouped_window_complete_flag", "Grouped window complete flag", "Indicator that the grouped 0 to 27-day climate window is fully observable inside the available climate series.", "binary flag", "quality_flag", "flagship_neonatal_bridge", "", False, "derived", "DATE|DTNASC", "derived.grouped_window_complete_flag", "Derived from the available climate window for grouped nonexact rows."),
        MartMetadataRow(NEONATAL_MART_NAME, "analysis_date", "Analysis date", "Exact analysis date for rows with exact neonatal day of life.", "date", "time", "flagship_neonatal_bridge", "", False, "derived", "DTNASC|AGE_VALUE", "derived.analysis_date", "Derived as birth cohort date plus exact day of life."),
        MartMetadataRow(NEONATAL_MART_NAME, "grouped_window_end_date", "Grouped window end date", "Upper bound of the grouped 0 to 27-day climate window when exact neonatal timing is unavailable.", "date", "time", "flagship_neonatal_bridge", "", False, "derived", "DTNASC", "derived.grouped_window_end_date", "Derived as the lesser of birth cohort date plus 27 days or the last climate date available."),
        MartMetadataRow(NEONATAL_MART_NAME, "births_at_risk_count", "Birth cohort live births count", "Count of live births in the birth cohort used as the bridge denominator surface.", "births", "denominator", "flagship_neonatal_bridge", "", False, "silver_sinasc_event", "CODMUNNASC|DATE", "derived.births_at_risk_count", "Grouped live-birth count by birth cohort and birth occurrence municipality."),
        MartMetadataRow(NEONATAL_MART_NAME, "neonatal_death_count", "Neonatal death count", "Count of neonatal deaths in the grouped bridge cell.", "events", "outcome", "flagship_neonatal_bridge", "", False, "silver_sim_individual", "CODMUNOCOR|DTNASC|AGE_UNIT_CODE|AGE_VALUE", "derived.neonatal_death_count", "Grouped death count by occurrence municipality, birth cohort, and day-of-life cell."),
        MartMetadataRow(NEONATAL_MART_NAME, "exact_timing_death_count", "Exact-timing neonatal death count", "Count of neonatal deaths in the cell with exact day-of-life support.", "events", "quality_flag", "flagship_neonatal_bridge", "", False, "silver_sim_individual", "AGE_UNIT_CODE|AGE_VALUE", "derived.exact_timing_death_count", "Grouped count limited to age records encoded in exact days."),
        MartMetadataRow(NEONATAL_MART_NAME, "grouped_timing_death_count", "Grouped-timing neonatal death count", "Count of neonatal deaths in the cell with grouped nonexact timing support.", "events", "quality_flag", "flagship_neonatal_bridge", "", False, "silver_sim_individual", "AGE_UNIT_CODE|AGE_VALUE", "derived.grouped_timing_death_count", "Grouped count limited to nonexact neonatal age records."),
        MartMetadataRow(NEONATAL_MART_NAME, "exact_timing_share", "Exact-timing share", "Share of deaths in the cell that retain exact day-of-life timing.", "proportion", "quality_flag", "flagship_neonatal_bridge", "", False, "derived", "exact_timing_death_count|neonatal_death_count", "derived.exact_timing_share", "Derived as exact_timing_death_count divided by neonatal_death_count when deaths are present."),
        MartMetadataRow(NEONATAL_MART_NAME, "macroregion", "Macroregion", "Brazil macroregion derived from the occurrence municipality code.", "label", "derived_stratifier", "flagship_neonatal_bridge", "", False, "derived", "CODMUNOCOR|CODMUNNASC", "derived.macroregion", "Derived from the municipality state prefix under the frozen Round 8 heterogeneity split."),
        MartMetadataRow(NEONATAL_MART_NAME, "tmax_exposure_value", "Neonatal bridge maximum temperature exposure", "Exposure value for the neonatal bridge cell using exact-day climate or grouped 0 to 27-day mean climate as appropriate.", "degrees Celsius", "exposure", "flagship_neonatal_bridge", "family_1_tmax", False, "gold_mnch_perinatal", "TMAX_C", "gold_mnch_perinatal.TMAX_C", "Exact-day rows use same-day climate; grouped rows use the mean across days 0 to 27 without pseudo-exact assignment."),
        MartMetadataRow(NEONATAL_MART_NAME, "approximate_heat_index_exposure_value", "Neonatal bridge heat index exposure", "Heat-index repeat exposure value for the neonatal bridge cell.", "degrees Celsius", "derived_exposure", "flagship_neonatal_bridge", "family_4_heat_index_sensitivity", False, "gold_mnch_perinatal", "HEAT_INDEX_APPROX_C", "gold_mnch_perinatal.HEAT_INDEX_APPROX_C", "Exact-day rows use same-day climate; grouped rows use the mean across days 0 to 27."),
        MartMetadataRow(NEONATAL_MART_NAME, "heatwave_duration_exposure_value", "Neonatal bridge heatwave duration exposure", "Heatwave-duration repeat exposure value for the neonatal bridge cell.", "days", "derived_exposure", "flagship_neonatal_bridge", "family_3_heatwave_duration", False, "gold_mnch_perinatal", "HEATWAVE_LEN_P95", "gold_mnch_perinatal.HEATWAVE_LEN_P95", "Exact-day rows use same-day climate; grouped rows use the mean across days 0 to 27."),
        MartMetadataRow(NEONATAL_MART_NAME, "rh_mean_pct_exposure_value", "Neonatal bridge relative humidity exposure", "Humidity control value for the neonatal bridge cell.", "percent", "covariate", "flagship_neonatal_bridge", "", False, "gold_mnch_perinatal", "RH_MEAN_PCT", "gold_mnch_perinatal.RH_MEAN_PCT", "Exact-day rows use same-day climate; grouped rows use the mean across days 0 to 27."),
        MartMetadataRow(NEONATAL_MART_NAME, "climate_days_observed_count", "Observed climate days in cell", "Number of municipality-day climate rows contributing to the bridge exposure value.", "days", "quality_flag", "flagship_neonatal_bridge", "", False, "gold_mnch_perinatal", "DATE", "derived.climate_days_observed_count", "Counts the climate rows used to derive the exposure value."),
        MartMetadataRow(NEONATAL_MART_NAME, "temperature_source_set", "Temperature source set", "Distinct temperature-source labels contributing to the bridge exposure value.", "source labels", "provenance", "flagship_neonatal_bridge", "", True, "gold_mnch_perinatal", "FONTE_TMAX_C_FINAL", "gold_mnch_perinatal.FONTE_TMAX_C_FINAL", "Distinct source labels aggregated across the contributing climate rows."),
        MartMetadataRow(NEONATAL_MART_NAME, "climate_correction_path_set", "Climate correction-path set", "Distinct climate-correction path labels contributing to the bridge exposure value.", "source labels", "provenance", "flagship_neonatal_bridge", "", True, "gold_mnch_perinatal", "CLIMATE_CORRECTION_PATH", "gold_mnch_perinatal.CLIMATE_CORRECTION_PATH", "Distinct correction-path labels aggregated across the contributing climate rows."),
        MartMetadataRow(NEONATAL_MART_NAME, "any_temperature_imputation_flag", "Any temperature imputation flag", "Indicator that at least one contributing climate row used temperature fallback imputation.", "binary flag", "quality_flag", "flagship_neonatal_bridge", "", False, "gold_mnch_perinatal", "IMPUTADO_TMAX_C", "gold_mnch_perinatal.IMPUTADO_TMAX_C", "Maximum of the daily temperature-imputation indicator across the contributing climate rows."),
        MartMetadataRow(NEONATAL_MART_NAME, "idh_z", "Standardized HDI", "Municipality-level standardized HDI modifier joined from the static context table.", "z score", "modifier", "flagship_neonatal_bridge", "", False, "gold_municipality_context_static", "IDH_EXPANDED", "gold_municipality_context_static.IDH_EXPANDED", "Z-score standardized over the full municipality context universe."),
        MartMetadataRow(NEONATAL_MART_NAME, "gini_z", "Standardized Gini", "Municipality-level standardized Gini modifier joined from the static context table.", "z score", "modifier", "flagship_neonatal_bridge", "", False, "gold_municipality_context_static", "GINI_EXPANDED", "gold_municipality_context_static.GINI_EXPANDED", "Z-score standardized over the full municipality context universe."),
    ]


def _metadata_frame() -> pd.DataFrame:
    rows = [row.__dict__ for row in (_gestational_metadata() + _stillbirth_metadata() + _neonatal_metadata())]
    return pd.DataFrame(rows).sort_values(["mart_name", "academic_variable_name"], ignore_index=True)


def _dictionary_frame(metadata_frame: pd.DataFrame) -> pd.DataFrame:
    return metadata_frame.loc[
        :,
        [
            "mart_name",
            "academic_variable_name",
            "academic_label",
            "description",
            "units",
            "public_role",
            "outcome_family",
            "exposure_family",
            "provenance_only",
        ],
    ].drop_duplicates().sort_values(["mart_name", "academic_variable_name"], ignore_index=True)


def _update_shared_mart_metadata(root: Path, inventory_rows: list[dict[str, Any]]) -> None:
    inventory_path = root / SHARED_MART_INVENTORY_PATH
    dict_path = root / SHARED_MART_DICT_PATH
    crosswalk_path = root / SHARED_MART_CROSSWALK_PATH

    existing_inventory = pd.read_csv(inventory_path)
    existing_inventory = existing_inventory[~existing_inventory["mart_name"].isin(
        [GESTATIONAL_MART_NAME, STILLBIRTH_MART_NAME, NEONATAL_MART_NAME]
    )]
    inventory = pd.concat([existing_inventory, pd.DataFrame(inventory_rows)], ignore_index=True).sort_values(
        ["mart_name", "partition_label"], ignore_index=True
    )
    inventory.to_csv(inventory_path, index=False)

    metadata = _metadata_frame()
    existing_crosswalk = pd.read_csv(crosswalk_path)
    existing_crosswalk = existing_crosswalk[~existing_crosswalk["mart_name"].isin(
        [GESTATIONAL_MART_NAME, STILLBIRTH_MART_NAME, NEONATAL_MART_NAME]
    )]
    crosswalk = pd.concat([existing_crosswalk, metadata], ignore_index=True).sort_values(
        ["mart_name", "academic_variable_name", "source_dataset"], ignore_index=True
    )
    crosswalk.to_csv(crosswalk_path, index=False)

    existing_dict = pd.read_csv(dict_path)
    existing_dict = existing_dict[~existing_dict["mart_name"].isin(
        [GESTATIONAL_MART_NAME, STILLBIRTH_MART_NAME, NEONATAL_MART_NAME]
    )]
    dictionary = pd.concat([existing_dict, _dictionary_frame(metadata)], ignore_index=True).sort_values(
        ["mart_name", "academic_variable_name"], ignore_index=True
    )
    dictionary.to_csv(dict_path, index=False)


def _validate_mart_columns(frame: pd.DataFrame) -> None:
    invalid = validate_public_output_columns(frame.columns.tolist(), set())
    if invalid:
        raise ValueError("Invalid public columns: " + ", ".join(sorted(invalid)))


def _primary_exposure_column_for_mart(mart_name: str) -> str:
    if mart_name == GESTATIONAL_MART_NAME:
        return "tmax_weekly_mean"
    if mart_name == STILLBIRTH_MART_NAME:
        return "tmax_weekly_mean"
    if mart_name == NEONATAL_MART_NAME:
        return "tmax_exposure_value"
    raise ValueError(f"Unknown mart name: {mart_name}")


def _non_missing_exposure_rows(path: Path, exposure_column: str) -> int:
    con = _con()
    row = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{path}')
        WHERE {exposure_column} IS NOT NULL
        """
    ).fetchone()
    con.close()
    return int(row[0] or 0)


def _non_missing_exposure_rows_many(paths: list[Path], exposure_column: str) -> int:
    total = 0
    for path in paths:
        total += _non_missing_exposure_rows(path, exposure_column)
    return total


def _surface_registry_rows(root: Path, inventory_rows: list[dict[str, Any]]) -> pd.DataFrame:
    path_map = {row["mart_name"]: row["output_path"] for row in inventory_rows if row["partition_label"] == "2024"}
    return pd.DataFrame(
        [
            {
                "surface_id": "s10_5_gestational_person_period",
                "mart_name": GESTATIONAL_MART_NAME,
                "domain_id": "flagship_preterm_primary",
                "grain": "one_row_per_pregnancy_week_at_risk",
                "municipality_axis": "maternal_residence_CODMUNRES",
                "source_backbone": "silver_sinasc_event_plus_gold_mnch_perinatal_climate",
                "status": "materialized",
                "example_output_path": path_map.get(GESTATIONAL_MART_NAME, ""),
                "notes": "Singleton live births only; risk starts at 22 completed weeks and censors at week 36 for the primary preterm domain.",
            },
            {
                "surface_id": "s10_5_stillbirth_weekly_far",
                "mart_name": STILLBIRTH_MART_NAME,
                "domain_id": "key_secondary_stillbirth",
                "grain": "grouped_residence_municipality_by_week_and_covariate_cell",
                "municipality_axis": "maternal_residence_CODMUNRES",
                "source_backbone": "silver_sinasc_event_plus_silver_stillbirth_individual_plus_gold_mnch_perinatal_climate",
                "status": "materialized",
                "example_output_path": path_map.get(STILLBIRTH_MART_NAME, ""),
                "notes": "One mart contains both primary_gte28 and sensitivity_gte22 threshold scenarios with explicit competing live birth counts.",
            },
            {
                "surface_id": "s10_5_neonatal_bridge",
                "mart_name": NEONATAL_MART_NAME,
                "domain_id": "key_secondary_neonatal",
                "grain": "occurrence_municipality_by_birth_cohort_date_and_day_of_life_cell",
                "municipality_axis": "mortality_occurrence_municipality_with_birth_cohort_bridge",
                "source_backbone": "silver_sinasc_event_plus_silver_sim_individual_plus_gold_mnch_perinatal_climate",
                "status": "materialized",
                "example_output_path": path_map.get(NEONATAL_MART_NAME, ""),
                "notes": "Grouped nonexact neonatal timing remains explicit as grouped_0_27_nonexact and is not coerced into an exact day.",
            },
        ]
    )


def build_flagship_execution_surfaces(root: Path, years: list[int] | None = None) -> dict[str, Any]:
    shared_root = analysis_shared_marts_root(root)
    results_root = root / ROUND10_5_RESULTS_DIR
    registry_root = root / ROUND10_5_REGISTRY_DIR
    results_root.mkdir(parents=True, exist_ok=True)
    registry_root.mkdir(parents=True, exist_ok=True)

    run_log = init_run_log(root, "round10_5_flagship_surface_materialization")
    con = _con()
    inventory_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    build_years = years or list(range(2020, 2025))
    for year in build_years:
        gestational_dir = analysis_shared_marts_root(root) / GESTATIONAL_MART_NAME / f"YEAR={year}"
        gestational_dir.mkdir(parents=True, exist_ok=True)
        gestational_paths = sorted(gestational_dir.glob("*.parquet"))
        if len(gestational_paths) != _expected_chunk_count(GESTATIONAL_MART_NAME):
            for stale in gestational_paths:
                stale.unlink()
            gestational_paths = []
            append_run_event(
                run_log,
                phase="round10_5_surface_build",
                dataset=GESTATIONAL_MART_NAME,
                year=year,
                command="materialize_shared_mart_partition_month_state_chunks",
                decision="build flagship gestational execution surface partition",
                status="started",
            )
            for month in range(1, 13):
                for state_prefix in STATE_PREFIXES:
                    output_path = _state_month_partition_path(root, GESTATIONAL_MART_NAME, year, month, state_prefix)
                    _write_parquet_query(con, _gestational_build_query(year, month, state_prefix), output_path)
                    gestational_paths.append(output_path)
            append_run_event(
                run_log,
                phase="round10_5_surface_build",
                dataset=GESTATIONAL_MART_NAME,
                year=year,
                command="materialize_shared_mart_partition_month_state_chunks",
                decision="build flagship gestational execution surface partition",
                status="completed",
            )
        gest_row_count, gest_column_count, gest_columns = _parquet_dataset_shape(gestational_paths)
        _validate_mart_columns(pd.DataFrame(columns=gest_columns))
        inventory_rows.append(
            _mart_inventory_row(
                root,
                GESTATIONAL_MART_NAME,
                str(year),
                gestational_dir,
                row_count=gest_row_count,
                column_count=gest_column_count,
            )
        )
        audit_rows.append(
            {
                "mart_name": GESTATIONAL_MART_NAME,
                "partition_year": year,
                "row_count": int(gest_row_count),
                "column_count": int(gest_column_count),
                "non_missing_exposure_rows": _non_missing_exposure_rows_many(
                    gestational_paths,
                    _primary_exposure_column_for_mart(GESTATIONAL_MART_NAME),
                ),
                "notes": "Partition materialized under Round 10.5 with delivery-month and state chunking",
            }
        )

        for mart_name, builder in (
            (STILLBIRTH_MART_NAME, _stillbirth_build_query),
            (NEONATAL_MART_NAME, _neonatal_build_query),
        ):
            mart_dir = analysis_shared_marts_root(root) / mart_name / f"YEAR={year}"
            mart_dir.mkdir(parents=True, exist_ok=True)
            chunk_paths = sorted(mart_dir.glob("*.parquet"))
            if len(chunk_paths) != _expected_chunk_count(mart_name):
                for stale in chunk_paths:
                    stale.unlink()
                chunk_paths = []
                append_run_event(
                    run_log,
                    phase="round10_5_surface_build",
                    dataset=mart_name,
                    year=year,
                    command="materialize_shared_mart_partition_month_state_chunks",
                    decision="build flagship execution surface partition",
                    status="started",
                )
                for month in range(1, 13):
                    for state_prefix in STATE_PREFIXES:
                        output_path = _state_month_partition_path(root, mart_name, year, month, state_prefix)
                        _write_parquet_query(con, builder(year, month, state_prefix), output_path)
                        chunk_paths.append(output_path)
                append_run_event(
                    run_log,
                    phase="round10_5_surface_build",
                    dataset=mart_name,
                    year=year,
                    command="materialize_shared_mart_partition_month_state_chunks",
                    decision="build flagship execution surface partition",
                    status="completed",
                )
            row_count, column_count, columns = _parquet_dataset_shape(chunk_paths)
            _validate_mart_columns(pd.DataFrame(columns=columns))
            inventory_rows.append(
                _mart_inventory_row(
                    root,
                    mart_name,
                    str(year),
                    mart_dir,
                    row_count=row_count,
                    column_count=column_count,
                )
            )
            audit_rows.append(
                {
                    "mart_name": mart_name,
                    "partition_year": year,
                    "row_count": int(row_count),
                    "column_count": int(column_count),
                    "non_missing_exposure_rows": _non_missing_exposure_rows_many(
                        chunk_paths,
                        _primary_exposure_column_for_mart(mart_name),
                    ),
                    "notes": "Partition materialized under Round 10.5 with month and state chunking",
                }
            )

    _update_shared_mart_metadata(root, inventory_rows)

    surface_audit = pd.DataFrame(audit_rows).sort_values(["mart_name", "partition_year"], ignore_index=True)
    surface_audit_path = root / ROUND10_5_RESULTS_DIR / "round10_5_surface_audit.csv"
    surface_audit.to_csv(surface_audit_path, index=False)

    surface_registry = _surface_registry_rows(root, inventory_rows)
    surface_registry_path = root / ROUND10_5_REGISTRY_DIR / "round10_5_surface_registry.csv"
    surface_registry.to_csv(surface_registry_path, index=False)

    return {
        "run_log": run_log,
        "surface_audit_path": surface_audit_path,
        "surface_registry_path": surface_registry_path,
    }


def _mart_paths(root: Path, mart_name: str) -> list[Path]:
    return sorted((analysis_shared_marts_root(root) / mart_name).glob("YEAR=*/*.parquet"))


def _expected_chunk_count(mart_name: str) -> int:
    if mart_name in {GESTATIONAL_MART_NAME, STILLBIRTH_MART_NAME, NEONATAL_MART_NAME}:
        return len(STATE_PREFIXES) * 12
    raise ValueError(f"Unknown mart name: {mart_name}")


def _dataset_batches(paths: list[Path], columns: list[str], batch_size: int = 250_000):
    dataset = ds.dataset([str(path) for path in paths], format="parquet")
    scanner = dataset.scanner(columns=columns, batch_size=batch_size)
    for batch in scanner.to_batches():
        yield batch.to_pandas()


def _single_frame_batch_factory(frame: pd.DataFrame):
    def batch_factory():
        yield frame.copy()

    return batch_factory


def _year_mart_glob(root: Path, mart_name: str, year: int) -> str:
    return str(analysis_shared_marts_root(root) / mart_name / f"YEAR={year}" / "*.parquet")


def _stillbirth_primary_aggregated_frame(root: Path, exposure_column: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in range(2020, 2025):
        con = _con()
        query = f"""
            SELECT
                'primary_gte28' AS threshold_scenario,
                ROUND(CAST({exposure_column} AS DOUBLE), {PRIMARY_EXPOSURE_BIN_DECIMALS}) AS {exposure_column},
                CAST(maternal_age_years AS INTEGER) AS maternal_age_years,
                CAST(gestational_week_completed AS INTEGER) AS gestational_week_completed,
                COALESCE(CAST(maternal_schooling_code AS VARCHAR), 'unknown') AS maternal_schooling_code,
                COALESCE(CAST(plurality_code AS VARCHAR), 'unknown') AS plurality_code,
                COALESCE(CAST(fetal_sex_code AS VARCHAR), 'unknown') AS fetal_sex_code,
                SUM(CAST(pregnancies_at_risk_count AS DOUBLE)) AS pregnancies_at_risk_count,
                SUM(CAST(stillbirth_event_count AS DOUBLE)) AS stillbirth_event_count
            FROM read_parquet('{_year_mart_glob(root, STILLBIRTH_MART_NAME, year)}', union_by_name=true)
            WHERE threshold_scenario = 'primary_gte28'
              AND pregnancies_at_risk_count > 0
              AND maternal_age_years IS NOT NULL
              AND {exposure_column} IS NOT NULL
            GROUP BY ALL
        """
        frames.append(con.execute(query).fetchdf())
        con.close()
    frame = pd.concat(frames, ignore_index=True)
    frame = (
        frame.groupby(
            [
                "threshold_scenario",
                exposure_column,
                "maternal_age_years",
                "gestational_week_completed",
                "maternal_schooling_code",
                "plurality_code",
                "fetal_sex_code",
            ],
            as_index=False,
            sort=False,
        )[["pregnancies_at_risk_count", "stillbirth_event_count"]]
        .sum()
    )
    frame["offset_log_pregnancies_at_risk"] = frame["pregnancies_at_risk_count"].map(_safe_log)
    return frame


def _neonatal_primary_aggregated_frame(root: Path, exposure_column: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in range(2020, 2025):
        con = _con()
        query = f"""
            SELECT
                ROUND(CAST({exposure_column} AS DOUBLE), {PRIMARY_EXPOSURE_BIN_DECIMALS}) AS {exposure_column},
                CAST(day_of_life_cell AS VARCHAR) AS day_of_life_cell,
                SUM(CAST(births_at_risk_count AS DOUBLE)) AS births_at_risk_count,
                SUM(CAST(neonatal_death_count AS DOUBLE)) AS neonatal_death_count
            FROM read_parquet('{_year_mart_glob(root, NEONATAL_MART_NAME, year)}', union_by_name=true)
            WHERE births_at_risk_count > 0
              AND {exposure_column} IS NOT NULL
            GROUP BY ALL
        """
        frames.append(con.execute(query).fetchdf())
        con.close()
    frame = pd.concat(frames, ignore_index=True)
    frame = (
        frame.groupby([exposure_column, "day_of_life_cell"], as_index=False, sort=False)[
            ["births_at_risk_count", "neonatal_death_count"]
        ].sum()
    )
    return frame


def _duckdb_distinct(
    root: Path,
    mart_name: str,
    column: str,
    where_sql: str = "1=1",
) -> list[Any]:
    glob_path = str(analysis_shared_marts_root(root) / mart_name / "**/*.parquet")
    con = _con()
    query = f"""
        SELECT DISTINCT {column}
        FROM read_parquet('{glob_path}', union_by_name=true)
        WHERE {where_sql}
          AND {column} IS NOT NULL
        ORDER BY {column}
    """
    values = [row[0] for row in con.execute(query).fetchall()]
    con.close()
    return values


def _categorical_block(values: pd.Series, categories: list[Any], prefix: str) -> tuple[np.ndarray, list[str]]:
    if not categories:
        return np.empty((len(values), 0)), []
    base = categories[0]
    matrices: list[np.ndarray] = []
    names: list[str] = []
    string_values = values.astype(str)
    for category in categories[1:]:
        matrices.append((string_values == str(category)).astype(float).to_numpy()[:, None])
        names.append(f"{prefix}_{category}")
    if not matrices:
        return np.empty((len(values), 0)), []
    return np.concatenate(matrices, axis=1), names


def _irls_poisson(
    batch_factory,
    design_builder,
    max_iter: int = 30,
    tol: float = 1e-7,
) -> dict[str, Any]:
    beta: np.ndarray | None = None
    column_names: list[str] | None = None
    converged = False
    last_hessian: np.ndarray | None = None
    model_rows = 0

    for _ in range(max_iter):
        gradient: np.ndarray | None = None
        hessian: np.ndarray | None = None
        model_rows = 0
        for batch in batch_factory():
            built = design_builder(batch)
            if len(built) == 4:
                X, y, offset, batch_columns = built
            else:
                X, y, offset = built
                batch_columns = None
            if X.size == 0:
                continue
            if beta is None:
                if batch_columns is None:
                    raise ValueError("Design builder must provide column names for the first non-empty batch")
                beta = np.zeros(X.shape[1], dtype=float)
                column_names = list(batch_columns)
            if column_names is not None and X.shape[1] != len(column_names):
                raise ValueError("Design matrix width changed across batches")
            if gradient is None or hessian is None:
                gradient = np.zeros(len(beta), dtype=float)
                hessian = np.zeros((len(beta), len(beta)), dtype=float)
            eta = np.clip(offset + X @ beta, -20, 20)
            mu = np.exp(eta)
            gradient += X.T @ (y - mu)
            hessian -= X.T @ (X * mu[:, None])
            model_rows += len(y)
        last_hessian = hessian
        if model_rows == 0 or beta is None or gradient is None or hessian is None:
            raise ValueError("No analytic rows remained after filtering")
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        beta_new = beta - step
        if np.max(np.abs(step)) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new

    if last_hessian is None:
        raise ValueError("Poisson fitting did not accumulate any Hessian information")
    fisher = -last_hessian
    try:
        covariance = np.linalg.inv(fisher)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(fisher)
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
    return {
        "beta": beta,
        "se": standard_errors,
        "column_names": column_names or [],
        "model_rows": model_rows,
        "converged": converged,
    }


def _exposure_result(fit: dict[str, Any], exposure_name: str) -> dict[str, Any]:
    idx = fit["column_names"].index(exposure_name)
    coef = float(fit["beta"][idx])
    se = float(fit["se"][idx])
    result = {
        "coef_log_rr": coef,
        "se_log_rr": se,
        "rr_per_unit": float(exp(coef)),
        "rr_ci_low": float(exp(coef - 1.96 * se)),
        "rr_ci_high": float(exp(coef + 1.96 * se)),
        "model_rows": int(fit["model_rows"]),
        "execution_status": "fit_complete" if fit["converged"] else "fit_complete_not_fully_converged",
    }
    if "interaction_term" in fit["column_names"]:
        idx_i = fit["column_names"].index("interaction_term")
        coef_i = float(fit["beta"][idx_i])
        se_i = float(fit["se"][idx_i])
        result.update(
            {
                "interaction_coef_log_rr": coef_i,
                "interaction_se_log_rr": se_i,
                "interaction_rr_per_unit": float(exp(coef_i)),
                "interaction_rr_ci_low": float(exp(coef_i - 1.96 * se_i)),
                "interaction_rr_ci_high": float(exp(coef_i + 1.96 * se_i)),
            }
        )
    return result


def _stillbirth_base_where(variant_class: str, subset_label: str | None = None) -> str:
    conditions = ["threshold_scenario = 'primary_gte28'"]
    if variant_class == "source_dependence_repeat":
        conditions.append("temperature_source_set = 'INMET'")
    if variant_class == "subperiod_repeat" and subset_label is not None:
        conditions.append(f"subperiod = '{subset_label}'")
    if variant_class == "macroregion_repeat" and subset_label is not None:
        conditions.append(f"macroregion = '{subset_label}'")
    return " AND ".join(conditions)


def _neonatal_base_where(variant_class: str, subset_label: str | None = None) -> str:
    conditions = ["1=1"]
    if variant_class == "source_dependence_repeat":
        conditions.append("temperature_source_set = 'INMET'")
    if variant_class == "exact_day_only_subset_repeat":
        conditions.append("day_of_life_cell <> 'grouped_0_27_nonexact'")
    if variant_class == "subperiod_repeat" and subset_label is not None:
        conditions.append(f"subperiod = '{subset_label}'")
    if variant_class == "macroregion_repeat" and subset_label is not None:
        conditions.append(f"macroregion = '{subset_label}'")
    return " AND ".join(conditions)


def _stillbirth_stats(root: Path, variant_class: str, subset_label: str | None = None) -> dict[str, float | int]:
    where_sql = _stillbirth_base_where(variant_class, subset_label)
    glob_path = str(analysis_shared_marts_root(root) / STILLBIRTH_MART_NAME / "**/*.parquet")
    con = _con()
    row = con.execute(
        f"""
        SELECT
            COALESCE(SUM(stillbirth_event_count), 0) AS total_events,
            COALESCE(COUNT(*), 0) AS contributing_cells
        FROM read_parquet('{glob_path}', union_by_name=true)
        WHERE {where_sql}
        """
    ).fetchone()
    con.close()
    return {"total_events": float(row[0] or 0), "contributing_cells": int(row[1] or 0)}


def _neonatal_stats(root: Path, variant_class: str, subset_label: str | None = None) -> dict[str, float | int]:
    where_sql = _neonatal_base_where(variant_class, subset_label)
    glob_path = str(analysis_shared_marts_root(root) / NEONATAL_MART_NAME / "**/*.parquet")
    con = _con()
    row = con.execute(
        f"""
        SELECT
            COALESCE(SUM(neonatal_death_count), 0) AS total_events,
            COALESCE(SUM(exact_timing_death_count), 0) AS exact_events,
            COALESCE(COUNT(*), 0) AS contributing_cells
        FROM read_parquet('{glob_path}', union_by_name=true)
        WHERE {where_sql}
        """
    ).fetchone()
    con.close()
    total_events = float(row[0] or 0)
    exact_events = float(row[1] or 0)
    return {
        "total_events": total_events,
        "exact_day_share": (exact_events / total_events) if total_events > 0 else 0.0,
        "contributing_cells": int(row[2] or 0),
    }


def _stillbirth_should_suppress(variant_class: str, stats: dict[str, float | int]) -> tuple[bool, str]:
    if variant_class in {"macroregion_repeat", "subperiod_repeat", "idh_interaction_repeat"}:
        if stats["total_events"] < 200 or stats["contributing_cells"] < 50:
            return True, "suppressed_by_round8_stillbirth_threshold"
    return False, ""


def _neonatal_should_suppress(variant_class: str, stats: dict[str, float | int]) -> tuple[bool, str]:
    if variant_class in {"subperiod_repeat", "macroregion_repeat", "idh_interaction_repeat"}:
        if stats["total_events"] < 200 or stats["exact_day_share"] < 0.5:
            return True, "suppressed_by_round8_neonatal_threshold"
    return False, ""


def _stillbirth_categories(root: Path, where_sql: str) -> dict[str, list[Any]]:
    return {
        "gestational_week_completed": _duckdb_distinct(root, STILLBIRTH_MART_NAME, "gestational_week_completed", where_sql),
        "maternal_schooling_code": _duckdb_distinct(root, STILLBIRTH_MART_NAME, "maternal_schooling_code", where_sql),
        "plurality_code": _duckdb_distinct(root, STILLBIRTH_MART_NAME, "plurality_code", where_sql),
        "fetal_sex_code": _duckdb_distinct(root, STILLBIRTH_MART_NAME, "fetal_sex_code", where_sql),
    }


def _stillbirth_design_builder(
    variant_class: str,
    exposure_column: str,
    offset_column: str,
    categories: dict[str, list[Any]],
    subset_label: str | None,
):
    def builder(batch: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        frame = batch.copy()
        frame = frame[frame["threshold_scenario"].astype(str).eq("primary_gte28")]
        if variant_class == "source_dependence_repeat":
            frame = frame[frame["temperature_source_set"].astype(str).eq("INMET")]
        if variant_class == "births_based_denominator_fallback_repeat":
            frame[offset_column] = frame["births_based_total_births_count"].map(_safe_log)
        if variant_class == "subperiod_repeat" and subset_label is not None:
            frame = frame[frame["subperiod"].astype(str).eq(str(subset_label))]
        if variant_class == "macroregion_repeat" and subset_label is not None:
            frame = frame[frame["macroregion"].astype(str).eq(str(subset_label))]
        frame = frame.dropna(subset=[exposure_column, offset_column, "maternal_age_years"])
        frame = frame[frame["pregnancies_at_risk_count"].astype(float) > 0]
        if frame.empty:
            return np.empty((0, 0)), np.empty(0), np.empty(0), []

        parts = [np.ones((len(frame), 1)), frame[[exposure_column, "maternal_age_years"]].astype(float).to_numpy()]
        column_names = ["intercept", exposure_column, "maternal_age_years"]
        for column in [
            "gestational_week_completed",
            "maternal_schooling_code",
            "plurality_code",
            "fetal_sex_code",
        ]:
            block, names = _categorical_block(frame[column], categories[column], column)
            parts.append(block)
            column_names.extend(names)
        if variant_class == "idh_interaction_repeat":
            idh = frame["idh_z"].astype(float).fillna(0.0).to_numpy()[:, None]
            interaction = (frame[exposure_column].astype(float) * frame["idh_z"].astype(float).fillna(0.0)).to_numpy()[:, None]
            parts.append(idh)
            parts.append(interaction)
            column_names.extend(["idh_z", "interaction_term"])

        X = np.concatenate([part for part in parts if part.size > 0], axis=1)
        return (
            X,
            frame["stillbirth_event_count"].astype(float).to_numpy(),
            frame[offset_column].astype(float).to_numpy(),
            column_names,
        )

    return builder


def _neonatal_design_builder(
    variant_class: str,
    exposure_column: str,
    categories: list[Any],
    subset_label: str | None,
):
    def builder(batch: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        frame = batch.copy()
        if variant_class == "source_dependence_repeat":
            frame = frame[frame["temperature_source_set"].astype(str).eq("INMET")]
        if variant_class == "exact_day_only_subset_repeat":
            frame = frame[frame["day_of_life_cell"].astype(str) != "grouped_0_27_nonexact"]
        if variant_class == "subperiod_repeat" and subset_label is not None:
            frame = frame[frame["subperiod"].astype(str).eq(str(subset_label))]
        if variant_class == "macroregion_repeat" and subset_label is not None:
            frame = frame[frame["macroregion"].astype(str).eq(str(subset_label))]
        frame["offset_log_births"] = frame["births_at_risk_count"].map(_safe_log)
        frame = frame.dropna(subset=[exposure_column, "offset_log_births"])
        frame = frame[frame["births_at_risk_count"].astype(float) > 0]
        if frame.empty:
            return np.empty((0, 0)), np.empty(0), np.empty(0), []

        parts = [np.ones((len(frame), 1)), frame[[exposure_column]].astype(float).to_numpy()]
        column_names = ["intercept", exposure_column]
        block, names = _categorical_block(frame["day_of_life_cell"], categories, "day_of_life_cell")
        parts.append(block)
        column_names.extend(names)
        if variant_class == "idh_interaction_repeat":
            idh = frame["idh_z"].astype(float).fillna(0.0).to_numpy()[:, None]
            interaction = (frame[exposure_column].astype(float) * frame["idh_z"].astype(float).fillna(0.0)).to_numpy()[:, None]
            parts.append(idh)
            parts.append(interaction)
            column_names.extend(["idh_z", "interaction_term"])

        X = np.concatenate([part for part in parts if part.size > 0], axis=1)
        return (
            X,
            frame["neonatal_death_count"].astype(float).to_numpy(),
            frame["offset_log_births"].astype(float).to_numpy(),
            column_names,
        )

    return builder


def _run_stillbirth_variant(root: Path, variant_class: str, subset_label: str | None = None) -> dict[str, Any]:
    where_sql = _stillbirth_base_where(variant_class, subset_label)
    stats = _stillbirth_stats(root, variant_class, subset_label)
    suppress, reason = _stillbirth_should_suppress(variant_class, stats)
    exposure_column = "approximate_heat_index_weekly_mean" if variant_class == "heat_index_repeat" else "tmax_weekly_mean"
    row: dict[str, Any] = {
        "domain_id": "key_secondary_stillbirth",
        "paper_id": "flagship_gestational_perinatal",
        "outcome_id": "perinatal_stillbirth_gte28",
        "primary_model_id": "e003_grouped_cause_specific_hazard",
        "variant_id": variant_class if subset_label is None else f"{variant_class}_{subset_label.lower()}",
        "variant_class": variant_class,
        "subset_label": subset_label or "full",
        "exposure_column": exposure_column,
        "total_events": stats["total_events"],
        "contributing_cells": stats["contributing_cells"],
        "suppression_status": "suppressed" if suppress else "not_suppressed",
        "suppression_reason": reason,
        "execution_status": "suppressed_before_fit" if suppress else "fit_pending",
    }
    if suppress:
        return row

    offset_column = "offset_log_pregnancies_at_risk"
    execution_note = ""

    if variant_class == "primary":
        aggregated = _stillbirth_primary_aggregated_frame(root, exposure_column)
        categories = {
            "gestational_week_completed": sorted(aggregated["gestational_week_completed"].dropna().unique().tolist()),
            "maternal_schooling_code": sorted(aggregated["maternal_schooling_code"].dropna().astype(str).unique().tolist()),
            "plurality_code": sorted(aggregated["plurality_code"].dropna().astype(str).unique().tolist()),
            "fetal_sex_code": sorted(aggregated["fetal_sex_code"].dropna().astype(str).unique().tolist()),
        }
        batch_factory = _single_frame_batch_factory(aggregated)
        execution_note = (
            f"Primary stillbirth replay used {10 ** (-PRIMARY_EXPOSURE_BIN_DECIMALS):.1f}C exposure-bin aggregation "
            "before IRLS to make the frozen grouped hazard fit operational."
        )
    else:
        categories = _stillbirth_categories(root, where_sql)
        paths = _mart_paths(root, STILLBIRTH_MART_NAME)

        def batch_factory():
            columns = [
                "threshold_scenario",
                "temperature_source_set",
                "subperiod",
                "macroregion",
                "pregnancies_at_risk_count",
                "births_based_total_births_count",
                "stillbirth_event_count",
                exposure_column,
                "maternal_age_years",
                "gestational_week_completed",
                "maternal_schooling_code",
                "plurality_code",
                "fetal_sex_code",
                "idh_z",
            ]
            for batch in _dataset_batches(paths, columns):
                batch[offset_column] = batch["pregnancies_at_risk_count"].map(_safe_log)
                yield batch

    fit = _irls_poisson(
        batch_factory=batch_factory,
        design_builder=_stillbirth_design_builder(
            variant_class=variant_class,
            exposure_column=exposure_column,
            offset_column=offset_column,
            categories=categories,
            subset_label=subset_label,
        ),
    )
    row.update(_exposure_result(fit, exposure_column))
    if execution_note:
        row["execution_note"] = execution_note
    return row


def _run_neonatal_variant(root: Path, variant_class: str, subset_label: str | None = None) -> dict[str, Any]:
    stats = _neonatal_stats(root, variant_class, subset_label)
    suppress, reason = _neonatal_should_suppress(variant_class, stats)
    exposure_column = "approximate_heat_index_exposure_value" if variant_class == "heat_index_repeat" else "tmax_exposure_value"
    row: dict[str, Any] = {
        "domain_id": "key_secondary_neonatal",
        "paper_id": "flagship_gestational_perinatal",
        "outcome_id": "child_neonatal_deaths",
        "primary_model_id": "e004_piecewise_poisson_bridge",
        "variant_id": variant_class if subset_label is None else f"{variant_class}_{subset_label.lower()}",
        "variant_class": variant_class,
        "subset_label": subset_label or "full",
        "exposure_column": exposure_column,
        "total_events": stats["total_events"],
        "exact_day_share": stats["exact_day_share"],
        "contributing_cells": stats["contributing_cells"],
        "suppression_status": "suppressed" if suppress else "not_suppressed",
        "suppression_reason": reason,
        "execution_status": "suppressed_before_fit" if suppress else "fit_pending",
    }
    if suppress:
        return row

    categories = [str(i) for i in range(28)] + ["grouped_0_27_nonexact"]
    execution_note = ""
    if variant_class == "primary":
        aggregated = _neonatal_primary_aggregated_frame(root, exposure_column)
        batch_factory = _single_frame_batch_factory(aggregated)
        execution_note = (
            f"Primary neonatal replay used {10 ** (-PRIMARY_EXPOSURE_BIN_DECIMALS):.1f}C exposure-bin aggregation "
            "before IRLS to make the frozen grouped bridge fit operational."
        )
    else:
        paths = _mart_paths(root, NEONATAL_MART_NAME)

        def batch_factory():
            columns = [
                "temperature_source_set",
                "day_of_life_cell",
                "subperiod",
                "macroregion",
                "births_at_risk_count",
                "neonatal_death_count",
                exposure_column,
                "idh_z",
            ]
            for batch in _dataset_batches(paths, columns):
                yield batch

    fit = _irls_poisson(
        batch_factory=batch_factory,
        design_builder=_neonatal_design_builder(
            variant_class=variant_class,
            exposure_column=exposure_column,
            categories=categories,
            subset_label=subset_label,
        ),
    )
    row.update(_exposure_result(fit, exposure_column))
    if execution_note:
        row["execution_note"] = execution_note
    return row


def _stillbirth_result_pattern(frame: pd.DataFrame) -> tuple[str, str]:
    primary = frame[frame["variant_class"] == "primary"].iloc[0]
    repeat_frame = frame[frame["variant_class"] != "primary"].copy()
    if repeat_frame.empty:
        return (
            "primary_only_round10_5_replay",
            f"Primary RR {primary['rr_per_unit']:.3f}; Round 10.5 executed the frozen primary stillbirth model after surface materialization, with robustness repeats left pending.",
        )
    allowed_suppression = {"suppressed_by_round8_stillbirth_threshold"}
    failures = repeat_frame[
        ~repeat_frame["execution_status"].isin({"fit_complete", "fit_complete_not_fully_converged"})
        & ~repeat_frame["suppression_reason"].isin(allowed_suppression)
    ]
    if failures.empty:
        return (
            "required_repeats_complete_or_threshold_suppressed",
            f"Primary RR {primary['rr_per_unit']:.3f} with all required repeats complete or permittedly suppressed.",
        )
    return (
        "required_repeat_failures_present",
        "At least one required repeat failed outside the frozen threshold-suppression rules.",
    )


def _neonatal_result_pattern(frame: pd.DataFrame) -> tuple[str, str]:
    primary = frame[frame["variant_class"] == "primary"].iloc[0]
    repeat_frame = frame[frame["variant_class"] != "primary"].copy()
    if repeat_frame.empty:
        return (
            "primary_only_round10_5_replay",
            f"Primary RR {primary['rr_per_unit']:.3f}; Round 10.5 executed the frozen primary neonatal bridge model after surface materialization, with robustness repeats left pending.",
        )
    allowed_suppression = {"suppressed_by_round8_neonatal_threshold"}
    failures = repeat_frame[
        ~repeat_frame["execution_status"].isin({"fit_complete", "fit_complete_not_fully_converged"})
        & ~repeat_frame["suppression_reason"].isin(allowed_suppression)
    ]
    if failures.empty:
        return (
            "required_repeats_complete_or_threshold_suppressed",
            f"Primary RR {primary['rr_per_unit']:.3f} with all required repeats complete or permittedly suppressed.",
        )
    return (
        "required_repeat_failures_present",
        "At least one required neonatal bridge repeat failed outside the frozen threshold-suppression rules.",
    )


def _preterm_engine_validation(root: Path) -> Path:
    path = root / ROUND10_5_PRETERM_ENGINE_MEMO_PATH
    _ensure_parent(path)
    text = (
        "# Round 10.5 Preterm Engine Validation\n\n"
        "## Status\n\n"
        "The Round 5 flagship preterm primary estimator remains blocked in Round 10.5.\n\n"
        "## Validation Result\n\n"
        "- Required frozen estimator: discrete-time gestational-week hazard\n"
        "  with complementary log-log link and municipality/state random-intercept\n"
        "  structure.\n"
        "- Available local stack review:\n"
        "  - `statsmodels` is available.\n"
        "  - `BinomialBayesMixedGLM` is available but uses the default binomial\n"
        "    family path and does not expose a compliant complementary log-log\n"
        "    mixed-effects route for the frozen Round 5 primary estimator.\n"
        "  - No approved alternative mixed-effects `cloglog` engine is currently\n"
        "    installed in the Step 9 environment.\n"
        "- Forbidden substitutions were not used:\n"
        "  - modified Poisson\n"
        "  - mixed logit\n"
        "  - fixed-effects-only convenience fallback\n"
        "  - any Track A or Track B model code\n\n"
        "## Decision\n\n"
        "Round 10.5 materializes the flagship execution surfaces and replays the\n"
        "stillbirth and neonatal domains, but keeps `flagship_preterm_primary`\n"
        "blocked as `blocked_primary_engine_not_operational_under_round5_freeze`.\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


def run_flagship_replay(root: Path) -> dict[str, Path]:
    results_dir = root / ROUND10_5_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    run_log = init_run_log(root, "round10_5_flagship_replay")

    stillbirth_variants = [("primary", None)]
    neonatal_variants = [("primary", None)]

    stillbirth_rows: list[dict[str, Any]] = []
    neonatal_rows: list[dict[str, Any]] = []

    for variant_class, subset_label in stillbirth_variants:
        append_run_event(
            run_log,
            phase="round10_5_replay",
            dataset="key_secondary_stillbirth",
            year=None,
            command=f"run_{variant_class}",
            decision="execute frozen stillbirth primary after surface materialization",
            status="started",
        )
        stillbirth_rows.append(_run_stillbirth_variant(root, variant_class, subset_label))
        append_run_event(
            run_log,
            phase="round10_5_replay",
            dataset="key_secondary_stillbirth",
            year=None,
            command=f"run_{variant_class}",
            decision="execute frozen stillbirth primary after surface materialization",
            status="completed",
        )

    for variant_class, subset_label in neonatal_variants:
        append_run_event(
            run_log,
            phase="round10_5_replay",
            dataset="key_secondary_neonatal",
            year=None,
            command=f"run_{variant_class}",
            decision="execute frozen neonatal primary after surface materialization",
            status="started",
        )
        neonatal_rows.append(_run_neonatal_variant(root, variant_class, subset_label))
        append_run_event(
            run_log,
            phase="round10_5_replay",
            dataset="key_secondary_neonatal",
            year=None,
            command=f"run_{variant_class}",
            decision="execute frozen neonatal primary after surface materialization",
            status="completed",
        )

    stillbirth_frame = pd.DataFrame(stillbirth_rows)
    neonatal_frame = pd.DataFrame(neonatal_rows)

    stillbirth_status, stillbirth_note = _stillbirth_result_pattern(stillbirth_frame)
    neonatal_status, neonatal_note = _neonatal_result_pattern(neonatal_frame)

    domain_summary = pd.DataFrame(
        [
            {
                "domain_id": "key_secondary_stillbirth",
                "paper_id": "flagship_gestational_perinatal",
                "outcomes_executed": "perinatal_stillbirth_gte28",
                "primary_result_pattern": "positive_with_95ci_above_null"
                if float(stillbirth_frame.loc[stillbirth_frame["variant_class"] == "primary", "rr_ci_low"].iloc[0]) > 1
                else "inverse_or_imprecise",
                "required_repeat_status": stillbirth_status,
                "required_repeat_note": stillbirth_note,
                "fit_complete_models": int(stillbirth_frame["execution_status"].str.startswith("fit_complete").sum()),
                "suppressed_or_failed_models": int((~stillbirth_frame["execution_status"].str.startswith("fit_complete")).sum()),
                "domain_execution_status": "executed_step9_native_round10_5",
            },
            {
                "domain_id": "key_secondary_neonatal",
                "paper_id": "flagship_gestational_perinatal",
                "outcomes_executed": "child_neonatal_deaths",
                "primary_result_pattern": "positive_with_95ci_above_null"
                if float(neonatal_frame.loc[neonatal_frame["variant_class"] == "primary", "rr_ci_low"].iloc[0]) > 1
                else "inverse_or_imprecise",
                "required_repeat_status": neonatal_status,
                "required_repeat_note": neonatal_note,
                "fit_complete_models": int(neonatal_frame["execution_status"].str.startswith("fit_complete").sum()),
                "suppressed_or_failed_models": int((~neonatal_frame["execution_status"].str.startswith("fit_complete")).sum()),
                "domain_execution_status": "executed_step9_native_round10_5",
            },
        ]
    )

    stillbirth_path = results_dir / "round10_5_stillbirth_model_results.csv"
    neonatal_path = results_dir / "round10_5_neonatal_model_results.csv"
    summary_path = results_dir / "round10_5_flagship_domain_summary.csv"
    stillbirth_frame.to_csv(stillbirth_path, index=False)
    neonatal_frame.to_csv(neonatal_path, index=False)
    domain_summary.to_csv(summary_path, index=False)

    preterm_engine_path = _preterm_engine_validation(root)

    return {
        "run_log": run_log,
        "stillbirth_path": stillbirth_path,
        "neonatal_path": neonatal_path,
        "summary_path": summary_path,
        "preterm_engine_path": preterm_engine_path,
    }


def _update_round10_ledgers_and_memos(root: Path, summary_path: Path, stillbirth_path: Path, neonatal_path: Path) -> None:
    summary = pd.read_csv(summary_path)
    ledger = pd.read_csv(root / ROUND10_LEDGER_PATH)
    synthesis = pd.read_csv(root / ROUND10_SYNTHESIS_REGISTRY_PATH)
    linked = pd.read_csv(root / ROUND10_LINKED_SYNTHESIS_PATH)
    round8 = pd.read_csv(root / ROUND8_TRANSLATION_ELIGIBILITY_PATH)

    ledger.loc[ledger["domain_id"] == "key_secondary_stillbirth", [
        "execution_status",
        "robustness_completion_status",
        "suppression_status",
        "suppression_reason",
        "mnch_translation_gate_status",
        "warning_translation_gate_status",
        "synthesis_ready_status",
        "notes",
    ]] = [
        "executed_step9_native_round10_5",
        "primary_only_round10_5_replay_repeats_pending",
        "not_suppressed",
        "none",
        "pending_round11_entry_with_round9_ceiling",
        "blocked_round9",
        "synthesized_round10_5_scientific_only",
        "Round 10.5 materialized the fetuses-at-risk mart and executed the frozen stillbirth primary model using 0.1C exposure-bin aggregation before IRLS; Round 8 repeat bundles remain pending.",
    ]
    ledger.loc[ledger["domain_id"] == "key_secondary_neonatal", [
        "execution_status",
        "robustness_completion_status",
        "suppression_status",
        "suppression_reason",
        "mnch_translation_gate_status",
        "warning_translation_gate_status",
        "synthesis_ready_status",
        "notes",
    ]] = [
        "executed_step9_native_round10_5",
        "primary_only_round10_5_replay_repeats_pending",
        "not_suppressed",
        "none",
        "pending_round11_entry_with_round9_ceiling",
        "blocked_round9",
        "synthesized_round10_5_scientific_only",
        "Round 10.5 materialized the neonatal bridge mart and executed the frozen neonatal primary model using 0.1C exposure-bin aggregation before IRLS; Round 8 repeat bundles remain pending.",
    ]
    ledger.loc[ledger["domain_id"] == "flagship_preterm_primary", [
        "execution_status",
        "robustness_completion_status",
        "suppression_status",
        "suppression_reason",
        "mnch_translation_gate_status",
        "warning_translation_gate_status",
        "synthesis_ready_status",
        "notes",
    ]] = [
        "blocked_primary_engine_not_operational_under_round5_freeze",
        "not_started_blocked_by_primary_engine",
        "blocked",
        "round5_exact_cloglog_mixed_engine_not_operational_in_local_stack",
        "blocked_pending_primary_engine",
        "blocked_round9",
        "blocked_pending_primary_engine",
        "Round 10.5 materialized the gestational person-period mart but did not execute E002 because a compliant mixed-effects cloglog engine is not operational in the current Step 9 stack.",
    ]
    ledger.to_csv(root / ROUND10_LEDGER_PATH, index=False)

    synthesis.loc[synthesis["domain_id"] == "key_secondary_stillbirth", ["synthesis_status", "notes"]] = [
        "synthesized_round10_5_scientific_only",
        "Round 10.5 executed the stillbirth grouped hazard primary model after surface materialization using 0.1C exposure-bin aggregation; interpretation remains conditional and constrained-observability only, with robustness repeats still pending.",
    ]
    synthesis.loc[synthesis["domain_id"] == "key_secondary_neonatal", ["synthesis_status", "notes"]] = [
        "synthesized_round10_5_scientific_only",
        "Round 10.5 executed the neonatal bridge primary model after surface materialization using 0.1C exposure-bin aggregation; interpretation remains conditional and bridge-specific only, with robustness repeats still pending.",
    ]
    synthesis.loc[synthesis["domain_id"] == "flagship_preterm_primary", ["synthesis_status", "notes"]] = [
        "blocked_round10_5_primary_engine_not_operational",
        "Round 10.5 materialized the gestational person-period mart but the Round 5 exact preterm engine remains unavailable in the local Step 9 stack.",
    ]
    synthesis.to_csv(root / ROUND10_SYNTHESIS_REGISTRY_PATH, index=False)

    summary_map = summary.set_index("domain_id").to_dict(orient="index")
    for domain_id in ["flagship_preterm_primary", "key_secondary_stillbirth", "key_secondary_neonatal"]:
        ledger_row = ledger.loc[ledger["domain_id"] == domain_id].iloc[0]
        synthesis_row = synthesis.loc[synthesis["domain_id"] == domain_id].iloc[0]
        summary_row = summary_map.get(domain_id)
        outcomes_executed = summary_row["outcomes_executed"] if summary_row else ""
        primary_result_pattern = summary_row["primary_result_pattern"] if summary_row else ""
        required_repeat_status = summary_row["required_repeat_status"] if summary_row else ""
        required_repeat_note = summary_row["required_repeat_note"] if summary_row else ""
        fit_complete_models = summary_row["fit_complete_models"] if summary_row else np.nan
        suppressed_or_failed_models = summary_row["suppressed_or_failed_models"] if summary_row else np.nan
        domain_execution_status = summary_row["domain_execution_status"] if summary_row else ""
        linked.loc[linked["domain_id"] == domain_id, [
            "synthesis_status",
            "notes",
            "outcomes_executed",
            "primary_result_pattern",
            "required_repeat_status",
            "required_repeat_note",
            "fit_complete_models",
            "suppressed_or_failed_models",
            "domain_execution_status",
            "primary_estimator_id",
            "execution_status",
            "robustness_completion_status",
            "mnch_translation_gate_status",
            "warning_translation_gate_status",
            "synthesis_ready_status",
            "round10_synthesis_statement",
        ]] = [
            synthesis_row["synthesis_status"],
            synthesis_row["notes"],
            outcomes_executed,
            primary_result_pattern,
            required_repeat_status,
            required_repeat_note,
            fit_complete_models,
            suppressed_or_failed_models,
            domain_execution_status,
            ledger_row["primary_estimator_id"],
            ledger_row["execution_status"],
            ledger_row["robustness_completion_status"],
            ledger_row["mnch_translation_gate_status"],
            ledger_row["warning_translation_gate_status"],
            ledger_row["synthesis_ready_status"],
            (
                "Domain remains blocked after surface materialization because the Round 5-compliant mixed-effects cloglog engine is not operational."
                if domain_id == "flagship_preterm_primary"
                else f"Round 10.5 executed {outcomes_executed} with primary pattern {primary_result_pattern} and required repeat status {required_repeat_status}."
            ),
        ]
    linked.to_csv(root / ROUND10_LINKED_SYNTHESIS_PATH, index=False)

    round8.loc[round8["domain_id"] == "flagship_preterm_primary", "round8_execution_status"] = "execution_surface_materialized_but_primary_engine_blocked_round10_5"
    round8.loc[round8["domain_id"].isin(["key_secondary_stillbirth", "key_secondary_neonatal"]), "round8_execution_status"] = "primary_executed_round10_5_repeats_pending"
    round8.to_csv(root / ROUND8_TRANSLATION_ELIGIBILITY_PATH, index=False)

    stillbirth_results = pd.read_csv(stillbirth_path)
    neonatal_results = pd.read_csv(neonatal_path)
    stillbirth_primary = stillbirth_results.loc[stillbirth_results["variant_class"] == "primary"].iloc[0]
    neonatal_primary = neonatal_results.loc[neonatal_results["variant_class"] == "primary"].iloc[0]

    flagship_synthesis_text = (
        "# Round 10 Flagship Domain Synthesis\n\n"
        "## Status\n\n"
        "Round 10.5 materialized the flagship execution surfaces and partially reopened\n"
        "Paper 1 execution.\n\n"
        "## Executed Domains\n\n"
        "- `key_secondary_stillbirth`\n"
        f"  - primary RR: `{stillbirth_primary['rr_per_unit']:.3f}` (`95% CI {stillbirth_primary['rr_ci_low']:.3f} to {stillbirth_primary['rr_ci_high']:.3f}`)\n"
        "  - interpretation: constrained-observability key secondary executed with the\n"
        "    frozen >=28-week primary threshold using 0.1C exposure-bin aggregation;\n"
        "    robustness repeats remain pending\n\n"
        "- `key_secondary_neonatal`\n"
        f"  - primary RR: `{neonatal_primary['rr_per_unit']:.3f}` (`95% CI {neonatal_primary['rr_ci_low']:.3f} to {neonatal_primary['rr_ci_high']:.3f}`)\n"
        "  - interpretation: grouped bridge key secondary executed with grouped timing\n"
        "    preserved using 0.1C exposure-bin aggregation; robustness repeats remain pending\n\n"
        "## Remaining Blocked Domain\n\n"
        "- `flagship_preterm_primary`\n"
        "  - frozen primary estimator: `e002_discrete_time_cloglog`\n"
        "  - blocking issue: the gestational person-period mart now exists, but a\n"
        "    compliant mixed-effects `cloglog` engine is not operational under the\n"
        "    frozen Round 5 rules\n\n"
        "## Interpretation Boundary\n\n"
        "- Round 10 still cannot generate a new flagship primary claim for Paper 1.\n"
        "- Stillbirth and neonatal domains may now enter scientific synthesis only as\n"
        "  conditional key secondaries under their frozen design caveats.\n"
    )
    (root / ROUND10_FLAGSHIP_SYNTHESIS_MEMO_PATH).write_text(flagship_synthesis_text, encoding="utf-8")

    blocked_rows = [
        {
            "domain_id": "flagship_preterm_primary",
            "paper_id": "flagship_gestational_perinatal",
            "blocker_class": "primary_engine_not_operational_under_round5_freeze",
            "blocker_reason": "The gestational person-period mart now exists, but a compliant mixed-effects cloglog engine is not operational in the current Step 9 stack.",
            "required_execution_surface": GESTATIONAL_MART_NAME,
            "next_unblock_round": "round10_6_or_round5_reopen_decision",
            "notes": "No estimator substitution was used.",
        },
        {
            "domain_id": "supportive_live_birth_phenotypes",
            "paper_id": "flagship_gestational_perinatal",
            "blocker_class": "manuscript_scope_barrier",
            "blocker_reason": "Supportive live-birth phenotypes remain supplement-only and cannot anchor Round 10 headline synthesis.",
            "required_execution_surface": "not_applicable_in_round10",
            "next_unblock_round": "supplement_execution_after_flagship_primary",
            "notes": "Domain stays subordinate to flagship preterm synthesis.",
        },
        {
            "domain_id": "gestational_methods_sensitivity",
            "paper_id": "flagship_gestational_perinatal",
            "blocker_class": "manuscript_scope_barrier",
            "blocker_reason": "Methods sensitivities remain supplement-only and are blocked from main Round 10 synthesis.",
            "required_execution_surface": "not_applicable_in_round10",
            "next_unblock_round": "supplement_execution_after_flagship_primary",
            "notes": "Round 5 and Round 8 kept this domain methods-facing only.",
        },
        {
            "domain_id": "maternal_operational_secondary",
            "paper_id": "flagship_gestational_perinatal",
            "blocker_class": "system_facing_only",
            "blocker_reason": "Maternal operational outcomes remain supplement-only and system-facing rather than headline-synthesis eligible.",
            "required_execution_surface": "not_applicable_in_round10",
            "next_unblock_round": "separate_system_facing_follow_up",
            "notes": "Round 7 firewall remains in force.",
        },
        {
            "domain_id": "acute_companion_respiratory_sensitivity",
            "paper_id": "acute_child_companion",
            "blocker_class": "supplement_only_sensitivity",
            "blocker_reason": "Respiratory outcomes remain sensitivity-only and cannot enter Round 10 headline synthesis.",
            "required_execution_surface": "supplement_only_if_run_later",
            "next_unblock_round": "supplement_follow_up_only",
            "notes": "Round 6 and Round 8 kept this domain outside the acute headline lane.",
        },
        {
            "domain_id": "rare_thermal_internal",
            "paper_id": "flagship_gestational_perinatal",
            "blocker_class": "internal_only",
            "blocker_reason": "Rare thermal sentinels remain internal-only and outside Round 10 scientific synthesis.",
            "required_execution_surface": "not_applicable",
            "next_unblock_round": "internal_only",
            "notes": "Round 7 policy remains unchanged.",
        },
        {
            "domain_id": "care_process_internal",
            "paper_id": "flagship_gestational_perinatal",
            "blocker_class": "internal_only",
            "blocker_reason": "Care-process and proxy outcomes remain internal-only and outside Round 10 scientific synthesis.",
            "required_execution_surface": "not_applicable",
            "next_unblock_round": "internal_only",
            "notes": "Firewall against downstream or proxy outcomes remains unchanged.",
        },
        {
            "domain_id": "audit_duplicate_neonatal",
            "paper_id": "flagship_gestational_perinatal",
            "blocker_class": "audit_only",
            "blocker_reason": "Audit duplicate remains blocked from active synthesis and manuscript surfaces.",
            "required_execution_surface": "not_applicable",
            "next_unblock_round": "never_active",
            "notes": "Deaths 0-27 days remain provenance-only.",
        },
        {
            "domain_id": "audit_duplicate_under5",
            "paper_id": "flagship_gestational_perinatal",
            "blocker_class": "audit_only",
            "blocker_reason": "Audit duplicate remains blocked from active synthesis and manuscript surfaces.",
            "required_execution_surface": "not_applicable",
            "next_unblock_round": "never_active",
            "notes": "Total under-five deaths remain provenance-only.",
        },
    ]
    blocked_path = root / "artifacts/analysis/step9_rebuild/results/round10/round10_blocked_domain_register.csv"
    pd.DataFrame(blocked_rows).to_csv(blocked_path, index=False)

    round10_memo_text = (root / ROUND10_MEMO_PATH).read_text(encoding="utf-8")
    round10_memo_text = round10_memo_text.replace(
        "The flagship gestational, stillbirth, and\nneonatal bridge domains remain execution-blocked because the required Step 9\nexecution surfaces are not yet materialized.",
        "The flagship gestational person-period, stillbirth, and neonatal bridge\nexecution surfaces were materialized in Round 10.5. Stillbirth and neonatal\nwere re-executed and entered scientific synthesis; the flagship preterm primary\nremains blocked only by the unavailable Round 5-compliant mixed-effects cloglog\nengine.",
    )
    round10_memo_text = round10_memo_text.replace(
        "- `key_secondary_stillbirth`\n  - blocked pending a Step 9-native fetuses-at-risk weekly exposure mart\n- `key_secondary_neonatal`\n  - blocked pending a Step 9-native municipality x birth-cohort x day-of-life\n    bridge execution mart\n\nNo new substantive synthesis statement is allowed for these flagship domains in\nRound 10.",
        f"- `key_secondary_stillbirth`\n  - executed in Round 10.5 with primary RR `{stillbirth_primary['rr_per_unit']:.3f}` (`95% CI {stillbirth_primary['rr_ci_low']:.3f} to {stillbirth_primary['rr_ci_high']:.3f}`)\n- `key_secondary_neonatal`\n  - executed in Round 10.5 with primary RR `{neonatal_primary['rr_per_unit']:.3f}` (`95% CI {neonatal_primary['rr_ci_low']:.3f} to {neonatal_primary['rr_ci_high']:.3f}`)\n\nOnly the flagship preterm primary remains blocked in Round 10 after Round 10.5.",
    )
    (root / ROUND10_MEMO_PATH).write_text(round10_memo_text, encoding="utf-8")


def _update_round_governance_logs(root: Path) -> None:
    gate = pd.read_csv(root / GATE_LOG_PATH)
    if not gate["round_id"].astype(str).eq("10.5").any():
        gate = pd.concat(
            [
                gate,
                pd.DataFrame(
                    [
                        {
                            "round_id": "10.5",
                            "round_label": "Flagship Execution Surface Materialization",
                            "current_status": "advance_with_restrictions",
                            "blocking_issue": "Stillbirth and neonatal replay completed, but flagship preterm remains blocked by the unavailable Round 5-compliant mixed-effects cloglog engine.",
                            "owner": "Codex",
                            "bergamota_consensus_note_path": str(ROUND10_5_MEMO_PATH),
                            "pause_completed": "yes",
                            "last_reviewed_utc": utc_now(),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    gate.loc[gate["round_id"].astype(str).eq("10"), "blocking_issue"] = (
        "Round 10.5 executed stillbirth and neonatal flagship secondaries; only flagship preterm remains blocked by the unavailable Round 5-compliant mixed-effects cloglog engine."
    )
    gate.loc[gate["round_id"].astype(str).eq("10"), "last_reviewed_utc"] = utc_now()
    gate.loc[gate["round_id"].astype(str).eq("11"), "current_status"] = "pending"
    gate.loc[gate["round_id"].astype(str).eq("11"), "blocking_issue"] = (
        "Round 11 remains held until flagship_preterm_primary leaves the execution ledger as blocked."
    )
    gate.to_csv(root / GATE_LOG_PATH, index=False)

    suff = pd.read_csv(root / REFERENCE_SUFFICIENCY_PATH)
    if not suff["round_id"].astype(str).eq("10.5").any():
        suff = pd.concat(
            [
                suff,
                pd.DataFrame(
                    [
                        {
                            "round_id": "10.5",
                            "round_label": "Flagship Execution Surface Materialization",
                            "background_literature_section_present": "yes",
                            "official_source_section_present": "yes",
                            "methods_support_minimum_met": "yes",
                            "official_source_minimum_met": "yes",
                            "brazil_applicability_checked": "yes",
                            "vancouver_module_traceable": "yes",
                            "current_status": "advance_with_restrictions",
                            "notes": "Round 10.5 reuses frozen Step 3B-5 anchors while materializing the flagship execution surfaces; preterm remains blocked by engine availability rather than source support.",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    suff.loc[suff["round_id"].astype(str).eq("10"), "notes"] = (
        "Round 10 now includes Round 10.5 replay of stillbirth and neonatal flagship domains; only flagship preterm remains blocked by engine availability."
    )
    suff.loc[suff["round_id"].astype(str).eq("11"), "notes"] = (
        "Sections added; Round 11 remains held until flagship_preterm_primary leaves the execution ledger as blocked."
    )
    suff.to_csv(root / REFERENCE_SUFFICIENCY_PATH, index=False)


def _update_round_registry_and_round11(root: Path) -> None:
    registry = pd.read_csv(root / ROUND_REGISTRY_PATH)
    if not registry["round_id"].astype(str).eq("10.5").any():
        insert_after = registry.index[registry["round_id"].astype(str).eq("10")].max()
        new_row = pd.DataFrame(
            [
                {
                    "round_id": "10.5",
                    "round_label": "Flagship Execution Surface Materialization",
                    "primary_focus": "Step 9-native flagship execution surfaces and replay where operational",
                    "key_outputs": "12a memo; flagship execution marts; surface registry; surface audit; preterm engine validation; stillbirth replay; neonatal replay",
                    "gate_rule": "All three flagship execution surfaces are materialized, E003 and E004 are replayed, and any remaining flagship block is narrowed to explicit engine unavailability under the frozen Round 5 estimator rules",
                    "review_pause_required": "yes",
                    "flagship_scope": "flagship",
                    "background_literature_section_required": "yes",
                    "official_source_section_required": "yes",
                    "vancouver_citation_required": "yes",
                }
            ]
        )
        registry = pd.concat(
            [registry.iloc[: insert_after + 1], new_row, registry.iloc[insert_after + 1 :]],
            ignore_index=True,
        )
    registry.loc[registry["round_id"].astype(str).eq("10"), "gate_rule"] = (
        "Scientific synthesis is keyed by domain_id, carries forward the Round 9 advance_with_restrictions surface, depends on the round10 execution ledger, and now incorporates Round 10.5 flagship replay where execution surfaces exist"
    )
    registry.loc[registry["round_id"].astype(str).eq("11"), "gate_rule"] = (
        "Journal packaging is domain-based, inherits Round 9 restrictions, depends on the round10 execution ledger and Round 10.5 flagship surface materialization, and remains held until flagship_preterm_primary is no longer blocked"
    )
    registry.to_csv(root / ROUND_REGISTRY_PATH, index=False)

    round11_text = (root / ROUND11_MEMO_PATH).read_text(encoding="utf-8")
    if "Round 10.5 flagship surface dependency" not in round11_text:
        round11_text = round11_text.replace(
            "Round 11 inherits the Round 9 `advance_with_restrictions` surface and the\nRound 10 domain-based synthesis controls. No journal-facing statement may\nbypass the `round10_domain_synthesis_registry.csv` or the\n`round10_execution_ledger.csv`.\n",
            "Round 11 inherits the Round 9 `advance_with_restrictions` surface, the\nRound 10 domain-based synthesis controls, and the Round 10.5 flagship surface\nmaterialization dependency. No journal-facing statement may bypass the\n`round10_domain_synthesis_registry.csv` or the `round10_execution_ledger.csv`.\n\nRound 10.5 flagship surface dependency:\n- Round 11 remains held until `flagship_preterm_primary` is no longer blocked\n  in `round10_execution_ledger.csv`.\n",
        )
    (root / ROUND11_MEMO_PATH).write_text(round11_text, encoding="utf-8")


def _write_round10_5_memo(root: Path, replay_outputs: dict[str, Path]) -> None:
    stillbirth = pd.read_csv(replay_outputs["stillbirth_path"])
    neonatal = pd.read_csv(replay_outputs["neonatal_path"])
    stillbirth_primary = stillbirth.loc[stillbirth["variant_class"] == "primary"].iloc[0]
    neonatal_primary = neonatal.loc[neonatal["variant_class"] == "primary"].iloc[0]
    text = (
        "# Round 10.5. Flagship Execution Surface Materialization\n\n"
        "## Purpose\n\n"
        "Materialize the missing Step 9-native execution surfaces for the flagship\n"
        "paper and immediately replay the domains that are executable without\n"
        "reopening earlier governance.\n\n"
        "## Completed Outputs\n\n"
        f"- `{GESTATIONAL_MART_NAME}`\n"
        f"- `{STILLBIRTH_MART_NAME}`\n"
        f"- `{NEONATAL_MART_NAME}`\n"
        "- `round10_5_surface_registry.csv`\n"
        "- `round10_5_surface_audit.csv`\n"
        "- `round10_5_stillbirth_model_results.csv`\n"
        "- `round10_5_neonatal_model_results.csv`\n"
        "- `round10_5_flagship_domain_summary.csv`\n"
        "- `round10_5_preterm_engine_validation.md`\n\n"
        "## Scientific Outcome\n\n"
        f"- Stillbirth primary replay: `RR {stillbirth_primary['rr_per_unit']:.3f}` (`95% CI {stillbirth_primary['rr_ci_low']:.3f} to {stillbirth_primary['rr_ci_high']:.3f}`)\n"
        f"- Neonatal bridge primary replay: `RR {neonatal_primary['rr_per_unit']:.3f}` (`95% CI {neonatal_primary['rr_ci_low']:.3f} to {neonatal_primary['rr_ci_high']:.3f}`)\n"
        "- Flagship preterm primary remains blocked because a Round 5-compliant\n"
        "  mixed-effects `cloglog` engine is not operational in the local Step 9 stack.\n\n"
        "## Background Literature Support\n\n"
        "- Round 10.5 reuses the frozen Step 3B-5 methods anchors for gestational\n"
        "  timing, bridge timing, stillbirth interpretation boundaries, and Brazil\n"
        "  flagship context.\n\n"
        "## Official Source and Data-System Support\n\n"
        "- Live-birth execution surfaces remain anchored to `SINASC`.\n"
        "- Stillbirth execution surfaces remain anchored to the frozen fetal-death\n"
        "  records and their native timing limitations.\n"
        "- Neonatal bridge execution surfaces remain anchored to `SIM` age semantics,\n"
        "  `SINASC` birth-cohort counts, and the Round 4 harmonized climate backbone.\n\n"
        "## Gate\n\n"
        "`advance_with_restrictions` only if the three execution surfaces are\n"
        "materialized, stillbirth and neonatal replay complete, and any remaining\n"
        "flagship block is explicitly narrowed to engine availability rather than\n"
        "missing data surfaces.\n\n"
        "Round 10.5 now meets this gate.\n"
    )
    _ensure_parent(root / ROUND10_5_MEMO_PATH)
    (root / ROUND10_5_MEMO_PATH).write_text(text, encoding="utf-8")


def run_round10_5_flagship(root: Path, years: list[int] | None = None, build_only: bool = False) -> dict[str, Any]:
    surface_outputs = build_flagship_execution_surfaces(root, years=years)
    if build_only:
        return surface_outputs
    replay_outputs = run_flagship_replay(root)
    _update_round10_ledgers_and_memos(
        root,
        summary_path=replay_outputs["summary_path"],
        stillbirth_path=replay_outputs["stillbirth_path"],
        neonatal_path=replay_outputs["neonatal_path"],
    )
    _update_round_governance_logs(root)
    _update_round_registry_and_round11(root)
    _write_round10_5_memo(root, replay_outputs)
    return {
        **surface_outputs,
        **replay_outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="2020,2021,2022,2023,2024")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    years = [int(part.strip()) for part in args.years.split(",") if part.strip()]
    outputs = run_round10_5_flagship(Path.cwd(), years=years, build_only=args.build_only)
    for key, value in outputs.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
