from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_track_b_config, load_track_b_window_spec
from .registry import ensure_track_b_directory_structure, track_b_event_level_root


CLIMATE_REQUIRED_COLUMNS = [
    "COD_MUNICIPIO",
    "DATE",
    "TMAX_C",
    "RH_MEAN_PCT",
    "HEAT_INDEX_C",
    "HEATWAVE_LEN_P95",
    "FONTE_CLIMA_FINAL",
    "IMPUTADO_POR_FALLBACK",
    "CLIMATE_UPSTREAM_QUARANTINE_FLAG",
]


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _normalize_municipality_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6).str.slice(0, 6)


def _read_parquet(root: Path, relative_path: str) -> pd.DataFrame:
    path = root / relative_path
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _load_climate(root: Path, years: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in years:
        frame = _read_parquet(root, f"data/silver/climate/YEAR={year}/climate_canonical_daily_{year}.parquet")
        frames.append(frame.loc[:, CLIMATE_REQUIRED_COLUMNS].copy())
    climate = pd.concat(frames, ignore_index=True)
    climate["municipality_code"] = _normalize_municipality_code(climate["COD_MUNICIPIO"])
    climate["date"] = pd.to_datetime(climate["DATE"])
    climate = climate.rename(
        columns={
            "TMAX_C": "tmax_c",
            "RH_MEAN_PCT": "rh_mean_pct",
            "HEAT_INDEX_C": "heat_index_c",
            "HEATWAVE_LEN_P95": "heatwave_len_p95",
            "FONTE_CLIMA_FINAL": "climate_source_final",
            "IMPUTADO_POR_FALLBACK": "fallback_imputation_flag",
            "CLIMATE_UPSTREAM_QUARANTINE_FLAG": "upstream_quarantine_flag",
        }
    )
    climate["heatwave_day_flag"] = climate["heatwave_len_p95"].fillna(0).gt(0).astype(np.int8)
    climate["era5_source_day_flag"] = climate["climate_source_final"].astype(str).str.startswith("ERA5").astype(np.int8)
    climate["inmet_source_day_flag"] = climate["climate_source_final"].astype(str).str.startswith("INMET").astype(np.int8)
    climate["fallback_imputation_flag"] = pd.to_numeric(climate["fallback_imputation_flag"], errors="coerce").fillna(0).astype(np.int8)
    climate["upstream_quarantine_flag"] = pd.to_numeric(climate["upstream_quarantine_flag"], errors="coerce").fillna(0).astype(np.int8)
    return climate.loc[
        :,
        [
            "municipality_code",
            "date",
            "tmax_c",
            "rh_mean_pct",
            "heat_index_c",
            "heatwave_day_flag",
            "era5_source_day_flag",
            "inmet_source_day_flag",
            "fallback_imputation_flag",
            "upstream_quarantine_flag",
        ],
    ]


def _build_fixed_windows(climate: pd.DataFrame) -> pd.DataFrame:
    climate = climate.sort_values(["municipality_code", "date"]).reset_index(drop=True)
    grouped = climate.groupby("municipality_code", group_keys=False)
    window_specs = {
        "days_0_6_before_event": 7,
        "days_0_13_before_event": 14,
        "days_0_27_before_event": 28,
    }
    for prefix, window in window_specs.items():
        climate[f"{prefix}_mean_tmax_c"] = grouped["tmax_c"].rolling(window, min_periods=window).mean().reset_index(level=0, drop=True)
        climate[f"{prefix}_mean_rh_mean_pct"] = grouped["rh_mean_pct"].rolling(window, min_periods=window).mean().reset_index(level=0, drop=True)
        climate[f"{prefix}_mean_heat_index_c"] = grouped["heat_index_c"].rolling(window, min_periods=window).mean().reset_index(level=0, drop=True)
        climate[f"{prefix}_heatwave_days"] = grouped["heatwave_day_flag"].rolling(window, min_periods=window).sum().reset_index(level=0, drop=True)
        climate[f"{prefix}_era5_day_share"] = grouped["era5_source_day_flag"].rolling(window, min_periods=window).mean().reset_index(level=0, drop=True)
        climate[f"{prefix}_inmet_day_share"] = grouped["inmet_source_day_flag"].rolling(window, min_periods=window).mean().reset_index(level=0, drop=True)
        climate[f"{prefix}_fallback_day_share"] = grouped["fallback_imputation_flag"].rolling(window, min_periods=window).mean().reset_index(level=0, drop=True)
        climate[f"{prefix}_quarantine_day_share"] = grouped["upstream_quarantine_flag"].rolling(window, min_periods=window).mean().reset_index(level=0, drop=True)
        climate[f"{prefix}_full_window_observed_flag"] = climate[f"{prefix}_mean_tmax_c"].notna().astype(np.int8)
    return climate


def _build_interval_lookup(climate: pd.DataFrame) -> pd.DataFrame:
    climate = climate.sort_values(["municipality_code", "date"]).reset_index(drop=True)
    for column in [
        "tmax_c",
        "rh_mean_pct",
        "heat_index_c",
        "heatwave_day_flag",
        "era5_source_day_flag",
        "inmet_source_day_flag",
        "fallback_imputation_flag",
        "upstream_quarantine_flag",
    ]:
        climate[f"cum_{column}"] = climate.groupby("municipality_code")[column].cumsum()
    interval = climate.loc[
        :,
        [
            "municipality_code",
            "date",
            "cum_tmax_c",
            "cum_rh_mean_pct",
            "cum_heat_index_c",
            "cum_heatwave_day_flag",
            "cum_era5_source_day_flag",
            "cum_inmet_source_day_flag",
            "cum_fallback_imputation_flag",
            "cum_upstream_quarantine_flag",
        ],
    ].copy()
    return interval


def _build_climate_coverage(climate: pd.DataFrame) -> pd.DataFrame:
    coverage = (
        climate.groupby("municipality_code", as_index=False)
        .agg(
            coverage_start_date=("date", "min"),
            coverage_end_date=("date", "max"),
        )
    )
    coverage["coverage_start_date"] = pd.to_datetime(coverage["coverage_start_date"]).dt.normalize()
    coverage["coverage_end_date"] = pd.to_datetime(coverage["coverage_end_date"]).dt.normalize()
    return coverage


def _load_live_births(root: Path, years: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in years:
        frame = _read_parquet(root, f"data/silver/sinasc/silver_sinasc_event/YEAR={year}/silver_sinasc_event_{year}.parquet")
        working = frame.loc[
            :,
            [
                "SINASC_EVENT_ID",
                "MUNICIPALITY_CODE",
                "DATE",
                "YEAR",
                "MONTH",
                "GA_WEEKS_COMPLETED_FINAL",
                "GA_DAYS_FINAL",
                "GA_VALID_FOR_ANALYSIS_FLAG",
                "GA_SOURCE",
                "GA_DUM_EXACT_FLAG",
                "PRETERM_LT37_FLAG",
                "PRETERM_LT32_FLAG",
                "PRETERM_DAYS_SHORT_OF_37W_FINAL",
            ],
        ].copy()
        working["record_type"] = "live_birth"
        working["record_id"] = working["SINASC_EVENT_ID"].astype(str)
        working["municipality_code"] = _normalize_municipality_code(working["MUNICIPALITY_CODE"])
        working["event_date"] = pd.to_datetime(working["DATE"])
        working["delivery_date"] = working["event_date"]
        working["calendar_year"] = working["YEAR"].astype(int)
        ga_days = pd.to_numeric(working["GA_DAYS_FINAL"], errors="coerce")
        ga_days = ga_days.where(ga_days.notna(), pd.to_numeric(working["GA_WEEKS_COMPLETED_FINAL"], errors="coerce") * 7.0)
        working["gestational_age_days_final"] = ga_days
        working["gestational_age_weeks_final"] = pd.to_numeric(working["GA_WEEKS_COMPLETED_FINAL"], errors="coerce")
        working["gestational_age_valid_for_analysis_flag"] = pd.to_numeric(
            working["GA_VALID_FOR_ANALYSIS_FLAG"], errors="coerce"
        ).fillna(0).astype(np.int8)
        working["gestational_age_source"] = working["GA_SOURCE"].fillna("missing").astype(str)
        working["gestational_age_missing_flag"] = working["gestational_age_weeks_final"].isna().astype(np.int8)
        working["ga_exact_measurement_flag"] = pd.to_numeric(working["GA_DUM_EXACT_FLAG"], errors="coerce").fillna(0).astype(np.int8)
        working["preterm_birth_flag"] = pd.to_numeric(working["PRETERM_LT37_FLAG"], errors="coerce").fillna(0).astype(np.int8)
        working["very_preterm_birth_flag"] = pd.to_numeric(working["PRETERM_LT32_FLAG"], errors="coerce").fillna(0).astype(np.int8)
        working["gestational_days_lost_below_37_weeks"] = pd.to_numeric(
            working["PRETERM_DAYS_SHORT_OF_37W_FINAL"], errors="coerce"
        ).fillna(0.0)
        working["stillbirth_primary_gte28_flag"] = np.int8(0)
        working["stillbirth_sensitivity_gte22_flag"] = np.int8(0)
        working["conception_date_estimated"] = working["event_date"] - pd.to_timedelta(
            working["gestational_age_days_final"].fillna(0.0), unit="D"
        )
        working["month_of_conception"] = working["conception_date_estimated"].dt.month.astype("Int64")
        frames.append(
            working.loc[
                :,
                [
                    "record_type",
                    "record_id",
                    "municipality_code",
                    "event_date",
                    "delivery_date",
                    "conception_date_estimated",
                    "calendar_year",
                    "month_of_conception",
                    "gestational_age_days_final",
                    "gestational_age_weeks_final",
                    "gestational_age_valid_for_analysis_flag",
                    "gestational_age_source",
                    "gestational_age_missing_flag",
                    "ga_exact_measurement_flag",
                    "preterm_birth_flag",
                    "very_preterm_birth_flag",
                    "gestational_days_lost_below_37_weeks",
                    "stillbirth_primary_gte28_flag",
                    "stillbirth_sensitivity_gte22_flag",
                ],
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _load_stillbirths(root: Path, years: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in years:
        frame = _read_parquet(root, f"data/silver/stillbirth/silver_stillbirth_individual/YEAR={year}/silver_stillbirth_individual_{year}.parquet")
        working = frame.loc[
            :,
            [
                "STILLBIRTH_RECORD_ID",
                "MUNICIPALITY_CODE",
                "EVENT_DATE",
                "YEAR",
                "GA_WEEKS_COMPLETED_FINAL",
                "GA_VALID_FOR_ANALYSIS_FLAG",
                "GA_SOURCE",
                "STILLBIRTH_GTE28_PRIMARY_FLAG",
                "STILLBIRTH_GTE22_SENSITIVITY_FLAG",
            ],
        ].copy()
        working["record_type"] = "stillbirth"
        working["record_id"] = working["STILLBIRTH_RECORD_ID"].astype(str)
        working["municipality_code"] = _normalize_municipality_code(working["MUNICIPALITY_CODE"])
        working["event_date"] = pd.to_datetime(working["EVENT_DATE"])
        working["delivery_date"] = working["event_date"]
        working["calendar_year"] = pd.to_numeric(working["YEAR"], errors="coerce").fillna(year).astype(int)
        working["month_of_conception"] = pd.NA
        working["gestational_age_weeks_final"] = pd.to_numeric(working["GA_WEEKS_COMPLETED_FINAL"], errors="coerce")
        working["gestational_age_days_final"] = working["gestational_age_weeks_final"] * 7.0
        working["gestational_age_valid_for_analysis_flag"] = pd.to_numeric(
            working["GA_VALID_FOR_ANALYSIS_FLAG"], errors="coerce"
        ).fillna(0).astype(np.int8)
        working["gestational_age_source"] = working["GA_SOURCE"].fillna("missing").astype(str)
        working["gestational_age_missing_flag"] = working["gestational_age_weeks_final"].isna().astype(np.int8)
        working["ga_exact_measurement_flag"] = np.int8(0)
        working["preterm_birth_flag"] = np.int8(0)
        working["very_preterm_birth_flag"] = np.int8(0)
        working["gestational_days_lost_below_37_weeks"] = 0.0
        working["stillbirth_primary_gte28_flag"] = pd.to_numeric(
            working["STILLBIRTH_GTE28_PRIMARY_FLAG"], errors="coerce"
        ).fillna(0).astype(np.int8)
        working["stillbirth_sensitivity_gte22_flag"] = pd.to_numeric(
            working["STILLBIRTH_GTE22_SENSITIVITY_FLAG"], errors="coerce"
        ).fillna(0).astype(np.int8)
        working["conception_date_estimated"] = working["event_date"] - pd.to_timedelta(
            working["gestational_age_days_final"].fillna(0.0), unit="D"
        )
        frames.append(
            working.loc[
                :,
                [
                    "record_type",
                    "record_id",
                    "municipality_code",
                    "event_date",
                    "delivery_date",
                    "conception_date_estimated",
                    "calendar_year",
                    "month_of_conception",
                    "gestational_age_days_final",
                    "gestational_age_weeks_final",
                    "gestational_age_valid_for_analysis_flag",
                    "gestational_age_source",
                    "gestational_age_missing_flag",
                    "ga_exact_measurement_flag",
                    "preterm_birth_flag",
                    "very_preterm_birth_flag",
                    "gestational_days_lost_below_37_weeks",
                    "stillbirth_primary_gte28_flag",
                    "stillbirth_sensitivity_gte22_flag",
                ],
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _attach_fixed_windows(events: pd.DataFrame, climate_windows: pd.DataFrame) -> pd.DataFrame:
    join_columns = [
        "municipality_code",
        "date",
        "days_0_6_before_event_mean_tmax_c",
        "days_0_6_before_event_mean_rh_mean_pct",
        "days_0_6_before_event_mean_heat_index_c",
        "days_0_6_before_event_heatwave_days",
        "days_0_6_before_event_era5_day_share",
        "days_0_6_before_event_inmet_day_share",
        "days_0_6_before_event_fallback_day_share",
        "days_0_6_before_event_quarantine_day_share",
        "days_0_6_before_event_full_window_observed_flag",
        "days_0_13_before_event_mean_tmax_c",
        "days_0_13_before_event_mean_rh_mean_pct",
        "days_0_13_before_event_mean_heat_index_c",
        "days_0_13_before_event_heatwave_days",
        "days_0_13_before_event_era5_day_share",
        "days_0_13_before_event_inmet_day_share",
        "days_0_13_before_event_fallback_day_share",
        "days_0_13_before_event_quarantine_day_share",
        "days_0_13_before_event_full_window_observed_flag",
        "days_0_27_before_event_mean_tmax_c",
        "days_0_27_before_event_mean_rh_mean_pct",
        "days_0_27_before_event_mean_heat_index_c",
        "days_0_27_before_event_heatwave_days",
        "days_0_27_before_event_era5_day_share",
        "days_0_27_before_event_inmet_day_share",
        "days_0_27_before_event_fallback_day_share",
        "days_0_27_before_event_quarantine_day_share",
        "days_0_27_before_event_full_window_observed_flag",
    ]
    keyed = climate_windows.loc[:, join_columns].copy()
    return events.merge(
        keyed,
        left_on=["municipality_code", "event_date"],
        right_on=["municipality_code", "date"],
        how="left",
    ).drop(columns=["date"])


def _attach_event_interval_endpoints(events: pd.DataFrame, climate_interval: pd.DataFrame) -> pd.DataFrame:
    event_lookup = climate_interval.rename(
        columns={
            "date": "event_date",
            "cum_tmax_c": "event_cum_tmax_c",
            "cum_rh_mean_pct": "event_cum_rh_mean_pct",
            "cum_heat_index_c": "event_cum_heat_index_c",
            "cum_heatwave_day_flag": "event_cum_heatwave_day_flag",
            "cum_era5_source_day_flag": "event_cum_era5_source_day_flag",
            "cum_inmet_source_day_flag": "event_cum_inmet_source_day_flag",
            "cum_fallback_imputation_flag": "event_cum_fallback_imputation_flag",
            "cum_upstream_quarantine_flag": "event_cum_upstream_quarantine_flag",
        }
    )
    events = events.merge(event_lookup, on=["municipality_code", "event_date"], how="left")
    return events


def _compute_dynamic_window(
    events: pd.DataFrame,
    climate_interval: pd.DataFrame,
    *,
    prefix: str,
    start_date_column: str,
    minimum_days: int,
    global_climate_start_date: pd.Timestamp,
) -> pd.DataFrame:
    start = pd.to_datetime(events[start_date_column]).dt.normalize()
    end = pd.to_datetime(events["event_date"]).dt.normalize()
    coverage_start = pd.to_datetime(events["coverage_start_date"]).dt.normalize()
    coverage_end = pd.to_datetime(events["coverage_end_date"]).dt.normalize()
    valid_window = start.notna() & end.notna() & start.le(end)
    duration_days = (end - start).dt.days + 1
    effective_coverage_start = coverage_start.where(coverage_start.notna(), global_climate_start_date)
    full_window = (
        valid_window
        & duration_days.ge(minimum_days)
        & effective_coverage_start.notna()
        & coverage_end.notna()
        & start.ge(effective_coverage_start)
        & end.le(coverage_end)
    )
    prior_lookup = climate_interval.rename(
        columns={
            "date": "interval_start_minus_one_date",
            "cum_tmax_c": "prior_cum_tmax_c",
            "cum_rh_mean_pct": "prior_cum_rh_mean_pct",
            "cum_heat_index_c": "prior_cum_heat_index_c",
            "cum_heatwave_day_flag": "prior_cum_heatwave_day_flag",
            "cum_era5_source_day_flag": "prior_cum_era5_source_day_flag",
            "cum_inmet_source_day_flag": "prior_cum_inmet_source_day_flag",
            "cum_fallback_imputation_flag": "prior_cum_fallback_imputation_flag",
            "cum_upstream_quarantine_flag": "prior_cum_upstream_quarantine_flag",
        }
    )
    working = events.copy()
    working["interval_start_minus_one_date"] = start - pd.Timedelta(days=1)
    working = working.merge(prior_lookup, on=["municipality_code", "interval_start_minus_one_date"], how="left")
    event_tmax = pd.to_numeric(working["event_cum_tmax_c"], errors="coerce")
    event_rh = pd.to_numeric(working["event_cum_rh_mean_pct"], errors="coerce")
    event_hi = pd.to_numeric(working["event_cum_heat_index_c"], errors="coerce")
    event_hw = pd.to_numeric(working["event_cum_heatwave_day_flag"], errors="coerce")
    event_era5 = pd.to_numeric(working["event_cum_era5_source_day_flag"], errors="coerce")
    event_inmet = pd.to_numeric(working["event_cum_inmet_source_day_flag"], errors="coerce")
    event_fallback = pd.to_numeric(working["event_cum_fallback_imputation_flag"], errors="coerce")
    event_quarantine = pd.to_numeric(working["event_cum_upstream_quarantine_flag"], errors="coerce")
    prior_tmax = pd.to_numeric(working["prior_cum_tmax_c"], errors="coerce").fillna(0.0)
    prior_rh = pd.to_numeric(working["prior_cum_rh_mean_pct"], errors="coerce").fillna(0.0)
    prior_hi = pd.to_numeric(working["prior_cum_heat_index_c"], errors="coerce").fillna(0.0)
    prior_hw = pd.to_numeric(working["prior_cum_heatwave_day_flag"], errors="coerce").fillna(0.0)
    prior_era5 = pd.to_numeric(working["prior_cum_era5_source_day_flag"], errors="coerce").fillna(0.0)
    prior_inmet = pd.to_numeric(working["prior_cum_inmet_source_day_flag"], errors="coerce").fillna(0.0)
    prior_fallback = pd.to_numeric(working["prior_cum_fallback_imputation_flag"], errors="coerce").fillna(0.0)
    prior_quarantine = pd.to_numeric(working["prior_cum_upstream_quarantine_flag"], errors="coerce").fillna(0.0)
    denominator = duration_days.where(full_window)
    working[f"{prefix}_mean_tmax_c"] = ((event_tmax - prior_tmax) / denominator).where(full_window)
    working[f"{prefix}_mean_rh_mean_pct"] = ((event_rh - prior_rh) / denominator).where(full_window)
    working[f"{prefix}_mean_heat_index_c"] = ((event_hi - prior_hi) / denominator).where(full_window)
    working[f"{prefix}_heatwave_days"] = (event_hw - prior_hw).where(full_window)
    working[f"{prefix}_era5_day_share"] = ((event_era5 - prior_era5) / denominator).where(full_window)
    working[f"{prefix}_inmet_day_share"] = ((event_inmet - prior_inmet) / denominator).where(full_window)
    working[f"{prefix}_fallback_day_share"] = ((event_fallback - prior_fallback) / denominator).where(full_window)
    working[f"{prefix}_quarantine_day_share"] = ((event_quarantine - prior_quarantine) / denominator).where(full_window)
    working[f"{prefix}_full_window_observed_flag"] = full_window.astype(np.int8)
    return working.drop(columns=["interval_start_minus_one_date", "prior_cum_tmax_c", "prior_cum_rh_mean_pct", "prior_cum_heat_index_c", "prior_cum_heatwave_day_flag", "prior_cum_era5_source_day_flag", "prior_cum_inmet_source_day_flag", "prior_cum_fallback_imputation_flag", "prior_cum_upstream_quarantine_flag"])


def _build_qc_report(frame: pd.DataFrame, years: list[int]) -> str:
    return (
        "# Track B Window QC Report\n\n"
        f"- Analytical years: `{', '.join(str(year) for year in years)}`\n"
        f"- Event rows: `{len(frame)}`\n"
        f"- Live births: `{int(frame['record_type'].eq('live_birth').sum())}`\n"
        f"- Stillbirths: `{int(frame['record_type'].eq('stillbirth').sum())}`\n"
        f"- Third trimester fully observed rows: `{int(frame['third_trimester_full_window_observed_flag'].sum())}`\n"
        f"- Whole pregnancy fully observed rows: `{int(frame['whole_pregnancy_full_window_observed_flag'].sum())}`\n"
        f"- Primary stillbirth 7-day window fully observed rows: `{int(frame['days_0_6_before_event_full_window_observed_flag'].sum())}`\n"
        f"- Third trimester rows with any fallback day: `{int(frame['third_trimester_fallback_day_share'].fillna(0).gt(0).sum())}`\n"
        f"- Third trimester rows with any upstream quarantine day: `{int(frame['third_trimester_quarantine_day_share'].fillna(0).gt(0).sum())}`\n"
    )


def _build_week_of_gestation_long(frame: pd.DataFrame) -> pd.DataFrame:
    live_births = frame.loc[frame["record_type"].eq("live_birth")].copy()
    return pd.DataFrame(
        {
            "record_id": live_births["record_id"],
            "municipality_code": live_births["municipality_code"],
            "event_date": live_births["event_date"],
            "maximum_gestational_week_supported": 40,
            "sensitivity_placeholder_flag": 1,
            "dataset_role": "week_of_gestation_scaffold_only",
        }
    )


def build_track_b_window_matrix(root: Path, config_path: Path | None = None) -> dict[str, object]:
    ensure_track_b_directory_structure(root)
    track_b_config = load_track_b_config(root, config_path=config_path)
    window_spec = load_track_b_window_spec(root)
    years = [int(year) for year in track_b_config["years"]]
    global_climate_start_date = pd.Timestamp(window_spec["window_spec"]["climate_start_date"]).normalize()

    climate = _load_climate(root, years)
    climate_fixed = _build_fixed_windows(climate)
    climate_interval = _build_interval_lookup(climate)
    climate_coverage = _build_climate_coverage(climate)

    live_births = _load_live_births(root, years)
    stillbirths = _load_stillbirths(root, years)
    events = pd.concat([live_births, stillbirths], ignore_index=True)
    events = events.merge(climate_coverage, on="municipality_code", how="left")
    events = _attach_fixed_windows(events, climate_fixed)
    events = _attach_event_interval_endpoints(events, climate_interval)

    events["third_trimester_start_date"] = (
        pd.to_datetime(events["conception_date_estimated"]).dt.normalize() + pd.to_timedelta(196, unit="D")
    )
    third_trimester_eligible = pd.to_numeric(events["gestational_age_days_final"], errors="coerce").ge(196)
    events.loc[~third_trimester_eligible, "third_trimester_start_date"] = pd.NaT

    events["whole_pregnancy_start_date"] = pd.to_datetime(events["conception_date_estimated"]).dt.normalize()
    events = _compute_dynamic_window(
        events,
        climate_interval,
        prefix="whole_pregnancy",
        start_date_column="whole_pregnancy_start_date",
        minimum_days=int(window_spec["window_spec"]["whole_pregnancy_minimum_days"]),
        global_climate_start_date=global_climate_start_date,
    )
    events = _compute_dynamic_window(
        events,
        climate_interval,
        prefix="third_trimester",
        start_date_column="third_trimester_start_date",
        minimum_days=int(window_spec["window_spec"]["third_trimester_minimum_days"]),
        global_climate_start_date=global_climate_start_date,
    )

    output_root = track_b_event_level_root(root)
    matrix_root = output_root / "perinatal_window_exposure_matrix.parquet"
    week_long_root = output_root / "perinatal_week_of_gestation_exposure_long.parquet"
    matrix_root.mkdir(parents=True, exist_ok=True)
    week_long_root.mkdir(parents=True, exist_ok=True)

    export_columns = [
        "record_type",
        "record_id",
        "municipality_code",
        "event_date",
        "delivery_date",
        "conception_date_estimated",
        "calendar_year",
        "month_of_conception",
        "gestational_age_weeks_final",
        "gestational_age_valid_for_analysis_flag",
        "gestational_age_source",
        "gestational_age_missing_flag",
        "ga_exact_measurement_flag",
        "preterm_birth_flag",
        "very_preterm_birth_flag",
        "gestational_days_lost_below_37_weeks",
        "stillbirth_primary_gte28_flag",
        "stillbirth_sensitivity_gte22_flag",
        "days_0_6_before_event_mean_tmax_c",
        "days_0_6_before_event_mean_rh_mean_pct",
        "days_0_6_before_event_mean_heat_index_c",
        "days_0_6_before_event_heatwave_days",
        "days_0_6_before_event_era5_day_share",
        "days_0_6_before_event_inmet_day_share",
        "days_0_6_before_event_fallback_day_share",
        "days_0_6_before_event_quarantine_day_share",
        "days_0_6_before_event_full_window_observed_flag",
        "days_0_13_before_event_mean_tmax_c",
        "days_0_13_before_event_mean_rh_mean_pct",
        "days_0_13_before_event_mean_heat_index_c",
        "days_0_13_before_event_heatwave_days",
        "days_0_13_before_event_era5_day_share",
        "days_0_13_before_event_inmet_day_share",
        "days_0_13_before_event_fallback_day_share",
        "days_0_13_before_event_quarantine_day_share",
        "days_0_13_before_event_full_window_observed_flag",
        "days_0_27_before_event_mean_tmax_c",
        "days_0_27_before_event_mean_rh_mean_pct",
        "days_0_27_before_event_mean_heat_index_c",
        "days_0_27_before_event_heatwave_days",
        "days_0_27_before_event_era5_day_share",
        "days_0_27_before_event_inmet_day_share",
        "days_0_27_before_event_fallback_day_share",
        "days_0_27_before_event_quarantine_day_share",
        "days_0_27_before_event_full_window_observed_flag",
        "third_trimester_mean_tmax_c",
        "third_trimester_mean_rh_mean_pct",
        "third_trimester_mean_heat_index_c",
        "third_trimester_heatwave_days",
        "third_trimester_era5_day_share",
        "third_trimester_inmet_day_share",
        "third_trimester_fallback_day_share",
        "third_trimester_quarantine_day_share",
        "third_trimester_full_window_observed_flag",
        "whole_pregnancy_mean_tmax_c",
        "whole_pregnancy_mean_rh_mean_pct",
        "whole_pregnancy_mean_heat_index_c",
        "whole_pregnancy_heatwave_days",
        "whole_pregnancy_era5_day_share",
        "whole_pregnancy_inmet_day_share",
        "whole_pregnancy_fallback_day_share",
        "whole_pregnancy_quarantine_day_share",
        "whole_pregnancy_full_window_observed_flag",
    ]
    export_frame = events.loc[:, export_columns].copy()

    for year, year_frame in export_frame.groupby("calendar_year", sort=True):
        year_root = matrix_root / f"YEAR={int(year)}"
        year_root.mkdir(parents=True, exist_ok=True)
        year_frame.to_parquet(year_root / f"perinatal_window_exposure_matrix_{int(year)}.parquet", index=False)

    week_long = _build_week_of_gestation_long(export_frame)
    for year, year_frame in week_long.assign(calendar_year=pd.to_datetime(week_long["event_date"]).dt.year).groupby("calendar_year", sort=True):
        year_root = week_long_root / f"YEAR={int(year)}"
        year_root.mkdir(parents=True, exist_ok=True)
        year_frame.drop(columns=["calendar_year"]).to_parquet(
            year_root / f"perinatal_week_of_gestation_exposure_long_{int(year)}.parquet",
            index=False,
        )

    qc_report_path = output_root / "window_qc_report.md"
    qc_report_path.write_text(_build_qc_report(export_frame, years), encoding="utf-8")

    return {
        "status": "PASS",
        "stage": "track_b_stage_b1_window_matrix",
        "analysis_id": track_b_config["analysis_id"],
        "event_row_count": int(len(export_frame)),
        "live_birth_row_count": int(export_frame["record_type"].eq("live_birth").sum()),
        "stillbirth_row_count": int(export_frame["record_type"].eq("stillbirth").sum()),
        "outputs": {
            "perinatal_window_exposure_matrix": _relative(root, matrix_root),
            "perinatal_week_of_gestation_exposure_long": _relative(root, week_long_root),
            "window_qc_report": _relative(root, qc_report_path),
        },
    }
