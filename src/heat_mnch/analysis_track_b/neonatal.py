from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from .config import load_track_b_config
from .registry import ensure_track_b_directory_structure, track_b_models_root
from .shared import floor_probability, normalize_municipality_code, pooled_continuous_proxy, relative_to_root, shape_proxy


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


LAG_SPECS = {
    "lag0_tmax_c": 0,
    "lag0_1_mean_tmax_c": 1,
    "lag0_3_mean_tmax_c": 3,
    "lag0_7_mean_tmax_c": 7,
    "lag0_14_mean_tmax_c": 14,
}


def _load_mortality(root: Path, years: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    columns = [
        "MUNICIPALITY_CODE",
        "DATE",
        "YEAR",
        "MONTH",
        "DEATHS_NEONATAL",
        "LIVE_BIRTHS_TOTAL_ANNUAL",
        "TMAX_C",
        "RH_MEAN_PCT",
        "FONTE_CLIMA_FINAL",
        "IMPUTADO_POR_FALLBACK",
        "OFFSET_VALID_FLAG",
        "DENOMINATOR_ZERO_FLAG",
        "DENOMINATOR_MISSING_FLAG",
        "RATE_NOT_COMPUTABLE_FLAG",
        "QC_STATUS",
    ]
    for year in years:
        path = root / f"data/gold/gold_mnch_mortality/YEAR={year}/gold_mnch_mortality_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Track B neonatal models require mortality gold at {path}.")
        frames.append(pd.read_parquet(path, columns=columns))
    frame = pd.concat(frames, ignore_index=True)
    frame["municipality_code"] = normalize_municipality_code(frame["MUNICIPALITY_CODE"])
    frame["observation_date"] = pd.to_datetime(frame["DATE"], errors="coerce")
    frame["calendar_year"] = pd.to_numeric(frame["YEAR"], errors="coerce").astype(int)
    frame["calendar_month"] = pd.to_numeric(frame["MONTH"], errors="coerce").astype(int)
    frame["day_of_week"] = frame["observation_date"].dt.dayofweek.fillna(-1).astype(int)
    frame["source_sensitive_flag"] = (
        pd.to_numeric(frame["IMPUTADO_POR_FALLBACK"], errors="coerce").fillna(0).eq(1)
    ).astype(np.int8)
    return frame


def _add_lags(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.sort_values(["municipality_code", "observation_date"]).copy()
    grouped = working.groupby("municipality_code", group_keys=False)
    working["lag0_tmax_c"] = pd.to_numeric(working["TMAX_C"], errors="coerce")
    for days in [2, 4, 8, 15]:
        column = {2: "lag0_1_mean_tmax_c", 4: "lag0_3_mean_tmax_c", 8: "lag0_7_mean_tmax_c", 15: "lag0_14_mean_tmax_c"}[days]
        working[column] = grouped["lag0_tmax_c"].rolling(days, min_periods=days).mean().reset_index(level=0, drop=True)
    return working


def _make_result_row(
    *,
    lag_or_window: str,
    exposure_label: str,
    result: dict[str, Any],
    frame: pd.DataFrame,
    model_label: str,
) -> dict[str, Any]:
    return {
        "outcome_variable": "neonatal_death_count",
        "outcome_label": "Neonatal mortality",
        "exposure_variable": lag_or_window,
        "exposure_label": exposure_label,
        "model_label": model_label,
        "estimate_scale": "rate_ratio_per_one_degree_celsius_increase",
        "primary_model_flag": "yes" if lag_or_window == "lag0_1_mean_tmax_c" and model_label == "poisson_fe_proxy" else "no",
        "lag_or_window": lag_or_window,
        "estimate": result["estimate"],
        "lower_ci": result["lower_ci"],
        "upper_ci": result["upper_ci"],
        "p_value": result["p_value"],
        "p_value_floor_applied_flag": result["p_value_floor_applied_flag"],
        "valid_strata_count": result["valid_strata_count"],
        "observation_count": result["observation_count"],
        "death_count": result["outcome_sum"],
        "annual_live_birth_denominator_sum": result["denominator_sum"],
        "overdispersion_ratio": result["overdispersion_ratio"],
        "internal_slope": result["internal_slope"],
        "internal_slope_se": result["internal_slope_se"],
        "source_sensitive_days": float(frame["source_sensitive_flag"].sum()),
    }


def _plot_neonatal_models(frame: pd.DataFrame, output_path: Path) -> None:
    figure = plt.figure(figsize=(10.5, 4.8))
    axis = figure.add_subplot(1, 1, 1)
    if frame.empty:
        axis.text(0.5, 0.5, "No neonatal model rows available.", ha="center", va="center")
    else:
        subset = frame.sort_values("lag_or_window")
        y_positions = np.arange(subset.shape[0])
        axis.errorbar(
            subset["estimate"],
            y_positions,
            xerr=[subset["estimate"] - subset["lower_ci"], subset["upper_ci"] - subset["estimate"]],
            fmt="o",
            color="#264653",
            ecolor="#555555",
            capsize=3,
        )
        axis.axvline(1.0, linestyle="--", color="black", linewidth=1)
        axis.set_yticks(y_positions, subset["exposure_label"].tolist())
        axis.set_xlabel("Rate ratio per one-degree Celsius increase")
    axis.set_title("Track B neonatal acute models")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _methods_note() -> str:
    return (
        "# Track B Neonatal Methods Note\n\n"
        "Stage B4 uses the municipal-day gold mortality layer and climate-enriched daily exposures.\n\n"
        "- The primary estimate is the `lag0_1_mean_tmax_c` association with neonatal deaths.\n"
        "- Municipality-year fixed effects are approximated through weighted within-stratum regression and inverse-variance pooling.\n"
        "- A conditional robustness proxy further conditions on municipality-year-epidemiologic-week strata.\n"
        "- The cumulative lag `0-14` result is represented through a lag-shape proxy rather than a spline-based distributed-lag non-linear model.\n"
    )


def build_track_b_neonatal_models(root: Path, config_path: Path | None = None) -> dict[str, object]:
    ensure_track_b_directory_structure(root)
    config = load_track_b_config(root, config_path=config_path)
    years = [int(year) for year in config["years"]]
    frame = _load_mortality(root, years)
    frame = _add_lags(frame)
    frame = frame.loc[
        frame["QC_STATUS"].astype(str).eq("PASS")
        & pd.to_numeric(frame["OFFSET_VALID_FLAG"], errors="coerce").fillna(0).eq(1)
        & pd.to_numeric(frame["DENOMINATOR_ZERO_FLAG"], errors="coerce").fillna(0).eq(0)
        & pd.to_numeric(frame["DENOMINATOR_MISSING_FLAG"], errors="coerce").fillna(0).eq(0)
        & pd.to_numeric(frame["RATE_NOT_COMPUTABLE_FLAG"], errors="coerce").fillna(0).eq(0)
    ].copy()
    frame["epidemiologic_week"] = frame["observation_date"].dt.isocalendar().week.astype(int)

    main_rows: list[dict[str, Any]] = []
    shape_components: list[dict[str, Any]] = []
    for lag_column, lag_day in LAG_SPECS.items():
        result = pooled_continuous_proxy(
            frame,
            exposure_column=lag_column,
            outcome_column="DEATHS_NEONATAL",
            denominator_column="LIVE_BIRTHS_TOTAL_ANNUAL",
            stratum_columns=["municipality_code", "calendar_year", "calendar_month", "day_of_week"],
            model_kind="count",
            exposure_scale_value=1.0,
        )
        result["lag_day"] = lag_day
        shape_components.append(result)
        main_rows.append(
            _make_result_row(
                lag_or_window=lag_column,
                exposure_label=lag_column.replace("_", " "),
                result=result,
                frame=frame,
                model_label="poisson_fe_proxy",
            )
        )

    conditional = pooled_continuous_proxy(
        frame,
        exposure_column="lag0_1_mean_tmax_c",
        outcome_column="DEATHS_NEONATAL",
        denominator_column="LIVE_BIRTHS_TOTAL_ANNUAL",
        stratum_columns=["municipality_code", "calendar_year", "epidemiologic_week"],
        model_kind="count",
        exposure_scale_value=1.0,
    )
    robustness_rows = [
        _make_result_row(
            lag_or_window="lag0_1_mean_tmax_c",
            exposure_label="lag0 1 mean tmax c",
            result=conditional,
            frame=frame,
            model_label="conditional_poisson_proxy",
        )
    ]
    lag_shape = shape_proxy(shape_components, x_column_name="lag_day", model_kind="count")
    dlnm_rows = [
        {
            "outcome_variable": "neonatal_death_count",
            "outcome_label": "Neonatal mortality",
            "exposure_variable": "lag_shape_0_1_3_7_14_days",
            "exposure_label": "Lag-shape cumulative temperature proxy",
            "model_label": "dlnm_style_lag_shape_proxy",
            "estimate_scale": "rate_ratio_per_additional_lag_day",
            "estimate": lag_shape["estimate"],
            "lower_ci": lag_shape["lower_ci"],
            "upper_ci": lag_shape["upper_ci"],
            "p_value": lag_shape["p_value"],
            "p_value_floor_applied_flag": lag_shape["p_value_floor_applied_flag"],
            "valid_strata_count": lag_shape["valid_strata_count"],
            "internal_slope": lag_shape["internal_slope"],
            "internal_slope_se": lag_shape["internal_slope_se"],
        }
    ]

    main_frame = pd.DataFrame(main_rows)
    dlnm_frame = pd.DataFrame(dlnm_rows)
    robustness_frame = pd.DataFrame(robustness_rows)
    for export_frame in [main_frame, dlnm_frame, robustness_frame]:
        export_frame["p_value"], export_frame["p_value_floor_applied_flag_numeric"] = floor_probability(export_frame["p_value"])

    models_root = track_b_models_root(root)
    plots_root = models_root / "neonatal_effect_plots"
    models_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)
    main_path = models_root / "neonatal_main_models.csv"
    dlnm_path = models_root / "neonatal_dlnm_results.csv"
    robustness_path = models_root / "neonatal_robustness_results.csv"
    methods_path = models_root / "neonatal_methods_note.md"
    figure_path = plots_root / "figure_01_neonatal_models.png"

    main_frame.to_csv(main_path, index=False)
    dlnm_frame.to_csv(dlnm_path, index=False)
    robustness_frame.to_csv(robustness_path, index=False)
    methods_path.write_text(_methods_note(), encoding="utf-8")
    _plot_neonatal_models(main_frame, figure_path)

    payload = {
        "status": "PASS",
        "stage": "track_b_stage_b4_neonatal_models",
        "analysis_id": config["analysis_id"],
        "main_model_row_count": int(main_frame.shape[0]),
        "dlnm_row_count": int(dlnm_frame.shape[0]),
        "robustness_row_count": int(robustness_frame.shape[0]),
        "outputs": {
            "neonatal_main_models": relative_to_root(root, main_path),
            "neonatal_dlnm_results": relative_to_root(root, dlnm_path),
            "neonatal_robustness_results": relative_to_root(root, robustness_path),
            "neonatal_methods_note": relative_to_root(root, methods_path),
        },
    }
    (models_root / "neonatal_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
