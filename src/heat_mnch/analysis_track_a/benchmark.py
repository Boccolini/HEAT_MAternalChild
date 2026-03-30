from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from .config import load_track_a_config
from .modeling import (
    ACUTE_OUTCOMES,
    ACUTE_TEMPERATURE_LAG_SUFFIXES,
    BASE_EXPOSURES,
    DELIVERY_WINDOW_SUFFIXES,
    MATERNAL_OUTCOMES,
    PERINATAL_OUTCOMES,
    WINDOW_SUFFIX_LABELS,
)
from .registry import track_a_root
from .screening_aggregations import PROBABILITY_FLOOR


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ALL_OUTCOME_SPECS = ACUTE_OUTCOMES + PERINATAL_OUTCOMES + MATERNAL_OUTCOMES
OUTCOME_SPEC_LOOKUP = {item["outcome_variable"]: item for item in ALL_OUTCOME_SPECS}
ACUTE_LAG_DAY_LOOKUP = {
    "lag0": 0,
    "lag0_1_mean": 1,
    "lag0_3_mean": 3,
    "lag0_7_mean": 7,
}
WINDOW_DAY_LOOKUP = {
    "days_0_6_before_event_mean": 6,
    "days_0_13_before_event_mean": 13,
    "days_0_27_before_event_mean": 27,
}
COUNT_MODEL_SCALE = "rate_ratio_per_interquartile_range_increase"
COUNT_SHAPE_SCALE = "rate_ratio_per_additional_day"
SEVERITY_MODEL_SCALE = "mean_difference_per_interquartile_range_increase"
SEVERITY_SHAPE_SCALE = "mean_difference_per_additional_day"


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _benchmark_root(root: Path) -> Path:
    return track_a_root(root) / "models"


def _screening_root(root: Path) -> Path:
    return track_a_root(root) / "screening"


def _mart_partition_path(root: Path, mart_name: str, year: int) -> Path:
    return track_a_root(root) / "analysis_marts" / mart_name / f"YEAR={year}" / f"{mart_name}_{year}.parquet"


def _load_triage(root: Path) -> pd.DataFrame:
    path = _screening_root(root) / "triage_signal_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(f"Stage 6 requires the Stage 4 triage matrix at {path}. Run track-a-screening first.")
    return pd.read_csv(path)


def _load_partition(root: Path, mart_name: str, year: int, columns: list[str]) -> pd.DataFrame:
    path = _mart_partition_path(root, mart_name, year)
    if not path.exists():
        raise FileNotFoundError(f"Stage 6 requires the Stage 2 mart partition at {path}. Run track-a-build-marts first.")
    return pd.read_parquet(path, columns=columns)


def _normal_two_sided_p_from_z(z_value: float) -> tuple[float, str]:
    if not math.isfinite(z_value):
        return float("nan"), "no"
    absolute_z = abs(z_value)
    probability = float(math.erfc(absolute_z / math.sqrt(2.0)))
    if probability < PROBABILITY_FLOOR:
        return PROBABILITY_FLOOR, "yes"
    return probability, "no"


def _split_acute_exposure_column(exposure_column: str) -> tuple[str, str]:
    for base_exposure in BASE_EXPOSURES:
        if base_exposure == exposure_column:
            return base_exposure, "lag0"
        for suffix in ACUTE_TEMPERATURE_LAG_SUFFIXES:
            candidate = f"{base_exposure}_{suffix}"
            if candidate == exposure_column:
                return base_exposure, suffix
    raise KeyError(f"Unable to resolve acute exposure column {exposure_column!r}")


def _split_window_exposure_column(exposure_column: str) -> tuple[str, str]:
    for base_exposure in BASE_EXPOSURES:
        for suffix in DELIVERY_WINDOW_SUFFIXES:
            candidate = f"{base_exposure}_{suffix}"
            if candidate == exposure_column:
                return base_exposure, suffix
    raise KeyError(f"Unable to resolve window exposure column {exposure_column!r}")


def _candidate_rows(triage: pd.DataFrame) -> list[dict[str, Any]]:
    eligible = triage.loc[triage["benchmark_stage_recommendation"].eq("yes")].copy()
    candidates: list[dict[str, Any]] = []
    for row in eligible.to_dict(orient="records"):
        outcome_variable = str(row["public_concept_name"])
        outcome_spec = OUTCOME_SPEC_LOOKUP.get(outcome_variable)
        if outcome_spec is None:
            continue
        outcome_family = str(outcome_spec["outcome_family"])
        if outcome_family == "acute_daily":
            mart_name = "acute_daily_mart"
            model_kind = "count"
        elif outcome_family == "perinatal_delivery_window_aggregate":
            mart_name = "perinatal_delivery_window_mart"
            model_kind = "severity" if outcome_variable == "gestational_days_lost_below_37_weeks" else "count"
        else:
            mart_name = "maternal_operational_mart"
            model_kind = "count"
        candidates.append(
            {
                "public_concept_name": outcome_variable,
                "public_concept_label": str(row["public_concept_label"]),
                "outcome_variable": outcome_variable,
                "outcome_label": str(outcome_spec["outcome_label"]),
                "outcome_family": outcome_family,
                "denominator_column": str(outcome_spec["denominator_column"]),
                "best_exposure_variable": str(row["best_exposure_variable"]),
                "best_lag_or_window": str(row["best_lag_or_window"]),
                "best_screening_domain": str(row["best_screening_domain"]),
                "stage1_classification": str(row["stage1_classification"]),
                "triage_priority_classification": str(row["triage_priority_classification"]),
                "mart_name": mart_name,
                "model_kind": model_kind,
            }
        )
    return candidates


def _benchmark_valid_rows(frame: pd.DataFrame, denominator_column: str, exposure_column: str) -> pd.DataFrame:
    valid = frame.loc[
        frame["quality_control_status"].astype(str).eq("PASS")
        & frame["offset_valid_flag"].eq(1)
        & frame["denominator_zero_flag"].eq(0)
        & frame["denominator_missing_flag"].eq(0)
        & frame["rate_not_computable_flag"].eq(0)
        & frame[denominator_column].fillna(0).gt(0)
        & frame[exposure_column].notna()
    ].copy()
    return valid


def _exposure_scale_value(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 1.0
    q25 = float(numeric.quantile(0.25))
    q75 = float(numeric.quantile(0.75))
    iqr = q75 - q25
    if math.isfinite(iqr) and iqr > 0:
        return iqr
    standard_deviation = float(numeric.std(ddof=0))
    if math.isfinite(standard_deviation) and standard_deviation > 0:
        return standard_deviation
    return 1.0


def _pooled_continuous_fe_proxy(
    frame: pd.DataFrame,
    exposure_column: str,
    outcome_column: str,
    denominator_column: str,
    stratum_columns: list[str],
    model_kind: str,
) -> dict[str, Any]:
    valid = _benchmark_valid_rows(frame, denominator_column, exposure_column)
    if valid.empty:
        return {
            "estimate": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "p_value": float("nan"),
            "p_value_floor_applied_flag": "no",
            "valid_strata_count": 0,
            "observation_count": 0,
            "outcome_sum": 0.0,
            "denominator_sum": 0.0,
            "overdispersion_ratio": float("nan"),
            "internal_slope": float("nan"),
            "internal_slope_se": float("nan"),
            "exposure_scale_value": 1.0,
        }

    valid["x"] = valid[exposure_column].astype(float)
    valid["w"] = valid[denominator_column].astype(float)
    exposure_scale_value = _exposure_scale_value(valid["x"])
    if model_kind == "count":
        valid["y"] = np.log((valid[outcome_column].astype(float) + 0.5) / valid[denominator_column].astype(float))
    else:
        valid["y"] = valid[outcome_column].astype(float) / valid[denominator_column].astype(float)

    valid["wx"] = valid["w"] * valid["x"]
    valid["wy"] = valid["w"] * valid["y"]
    valid["wxx"] = valid["w"] * valid["x"] * valid["x"]
    valid["wxy"] = valid["w"] * valid["x"] * valid["y"]
    valid["wyy"] = valid["w"] * valid["y"] * valid["y"]

    grouped = valid.groupby(stratum_columns, as_index=False).agg(
        observation_count=("x", "size"),
        sw=("w", "sum"),
        swx=("wx", "sum"),
        swy=("wy", "sum"),
        swxx=("wxx", "sum"),
        swxy=("wxy", "sum"),
        swyy=("wyy", "sum"),
        outcome_sum=(outcome_column, "sum"),
        denominator_sum=(denominator_column, "sum"),
    )
    grouped["det"] = (grouped["sw"] * grouped["swxx"]) - (grouped["swx"] ** 2)
    grouped["degrees_of_freedom"] = grouped["observation_count"] - 2
    grouped = grouped.loc[
        grouped["observation_count"].ge(3)
        & grouped["det"].gt(0)
        & grouped["degrees_of_freedom"].gt(0)
    ].copy()
    if grouped.empty:
        return {
            "estimate": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "p_value": float("nan"),
            "p_value_floor_applied_flag": "no",
            "valid_strata_count": 0,
            "observation_count": int(valid.shape[0]),
            "outcome_sum": float(valid[outcome_column].sum()),
            "denominator_sum": float(valid[denominator_column].sum()),
            "overdispersion_ratio": float("nan"),
            "internal_slope": float("nan"),
            "internal_slope_se": float("nan"),
            "exposure_scale_value": exposure_scale_value,
        }

    grouped["slope"] = ((grouped["sw"] * grouped["swxy"]) - (grouped["swx"] * grouped["swy"])) / grouped["det"]
    grouped["intercept"] = (grouped["swy"] - (grouped["slope"] * grouped["swx"])) / grouped["sw"]
    grouped["weighted_sse"] = grouped["swyy"] - (
        (grouped["intercept"] * grouped["swy"]) + (grouped["slope"] * grouped["swxy"])
    )
    grouped["weighted_sse"] = grouped["weighted_sse"].clip(lower=0.0)
    grouped["sigma_squared"] = grouped["weighted_sse"] / grouped["degrees_of_freedom"]
    grouped["slope_variance"] = (grouped["sigma_squared"] * grouped["sw"]) / grouped["det"]
    grouped = grouped.loc[grouped["slope_variance"].gt(0)].copy()
    if grouped.empty:
        return {
            "estimate": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "p_value": float("nan"),
            "p_value_floor_applied_flag": "no",
            "valid_strata_count": 0,
            "observation_count": int(valid.shape[0]),
            "outcome_sum": float(valid[outcome_column].sum()),
            "denominator_sum": float(valid[denominator_column].sum()),
            "overdispersion_ratio": float("nan"),
            "internal_slope": float("nan"),
            "internal_slope_se": float("nan"),
            "exposure_scale_value": exposure_scale_value,
        }

    grouped["weight"] = 1.0 / grouped["slope_variance"]
    pooled_slope = float(np.average(grouped["slope"], weights=grouped["weight"]))
    pooled_se = float(np.sqrt(1.0 / grouped["weight"].sum()))
    z_value = pooled_slope / pooled_se if pooled_se > 0 else float("nan")
    p_value, p_floor_flag = _normal_two_sided_p_from_z(z_value)
    overdispersion_ratio = float(grouped["weighted_sse"].sum() / grouped["degrees_of_freedom"].sum())

    scaled_slope = pooled_slope * exposure_scale_value
    scaled_se = pooled_se * exposure_scale_value
    if model_kind == "count":
        lower_ci = math.exp(scaled_slope - (1.96 * scaled_se))
        upper_ci = math.exp(scaled_slope + (1.96 * scaled_se))
        estimate = math.exp(scaled_slope)
    else:
        lower_ci = scaled_slope - (1.96 * scaled_se)
        upper_ci = scaled_slope + (1.96 * scaled_se)
        estimate = scaled_slope

    return {
        "estimate": float(estimate),
        "lower_ci": float(lower_ci),
        "upper_ci": float(upper_ci),
        "p_value": float(p_value),
        "p_value_floor_applied_flag": p_floor_flag,
        "valid_strata_count": int(grouped.shape[0]),
        "observation_count": int(grouped["observation_count"].sum()),
        "outcome_sum": float(grouped["outcome_sum"].sum()),
        "denominator_sum": float(grouped["denominator_sum"].sum()),
        "overdispersion_ratio": overdispersion_ratio,
        "internal_slope": pooled_slope,
        "internal_slope_se": pooled_se,
        "exposure_scale_value": exposure_scale_value,
    }


def _dispersion_adjusted_result(base_result: dict[str, Any], model_kind: str) -> dict[str, Any]:
    phi = max(float(base_result["overdispersion_ratio"]), 1.0)
    scale_value = float(base_result.get("exposure_scale_value", 1.0))
    adjusted_se = float(base_result["internal_slope_se"]) * math.sqrt(phi)
    scaled_slope = float(base_result["internal_slope"]) * scale_value
    scaled_adjusted_se = adjusted_se * scale_value
    z_value = scaled_slope / scaled_adjusted_se if scaled_adjusted_se > 0 else float("nan")
    p_value, p_floor_flag = _normal_two_sided_p_from_z(z_value)
    if model_kind == "count":
        estimate = math.exp(scaled_slope)
        lower_ci = math.exp(scaled_slope - (1.96 * scaled_adjusted_se))
        upper_ci = math.exp(scaled_slope + (1.96 * scaled_adjusted_se))
    else:
        estimate = scaled_slope
        lower_ci = scaled_slope - (1.96 * scaled_adjusted_se)
        upper_ci = scaled_slope + (1.96 * scaled_adjusted_se)
    return {
        **base_result,
        "estimate": float(estimate),
        "lower_ci": float(lower_ci),
        "upper_ci": float(upper_ci),
        "p_value": float(p_value),
        "p_value_floor_applied_flag": p_floor_flag,
        "internal_slope_se": adjusted_se,
        "exposure_scale_value": scale_value,
    }


def _shape_proxy(
    components: list[dict[str, Any]],
    x_column_name: str,
    model_kind: str,
) -> dict[str, Any]:
    frame = pd.DataFrame(components)
    valid = frame.loc[
        frame["internal_slope"].notna()
        & frame["internal_slope_se"].notna()
        & frame["internal_slope_se"].astype(float).gt(0)
        & frame[x_column_name].notna()
    ].copy()
    if valid.shape[0] < 2:
        return {
            "estimate": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "p_value": float("nan"),
            "p_value_floor_applied_flag": "no",
            "valid_strata_count": 0,
            "exposure_scale_value": 1.0,
        }

    x = valid[x_column_name].astype(float).to_numpy()
    y = valid["internal_slope"].astype(float).to_numpy()
    w = 1.0 / (valid["internal_slope_se"].astype(float).to_numpy() ** 2)
    sw = w.sum()
    swx = (w * x).sum()
    swy = (w * y).sum()
    swxx = (w * x * x).sum()
    swxy = (w * x * y).sum()
    det = (sw * swxx) - (swx ** 2)
    if det <= 0:
        return {
            "estimate": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "p_value": float("nan"),
            "p_value_floor_applied_flag": "no",
            "valid_strata_count": int(valid.shape[0]),
            "exposure_scale_value": 1.0,
        }
    slope = ((sw * swxy) - (swx * swy)) / det
    slope_se = math.sqrt(sw / det) if det > 0 and sw > 0 else float("nan")
    if not math.isfinite(slope_se) or slope_se <= 0:
        return {
            "estimate": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "p_value": float("nan"),
            "p_value_floor_applied_flag": "no",
            "valid_strata_count": int(valid.shape[0]),
            "exposure_scale_value": 1.0,
        }
    z_value = slope / slope_se
    p_value, p_floor_flag = _normal_two_sided_p_from_z(z_value)
    if model_kind == "count":
        return {
            "estimate": float(math.exp(slope)),
            "lower_ci": float(math.exp(slope - (1.96 * slope_se))),
            "upper_ci": float(math.exp(slope + (1.96 * slope_se))),
            "p_value": float(p_value),
            "p_value_floor_applied_flag": p_floor_flag,
            "valid_strata_count": int(valid.shape[0]),
            "internal_slope": float(slope),
            "internal_slope_se": float(slope_se),
            "exposure_scale_value": 1.0,
        }
    return {
        "estimate": float(slope),
        "lower_ci": float(slope - (1.96 * slope_se)),
        "upper_ci": float(slope + (1.96 * slope_se)),
        "p_value": float(p_value),
        "p_value_floor_applied_flag": p_floor_flag,
        "valid_strata_count": int(valid.shape[0]),
        "internal_slope": float(slope),
        "internal_slope_se": float(slope_se),
        "exposure_scale_value": 1.0,
    }


def _acute_columns_for_candidates(candidates: list[dict[str, Any]]) -> list[str]:
    exposure_columns: set[str] = set()
    outcome_columns: set[str] = set()
    for candidate in candidates:
        if candidate["outcome_family"] != "acute_daily":
            continue
        outcome_columns.add(str(candidate["outcome_variable"]))
        base_exposure, _ = _split_acute_exposure_column(str(candidate["best_exposure_variable"]))
        for suffix in ACUTE_TEMPERATURE_LAG_SUFFIXES:
            exposure_columns.add(f"{base_exposure}_{suffix}")
    return sorted(
        {
            "municipality_code",
            "observation_date",
            "calendar_year",
            "quality_control_status",
            "offset_valid_flag",
            "denominator_zero_flag",
            "denominator_missing_flag",
            "rate_not_computable_flag",
            "annual_live_birth_count",
            "under_five_population_count",
            *exposure_columns,
            *outcome_columns,
        }
    )


def _window_columns_for_candidates(candidates: list[dict[str, Any]], mart_name: str) -> list[str]:
    exposure_columns: set[str] = set()
    outcome_columns: set[str] = set()
    denominator_columns: set[str] = set()
    for candidate in candidates:
        if candidate["mart_name"] != mart_name:
            continue
        outcome_columns.add(str(candidate["outcome_variable"]))
        denominator_columns.add(str(candidate["denominator_column"]))
        base_exposure, _ = _split_window_exposure_column(str(candidate["best_exposure_variable"]))
        for suffix in DELIVERY_WINDOW_SUFFIXES:
            exposure_columns.add(f"{base_exposure}_{suffix}")
    return sorted(
        {
            "municipality_code",
            "observation_date",
            "calendar_year",
            "quality_control_status",
            "offset_valid_flag",
            "denominator_zero_flag",
            "denominator_missing_flag",
            "rate_not_computable_flag",
            *denominator_columns,
            *exposure_columns,
            *outcome_columns,
        }
    )


def _make_result_row(
    candidate: dict[str, Any],
    exposure_variable: str,
    exposure_label: str,
    exposure_family_id: str,
    exposure_family_name: str,
    lag_or_window: str,
    model_label: str,
    model_family: str,
    estimate_scale: str,
    model_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "public_concept_name": candidate["public_concept_name"],
        "public_concept_label": candidate["public_concept_label"],
        "outcome_variable": candidate["outcome_variable"],
        "outcome_label": candidate["outcome_label"],
        "outcome_family": candidate["outcome_family"],
        "stage1_classification": candidate["stage1_classification"],
        "triage_priority_classification": candidate["triage_priority_classification"],
        "benchmark_stage_recommendation": "yes",
        "exposure_variable": exposure_variable,
        "exposure_label": exposure_label,
        "exposure_family_id": exposure_family_id,
        "exposure_family_name": exposure_family_name,
        "lag_or_window": lag_or_window,
        "model_label": model_label,
        "model_family": model_family,
        "estimate_scale": estimate_scale,
        "exposure_scale_value": model_result.get("exposure_scale_value", 1.0),
        "estimate": model_result["estimate"],
        "lower_ci": model_result["lower_ci"],
        "upper_ci": model_result["upper_ci"],
        "p_value": model_result["p_value"],
        "p_value_floor_applied_flag": model_result.get("p_value_floor_applied_flag", "no"),
        "valid_strata_count": model_result["valid_strata_count"],
        "observation_count": model_result.get("observation_count", 0),
        "outcome_sum": model_result.get("outcome_sum", 0.0),
        "denominator_sum": model_result.get("denominator_sum", 0.0),
        "overdispersion_ratio": model_result.get("overdispersion_ratio", float("nan")),
    }


def _make_diagnostic_row(
    candidate: dict[str, Any],
    exposure_variable: str,
    lag_or_window: str,
    model_label: str,
    model_result: dict[str, Any],
    preferred_distribution: str,
) -> dict[str, Any]:
    return {
        "public_concept_name": candidate["public_concept_name"],
        "public_concept_label": candidate["public_concept_label"],
        "outcome_variable": candidate["outcome_variable"],
        "outcome_family": candidate["outcome_family"],
        "exposure_variable": exposure_variable,
        "lag_or_window": lag_or_window,
        "model_label": model_label,
        "valid_strata_count": model_result.get("valid_strata_count", 0),
        "observation_count": model_result.get("observation_count", 0),
        "outcome_sum": model_result.get("outcome_sum", 0.0),
        "denominator_sum": model_result.get("denominator_sum", 0.0),
        "internal_slope": model_result.get("internal_slope", float("nan")),
        "internal_slope_se": model_result.get("internal_slope_se", float("nan")),
        "overdispersion_ratio": model_result.get("overdispersion_ratio", float("nan")),
        "preferred_distribution": preferred_distribution,
    }


def _acute_effect_label(base_exposure: str, suffix: str) -> str:
    return f"{BASE_EXPOSURES[base_exposure]['label']} ({WINDOW_SUFFIX_LABELS[suffix]})"


def _window_effect_label(base_exposure: str, suffix: str) -> str:
    return f"{BASE_EXPOSURES[base_exposure]['label']} ({WINDOW_SUFFIX_LABELS[suffix]})"


def _plot_model_comparison(results: pd.DataFrame, output_path: Path, title: str, outcome_family: str) -> None:
    subset = results.loc[results["outcome_family"].eq(outcome_family)].copy()
    figure = plt.figure(figsize=(12, 7))
    axis = figure.add_subplot(1, 1, 1)
    if subset.empty:
        axis.text(0.5, 0.5, "No benchmark results available.", ha="center", va="center")
    else:
        subset = subset.sort_values(["public_concept_label", "model_label"])
        labels = [f"{row['public_concept_label']}\n{row['model_label']}" for row in subset.to_dict(orient="records")]
        positions = np.arange(subset.shape[0])
        axis.errorbar(
            positions,
            subset["estimate"],
            yerr=[
                subset["estimate"] - subset["lower_ci"],
                subset["upper_ci"] - subset["estimate"],
            ],
            fmt="o",
            color="#0b6e4f",
            ecolor="#6c757d",
            capsize=3,
        )
        if subset["estimate_scale"].astype(str).str.startswith("rate_ratio").all():
            axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
            axis.set_ylabel("Rate ratio")
        else:
            axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
            axis.set_ylabel("Mean difference")
        axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_title(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _plot_diagnostics(diagnostics: pd.DataFrame, output_path: Path) -> None:
    subset = diagnostics.loc[
        diagnostics["overdispersion_ratio"].notna()
        & diagnostics["model_label"].isin(["poisson_fe_proxy", "gaussian_fe_proxy"])
    ].copy()
    figure = plt.figure(figsize=(12, 6))
    axis = figure.add_subplot(1, 1, 1)
    if subset.empty:
        axis.text(0.5, 0.5, "No dispersion diagnostics available.", ha="center", va="center")
    else:
        subset = subset.sort_values("overdispersion_ratio", ascending=False)
        positions = np.arange(subset.shape[0])
        axis.bar(positions, subset["overdispersion_ratio"], color="#c65d00")
        axis.axhline(1.5, color="black", linestyle="--", linewidth=1)
        axis.set_xticks(
            positions,
            [f"{row['public_concept_label']}\n{row['lag_or_window']}" for row in subset.to_dict(orient="records")],
            rotation=25,
            ha="right",
        )
        axis.set_ylabel("Overdispersion ratio")
    axis.set_title("Benchmark diagnostics: overdispersion by concept")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _render_methods_note() -> str:
    return (
        "# Track A Benchmark Methods Note\n\n"
        "Stage 6 operationalizes a benchmark ladder on top of the Stage 4 triage matrix. Only concepts with `benchmark_stage_recommendation=yes` are advanced into the benchmark package.\n\n"
        "## Benchmark Design Adaptation\n\n"
        "- The repository environment does not vend `statsmodels`, so the Track A benchmark ladder is implemented as a fixed-effects proxy framework using weighted within-stratum regression and inverse-variance pooling.\n"
        "- This adaptation is explicit in the model labels. The package does not pretend that these rows are exact maximum-likelihood Poisson or negative-binomial fits.\n"
        "- Acute robustness uses municipality-year-week conditioning as a conditional Poisson proxy. Delivery-window and event-window robustness uses window-shape meta-regression across the 7-day, 14-day, and 28-day observable windows.\n\n"
        "## Outcome Handling\n\n"
        "- Count outcomes are modeled on continuity-corrected log rates with denominator weights.\n"
        "- Main benchmark effect sizes are reported per empirical interquartile-range increase in the selected exposure. If the interquartile range is degenerate, the package falls back to the exposure standard deviation and then to a one-unit scale.\n"
        "- `gestational_days_lost_below_37_weeks` is modeled as a weighted per-birth severity outcome using an additive fixed-effects proxy.\n"
        "- Non-benchmarkable concepts from Stage 4 remain excluded from the benchmark stage.\n\n"
        "## Diagnostics\n\n"
        "- Overdispersion is summarized as a weighted residual-dispersion ratio. Ratios above 1.5 trigger a negative-binomial proxy rung with dispersion-adjusted uncertainty.\n"
        "- All benchmark outputs remain in academic English and are linked to the Stage 4 triage selections.\n"
    )


def build_track_a_benchmark(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    config = load_track_a_config(root, config_path=config_path)
    years = [int(year) for year in config["years"]]
    triage = _load_triage(root)
    candidates = _candidate_rows(triage)

    acute_candidates = [candidate for candidate in candidates if candidate["mart_name"] == "acute_daily_mart"]
    perinatal_candidates = [candidate for candidate in candidates if candidate["mart_name"] == "perinatal_delivery_window_mart"]
    maternal_candidates = [candidate for candidate in candidates if candidate["mart_name"] == "maternal_operational_mart"]

    acute_frames = {
        year: _load_partition(root, "acute_daily_mart", year, _acute_columns_for_candidates(acute_candidates))
        for year in years
    } if acute_candidates else {}
    for frame in acute_frames.values():
        frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce")
        frame["epidemiologic_week"] = frame["observation_date"].dt.isocalendar().week.astype("Int64")

    perinatal_frames = {
        year: _load_partition(root, "perinatal_delivery_window_mart", year, _window_columns_for_candidates(perinatal_candidates, "perinatal_delivery_window_mart"))
        for year in years
    } if perinatal_candidates else {}
    maternal_frames = {
        year: _load_partition(root, "maternal_operational_mart", year, _window_columns_for_candidates(maternal_candidates, "maternal_operational_mart"))
        for year in years
    } if maternal_candidates else {}

    results_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for candidate in acute_candidates:
        base_exposure, selected_suffix = _split_acute_exposure_column(candidate["best_exposure_variable"])
        exposure_meta = BASE_EXPOSURES[base_exposure]
        frame = pd.concat(
            [
                year_frame.loc[
                    :,
                    [
                        "municipality_code",
                        "calendar_year",
                        "epidemiologic_week",
                        "quality_control_status",
                        "offset_valid_flag",
                        "denominator_zero_flag",
                        "denominator_missing_flag",
                        "rate_not_computable_flag",
                        candidate["denominator_column"],
                        candidate["outcome_variable"],
                        *[f"{base_exposure}_{suffix}" for suffix in ACUTE_TEMPERATURE_LAG_SUFFIXES],
                    ],
                ]
                for year_frame in acute_frames.values()
            ],
            ignore_index=True,
        )

        component_results: list[dict[str, Any]] = []
        for suffix in ACUTE_TEMPERATURE_LAG_SUFFIXES:
            exposure_column = f"{base_exposure}_{suffix}"
            component = _pooled_continuous_fe_proxy(
                frame,
                exposure_column=exposure_column,
                outcome_column=candidate["outcome_variable"],
                denominator_column=candidate["denominator_column"],
                stratum_columns=["municipality_code", "calendar_year"],
                model_kind="count",
            )
            component["lag_day"] = ACUTE_LAG_DAY_LOOKUP[suffix]
            component["lag_or_window"] = suffix
            component_results.append(component)
            if suffix == selected_suffix:
                results_rows.append(
                    _make_result_row(
                        candidate,
                        exposure_variable=exposure_column,
                        exposure_label=_acute_effect_label(base_exposure, suffix),
                        exposure_family_id=exposure_meta["family_id"],
                        exposure_family_name=exposure_meta["family_name"],
                        lag_or_window=suffix,
                        model_label="poisson_fe_proxy",
                        model_family="acute_benchmark_ladder",
                        estimate_scale=COUNT_MODEL_SCALE,
                        model_result=component,
                    )
                )
                diagnostic_rows.append(
                    _make_diagnostic_row(
                        candidate,
                        exposure_variable=exposure_column,
                        lag_or_window=suffix,
                        model_label="poisson_fe_proxy",
                        model_result=component,
                        preferred_distribution="negative_binomial" if float(component["overdispersion_ratio"]) > 1.5 else "poisson",
                    )
                )
                if math.isfinite(float(component["overdispersion_ratio"])) and float(component["overdispersion_ratio"]) > 1.5:
                    nb_result = _dispersion_adjusted_result(component, model_kind="count")
                    results_rows.append(
                        _make_result_row(
                            candidate,
                            exposure_variable=exposure_column,
                            exposure_label=_acute_effect_label(base_exposure, suffix),
                            exposure_family_id=exposure_meta["family_id"],
                            exposure_family_name=exposure_meta["family_name"],
                            lag_or_window=suffix,
                            model_label="negative_binomial_fe_proxy",
                            model_family="acute_benchmark_ladder",
                            estimate_scale=COUNT_MODEL_SCALE,
                            model_result=nb_result,
                        )
                    )

        conditional_result = _pooled_continuous_fe_proxy(
            frame,
            exposure_column=f"{base_exposure}_{selected_suffix}",
            outcome_column=candidate["outcome_variable"],
            denominator_column=candidate["denominator_column"],
            stratum_columns=["municipality_code", "calendar_year", "epidemiologic_week"],
            model_kind="count",
        )
        results_rows.append(
            _make_result_row(
                candidate,
                exposure_variable=f"{base_exposure}_{selected_suffix}",
                exposure_label=_acute_effect_label(base_exposure, selected_suffix),
                exposure_family_id=exposure_meta["family_id"],
                exposure_family_name=exposure_meta["family_name"],
                lag_or_window=selected_suffix,
                model_label="conditional_poisson_proxy",
                model_family="acute_benchmark_ladder",
                estimate_scale=COUNT_MODEL_SCALE,
                model_result=conditional_result,
            )
        )
        diagnostic_rows.append(
            _make_diagnostic_row(
                candidate,
                exposure_variable=f"{base_exposure}_{selected_suffix}",
                lag_or_window=selected_suffix,
                model_label="conditional_poisson_proxy",
                model_result=conditional_result,
                preferred_distribution="conditional_proxy",
            )
        )

        lag_shape_result = _shape_proxy(component_results, x_column_name="lag_day", model_kind="count")
        results_rows.append(
            _make_result_row(
                candidate,
                exposure_variable=base_exposure,
                exposure_label=f"{BASE_EXPOSURES[base_exposure]['label']} lag-shape proxy",
                exposure_family_id=exposure_meta["family_id"],
                exposure_family_name=exposure_meta["family_name"],
                lag_or_window="lag_shape_0_1_3_7_days",
                model_label="lag_shape_linear_proxy",
                model_family="acute_benchmark_ladder",
                estimate_scale=COUNT_SHAPE_SCALE,
                model_result=lag_shape_result,
            )
        )

    for candidate_group, frames_by_year, model_family in (
        (perinatal_candidates, perinatal_frames, "delivery_window_benchmark_ladder"),
        (maternal_candidates, maternal_frames, "delivery_window_benchmark_ladder"),
    ):
        for candidate in candidate_group:
            base_exposure, selected_suffix = _split_window_exposure_column(candidate["best_exposure_variable"])
            exposure_meta = BASE_EXPOSURES[base_exposure]
            frame = pd.concat(
                [
                    year_frame.loc[
                        :,
                        [
                            "municipality_code",
                            "calendar_year",
                            "quality_control_status",
                            "offset_valid_flag",
                            "denominator_zero_flag",
                            "denominator_missing_flag",
                            "rate_not_computable_flag",
                            candidate["denominator_column"],
                            candidate["outcome_variable"],
                            *[f"{base_exposure}_{suffix}" for suffix in DELIVERY_WINDOW_SUFFIXES],
                        ],
                    ]
                    for year_frame in frames_by_year.values()
                ],
                ignore_index=True,
            )

            window_components: list[dict[str, Any]] = []
            for suffix in DELIVERY_WINDOW_SUFFIXES:
                exposure_column = f"{base_exposure}_{suffix}"
                component = _pooled_continuous_fe_proxy(
                    frame,
                    exposure_column=exposure_column,
                    outcome_column=candidate["outcome_variable"],
                    denominator_column=candidate["denominator_column"],
                    stratum_columns=["municipality_code", "calendar_year"],
                    model_kind=candidate["model_kind"],
                )
                component["window_day"] = WINDOW_DAY_LOOKUP[suffix]
                component["lag_or_window"] = suffix
                window_components.append(component)
                if suffix == selected_suffix:
                    primary_label = "gaussian_fe_proxy" if candidate["model_kind"] == "severity" else "poisson_fe_proxy"
                    primary_scale = SEVERITY_MODEL_SCALE if candidate["model_kind"] == "severity" else COUNT_MODEL_SCALE
                    results_rows.append(
                        _make_result_row(
                            candidate,
                            exposure_variable=exposure_column,
                            exposure_label=_window_effect_label(base_exposure, suffix),
                            exposure_family_id=exposure_meta["family_id"],
                            exposure_family_name=exposure_meta["family_name"],
                            lag_or_window=suffix,
                            model_label=primary_label,
                            model_family=model_family,
                            estimate_scale=primary_scale,
                            model_result=component,
                        )
                    )
                    diagnostic_rows.append(
                        _make_diagnostic_row(
                            candidate,
                            exposure_variable=exposure_column,
                            lag_or_window=suffix,
                            model_label=primary_label,
                            model_result=component,
                            preferred_distribution="negative_binomial" if candidate["model_kind"] == "count" and float(component["overdispersion_ratio"]) > 1.5 else ("gaussian" if candidate["model_kind"] == "severity" else "poisson"),
                        )
                    )
                    if candidate["model_kind"] == "count" and math.isfinite(float(component["overdispersion_ratio"])) and float(component["overdispersion_ratio"]) > 1.5:
                        nb_result = _dispersion_adjusted_result(component, model_kind="count")
                        results_rows.append(
                            _make_result_row(
                                candidate,
                                exposure_variable=exposure_column,
                                exposure_label=_window_effect_label(base_exposure, suffix),
                                exposure_family_id=exposure_meta["family_id"],
                                exposure_family_name=exposure_meta["family_name"],
                                lag_or_window=suffix,
                                model_label="negative_binomial_fe_proxy",
                                model_family=model_family,
                                estimate_scale=COUNT_MODEL_SCALE,
                                model_result=nb_result,
                            )
                        )

            shape_result = _shape_proxy(
                window_components,
                x_column_name="window_day",
                model_kind="count" if candidate["model_kind"] == "count" else "severity",
            )
            results_rows.append(
                _make_result_row(
                    candidate,
                    exposure_variable=base_exposure,
                    exposure_label=f"{BASE_EXPOSURES[base_exposure]['label']} window-shape proxy",
                    exposure_family_id=exposure_meta["family_id"],
                    exposure_family_name=exposure_meta["family_name"],
                    lag_or_window="window_shape_7_14_28_days",
                    model_label="window_shape_linear_proxy",
                    model_family=model_family,
                    estimate_scale=COUNT_SHAPE_SCALE if candidate["model_kind"] == "count" else SEVERITY_SHAPE_SCALE,
                    model_result=shape_result,
                )
            )

    results_frame = pd.DataFrame(results_rows)
    diagnostics_frame = pd.DataFrame(diagnostic_rows)
    if results_frame.empty:
        raise ValueError("Stage 6 produced no benchmark rows. Verify Stage 4 triage and Stage 2 marts.")

    benchmark_root = _benchmark_root(root)
    plots_root = benchmark_root / "effect_plots"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    results_frame = results_frame.sort_values(
        ["outcome_family", "public_concept_label", "model_label", "lag_or_window"],
        ascending=[True, True, True, True],
        na_position="last",
    )
    diagnostics_frame = diagnostics_frame.sort_values(
        ["outcome_family", "public_concept_label", "model_label", "lag_or_window"],
        ascending=[True, True, True, True],
        na_position="last",
    )

    benchmark_results_path = benchmark_root / "benchmark_results.csv"
    diagnostics_path = benchmark_root / "model_diagnostics.csv"
    methods_note_path = benchmark_root / "model_methods_note.md"

    results_frame.to_csv(benchmark_results_path, index=False)
    diagnostics_frame.to_csv(diagnostics_path, index=False)
    methods_note_path.write_text(_render_methods_note(), encoding="utf-8")

    figure_paths = [
        plots_root / "figure_01_acute_benchmark_models.png",
        plots_root / "figure_02_perinatal_benchmark_models.png",
        plots_root / "figure_03_maternal_benchmark_models.png",
        plots_root / "figure_04_model_diagnostics.png",
    ]
    _plot_model_comparison(results_frame, figure_paths[0], "Acute benchmark ladder", "acute_daily")
    _plot_model_comparison(results_frame, figure_paths[1], "Perinatal benchmark ladder", "perinatal_delivery_window_aggregate")
    _plot_model_comparison(results_frame, figure_paths[2], "Maternal operational benchmark ladder", "maternal_operational")
    _plot_diagnostics(diagnostics_frame, figure_paths[3])

    summary_payload = {
        "status": "PASS",
        "stage": "stage6_progressive_benchmark_modeling",
        "analysis_id": config["analysis_id"],
        "benchmark_row_count": int(results_frame.shape[0]),
        "diagnostic_row_count": int(diagnostics_frame.shape[0]),
        "figure_count": len(figure_paths),
        "modeled_concept_count": int(results_frame["public_concept_name"].nunique()),
        "outputs": {
            "benchmark_results": _relative(root, benchmark_results_path),
            "model_diagnostics": _relative(root, diagnostics_path),
            "model_methods_note": _relative(root, methods_note_path),
            "effect_plots_directory": _relative(root, plots_root),
        },
    }
    (benchmark_root / "benchmark_summary.json").write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    return summary_payload
