from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROBABILITY_FLOOR = 1e-300
UF_TO_MACROREGION = {
    "AC": "North",
    "AL": "Northeast",
    "AP": "North",
    "AM": "North",
    "BA": "Northeast",
    "CE": "Northeast",
    "DF": "Central-West",
    "ES": "Southeast",
    "GO": "Central-West",
    "MA": "Northeast",
    "MT": "Central-West",
    "MS": "Central-West",
    "MG": "Southeast",
    "PA": "North",
    "PB": "Northeast",
    "PR": "South",
    "PE": "Northeast",
    "PI": "Northeast",
    "RJ": "Southeast",
    "RN": "Northeast",
    "RS": "South",
    "RO": "North",
    "RR": "North",
    "SC": "South",
    "SP": "Southeast",
    "SE": "Northeast",
    "TO": "North",
}


def relative_to_root(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def normalize_municipality_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6).str.slice(0, 6)


def normal_two_sided_p_from_z(z_value: float) -> tuple[float, str]:
    if not math.isfinite(z_value):
        return float("nan"), "no"
    probability = float(math.erfc(abs(z_value) / math.sqrt(2.0)))
    if probability < PROBABILITY_FLOOR:
        return PROBABILITY_FLOOR, "yes"
    return probability, "no"


def floor_probability(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce")
    floor_flag = numeric.lt(PROBABILITY_FLOOR) & numeric.notna()
    floored = numeric.where(~floor_flag, PROBABILITY_FLOOR)
    return floored, floor_flag.astype(int)


def pooled_continuous_proxy(
    frame: pd.DataFrame,
    *,
    exposure_column: str,
    outcome_column: str,
    denominator_column: str,
    stratum_columns: list[str],
    model_kind: str,
    exposure_scale_value: float = 1.0,
) -> dict[str, Any]:
    valid = frame.loc[
        frame[exposure_column].notna()
        & frame[denominator_column].fillna(0).gt(0)
        & frame[outcome_column].notna()
    ].copy()
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
            "exposure_scale_value": exposure_scale_value,
        }

    valid["x"] = pd.to_numeric(valid[exposure_column], errors="coerce")
    valid["w"] = pd.to_numeric(valid[denominator_column], errors="coerce")
    if model_kind == "count":
        valid["y"] = np.log((pd.to_numeric(valid[outcome_column], errors="coerce") + 0.5) / valid["w"])
    else:
        valid["y"] = pd.to_numeric(valid[outcome_column], errors="coerce") / valid["w"]
    valid = valid.loc[valid["x"].notna() & valid["y"].notna() & valid["w"].gt(0)].copy()
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
            "exposure_scale_value": exposure_scale_value,
        }

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
        & grouped["degrees_of_freedom"].gt(0)
        & grouped["det"].gt(0)
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
    grouped["intercept"] = (grouped["swy"] - grouped["slope"] * grouped["swx"]) / grouped["sw"]
    grouped["weighted_sse"] = grouped["swyy"] - (grouped["intercept"] * grouped["swy"]) - (grouped["slope"] * grouped["swxy"])
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
    p_value, p_floor_flag = normal_two_sided_p_from_z(z_value)
    scaled_slope = pooled_slope * exposure_scale_value
    scaled_se = pooled_se * exposure_scale_value
    if model_kind == "count":
        estimate = math.exp(scaled_slope)
        lower_ci = math.exp(scaled_slope - 1.96 * scaled_se)
        upper_ci = math.exp(scaled_slope + 1.96 * scaled_se)
    else:
        estimate = scaled_slope
        lower_ci = scaled_slope - 1.96 * scaled_se
        upper_ci = scaled_slope + 1.96 * scaled_se
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
        "overdispersion_ratio": float(grouped["weighted_sse"].sum() / grouped["degrees_of_freedom"].sum()),
        "internal_slope": pooled_slope,
        "internal_slope_se": pooled_se,
        "exposure_scale_value": exposure_scale_value,
    }


def shape_proxy(components: list[dict[str, Any]], *, x_column_name: str, model_kind: str) -> dict[str, Any]:
    frame = pd.DataFrame(components)
    valid = frame.loc[
        frame[x_column_name].notna()
        & frame["internal_slope"].notna()
        & frame["internal_slope_se"].notna()
        & pd.to_numeric(frame["internal_slope_se"], errors="coerce").gt(0)
    ].copy()
    if valid.shape[0] < 2:
        return {
            "estimate": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "p_value": float("nan"),
            "p_value_floor_applied_flag": "no",
            "valid_strata_count": int(valid.shape[0]),
            "internal_slope": float("nan"),
            "internal_slope_se": float("nan"),
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
            "internal_slope": float("nan"),
            "internal_slope_se": float("nan"),
            "exposure_scale_value": 1.0,
        }
    slope = ((sw * swxy) - (swx * swy)) / det
    slope_se = math.sqrt(sw / det) if det > 0 else float("nan")
    z_value = slope / slope_se if slope_se > 0 else float("nan")
    p_value, p_floor_flag = normal_two_sided_p_from_z(z_value)
    if model_kind == "count":
        estimate = math.exp(slope)
        lower_ci = math.exp(slope - 1.96 * slope_se)
        upper_ci = math.exp(slope + 1.96 * slope_se)
    else:
        estimate = slope
        lower_ci = slope - 1.96 * slope_se
        upper_ci = slope + 1.96 * slope_se
    return {
        "estimate": float(estimate),
        "lower_ci": float(lower_ci),
        "upper_ci": float(upper_ci),
        "p_value": float(p_value),
        "p_value_floor_applied_flag": p_floor_flag,
        "valid_strata_count": int(valid.shape[0]),
        "internal_slope": float(slope),
        "internal_slope_se": float(slope_se),
        "exposure_scale_value": 1.0,
    }


def assign_grouped_quantile_bins(
    frame: pd.DataFrame,
    *,
    exposure_column: str,
    group_columns: list[str],
    bin_count: int = 10,
) -> pd.Series:
    working = frame.loc[:, group_columns + [exposure_column]].copy()
    working["__rank_pct"] = working.groupby(group_columns)[exposure_column].rank(method="first", pct=True)
    bins = np.ceil(working["__rank_pct"] * bin_count).clip(lower=1, upper=bin_count)
    insufficient = working.groupby(group_columns)[exposure_column].transform("count").lt(3)
    return bins.where(~insufficient)


def load_municipality_lookup(root: Path) -> pd.DataFrame:
    lookup = pd.read_csv(root / "metadata/era5_municipio_grid_lookup.csv")
    lookup["municipality_code"] = normalize_municipality_code(lookup["COD_MUNICIPIO"])
    lookup["municipality_name"] = lookup["NM_MUNICIPIO"].astype(str)
    lookup["state_code"] = lookup["UF"].astype(str)
    lookup["macroregion"] = lookup["state_code"].map(UF_TO_MACROREGION).fillna("Unknown")
    return lookup.loc[:, ["municipality_code", "municipality_name", "state_code", "macroregion"]].drop_duplicates()


def load_context_lookup(root: Path) -> pd.DataFrame:
    path = root / "artifacts/track_a/analysis_marts/context_lookup.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Track B inequity requires Track A context lookup at {path}.")
    frame = pd.read_parquet(path)
    frame["municipality_code"] = frame["municipality_code"].astype(str).str.zfill(6)
    return frame


def load_track_b_matrix_partitions(root: Path, years: list[int], columns: list[str] | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    matrix_root = root / "artifacts/track_b/event_level/perinatal_window_exposure_matrix.parquet"
    for year in years:
        path = matrix_root / f"YEAR={year}" / f"perinatal_window_exposure_matrix_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Track B stage requires the event matrix partition at {path}. Run track-b-build-window-matrix first.")
        frames.append(pd.read_parquet(path, columns=columns))
    return pd.concat(frames, ignore_index=True)
