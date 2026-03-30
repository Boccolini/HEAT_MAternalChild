from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


PROBABILITY_FLOOR = 1e-300
_LOG_PROBABILITY_FLOOR = math.log(PROBABILITY_FLOOR)
_SQRT_TWO = math.sqrt(2.0)
_LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)


def aggregate_binary_outcomes(
    frame: pd.DataFrame,
    index_columns: list[str],
    outcome_columns: list[str],
    denominator_column: str,
    group_column: str,
) -> pd.DataFrame:
    subset = frame.loc[frame[group_column].isin(["exposed", "reference"])].copy()
    if subset.empty:
        return pd.DataFrame()
    aggregations: dict[str, tuple[str, str]] = {
        "denominator_sum": (denominator_column, "sum"),
        "row_count": (group_column, "size"),
    }
    for outcome_column in outcome_columns:
        aggregations[f"{outcome_column}_sum"] = (outcome_column, "sum")
    return subset.groupby(index_columns + [group_column], as_index=False).agg(**aggregations)


def aggregate_quintile_outcomes(
    frame: pd.DataFrame,
    index_columns: list[str],
    outcome_columns: list[str],
    denominator_column: str,
    quintile_column: str,
) -> pd.DataFrame:
    subset = frame.loc[frame[quintile_column].notna()].copy()
    if subset.empty:
        return pd.DataFrame()
    subset["exposure_quintile"] = subset[quintile_column].astype(int)
    aggregations: dict[str, tuple[str, str]] = {
        "denominator_sum": (denominator_column, "sum"),
        "row_count": (quintile_column, "size"),
    }
    for outcome_column in outcome_columns:
        aggregations[f"{outcome_column}_sum"] = (outcome_column, "sum")
    return subset.groupby(index_columns + ["exposure_quintile"], as_index=False).agg(**aggregations)


def grouped_wide_to_long(
    grouped: pd.DataFrame,
    id_columns: list[str],
    outcome_columns: list[str],
) -> pd.DataFrame:
    if grouped.empty:
        return pd.DataFrame()
    value_columns = [f"{outcome_column}_sum" for outcome_column in outcome_columns]
    long_frame = grouped.melt(
        id_vars=id_columns + ["denominator_sum", "row_count"],
        value_vars=value_columns,
        var_name="outcome_variable",
        value_name="outcome_sum",
    )
    long_frame["outcome_variable"] = long_frame["outcome_variable"].str.removesuffix("_sum")
    return long_frame


def _normal_two_sided_p_from_z(z_value: float) -> tuple[float, bool]:
    if not math.isfinite(z_value):
        return float("nan"), False
    absolute_z = abs(z_value)
    if absolute_z < 8.0:
        probability = math.erfc(absolute_z / _SQRT_TWO)
        if probability < PROBABILITY_FLOOR:
            return PROBABILITY_FLOOR, True
        return float(probability), False

    log_probability = math.log(2.0) - 0.5 * absolute_z * absolute_z - math.log(absolute_z) - _LOG_SQRT_TWO_PI
    if log_probability <= _LOG_PROBABILITY_FLOOR:
        return PROBABILITY_FLOOR, True
    return float(math.exp(log_probability)), False


def apply_probability_floor(frame: pd.DataFrame) -> pd.DataFrame:
    stabilized = frame.copy()
    for column_name, flag_name in (
        ("p_value", "p_value_floor_applied_flag"),
        ("q_value", "q_value_floor_applied_flag"),
    ):
        if column_name not in stabilized.columns:
            continue
        existing_flag = (
            stabilized[flag_name].astype(str).eq("yes")
            if flag_name in stabilized.columns
            else pd.Series(False, index=stabilized.index)
        )
        mask = stabilized[column_name].notna() & stabilized[column_name].astype(float).lt(PROBABILITY_FLOOR)
        stabilized[column_name] = stabilized[column_name].mask(mask, PROBABILITY_FLOOR)
        stabilized[flag_name] = np.where(mask | existing_flag, "yes", "no")
    return stabilized


def pool_binary_rate_ratios(
    grouped: pd.DataFrame,
    combination_columns: list[str],
    stratum_columns: list[str],
    group_column: str,
) -> pd.DataFrame:
    if grouped.empty:
        return pd.DataFrame(columns=combination_columns)

    all_combinations = grouped.loc[:, combination_columns].drop_duplicates()
    pivot = grouped.pivot_table(
        index=combination_columns + stratum_columns,
        columns=group_column,
        values=["outcome_sum", "denominator_sum", "row_count"],
        aggfunc="first",
        fill_value=0,
    )
    pivot.columns = [f"{metric}_{group_label}" for metric, group_label in pivot.columns.to_flat_index()]
    pivot = pivot.reset_index()

    for column_name in [
        "outcome_sum_exposed",
        "outcome_sum_reference",
        "denominator_sum_exposed",
        "denominator_sum_reference",
        "row_count_exposed",
        "row_count_reference",
    ]:
        if column_name not in pivot.columns:
            pivot[column_name] = 0.0

    base_summary = pivot.groupby(combination_columns, as_index=False).agg(
        exposed_row_count=("row_count_exposed", "sum"),
        reference_row_count=("row_count_reference", "sum"),
        exposed_outcome_sum=("outcome_sum_exposed", "sum"),
        reference_outcome_sum=("outcome_sum_reference", "sum"),
        exposed_denominator_sum=("denominator_sum_exposed", "sum"),
        reference_denominator_sum=("denominator_sum_reference", "sum"),
    )

    valid = pivot.loc[
        pivot["row_count_exposed"].gt(0)
        & pivot["row_count_reference"].gt(0)
        & pivot["denominator_sum_exposed"].gt(0)
        & pivot["denominator_sum_reference"].gt(0)
    ].copy()

    if valid.empty:
        result = all_combinations.merge(base_summary, on=combination_columns, how="left")
        result["estimate"] = float("nan")
        result["lower_ci"] = float("nan")
        result["upper_ci"] = float("nan")
        result["p_value"] = float("nan")
        result["p_value_floor_applied_flag"] = "no"
        result["valid_strata_count"] = 0
        return result

    valid["exposed_cases"] = valid["outcome_sum_exposed"].astype(float) + 0.5
    valid["reference_cases"] = valid["outcome_sum_reference"].astype(float) + 0.5
    valid["log_effect"] = np.log(valid["exposed_cases"] / valid["denominator_sum_exposed"].astype(float)) - np.log(
        valid["reference_cases"] / valid["denominator_sum_reference"].astype(float)
    )
    valid["variance"] = (1.0 / valid["exposed_cases"]) + (1.0 / valid["reference_cases"])
    valid["weight"] = 1.0 / valid["variance"]
    valid["weighted_log_effect"] = valid["weight"] * valid["log_effect"]

    pooled = valid.groupby(combination_columns, as_index=False).agg(
        sum_weight=("weight", "sum"),
        sum_weighted_log_effect=("weighted_log_effect", "sum"),
        valid_strata_count=("weight", "size"),
    )
    pooled["pooled_log_effect"] = pooled["sum_weighted_log_effect"] / pooled["sum_weight"]
    pooled["pooled_se"] = np.sqrt(1.0 / pooled["sum_weight"])
    pooled["z_value"] = pooled["pooled_log_effect"] / pooled["pooled_se"]
    pooled["estimate"] = np.exp(pooled["pooled_log_effect"])
    pooled["lower_ci"] = np.exp(pooled["pooled_log_effect"] - (1.96 * pooled["pooled_se"]))
    pooled["upper_ci"] = np.exp(pooled["pooled_log_effect"] + (1.96 * pooled["pooled_se"]))

    probability_rows = pooled["z_value"].apply(_normal_two_sided_p_from_z)
    pooled["p_value"] = probability_rows.map(lambda row: row[0])
    pooled["p_value_floor_applied_flag"] = probability_rows.map(lambda row: "yes" if row[1] else "no")

    result = all_combinations.merge(base_summary, on=combination_columns, how="left").merge(
        pooled.loc[
            :,
            combination_columns
            + [
                "estimate",
                "lower_ci",
                "upper_ci",
                "p_value",
                "p_value_floor_applied_flag",
                "valid_strata_count",
            ],
        ],
        on=combination_columns,
        how="left",
    )

    result["p_value_floor_applied_flag"] = result["p_value_floor_applied_flag"].fillna("no")
    result["valid_strata_count"] = result["valid_strata_count"].fillna(0).astype(int)
    return result


def pool_quintile_trends(
    grouped: pd.DataFrame,
    combination_columns: list[str],
    stratum_columns: list[str],
) -> pd.DataFrame:
    if grouped.empty:
        return pd.DataFrame(columns=combination_columns)

    all_combinations = grouped.loc[:, combination_columns].drop_duplicates()
    q1 = grouped.loc[grouped["exposure_quintile"].eq(1)]
    q5 = grouped.loc[grouped["exposure_quintile"].eq(5)]
    base_summary = all_combinations.copy()
    if not q1.empty:
        q1_summary = q1.groupby(combination_columns, as_index=False).agg(
            reference_row_count=("row_count", "sum"),
            reference_outcome_sum=("outcome_sum", "sum"),
            reference_denominator_sum=("denominator_sum", "sum"),
        )
        base_summary = base_summary.merge(q1_summary, on=combination_columns, how="left")
    if not q5.empty:
        q5_summary = q5.groupby(combination_columns, as_index=False).agg(
            exposed_row_count=("row_count", "sum"),
            exposed_outcome_sum=("outcome_sum", "sum"),
            exposed_denominator_sum=("denominator_sum", "sum"),
        )
        base_summary = base_summary.merge(q5_summary, on=combination_columns, how="left")

    valid_rows = grouped.loc[
        grouped["denominator_sum"].gt(0)
        & grouped["row_count"].gt(0)
        & grouped["exposure_quintile"].notna()
    ].copy()
    if valid_rows.empty:
        result = base_summary.copy()
        result["estimate"] = float("nan")
        result["lower_ci"] = float("nan")
        result["upper_ci"] = float("nan")
        result["p_value"] = float("nan")
        result["p_value_floor_applied_flag"] = "no"
        result["valid_strata_count"] = 0
        for column_name in [
            "exposed_row_count",
            "reference_row_count",
            "exposed_outcome_sum",
            "reference_outcome_sum",
            "exposed_denominator_sum",
            "reference_denominator_sum",
        ]:
            if column_name not in result.columns:
                result[column_name] = 0.0
            else:
                result[column_name] = result[column_name].fillna(0.0)
        return result

    valid_rows["x"] = valid_rows["exposure_quintile"].astype(float)
    valid_rows["w"] = valid_rows["denominator_sum"].astype(float)
    valid_rows["y"] = np.log((valid_rows["outcome_sum"].astype(float) + 0.5) / valid_rows["denominator_sum"].astype(float))
    valid_rows["wx"] = valid_rows["w"] * valid_rows["x"]
    valid_rows["wy"] = valid_rows["w"] * valid_rows["y"]
    valid_rows["wxx"] = valid_rows["w"] * valid_rows["x"] * valid_rows["x"]
    valid_rows["wxy"] = valid_rows["w"] * valid_rows["x"] * valid_rows["y"]
    valid_rows["wyy"] = valid_rows["w"] * valid_rows["y"] * valid_rows["y"]

    stratum_frame = valid_rows.groupby(combination_columns + stratum_columns, as_index=False).agg(
        sw=("w", "sum"),
        swx=("wx", "sum"),
        swy=("wy", "sum"),
        swxx=("wxx", "sum"),
        swxy=("wxy", "sum"),
        swyy=("wyy", "sum"),
        observed_quintile_count=("exposure_quintile", "nunique"),
    )
    stratum_frame["det"] = (stratum_frame["sw"] * stratum_frame["swxx"]) - (stratum_frame["swx"] ** 2)
    stratum_frame["degrees_of_freedom"] = stratum_frame["observed_quintile_count"] - 2

    valid_strata = stratum_frame.loc[
        stratum_frame["observed_quintile_count"].ge(3)
        & stratum_frame["det"].gt(0)
        & stratum_frame["degrees_of_freedom"].gt(0)
    ].copy()

    if valid_strata.empty:
        result = base_summary.copy()
        result["estimate"] = float("nan")
        result["lower_ci"] = float("nan")
        result["upper_ci"] = float("nan")
        result["p_value"] = float("nan")
        result["p_value_floor_applied_flag"] = "no"
        result["valid_strata_count"] = 0
        for column_name in [
            "exposed_row_count",
            "reference_row_count",
            "exposed_outcome_sum",
            "reference_outcome_sum",
            "exposed_denominator_sum",
            "reference_denominator_sum",
        ]:
            if column_name not in result.columns:
                result[column_name] = 0.0
            else:
                result[column_name] = result[column_name].fillna(0.0)
        return result

    valid_strata["slope"] = (
        (valid_strata["sw"] * valid_strata["swxy"]) - (valid_strata["swx"] * valid_strata["swy"])
    ) / valid_strata["det"]
    valid_strata["intercept"] = (valid_strata["swy"] - (valid_strata["slope"] * valid_strata["swx"])) / valid_strata["sw"]
    valid_strata["weighted_sse"] = valid_strata["swyy"] - (
        (valid_strata["intercept"] * valid_strata["swy"]) + (valid_strata["slope"] * valid_strata["swxy"])
    )
    valid_strata["weighted_sse"] = valid_strata["weighted_sse"].clip(lower=0.0)
    valid_strata["sigma_squared"] = valid_strata["weighted_sse"] / valid_strata["degrees_of_freedom"]
    valid_strata["slope_variance"] = (valid_strata["sigma_squared"] * valid_strata["sw"]) / valid_strata["det"]
    valid_strata = valid_strata.loc[valid_strata["slope_variance"].gt(0)].copy()

    if valid_strata.empty:
        result = base_summary.copy()
        result["estimate"] = float("nan")
        result["lower_ci"] = float("nan")
        result["upper_ci"] = float("nan")
        result["p_value"] = float("nan")
        result["p_value_floor_applied_flag"] = "no"
        result["valid_strata_count"] = 0
        return result.fillna(0.0)

    valid_strata["weight"] = 1.0 / valid_strata["slope_variance"]
    valid_strata["weighted_slope"] = valid_strata["weight"] * valid_strata["slope"]

    pooled = valid_strata.groupby(combination_columns, as_index=False).agg(
        sum_weight=("weight", "sum"),
        sum_weighted_slope=("weighted_slope", "sum"),
        valid_strata_count=("weight", "size"),
    )
    pooled["pooled_slope"] = pooled["sum_weighted_slope"] / pooled["sum_weight"]
    pooled["pooled_se"] = np.sqrt(1.0 / pooled["sum_weight"])
    pooled["z_value"] = pooled["pooled_slope"] / pooled["pooled_se"]
    pooled["estimate"] = np.exp(pooled["pooled_slope"])
    pooled["lower_ci"] = np.exp(pooled["pooled_slope"] - (1.96 * pooled["pooled_se"]))
    pooled["upper_ci"] = np.exp(pooled["pooled_slope"] + (1.96 * pooled["pooled_se"]))

    probability_rows = pooled["z_value"].apply(_normal_two_sided_p_from_z)
    pooled["p_value"] = probability_rows.map(lambda row: row[0])
    pooled["p_value_floor_applied_flag"] = probability_rows.map(lambda row: "yes" if row[1] else "no")

    result = base_summary.merge(
        pooled.loc[
            :,
            combination_columns
            + [
                "estimate",
                "lower_ci",
                "upper_ci",
                "p_value",
                "p_value_floor_applied_flag",
                "valid_strata_count",
            ],
        ],
        on=combination_columns,
        how="left",
    )

    result["p_value_floor_applied_flag"] = result["p_value_floor_applied_flag"].fillna("no")
    result["valid_strata_count"] = result["valid_strata_count"].fillna(0).astype(int)
    for column_name in [
        "exposed_row_count",
        "reference_row_count",
        "exposed_outcome_sum",
        "reference_outcome_sum",
        "exposed_denominator_sum",
        "reference_denominator_sum",
    ]:
        if column_name not in result.columns:
            result[column_name] = 0.0
        else:
            result[column_name] = result[column_name].fillna(0.0)
    return result
