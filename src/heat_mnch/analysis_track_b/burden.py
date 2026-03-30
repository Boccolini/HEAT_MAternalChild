from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from .config import load_track_b_burden_config, load_track_b_config
from .registry import ensure_track_b_directory_structure, track_b_burden_root, track_b_models_root
from .shared import (
    assign_grouped_quantile_bins,
    load_municipality_lookup,
    load_track_b_matrix_partitions,
    relative_to_root,
)
from .neonatal import _add_lags, _load_mortality


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load_primary_slopes(root: Path) -> dict[str, float]:
    models_root = track_b_models_root(root)
    preterm = pd.read_csv(models_root / "preterm_main_models.csv")
    stillbirth = pd.read_csv(models_root / "stillbirth_main_models.csv")
    neonatal = pd.read_csv(models_root / "neonatal_main_models.csv")
    severity = pd.read_csv(models_root / "gest_days_lost_models.csv")
    def _clean(value: float) -> float:
        return float(value) if np.isfinite(value) else 0.0

    return {
        "preterm_beta": _clean(float(preterm.loc[preterm["primary_model_flag"].astype(str).eq("yes"), "internal_slope"].iloc[0])),
        "stillbirth_beta": _clean(float(stillbirth.loc[stillbirth["primary_model_flag"].astype(str).eq("yes"), "internal_slope"].iloc[0])),
        "neonatal_beta": _clean(float(neonatal.loc[neonatal["primary_model_flag"].astype(str).eq("yes"), "internal_slope"].iloc[0])),
        "severity_beta": _clean(float(severity.loc[severity["primary_model_flag"].astype(str).eq("yes"), "internal_slope"].iloc[0])),
    }


def _municipality_mrt(
    frame: pd.DataFrame,
    *,
    exposure_column: str,
    outcome_column: str,
    denominator_column: str,
) -> pd.DataFrame:
    working = frame.loc[
        frame[exposure_column].notna()
        & frame[outcome_column].notna()
        & frame[denominator_column].fillna(0).gt(0)
    ].copy()
    working["exposure_decile"] = assign_grouped_quantile_bins(
        working,
        exposure_column=exposure_column,
        group_columns=["municipality_code"],
        bin_count=10,
    )
    deciles = (
        working.loc[working["exposure_decile"].notna()]
        .groupby(["municipality_code", "exposure_decile"], as_index=False)
        .agg(
            denominator_total=(denominator_column, "sum"),
            outcome_total=(outcome_column, "sum"),
            exposure_mean=(exposure_column, "mean"),
        )
    )
    deciles["rate"] = (deciles["outcome_total"] + 0.5) / deciles["denominator_total"]
    best = deciles.sort_values(["municipality_code", "rate", "exposure_mean"]).drop_duplicates("municipality_code")
    fallback = working.groupby("municipality_code", as_index=False).agg(
        municipality_median_exposure=(exposure_column, "median")
    )
    mrt = fallback.merge(best.loc[:, ["municipality_code", "exposure_mean"]], on="municipality_code", how="left")
    mrt["mrt_temperature_c"] = mrt["exposure_mean"].where(mrt["exposure_mean"].notna(), mrt["municipality_median_exposure"])
    mrt["mrt_source"] = np.where(mrt["exposure_mean"].notna(), "empirical_minimum_decile_rate", "municipality_median_fallback")
    return mrt.loc[:, ["municipality_code", "mrt_temperature_c", "mrt_source"]]


def _attributable_fraction(beta: float, excess_temperature: pd.Series) -> pd.Series:
    positive_excess = pd.to_numeric(excess_temperature, errors="coerce").clip(lower=0)
    if beta <= 0:
        return pd.Series(np.zeros(len(positive_excess)), index=positive_excess.index)
    return 1.0 - np.exp(-beta * positive_excess)


def _aggregate_burden(
    frame: pd.DataFrame,
    *,
    lookup: pd.DataFrame,
    outcome_label: str,
    outcome_column: str,
    attributable_column: str,
    year_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = frame.merge(lookup, on="municipality_code", how="left")
    year_macro = (
        working.groupby([year_column, "macroregion"], as_index=False)
        .agg(
            observed_events=(outcome_column, "sum"),
            attributable_events=(attributable_column, "sum"),
        )
        .rename(columns={year_column: "calendar_year"})
    )
    year_macro["outcome_label"] = outcome_label
    state = (
        working.groupby([year_column, "state_code"], as_index=False)
        .agg(
            observed_events=(outcome_column, "sum"),
            attributable_events=(attributable_column, "sum"),
        )
        .rename(columns={year_column: "calendar_year"})
    )
    state["outcome_label"] = outcome_label
    return year_macro, state


def _plot_burden_map(frame: pd.DataFrame, output_path: Path, title: str) -> None:
    pivot = frame.pivot_table(index="macroregion", columns="calendar_year", values="attributable_events", aggfunc="sum").fillna(0)
    figure = plt.figure(figsize=(9.5, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    if pivot.empty:
        axis.text(0.5, 0.5, "No burden rows available.", ha="center", va="center")
    else:
        pivot.plot(kind="bar", ax=axis)
        axis.set_ylabel("Attributable events")
        axis.legend(title="Year", fontsize=8)
    axis.set_title(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _methods_note() -> str:
    return (
        "# Track B Burden Methods Note\n\n"
        "Stage B5 computes attributable burden using a municipality-specific empirical minimum-risk temperature counterfactual.\n\n"
        "- Municipality-specific MRT is approximated as the mean exposure of the municipality-specific exposure decile with the lowest observed event rate; when a municipality lacks sufficient decile support, the municipality median exposure is used as a fallback.\n"
        "- Attributable burden is computed only for temperatures above the municipality-specific MRT.\n"
        "- State-level results are published only when observed events are at least 100; otherwise burden is interpreted at macro-region level.\n"
    )


def _gather_note() -> str:
    return (
        "# GATHER-Compatible Burden Appendix\n\n"
        "- Counterfactual: municipality-specific empirical minimum-risk temperature.\n"
        "- Outcome-specific drivers: Track B primary preterm, stillbirth, and neonatal models plus the preterm severity model for gestational days lost.\n"
        "- Stability rule: state-level burden is suppressed when observed events are below 100.\n"
        "- Burden estimates are aggregated by calendar year and macro-region in the main workbook.\n"
    )


def build_track_b_burden(root: Path, config_path: Path | None = None) -> dict[str, object]:
    ensure_track_b_directory_structure(root)
    config = load_track_b_config(root, config_path=config_path)
    burden_config = load_track_b_burden_config(root)
    years = [int(year) for year in config["years"]]
    lookup = load_municipality_lookup(root)
    slopes = _load_primary_slopes(root)

    matrix_columns = [
        "record_type",
        "municipality_code",
        "calendar_year",
        "preterm_birth_flag",
        "stillbirth_primary_gte28_flag",
        "gestational_days_lost_below_37_weeks",
        "third_trimester_mean_tmax_c",
        "third_trimester_full_window_observed_flag",
        "days_0_6_before_event_mean_tmax_c",
        "days_0_6_before_event_full_window_observed_flag",
    ]
    event_matrix = load_track_b_matrix_partitions(root, years, columns=matrix_columns)
    event_matrix["birth_count"] = 1.0

    preterm = event_matrix.loc[
        event_matrix["record_type"].eq("live_birth")
        & pd.to_numeric(event_matrix["third_trimester_full_window_observed_flag"], errors="coerce").fillna(0).eq(1)
    ].copy()
    preterm_mrt = _municipality_mrt(
        preterm,
        exposure_column="third_trimester_mean_tmax_c",
        outcome_column="preterm_birth_flag",
        denominator_column="birth_count",
    )
    preterm = preterm.merge(preterm_mrt, on="municipality_code", how="left")
    preterm["attributable_fraction"] = _attributable_fraction(
        slopes["preterm_beta"],
        preterm["third_trimester_mean_tmax_c"] - preterm["mrt_temperature_c"],
    )
    preterm["attributable_preterm_births"] = preterm["preterm_birth_flag"] * preterm["attributable_fraction"]
    preterm["attributable_gestational_days_lost"] = (
        np.maximum(slopes["severity_beta"], 0.0)
        * (pd.to_numeric(preterm["third_trimester_mean_tmax_c"], errors="coerce") - pd.to_numeric(preterm["mrt_temperature_c"], errors="coerce")).clip(lower=0)
        * preterm["birth_count"]
    )

    stillbirth = event_matrix.loc[
        pd.to_numeric(event_matrix["days_0_6_before_event_full_window_observed_flag"], errors="coerce").fillna(0).eq(1)
    ].copy()
    stillbirth_mrt = _municipality_mrt(
        stillbirth,
        exposure_column="days_0_6_before_event_mean_tmax_c",
        outcome_column="stillbirth_primary_gte28_flag",
        denominator_column="birth_count",
    )
    stillbirth = stillbirth.merge(stillbirth_mrt, on="municipality_code", how="left")
    stillbirth["attributable_fraction"] = _attributable_fraction(
        slopes["stillbirth_beta"],
        stillbirth["days_0_6_before_event_mean_tmax_c"] - stillbirth["mrt_temperature_c"],
    )
    stillbirth["attributable_stillbirths"] = stillbirth["stillbirth_primary_gte28_flag"] * stillbirth["attributable_fraction"]

    neonatal = _add_lags(_load_mortality(root, years))
    neonatal = neonatal.loc[
        neonatal["QC_STATUS"].astype(str).eq("PASS")
        & pd.to_numeric(neonatal["OFFSET_VALID_FLAG"], errors="coerce").fillna(0).eq(1)
        & pd.to_numeric(neonatal["DENOMINATOR_ZERO_FLAG"], errors="coerce").fillna(0).eq(0)
        & pd.to_numeric(neonatal["DENOMINATOR_MISSING_FLAG"], errors="coerce").fillna(0).eq(0)
    ].copy()
    neonatal_mrt = _municipality_mrt(
        neonatal.rename(columns={"lag0_1_mean_tmax_c": "primary_exposure"}),
        exposure_column="primary_exposure",
        outcome_column="DEATHS_NEONATAL",
        denominator_column="LIVE_BIRTHS_TOTAL_ANNUAL",
    ).rename(columns={"mrt_temperature_c": "mrt_temperature_c", "mrt_source": "mrt_source"})
    neonatal = neonatal.merge(neonatal_mrt, on="municipality_code", how="left")
    neonatal["attributable_fraction"] = _attributable_fraction(
        slopes["neonatal_beta"],
        neonatal["lag0_1_mean_tmax_c"] - neonatal["mrt_temperature_c"],
    )
    neonatal["attributable_neonatal_deaths"] = neonatal["DEATHS_NEONATAL"] * neonatal["attributable_fraction"]
    neonatal["calendar_year"] = neonatal["calendar_year"].astype(int)

    preterm_year_macro, preterm_state = _aggregate_burden(
        preterm,
        lookup=lookup,
        outcome_label="Preterm birth",
        outcome_column="preterm_birth_flag",
        attributable_column="attributable_preterm_births",
        year_column="calendar_year",
    )
    stillbirth_year_macro, stillbirth_state = _aggregate_burden(
        stillbirth,
        lookup=lookup,
        outcome_label="Stillbirth",
        outcome_column="stillbirth_primary_gte28_flag",
        attributable_column="attributable_stillbirths",
        year_column="calendar_year",
    )
    neonatal_year_macro, neonatal_state = _aggregate_burden(
        neonatal,
        lookup=lookup,
        outcome_label="Neonatal mortality",
        outcome_column="DEATHS_NEONATAL",
        attributable_column="attributable_neonatal_deaths",
        year_column="calendar_year",
    )
    gest_days_year_macro = (
        preterm.merge(lookup, on="municipality_code", how="left")
        .groupby(["calendar_year", "macroregion"], as_index=False)
        .agg(attributable_gestational_days_lost=("attributable_gestational_days_lost", "sum"))
    )

    state_minimum_events = int(burden_config["burden_spec"]["state_publication_minimum_events"])
    state_tables = []
    for state_frame in [preterm_state, stillbirth_state, neonatal_state]:
        state_tables.append(
            state_frame.assign(
                publish_state_flag=state_frame["observed_events"].ge(state_minimum_events).astype(int)
            )
        )
    state_table = pd.concat(state_tables, ignore_index=True)

    burden_root = track_b_burden_root(root)
    maps_root = burden_root / "burden_maps"
    burden_root.mkdir(parents=True, exist_ok=True)
    maps_root.mkdir(parents=True, exist_ok=True)
    workbook_path = burden_root / "flagship_burden_estimates.xlsx"
    methods_path = burden_root / "burden_methods_note.md"
    gather_path = burden_root / "gather_burden_appendix.md"
    detail_path = burden_root / "municipality_burden_detail.csv"
    with pd.ExcelWriter(workbook_path) as writer:
        preterm_year_macro.to_excel(writer, sheet_name="preterm_year_macroregion", index=False)
        stillbirth_year_macro.to_excel(writer, sheet_name="stillbirth_year_macroregion", index=False)
        neonatal_year_macro.to_excel(writer, sheet_name="neonatal_year_macroregion", index=False)
        gest_days_year_macro.to_excel(writer, sheet_name="gest_days_lost_year_macro", index=False)
        state_table.to_excel(writer, sheet_name="state_stability", index=False)
    methods_path.write_text(_methods_note(), encoding="utf-8")
    gather_path.write_text(_gather_note(), encoding="utf-8")

    municipality_detail = pd.concat(
        [
            preterm.merge(lookup, on="municipality_code", how="left").assign(
                outcome_label="Preterm birth",
                attributable_events=preterm["attributable_preterm_births"],
                observed_events=preterm["preterm_birth_flag"],
            )[
                ["municipality_code", "calendar_year", "municipality_name", "state_code", "macroregion", "outcome_label", "observed_events", "attributable_events", "mrt_temperature_c", "mrt_source"]
            ],
            stillbirth.merge(lookup, on="municipality_code", how="left").assign(
                outcome_label="Stillbirth",
                attributable_events=stillbirth["attributable_stillbirths"],
                observed_events=stillbirth["stillbirth_primary_gte28_flag"],
            )[
                ["municipality_code", "calendar_year", "municipality_name", "state_code", "macroregion", "outcome_label", "observed_events", "attributable_events", "mrt_temperature_c", "mrt_source"]
            ],
            neonatal.merge(lookup, on="municipality_code", how="left").assign(
                outcome_label="Neonatal mortality",
                attributable_events=neonatal["attributable_neonatal_deaths"],
                observed_events=neonatal["DEATHS_NEONATAL"],
            )[
                ["municipality_code", "calendar_year", "municipality_name", "state_code", "macroregion", "outcome_label", "observed_events", "attributable_events", "mrt_temperature_c", "mrt_source"]
            ],
        ],
        ignore_index=True,
    )
    municipality_detail.to_csv(detail_path, index=False)

    _plot_burden_map(preterm_year_macro, maps_root / "figure_01_preterm_burden_macroregion.png", "Attributable preterm births by macro-region")
    _plot_burden_map(stillbirth_year_macro, maps_root / "figure_02_stillbirth_burden_macroregion.png", "Attributable stillbirths by macro-region")
    _plot_burden_map(neonatal_year_macro, maps_root / "figure_03_neonatal_burden_macroregion.png", "Attributable neonatal deaths by macro-region")

    payload = {
        "status": "PASS",
        "stage": "track_b_stage_b5_burden",
        "analysis_id": config["analysis_id"],
        "macroregion_rows": int(preterm_year_macro.shape[0] + stillbirth_year_macro.shape[0] + neonatal_year_macro.shape[0]),
        "state_rows": int(state_table.shape[0]),
        "outputs": {
            "flagship_burden_estimates": relative_to_root(root, workbook_path),
            "burden_methods_note": relative_to_root(root, methods_path),
            "burden_maps_directory": relative_to_root(root, maps_root),
            "gather_burden_appendix": relative_to_root(root, gather_path),
            "municipality_burden_detail": relative_to_root(root, detail_path),
        },
    }
    (burden_root / "burden_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
