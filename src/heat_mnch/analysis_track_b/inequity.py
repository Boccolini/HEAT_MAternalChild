from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

from .config import load_track_b_config
from .registry import ensure_track_b_directory_structure, track_b_burden_root, track_b_inequity_root
from .shared import load_context_lookup, normalize_municipality_code, relative_to_root


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _tertile(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.nunique() < 3:
        return pd.Series(["middle"] * len(series), index=series.index)
    lower = float(valid.quantile(1 / 3))
    upper = float(valid.quantile(2 / 3))
    return pd.Series(
        pd.cut(
            numeric,
            bins=[float("-inf"), lower, upper, float("inf")],
            labels=["low", "middle", "high"],
            include_lowest=True,
        ).astype(str),
        index=series.index,
    )


def _plot_modifier(frame: pd.DataFrame, output_path: Path, modifier_label: str) -> None:
    figure = plt.figure(figsize=(9.0, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    if frame.empty:
        axis.text(0.5, 0.5, "No inequity rows available.", ha="center", va="center")
    else:
        pivot = frame.pivot_table(index="modifier_level", columns="outcome_label", values="attributable_events", aggfunc="sum").fillna(0)
        pivot.plot(kind="bar", ax=axis)
        axis.set_ylabel("Attributable events")
    axis.set_title(modifier_label)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _methods_note() -> str:
    return (
        "# Track B Inequity and Heterogeneity Note\n\n"
        "Stage B6 summarizes vulnerability through one modifier at a time using municipality context inherited from the validated Track A context lookup.\n\n"
        "- Urban population proportion, expanded Gini index, expanded human development index, and mean altitude are summarized in tertiles.\n"
        "- Coastal municipality status is kept binary.\n"
        "- Results are reported as burden and observed events by modifier stratum rather than as a fully interacted likelihood model.\n"
    )


def build_track_b_inequity(root: Path, config_path: Path | None = None) -> dict[str, object]:
    ensure_track_b_directory_structure(root)
    config = load_track_b_config(root, config_path=config_path)
    burden_detail_path = track_b_burden_root(root) / "municipality_burden_detail.csv"
    if not burden_detail_path.exists():
        raise FileNotFoundError(f"Track B inequity requires municipality burden detail at {burden_detail_path}. Run track-b-burden first.")
    aggregated_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        burden_detail_path,
        usecols=["municipality_code", "outcome_label", "observed_events", "attributable_events"],
        low_memory=False,
        chunksize=500_000,
    ):
        chunk["municipality_code"] = normalize_municipality_code(chunk["municipality_code"])
        aggregated_chunks.append(
            chunk.groupby(["municipality_code", "outcome_label"], as_index=False)
            .agg(
                observed_events=("observed_events", "sum"),
                attributable_events=("attributable_events", "sum"),
            )
        )
    burden_detail = (
        pd.concat(aggregated_chunks, ignore_index=True)
        .groupby(["municipality_code", "outcome_label"], as_index=False)
        .agg(
            observed_events=("observed_events", "sum"),
            attributable_events=("attributable_events", "sum"),
        )
    )
    context = load_context_lookup(root)
    context["urbanicity_tertile"] = _tertile(context["urban_population_proportion"])
    context["inequality_tertile"] = _tertile(context["expanded_gini_index"])
    context["human_development_tertile"] = _tertile(context["expanded_human_development_index"])
    context["altitude_tertile"] = _tertile(context["mean_altitude_m"])
    context["coastal_status"] = context["coastal_municipality_flag"].map({1: "coastal", 0: "inland"}).fillna("unknown")
    merged = burden_detail.merge(context, on="municipality_code", how="left")

    modifier_specs = [
        ("urbanicity_tertile", "Urbanicity tertile"),
        ("inequality_tertile", "Expanded Gini tertile"),
        ("human_development_tertile", "Expanded human development tertile"),
        ("altitude_tertile", "Altitude tertile"),
        ("coastal_status", "Coastal status"),
    ]
    rows: list[dict[str, object]] = []
    inequity_root = track_b_inequity_root(root)
    maps_root = inequity_root / "flagship_vulnerability_maps"
    inequity_root.mkdir(parents=True, exist_ok=True)
    maps_root.mkdir(parents=True, exist_ok=True)
    for index, (modifier_column, modifier_label) in enumerate(modifier_specs, start=1):
        grouped = (
            merged.groupby(["outcome_label", modifier_column], as_index=False)
            .agg(
                municipality_count=("municipality_code", "nunique"),
                observed_events=("observed_events", "sum"),
                attributable_events=("attributable_events", "sum"),
            )
            .rename(columns={modifier_column: "modifier_level"})
        )
        grouped["modifier_name"] = modifier_label
        rows.extend(grouped.to_dict(orient="records"))
        _plot_modifier(grouped, maps_root / f"figure_{index:02d}_{modifier_column}.png", modifier_label)

    inequity_frame = pd.DataFrame(rows)
    results_path = inequity_root / "flagship_inequity_results.csv"
    note_path = inequity_root / "heterogeneity_note.md"
    inequity_frame.to_csv(results_path, index=False)
    note_path.write_text(_methods_note(), encoding="utf-8")

    payload = {
        "status": "PASS",
        "stage": "track_b_stage_b6_inequity",
        "analysis_id": config["analysis_id"],
        "inequity_row_count": int(inequity_frame.shape[0]),
        "outputs": {
            "flagship_inequity_results": relative_to_root(root, results_path),
            "flagship_vulnerability_maps_directory": relative_to_root(root, maps_root),
            "heterogeneity_note": relative_to_root(root, note_path),
        },
    }
    (inequity_root / "inequity_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
