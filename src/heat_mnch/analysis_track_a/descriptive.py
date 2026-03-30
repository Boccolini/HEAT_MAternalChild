from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from .config import load_track_a_config
from .registry import track_a_root


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


EXPOSURE_COLUMNS = [
    "maximum_daily_temperature_c",
    "mean_daily_relative_humidity_pct",
    "heatwave_duration_days_p95_threshold",
    "approximate_heat_index_c",
]
FALLBACK_COLUMN = "any_climate_fallback_imputation_flag"

ACUTE_OUTCOMES = [
    "neonatal_death_count",
    "infant_death_count",
    "under_five_death_count",
    "acute_care_proxy_count",
]
PERINATAL_OUTCOMES = [
    "preterm_birth_count",
    "very_preterm_birth_count",
    "gestational_days_lost_below_37_weeks",
    "stillbirth_primary_gte28_count",
    "stillbirth_sensitivity_gte22_count",
]
MATERNAL_OUTCOMES = [
    "obstetric_admission_count",
    "hypertensive_disorder_of_pregnancy_count",
    "severe_hypertensive_disorder_of_pregnancy_count",
    "preterm_labor_admission_count",
    "gestational_diabetes_admission_count",
]
DENOMINATOR_FLAG_COLUMNS = [
    "denominator_zero_flag",
    "denominator_missing_flag",
    "rate_not_computable_flag",
    "offset_valid_flag",
]
CONTEXT_COLUMNS = [
    "urban_population_proportion",
    "expanded_gini_index",
    "expanded_human_development_index",
    "mean_altitude_m",
    "coastal_municipality_flag",
    "context_join_status",
]

STATE_CODE_TO_MACROREGION = {
    "11": "North",
    "12": "North",
    "13": "North",
    "14": "North",
    "15": "North",
    "16": "North",
    "17": "North",
    "21": "Northeast",
    "22": "Northeast",
    "23": "Northeast",
    "24": "Northeast",
    "25": "Northeast",
    "26": "Northeast",
    "27": "Northeast",
    "28": "Northeast",
    "29": "Northeast",
    "31": "Southeast",
    "32": "Southeast",
    "33": "Southeast",
    "35": "Southeast",
    "41": "South",
    "42": "South",
    "43": "South",
    "50": "Center-West",
    "51": "Center-West",
    "52": "Center-West",
    "53": "Center-West",
}


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _mart_partition_path(root: Path, mart_name: str, year: int) -> Path:
    return track_a_root(root) / "analysis_marts" / mart_name / f"YEAR={year}" / f"{mart_name}_{year}.parquet"


def _load_partition(root: Path, mart_name: str, year: int, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(_mart_partition_path(root, mart_name, year), columns=columns)


def _normalize_municipality_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(7)


def _with_macro_region(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["municipality_code"] = _normalize_municipality_code(enriched["municipality_code"])
    enriched["macro_region"] = (
        enriched["municipality_code"].str[:2].map(STATE_CODE_TO_MACROREGION).fillna("Unknown")
    )
    return enriched


def _freeze_root(root: Path) -> Path:
    return track_a_root(root) / "freeze"


def _load_readiness_assets(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    readiness_path = _freeze_root(root) / "analysis_ready_matrix.csv"
    if not readiness_path.exists():
        raise FileNotFoundError(
            f"Stage 3 requires the Stage 1 readiness matrix at {readiness_path}. Run track-a-freeze first."
        )
    readiness = pd.read_csv(readiness_path)
    if "headline_eligible_flag" not in readiness.columns:
        readiness["headline_eligible_flag"] = readiness["stage1_classification"].eq("benchmark_ready").map(
            {True: "yes", False: "no"}
        )
    if "public_concept_name" not in readiness.columns:
        readiness["public_concept_name"] = readiness["academic_variable_name"]
    if "public_concept_label" not in readiness.columns:
        readiness["public_concept_label"] = readiness["academic_label"]
    if "duplicate_prompt_concept_flag" not in readiness.columns:
        readiness["duplicate_prompt_concept_flag"] = "no"

    concept_registry_path = _freeze_root(root) / "public_concept_registry.csv"
    if concept_registry_path.exists():
        concept_registry = pd.read_csv(concept_registry_path)
    else:
        concept_registry = (
            readiness.groupby(["public_concept_name", "public_concept_label"], as_index=False)
            .agg(
                prompt_labels=("prompt_label", lambda series: ";".join(sorted(series.astype(str).tolist()))),
                prompt_label_count=("prompt_label", "count"),
                stage1_classifications=("stage1_classification", lambda series: ";".join(sorted(set(series.astype(str))))),
                semantic_alignments=("semantic_alignment", lambda series: ";".join(sorted(set(series.astype(str))))),
                headline_eligible_flag=("headline_eligible_flag", lambda series: "yes" if (series == "yes").any() else "no"),
            )
        )
        concept_registry["concept_resolution_status"] = concept_registry["prompt_label_count"].gt(1).map(
            {True: "shared_public_concept", False: "one_prompt_one_public_concept"}
        )
        concept_registry["concept_resolution_note"] = concept_registry["concept_resolution_status"].map(
            {
                "shared_public_concept": "Multiple prompt-level concepts map to a single public-facing Track A concept and should be reported once.",
                "one_prompt_one_public_concept": "The prompt-level concept maps uniquely to a single public-facing Track A concept.",
            }
        )
    return readiness, concept_registry


def _eligibility_lookup(readiness: pd.DataFrame) -> dict[str, dict[str, str]]:
    frame = readiness.sort_values(["academic_variable_name", "stage1_classification"]).drop_duplicates(
        "academic_variable_name"
    )
    return {
        str(row["academic_variable_name"]): {
            "public_concept_name": str(row["public_concept_name"]),
            "public_concept_label": str(row["public_concept_label"]),
            "stage1_classification": str(row["stage1_classification"]),
            "headline_eligible_flag": str(row["headline_eligible_flag"]),
        }
        for row in frame.to_dict(orient="records")
    }


def _round_numeric(frame: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    rounded = frame.copy()
    numeric_columns = rounded.select_dtypes(include=["number"]).columns
    rounded[numeric_columns] = rounded[numeric_columns].round(digits)
    return rounded


def _compute_exposure_annual_summary(root: Path, years: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year in years:
        frame = _load_partition(
            root,
            "acute_daily_mart",
            year,
            columns=["municipality_code", *EXPOSURE_COLUMNS, FALLBACK_COLUMN],
        )
        fallback_share = float(frame[FALLBACK_COLUMN].mean())
        for exposure in EXPOSURE_COLUMNS:
            series = frame[exposure].astype(float)
            rows.append(
                {
                    "calendar_year": year,
                    "exposure_variable": exposure,
                    "municipality_day_count": int(series.shape[0]),
                    "mean_value": float(series.mean()),
                    "standard_deviation": float(series.std()),
                    "median_value": float(series.median()),
                    "p95_value": float(series.quantile(0.95)),
                    "minimum_value": float(series.min()),
                    "maximum_value": float(series.max()),
                    "fallback_share": fallback_share,
                }
            )
    return pd.DataFrame(rows)


def _compute_exposure_macro_region_summary(root: Path, years: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year in years:
        frame = _with_macro_region(
            _load_partition(
                root,
                "acute_daily_mart",
                year,
                columns=["municipality_code", *EXPOSURE_COLUMNS, FALLBACK_COLUMN],
            )
        )
        for macro_region, macro_frame in frame.groupby("macro_region", sort=True):
            fallback_share = float(macro_frame[FALLBACK_COLUMN].mean())
            for exposure in EXPOSURE_COLUMNS:
                series = macro_frame[exposure].astype(float)
                rows.append(
                    {
                        "calendar_year": year,
                        "macro_region": macro_region,
                        "exposure_variable": exposure,
                        "municipality_day_count": int(series.shape[0]),
                        "mean_value": float(series.mean()),
                        "p95_value": float(series.quantile(0.95)),
                        "fallback_share": fallback_share,
                    }
                )
    return pd.DataFrame(rows)


def _compute_municipal_exposure_summary(root: Path, years: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in years:
        frame = _load_partition(
            root,
            "acute_daily_mart",
            year,
            columns=["municipality_code", *EXPOSURE_COLUMNS, FALLBACK_COLUMN],
        )
        frames.append(frame)
    combined = _with_macro_region(pd.concat(frames, ignore_index=True))
    grouped = combined.groupby(["municipality_code", "macro_region"], as_index=False).agg(
        maximum_daily_temperature_c_mean=("maximum_daily_temperature_c", "mean"),
        maximum_daily_temperature_c_p95=("maximum_daily_temperature_c", lambda series: series.quantile(0.95)),
        maximum_daily_temperature_c_variance=("maximum_daily_temperature_c", "var"),
        mean_daily_relative_humidity_pct_mean=("mean_daily_relative_humidity_pct", "mean"),
        mean_daily_relative_humidity_pct_p95=("mean_daily_relative_humidity_pct", lambda series: series.quantile(0.95)),
        mean_daily_relative_humidity_pct_variance=("mean_daily_relative_humidity_pct", "var"),
        heatwave_duration_days_p95_threshold_mean=("heatwave_duration_days_p95_threshold", "mean"),
        heatwave_duration_days_p95_threshold_p95=("heatwave_duration_days_p95_threshold", lambda series: series.quantile(0.95)),
        heatwave_duration_days_p95_threshold_variance=("heatwave_duration_days_p95_threshold", "var"),
        approximate_heat_index_c_mean=("approximate_heat_index_c", "mean"),
        approximate_heat_index_c_p95=("approximate_heat_index_c", lambda series: series.quantile(0.95)),
        approximate_heat_index_c_variance=("approximate_heat_index_c", "var"),
        climate_fallback_share=(FALLBACK_COLUMN, "mean"),
    )
    return grouped


def _summarize_outcome_partition(
    frame: pd.DataFrame,
    year: int,
    mart_name: str,
    outcome_columns: list[str],
    eligibility: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for outcome in outcome_columns:
        series = frame[outcome].astype(float)
        flags = eligibility.get(
            outcome,
            {
                "public_concept_name": outcome,
                "public_concept_label": outcome.replace("_", " ").title(),
                "stage1_classification": "unclassified",
                "headline_eligible_flag": "no",
            },
        )
        rows.append(
            {
                "mart_name": mart_name,
                "calendar_year": year,
                "public_concept_name": flags["public_concept_name"],
                "public_concept_label": flags["public_concept_label"],
                "stage1_classification": flags["stage1_classification"],
                "headline_eligible_flag": flags["headline_eligible_flag"],
                "total_count": float(series.sum()),
                "mean_daily_count": float(series.mean()),
                "standard_deviation_daily_count": float(series.std()),
                "nonzero_day_count": int(series.gt(0).sum()),
                "municipality_day_count": int(series.shape[0]),
            }
        )
    return rows


def _compute_outcome_annual_summary(
    root: Path,
    years: list[int],
    readiness: pd.DataFrame,
) -> pd.DataFrame:
    eligibility = _eligibility_lookup(readiness)
    rows: list[dict[str, Any]] = []
    mart_definitions = [
        ("acute_daily_mart", ACUTE_OUTCOMES),
        ("perinatal_delivery_window_mart", PERINATAL_OUTCOMES),
        ("maternal_operational_mart", MATERNAL_OUTCOMES),
    ]
    for mart_name, outcome_columns in mart_definitions:
        for year in years:
            frame = _load_partition(root, mart_name, year, columns=outcome_columns)
            rows.extend(_summarize_outcome_partition(frame, year, mart_name, outcome_columns, eligibility))
    return pd.DataFrame(rows)


def _compute_denominator_validity_summary(root: Path, years: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mart_name in ["acute_daily_mart", "perinatal_delivery_window_mart", "maternal_operational_mart"]:
        for year in years:
            frame = _load_partition(root, mart_name, year, columns=DENOMINATOR_FLAG_COLUMNS)
            rows.append(
                {
                    "mart_name": mart_name,
                    "calendar_year": year,
                    "rows_total": int(frame.shape[0]),
                    "denominator_zero_row_count": int(frame["denominator_zero_flag"].sum()),
                    "denominator_missing_row_count": int(frame["denominator_missing_flag"].sum()),
                    "rate_not_computable_row_count": int(frame["rate_not_computable_flag"].sum()),
                    "offset_valid_row_count": int(frame["offset_valid_flag"].sum()),
                    "offset_invalid_row_count": int(frame.shape[0] - frame["offset_valid_flag"].sum()),
                    "denominator_zero_share": float(frame["denominator_zero_flag"].mean()),
                    "denominator_missing_share": float(frame["denominator_missing_flag"].mean()),
                    "rate_not_computable_share": float(frame["rate_not_computable_flag"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _compute_context_summary(root: Path) -> pd.DataFrame:
    frame = pd.read_parquet(track_a_root(root) / "analysis_marts" / "context_lookup.parquet")
    rows: list[dict[str, Any]] = []
    for column in [
        "urban_population_proportion",
        "expanded_gini_index",
        "expanded_human_development_index",
        "mean_altitude_m",
    ]:
        series = frame[column].astype(float)
        for statistic, value in {
            "mean": float(series.mean()),
            "standard_deviation": float(series.std()),
            "median": float(series.median()),
            "p95": float(series.quantile(0.95)),
            "minimum": float(series.min()),
            "maximum": float(series.max()),
            "missing_count": float(series.isna().sum()),
        }.items():
            rows.append(
                {
                    "summary_domain": "numeric_context",
                    "variable_name": column,
                    "statistic": statistic,
                    "value": value,
                }
            )

    coastal_counts = frame["coastal_municipality_flag"].value_counts(dropna=False).sort_index()
    for category, count in coastal_counts.items():
        rows.append(
            {
                "summary_domain": "coastal_flag_frequency",
                "variable_name": "coastal_municipality_flag",
                "statistic": str(category),
                "value": float(count),
            }
        )

    join_status_counts = frame["context_join_status"].value_counts(dropna=False)
    for category, count in join_status_counts.items():
        rows.append(
            {
                "summary_domain": "context_join_status_frequency",
                "variable_name": "context_join_status",
                "statistic": str(category),
                "value": float(count),
            }
        )
    return pd.DataFrame(rows)


def _compute_context_readiness_summary(root: Path) -> pd.DataFrame:
    frame = pd.read_parquet(track_a_root(root) / "analysis_marts" / "context_lookup.parquet")
    rows = [
        {"check_name": "row_count", "check_value": str(int(frame.shape[0]))},
        {"check_name": "duplicate_municipality_code_count", "check_value": str(int(frame["municipality_code"].duplicated().sum()))},
        {"check_name": "required_fields_present", "check_value": "yes" if set(CONTEXT_COLUMNS).issubset(frame.columns) else "no"},
        {"check_name": "missing_urban_population_proportion_count", "check_value": str(int(frame["urban_population_proportion"].isna().sum()))},
        {"check_name": "missing_expanded_gini_index_count", "check_value": str(int(frame["expanded_gini_index"].isna().sum()))},
        {"check_name": "missing_expanded_human_development_index_count", "check_value": str(int(frame["expanded_human_development_index"].isna().sum()))},
        {"check_name": "missing_mean_altitude_m_count", "check_value": str(int(frame["mean_altitude_m"].isna().sum()))},
        {"check_name": "recommended_use", "check_value": "stratification_and_heterogeneity_only"},
        {"check_name": "endpoint_readiness_status", "check_value": "not_an_endpoint_asset"},
    ]
    return pd.DataFrame(rows)


def _write_csv_outputs(atlas_dir: Path, tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for name, frame in tables.items():
        path = atlas_dir / f"{name}.csv"
        _round_numeric(frame).to_csv(path, index=False)
        outputs[name] = str(path)
    return outputs


def _style_workbook(path: Path) -> None:
    workbook = load_workbook(path)
    header_fill = PatternFill(fill_type="solid", fgColor="D9EAF7")
    header_font = Font(bold=True)
    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
        for column_cells in worksheet.columns:
            max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
            worksheet.column_dimensions[get_column_letter(column_cells[0].column)].width = min(max_length + 2, 40)
    workbook.save(path)


def _write_workbook(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in tables.items():
            _round_numeric(frame).to_excel(writer, index=False, sheet_name=name[:31])
    _style_workbook(path)


def _plot_exposure_annual_trends(path: Path, exposure_annual: pd.DataFrame) -> None:
    selected = exposure_annual.loc[
        exposure_annual["exposure_variable"].isin(
            ["maximum_daily_temperature_c", "approximate_heat_index_c"]
        )
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for axis, exposure in zip(axes, ["maximum_daily_temperature_c", "approximate_heat_index_c"]):
        frame = selected.loc[selected["exposure_variable"].eq(exposure)].copy()
        axis.plot(frame["calendar_year"], frame["mean_value"], marker="o", label="Mean")
        axis.plot(frame["calendar_year"], frame["p95_value"], marker="s", label="P95")
        axis.set_title(exposure.replace("_", " ").title())
        axis.set_xlabel("Calendar year")
        axis.legend()
    fig.suptitle("Annual Exposure Trends")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_fallback_share(path: Path, exposure_annual: pd.DataFrame) -> None:
    frame = exposure_annual.loc[
        exposure_annual["exposure_variable"].eq("maximum_daily_temperature_c"),
        ["calendar_year", "fallback_share"],
    ].copy()
    fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    axis.bar(frame["calendar_year"].astype(str), frame["fallback_share"])
    axis.set_title("Climate Fallback Share by Year")
    axis.set_xlabel("Calendar year")
    axis.set_ylabel("Share")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_outcome_annual_totals(path: Path, outcome_annual: pd.DataFrame) -> None:
    frame = outcome_annual.loc[outcome_annual["headline_eligible_flag"].eq("yes")].copy()
    top_concepts = (
        frame.groupby("public_concept_name", as_index=False)["total_count"]
        .mean()
        .sort_values("total_count", ascending=False)
        .head(4)["public_concept_name"]
        .tolist()
    )
    plot_frame = frame.loc[frame["public_concept_name"].isin(top_concepts)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, public_concept_name in zip(axes.flatten(), top_concepts):
        concept_frame = plot_frame.loc[plot_frame["public_concept_name"].eq(public_concept_name)].copy()
        axis.bar(concept_frame["calendar_year"].astype(str), concept_frame["total_count"])
        axis.set_title(concept_frame["public_concept_label"].iloc[0])
        axis.set_xlabel("Calendar year")
    fig.suptitle("Annual Outcome Totals for Headline-Eligible Concepts")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_macro_region_temperature(path: Path, exposure_macro_region: pd.DataFrame, latest_year: int) -> None:
    frame = exposure_macro_region.loc[
        exposure_macro_region["calendar_year"].eq(latest_year)
        & exposure_macro_region["exposure_variable"].eq("maximum_daily_temperature_c")
    ].copy()
    frame = frame.sort_values("mean_value", ascending=False)
    fig, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
    axis.bar(frame["macro_region"], frame["mean_value"])
    axis.set_title(f"Mean Maximum Daily Temperature by Macro-Region ({latest_year})")
    axis.set_xlabel("Macro-region")
    axis.set_ylabel("Degrees Celsius")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _render_context_readiness_report(context_readiness: pd.DataFrame) -> str:
    lines = [
        "# Track A Context Readiness Report",
        "",
        "The municipality context asset is approved for descriptive, stratification, and heterogeneity use only. It is not an endpoint-certification asset.",
        "",
        "## Checks",
        "",
    ]
    for row in context_readiness.to_dict(orient="records"):
        lines.append(f"- {row['check_name']}: `{row['check_value']}`")
    return "\n".join(lines) + "\n"


def _render_descriptive_methods_note(outputs: dict[str, str]) -> str:
    return (
        "# Track A Descriptive Methods Note\n\n"
        "Stage 3 summarizes the frozen Track A marts without modifying canonical gold assets.\n\n"
        "- Exposure summaries are derived from `acute_daily_mart`, which provides the shared municipality-day climate grid.\n"
        "- Outcome summaries are concept-resolved to public-facing Track A concepts, so duplicated prompt-level proxy concepts are reported once.\n"
        "- Headline figures prioritize outcomes with `headline_eligible_flag=yes`; screening-only and descriptive-only concepts remain visible in tables.\n"
        "- Denominator-invalid rows are preserved and summarized explicitly in the denominator-validity table.\n"
        "- Delivery-aligned and event-aligned exposure windows remain descriptive operational summaries and are not interpreted as trimester or whole-pregnancy exposures.\n\n"
        "## Outputs\n\n"
        f"- Atlas directory: `{outputs['descriptive_atlas_directory']}`\n"
        f"- Summary workbook: `{outputs['descriptive_summary_tables']}`\n"
        f"- Context readiness report: `{outputs['context_readiness_report']}`\n"
    )


def build_track_a_descriptive_atlas(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    config = load_track_a_config(root, config_path=config_path)
    years = list(config["years"])
    readiness, concept_registry = _load_readiness_assets(root)

    descriptive_root = track_a_root(root) / "descriptive"
    atlas_dir = descriptive_root / "descriptive_atlas"
    descriptive_root.mkdir(parents=True, exist_ok=True)
    atlas_dir.mkdir(parents=True, exist_ok=True)

    exposure_annual = _compute_exposure_annual_summary(root, years)
    exposure_macro_region = _compute_exposure_macro_region_summary(root, years)
    municipal_exposure = _compute_municipal_exposure_summary(root, years)
    outcome_annual = _compute_outcome_annual_summary(root, years, readiness)
    denominator_validity = _compute_denominator_validity_summary(root, years)
    context_summary = _compute_context_summary(root)
    context_readiness = _compute_context_readiness_summary(root)
    headline_eligibility = readiness.loc[
        :,
        [
            "prompt_label",
            "public_concept_name",
            "public_concept_label",
            "stage1_classification",
            "headline_eligible_flag",
            "duplicate_prompt_concept_flag",
            "semantic_alignment",
        ],
    ].copy()

    tables = {
        "exposure_annual_summary": exposure_annual,
        "exposure_macro_region_summary": exposure_macro_region,
        "municipal_exposure_summary": municipal_exposure,
        "outcome_annual_summary": outcome_annual,
        "denominator_validity_summary": denominator_validity,
        "context_summary": context_summary,
        "headline_eligibility_summary": headline_eligibility,
        "public_concept_registry": concept_registry,
        "context_readiness_summary": context_readiness,
    }
    csv_outputs = _write_csv_outputs(atlas_dir, tables)

    figure_outputs = {
        "figure_01_exposure_annual_trends": atlas_dir / "figure_01_exposure_annual_trends.png",
        "figure_02_fallback_share_by_year": atlas_dir / "figure_02_fallback_share_by_year.png",
        "figure_03_headline_outcome_annual_totals": atlas_dir / "figure_03_headline_outcome_annual_totals.png",
        "figure_04_macro_region_temperature_latest_year": atlas_dir / "figure_04_macro_region_temperature_latest_year.png",
    }
    _plot_exposure_annual_trends(figure_outputs["figure_01_exposure_annual_trends"], exposure_annual)
    _plot_fallback_share(figure_outputs["figure_02_fallback_share_by_year"], exposure_annual)
    _plot_outcome_annual_totals(figure_outputs["figure_03_headline_outcome_annual_totals"], outcome_annual)
    _plot_macro_region_temperature(
        figure_outputs["figure_04_macro_region_temperature_latest_year"],
        exposure_macro_region,
        latest_year=max(years),
    )

    workbook_path = descriptive_root / "descriptive_summary_tables.xlsx"
    methods_note_path = descriptive_root / "descriptive_methods_note.md"
    context_readiness_report_path = descriptive_root / "context_readiness_report.md"
    _write_workbook(workbook_path, tables)

    outputs = {
        "descriptive_atlas_directory": _relative(root, atlas_dir),
        "descriptive_summary_tables": _relative(root, workbook_path),
        "descriptive_methods_note": _relative(root, methods_note_path),
        "context_readiness_report": _relative(root, context_readiness_report_path),
        "csv_outputs": {name: _relative(root, Path(path)) for name, path in csv_outputs.items()},
        "figure_outputs": {name: _relative(root, path) for name, path in figure_outputs.items()},
    }
    methods_note_path.write_text(_render_descriptive_methods_note(outputs), encoding="utf-8")
    context_readiness_report_path.write_text(
        _render_context_readiness_report(context_readiness),
        encoding="utf-8",
    )

    return {
        "status": "PASS",
        "stage": "stage3_descriptive_atlas",
        "analysis_id": config["analysis_id"],
        "analysis_title": config["analysis_title"],
        "outputs": outputs,
        "table_row_counts": {
            name: int(frame.shape[0]) for name, frame in tables.items()
        },
        "figure_count": len(figure_outputs),
    }
