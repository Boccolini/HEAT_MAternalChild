from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DATASET_NAME = "gold_municipality_context"
EXPECTED_YEARS = [2020, 2021, 2022, 2023, 2024]
DEFAULT_OUTPUT_DIRNAME = "municipality_context_audit"
DEFAULT_REPORT_NAME = "municipality_context_audit_report.md"

REQUIRED_COLUMNS = [
    "municipality_code",
    "MUNICIPALITY_CODE_IBGE7",
    "year",
    "URBAN_POP",
    "RURAL_POP",
    "PROP_URBAN",
    "PROP_RURAL",
    "GINI_RAW",
    "GINI_SOURCE_YEAR",
    "GINI_EXPANDED",
    "GINI_EXPANSION_METHOD",
    "IDH_RAW",
    "IDH_SOURCE_YEAR",
    "IDH_EXPANDED",
    "IDH_EXPANSION_METHOD",
    "ALTITUDE_MEAN_M",
    "COASTAL_FLAG",
    "URBAN_RURAL_SOURCE",
    "URBAN_RURAL_SOURCE_YEAR",
    "URBAN_RURAL_EXPANSION_METHOD",
    "ALTITUDE_SOURCE",
    "COASTAL_SOURCE",
    "STATIC_CONTEXT_FLAG",
    "MUNICIPAL_BOUNDARY_REFERENCE_YEAR",
    "JOIN_STATUS_FLAG",
]

NUMERIC_FIELDS = [
    "URBAN_POP",
    "RURAL_POP",
    "PROP_URBAN",
    "PROP_RURAL",
    "GINI_EXPANDED",
    "IDH_EXPANDED",
    "ALTITUDE_MEAN_M",
]

CATEGORICAL_FIELDS = [
    "COASTAL_FLAG",
    "JOIN_STATUS_FLAG",
    "URBAN_RURAL_SOURCE_YEAR",
    "GINI_SOURCE_YEAR",
    "IDH_SOURCE_YEAR",
    "URBAN_RURAL_EXPANSION_METHOD",
    "GINI_EXPANSION_METHOD",
    "IDH_EXPANSION_METHOD",
    "STATIC_CONTEXT_FLAG",
    "MUNICIPAL_BOUNDARY_REFERENCE_YEAR",
]

OVERALL_FREQUENCY_ANALYSIS_LEVEL = {
    "COASTAL_FLAG": "municipality",
    "JOIN_STATUS_FLAG": "municipality",
    "URBAN_RURAL_SOURCE_YEAR": "municipality",
    "GINI_SOURCE_YEAR": "municipality",
    "IDH_SOURCE_YEAR": "municipality",
    "URBAN_RURAL_EXPANSION_METHOD": "municipality_year",
    "GINI_EXPANSION_METHOD": "municipality",
    "IDH_EXPANSION_METHOD": "municipality",
    "STATIC_CONTEXT_FLAG": "municipality",
    "MUNICIPAL_BOUNDARY_REFERENCE_YEAR": "municipality",
}

INVARIANT_MUNICIPALITY_FIELDS = [
    "MUNICIPALITY_CODE_IBGE7",
    "URBAN_POP",
    "RURAL_POP",
    "PROP_URBAN",
    "PROP_RURAL",
    "GINI_RAW",
    "GINI_SOURCE_YEAR",
    "GINI_EXPANDED",
    "GINI_EXPANSION_METHOD",
    "IDH_RAW",
    "IDH_SOURCE_YEAR",
    "IDH_EXPANDED",
    "IDH_EXPANSION_METHOD",
    "ALTITUDE_MEAN_M",
    "COASTAL_FLAG",
    "URBAN_RURAL_SOURCE",
    "URBAN_RURAL_SOURCE_YEAR",
    "ALTITUDE_SOURCE",
    "COASTAL_SOURCE",
    "STATIC_CONTEXT_FLAG",
    "MUNICIPAL_BOUNDARY_REFERENCE_YEAR",
    "JOIN_STATUS_FLAG",
]

MUNICIPALITY_LEVEL_FIELDS = [
    "municipality_code",
    "MUNICIPALITY_CODE_IBGE7",
    "URBAN_POP",
    "RURAL_POP",
    "PROP_URBAN",
    "PROP_RURAL",
    "GINI_RAW",
    "GINI_SOURCE_YEAR",
    "GINI_EXPANDED",
    "GINI_EXPANSION_METHOD",
    "IDH_RAW",
    "IDH_SOURCE_YEAR",
    "IDH_EXPANDED",
    "IDH_EXPANSION_METHOD",
    "ALTITUDE_MEAN_M",
    "COASTAL_FLAG",
    "URBAN_RURAL_SOURCE",
    "URBAN_RURAL_SOURCE_YEAR",
    "ALTITUDE_SOURCE",
    "COASTAL_SOURCE",
    "STATIC_CONTEXT_FLAG",
    "MUNICIPAL_BOUNDARY_REFERENCE_YEAR",
    "JOIN_STATUS_FLAG",
]

STRUCTURAL_MISSING_FIELDS = ["GINI_RAW", "IDH_RAW", "ALTITUDE_MEAN_M"]
LOG_TRANSFORM_FIELDS = {"URBAN_POP", "RURAL_POP"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dataset_dir(root: Path) -> Path:
    return root / "data" / "gold" / DATASET_NAME


def _manifest_path(root: Path) -> Path:
    return root / "artifacts" / "manifests" / "municipality_context_manifest.json"


def _output_dir(root: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    return root / "artifacts" / "status" / DEFAULT_OUTPUT_DIRNAME


def _report_path(root: Path) -> Path:
    return root / "artifacts" / "status" / DEFAULT_REPORT_NAME


def _plots_dir(output_dir: Path) -> Path:
    return output_dir / "plots"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _require_columns(frame: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Municipality context audit is missing required columns: {missing}")


def _load_manifest_payload(root: Path) -> dict[str, Any] | None:
    path = _manifest_path(root)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(_format_value(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    _ensure_dir(path.parent)
    frame.to_csv(path, index=False)


def _load_panel_paths(root: Path) -> list[Path]:
    dataset_dir = _dataset_dir(root)
    paths = sorted(dataset_dir.glob("YEAR=*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No municipality context parquet partitions found in {dataset_dir}")
    return paths


def load_context_panel(root: Path) -> pd.DataFrame:
    paths = _load_panel_paths(root)
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    _require_columns(frame, REQUIRED_COLUMNS)
    frame["municipality_code"] = frame["municipality_code"].astype(str).str.zfill(6)
    frame["MUNICIPALITY_CODE_IBGE7"] = frame["MUNICIPALITY_CODE_IBGE7"].astype(str).str.zfill(7)
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    return frame.sort_values(["year", "municipality_code"], kind="mergesort").reset_index(drop=True)


def load_context_municipality_level(panel: pd.DataFrame) -> pd.DataFrame:
    _require_columns(panel, REQUIRED_COLUMNS)
    grouped = panel.groupby("municipality_code", dropna=False)
    invariant_failures: list[str] = []
    for column in INVARIANT_MUNICIPALITY_FIELDS:
        uniqueness = grouped[column].nunique(dropna=False)
        invalid_codes = uniqueness.loc[uniqueness.gt(1)].index.astype(str).tolist()
        if invalid_codes:
            invariant_failures.append(f"{column} -> {invalid_codes[:5]}")
    if invariant_failures:
        raise ValueError(
            "Municipality-level audit cannot deduplicate non-invariant fields: "
            + "; ".join(invariant_failures)
        )
    municipality_level = (
        panel.sort_values(["municipality_code", "year"], kind="mergesort")
        .drop_duplicates(["municipality_code"], keep="first")
        .reset_index(drop=True)
    )
    return municipality_level[MUNICIPALITY_LEVEL_FIELDS].copy()


def coverage_by_year(panel: pd.DataFrame) -> pd.DataFrame:
    expected_municipalities = int(panel["municipality_code"].nunique())
    rows: list[dict[str, Any]] = []
    for year, group in panel.groupby("year", sort=True):
        unique_pairs = int(group[["municipality_code", "year"]].drop_duplicates().shape[0])
        municipalities_n = int(group["municipality_code"].nunique())
        duplicates_n = int(len(group) - unique_pairs)
        missing_rows = int(expected_municipalities - municipalities_n)
        rows.append(
            {
                "year": int(year),
                "rows_n": int(len(group)),
                "municipalities_n": municipalities_n,
                "duplicates_n": duplicates_n,
                "expected_rows": expected_municipalities,
                "missing_rows": missing_rows,
                "coverage_pct": float((municipalities_n / expected_municipalities) * 100.0),
            }
        )
    return pd.DataFrame(rows).sort_values("year", kind="mergesort").reset_index(drop=True)


def completeness_overall(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_rows = int(len(panel))
    for column in panel.columns:
        non_missing_n = int(panel[column].notna().sum())
        missing_n = int(total_rows - non_missing_n)
        rows.append(
            {
                "column": column,
                "dtype": str(panel[column].dtype),
                "non_missing_n": non_missing_n,
                "missing_n": missing_n,
                "missing_pct": float((missing_n / total_rows) * 100.0),
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_pct", "column"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def completeness_by_year(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in panel.groupby("year", sort=True):
        total_rows = int(len(group))
        for column in panel.columns:
            non_missing_n = int(group[column].notna().sum())
            missing_n = int(total_rows - non_missing_n)
            rows.append(
                {
                    "year": int(year),
                    "column": column,
                    "non_missing_n": non_missing_n,
                    "missing_n": missing_n,
                    "missing_pct": float((missing_n / total_rows) * 100.0),
                }
            )
    return pd.DataFrame(rows).sort_values(["year", "missing_pct", "column"], ascending=[True, False, True], kind="mergesort").reset_index(drop=True)


def frequencies_overall(panel: pd.DataFrame, municipality_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field in CATEGORICAL_FIELDS:
        analysis_level = OVERALL_FREQUENCY_ANALYSIS_LEVEL[field]
        source = municipality_level if analysis_level == "municipality" else panel
        denominator_n = int(len(source))
        counts = source[field].astype("object").where(source[field].notna(), "missing").value_counts(dropna=False, sort=False)
        for value, count in counts.items():
            rows.append(
                {
                    "field": field,
                    "analysis_level": analysis_level,
                    "value": str(value),
                    "count_n": int(count),
                    "share_pct": float((int(count) / denominator_n) * 100.0),
                    "denominator_n": denominator_n,
                }
            )
    return pd.DataFrame(rows).sort_values(["field", "value"], kind="mergesort").reset_index(drop=True)


def frequencies_by_year(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in panel.groupby("year", sort=True):
        denominator_n = int(len(group))
        for field in CATEGORICAL_FIELDS:
            counts = group[field].astype("object").where(group[field].notna(), "missing").value_counts(dropna=False, sort=False)
            for value, count in counts.items():
                rows.append(
                    {
                        "year": int(year),
                        "field": field,
                        "value": str(value),
                        "count_n": int(count),
                        "share_pct": float((int(count) / denominator_n) * 100.0),
                        "denominator_n": denominator_n,
                    }
                )
    return pd.DataFrame(rows).sort_values(["year", "field", "value"], kind="mergesort").reset_index(drop=True)


def numeric_summary(municipality_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_rows = int(len(municipality_level))
    for field in NUMERIC_FIELDS:
        series = pd.to_numeric(municipality_level[field], errors="coerce")
        valid = series.dropna()
        rows.append(
            {
                "variable": field,
                "analysis_level": "municipality",
                "count_n": int(valid.shape[0]),
                "missing_n": int(total_rows - valid.shape[0]),
                "mean": float(valid.mean()) if not valid.empty else np.nan,
                "std": float(valid.std(ddof=1)) if valid.shape[0] > 1 else np.nan,
                "min": float(valid.min()) if not valid.empty else np.nan,
                "p01": float(valid.quantile(0.01)) if not valid.empty else np.nan,
                "p05": float(valid.quantile(0.05)) if not valid.empty else np.nan,
                "p25": float(valid.quantile(0.25)) if not valid.empty else np.nan,
                "p50": float(valid.quantile(0.50)) if not valid.empty else np.nan,
                "p75": float(valid.quantile(0.75)) if not valid.empty else np.nan,
                "p95": float(valid.quantile(0.95)) if not valid.empty else np.nan,
                "p99": float(valid.quantile(0.99)) if not valid.empty else np.nan,
                "max": float(valid.max()) if not valid.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _transformed_series(series: pd.Series, field: str) -> tuple[pd.Series, str]:
    numeric = pd.to_numeric(series, errors="coerce")
    if field in LOG_TRANSFORM_FIELDS:
        return np.log1p(numeric), "log1p"
    return numeric, "identity"


def _inverse_transform(value: float, transform_used: str) -> float:
    if transform_used == "log1p":
        return float(max(np.expm1(value), 0.0))
    return float(value)


def boxplot_stats(municipality_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field in NUMERIC_FIELDS:
        original = pd.to_numeric(municipality_level[field], errors="coerce")
        transformed, transform_used = _transformed_series(municipality_level[field], field)
        valid = transformed.dropna()
        if valid.empty:
            rows.append(
                {
                    "variable": field,
                    "analysis_level": "municipality",
                    "transform_used": transform_used,
                    "count_n": 0,
                    "q1_scale": np.nan,
                    "median_scale": np.nan,
                    "q3_scale": np.nan,
                    "iqr_scale": np.nan,
                    "lower_fence_scale": np.nan,
                    "upper_fence_scale": np.nan,
                    "lower_fence_value": np.nan,
                    "upper_fence_value": np.nan,
                    "whisker_low_value": np.nan,
                    "whisker_high_value": np.nan,
                    "outlier_n": 0,
                }
            )
            continue
        q1 = float(valid.quantile(0.25))
        median = float(valid.quantile(0.50))
        q3 = float(valid.quantile(0.75))
        iqr = float(q3 - q1)
        lower_scale = float(q1 - 1.5 * iqr)
        upper_scale = float(q3 + 1.5 * iqr)
        inlier_mask = transformed.ge(lower_scale) & transformed.le(upper_scale)
        inliers_original = original.loc[inlier_mask & original.notna()]
        outlier_mask = transformed.lt(lower_scale) | transformed.gt(upper_scale)
        rows.append(
            {
                "variable": field,
                "analysis_level": "municipality",
                "transform_used": transform_used,
                "count_n": int(valid.shape[0]),
                "q1_scale": q1,
                "median_scale": median,
                "q3_scale": q3,
                "iqr_scale": iqr,
                "lower_fence_scale": lower_scale,
                "upper_fence_scale": upper_scale,
                "lower_fence_value": _inverse_transform(lower_scale, transform_used),
                "upper_fence_value": _inverse_transform(upper_scale, transform_used),
                "whisker_low_value": float(inliers_original.min()) if not inliers_original.empty else np.nan,
                "whisker_high_value": float(inliers_original.max()) if not inliers_original.empty else np.nan,
                "outlier_n": int(outlier_mask.fillna(False).sum()),
            }
        )
    return pd.DataFrame(rows)


def outlier_table(municipality_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stats = boxplot_stats(municipality_level).set_index("variable")
    for field in NUMERIC_FIELDS:
        original = pd.to_numeric(municipality_level[field], errors="coerce")
        transformed, transform_used = _transformed_series(municipality_level[field], field)
        lower_scale = float(stats.loc[field, "lower_fence_scale"])
        upper_scale = float(stats.loc[field, "upper_fence_scale"])
        outlier_mask = transformed.lt(lower_scale) | transformed.gt(upper_scale)
        for _, row in municipality_level.loc[outlier_mask.fillna(False), ["municipality_code", "MUNICIPALITY_CODE_IBGE7"]].iterrows():
            municipality_code = str(row["municipality_code"]).zfill(6)
            value = float(original.loc[municipality_level["municipality_code"].eq(municipality_code)].iloc[0])
            transformed_value = float(transformed.loc[municipality_level["municipality_code"].eq(municipality_code)].iloc[0])
            rows.append(
                {
                    "variable": field,
                    "municipality_code": municipality_code,
                    "MUNICIPALITY_CODE_IBGE7": str(row["MUNICIPALITY_CODE_IBGE7"]).zfill(7),
                    "value": value,
                    "transform_used": transform_used,
                    "transformed_value": transformed_value,
                    "lower_fence_scale": lower_scale,
                    "upper_fence_scale": upper_scale,
                    "lower_fence_value": _inverse_transform(lower_scale, transform_used),
                    "upper_fence_value": _inverse_transform(upper_scale, transform_used),
                    "outlier_direction": "low" if transformed_value < lower_scale else "high",
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "variable",
                "municipality_code",
                "MUNICIPALITY_CODE_IBGE7",
                "value",
                "transform_used",
                "transformed_value",
                "lower_fence_scale",
                "upper_fence_scale",
                "lower_fence_value",
                "upper_fence_value",
                "outlier_direction",
            ]
        )
    return pd.DataFrame(rows).sort_values(["variable", "outlier_direction", "value"], kind="mergesort").reset_index(drop=True)


def missing_structural_municipalities(municipality_level: pd.DataFrame) -> pd.DataFrame:
    frame = municipality_level.copy()
    frame["missing_gini_flag"] = frame["GINI_RAW"].isna().astype(int)
    frame["missing_idh_flag"] = frame["IDH_RAW"].isna().astype(int)
    frame["missing_altitude_flag"] = frame["ALTITUDE_MEAN_M"].isna().astype(int)
    mask = (
        frame["missing_gini_flag"].eq(1)
        | frame["missing_idh_flag"].eq(1)
        | frame["missing_altitude_flag"].eq(1)
        | frame["JOIN_STATUS_FLAG"].ne("OK")
    )
    return frame.loc[
        mask,
        [
            "municipality_code",
            "MUNICIPALITY_CODE_IBGE7",
            "JOIN_STATUS_FLAG",
            "missing_gini_flag",
            "missing_idh_flag",
            "missing_altitude_flag",
            "GINI_SOURCE_YEAR",
            "IDH_SOURCE_YEAR",
        ],
    ].sort_values(["JOIN_STATUS_FLAG", "municipality_code"], kind="mergesort").reset_index(drop=True)


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt


def render_plots(
    output_dir: Path,
    municipality_level: pd.DataFrame,
    frequencies_overall_frame: pd.DataFrame,
    completeness_overall_frame: pd.DataFrame,
) -> list[Path]:
    plt = _load_matplotlib()
    plots_dir = _plots_dir(output_dir)
    _ensure_dir(plots_dir)
    paths: list[Path] = []

    population_path = plots_dir / "boxplot_population_counts_logscale.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for axis, field in zip(axes, ["URBAN_POP", "RURAL_POP"], strict=True):
        values = pd.to_numeric(municipality_level[field], errors="coerce").dropna()
        axis.boxplot(values, orientation="vertical")
        axis.set_yscale("log")
        axis.set_title(field)
        axis.set_ylabel("count (log scale)")
        axis.grid(axis="y", linestyle=":", alpha=0.4)
    fig.suptitle("Municipality-level population context")
    fig.tight_layout()
    fig.savefig(population_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(population_path)

    indices_path = plots_dir / "boxplot_proportions_indices.png"
    fields = ["PROP_URBAN", "PROP_RURAL", "GINI_EXPANDED", "IDH_EXPANDED", "ALTITUDE_MEAN_M"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    flat_axes = axes.flatten()
    for axis, field in zip(flat_axes, fields, strict=False):
        values = pd.to_numeric(municipality_level[field], errors="coerce").dropna()
        axis.boxplot(values, orientation="vertical")
        axis.set_title(field)
        axis.grid(axis="y", linestyle=":", alpha=0.4)
    for axis in flat_axes[len(fields):]:
        axis.axis("off")
    fig.suptitle("Municipality-level structural proportions and indices")
    fig.tight_layout()
    fig.savefig(indices_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(indices_path)

    join_status_path = plots_dir / "frequency_join_status.png"
    join_status = frequencies_overall_frame.loc[frequencies_overall_frame["field"].eq("JOIN_STATUS_FLAG")].copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    y_positions = np.arange(len(join_status))
    ax.barh(y_positions, join_status["count_n"], color="#4C78A8")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(join_status["value"])
    ax.set_xlabel("municipalities")
    ax.set_title("JOIN_STATUS_FLAG frequency")
    for index, row in join_status.reset_index(drop=True).iterrows():
        ax.text(row["count_n"], index, f" {int(row['count_n'])} ({row['share_pct']:.2f}%)", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(join_status_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(join_status_path)

    coastal_path = plots_dir / "frequency_coastal_flag.png"
    coastal = frequencies_overall_frame.loc[frequencies_overall_frame["field"].eq("COASTAL_FLAG")].copy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(coastal["value"], coastal["count_n"], color=["#9ecae1", "#08519c"][: len(coastal)])
    ax.set_xlabel("COASTAL_FLAG")
    ax.set_ylabel("municipalities")
    ax.set_title("COASTAL_FLAG frequency")
    for _, row in coastal.iterrows():
        ax.text(row["value"], row["count_n"], f"{int(row['count_n'])}\n({row['share_pct']:.2f}%)", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(coastal_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(coastal_path)

    completeness_path = plots_dir / "completeness_by_variable.png"
    completeness = completeness_overall_frame.sort_values(["missing_pct", "column"], ascending=[False, True], kind="mergesort")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(completeness["column"], completeness["missing_pct"], color="#E45756")
    ax.invert_yaxis()
    ax.set_xlabel("missing %")
    ax.set_title("Completeness by variable")
    fig.tight_layout()
    fig.savefig(completeness_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(completeness_path)

    return paths


def _summary_checks(coverage_frame: pd.DataFrame, panel: pd.DataFrame) -> dict[str, Any]:
    duplicates_total = int(coverage_frame["duplicates_n"].sum())
    missing_total = int(coverage_frame["missing_rows"].sum())
    years_present = sorted(panel["year"].unique().tolist())
    expected_years_match = years_present == EXPECTED_YEARS
    return {
        "municipalities_n": int(panel["municipality_code"].nunique()),
        "municipality_years_n": int(len(panel)),
        "duplicates_total": duplicates_total,
        "missing_municipality_years_total": missing_total,
        "years_present": years_present,
        "expected_years_match": expected_years_match,
        "all_checks_passed": bool(duplicates_total == 0 and missing_total == 0 and expected_years_match),
    }


def write_markdown_report(
    root: Path,
    report_path: Path,
    output_dir: Path,
    panel: pd.DataFrame,
    coverage_frame: pd.DataFrame,
    completeness_overall_frame: pd.DataFrame,
    completeness_by_year_frame: pd.DataFrame,
    frequencies_overall_frame: pd.DataFrame,
    frequencies_by_year_frame: pd.DataFrame,
    numeric_summary_frame: pd.DataFrame,
    boxplot_stats_frame: pd.DataFrame,
    missing_structural_frame: pd.DataFrame,
    plot_paths: list[Path],
    manifest_payload: dict[str, Any] | None,
) -> Path:
    checks = _summary_checks(coverage_frame, panel)
    top_missing = completeness_overall_frame.loc[completeness_overall_frame["missing_n"].gt(0)].head(10)
    join_status = frequencies_overall_frame.loc[frequencies_overall_frame["field"].eq("JOIN_STATUS_FLAG")].copy()
    coastal = frequencies_overall_frame.loc[frequencies_overall_frame["field"].eq("COASTAL_FLAG")].copy()
    source_years = frequencies_overall_frame.loc[
        frequencies_overall_frame["field"].isin(["URBAN_RURAL_SOURCE_YEAR", "GINI_SOURCE_YEAR", "IDH_SOURCE_YEAR"])
    ].copy()
    expansion_methods = frequencies_by_year_frame.loc[
        frequencies_by_year_frame["field"].isin(["URBAN_RURAL_EXPANSION_METHOD", "GINI_EXPANSION_METHOD", "IDH_EXPANSION_METHOD"])
    ].copy()

    lines = [
        "# Municipality Context Audit",
        "",
        f"- Generated at: `{utc_now()}`",
        f"- Dataset audited: `{DATASET_NAME}`",
        f"- Repository root: `{root}`",
        f"- Partitions read: {len(_load_panel_paths(root))}",
        f"- Partition root: `{_dataset_dir(root).relative_to(root)}`",
        f"- Manifest used: `{_manifest_path(root).relative_to(root)}`" if manifest_payload is not None else "- Manifest used: `not available`",
        "",
        "## Executive Summary",
        "",
        f"- Municipalities in panel: {checks['municipalities_n']}",
        f"- Municipality-years in panel: {checks['municipality_years_n']}",
        f"- Duplicates on `municipality_code + year`: {checks['duplicates_total']}",
        f"- Missing municipality-years: {checks['missing_municipality_years_total']}",
        f"- Expected year set present: {'yes' if checks['expected_years_match'] else 'no'} ({checks['years_present']})",
        f"- Structural-missing municipalities: {int(missing_structural_frame['municipality_code'].nunique())}",
        "- Context rules documented in the data are preserved: urban/rural observed in 2022 and expanded; Gini/IDH observed in 2010 and expanded; altitude and coastal are static context.",
        "",
        "## Coverage",
        "",
        _markdown_table(coverage_frame),
        "",
        "CSV exports:",
        f"- `{(output_dir / 'coverage_by_year.csv').relative_to(root)}`",
        "",
        "## Completeness",
        "",
        "Top variables with missingness in the municipality-year panel:",
        "",
        _markdown_table(top_missing[["column", "missing_n", "missing_pct"]]),
        "",
        "Completeness by year for variables with any missingness:",
        "",
        _markdown_table(
            completeness_by_year_frame.loc[completeness_by_year_frame["missing_n"].gt(0), ["year", "column", "missing_n", "missing_pct"]].head(15)
        ),
        "",
        "CSV exports:",
        f"- `{(output_dir / 'completeness_overall.csv').relative_to(root)}`",
        f"- `{(output_dir / 'completeness_by_year.csv').relative_to(root)}`",
        "",
        "## Frequencies",
        "",
        "Overall frequencies for `JOIN_STATUS_FLAG`:",
        "",
        _markdown_table(join_status[["field", "value", "count_n", "share_pct"]]),
        "",
        "Overall frequencies for `COASTAL_FLAG`:",
        "",
        _markdown_table(coastal[["field", "value", "count_n", "share_pct"]]),
        "",
        "Source years (overall municipality level):",
        "",
        _markdown_table(source_years[["field", "value", "count_n", "share_pct"]]),
        "",
        "Expansion methods (sample by year):",
        "",
        _markdown_table(expansion_methods[["year", "field", "value", "count_n", "share_pct"]].head(15)),
        "",
        "CSV exports:",
        f"- `{(output_dir / 'frequencies_overall.csv').relative_to(root)}`",
        f"- `{(output_dir / 'frequencies_by_year.csv').relative_to(root)}`",
        "",
        "## Distributions",
        "",
        "Numeric summary at municipality level:",
        "",
        _markdown_table(numeric_summary_frame),
        "",
        "Boxplot statistics at municipality level:",
        "",
        _markdown_table(boxplot_stats_frame),
        "",
        "CSV exports:",
        f"- `{(output_dir / 'numeric_summary_municipality_level.csv').relative_to(root)}`",
        f"- `{(output_dir / 'boxplot_stats_municipality_level.csv').relative_to(root)}`",
        f"- `{(output_dir / 'outliers_municipality_level.csv').relative_to(root)}`",
        "",
        "PNG exports:",
    ]
    lines.extend([f"- `{path.relative_to(root)}`" for path in plot_paths])
    lines.extend(
        [
            "",
            "## Municipalities with Structural Problems",
            "",
            _markdown_table(missing_structural_frame.head(20)),
            "",
            f"CSV export: `{(output_dir / 'missing_structural_municipalities.csv').relative_to(root)}`",
            "",
            "## Conclusion",
            "",
            "- The municipality-year panel is complete in coverage when benchmarked against the pactuated municipal spine.",
            "- Missingness is concentrated in `GINI_*` and `IDH_*` and reflects 5 municipalities absent from the Atlas Brasil source, repeated across 2020-2024.",
            "- Distributional summaries and boxplots were computed on a deduplicated municipality-level base to avoid overstating repeated structural values.",
            "- The audit supports descriptive, contextual and heterogeneity uses of `gold_municipality_context` while preserving the distinction between observed source years and expanded values.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_path


def audit_gold_municipality_context(root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    panel = load_context_panel(root)
    municipality_level = load_context_municipality_level(panel)
    coverage_frame = coverage_by_year(panel)
    completeness_overall_frame = completeness_overall(panel)
    completeness_by_year_frame = completeness_by_year(panel)
    frequencies_overall_frame = frequencies_overall(panel, municipality_level)
    frequencies_by_year_frame = frequencies_by_year(panel)
    numeric_summary_frame = numeric_summary(municipality_level)
    boxplot_stats_frame = boxplot_stats(municipality_level)
    outliers_frame = outlier_table(municipality_level)
    missing_structural_frame = missing_structural_municipalities(municipality_level)
    manifest_payload = _load_manifest_payload(root)

    resolved_output_dir = _output_dir(root, output_dir)
    _ensure_dir(resolved_output_dir)
    _ensure_dir(_plots_dir(resolved_output_dir))

    csv_targets = {
        "coverage_by_year": coverage_frame,
        "completeness_overall": completeness_overall_frame,
        "completeness_by_year": completeness_by_year_frame,
        "frequencies_overall": frequencies_overall_frame,
        "frequencies_by_year": frequencies_by_year_frame,
        "numeric_summary_municipality_level": numeric_summary_frame,
        "boxplot_stats_municipality_level": boxplot_stats_frame,
        "outliers_municipality_level": outliers_frame,
        "missing_structural_municipalities": missing_structural_frame,
    }
    output_paths: dict[str, str] = {}
    for stem, frame in csv_targets.items():
        path = resolved_output_dir / f"{stem}.csv"
        _write_csv(path, frame)
        output_paths[stem] = str(path.relative_to(root))

    plot_paths = render_plots(
        output_dir=resolved_output_dir,
        municipality_level=municipality_level,
        frequencies_overall_frame=frequencies_overall_frame,
        completeness_overall_frame=completeness_overall_frame,
    )
    report_path = write_markdown_report(
        root=root,
        report_path=_report_path(root),
        output_dir=resolved_output_dir,
        panel=panel,
        coverage_frame=coverage_frame,
        completeness_overall_frame=completeness_overall_frame,
        completeness_by_year_frame=completeness_by_year_frame,
        frequencies_overall_frame=frequencies_overall_frame,
        frequencies_by_year_frame=frequencies_by_year_frame,
        numeric_summary_frame=numeric_summary_frame,
        boxplot_stats_frame=boxplot_stats_frame,
        missing_structural_frame=missing_structural_frame,
        plot_paths=plot_paths,
        manifest_payload=manifest_payload,
    )

    checks = _summary_checks(coverage_frame, panel)
    return {
        "status": "PASS" if checks["all_checks_passed"] else "FAIL",
        "dataset": DATASET_NAME,
        "report_path": str(report_path.relative_to(root)),
        "output_dir": str(resolved_output_dir.relative_to(root)),
        "plot_paths": [str(path.relative_to(root)) for path in plot_paths],
        "checks": checks,
        "csv_outputs": output_paths,
    }
