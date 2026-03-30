from __future__ import annotations

import json
import math
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pandas as pd
import shapefile

from .dashboard_registry import ContextVariableDefinition, MetricDefinition, load_context_registry, load_metric_registry, load_page_registry


UF_CODE_TO_SIGLA = {
    "11": "RO",
    "12": "AC",
    "13": "AM",
    "14": "RR",
    "15": "PA",
    "16": "AP",
    "17": "TO",
    "21": "MA",
    "22": "PI",
    "23": "CE",
    "24": "RN",
    "25": "PB",
    "26": "PE",
    "27": "AL",
    "28": "SE",
    "29": "BA",
    "31": "MG",
    "32": "ES",
    "33": "RJ",
    "35": "SP",
    "41": "PR",
    "42": "SC",
    "43": "RS",
    "50": "MS",
    "51": "MT",
    "52": "GO",
    "53": "DF",
}

UF_TO_STATE_NAME = {
    "RO": "Rondonia",
    "AC": "Acre",
    "AM": "Amazonas",
    "RR": "Roraima",
    "PA": "Para",
    "AP": "Amapa",
    "TO": "Tocantins",
    "MA": "Maranhao",
    "PI": "Piaui",
    "CE": "Ceara",
    "RN": "Rio Grande do Norte",
    "PB": "Paraiba",
    "PE": "Pernambuco",
    "AL": "Alagoas",
    "SE": "Sergipe",
    "BA": "Bahia",
    "MG": "Minas Gerais",
    "ES": "Espirito Santo",
    "RJ": "Rio de Janeiro",
    "SP": "Sao Paulo",
    "PR": "Parana",
    "SC": "Santa Catarina",
    "RS": "Rio Grande do Sul",
    "MS": "Mato Grosso do Sul",
    "MT": "Mato Grosso",
    "GO": "Goias",
    "DF": "Distrito Federal",
}

MONTH_TO_SEASON = {
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
    12: "DJF",
}

ANALYTICAL_MUNICIPALITY_COUNT = 5570
EXCLUDED_NON_MUNICIPAL_GEOCODES = {"4300001", "4300002"}

DASHBOARD_METRICS_COLUMNS = [
    "page_id",
    "metric_id",
    "metric_family",
    "source_table",
    "definition_type",
    "geo_level",
    "uf",
    "municipality_code",
    "date_bucket",
    "year",
    "month",
    "epiweek",
    "season",
    "value_count",
    "value_rate",
    "denominator_value",
    "unit",
    "quality_flag_any",
    "quality_flag_detail",
    "context_join_year",
    "last_updated",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _path_for_manifest(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _gold_path(root: Path, dataset: str, year: int) -> Path:
    return root / "data" / "gold" / dataset / f"YEAR={year}" / f"{dataset}_{year}.parquet"


def _read_gold_years(root: Path, dataset: str, years: Iterable[int], columns: list[str] | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in years:
        path = _gold_path(root, dataset, year)
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=columns)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=columns or [])
    return pd.concat(frames, ignore_index=True)


def _normalize_municipality_code(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(7)


def _season_from_month(month: int | float | None) -> str | None:
    if month is None or pd.isna(month):
        return None
    return MONTH_TO_SEASON.get(int(month))


def _quantile_labels(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{index}" for index in range(1, count + 1)]


def _ranked_quantiles(series: pd.Series, quantiles: int, labels: list[str], missing_label: str) -> pd.Series:
    output = pd.Series([missing_label] * len(series), index=series.index, dtype="string")
    mask = series.notna()
    if mask.any():
        ranks = series[mask].rank(method="first")
        output.loc[mask] = pd.qcut(ranks, q=quantiles, labels=labels)
    return output


def _urbanicity_band(series: pd.Series, thresholds: tuple[float, ...], missing_label: str) -> pd.Series:
    output = pd.Series([missing_label] * len(series), index=series.index, dtype="string")
    mask = series.notna()
    if not mask.any():
        return output
    low_threshold, high_threshold = thresholds
    output.loc[mask & (series < low_threshold)] = "low"
    output.loc[mask & (series >= low_threshold) & (series < high_threshold)] = "medium"
    output.loc[mask & (series >= high_threshold)] = "high"
    return output


def _binary_flag(series: pd.Series, missing_label: str) -> pd.Series:
    output = pd.Series([missing_label] * len(series), index=series.index, dtype="string")
    mask = series.notna()
    output.loc[mask & (series.astype("Int64") == 1)] = "coastal"
    output.loc[mask & (series.astype("Int64") != 1)] = "non_coastal"
    return output


def _json_quality_summary_from_row(row: pd.Series, quality_flags: tuple[str, ...]) -> str:
    counts: dict[str, int] = {}
    for flag in quality_flags:
        value = row.get(flag, 0)
        if pd.notna(value) and int(value) > 0:
            counts[flag] = int(value)
    qc_value = row.get("__qc_non_pass", 0)
    if pd.notna(qc_value) and int(qc_value) > 0:
        counts["qc_status_non_pass_rows"] = int(qc_value)
    return json.dumps(counts, sort_keys=True)


def _as_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def _extract_geography_lookup_from_zip(source_zip: Path) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="observa_infancia_geo_") as tmpdir:
        with zipfile.ZipFile(source_zip) as archive:
            archive.extractall(tmpdir)
        base = Path(tmpdir) / "BR_Municipios_2022"
        reader = shapefile.Reader(str(base.with_suffix(".shp")))
        rows: list[dict[str, str]] = []
        for record in reader.records():
            municipality_code = str(record["CD_MUN"]).zfill(7)
            uf = str(record["SIGLA_UF"])
            rows.append(
                {
                    "municipality_code": municipality_code,
                    "municipality_name": str(record["NM_MUN"]),
                    "uf": uf,
                    "state_name": UF_TO_STATE_NAME[uf],
                }
            )
        frame = pd.DataFrame(rows).drop_duplicates(subset=["municipality_code"]).sort_values(["uf", "municipality_name"])
        return frame.reset_index(drop=True)


def build_analytical_municipality_spine(root: Path) -> set[str]:
    static_path = root / "data" / "gold" / "gold_municipality_context_static" / "gold_municipality_context_static.parquet"
    static_frame = pd.read_parquet(static_path, columns=["MUNICIPALITY_CODE_IBGE7"])
    municipality_codes = set(_normalize_municipality_code(static_frame["MUNICIPALITY_CODE_IBGE7"]).dropna().tolist())
    municipality_codes -= EXCLUDED_NON_MUNICIPAL_GEOCODES
    if len(municipality_codes) != ANALYTICAL_MUNICIPALITY_COUNT:
        raise ValueError(
            "Analytical municipality spine does not match the expected 5,570 municipalities: "
            f"observed {len(municipality_codes)} unique codes."
        )
    return municipality_codes


def build_dashboard_geographies(root: Path) -> pd.DataFrame:
    source_zip = root / "artifacts" / "reference" / "era5" / "BR_Municipios_2022.zip"
    analytical_spine = build_analytical_municipality_spine(root)
    lookup = _extract_geography_lookup_from_zip(source_zip)
    filtered = lookup.loc[lookup["municipality_code"].isin(analytical_spine)].copy()
    if filtered["municipality_code"].nunique() != len(analytical_spine):
        raise ValueError(
            "Dashboard geography lookup does not align with the analytical municipality spine after filtering."
        )
    return filtered.reset_index(drop=True)


def _build_context_frame(root: Path, years: list[int], definitions: list[ContextVariableDefinition]) -> tuple[pd.DataFrame, dict[str, Any]]:
    static_path = root / "data" / "gold" / "gold_municipality_context_static" / "gold_municipality_context_static.parquet"
    static_frame = pd.read_parquet(static_path)
    static_frame["municipality_code"] = _normalize_municipality_code(static_frame["MUNICIPALITY_CODE_IBGE7"])
    manifest_bins: dict[str, Any] = {}

    idh_labels = _quantile_labels("Q", 5)
    gini_labels = _quantile_labels("Q", 5)
    altitude_labels = _quantile_labels("Q", 4)
    static_frame["idh_quintile"] = _ranked_quantiles(static_frame["IDH_EXPANDED"], 5, idh_labels, "missing")
    static_frame["gini_quintile"] = _ranked_quantiles(static_frame["GINI_EXPANDED"], 5, gini_labels, "missing")
    static_frame["altitude_quartile"] = _ranked_quantiles(static_frame["ALTITUDE_MEAN_M"], 4, altitude_labels, "missing")
    static_frame["urbanicity_band"] = _urbanicity_band(static_frame["PROP_URBAN"], (0.33, 0.66), "missing")
    static_frame["coastal_flag"] = _binary_flag(static_frame["COASTAL_FLAG"], "missing")

    manifest_bins["idh_quintile"] = {"labels": idh_labels + ["missing"]}
    manifest_bins["gini_quintile"] = {"labels": gini_labels + ["missing"]}
    manifest_bins["altitude_quartile"] = {"labels": altitude_labels + ["missing"]}
    manifest_bins["urbanicity_band"] = {"labels": ["low", "medium", "high", "missing"], "thresholds": [0.33, 0.66]}
    manifest_bins["coastal_flag"] = {"labels": ["non_coastal", "coastal", "missing"]}

    keep_columns = [
        "municipality_code",
        "year",
        "IDH_EXPANDED",
        "GINI_EXPANDED",
        "PROP_URBAN",
        "ALTITUDE_MEAN_M",
        "COASTAL_FLAG",
        "idh_quintile",
        "gini_quintile",
        "urbanicity_band",
        "altitude_quartile",
        "coastal_flag",
        "JOIN_STATUS_FLAG",
        "MUNICIPALITY_CODE_IBGE7",
    ]

    frames: list[pd.DataFrame] = []
    for year in years:
        path = _gold_path(root, "gold_municipality_context", year)
        frame = pd.read_parquet(path)
        frame["municipality_code"] = _normalize_municipality_code(frame["MUNICIPALITY_CODE_IBGE7"])
        merged = frame.merge(
            static_frame[
                [
                    "municipality_code",
                    "idh_quintile",
                    "gini_quintile",
                    "urbanicity_band",
                    "altitude_quartile",
                    "coastal_flag",
                ]
            ],
            on="municipality_code",
            how="left",
        )
        merged["context_missing_flag"] = merged["JOIN_STATUS_FLAG"].fillna("").ne("OK").astype(int)
        merged["year"] = merged["year"].astype(int)
        frames.append(
            merged[
                [
                    "municipality_code",
                    "year",
                    "IDH_EXPANDED",
                    "GINI_EXPANDED",
                    "PROP_URBAN",
                    "ALTITUDE_MEAN_M",
                    "COASTAL_FLAG",
                    "idh_quintile",
                    "gini_quintile",
                    "urbanicity_band",
                    "altitude_quartile",
                    "coastal_flag",
                    "context_missing_flag",
                ]
            ].copy()
        )
    context_frame = pd.concat(frames, ignore_index=True).sort_values(["municipality_code", "year"]).reset_index(drop=True)
    return context_frame, manifest_bins


def _metric_source_columns(metric: MetricDefinition) -> list[str]:
    columns = ["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH", "QC_STATUS"]
    columns.append(metric.numerator_field)
    if metric.denominator_field:
        columns.append(metric.denominator_field)
    if metric.aggregation_weight_field:
        columns.append(metric.aggregation_weight_field)
    columns.extend(metric.quality_flags)
    deduped: list[str] = []
    for column in columns:
        if column and column not in deduped:
            deduped.append(column)
    return deduped


def _metrics_grouped_by_source(metrics: list[MetricDefinition]) -> dict[str, list[MetricDefinition]]:
    by_source: dict[str, list[MetricDefinition]] = {}
    for metric in metrics:
        by_source.setdefault(metric.source_table, []).append(metric)
    return by_source


def _empty_metrics_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=DASHBOARD_METRICS_COLUMNS)


def _build_period_frame(frame: pd.DataFrame, metric: MetricDefinition, last_updated: str) -> pd.DataFrame:
    if frame.empty:
        return _empty_metrics_frame()

    working = frame.copy()
    working["MUNICIPALITY_CODE"] = _normalize_municipality_code(working["MUNICIPALITY_CODE"])
    working["DATE"] = pd.to_datetime(working["DATE"])
    working["YEAR"] = working["YEAR"].astype(int)
    working["MONTH"] = working["MONTH"].astype(int)
    working["uf"] = working["MUNICIPALITY_CODE"].str[:2].map(UF_CODE_TO_SIGLA)
    working["season"] = working["MONTH"].map(MONTH_TO_SEASON)

    if metric.aggregation_method == "weighted_mean":
        if metric.aggregation_weight_field and metric.aggregation_weight_field in working.columns:
            weights = pd.to_numeric(working[metric.aggregation_weight_field], errors="coerce").fillna(0.0)
        else:
            weights = working[metric.numerator_field].notna().astype(float)
        values = pd.to_numeric(working[metric.numerator_field], errors="coerce")
        working["_value_component"] = values.fillna(0.0) * weights
        working["_denominator_component"] = weights
    else:
        working["_value_component"] = pd.to_numeric(working[metric.numerator_field], errors="coerce").fillna(0.0)
        if metric.denominator_field and metric.denominator_field in working.columns:
            working["_denominator_component"] = pd.to_numeric(working[metric.denominator_field], errors="coerce")
        else:
            working["_denominator_component"] = pd.NA
    working["__qc_non_pass"] = working.get("QC_STATUS", pd.Series(["PASS"] * len(working))).ne("PASS").astype(int)

    group_specs = [
        ("month", ["MUNICIPALITY_CODE", "uf", "YEAR", "MONTH", "season"]),
        ("season", ["MUNICIPALITY_CODE", "uf", "YEAR", "season"]),
        ("year", ["MUNICIPALITY_CODE", "uf", "YEAR"]),
    ]

    frames: list[pd.DataFrame] = []
    for granularity, keys in group_specs:
        agg_map: dict[str, str] = {"_value_component": "sum", "__qc_non_pass": "max"}
        if metric.aggregation_method == "weighted_mean":
            agg_map["_denominator_component"] = "sum"
        elif metric.denominator_field:
            agg_map["_denominator_component"] = metric.denominator_aggregation or "max"
        for flag in metric.quality_flags:
            if flag in working.columns:
                agg_map[flag] = "max"

        grouped = working.groupby(keys, dropna=False, sort=True, observed=True).agg(agg_map).reset_index()
        grouped["year"] = grouped["YEAR"].astype(int)
        grouped["month"] = grouped["MONTH"].astype("Int64") if "MONTH" in grouped.columns else pd.Series([pd.NA] * len(grouped), dtype="Int64")
        grouped["season_value"] = grouped["season"].astype("string") if "season" in grouped.columns else pd.Series([pd.NA] * len(grouped), dtype="string")
        grouped["denominator_value"] = grouped["_denominator_component"] if "_denominator_component" in grouped.columns else pd.NA
        grouped["value_rate"] = pd.NA
        if metric.denominator_field and metric.rate_multiplier:
            denominator = pd.to_numeric(grouped["denominator_value"], errors="coerce")
            numerator = pd.to_numeric(grouped["_value_component"], errors="coerce")
            valid = denominator.ge(metric.minimum_denominator_threshold) & denominator.gt(0)
            grouped.loc[valid, "value_rate"] = metric.rate_multiplier * numerator[valid] / denominator[valid]
        grouped["quality_flag_any"] = grouped[["__qc_non_pass", *[flag for flag in metric.quality_flags if flag in grouped.columns]]].max(axis=1).fillna(0).astype(int)
        grouped["quality_flag_detail"] = grouped.apply(lambda row: _json_quality_summary_from_row(row, metric.quality_flags), axis=1)

        def _bucket(row: pd.Series) -> str:
            if granularity == "month":
                return datetime(int(row["year"]), int(row["month"]), 1, tzinfo=timezone.utc).date().isoformat()
            if granularity == "season":
                season_start_month = {"DJF": 1, "MAM": 3, "JJA": 6, "SON": 9}[str(row["season_value"])]
                return datetime(int(row["year"]), season_start_month, 1, tzinfo=timezone.utc).date().isoformat()
            return datetime(int(row["year"]), 1, 1, tzinfo=timezone.utc).date().isoformat()

        grouped["date_bucket"] = grouped.apply(_bucket, axis=1)
        grouped["page_id"] = metric.page_id
        grouped["metric_id"] = metric.metric_id
        grouped["metric_family"] = metric.metric_family
        grouped["source_table"] = metric.source_table
        grouped["definition_type"] = metric.definition_type
        grouped["geo_level"] = "municipality"
        grouped["municipality_code"] = grouped["MUNICIPALITY_CODE"]
        grouped["context_join_year"] = grouped["year"]
        grouped["last_updated"] = last_updated
        grouped["unit"] = metric.display_unit
        grouped["epiweek"] = pd.NA
        grouped["value_count"] = pd.to_numeric(grouped["_value_component"], errors="coerce")

        frames.append(
            grouped[
                [
                    "page_id",
                    "metric_id",
                    "metric_family",
                    "source_table",
                    "definition_type",
                    "geo_level",
                    "uf",
                    "municipality_code",
                    "date_bucket",
                    "year",
                    "month",
                    "epiweek",
                    "season_value",
                    "value_count",
                    "value_rate",
                    "denominator_value",
                    "unit",
                    "quality_flag_any",
                    "quality_flag_detail",
                    "context_join_year",
                    "last_updated",
                ]
            ]
            .rename(columns={"season_value": "season"})
            .copy()
        )

    if not frames:
        return _empty_metrics_frame()

    metrics_frame = pd.concat(frames, ignore_index=True)
    return metrics_frame[DASHBOARD_METRICS_COLUMNS]


def build_dashboard_marts(
    root: Path,
    start_year: int,
    end_year: int,
    output_dir: Path,
    metric_ids: list[str] | None = None,
) -> dict[str, Any]:
    years = list(range(start_year, end_year + 1))
    _ensure_dir(output_dir)
    pages = load_page_registry(root)
    metrics = load_metric_registry(root)
    if metric_ids:
        requested = set(metric_ids)
        metrics = [metric for metric in metrics if metric.metric_id in requested]
    context_variables = load_context_registry(root)
    build_timestamp = _now_iso()

    geography_frame = build_dashboard_geographies(root)
    geography_path = output_dir / "dashboard_geographies.parquet"
    geography_frame.to_parquet(geography_path, index=False)

    context_frame, context_bins = _build_context_frame(root, years, context_variables)
    context_path = output_dir / "dashboard_context.parquet"
    context_frame.to_parquet(context_path, index=False)

    duckdb_path = output_dir / "observa_infancia.duckdb"
    if duckdb_path.exists():
        duckdb_path.unlink()

    connection = duckdb.connect(str(duckdb_path))
    connection.execute(
        """
        CREATE TABLE dashboard_metrics (
            page_id VARCHAR,
            metric_id VARCHAR,
            metric_family VARCHAR,
            source_table VARCHAR,
            definition_type VARCHAR,
            geo_level VARCHAR,
            uf VARCHAR,
            municipality_code VARCHAR,
            date_bucket VARCHAR,
            year INTEGER,
            month INTEGER,
            epiweek INTEGER,
            season VARCHAR,
            value_count DOUBLE,
            value_rate DOUBLE,
            denominator_value DOUBLE,
            unit VARCHAR,
            quality_flag_any INTEGER,
            quality_flag_detail VARCHAR,
            context_join_year INTEGER,
            last_updated VARCHAR
        )
        """
    )

    by_source = _metrics_grouped_by_source(metrics)
    try:
        for source_table, source_metrics in by_source.items():
            source_columns = sorted({column for metric in source_metrics for column in _metric_source_columns(metric)})
            source_frame = _read_gold_years(root, source_table, years, source_columns)
            for metric in source_metrics:
                metric_frame = _build_period_frame(source_frame, metric, build_timestamp)
                if metric_frame.empty:
                    continue
                connection.register("metric_frame_df", metric_frame)
                try:
                    connection.execute("INSERT INTO dashboard_metrics SELECT * FROM metric_frame_df")
                finally:
                    connection.unregister("metric_frame_df")
                del metric_frame
            del source_frame

        metrics_path = output_dir / "dashboard_metrics.parquet"
        connection.execute(
            f"""
            COPY (
                SELECT *
                FROM dashboard_metrics
                ORDER BY page_id, metric_id, municipality_code, year, month NULLS LAST
            )
            TO '{metrics_path.as_posix()}'
            (FORMAT PARQUET)
            """
        )

        profiles_path = output_dir / "municipality_profiles.parquet"
        connection.execute(
            """
            CREATE OR REPLACE TABLE municipality_profiles AS
            SELECT
                municipality_code,
                uf,
                year,
                page_id,
                metric_id,
                definition_type,
                value_count,
                value_rate,
                denominator_value,
                unit,
                quality_flag_any,
                quality_flag_detail,
                last_updated
            FROM dashboard_metrics
            WHERE month IS NULL AND season IS NULL
            ORDER BY municipality_code, page_id, metric_id, year
            """
        )
        connection.execute(f"COPY municipality_profiles TO '{profiles_path.as_posix()}' (FORMAT PARQUET)")

        connection.execute(f"CREATE OR REPLACE TABLE dashboard_context AS SELECT * FROM read_parquet('{context_path.as_posix()}')")
        connection.execute(f"CREATE OR REPLACE TABLE dashboard_geographies AS SELECT * FROM read_parquet('{geography_path.as_posix()}')")

        metric_rows = int(connection.execute("SELECT COUNT(*) FROM dashboard_metrics").fetchone()[0])

        manifest = {
            "dashboard_version": "v1",
            "dashboard_name": "Child Health Observatory (Observa Infancia)",
            "build_generated_at": build_timestamp,
            "years_covered": years,
            "granularities": ["month", "season", "year"],
            "geo_levels": ["national", "state", "municipality"],
            "pages": [page.to_dict() for page in pages],
            "metrics": [metric.to_dict() for metric in metrics],
            "context_variables": [definition.to_dict() for definition in context_variables],
            "context_bins": context_bins,
            "source_tables": sorted({metric.source_table for metric in metrics}),
            "artifacts": {
                "dashboard_metrics": _path_for_manifest(root, metrics_path),
                "dashboard_context": _path_for_manifest(root, context_path),
                "municipality_profiles": _path_for_manifest(root, profiles_path),
                "dashboard_geographies": _path_for_manifest(root, geography_path),
                "observa_infancia_duckdb": _path_for_manifest(root, duckdb_path),
            },
            "coverage": {
                "municipalities": int(geography_frame["municipality_code"].nunique()),
                "states": int(geography_frame["uf"].nunique()),
                "metric_rows": metric_rows,
            },
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest
    finally:
        connection.close()
