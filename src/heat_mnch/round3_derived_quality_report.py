from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .round3_gold_report import (
    PLOT_SAMPLE_ROWS,
    GoldTableSpec,
    _build_catalog,
    _chunks,
    _quote_ident,
    _relation_sql,
    _table_pages,
    _table_specs,
    _utc_now,
)


@dataclass(frozen=True)
class DerivedQualityArtifacts:
    output_dir: Path
    tables_dir: Path
    pdf_path: Path
    manifest_path: Path
    missingness_path: Path
    outliers_path: Path


def _artifact_paths(root: Path, output_dir: Path | None = None) -> DerivedQualityArtifacts:
    report_dir = output_dir or root / "reports/round3"
    tables_dir = report_dir / "tables"
    return DerivedQualityArtifacts(
        output_dir=report_dir,
        tables_dir=tables_dir,
        pdf_path=report_dir / "round3_derived_missing_outliers.pdf",
        manifest_path=report_dir / "round3_derived_missing_outliers_manifest.json",
        missingness_path=tables_dir / "round3_derived_missingness.csv",
        outliers_path=tables_dir / "round3_derived_outlier_summary.csv",
    )


def _derived_specs(root: Path) -> list[GoldTableSpec]:
    return [spec for spec in _table_specs(root) if spec.layer == "derived_gold"]


def _load_samples(con: duckdb.DuckDBPyConnection, specs: list[GoldTableSpec]) -> dict[str, pd.DataFrame]:
    samples: dict[str, pd.DataFrame] = {}
    for spec in specs:
        samples[spec.table_name] = con.execute(
            f"SELECT * FROM {_relation_sql(spec)} USING SAMPLE reservoir({PLOT_SAMPLE_ROWS} ROWS) REPEATABLE (42)"
        ).df()
    return samples


def _build_outlier_summary(
    con: duckdb.DuckDBPyConnection,
    numeric_catalog: pd.DataFrame,
    spec_map: dict[str, GoldTableSpec],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for table_name, group in numeric_catalog.groupby("table_name", sort=False):
        spec = spec_map[table_name]
        relation = _relation_sql(spec)
        stats_rows: list[dict[str, Any]] = []
        for chunk in _chunks(group["variable_name"].tolist(), 12):
            exprs: list[str] = []
            for variable_name in chunk:
                qcol = _quote_ident(variable_name)
                exprs.extend(
                    [
                        f"COUNT({qcol}) AS {_quote_ident(variable_name + '__n')}",
                        f"MIN(CAST({qcol} AS DOUBLE)) AS {_quote_ident(variable_name + '__min')}",
                        f"QUANTILE_CONT(CAST({qcol} AS DOUBLE), 0.25) AS {_quote_ident(variable_name + '__q1')}",
                        f"QUANTILE_CONT(CAST({qcol} AS DOUBLE), 0.5) AS {_quote_ident(variable_name + '__median')}",
                        f"QUANTILE_CONT(CAST({qcol} AS DOUBLE), 0.75) AS {_quote_ident(variable_name + '__q3')}",
                        f"MAX(CAST({qcol} AS DOUBLE)) AS {_quote_ident(variable_name + '__max')}",
                    ]
                )
            stats = con.execute(f"SELECT {', '.join(exprs)} FROM {relation}").df().iloc[0]
            for variable_name in chunk:
                meta = group.loc[group["variable_name"] == variable_name].iloc[0]
                q1 = stats[f"{variable_name}__q1"]
                q3 = stats[f"{variable_name}__q3"]
                if pd.isna(q1) or pd.isna(q3):
                    iqr = np.nan
                    lower_fence = np.nan
                    upper_fence = np.nan
                    iqr_zero_flag = 0
                else:
                    iqr = float(q3) - float(q1)
                    lower_fence = float(q1) - 1.5 * iqr
                    upper_fence = float(q3) + 1.5 * iqr
                    iqr_zero_flag = int(iqr == 0.0)
                stats_rows.append(
                    {
                        "table_name": table_name,
                        "table_label": meta["table_label"],
                        "variable_name": variable_name,
                        "label": meta["label"],
                        "theme_group": meta["theme_group"],
                        "distribution_bucket": meta["distribution_bucket"],
                        "n_non_missing": int(stats[f"{variable_name}__n"]),
                        "min": stats[f"{variable_name}__min"],
                        "q1": q1,
                        "median": stats[f"{variable_name}__median"],
                        "q3": q3,
                        "max": stats[f"{variable_name}__max"],
                        "iqr": iqr,
                        "lower_fence": lower_fence,
                        "upper_fence": upper_fence,
                        "iqr_zero_flag": iqr_zero_flag,
                        "missing_rows": int(meta["missing_rows"]),
                        "missing_pct": float(meta["missing_pct"]),
                    }
                )
        stats_frame = pd.DataFrame(stats_rows)
        outlier_counts: list[dict[str, Any]] = []
        for chunk in _chunks(stats_frame["variable_name"].tolist(), 12):
            exprs = []
            for variable_name in chunk:
                row = stats_frame.loc[stats_frame["variable_name"] == variable_name].iloc[0]
                qcol = _quote_ident(variable_name)
                if pd.isna(row["lower_fence"]) or pd.isna(row["upper_fence"]) or int(row["iqr_zero_flag"]) == 1:
                    exprs.append(f"CAST(NULL AS BIGINT) AS {_quote_ident(variable_name + '__outliers')}")
                    continue
                exprs.append(
                    f"""
                    SUM(
                        CASE
                            WHEN {qcol} IS NOT NULL
                             AND (CAST({qcol} AS DOUBLE) < {float(row['lower_fence'])}
                               OR CAST({qcol} AS DOUBLE) > {float(row['upper_fence'])})
                            THEN 1
                            ELSE 0
                        END
                    ) AS {_quote_ident(variable_name + '__outliers')}
                    """
                )
            counts = con.execute(f"SELECT {', '.join(exprs)} FROM {relation}").df().iloc[0]
            for variable_name in chunk:
                value = counts[f"{variable_name}__outliers"]
                outlier_counts.append(
                    {
                        "table_name": table_name,
                        "variable_name": variable_name,
                        "outlier_rows": pd.NA if pd.isna(value) else int(value),
                    }
                )
        outlier_frame = pd.DataFrame(outlier_counts)
        merged = stats_frame.merge(outlier_frame, on=["table_name", "variable_name"], how="left")
        merged["outlier_pct"] = np.where(
            merged["n_non_missing"] > 0,
                merged["outlier_rows"] / merged["n_non_missing"],
                np.nan,
        )
        merged["outlier_status"] = np.where(
            merged["iqr_zero_flag"] == 1,
            "IQR zero; revisar como cauda rara ou zero-inflada",
            "Tukey calculado",
        )
        frames.append(merged)
    if not frames:
        return pd.DataFrame()
    summary = pd.concat(frames, ignore_index=True)
    return summary.sort_values(["outlier_pct", "table_name", "variable_name"], ascending=[False, True, True], kind="mergesort").reset_index(drop=True)


def _render_cover(pdf: PdfPages, missingness: pd.DataFrame, outliers: pd.DataFrame, specs: list[GoldTableSpec], artifacts: DerivedQualityArtifacts) -> None:
    rankable = outliers[(outliers["iqr_zero_flag"] == 0) & outliers["outlier_pct"].notna()].copy()
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.text(0.03, 0.94, "Round 3: missings e outliers das variaveis derivadas", fontsize=20, fontweight="bold", transform=ax.transAxes)
    ax.text(0.03, 0.86, "Escopo", fontsize=13, fontweight="bold", transform=ax.transAxes)
    ax.text(
        0.03,
        0.78,
        "\n".join(
            [
                "Analise restrita as tabelas derived_gold em data/gold.",
                "Missingness calculado exatamente para todas as colunas das views derivadas.",
                "Outliers calculados nas variaveis numericas via regra de Tukey (Q1 - 1.5*IQR; Q3 + 1.5*IQR).",
                "Variaveis com IQR zero ficam marcadas como zero-infladas/cauda rara e nao entram no ranking de outliers.",
                f"Saida principal: {artifacts.pdf_path.name}",
            ]
        ),
        fontsize=11,
        va="top",
        transform=ax.transAxes,
    )
    ax.text(0.03, 0.46, "Cobertura", fontsize=13, fontweight="bold", transform=ax.transAxes)
    ax.text(
        0.03,
        0.38,
        "\n".join(
            [
                f"Tabelas derived_gold: {len(specs)}",
                f"Variaveis avaliadas para missingness: {missingness.shape[0]}",
                f"Variaveis numericas avaliadas para outliers: {outliers.shape[0]}",
                f"Maior missing observado: {missingness['missing_pct'].max()*100:.2f}%",
                f"Maior percentual de outliers rankeavel: {rankable['outlier_pct'].max()*100:.2f}%" if not rankable.empty else "Maior percentual de outliers rankeavel: n/a",
            ]
        ),
        fontsize=11,
        va="top",
        transform=ax.transAxes,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_top_missingness(pdf: PdfPages, missingness: pd.DataFrame) -> None:
    top = missingness.sort_values("missing_pct", ascending=False).head(20).copy()
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.barh(top["table_name"] + " | " + top["variable_name"], top["missing_pct"] * 100, color="#a63a50")
    ax.invert_yaxis()
    ax.set_title("Top 20 missings nas views derivadas", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("% missing")
    ax.grid(axis="x", alpha=0.25)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_top_outliers(pdf: PdfPages, outliers: pd.DataFrame) -> None:
    top = outliers[(outliers["iqr_zero_flag"] == 0) & outliers["outlier_pct"].notna()].sort_values("outlier_pct", ascending=False).head(20).copy()
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.barh(top["table_name"] + " | " + top["variable_name"], top["outlier_pct"] * 100, color="#d98e73")
    ax.invert_yaxis()
    ax.set_title("Top 20 variaveis derivadas com maior percentual de outliers", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("% de outliers (Tukey IQR)")
    ax.grid(axis="x", alpha=0.25)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_boxplots(pdf: PdfPages, outliers: pd.DataFrame, samples: dict[str, pd.DataFrame]) -> None:
    top = outliers[(outliers["iqr_zero_flag"] == 0) & outliers["outlier_pct"].notna()].sort_values("outlier_pct", ascending=False).head(18).to_dict(orient="records")
    if not top:
        return
    for page_start in range(0, len(top), 6):
        page = top[page_start : page_start + 6]
        fig, axes = plt.subplots(len(page), 1, figsize=(11.69, 8.27))
        if len(page) == 1:
            axes = [axes]
        fig.suptitle("Boxplots das variaveis derivadas com maior sinal de outlier", x=0.03, y=0.98, ha="left", fontsize=16, fontweight="bold")
        for axis, row in zip(axes, page, strict=False):
            frame = samples[row["table_name"]]
            values = pd.to_numeric(frame[row["variable_name"]], errors="coerce").dropna()
            if values.empty:
                axis.text(0.5, 0.5, "Sem valores nao nulos na amostra", ha="center", va="center")
                axis.axis("off")
                continue
            axis.boxplot(values.to_numpy(), vert=False, patch_artist=True, whis=(5, 95), boxprops={"facecolor": "#9cc3ad"})
            axis.set_title(f"{row['label']} [{row['table_name']}]", loc="left", fontsize=11)
            axis.grid(axis="x", alpha=0.2)
            axis.text(
                0.99,
                0.82,
                f"outliers={row['outlier_pct']*100:.2f}% | missing={row['missing_pct']*100:.2f}%",
                ha="right",
                va="center",
                transform=axis.transAxes,
                fontsize=9,
                color="#444444",
            )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def build_round3_derived_quality_report(root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    artifacts = _artifact_paths(root, output_dir=output_dir)
    artifacts.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts.tables_dir.mkdir(parents=True, exist_ok=True)

    specs = _derived_specs(root)
    spec_map = {spec.table_name: spec for spec in specs}
    con = duckdb.connect()
    catalog, table_inventory = _build_catalog(root, con, specs)

    missingness = catalog[
        [
            "table_name",
            "table_label",
            "variable_name",
            "label",
            "theme_group",
            "distribution_bucket",
            "total_rows",
            "missingness_denominator_rows",
            "missingness_scope",
            "non_missing_rows",
            "missing_rows",
            "missing_pct",
            "approx_distinct_non_missing",
        ]
    ].sort_values(["missing_pct", "table_name", "variable_name"], ascending=[False, True, True], kind="mergesort")
    numeric_catalog = catalog[
        catalog["distribution_bucket"].isin({"continuous", "count"}) & (catalog["non_missing_rows"] > 0)
    ].copy()
    outliers = _build_outlier_summary(con, numeric_catalog, spec_map)

    missingness.to_csv(artifacts.missingness_path, index=False)
    outliers.to_csv(artifacts.outliers_path, index=False)

    samples = _load_samples(con, specs)
    with PdfPages(artifacts.pdf_path) as pdf:
        _render_cover(pdf, missingness, outliers, specs, artifacts)
        _render_top_missingness(pdf, missingness)
        _render_top_outliers(pdf, outliers)
        _render_boxplots(pdf, outliers, samples)
        _table_pages(
            pdf,
            missingness,
            title="Missingness nas variaveis derivadas",
            subtitle="Appendix completa das colunas das views derived_gold.",
            columns=[
                "table_name",
                "variable_name",
                "theme_group",
                "distribution_bucket",
                "missingness_denominator_rows",
                "missingness_scope",
                "missing_rows",
                "missing_pct",
                "approx_distinct_non_missing",
            ],
            rename={
                "table_name": "Tabela",
                "variable_name": "Variavel",
                "theme_group": "Grupo",
                "distribution_bucket": "Tipo",
                "missingness_denominator_rows": "Linhas avaliadas",
                "missingness_scope": "Escopo",
                "missing_rows": "Missings",
                "missing_pct": "% missing",
                "approx_distinct_non_missing": "Distintos aprox",
            },
            rows_per_page=26,
        )
        _table_pages(
            pdf,
            outliers,
            title="Outliers nas variaveis derivadas numericas",
            subtitle="Regra de Tukey IQR aplicada quando ha dispersao observavel; IQR zero fica marcado como nao rankeavel.",
            columns=["table_name", "variable_name", "distribution_bucket", "n_non_missing", "outlier_rows", "outlier_pct", "iqr_zero_flag", "outlier_status", "q1", "median", "q3", "lower_fence", "upper_fence"],
            rename={
                "table_name": "Tabela",
                "variable_name": "Variavel",
                "distribution_bucket": "Tipo",
                "n_non_missing": "N nao missing",
                "outlier_rows": "Outliers",
                "outlier_pct": "% outliers",
                "iqr_zero_flag": "IQR zero",
                "outlier_status": "Status",
                "q1": "Q1",
                "median": "Mediana",
                "q3": "Q3",
                "lower_fence": "Fence inf",
                "upper_fence": "Fence sup",
            },
            rows_per_page=22,
        )

    manifest = {
        "generated_at_utc": _utc_now(),
        "scope": "derived_gold_only",
        "source_tables": table_inventory.to_dict(orient="records"),
        "outlier_method": "Tukey IQR fences (Q1 - 1.5*IQR, Q3 + 1.5*IQR)",
        "sample_rows_per_table_for_boxplots": PLOT_SAMPLE_ROWS,
        "outputs": {
            "pdf": str(artifacts.pdf_path.relative_to(root)),
            "missingness_csv": str(artifacts.missingness_path.relative_to(root)),
            "outliers_csv": str(artifacts.outliers_path.relative_to(root)),
        },
    }
    artifacts.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    con.close()
    return {
        "pdf_path": str(artifacts.pdf_path),
        "manifest_path": str(artifacts.manifest_path),
        "missingness_path": str(artifacts.missingness_path),
        "outliers_path": str(artifacts.outliers_path),
    }
