from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from heat_mnch.dashboard_build import build_dashboard_marts


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dashboard_output_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = _root()
    output_dir = tmp_path_factory.mktemp("dashboard") / "v1"
    build_dashboard_marts(
        root=root,
        start_year=2024,
        end_year=2024,
        output_dir=output_dir,
        metric_ids=[
            "deaths_u5",
            "preterm_flag",
            "low_birth_weight_lt2500_total",
            "aph_antepartum_total",
            "principal_diagnosis_child_dehydration_primary_total",
            "tmax_c",
        ],
    )
    return output_dir


def test_build_dashboard_marts_creates_expected_artifacts(dashboard_output_dir: Path) -> None:
    root = _root()
    output_dir = dashboard_output_dir
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["dashboard_version"] == "v1"
    assert (output_dir / "dashboard_metrics.parquet").exists()
    assert (output_dir / "dashboard_context.parquet").exists()
    assert (output_dir / "municipality_profiles.parquet").exists()
    assert (output_dir / "dashboard_geographies.parquet").exists()
    assert (output_dir / "manifest.json").exists()

    metrics = pd.read_parquet(output_dir / "dashboard_metrics.parquet")
    expected_columns = {
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
    }
    assert expected_columns.issubset(set(metrics.columns))
    assert set(metrics["metric_id"].unique()) == {
        "deaths_u5",
        "preterm_flag",
        "low_birth_weight_lt2500_total",
        "aph_antepartum_total",
        "principal_diagnosis_child_dehydration_primary_total",
        "tmax_c",
    }
    assert set(metrics.loc[metrics["month"].isna() & metrics["season"].isna(), "year"]) == {2024}


def test_build_dashboard_marts_matches_gold_annual_total_for_under_five_deaths(dashboard_output_dir: Path) -> None:
    root = _root()
    mart = pd.read_parquet(dashboard_output_dir / "dashboard_metrics.parquet")
    annual = mart.loc[
        (mart["metric_id"] == "deaths_u5")
        & (mart["year"] == 2024)
        & (mart["month"].isna())
        & (mart["season"].isna())
    ]
    mart_total = float(annual["value_count"].sum())

    source = pd.read_parquet(root / "data" / "gold" / "gold_mnch_mortality" / "YEAR=2024" / "gold_mnch_mortality_2024.parquet")
    source_total = float(pd.to_numeric(source["DEATHS_U5"], errors="coerce").fillna(0).sum())

    assert mart_total == source_total


def test_build_dashboard_marts_context_bins_include_missing_category(dashboard_output_dir: Path) -> None:
    manifest = json.loads((dashboard_output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "missing" in manifest["context_bins"]["idh_quintile"]["labels"]
    assert "missing" in manifest["context_bins"]["gini_quintile"]["labels"]


def test_build_dashboard_marts_geographies_follow_analytical_spine(dashboard_output_dir: Path) -> None:
    geographies = pd.read_parquet(dashboard_output_dir / "dashboard_geographies.parquet")
    municipality_codes = geographies["municipality_code"].astype(str)

    assert municipality_codes.nunique() == 5570
    assert not municipality_codes.isin(["4300001", "4300002"]).any()
