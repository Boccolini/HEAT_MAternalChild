from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from heat_mnch.dashboard_api.app import create_app
from heat_mnch.dashboard_build import build_dashboard_marts
from heat_mnch.dashboard_geometry import build_dashboard_geometry


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dashboard_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = _root()
    dashboard_dir = tmp_path_factory.mktemp("dashboard_api") / "v1"
    build_dashboard_marts(
        root=root,
        start_year=2024,
        end_year=2024,
        output_dir=dashboard_dir,
        metric_ids=[
            "deaths_u5",
            "deaths_neonatal",
            "preterm_flag",
            "low_birth_weight_lt2500_total",
            "aph_antepartum_total",
            "principal_diagnosis_child_dehydration_primary_total",
            "underlying_basic_cause_direct_heat_any_total",
            "tmax_c",
        ],
    )
    build_dashboard_geometry(
        root=root,
        source_zip=root / "artifacts" / "reference" / "era5" / "BR_Municipios_2022.zip",
        output_dir=dashboard_dir,
        tolerance=0.05,
    )
    return dashboard_dir


def test_dashboard_api_metadata_endpoints(dashboard_dir: Path) -> None:
    client = TestClient(create_app(dashboard_dir))

    manifest_response = client.get("/api/v1/meta/manifest")
    assert manifest_response.status_code == 200
    manifest = manifest_response.json()
    assert manifest["dashboard_version"] == "v1"

    filters_response = client.get("/api/v1/meta/filters")
    assert filters_response.status_code == 200
    filters = filters_response.json()
    assert 2024 in filters["years"]
    assert any(state["geo_code"] == "SP" for state in filters["states"])

    metrics_response = client.get("/api/v1/meta/metrics")
    assert metrics_response.status_code == 200
    metrics = metrics_response.json()
    assert any(metric["metric_id"] == "deaths_u5" for metric in metrics["metrics"])
    assert any(metric["metric_id"] == "low_birth_weight_lt2500_total" for metric in metrics["metrics"])
    assert any(metric["metric_id"] == "aph_antepartum_total" for metric in metrics["metrics"])
    assert any(metric["metric_id"] == "principal_diagnosis_child_dehydration_primary_total" for metric in metrics["metrics"])

    methods_response = client.get("/api/v1/meta/methods")
    assert methods_response.status_code == 200
    methods = methods_response.json()
    assert methods["technical_overview"]["metric_count"] >= len(metrics["metrics"])
    assert methods["technical_overview"]["stratifier_count"] >= len(metrics["context_variables"])
    assert methods["medallion_strategy"]["layers"][-1]["layer_id"] == "dashboard_mart"
    indicator_sheets = methods["indicator_sheets"]
    stratifier_sheets = methods["stratifier_sheets"]
    assert any(sheet["metric_id"] == "deaths_u5" for sheet in indicator_sheets)
    assert all(sheet["numerator"]["fields"] for sheet in indicator_sheets)
    assert all(sheet["source_system_ids"] for sheet in indicator_sheets)
    assert any(sheet["context_variable"] == "idh_quintile" for sheet in stratifier_sheets)
    assert all(sheet["threshold_labels"] for sheet in stratifier_sheets)


def test_dashboard_api_series_and_map_queries(dashboard_dir: Path) -> None:
    client = TestClient(create_app(dashboard_dir))

    payload = {
        "page_id": "child_mortality",
        "metric_id": "deaths_u5",
        "years": [2024],
        "granularity": "month",
        "geo_level": "state",
        "geo_codes": [],
        "value_mode": "count",
        "context_bins": [],
        "compare_mode": "none",
    }

    series_response = client.post("/api/v1/query/series", json=payload)
    assert series_response.status_code == 200
    series = series_response.json()
    assert len(series) == 12

    cards_response = client.post("/api/v1/query/cards", json=payload)
    assert cards_response.status_code == 200
    cards = cards_response.json()
    assert cards[0]["period_label"] == "2024"
    yearly_card_series_response = client.post(
        "/api/v1/query/series",
        json={**payload, "granularity": "year"},
    )
    assert yearly_card_series_response.status_code == 200
    yearly_card_series = yearly_card_series_response.json()
    assert cards[0]["value"] == yearly_card_series[0]["y"]

    map_response = client.post("/api/v1/query/map", json={**payload, "granularity": "year"})
    assert map_response.status_code == 200
    map_rows = map_response.json()
    assert any(row["geo_code"] == "SP" for row in map_rows)

    child_rate_response = client.post(
        "/api/v1/query/series",
        json={
            **payload,
            "metric_id": "principal_diagnosis_child_dehydration_primary_total",
            "value_mode": "rate",
        },
    )
    assert child_rate_response.status_code == 200
    child_rate_rows = child_rate_response.json()
    assert len(child_rate_rows) == 12
    assert any(row["y"] is not None for row in child_rate_rows)


def test_dashboard_api_cards_use_combined_period_for_multi_year_selection(dashboard_dir: Path) -> None:
    client = TestClient(create_app(dashboard_dir))

    payload = {
        "page_id": "child_mortality",
        "metric_id": "deaths_u5",
        "years": [2020, 2021, 2022, 2023, 2024],
        "granularity": "month",
        "geo_level": "national",
        "geo_codes": [],
        "value_mode": "count",
        "context_bins": [],
        "compare_mode": "none",
    }

    cards_response = client.post("/api/v1/query/cards", json=payload)
    assert cards_response.status_code == 200
    card = cards_response.json()[0]

    yearly_series_response = client.post(
        "/api/v1/query/series",
        json={**payload, "granularity": "year"},
    )
    assert yearly_series_response.status_code == 200
    yearly_series = yearly_series_response.json()

    assert card["period_label"] == "Combined 2020-24"
    assert card["value"] == sum(row["y"] or 0 for row in yearly_series)


def test_dashboard_api_relationship_query(dashboard_dir: Path) -> None:
    client = TestClient(create_app(dashboard_dir))

    payload = {
        "x_metric_id": "tmax_c",
        "y_metric_id": "preterm_flag",
        "years": [2024],
        "granularity": "month",
        "geo_level": "state",
        "geo_codes": [],
        "x_value_mode": "count",
        "y_value_mode": "rate",
        "context_variable": "idh_quintile",
        "context_bins": [],
    }

    response = client.post("/api/v1/query/relationship", json=payload)
    assert response.status_code == 200
    rows = response.json()
    assert rows
    assert "x_value" in rows[0]
    assert "y_value" in rows[0]
    assert "context_stratum" in rows[0]
    assert rows[0]["tooltip_meta"]["x_metric_id"] == "tmax_c"
    assert rows[0]["tooltip_meta"]["y_metric_id"] == "preterm_flag"

    regional_response = client.post(
        "/api/v1/query/relationship",
        json={**payload, "geo_level": "region", "context_variable": None},
    )
    assert regional_response.status_code == 200
    regional_rows = regional_response.json()
    assert regional_rows
    assert regional_rows[0]["label"] in {"North", "Northeast", "Southeast", "South", "Central-West"}


def test_dashboard_api_municipality_profile_endpoint(dashboard_dir: Path) -> None:
    client = TestClient(create_app(dashboard_dir))

    response = client.get("/api/v1/municipalities/1100015/profile")
    assert response.status_code == 200
    payload = response.json()
    assert payload["municipality_code"] == "1100015"
    assert "overview" in payload


def test_dashboard_geometry_uses_analytical_spine(dashboard_dir: Path) -> None:
    municipalities_topology = json.loads((dashboard_dir / "municipalities_2022_simplified.topojson").read_text(encoding="utf-8"))
    geometries = municipalities_topology["objects"]["municipalities"]["geometries"]
    municipality_codes = {geometry["properties"]["municipality_code"] for geometry in geometries}

    assert len(municipality_codes) == 5570
    assert "4300001" not in municipality_codes
    assert "4300002" not in municipality_codes
