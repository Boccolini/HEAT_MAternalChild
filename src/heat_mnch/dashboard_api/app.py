from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..config import repository_root
from .repository import DashboardRepository
from .schemas import (
    DashboardFilters,
    MapDatum,
    MetricCard,
    MethodsPayload,
    MunicipalityProfile,
    RankingRow,
    RelationshipFilters,
    RelationshipPoint,
    SeriesPoint,
    StratificationRow,
)


def create_app(dashboard_dir: Path | None = None) -> FastAPI:
    root = repository_root()
    resolved_dashboard_dir = dashboard_dir or root / "artifacts" / "dashboard" / "v1"
    repository = DashboardRepository(root=root, dashboard_dir=resolved_dashboard_dir)

    app = FastAPI(title="Observa Infancia Dashboard API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/dashboard-assets", StaticFiles(directory=resolved_dashboard_dir), name="dashboard-assets")

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {"status": "ok", "dashboard_dir": str(resolved_dashboard_dir)}

    @app.get("/api/v1/meta/manifest")
    def meta_manifest() -> dict[str, object]:
        return repository.get_manifest()

    @app.get("/api/v1/meta/filters")
    def meta_filters() -> dict[str, object]:
        return repository.get_filters()

    @app.get("/api/v1/meta/metrics")
    def meta_metrics() -> dict[str, object]:
        return repository.get_metrics()

    @app.get("/api/v1/meta/methods", response_model=MethodsPayload)
    def meta_methods() -> dict[str, object]:
        return repository.get_methods_payload()

    @app.post("/api/v1/query/cards", response_model=list[MetricCard])
    def query_cards(filters: DashboardFilters) -> list[dict[str, object]]:
        try:
            return repository.query_cards(filters)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/query/series", response_model=list[SeriesPoint])
    def query_series(filters: DashboardFilters) -> list[dict[str, object]]:
        try:
            return repository.query_series(filters)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/query/map", response_model=list[MapDatum])
    def query_map(filters: DashboardFilters) -> list[dict[str, object]]:
        try:
            return repository.query_map(filters)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/query/ranking", response_model=list[RankingRow])
    def query_ranking(filters: DashboardFilters) -> list[dict[str, object]]:
        try:
            return repository.query_ranking(filters)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/query/stratification", response_model=list[StratificationRow])
    def query_stratification(filters: DashboardFilters) -> list[dict[str, object]]:
        try:
            return repository.query_stratification(filters)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/query/relationship", response_model=list[RelationshipPoint])
    def query_relationship(filters: RelationshipFilters) -> list[dict[str, object]]:
        try:
            return repository.query_relationship(filters)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/v1/municipalities/{municipality_code}/profile", response_model=MunicipalityProfile)
    def municipality_profile(municipality_code: str) -> dict[str, object]:
        try:
            return repository.municipality_profile(municipality_code)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Municipality {municipality_code} was not found.") from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
