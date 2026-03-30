from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..config import repository_root
from .registry import StatCalcRegistry
from .runner import TrackBWorker, submit_run
from .schemas import AnalysisRun, AnalysisRunResult, AnalysisRunSpec, SavedPreset, SavedPresetCreate, SavedPresetUpdate
from .storage import StatCalcStorage


def create_app(
    *,
    root: Path | None = None,
    storage_dir: Path | None = None,
    frontend_dist: Path | None = None,
    start_worker: bool = True,
) -> FastAPI:
    resolved_root = root or repository_root()
    resolved_storage_dir = storage_dir or resolved_root / "artifacts" / "statcalc" / "v1"
    resolved_frontend_dist = frontend_dist or resolved_root / "apps" / "observa-infancia-statcalc-web" / "dist"
    storage = StatCalcStorage(resolved_storage_dir)
    registry = StatCalcRegistry(resolved_root)
    worker = TrackBWorker(resolved_root, resolved_storage_dir)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if start_worker:
            worker.start()
        try:
            yield
        finally:
            if start_worker:
                worker.stop()

    app = FastAPI(title="StatCalc API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/statcalc-assets", StaticFiles(directory=resolved_storage_dir), name="statcalc-assets")
    if resolved_frontend_dist.exists():
        assets_dir = resolved_frontend_dist / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {
            "status": "ok",
            "root": str(resolved_root),
            "storage_dir": str(resolved_storage_dir),
            "frontend_dist": str(resolved_frontend_dist),
        }

    @app.get("/api/v1/statcalc/meta/recipes")
    def meta_recipes() -> list[dict[str, object]]:
        return [recipe.model_dump(mode="json") for recipe in registry.get_recipes()]

    @app.get("/api/v1/statcalc/meta/stratifiers")
    def meta_stratifiers() -> list[dict[str, object]]:
        return [item.model_dump(mode="json") for item in registry.get_stratifiers()]

    @app.get("/api/v1/statcalc/meta/presets", response_model=list[SavedPreset])
    def meta_presets() -> list[SavedPreset]:
        return storage.list_presets()

    @app.get("/api/v1/statcalc/runs", response_model=list[AnalysisRun])
    def list_runs(limit: int = 50) -> list[AnalysisRun]:
        return storage.list_runs(limit=limit)

    @app.post("/api/v1/statcalc/runs", response_model=AnalysisRun)
    def create_run(spec: AnalysisRunSpec) -> AnalysisRun:
        try:
            run, _ = submit_run(resolved_root, storage, registry, spec)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return run

    @app.get("/api/v1/statcalc/runs/{run_id}", response_model=AnalysisRun)
    def get_run(run_id: str) -> AnalysisRun:
        try:
            return storage.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Run {run_id} was not found.") from exc

    @app.get("/api/v1/statcalc/runs/{run_id}/result", response_model=AnalysisRunResult)
    def get_run_result(run_id: str) -> AnalysisRunResult:
        try:
            run = storage.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Run {run_id} was not found.") from exc
        result_path = Path(run.run_dir) / "result.json"
        if not result_path.exists():
            raise HTTPException(status_code=404, detail=f"Run {run_id} does not have a completed result yet.")
        return AnalysisRunResult.model_validate(json.loads(result_path.read_text(encoding="utf-8")))

    @app.post("/api/v1/statcalc/presets", response_model=SavedPreset)
    def create_preset(payload: SavedPresetCreate) -> SavedPreset:
        try:
            registry.validate_spec(payload.run_spec)
            return storage.create_preset(
                name=payload.name,
                description=payload.description,
                run_spec=payload.run_spec,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/api/v1/statcalc/presets/{preset_id}", response_model=SavedPreset)
    def update_preset(preset_id: str, payload: SavedPresetUpdate) -> SavedPreset:
        try:
            registry.validate_spec(payload.run_spec)
            return storage.update_preset(
                preset_id,
                name=payload.name,
                description=payload.description,
                run_spec=payload.run_spec,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Preset {preset_id} was not found.") from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/v1/statcalc/presets/{preset_id}")
    def delete_preset(preset_id: str) -> dict[str, object]:
        storage.delete_preset(preset_id)
        return {"status": "deleted", "preset_id": preset_id}

    @app.get("/{full_path:path}", response_model=None)
    def serve_frontend(full_path: str):
        if not resolved_frontend_dist.exists():
            return HTMLResponse(
                "<html><body><h1>StatCalc frontend not built</h1><p>Build apps/observa-infancia-statcalc-web first.</p></body></html>",
                status_code=200,
            )
        candidate = resolved_frontend_dist / full_path
        if full_path and candidate.exists() and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(resolved_frontend_dist / "index.html")

    return app
