from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schemas import AnalysisRun, AnalysisRunSpec, SavedPreset


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def spec_hash(spec: AnalysisRunSpec) -> str:
    payload = json.dumps(spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return __import__("hashlib").sha256(payload.encode("utf-8")).hexdigest()


class StatCalcStorage:
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir = self.storage_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = self.storage_dir / "statcalc.sqlite"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.sqlite_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    track_id TEXT NOT NULL,
                    recipe_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    execution_mode TEXT NOT NULL,
                    is_reference INTEGER NOT NULL,
                    spec_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    warnings_json TEXT NOT NULL,
                    error_message TEXT,
                    run_spec_json TEXT NOT NULL,
                    run_dir TEXT NOT NULL,
                    result_json_path TEXT
                );

                CREATE TABLE IF NOT EXISTS job_queue (
                    run_id TEXT PRIMARY KEY,
                    queue_status TEXT NOT NULL,
                    enqueued_at TEXT NOT NULL,
                    claimed_at TEXT,
                    completed_at TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS presets (
                    preset_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    run_spec_json TEXT NOT NULL
                );
                """
            )

    def mark_interrupted_running_jobs(self) -> None:
        now = utc_now_iso()
        with self._connect() as connection:
            running = connection.execute(
                "SELECT run_id, warnings_json FROM runs WHERE status = 'running'"
            ).fetchall()
            for row in running:
                warnings = json.loads(row["warnings_json"])
                warnings.append("Run canceled because the statcalc service restarted before the job finished.")
                connection.execute(
                    """
                    UPDATE runs
                    SET status = 'failed',
                        updated_at = ?,
                        completed_at = ?,
                        warnings_json = ?,
                        error_message = COALESCE(error_message, 'Interrupted by service restart.')
                    WHERE run_id = ?
                    """,
                    (now, now, json.dumps(warnings), row["run_id"]),
                )
                connection.execute(
                    """
                    UPDATE job_queue
                    SET queue_status = 'failed',
                        completed_at = ?
                    WHERE run_id = ? AND queue_status = 'claimed'
                    """,
                    (now, row["run_id"]),
                )

    def _row_to_run(self, row: sqlite3.Row) -> AnalysisRun:
        spec = AnalysisRunSpec.model_validate(json.loads(row["run_spec_json"]))
        warnings = json.loads(row["warnings_json"])
        return AnalysisRun(
            run_id=row["run_id"],
            track_id=row["track_id"],
            recipe_id=row["recipe_id"],
            status=row["status"],
            execution_mode=row["execution_mode"],
            is_reference=bool(row["is_reference"]),
            spec_hash=row["spec_hash"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            warnings=warnings,
            error_message=row["error_message"],
            run_dir=row["run_dir"],
            result_available=bool(row["result_json_path"]),
            spec=spec,
        )

    def create_run(
        self,
        *,
        track_id: str,
        recipe_id: str,
        execution_mode: str,
        is_reference: bool,
        status: str,
        spec: AnalysisRunSpec,
        warnings: list[str] | None = None,
    ) -> AnalysisRun:
        run_id = uuid.uuid4().hex
        now = utc_now_iso()
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, track_id, recipe_id, status, execution_mode, is_reference,
                    spec_hash, created_at, updated_at, warnings_json, run_spec_json, run_dir
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    track_id,
                    recipe_id,
                    status,
                    execution_mode,
                    1 if is_reference else 0,
                    spec_hash(spec),
                    now,
                    now,
                    json.dumps(warnings or []),
                    json.dumps(spec.model_dump(mode="json"), sort_keys=True),
                    str(run_dir),
                ),
            )
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._row_to_run(row)

    def mark_run_running(self, run_id: str) -> AnalysisRun:
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET status = 'running', started_at = ?, updated_at = ? WHERE run_id = ?",
                (now, now, run_id),
            )
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return self._row_to_run(row)

    def enqueue_run(self, run_id: str) -> None:
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO job_queue (run_id, queue_status, enqueued_at, claimed_at, completed_at)
                VALUES (?, 'queued', ?, NULL, NULL)
                """,
                (run_id, now),
            )
            connection.execute(
                "UPDATE runs SET status = 'queued', updated_at = ? WHERE run_id = ?",
                (now, run_id),
            )

    def claim_next_queued_run(self) -> str | None:
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT run_id
                FROM job_queue
                WHERE queue_status = 'queued'
                ORDER BY enqueued_at
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            run_id = str(row["run_id"])
            connection.execute(
                "UPDATE job_queue SET queue_status = 'claimed', claimed_at = ? WHERE run_id = ?",
                (now, run_id),
            )
            connection.execute(
                "UPDATE runs SET status = 'running', started_at = ?, updated_at = ? WHERE run_id = ?",
                (now, now, run_id),
            )
            connection.commit()
            return run_id

    def mark_job_finished(self, run_id: str, status: str) -> None:
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                "UPDATE job_queue SET queue_status = ?, completed_at = ? WHERE run_id = ?",
                (status, now, run_id),
            )

    def get_run(self, run_id: str) -> AnalysisRun:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return self._row_to_run(row)

    def list_runs(self, limit: int = 50) -> list[AnalysisRun]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def set_run_result(
        self,
        run_id: str,
        *,
        status: str,
        warnings: list[str],
        error_message: str | None = None,
        result_json_path: str | None = None,
    ) -> AnalysisRun:
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET status = ?,
                    updated_at = ?,
                    completed_at = ?,
                    warnings_json = ?,
                    error_message = ?,
                    result_json_path = ?
                WHERE run_id = ?
                """,
                (status, now, now, json.dumps(warnings), error_message, result_json_path, run_id),
            )
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return self._row_to_run(row)

    def run_result_path(self, run_id: str) -> Path:
        run = self.get_run(run_id)
        return Path(run.run_dir) / "result.json"

    def get_run_spec(self, run_id: str) -> AnalysisRunSpec:
        return self.get_run(run_id).spec

    def create_preset(self, *, name: str, description: str | None, run_spec: AnalysisRunSpec) -> SavedPreset:
        preset_id = uuid.uuid4().hex
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO presets (preset_id, name, description, created_at, updated_at, run_spec_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    preset_id,
                    name,
                    description,
                    now,
                    now,
                    json.dumps(run_spec.model_dump(mode="json"), sort_keys=True),
                ),
            )
            row = connection.execute("SELECT * FROM presets WHERE preset_id = ?", (preset_id,)).fetchone()
        return self._row_to_preset(row)

    def update_preset(self, preset_id: str, *, name: str, description: str | None, run_spec: AnalysisRunSpec) -> SavedPreset:
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE presets
                SET name = ?, description = ?, updated_at = ?, run_spec_json = ?
                WHERE preset_id = ?
                """,
                (
                    name,
                    description,
                    now,
                    json.dumps(run_spec.model_dump(mode="json"), sort_keys=True),
                    preset_id,
                ),
            )
            row = connection.execute("SELECT * FROM presets WHERE preset_id = ?", (preset_id,)).fetchone()
        if row is None:
            raise KeyError(preset_id)
        return self._row_to_preset(row)

    def delete_preset(self, preset_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM presets WHERE preset_id = ?", (preset_id,))

    def list_presets(self) -> list[SavedPreset]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM presets ORDER BY updated_at DESC").fetchall()
        return [self._row_to_preset(row) for row in rows]

    def _row_to_preset(self, row: sqlite3.Row) -> SavedPreset:
        return SavedPreset(
            preset_id=row["preset_id"],
            name=row["name"],
            description=row["description"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            run_spec=AnalysisRunSpec.model_validate(json.loads(row["run_spec_json"])),
        )
