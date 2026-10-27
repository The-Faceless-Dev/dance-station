"""Small durable SQLite repository for the standalone dataset manager."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

from .contracts import now_iso, new_id


class DatasetRepository:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "dataset.sqlite3"
        self._lock = threading.RLock()
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS songs (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    artist TEXT NOT NULL DEFAULT '',
                    source_filename TEXT NOT NULL,
                    language TEXT NOT NULL,
                    lyrics TEXT NOT NULL DEFAULT '',
                    lyrics_source TEXT NOT NULL DEFAULT 'missing',
                    duration_ms INTEGER NOT NULL DEFAULT 0,
                    sample_rate INTEGER NOT NULL DEFAULT 0,
                    channels INTEGER NOT NULL DEFAULT 0,
                    source_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'New',
                    has_unsaved_changes INTEGER NOT NULL DEFAULT 0,
                    vocals_only INTEGER NOT NULL DEFAULT 0,
                    selected_vocal_source TEXT NOT NULL DEFAULT 'machine',
                    active_run_id TEXT,
                    active_draft_id TEXT,
                    approved_revision_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_error TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS songs_updated_idx ON songs(updated_at DESC);
                CREATE TABLE IF NOT EXISTS extraction_runs (
                    id TEXT PRIMARY KEY,
                    song_id TEXT NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
                    pipeline_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    current_stage TEXT NOT NULL DEFAULT '',
                    config_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL DEFAULT '{}',
                    artifacts_json TEXT NOT NULL DEFAULT '{}',
                    warnings_json TEXT NOT NULL DEFAULT '[]',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS runs_song_idx ON extraction_runs(song_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS extraction_stages (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES extraction_runs(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT '',
                    provider TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '',
                    started_at TEXT,
                    completed_at TEXT,
                    error TEXT NOT NULL DEFAULT '',
                    artifacts_json TEXT NOT NULL DEFAULT '{}',
                    UNIQUE(run_id, name)
                );
                CREATE TABLE IF NOT EXISTS annotation_drafts (
                    id TEXT PRIMARY KEY,
                    song_id TEXT NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
                    version INTEGER NOT NULL,
                    annotation_json TEXT NOT NULL,
                    validation_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(song_id)
                );
                CREATE TABLE IF NOT EXISTS approved_revisions (
                    id TEXT PRIMARY KEY,
                    song_id TEXT NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
                    parent_revision_id TEXT,
                    version INTEGER NOT NULL,
                    annotation_json TEXT NOT NULL,
                    validation_json TEXT NOT NULL,
                    export_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS revisions_song_idx ON approved_revisions(song_id, version DESC);
                CREATE TABLE IF NOT EXISTS dataset_exports (
                    id TEXT PRIMARY KEY,
                    revision_ids_json TEXT NOT NULL,
                    manifest_path TEXT NOT NULL,
                    validation_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))

    @staticmethod
    def _loads(value: str, default: Any) -> Any:
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return default

    def create_song(self, values: dict[str, Any]) -> dict[str, Any]:
        song_id = str(values.get("id") or new_id("song"))
        timestamp = now_iso()
        fields = {
            "id": song_id,
            "title": str(values.get("title") or values.get("source_filename") or song_id),
            "artist": str(values.get("artist") or ""),
            "source_filename": str(values["source_filename"]),
            "language": str(values.get("language") or "en"),
            "lyrics": str(values.get("lyrics") or ""),
            "lyrics_source": "user_provided" if values.get("lyrics") else "missing",
            "duration_ms": int(values.get("duration_ms") or 0),
            "sample_rate": int(values.get("sample_rate") or 0),
            "channels": int(values.get("channels") or 0),
            "source_sha256": str(values["source_sha256"]),
            "status": "New",
            "has_unsaved_changes": 0,
            "vocals_only": int(bool(values.get("vocals_only"))),
            "selected_vocal_source": "uploaded" if values.get("vocals_only") else "machine",
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO songs
                (id,title,artist,source_filename,language,lyrics,lyrics_source,duration_ms,sample_rate,channels,
                 source_sha256,status,has_unsaved_changes,vocals_only,selected_vocal_source,created_at,updated_at)
                VALUES (:id,:title,:artist,:source_filename,:language,:lyrics,:lyrics_source,:duration_ms,:sample_rate,:channels,
                 :source_sha256,:status,:has_unsaved_changes,:vocals_only,:selected_vocal_source,:created_at,:updated_at)""",
                fields,
            )
        return self.get_song(song_id) or fields

    def _song(self, row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        for key in ("has_unsaved_changes", "vocals_only"):
            value[key] = bool(value[key])
        return value

    def get_song(self, song_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM songs WHERE id = ?", (song_id,)).fetchone()
        return self._song(row) if row else None

    def list_songs(self, search: str = "", status: str = "") -> list[dict[str, Any]]:
        query = "SELECT * FROM songs WHERE 1=1"
        params: list[Any] = []
        if search.strip():
            query += " AND (title LIKE ? OR source_filename LIKE ? OR artist LIKE ?)"
            token = f"%{search.strip()}%"
            params.extend((token, token, token))
        if status.strip() and status.strip().lower() != "all":
            query += " AND status = ?"
            params.append(status.strip())
        query += " ORDER BY created_at DESC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._song(row) for row in rows]

    def update_song(self, song_id: str, **values: Any) -> dict[str, Any]:
        allowed = {
            "title", "artist", "language", "lyrics", "lyrics_source", "status", "has_unsaved_changes",
            "selected_vocal_source", "active_run_id", "active_draft_id", "approved_revision_id", "last_error",
        }
        assignments = [(key, value) for key, value in values.items() if key in allowed]
        if not assignments:
            return self.get_song(song_id) or {}
        assignments.append(("updated_at", now_iso()))
        clause = ", ".join(f"{key} = ?" for key, _ in assignments)
        with self._lock, self._connect() as connection:
            connection.execute(f"UPDATE songs SET {clause} WHERE id = ?", [value for _, value in assignments] + [song_id])
        return self.get_song(song_id) or {}

    def create_run(self, song_id: str, config: dict[str, Any]) -> dict[str, Any]:
        run_id = new_id("run")
        timestamp = now_iso()
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO extraction_runs
                (id,song_id,pipeline_version,status,progress,current_stage,config_json,created_at,updated_at)
                VALUES (?,?,?, ?,0,'',?,?,?)""",
                (run_id, song_id, "dataset-manager-v1", "queued", self._json(config), timestamp, timestamp),
            )
            connection.execute(
                "UPDATE songs SET active_run_id = ?, status = 'Queued', last_error = '', updated_at = ? WHERE id = ?",
                (run_id, timestamp, song_id),
            )
        return self.get_run(run_id) or {"id": run_id, "song_id": song_id, "status": "queued"}

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM extraction_runs WHERE id = ?", (run_id,)).fetchone()
            stages = connection.execute(
                "SELECT * FROM extraction_stages WHERE run_id = ? ORDER BY rowid", (run_id,)
            ).fetchall()
        if not row:
            return None
        value = dict(row)
        for key, default in (("config_json", {}), ("provenance_json", {}), ("artifacts_json", {}), ("warnings_json", [])):
            value[key.removesuffix("_json")] = self._loads(value.pop(key), default)
        value["stages"] = [self._stage(item) for item in stages]
        return value

    def list_runs(self, song_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT id FROM extraction_runs WHERE song_id=? ORDER BY created_at DESC", (song_id,)).fetchall()
        return [run for row in rows if (run := self.get_run(row["id"])) is not None]

    def _stage(self, row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["artifacts"] = self._loads(value.pop("artifacts_json"), {})
        return value

    def upsert_stage(self, run_id: str, name: str, **values: Any) -> dict[str, Any]:
        with self._connect() as connection:
            existing = connection.execute("SELECT * FROM extraction_stages WHERE run_id=? AND name=?", (run_id, name)).fetchone()
        defaults = {
            "status": "queued", "progress": 0.0, "message": "", "provider": "", "model": "",
            "started_at": None, "completed_at": None, "error": "", "artifacts_json": self._json(values.get("artifacts", {})),
        }
        for key in ("status", "progress", "message", "provider", "model", "started_at", "completed_at", "error"):
            if key in values:
                defaults[key] = values[key]
            elif existing is not None:
                defaults[key] = existing[key]
        if "artifacts" not in values and existing is not None:
            defaults["artifacts_json"] = existing["artifacts_json"]
        stage_id = str(values.get("id") or new_id("stage"))
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO extraction_stages
                (id,run_id,name,status,progress,message,provider,model,started_at,completed_at,error,artifacts_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(run_id,name) DO UPDATE SET status=excluded.status,progress=excluded.progress,
                message=excluded.message,provider=excluded.provider,model=excluded.model,started_at=excluded.started_at,
                completed_at=excluded.completed_at,error=excluded.error,artifacts_json=excluded.artifacts_json""",
                (stage_id, run_id, name, defaults["status"], float(defaults["progress"]), defaults["message"], defaults["provider"], defaults["model"], defaults["started_at"], defaults["completed_at"], defaults["error"], defaults["artifacts_json"]),
            )
        run = self.get_run(run_id) or {}
        return next((stage for stage in run.get("stages", []) if stage["name"] == name), {})

    def update_run(self, run_id: str, **values: Any) -> dict[str, Any]:
        mapping = {"config": "config_json", "provenance": "provenance_json", "artifacts": "artifacts_json", "warnings": "warnings_json"}
        allowed = {"status", "progress", "current_stage", "error", "completed_at", "config_json", "provenance_json", "artifacts_json", "warnings_json"}
        assignments: list[tuple[str, Any]] = []
        for key, value in values.items():
            target = mapping.get(key, key)
            if target not in allowed:
                continue
            if target.endswith("_json") and not isinstance(value, str):
                value = self._json(value)
            assignments.append((target, value))
        assignments.append(("updated_at", now_iso()))
        clause = ", ".join(f"{key} = ?" for key, _ in assignments)
        with self._lock, self._connect() as connection:
            connection.execute(f"UPDATE extraction_runs SET {clause} WHERE id = ?", [v for _, v in assignments] + [run_id])
        return self.get_run(run_id) or {}

    def recover_running_runs(self) -> int:
        with self._lock, self._connect() as connection:
            rows = connection.execute("SELECT id,song_id FROM extraction_runs WHERE status IN ('queued','running')").fetchall()
            for row in rows:
                connection.execute("UPDATE extraction_runs SET status='failed',error=?,updated_at=? WHERE id=?", ("Manager restarted before this run completed.", now_iso(), row["id"]))
                connection.execute("UPDATE songs SET status='Failed',last_error=?,updated_at=? WHERE id=?", ("Manager restarted before this run completed.", now_iso(), row["song_id"]))
        return len(rows)

    def get_draft(self, song_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM annotation_drafts WHERE song_id = ?", (song_id,)).fetchone()
        if not row:
            return None
        value = dict(row)
        value["annotation"] = self._loads(value.pop("annotation_json"), {})
        value["validation"] = self._loads(value.pop("validation_json"), {})
        return value

    def delete_draft(self, song_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM annotation_drafts WHERE song_id = ?", (song_id,))
            connection.execute(
                "UPDATE songs SET active_draft_id=NULL,has_unsaved_changes=0 WHERE id=?",
                (song_id,),
            )

    def upsert_draft(self, song_id: str, annotation: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
        existing = self.get_draft(song_id)
        draft_id = existing["id"] if existing else new_id("draft")
        version = int(existing["version"] if existing else 0) + 1
        timestamp = now_iso()
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO annotation_drafts (id,song_id,version,annotation_json,validation_json,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(song_id) DO UPDATE SET id=excluded.id,version=excluded.version,annotation_json=excluded.annotation_json,
                validation_json=excluded.validation_json,updated_at=excluded.updated_at""",
                (draft_id, song_id, version, self._json(annotation), self._json(validation), existing["created_at"] if existing else timestamp, timestamp),
            )
            connection.execute("UPDATE songs SET active_draft_id=?,has_unsaved_changes=1,status=CASE WHEN status='Saved' THEN 'Editing' ELSE status END,updated_at=? WHERE id=?", (draft_id, timestamp, song_id))
        return self.get_draft(song_id) or {}

    def create_revision(self, song_id: str, annotation: dict[str, Any], validation: dict[str, Any], export: dict[str, Any]) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT approved_revision_id FROM songs WHERE id=?", (song_id,)).fetchone()
            parent = row["approved_revision_id"] if row else None
            version_row = connection.execute("SELECT COALESCE(MAX(version),0)+1 AS next_version FROM approved_revisions WHERE song_id=?", (song_id,)).fetchone()
            version = int(version_row["next_version"])
            revision_id = new_id("revision")
            timestamp = now_iso()
            connection.execute("INSERT INTO approved_revisions (id,song_id,parent_revision_id,version,annotation_json,validation_json,export_json,created_at) VALUES (?,?,?,?,?,?,?,?)", (revision_id,song_id,parent,version,self._json(annotation),self._json(validation),self._json(export),timestamp))
            connection.execute("UPDATE songs SET approved_revision_id=?,has_unsaved_changes=0,status='Saved',updated_at=?,last_error='' WHERE id=?", (revision_id,timestamp,song_id))
        return {"id": revision_id, "song_id": song_id, "version": version, "parent_revision_id": parent, "annotation": annotation, "validation": validation, "export": export, "created_at": timestamp}

    def get_revision(self, revision_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM approved_revisions WHERE id=?", (revision_id,)).fetchone()
        if not row:
            return None
        value = dict(row)
        for key, default in (("annotation_json", {}), ("validation_json", {}), ("export_json", {})):
            value[key.removesuffix("_json")] = self._loads(value.pop(key), default)
        return value

    def stats(self) -> dict[str, Any]:
        with self._connect() as connection:
            totals = connection.execute("SELECT COUNT(*) AS songs, COALESCE(SUM(duration_ms),0) AS duration_ms FROM songs").fetchone()
            saved = connection.execute("SELECT COUNT(*) AS songs FROM songs WHERE approved_revision_id IS NOT NULL").fetchone()
            review = connection.execute("SELECT COUNT(*) AS songs FROM songs WHERE status IN ('New','Ready for Review','Editing')").fetchone()
            failed = connection.execute("SELECT COUNT(*) AS songs FROM songs WHERE status='Failed'").fetchone()
            jobs = connection.execute("SELECT COUNT(*) AS jobs FROM extraction_runs WHERE status IN ('queued','running')").fetchone()
            approved_duration = connection.execute("SELECT COALESCE(SUM(duration_ms),0) AS duration_ms FROM songs WHERE approved_revision_id IS NOT NULL").fetchone()
        return {"songs": int(totals["songs"]), "duration_ms": int(totals["duration_ms"]), "saved_songs": int(saved["songs"]), "approved_duration_ms": int(approved_duration["duration_ms"]), "awaiting_review": int(review["songs"]), "failed_songs": int(failed["songs"]), "active_jobs": int(jobs["jobs"])}

    def create_export(self, revision_ids: Iterable[str], manifest_path: str, validation: dict[str, Any]) -> dict[str, Any]:
        export_id = new_id("export")
        timestamp = now_iso()
        revision_ids = list(revision_ids)
        with self._lock, self._connect() as connection:
            connection.execute("INSERT INTO dataset_exports (id,revision_ids_json,manifest_path,validation_json,created_at) VALUES (?,?,?,?,?)", (export_id,self._json(revision_ids),manifest_path,self._json(validation),timestamp))
        return {"id": export_id, "revision_ids": revision_ids, "manifest_path": manifest_path, "validation": validation, "created_at": timestamp}
