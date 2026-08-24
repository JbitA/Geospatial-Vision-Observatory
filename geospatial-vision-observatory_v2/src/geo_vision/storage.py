from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .config import Settings
from .domain import AnalysisRecord, FrameRecord, ObservationMetadata
from .security import chain_digest, sha256_bytes

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS frames (
 id INTEGER PRIMARY KEY, source_id TEXT NOT NULL UNIQUE, captured_at TEXT NOT NULL,
 fetched_at TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE, byte_size INTEGER NOT NULL,
 media_type TEXT NOT NULL, relative_path TEXT NOT NULL, metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS analyses (
 id INTEGER PRIMARY KEY, frame_id INTEGER NOT NULL REFERENCES frames(id),
 processor TEXT NOT NULL, processor_version TEXT NOT NULL, status TEXT NOT NULL,
 result_json TEXT NOT NULL, duration_ms REAL NOT NULL, created_at TEXT NOT NULL,
 UNIQUE(frame_id, processor, processor_version)
);
CREATE TABLE IF NOT EXISTS request_log (
 id INTEGER PRIMARY KEY, request_id TEXT, attempted_at TEXT NOT NULL, host TEXT NOT NULL,
 purpose TEXT NOT NULL, outcome TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit_log (
 id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, event_json TEXT NOT NULL,
 previous_digest TEXT NOT NULL, digest TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_request_log_attempted_at ON request_log(attempted_at);
CREATE INDEX IF NOT EXISTS idx_frames_captured_at ON frames(captured_at);
CREATE INDEX IF NOT EXISTS idx_analyses_frame_id ON analyses(frame_id);
"""


class Store:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.data_dir.mkdir(parents=True, exist_ok=True, mode=0o750)
        (self.settings.data_dir / "frames").mkdir(exist_ok=True, mode=0o750)
        self._lock = threading.RLock()
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(request_log)")}
            if "request_id" not in columns:
                conn.execute("ALTER TABLE request_log ADD COLUMN request_id TEXT")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_request_log_request_id "
                "ON request_log(request_id) WHERE request_id IS NOT NULL"
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.settings.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def acquire_request_budget(self, host: str, purpose: str, now: datetime) -> str | None:
        hour_cutoff = (now - timedelta(hours=1)).isoformat()
        day_cutoff = (now - timedelta(days=1)).isoformat()
        with self._lock, self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            hourly = conn.execute(
                "SELECT COUNT(*) FROM request_log WHERE attempted_at >= ?", (hour_cutoff,)
            ).fetchone()[0]
            daily = conn.execute(
                "SELECT COUNT(*) FROM request_log WHERE attempted_at >= ?", (day_cutoff,)
            ).fetchone()[0]
            last = conn.execute(
                "SELECT attempted_at FROM request_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
            spaced = (
                not last
                or (now - datetime.fromisoformat(last[0])).total_seconds()
                >= self.settings.min_request_spacing_seconds
            )
            if (
                hourly >= self.settings.hourly_request_budget
                or daily >= self.settings.daily_request_budget
                or not spaced
            ):
                return None
            request_id = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO request_log(request_id,attempted_at,host,purpose,outcome) "
                "VALUES(?,?,?,?,?)",
                (request_id, now.isoformat(), host, purpose, "started"),
            )
            return request_id

    def seconds_until_next_request(self, now: datetime) -> float:
        with self.connect() as conn:
            last = conn.execute(
                "SELECT attempted_at FROM request_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if last is None:
            return 0.0
        elapsed = (now - datetime.fromisoformat(last[0])).total_seconds()
        return max(0.0, self.settings.min_request_spacing_seconds - elapsed)

    def finish_request(self, request_id: str, outcome: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE request_log SET outcome=? WHERE request_id=? AND outcome='started'",
                (outcome, request_id),
            )

    def get_state(self, key: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def set_state(self, key: str, value: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO state(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def contains_source(self, source_id: str) -> bool:
        with self.connect() as conn:
            return (
                conn.execute("SELECT 1 FROM frames WHERE source_id=?", (source_id,)).fetchone()
                is not None
            )

    def save_frame(
        self, metadata: ObservationMetadata, payload: bytes, media_type: str
    ) -> FrameRecord:
        digest = sha256_bytes(payload)
        extension = ".png" if media_type == "image/png" else ".jpg"
        relative = Path("frames") / metadata.date.strftime("%Y/%m/%d") / f"{digest}{extension}"
        destination = self.settings.data_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
        destination_existed = destination.exists()
        fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".incoming-")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o640)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        now = datetime.now(UTC).isoformat()
        metadata_json = metadata.model_dump_json()
        try:
            with self._lock, self.connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                cursor = conn.execute(
                    "INSERT INTO frames(source_id,captured_at,fetched_at,sha256,byte_size,"
                    "media_type,"
                    "relative_path,metadata_json) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        metadata.identifier,
                        metadata.date.isoformat(),
                        now,
                        digest,
                        len(payload),
                        media_type,
                        str(relative),
                        metadata_json,
                    ),
                )
                if cursor.lastrowid is None:
                    raise RuntimeError("database did not return a frame id")
                frame_id = int(cursor.lastrowid)
                self._append_audit(conn, "frame.saved", {"frame_id": frame_id, "sha256": digest})
        except Exception:
            if not destination_existed:
                destination.unlink(missing_ok=True)
            raise
        return FrameRecord(
            id=frame_id,
            source_id=metadata.identifier,
            captured_at=metadata.date.isoformat(),
            fetched_at=now,
            sha256=digest,
            byte_size=len(payload),
            media_type=media_type,
            relative_path=str(relative),
            metadata_json=metadata_json,
        )

    def record_analysis(self, record: AnalysisRecord) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO analyses(frame_id,processor,processor_version,status,"
                "result_json,duration_ms,created_at) VALUES(?,?,?,?,?,?,?)",
                (
                    record.frame_id,
                    record.processor,
                    record.processor_version,
                    record.status,
                    json.dumps(record.result, sort_keys=True, separators=(",", ":")),
                    record.duration_ms,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def latest_frames(self, limit: int = 20) -> list[dict[str, object]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT f.*, (SELECT json_group_array(json_object('processor',processor,'status',"
                "status,'result',json(result_json),'duration_ms',duration_ms)) FROM analyses a "
                "WHERE a.frame_id=f.id) analyses FROM frames f ORDER BY captured_at DESC LIMIT ?",
                (min(max(limit, 1), 100),),
            ).fetchall()
        return [dict(row) for row in rows]

    def frame_path(self, frame_id: int) -> Path | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT relative_path FROM frames WHERE id=?", (frame_id,)
            ).fetchone()
        if row is None:
            return None
        candidate = (self.settings.data_dir / row[0]).resolve()
        root = self.settings.data_dir.resolve()
        if root not in candidate.parents:
            raise RuntimeError("stored frame path escaped data root")
        return Path(candidate)

    def frame_media_type(self, frame_id: int) -> str | None:
        with self.connect() as conn:
            row = conn.execute("SELECT media_type FROM frames WHERE id=?", (frame_id,)).fetchone()
        return None if row is None else str(row[0])

    def audit(self, event_type: str, details: dict[str, object]) -> None:
        with self._lock, self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._append_audit(conn, event_type, details)

    def _append_audit(
        self, conn: sqlite3.Connection, event_type: str, details: dict[str, object]
    ) -> None:
        event = json.dumps(
            {"type": event_type, "details": details}, sort_keys=True, separators=(",", ":")
        )
        key_secret = self.settings.audit_hmac_key
        key = None if key_secret is None else key_secret.get_secret_value().encode()
        previous_row = conn.execute(
            "SELECT digest FROM audit_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        previous = "0" * 64 if previous_row is None else str(previous_row[0])
        digest = chain_digest(previous, event.encode(), key)
        conn.execute(
            "INSERT INTO audit_log(created_at,event_json,previous_digest,digest) VALUES(?,?,?,?)",
            (datetime.now(UTC).isoformat(), event, previous, digest),
        )

    def prune_request_log(self, now: datetime, retention_days: int = 35) -> int:
        cutoff = (now - timedelta(days=retention_days)).isoformat()
        with self.connect() as conn:
            cursor = conn.execute("DELETE FROM request_log WHERE attempted_at < ?", (cutoff,))
            return cursor.rowcount

    def verify_audit_chain(self) -> bool:
        key_secret = self.settings.audit_hmac_key
        key = None if key_secret is None else key_secret.get_secret_value().encode()
        previous = "0" * 64
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM audit_log ORDER BY id").fetchall()
        for row in rows:
            if row["previous_digest"] != previous:
                return False
            expected = chain_digest(previous, row["event_json"].encode(), key)
            if expected != row["digest"]:
                return False
            previous = row["digest"]
        return True
